"""Tiles (docs/gpu_tiles.md, Stage 0): the portable-core contract.

Everything runs through parsed source (parse -> ... -> interpreter /
native), per the house conventions.  Three layers are pinned:

  1. INTERPRETER SEMANTICS — the reference: op values, pinned accumulation
     order, the repr format, and the loud dynamic errors.
  2. COMPILE-TIME SHAPE ERRORS — tile_shape_check's `type-tile-shape`
     rejections for statically visible misuse (zero false positives: the
     branch-soundness cases must stay accepted).
  3. NATIVE DIFFERENTIALS — clang-compiled output byte-identical to the
     interpreter, including tile reprs, float sums (the %g print-parity
     fix this work forced), catchable bounds raises, and zero
     placeholders in the emitted module.
"""
from __future__ import annotations

import pytest

from metaxu.compiler.frozen_borrow_checker import TypeCheckError
from metaxu.compiler.mir_interp import InterpError
from metaxu.compiler.pipeline import build_context_from_source, run_pipeline_ctx

from metaxu.compiler.tests.test_codegen_llvm import (
    assert_native_matches_interp,
    count_placeholders,
    interp_run,
    llvm_from_source,
    needs_clang,
)


# ---------------------------------------------------------------------------
# 1. Interpreter semantics (the reference)
# ---------------------------------------------------------------------------

_CORE_SRC = """
fn main() -> int {
    let a = Tile.arange(2, 3);
    let b = Tile.transpose(a);
    let c = Tile.dot(a, b);
    print(Tile.sum(c));
    print(Tile.get(c, 1, 1));
    let z = Tile.filled(2, 2, 1.5);
    print(Tile.sum(Tile.scale(z, 2.0)));
    let v = Tile.to_vec(a);
    print(v[5]);
    let t2 = Tile.from_vec(v, 3, 2);
    print(Tile.get(t2, 2, 1));
    print(a);
    print(z);
    print(Tile.rows(c) + Tile.cols(c));
    print(Tile.sum(Tile.mul(a, a)));
    print(Tile.sum(Tile.add(z, z)));
    0
}
"""

_CORE_OUT = [
    "83",          # sum(arange(2,3) . its transpose) = 5+14+14+50
    "50",          # c[1][1]
    "12.0",        # sum(filled(2,2,1.5) * 2.0)
    "5",           # to_vec row-major last element
    "5",           # from_vec reshaped 3x2, get(2,1)
    "tile[2x3](0, 1, 2; 3, 4, 5)",
    "tile[2x2](1.5, 1.5; 1.5, 1.5)",
    "4",           # rows+cols of the 2x2 dot result
    "55",          # sum of squares 0..5
    "12.0",        # sum(z + z)
]


def test_interp_core_semantics():
    result, out = interp_run(_CORE_SRC)
    assert result == 0
    assert out.splitlines() == _CORE_OUT


def test_interp_dynamic_errors_are_loud_and_catchable():
    # Only DYNAMIC misuse reaches the runtime (static misuse is rejected
    # at compile time below); the messages are the language-visible
    # contract both engines share.
    src = """
fn main() -> int {
    let a = Tile.arange(2, 3);
    let big = Tile.rows(a) + 3;
    let e1 = try { Tile.get(a, big, 0); 0 } catch m { print(m); 1 };
    let @mut v = Vec.new();
    v.push(1);
    let e2 = try { Tile.sum(Tile.from_vec(v, 2, 2)); 0 } catch m { print(m); 2 };
    e1 + e2
}
"""
    result, out = interp_run(src)
    assert result == 3
    assert out.splitlines() == [
        "Tile.get: index out of bounds: (5, 0) (shape 2x3)",
        "Tile.from_vec: Vec length 1 does not fill 2x2 (= 4 elements)",
    ]


def test_interp_mixed_element_vec_is_rejected():
    src = """
fn main() -> int {
    let @mut v = Vec.new();
    v.push(1);
    v.push(1.5);
    try { Tile.sum(Tile.from_vec(v, 1, 2)); 0 } catch m { print(m); 1 }
}
"""
    result, out = interp_run(src)
    assert result == 1
    assert out.splitlines() == [
        "Tile.from_vec: mixed int and float elements in the Vec"]


def test_interp_ops_are_functional_not_in_place():
    # Tiles are immutable values: ops never mutate their operands.
    src = """
fn main() -> int {
    let a = Tile.arange(1, 3);
    let b = Tile.scale(a, 10);
    print(Tile.sum(a));
    print(Tile.sum(b));
    0
}
"""
    _res, out = interp_run(src)
    assert out.splitlines() == ["3", "30"]


# ---------------------------------------------------------------------------
# 2. Compile-time shape errors (type-tile-shape -> TypeCheckError)
# ---------------------------------------------------------------------------

def _expect_reject(src: str, fragment: str) -> None:
    with pytest.raises(TypeCheckError) as ei:
        run_pipeline_ctx(build_context_from_source(src))
    assert fragment in str(ei.value)


@pytest.mark.parametrize("src,fragment", [
    ("fn main() -> int { let a = Tile.zeros(2,3); let b = Tile.zeros(3,2);"
     " Tile.sum(Tile.add(a,b)); 0 }",
     "Tile.add: shape mismatch: 2x3 vs 3x2"),
    ("fn main() -> int { let a = Tile.arange(2,3); let b = Tile.arange(2,3);"
     " Tile.sum(Tile.dot(a,b)); 0 }",
     "Tile.dot: shape mismatch: 2x3 · 2x3 (inner dims 3 and 2)"),
    ("fn main() -> int { let a = Tile.zeros(0,3); 0 }",
     "Tile.zeros: tile shape must be positive, got 0x3"),
    ("fn main() -> int { let a = Tile.arange(2,3);"
     " print(Tile.get(a, 2, 0)); 0 }",
     "Tile.get: index out of bounds: (2, 0) (shape 2x3)"),
    ("fn main() -> int { let a = Tile.arange(2,3);"
     " Tile.sum(Tile.add(a, Tile.zeros(2,3))); 0 }",
     "Tile.add: element kinds differ (int vs float)"),
    ("fn main() -> int { let a = Tile.arange(2,2);"
     " Tile.sum(Tile.scale(a, 1.5)); 0 }",
     "Tile.scale: scalar kind must match tile elements "
     "(int tile, float scalar)"),
    ("fn main() -> int { Tile.sum(Tile.dot(Tile.zeros(2,2))); 0 }",
     "Tile.dot: expects 2 arguments, got 1"),
])
def test_static_shape_misuse_is_a_compile_error(src, fragment):
    _expect_reject(src, fragment)


def test_branch_rebind_suppresses_static_checking():
    # Zero-false-positive contract: a rebind inside a branch makes the
    # shape unknown, so this (dynamically fine) program must compile and
    # run — the dynamic checks remain the backstop.
    src = """
fn main() -> int {
    let mut a = Tile.arange(2, 3);
    if 1 == 1 { a = Tile.arange(3, 2); } else { () };
    print(Tile.sum(Tile.add(a, Tile.arange(3, 2))));
    0
}
"""
    _res, out = interp_run(src)
    assert out.splitlines() == ["30"]


def test_dynamic_shapes_stay_runtime_checked():
    # Non-literal ctor shapes: statically unknown (never a false
    # positive), still loud at run time through the same messages.
    src = """
fn zshape() -> int { 0 }
fn main() -> int {
    let r = zshape();
    try { Tile.sum(Tile.zeros(r, 3)); 0 } catch m { print(m); 1 }
}
"""
    result, out = interp_run(src)
    assert result == 1
    assert out.splitlines() == [
        "Tile.zeros: tile shape must be positive, got 0x3"]


# ---------------------------------------------------------------------------
# 3. Native differentials (byte-identical to the interpreter)
# ---------------------------------------------------------------------------

@needs_clang
def test_native_core_matches_interp(tmp_path):
    ir = assert_native_matches_interp(_CORE_SRC, tmp_path)
    assert count_placeholders(ir) == 0
    # Structural: shapes are burned into kinds, ops are mx_tile_* calls.
    assert "declare ptr @mx_tile_dot(ptr, ptr, i64)" in ir
    assert "call ptr @mx_tile_arange" in ir
    assert "call ptr @mx_tile_to_str" in ir  # repr parity path


@needs_clang
def test_native_dynamic_errors_match_interp(tmp_path):
    src = """
fn main() -> int {
    let a = Tile.arange(2, 3);
    let big = Tile.rows(a) + 3;
    let e1 = try { Tile.get(a, big, 0); 0 } catch m { print(m); 1 };
    let @mut v = Vec.new();
    v.push(1);
    let e2 = try { Tile.sum(Tile.from_vec(v, 2, 2)); 0 } catch m { print(m); 2 };
    e1 + e2
}
"""
    ir = assert_native_matches_interp(src, tmp_path)
    assert count_placeholders(ir) == 0


@needs_clang
def test_native_float_print_parity(tmp_path):
    # The tile differentials caught native print(f64) using %g ("12"
    # for 12.0) where the interpreter prints Python repr ("12.0"); the
    # print helper now routes through mx_f64_to_str.  Pin the parity on
    # the shapes %g got wrong (integral, tiny, precision-heavy).
    src = """
fn main() -> int {
    print(12.0);
    print(0.1);
    print(1.0 / 3.0);
    print(0.00001);
    print(2.5, 4.0);
    0
}
"""
    ir = assert_native_matches_interp(src, tmp_path)
    assert count_placeholders(ir) == 0


@needs_clang
def test_native_vec_of_tiles_matches_interp(tmp_path):
    # Tiles are word kinds: they ride Vec slots as opaque pointers on
    # both engines (identity through the slot, like any pointer word).
    src = """
fn main() -> int {
    let @mut v = Vec.new();
    v.push(Tile.arange(2, 2));
    v.push(Tile.filled(2, 2, 7));
    print(Tile.sum(Tile.add(v[0], v[1])));
    0
}
"""
    ir = assert_native_matches_interp(src, tmp_path)
    assert count_placeholders(ir) == 0


@needs_clang
def test_native_int_dot_float_dot_differential(tmp_path):
    # Pinned accumulation order makes float dot/sum bit-identical across
    # engines; ints are exact.  A non-trivial K exercises the k-loop.
    src = """
fn main() -> int {
    let a = Tile.from_vec(Tile.to_vec(Tile.arange(3, 4)), 3, 4);
    let b = Tile.transpose(a);
    print(Tile.dot(a, b));
    let fa = Tile.scale(Tile.filled(3, 4, 0.125), 3.0);
    print(Tile.sum(Tile.dot(fa, Tile.transpose(fa))));
    0
}
"""
    ir = assert_native_matches_interp(src, tmp_path)
    assert count_placeholders(ir) == 0


def test_non_static_shape_demotes_native_never_wrong_code():
    # A tile whose shape codegen cannot resolve to constants DEMOTES the
    # function (comment-only placeholder) instead of guessing — the
    # interpreter path above proves the program itself is fine.
    src = """
fn zshape() -> int { 2 }
fn main() -> int {
    let r = zshape();
    print(Tile.sum(Tile.zeros(r, 3)));
    0
}
"""
    ir = llvm_from_source(src)
    assert count_placeholders(ir) >= 1
    assert "not statically resolvable" in ir


# ---------------------------------------------------------------------------
# Stage 1: the buffer <-> tile boundary (load/store + masked forms)
# ---------------------------------------------------------------------------

_LOAD_STORE_SRC = """
fn main() -> int {
    let @mut v = Vec.new();
    let mut i = 0;
    while i < 10 { v.push(i * i); i = i + 1 };
    let t = Tile.load(v, 2, 2, 3);
    print(t);
    Tile.store(v, 0, Tile.scale(t, 10));
    print(v[0]);
    print(v[5]);
    let edge = Tile.load_or(v, 8, 1, 4, 0 - 1);
    print(edge);
    Tile.store_clipped(v, 8, Tile.filled(1, 4, 7));
    print(v[9]);
    print(len(v));
    let e = try { Tile.load(v, 8, 1, 4); 0 } catch m { print(m); 0 - 1 };
    print(e);
    0
}
"""

_LOAD_STORE_OUT = [
    "tile[2x3](4, 9, 16; 25, 36, 49)",
    "40",    # store rewrote v[0..6) with 10x the loaded tile
    "490",
    "tile[1x4](64, 81, -1, -1)",   # masked load fills `other` past the end
    "7",     # clipped store wrote only the in-range elements
    "10",    # ...and never grew the Vec
    "Tile.load: range [8, 12) outside Vec length 10",
    "-1",
]


def test_interp_load_store_and_masked_semantics():
    result, out = interp_run(_LOAD_STORE_SRC)
    assert result == 0
    assert out.splitlines() == _LOAD_STORE_OUT


@needs_clang
def test_native_load_store_matches_interp(tmp_path):
    ir = assert_native_matches_interp(_LOAD_STORE_SRC, tmp_path)
    assert count_placeholders(ir) == 0
    assert "call ptr @mx_tile_load_or" in ir
    assert "call void @mx_tile_store_clipped" in ir


def test_interp_store_range_is_loud():
    src = """
fn main() -> int {
    let @mut v = Vec.new();
    v.push(1);
    try { Tile.store(v, 0, Tile.filled(1, 2, 5)); 0 }
    catch m { print(m); 1 }
}
"""
    result, out = interp_run(src)
    assert result == 1
    assert out.splitlines() == [
        "Tile.store: range [0, 2) outside Vec length 1"]


# Tile stores are Vec WRITES: the contended-write permission applies
# exactly as it does to push/pop/index stores.  A spawned kernel writing
# a crossed buffer without a lock raises the canonical message on BOTH
# engines (the same wording test_contention pins for the other mutators).
from metaxu.compiler.tests.test_threads import THREAD_EFFECT  # noqa: E402

_CONTENDED_TILE_STORE_SRC = THREAD_EFFECT + """
fn main() -> int {
    let @mut v = Vec.new();
    v.push(1);
    v.push(2);
    let @mut handles = Vec.new();
    unsafe {
        let t = perform Thread.spawn(|| {
            let msg = try { Tile.store(v, 0, Tile.filled(1, 1, 9)); "no error" }
                      catch e { e };
            print(msg);
            0
        });
        handles.push(t);
    }
    perform Thread.join(handles[0]);
    print(v[0]);
    0
}
"""


def test_interp_tile_store_takes_the_contended_guard():
    from metaxu.compiler.mir_interp import _CONTENDED_WRITE_MSG
    result, out = interp_run(_CONTENDED_TILE_STORE_SRC)
    assert result == 0
    assert out.splitlines() == [_CONTENDED_WRITE_MSG, "1"]


@needs_clang
def test_native_tile_store_contended_matches_interp(tmp_path):
    ir = assert_native_matches_interp(_CONTENDED_TILE_STORE_SRC, tmp_path)
    assert count_placeholders(ir) == 0


# ---------------------------------------------------------------------------
# Stage 1: the kernel seam — std.gpu's Gpu.launch effect
# ---------------------------------------------------------------------------

_VECADD_SRC = """
from std.gpu import Gpu, run_grid;

fn vecadd_kernel(pid: int, a: Vec, b: Vec, out: Vec) -> () {
    let ta = Tile.load(a, pid * 4, 1, 4);
    let tb = Tile.load(b, pid * 4, 1, 4);
    Tile.store(out, pid * 4, Tile.add(ta, tb));
    ()
}

fn main() -> int {
    let @mut a = Vec.new();
    let @mut b = Vec.new();
    let @mut out = Vec.new();
    let mut i = 0;
    while i < 8 { a.push(i); b.push(i * 10); out.push(0); i = i + 1 };
    perform Gpu.launch(2, fn(pid: int) -> vecadd_kernel(pid, a, b, out));
    print(out[0]);
    print(out[3]);
    print(out[7]);
    0
}
"""

_MATMUL_SRC = """
from std.gpu import Gpu, run_grid;

fn mm_kernel(pid: int, a: Vec, b: Vec, c: Vec) -> () {
    let ti = pid / 2;
    let tj = pid % 2;
    let mut k = 0;
    let mut r = Tile.filled(2, 2, 0);
    while k < 2 {
        let ta = Tile.load_rows(a, (ti * 2) * 4 + k * 2, 4, 2, 2, 0);
        let tb = Tile.load_rows(b, (k * 2) * 4 + tj * 2, 4, 2, 2, 0);
        r = Tile.add(r, Tile.dot(ta, tb));
        k = k + 1
    };
    Tile.store_rows(c, (ti * 2) * 4 + tj * 2, 4, r);
    ()
}

fn main() -> int {
    let @mut a = Vec.new();
    let @mut b = Vec.new();
    let @mut c = Vec.new();
    let mut i = 0;
    while i < 16 {
        a.push(i);
        b.push(if i % 5 == 0 { 1 } else { 0 });
        c.push(0);
        i = i + 1
    };
    perform Gpu.launch(4, fn(pid: int) -> mm_kernel(pid, a, b, c));
    print(Tile.from_vec(c, 4, 4));
    0
}
"""

_RAGGED_SRC = """
from std.gpu import Gpu, run_grid;

fn double_kernel(pid: int, v: Vec) -> () {
    let t = Tile.load_or(v, pid * 4, 1, 4, 0);
    Tile.store_clipped(v, pid * 4, Tile.scale(t, 2));
    ()
}

fn main() -> int {
    let @mut v = Vec.new();
    let mut i = 0;
    while i < 10 { v.push(i + 1); i = i + 1 };
    perform Gpu.launch(3, fn(pid: int) -> double_kernel(pid, v));
    print(v[0]);
    print(v[9]);
    print(len(v));
    0
}
"""


def test_interp_vecadd_kernel():
    _res, out = interp_run(_VECADD_SRC)
    assert out.splitlines() == ["0", "33", "77"]


def test_interp_matmul_kernel():
    # TRUE 2D tiled matmul via the row-strided forms.  A = row-major
    # iota(4x4); B = indicator of multiples of 5, which in a 4x4 row-major
    # layout is exactly the IDENTITY (positions 0, 5, 10, 15) — so A · B
    # must equal A, checkable at sight.  History: the first version of
    # this test used the FLAT load/store forms and pinned a value that was
    # not a matrix product at all — both engines agreed (differentials
    # cannot catch a shared wrong idiom), and only an independent
    # ground-truth check exposed it.  test_interp_matmul_ground_truth
    # below keeps that check permanent.
    _res, out = interp_run(_MATMUL_SRC)
    assert out.splitlines() == [
        "tile[4x4](0, 1, 2, 3; 4, 5, 6, 7; 8, 9, 10, 11; 12, 13, 14, 15)"]


def test_interp_matmul_ground_truth():
    # 8x8, 2x2 tiles, non-trivial A and B: the expected product is
    # computed HERE, independently, in Python — never by the thing under
    # test.  This is the check differentials structurally cannot provide.
    n, nt = 8, 4
    src = """
from std.gpu import Gpu, run_grid;

fn mm_kernel(pid: int, a: Vec, b: Vec, c: Vec) -> () {
    let ti = pid / 4;
    let tj = pid % 4;
    let mut k = 0;
    let mut r = Tile.filled(2, 2, 0);
    while k < 4 {
        let ta = Tile.load_rows(a, (ti * 2) * 8 + k * 2, 8, 2, 2, 0);
        let tb = Tile.load_rows(b, (k * 2) * 8 + tj * 2, 8, 2, 2, 0);
        r = Tile.add(r, Tile.dot(ta, tb));
        k = k + 1
    };
    Tile.store_rows(c, (ti * 2) * 8 + tj * 2, 8, r);
    ()
}

fn main() -> int {
    let n = 8;
    let @mut a = Vec.new();
    let @mut b = Vec.new();
    let @mut c = Vec.new();
    let mut i = 0;
    while i < n * n { a.push(i % 7); b.push((i * 3) % 11); c.push(0); i = i + 1 };
    perform Gpu.launch(16, fn(pid: int) -> mm_kernel(pid, a, b, c));
    let mut j = 0;
    while j < n * n { print(c[j]); j = j + 1 };
    0
}
"""
    _res, out = interp_run(src)
    got = [int(x) for x in out.split()]
    A = [[(r * n + cc) % 7 for cc in range(n)] for r in range(n)]
    B = [[((r * n + cc) * 3) % 11 for cc in range(n)] for r in range(n)]
    truth = [sum(A[r][k] * B[k][cc] for k in range(n))
             for r in range(n) for cc in range(n)]
    assert got == truth


def test_interp_ragged_grid_uses_masked_forms():
    # 10 elements, 3 instances of width 4: the last instance is ragged and
    # must neither raise nor grow the Vec.
    _res, out = interp_run(_RAGGED_SRC)
    assert out.splitlines() == ["2", "20", "10"]


def test_interp_launch_is_pid_ordered():
    # The reference semantics: sequential, pid ascending — pinned so any
    # future backend handler has a defined baseline to differ from.
    src = """
from std.gpu import Gpu, run_grid;
fn main() -> int {
    let @mut order = Vec.new();
    perform Gpu.launch(3, fn(pid: int) -> order.push(pid * pid));
    print(order[0]);
    print(order[1]);
    print(order[2]);
    0
}
"""
    _res, out = interp_run(src)
    assert out.splitlines() == ["0", "1", "4"]


def test_interp_launch_is_virtualizable_by_handlers():
    # The whole point of launch-as-effect: a handler can observe or replace
    # the launch without the kernel changing.
    src = """
from std.gpu import Gpu, run_grid;
fn main() -> int {
    let @mut out = Vec.new();
    let mut i = 0;
    while i < 4 { out.push(0); i = i + 1 };
    let last = handle Gpu with {
        launch(n, f) -> {
            print("virtual launch of");
            print(n);
            run_grid(n, f);
            resume(())
        }
    } in {
        perform Gpu.launch(4, fn(pid: int) ->
            Tile.store(out, pid, Tile.filled(1, 1, pid * pid)));
        out[3]
    };
    print(last);
    0
}
"""
    _res, out = interp_run(src)
    assert out.splitlines() == ["virtual launch of", "4", "9"]


@needs_clang
@pytest.mark.parametrize("src", [_VECADD_SRC, _MATMUL_SRC, _RAGGED_SRC],
                         ids=["vecadd", "matmul", "ragged"])
def test_native_kernels_match_interp(src, tmp_path):
    ir = assert_native_matches_interp(src, tmp_path)
    assert count_placeholders(ir) == 0
