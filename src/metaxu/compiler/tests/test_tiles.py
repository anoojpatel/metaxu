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
    # f32 element kind (Stage 1d): f32 mixes with neither int nor f64.
    ("fn main() -> int { let f = Tile.to_f32(Tile.zeros(2,2));"
     " Tile.sum(Tile.add(f, Tile.zeros(2,2))); 0 }",
     "Tile.add: element kinds differ (f32 vs float)"),
    ("fn main() -> int { let f = Tile.to_f32(Tile.arange(2,2));"
     " Tile.sum(Tile.dot(f, Tile.arange(2,2))); 0 }",
     "Tile.dot: element kinds differ (f32 vs int)"),
    ("fn main() -> int { let f = Tile.to_f32(Tile.zeros(2,2));"
     " Tile.sum(Tile.scale(f, 2)); 0 }",
     "Tile.scale: scalar kind must match tile elements (f32 tile, "
     "int scalar; f32 tiles scale by float scalars)"),
    # f16 element kind (Stage 1f): f16 mixes with NOTHING — not int, not
    # f64, not f32.
    ("fn main() -> int { let h = Tile.to_f16(Tile.zeros(2,2));"
     " Tile.sum(Tile.add(h, Tile.zeros(2,2))); 0 }",
     "Tile.add: element kinds differ (f16 vs float)"),
    ("fn main() -> int { let h = Tile.to_f16(Tile.arange(2,2));"
     " Tile.sum(Tile.dot(h, Tile.arange(2,2))); 0 }",
     "Tile.dot: element kinds differ (f16 vs int)"),
    ("fn main() -> int { let h = Tile.to_f16(Tile.zeros(2,2));"
     " Tile.sum(Tile.mul(h, Tile.to_f32(Tile.zeros(2,2)))); 0 }",
     "Tile.mul: element kinds differ (f16 vs f32)"),
    ("fn main() -> int { let h = Tile.to_f16(Tile.zeros(2,2));"
     " Tile.sum(Tile.scale(h, 2)); 0 }",
     "Tile.scale: scalar kind must match tile elements (f16 tile, "
     "int scalar; f16 tiles scale by float scalars)"),
])
def test_static_shape_misuse_is_a_compile_error(src, fragment):
    _expect_reject(src, fragment)


# f32 (docs/gpu_tiles.md Stage 1d): elements are the f32-representable
# double; every op rounds once through float.  The values below are
# PINNED — they are what real float arithmetic produces (0.1f is
# 0.10000000149011612, four of them sum to 0.4000000059604645, etc.),
# and all three engines must reproduce them bit for bit.
_F32_SRC = """
fn main() -> int {
    let a = Tile.filled(2, 2, 0.1);
    let f = Tile.to_f32(a);
    print(f);
    print(Tile.sum(f));
    print(Tile.add(f, f));
    print(Tile.dot(f, f));
    print(Tile.scale(f, 3.0));
    print(Tile.mul(f, f));
    print(Tile.get(Tile.transpose(f), 1, 0));
    print(Tile.to_f64(f));
    print(Tile.to_f32(Tile.arange(2, 2)));
    print(Tile.sum(Tile.to_f64(Tile.arange(2, 2))));
    let v = Tile.to_vec(f);
    print(v[3]);
    0
}
"""

_F32_OUT = [
    "tile[2x2](0.10000000149011612, 0.10000000149011612; "
    "0.10000000149011612, 0.10000000149011612)",
    "0.4000000059604645",       # f32 accumulation, widened
    "tile[2x2](0.20000000298023224, 0.20000000298023224; "
    "0.20000000298023224, 0.20000000298023224)",
    "tile[2x2](0.020000001415610313, 0.020000001415610313; "
    "0.020000001415610313, 0.020000001415610313)",
    "tile[2x2](0.30000001192092896, 0.30000001192092896; "
    "0.30000001192092896, 0.30000001192092896)",  # factor rounds to f32
    "tile[2x2](0.010000000707805157, 0.010000000707805157; "
    "0.010000000707805157, 0.010000000707805157)",
    "0.10000000149011612",
    "tile[2x2](0.10000000149011612, 0.10000000149011612; "
    "0.10000000149011612, 0.10000000149011612)",  # widening is exact
    "tile[2x2](0.0, 1.0; 2.0, 3.0)",              # int -> f32 is total
    "6.0",
    "0.10000000149011612",                        # to_vec widens
]


def test_interp_f32_rounding_semantics_pinned():
    result, out = interp_run(_F32_SRC)
    assert result == 0
    assert out.splitlines() == _F32_OUT


# f16 (docs/gpu_tiles.md Stage 1f): elements are the f16-representable
# double; every op rounds once through IEEE binary16.  The values below
# are PINNED — computed with Python's struct 'e' round-trip
# (struct.unpack("e", struct.pack("e", x))[0]: f16(0.1) is
# 0.0999755859375, four of them sum to 0.39990234375, etc.), which IS
# real _Float16 arithmetic, and all three engines must reproduce them bit
# for bit.
_F16_SRC = """
fn main() -> int {
    let a = Tile.filled(2, 2, 0.1);
    let h = Tile.to_f16(a);
    print(h);
    print(Tile.sum(h));
    print(Tile.add(h, h));
    print(Tile.dot(h, h));
    print(Tile.scale(h, 3.0));
    print(Tile.mul(h, h));
    print(Tile.get(Tile.transpose(h), 1, 0));
    print(Tile.to_f64(h));
    print(Tile.to_f32(h));
    print(Tile.to_f16(Tile.arange(2, 2)));
    print(Tile.to_f16(Tile.to_f32(a)));
    let v = Tile.to_vec(h);
    print(v[3]);
    0
}
"""

_F16_OUT = [
    "tile[2x2](0.0999755859375, 0.0999755859375; "
    "0.0999755859375, 0.0999755859375)",
    "0.39990234375",            # f16 accumulation, widened
    "tile[2x2](0.199951171875, 0.199951171875; "
    "0.199951171875, 0.199951171875)",
    "tile[2x2](0.019989013671875, 0.019989013671875; "
    "0.019989013671875, 0.019989013671875)",  # round product, then acc
    "tile[2x2](0.2998046875, 0.2998046875; "
    "0.2998046875, 0.2998046875)",            # factor rounds to f16
    "tile[2x2](0.0099945068359375, 0.0099945068359375; "
    "0.0099945068359375, 0.0099945068359375)",
    "0.0999755859375",
    "tile[2x2](0.0999755859375, 0.0999755859375; "
    "0.0999755859375, 0.0999755859375)",      # widening is exact
    "tile[2x2](0.0999755859375, 0.0999755859375; "
    "0.0999755859375, 0.0999755859375)",      # f16 -> f32 is exact too
    "tile[2x2](0.0, 1.0; 2.0, 3.0)",          # int -> f16 is total
    "tile[2x2](0.0999755859375, 0.0999755859375; "
    "0.0999755859375, 0.0999755859375)",      # f16(f32(x)) == f16(x)
    "0.0999755859375",                        # to_vec widens
]


def test_interp_f16_rounding_semantics_pinned():
    result, out = interp_run(_F16_SRC)
    assert result == 0
    assert out.splitlines() == _F16_OUT


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


@needs_clang
def test_native_f32_semantics_match_interp(tmp_path):
    # The pinned f32 rounding program, natively: mx_tile_* with ekind
    # code 2 must reproduce the interpreter's struct-rounded values byte
    # for byte (including reprs and the widened scalar prints).
    ir = assert_native_matches_interp(_F32_SRC, tmp_path)
    assert count_placeholders(ir) == 0
    assert "mx_tile_to_f32" in ir and "mx_tile_to_f64" in ir


_F32_KERNEL_SRC = """
from std.gpu import Gpu, run_grid;

fn fk(pid: int, a: Vec, out: Vec) -> () {
    let t = Tile.to_f32(Tile.load_rows(a, pid * 4, 4, 1, 4, 0.0));
    Tile.store_rows(out, pid * 4, 4, Tile.scale(t, 0.5));
    ()
}

fn main() -> int {
    let @mut a = Vec.new();
    let @mut out = Vec.new();
    let mut i = 0;
    let mut x = 0.1;
    while i < 8 { a.push(x); out.push(0.0); x = x + 1.0; i = i + 1 };
    perform Gpu.launch(2, fn(pid: int) -> fk(pid, a, out));
    let mut j = 0;
    while j < 8 { print(out[j]); j = j + 1 };
    0
}
"""


@needs_clang
def test_native_f32_kernel_matches_interp(tmp_path):
    # An f32 kernel through std.gpu's reference launch, natively: the
    # fused-load idiom (to_f32 of a float-filled strided load) runs
    # unfused on CPU — load f64s, round, scale with per-op rounding —
    # and must match the interpreter bit for bit.
    ir = assert_native_matches_interp(_F32_KERNEL_SRC, tmp_path)
    assert count_placeholders(ir) == 0


@needs_clang
def test_native_f16_semantics_match_interp(tmp_path):
    # The pinned f16 rounding program, natively: mx_tile_* with ekind
    # code 3 must reproduce the interpreter's struct-'e'-rounded values
    # byte for byte (mx_f16r is (double)(_Float16)x — the same IEEE
    # binary16 round-to-nearest-even).
    ir = assert_native_matches_interp(_F16_SRC, tmp_path)
    assert count_placeholders(ir) == 0
    assert "mx_tile_to_f16" in ir


_F16_KERNEL_SRC = """
from std.gpu import Gpu, run_grid;

fn hk(pid: int, a: Vec, out: Vec) -> () {
    let h = Tile.to_f16(Tile.to_f32(
        Tile.load_rows(a, pid * 4, 4, 1, 4, 0.0)));
    Tile.store_rows(out, pid * 4, 4, Tile.to_f32(Tile.scale(h, 0.3)));
    ()
}

fn main() -> int {
    let @mut a = Vec.new();
    let @mut out = Vec.new();
    let mut i = 0;
    let mut x = 0.1;
    while i < 8 { a.push(x); out.push(0.0); x = x + 1.0; i = i + 1 };
    perform Gpu.launch(2, fn(pid: int) -> hk(pid, a, out));
    let mut j = 0;
    while j < 8 { print(out[j]); j = j + 1 };
    0
}
"""


@needs_clang
def test_native_f16_kernel_matches_interp(tmp_path):
    # An f16 compute kernel through std.gpu's reference launch, natively:
    # f16 tiles arise via Tile.to_f16 and convert back through Tile.to_f32
    # before storing (the MSL compute-only rule, exercised here on CPU) —
    # the per-op f16 rounding must match the interpreter bit for bit.
    ir = assert_native_matches_interp(_F16_KERNEL_SRC, tmp_path)
    assert count_placeholders(ir) == 0
    assert "mx_tile_to_f16" in ir


# ---------------------------------------------------------------------------
# 7. Higher-order tile ops (docs/gpu_tiles.md): map / zip / reduce_rows /
#    reduce_cols / broadcast_rows / broadcast_cols take a Metaxu function per
#    element; f32/f16 elements reach it in NARROW MODE (every op rounds to
#    the width), which is what std/tile.mx builds exp, row_max, sub_rows and
#    the rest on.
# ---------------------------------------------------------------------------

_HOF_INT_SRC = """
fn main() -> int {
    let a = Tile.arange(2, 3);
    print(Tile.map(a, fn(x: int) -> x * x));
    let rs = Tile.reduce_rows(a, 0, fn(acc: int, x: int) -> acc + x);
    print(rs);
    let cs = Tile.reduce_cols(a, 100, fn(acc: int, x: int) -> acc - x);
    print(cs);
    print(Tile.zip(a, Tile.map(a, fn(x: int) -> x * x), fn(x: int, y: int) -> y - x));
    print(Tile.broadcast_rows(a, rs, fn(x: int, s: int) -> x * 10 + s));
    print(Tile.broadcast_cols(a, cs, fn(x: int, s: int) -> s - x));
    print(Tile.reduce_rows(a, 0, fn(acc: int, x: int) -> max(acc, x)));
    print(max(3, 7) + min(3, 7));
    0
}
"""

_HOF_INT_OUT = [
    "tile[2x3](0, 1, 4; 9, 16, 25)",
    "tile[2x1](3; 12)",
    "tile[1x3](97, 95, 93)",          # 100 - (0+3), 100 - (1+4), 100 - (2+5)
    "tile[2x3](0, 0, 2; 6, 12, 20)",
    "tile[2x3](3, 13, 23; 42, 52, 62)",
    "tile[2x3](97, 94, 91; 94, 91, 88)",
    "tile[2x1](2; 5)",
    "10",
]


def test_interp_higher_order_int_semantics():
    result, out = interp_run(_HOF_INT_SRC)
    assert result == 0
    assert out.splitlines() == _HOF_INT_OUT


def _f32(x):
    import struct
    return struct.unpack("f", struct.pack("f", x))[0]


def _f16(x):
    import struct
    return struct.unpack("e", struct.pack("e", x))[0]


def test_interp_narrow_mode_rounds_per_op():
    # x*x + x on an f32 tile: in f32 arithmetic each step rounds, which
    # differs from evaluating the body in f64 and rounding once at the end.
    import math
    src = """
fn main() -> int {
    let f = Tile.to_f32(Tile.filled(1, 2, 1.1));
    print(Tile.map(f, fn(x: float) -> x * x + x));
    print(Tile.map(f, fn(x: float) -> exp(x) - 1.0));
    print(Tile.reduce_rows(f, 0.0, fn(acc: float, x: float) -> acc + x));
    let h = Tile.to_f16(f);
    print(Tile.map(h, fn(x: float) -> x * x + x));
    print(Tile.map(Tile.filled(1, 1, 1.1), fn(x: float) -> x * x + x));
    let c = 0.1;
    print(Tile.map(f, fn(x: float) -> x * 0.1));
    print(Tile.map(f, fn(x: float) -> x + c));
    print(Tile.map(f, fn(x: float) -> if x > 1.0 { 0.1 } else { x }));
    0
}
"""
    result, out = interp_run(src)
    x = _f32(1.1)
    per_op = _f32(_f32(x * x) + x)
    once = _f32(x * x + x)                     # the body in f64, rounded once
    assert per_op != once                      # the two definitions differ here
    assert out.splitlines()[0] == f"tile[1x2]({per_op!r}, {per_op!r})"
    e = _f32(_f32(math.exp(x)) - 1.0)
    assert out.splitlines()[1] == f"tile[1x2]({e!r}, {e!r})"
    s = _f32(_f32(0.0 + x) + x)
    assert out.splitlines()[2] == f"tile[1x1]({s!r})"
    y = _f16(1.1)
    h = _f16(_f16(y * y) + y)
    assert out.splitlines()[3] == f"tile[1x2]({h!r}, {h!r})"
    assert out.splitlines()[4] == f"tile[1x1]({1.1 * 1.1 + 1.1!r})"   # f64: no rounding
    # Narrow mode is the CALL's: a literal, a captured float and a value
    # chosen by a branch are all at the width, not just the operands that
    # came out of the tile (what a float-typed body computes on the device).
    lit = _f32(x * _f32(0.1))
    assert lit != _f32(x * 0.1)
    assert out.splitlines()[5] == f"tile[1x2]({lit!r}, {lit!r})"
    cap = _f32(x + _f32(0.1))
    assert out.splitlines()[6] == f"tile[1x2]({cap!r}, {cap!r})"
    assert out.splitlines()[7] == f"tile[1x2]({_f32(0.1)!r}, {_f32(0.1)!r})"


def test_interp_higher_order_errors_are_loud():
    # Shapes and element kinds the checker cannot see statically (a shape
    # from a call, a Vec's runtime contents) are checked at run time,
    # loudly and catchably, with the same wording the static check uses.
    cases = [
        ("Tile.zip(Tile.arange(2, 2), Tile.arange(pick(2, 3, false), 3), fn(a: int, b: int) -> a + b)",
         "Tile.zip: shape mismatch: 2x2 vs 3x3"),
        ("Tile.broadcast_rows(Tile.arange(2, 3), Tile.arange(pick(1, 2, true), 3), fn(a: int, b: int) -> a + b)",
         "Tile.broadcast_rows: shape mismatch: expected a 2x1 column for a 2x3 tile, got 1x3"),
        ("Tile.map(Tile.arange(1, 2), fn(x: int) -> 1.5)",
         "Tile.map: the function must return an int for an int tile, got 'Float'"),
        ("Tile.reduce_rows(Tile.from_vec(floats(), 1, 2), 0, fn(a: float, b: float) -> a + b)",
         "Tile.reduce_rows: init must be a float for a float tile, got 'Int'"),
        ("Tile.map(Tile.arange(1, 2), fn(x: int, y: int) -> x)",
         "Tile.map: the function takes 2 parameter(s), expected 1"),
    ]
    for expr, msg in cases:
        src = ("fn pick(a: int, b: int, flag: bool) -> int { if flag { a } else { b } }\n"
               "fn floats() -> Vec { let @mut v = Vec.new(); v.push(1.5); v.push(2.5); v }\n"
               "fn main() -> int {\n"
               f"    let r = try {{ let t = {expr}; 0 }} catch e {{ print(e); 1 }};\n"
               "    r\n}\n")
        result, out = interp_run(src)
        assert result == 1, (expr, out)
        assert out.strip() == msg, (expr, out)


@pytest.mark.parametrize("src, fragment", [
    ("fn main() -> int { let t = Tile.zip(Tile.arange(2, 2), Tile.arange(3, 2), fn(a: int, b: int) -> a); 0 }",
     "Tile.zip: shape mismatch: 2x2 vs 3x2"),
    ("fn main() -> int { let t = Tile.broadcast_cols(Tile.arange(2, 3), Tile.arange(2, 1), fn(a: int, b: int) -> a); 0 }",
     "Tile.broadcast_cols: shape mismatch: expected a 1x3 row for a 2x3 tile, got 2x1"),
    ("fn main() -> int { let t = Tile.reduce_rows(Tile.arange(2, 3), 0.0, fn(a: int, b: int) -> a); 0 }",
     "Tile.reduce_rows: init must be an int for an int tile, got a float"),
    ("fn main() -> int { let t = Tile.add(Tile.reduce_rows(Tile.arange(2, 3), 0, fn(a: int, b: int) -> a), Tile.arange(1, 2)); 0 }",
     "Tile.add: shape mismatch: 2x1 vs 1x2"),
])
def test_higher_order_shape_misuse_is_a_compile_error(src, fragment):
    ctx = build_context_from_source(src)
    with pytest.raises(TypeCheckError) as ei:
        run_pipeline_ctx(ctx)
    assert fragment in str(ei.value)


_HOF_NATIVE_SRC = """
fn main() -> int {
    let a = Tile.arange(2, 3);
    print(Tile.map(a, fn(x: int) -> x * x));
    let rs = Tile.reduce_rows(a, 0, fn(acc: int, x: int) -> acc + x);
    print(rs);
    let cs = Tile.reduce_cols(a, 100, fn(acc: int, x: int) -> acc - x);
    print(cs);
    print(Tile.zip(a, Tile.map(a, fn(x: int) -> x * x), fn(x: int, y: int) -> y - x));
    print(Tile.broadcast_rows(a, rs, fn(x: int, s: int) -> x * 10 + s));
    print(Tile.broadcast_cols(a, cs, fn(x: int, s: int) -> s - x));
    print(Tile.reduce_rows(a, 0, fn(acc: int, x: int) -> max(acc, x)));
    print(max(3, 7) + min(3, 7));
    let d = Tile.filled(2, 2, 1.1);
    print(Tile.map(d, fn(x: float) -> x * x + x));
    print(Tile.reduce_rows(d, 0.5, fn(acc: float, x: float) -> acc + exp(x) - log(x)));
    print(Tile.broadcast_cols(d, Tile.reduce_cols(d, 0.0, fn(p: float, q: float) -> max(p, q)),
                              fn(x: float, m: float) -> x - m));
    let k = 3;
    print(Tile.map(a, fn(x: int) -> x * k));
    print(max(1.5, 2.5));
    print(min(1.5, 2.5));
    let r = try {
        let bad = Tile.zip(a, Tile.arange(pick(2, 3, false), 3), fn(x: int, y: int) -> x + y);
        0
    } catch e { print(e); 1 };
    r
}
fn pick(a: int, b: int, flag: bool) -> int { if flag { a } else { b } }
"""


def test_higher_order_int_and_f64_ops_lower_natively():
    ir = llvm_from_source(_HOF_NATIVE_SRC)
    assert count_placeholders(ir) == 0
    for sym in ("mx_tile_map", "mx_tile_zip", "mx_tile_reduce_rows",
                "mx_tile_reduce_cols", "mx_tile_broadcast_rows",
                "mx_tile_broadcast_cols"):
        assert f"@{sym}(" in ir, sym
    assert "define internal i64 @mx.tth." in ir      # the per-site thunks
    assert "call double @exp(double" in ir and "call double @log(double" in ir
    assert "fcmp ogt double" in ir and "icmp sgt i64" in ir   # max, inline


@needs_clang
def test_native_higher_order_ops_match_interp(tmp_path):
    # Captured scalars, math builtins, max/min, every op, and the catchable
    # dynamic shape mismatch raise, byte for byte.
    assert_native_matches_interp(_HOF_NATIVE_SRC, tmp_path)


_NARROW_NATIVE_SRC = """
fn big() -> float { 3.5 }
fn half_of(x: float) -> float { x * 0.5 }
fn main() -> int {
    let x = 0.1;
    let f = Tile.to_f32(Tile.filled(1, 2, 1.1));
    print(Tile.map(f, fn(v: float) -> v + x * 3.0 + big()));
    print(Tile.map(f, fn(v: float) -> v * v + v));
    print(Tile.map(f, fn(v: float) -> exp(v) * 0.1 - log(v) / 7.0));
    print(Tile.map(f, fn(v: float) -> if v > 1.0 { sqrt(v) } else { half_of(v) }));
    print(Tile.zip(f, Tile.map(f, fn(v: float) -> v * 0.3), fn(a: float, b: float) -> max(a, b) - min(a, b)));
    print(Tile.reduce_rows(f, 0.25, fn(acc: float, v: float) -> acc + v * 0.3));
    print(Tile.broadcast_cols(f, Tile.reduce_cols(f, 0.0, fn(p: float, q: float) -> p + q),
                              fn(v: float, s: float) -> v / s));
    let h = Tile.to_f16(f);
    print(Tile.map(h, fn(v: float) -> v * v + v));
    print(Tile.map(h, fn(v: float) -> exp(v) * 0.1));
    print(Tile.reduce_rows(h, 0.0, fn(a: float, b: float) -> a + b * 0.3));
    print(Tile.broadcast_rows(h, Tile.reduce_rows(h, 0.0, fn(a: float, b: float) -> a + b),
                              fn(v: float, s: float) -> v - s * x));
    0
}
"""


def test_narrow_lambdas_lower_natively_with_per_op_rounding():
    # The lambda of a map over an f32/f16 tile is compiled as a NARROW
    # LAMBDA: its module calls are inlined and every f64 it binds rounds
    # through float/half, the interpreter's narrow-mode rule.
    ir = llvm_from_source(_NARROW_NATIVE_SRC)
    assert count_placeholders(ir) == 0
    assert "fptrunc double %a.v to float  ; narrow lambda: parameter v rounds to f32" in ir
    assert "narrow lambda: capture x_" in ir
    assert "to half  ; narrow lambda:" in ir
    assert "fpext half" in ir and "fpext float" in ir
    # the helper was inlined into the lambda, not called from it
    assert "narrow lambda: c1$i" in ir
    assert "call double @exp(double" in ir


@needs_clang
def test_native_narrow_lambdas_match_interp(tmp_path):
    assert_native_matches_interp(_NARROW_NATIVE_SRC, tmp_path)


@pytest.mark.parametrize("src,fragment", [
    # the closure is also called directly: that call would compute in f64
    ("fn main() -> int { let f = Tile.to_f32(Tile.filled(1, 2, 1.1));"
     " let g = fn(v: float) -> v * v; print(g(2.0)); print(Tile.map(f, g)); 0 }",
     "is also used by a call of it, which would compute in f64"),
    # one function, two widths
    ("fn main() -> int { let f = Tile.to_f32(Tile.filled(1, 2, 1.1));"
     " let g = fn(v: float) -> v * v; print(Tile.map(f, g));"
     " print(Tile.map(Tile.to_f16(f), g)); 0 }",
     "applied to tiles of f32 and of f16 elements (one width per function)"),
    # a recursive helper has no finite inlining
    ("fn again(x: float, n: int) -> float { if n == 0 { x } else { again(x * 0.5, n - 1) } }"
     " fn main() -> int { let f = Tile.to_f32(Tile.filled(1, 2, 1.1));"
     " print(Tile.map(f, fn(v: float) -> again(v, 2))); 0 }",
     "recursive call to 'again' cannot be inlined"),
    # a function value called from inside the lambda
    ("fn main() -> int { let f = Tile.to_f32(Tile.filled(1, 2, 1.1));"
     " let k = fn(y: float) -> y * 0.5;"
     " print(Tile.map(f, fn(v: float) -> k(v) + 1.0)); 0 }",
     "calls a function value"),
])
def test_narrow_lambda_misuse_demotes_with_a_reason(src, fragment):
    ir = llvm_from_source(src)
    assert count_placeholders(ir) >= 1
    assert fragment in ir
