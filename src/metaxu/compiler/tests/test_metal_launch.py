"""The Metal launch handler (docs/gpu_tiles.md — handlers as backends).

`std.gpu.run_metal(n, f)` hands a launch to the runtime's Metal engine:
the closure is introspected (named kernel + captured Vec buffers), the
kernel's MSL is emitted, and the grid runs on the best engine — mlx on
Apple silicon, the bit-exact clang++ shim here.  The shim engine is what
makes the HANDLER differentially testable with no Metal in the
container: installing the Metal handler over `Gpu.launch` must produce
byte-identical program output to the default CPU reference handler.

Everything outside the contract is a LOUD, catchable error — a kernel
outside the MSL subset, a non-Vec buffer, a non-canonical closure, a
float buffer holding non-f32-representable values, a contended write
without permission — never a silent CPU fallback.
"""
from __future__ import annotations

import shutil
import struct

import pytest

from metaxu.compiler.metal_launch import available_engine

from metaxu.compiler.tests.test_codegen_llvm import interp_run
from metaxu.compiler.tests.test_threads import THREAD_EFFECT

needs_clangxx = pytest.mark.skipif(shutil.which("clang++") is None,
                                   reason="clang++ not on PATH")


def _f32r(x: float) -> float:
    return struct.unpack("f", struct.pack("f", float(x)))[0]


# The same program body, launched under either handler: the METAL
# handler (run_metal) must reproduce the DEFAULT handler byte for byte.

def _mm_program(metal: bool) -> str:
    launch = ("""
    handle Gpu with {
        launch(n, f) -> { run_metal(n, f); resume(()) }
    } in {
        perform Gpu.launch(4, fn(pid: int) -> mm_kernel(pid, a, b, c));
        ()
    };""" if metal else """
    perform Gpu.launch(4, fn(pid: int) -> mm_kernel(pid, a, b, c));""")
    return """
from std.gpu import Gpu, run_grid, Metal, run_metal;

fn mm_kernel(pid: int, a: Vec, b: Vec, c: Vec) -> () {
    let ti = pid / 2;
    let tj = pid % 2;
    let mut k = 0;
    let mut r = Tile.filled(2, 2, 0);
    while k < 2 {
        let ta = Tile.load_rows(a, (ti * 2) * 4 + k * 2, 4, 2, 2, 0);
        r = Tile.add(r, Tile.dot(ta,
                Tile.load_rows(b, (k * 2) * 4 + tj * 2, 4, 2, 2, 0)));
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
""" + launch + """
    let mut j = 0;
    while j < 16 { print(c[j]); j = j + 1 };
    0
}
"""


def test_engine_available_here():
    # The container has clang++ and no mlx: the shim engine must be the
    # selection, so every test below exercises a REAL dispatch path.
    assert available_engine() == "shim"


@needs_clangxx
def test_metal_handler_matches_cpu_handler_int_matmul():
    _res, cpu = interp_run(_mm_program(metal=False))
    _res, metal = interp_run(_mm_program(metal=True))
    assert metal == cpu
    # B is the identity, so beyond the differential this is ground truth:
    assert metal.splitlines() == [str(i) for i in range(16)]


@needs_clangxx
def test_metal_handler_matches_cpu_handler_f32():
    a_vals = [_f32r((i + 1) * 0.3) for i in range(10)]
    pushes = "\n".join(f"    v.push({v!r});" for v in a_vals)

    def program(metal: bool) -> str:
        launch = ("""
    handle Gpu with {
        launch(n, f) -> { run_metal(n, f); resume(()) }
    } in { perform Gpu.launch(3, fn(pid: int) -> fk(pid, v)); () };"""
                  if metal else """
    perform Gpu.launch(3, fn(pid: int) -> fk(pid, v));""")
        return """
from std.gpu import Gpu, run_grid, Metal, run_metal;

fn fk(pid: int, v: Vec) -> () {
    let t = Tile.to_f32(Tile.load_or(v, pid * 4, 1, 4, 0.0));
    Tile.store_clipped(v, pid * 4, Tile.scale(t, 2.5));
    ()
}

fn main() -> int {
    let @mut v = Vec.new();
""" + pushes + launch + """
    let mut j = 0;
    while j < 10 { print(v[j]); j = j + 1 };
    0
}
"""

    _res, cpu = interp_run(program(metal=False))
    _res, metal = interp_run(program(metal=True))
    assert metal == cpu  # bit parity, float formatting included


# ---------------------------------------------------------------------------
# Loud, catchable errors — never a silent CPU fallback
# ---------------------------------------------------------------------------

def _catch_program(kernel: str, launch_body: str, setup: str = "") -> str:
    return """
from std.gpu import Gpu, run_grid, Metal, run_metal;
""" + kernel + """
fn main() -> int {
    let @mut v = Vec.new();
    v.push(1); v.push(2);
""" + setup + """
    let msg = try { """ + launch_body + """; "no error" } catch e { e };
    print(msg);
    0
}
"""


def test_kernel_outside_subset_is_loud_and_catchable():
    src = _catch_program(
        "fn strictk(pid: int, v: Vec) -> () {\n"
        "    let t = Tile.load(v, 0, 1, 2);\n"
        "    Tile.store_clipped(v, 0, t);\n"
        "    ()\n}",
        "run_metal(1, fn(pid: int) -> strictk(pid, v))")
    _res, out = interp_run(src)
    assert "outside the MSL subset" in out
    assert "masked forms" in out  # the emitter's hint survives


def test_non_vec_buffer_capture_is_loud():
    src = _catch_program(
        "fn k(pid: int, v: Vec) -> () {\n"
        "    Tile.store_clipped(v, pid, Tile.filled(1, 1, 7));\n"
        "    ()\n}",
        "run_metal(1, fn(pid: int) -> k(pid, x))",
        setup="    let x = 5;\n")
    _res, out = interp_run(src)
    assert "expected a Vec" in out


def test_non_canonical_closure_is_loud():
    src = _catch_program(
        "fn k(pid: int, v: Vec) -> () {\n"
        "    Tile.store_clipped(v, pid, Tile.filled(1, 1, 7));\n"
        "    ()\n}",
        "run_metal(1, fn(pid: int) -> { k(pid, v); k(pid, v) })")
    _res, out = interp_run(src)
    assert "more than one call" in out


def test_non_representable_float_buffer_is_loud():
    # 0.1 is not f32-representable: the device buffer is float32, so a
    # silent round would diverge from the CPU handler on unwritten
    # elements — the launch must refuse instead.
    src = """
from std.gpu import Gpu, run_grid, Metal, run_metal;

fn fk(pid: int, v: Vec) -> () {
    let t = Tile.to_f32(Tile.load_or(v, pid * 4, 1, 4, 0.0));
    Tile.store_clipped(v, pid * 4, Tile.scale(t, 2.0));
    ()
}

fn main() -> int {
    let @mut v = Vec.new();
    v.push(0.1); v.push(0.5);
    let msg = try { run_metal(1, fn(pid: int) -> fk(pid, v)); "no error" }
              catch e { e };
    print(msg);
    0
}
"""
    _res, out = interp_run(src)
    assert "f32-representable" in out


def test_native_metal_program_compiles_placeholder_free():
    # Native binaries cannot dispatch Metal (the MSL emitter and closure
    # introspection live in the interpreter/Mac host), but a program
    # using run_metal must still COMPILE clean: the Metal thunk routes
    # to mx_metal_launch, which aborts loudly at dispatch time — an
    # honest hard stop, never a silent CPU fallback and never a
    # placeholder infecting every module that imports std.gpu.
    from metaxu.compiler.tests.test_codegen_llvm import (
        count_placeholders, llvm_from_source)
    ir = llvm_from_source(_mm_program(metal=True))
    assert count_placeholders(ir) == 0
    assert "mx_metal_launch" in ir


def test_contended_launch_write_takes_the_guard():
    # The launch writes its output Vecs: crossing the Vec into a spawned
    # thread marks it contended, and an unpermitted launch write must
    # fail with the SAME wording as every other contended Vec write.
    from metaxu.compiler.mir_interp import _CONTENDED_WRITE_MSG
    src = THREAD_EFFECT + """
from std.gpu import Gpu, run_grid, Metal, run_metal;

fn k(pid: int, v: Vec) -> () {
    Tile.store_clipped(v, pid, Tile.filled(1, 1, 9));
    ()
}

fn main() -> int {
    let @mut v = Vec.new();
    v.push(1);
    v.push(2);
    let @mut handles = Vec.new();
    unsafe {
        let t = perform Thread.spawn(|| {
            let msg = try { run_metal(1, fn(pid: int) -> k(pid, v));
                            "no error" }
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
    result, out = interp_run(src)
    assert result == 0
    assert out.splitlines() == [_CONTENDED_WRITE_MSG, "1"]
