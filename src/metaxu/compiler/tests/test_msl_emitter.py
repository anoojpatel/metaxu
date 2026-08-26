"""MSL emission (docs/gpu_tiles.md, Stage 1c).

No Metal exists in this container, and the tests are REAL anyway: the
emitted kernel body is C++-compatible MSL by design (switch-machine CFG,
plain pointers, `thread_position_in_grid` shimmed in five lines), so the
differential here compiles the emitted body with clang++ and races it
against the interpreter running the same kernel through std.gpu's
sequential reference launch.  What stays untested here — the MLX binding
and Metal address spaces — is exactly what the generated Mac harness
self-checks (structure pinned below).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import pytest

from metaxu.compiler.emit_msl import MslError, emit_msl_kernel

from metaxu.compiler.tests.test_codegen_llvm import interp_run

needs_clangxx = pytest.mark.skipif(shutil.which("clang++") is None,
                                   reason="clang++ not on PATH")


# ---------------------------------------------------------------------------
# Kernels under test (each also runs through std.gpu in main so the
# interpreter output IS the reference the shim must match)
# ---------------------------------------------------------------------------

def _driver(kernel_src: str, kernel: str, grid: int, bufs: dict) -> str:
    """A main() that builds the buffers, launches the kernel over the
    grid through std.gpu, and prints every buffer element."""
    lines = ["from std.gpu import Gpu, run_grid;", kernel_src,
             "fn main() -> int {"]
    for name, vals in bufs.items():
        lines.append(f"    let @mut {name} = Vec.new();")
        for v in vals:
            lines.append(f"    {name}.push({v});")
    args = ", ".join(bufs)
    lines.append(f"    perform Gpu.launch({grid}, fn(pid: int) -> "
                 f"{kernel}(pid, {args}));")
    for name, vals in bufs.items():
        lines.append("    let mut i_%s = 0;" % name)
        lines.append(f"    while i_{name} < {len(vals)} "
                     f"{{ print({name}[i_{name}]); "
                     f"i_{name} = i_{name} + 1 }};")
    lines.append("    0")
    lines.append("}")
    return "\n".join(lines)


def _run_shim(k, grid: int, bufs: dict) -> list:
    """Compile the emitted body with clang++ and run it over the grid."""
    with tempfile.TemporaryDirectory(prefix="mx_msl_") as d:
        cpp = os.path.join(d, "k.cpp")
        with open(cpp, "w") as fh:
            fh.write(k.cpp_wrapper())
        exe = os.path.join(d, "k")
        r = subprocess.run(["clang++", "-O2", "-std=c++17", cpp, "-o", exe],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"emitted body failed C++: {r.stderr}"
        feed = [str(len(bufs))]
        for name in k.in_bufs:
            vals = bufs[name]
            feed.append(str(len(vals)))
            feed += [str(x) for x in vals]
        p = subprocess.run([exe, str(grid)], input=" ".join(feed),
                           capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        return [int(x) for x in p.stdout.split()]


def _differential(kernel_src: str, kernel: str, grid: int,
                  bufs: dict) -> None:
    """interpreter-through-std.gpu == clang++-compiled emitted body."""
    src = _driver(kernel_src, kernel, grid, bufs)
    _res, out = interp_run(src)
    interp_flat = [int(x) for x in out.split()]
    k = emit_msl_kernel(src, kernel)
    shim = _run_shim(k, grid, bufs)
    # the shim prints only the WRITTEN buffers (merged); compare those
    # against the same slices of the interpreter's full dump
    offsets = {}
    pos = 0
    for name in bufs:
        offsets[name] = pos
        pos += len(bufs[name])
    expect = []
    for name in k.out_bufs:
        expect += interp_flat[offsets[name]:offsets[name] + len(bufs[name])]
    assert shim == expect, (shim, expect)


# TRUE 2D tiled matmul (row-strided forms — the flat forms silently read
# the wrong elements for 2D tiles, a bug both engines shared until a
# ground-truth check caught it; see test_tiles.py's matmul history note).
_MM = """
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
"""

_VECADD = """
fn va_kernel(pid: int, a: Vec, b: Vec, out: Vec) -> () {
    let ta = Tile.load_or(a, pid * 4, 1, 4, 0);
    let tb = Tile.load_or(b, pid * 4, 1, 4, 0);
    Tile.store_clipped(out, pid * 4, Tile.add(ta, tb));
    ()
}
"""

_RAGGED = """
fn dbl_kernel(pid: int, v: Vec) -> () {
    let t = Tile.load_or(v, pid * 4, 1, 4, 0);
    Tile.store_clipped(v, pid * 4, Tile.scale(t, 2));
    ()
}
"""

_TRANSPOSE = """
fn tr_kernel(pid: int, a: Vec, out: Vec) -> () {
    let t = Tile.load_or(a, pid * 6, 2, 3, 0);
    let tt = Tile.transpose(t);
    Tile.store_clipped(out, pid * 6, Tile.scale(tt,
        Tile.sum(Tile.arange(1, 2)) + 1));
    ()
}
"""


@needs_clangxx
def test_matmul_kernel_shim_matches_interp():
    # B is the 4x4 identity (i % 5 == 0 hits 0, 5, 10, 15), so the shim
    # must reproduce C == A — a ground-truth pin, not just a differential.
    bufs = {
        "a": list(range(16)),
        "b": [1 if i % 5 == 0 else 0 for i in range(16)],
        "c": [0] * 16,
    }
    _differential(_MM, "mm_kernel", 4, bufs)
    src = _driver(_MM, "mm_kernel", 4, bufs)
    k = emit_msl_kernel(src, "mm_kernel")
    assert _run_shim(k, 4, bufs) == list(range(16))  # C == A


@needs_clangxx
def test_vecadd_kernel_shim_matches_interp():
    _differential(_VECADD, "va_kernel", 2, {
        "a": list(range(8)),
        "b": [i * 10 for i in range(8)],
        "out": [0] * 8,
    })


@needs_clangxx
def test_ragged_kernel_masked_semantics_shim_matches_interp():
    # 10 elements, 3 instances of width 4: the emitted mask merge must
    # reproduce the clipped-store semantics exactly.
    _differential(_RAGGED, "dbl_kernel", 3, {"v": [i + 1 for i in range(10)]})


@needs_clangxx
def test_transpose_sum_kernel_shim_matches_interp():
    _differential(_TRANSPOSE, "tr_kernel", 2, {
        "a": list(range(12)),
        "out": [0] * 12,
    })


# ---------------------------------------------------------------------------
# Subset rejections: out-of-subset kernels fail LOUDLY with the reason
# ---------------------------------------------------------------------------

def _kernel_module(body: str) -> str:
    return ("from std.gpu import Gpu, run_grid;\n" + body
            + "\nfn main() -> int { 0 }")


@pytest.mark.parametrize("body,fragment", [
    ("fn k(pid: int, v: Vec) -> () { let t = Tile.load(v, 0, 1, 2);"
     " Tile.store_clipped(v, 0, t); () }",
     "masked forms"),
    ("fn k(pid: int, v: Vec) -> () {"
     " Tile.store_clipped(v, 0, Tile.zeros(1, 2)); () }",
     "Metal has no f64"),
    ("fn k(pid: int, v: Vec) -> () { print(pid); () }",
     "outside the MSL subset"),
    ("fn k(pid: int, v: Vec) -> () {"
     " let n = len(v);"
     " Tile.store_clipped(v, n, Tile.filled(1, 1, 1)); () }",
     "outside the MSL subset"),
])
def test_out_of_subset_kernels_are_rejected(body, fragment):
    with pytest.raises(MslError) as ei:
        emit_msl_kernel(_kernel_module(body), "k")
    assert fragment in str(ei.value)


def test_unknown_kernel_name_is_loud():
    with pytest.raises(MslError) as ei:
        emit_msl_kernel(_kernel_module(
            "fn k(pid: int, v: Vec) -> () { () }"), "nope")
    assert "not found" in str(ei.value)


def test_dynamic_tile_shape_in_kernel_is_rejected():
    body = ("fn k(pid: int, v: Vec) -> () {"
            " let t = Tile.load_or(v, 0, 1, pid + 1, 0);"
            " Tile.store_clipped(v, 0, t); () }")
    with pytest.raises(MslError) as ei:
        emit_msl_kernel(_kernel_module(body), "k")
    assert "integer literal" in str(ei.value)


# ---------------------------------------------------------------------------
# The Mac harness: structure pinned (it cannot run here)
# ---------------------------------------------------------------------------

def test_mlx_harness_structure():
    src = _driver(_VECADD, "va_kernel", 2,
                  {"a": list(range(8)), "b": [0] * 8, "out": [0] * 8})
    k = emit_msl_kernel(src, "va_kernel")
    h = k.mlx_harness(2, {"a": list(range(8)), "b": [0] * 8,
                          "out": [0] * 8},
                      {"out": list(range(8))})
    assert "mx.fast.metal_kernel(" in h
    assert "input_names=['a_in', 'b_in', 'out_in', 'lens']" in h
    assert "output_names=['out_out', 'out_wm']" in h
    assert "thread_position_in_grid" in h
    assert "EXPECTED_out = [0, 1, 2, 3, 4, 5, 6, 7]" in h
    assert "sys.exit(0 if ok else 1)" in h
    # the body inside the harness is the same fragment the shim compiled
    assert "while (__run) { switch (__bb) {" in h


# ---------------------------------------------------------------------------
# The user-facing generator script
# ---------------------------------------------------------------------------

def test_emit_metal_harness_script(tmp_path):
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[4]
    src = tmp_path / "kernels.mx"
    src.write_text("from std.gpu import Gpu, run_grid;\n" + _VECADD
                   + "\nfn main() -> int { 0 }\n")
    out = tmp_path / "harness.py"
    r = subprocess.run(
        [sys.executable, str(repo / "scripts" / "emit_metal_harness.py"),
         str(src), "va_kernel", "--grid", "2",
         "--buf", "a=1,2,3,4,5,6,7,8",
         "--buf", "b=10,20,30,40,50,60,70,80",
         "--buf", "out=0,0,0,0,0,0,0,0",
         "-o", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "expected out: [11, 22, 33, 44, 55, 66, 77, 88]" in r.stdout
    h = out.read_text()
    assert "EXPECTED_out = [11, 22, 33, 44, 55, 66, 77, 88]" in h
    # without mlx the harness exits 2 with a clear message, never a crash
    p = subprocess.run([sys.executable, str(out)],
                       capture_output=True, text=True)
    assert p.returncode in (0, 2), p.stderr
