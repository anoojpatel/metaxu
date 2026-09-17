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


def _f32r(x) -> float:
    """Round a value to the f32-representable double — the boundary rule
    for float buffers (they are float32 on the device)."""
    import struct
    return struct.unpack("f", struct.pack("f", float(x)))[0]


def _run_shim(k, grid: int, bufs: dict) -> list:
    """Compile the emitted body with clang++ and run it over the grid.

    -ffp-contract=off pins the f32 rounding order (an FMA would produce
    a different — better-rounded but DIVERGENT — dot product)."""
    with tempfile.TemporaryDirectory(prefix="mx_msl_") as d:
        cpp = os.path.join(d, "k.cpp")
        with open(cpp, "w") as fh:
            fh.write(k.cpp_wrapper())
        exe = os.path.join(d, "k")
        r = subprocess.run(["clang++", "-O2", "-std=c++17",
                            "-ffp-contract=off", cpp, "-o", exe],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"emitted body failed C++: {r.stderr}"
        feed = [str(len(bufs))]
        for name in k.in_bufs:
            vals = bufs[name]
            feed.append(str(len(vals)))
            feed += [repr(x) for x in vals]
        p = subprocess.run([exe, str(grid)], input=" ".join(feed),
                           capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        toks = p.stdout.split()
        out, pos = [], 0
        for name in k.out_bufs:
            conv = float if k.buf_types[name] == "float" else int
            n = len(bufs[name])
            out += [conv(x) for x in toks[pos:pos + n]]
            pos += n
        return out


def _differential(kernel_src: str, kernel: str, grid: int,
                  bufs: dict) -> None:
    """interpreter-through-std.gpu == clang++-compiled emitted body.

    Float comparisons are EXACT equality: both engines produce
    f32-representable doubles, %.17g round-trips them, and the shim's
    contraction is off — bit parity is the contract, not closeness."""
    src = _driver(kernel_src, kernel, grid, bufs)
    _res, out = interp_run(src)
    k = emit_msl_kernel(src, kernel)
    toks = out.split()
    vals, pos = {}, 0
    for name in bufs:
        conv = float if k.buf_types[name] == "float" else int
        n = len(bufs[name])
        vals[name] = [conv(x) for x in toks[pos:pos + n]]
        pos += n
    shim = _run_shim(k, grid, bufs)
    expect = []
    for name in k.out_bufs:
        expect += vals[name]
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


# f32 kernels (docs/gpu_tiles.md Stage 1d): float buffers via the
# to_f32-wrapped-load fusion; float inputs are f32-representable by the
# boundary rule (they are float32 on the device).

_FMM = """
fn fmm_kernel(pid: int, a: Vec, b: Vec, c: Vec) -> () {
    let ti = pid / 2;
    let tj = pid % 2;
    let mut k = 0;
    let mut r = Tile.to_f32(Tile.filled(2, 2, 0));
    while k < 2 {
        let ta = Tile.to_f32(
            Tile.load_rows(a, (ti * 2) * 4 + k * 2, 4, 2, 2, 0.0));
        let tb = Tile.to_f32(
            Tile.load_rows(b, (k * 2) * 4 + tj * 2, 4, 2, 2, 0.0));
        r = Tile.add(r, Tile.dot(ta, tb));
        k = k + 1
    };
    Tile.store_rows(c, (ti * 2) * 4 + tj * 2, 4, r);
    ()
}
"""

_FSCALE = """
fn fscale_kernel(pid: int, v: Vec) -> () {
    let t = Tile.to_f32(Tile.load_or(v, pid * 4, 1, 4, 0.0));
    Tile.store_clipped(v, pid * 4, Tile.scale(t, 2.5));
    ()
}
"""


@needs_clangxx
def test_f32_matmul_kernel_shim_matches_interp():
    # B is the 4x4 float identity, so beyond the differential the shim
    # must reproduce C == A exactly (multiplying by 1.0f and adding 0.0f
    # are exact in f32) — a ground-truth pin.
    a = [_f32r(i * 0.1) for i in range(16)]
    bufs = {
        "a": a,
        "b": [1.0 if i % 5 == 0 else 0.0 for i in range(16)],
        "c": [0.0] * 16,
    }
    _differential(_FMM, "fmm_kernel", 4, bufs)
    src = _driver(_FMM, "fmm_kernel", 4, bufs)
    k = emit_msl_kernel(src, "fmm_kernel")
    assert k.buf_types == {"a": "float", "b": "float", "c": "float"}
    assert _run_shim(k, 4, bufs) == a  # C == A


@needs_clangxx
def test_f32_matmul_ground_truth_independent():
    # The differential alone cannot catch an idiom both engines share
    # (the flat-vs-strided lesson): pin the f32 matmul against a product
    # computed HERE, in Python, with explicit per-op f32 rounding.
    import random
    rng = random.Random(7)
    a = [_f32r(rng.uniform(-1, 1)) for _ in range(16)]
    b = [_f32r(rng.uniform(-1, 1)) for _ in range(16)]
    bufs = {"a": a, "b": b, "c": [0.0] * 16}
    # Mirror the kernel's BLOCKED accumulation: each 2-wide k-block is a
    # Tile.dot (inner rounding per product and per accumulate), and the
    # blocks combine through Tile.add — a different f32 association than
    # a flat k loop, and part of the pinned semantics.
    expect = [0.0] * 16
    for i in range(4):
        for j in range(4):
            r = 0.0
            for kb in range(2):
                acc = 0.0
                for kk in range(2):  # pinned order: k ascending
                    gk = kb * 2 + kk
                    acc = _f32r(acc + _f32r(a[i * 4 + gk] * b[gk * 4 + j]))
                r = _f32r(r + acc)
            expect[i * 4 + j] = r
    src = _driver(_FMM, "fmm_kernel", 4, bufs)
    k = emit_msl_kernel(src, "fmm_kernel")
    assert _run_shim(k, 4, bufs) == expect
    _differential(_FMM, "fmm_kernel", 4, bufs)


@needs_clangxx
def test_f32_ragged_scale_kernel_shim_matches_interp():
    # Masked semantics with a float buffer: the merge keeps unwritten
    # tail elements bit-identical (the f32-representable boundary rule).
    vals = [_f32r((i + 1) * 0.3) for i in range(10)]
    _differential(_FSCALE, "fscale_kernel", 3, {"v": vals})


# f16 kernels (docs/gpu_tiles.md Stage 1f): COMPUTE-ONLY — buffers stay
# float, f16 tiles arise via Tile.to_f16 inside the kernel and convert
# back through Tile.to_f32 before storing.  The C++ shim maps `half` to
# _Float16 and stays the bit-exact leg.

def _f16r(x) -> float:
    """Round a value to the f16-representable double (IEEE binary16,
    round-to-nearest-even — Python's 'e' struct format)."""
    import struct
    return struct.unpack("e", struct.pack("e", float(x)))[0]


_HMM = """
fn hmm_kernel(pid: int, a: Vec, b: Vec, c: Vec) -> () {
    let ti = pid / 2;
    let tj = pid % 2;
    let mut k = 0;
    let mut r = Tile.to_f16(Tile.filled(2, 2, 0));
    while k < 2 {
        let ta = Tile.to_f16(Tile.to_f32(
            Tile.load_rows(a, (ti * 2) * 4 + k * 2, 4, 2, 2, 0.0)));
        let tb = Tile.to_f16(Tile.to_f32(
            Tile.load_rows(b, (k * 2) * 4 + tj * 2, 4, 2, 2, 0.0)));
        r = Tile.add(r, Tile.dot(ta, tb));
        k = k + 1
    };
    Tile.store_rows(c, (ti * 2) * 4 + tj * 2, 4, Tile.to_f32(r));
    ()
}
"""

_HSCALE = """
fn hscale_kernel(pid: int, v: Vec) -> () {
    let t = Tile.to_f16(Tile.to_f32(Tile.load_or(v, pid * 4, 1, 4, 0.0)));
    Tile.store_clipped(v, pid * 4, Tile.to_f32(Tile.scale(t, 0.3)));
    ()
}
"""


@needs_clangxx
def test_f16_matmul_kernel_shim_matches_interp():
    # B is the 4x4 identity, so beyond the differential the shim must
    # reproduce C == round_f16(A) exactly (multiplying by 1.0 and adding
    # 0.0 are exact in f16; the store widens exactly to f32).
    a = [_f32r(i * 0.1) for i in range(16)]
    bufs = {
        "a": a,
        "b": [1.0 if i % 5 == 0 else 0.0 for i in range(16)],
        "c": [0.0] * 16,
    }
    _differential(_HMM, "hmm_kernel", 4, bufs)
    src = _driver(_HMM, "hmm_kernel", 4, bufs)
    k = emit_msl_kernel(src, "hmm_kernel")
    assert k.buf_types == {"a": "float", "b": "float", "c": "float"}
    assert k.uses_half
    assert _run_shim(k, 4, bufs) == [_f16r(x) for x in a]


@needs_clangxx
def test_f16_matmul_ground_truth_independent():
    # The differential alone cannot catch an idiom both engines share:
    # pin the f16 matmul against a product computed HERE, in Python, with
    # explicit per-op 'e' rounding and the kernel's BLOCKED accumulation
    # (per-block dot — round product, then accumulation — combined
    # through Tile.add, each step rounding once to f16).
    import random
    rng = random.Random(11)
    a = [_f32r(rng.uniform(-1, 1)) for _ in range(16)]
    b = [_f32r(rng.uniform(-1, 1)) for _ in range(16)]
    bufs = {"a": a, "b": b, "c": [0.0] * 16}
    ah = [_f16r(x) for x in a]   # Tile.to_f16 of the loaded f32 tiles
    bh = [_f16r(x) for x in b]
    expect = [0.0] * 16
    for i in range(4):
        for j in range(4):
            r = 0.0
            for kb in range(2):
                acc = 0.0
                for kk in range(2):  # pinned order: k ascending
                    gk = kb * 2 + kk
                    acc = _f16r(acc + _f16r(ah[i * 4 + gk] *
                                            bh[gk * 4 + j]))
                r = _f16r(r + acc)   # Tile.add rounds once per element
            expect[i * 4 + j] = r    # Tile.to_f32 widens exactly
    src = _driver(_HMM, "hmm_kernel", 4, bufs)
    k = emit_msl_kernel(src, "hmm_kernel")
    assert _run_shim(k, 4, bufs) == expect
    _differential(_HMM, "hmm_kernel", 4, bufs)


@needs_clangxx
def test_f16_ragged_scale_kernel_shim_matches_interp():
    # Masked semantics through an f16 round-trip: the factor 0.3 rounds
    # once to f16, each product rounds once, the store widens exactly,
    # and the merge keeps unwritten tail elements bit-identical.
    vals = [_f32r((i + 1) * 0.7) for i in range(10)]
    _differential(_HSCALE, "hscale_kernel", 3, {"v": vals})


def test_f16_shim_typedef_present_only_when_used():
    # The _Float16 typedef (and its loud #error guard) appears exactly
    # when the kernel declares half values — an int kernel's shim must
    # stay compilable on toolchains without _Float16.
    hk = emit_msl_kernel(_driver(_HSCALE, "hscale_kernel", 1,
                                 {"v": [0.0] * 4}), "hscale_kernel")
    assert hk.uses_half
    assert "typedef _Float16 half;" in hk.cpp_wrapper()
    assert "__FLT16_MANT_DIG__" in hk.cpp_wrapper()
    ik = emit_msl_kernel(_driver(_MM, "mm_kernel", 4, {
        "a": [0] * 16, "b": [0] * 16, "c": [0] * 16}), "mm_kernel")
    assert not ik.uses_half
    assert "_Float16" not in ik.cpp_wrapper()


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
    # f32 subset edges (Stage 1d).  The mixed-kind cases use COMPUTED
    # values so the static tile shape checker cannot see them — the
    # statically visible versions are already TypeCheckErrors before the
    # emitter runs (pinned in test_tiles.py).
    ("fn k(pid: int, v: Vec) -> () {"
     " let t = Tile.to_f32(Tile.filled(1, 2, 0));"
     " Tile.store_clipped(v, 0, Tile.to_f64(t)); () }",
     "Metal has no f64"),
    ("fn k(pid: int, v: Vec) -> () {"
     " let t = Tile.load_or(v, 0, 1, 2, 0.0);"
     " Tile.store_clipped(v, 0, t); () }",
     "wrap the load directly"),
    ("fn k(pid: int, v: Vec) -> () {"
     " Tile.store_clipped(v, 0, Tile.filled(1, 2, 1.5)); () }",
     "Tile.to_f32(Tile.filled"),
    ("fn k(pid: int, v: Vec) -> () {"
     " let x = 1.5 + 2.5;"
     " Tile.store_clipped(v, 0, Tile.filled(1, 1, 1)); () }",
     "float scalar arithmetic"),
    ("fn k(pid: int, v: Vec) -> () {"
     " let a = Tile.to_f32(Tile.filled(1, 2, 0));"
     " let b = Tile.filled(1, 2, pid * 0);"
     " Tile.store_clipped(v, 0, Tile.add(a, b)); () }",
     "element kinds differ"),
    ("fn k(pid: int, v: Vec) -> () {"
     " let a = Tile.to_f32(Tile.filled(1, 2, 0));"
     " Tile.store_clipped(v, 0, Tile.scale(a, pid)); () }",
     "Tile.scale"),
    # f16 subset edges (Stage 1f): compute-only — stores need to_f32
    # first; f16 mixes with nothing; scale factors are float literals.
    ("fn k(pid: int, v: Vec) -> () {"
     " let t = Tile.to_f16(Tile.filled(1, 2, 0));"
     " Tile.store_clipped(v, 0, t); () }",
     "compute-only"),
    ("fn k(pid: int, v: Vec) -> () {"
     " let a = Tile.to_f16(Tile.filled(1, 2, 0));"
     " let b = Tile.filled(1, 2, pid * 0);"
     " Tile.store_clipped(v, 0, Tile.to_f32(Tile.add(a, b))); () }",
     "element kinds differ"),
    ("fn k(pid: int, v: Vec) -> () {"
     " let a = Tile.to_f16(Tile.filled(1, 2, 0));"
     " Tile.store_clipped(v, 0, Tile.to_f32(Tile.scale(a, pid))); () }",
     "f16 tiles scale by float literals"),
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


def test_mlx_harness_float_structure():
    a = [_f32r(i * 0.5) for i in range(8)]
    bufs = {"a": a, "out": [0.0] * 8}
    src = _driver(
        "fn fs2_kernel(pid: int, a: Vec, out: Vec) -> () {\n"
        "    let t = Tile.to_f32(Tile.load_or(a, pid * 4, 1, 4, 0.0));\n"
        "    Tile.store_clipped(out, pid * 4, Tile.scale(t, 2.0));\n"
        "    ()\n}",
        "fs2_kernel", 2, bufs)
    k = emit_msl_kernel(src, "fs2_kernel")
    assert k.buf_types == {"a": "float", "out": "float"}
    h = k.mlx_harness(2, bufs, {"out": [x * 2 for x in a]})
    assert "mx.array(BUF_a, dtype=mx.float32)" in h
    assert "output_dtypes += [mx.float32, mx.int64]" in h
    assert "FLOAT_TOL" in h
    # int masks stay exact; float outputs compare with the tolerance
    assert "abs(m - e) <= FLOAT_TOL" in h


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


# ---------------------------------------------------------------------------
# Per-simdgroup lowering (docs/simdgroup_plan.md, Option A): one instance is
# one 32-lane simdgroup and 8x8 f32/f16 dots take the matrix units.  The
# shim emulates the collectives in the pinned rounding order, so it stays
# the bit-exact leg; the device leg keeps its float tolerance.
# ---------------------------------------------------------------------------

_FMM8 = """
fn fmm8_kernel(pid: int, a: Vec, b: Vec, c: Vec) -> () {
    let ti = pid / 2;
    let tj = pid % 2;
    let mut k = 0;
    let mut r = Tile.to_f32(Tile.filled(8, 8, 0));
    while k < 2 {
        let ta = Tile.to_f32(
            Tile.load_rows(a, (ti * 8) * 16 + k * 8, 16, 8, 8, 0.0));
        let tb = Tile.to_f32(
            Tile.load_rows(b, (k * 8) * 16 + tj * 8, 16, 8, 8, 0.0));
        r = Tile.add(r, Tile.dot(ta, tb));
        k = k + 1
    };
    Tile.store_rows(c, (ti * 8) * 16 + tj * 8, 16, r);
    ()
}
"""

_HMM8 = """
fn hmm8_kernel(pid: int, a: Vec, b: Vec, c: Vec) -> () {
    let ti = pid / 2;
    let tj = pid % 2;
    let mut k = 0;
    let mut r = Tile.to_f16(Tile.filled(8, 8, 0));
    while k < 2 {
        let ta = Tile.to_f16(Tile.to_f32(
            Tile.load_rows(a, (ti * 8) * 16 + k * 8, 16, 8, 8, 0.0)));
        let tb = Tile.to_f16(Tile.to_f32(
            Tile.load_rows(b, (k * 8) * 16 + tj * 8, 16, 8, 8, 0.0)));
        r = Tile.add(r, Tile.dot(ta, tb));
        k = k + 1
    };
    Tile.store_rows(c, (ti * 8) * 16 + tj * 8, 16, Tile.to_f32(r));
    ()
}
"""


def _random_16x16(seed: int):
    import random
    rng = random.Random(seed)
    a = [_f32r(rng.uniform(-1, 1)) for _ in range(256)]
    b = [_f32r(rng.uniform(-1, 1)) for _ in range(256)]
    return {"a": a, "b": b, "c": [0.0] * 256}


def _blocked_16x16(a, b, rnd):
    """The kernels' blocked accumulation, computed here with explicit
    per-op rounding: two 8-wide k-blocks, each a dot (round the product,
    then the accumulate, k ascending), combined through Tile.add."""
    expect = [0.0] * 256
    for i in range(16):
        for j in range(16):
            r = 0.0
            for kb in range(2):
                acc = 0.0
                for kk in range(8):
                    gk = kb * 8 + kk
                    acc = rnd(acc + rnd(a[i * 16 + gk] * b[gk * 16 + j]))
                r = rnd(r + acc)
            expect[i * 16 + j] = r
    return expect


def test_simdgroup_lowering_is_chosen_for_8x8_dots_only():
    bufs = _random_16x16(1)
    k8 = emit_msl_kernel(_driver(_FMM8, "fmm8_kernel", 4, bufs), "fmm8_kernel")
    assert k8.simdgroup
    small = {"a": [_f32r(i * 0.1) for i in range(16)],
             "b": [1.0] * 16, "c": [0.0] * 16}
    k2 = emit_msl_kernel(_driver(_FMM, "fmm_kernel", 4, small), "fmm_kernel")
    assert not k2.simdgroup           # a 2x2 dot has no matrix-unit form
    ki = emit_msl_kernel(_driver(_MM, "mm_kernel", 4, {
        "a": [0] * 16, "b": [0] * 16, "c": [0] * 16}), "mm_kernel")
    assert not ki.simdgroup           # int tiles never do
    # forcing either way is honored
    assert emit_msl_kernel(_driver(_FMM8, "fmm8_kernel", 4, bufs),
                           "fmm8_kernel", simdgroup=False).simdgroup is False
    assert emit_msl_kernel(_driver(_FMM, "fmm_kernel", 4, small),
                           "fmm_kernel", simdgroup=True).simdgroup is True


def test_simdgroup_body_structure():
    bufs = _random_16x16(2)
    k = emit_msl_kernel(_driver(_FMM8, "fmm8_kernel", 4, bufs), "fmm8_kernel")
    body = k.body
    # pid is the simdgroup index; lanes are replicated except where guarded
    assert "(long)(thread_position_in_grid.x / 32)" in body
    assert "const uint __lane = thread_position_in_grid.x % 32;" in body
    assert "threadgroup float " in body
    assert "if (__lane == 0) {" in body
    assert "metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);" in body
    # the collective for the 8x8 dot
    assert "metal::simdgroup_float8x8 __A, __B;" in body
    assert "metal::simdgroup_load(__A," in body
    assert "metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f)" in body
    assert "metal::simdgroup_multiply_accumulate(__C, __A, __B, __C);" in body
    assert "metal::simdgroup_store(__C," in body
    # no per-thread dot loop remains for it
    assert "__acc +=" not in body
    # grid arithmetic: 32 threads per instance, one simdgroup per group
    assert k.grid(4) == (128, 32)
    # the shim carries the emulation exactly when the lowering needs it
    assert "namespace metal {" in k.cpp_wrapper()
    assert "(unsigned)pid * 32u" in k.cpp_wrapper()
    kt = emit_msl_kernel(_driver(_FMM8, "fmm8_kernel", 4, bufs), "fmm8_kernel",
                         simdgroup=False)
    assert "namespace metal" not in kt.cpp_wrapper()
    assert kt.grid(4) == (4, 4)


def test_simdgroup_harness_dispatches_32_threads_per_instance():
    bufs = _random_16x16(3)
    src = _driver(_FMM8, "fmm8_kernel", 4, bufs)
    k = emit_msl_kernel(src, "fmm8_kernel")
    h = k.mlx_harness(4, bufs, {"c": [0.0] * 256})
    assert "GRID_X = 128" in h and "THREADGROUP_X = 32" in h
    assert "grid=(GRID_X, 1, 1)" in h
    assert "threadgroup=(THREADGROUP_X, 1, 1)" in h
    assert "per-simdgroup" in h


@needs_clangxx
def test_f32_8x8_matmul_simdgroup_shim_matches_interp_and_ground_truth():
    bufs = _random_16x16(7)
    _differential(_FMM8, "fmm8_kernel", 4, bufs)   # auto: per-simdgroup
    src = _driver(_FMM8, "fmm8_kernel", 4, bufs)
    k = emit_msl_kernel(src, "fmm8_kernel")
    assert k.simdgroup
    assert _run_shim(k, 4, bufs) == _blocked_16x16(bufs["a"], bufs["b"], _f32r)
    # both lowerings of the same kernel agree bit for bit on the shim
    kt = emit_msl_kernel(src, "fmm8_kernel", simdgroup=False)
    assert _run_shim(kt, 4, bufs) == _run_shim(k, 4, bufs)


@needs_clangxx
def test_f16_8x8_matmul_simdgroup_shim_matches_interp_and_ground_truth():
    bufs = _random_16x16(11)
    _differential(_HMM8, "hmm8_kernel", 4, bufs)
    src = _driver(_HMM8, "hmm8_kernel", 4, bufs)
    k = emit_msl_kernel(src, "hmm8_kernel")
    assert k.simdgroup and k.uses_half
    assert "metal::simdgroup_half8x8" in k.body
    assert "typedef simdgroup_matrix8x8<half> simdgroup_half8x8;" in k.cpp_wrapper()
    ah = [_f16r(x) for x in bufs["a"]]
    bh = [_f16r(x) for x in bufs["b"]]
    assert _run_shim(k, 4, bufs) == _blocked_16x16(ah, bh, _f16r)


@needs_clangxx
@pytest.mark.parametrize("kernel_src,kernel,bufs", [
    (_MM, "mm_kernel", {"a": list(range(16)),
                        "b": [1 if i % 5 == 0 else 0 for i in range(16)],
                        "c": [0] * 16}),
    (_TRANSPOSE, "tr_kernel", {"a": list(range(12)), "out": [0] * 12}),
    (_RAGGED, "dbl_kernel", {"v": [i + 1 for i in range(10)]}),
    (_FMM, "fmm_kernel", {"a": [_f32r(i * 0.1) for i in range(16)],
                          "b": [1.0 if i % 5 == 0 else 0.0 for i in range(16)],
                          "c": [0.0] * 16}),
    (_HSCALE, "hscale_kernel", {"v": [_f32r((i + 1) * 0.7) for i in range(10)]}),
])
def test_forced_simdgroup_lowering_matches_interp_for_every_op(kernel_src, kernel, bufs):
    # Every op has a per-simdgroup form (lane 0 behind a barrier, sums
    # replicated); forcing the mode on the whole kernel corpus must keep
    # the interpreter parity, including a Tile.sum steering a scalar.
    src = _driver(kernel_src, kernel, 4 if "c" in bufs else 3 if "v" in bufs else 2, bufs)
    grid = 4 if "c" in bufs else 3 if "v" in bufs else 2
    _res, out = interp_run(src)
    k = emit_msl_kernel(src, kernel, simdgroup=True)
    assert k.simdgroup
    toks = out.split()
    vals, pos = {}, 0
    for name in bufs:
        conv = float if k.buf_types[name] == "float" else int
        n = len(bufs[name])
        vals[name] = [conv(x) for x in toks[pos:pos + n]]
        pos += n
    expect = []
    for name in k.out_bufs:
        expect += vals[name]
    assert _run_shim(k, grid, bufs) == expect
