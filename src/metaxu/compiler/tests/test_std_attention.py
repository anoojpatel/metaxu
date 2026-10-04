"""std.attention: the FlashAttention-2 kernels over std.tile.

Three layers of evidence, each through parsed source:

* the f32 kernels run on the interpreter (through std.gpu's CPU handler)
  agree with `attention_ref`, the plain f64 definition in the same
  module, and with an independent Python reference, to f32 tolerance;
* the MSL emitter inlines the library and lambdas into one device body
  that the clang++ shim runs BIT-IDENTICALLY to the interpreter, in the
  automatic per-simdgroup lowering (8x8 dots on the matrix units,
  elementwise work lane-strided through MX_EACH) and the per-thread one;
* the `run_metal` handler path reproduces the CPU handler byte for byte.
"""
from __future__ import annotations

import math
import random
import re

import pytest

from metaxu.compiler.emit_msl import emit_msl_kernel
from metaxu.compiler.tests.test_codegen_llvm import (
    assert_native_matches_interp, count_placeholders, interp_run,
    llvm_from_source, needs_clang)
from metaxu.compiler.tests.test_msl_emitter import _f32r, _run_shim, needs_clangxx

N, D = 16, 8
_IMPORT = "from std.attention import attention8, causal_attention8, attention_ref;"


def _inputs(seed: int = 7) -> dict:
    rng = random.Random(seed)
    return {
        "q": [_f32r(rng.uniform(-1, 1)) for _ in range(N * D)],
        "k": [_f32r(rng.uniform(-1, 1)) for _ in range(N * D)],
        "v": [_f32r(rng.uniform(-1, 1)) for _ in range(N * D)],
        "out": [0.0] * (N * D),
        "meta": [N],
    }


def _pushes(bufs: dict) -> str:
    lines = []
    for name, vals in bufs.items():
        lines.append(f"    let @mut {name} = Vec.new();")
        lines += [f"    {name}.push({v!r});" for v in vals]
    return "\n".join(lines)


def _program(kernel: str, bufs: dict, metal: bool = False,
             with_ref: bool = False) -> str:
    """main(): build the buffers, launch `kernel` over N/8 instances
    through std.gpu (the CPU handler, or `run_metal`), print `out`, and
    optionally the f64 reference after it."""
    call = f"perform Gpu.launch({N // 8}, fn(pid: int) -> {kernel}(pid, q, k, v, out, meta))"
    launch = (f"""
    handle Gpu with {{
        launch(n, f) -> {{ run_metal(n, f); resume(()) }}
    }} in {{ {call}; () }};""" if metal else f"\n    {call};")
    ref = ""
    if with_ref:
        causal = "true" if kernel.startswith("causal") else "false"
        ref = f"""
    let r = attention_ref(q, k, v, {N}, {D}, {causal});
    let mut j = 0;
    while j < {N * D} {{ print(r[j]); j = j + 1 }};"""
    return f"""
from std.gpu import Gpu, run_grid, Metal, run_metal;
{_IMPORT}

fn main() -> int {{
{_pushes(bufs)}{launch}
    let mut i = 0;
    while i < {N * D} {{ print(out[i]); i = i + 1 }};{ref}
    0
}}
"""


def _python_ref(bufs: dict, causal: bool) -> list:
    q, k, v = bufs["q"], bufs["k"], bufs["v"]
    scale = 1.0 / math.sqrt(D)
    out = []
    for i in range(N):
        limit = i + 1 if causal else N
        scores = [sum(q[i * D + c] * k[j * D + c] for c in range(D)) * scale
                  for j in range(limit)]
        m = max(scores)
        e = [math.exp(s - m) for s in scores]
        denom = sum(e)
        for c in range(D):
            out.append(sum(e[j] * v[j * D + c] for j in range(limit)) / denom)
    return out


def _run_cpu(kernel: str, bufs: dict, with_ref: bool = False):
    result, out = interp_run(_program(kernel, bufs, with_ref=with_ref))
    assert result == 0
    vals = [float(x) for x in out.split()]
    if with_ref:
        return vals[:N * D], vals[N * D:]
    return vals


@pytest.mark.parametrize("kernel", ["attention8", "causal_attention8"])
def test_kernel_matches_the_f64_reference(kernel):
    bufs = _inputs()
    got, ref = _run_cpu(kernel, bufs, with_ref=True)
    assert len(got) == len(ref) == N * D
    # The kernel accumulates in f32 with per-op rounding; the reference
    # is plain f64.  The inputs are in [-1, 1] and the outputs are convex
    # combinations of v rows, so 1e-5 is loose for f32 and tight against
    # any ordering or masking mistake.
    assert max(abs(a - b) for a, b in zip(got, ref)) < 1e-5
    # And the module's reference itself is the textbook definition.
    py = _python_ref(bufs, causal=kernel.startswith("causal"))
    assert max(abs(a - b) for a, b in zip(ref, py)) < 1e-12
    # Every kernel output is f32-representable: the store is a float32 buffer.
    assert all(_f32r(x) == x for x in got)


def test_causal_mask_changes_only_the_masked_rows():
    bufs = _inputs()
    full = _run_cpu("attention8", bufs)
    causal = _run_cpu("causal_attention8", bufs)
    # Row 0 attends to key 0 only under the mask: it is v[0] exactly
    # (softmax of one score is 1); the full kernel mixes all keys.
    assert causal[:D] == bufs["v"][:D]
    assert full[:D] != bufs["v"][:D]
    # The last row sees every key either way, so the two agree there to
    # f32 noise (the block order of the running maximum differs).
    assert max(abs(a - b) for a, b in zip(full[-D:], causal[-D:])) < 1e-6


def test_kernels_select_the_simdgroup_lowering():
    bufs = _inputs()
    for kernel in ("attention8", "causal_attention8"):
        k = emit_msl_kernel(_program(kernel, bufs), f"std.attention.{kernel}")
        assert k.simdgroup                       # 8x8 f32 dots
        assert "metal::simdgroup_multiply_accumulate" in k.body
        assert "MX_EACH(__i, 64)" in k.body      # lane-strided online softmax
        assert "metal::exp(" in k.body
        assert "std.tile" not in k.body and "$" not in k.body
        assert set(k.in_bufs) == {"q", "k", "v", "out", "meta"}
        assert k.out_bufs == ["out"]
        assert k.buf_types["meta"] != "float"     # the row count buffer is integer
        assert k.buf_types["out"] == "float"


@needs_clangxx
@pytest.mark.parametrize("kernel", ["attention8", "causal_attention8"])
@pytest.mark.parametrize("simdgroup", [None, False, True],
                         ids=["auto", "thread", "simdgroup"])
def test_shim_matches_interp_bit_for_bit(kernel, simdgroup):
    bufs = _inputs()
    src = _program(kernel, bufs)
    expect = _run_cpu(kernel, bufs)
    kw = {} if simdgroup is None else {"simdgroup": simdgroup}
    k = emit_msl_kernel(src, f"std.attention.{kernel}", **kw)
    assert k.simdgroup is (True if simdgroup is None else simdgroup)
    assert _run_shim(k, N // 8, bufs) == expect


@pytest.mark.parametrize("kernel", ["attention8", "causal_attention8"])
def test_kernels_compile_natively(kernel):
    # The CPU-handler program (kernel through std.gpu's sequential grid,
    # plus the f64 reference) compiles natively: the std.tile helpers are
    # inlined at each site with a private narrow lambda (so `exp` serves
    # 8x8 and 8x1 tiles), the 8x8 dots are the C runtime's.  The only
    # placeholders are std.tile functions this program never calls.
    ir = llvm_from_source(_program(kernel, _inputs(), with_ref=True))
    placeholders = re.findall(r"^; function @mx_(\S+): placeholder", ir, re.M)
    assert placeholders and all(p.startswith("std_tile_") for p in placeholders), placeholders
    assert "@mx_main: placeholder" not in ir
    assert re.search(rf"^define \S+ @mx_std_attention_{kernel}\(", ir, re.M)
    assert re.search(r"^define \S+ @mx_std_tile_exp_lambda1_i\d+\(", ir, re.M)
    assert "@mx_std_tile_exp(" not in ir           # inlined away
    assert "narrow lambda:" in ir and "to float" in ir


@needs_clang
@pytest.mark.parametrize("kernel", ["attention8", "causal_attention8"])
def test_native_matches_interp(kernel, tmp_path):
    # Native stdout == interpreter stdout, byte for byte: the f32 kernel's
    # per-op rounding and the f64 reference alike.
    assert_native_matches_interp(_program(kernel, _inputs(seed=3), with_ref=True),
                                 tmp_path)


@needs_clangxx
@pytest.mark.parametrize("kernel", ["attention8", "causal_attention8"])
def test_metal_handler_matches_cpu_handler(kernel, monkeypatch):
    monkeypatch.delenv("METAXU_METAL_LOWERING", raising=False)
    bufs = _inputs(seed=11)
    _res, cpu = interp_run(_program(kernel, bufs, metal=False))
    _res, metal = interp_run(_program(kernel, bufs, metal=True))
    assert metal == cpu
    assert len(metal.splitlines()) == N * D
