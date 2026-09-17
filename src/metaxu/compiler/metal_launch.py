"""The Metal launch engine (docs/gpu_tiles.md — the runtime handler).

Runs one emitted ``MslKernel`` over a grid on the best engine available
and returns the mask-merged output buffers:

  * **mlx** (Apple silicon): the kernel body binds through
    ``mx.fast.metal_kernel`` — a real GPU dispatch.  Metal compiles
    fast-math, so float results carry the documented tolerance there.
  * **clang++ shim** (anywhere, this container included): compiles
    ``MslKernel.cpp_wrapper()`` with ``-ffp-contract=off`` and runs the
    grid sequentially — bit-exact with the CPU reference, which is what
    makes the Metal HANDLER differentially testable with no Metal.

Engine selection is automatic (mlx if importable, else the shim) and a
missing engine is a loud ``MetalLaunchError``, never a silent CPU
fallback — the caller asked for the Metal backend and must be told when
there isn't one.

Buffer contract (the float boundary rule): a ``float``-typed buffer is
float32 on the device, so its host values must already be
F32-REPRESENTABLE; the launch validates and refuses others — silently
rounding here would make the Metal handler disagree with the CPU
handler on unwritten (mask-merged) elements.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import struct
import subprocess
import tempfile
from typing import Dict, List, Optional

from .emit_msl import MslKernel


class MetalLaunchError(Exception):
    """The launch could not run; the message says why."""


def _f32_representable(x: float) -> bool:
    try:
        return struct.unpack("f", struct.pack("f", x))[0] == x
    except (OverflowError, struct.error):
        return False


def validate_buffers(kern: MslKernel,
                     buffers: Dict[str, List]) -> None:
    """Refuse buffer contents the device could not hold faithfully."""
    for b in kern.in_bufs:
        vals = buffers[b]
        if kern.buf_types.get(b) == "float":
            for i, v in enumerate(vals):
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise MetalLaunchError(
                        f"buffer {b!r} is float-typed but element {i} is "
                        f"{type(v).__name__}")
                if isinstance(v, int) or not _f32_representable(v):
                    raise MetalLaunchError(
                        f"buffer {b!r} is float32 on the device, but "
                        f"element {i} ({v!r}) is not f32-representable — "
                        "round host values once at the boundary "
                        "(e.g. through Tile.to_f32) so unwritten elements "
                        "merge bit-identically")
        else:
            for i, v in enumerate(vals):
                if isinstance(v, bool) or not isinstance(v, int):
                    raise MetalLaunchError(
                        f"buffer {b!r} is int-typed but element {i} is "
                        f"{type(v).__name__}")


def available_engine() -> Optional[str]:
    """'mlx', 'shim', or None — what a launch here would run on."""
    try:  # pragma: no cover — no Metal in the dev container
        import mlx.core  # noqa: F401
        return "mlx"
    except ImportError:
        pass
    if shutil.which("clang++"):
        return "shim"
    return None


def run(kern: MslKernel, grid_n: int,
        buffers: Dict[str, List]) -> Dict[str, List]:
    """Run the kernel; return merged outputs keyed by written buffer."""
    missing = [b for b in kern.in_bufs if b not in buffers]
    if missing:
        raise MetalLaunchError(
            f"missing buffers for launch: {', '.join(missing)}")
    validate_buffers(kern, buffers)
    engine = available_engine()
    if engine == "mlx":  # pragma: no cover — no Metal in the container
        return _run_mlx(kern, grid_n, buffers)
    if engine == "shim":
        return _run_shim(kern, grid_n, buffers)
    raise MetalLaunchError(
        "no Metal engine available: mlx is not installed and clang++ is "
        "not on PATH (the launch never silently falls back to the CPU "
        "reference — install one, or run under the default Gpu handler)")


# -- the clang++ shim engine -------------------------------------------------

# Compiled shims cached per kernel body for the life of the process
# (launching in a loop must not recompile every call).
_shim_cache: Dict[str, str] = {}
_shim_dir: Optional[str] = None


def _shim_exe(kern: MslKernel) -> str:
    global _shim_dir
    key = hashlib.sha256(kern.cpp_wrapper().encode()).hexdigest()[:24]
    exe = _shim_cache.get(key)
    if exe is not None and os.path.exists(exe):
        return exe
    if _shim_dir is None:
        _shim_dir = tempfile.mkdtemp(prefix="mx_metal_shim_")
    cpp = os.path.join(_shim_dir, key + ".cpp")
    exe = os.path.join(_shim_dir, key)
    with open(cpp, "w") as fh:
        fh.write(kern.cpp_wrapper())
    r = subprocess.run(
        ["clang++", "-O2", "-std=c++17", "-ffp-contract=off", cpp,
         "-o", exe],
        capture_output=True, text=True)
    if r.returncode != 0:  # an emitter bug, not a user error — still loud
        raise MetalLaunchError(
            f"generated shim failed to compile: {r.stderr.strip()}")
    _shim_cache[key] = exe
    return exe


def _run_shim(kern: MslKernel, grid_n: int,
              buffers: Dict[str, List]) -> Dict[str, List]:
    exe = _shim_exe(kern)
    feed = [str(len(kern.in_bufs))]
    for b in kern.in_bufs:
        vals = buffers[b]
        feed.append(str(len(vals)))
        feed += [repr(v) for v in vals]
    p = subprocess.run([exe, str(grid_n)], input=" ".join(feed),
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise MetalLaunchError(
            f"shim run failed (exit {p.returncode}): {p.stderr.strip()}")
    toks = p.stdout.split()
    out: Dict[str, List] = {}
    pos = 0
    for b in kern.out_bufs:
        conv = float if kern.buf_types.get(b) == "float" else int
        n = len(buffers[b])
        out[b] = [conv(x) for x in toks[pos:pos + n]]
        pos += n
    return out


# -- the mlx engine (Apple silicon) -----------------------------------------

def _run_mlx(kern: MslKernel, grid_n: int,
             buffers: Dict[str, List]) -> Dict[str, List]:  # pragma: no cover
    import mlx.core as mx

    def dt(b: str):
        return mx.float32 if kern.buf_types.get(b) == "float" else mx.int64

    inputs = [mx.array(buffers[b], dtype=dt(b)) for b in kern.in_bufs]
    inputs.append(mx.array([len(buffers[b]) for b in kern.in_bufs],
                           dtype=mx.int64))
    output_names: List[str] = []
    output_shapes: List[tuple] = []
    output_dtypes: List = []
    for b in kern.out_bufs:
        output_names += [f"{b}_out", f"{b}_wm"]
        output_shapes += [(len(buffers[b]),), (len(buffers[b]),)]
        output_dtypes += [dt(b), mx.int64]
    kernel = mx.fast.metal_kernel(
        name=kern.name,
        input_names=[f"{b}_in" for b in kern.in_bufs] + ["lens"],
        output_names=output_names,
        source=kern.body,
    )
    # per-simdgroup lowering: 32 threads per instance, one simdgroup per
    # threadgroup (MslKernel.grid does the arithmetic for both modes)
    gx, tg = kern.grid(grid_n)
    outs = kernel(
        inputs=inputs,
        grid=(gx, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        init_value=0,
    )
    merged: Dict[str, List] = {}
    for i, b in enumerate(kern.out_bufs):
        out = outs[2 * i].tolist()
        wm = outs[2 * i + 1].tolist()
        merged[b] = [o if m else v
                     for o, m, v in zip(out, wm, buffers[b])]
    return merged
