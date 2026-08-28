"""Generate a self-checking Metal harness for a metaxu tile kernel.

    uv run python scripts/emit_metal_harness.py kernels.mx mm_kernel \
        --grid 4 --buf a=0,1,2,3 --buf out=0,0,0,0 -o harness_mm.py

Then, on a Mac with mlx installed:  python3 harness_mm.py
(exit 0 = Metal output matches the interpreter reference bit for bit).

The expected outputs are computed HERE by the interpreter — the
semantics reference — running the kernel sequentially over the grid
(std.gpu's run_grid semantics); the harness bakes them in and compares
the Metal results after the written-mask merge (docs/gpu_tiles.md
Stage 1c).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="metaxu file defining the kernel")
    ap.add_argument("kernel", help="kernel function name "
                                   "(fn k(pid: int, bufs: Vec...) -> ())")
    ap.add_argument("--grid", type=int, required=True,
                    help="number of kernel instances")
    ap.add_argument("--buf", action="append", default=[],
                    metavar="NAME=v0,v1,...",
                    help="one per kernel buffer parameter, in order")
    ap.add_argument("-o", "--out", default=None,
                    help="harness path (default: harness_<kernel>.py)")
    args = ap.parse_args()

    from metaxu.compiler.emit_msl import MslError, emit_msl_kernel
    from metaxu.compiler.hir import HIRBuilder
    from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
    from metaxu.compiler.mir_interp import MirInterpreter, MxVec
    from metaxu.compiler.pipeline import (build_context_from_source,
                                          run_pipeline_ctx)

    source = Path(args.source).read_text()

    # Values parse as ints unless written with a decimal point / exponent
    # (float buffers are the f32 increment: 1.5 stays float, 3 stays int).
    def num(x: str):
        try:
            return int(x)
        except ValueError:
            return float(x)

    buffers: dict[str, list] = {}
    for spec in args.buf:
        name, _, csv = spec.partition("=")
        buffers[name] = [num(x) for x in csv.split(",") if x != ""]

    try:
        kern = emit_msl_kernel(source, args.kernel)
    except MslError as e:
        print(f"kernel not emittable: {e}", file=sys.stderr)
        return 1
    # A float-typed buffer holds F32-REPRESENTABLE values everywhere: on
    # the device it IS float32, so the host rounds once at the boundary —
    # otherwise unwritten elements would differ between the f64 CPU
    # reference Vec and the float32 Metal buffer after the mask merge.
    import struct

    def f32r(x) -> float:
        return struct.unpack("f", struct.pack("f", float(x)))[0]

    for b, vals in list(buffers.items()):
        if kern.buf_types.get(b) == "float":
            buffers[b] = [f32r(x) for x in vals]

    missing = [b for b in kern.in_bufs if b not in buffers]
    if missing:
        print(f"missing --buf for kernel buffers: {', '.join(missing)} "
              f"(kernel buffers, in order: {', '.join(kern.in_bufs)})",
              file=sys.stderr)
        return 1

    # Expected outputs: the interpreter runs the kernel sequentially over
    # the grid on copies of the buffers (run_grid reference semantics).
    ctx = build_context_from_source(source, file_path=args.source)
    run_pipeline_ctx(ctx)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    vecs = {b: MxVec(list(buffers[b])) for b in kern.in_bufs}
    for pid in range(args.grid):
        interp.call(args.kernel,
                    [pid] + [vecs[b] for b in kern.in_bufs])
    expected = {b: list(vecs[b].items) for b in kern.out_bufs}

    out_path = Path(args.out or f"harness_{args.kernel}.py")
    out_path.write_text(kern.mlx_harness(args.grid, buffers, expected))
    print(f"wrote {out_path}")
    print(f"  kernel:  {kern.name}  (grid {args.grid})")
    print(f"  buffers: {', '.join(kern.in_bufs)}  "
          f"(written: {', '.join(kern.out_bufs) or 'none'})")
    for b in kern.out_bufs:
        print(f"  expected {b}: {expected[b]}")
    print("run it on a Mac:  python3 " + str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
