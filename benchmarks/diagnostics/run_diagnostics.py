"""Emission diagnostics: eight tiny paired programs, each isolating ONE
code shape the backend emits, raced against a C twin doing the identical
work.  Unlike benchmarks/suite (whole showcase programs), each pair here
answers a single question — "what does THIS emission pattern cost?" — so a
regression or a win in the emitted IR shows up as one moving row instead
of a blended number.

    uv run python benchmarks/diagnostics/run_diagnostics.py
    uv run python benchmarks/diagnostics/run_diagnostics.py --rounds 5

Pairs (ratio = metaxu/C, ~1.00 is parity):
    vecread   unit-stride Vec reads          (sum 5M elements x4)
    vecwrite  unit-stride Vec writes         (fill 5M elements x4)
    vecpush   append-heavy growth            (push 5M x4)
    modloop   pure integer ALU loop          (no memory traffic)
    closure   indirect calls through a lambda (vs C fn pointer)
    enum      construct+match a 3-case enum in a loop
    string    int -> string -> len           (allocation semantics)
    struct    by-value struct rebuild in a loop
    tilemm    8x8-tiled 128^3 int matmul through Gpu.launch (Stage 1
              kernel seam) vs plain C loops — the CPU tile-op cost:
              every op is an opaque runtime call ALLOCATING a fresh
              block (functional semantics).  Measured ~4x C at
              landing — and proven INCIDENTAL, not intrinsic: the MSL
              emitter's lowering of the SAME kernel (tiles as stack
              arrays, zero allocation, ops inlined) runs at ~0.9-1.0x C
              through the C++ shim.  The tile MODEL is at C parity; the
              4x is entirely the reference lowering's allocation-per-op
              + opaque calls.  The payoff layer is Metal + Stage 2
              layouts; the known CPU wins (op fusion, arena/reuse
              allocation, the Vec-fast-path treatment) are recorded in
              docs/gpu_tiles.md rather than chased early.

History: the first run of these diagnostics (2026-08, pre-inlining) put
vecread at 2.57x, vecwrite at 2.86x and vecpush at 1.68x — every Vec
access was an opaque runtime call, which blocks register caching, LICM
and strength reduction around the loop.  That measurement motivated the
inline Vec fast paths in codegen_llvm, landed in three pieces:

  1. hot path inline against the mx_vec header, cold path calls the
     runtime (byte-identical diagnostics)      -> 1.93x / 2.19x / 1.37x
  2. get/set/pop/len misses are ALWAYS failures, so their cold blocks
     became noreturn mx__vec_*_fail terminators with a narrow memory
     contract — without it the maybe-taken call clobbers every header
     load and LICM hoists nothing               -> 1.64x / 1.79x
  3. TBAA tags separating header words from element words (distinct
     allocations), so element stores don't pin the hoisted len/data
     loads; clang then unrolls 4x and batches the bounds checks
                                                -> ~1.5x / ~1.7x / ~1.3x

The residual is the checked-indexing price — null test + bounds check
(+ contended-write guard on mutation) per element against C's raw
pointer arithmetic — not call overhead; closing it further means
eliminating checks (iterator fusion / unsafe), a language question
rather than an emission one.  The other rows were already at parity
(closure BELOW 1.0: pinned closures beat C's opaque function pointers);
string's ~1.45x is allocation semantics, tracked separately.

Methodology (inherited from benchmarks/contention, earned the hard way):
-falign-functions=64 on every object and binary (layout luck otherwise
swings +/-30%), per-round run order rotates, medians over the rounds,
and output equality between the independent implementations is asserted
before any timing.
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))

NATIVE = REPO / "src" / "metaxu" / "runtime" / "native"
RT_FILES = ("metaxu_rt", "metaxu_effects", "metaxu_threads")
ALIGN = "-falign-functions=64"
CFLAGS = ["-std=c11", "-O2", "-g", "-fPIC", "-Wall", "-Wextra",
          "-pthread", ALIGN]

PAIRS = ["vecread", "vecwrite", "vecpush", "modloop", "closure", "enum",
         "string", "struct", "tilemm"]


def build_runtime(out: Path) -> list[str]:
    objs = []
    for f in RT_FILES:
        o = out / f"{f}.o"
        subprocess.run(["clang", *CFLAGS, "-c", str(NATIVE / f"{f}.c"),
                        "-o", str(o), f"-I{NATIVE}"], check=True,
                       capture_output=True)
        objs.append(str(o))
    return objs


def build_metaxu(name: str, rt: list[str], work: Path) -> Path:
    from metaxu.compiler.pipeline import emit_llvm_from_source
    from metaxu.compiler.llvm_run import compile_and_run
    src = (HERE / f"{name}.mx").read_text()
    ir = emit_llvm_from_source(src, file_path=str(HERE / f"{name}.mx"))
    wd = work / f"wd_{name}"
    wd.mkdir()
    compile_and_run(ir, "main", workdir=str(wd))  # synthesizes @main driver
    binary = work / f"mx_{name}"
    subprocess.run(["clang", "-O2", ALIGN, str(wd / "prog.ll"), *rt,
                    "-pthread", "-lm", "-o", str(binary)],
                   check=True, capture_output=True)
    return binary


def build_c(name: str, work: Path) -> Path:
    binary = work / f"c_{name}"
    subprocess.run(["clang", "-O2", ALIGN, str(HERE / "c" / f"{name}.c"),
                    "-pthread", "-lm", "-o", str(binary)],
                   check=True, capture_output=True)
    return binary


def run_out(path: Path) -> tuple[int, str]:
    p = subprocess.run([str(path)], capture_output=True, text=True)
    return p.returncode, p.stdout


def timeit(path: Path) -> float:
    t0 = time.perf_counter()
    subprocess.run([str(path)], stdout=subprocess.DEVNULL, check=True)
    return (time.perf_counter() - t0) * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=9)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="mx_diag_") as td:
        work = Path(td)
        rt = build_runtime(work)
        rows = []
        for name in PAIRS:
            mx = build_metaxu(name, rt, work)
            cc = build_c(name, work)
            om, oc = run_out(mx), run_out(cc)
            if om != oc:
                print(f"!! {name}: OUTPUT DIVERGENCE metaxu vs C\n"
                      f"   metaxu ({om[0]}): {om[1]!r}\n"
                      f"   c      ({oc[0]}): {oc[1]!r}")
                return 1
            tm: list[float] = []
            tc: list[float] = []
            timeit(mx), timeit(cc)  # warmup
            for r in range(args.rounds):
                pair = [(tm, mx), (tc, cc)] if r % 2 == 0 \
                    else [(tc, cc), (tm, mx)]
                for acc, b in pair:
                    acc.append(timeit(b))
            rows.append((name, statistics.median(tm), min(tm),
                         statistics.median(tc), min(tc)))

    print(f"\n== Emission diagnostics  (medians over {args.rounds} "
          "alternating rounds, ms;\n   outputs verified equal first; "
          "ratio = metaxu/C, ~1.00 is parity)")
    print(f"{'pattern':10} {'metaxu':>9} {'min':>8} {'C':>9} {'min':>8} "
          f"{'ratio':>7}")
    for name, m, mlo, c, clo in rows:
        print(f"{name:10} {m:9.1f} {mlo:8.1f} {c:9.1f} {clo:8.1f} "
              f"{m / c:6.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
