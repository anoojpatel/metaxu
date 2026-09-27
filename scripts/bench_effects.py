"""Effects microbenchmark: what one perform costs natively.

Three programs, each doing N events, compiled through the LLVM backend
and timed as whole processes (best of REPEATS runs):

  baseline   a loop calling a plain function          -- the floor
  counter    the same loop performing `Counter.tick`  -- one tail-resuming
             handler case per event (Class A: runs on the current stack)
  abortive   a handler whose arm tail-resumes until a bound and then
             returns without resuming (take-shaped: Class A as well)
  general    an arm that does work AFTER the resume (`resume(v) + 0`):
             the coroutine path, one context switch pair per event
  pipeline   std.stream sum(map(filter(iota(N)))) -- three nested scopes

Per-event cost is (t - t_baseline) / N.  Usage:

    uv run python scripts/bench_effects.py [--n 2000000] [--repeats 3] [--interp]

--interp also times the counter program on the MIR interpreter at n/100
(it is two to three orders of magnitude slower; the number is for scale).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaxu.compiler.llvm_run import compile_to_binary  # noqa: E402
from metaxu.compiler.pipeline import emit_llvm_from_source  # noqa: E402

BASELINE = """
fn tick(n: int) -> int { n + 1 }
fn main() -> int {
    let @mut acc = 0;
    let @mut i = 0;
    while i < {N} { acc = tick(acc); i = i + 1 };
    print(acc);
    0
}
"""

COUNTER = """
effect Counter { tick(n: int) -> int }
fn loop_it() -> int {
    let @mut acc = 0;
    let @mut i = 0;
    while i < {N} { acc = perform Counter.tick(acc); i = i + 1 };
    acc
}
fn main() -> int {
    let r = handle Counter with {
        tick(n) -> resume(n + 1)
    } in { loop_it() };
    print(r);
    0
}
"""

ABORTIVE = """
effect Counter { tick(n: int) -> int }
fn loop_it() -> int {
    let @mut acc = 0;
    while true { acc = perform Counter.tick(acc) };
    acc
}
fn main() -> int {
    let r = handle Counter with {
        tick(n) -> if n < {N} { resume(n + 1) } else { n }
    } in { loop_it() };
    print(r);
    0
}
"""

GENERAL = """
effect Counter { tick(n: int) -> int }
fn loop_it() -> int {
    let @mut acc = 0;
    let @mut i = 0;
    while i < {N} { acc = perform Counter.tick(acc); i = i + 1 };
    acc
}
fn main() -> int {
    let r = handle Counter with {
        tick(n) -> { let v = resume(n + 1); v + 0 }
    } in { loop_it() };
    print(r);
    0
}
"""

PIPELINE = """
from std.stream import iota, sum, map, filter;
fn main() -> int {
    print(sum(map(filter(iota({N}), fn(x: int) -> x % 3 == 0), fn(x: int) -> x * x)));
    0
}
"""

# (name, source, cap on N): the general arm keeps one pending frame per
# event by semantics (work after the resume is foldr's shape), so it runs
# at a bounded N and its per-event figure is scaled from there.
PROGRAMS = [("baseline", BASELINE, None), ("counter", COUNTER, None),
            ("abortive", ABORTIVE, None), ("general", GENERAL, 10_000),
            ("pipeline", PIPELINE, None)]


def build(name: str, src: str, n: int, workdir: Path) -> tuple[str, bool]:
    llvm = emit_llvm_from_source(src.replace("{N}", str(n)))
    direct = "@mx_handle_direct(" in llvm
    if "; function @mx_main: placeholder" in llvm:
        raise SystemExit(f"{name}: main demoted:\n" + "\n".join(
            l for l in llvm.splitlines() if "reason" in l)[:2000])
    return compile_to_binary(llvm, "main", out_path=str(workdir / f"{name}.bin"),
                             timeout=600), direct


def time_binary(path: str, repeats: int) -> tuple[float, str]:
    best = float("inf")
    out = ""
    for _ in range(repeats):
        t = time.perf_counter()
        proc = subprocess.run([path], capture_output=True, text=True, check=True)
        best = min(best, time.perf_counter() - t)
        out = proc.stdout.strip()
    return best, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--interp", action="store_true")
    ap.add_argument("--no-direct", action="store_true",
                    help="A/B: keep every scope on the coroutine path")
    args = ap.parse_args()
    n = args.n
    if args.no_direct:
        from metaxu.compiler import codegen_llvm
        codegen_llvm._program_direct_cases = lambda funcs: frozenset()
    with tempfile.TemporaryDirectory(prefix="mx_bench_effects_") as d:
        workdir = Path(d)
        rows = []
        base_per_event = 0.0
        for name, src, cap in PROGRAMS:
            m = n if cap is None else min(n, cap)
            binary, direct = build(name, src, m, workdir)
            t, out = time_binary(binary, args.repeats)
            if name == "baseline":
                base_per_event = t / m
            per_event = (t / m - base_per_event) * 1e9
            rows.append((name, m, t, per_event, direct, out))
        print(f"best of {args.repeats} runs; ns/event is over the baseline loop\n")
        print(f"{'program':<10} {'N':>10} {'wall (s)':>9} {'ns/event':>10}  scope")
        for name, m, t, per_event, direct, out in rows:
            scope = "" if name == "baseline" else ("direct" if direct else "coroutine")
            print(f"{name:<10} {m:>10,} {t:>9.3f} {per_event:>10.1f}  {scope}")
        if args.interp:
            from metaxu.compiler.hir import HIRBuilder
            from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
            from metaxu.compiler.mir_interp import MirInterpreter
            from metaxu.compiler.pipeline import build_context_from_source
            m = max(n // 100, 1000)
            ctx = build_context_from_source(COUNTER.replace("{N}", str(m)))
            hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
            interp = MirInterpreter()
            interp.load(lower_hir_to_mir(hir))
            interp.register_builtin("print", lambda *a: None)
            t = time.perf_counter()
            interp.call("main", [])
            dt = time.perf_counter() - t
            print(f"\ninterpreter counter, N = {m:,}: {dt:.2f} s, {dt / m * 1e9:,.0f} ns/event")
    return 0


if __name__ == "__main__":
    sys.exit(main())
