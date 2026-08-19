"""Metaxu showcase suite: speed against C, plus the language's own tricks.

    uv run python benchmarks/suite/run_suite.py            # full (minutes)
    uv run python benchmarks/suite/run_suite.py --rounds 5 # quicker

Four speed benchmarks (fib, nsieve, mandelbrot, orbit) are raced against
hand-written C twins compiled by the same clang at the same -O2; output
equality between the two INDEPENDENT implementations is asserted before
any timing (a wrong answer fast is not a result). par_sum reports 4-thread
scaling implicitly (its binary runs the serial and parallel reductions
back to back and self-checks); pipeline prices effects-as-iterators
against the hand-written loop inside one binary the same way.

Methodology (inherited from benchmarks/contention/run_bench.py, where
both controls were earned the hard way): every object and binary is
built with -falign-functions=64, and per-round run order rotates.
Numbers are medians over the rounds; both distributions' minima are
shown so overlap is visible.
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

RACES = ["fib", "nsieve", "mandelbrot", "orbit", "par_sum"]
SOLO = ["pipeline"]  # metaxu-only: the comparison lives inside the program


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


def outputs_equal(a: tuple[int, str], b: tuple[int, str]) -> bool:
    """Line-wise equality; float lines compare by VALUE (C's %.17g and
    Metaxu's shortest-round-trip repr print the same double differently —
    the doubles themselves must be bit-equal)."""
    if a[0] != b[0]:
        return False
    la, lb = a[1].splitlines(), b[1].splitlines()
    if len(la) != len(lb):
        return False
    for x, y in zip(la, lb):
        if x == y:
            continue
        try:
            if float(x) == float(y):   # exact: same double, different text
                continue
        except ValueError:
            pass
        return False
    return True


def timeit(path: Path) -> float:
    t0 = time.perf_counter()
    subprocess.run([str(path)], stdout=subprocess.DEVNULL, check=True)
    return (time.perf_counter() - t0) * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=11)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="mx_suite_") as td:
        work = Path(td)
        rt = build_runtime(work)
        rows = []
        for name in RACES:
            mx = build_metaxu(name, rt, work)
            cc = build_c(name, work)
            code_m, out_m = run_out(mx)
            code_c, out_c = run_out(cc)
            if not outputs_equal((code_m, out_m), (code_c, out_c)):
                print(f"!! {name}: OUTPUT DIVERGENCE metaxu vs C\n"
                      f"   metaxu ({code_m}): {out_m!r}\n"
                      f"   c      ({code_c}): {out_c!r}")
                return 1
            tm: list[float] = []
            tc: list[float] = []
            timeit(mx), timeit(cc)  # warmup
            for r in range(args.rounds):
                pair = [(tm, mx), (tc, cc)] if r % 2 == 0 else [(tc, cc), (tm, mx)]
                for acc, b in pair:
                    acc.append(timeit(b))
            rows.append((name, statistics.median(tm), min(tm),
                         statistics.median(tc), min(tc), out_m))
        solos = []
        for name in SOLO:
            mx = build_metaxu(name, rt, work)
            code, out = run_out(mx)
            if code != 0:
                print(f"!! {name}: self-check failed ({code}): {out!r}")
                return 1
            ts = []
            timeit(mx)
            for _ in range(args.rounds):
                ts.append(timeit(mx))
            solos.append((name, statistics.median(ts), min(ts), out))

    print(f"\n== Metaxu vs C  (medians over {args.rounds} alternating rounds, "
          f"ms; outputs verified equal first)")
    print(f"{'benchmark':12} {'metaxu':>9} {'min':>8} {'C':>9} {'min':>8} "
          f"{'ratio':>7}")
    for name, m, mlo, c, clo, _out in rows:
        print(f"{name:12} {m:9.1f} {mlo:8.1f} {c:9.1f} {clo:8.1f} "
              f"{m / c:6.2f}x")
    print(f"\n== Metaxu-only exhibits")
    for name, med, lo, out in solos:
        print(f"{name:12} {med:9.1f} {lo:8.1f}   output={out.strip()!r}")
    print("\npar_sum note: its binary runs the SAME reduction serially and "
          "on 4 threads\nand self-checks equality; the wall-clock above "
          "contains both, so the 4-thread\nspeedup shows up as the whole "
          "binary approaching (serial + serial/4) time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
