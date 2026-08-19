"""Four-way contention-guard benchmark (docs/contention_as_permission.md).

Compares runtime variants on the two paths where they can differ:
a 30M-iteration Vec-mutation loop and 1M lock/unlock pairs.

    uv run python benchmarks/contention/run_bench.py

Variants built from this tree:
  current  - the shipped permit guard (design B)
  noguard  - current source with the vec guard compiled out
             (layout control: must match `current` on locks and the
             cheaper variants on mutation, or the harness is broken)
  freeze   - benchmarks/contention/freeze-variant.patch applied
             (design C: contended -> writes always refused, no TLS).
             The patch applying cleanly is part of the check: if the
             guard code has changed, the patch fails loudly and this
             file needs updating rather than silently measuring the
             wrong thing.

METHODOLOGY — both of these bit real measurements before being fixed
here; do not remove them:
  * every object is compiled with -falign-functions=64: unaligned
    builds swing +/-30% on linker code placement alone, which dwarfs
    the guard being measured;
  * run order rotates every round: a fixed order penalizes whichever
    binary runs first on cold caches (this artifact once produced a
    NEGATIVE cost for added code).
"""
from __future__ import annotations

import shutil
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
CFLAGS = ["-std=c11", "-O2", "-g", "-fPIC", "-Wall", "-Wextra",
          "-pthread", "-falign-functions=64"]
ROUNDS = 24


def build_variant(name: str, src_dir: Path, out: Path) -> list[str]:
    objs = []
    for f in RT_FILES:
        o = out / f"{f}_{name}.o"
        subprocess.run(["clang", *CFLAGS, "-c", str(src_dir / f"{f}.c"),
                        "-o", str(o), f"-I{src_dir}"], check=True,
                       capture_output=True)
        objs.append(str(o))
    return objs


def make_sources(work: Path) -> dict[str, Path]:
    dirs = {}
    for name in ("current", "noguard", "freeze"):
        d = work / name
        d.mkdir()
        for f in NATIVE.glob("*.c"):
            shutil.copy(f, d)
        for f in NATIVE.glob("*.h"):
            shutil.copy(f, d)
        dirs[name] = d
    # noguard: compile the vec guard out (layout control)
    rt = dirs["noguard"] / "metaxu_rt.c"
    s = rt.read_text()
    guarded = ("atomic_load_explicit(&v->contended, memory_order_relaxed)\n"
               "            && mx__tls_write_permit == 0, 0)")
    if guarded not in s:
        raise SystemExit("noguard: guard shape changed; update run_bench.py")
    rt.write_text(s.replace(guarded, "0, 0)", 1))
    # freeze: apply the checked-in patch (paths are a/<file> b/<file>, so
    # -p1 lands on basenames in the flat variant dir). A failed apply is
    # the SIGNAL: the guard changed and the patch needs regenerating.
    proc = subprocess.run(
        ["patch", "-p1", "-d", str(dirs["freeze"]), "--no-backup-if-mismatch",
         "-i", str(HERE / "freeze-variant.patch")],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"freeze-variant.patch no longer applies — the guard changed; "
            f"regenerate the patch:\n{proc.stderr or proc.stdout}")
    return dirs


def emit_ir(work: Path) -> dict[str, Path]:
    from metaxu.compiler.pipeline import emit_llvm_from_source
    from metaxu.compiler.llvm_run import compile_and_run
    irs = {}
    for prog in ("bench_mut", "bench_lock"):
        src = (HERE / f"{prog}.mx").read_text()
        ir = emit_llvm_from_source(src, file_path=str(HERE / f"{prog}.mx"))
        assert "placeholder -- unsupported" not in ir, prog
        # compile_and_run synthesizes the @main driver; keep its combined .ll
        wd = work / f"wd_{prog}"
        wd.mkdir()
        compile_and_run(ir, "main", workdir=str(wd))
        irs[prog] = wd / "prog.ll"
    return irs


def timeit(path: Path) -> float:
    t0 = time.perf_counter()
    subprocess.run([str(path)], stdout=subprocess.DEVNULL, check=True)
    return (time.perf_counter() - t0) * 1000


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mx_contention_bench_") as td:
        work = Path(td)
        dirs = make_sources(work)
        objs = {n: build_variant(n, d, work) for n, d in dirs.items()}
        irs = emit_ir(work)
        bins: dict[tuple[str, str], Path] = {}
        for prog, ll in irs.items():
            for name, os_ in objs.items():
                b = work / f"{prog}_{name}"
                subprocess.run(["clang", "-O2", "-falign-functions=64",
                                str(ll), *os_, "-pthread", "-lm",
                                "-o", str(b)], check=True, capture_output=True)
                bins[(prog, name)] = b
        variants = ["current", "noguard", "freeze"]
        for prog in ("bench_mut", "bench_lock"):
            times: dict[str, list[float]] = {v: [] for v in variants}
            for v in variants:
                timeit(bins[(prog, v)])  # warmup
            for r in range(ROUNDS):
                order = variants[r % len(variants):] + variants[:r % len(variants)]
                for v in order:
                    times[v].append(timeit(bins[(prog, v)]))
            base = statistics.median(times["noguard"])
            print(f"== {prog} (median/min ms, {ROUNDS} rotated rounds)")
            for v in variants:
                med = statistics.median(times[v])
                print(f"  {v:8} {med:8.1f} {min(times[v]):8.1f}"
                      f"   {100 * (med - base) / base:+6.2f}% vs noguard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
