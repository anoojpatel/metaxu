"""Run every example program through the compiler pipeline and report status.

Usage:
    uv run python scripts/run_examples.py [--stage parse|pipeline] [paths...]

With no paths, runs all examples/*.mx and repo-root test_*.mx files.
Exit code is the number of failing files (0 = all pass), so this can gate CI
and merges.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))


def default_targets() -> list[str]:
    return sorted(glob.glob(os.path.join(REPO_ROOT, "examples", "*.mx"))) + sorted(
        glob.glob(os.path.join(REPO_ROOT, "test_*.mx"))
    )


def run_one(path: str, stage: str) -> tuple[bool, str]:
    source = open(path).read()
    try:
        if stage == "parse":
            from metaxu.parser import Parser

            Parser().parse(source, file_path=path)
            return True, "parsed"
        from metaxu.compiler.pipeline import run_pipeline_from_source

        _ast, hir, mir, clif = run_pipeline_from_source(source)
        return True, f"hir={len(hir)}b mir={len(mir)}b clif={len(clif)}b"
    except Exception as exc:  # noqa: BLE001 - report every failure uniformly
        last = traceback.format_exc().strip().splitlines()[-1]
        return False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]} ({last[:80]})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["parse", "pipeline"], default="pipeline")
    ap.add_argument("paths", nargs="*")
    args = ap.parse_args()

    targets = [os.path.abspath(p) for p in args.paths] or default_targets()
    failures = 0
    for path in targets:
        ok, msg = run_one(path, args.stage)
        failures += 0 if ok else 1
        print(f"{'OK  ' if ok else 'FAIL'} {os.path.relpath(path, REPO_ROOT):45} {msg}")
    print(f"\n{len(targets) - failures}/{len(targets)} pass ({args.stage})")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
