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


# Negative fixtures: the pipeline must REJECT these files, with a diagnostic
# containing the given substring. (Parsing must still succeed.)
EXPECTED_PIPELINE_ERROR = {
    "test_borrow_check.mx": "borrow",
    "test_type_error.mx": "type",
}


def run_one(path: str, stage: str) -> tuple[bool, str]:
    source = open(path).read()
    expected_error = EXPECTED_PIPELINE_ERROR.get(os.path.basename(path)) if stage != "parse" else None
    try:
        if stage == "parse":
            from metaxu.parser import Parser

            Parser().parse(source, file_path=path)
            return True, "parsed"
        if stage == "run":
            from metaxu.compiler.pipeline import (
                build_context_from_source,
                run_pipeline_ctx,
            )
            from metaxu.compiler.hir import HIRBuilder
            from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
            from metaxu.compiler.mir_interp import MirInterpreter, UNIT

            ctx = build_context_from_source(source, file_path=path)
            run_pipeline_ctx(ctx)  # strict checks (raises on negative fixtures)
            hir_funcs = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
            interp = MirInterpreter()
            interp.load(lower_hir_to_mir(hir_funcs))
            interp.register_builtin("print", lambda *a: UNIT)

            # Entry point: main(), falling back to a zero-arg example() for
            # files that demonstrate a library (e.g. 10_traits_and_structs.mx).
            entry = None
            if "main" in interp._funcs:
                entry = "main"
            elif ("example" in interp._funcs
                  and not interp._funcs["example"].param_names()):
                entry = "example"
            if entry is None:
                return True, "no main (compile-only)"
            result = interp.call(entry, [])
            if expected_error:
                return False, f"expected a '{expected_error}' diagnostic but {entry}() ran"
            return True, f"{entry}() = {result!r}"[:60]
        from metaxu.compiler.pipeline import run_pipeline_from_source

        _ast, hir, mir, clif = run_pipeline_from_source(source, file_path=path)
        if expected_error:
            return False, f"expected a '{expected_error}' diagnostic but the pipeline accepted the file"
        return True, f"hir={len(hir)}b mir={len(mir)}b clif={len(clif)}b"
    except Exception as exc:  # noqa: BLE001 - report every failure uniformly
        if expected_error and expected_error in str(exc).lower():
            return True, f"rejected as expected: {type(exc).__name__}"
        last = traceback.format_exc().strip().splitlines()[-1]
        return False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]} ({last[:80]})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["parse", "pipeline", "run"], default="pipeline")
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
