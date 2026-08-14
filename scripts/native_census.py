"""Census of native placeholders by reason, for the gate corpus.

Prints, per file: number of real `define`s, number of placeholder comments,
and a histogram of placeholder reasons. Used to measure native-coverage
levers before/after a change.
"""
from __future__ import annotations

import collections
import glob
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from metaxu.compiler.pipeline import emit_llvm_from_source  # noqa: E402

# _emit_placeholder writes a header line per demoted function followed by one
# `;   reason: ...` line per reason, so functions and reasons count separately.
PLACEHOLDER_RE = re.compile(r"^; function @\S+: placeholder ", re.M)
REASON_RE = re.compile(r"^;\s+reason:\s*(.*)$", re.M)


def normalize(reason: str) -> str:
    """Collapse instance-specific detail so reasons group into causes."""
    reason = re.sub(r"'[^']*'", "'X'", reason)
    reason = re.sub(r"\benum:\w+\{[^}]*\}", "enum:E{...}", reason)
    reason = re.sub(r"\b\d+\b", "N", reason)
    return reason.strip()


def census(path: str):
    src = open(path).read()
    try:
        ir = emit_llvm_from_source(src, file_path=path)
    except Exception as exc:  # noqa: BLE001 - a failing file is a data point
        return None, f"{type(exc).__name__}: {str(exc).splitlines()[0][:80]}"
    defines = len(re.findall(r"^define ", ir, re.M))
    holes = len(PLACEHOLDER_RE.findall(ir))
    reasons = [normalize(r) for r in REASON_RE.findall(ir)]
    return (defines, holes, reasons), None


def main() -> int:
    targets = sys.argv[1:] or (
        sorted(glob.glob(os.path.join(REPO_ROOT, "examples", "*.mx")))
        + [os.path.join(REPO_ROOT, "examples", "app", "main.mx")]
    )
    total = collections.Counter()
    tot_def = tot_ph = 0
    for path in targets:
        data, err = census(path)
        rel = os.path.relpath(path, REPO_ROOT)
        if err:
            print(f"{rel:42} ERROR {err}")
            continue
        defines, holes, reasons = data
        tot_def += defines
        tot_ph += holes
        total.update(reasons)
        print(f"{rel:42} {defines:4} defines {holes:4} placeholders"
              f" ({len(reasons)} reasons)")
    print(f"\nTOTAL {tot_def} defines / {tot_ph} placeholders")
    print("\nBy reason:")
    for reason, n in total.most_common():
        print(f"{n:5}  {reason[:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
