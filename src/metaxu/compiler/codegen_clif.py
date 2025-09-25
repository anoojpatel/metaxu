from __future__ import annotations

from typing import Sequence

from .mir import MirFunc


def emit_clif(funcs: Sequence[MirFunc]) -> str:
    """String emitter for CLIF.

    This is intentionally a simple textual form for golden tests.
    """
    out: list[str] = []
    for f in funcs:
        out.append(f"; function {f.name}")
        out.append("function %" + f.name + "() {")
        out.append("  block0:")
        out.append("    ; ...")
        out.append("    return")
        out.append("}")
    return "\n".join(out)
