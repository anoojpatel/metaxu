from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(slots=True)
class MirBlock:
    ops: list[tuple]
    term: tuple


@dataclass(slots=True)
class MirFunc:
    name: str
    ty_sig: Any
    blocks: list[MirBlock]
    suspending: bool


def dump_mir(funcs: Sequence[MirFunc]) -> str:
    out: list[str] = []
    for f in funcs:
        out.append(f"func {f.name} suspending={f.suspending}")
        for bi, b in enumerate(f.blocks):
            out.append(f"  bb{bi}:")
            for op in b.ops:
                out.append(f"    {op}")
            out.append(f"    term {b.term}")
    return "\n".join(out)
