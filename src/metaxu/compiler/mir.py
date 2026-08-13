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
    # Module-level constant names initialized by this function (only set on
    # the synthesized __module_init): the interpreter runs it before the
    # entry point and publishes exactly these bindings as globals.
    globals_decl: tuple = ()

    def param_names(self) -> tuple:
        """Parameter names from the entry block's params op (empty if none)."""
        for op in (self.blocks[0].ops if self.blocks else ()):
            if op[0] == "params":
                return tuple(op[1])
        return ()


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
