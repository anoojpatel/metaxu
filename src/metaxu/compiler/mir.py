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
    # Parameter names with pass-by-reference (write-back) semantics: params
    # declared @mut (uniqueness 'mutable'/'exclusive') plus non-@const
    # method receivers (`self` of an __impl$... function). Both engines
    # write a rebound struct param back to the caller ONLY for these names;
    # plain params keep value semantics (rebinding stays callee-local).
    mut_params: tuple = ()
    # Pre-monomorphization display name ("" = same as name). Runtime
    # failure messages that embed a function name (match_fail) use this so
    # the monomorphized native lane and interp-on-monomorphized-MIR runs
    # bind the SAME string the unspecialized interpreter binds. Not printed
    # by dump_mir (golden MIR text is name-keyed and unchanged).
    origin_name: str = ""
    # errors.SourceLocation of the function's DECLARATION, carried so a
    # runtime error can name the function it happened in. Individual ops
    # have no locations: MIR ops are positional tuples (and are dumped
    # verbatim into the golden MIR text), so per-op spans would mean
    # reshaping every op — see docs/diagnostics_locations.md.
    location: Any = None

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
