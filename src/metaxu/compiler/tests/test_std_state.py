"""std.state: State as an algebraic effect over a mutable capture cell.

Parsed-source end to end (module loader resolves std/state.mx; the
handler arms share one cell-backed variable).
"""
from __future__ import annotations

import os

from metaxu.compiler.pipeline import build_context_from_source
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import MirInterpreter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def run_main(source: str):
    path = os.path.join(REPO_ROOT, "__std_state_probe__.mx")
    ctx = build_context_from_source(source, file_path=path)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    return interp.call("main", [])


def test_get_put_roundtrip():
    assert run_main("""
from std.state import with_state;

fn counter() -> int {
    perform State.put(perform State.get() + 1);
    perform State.put(perform State.get() + 1);
    perform State.get()
}

from std.state import State;

fn main() -> int {
    with_state(40, fn() -> counter())
}
""") == 42


def test_state_reaches_called_functions_deeply():
    assert run_main("""
from std.state import State, with_state, modify;

fn bump_twice() -> int {
    modify(fn(s: int) -> s * 2);
    modify(fn(s: int) -> s + 1);
    perform State.get()
}

fn main() -> int {
    with_state(10, fn() -> bump_twice())
}
""") == 21


def test_nested_with_state_scopes_are_independent():
    """Inner with_state gets its own cell; the outer scope's state is
    untouched by inner puts (innermost handler wins per effect
    semantics). Block-bodied lambdas don't parse yet (grammar gap, noted
    in std/README.md), so the bodies are named helpers."""
    assert run_main("""
from std.state import State, with_state;

fn inner_body() -> int {
    perform State.put(perform State.get() + 1);
    perform State.get()
}

fn outer_body() -> int {
    let inner = with_state(100, fn() -> inner_body());
    perform State.put(perform State.get() + inner);
    perform State.get()
}

fn main() -> int {
    with_state(1, fn() -> outer_body())
}
""") == 102
