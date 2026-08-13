"""Statement-position (else-less) `if` / `if let` must evaluate to unit.

An `if` without an else branch is by definition unit-valued (the standard
Rust statement rule), regardless of where it appears: the then-arm still
executes for its effects (assignments, early `return` via the epilogue),
but its value is discarded instead of merging with an implicit unit else
arm. An `if/else` in value position keeps merging both arms as before.

The accident this pins against: `examples/linked_list.mx`'s `main` ends in
a trailing else-less `if let Some(node) = owned_node { node.data = 42; }`,
which used to leak the then-arm's `struct Node` value into the function
result (merged with unit), so main "returned either" a Node or unit.

All tests go through parsed source (parse -> ... -> interpreter), per the
project convention.
"""
from __future__ import annotations

import os

from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import MirInterpreter, UNIT, MxUnit
from metaxu.compiler.pipeline import build_context_from_source, run_pipeline_ctx

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def run_main(source: str, entry: str = "main", strict: bool = False):
    """Compile ``source`` from text and execute ``entry`` in the interpreter."""
    ctx = build_context_from_source(source)
    if strict:
        run_pipeline_ctx(ctx)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    interp.register_builtin("print", lambda *a: UNIT)
    return interp.call(entry, [])


# ---------------------------------------------------------------------------
# Else-less `if` yields unit on BOTH paths (block-tail position)
# ---------------------------------------------------------------------------

def test_elseless_if_taken_is_unit():
    src = """
fn main() {
    let c = 1;
    if c > 0 { 42 }
}
"""
    assert isinstance(run_main(src), MxUnit)


def test_elseless_if_not_taken_is_unit():
    src = """
fn main() {
    let c = 0;
    if c > 0 { 42 }
}
"""
    assert isinstance(run_main(src), MxUnit)


def test_elseless_if_then_arm_still_runs_for_effects():
    # The arm's value is discarded, but the arm itself must execute.
    src = """
fn main() -> int {
    let @mut t = 0;
    if 1 > 0 {
        t = 7;
    }
    t
}
"""
    assert run_main(src) == 7


# ---------------------------------------------------------------------------
# Early `return` inside an else-less if still returns properly
# ---------------------------------------------------------------------------

def test_elseless_if_early_return_taken_and_fallthrough():
    src = """
fn f(c: int) -> int {
    if c > 0 {
        return 10
    }
    5
}

fn main() -> int {
    f(1) * 100 + f(0)
}
"""
    assert run_main(src) == 10 * 100 + 5


# ---------------------------------------------------------------------------
# `if/else` in value position still merges both arms
# ---------------------------------------------------------------------------

def test_if_else_in_value_position_still_merges():
    src = """
fn f(c: int) -> int {
    if c > 0 { 1 } else { 2 }
}

fn main() -> int {
    f(1) * 10 + f(0)
}
"""
    assert run_main(src) == 1 * 10 + 2


# ---------------------------------------------------------------------------
# Else-less `if let` (desugared to Match) yields unit on both paths
# ---------------------------------------------------------------------------

def test_elseless_if_let_hit_is_unit():
    src = """
fn main() {
    let v = Some(5);
    if let Some(x) = v {
        x + 1;
    }
}
"""
    assert isinstance(run_main(src), MxUnit)


def test_elseless_if_let_miss_is_unit():
    src = """
fn probe(v: Option[int]) {
    if let Some(x) = v {
        x + 1;
    }
}

fn main() {
    probe(None)
}
"""
    assert isinstance(run_main(src), MxUnit)


def test_elseless_if_let_arm_effects_and_early_return():
    # take(Some(9)) returns 9 via the early return inside the if-let arm;
    # take(None) falls through to the tail expression.
    src = """
fn take(v: Option[int]) -> int {
    if let Some(n) = v {
        return n
    }
    0 - 1
}

fn main() -> int {
    take(Some(9)) * 1000 + take(None)
}
"""
    assert run_main(src) == 9 * 1000 - 1


def test_if_let_with_else_still_merges_arm_values():
    src = """
fn unwrap_or(v: Option[int], dflt: int) -> int {
    if let Some(x) = v { x } else { dflt }
}

fn main() -> int {
    unwrap_or(Some(3), 0) * 10 + unwrap_or(None, 4)
}
"""
    assert run_main(src) == 3 * 10 + 4


# ---------------------------------------------------------------------------
# linked_list.mx: main ends in a trailing else-less `if let` -> unit
# ---------------------------------------------------------------------------

def test_linked_list_main_returns_unit():
    # Before the fix, main's trailing else-less `if let Some(node) = ...`
    # leaked the taken arm's value: main returned
    #   Node { data=42, next=Some(Node { data=3, next=None }) }
    # when the pattern matched. main has no tail expression after the if-let
    # and no trailing `return`, so under the statement rule it is unit.
    source = open(os.path.join(REPO_ROOT, "examples", "linked_list.mx")).read()
    result = run_main(source, strict=True)
    assert isinstance(result, MxUnit)
