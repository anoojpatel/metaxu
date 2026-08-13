"""Regression tests for parsed-source semantics found by adversarial review.

Every test here goes through the REAL front end (parse -> desugar -> freeze ->
infer -> HIR -> MIR -> interpreter). The bugs these pin down all shared one
root cause: unit tests exercised hand-built HIR/frozen fixtures and missed
parser/freeze/HIR seams where constructs silently degraded (patterns became
wildcards, modes were dropped, Some/None arguments vanished).
"""
from __future__ import annotations

import pytest

from metaxu.compiler.pipeline import build_context_from_source, run_pipeline_from_source
from metaxu.compiler.frozen_borrow_checker import BorrowCheckError
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import MirInterpreter


def call(source: str, fn: str, args: list):
    ctx = build_context_from_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    return interp.call(fn, args)


# ---------------------------------------------------------------------------
# Match patterns from parsed source (were all degrading to wildcards)
# ---------------------------------------------------------------------------

MATCH_LITERALS = """
fn f(x: int) -> int {
    match x {
        1 => 10,
        2 => 20,
        y => y
    }
}
"""


def test_parsed_literal_patterns_discriminate():
    assert call(MATCH_LITERALS, "f", [1]) == 10
    assert call(MATCH_LITERALS, "f", [2]) == 20


def test_parsed_var_pattern_binds():
    assert call(MATCH_LITERALS, "f", [7]) == 7


def test_parsed_ctor_patterns_and_option_constructors():
    src = """
enum Option { Some(value: int), None }

fn g(o: Option) -> int {
    match o {
        Some(v) => v,
        None => 0
    }
}

fn main() -> int {
    g(Some(41)) + g(None)
}
"""
    assert call(src, "main", []) == 41


def test_parsed_wildcard_pattern():
    src = """
fn f(x: int) -> int {
    match x {
        1 => 100,
        _ => 5
    }
}
"""
    assert call(src, "f", [1]) == 100
    assert call(src, "f", [9]) == 5


# ---------------------------------------------------------------------------
# Mode annotations from parsed source (were dropped at freeze)
# ---------------------------------------------------------------------------

def test_local_binding_cannot_escape_via_return():
    src = """
fn f() -> int {
    let @local x = 1;
    return x;
}
"""
    with pytest.raises(BorrowCheckError, match="escape"):
        run_pipeline_from_source(src)


def test_global_binding_may_return():
    src = """
fn f() -> int {
    let x = 1;
    return x;
}
"""
    run_pipeline_from_source(src)  # must not raise


# ---------------------------------------------------------------------------
# Borrow state is per-function (moves were leaking across functions)
# ---------------------------------------------------------------------------

def test_move_in_one_function_does_not_poison_another():
    src = """
fn f() -> int {
    let x = 1;
    let y = move(x);
    y
}

fn g() -> int {
    let x = 2;
    x
}
"""
    run_pipeline_from_source(src)  # must not raise


def test_use_after_move_still_rejected_within_function():
    src = """
fn f() -> int {
    let x = 1;
    let y = move(x);
    x
}
"""
    with pytest.raises(BorrowCheckError, match="moved"):
        run_pipeline_from_source(src)


# ---------------------------------------------------------------------------
# If-expression conditions get real type constraints (branch was dead code)
# ---------------------------------------------------------------------------

def test_if_expression_emits_bool_constraint():
    src = """
fn f(c: bool) -> int {
    if c { 1 } else { 2 }
}
"""
    ctx = build_context_from_source(src)
    kinds = set()
    def walk(n):
        kinds.add(n.kind)
        for ch in n.children:
            walk(ch)
    walk(ctx.frozen_root)
    assert "IfExpression" in kinds  # the emitter's condition branch must cover this kind
    run_pipeline_from_source(src)


# ---------------------------------------------------------------------------
# Named borrows survive call-argument borrow release
# ---------------------------------------------------------------------------

def test_named_borrow_still_conflicts_after_call_release():
    """f(@mut x) must not erase a live named borrow of x taken with let."""
    src = """
fn f(@mut p: int) -> int { p }

fn main() -> int {
    let @mut x = 1;
    let r = @mut x;
    f(@mut x);
    let r2 = @mut x;
    0
}
"""
    with pytest.raises(BorrowCheckError):
        run_pipeline_from_source(src)
