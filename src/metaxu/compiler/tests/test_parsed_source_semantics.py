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
    """The emitter must emit a Bool class constraint for if-expression
    conditions (spied directly — compiling successfully is not evidence,
    since the branch was silently dead before)."""
    from unittest.mock import patch
    from metaxu.compiler.simplesub_adapter import SimpleSubFacade

    src = """
fn f(c: bool) -> int {
    if c { 1 } else { 2 }
}
"""
    recorded: list[str] = []
    original = SimpleSubFacade.add_class_constraint

    def spy(self, cls, args, node_id=None):
        recorded.append(cls)
        return original(self, cls, args, node_id)

    with patch.object(SimpleSubFacade, "add_class_constraint", spy):
        ctx = build_context_from_source(src)
    kinds = set()
    def walk(n):
        kinds.add(n.kind)
        for ch in n.children:
            walk(ch)
    walk(ctx.frozen_root)
    assert "IfExpression" in kinds  # parser emits IfExpression, not IfStatement
    assert "Bool" in recorded, "no Bool constraint emitted for the if condition"


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


# ---------------------------------------------------------------------------
# Runtime semantics of modes example constructs (example 01 regressions)
# ---------------------------------------------------------------------------

def test_borrow_argument_passes_value():
    """`f(&mut x)` passes x's value; it must not be silently dropped from the
    argument list (which previously shifted every later argument left)."""
    src = """
struct Counter { value: int }

fn bump(@mut c: Counter, delta: int) -> int {
    c.value + delta
}

fn main() -> int {
    let counter = Counter { value: 40 };
    bump(&mut counter, 2)
}
"""
    assert call(src, "main", []) == 42


def test_field_assignment_writes_through():
    """`c.value = expr` updates the struct in the variable's slot, instead of
    creating a phantom local named 'c.value'."""
    src = """
struct Counter { value: int }

fn main() -> int {
    let @mut c = Counter { value: 1 };
    c.value = c.value + 10;
    c.value
}
"""
    assert call(src, "main", []) == 11


def test_to_string_builtin_method():
    src = """
fn main() -> string {
    let x = 7;
    "v=" + x.to_string()
}
"""
    assert call(src, "main", []) == "v=7"


def test_field_access_with_string_base():
    """The parser stores `c.name`'s base as a raw string; it must still read."""
    src = """
struct P { name: string }

fn main() -> string {
    let p = P { name: "ok" };
    p.name
}
"""
    assert call(src, "main", []) == "ok"


# ---------------------------------------------------------------------------
# Second adversarial-review round regressions
# ---------------------------------------------------------------------------

def test_bare_variant_pattern_is_ctor_not_binding():
    """`Point => ...` must match only the Point variant, not everything."""
    src = """
enum Shape { Point, Circle(r: int) }

fn f(s: Shape) -> int {
    match s {
        Point => 0,
        Circle(r) => r
    }
}

fn main() -> int {
    f(Circle(5)) + f(Point)
}
"""
    assert call(src, "main", []) == 5


def test_negative_literal_pattern():
    src = """
fn f(x: int) -> int {
    match x {
        -1 => 10,
        _ => 0
    }
}
"""
    assert call(src, "f", [-1]) == 10
    assert call(src, "f", [5]) == 0


def test_shadow_in_nested_block_does_not_unmove_outer():
    """An inner-scope `let x` must not clear the outer x's moved flag."""
    src = """
fn f() -> int {
    let x = 1;
    let y = move(x);
    {
        let x = 2;
        x
    };
    x
}
"""
    with pytest.raises(BorrowCheckError, match="moved"):
        run_pipeline_from_source(src)


def test_function_decl_between_borrows_keeps_outer_borrow():
    """Walking a nested fn whose param shares a name with an outer borrowed
    variable must not erase the outer borrow."""
    src = """
fn main() -> int {
    let @mut x = 1;
    let r = @mut x;
    let r2 = @mut x;
    0
}

fn f(x: int) -> int { x }
"""
    with pytest.raises(BorrowCheckError):
        run_pipeline_from_source(src)


def test_shadowed_reference_holder_does_not_block_new_binding():
    """After `let x = 2` shadows x, the old named reference to the old x must
    not make call-site borrows of the NEW x unreleasable."""
    src = """
fn f(@const p: int) -> int { p }
fn g(@mut p: int) -> int { p }

fn main() -> int {
    let x = 1;
    let r = &x;
    let x = 2;
    f(&x);
    g(@mut x);
    0
}
"""
    run_pipeline_from_source(src)  # must not raise
