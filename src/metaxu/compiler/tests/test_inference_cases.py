"""End-to-end inference cases (docs/type_inference_plan.md).

Same contract as the book harness: each case is a complete program run
through the real pipeline; ``ok`` cases pin stdout, ``error`` cases pin
a fragment of the diagnostic. Cases for behavior that is planned but
not landed are marked xfail so the suite documents the target without
going red; drop the marker as each lands.
"""
from __future__ import annotations

import pytest

from metaxu.compiler.frozen_borrow_checker import (BorrowCheckError,
                                                   TypeCheckError)
from metaxu.compiler.pipeline import build_context_from_source
from metaxu.compiler.tests.test_codegen_llvm import interp_run

P1 = pytest.mark.xfail(reason="plan part 1: call edges in conflict "
                       "detection", strict=False)
P2 = pytest.mark.xfail(reason="plan part 2: let-polymorphism on the "
                       "compile path", strict=False)
P3 = pytest.mark.xfail(reason="plan part 3: two-location conflict "
                       "diagnostics", strict=False)


def run_ok(src: str, expected: str) -> None:
    result, out = interp_run(src)
    assert result == 0, out
    assert out.rstrip("\n") == expected.rstrip("\n"), out


def run_error(src: str, fragment: str) -> None:
    with pytest.raises((TypeCheckError, BorrowCheckError)) as ei:
        interp_run(src)
    assert fragment in str(ei.value), str(ei.value)


def principal_types_of_calls(src: str, callee: str) -> list[str]:
    """Principal types of every `callee(...)` call node, in source order."""
    ctx = build_context_from_source(src)
    out = []

    def walk(n):
        if n.kind == "FunctionCall" and isinstance(n.value, dict) \
                and n.value.get("name") == callee:
            out.append(ctx.tables.facade.principal_type_of(n.node_id))
        for c in n.children:
            walk(c)

    walk(ctx.frozen_root)
    return out


# --- part 1: call edges ------------------------------------------------

@P1
def test_lambda_argument_body_mismatch_is_rejected():
    # `s` receives an Int through apply's call edge and the body needs a
    # String; today this compiles and dies at run time
    run_error("""
fn apply(f: fn(int) -> int) -> int {
    f(20)
}

fn main() -> int {
    print(apply(fn(s) -> s + "!"));
    0
}
""", "Int and String")


def test_lambda_argument_flows_through_call_edge():
    run_ok("""
fn apply(f: fn(int) -> int) -> int {
    f(20)
}

fn main() -> int {
    print(apply(fn(s) -> s * 2));
    0
}
""", "40")


# --- part 2: let-polymorphism -------------------------------------------

LET_IDENTITY = """
fn main() -> int {
    let same = fn(x) -> x;
    print(same(1));
    print(same("a"));
    0
}
"""


def test_let_identity_used_at_two_types_compiles_today():
    # accepted because call edges are not merged, not because `same` is
    # generalized (chapter 18 explains); part 1 must not break this, and
    # part 2 is what keeps it compiling once part 1 lands
    run_ok(LET_IDENTITY, "1\na")


def test_let_identity_is_monomorphic_today():
    tys = principal_types_of_calls(LET_IDENTITY, "same")
    assert tys == ["Int ∨ String", "Int ∨ String"]


@P2
def test_let_identity_instantiates_per_use():
    tys = principal_types_of_calls(LET_IDENTITY, "same")
    assert tys == ["Int", "String"]


@P1
def test_captured_outer_binding_stays_monomorphic():
    # k is Int and the lambda's parameter unifies with k through `+`, so
    # a String argument must collide with or without generalization; it
    # reaches the parameter through a call edge, so today it slips past
    # the detector and dies at run time (same gap as the apply case)
    run_error("""
fn main() -> int {
    let k = 1;
    let addk = fn(x) -> x + k;
    print(addk(2));
    print(addk("a"));
    0
}
""", "Int and String")


def test_named_generic_function_is_polymorphic():
    run_ok("""
fn ident<T>(x: T) -> T {
    x
}

fn main() -> int {
    print(ident(1));
    print(ident("a"));
    0
}
""", "1\na")


# --- part 3: conflict diagnostics --------------------------------------

def test_conflict_names_both_requirements():
    run_error("""
fn main() -> int {
    print(1 + "a");
    0
}
""", "one value is required to be Int and String")


@P3
def test_conflict_reports_where_each_requirement_came_from():
    run_error("""
fn main() -> int {
    let n = 1 + "a";
    print(n);
    0
}
""", "comes from here")


# --- invariants the plan must not break --------------------------------

def test_if_arms_still_unify():
    run_error("""
fn main() -> int {
    let label = if 1 < 2 { "small" } else { 0 };
    print(label);
    0
}
""", "Int and String")


def test_declared_parameter_types_still_enforced():
    run_error("""
fn half(n: int) -> int {
    n / 2
}

fn main() -> int {
    print(half("ten"));
    0
}
""", "Int and String")
