"""End-to-end inference cases (docs/type_inference_plan.md).

Same contract as the book harness: each case is a complete program run
through the real pipeline; ``ok`` cases pin stdout, ``error`` cases pin
a fragment of the diagnostic. Cases for behavior that is planned but
not landed are marked xfail so the suite documents the target without
going red; drop the marker as each lands.

This module needs the v1 pipeline (``interp_run``); on a tree without
it the whole module skips.
"""
from __future__ import annotations

import pytest

try:
    from metaxu.compiler.frozen_borrow_checker import (BorrowCheckError,
                                                       TypeCheckError)
    from metaxu.compiler.tests.test_codegen_llvm import interp_run
except Exception:  # pragma: no cover - old trees
    pytest.skip("inference cases need the v1 pipeline (interp_run)",
                allow_module_level=True)

P1 = pytest.mark.xfail(reason="plan part 1: let-polymorphism by "
                       "constraint replay", strict=False)
P2 = pytest.mark.xfail(reason="plan part 2: two-location conflict "
                       "diagnostics", strict=False)


def run_ok(src: str, expected: str) -> None:
    result, out = interp_run(src)
    assert result == 0, out
    assert out.rstrip("\n") == expected.rstrip("\n"), out


def run_error(src: str, fragment: str) -> None:
    with pytest.raises((TypeCheckError, BorrowCheckError)) as ei:
        interp_run(src)
    assert fragment in str(ei.value), str(ei.value)


# --- part 1: let-polymorphism ------------------------------------------

@P1
def test_let_identity_used_at_two_types():
    run_ok("""
fn main() -> int {
    let same = fn(x) -> x;
    print(same(1));
    print(same("a"));
    0
}
""", "1\na")


@P1
def test_first_class_use_instantiates_too():
    run_ok("""
fn main() -> int {
    let same = fn(x) -> x;
    let g = same;
    print(g(2));
    print(g("b"));
    0
}
""", "2\nb")


def test_captured_outer_binding_stays_monomorphic():
    # k is Int; the lambda's parameter must unify with k, so applying
    # it to a string is a conflict with or without generalization
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
fn ident[T](x: T) -> T {
    x
}

fn main() -> int {
    print(ident(1));
    print(ident("a"));
    0
}
""", "1\na")


@P1
def test_value_restriction_keeps_call_results_monomorphic():
    # `let f = make();` binds a value produced by a call, not a lambda
    # literal, so it is not generalized: using it at two types conflicts
    run_error("""
fn make() -> fn(int) -> int {
    fn(x: int) -> x
}

fn main() -> int {
    let f = make();
    print(f(1));
    print(f("a"));
    0
}
""", "Int and String")


# --- part 2: conflict diagnostics --------------------------------------

def test_conflict_names_both_requirements():
    run_error("""
fn main() -> int {
    print(1 + "a");
    0
}
""", "one value is required to be Int and String")


@P2
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
