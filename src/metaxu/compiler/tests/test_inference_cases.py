"""End-to-end inference cases (docs/type_inference_plan.md).

Same contract as the book harness: each case is a complete program run
through the real pipeline; ``ok`` cases pin stdout, ``error`` cases pin
a fragment of the diagnostic. Parts 1 (call edges in conflict detection)
and 2 (let-polymorphism for let-bound lambdas) have landed and are
pinned below without markers; part 3 (two-location diagnostics) is
still xfail so the suite documents the target without going red.
"""
from __future__ import annotations

import pytest

from metaxu.compiler.frozen_borrow_checker import (BorrowCheckError,
                                                   TypeCheckError)
from metaxu.compiler.mir_interp import InterpError, _eval_binop
from metaxu.compiler.pipeline import build_context_from_source
from metaxu.compiler.tests.test_codegen_llvm import interp_run

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


def principal_types_of(src: str, kind: str) -> list[str]:
    """Principal types of every node of `kind`, in source order."""
    ctx = build_context_from_source(src)
    out = []

    def walk(n):
        if n.kind == kind:
            out.append(ctx.tables.facade.principal_type_of(n.node_id))
        for c in n.children:
            walk(c)

    walk(ctx.frozen_root)
    return out


# --- part 1: call edges ------------------------------------------------

APPLY = """
fn apply(f: fn(int) -> int) -> int {
    f(20)
}
"""


def test_lambda_argument_body_mismatch_is_rejected():
    # `s` receives an Int through apply's declared parameter type and the
    # body needs a String; reported at the string, the value that made
    # the conflict apparent
    run_error(APPLY + """
fn main() -> int {
    print(apply(fn(s) -> s + "!"));
    0
}
""", "<mem>:7:30: type mismatch: one value is required to be Int and String")


def test_lambda_argument_flows_through_call_edge():
    run_ok(APPLY + """
fn main() -> int {
    print(apply(fn(s) -> s * 2));
    0
}
""", "40")


def test_let_bound_lambda_passed_as_argument_is_checked():
    run_error(APPLY + """
fn main() -> int {
    let shout = fn(s) -> s + "!";
    print(apply(shout));
    0
}
""", "Int and String")


def test_let_bound_lambda_passed_as_argument_runs():
    run_ok(APPLY + """
fn main() -> int {
    let twice = fn(s) -> s * 2;
    print(apply(twice));
    print(twice(4));
    0
}
""", "40\n8")


def test_let_bound_lambda_called_at_wrong_type_is_rejected():
    run_error("""
fn main() -> int {
    let shout = fn(s) -> s + "!";
    print(shout(1));
    0
}
""", "Int and String")


def test_nested_function_type_parameter():
    run_ok("""
fn run(g: fn(fn(int) -> int) -> int) -> int {
    g(fn(n) -> n + 1)
}

fn main() -> int {
    print(run(fn(h) -> h(41)));
    0
}
""", "42")


def test_declared_function_type_is_in_the_frozen_payload():
    ctx = build_context_from_source(APPLY + "fn main() -> int { 0 }\n")
    decls = {}

    def walk(n):
        if n.kind == "FunctionDeclaration":
            decls[n.value["name"]] = n.value
        for c in n.children:
            walk(c)

    walk(ctx.frozen_root)
    assert decls["apply"]["param_types"] == ["fn(int) -> int"]


def test_interpreter_never_leaks_a_host_type_error():
    # the checker's backstop: a value reaching an operator at a type the
    # checker did not see is a catchable Metaxu error, not a Python one
    with pytest.raises(InterpError) as ei:
        _eval_binop("+", 1, "a")
    assert str(ei.value) == \
        "binary operator '+' cannot be applied to Int and String"


# --- part 2: let-polymorphism -------------------------------------------

LET_IDENTITY = """
fn main() -> int {
    let same = fn(x) -> x;
    print(same(1));
    print(same("a"));
    0
}
"""


def test_let_identity_used_at_two_types_compiles():
    run_ok(LET_IDENTITY, "1\na")


def test_let_identity_stays_polymorphic():
    # Before generalization both uses flowed into the one lambda type and
    # the biunifier rendered it `(Int ∧ String) -> (Int ∨ String)`; each
    # use now talks to its own instance, so the lambda itself is the
    # polymorphic `'a -> 'a`. (The call RESULTS are not pinned: the flat
    # solver fuses every statement of `main` into one representative, so
    # their principal types are the chain's, not the call's; sharpening
    # those edges is plan part 4.)
    assert principal_types_of(LET_IDENTITY, "LambdaExpression") == ["'a -> 'a"]


def test_let_bound_lambda_is_generic_in_its_own_variables_only():
    # the lambda's parameter and body are generalized; `k`, bound outside,
    # is shared by every instance and stays Int
    run_ok("""
fn main() -> int {
    let k = 1;
    let addk = fn(x) -> x + k;
    let pair = fn(a) -> a;
    print(addk(2));
    print(pair("s"));
    print(pair(addk(3)));
    0
}
""", "3\ns\n4")


def test_captured_outer_binding_stays_monomorphic():
    # k is Int and the lambda's parameter unifies with k through `+`, so
    # a String argument collides even though `addk` is generalized
    run_error("""
fn main() -> int {
    let k = 1;
    let addk = fn(x) -> x + k;
    print(addk(2));
    print(addk("a"));
    0
}
""", "Int and String")


def test_mutable_lambda_binding_is_not_generalized():
    # the value restriction: a `let mut` lambda can be reassigned, so it
    # keeps one type across its uses
    run_error("""
fn main() -> int {
    let mut same = fn(x) -> x;
    print(same(1));
    print(same("a"));
    0
}
""", "Int and String")


def test_shadowing_a_generalized_lambda_drops_its_scheme():
    run_ok("""
fn main() -> int {
    let same = fn(x) -> x;
    let same = 7;
    print(same + 1);
    0
}
""", "8")


def test_once_lambda_instances_count_as_one_callable():
    run_error("""
fn main() -> int {
    let @once same = fn(x) -> x;
    print(same(1));
    print(same("a"));
    0
}
""", "once")


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


def test_named_function_with_bare_parameter_is_polymorphic():
    # a named function with an undeclared parameter keeps one shared type
    # (its call edges are not folded), exactly as before part 1
    run_ok("""
fn same(x) {
    x
}

fn main() -> int {
    print(same(1));
    print(same("a"));
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
