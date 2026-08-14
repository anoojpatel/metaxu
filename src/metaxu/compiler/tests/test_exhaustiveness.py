"""Compile-time match exhaustiveness checking.

Non-exhaustive matches are rejected at compile time (kind
"type-nonexhaustive-match", raised as TypeCheckError by the strict
pipeline) instead of failing at runtime with match_fail. The check lives in
frozen_constraint_emitter._check_match_exhaustiveness, driven by the arm
descriptors mutaxu_ast freezes into MatchExpression payloads.

Coverage rule: a variant is covered iff a wildcard/binding arm exists, or
the arms naming its constructor cover its whole payload space — computed by
the textbook specialization recursion, so NESTED constructor patterns count
(`Some(TInt(n)) | Some(TName(s)) | Some(TOp(s)) | None` is exhaustive).
The recursion only ever answers "provably exhaustive": literal columns,
unknown enums and mixed ctor/literal columns answer no, so literal
completeness itself is still not analyzed (`Some(1) | Some(n)` is covered by
the binding arm, `Some(1)` alone is not). Matches whose scrutinee type
cannot be determined from the patterns (or that contain opaque pattern
forms) are not checked — no false positives.

An arm following a wildcard/binding arm is unreachable; that is a
warning-level advisory on the -1 diagnostics channel (like "Unresolved
..." advisories), never an error.

All tests go through parsed source per the repo convention.
"""
from __future__ import annotations

import pytest

from metaxu.compiler.pipeline import (
    build_context_from_source,
    run_pipeline_ctx,
    run_pipeline_from_source,
)
from metaxu.compiler.frozen_borrow_checker import TypeCheckError


def compile_src(src: str):
    return run_pipeline_from_source(src)


def reject(src: str) -> str:
    """Compile expecting a TypeCheckError; return the joined messages."""
    with pytest.raises(TypeCheckError) as excinfo:
        compile_src(src)
    errors = excinfo.value.errors
    assert any(
        getattr(e, "kind", "") == "type-nonexhaustive-match" for e in errors
    ), f"expected a type-nonexhaustive-match error, got: {errors}"
    return "; ".join(getattr(e, "message", str(e)) for e in errors)


# ---------------------------------------------------------------------------
# Declared enums
# ---------------------------------------------------------------------------

def test_missing_variants_rejected_with_names():
    msg = reject("""
enum Shape { Circle(r: int), Square(s: int), Dot }

fn f(s: Shape) -> int {
    match s {
        Dot => 0
    }
}
""")
    assert "non-exhaustive match" in msg
    assert "Circle" in msg and "Square" in msg


def test_single_missing_variant_named():
    msg = reject("""
enum Shape { Circle(r: int), Square(s: int), Dot }

fn f(s: Shape) -> int {
    match s {
        Circle(r) => r,
        Square(x) => x
    }
}
""")
    assert "missing variants Dot" in msg


def test_full_variant_coverage_accepted():
    compile_src("""
enum Shape { Circle(r: int), Square(s: int), Dot }

fn f(s: Shape) -> int {
    match s {
        Circle(r) => r,
        Square(x) => x,
        Dot => 0
    }
}
""")


def test_wildcard_arm_accepted():
    compile_src("""
enum Shape { Circle(r: int), Square(s: int), Dot }

fn f(s: Shape) -> int {
    match s {
        Circle(r) => r,
        _ => 0
    }
}
""")


def test_binding_arm_accepted():
    compile_src("""
enum Shape { Circle(r: int), Dot }

fn f(s: Shape) -> int {
    match s {
        Dot => 0,
        other => 1
    }
}
""")


def test_qualified_ctor_patterns_checked():
    msg = reject("""
enum Shape { Circle(r: int), Dot }

fn f(s: Shape) -> int {
    match s {
        Shape::Circle(r) => r
    }
}
""")
    assert "missing variants Dot" in msg


# ---------------------------------------------------------------------------
# Ctor arms with refutable (literal) subpatterns do not cover their variant
# ---------------------------------------------------------------------------

def test_ctor_with_literal_subpattern_is_not_coverage():
    msg = reject("""
fn f(o: Option) -> int {
    match o {
        Some(1) => 1,
        None => 0
    }
}
""")
    assert "missing variants Some" in msg


def test_literal_subpattern_completed_by_binding_arm():
    # Some(1) | Some(n): the binding arm covers the variant (documented
    # shallow rule — literal completeness itself is not analyzed).
    compile_src("""
fn f(o: Option) -> int {
    match o {
        Some(1) => 1,
        Some(n) => n,
        None => 0
    }
}
""")


# ---------------------------------------------------------------------------
# Builtin Option / Result enums
# ---------------------------------------------------------------------------

def test_builtin_option_full_coverage_accepted():
    compile_src("""
fn f(o: Option) -> int {
    match o {
        Some(v) => v,
        None => 0
    }
}
""")


def test_builtin_option_missing_none_rejected():
    msg = reject("""
fn f(o: Option) -> int {
    match o {
        Some(v) => v
    }
}
""")
    assert "missing variants None" in msg


def test_builtin_result_missing_err_rejected():
    msg = reject("""
fn f() -> int {
    match Ok(1) {
        Ok(v) => v
    }
}
""")
    assert "missing variants Err" in msg


def test_builtin_result_full_coverage_accepted():
    compile_src("""
fn f() -> int {
    match Ok(1) {
        Ok(v) => v,
        Err(e) => 0
    }
}
""")


# ---------------------------------------------------------------------------
# Bool scrutinees
# ---------------------------------------------------------------------------

def test_bool_true_false_accepted():
    compile_src("""
fn f(b: bool) -> int {
    match b {
        true => 1,
        false => 0
    }
}
""")


def test_bool_missing_false_rejected():
    msg = reject("""
fn f(b: bool) -> int {
    match b {
        true => 1
    }
}
""")
    assert "false" in msg


def test_bool_wildcard_accepted():
    compile_src("""
fn f(b: bool) -> int {
    match b {
        true => 1,
        _ => 0
    }
}
""")


# ---------------------------------------------------------------------------
# Int / String literal scrutinees can never be exhaustive without a catch-all
# ---------------------------------------------------------------------------

def test_int_literals_without_wildcard_rejected():
    msg = reject("""
fn f(x: int) -> int {
    match x {
        1 => 10,
        2 => 20
    }
}
""")
    assert "can never be exhaustive" in msg


def test_int_literals_with_wildcard_accepted():
    compile_src("""
fn f(x: int) -> int {
    match x {
        1 => 10,
        _ => 0
    }
}
""")


def test_int_literals_with_binding_accepted():
    compile_src("""
fn f(x: int) -> int {
    match x {
        1 => 10,
        y => y
    }
}
""")


def test_string_literals_without_wildcard_rejected():
    msg = reject("""
fn f(s: str) -> int {
    match s {
        "a" => 1,
        "b" => 2
    }
}
""")
    assert "can never be exhaustive" in msg


# ---------------------------------------------------------------------------
# Unknown scrutinee types stay unchecked (no false positives)
# ---------------------------------------------------------------------------

def _no_exhaustiveness_error(src: str) -> None:
    """Check + infer the source and assert no non-exhaustive-match diagnostic.

    These two cases stop BEFORE HIR: their whole point is an opaque pattern
    shape, and HIR lowering now rejects opaque patterns loudly (a pattern it
    cannot convert used to degrade to a match-anything wildcard). The fact
    being pinned here belongs to the exhaustiveness checker, which runs during
    build_context_from_source.
    """
    ctx = build_context_from_source(src)
    errs = [e for diags in ctx.tables.constraints.values() for e in diags]
    assert not [e for e in errs
                if getattr(e, "kind", "") == "type-nonexhaustive-match"], errs


def test_unknown_type_scrutinee_unchecked():
    # The ctor pattern names no known variant, so the scrutinee's type is
    # not statically known here — the match must stay permissive.
    _no_exhaustiveness_error("""
fn g<T>(x: T) -> int {
    match x {
        WeirdThing(a) => 1
    }
}

fn main() -> int { 0 }
""")


def test_opaque_pattern_shape_unchecked():
    # A lambda-shaped pattern is an opaque form: the whole match is skipped
    # even though another arm has a resolvable ctor pattern.
    _no_exhaustiveness_error("""
fn f(o: Option) -> int {
    match o {
        Some(v) => v,
        fn(x) -> x => 0
    }
}
""")


def test_opaque_pattern_shapes_are_rejected_by_hir():
    """...and the pipeline as a whole still refuses them: an opaque pattern
    that HIR cannot convert would become a match-anything wildcard, silently
    making every later arm dead code."""
    from metaxu.compiler.hir import UnsupportedConstruct

    with pytest.raises(UnsupportedConstruct, match="not a known enum variant"):
        compile_src("""
fn g<T>(x: T) -> int {
    match x {
        WeirdThing(a) => 1
    }
}

fn main() -> int { 0 }
""")
    with pytest.raises(UnsupportedConstruct, match="LambdaExpression"):
        compile_src("""
fn f(o: Option) -> int {
    match o {
        Some(v) => v,
        fn(x) -> x => 0
    }
}
""")


# ---------------------------------------------------------------------------
# Unreachable-arm advisory (-1 channel, warning-level, not an error)
# ---------------------------------------------------------------------------

def _advisories(src: str) -> list[str]:
    ctx = build_context_from_source(src)
    return [str(e) for e in ctx.tables.constraints.get(-1, ())]


def test_unreachable_arm_after_wildcard_gets_advisory():
    src = """
fn f(x: int) -> int {
    match x {
        _ => 0,
        1 => 10
    }
}
"""
    msgs = _advisories(src)
    assert any("Unreachable match arm" in m for m in msgs)
    # Advisory only: the strict pipeline still compiles the program.
    ctx = build_context_from_source(src)
    run_pipeline_ctx(ctx)


def test_unreachable_arm_after_binding_gets_advisory():
    msgs = _advisories("""
fn f(x: int) -> int {
    match x {
        y => y,
        1 => 10
    }
}
""")
    assert any("Unreachable match arm" in m for m in msgs)


def test_no_advisory_for_final_wildcard():
    msgs = _advisories("""
fn f(x: int) -> int {
    match x {
        1 => 10,
        _ => 0
    }
}
""")
    assert not any("Unreachable match arm" in m for m in msgs)


# ---------------------------------------------------------------------------
# Nested constructor patterns (found writing examples/app's parser, which
# matches `Some(TInt(n))` over Option[Token]).  Coverage is computed by the
# textbook specialization recursion, so a nested pattern set that really
# does cover the payload space is accepted -- the old "all subpatterns must
# be irrefutable" rule rejected it with "missing variants Some", naming a
# variant that was right there in the match.
# ---------------------------------------------------------------------------

TOKENS = """
enum Token { TInt(int), TName(str), TOp(str) }
"""


def test_nested_ctor_patterns_covering_the_payload_are_accepted():
    compile_src(TOKENS + """
fn describe(o: Option) -> str {
    match o {
        Some(TInt(n)) => "int",
        Some(TName(s)) => "name",
        Some(TOp(s)) => "op",
        None => "eof"
    }
}

fn main() -> int { 0 }
""")


def test_nested_ctor_patterns_missing_an_inner_variant_rejected():
    msg = reject(TOKENS + """
fn describe(o: Option) -> str {
    match o {
        Some(TInt(n)) => "int",
        Some(TName(s)) => "name",
        None => "eof"
    }
}

fn main() -> int { 0 }
""")
    assert "missing variants Some" in msg


def test_nested_ctor_patterns_completed_by_a_binding_arm_accepted():
    compile_src(TOKENS + """
fn describe(o: Option) -> str {
    match o {
        Some(TInt(n)) => "int",
        Some(other) => "other",
        None => "eof"
    }
}

fn main() -> int { 0 }
""")


def test_two_levels_of_nesting_are_analyzed():
    compile_src("""
enum Inner { A, B }
enum Outer { Wrap(Inner), Empty }

fn f(o: Option) -> int {
    match o {
        Some(Wrap(A)) => 1,
        Some(Wrap(B)) => 2,
        Some(Empty) => 3,
        None => 4
    }
}

fn main() -> int { 0 }
""")


def test_two_levels_of_nesting_with_a_hole_rejected():
    msg = reject("""
enum Inner { A, B }
enum Outer { Wrap(Inner), Empty }

fn f(o: Option) -> int {
    match o {
        Some(Wrap(A)) => 1,
        Some(Empty) => 3,
        None => 4
    }
}

fn main() -> int { 0 }
""")
    assert "missing variants Some" in msg


def test_multi_field_variant_needs_every_combination():
    # Pair(A|B, A|B): three of the four combinations is not coverage.
    msg = reject("""
enum Inner { A, B }
enum Both { Pair(Inner, Inner) }

fn f(b: Both) -> int {
    match b {
        Pair(A, A) => 1,
        Pair(A, B) => 2,
        Pair(B, A) => 3
    }
}

fn main() -> int { 0 }
""")
    assert "missing variants Pair" in msg


def test_multi_field_variant_with_every_combination_accepted():
    compile_src("""
enum Inner { A, B }
enum Both { Pair(Inner, Inner) }

fn f(b: Both) -> int {
    match b {
        Pair(A, A) => 1,
        Pair(A, B) => 2,
        Pair(B, A) => 3,
        Pair(B, B) => 4
    }
}

fn main() -> int { 0 }
""")


def test_payload_less_spelling_of_a_ctor_still_covers_it():
    """`Circle => ...` for a Circle(int) is the payload-less spelling the
    checker has always read as irrefutable; the recursion keeps that."""
    compile_src("""
enum Shape { Circle(r: int), Dot }

fn f(s: Shape) -> int {
    match s {
        Circle => 1,
        Dot => 0
    }
}

fn main() -> int { 0 }
""")
