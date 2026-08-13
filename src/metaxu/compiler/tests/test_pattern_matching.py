"""Tests for pattern matching.

Part 1: type-checker smoke tests (frozen AST -> inference tables).
Part 2: genuine end-to-end semantics tests: patterns compile to a decision
tree of test-and-branch MIR blocks and the interpreter produces the right
values (literal, wildcard, variable binding, Some/None constructors, nested
constructors, first-match-wins ordering, and match failure).
"""

import pytest

import metaxu.metaxu_ast as fast
from metaxu.compiler.hir import HExpr, HFun, HPattern, HIRBuilder
from metaxu.compiler.infer_tables import build_tables_from_frozen_via_simplesub
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import MirInterpreter, MxVariant, InterpError
from metaxu.compiler.mutaxu_ast import build_frozen_ast_with_map
from metaxu.compiler.types import EffectSet


def test_pattern_matching_some_none():
    """Test pattern matching on Option type (Some/None)."""
    parsed = fast.Block([
        fast.MatchExpression(
            fast.Variable("x"),
            [
                (fast.VariablePattern("val"), fast.Variable("val")),
                (fast.WildcardPattern(), fast.Literal(0))
            ]
        )
    ])
    frozen, _ = build_frozen_ast_with_map(parsed)
    tables = build_tables_from_frozen_via_simplesub(frozen)
    
    # Should have types for match expression
    assert 1 in tables.types  # Match expression
    print("Pattern matching test passed - types inferred")


def test_pattern_matching_with_literal():
    """Test pattern matching with literal values."""
    parsed = fast.Block([
        fast.MatchExpression(
            fast.Literal(42),
            [
                (fast.LiteralPattern(fast.Literal(42)), fast.Literal(1)),
                (fast.LiteralPattern(fast.Literal(0)), fast.Literal(0))
            ]
        )
    ])
    frozen, _ = build_frozen_ast_with_map(parsed)
    tables = build_tables_from_frozen_via_simplesub(frozen)
    
    # Should have types for match expression
    assert 1 in tables.types  # Match expression
    print("Literal pattern matching test passed")


def test_pattern_matching_type_inference():
    """Test that pattern matching infers types correctly."""
    parsed = fast.Block([
        fast.MatchExpression(
            fast.Literal(42),
            [
                (fast.VariablePattern("x"), fast.Variable("x")),
                (fast.WildcardPattern(), fast.Literal(0))
            ]
        )
    ])
    frozen, _ = build_frozen_ast_with_map(parsed)
    tables = build_tables_from_frozen_via_simplesub(frozen)
    
    # Should infer types for nodes
    print(f"Types in tables: {list(tables.types.keys())}")
    assert len(tables.types) > 0, "Should have types for some nodes"
    print("Pattern matching type inference test passed")


# ---------------------------------------------------------------------------
# End-to-end: HIR match -> MIR decision tree -> interpreter
# ---------------------------------------------------------------------------

def hx(op: str, ty="Int", **kw) -> HExpr:
    return HExpr(node_id=0, kind="Expr", args=(), ty=ty,
                 effects=EffectSet(frozenset()), suspends=False, sym=None,
                 span=None, op=op, **kw)


def lit(v) -> HExpr:
    return hx("Literal", literal=v)


def var(name: str) -> HExpr:
    return hx("Var", var_name=name)


def match(scrutinee: HExpr, arms) -> HExpr:
    return hx("Match", scrutinee=scrutinee, match_arms=tuple(arms))


def mkv(enum: str, tag: str, *payload: HExpr) -> HExpr:
    return hx("MakeVariant", enum_name=enum, variant_name=tag, operands=tuple(payload))


def p_lit(v) -> HPattern:
    return HPattern(kind="literal", value=v)


def p_var(name: str) -> HPattern:
    return HPattern(kind="var", name=name)


def p_wild() -> HPattern:
    return HPattern(kind="wildcard")


def p_ctor(tag: str, *subs: HPattern, enum: str | None = None) -> HPattern:
    return HPattern(kind="ctor", name=tag, enum_name=enum, subpatterns=tuple(subs))


def run_body(body: HExpr, params: list[str] = [], args: list = []):
    f = HFun(sym="f", params=[(p, "Int") for p in params], dict_params=[],
             ret_ty="Int", where_cls=[], body=body)
    mir = lower_hir_to_mir([f])
    interp = MirInterpreter()
    interp.load(mir)
    return interp.call("f", args)


def test_match_literal_e2e():
    body = match(var("x"), [
        (p_lit(1), lit(10)),
        (p_lit(2), lit(20)),
        (p_wild(), lit(99)),
    ])
    assert run_body(body, ["x"], [1]) == 10
    assert run_body(body, ["x"], [2]) == 20
    assert run_body(body, ["x"], [3]) == 99


def test_match_wildcard_e2e():
    body = match(lit(5), [
        (p_lit(1), lit(10)),
        (p_wild(), lit(99)),
    ])
    assert run_body(body) == 99


def test_match_variable_binding_e2e():
    # match x { y => y + 1 }
    body = match(var("x"), [
        (p_var("y"), hx("BinOp", binop="+", left=var("y"), right=lit(1))),
    ])
    assert run_body(body, ["x"], [7]) == 8


def test_match_some_none_e2e():
    # match opt { Some(v) => v, None => 0 }
    def match_on(opt_expr: HExpr) -> HExpr:
        return match(opt_expr, [
            (p_ctor("Some", p_var("v"), enum="Option"), var("v")),
            (p_ctor("None", enum="Option"), lit(0)),
        ])
    assert run_body(match_on(mkv("Option", "Some", lit(5)))) == 5
    assert run_body(match_on(mkv("Option", "None"))) == 0


def test_match_multi_arm_ordering_first_wins():
    # match 1 { _ => 100, 1 => 1 }  -- top-to-bottom, first match wins
    body = match(lit(1), [
        (p_wild(), lit(100)),
        (p_lit(1), lit(1)),
    ])
    assert run_body(body) == 100


def test_match_literal_before_wildcard():
    # ...and the specific arm wins when it comes first
    body = match(lit(1), [
        (p_lit(1), lit(1)),
        (p_wild(), lit(100)),
    ])
    assert run_body(body) == 1


def test_match_nested_ctor_patterns():
    # match Cons(1, Cons(2, Nil)) { Cons(h, Cons(h2, _)) => h + h2, _ => 0 }
    lst = mkv("List", "Cons", lit(1), mkv("List", "Cons", lit(2), mkv("List", "Nil")))
    body = match(lst, [
        (p_ctor("Cons", p_var("h"), p_ctor("Cons", p_var("h2"), p_wild())),
         hx("BinOp", binop="+", left=var("h"), right=var("h2"))),
        (p_wild(), lit(0)),
    ])
    assert run_body(body) == 3


def test_match_nested_ctor_falls_through():
    # A single-element list does not match Cons(h, Cons(...)), falls to _
    lst = mkv("List", "Cons", lit(1), mkv("List", "Nil"))
    body = match(lst, [
        (p_ctor("Cons", p_var("h"), p_ctor("Cons", p_var("h2"), p_wild())),
         hx("BinOp", binop="+", left=var("h"), right=var("h2"))),
        (p_wild(), lit(0)),
    ])
    assert run_body(body) == 0


def test_match_no_arm_matches_raises():
    body = match(lit(3), [
        (p_lit(1), lit(10)),
        (p_lit(2), lit(20)),
    ])
    with pytest.raises(InterpError, match="match failure"):
        run_body(body)


def test_match_bindings_do_not_leak_across_arms():
    # match x { 1 => 10, y => y }: the second arm's y binds the scrutinee,
    # and the first arm's compilation must not leave stale bindings around.
    body = match(var("x"), [
        (p_lit(1), lit(10)),
        (p_var("y"), var("y")),
    ])
    assert run_body(body, ["x"], [1]) == 10
    assert run_body(body, ["x"], [42]) == 42


def test_match_guards_scrutinee_evaluated_once():
    """The scrutinee expression is lowered once, not once per arm."""
    body = match(var("x"), [
        (p_lit(1), lit(10)),
        (p_lit(2), lit(20)),
        (p_wild(), lit(99)),
    ])
    f = HFun(sym="f", params=[("x", "Int")], dict_params=[], ret_ty="Int",
             where_cls=[], body=body)
    mir = lower_hir_to_mir([f])
    # No op should re-evaluate the scrutinee: 'x' is a param, and every
    # comparison references it directly.
    all_ops = [op for b in mir[0].blocks for op in b.ops]
    eq_ops = [op for op in all_ops if op[0] == "let" and op[2] == ("binop", "==")]
    assert len(eq_ops) == 2
    assert all(op[3][0] == "x" for op in eq_ops)


# ---------------------------------------------------------------------------
# End-to-end: AST patterns -> HIRBuilder -> MIR -> interpreter
# ---------------------------------------------------------------------------

def _run_ast_main(stmts, enum_defs=()):
    fdecl = fast.FunctionDeclaration("main", [], list(stmts))
    program = fast.Block(list(enum_defs) + [fdecl])
    frozen, id_map = build_frozen_ast_with_map(program)
    tables = build_tables_from_frozen_via_simplesub(frozen)
    hir = HIRBuilder(tables, id_map=id_map).build(frozen)
    mir = lower_hir_to_mir(hir)
    interp = MirInterpreter()
    interp.load(mir)
    return interp.call("main", [])


def test_ast_match_literal_and_wildcard():
    m = fast.MatchExpression(fast.Literal(2), [
        (fast.LiteralPattern(fast.Literal(1)), fast.Literal(10)),
        (fast.LiteralPattern(fast.Literal(2)), fast.Literal(20)),
        (fast.WildcardPattern(), fast.Literal(99)),
    ])
    assert _run_ast_main([m]) == 20


def test_ast_match_variable_pattern_binds():
    m = fast.MatchExpression(fast.Literal(41), [
        (fast.VariablePattern("v"),
         fast.BinaryOperation(fast.Variable("v"), "+", fast.Literal(1))),
    ])
    assert _run_ast_main([m]) == 42


def test_ast_match_variant_pattern():
    enum = fast.EnumDefinition("Option", [
        fast.VariantDefinition("Some", [("value", "Int")]),
        fast.VariantDefinition("None", []),
    ])
    m = fast.MatchExpression(
        fast.FunctionCall("Some", [fast.Literal(9)]),
        [
            (fast.VariantPattern("Option", "Some", [fast.VariablePattern("v")]),
             fast.Variable("v")),
            (fast.VariantPattern("Option", "None", []), fast.Literal(0)),
        ],
    )
    assert _run_ast_main([m], [enum]) == 9


def test_ast_match_desugared_if_shape():
    """IfDesugarPass rewrites if/else into match with True/False literal
    patterns; that shape must execute correctly."""
    m = fast.MatchExpression(fast.Literal(True), [
        (fast.LiteralPattern(True), fast.Literal(1)),
        (fast.LiteralPattern(False), fast.Literal(2)),
    ])
    assert _run_ast_main([m]) == 1
    m2 = fast.MatchExpression(fast.Literal(False), [
        (fast.LiteralPattern(True), fast.Literal(1)),
        (fast.LiteralPattern(False), fast.Literal(2)),
    ])
    assert _run_ast_main([m2]) == 2
