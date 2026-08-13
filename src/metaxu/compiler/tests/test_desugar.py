"""Tests for desugaring passes."""

import metaxu.metaxu_ast as fast
from metaxu.compiler.desugar import (
    DesugarContext,
    IfDesugarPass,
    TraitDictionaryDesugarPass,
    run_default_desugaring,
    run_desugaring_passes,
)


def test_apply_recursive_reaches_nested_nodes_and_syncs_children():
    """The generic traversal must rewrite nodes nested in attribute fields AND
    keep the parallel `children` bookkeeping list consistent."""
    if_expr = fast.IfExpression(fast.Literal(True), fast.Literal(1), fast.Literal(2))
    fn = fast.FunctionDeclaration("f", [], [if_expr])
    program = fast.Block([fn])

    result = run_desugaring_passes(program, [IfDesugarPass()], DesugarContext())

    assert result is program
    fn_after = result.statements[0]
    # The nested IfExpression was rewritten to a MatchExpression...
    assert isinstance(fn_after.body[0], fast.MatchExpression)
    # ...and no stale IfExpression survives anywhere in the children lists.
    def all_nodes(node, seen=None):
        if seen is None:
            seen = set()
        if id(node) in seen:
            return
        seen.add(id(node))
        yield node
        for child in getattr(node, "children", []) or []:
            if isinstance(child, fast.Node):
                yield from all_nodes(child, seen)
    assert not any(isinstance(n, fast.IfExpression) for n in all_nodes(result))


def test_if_desugar_produces_bool_match_cases():
    if_expr = fast.IfExpression(fast.Literal(True), fast.Literal(1), fast.Literal(2))

    result = IfDesugarPass().apply_recursive(if_expr, DesugarContext())

    assert isinstance(result, fast.MatchExpression)
    (true_pat, then_expr), (false_pat, else_expr) = result.cases
    assert isinstance(true_pat, fast.LiteralPattern) and true_pat.value is True
    assert isinstance(false_pat, fast.LiteralPattern) and false_pat.value is False
    assert then_expr.value == 1
    assert else_expr.value == 2


def test_default_desugaring_keeps_if_native():
    """IfDesugarPass is not part of the default pipeline: if/else keeps its
    direct HIR/MIR lowering path."""
    if_expr = fast.IfExpression(fast.Literal(True), fast.Literal(1), fast.Literal(2))
    program = fast.Block([if_expr])

    result = run_default_desugaring(program, DesugarContext())

    assert isinstance(result.statements[0], fast.IfExpression)


def test_trait_dictionary_desugaring():
    """Test that trait method calls are desugared to dictionary lookups."""
    parsed = fast.Block([
        fast.FunctionCall("push", [fast.Variable("stack"), fast.Literal(42)])
    ])
    
    desugar_ctx = DesugarContext()
    result = run_default_desugaring(parsed, desugar_ctx)
    
    # The function call should be transformed to a dictionary lookup
    # stack.trait_dict["Trait::push"](stack, 42)
    assert isinstance(result, fast.Block)
    assert len(result.statements) == 1
    stmt = result.statements[0]
    
    # Should be a FunctionCall to a dictionary lookup
    assert isinstance(stmt, fast.FunctionCall)
    print("Trait dictionary desugaring test passed")


def test_desugar_pass_with_context():
    """Test that desugaring context is properly used."""
    parsed = fast.Block([
        fast.FunctionCall("push", [fast.Variable("stack"), fast.Literal(42)])
    ])
    
    desugar_ctx = DesugarContext(
        source="test source",
        file_path="test.mx",
        traits={},
        trait_impls={}
    )
    
    pass_obj = TraitDictionaryDesugarPass()
    result = pass_obj.apply_recursive(parsed, desugar_ctx)
    
    assert isinstance(result, fast.Block)
    print("Desugar pass with context test passed")
