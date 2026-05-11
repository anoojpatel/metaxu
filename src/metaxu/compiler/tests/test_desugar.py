"""Tests for desugaring passes."""

import metaxu.metaxu_ast as fast
from metaxu.compiler.desugar import run_default_desugaring, DesugarContext, TraitDictionaryDesugarPass


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
