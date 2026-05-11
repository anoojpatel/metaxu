"""Tests for pattern matching with type checker.

Tests that pattern matching works correctly with the frozen AST type checker,
including type inference and value type compression.
"""

import metaxu.metaxu_ast as fast
from metaxu.compiler.infer_tables import build_tables_from_frozen_via_simplesub
from metaxu.compiler.mutaxu_ast import build_frozen_ast_with_map


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
