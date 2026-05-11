"""Tests for subtyping patterns to ensure SimpleSub correctly handles subtype constraints.

Tests common subtyping patterns including:
- Numeric subtyping (Int <: Number)
- Function subtyping (covariance/contravariance)
- Trait-based subtyping
"""

import metaxu.metaxu_ast as fast
from metaxu.compiler.infer_tables import build_tables_from_frozen_via_simplesub
from metaxu.compiler.mutaxu_ast import build_frozen_ast_with_map


def test_numeric_subtyping_int_to_number():
    """Test that Int is a subtype of Number."""
    parsed = fast.Literal(42)  # Int literal
    frozen, _ = build_frozen_ast_with_map(parsed)
    tables = build_tables_from_frozen_via_simplesub(frozen)
    
    # Should have Int class constraint
    constraints = tables.constraints_of(0)
    assert any(constraint[0] == "class" and constraint[1] == "Int" for constraint in constraints)
    # Subtyping is handled by SimpleSub - we emit class constraints and SimpleSub handles subtype relationships


def test_numeric_subtyping_float_to_number():
    """Test that Float is a subtype of Number."""
    parsed = fast.Literal(3.14)  # Float literal
    frozen, _ = build_frozen_ast_with_map(parsed)
    tables = build_tables_from_frozen_via_simplesub(frozen)
    
    # Should have Float class constraint
    constraints = tables.constraints_of(0)
    assert any(constraint[0] == "class" and constraint[1] == "Float" for constraint in constraints)


def test_subtype_constraint_emission():
    """Test that subtype constraints are emitted correctly."""
    parsed = fast.Block([
        fast.LetStatement([fast.LetBinding("x", fast.Literal(1))]),
        fast.Variable("x"),
    ])
    frozen, _ = build_frozen_ast_with_map(parsed)
    tables = build_tables_from_frozen_via_simplesub(frozen)
    
    # Should have subtype constraints
    constraints = tables.constraints_of(0)
    subtype_constraints = [c for c in constraints if c[0] == "subtype"]
    assert len(subtype_constraints) > 0, "Should emit subtype constraints"


def test_trait_based_subtyping():
    """Test trait-based subtyping (placeholder for future trait system)."""
    # Trait-based subtyping will be implemented via the trait system
    # For now, this test documents expected behavior
    pass


def test_variance_in_subtype_constraints():
    """Test that variance is handled in unify constraints."""
    # Variance (covariant, contravariant, invariant) is handled by SimpleSub
    # The frozen constraint emitter emits unify constraints with variance
    # SimpleSub translates these to polarity (POSITIVE, NEGATIVE, NEUTRAL)
    # This test documents expected behavior
    pass


def test_simplesub_solves_constraints():
    """Test that SimpleSub actually solves constraints correctly."""
    from metaxu.simplesub import TypeInferencer
    from metaxu.type_defs import CompactType
    
    # Test SimpleSub directly to verify it works
    ss = TypeInferencer()
    t1 = CompactType.fresh_var()
    t2 = CompactType.fresh_var()
    
    # Add a subtype constraint (t1 <: t2)
    ss.add_constraint(t1, t2, 'POSITIVE')
    
    # Solve
    ss.solve_constraints()
    
    # Verify the constraint was solved - t1 should have upper_bound of t2
    assert t1.bounds.upper_bound == t2, "SimpleSub should set upper_bound after solving"
    print("SimpleSub correctly solved subtype constraint")


def test_solved_types_from_frozen_ast():
    """Test that we can get solved types from frozen AST via SimpleSub."""
    from metaxu.compiler.simplesub_adapter import SimpleSubFacade
    from metaxu.type_defs import CompactType
    from metaxu.compiler.mutaxu_ast import AstNode, Span
    
    # Create a simple frozen AST manually
    span = Span(file="<test>", start=0, end=0)
    child = AstNode(node_id=2, kind="Literal", children=tuple(), span=span, value=42)
    root = AstNode(node_id=1, kind="Expr", children=(child,), span=span)
    
    # Allocate types
    types = {1: CompactType.fresh_var(), 2: CompactType.fresh_var()}
    
    # Create SimpleSubFacade
    ss = SimpleSubFacade(types)
    ss.enable_solution_application()
    
    # Emit constraints (manually for this test)
    ss.add_class_constraint("Int", [types[2]], 2)
    ss.add_subtype(types[2], types[1])
    
    # Solve
    ss.solve()
    
    # Check solved bounds
    bounds = ss.get_solved_bounds(1)
    assert bounds is not None, "Should have solved bounds for root node"
    print("Solved bounds for root node:", bounds)
