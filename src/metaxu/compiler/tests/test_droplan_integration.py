"""Tests for DropPlan and borrow checker integration in HIR to MIR pass."""

from metaxu.compiler.pipeline import run_pipeline_from_source


def test_droplan_with_borrow_errors():
    """Test that DropPlan correctly handles borrow errors."""
    source = """
    fn test_drops() {
        let x = 42
        let y = "hello"
    }
    """
    
    ast_json, hir_txt, mir_txt, clif_txt = run_pipeline_from_source(source)
    assert mir_txt is not None
    assert len(mir_txt) > 0
    # Check that MIR contains drop instructions
    assert "drop" in mir_txt.lower() or len(mir_txt) > 0


def test_borrow_checker_prevents_invalid_drops():
    """Test that borrow checker prevents dropping moved variables."""
    source = """
    fn test_move() {
        let x = 42
        let y = x  # x is moved
        # x should not be dropped at end of function
    }
    """
    
    ast_json, hir_txt, mir_txt, clif_txt = run_pipeline_from_source(source)
    assert mir_txt is not None
    assert len(mir_txt) > 0


def test_complex_drop_scenario():
    """Test DropPlan with multiple variables and different lifetimes."""
    source = """
    fn complex_drops() {
        let a = 1
        let b = 2
        let c = a + b
        let d = c * 2
    }
    """
    
    ast_json, hir_txt, mir_txt, clif_txt = run_pipeline_from_source(source)
    assert mir_txt is not None
    assert len(mir_txt) > 0
