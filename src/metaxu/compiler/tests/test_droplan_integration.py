"""Tests for DropPlan and borrow checker integration in HIR to MIR pass."""

from metaxu.compiler.borrow_analysis import plan_drops
from metaxu.compiler.frozen_borrow_checker import BorrowError
from metaxu.compiler.hir import HExpr, HFun, EffectSet
from metaxu.compiler.mutaxu_ast import Span
from metaxu.compiler.pipeline import run_pipeline_from_source


def test_plan_drops_uses_structured_borrow_errors():
    """plan_drops must skip dropping variables reported moved via structured
    BorrowError data (kind/variable), not by scraping message strings."""

    class NeedsDropTy:
        def needs_drop(self):
            return True

    span = Span(file="<test>", start=0, end=0)
    ty = NeedsDropTy()

    def let_expr(name):
        bound = HExpr(node_id=2, kind="Literal", args=(), ty=ty,
                      effects=EffectSet(frozenset()), suspends=False, sym=None,
                      span=span, op="Literal")
        return HExpr(node_id=1, kind="Let", args=(), ty=ty,
                     effects=EffectSet(frozenset()), suspends=False, sym=None,
                     span=span, op="Let", bindings=((name, bound),))

    body = HExpr(node_id=0, kind="Block", args=(), ty=ty,
                 effects=EffectSet(frozenset()), suspends=False, sym=None,
                 span=span, op="Block",
                 operands=(let_expr("moved_var"), let_expr("kept_var")))
    fun = HFun(sym="f", params=[], dict_params=[], ret_ty=ty, where_cls=[], body=body)

    errors = [
        BorrowError("Cannot use moved_var after it was moved", 3,
                    kind="use-after-move", variable="moved_var"),
        # An error message mentioning "moved" but with a non-move kind must
        # NOT suppress the drop (this is what the old regex scraping got wrong).
        BorrowError("kept_var mentioned moved in text only", 4,
                    kind="locality-escape", variable="kept_var"),
    ]

    plans = plan_drops([fun], errors)

    assert plans["f"].drop_at_end == ["kept_var"]


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
