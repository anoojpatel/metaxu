from __future__ import annotations

from pathlib import Path

from metaxu.compiler.mir import dump_mir
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.hir import HFun, HExpr, ModeInfo
from metaxu.compiler.types import EffectSet


def make_binop_fun() -> HFun:
    # Build HIR: body = (1 + 2)
    left = HExpr(
        node_id=2,
        kind="Expr",
        args=tuple(),
        ty="Int",
        effects=EffectSet(frozenset()),
        suspends=False,
        sym=None,
        span=None,  # not used in this test
        op="Literal",
        literal=1,
    )
    right = HExpr(
        node_id=3,
        kind="Expr",
        args=tuple(),
        ty="Int",
        effects=EffectSet(frozenset()),
        suspends=False,
        sym=None,
        span=None,
        op="Literal",
        literal=2,
    )
    body = HExpr(
        node_id=1,
        kind="Expr",
        args=tuple(),
        ty="Int",
        effects=EffectSet(frozenset()),
        suspends=False,
        sym=None,
        span=None,
        op="BinOp",
        binop="+",
        left=left,
        right=right,
    )
    return HFun(sym="main", params=[], dict_params=[], ret_ty="Int", where_cls=[], body=body)


def test_mir_binop_golden():
    f = make_binop_fun()
    mir_funcs = lower_hir_to_mir([f])
    got = dump_mir(mir_funcs)
    golden = (Path(__file__).parent / "golden" / "sample2.mir.txt").read_text()
    assert got.strip() == golden.strip()
