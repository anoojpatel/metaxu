"""While-loop lowering and interpretation tests.

Loops lower to header/body/exit blocks with a back-edge from the body to the
header; assignments write through to the variable's runtime slot so the header
re-reads updated values each iteration.
"""
from __future__ import annotations

import pytest

from metaxu.compiler.hir import HExpr, HFun
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import MirInterpreter, MxUnit
from metaxu.compiler.types import EffectSet


def hx(op: str, ty="Int", **kw) -> HExpr:
    return HExpr(node_id=0, kind="Expr", args=(), ty=ty,
                 effects=EffectSet(frozenset()), suspends=False, sym=None,
                 span=None, op=op, **kw)


def lit(v) -> HExpr:
    return hx("Literal", literal=v)


def var(name: str) -> HExpr:
    return hx("Var", var_name=name)


def binop(op: str, l: HExpr, r: HExpr) -> HExpr:
    return hx("BinOp", binop=op, left=l, right=r)


def assign(name: str, value: HExpr) -> HExpr:
    return hx("Assign", var_name=name, assign_value=value)


def let(name: str, value: HExpr) -> HExpr:
    return hx("Let", bindings=((name, value),))


def while_(cond: HExpr, body) -> HExpr:
    return hx("While", cond=cond, loop_body=tuple(body))


def fn(name: str, params: list[str], body: HExpr) -> HFun:
    return HFun(sym=name, params=[(p, "Int") for p in params], dict_params=[],
                ret_ty="Int", where_cls=[], body=body)


def run(hfun: HFun, args: list):
    mir = lower_hir_to_mir([hfun])
    interp = MirInterpreter()
    interp.load(mir)
    return interp.call(str(hfun.sym), args)


# ---------------------------------------------------------------------------
# Countdown
# ---------------------------------------------------------------------------

def test_while_countdown():
    # f(n) = { let i = n; while i > 0 { i = i - 1 }; i }  -> 0
    body = hx("Block", operands=(
        let("i", var("n")),
        while_(binop(">", var("i"), lit(0)),
               [assign("i", binop("-", var("i"), lit(1)))]),
        var("i"),
    ))
    f = fn("countdown", ["n"], body)
    assert run(f, [5]) == 0
    assert run(f, [0]) == 0
    assert run(f, [1]) == 0


def test_while_counts_iterations():
    # f(n) = { let i = n; let c = 0; while i > 0 { i = i - 1; c = c + 1 }; c }
    body = hx("Block", operands=(
        let("i", var("n")),
        let("c", lit(0)),
        while_(binop(">", var("i"), lit(0)),
               [assign("i", binop("-", var("i"), lit(1))),
                assign("c", binop("+", var("c"), lit(1)))]),
        var("c"),
    ))
    f = fn("count", ["n"], body)
    assert run(f, [7]) == 7
    assert run(f, [0]) == 0


# ---------------------------------------------------------------------------
# Sum loop
# ---------------------------------------------------------------------------

def test_while_sum_loop():
    # sum_to(n) = { let s = 0; let i = 1; while i <= n { s = s + i; i = i + 1 }; s }
    body = hx("Block", operands=(
        let("s", lit(0)),
        let("i", lit(1)),
        while_(binop("<=", var("i"), var("n")),
               [assign("s", binop("+", var("s"), var("i"))),
                assign("i", binop("+", var("i"), lit(1)))]),
        var("s"),
    ))
    f = fn("sum_to", ["n"], body)
    assert run(f, [10]) == 55
    assert run(f, [1]) == 1
    assert run(f, [0]) == 0


def test_while_body_never_entered():
    # f() = { let x = 1; while 0 { x = 99 }; x }  -> 1
    body = hx("Block", operands=(
        let("x", lit(1)),
        while_(lit(0), [assign("x", lit(99))]),
        var("x"),
    ))
    assert run(fn("f", [], body), []) == 1


def test_while_evaluates_to_unit():
    body = while_(lit(0), [])
    result = run(fn("f", [], body), [])
    assert isinstance(result, MxUnit)


# ---------------------------------------------------------------------------
# Nested while loops
# ---------------------------------------------------------------------------

def test_nested_while_multiplication():
    # mul(a, b) = { let s = 0; let i = a;
    #               while i > 0 { let j = b; while j > 0 { s = s + 1; j = j - 1 };
    #                             i = i - 1 };
    #               s }
    inner = while_(binop(">", var("j"), lit(0)),
                   [assign("s", binop("+", var("s"), lit(1))),
                    assign("j", binop("-", var("j"), lit(1)))])
    body = hx("Block", operands=(
        let("s", lit(0)),
        let("i", var("a")),
        while_(binop(">", var("i"), lit(0)),
               [let("j", var("b")),
                inner,
                assign("i", binop("-", var("i"), lit(1)))]),
        var("s"),
    ))
    f = fn("mul", ["a", "b"], body)
    assert run(f, [3, 4]) == 12
    assert run(f, [4, 3]) == 12
    assert run(f, [0, 9]) == 0
    assert run(f, [9, 0]) == 0


# ---------------------------------------------------------------------------
# AST -> HIR path for WhileStatement (parser fix for statements is landing
# separately, so build the AST by hand and go through HIRBuilder)
# ---------------------------------------------------------------------------

def test_while_from_ast_via_hirbuilder():
    import metaxu.metaxu_ast as fast
    from metaxu.compiler.mutaxu_ast import build_frozen_ast_with_map
    from metaxu.compiler.infer_tables import build_tables_from_frozen_via_simplesub
    from metaxu.compiler.hir import HIRBuilder

    # fn f(n: Int) -> Int { let i = n; while (i > 0) { i = i - 1 }; i }
    # LetStatement/LetBinding construction differs across parser versions, so
    # use an Assignment to introduce `i` (first assignment creates the slot).
    fdecl = fast.FunctionDeclaration(
        "f",
        [fast.Parameter("n", None)],
        [
            fast.Assignment("i", fast.Variable("n")),
            fast.WhileStatement(
                fast.ComparisonExpression(fast.Variable("i"), ">", fast.Literal(0)),
                fast.Block([
                    fast.Assignment("i", fast.BinaryOperation(fast.Variable("i"), "-", fast.Literal(1))),
                ]),
            ),
            fast.Variable("i"),
        ],
    )
    frozen, id_map = build_frozen_ast_with_map(fdecl)
    tables = build_tables_from_frozen_via_simplesub(frozen)
    hir = HIRBuilder(tables, id_map=id_map).build(frozen)
    mir = lower_hir_to_mir(hir)
    interp = MirInterpreter()
    interp.load(mir)
    assert interp.call("f", [5]) == 0
