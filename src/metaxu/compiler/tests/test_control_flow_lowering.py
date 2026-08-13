"""Control-flow lowering tests: nested ifs, if-in-else, sequential ifs, if-in-loop.

These are real end-to-end assertions (HIR -> MIR -> interpreter -> value),
regression tests for the label-allocation bug where then/else/join labels were
computed from len(blocks) *before* lowering the arms, so any arm that created
blocks invalidated the labels and produced infinite br loops (bb3 -> br 4,
bb4 -> br 3).
"""
from __future__ import annotations

import pytest

from metaxu.compiler.hir import HExpr, HFun
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import MirInterpreter
from metaxu.compiler.types import EffectSet


# ---------------------------------------------------------------------------
# Hand-built HIR helpers
# ---------------------------------------------------------------------------

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


def if_(cond: HExpr, then_ops, else_ops=()) -> HExpr:
    return hx("If", cond=cond, then_ops=tuple(then_ops), else_ops=tuple(else_ops))


def fn(name: str, params: list[str], body: HExpr) -> HFun:
    return HFun(sym=name, params=[(p, "Int") for p in params], dict_params=[],
                ret_ty="Int", where_cls=[], body=body)


def run(hfun: HFun, args: list):
    mir = lower_hir_to_mir([hfun])
    interp = MirInterpreter()
    interp.load(mir)
    return interp.call(str(hfun.sym), args)


# ---------------------------------------------------------------------------
# Nested if in the then-arm (the original hang)
# ---------------------------------------------------------------------------

def test_nested_if_in_then():
    # f(c) = if c { if c { 1 } else { 2 } } else { 3 }
    body = if_(var("c"),
               [if_(var("c"), [lit(1)], [lit(2)])],
               [lit(3)])
    f = fn("f", ["c"], body)
    assert run(f, [1]) == 1
    assert run(f, [0]) == 3


def test_nested_if_terminators_are_valid():
    """Every branch target must be a real block index (no dangling labels)."""
    body = if_(var("c"),
               [if_(var("c"), [lit(1)], [lit(2)])],
               [lit(3)])
    mir = lower_hir_to_mir([fn("f", ["c"], body)])
    f = mir[0]
    n = len(f.blocks)
    for b in f.blocks:
        term = b.term
        if term[0] == "br":
            assert 0 <= term[1] < n
        elif term[0] == "br_if":
            assert 0 <= term[2] < n and 0 <= term[3] < n
        else:
            assert term[0] in ("ret", "unreachable")


def test_nested_if_via_pipeline():
    """The exact audit repro, end-to-end from source (used to hang)."""
    from metaxu.compiler.pipeline import build_context_from_source
    from metaxu.compiler.hir import HIRBuilder

    src = "fn f(c: Int) -> Int { if (c) { if (c) { 1 } else { 2 } } else { 3 } }"
    ctx = build_context_from_source(src)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    mir = lower_hir_to_mir(hir)
    interp = MirInterpreter()
    interp.load(mir)
    assert interp.call("f", [1]) == 1
    assert interp.call("f", [0]) == 3


# ---------------------------------------------------------------------------
# Nested if in the else-arm
# ---------------------------------------------------------------------------

def test_if_in_else():
    # f(a, b) = if a { 1 } else { if b { 2 } else { 3 } }
    body = if_(var("a"),
               [lit(1)],
               [if_(var("b"), [lit(2)], [lit(3)])])
    f = fn("f", ["a", "b"], body)
    assert run(f, [1, 0]) == 1
    assert run(f, [1, 1]) == 1
    assert run(f, [0, 1]) == 2
    assert run(f, [0, 0]) == 3


def test_if_in_both_arms():
    # f(a, b) = if a { if b { 11 } else { 10 } } else { if b { 1 } else { 0 } }
    body = if_(var("a"),
               [if_(var("b"), [lit(11)], [lit(10)])],
               [if_(var("b"), [lit(1)], [lit(0)])])
    f = fn("f", ["a", "b"], body)
    assert run(f, [1, 1]) == 11
    assert run(f, [1, 0]) == 10
    assert run(f, [0, 1]) == 1
    assert run(f, [0, 0]) == 0


# ---------------------------------------------------------------------------
# Sequential ifs
# ---------------------------------------------------------------------------

def test_sequential_ifs():
    # f(a, b) = { if a { 1 } else { 2 }; if b { 10 } else { 20 } }
    # Block value is the last expression's value.
    body = hx("Block", operands=(
        if_(var("a"), [lit(1)], [lit(2)]),
        if_(var("b"), [lit(10)], [lit(20)]),
    ))
    f = fn("f", ["a", "b"], body)
    assert run(f, [1, 1]) == 10
    assert run(f, [0, 0]) == 20
    assert run(f, [1, 0]) == 20


def test_sequential_ifs_first_feeds_second():
    # f(a) = { let x = if a { 10 } else { 20 }; x + (if a { 1 } else { 2 }) }
    body = hx("Block", operands=(
        hx("Let", bindings=(("x", if_(var("a"), [lit(10)], [lit(20)])),)),
        binop("+", var("x"), if_(var("a"), [lit(1)], [lit(2)])),
    ))
    f = fn("f", ["a"], body)
    assert run(f, [1]) == 11
    assert run(f, [0]) == 22


def test_if_without_else_returns_unit_on_false():
    from metaxu.compiler.mir_interp import MxUnit
    body = if_(var("a"), [lit(1)])
    f = fn("f", ["a"], body)
    assert run(f, [1]) == 1
    assert isinstance(run(f, [0]), MxUnit)


# ---------------------------------------------------------------------------
# If inside a loop body
# ---------------------------------------------------------------------------

def test_if_in_loop():
    # f(n) = { let s = 0; let i = n;
    #          while i > 0 { s = s + (if i % 2 == 0 { i } else { 0 }); i = i - 1 };
    #          s }
    # Sum of even numbers in 1..=n. f(4) = 4 + 2 = 6, f(5) = 4 + 2 = 6, f(6) = 12.
    even_add = hx("Assign", var_name="s",
                  assign_value=binop("+", var("s"),
                                     if_(binop("==", binop("%", var("i"), lit(2)), lit(0)),
                                         [var("i")], [lit(0)])))
    decr = hx("Assign", var_name="i", assign_value=binop("-", var("i"), lit(1)))
    body = hx("Block", operands=(
        hx("Let", bindings=(("s", lit(0)),)),
        hx("Let", bindings=(("i", var("n")),)),
        hx("While", cond=binop(">", var("i"), lit(0)), loop_body=(even_add, decr)),
        var("s"),
    ))
    f = fn("f", ["n"], body)
    assert run(f, [4]) == 6
    assert run(f, [5]) == 6
    assert run(f, [6]) == 12
    assert run(f, [0]) == 0
