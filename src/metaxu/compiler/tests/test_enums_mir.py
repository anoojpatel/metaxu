"""Enum/variant tests: MxVariant runtime value, make_variant / variant_tag /
variant_field MIR ops, HIR MakeVariant lowering, and enum constructors coming
from the AST (FunctionCall to a known variant, VariantInstance).
"""
from __future__ import annotations

import pytest

from metaxu.compiler.hir import HExpr, HFun
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir import MirBlock, MirFunc
from metaxu.compiler.mir_interp import MirInterpreter, MxVariant, InterpError
from metaxu.compiler.types import EffectSet


def hx(op: str, ty="Int", **kw) -> HExpr:
    return HExpr(node_id=0, kind="Expr", args=(), ty=ty,
                 effects=EffectSet(frozenset()), suspends=False, sym=None,
                 span=None, op=op, **kw)


def fn(name: str, params: list[str], body: HExpr) -> HFun:
    return HFun(sym=name, params=[(p, "Int") for p in params], dict_params=[],
                ret_ty="Int", where_cls=[], body=body)


def run(hfuns: list[HFun], entry: str, args: list):
    mir = lower_hir_to_mir(hfuns)
    interp = MirInterpreter()
    interp.load(mir)
    return interp.call(entry, args)


# ---------------------------------------------------------------------------
# Raw MIR ops
# ---------------------------------------------------------------------------

def test_make_variant_op():
    f = MirFunc(name="f", ty_sig=None, suspending=False, blocks=[
        MirBlock(ops=[
            ("let", "x", ("const", 5), ()),
            ("let", "v", ("make_variant", "Option", "Some"), ("x",)),
        ], term=("ret", "v")),
    ])
    interp = MirInterpreter()
    interp.load([f])
    result = interp.call("f", [])
    assert isinstance(result, MxVariant)
    assert result.enum_name == "Option"
    assert result.tag == "Some"
    assert result.fields == (5,)


def test_variant_tag_and_field_ops():
    f = MirFunc(name="f", ty_sig=None, suspending=False, blocks=[
        MirBlock(ops=[
            ("let", "h", ("const", 1), ()),
            ("let", "t", ("const", 2), ()),
            ("let", "v", ("make_variant", "List", "Cons"), ("h", "t")),
            ("let", "tag", ("variant_tag",), ("v",)),
            ("let", "f0", ("variant_field", 0), ("v",)),
            ("let", "f1", ("variant_field", 1), ("v",)),
            ("let", "s", ("binop", "+"), ("f0", "f1")),
        ], term=("ret", "s")),
    ])
    interp = MirInterpreter()
    interp.load([f])
    assert interp.call("f", []) == 3


def test_variant_tag_on_non_variant_raises():
    f = MirFunc(name="f", ty_sig=None, suspending=False, blocks=[
        MirBlock(ops=[
            ("let", "x", ("const", 5), ()),
            ("let", "tag", ("variant_tag",), ("x",)),
        ], term=("ret", "tag")),
    ])
    interp = MirInterpreter()
    interp.load([f])
    with pytest.raises(InterpError, match="variant_tag"):
        interp.call("f", [])


def test_variant_field_out_of_range_raises():
    f = MirFunc(name="f", ty_sig=None, suspending=False, blocks=[
        MirBlock(ops=[
            ("let", "v", ("make_variant", "Option", "None"), ()),
            ("let", "x", ("variant_field", 0), ("v",)),
        ], term=("ret", "x")),
    ])
    interp = MirInterpreter()
    interp.load([f])
    with pytest.raises(InterpError, match="out of range"):
        interp.call("f", [])


def test_variant_repr():
    assert repr(MxVariant("Option", "None")) == "Option::None"
    assert repr(MxVariant("Option", "Some", (5,))) == "Option::Some(5)"


# ---------------------------------------------------------------------------
# HIR MakeVariant lowering
# ---------------------------------------------------------------------------

def test_hir_make_variant_lowering():
    body = hx("MakeVariant", enum_name="Option", variant_name="Some",
              operands=(hx("Literal", literal=42),))
    result = run([fn("some42", [], body)], "some42", [])
    assert isinstance(result, MxVariant)
    assert result.tag == "Some"
    assert result.fields == (42,)


def test_hir_make_variant_nested():
    # Cons(1, Cons(2, Nil))
    nil = hx("MakeVariant", enum_name="List", variant_name="Nil", operands=())
    inner = hx("MakeVariant", enum_name="List", variant_name="Cons",
               operands=(hx("Literal", literal=2), nil))
    outer = hx("MakeVariant", enum_name="List", variant_name="Cons",
               operands=(hx("Literal", literal=1), inner))
    result = run([fn("l", [], outer)], "l", [])
    assert result.tag == "Cons"
    assert result.fields[0] == 1
    assert result.fields[1].tag == "Cons"
    assert result.fields[1].fields[0] == 2
    assert result.fields[1].fields[1].tag == "Nil"


# ---------------------------------------------------------------------------
# AST -> HIR: enum constructor calls
# ---------------------------------------------------------------------------

def _build_hir_from_ast(fdecl_body, enum_defs):
    import metaxu.metaxu_ast as fast
    from metaxu.compiler.mutaxu_ast import build_frozen_ast_with_map
    from metaxu.compiler.infer_tables import build_tables_from_frozen_via_simplesub
    from metaxu.compiler.hir import HIRBuilder

    fdecl = fast.FunctionDeclaration("main", [], fdecl_body)
    program = fast.Block(list(enum_defs) + [fdecl])
    frozen, id_map = build_frozen_ast_with_map(program)
    tables = build_tables_from_frozen_via_simplesub(frozen)
    return HIRBuilder(tables, id_map=id_map).build(frozen)


def test_enum_ctor_via_function_call():
    """`Some(5)` as a bare FunctionCall resolves to a variant constructor when
    an EnumDefinition declares the variant."""
    import metaxu.metaxu_ast as fast

    enum = fast.EnumDefinition("Option", [
        fast.VariantDefinition("Some", [("value", "Int")]),
        fast.VariantDefinition("None", []),
    ])
    hir = _build_hir_from_ast([fast.FunctionCall("Some", [fast.Literal(5)])], [enum])
    mir = lower_hir_to_mir(hir)
    interp = MirInterpreter()
    interp.load(mir)
    result = interp.call("main", [])
    assert isinstance(result, MxVariant)
    assert result.enum_name == "Option"
    assert result.tag == "Some"
    assert result.fields == (5,)


def test_enum_ctor_via_variant_instance():
    """`Option::Some(value=7)` (VariantInstance node) constructs a variant."""
    import metaxu.metaxu_ast as fast

    vi = fast.VariantInstance("Option", "Some", [("value", fast.Literal(7))])
    hir = _build_hir_from_ast([vi], [])
    mir = lower_hir_to_mir(hir)
    interp = MirInterpreter()
    interp.load(mir)
    result = interp.call("main", [])
    assert isinstance(result, MxVariant)
    assert result.enum_name == "Option"
    assert result.tag == "Some"
    assert result.fields == (7,)


def test_non_variant_call_stays_a_call():
    """A FunctionCall whose name is not a declared variant remains a Call."""
    import metaxu.metaxu_ast as fast

    hir = _build_hir_from_ast([fast.FunctionCall("Some", [fast.Literal(5)])], [])
    # No EnumDefinition in scope: "Some" is an unknown callee, not a variant.
    mir = lower_hir_to_mir(hir)
    interp = MirInterpreter()
    interp.load(mir)
    with pytest.raises(InterpError, match="Unknown callee"):
        interp.call("main", [])
