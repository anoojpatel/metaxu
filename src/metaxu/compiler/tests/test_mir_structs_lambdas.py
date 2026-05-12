"""Tests for struct and lambda/closure support in the MIR interpreter.

Covers:
- Struct allocation (local and global/heap)
- Field access (field_get)
- Field update (field_set, value semantics)
- Lambda creation (make_closure)
- Closure call (call_closure)
- Closure capturing from outer env
- Locality propagation
"""
from __future__ import annotations

import pytest

from metaxu.compiler.mir import MirBlock, MirFunc
from metaxu.compiler.mir_interp import (
    MirInterpreter, MxStruct, MxClosure, MxUnit, UNIT, InterpError
)


def make_func(name: str, blocks: list[MirBlock], suspending: bool = False) -> MirFunc:
    return MirFunc(name=name, ty_sig=None, blocks=blocks, suspending=suspending)


def simple_block(ops: list[tuple], term: tuple) -> MirBlock:
    return MirBlock(ops=ops, term=term)


# ---------------------------------------------------------------------------
# Struct: alloc_struct
# ---------------------------------------------------------------------------

def test_alloc_struct_local():
    f = make_func("f", [simple_block(
        [
            ("let", "x", ("const", 10), ()),
            ("let", "y", ("const", 20), ()),
            ("let", "pt", ("alloc_struct", "Point", "local"), (("x", "x"), ("y", "y"))),
        ],
        ("ret", "pt")
    )])
    interp = MirInterpreter()
    interp.load([f])
    result = interp.call("f", [])
    assert isinstance(result, MxStruct)
    assert result.name == "Point"
    assert result.fields == {"x": 10, "y": 20}
    assert result.locality == "local"


def test_alloc_struct_global():
    f = make_func("f", [simple_block(
        [
            ("let", "n", ("const", 42), ()),
            ("let", "box", ("alloc_struct", "Box", "global"), (("val", "n"),)),
        ],
        ("ret", "box")
    )])
    interp = MirInterpreter()
    interp.load([f])
    result = interp.call("f", [])
    assert isinstance(result, MxStruct)
    assert result.locality == "global"
    assert result.fields["val"] == 42


def test_struct_repr():
    s = MxStruct(name="Point", fields={"x": 1, "y": 2})
    assert "Point" in repr(s)
    assert "x=1" in repr(s)


# ---------------------------------------------------------------------------
# Struct: field_get
# ---------------------------------------------------------------------------

def test_field_get():
    f = make_func("f", [simple_block(
        [
            ("let", "v", ("const", 99), ()),
            ("let", "s", ("alloc_struct", "Wrapper", "local"), (("val", "v"),)),
            ("let", "r", ("field_get", "val"), ("s",)),
        ],
        ("ret", "r")
    )])
    interp = MirInterpreter()
    interp.load([f])
    assert interp.call("f", []) == 99


def test_field_get_missing_raises():
    f = make_func("f", [simple_block(
        [
            ("let", "s", ("alloc_struct", "Empty", "local"), ()),
            ("let", "r", ("field_get", "missing"), ("s",)),
        ],
        ("ret", "r")
    )])
    interp = MirInterpreter()
    interp.load([f])
    with pytest.raises(KeyError, match="missing"):
        interp.call("f", [])


def test_field_get_non_struct_raises():
    f = make_func("f", [simple_block(
        [
            ("let", "x", ("const", 5), ()),
            ("let", "r", ("field_get", "val"), ("x",)),
        ],
        ("ret", "r")
    )])
    interp = MirInterpreter()
    interp.load([f])
    with pytest.raises(InterpError, match="field_get"):
        interp.call("f", [])


# ---------------------------------------------------------------------------
# Struct: field_set (value semantics — returns updated copy)
# ---------------------------------------------------------------------------

def test_field_set_returns_new_struct():
    f = make_func("f", [simple_block(
        [
            ("let", "old", ("const", 1), ()),
            ("let", "new_v", ("const", 99), ()),
            ("let", "s", ("alloc_struct", "Cell", "local"), (("val", "old"),)),
            ("let", "s2", ("field_set", "val"), ("s", "new_v")),
            ("let", "r", ("field_get", "val"), ("s2",)),
        ],
        ("ret", "r")
    )])
    interp = MirInterpreter()
    interp.load([f])
    assert interp.call("f", []) == 99


def test_field_set_original_unchanged():
    """Value semantics: original struct must not be mutated."""
    f = make_func("f", [simple_block(
        [
            ("let", "old", ("const", 1), ()),
            ("let", "new_v", ("const", 99), ()),
            ("let", "s", ("alloc_struct", "Cell", "local"), (("val", "old"),)),
            ("let", "s2", ("field_set", "val"), ("s", "new_v")),
            ("let", "r", ("field_get", "val"), ("s",)),  # read from original
        ],
        ("ret", "r")
    )])
    interp = MirInterpreter()
    interp.load([f])
    assert interp.call("f", []) == 1  # original unchanged


# ---------------------------------------------------------------------------
# Lambda: make_closure + call_closure
# ---------------------------------------------------------------------------

def test_make_closure_and_call():
    """make_closure wraps a named func; call_closure invokes it."""
    add_one = make_func("add_one", [simple_block(
        [
            ("params", ("n",)),
            ("let", "one", ("const", 1), ()),
            ("let", "r", ("binop", "+"), ("n", "one")),
        ],
        ("ret", "r")
    )])
    caller = make_func("caller", [simple_block(
        [
            ("let", "cl", ("make_closure", "add_one", ("n",)), ()),
            ("let", "arg", ("const", 41), ()),
            ("let", "r", ("call_closure",), ("cl", "arg")),
        ],
        ("ret", "r")
    )])
    interp = MirInterpreter()
    interp.load([add_one, caller])
    assert interp.call("caller", []) == 42


def test_closure_captures_env():
    """Closure captures a value from its creation env."""
    adder = make_func("adder", [simple_block(
        [
            ("params", ("x",)),
            ("let", "r", ("binop", "+"), ("x", "offset")),
        ],
        ("ret", "r")
    )])
    maker = make_func("make_adder", [simple_block(
        [
            # offset = 10 in this env
            ("let", "offset", ("const", 10), ()),
            # make_closure: func=adder, captures offset from current env
            ("let", "cl", ("make_closure", "adder", ("x",)), (("offset", "offset"),)),
        ],
        ("ret", "cl")
    )])
    caller = make_func("caller", [simple_block(
        [
            ("let", "adder_cl", ("call", "make_adder"), ()),
            ("let", "arg", ("const", 32), ()),
            ("let", "r", ("call_closure",), ("adder_cl", "arg")),
        ],
        ("ret", "r")
    )])
    interp = MirInterpreter()
    interp.load([adder, maker, caller])
    assert interp.call("caller", []) == 42


def test_call_non_closure_raises():
    f = make_func("f", [simple_block(
        [
            ("let", "x", ("const", 5), ()),
            ("let", "r", ("call_closure",), ("x",)),
        ],
        ("ret", "r")
    )])
    interp = MirInterpreter()
    interp.load([f])
    with pytest.raises(InterpError, match="call_closure"):
        interp.call("f", [])


# ---------------------------------------------------------------------------
# Struct + Lambda combined: pass struct to closure, access field
# ---------------------------------------------------------------------------

def test_lambda_receives_struct_field():
    get_x = make_func("get_x", [simple_block(
        [
            ("params", ("pt",)),
            ("let", "r", ("field_get", "x"), ("pt",)),
        ],
        ("ret", "r")
    )])
    f = make_func("f", [simple_block(
        [
            ("let", "vx", ("const", 7), ()),
            ("let", "vy", ("const", 3), ()),
            ("let", "pt", ("alloc_struct", "Point", "local"), (("x", "vx"), ("y", "vy"))),
            ("let", "cl", ("make_closure", "get_x", ("pt",)), ()),
            ("let", "r", ("call_closure",), ("cl", "pt")),
        ],
        ("ret", "r")
    )])
    interp = MirInterpreter()
    interp.load([get_x, f])
    assert interp.call("f", []) == 7


# ---------------------------------------------------------------------------
# Locality: local structs are MxStruct with locality="local"
#           global structs are MxStruct with locality="global"
# ---------------------------------------------------------------------------

def test_locality_local_vs_global():
    f = make_func("f", [simple_block(
        [
            ("let", "n", ("const", 1), ()),
            ("let", "ls", ("alloc_struct", "S", "local"), (("n", "n"),)),
            ("let", "gs", ("alloc_struct", "S", "global"), (("n", "n"),)),
        ],
        ("ret", "gs")
    )])
    interp = MirInterpreter()
    interp.load([f])
    result = interp.call("f", [])
    assert result.locality == "global"
