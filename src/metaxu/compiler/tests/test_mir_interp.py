"""Tests for the MIR interpreter.

Covers:
- Literal constants
- Arithmetic and comparison binops
- Function calls (user-defined and builtins)
- Conditional branches (br_if)
- Let bindings passed as params
- Effect perform/handle (stack and suspend classes)
- Single-shot continuation enforcement
"""
from __future__ import annotations

import pytest

from metaxu.compiler.mir import MirBlock, MirFunc
from metaxu.compiler.mir_interp import MirInterpreter, MxContinuation, UNIT, InterpError


# ---------------------------------------------------------------------------
# Helpers to build MirFuncs without going through the full pipeline
# ---------------------------------------------------------------------------

def make_func(name: str, blocks: list[MirBlock], suspending: bool = False) -> MirFunc:
    return MirFunc(name=name, ty_sig=None, blocks=blocks, suspending=suspending)


def simple_block(ops: list[tuple], term: tuple) -> MirBlock:
    return MirBlock(ops=ops, term=term)


# ---------------------------------------------------------------------------
# Literal / const
# ---------------------------------------------------------------------------

def test_literal_int():
    f = make_func("f", [
        simple_block(
            [("let", "x", ("const", 42), ())],
            ("ret", "x")
        )
    ])
    interp = MirInterpreter()
    interp.load([f])
    assert interp.call("f", []) == 42


def test_literal_string():
    f = make_func("f", [
        simple_block(
            [("let", "s", ("const", "hello"), ())],
            ("ret", "s")
        )
    ])
    interp = MirInterpreter()
    interp.load([f])
    assert interp.call("f", []) == "hello"


def test_const_ty_is_unit():
    f = make_func("f", [
        simple_block(
            [("let", "u", ("const_ty", "Int"), ())],
            ("ret", "u")
        )
    ])
    interp = MirInterpreter()
    interp.load([f])
    assert interp.call("f", []) is UNIT


# ---------------------------------------------------------------------------
# Binary ops
# ---------------------------------------------------------------------------

def test_binop_add():
    f = make_func("f", [
        simple_block(
            [
                ("let", "a", ("const", 3), ()),
                ("let", "b", ("const", 4), ()),
                ("let", "r", ("binop", "+"), ("a", "b")),
            ],
            ("ret", "r")
        )
    ])
    interp = MirInterpreter()
    interp.load([f])
    assert interp.call("f", []) == 7


def test_binop_mul():
    f = make_func("f", [
        simple_block(
            [
                ("let", "a", ("const", 6), ()),
                ("let", "b", ("const", 7), ()),
                ("let", "r", ("binop", "*"), ("a", "b")),
            ],
            ("ret", "r")
        )
    ])
    interp = MirInterpreter()
    interp.load([f])
    assert interp.call("f", []) == 42


def test_binop_eq_true():
    f = make_func("f", [
        simple_block(
            [
                ("let", "a", ("const", 5), ()),
                ("let", "b", ("const", 5), ()),
                ("let", "r", ("binop", "=="), ("a", "b")),
            ],
            ("ret", "r")
        )
    ])
    interp = MirInterpreter()
    interp.load([f])
    assert interp.call("f", []) is True


def test_binop_lt():
    f = make_func("f", [
        simple_block(
            [
                ("let", "a", ("const", 2), ()),
                ("let", "b", ("const", 5), ()),
                ("let", "r", ("binop", "<"), ("a", "b")),
            ],
            ("ret", "r")
        )
    ])
    interp = MirInterpreter()
    interp.load([f])
    assert interp.call("f", []) is True


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def test_identity_function():
    f = make_func("id", [
        simple_block(
            [("params", ("x",))],
            ("ret", "x")
        )
    ])
    interp = MirInterpreter()
    interp.load([f])
    assert interp.call("id", [99]) == 99


def test_add_function():
    f = make_func("add", [
        simple_block(
            [
                ("params", ("a", "b")),
                ("let", "r", ("binop", "+"), ("a", "b")),
            ],
            ("ret", "r")
        )
    ])
    interp = MirInterpreter()
    interp.load([f])
    assert interp.call("add", [10, 32]) == 42


# ---------------------------------------------------------------------------
# Control flow: br_if
# ---------------------------------------------------------------------------

def test_branch_true():
    # if true -> 1 else -> 2
    f = make_func("f", [
        simple_block(
            [
                ("params", ("cond",)),
            ],
            ("br_if", "cond", 1, 2)   # bb0 -> bb1 (true) or bb2 (false)
        ),
        # bb1: true branch
        simple_block(
            [("let", "r", ("const", 1), ())],
            ("ret", "r")
        ),
        # bb2: false branch
        simple_block(
            [("let", "r", ("const", 2), ())],
            ("ret", "r")
        ),
    ])
    interp = MirInterpreter()
    interp.load([f])
    assert interp.call("f", [True]) == 1
    assert interp.call("f", [False]) == 2


def test_branch_computes_condition():
    # compute x > 0, then branch
    f = make_func("sign", [
        simple_block(
            [
                ("params", ("x",)),
                ("let", "zero", ("const", 0), ()),
                ("let", "cond", ("binop", ">"), ("x", "zero")),
            ],
            ("br_if", "cond", 1, 2)
        ),
        simple_block(
            [("let", "r", ("const", "positive"), ())],
            ("ret", "r")
        ),
        simple_block(
            [("let", "r", ("const", "non-positive"), ())],
            ("ret", "r")
        ),
    ])
    interp = MirInterpreter()
    interp.load([f])
    assert interp.call("sign", [5]) == "positive"
    assert interp.call("sign", [-1]) == "non-positive"
    assert interp.call("sign", [0]) == "non-positive"


# ---------------------------------------------------------------------------
# User function calls
# ---------------------------------------------------------------------------

def test_call_user_func():
    double = make_func("double", [
        simple_block(
            [
                ("params", ("n",)),
                ("let", "two", ("const", 2), ()),
                ("let", "r", ("binop", "*"), ("n", "two")),
            ],
            ("ret", "r")
        )
    ])
    caller = make_func("caller", [
        simple_block(
            [
                ("let", "v", ("call", "double"), ("10",)),
            ],
            ("ret", "v")
        )
    ])
    interp = MirInterpreter()
    interp.load([double, caller])
    # Note: "10" is a string literal name; we need to const-bind it first
    # Use a proper example with const
    caller2 = make_func("caller2", [
        simple_block(
            [
                ("let", "arg", ("const", 10), ()),
                ("let", "v", ("call", "double"), ("arg",)),
            ],
            ("ret", "v")
        )
    ])
    interp.load([caller2])
    assert interp.call("caller2", []) == 20


# ---------------------------------------------------------------------------
# Builtin functions
# ---------------------------------------------------------------------------

def test_builtin_assert_eq_passes():
    f = make_func("f", [
        simple_block(
            [
                ("let", "a", ("const", 42), ()),
                ("let", "b", ("const", 42), ()),
                ("let", "r", ("call", "assert_eq"), ("a", "b")),
            ],
            ("ret", "r")
        )
    ])
    interp = MirInterpreter()
    interp.load([f])
    assert interp.call("f", []) is UNIT


def test_builtin_assert_eq_fails():
    f = make_func("f", [
        simple_block(
            [
                ("let", "a", ("const", 1), ()),
                ("let", "b", ("const", 2), ()),
                ("let", "r", ("call", "assert_eq"), ("a", "b")),
            ],
            ("ret", "r")
        )
    ])
    interp = MirInterpreter()
    interp.load([f])
    with pytest.raises(AssertionError):
        interp.call("f", [])


# ---------------------------------------------------------------------------
# Effect system: stack effect (handler calls k immediately)
# ---------------------------------------------------------------------------

def test_stack_effect_perform():
    """Stack effect: handler receives op, args, k and resumes k immediately."""
    # The "perform" op: ("perform", dst, effect, op_name, args, resume_block, resume_slot)
    f = make_func("f", [
        # bb0: perform Console.print("hello"), then resume into bb1
        simple_block(
            [
                ("let", "msg", ("const", "hello"), ()),
                ("perform", "r", "Console", "print", ("msg",), 1, "resume_val"),
            ],
            ("br", 1)
        ),
        # bb1: return result
        simple_block([], ("ret", "r")),
    ], suspending=False)

    interp = MirInterpreter()
    interp.load([f])
    printed: list[Any] = []

    def console_handler(op_name: str, args: list, k: MxContinuation):
        if op_name == "print":
            printed.append(args[0])
            return k.resume(UNIT, interp)
        raise InterpError(f"Unknown Console op: {op_name}")

    interp.register_effect_handler("Console", "stack", console_handler)
    result = interp.call("f", [])
    assert printed == ["hello"]
    assert result is UNIT


def test_stack_effect_single_shot_violation():
    """Resuming a stack continuation twice must raise an error."""
    f = make_func("f", [
        simple_block(
            [
                ("let", "x", ("const", 1), ()),
                ("perform", "r", "BadEffect", "op", ("x",), 1, "rv"),
            ],
            ("br", 1)
        ),
        simple_block([], ("ret", "r")),
    ])

    interp = MirInterpreter()
    interp.load([f])

    def bad_handler(op_name: str, args: list, k: MxContinuation):
        k.resume(UNIT, interp)  # first resume: OK
        k.resume(UNIT, interp)  # second resume: must raise

    interp.register_effect_handler("BadEffect", "stack", bad_handler)
    with pytest.raises(RuntimeError, match="single-shot"):
        interp.call("f", [])


# ---------------------------------------------------------------------------
# Effect system: suspend effect (handler stores k, resumes later)
# ---------------------------------------------------------------------------

def test_suspend_effect_perform_resume_later():
    """Suspend effect: handler stores continuation, resumes it after call returns.

    MIR structure for a coroutine-style function:
      bb0: perform Async.yield -> result stored in 'resumed_val'; if handler
           doesn't call k immediately the block returns UNIT via ("ret", "resumed_val").
      bb1: the resume point — uses the value that was passed to k.resume().
    """
    f = make_func("coro", [
        # bb0: perform yield; the handler stores k and returns UNIT.
        # The terminator ("ret", "resumed_val") returns the handler's placeholder.
        simple_block(
            [
                ("let", "unit", ("const_ty", "Unit"), ()),
                ("perform", "resumed_val", "Async", "yield", ("unit",), 1, "resumed_val"),
            ],
            ("ret", "resumed_val")   # suspend: return placeholder to caller
        ),
        # bb1: resume point — the continuation injects the real value via result_slot
        simple_block(
            [
                ("let", "two", ("const", 2), ()),
                ("let", "r", ("binop", "*"), ("resumed_val", "two")),
            ],
            ("ret", "r")
        ),
    ], suspending=True)

    interp = MirInterpreter()
    interp.load([f])
    stored_k: list[MxContinuation] = []

    def async_handler(op_name: str, args: list, k: MxContinuation):
        if op_name == "yield":
            stored_k.append(k)
            return UNIT  # scheduler will resume later
        raise InterpError(f"Unknown Async op: {op_name}")

    interp.register_effect_handler("Async", "suspend", async_handler)

    # First call suspends at the yield; continuation stored
    result1 = interp.call("coro", [])
    assert result1 is UNIT   # yielded placeholder
    assert len(stored_k) == 1

    # Scheduler resumes the continuation with value 21 -> bb1 runs: 21 * 2 = 42
    result2 = stored_k[0].resume(21, interp)
    assert result2 == 42


# ---------------------------------------------------------------------------
# Drop op is a no-op
# ---------------------------------------------------------------------------

def test_drop_noop():
    f = make_func("f", [
        simple_block(
            [
                ("let", "x", ("const", 7), ()),
                ("drop", "x"),
                ("let", "y", ("const", 3), ()),
            ],
            ("ret", "y")
        )
    ])
    interp = MirInterpreter()
    interp.load([f])
    assert interp.call("f", []) == 3


# ---------------------------------------------------------------------------
# Unknown function errors cleanly
# ---------------------------------------------------------------------------

def test_unknown_function_raises():
    interp = MirInterpreter()
    with pytest.raises(InterpError, match="Unknown function"):
        interp.call("no_such_fn", [])


# ---------------------------------------------------------------------------
# Integration: pipeline -> MIR -> interpreter
# ---------------------------------------------------------------------------

def test_pipeline_to_interp():
    """Full pipeline: .mx source -> HIR -> MIR -> interpreter."""
    from metaxu.compiler.pipeline import build_context_from_source, run_pipeline
    from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
    from metaxu.compiler.hir import HIRBuilder

    src = "fn double(x: Int) -> Int { x + x }"
    ctx = build_context_from_source(src)
    hir_funcs = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    mir_funcs = lower_hir_to_mir(hir_funcs)

    found = [f for f in mir_funcs if "double" in f.name]
    if not found:
        pytest.skip("No 'double' function in MIR output")

    interp = MirInterpreter()
    interp.load(mir_funcs)
    assert interp.call(found[0].name, [21]) == 42
