"""try/catch semantics (docs/try_catch.md): delimited dynamic error recovery.

All tests go through parsed source to the interpreter.
"""
from __future__ import annotations

import pytest

from metaxu.compiler.pipeline import build_context_from_source
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import MirInterpreter, UNIT


def call(source: str, fn: str = "main", args: list | None = None):
    ctx = build_context_from_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    interp.register_builtin("print", lambda *a: UNIT)
    return interp.call(fn, args or [])


def test_try_value_when_no_failure():
    assert call("""
fn main() -> int {
    try { 41 + 1 } catch e { 0 }
}
""") == 42


def test_catch_unhandled_perform():
    """An unhandled effect perform inside try takes the catch path."""
    assert call("""
effect Parser { parse(input: string) -> int }

fn main() -> int {
    try {
        let v = perform Parser.parse("123");
        v
    } catch e {
        -1
    }
}
""") == -1


def test_catch_binds_error_message():
    result = call("""
effect Parser { parse(input: string) -> int }

fn main() -> string {
    try {
        perform Parser.parse("x");
        "ok"
    } catch e {
        "caught: " + e
    }
}
""")
    assert result.startswith("caught: ")
    assert "Parser" in result


def test_catch_builtin_contract_violation():
    """pop on an empty Vec is a catchable runtime failure."""
    assert call("""
fn main() -> int {
    try {
        let v = Vec<int>::new();
        v.pop();
        1
    } catch e {
        2
    }
}
""") == 2


def test_abort_semantics_rest_of_body_skipped():
    _, prints = _call_with_prints("""
effect Fail { boom() -> int }

fn main() -> int {
    try {
        perform Fail.boom();
        print("unreachable");
        1
    } catch e {
        0
    }
}
""")
    assert prints == []


def test_failure_in_called_function_caught():
    """The delimited extent includes called functions."""
    assert call("""
effect Fail { boom() -> int }

fn helper() performs Fail -> int {
    perform Fail.boom()
}

fn main() -> int {
    try { helper() } catch e { 7 }
}
""") == 7


def test_nested_try_innermost_wins():
    assert call("""
effect Fail { boom() -> int }

fn main() -> int {
    try {
        try { perform Fail.boom() } catch inner { 10 }
    } catch outer {
        20
    }
}
""") == 10


def test_failure_in_catch_propagates_outward():
    assert call("""
effect Fail { boom() -> int }

fn main() -> int {
    try {
        try {
            perform Fail.boom()
        } catch inner {
            perform Fail.boom()
        }
    } catch outer {
        30
    }
}
""") == 30


def test_installed_handler_wins_over_try():
    """A real handler gives the effect meaning; try only contains failures."""
    assert call("""
effect Ask { ask() -> int }

fn main() -> int {
    handle Ask with {
        ask() -> resume(5)
    } in {
        try { perform Ask.ask() + 1 } catch e { -1 }
    }
}
""") == 6


def _call_with_prints(source: str):
    ctx = build_context_from_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    prints: list[str] = []
    interp.register_builtin("print", lambda *a: (prints.append(" ".join(str(x) for x in a)), UNIT)[1])
    return interp.call("main", []), prints
