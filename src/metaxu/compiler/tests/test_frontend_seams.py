"""Regression tests for front-end silent-degradation seams.

All tests go through parsed source (parse -> freeze -> infer -> HIR -> MIR ->
interpreter), never hand-built HIR/MIR fixtures, per the repo convention.

Seam 1: lambdas created inside loop (and block) bodies used to get EMPTY
capture lists — the parser's scope-based capture analysis never links or
populates loop/block scopes, and the HIR builder trusted it blindly. The
interpreter then raised "Unbound variable" when the lambda body ran.

Seam 2: `if let`, `while let` and mode-annotated expressions were not
handled by the HIR builder at all (silently lowered to nothing, so whole
function bodies degraded to unit), and `return` was lowered to just its
operand expression, which is only correct in tail position — early returns
inside loops, `if` arms and match arms silently fell through. Struct
mutations through @mut parameters additionally vanished in the interpreter
because struct arguments were copied on call with no write-back.
"""
from __future__ import annotations

import pytest

from metaxu.compiler.pipeline import build_context_from_source, run_pipeline_ctx
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import MirInterpreter, UNIT


def run_main(source: str, entry: str = "main", strict: bool = False):
    """Compile ``source`` from text and execute ``entry`` in the interpreter."""
    ctx = build_context_from_source(source)
    if strict:
        run_pipeline_ctx(ctx)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    interp.register_builtin("print", lambda *a: UNIT)
    return interp.call(entry, [])


# ---------------------------------------------------------------------------
# Seam 1: lambda captures inside loop / block scopes
# ---------------------------------------------------------------------------

def test_lambda_in_while_loop_captures_loop_local_and_outer_mut():
    # i=0: base=0,  f(0)=0   -> total 0
    # i=1: base=10, f(1)=11  -> total 11
    # i=2: base=20, f(2)=22  -> total 33
    src = """
fn main() -> int {
    let @mut total = 0;
    let @mut i = 0;
    while i < 3 {
        let base = i * 10;
        let f = fn(x: int) -> x + base;
        total = total + f(i);
        i = i + 1;
    }
    total
}
"""
    assert run_main(src) == 33


def test_lambda_in_nested_block_inside_loop_captures():
    # The lambda lives one block deeper than the loop body; both `base`
    # (loop-local) and `bonus` (nested-block-local) must be captured.
    # i=0: bonus=1, g(0)=0+0+1=1;  i=1: bonus=2, g(1)=1+10+2=13 -> 14
    src = """
fn main() -> int {
    let @mut acc = 0;
    let @mut i = 0;
    while i < 2 {
        let base = i * 10;
        {
            let bonus = i + 1;
            let g = fn(x: int) -> x + base + bonus;
            acc = acc + g(i);
        }
        i = i + 1;
    }
    acc
}
"""
    assert run_main(src) == 14


def test_lambda_in_plain_nested_block_captures():
    src = """
fn main() -> int {
    let a = 5;
    let @mut out = 0;
    {
        let b = 7;
        let h = fn(x: int) -> x + a + b;
        out = h(100);
    }
    out
}
"""
    assert run_main(src) == 112


def test_lambda_captured_value_is_current_iteration_value():
    # The capture must be the value at closure-creation time within the
    # iteration that calls it, not a stale first-iteration value.
    src = """
fn main() -> int {
    let @mut last = 0;
    let @mut i = 0;
    while i < 4 {
        let f = fn(y: int) -> y * i;
        last = f(10);
        i = i + 1;
    }
    last
}
"""
    assert run_main(src) == 30  # 10 * 3 on the final iteration


# ---------------------------------------------------------------------------
# Seam 2a: `if let` with early return (pop_front shape)
# ---------------------------------------------------------------------------

POP_SRC = """
fn pop_value(v: Option[int]) -> Option[int] {
    if let Some(x) = v {
        return Some(x + 100)
    } else {
        return None
    }
}

fn unwrap_or(v: Option[int], dflt: int) -> int {
    if let Some(x) = v {
        return x
    } else {
        return dflt
    }
}

fn main() -> int {
    let a = unwrap_or(pop_value(Some(5)), 0 - 1);
    let b = unwrap_or(pop_value(None), 0 - 1);
    a * 1000 + b
}
"""


def test_if_let_with_early_returns_executes_both_branches():
    # pop_value(Some(5)) -> Some(105); pop_value(None) -> None -> -1
    assert run_main(POP_SRC) == 105 * 1000 - 1


def test_if_let_without_else_falls_through():
    src = """
fn describe(v: Option[int]) -> int {
    if let Some(x) = v {
        return x * 2
    }
    return 0 - 7
}

fn main() -> int {
    describe(Some(21)) + describe(None)
}
"""
    assert run_main(src) == 42 - 7


def test_pop_front_shaped_struct_mutation_via_mut_param():
    # The linked_list.mx pop_front shape end-to-end: an if-let that both
    # mutates the container through a @mut param and early-returns the
    # popped payload.
    src = """
struct Box {
    @mut item: Option[int]
}

fn pop_it(b: @mut Box) -> Option[int] {
    if let Some(x) = b.item {
        b.item = None;
        return Some(x)
    } else {
        return None
    }
}

fn main() -> int {
    let @mut b = Box { item: Some(9) };
    let first = pop_it(b);
    let second = pop_it(b);
    let @mut got = 0;
    if let Some(x) = first {
        got = got + x;
    }
    if let Some(y) = second {
        got = got + 1000;
    }
    got
}
"""
    # first pop yields 9; second pop must see the mutation and yield None
    assert run_main(src) == 9


# ---------------------------------------------------------------------------
# Seam 2b: early `return` in non-tail positions
# ---------------------------------------------------------------------------

def test_early_return_inside_while_loop():
    src = """
fn first_square_at_least(limit: int) -> int {
    let @mut i = 0;
    while i < 100 {
        if i * i >= limit {
            return i
        }
        i = i + 1;
    }
    return 0 - 1
}

fn main() -> int {
    first_square_at_least(44)
}
"""
    assert run_main(src) == 7  # 7*7 = 49 >= 44, 6*6 = 36 < 44


def test_return_inside_if_guard_skips_rest_of_body():
    # The 10_traits_and_structs.mx pop() shape: a guard `if` with a return
    # and NO else, followed by more statements. The guard firing must skip
    # the rest of the body.
    src = """
fn guarded(flag: int) -> int {
    if flag == 1 {
        return 111
    }
    return 222
}

fn main() -> int {
    guarded(1) * 1000 + guarded(0)
}
"""
    assert run_main(src) == 111 * 1000 + 222


def test_return_inside_match_arm_with_code_after_match():
    src = """
fn f(v: Option[int]) -> int {
    match v {
        Some(x) => { return x * 2 },
        None => 0,
    }
    return 99
}

fn main() -> int {
    f(Some(3)) * 1000 + f(None)
}
"""
    assert run_main(src) == 6 * 1000 + 99


# ---------------------------------------------------------------------------
# Seam 2c: `while let`
# ---------------------------------------------------------------------------

def test_while_let_loop_binds_and_terminates():
    # Counts down 5,4,3,2,1 accumulating the sum, then exits on None.
    src = """
fn step(x: int) -> Option[int] {
    if x > 1 {
        return Some(x - 1)
    } else {
        return None
    }
}

fn main() -> int {
    let @mut total = 0;
    let @mut cur = Some(5);
    while let Some(x) = cur {
        total = total + x;
        cur = step(x);
    }
    total
}
"""
    assert run_main(src) == 5 + 4 + 3 + 2 + 1


def test_while_let_with_early_return_inside_body():
    # The linked_list.mx get() shape: while-let traversal with an early
    # return out of the loop body.
    src = """
fn find_at(start: int, index: int) -> Option[int] {
    let @mut current = Some(start);
    let @mut i = 0;
    while let Some(x) = current {
        if i == index {
            return Some(x * 10)
        }
        current = if x > 0 { Some(x - 1) } else { None };
        i = i + 1;
    }
    return None
}

fn unwrap_or(v: Option[int], dflt: int) -> int {
    if let Some(x) = v {
        return x
    }
    return dflt
}

fn main() -> int {
    let hit = unwrap_or(find_at(9, 3), 0 - 1);
    let miss = unwrap_or(find_at(2, 50), 0 - 1);
    hit * 1000 + miss
}
"""
    # find_at(9, 3): values 9,8,7,6 -> index 3 is 6 -> Some(60)
    assert run_main(src) == 60 * 1000 - 1


# ---------------------------------------------------------------------------
# Seam 2d: mode-annotated expressions keep their value
# ---------------------------------------------------------------------------

def test_mode_annotated_expression_payload_survives():
    # `Some(@const s.field)` used to silently drop the payload, producing a
    # nullary Some that exploded at pattern-match time.
    src = """
struct Cell {
    data: int
}

fn read(c: Cell) -> Option[int] {
    return Some(@const c.data)
}

fn main() -> int {
    if let Some(v) = read(Cell { data: 77 }) {
        return v
    }
    return 0 - 1
}
"""
    assert run_main(src) == 77


# ---------------------------------------------------------------------------
# Loudness: unsupported if-let pattern shapes must not silently wildcard
# ---------------------------------------------------------------------------

def test_if_let_with_unsupported_pattern_fails_loudly():
    src = """
fn main() -> int {
    let v = 3;
    if let x + 1 = v {
        return 1
    }
    return 0
}
"""
    ctx = build_context_from_source(src)
    with pytest.raises(NotImplementedError, match="if let"):
        HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)


# ---------------------------------------------------------------------------
# The linked_list example itself must exercise its real bodies
# ---------------------------------------------------------------------------

def test_linked_list_pop_front_is_not_unit():
    import os
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    source = open(os.path.join(repo_root, "examples", "linked_list.mx")).read()
    ctx = build_context_from_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    mir = lower_hir_to_mir(hir)
    by_name = {f.name: f for f in mir}
    # pop_front used to lower to a single trivial unit block.
    pop = by_name["pop_front"]
    assert len(pop.blocks) > 1
    interp = MirInterpreter()
    interp.load(mir)
    interp.register_builtin("print", lambda *a: UNIT)
    lst = interp.call("new_list", [])
    interp.call("push_front", [lst, 3])
    # @mut write-back: main's env isn't in play here, so re-fetch by running
    # the example's own main to exercise push/pop/get/take end-to-end.
    result = interp.call("main", [])
    # main's last statement takes node index 1 (data 2) and sets data=42.
    assert str(result) != "()"
