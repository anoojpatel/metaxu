"""Pin examples/effect_mapping.mx and the `with SYMBOL` runtime mapping.

The example declares Thread and Mutex effects whose ops carry `with
EFFECT_*` clauses ("C runtime mappings", per its comments) and performs
them with no handler in scope. Each mapped op compiles to a thunk
__effect_runtime$Effect$op whose body calls __mx_effect_runtime$SYMBOL;
the MIR interpreter dispatches that callee to a runtime shim table.

Interpreter execution model (documented on the shims): single logical
thread — EFFECT_SPAWN runs the child closure to completion at spawn time
(one legal schedule of real thread semantics) and EFFECT_JOIN returns its
stored result. Mutexes are therefore exact, not simulated: locking an
already-locked mutex can never succeed later, so it is a deadlock and a
loud error; unlocking an unlocked mutex and joining a thread twice are
loud errors too. Handlers in scope always win over the runtime mapping
(effects stay virtualizable); a mapped symbol with no shim fails loudly.

All tests go through parsed source (parse -> pipeline -> HIR -> MIR ->
interpreter), never hand-built fixtures.
"""
from __future__ import annotations

import os

import pytest

from metaxu.compiler.pipeline import build_context_from_source, run_pipeline_ctx
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import InterpError, MirInterpreter, UNIT

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

MUTEX_EFFECT = """
extern type Mutex;

effect Mutex = {
    fn create() -> @global Mutex with EFFECT_MUTEX_CREATE
    fn lock(mutex: @global Mutex) -> () with EFFECT_MUTEX_LOCK
    fn unlock(mutex: @global Mutex) -> () with EFFECT_MUTEX_UNLOCK
}
"""

THREAD_EFFECT = """
extern type Thread[T];

effect Thread = {
    fn spawn[T](f: fn() -> @global T) -> @global Thread[T] with EFFECT_SPAWN
    fn join[T](thread: @global Thread[T]) -> @global T with EFFECT_JOIN
}
"""


def run_source(source: str):
    """Full run-gate path: parse -> strict pipeline -> HIR -> MIR -> interp."""
    ctx = build_context_from_source(source)
    run_pipeline_ctx(ctx)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    prints: list[str] = []
    interp.register_builtin(
        "print", lambda *a: (prints.append(" ".join(str(x) for x in a)), UNIT)[1])
    return interp.call("main", []), prints


def test_effect_mapping_example_executes():
    """The shipped example runs end-to-end: mutex created, thread spawned
    (its closure locks and unlocks the captured mutex across the effect
    boundary), thread joined; main returns unit."""
    source = open(os.path.join(REPO_ROOT, "examples", "effect_mapping.mx")).read()
    result, _prints = run_source(source)
    assert result is UNIT


def test_spawn_runs_closure_and_join_returns_its_value():
    """EFFECT_SPAWN actually executes the closure (with its captured env)
    and EFFECT_JOIN yields the closure's result — not a placeholder."""
    source = THREAD_EFFECT + """
fn main() -> () {
    let x = 20;
    let t = perform Thread.spawn(|| { x + 22 });
    let v = perform Thread.join(t);
    print(v.to_string());
}
"""
    _result, prints = run_source(source)
    assert prints == ["42"]


def test_lock_unlock_lock_again_is_legal():
    """Mutex state is tracked: unlock releases, so a second lock succeeds."""
    source = MUTEX_EFFECT + """
fn main() -> () {
    let m = perform Mutex.create();
    perform Mutex.lock(m);
    perform Mutex.unlock(m);
    perform Mutex.lock(m);
    perform Mutex.unlock(m);
}
"""
    result, _prints = run_source(source)
    assert result is UNIT


def test_locking_locked_mutex_is_deadlock_error():
    source = MUTEX_EFFECT + """
fn main() -> () {
    let m = perform Mutex.create();
    perform Mutex.lock(m);
    perform Mutex.lock(m);
}
"""
    with pytest.raises(InterpError, match="deadlock"):
        run_source(source)


def test_unlocking_unlocked_mutex_errors():
    source = MUTEX_EFFECT + """
fn main() -> () {
    let m = perform Mutex.create();
    perform Mutex.unlock(m);
}
"""
    with pytest.raises(InterpError, match="not locked"):
        run_source(source)


def test_joining_thread_twice_errors():
    source = THREAD_EFFECT + """
fn main() -> () {
    let t = perform Thread.spawn(|| { 1 });
    perform Thread.join(t);
    perform Thread.join(t);
}
"""
    with pytest.raises(InterpError, match="already joined"):
        run_source(source)


def test_unmapped_runtime_symbol_fails_loudly():
    """A `with SYMBOL` mapping the interpreter has no shim for must error
    precisely at perform time — never silently no-op."""
    source = """
effect Exotic = {
    fn zap() -> () with EFFECT_TELEPORT
}

fn main() -> () {
    perform Exotic.zap();
}
"""
    with pytest.raises(InterpError, match="EFFECT_TELEPORT"):
        run_source(source)


def test_handler_in_scope_overrides_runtime_mapping():
    """Effects stay virtualizable: a handle scope intercepting a mapped op
    wins over the runtime shim (no real mutex is ever created here)."""
    source = MUTEX_EFFECT + """
fn main() -> () {
    handle Mutex with {
        create() -> {
            print("intercepted");
            resume(())
        }
    } in {
        let m = perform Mutex.create();
    }
}
"""
    _result, prints = run_source(source)
    assert prints == ["intercepted"]
