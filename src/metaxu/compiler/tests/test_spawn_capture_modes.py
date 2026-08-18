"""Thread-safety mode enforcement at spawn boundaries
(compiler/spawn_capture_check.py; docs/threads_runtime.md § Modes).

A closure that flows into a `with EFFECT_SPAWN`-mapped operation must not
capture (1) a variable declared `@local` — kind ``locality-spawn-capture``:
the spawning frame's stack region may be reclaimed while the child thread
still runs — nor (2) a binding involved in an active `@mut` borrow — kind
``borrow-spawn-capture``: an exclusive borrow must not be shared across
threads.  Matching is on the RUNTIME SYMBOL, never the effect/op spelling,
and the check is deliberately syntactic on the spawn-mapped op: it applies
even when a handler in scope overrides the runtime mapping (a virtualized
spawn still promises thread-compatibility by its signature).

Everything documented as working stays working, pinned here in the
anti-false-positive direction: the mutex-counter pattern (Vec + Mutex
captured, mutated under lock) still compiles AND runs, as do captures of
plain scalars/strings/structs and Thread/Mutex handles.

Non-vacuity: verified during development by replacing the registration in
`pipeline.build_context_from_source` with `spawn_errors = []` — every
negative test below then compiled clean (6 failures in the development
battery), and restoring the registration turned them back into
BorrowCheckErrors.  All tests go through parsed source (parse -> pipeline
-> HIR -> MIR -> interpreter), per repo convention.
"""
from __future__ import annotations

import pytest

from metaxu.compiler.frozen_borrow_checker import BorrowCheckError
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import MirInterpreter, UNIT
from metaxu.compiler.pipeline import build_context_from_source, run_pipeline_ctx
from metaxu.compiler.spawn_capture_check import (LOCAL_CAPTURE_KIND,
                                                 MUT_BORROW_CAPTURE_KIND)

THREAD_EFFECT = """
extern type Thread[T];

effect Thread = {
    fn spawn[T](f: fn() -> @global T) -> @global Thread[T] with EFFECT_SPAWN
    fn join[T](thread: @global Thread[T]) -> @global T with EFFECT_JOIN
}
"""

MUTEX_EFFECT = """
extern type Mutex;

effect Mutex = {
    fn create() -> @global Mutex with EFFECT_MUTEX_CREATE
    fn lock(mutex: @global Mutex) -> () with EFFECT_MUTEX_LOCK
    fn unlock(mutex: @global Mutex) -> () with EFFECT_MUTEX_UNLOCK
}
"""


def compile_source(source: str, file_path: str = "<spawn-test>"):
    ctx = build_context_from_source(source, file_path=file_path)
    run_pipeline_ctx(ctx)
    return ctx


def run_source(source: str):
    """Full run-gate path: parse -> strict pipeline -> HIR -> MIR -> interp."""
    ctx = compile_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    prints: list[str] = []
    interp.register_builtin(
        "print", lambda *a: (prints.append(" ".join(str(x) for x in a)), UNIT)[1])
    return interp.call("main", []), prints


def spawn_capture_errors(excinfo, kind):
    """The structured errors of `kind` carried by a BorrowCheckError."""
    return [e for e in excinfo.value.errors
            if getattr(e, "kind", None) == kind]


# ---------------------------------------------------------------------------
# NEGATIVE: @local captures must not cross the spawn boundary (rule 1)
# ---------------------------------------------------------------------------

def test_local_let_captured_by_spawn_closure_is_rejected():
    source = THREAD_EFFECT + """
fn main() -> int {
    let @local x = 5;
    let t = perform Thread.spawn(|| { x + 1 });
    perform Thread.join(t)
}
"""
    with pytest.raises(BorrowCheckError) as excinfo:
        compile_source(source)
    errors = spawn_capture_errors(excinfo, LOCAL_CAPTURE_KIND)
    assert errors, str(excinfo.value)
    # The message names the variable, its mode, and the dangling rationale.
    assert errors[0].variable == "x"
    assert "@local variable 'x'" in errors[0].message
    assert "EFFECT_SPAWN" in errors[0].message
    assert "spawning frame" in errors[0].message
    # Diagnostic carries a real source location (docs/diagnostics_locations.md).
    assert errors[0].location is not None
    assert "<spawn-test>" in str(errors[0])


def test_local_param_captured_by_spawn_closure_is_rejected():
    source = THREAD_EFFECT + """
fn go(@local n: int) -> int {
    let t = perform Thread.spawn(|| { n });
    perform Thread.join(t)
}

fn main() -> int { go(3) }
"""
    with pytest.raises(BorrowCheckError) as excinfo:
        compile_source(source)
    errors = spawn_capture_errors(excinfo, LOCAL_CAPTURE_KIND)
    assert errors and errors[0].variable == "n"


def test_local_captured_via_nested_inner_lambda_is_rejected():
    """The capture crosses the thread boundary even when only an inner
    lambda inside the spawned closure reads it."""
    source = THREAD_EFFECT + """
fn apply(f: fn() -> int) -> int { f() }

fn main() -> int {
    let @local secret = 9;
    let t = perform Thread.spawn(|| {
        let g = || { secret + 1 };
        apply(g)
    });
    perform Thread.join(t)
}
"""
    with pytest.raises(BorrowCheckError) as excinfo:
        compile_source(source)
    errors = spawn_capture_errors(excinfo, LOCAL_CAPTURE_KIND)
    assert errors and errors[0].variable == "secret"


def test_bound_lambda_flowing_into_spawn_by_name_is_rejected():
    """`let work = || ...; perform Thread.spawn(work)` resolves the closure
    through the binding, not just the inline-lambda spelling."""
    source = THREAD_EFFECT + """
fn main() -> int {
    let @local x = 5;
    let work = || { x * 2 };
    let t = perform Thread.spawn(work);
    perform Thread.join(t)
}
"""
    with pytest.raises(BorrowCheckError) as excinfo:
        compile_source(source)
    errors = spawn_capture_errors(excinfo, LOCAL_CAPTURE_KIND)
    assert errors and errors[0].variable == "x"


def test_renamed_effect_is_matched_by_runtime_symbol_not_spelling():
    """Users can rename the effect and the op; `with EFFECT_SPAWN` is what
    makes it a spawn."""
    source = """
extern type Tsk[T];

effect Tsk = {
    fn go[T](f: fn() -> @global T) -> @global Tsk[T] with EFFECT_SPAWN
    fn wait[T](t: @global Tsk[T]) -> @global T with EFFECT_JOIN
}

fn main() -> int {
    let @local x = 1;
    let t = perform Tsk.go(|| { x });
    perform Tsk.wait(t)
}
"""
    with pytest.raises(BorrowCheckError) as excinfo:
        compile_source(source)
    errors = spawn_capture_errors(excinfo, LOCAL_CAPTURE_KIND)
    assert errors and errors[0].variable == "x"
    assert "Tsk.go" in errors[0].message


def test_handler_overridden_spawn_is_still_checked():
    """Documented choice (docs/threads_runtime.md § Modes): the check is
    syntactic on the spawn-mapped op regardless of handler override — a
    virtualized spawn that never really threads still promises
    thread-compatibility by its signature."""
    source = THREAD_EFFECT + """
fn main() -> int {
    let @local n = 7;
    handle Thread with {
        spawn(f) -> resume(f())
        join(t) -> resume(t)
    } in {
        let t = perform Thread.spawn(|| { n });
        perform Thread.join(t)
    }
}
"""
    with pytest.raises(BorrowCheckError) as excinfo:
        compile_source(source)
    errors = spawn_capture_errors(excinfo, LOCAL_CAPTURE_KIND)
    assert errors and errors[0].variable == "n"


# ---------------------------------------------------------------------------
# NEGATIVE: active @mut borrows must not cross the spawn boundary (rule 2)
# ---------------------------------------------------------------------------

def test_captured_mut_borrow_reference_is_rejected():
    source = THREAD_EFFECT + """
fn main() -> int {
    let v = 1;
    let r = &mut v;
    let t = perform Thread.spawn(|| { r });
    perform Thread.join(t);
    0
}
"""
    with pytest.raises(BorrowCheckError) as excinfo:
        compile_source(source)
    errors = spawn_capture_errors(excinfo, MUT_BORROW_CAPTURE_KIND)
    assert errors, str(excinfo.value)
    assert errors[0].variable == "r"
    assert "active @mut borrow of 'v'" in errors[0].message


def test_capture_of_variable_under_active_mut_borrow_is_rejected():
    source = THREAD_EFFECT + """
fn main() -> int {
    let v = 1;
    let r = &mut v;
    let t = perform Thread.spawn(|| { v });
    perform Thread.join(t)
}
"""
    with pytest.raises(BorrowCheckError) as excinfo:
        compile_source(source)
    errors = spawn_capture_errors(excinfo, MUT_BORROW_CAPTURE_KIND)
    assert errors and errors[0].variable == "v"
    assert "'r' holds an active @mut borrow" in errors[0].message


# ---------------------------------------------------------------------------
# POSITIVE: the documented working patterns keep compiling AND running
# ---------------------------------------------------------------------------

def test_mutex_counter_pattern_still_compiles_and_runs():
    """THE canonical cross-thread pattern (docs/threads_runtime.md): a Vec
    (shared identity) and a Mutex handle captured by worker closures, the
    Vec mutated under the lock.  Schedule-independent final value N*M."""
    source = THREAD_EFFECT + MUTEX_EFFECT + """
fn main() -> int {
    let m = perform Mutex.create();
    let @mut counter = Vec.new();
    counter.push(0);
    let @mut handles = Vec.new();
    let @mut i = 0;
    while i < 2 {
        let t = perform Thread.spawn(|| {
            let @mut j = 0;
            while j < 50 {
                perform Mutex.lock(m);
                counter[0] = counter[0] + 1;
                perform Mutex.unlock(m);
                j = j + 1
            };
            0
        });
        handles.push(t);
        i = i + 1
    };
    let @mut k = 0;
    while k < 2 {
        perform Thread.join(handles[k]);
        k = k + 1
    };
    counter[0]
}
"""
    result, _ = run_source(source)
    assert result == 100


def test_plain_scalar_string_struct_captures_still_run():
    source = THREAD_EFFECT + """
struct Point { x: int, y: int }

fn main() -> int {
    let n = 41;
    let s = "hi";
    let p = Point { x: 1, y: 2 };
    let t = perform Thread.spawn(|| { print(s); n + p.x });
    perform Thread.join(t)
}
"""
    result, prints = run_source(source)
    assert result == 42
    assert prints == ["hi"]


def test_handler_overridden_spawn_with_clean_captures_still_runs():
    """The other half of the documented choice: an overridden spawn whose
    closure captures only thread-compatible values is untouched."""
    source = THREAD_EFFECT + """
fn main() -> int {
    let n = 7;
    handle Thread with {
        spawn(f) -> resume(f())
        join(t) -> resume(t)
    } in {
        let t = perform Thread.spawn(|| { n });
        perform Thread.join(t)
    }
}
"""
    result, _ = run_source(source)
    assert result == 7


def test_mutex_handle_capture_still_runs():
    """examples/effect_mapping.mx's shape: the spawned closure captures the
    Mutex handle (an opaque runtime word) — explicitly allowed."""
    source = THREAD_EFFECT + MUTEX_EFFECT + """
fn main() -> int {
    let mutex = perform Mutex.create();
    let thread = perform Thread.spawn(|| {
        perform Mutex.lock(mutex);
        perform Mutex.unlock(mutex);
        5
    });
    perform Thread.join(thread)
}
"""
    result, _ = run_source(source)
    assert result == 5


def test_local_used_outside_spawn_is_not_flagged():
    """Anti-false-positive: an @local in the same function is fine as long
    as the spawned closure does not capture it."""
    source = THREAD_EFFECT + """
fn main() -> int {
    let @local x = 5;
    let y = x + 1;
    let n = 2;
    let t = perform Thread.spawn(|| { n });
    perform Thread.join(t)
}
"""
    result, _ = run_source(source)
    assert result == 2


def test_shadowing_inside_closure_is_not_a_capture():
    """Anti-false-positive: a let inside the spawned closure that shadows
    an outer @local is the closure's own binding, not a capture."""
    source = THREAD_EFFECT + """
fn main() -> int {
    let @local x = 5;
    let y = x + 1;
    let t = perform Thread.spawn(|| { let x = 1; x + 1 });
    perform Thread.join(t)
}
"""
    result, _ = run_source(source)
    assert result == 2


def test_programs_without_spawn_mapped_effects_are_untouched():
    """The pass is a no-op for programs that declare no EFFECT_SPAWN op —
    an @local captured by an ordinary (non-spawn) lambda stays legal."""
    source = """
fn main() -> int {
    let @local x = 5;
    let f = || { x + 1 };
    f()
}
"""
    result, _ = run_source(source)
    assert result == 6
