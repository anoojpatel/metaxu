"""Real-threads semantics behind EFFECT_SPAWN/JOIN and EFFECT_MUTEX_*
(docs/threads_runtime.md; the interpreter is the semantics REFERENCE).

Execution model under test: `perform Thread.spawn(f)` starts `f` on its
own OS thread immediately and runs it concurrently; `perform
Thread.join(t)` blocks and returns the child's result exactly once (a
second join errors loudly); mutexes are ERRORCHECK (lock held by ANOTHER
thread blocks, self-relock and unlock-not-held error loudly); each
spawned thread owns its OWN effect scope stack (a child's perform never
reaches the spawner's in-scope handlers), while a handler in scope ON THE
PERFORMING THREAD still overrides the runtime mapping.

Determinism policy (spec § scheduling): every assertion here is
schedule-independent — final counter values, join results, error
messages — never interleavings.  All tests go through parsed source
(parse -> pipeline -> HIR -> MIR -> interpreter), per repo convention.
Native differentials for the schedule-independent programs live in
test_codegen_llvm.py (thread increment section).
"""
from __future__ import annotations

import pytest

from metaxu.compiler.pipeline import build_context_from_source, run_pipeline_ctx
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import InterpError, MirInterpreter, UNIT

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


# ---------------------------------------------------------------------------
# Concurrency: the mutex-protected shared counter (N threads x M increments)
# ---------------------------------------------------------------------------

# A Vec captured by the worker closures: Vec has IDENTITY semantics on both
# engines, so all threads increment ONE slot.  The mutex makes the read-
# modify-write atomic; the final value N*M is schedule-independent.
COUNTER_SRC = THREAD_EFFECT + MUTEX_EFFECT + """
fn main() -> int {
    let m = perform Mutex.create();
    let @mut counter = Vec.new();
    counter.push(0);
    let @mut handles = Vec.new();
    let @mut i = 0;
    # unsafe: exercises the RAW mutex primitives on purpose (manual
    # lock/unlock around a bare shared Vec); the blessed non-unsafe
    # spelling is std.sync.protect (docs/separate_send_sync.md).  Since
    # the contention work this manual discipline has a dynamic net: the
    # vec is marked contended at spawn, and the held lock is what makes
    # the writes legal (docs/contention_as_permission.md).
    unsafe {
        while i < 4 {
            let t = perform Thread.spawn(|| {
                let @mut j = 0;
                while j < 250 {
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
    }
    let @mut k = 0;
    while k < 4 {
        perform Thread.join(handles[k]);
        k = k + 1
    };
    counter[0]
}
"""


def test_mutex_counter_reaches_n_times_m():
    result, _ = run_source(COUNTER_SRC)
    assert result == 1000


def test_lock_held_by_another_thread_blocks_until_released():
    """The semantic change from the old single-threaded model: a lock held
    by ANOTHER thread blocks (it does not 'deadlock'-error).  The parent
    holds the mutex across the spawn; the child's lock can only succeed
    after the parent's unlock, and join returning the child's value proves
    the child blocked and then proceeded — schedule-independent."""
    source = THREAD_EFFECT + MUTEX_EFFECT + """
fn main() -> int {
    let m = perform Mutex.create();
    perform Mutex.lock(m);
    let t = perform Thread.spawn(|| {
        perform Mutex.lock(m);
        perform Mutex.unlock(m);
        7
    });
    perform Mutex.unlock(m);
    perform Thread.join(t)
}
"""
    result, _ = run_source(source)
    assert result == 7


# ---------------------------------------------------------------------------
# Join semantics
# ---------------------------------------------------------------------------

def test_join_returns_child_value():
    source = THREAD_EFFECT + """
fn main() -> int {
    let x = 40;
    let t = perform Thread.spawn(|| { x + 2 });
    perform Thread.join(t)
}
"""
    result, _ = run_source(source)
    assert result == 42


def test_double_join_errors_loudly():
    source = THREAD_EFFECT + """
fn main() -> int {
    let t = perform Thread.spawn(|| { 1 });
    perform Thread.join(t);
    perform Thread.join(t)
}
"""
    with pytest.raises(InterpError, match="thread already joined"):
        run_source(source)


def test_child_failure_surfaces_at_join_not_at_spawn():
    """A catchable failure in the child does NOT unwind into the parent at
    failure time; it is re-raised (with the child's own message) by join."""
    source = THREAD_EFFECT + """
fn main() -> int {
    let t = perform Thread.spawn(|| {
        let @mut v = Vec.new();
        v.pop()
    });
    print("spawned");
    perform Thread.join(t)
}
"""
    with pytest.raises(InterpError, match="pop: Vec is empty"):
        run_source(source)


def test_child_failure_is_catchable_around_join():
    """try around the JOIN recovers the child's failure message — the
    native contract (the child's mx_try pad captures it; join re-raises
    catchably on the joining thread)."""
    source = THREAD_EFFECT + """
fn main() -> str {
    let t = perform Thread.spawn(|| {
        let @mut v = Vec.new();
        v.pop()
    });
    try {
        perform Thread.join(t);
        "no failure"
    } catch e {
        "caught: " + e
    }
}
"""
    result, _ = run_source(source)
    assert result == "caught: pop: Vec is empty"


# ---------------------------------------------------------------------------
# Mutex error cases (ERRORCHECK semantics)
# ---------------------------------------------------------------------------

def test_self_relock_is_deadlock_error():
    source = MUTEX_EFFECT + """
fn main() -> () {
    let m = perform Mutex.create();
    perform Mutex.lock(m);
    perform Mutex.lock(m);
}
"""
    with pytest.raises(InterpError,
                       match="deadlock: EFFECT_MUTEX_LOCK .* already holds it"):
        run_source(source)


def test_unlock_of_unlocked_mutex_errors():
    source = MUTEX_EFFECT + """
fn main() -> () {
    let m = perform Mutex.create();
    perform Mutex.unlock(m);
}
"""
    with pytest.raises(InterpError,
                       match="EFFECT_MUTEX_UNLOCK .* does not hold it"):
        run_source(source)


def test_unlock_by_non_owner_thread_errors():
    """The parent holds the mutex; the CHILD's unlock is an EPERM-style
    loud error (same unified message as unlock-of-unlocked), surfacing at
    join."""
    source = THREAD_EFFECT + MUTEX_EFFECT + """
fn main() -> int {
    let m = perform Mutex.create();
    perform Mutex.lock(m);
    let t = perform Thread.spawn(|| {
        perform Mutex.unlock(m);
        1
    });
    let r = try {
        perform Thread.join(t)
    } catch e {
        print(e);
        -1
    };
    perform Mutex.unlock(m);
    r
}
"""
    result, prints = run_source(source)
    assert result == -1
    assert prints and "does not hold it" in prints[0]


def test_mutex_owner_spans_handle_body_seam():
    """Mutex ownership is per LOGICAL thread: a handle body runs on a
    parked helper thread but is the same thread of control, so locking
    inside the handle body after locking outside is still a SELF-relock
    (deadlock error), not a block."""
    source = MUTEX_EFFECT + """
effect Ping = {
    fn ping() -> int
}

fn main() -> () {
    let m = perform Mutex.create();
    perform Mutex.lock(m);
    handle Ping with {
        ping() -> { resume(1) }
    } in {
        perform Mutex.lock(m);
    }
}
"""
    with pytest.raises(InterpError,
                       match="deadlock: EFFECT_MUTEX_LOCK .* already holds it"):
        run_source(source)


# ---------------------------------------------------------------------------
# Effect-scope isolation and virtualization
# ---------------------------------------------------------------------------

def test_child_perform_does_not_reach_parents_in_scope_handler():
    """Scope isolation (spec § effect-scope isolation): the child performs
    Ping.ping while the PARENT has a Ping handler in scope; the child's
    scope stack is its own and empty, so the perform fails with the usual
    unhandled-effect error — surfaced at join — and the parent's handler
    never fires."""
    source = THREAD_EFFECT + """
effect Ping = {
    fn ping() -> int
}

fn main() -> str {
    handle Ping with {
        ping() -> {
            print("parent handler fired");
            resume(99)
        }
    } in {
        let t = perform Thread.spawn(|| {
            perform Ping.ping()
        });
        try {
            perform Thread.join(t);
            "handled?"
        } catch e {
            e
        }
    }
}
"""
    result, prints = run_source(source)
    assert result == "No handler for effect 'Ping'"
    assert prints == []  # the parent's handler never ran


def test_child_installs_and_uses_its_own_handler():
    """Isolation cuts inheritance, not capability: a handle INSIDE the
    child works normally."""
    source = THREAD_EFFECT + """
effect Ping = {
    fn ping() -> int
}

fn main() -> int {
    let t = perform Thread.spawn(|| {
        handle Ping with {
            ping() -> { resume(21) }
        } in {
            perform Ping.ping() * 2
        }
    });
    perform Thread.join(t)
}
"""
    result, _ = run_source(source)
    assert result == 42


def test_in_scope_handler_overrides_spawn_runtime_mapping():
    """Virtualization preserved: a Thread.spawn handler in scope on the
    performing thread intercepts the perform — no OS thread is created,
    and the handler decides the value."""
    source = THREAD_EFFECT + """
fn main() -> int {
    handle Thread with {
        spawn(f) -> {
            print("intercepted");
            resume(1234)
        }
    } in {
        perform Thread.spawn(|| { 1 })
    }
}
"""
    result, prints = run_source(source)
    assert result == 1234
    assert prints == ["intercepted"]
