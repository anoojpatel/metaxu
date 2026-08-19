"""Contention as effect-granted permission (docs/contention_as_permission.md).

Under test, on BOTH engines (any asymmetry is a bug):

* A Vec captured by a closure crossing a REAL spawn is marked contended
  — direct Vec captures plus recursion through struct fields by static
  layout, STOPPING at Vec elements (the vec-of-vecs hole is pinned here
  on purpose so it stays visible, never silently divergent).
* MUTATING a contended Vec (push / pop / index-set) on a thread holding
  no runtime mutex raises CATCHABLY with the spec's exact wording,
  byte-identical across engines.  Reads stay free.
* Permission is the per-logical-thread held-mutex count, granted by the
  mutex runtime (lock +1 / unlock -1) — so the same write under a lock
  passes, and std.sync's protect/update idiom is untouched (its cell IS
  marked, and every access holds the lock).

All tests go through parsed source (repo convention); the static
separateness check is deliberately bypassed with `unsafe { }`, which is
exactly the flow this dynamic net exists for (spec § interaction with
the static layer).

NON-VACUITY (verified while landing this file, 2026-08-19): with the
marking walk in mir_interp._rt_thread_spawn commented out, every
raises-at-runtime test here fails (the unprotected writes succeed
silently); with the walk restored they pass.  The native differentials
repeat the same experiment through the emitted mx_vec_mark_contended
calls — deleting the emission makes the native halves of the
differentials print "no error" and diverge.
"""
from __future__ import annotations

import pytest

from metaxu.compiler.mir_interp import InterpError

from metaxu.compiler.tests.test_codegen_llvm import (
    assert_native_matches_interp,
    compile_and_run,
    count_placeholders,
    interp_run,
    llvm_from_source,
    needs_clang,
    needs_tsan,
)
from metaxu.compiler.tests.test_threads import (
    MUTEX_EFFECT,
    THREAD_EFFECT,
    run_source,
)

# The spec's exact wording (docs/contention_as_permission.md § the idea);
# shared byte-for-byte by mir_interp._CONTENDED_WRITE_MSG and
# metaxu_rt.c mx__vec_write_check.  The caught value is language-visible.
CONTENDED_MSG = (
    "write to contended Vec without a held lock: this value crossed "
    "a thread boundary at spawn; mutate it under a mutex "
    "(std.sync.with_lock) or keep it thread-local")

DECLS = THREAD_EFFECT + MUTEX_EFFECT


# ---------------------------------------------------------------------------
# The core rule: unprotected writes raise (catchably); locked writes pass;
# reads stay free
# ---------------------------------------------------------------------------

# The child catches its own violation with `try` — proving the raise is
# CATCHABLE — and returns; main then reads the vec, proving the write
# never landed.  Every observation is schedule-independent.
UNPROTECTED_WRITE_SRC = DECLS + """
fn main() -> int {
    let @mut v = Vec.new();
    v.push(7);
    let @mut handles = Vec.new();
    unsafe {
        let t = perform Thread.spawn(|| {
            let msg = try { v[0] = 1; "no error" } catch e { e };
            print(msg);
            0
        });
        handles.push(t);
    }
    perform Thread.join(handles[0]);
    v[0]
}
"""


def test_unprotected_write_to_spawn_captured_vec_raises_catchably():
    result, prints = run_source(UNPROTECTED_WRITE_SRC)
    assert prints == [CONTENDED_MSG]
    assert result == 7  # the write never landed


def test_uncaught_violation_surfaces_at_join_with_exact_message():
    """Without a `try` in the child, the violation is the child's failure
    and join re-raises it on the joining thread (docs/threads_runtime.md
    § failure in a child thread) — with the spec's exact wording."""
    source = DECLS + """
fn main() -> int {
    let @mut v = Vec.new();
    v.push(0);
    let @mut handles = Vec.new();
    unsafe {
        let t = perform Thread.spawn(|| { v.push(1); 0 });
        handles.push(t);
    }
    perform Thread.join(handles[0]);
    0
}
"""
    with pytest.raises(InterpError) as excinfo:
        run_source(source)
    assert excinfo.value.message == CONTENDED_MSG


def test_push_and_pop_are_guarded_too():
    """Every Vec MUTATOR is checked: push, pop and index-set all raise
    the same wording on a contended vec with no lock held."""
    source = DECLS + """
fn main() -> int {
    let @mut v = Vec.new();
    v.push(1);
    let @mut handles = Vec.new();
    unsafe {
        let t = perform Thread.spawn(|| {
            let a = try { v.push(2); "no error" } catch e { e };
            print(a);
            let b = try { v.pop(); "no error" } catch e { e };
            print(b);
            let c = try { v[0] = 9; "no error" } catch e { e };
            print(c);
            0
        });
        handles.push(t);
    }
    perform Thread.join(handles[0]);
    v.len()
}
"""
    result, prints = run_source(source)
    assert prints == [CONTENDED_MSG, CONTENDED_MSG, CONTENDED_MSG]
    assert result == 1  # nothing mutated


LOCKED_WRITE_SRC = DECLS + """
fn main() -> int {
    let m = perform Mutex.create();
    let @mut v = Vec.new();
    v.push(0);
    let @mut handles = Vec.new();
    unsafe {
        let t = perform Thread.spawn(|| {
            let @mut j = 0;
            while j < 100 {
                perform Mutex.lock(m);
                v[0] = v[0] + 1;
                perform Mutex.unlock(m);
                j = j + 1
            };
            0
        });
        handles.push(t);
    }
    perform Thread.join(handles[0]);
    v[0]
}
"""


def test_same_write_under_lock_passes():
    result, _ = run_source(LOCKED_WRITE_SRC)
    assert result == 100


def test_reads_of_contended_vec_stay_free():
    """Contention weakens access (writes need permission); it never
    revokes it: len / index reads / iteration of a contended vec need no
    lock, in the child AND in the spawner afterwards."""
    source = DECLS + """
fn main() -> int {
    let @mut v = Vec.new();
    v.push(10);
    v.push(20);
    let @mut handles = Vec.new();
    unsafe {
        let t = perform Thread.spawn(|| { v[0] + v[1] + v.len() });
        handles.push(t);
    }
    let child_sum = perform Thread.join(handles[0]);
    let @mut total = 0;
    for x in v { total = total + x };
    child_sum + total + v.len()
}
"""
    result, _ = run_source(source)
    assert result == 32 + 30 + 2


def test_spawner_also_needs_the_lock_after_the_crossing():
    """The mark is on the VALUE, not the thread: after the spawn, the
    SPAWNING thread's own unprotected writes to the shared vec raise
    too (both sides of the race are caught)."""
    source = DECLS + """
fn main() -> int {
    let m = perform Mutex.create();
    let @mut v = Vec.new();
    v.push(0);
    let @mut handles = Vec.new();
    unsafe {
        let t = perform Thread.spawn(|| {
            perform Mutex.lock(m);
            v[0] = v[0] + 1;
            perform Mutex.unlock(m);
            0
        });
        handles.push(t);
    }
    let msg = try { v[0] = 5; "no error" } catch e { e };
    print(msg);
    perform Thread.join(handles[0]);
    perform Mutex.lock(m);
    let out = v[0];
    perform Mutex.unlock(m);
    out
}
"""
    result, prints = run_source(source)
    assert prints == [CONTENDED_MSG]
    assert result == 1


# ---------------------------------------------------------------------------
# std.sync stays untouched: the Protected cell is marked, but every
# access holds the lock
# ---------------------------------------------------------------------------

def test_std_sync_protect_update_counter_passes_untouched():
    """The blessed idiom needs no exemption machinery: protect()'s cell
    IS marked contended when the Protected handle crosses (struct-field
    recursion reaches the Vec cell), and every accessor holds the mutex,
    so the permission is present exactly when the writes happen."""
    source = THREAD_EFFECT + """
from std.sync import protect, update, read;

fn main() -> int {
    let p = protect(0);
    let @mut handles = Vec.new();
    let @mut i = 0;
    while i < 2 {
        let t = perform Thread.spawn(|| {
            let @mut j = 0;
            while j < 50 {
                update(p, fn(n: int) -> n + 1);
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
    read(p)
}
"""
    result, _ = run_source(source)
    assert result == 100


# ---------------------------------------------------------------------------
# Marking rule: struct fields recurse; vec elements do NOT (pinned hole)
# ---------------------------------------------------------------------------

STRUCT_FIELD_SRC = DECLS + """
struct Holder { tag: int, data: Vec }

fn main() -> int {
    let @mut inner = Vec.new();
    inner.push(3);
    let h = Holder { tag: 1, data: inner };
    let @mut handles = Vec.new();
    unsafe {
        let t = perform Thread.spawn(|| {
            let msg = try { h.data[0] = 9; "no error" } catch e { e };
            print(msg);
            0
        });
        handles.push(t);
    }
    perform Thread.join(handles[0]);
    inner[0]
}
"""


def test_struct_field_vec_is_marked_through_static_layout():
    """A struct captured by the spawn is walked field-by-field: the Vec
    inside it (shared identity — the struct copy aliases one vector) is
    marked, so the child's unprotected write through the field raises."""
    result, prints = run_source(STRUCT_FIELD_SRC)
    assert prints == [CONTENDED_MSG]
    assert result == 3


def test_vec_of_vecs_inner_write_does_not_raise_the_documented_hole():
    """THE PINNED HOLE (docs/contention_as_permission.md § marking rule):
    marking STOPS at Vec elements.  A captured Vec-of-Vecs marks only the
    OUTER vector — enumerating runtime elements is possible for the
    interpreter but not for native code, and the engines must agree — so
    an unprotected write to an INNER vec reached through the outer one
    does NOT raise.  This test asserts the hole so it stays visible; if
    marking ever starts descending into elements, update the spec first,
    then this test."""
    source = DECLS + """
fn main() -> int {
    let @mut outer = Vec.new();
    let @mut inner = Vec.new();
    inner.push(0);
    outer.push(inner);
    let @mut handles = Vec.new();
    unsafe {
        let t = perform Thread.spawn(|| {
            let mine = outer[0];
            let a = try { mine[0] = 42; "no error" } catch e { e };
            print(a);
            let b = try { outer.push(mine); "outer raised not" } catch e { "outer raised" };
            print(b);
            0
        });
        handles.push(t);
    }
    perform Thread.join(handles[0]);
    let back = outer[0];
    back[0]
}
"""
    result, prints = run_source(source)
    # Inner write sails through (the hole); the OUTER vec is marked.
    assert prints == ["no error", "outer raised"]
    assert result == 42


# ---------------------------------------------------------------------------
# Native differentials: stdout/exit and the caught message byte-identical
# ---------------------------------------------------------------------------

# print-based variants (native exit codes only carry ints mod 256; the
# differential helper compares stdout and exit).

_NATIVE_UNPROTECTED_SRC = UNPROTECTED_WRITE_SRC.replace(
    "    perform Thread.join(handles[0]);\n    v[0]\n}",
    "    perform Thread.join(handles[0]);\n"
    "    print(v[0].to_string());\n    0\n}")

_NATIVE_LOCKED_SRC = LOCKED_WRITE_SRC.replace(
    "    perform Thread.join(handles[0]);\n    v[0]\n}",
    "    perform Thread.join(handles[0]);\n"
    "    print(v[0].to_string());\n    0\n}")

_NATIVE_STRUCT_FIELD_SRC = STRUCT_FIELD_SRC.replace(
    "    perform Thread.join(handles[0]);\n    inner[0]\n}",
    "    perform Thread.join(handles[0]);\n"
    "    print(inner[0].to_string());\n    0\n}")


@needs_clang
def test_native_unprotected_write_matches_interp(tmp_path):
    """The caught contended-write message is a language-visible value:
    native output must be byte-identical to the interpreter's."""
    ir = assert_native_matches_interp(_NATIVE_UNPROTECTED_SRC, tmp_path)
    assert count_placeholders(ir) == 0
    assert "call void @mx_vec_mark_contended(ptr" in ir


@needs_clang
def test_native_locked_write_matches_interp(tmp_path):
    ir = assert_native_matches_interp(_NATIVE_LOCKED_SRC, tmp_path)
    assert count_placeholders(ir) == 0


@needs_clang
def test_native_struct_field_marking_matches_interp(tmp_path):
    """Struct-field recursion is emitted from the STATIC layout at the
    spawn thunk; the nested write raises identically on both engines."""
    ir = assert_native_matches_interp(_NATIVE_STRUCT_FIELD_SRC, tmp_path)
    assert count_placeholders(ir) == 0
    assert "call void @mx_vec_mark_contended(ptr" in ir


@needs_clang
def test_native_vec_of_vecs_hole_matches_interp(tmp_path):
    """The pinned hole is pinned on BOTH engines: the inner write is free,
    the outer mutation raises, natively exactly as in the interpreter."""
    source = DECLS + """
fn main() -> int {
    let @mut outer = Vec.new();
    let @mut inner = Vec.new();
    inner.push(0);
    outer.push(inner);
    let @mut handles = Vec.new();
    unsafe {
        let t = perform Thread.spawn(|| {
            let mine = outer[0];
            let a = try { mine[0] = 42; "no error" } catch e { e };
            print(a);
            0
        });
        handles.push(t);
    }
    perform Thread.join(handles[0]);
    let back = outer[0];
    print(back[0].to_string());
    0
}
"""
    ir = assert_native_matches_interp(source, tmp_path)
    assert count_placeholders(ir) == 0


# ---------------------------------------------------------------------------
# The old TSan non-vacuity experiment, inverted: deleting the locks from
# the counter now aborts deterministically instead of racing
# ---------------------------------------------------------------------------

# test_codegen_llvm._THREAD_COUNTER_SRC with the lock/unlock lines
# DELETED and each child catching its own failure: every child's FIRST
# unprotected write raises (deterministically — the vec was marked at
# spawn, before the child existed), so no data race ever happens and the
# counter stays 0.  Schedule-independent on both engines.
_RACY_EXPERIMENT_SRC = DECLS + """
fn main() -> int {
    let @mut counter = Vec.new();
    counter.push(0);
    let @mut handles = Vec.new();
    let @mut i = 0;
    unsafe {
        while i < 4 {
            let t = perform Thread.spawn(|| {
                let msg = try {
                    counter[0] = counter[0] + 1;
                    "no error"
                } catch e { e };
                print(msg);
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
    print(counter[0].to_string());
    0
}
"""


def test_racy_counter_without_locks_now_raises_deterministically():
    """The TSan non-vacuity experiment from the threads work, with its
    semantics upgraded by this feature: deleting the lock/unlock lines no
    longer races — every child's first unprotected write raises the
    contended-write error before touching the slot, and the counter
    stays 0 (docs/contention_as_permission.md § interaction with the
    static layer)."""
    result, prints = run_source(_RACY_EXPERIMENT_SRC)
    assert prints == [CONTENDED_MSG] * 4 + ["0"]
    assert result == 0


@needs_tsan
def test_native_racy_counter_aborts_with_error_not_a_race_tsan(tmp_path):
    """Same program natively, under -fsanitize=thread: the unprotected
    writes raise (identical stdout) BEFORE any racing store happens, so
    TSan sees zero reports (a report would change the exit code and break
    the differential).  Before this feature the locks-deleted counter was
    the documented TSan data-race reproducer."""
    result, expected_out = interp_run(_RACY_EXPERIMENT_SRC)
    assert result == 0
    ir = llvm_from_source(_RACY_EXPERIMENT_SRC)
    assert count_placeholders(ir) == 0
    exit_code, stdout = compile_and_run(
        ir, "main", workdir=str(tmp_path),
        clang_args=("-fsanitize=thread",))
    assert stdout == expected_out
    assert exit_code == 0


@needs_tsan
def test_native_locked_contended_write_tsan_clean(tmp_path):
    """The contended-and-locked path (the one that pays the thread-local
    permit read) is TSan-clean: marking happens before pthread_create
    (happens-before the child) and the flag is a relaxed atomic."""
    result, expected_out = interp_run(_NATIVE_LOCKED_SRC)
    assert result == 0
    ir = llvm_from_source(_NATIVE_LOCKED_SRC)
    exit_code, stdout = compile_and_run(
        ir, "main", workdir=str(tmp_path),
        clang_args=("-fsanitize=thread",))
    assert stdout == expected_out
    assert exit_code == 0


# ---------------------------------------------------------------------------
# Virtualized spawns do NOT mark (handler presence subtracts the crossing)
# ---------------------------------------------------------------------------

def test_handler_virtualized_spawn_does_not_mark():
    """A spawn intercepted by an in-scope handler never crosses a real
    thread: the closure's captures are NOT marked, and a later write
    needs no lock.  (Natively the marking lives inside the EFFECT_SPAWN
    runtime thunk, which an intercepted perform never calls — the same
    routing decides both engines.)"""
    source = DECLS + """
fn main() -> int {
    let @mut v = Vec.new();
    v.push(1);
    let r = handle Thread with {
        spawn(f) -> { resume(0) }
    } in {
        perform Thread.spawn(|| { v[0] = 5; 0 });
        0
    };
    v[0] = v[0] + 10;
    v[0] + r
}
"""
    result, _ = run_source(source)
    assert result == 11
