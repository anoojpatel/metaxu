/* metaxu_threads.h -- pthreads-backed Thread/Mutex effect primitives.
 *
 * The native implementation of the `with EFFECT_*` runtime mappings in
 * effect_mapping.mx (spec: docs/threads_runtime.md; reference semantics:
 * mir_interp._rt_thread_* / _rt_mutex_*, whose error-message wording is
 * shared byte for byte -- the caught value is language-visible).
 *
 * | symbol              | signature                                  |
 * |---------------------|--------------------------------------------|
 * | EFFECT_SPAWN        | int64_t mx_thread_spawn(void *fn, void *env)|
 * | EFFECT_JOIN         | int64_t mx_thread_join(int64_t handle)     |
 * | EFFECT_MUTEX_CREATE | int64_t mx_mutex_create(void)              |
 * | EFFECT_MUTEX_LOCK   | int64_t mx_mutex_lock(int64_t m)           |
 * | EFFECT_MUTEX_UNLOCK | int64_t mx_mutex_unlock(int64_t m)         |
 *
 * mx_thread_spawn starts `fn` -- a word-uniform zero-argument closure
 * entry, `int64_t fn(void *env)` -- on a NEW pthread immediately and
 * returns an opaque handle word without waiting.  The env must be an
 * immortal heap block (the compiler forces heap-env for every lambda
 * whose closure reaches a spawn -- the child dereferences it after the
 * spawning frame has moved on).  The child's effects-scheduler state is
 * _Thread_local (metaxu_effects.c), so it starts with an EMPTY scope
 * stack: effect-scope isolation holds by construction.  The child body
 * runs under an mx_try landing pad: a CATCHABLE failure the child does
 * not recover itself is stored on the handle and re-raised -- catchably,
 * via mx_raise -- by mx_thread_join on the joining thread.  FATAL
 * failures (mx__fatal / mx_rt_fail) still abort the whole process at the
 * failure point, on whatever thread (documented divergence: the
 * interpreter surfaces them at join; both end the program naming the
 * failure, and only schedule-independent observations are specified).
 *
 * mx_thread_join blocks (pthread_join) and returns the child's result
 * word exactly once; a second join of the same handle raises the
 * interpreter's exact catchable error ("EFFECT_JOIN on <Thread#N
 * joined>: thread already joined").  The single join is claimed with an
 * atomic exchange, so two racing joiners resolve safely: one wins, one
 * raises.  Handles are IMMORTAL (never freed, leak by design) -- a
 * double join is a flag check, never a use-after-free.  A handle that is
 * never joined is a detached thread: process exit does not wait for it.
 *
 * Mutexes are PTHREAD_MUTEX_ERRORCHECK behind opaque immortal handle
 * words: lock of a mutex held by ANOTHER thread BLOCKS; self-relock
 * (EDEADLK) raises "deadlock: EFFECT_MUTEX_LOCK on <Mutex#N>: this
 * thread already holds it"; unlock of a mutex this thread does not hold
 * (EPERM -- covers both "unlocked" and "held by another thread") raises
 * "EFFECT_MUTEX_UNLOCK on <Mutex#N>: this thread does not hold it".
 * Both are CATCHABLE (the interpreter raises InterpError there).
 * lock/unlock return the unit word 0.
 *
 * Ids: thread and mutex ids are process-wide C11 atomic counters starting
 * at 1 (they appear in the error messages above; creation order is only
 * deterministic when all creation happens on one thread -- tests must
 * stay schedule-independent).
 *
 * Everything here is thread-safe by construction: handles are written by
 * one side at a time (spawn before the child exists; result/error by the
 * child before it exits, read by join after pthread_join's happens-before
 * edge), and the id counters are atomics.
 */
#ifndef METAXU_THREADS_H
#define METAXU_THREADS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int64_t mx_thread_spawn(void *fn, void *env);
int64_t mx_thread_join(int64_t handle);

int64_t mx_mutex_create(void);
int64_t mx_mutex_lock(int64_t handle);
int64_t mx_mutex_unlock(int64_t handle);

/* Contention permission (docs/contention_as_permission.md): the calling
 * thread's held-mutex count.  _Thread_local, bumped by mx_mutex_lock
 * (+1, only after a SUCCESSFUL lock -- a failed ERRORCHECK lock never
 * bumps) and mx_mutex_unlock (-1, restored on the EPERM error path).
 * Spawned threads start at 0 (fresh TLS); handle bodies are ucontext
 * fibers on the SAME OS thread, so the permission follows the logical
 * thread exactly like mutex ownership does.  Read by the Vec mutators'
 * contended-write guard (metaxu_rt.c mx__vec_write_check).
 *
 * The counter is exported as a variable, not only through the accessor:
 * the guard sits on the mutator fast path, and an opaque cross-TU CALL
 * in that path (even behind a never-taken branch) measurably bloats the
 * uncrossed case -- the direct TLS load keeps it to a compare.  The
 * accessor exists for tests/tools; runtime-internal readers use the
 * variable.  tls_model("initial-exec"): under the default -fPIC these
 * objects would use the general-dynamic __tls_get_addr CALL sequence,
 * whose clobbers force register spills in the mutators even on the
 * not-taken path (measured; see the spec's numbers); IE is a plain
 * %fs-relative load and is always valid here because the runtime links
 * into executables, never into dlopen'd libraries. */
extern _Thread_local int64_t mx__tls_write_permit
    __attribute__((tls_model("initial-exec")));
int64_t mx__write_permit(void);

#ifdef __cplusplus
}
#endif

#endif /* METAXU_THREADS_H */
