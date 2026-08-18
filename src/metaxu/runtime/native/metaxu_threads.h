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

#ifdef __cplusplus
}
#endif

#endif /* METAXU_THREADS_H */
