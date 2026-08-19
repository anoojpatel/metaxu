/* metaxu_threads.c -- pthreads-backed Thread/Mutex effect primitives.
 *
 * See metaxu_threads.h for the full contract and docs/threads_runtime.md
 * for the spec.  The reference semantics is the MIR interpreter
 * (mir_interp._rt_thread_* / _rt_mutex_*); every catchable error message
 * here matches its wording byte for byte.
 *
 * Composition with the effects runtime (metaxu_effects.c): all of its
 * scheduler state is _Thread_local, so a child thread's first touch of
 * any effect machinery sees a fresh zero-initialized instance -- no
 * explicit per-thread initialization is needed here beyond running the
 * closure under an mx_try pad (the thread's outermost landing pad, which
 * captures catchable failures for re-raise at join). */
#define _XOPEN_SOURCE 700

#include "metaxu_threads.h"
#include "metaxu_effects.h"

#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* FATAL failures with no interpreter counterpart (allocation / OS errors):
 * same shape as mx_rt_fail / mx__fatal -- flush stdout first (abort() does
 * not), name the failure, abort. */
static void mx_thr_fatal(const char *msg, int err) {
    fflush(stdout);
    if (err != 0) {
        fprintf(stderr, "metaxu threads runtime: %s: %s\n", msg,
                strerror(err));
    } else {
        fprintf(stderr, "metaxu threads runtime: %s\n", msg);
    }
    fflush(stderr);
    abort();
}

/* Process-wide id counters (ids appear in the catchable error messages;
 * the interpreter numbers from 1 in creation order too). */
static _Atomic int64_t g_next_thread_id = 1;
static _Atomic int64_t g_next_mutex_id = 1;

/* Contention permission (docs/contention_as_permission.md): a per-thread
 * count of mutexes held through this runtime.  _Thread_local, so spawned
 * threads start at 0 and handle-body fibers (same OS thread) share the
 * logical thread's grant, exactly like pthread mutex ownership.  Bumped
 * only on SUCCESSFUL lock/unlock -- see the error paths below. */
static _Thread_local int64_t g_write_permit = 0;

int64_t mx__write_permit(void) {
    return g_write_permit;
}

/* ------------------------------------------------------------------------
 * Threads
 * ---------------------------------------------------------------------- */

/* The IMMORTAL thread handle (never freed, leak by design): double-join is
 * a flag check, never a use-after-free, and the handle word can be stored,
 * copied and shared freely.  Field-access safety: fn/env/thread_id are
 * written before pthread_create publishes them to the child; result/error
 * are written by the child and read by the joiner strictly after
 * pthread_join's happens-before edge; `joined` is claimed by atomic
 * exchange. */
typedef struct {
    pthread_t tid;
    int64_t (*fn)(void *env);   /* word-uniform closure entry */
    void *env;                  /* immortal heap env (compiler-forced) */
    int64_t thread_id;
    int64_t result;             /* closure's result word */
    char *error;                /* child's catchable failure msg, or NULL */
    atomic_bool joined;
} mx_thread;

static int64_t mx__thread_body(void *arg) {
    mx_thread *t = (mx_thread *)arg;
    return t->fn(t->env);
}

static int64_t mx__thread_catch(void *arg, const char *msg) {
    /* The child's outermost catch: store the failure for the joiner.  The
     * copy is immortal alongside the handle (mx_raise at join copies it
     * again for the catch binding, per its own contract). */
    mx_thread *t = (mx_thread *)arg;
    size_t n = strlen(msg) + 1;
    char *copy = (char *)malloc(n);
    if (copy == NULL)
        mx_thr_fatal("out of memory (child failure message)", 0);
    memcpy(copy, msg, n);
    t->error = copy;
    return 0;
}

static void *mx__thread_main(void *arg) {
    /* This thread's effects state (scope stack, pad chain) is fresh
     * _Thread_local storage: effect-scope isolation by construction.
     * mx_try installs the thread's outermost pad, so a catchable failure
     * anywhere in the child that no inner try recovers lands in
     * mx__thread_catch instead of aborting the process -- the parent must
     * never see a child failure before join (docs/threads_runtime.md).
     * Fatal failures still abort the process at the failure point. */
    mx_thread *t = (mx_thread *)arg;
    t->result = mx_try(mx__thread_body, t, mx__thread_catch, t);
    return NULL;
}

int64_t mx_thread_spawn(void *fn, void *env) {
    if (fn == NULL)
        mx_thr_fatal("EFFECT_SPAWN: NULL closure function", 0);
    mx_thread *t = (mx_thread *)calloc(1, sizeof(mx_thread));
    if (t == NULL)
        mx_thr_fatal("out of memory (thread handle)", 0);
    t->fn = (int64_t (*)(void *))fn;
    t->env = env;
    t->thread_id = atomic_fetch_add(&g_next_thread_id, 1);
    int rc = pthread_create(&t->tid, NULL, mx__thread_main, t);
    if (rc != 0)
        mx_thr_fatal("EFFECT_SPAWN: pthread_create failed", rc);
    return (int64_t)(intptr_t)t;
}

int64_t mx_thread_join(int64_t handle) {
    mx_thread *t = (mx_thread *)(intptr_t)handle;
    if (t == NULL)
        mx_thr_fatal("EFFECT_JOIN: NULL thread handle", 0);
    if (atomic_exchange(&t->joined, true)) {
        /* CATCHABLE, and byte-identical to the interpreter's wording
         * (mir_interp._rt_thread_join). */
        mx_raisef("EFFECT_JOIN on <Thread#%lld joined>: "
                  "thread already joined", (long long)t->thread_id);
    }
    int rc = pthread_join(t->tid, NULL);
    if (rc != 0)
        mx_thr_fatal("EFFECT_JOIN: pthread_join failed", rc);
    if (t->error != NULL) {
        /* Re-raise the child's catchable failure on the JOINING thread --
         * an enclosing try here binds the child's exact message. */
        mx_raise(t->error);
    }
    return t->result;
}

/* ------------------------------------------------------------------------
 * Mutexes (PTHREAD_MUTEX_ERRORCHECK; immortal handles)
 * ---------------------------------------------------------------------- */

typedef struct {
    pthread_mutex_t mu;
    int64_t mutex_id;
} mx_mutex;

int64_t mx_mutex_create(void) {
    mx_mutex *m = (mx_mutex *)calloc(1, sizeof(mx_mutex));
    if (m == NULL)
        mx_thr_fatal("out of memory (mutex)", 0);
    pthread_mutexattr_t attr;
    int rc = pthread_mutexattr_init(&attr);
    if (rc != 0)
        mx_thr_fatal("EFFECT_MUTEX_CREATE: mutexattr_init failed", rc);
    /* ERRORCHECK is the semantics contract, not a debug nicety: self-
     * relock must return EDEADLK (default mutexes DEADLOCK there) and
     * unlock-not-held must return EPERM (default mutexes are UB there). */
    rc = pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);
    if (rc != 0)
        mx_thr_fatal("EFFECT_MUTEX_CREATE: mutexattr_settype failed", rc);
    rc = pthread_mutex_init(&m->mu, &attr);
    if (rc != 0)
        mx_thr_fatal("EFFECT_MUTEX_CREATE: mutex_init failed", rc);
    pthread_mutexattr_destroy(&attr);
    m->mutex_id = atomic_fetch_add(&g_next_mutex_id, 1);
    return (int64_t)(intptr_t)m;
}

int64_t mx_mutex_lock(int64_t handle) {
    mx_mutex *m = (mx_mutex *)(intptr_t)handle;
    if (m == NULL)
        mx_thr_fatal("EFFECT_MUTEX_LOCK: NULL mutex handle", 0);
    int rc = pthread_mutex_lock(&m->mu); /* held by another thread: BLOCKS */
    if (rc == EDEADLK) {
        /* CATCHABLE; interpreter wording (mir_interp._rt_mutex_lock). */
        mx_raisef("deadlock: EFFECT_MUTEX_LOCK on <Mutex#%lld>: "
                  "this thread already holds it", (long long)m->mutex_id);
    }
    if (rc != 0)
        mx_thr_fatal("EFFECT_MUTEX_LOCK: pthread_mutex_lock failed", rc);
    /* Lock HELD from here: grant write permission.  Strictly after the
     * error paths -- a failed ERRORCHECK lock must not bump. */
    g_write_permit += 1;
    return 0; /* unit */
}

int64_t mx_mutex_unlock(int64_t handle) {
    mx_mutex *m = (mx_mutex *)(intptr_t)handle;
    if (m == NULL)
        mx_thr_fatal("EFFECT_MUTEX_UNLOCK: NULL mutex handle", 0);
    /* Revoke write permission BEFORE the pthread call: once the unlock
     * succeeds the lock is gone, and no code on this thread runs between
     * the two statements to observe an inconsistent counter.  The EPERM
     * error path (unlock of a mutex this thread does not hold) restores
     * the counter before raising -- a failed unlock changes nothing. */
    g_write_permit -= 1;
    int rc = pthread_mutex_unlock(&m->mu);
    if (rc == EPERM) {
        g_write_permit += 1;
        /* One unified message for "unlocked" and "held by another
         * thread" (EPERM covers both, and the momentary state is racy to
         * print) -- interpreter wording (mir_interp._rt_mutex_unlock). */
        mx_raisef("EFFECT_MUTEX_UNLOCK on <Mutex#%lld>: "
                  "this thread does not hold it", (long long)m->mutex_id);
    }
    if (rc != 0)
        mx_thr_fatal("EFFECT_MUTEX_UNLOCK: pthread_mutex_unlock failed", rc);
    return 0; /* unit */
}
