/* metaxu_effects.c -- native algebraic effects: delimited single-shot
 * deep-handler continuations on stackful ucontext coroutines.
 *
 * See metaxu_effects.h for the full contract.  The reference semantics is
 * the MIR interpreter (src/metaxu/compiler/mir_interp.py); the mapping is:
 *
 *   interpreter                          this file
 *   -----------------------------------  --------------------------------
 *   handle body thread (parked)          body coroutine (ucontext + stack)
 *   _mir_handler_frames (global list)    g_top scope stack (innermost first)
 *   _find_mir_frame (busy skip, effect   mx__find_scope
 *     match, innermost op match)
 *   _pump_scope PERFORM -> case call     dispatcher call in mx_handle /
 *     with busy=True                       mx_resume with s->busy = 1
 *   resume() -> unpark body, wait for    mx_resume: swap to the perform
 *     next event (busy off meanwhile)      site, dispatch next event
 *   _EffectAbort unwind to handle_scope  longjmp(s->abort_jmp)
 *   _ScopeAbort cascade teardown of      mx__teardown_to frees the parked
 *     nested parked bodies                 coroutine stacks outright (C has
 *                                          no destructors to run on them)
 *
 * Control-transfer invariants (why this is safe):
 *   - all handler-side activity for a scope S (mx_handle's dispatch, every
 *     mx_resume on S's continuations, longjmp to S's abort_jmp) runs on
 *     S's OWNER stack -- the context that called mx_handle.  The compiler
 *     guarantees this by only emitting `resume` inside the handler-case
 *     functions of the scope that received the continuation (a resume
 *     smuggled into a nested scope body demotes to the interpreter).
 *   - exactly one side of a scope runs at a time; events (ev_*) are
 *     written by the body side strictly before switching to the handler
 *     side, and the resume value is written strictly before switching
 *     back, so no field is ever concurrently accessed.
 *   - at normal completion (DONE) the scope is the innermost live scope
 *     (nested handles inside the body completed structurally before the
 *     body returned); at abort, every scope above it on the stack belongs
 *     to its parked body chain and is freed with it.
 */
#define _XOPEN_SOURCE 700

#include "metaxu_effects.h"

#include <setjmp.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <ucontext.h>

/* ASan fiber-switch annotations: without them the fake-stack machinery
 * misattributes frames after swapcontext and reports false positives. */
#if defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define MX_ASAN 1
#  endif
#endif
#if !defined(MX_ASAN) && defined(__SANITIZE_ADDRESS__)
#  define MX_ASAN 1
#endif
#ifdef MX_ASAN
void __sanitizer_start_switch_fiber(void **fake_stack_save,
                                    const void *bottom, size_t size);
void __sanitizer_finish_switch_fiber(void *fake_stack_save,
                                     const void **bottom_old,
                                     size_t *size_old);
#endif

#define MX_EFFECT_STACK_SIZE ((size_t)1 << 20) /* 1 MiB per body coroutine */

enum { MX_EV_DONE = 0, MX_EV_PERFORM = 1 };

typedef struct {
    const void *bottom;
    size_t size;
} mx_stackinfo;

typedef struct mx_scope mx_scope;

struct mx_k {
    mx_scope *scope;
    ucontext_t resume_ctx;  /* the parked perform site */
    int64_t resume_value;   /* handler -> body payload */
    int used;               /* single-shot flag; checked FIRST in mx_resume */
    mx_stackinfo stack;     /* the parked fiber's stack (ASan annotation) */
    struct mx_k *next;      /* scope's list, freed at teardown */
};

struct mx_scope {
    /* configuration (from mx_handle) */
    mx_handler_fn handler;
    void *handler_env;
    const char *effect;             /* "" or NULL = matches any effect */
    const char *const *op_names;
    const int64_t *op_nparams;
    int64_t nops;
    /* body coroutine */
    mx_body_fn body;
    void *body_env;
    ucontext_t body_ctx;            /* initial makecontext target */
    void *stack;
    /* handler-side wait point: every body->handler transfer targets this;
     * re-saved by each mx_resume before switching into the body. */
    ucontext_t handler_ctx;
    mx_stackinfo owner_stack;       /* owner context's stack (ASan) */
    int owner_was_main;
    /* dispatch state */
    int busy;                       /* a handler case of this scope is running */
    /* event: body side fills, then switches to handler_ctx */
    int ev_kind;                    /* MX_EV_DONE / MX_EV_PERFORM */
    int64_t ev_op_index;
    int64_t ev_args[MX_EFFECT_MAX_ARGS];
    int64_t done_value;
    mx_k *ev_k;
    /* abort unwinding (handler case returned without resuming while inside
     * a nested mx_resume pump): longjmp back to this scope's mx_handle */
    jmp_buf abort_jmp;
    int64_t abort_value;
    /* bookkeeping */
    mx_k *k_list;
    mx_scope *prev;                 /* global scope stack link */
};

/* Global scope stack, innermost scope first (single-threaded by design). */
static mx_scope *g_top = NULL;

/* The currently running fiber's stack ({NULL, 0} = the main thread before
 * its bounds are learned, or unknown); maintained across every switch so
 * mx_perform can stamp continuations with the stack they park on. */
static mx_stackinfo g_cur = {NULL, 0};
/* Main-thread stack bounds, learned from the first coroutine entered
 * directly from the main context (ASan reports them at fiber entry). */
static mx_stackinfo g_main = {NULL, 0};

static void mx__fatal(const char *msg) {
    fprintf(stderr, "metaxu effects runtime: %s\n", msg);
    fflush(stderr);
    abort();
}

/* Switch to `to`, saving the current context in `save`; returns when some
 * later switch targets `save`.  `to_stack` is the target fiber's stack
 * (ASan annotation only). */
static void mx__switch(ucontext_t *save, ucontext_t *to, mx_stackinfo to_stack) {
    mx_stackinfo my = g_cur;
#ifdef MX_ASAN
    void *fake = NULL;
    __sanitizer_start_switch_fiber(&fake, to_stack.bottom, to_stack.size);
#endif
    g_cur = to_stack;
    if (swapcontext(save, to) != 0)
        mx__fatal("swapcontext failed");
    /* control has come back to this context */
#ifdef MX_ASAN
    __sanitizer_finish_switch_fiber(fake, NULL, NULL);
#endif
    g_cur = my;
}

/* Final switch away from a dying fiber (its fake stack is destroyed). */
static void mx__switch_dead(ucontext_t *to, mx_stackinfo to_stack) {
#ifdef MX_ASAN
    __sanitizer_start_switch_fiber(NULL, to_stack.bottom, to_stack.size);
#endif
    g_cur = to_stack;
    setcontext(to);
    mx__fatal("setcontext returned"); /* unreachable */
}

/* Free every scope from the top of the stack down to and including `s`.
 * Called on s's owner context.  Scopes above s (abort case) are the ones
 * nested inside s's parked body chain: their coroutines are parked and
 * will never run again, and C has no destructors, so freeing their stacks
 * outright is observationally equivalent to the interpreter's cascading
 * _ScopeAbort unwind.  Continuation records are freed with their scope
 * (the surface language cannot store a continuation past its scope). */
static void mx__teardown_to(mx_scope *s) {
    for (;;) {
        mx_scope *x = g_top;
        if (x == NULL)
            mx__fatal("scope stack corrupted (teardown past bottom)");
        g_top = x->prev;
        mx_k *k = x->k_list;
        while (k != NULL) {
            mx_k *next = k->next;
            free(k);
            k = next;
        }
        free(x->stack);
        int done = (x == s);
        free(x);
        if (done)
            return;
    }
}

/* Body coroutine entry point (makecontext passes the scope pointer split
 * into two unsigned ints for portability). */
static void mx__trampoline(unsigned int hi, unsigned int lo) {
    mx_scope *s = (mx_scope *)(void *)(((uintptr_t)hi << 32) | (uintptr_t)lo);
#ifdef MX_ASAN
    {
        const void *ob = NULL;
        size_t osz = 0;
        __sanitizer_finish_switch_fiber(NULL, &ob, &osz);
        s->owner_stack.bottom = ob;
        s->owner_stack.size = osz;
        if (s->owner_was_main && g_main.bottom == NULL)
            g_main = s->owner_stack;
    }
#endif
    g_cur.bottom = s->stack;
    g_cur.size = MX_EFFECT_STACK_SIZE;
    int64_t v = s->body(s->body_env);
    s->ev_kind = MX_EV_DONE;
    s->done_value = v;
    mx__switch_dead(&s->handler_ctx, s->owner_stack);
}

/* Innermost non-busy scope handling (effect, op); writes the matched op
 * index through *op_index.  NULL when unhandled.  Mirrors
 * mir_interp._find_mir_frame: busy scopes are skipped (a handler's own
 * performs route outward), the effect name only constrains the match when
 * BOTH sides name one. */
static mx_scope *mx__find_scope(const char *effect, const char *op,
                                int64_t *op_index) {
    for (mx_scope *s = g_top; s != NULL; s = s->prev) {
        if (s->busy)
            continue;
        int64_t found = -1;
        for (int64_t i = 0; i < s->nops; i++) {
            if (strcmp(s->op_names[i], op) == 0) {
                found = i;
                break;
            }
        }
        if (found < 0)
            continue;
        if (effect != NULL && effect[0] != '\0' &&
            s->effect != NULL && s->effect[0] != '\0' &&
            strcmp(s->effect, effect) != 0)
            continue;
        *op_index = found;
        return s;
    }
    return NULL;
}

int64_t mx_handle(mx_body_fn body, void *body_env,
                  mx_handler_fn handler, void *handler_env,
                  const char *effect,
                  const char *const *op_names,
                  const int64_t *op_nparams, int64_t nops) {
    for (int64_t i = 0; i < nops; i++) {
        if (op_nparams[i] < 0 || op_nparams[i] > MX_EFFECT_MAX_ARGS)
            mx__fatal("handler case declares more parameters than "
                      "MX_EFFECT_MAX_ARGS");
    }
    mx_scope *volatile s = calloc(1, sizeof(mx_scope));
    if (s == NULL)
        mx__fatal("out of memory (scope)");
    s->handler = handler;
    s->handler_env = handler_env;
    s->effect = effect;
    s->op_names = op_names;
    s->op_nparams = op_nparams;
    s->nops = nops;
    s->body = body;
    s->body_env = body_env;
    s->stack = malloc(MX_EFFECT_STACK_SIZE);
    if (s->stack == NULL)
        mx__fatal("out of memory (coroutine stack)");
    s->owner_was_main = (g_cur.bottom == NULL);
    s->prev = g_top;
    g_top = s;

    if (getcontext(&s->body_ctx) != 0)
        mx__fatal("getcontext failed");
    s->body_ctx.uc_stack.ss_sp = s->stack;
    s->body_ctx.uc_stack.ss_size = MX_EFFECT_STACK_SIZE;
    s->body_ctx.uc_link = NULL;
    uintptr_t sp = (uintptr_t)(void *)s;
    makecontext(&s->body_ctx, (void (*)(void))mx__trampoline, 2,
                (unsigned int)(sp >> 32), (unsigned int)(sp & 0xffffffffu));

    if (setjmp(s->abort_jmp) != 0) {
        /* A handler case reached through a nested mx_resume pump returned
         * without resuming: its value is the handle expression's value
         * (the interpreter's _EffectAbort caught at its own handle_scope). */
        int64_t v = s->abort_value;
        mx__teardown_to(s);
        return v;
    }

    /* Start the body; control returns here at its first event. */
    mx_stackinfo body_stack = {s->stack, MX_EFFECT_STACK_SIZE};
    mx__switch(&s->handler_ctx, &s->body_ctx, body_stack);
    if (g_cur.bottom == NULL)
        g_cur = s->owner_stack; /* main-thread bounds learned at fiber entry */

    if (s->ev_kind == MX_EV_DONE) {
        int64_t v = s->done_value;
        mx__teardown_to(s);
        return v;
    }
    /* MX_EV_PERFORM: dispatch the first perform.  Deep semantics make one
     * dispatch enough -- every later perform against this scope is pumped
     * recursively inside mx_resume (the interpreter calls _pump_scope
     * exactly once per handle_scope, too). */
    mx_k *k = s->ev_k;
    s->busy = 1;
    int64_t hres = s->handler(s->handler_env, s->ev_op_index, s->ev_args, k);
    s->busy = 0;
    /* k->used: the case resumed, so hres is already the WHOLE delimited
     * body's completion value (deep).  !k->used: the case declined to
     * resume -- abort semantics make its value the handle's value.  Either
     * way hres is the result; teardown frees the (completed or parked)
     * body coroutine and anything nested inside it. */
    int64_t v = hres;
    mx__teardown_to(s);
    return v;
}

int64_t mx_perform(const char *effect, const char *op,
                   const int64_t *args, int64_t nargs) {
    int64_t op_index = 0;
    mx_scope *s = mx__find_scope(effect, op, &op_index);
    if (s == NULL) {
        char buf[256];
        snprintf(buf, sizeof buf, "Unhandled effect operation: '%s'",
                 op ? op : "(null)");
        mx__fatal(buf);
    }
    int64_t nparams = s->op_nparams[op_index];
    if (nargs > nparams) {
        /* The interpreter's arity error: fewer args than params is
         * legitimate (synthesized case params pad with UNIT), more is a
         * program bug that must error, never silently truncate. */
        char buf[256];
        snprintf(buf, sizeof buf,
                 "Effect op '%s' performed with %lld argument(s) but its "
                 "handler case declares only %lld parameter(s)",
                 op, (long long)nargs, (long long)nparams);
        mx__fatal(buf);
    }
    mx_k *k = calloc(1, sizeof(mx_k));
    if (k == NULL)
        mx__fatal("out of memory (continuation)");
    k->scope = s;
    k->stack = (g_cur.bottom != NULL) ? g_cur : g_main;
    k->next = s->k_list;
    s->k_list = k;
    for (int64_t i = 0; i < nparams; i++)
        s->ev_args[i] = (i < nargs) ? args[i] : 0; /* pad = UNIT = 0 */
    s->ev_kind = MX_EV_PERFORM;
    s->ev_op_index = op_index;
    s->ev_k = k;
    /* Park this whole call stack at the perform site: transfer to the
     * scope's handler side and stay parked until mx_resume targets us. */
    mx__switch(&k->resume_ctx, &s->handler_ctx, s->owner_stack);
    return k->resume_value;
}

int64_t mx_resume(mx_k *k, int64_t value) {
    if (k == NULL)
        mx__fatal("resume: NULL continuation");
    if (k->used)
        mx__fatal("Continuation already consumed (single-shot violation)");
    k->used = 1;
    mx_scope *s = k->scope;
    /* Re-arm the scope's delimitation while its body runs (the interpreter
     * clears busy across the resume and restores it for the post-resume
     * handler code, which evaluates outside its own delimitation). */
    int was_busy = s->busy;
    s->busy = 0;
    k->resume_value = value;
    /* Unpark the body at its perform site; the handler side now waits
     * here (handler_ctx re-saved) for the scope's next event. */
    mx__switch(&s->handler_ctx, &k->resume_ctx, k->stack);
    if (g_cur.bottom == NULL)
        g_cur = s->owner_stack;

    if (s->ev_kind == MX_EV_DONE) {
        s->busy = was_busy;
        return s->done_value; /* deep: the WHOLE body's completion value */
    }
    /* The body performed against this scope again: pump recursively (the
     * interpreter's _pump_scope recursion, as plain C recursion on the
     * owner stack). */
    mx_k *k2 = s->ev_k;
    s->busy = 1;
    int64_t hres = s->handler(s->handler_env, s->ev_op_index, s->ev_args, k2);
    s->busy = was_busy;
    if (k2->used)
        return hres; /* that case resumed: hres is the body's final value */
    /* The nested case declined to resume: unwind every handler-side frame
     * between here and the scope's mx_handle (the interpreter's
     * _EffectAbort propagating through resume() calls). */
    s->abort_value = hres;
    longjmp(s->abort_jmp, 1);
}
