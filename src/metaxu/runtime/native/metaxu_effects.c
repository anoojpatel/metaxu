/* metaxu_effects.c -- native algebraic effects: delimited single-shot
 * deep-handler continuations on stackful ucontext coroutines.
 *
 * See metaxu_effects.h for the full contract.  The reference semantics is
 * the MIR interpreter (src/metaxu/compiler/mir_interp.py); the mapping is:
 *
 *   interpreter                          this file
 *   -----------------------------------  --------------------------------
 *   handle body thread (parked)          body coroutine (ucontext + stack)
 *   _mir_handler_frames (per logical     g_top scope stack (innermost
 *     thread; see _ThreadCtx)              first; _Thread_local)
 *   _find_mir_frame (busy skip, effect   mx__find_scope
 *     match, innermost op match)
 *   _pump_scope PERFORM -> case call     dispatcher call in mx__pump_events
 *     with busy=True                       with s->busy = 1
 *   resume() -> unpark body, wait for    mx_resume: swap to the perform
 *     next event (busy off meanwhile)      site, dispatch next event
 *   _pump_scope tail-resume loop         mx_resume_tail records (k, v);
 *     (_TailResume unwound to the pump)    mx__pump_events does the switch
 *   _EffectAbort unwind to handle_scope  longjmp(s->abort_jmp)
 *   _ScopeAbort cascade teardown of      mx__teardown_to frees the parked
 *     nested parked bodies                 coroutine stacks outright (C has
 *                                          no destructors to run on them)
 *   no frame -> __effect_default$E$op    mx_perform_or_default's dflt call
 *     called in the performing frame       on the performing stack
 *   try_scope delimited error boundary   mx_try landing pad (setjmp)
 *   InterpError raised in the extent     mx_raise -> longjmp to the pad
 *   ("error", exc) body thread message   MX_EV_ERROR + re-raise on the
 *     re-raised by _pump_scope             scope's owner stack
 *
 * THE PUMP MODEL.  All handler-case dispatch for a scope runs in ONE
 * owner-side event loop, mx__pump_events, entered from mx_handle (the
 * body's first perform) or from mx_resume (a general resume waiting for
 * the resumed body's next event).  Each iteration dispatches one event's
 * case and then looks at how the case ended:
 *
 *   - TAIL RESUME (the compiler proved the resume's value IS the case's
 *     return value, nothing after it -- compiler/effect_tail.py): the
 *     case called mx_resume_tail, which only RECORDS (k, v) on the scope
 *     and returns; the case returns to the pump, and the pump performs
 *     the switch into the body from its own CONSTANT frame, then loops
 *     for the scope's next event.  Handler-side stack usage is O(1) in
 *     the number of events -- the fix for the stream-pipeline stack
 *     cliff (a recursive pump grew (case + resume frame) per element).
 *     The consumed continuation record is freed right after the switch
 *     comes back (single-shot, provably unreachable: a tail resume is
 *     the case's last action and the compiler demotes any k that
 *     escapes into an env or closure).
 *   - GENERAL RESUME (mx_resume, anywhere but tail position): keeps its
 *     recursive semantics -- the resumed body's next perform is pumped
 *     from inside mx_resume, so the C stack tracks handler-code nesting
 *     (e.g. fold's f(x, resume(())) pending applications, which ARE the
 *     foldr semantics).  hres is then already the whole body's
 *     completion value (deep), and the pump returns it.
 *   - NO RESUME: abort -- the case's value is the handle expression's
 *     value; the pump reports it (mx_handle returns it; mx_resume
 *     longjmps the intervening handler frames away, as before).
 *
 * EQUIVALENCE WITH THE RECURSIVE PUMP (the interpreter's _pump_scope is
 * the model; induction on the number of events the body still produces):
 * under deep handling, a TAIL case's value is resume's value, which is
 * the WHOLE body's eventual completion value.  In the recursive scheme
 * that value is produced at the innermost recursion (the DONE event),
 * and every tail case on the way out returns it UNCHANGED (that is what
 * tail position means: no op after the resume, the ret returns its
 * value).  So collapsing the chain -- the pump switches into the body
 * itself and, at MX_EV_DONE, takes done_value directly as the scope's
 * value -- yields the identical result.  A non-tail case in the chain
 * re-enters via mx_resume exactly as before, so mixed chains compose:
 * the pump's constant frame stands in for the deleted tail frames only.
 * Abort: a case declining to resume makes its value the handle value in
 * both schemes (recursive: propagated/longjmp'd through the tail frames,
 * which would have returned it unchanged; pump: returned directly).
 * Errors: MX_EV_ERROR raises on the owner stack in both schemes, and the
 * pad chain is identical -- a tail case cannot have a live pad at its
 * resume (a pad around the resume would make it non-tail; pads it
 * installed earlier were popped on its normal path), so the deleted
 * frames held no pads.  busy flags: the pump holds busy=1 exactly while
 * a case runs and 0 while the body runs, which is what the recursive
 * chain maintained at every level (mx_resume cleared it before switching
 * and re-set it around the nested dispatch).
 *
 * Perform precedence (mir_interp's `perform` op, mirrored in
 * mx__perform_impl): innermost non-busy handler frame, then the declared
 * `with SYMBOL` runtime mapping, then the declared `= expr` default, then
 * a loud error.  The two fallback rungs share one mechanism here: the
 * compiler passes the op's fallback thunk (runtime-mapping thunk if one
 * is declared, else the default thunk) as mx_perform_or_default's dflt.
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
#include <stdarg.h>
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

/* MX_EV_ERROR is the interpreter's ("error", exc) message from a parked
 * body thread to _pump_scope: a catchable failure reached the bottom of a
 * body coroutine with no landing pad on it, so it must be re-raised on the
 * scope's OWNER stack (where an enclosing try's pad lives). */
enum { MX_EV_DONE = 0, MX_EV_PERFORM = 1, MX_EV_ERROR = 2 };

typedef struct {
    const void *bottom;
    size_t size;
} mx_stackinfo;

typedef struct mx_scope mx_scope;
typedef struct mx_pad mx_pad;

/* One `try` landing pad: a stack object in the frame that called mx_try. */
struct mx_pad {
    jmp_buf jb;
    mx_pad *prev;          /* enclosing pad ON THE SAME FIBER */
    mx_scope *scope_top;   /* g_top at install: catching unwinds down to it */
    int scope_busy;        /* scope_top->busy at install (restored on catch) */
};

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
    int ev_kind;                    /* MX_EV_DONE / MX_EV_PERFORM / _ERROR */
    int64_t ev_op_index;
    int64_t ev_args[MX_EFFECT_MAX_ARGS];
    int64_t done_value;
    char *ev_error;                 /* MX_EV_ERROR: owned message text */
    mx_k *ev_k;
    /* Landing-pad chain of the OWNER context when this scope was created:
     * restored before the abort longjmp, whose target frame sits below any
     * pad installed inside a handler case (those frames die with it). */
    mx_pad *owner_pad;
    /* abort unwinding (handler case returned without resuming while inside
     * a nested mx_resume pump): longjmp back to this scope's mx_handle */
    jmp_buf abort_jmp;
    int64_t abort_value;
    /* Pending tail resume: mx_resume_tail RECORDS (k, value) here and
     * returns; the pump that dispatched the case consumes it and performs
     * the switch from its own constant frame.  Only ever set between a
     * case's mx_resume_tail call and its return to the pump (one side of
     * a scope runs at a time, so no other dispatch can intervene). */
    mx_k *tail_k;
    int64_t tail_value;
    /* bookkeeping */
    mx_k *k_list;
    mx_scope *prev;                 /* global scope stack link */
};

/* ALL scheduler state is _Thread_local (docs/threads_runtime.md): each OS
 * thread -- the main thread and every mx_thread_spawn child -- owns its
 * own scope stack, pad chain and fiber bookkeeping, zero-initialized at
 * thread start.  That is the effect-scope-isolation contract: a spawned
 * child starts with NO scopes in view (its unhandled performs route to
 * runtime mappings / defaults, never to a scope rooted in another
 * thread's stack), and the per-fiber pad-chain semantics are unchanged
 * WITHIN each thread.  Fibers (ucontext coroutines) never migrate
 * between threads, so thread-locals read inside a fiber always belong to
 * the thread that runs it. */

/* Scope stack, innermost scope first (per thread). */
static _Thread_local mx_scope *g_top = NULL;

/* The currently running fiber's stack ({NULL, 0} = the thread's root
 * context before its bounds are learned, or unknown); maintained across
 * every switch so mx_perform can stamp continuations with the stack they
 * park on. */
static _Thread_local mx_stackinfo g_cur = {NULL, 0};
/* Root-stack bounds of THIS thread, learned from the first coroutine
 * entered directly from the root context (ASan reports them at fiber
 * entry). */
static _Thread_local mx_stackinfo g_main = {NULL, 0};

/* Landing-pad chain OF THE RUNNING FIBER (innermost first).  A longjmp may
 * only target a frame on the stack it runs on, so this -- like g_cur -- is
 * saved and restored around every context switch and starts empty on a
 * fresh coroutine.  See "COMPOSITION WITH EFFECT SCOPES" in the header. */
static _Thread_local mx_pad *g_pad = NULL;
/* The scope whose BODY COROUTINE is currently running (NULL on an owner
 * stack / the thread's root context): the target of a failure that finds
 * no pad. */
static _Thread_local mx_scope *g_fiber = NULL;
/* The message of the raise currently in flight.  Not a pad field: the pad
 * is an automatic object and writing it between setjmp and longjmp would
 * make its value indeterminate.  Thread-local because a raise always
 * resolves on the thread it was raised on.  Only ever read immediately
 * after a longjmp lands, and each raise heap-copies its own text. */
static _Thread_local char *g_raise_msg = NULL;

static void mx__fatal(const char *msg) {
    /* stdout FIRST: abort() does not flush it, and a native program that
     * printed before dying must show the same prefix the interpreter shows
     * before it raises (the differential tests compare exactly that). */
    fflush(stdout);
    fprintf(stderr, "metaxu effects runtime: %s\n", msg);
    fflush(stderr);
    abort();
}

/* Switch to `to`, saving the current context in `save`; returns when some
 * later switch targets `save`.  `to_stack` is the target fiber's stack
 * (ASan annotation only). */
static void mx__switch(ucontext_t *save, ucontext_t *to, mx_stackinfo to_stack) {
    mx_stackinfo my = g_cur;
    /* The pad chain and the running-fiber marker belong to THIS context and
     * are restored when control comes back; the target sets its own (a
     * fresh fiber in mx__trampoline, a parked one by returning from its own
     * mx__switch below).  Without this, a longjmp on the handler side could
     * target a frame on a parked coroutine stack. */
    mx_pad *my_pad = g_pad;
    mx_scope *my_fiber = g_fiber;
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
    g_pad = my_pad;
    g_fiber = my_fiber;
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

/* Unlink one continuation record from its scope's list and free it.
 * Only for a record the caller proved consumed AND unreachable: the pump
 * calls this for tail-resumed continuations after the switch back (the
 * body has read resume_value by then, the ucontext was consumed by the
 * switch, and a tail resume is its case's last action so no later
 * single-shot check can touch the record -- the compiler demotes any k
 * that escapes into an env/closure).  GENERAL (non-tail) resumes never
 * free: the record must stay allocated so a later double resume hits the
 * `used` flag (the clean single-shot fatal), not freed memory; those
 * records are reclaimed at scope teardown as before. */
static void mx__free_k(mx_scope *s, mx_k *k) {
    mx_k **pp = &s->k_list;
    while (*pp != NULL && *pp != k)
        pp = &(*pp)->next;
    if (*pp == NULL)
        mx__fatal("continuation record missing from its scope's list");
    *pp = k->next;
    free(k);
}

/* Free every scope ABOVE `top` (exclusive), leaving `top` itself installed.
 * The try landing pad's unwind: every scope pushed inside the delimited
 * extent is gone -- coroutine stacks, continuation records and scope
 * records -- which is the interpreter's `finally: self._abort_scope(scope);
 * self._mir_handler_frames.remove(frame)` running for each handle_scope the
 * failure propagated out of. */
static void mx__teardown_above(mx_scope *top) {
    while (g_top != top) {
        if (g_top == NULL)
            mx__fatal("scope stack corrupted (unwind past bottom)");
        mx__teardown_to(g_top);
    }
}

/* Defined with the try/catch machinery at the bottom of this file; used by
 * mx_handle / mx_resume to re-raise a failure that escaped a body fiber. */
static _Noreturn void mx__raise_owned(char *owned);

static char *mx__dup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = malloc(n);
    if (p == NULL)
        mx__fatal("out of memory (failure message)");
    memcpy(p, s, n);
    return p;
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
    g_pad = NULL;    /* a fresh fiber has no landing pads of its own */
    g_fiber = s;
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

typedef struct {
    int64_t value;
    int aborted;   /* the last case declined to resume */
} mx_pump_result;

/* The owner-side event pump (see THE PUMP MODEL in the file header).
 * Precondition: s->ev_kind == MX_EV_PERFORM and the caller runs on s's
 * owner stack.  Dispatches the pending event's case, then:
 *
 *   - the case TAIL-RESUMED (mx_resume_tail recorded s->tail_k): switch
 *     into the body from THIS frame, free the consumed record, and loop
 *     on the scope's next event -- MX_EV_DONE returns done_value as the
 *     scope's final value (exactly what the deleted chain of tail frames
 *     would have propagated, see the equivalence note in the header),
 *     MX_EV_ERROR re-raises on this owner stack, MX_EV_PERFORM iterates;
 *   - the case GENERAL-resumed (k->used, no tail record): its return
 *     value is already the whole body's completion value (deep) --
 *     return it, aborted = 0;
 *   - the case did NOT resume: return its value with aborted = 1 (the
 *     caller decides between returning it as the handle value and
 *     longjmp-unwinding nested handler frames). */
static mx_pump_result mx__pump_events(mx_scope *s) {
    for (;;) {
        mx_k *k = s->ev_k;
        s->tail_k = NULL;
        s->busy = 1;
        int64_t hres = s->handler(s->handler_env, s->ev_op_index,
                                  s->ev_args, k);
        s->busy = 0;
        if (s->tail_k == NULL) {
            mx_pump_result r = { hres, !k->used };
            return r;
        }
        /* Tail resume recorded.  It must be THIS dispatch's continuation:
         * the compiler only emits mx_resume_tail on a case's own trailing
         * __k parameter. */
        mx_k *tk = s->tail_k;
        s->tail_k = NULL;
        if (tk != k)
            mx__fatal("tail resume of a foreign continuation "
                      "(pump invariant violated)");
        tk->resume_value = s->tail_value;
        /* Unpark the body at its perform site; the handler side now waits
         * HERE -- every later body->handler transfer re-enters this loop
         * at constant depth. */
        mx__switch(&s->handler_ctx, &tk->resume_ctx, tk->stack);
        if (g_cur.bottom == NULL)
            g_cur = s->owner_stack;
        /* The body has moved past the perform site (it read resume_value
         * strictly before switching back).  The record is consumed and
         * unreachable: reclaim it now -- this is what keeps a stream's
         * heap flat instead of one leaked mx_k (with its ~1 KB ucontext)
         * per element. */
        mx__free_k(s, tk);
        if (s->ev_kind == MX_EV_DONE) {
            mx_pump_result r = { s->done_value, 0 };
            return r;
        }
        if (s->ev_kind == MX_EV_ERROR) {
            /* Same re-raise point as mx_handle/mx_resume: the failure
             * escaped the body fiber and must surface on the owner
             * stack.  The deleted tail frames held no pads (see the
             * equivalence note), so the pad chain here is exactly what
             * the recursive unwind would have seen. */
            mx__raise_owned(s->ev_error);
        }
        /* MX_EV_PERFORM: next element's event -- loop, constant frame. */
    }
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
    s->owner_pad = g_pad;
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
    if (s->ev_kind == MX_EV_ERROR) {
        /* The body failed with no landing pad on its coroutine: re-raise on
         * THIS (owner) stack, where an enclosing try's pad lives.  No
         * teardown here -- that pad's unwind frees this scope, and if the
         * pad is inside one of our own handler cases the scope must SURVIVE
         * (the interpreter keeps the frame until handle_scope's finally). */
        mx__raise_owned(s->ev_error);
    }
    /* MX_EV_PERFORM: enter the event pump.  Tail-resuming cases loop
     * inside it at constant depth; a general resume recurses inside
     * mx_resume (the interpreter calls _pump_scope exactly once per
     * handle_scope, too).  Whatever the pump reports is the handle
     * value: the DONE value reached through tail resumes, a
     * general-resuming case's return value (already the WHOLE delimited
     * body's completion value, deep), or a non-resuming case's return
     * value (abort semantics).  Teardown frees the (completed or parked)
     * body coroutine and anything nested inside it. */
    mx_pump_result r = mx__pump_events(s);
    int64_t v = r.value;
    mx__teardown_to(s);
    return v;
}

/* Shared body of mx_perform / mx_perform_or_default.  `dflt` is the op's
 * declared `= expr` default (NULL when it has none): the ONLY thing it
 * changes is what happens when no scope matches -- the abort path of
 * mx_perform is untouched for ops without a default.
 *
 * Precedence mirrored from mir_interp (perform op, in order):
 *   1. innermost non-busy scope handling (effect, op)   -> park + dispatch
 *   2. declared `with SYMBOL` runtime mapping           -> plain call HERE
 *   3. declared `= expr` default                        -> plain call HERE
 *   4. loud "Unhandled effect operation" error
 * Rungs 2 and 3 share this one entry point: an op can declare a mapping,
 * a default, or neither, and the compiler passes the HIGHER-precedence
 * thunk it has (the __effect_runtime$E$op thunk when a mapping exists --
 * its body calls the metaxu_threads.c primitives -- else the
 * __effect_default$E$op thunk) as `dflt`, so the ordering is decided at
 * compile time and this runtime only ever sees one fallback.
 *
 * The default runs RIGHT HERE, on the performing stack: a default is an
 * expression, not a suspension.  Nothing is parked, no scope is pushed,
 * no continuation exists, and the scope stack is bit-for-bit what it was
 * at the perform -- so a perform inside the default routes (and parks)
 * exactly as one written at the perform site would, which is what the
 * interpreter does when it evaluates __effect_default$E$op in the
 * performing frame's context. */
static int64_t mx__perform_impl(const char *effect, const char *op,
                                const int64_t *args, int64_t nargs,
                                mx_default_fn dflt, void *dflt_env) {
    int64_t op_index = 0;
    mx_scope *s = mx__find_scope(effect, op, &op_index);
    if (s == NULL) {
        if (dflt != NULL) {
            if (nargs < 0 || nargs > MX_EFFECT_MAX_ARGS)
                mx__fatal("effect op performed with more arguments than "
                          "MX_EFFECT_MAX_ARGS");
            /* The thunk reads exactly the words its default declares; the
             * compiler proved the arities agree (it demotes otherwise). */
            return dflt(dflt_env, args);
        }
        /* CATCHABLE (mir_interp's perform op raises InterpError here, so a
         * `try` recovers from it -- this is exactly what example 04's
         * speculative `perform Parser.parse` relies on).  The wording names
         * the EFFECT, not the op: that is the message the MIR perform path
         * raises ("No handler for effect 'Parser'"), and the caught value
         * must be byte-identical to the interpreter's. */
        mx_raisef("No handler for effect '%s'", effect ? effect : "");
    }
    int64_t nparams = s->op_nparams[op_index];
    if (nargs > nparams) {
        /* The interpreter's arity error: fewer args than params is
         * legitimate (synthesized case params pad with UNIT), more is a
         * program bug that must error, never silently truncate.  CATCHABLE
         * (InterpError in _pump_scope), same wording. */
        mx_raisef("Effect op '%s' performed with %lld argument(s) but its "
                  "handler case declares only %lld parameter(s)",
                  op, (long long)nargs, (long long)nparams);
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

int64_t mx_perform(const char *effect, const char *op,
                   const int64_t *args, int64_t nargs) {
    return mx__perform_impl(effect, op, args, nargs, NULL, NULL);
}

int64_t mx_perform_or_default(const char *effect, const char *op,
                              const int64_t *args, int64_t nargs,
                              mx_default_fn dflt, void *dflt_env) {
    return mx__perform_impl(effect, op, args, nargs, dflt, dflt_env);
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
     * here (handler_ctx re-saved) for the scope's next event.  NOTE: k is
     * NOT freed on this (general) path -- it must stay allocated so a
     * later resume of the same continuation hits the `used` flag above
     * (the clean single-shot fatal) instead of freed memory.  It is
     * reclaimed at scope teardown, so the cost is transient, one record
     * per still-live general resume of the scope. */
    mx__switch(&s->handler_ctx, &k->resume_ctx, k->stack);
    if (g_cur.bottom == NULL)
        g_cur = s->owner_stack;

    if (s->ev_kind == MX_EV_DONE) {
        s->busy = was_busy;
        return s->done_value; /* deep: the WHOLE body's completion value */
    }
    if (s->ev_kind == MX_EV_ERROR) {
        /* The RESUMED body failed with no pad of its own: the interpreter's
         * _pump_scope re-raises here, inside resume(), so a try installed by
         * this very handler case can catch it.  busy stays as the pad
         * recorded it (mx_try restores scope_top->busy on catch), matching
         * the interpreter's resume() `finally: frame["busy"] = was_busy`. */
        mx__raise_owned(s->ev_error);
    }
    /* The body performed against this scope again: pump from here (the
     * interpreter's _pump_scope recursion -- one C recursion per GENERAL
     * resume on the owner stack; the pump's own tail loop handles any
     * chain of tail-resuming dispatches at constant depth). */
    mx_pump_result r = mx__pump_events(s);
    s->busy = was_busy;
    if (!r.aborted)
        return r.value; /* the body's final value (deep) */
    /* A nested case declined to resume: unwind every handler-side frame
     * between here and the scope's mx_handle (the interpreter's
     * _EffectAbort propagating through resume() calls). */
    s->abort_value = r.value;
    g_pad = s->owner_pad;  /* pads inside the abandoned frames die with them */
    longjmp(s->abort_jmp, 1);
}

int64_t mx_resume_tail(mx_k *k, int64_t value) {
    /* The TAIL form of resume, emitted by the compiler ONLY where the
     * resume's value is returned as the case's value with nothing after
     * it (compiler/effect_tail.py -- strict; anything doubtful keeps
     * mx_resume).  Records the resume on the scope and returns
     * immediately: the case then returns to the pump that dispatched it,
     * and THE PUMP performs the switch into the body from its constant
     * frame.  The return value is a placeholder for the case's dataflow
     * (its ret still needs a word); the pump ignores the case's return
     * value whenever a tail resume is recorded. */
    if (k == NULL)
        mx__fatal("resume: NULL continuation");
    if (k->used)
        mx__fatal("Continuation already consumed (single-shot violation)");
    k->used = 1;
    mx_scope *s = k->scope;
    if (s->tail_k != NULL)
        mx__fatal("tail resume recorded twice in one dispatch "
                  "(pump invariant violated)");
    s->tail_k = k;
    s->tail_value = value;
    return 0;
}

/* ------------------------------------------------------------------------
 * Delimited failure recovery: mx_try / mx_raise (docs/try_catch.md).
 * See metaxu_effects.h for the mechanism, the composition rules and the
 * catchable/fatal split.
 * ---------------------------------------------------------------------- */

int64_t mx_try(mx_try_body_fn body, void *body_env,
               mx_try_catch_fn katch, void *catch_env) {
    /* volatile: these are read after longjmp, so they must not live only in
     * a caller-saved register the unwind does not restore. */
    mx_try_catch_fn volatile vcatch = katch;
    void *volatile vcatch_env = catch_env;
    mx_pad pad;
    pad.prev = g_pad;
    pad.scope_top = g_top;
    pad.scope_busy = (g_top != NULL) ? g_top->busy : 0;
    g_pad = &pad;
    if (setjmp(pad.jb) != 0) {
        /* A catchable failure occurred in the delimited extent.  Pop this
         * pad FIRST (a failure raised by the catch block propagates
         * outward, never to its own try -- docs/try_catch.md rule 4), then
         * tear down every effect scope the failure escaped and restore the
         * innermost surviving scope's busy flag. */
        g_pad = pad.prev;
        mx__teardown_above(pad.scope_top);
        if (pad.scope_top != NULL)
            pad.scope_top->busy = pad.scope_busy;
        const char *msg = g_raise_msg ? g_raise_msg : "";
        g_raise_msg = NULL;
        return vcatch(vcatch_env, msg);
    }
    int64_t v = body(body_env);
    g_pad = pad.prev;
    return v;
}

/* The raise machinery, over a message this function takes ownership of.
 * Re-raising an escaped fiber failure reuses its existing copy rather than
 * duplicating it again (mx_handle / mx_resume call this directly). */
static _Noreturn void mx__raise_owned(char *owned) {
    if (g_pad != NULL) {
        g_raise_msg = owned;
        longjmp(g_pad->jb, 1);
    }
    if (g_fiber != NULL) {
        /* No pad on this coroutine: hand the failure to the scope's owner
         * side (the interpreter's ("error", exc) message) and abandon this
         * fiber -- its stack is freed when the scope is torn down. */
        mx_scope *s = g_fiber;
        s->ev_kind = MX_EV_ERROR;
        s->ev_error = owned;
        mx__switch_dead(&s->handler_ctx, s->owner_stack);
    }
    /* Uncaught: identical to the pre-try/catch behavior of mx_rt_fail --
     * flush what the program already printed, name the failure, abort. */
    fflush(stdout);
    fprintf(stderr, "metaxu runtime error: %s\n", owned);
    fflush(stderr);
    abort();
}

_Noreturn void mx_raise(const char *msg) {
    /* Heap-copied once, here: the caught value is an ordinary metaxu string
     * that may outlive the try (leak by design, like every produced string
     * the backend cannot prove dead), and the caller's buffer is usually a
     * stack scratch buffer the longjmp is about to destroy. */
    mx__raise_owned(mx__dup(msg != NULL ? msg : ""));
}

_Noreturn void mx_raisef(const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    mx_raise(buf);
}
