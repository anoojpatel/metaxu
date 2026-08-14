/* metaxu_effects.h -- native algebraic effects runtime for Metaxu.
 *
 * Implements delimited, SINGLE-SHOT, DEEP-handler continuations for the
 * MIR ops handle_scope / perform / resume, matching the reference
 * semantics of the MIR interpreter (src/metaxu/compiler/mir_interp.py)
 * exactly:
 *
 *   - every handle_scope body runs on its own STACKFUL COROUTINE
 *     (ucontext_t + a malloc'd stack), mirroring the interpreter's
 *     parked-thread model: a perform ANYWHERE in the delimited dynamic
 *     extent -- including deep inside called functions -- parks the whole
 *     body-side call stack at the perform site and transfers control to
 *     the handler side;
 *   - handler dispatch is DYNAMIC, over a global (process-wide) stack of
 *     active scopes: the innermost non-busy scope whose op-name set
 *     contains the performed op (and whose effect name matches, when both
 *     sides name one) wins -- exactly mir_interp._find_mir_frame.  A
 *     scope is `busy` while one of its handler cases is running, so a
 *     handler's own performs route OUTWARD (never to its own scope);
 *     mx_resume un-busies the scope while the body runs (deep-handler
 *     re-arming) and restores the flag afterwards;
 *   - DEEP semantics: mx_resume(k, v) returns the completion value of the
 *     WHOLE delimited body.  Any further perform the resumed body makes
 *     against this scope is dispatched recursively inside mx_resume (a C
 *     recursion on the handler side, mirroring the interpreter's
 *     _pump_scope recursion);
 *   - ABORT: a handler case that returns without resuming makes its
 *     return value the handle expression's value and tears down the
 *     parked body.  When the non-resuming case ran inside a nested
 *     mx_resume pump, longjmp unwinds the handler-side C frames back to
 *     the owning mx_handle (the interpreter's _EffectAbort).  Teardown
 *     frees the body coroutine stacks of the aborted scope AND of every
 *     scope nested inside its parked body chain (the interpreter's
 *     cascading _ScopeAbort) -- no code ever runs on those stacks again,
 *     and C has no destructors to run, so freeing them outright is
 *     observationally equivalent to the interpreter's cascade;
 *   - SINGLE-SHOT: resuming a consumed continuation prints the
 *     interpreter's message ("Continuation already consumed (single-shot
 *     violation)") and aborts the process.  Continuation records are
 *     freed at scope teardown; `used` is checked before anything else is
 *     touched.
 *
 * ABI conventions (all effect-boundary values are opaque 8-byte words:
 * i64 as-is, double bit-cast, pointers ptrtoint -- the compiler knows the
 * static kinds on both sides and reinterprets):
 *
 * | symbol      | signature                                              |
 * |-------------|--------------------------------------------------------|
 * | mx_handle   | int64_t (mx_body_fn body, void *body_env,              |
 * |             |          mx_handler_fn handler, void *handler_env,     |
 * |             |          const char *effect,                           |
 * |             |          const char *const *op_names,                  |
 * |             |          const int64_t *op_nparams, int64_t nops)      |
 * | mx_perform  | int64_t (const char *effect, const char *op,           |
 * |             |          const int64_t *args, int64_t nargs)           |
 * | mx_perform_ | int64_t (const char *effect, const char *op,           |
 * | or_default  |          const int64_t *args, int64_t nargs,           |
 * |             |          mx_default_fn dflt, void *dflt_env)           |
 * | mx_resume   | int64_t (mx_k *k, int64_t value)                       |
 *
 *   mx_body_fn    = int64_t (*)(void *env)
 *   mx_handler_fn = int64_t (*)(void *env, int64_t op_index,
 *                               const int64_t *args, mx_k *k)
 *   mx_default_fn = int64_t (*)(void *env, const int64_t *args)
 *
 * mx_handle runs `body` on a fresh coroutine under a newly-installed
 * scope and returns the handle expression's value: the body's completion
 * value when it finishes without performing against this scope, else the
 * (first) handler case invocation's return value -- which under deep
 * semantics already reflects the whole body when the case resumed, and is
 * the abort value when it did not.  `handler` is the compiled per-site
 * dispatcher: it switches on op_index (the position in op_names) and
 * calls the matching handler-case function with `args` (exactly
 * op_nparams[op_index] words: the runtime pads missing trailing args with
 * 0 -- the interpreter's UNIT padding for synthesized case params -- and
 * aborts when a perform supplies MORE args than the case declares,
 * mirroring the interpreter's arity error).
 *
 * mx_perform routes to the innermost matching scope as described and
 * returns the value passed to mx_resume; when no scope matches it prints
 * "Unhandled effect operation: '<op>'" and aborts (the interpreter's
 * InterpError).
 *
 * mx_perform_or_default is mx_perform for an op that ALSO declares a
 * `= expr` default (the surface `op(x) -> T = expr` form).  Routing is
 * unchanged whenever a scope matches -- same innermost-non-busy lookup,
 * same padding/arity rules, same parking -- but where mx_perform aborts,
 * this entry point calls `dflt(dflt_env, args)` and returns its value.
 * That mirrors the interpreter's perform precedence exactly:
 *
 *     in-scope handler frame  >  declared `= expr` default  >  error
 *
 * (the interpreter's third rung, a `with SYMBOL` runtime mapping, sits
 * between the two and has no native implementation at all -- the compiler
 * demotes any op that declares one, so no lowering can reach this entry
 * point with a mapping in play).
 *
 * THE DEFAULT RUNS ON THE PERFORMING STACK.  A default is an ordinary
 * expression, not a suspension: no coroutine is created, no scope is
 * installed, nothing is parked, and no continuation record exists (there
 * is nothing to resume -- the perform simply becomes a call).  The scope
 * stack is untouched across the call, so a perform INSIDE the default
 * routes exactly as a perform written at the original site would: it sees
 * the same scopes with the same busy flags, and if it needs to park, it
 * parks this same fiber.  That is precisely what the interpreter does
 * (it evaluates the default function in the performing frame's context
 * with the handler-frame list unchanged), so the two agree by
 * construction, including for recursive and re-entrant defaults.
 *
 * `dflt` receives the caller's `args` pointer UNPADDED: the compiler emits
 * a per-op thunk that reads exactly the argument words the default
 * function declares, and demotes any perform whose arity disagrees with
 * its default (the scope-routed path keeps the runtime's own padding /
 * arity checks).  Passing dflt == NULL is exactly mx_perform.
 *
 * Limits: at most MX_EFFECT_MAX_ARGS (8) op arguments / case parameters
 * (the compiler demotes anything larger); coroutine stacks are
 * MX_EFFECT_STACK_SIZE bytes (1 MiB).
 *
 * Memory: coroutine stacks, scope records and continuation records are
 * freed at scope completion/abort -- effect machinery itself is
 * leak-clean under ASan/LSan.  Values flowing through performs follow the
 * backend's usual contracts (boxes/strings may leak by design).
 *
 * ASan: switches are annotated with __sanitizer_start_switch_fiber /
 * __sanitizer_finish_switch_fiber when compiled with -fsanitize=address,
 * so sanitized differential tests see no false positives.
 *
 * Thread-unsafe by design (single-threaded native programs; the scope
 * stack is a process-wide global), like the rest of the native runtime.
 */
#ifndef METAXU_EFFECTS_H
#define METAXU_EFFECTS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MX_EFFECT_MAX_ARGS 8

typedef struct mx_k mx_k;

typedef int64_t (*mx_body_fn)(void *env);
typedef int64_t (*mx_handler_fn)(void *env, int64_t op_index,
                                 const int64_t *args, mx_k *k);
/* An op's declared `= expr` default, compiled to a per-op thunk that
 * decodes the argument words and word-encodes the result.  `env` is the
 * thunk's environment (NULL for the top-level __effect_default$E$op
 * functions the compiler emits today; the parameter exists so a capturing
 * default can be added without another ABI change). */
typedef int64_t (*mx_default_fn)(void *env, const int64_t *args);

int64_t mx_handle(mx_body_fn body, void *body_env,
                  mx_handler_fn handler, void *handler_env,
                  const char *effect,
                  const char *const *op_names,
                  const int64_t *op_nparams, int64_t nops);

int64_t mx_perform(const char *effect, const char *op,
                   const int64_t *args, int64_t nargs);

int64_t mx_perform_or_default(const char *effect, const char *op,
                              const int64_t *args, int64_t nargs,
                              mx_default_fn dflt, void *dflt_env);

int64_t mx_resume(mx_k *k, int64_t value);

#ifdef __cplusplus
}
#endif

#endif /* METAXU_EFFECTS_H */
