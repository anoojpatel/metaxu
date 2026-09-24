# try/catch semantics (v1 proposal)

Status: proposed + implemented on the v1 branch; awaiting language-owner
review. Example 04 (`try_parse`) is the motivating use.

## Semantics

`try { body } catch e { handler }` is **delimited dynamic error recovery**,
built on the same delimitation mechanism as effect handlers:

1. `body` evaluates as a delimited scope. If it completes with value `v`,
   the try expression's value is `v` and the catch block never runs.
2. If a **runtime failure** occurs anywhere in the delimited extent —
   in `body` itself or in anything it calls — the failure is materialized
   as an error *value*, bound to `e`, and the `handler` block's value
   becomes the try expression's value. The rest of `body` never runs
   (abort semantics, matching a non-resuming effect handler).
3. Runtime failures are: an unhandled effect `perform` (no handler
   installed dynamically), builtin contract violations (pop on empty Vec,
   index out of bounds, wrong-arity effect op, missing trait impl),
   match failures, the IO effects' failures (`read: mx.toml: No such
   file or directory`, docs/io_runtime.md), and an explicit
   `raise(message)`, the program's own failure with exactly that text.
   Borrow/type errors are compile-time and are NOT catchable.
4. Nested `try` scopes: the innermost one catches. A failure raised inside
   a `catch` handler propagates outward; it is not caught by its own try.
5. The error value in v1 is the failure's message string (example 04 does
   `Err(error)` and later prints `"Error: " + e`, so a string composes).
   A structured error type can replace it later without changing the
   delimitation semantics.

## What try/catch deliberately does NOT do

- It does **not** discharge `performs` obligations statically. A
  `perform` that is only covered by an enclosing `try` still gets the
  unhandled-effect advisory: catching the failure is dynamic *recovery*,
  not a handler installation. This keeps the static story honest — a
  handler gives the effect meaning; try/catch only contains its absence.
- It does not resume. There is no continuation exposed in the catch block;
  if resumable recovery is wanted, that is what `handle` is for.

## Rationale

Example 04 treats `perform Parser.parse<T>(input)` as *speculative*: with
no `Parser` handler installed, `try_parse` should take the `Err` path
rather than crash. Making the unhandled-perform failure catchable gives
that program well-defined behavior without inventing a second exception
system: one failure channel, one delimitation mechanism, and `handle`
remains the way to give an effect real semantics.

## Implementation

- HIR: `op="Try"` with the body expression, catch parameter name, and
  catch body (`hir.py`, converted from the parsed `TryCatch` node).
- MIR: body and catch compile to standalone subfunctions (exactly like
  handle bodies and handler cases); the op is
  `("try_scope", body_fn, catch_fn, catch_param)` with captured locals
  passed the same way as `handle_scope`.
- Interpreter: runs the body subfunction under a delimited error boundary;
  on `InterpError` (and only `InterpError` — scope-teardown control
  exceptions pass through untouched) it calls the catch subfunction with
  the message string. `mir_interp.py`.
- Native (LLVM): the op lowers to an env fill plus
  `mx_try(body_thunk, env, catch_thunk, env)` — the handle-site machinery
  reused verbatim, since a try IS a handle site with no cases.  See
  "Native lowering" below.

## Native lowering

`metaxu_effects.c` implements `mx_try` / `mx_raise` with **setjmp/longjmp
landing pads**, not LLVM's `invoke`/`landingpad`.  Two reasons, both
decisive: the failure sites are C runtime functions compiled without
unwind tables, and a failure raised inside a `handle` body runs on a
**ucontext coroutine stack** that no DWARF unwinder can walk back to the
owner stack.  The file already transfers control non-locally with
setjmp/longjmp (a non-resuming handler case unwinds to its scope's
`mx_handle` through `abort_jmp`), so try/catch reuses that machinery
instead of adding a second, incompatible one.

Three invariants make it compose with in-flight effect scopes:

1. **The pad chain is per-fiber.**  A longjmp may only target a frame on
   the stack it runs on, so the chain is saved/restored across every
   context switch (exactly like the current-stack marker), and a fresh body
   coroutine starts with an empty chain.  A `try` inside a handle body is
   therefore invisible to the handler side while the body is parked, and
   visible again after a resume — which is why `try { ... perform ... }`
   with the handler *outside* the try works, and why a failure on the
   handler side can never longjmp into a parked frame.
2. **A failure on a body fiber with no local pad escapes to its owner.**
   It becomes an `MX_EV_ERROR` event, which `mx_handle`/`mx_resume` — and
   the owner-side event pump that trampolines tail resumes
   (`mx__pump_events`, see metaxu_effects.c "THE PUMP MODEL") — re-raise
   on the owner stack.  That is the interpreter's `("error", exc)` message
   from the body thread to `_pump_scope`, and it is what makes
   `try { handle { ... fail ... } }` — and a `try` inside a handler case
   catching the *resumed* body's failure — behave identically.
3. **Catching unwinds the scope stack.**  Each pad records the scope-stack
   top (and that scope's `busy` flag) at install; reaching the pad tears
   down every scope pushed since — coroutine stacks, continuation records
   and scope records freed — and restores the flag.  That is the
   interpreter's `finally: _abort_scope(scope); frames.remove(frame)` plus
   `_pump_scope`'s `finally: frame["busy"] = False`.

### Try-body captures: read-before-rebind names are env fields

The body and catch compile to sub-functions receiving the try site's env
struct.  Which enclosing names land in that env comes from the codegen
free-name fixpoint, and it has one non-obvious rule (added 2026-08 after
the inline-Vec work unmasked a soundness hole): a name that the body
both USES and DEFINES is still free when it is **upward-exposed** —
possibly read before its first local def.  The shape that needs it is a
field or index update on an enclosing struct: `h.data[0] = 9` lowers to
read `h` → element store → REBIND `h` (the functional store-back), so
plain `uses - defs` called `h` local and the body read an uninitialized
slot.  The all-call emission survived that by accident (the garbage
word reached `mx_vec_set`, whose contended check happened to fire the
"right" raise); the inline emission dereferenced it and crashed.
`_upward_exposed` (block-level backward liveness in codegen_llvm.py)
now keeps such names free — as **by-value** env fields, which is
interpreter parity: there too the rebind stays local to the body's env
copy while the Vec mutation travels by identity.  Exposure is
intersected with the site's captures because liveness also sees the
impossible br-to-join path after a raising `match_fail`, which would
otherwise drag match-result temps into the env.  Pinned by
`test_try_body_field_update_sees_enclosing_struct` (differential, plus
the env-struct shape).

### Catchable vs fatal, natively

A failure is catchable **iff** the interpreter raises `InterpError` for it
*and* the runtime can produce the interpreter's exact message text (the
caught value is language-visible, so it must be byte-identical).

Catchable: `No handler for effect '<E>'`; the effect-op arity error; and
the `metaxu_rt.c` contract violations that mirror an `InterpError` — pop on
empty, index out of bounds, index assignment out of bounds, slice step,
vector size mismatch, zip length mismatch, comprehension length, `as_ptr`
byte range, shift-count range.

Fatal on both sides, because the interpreter does not raise `InterpError`
either: `assert` failure (`AssertionError`), integer division by zero
(`ZeroDivisionError`), double `resume` (`RuntimeError`).  Also fatal, with
no interpreter counterpart at all: allocation failure, NULL receivers,
capacity overflow, internal formatting invariants, `swapcontext` failure.

The last asymmetry is gone (2026-08-18): a match failure IS catchable in
the interpreter, and its message embeds a function name that
monomorphization used to rename (`classify` -> `classify$Int`).  The
pre-monomorphization name is now carried through MIR
(`HFun.origin_sym` -> `MirFunc.origin_name`, with subfunction names
substituted back so handler/lambda subfunctions match the unspecialized
lane too), so `match_fail` lowers to `mx_raise` with the interpreter's
exact wording — `match failure in 'classify': no pattern matched` — from
the clone as well.  The former extent walk that demoted any try whose
body could transitively reach a `match_fail` (or an indirect call, which
made the extent unknowable) is deleted: with every failure now raising
byte-identical text, extent knowledge is unnecessary.

### Memory

The pad is a stack object; the caught message is a fresh heap copy that
**leaks by design** (it is an ordinary produced `str` that may outlive the
try, like every string the backend cannot prove dead).  The scheduler
itself stays leak-clean across a caught failure: ASan with leak checking on
reports only `mx__dup` allocations.

## The one `InterpError` `catch` does not catch

`mir_interp.RecursionLimitExceeded` — the interpreter running out of
recursion budget (see docs/v1_gap_analysis.md § "Recursion depth") — is an
`InterpError` so that it renders as a located Metaxu diagnostic, but
`try_scope` re-raises it instead of routing it to the catch arm.

It is interpreter resource exhaustion, not a failure the program produced.
A catch arm would run with the stack still at the ceiling, so it would
either overflow again immediately or "recover" onto a stack that can no
longer do useful work. The native backend uses the real machine stack and
has no recoverable equivalent either, so leaving it uncatchable keeps the
two backends from diverging on the recovery path — the same reason the
catch binding is plain text. Pinned by
`test_recursion_depth.test_try_catch_does_not_swallow_recursion_exhaustion`.

## What the catch binding is, exactly

The value bound to `e` is `InterpError.message`: the **plain failure text
as raised**, and nothing else. It carries no compiler context and no
filesystem paths, so the same program produces the same caught string on
every machine and every backend. It is a language-visible value, not a
diagnostic.

The compiler's own "[in function 'f' declared at f.mx:3:1]" context lives
in `InterpError.note` / `str(exc)` instead, where uncaught failures and
tracebacks still show it (see docs/diagnostics_locations.md). Appending
that note to the caught value — which `locate` used to do by rewriting
`args[0]` — made a caught message depend on the absolute path the file
happened to be compiled from, and diverged from the native backend.

The native backend binds this same plain text: `mx_try`'s catch thunk
receives the runtime's message pointer unchanged, and every catchable
native failure is raised through `mx_raise` with the interpreter's exact
wording (see "Native lowering" above).  A differential test compares the
caught value byte for byte.
