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
   index out of bounds, wrong-arity effect op, missing trait impl), and
   match failures. Borrow/type errors are compile-time and are NOT
   catchable.
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
   It becomes an `MX_EV_ERROR` event, which `mx_handle`/`mx_resume` re-raise
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

One asymmetry remains and is handled by **demotion, not by guessing**: a
match failure IS catchable in the interpreter, but its message embeds the
MIR function name (`match failure in 'classify': no pattern matched`) and
the native lane runs monomorphization, which renames a specialized generic
(`classify` -> `classify$Int`).  Emitting the message with the name codegen
sees would bind a *different* string; emitting nothing would fail to catch
what the interpreter catches.  So `codegen_llvm` computes the try body's
transitive extent and demotes the owner when it can reach a `match_fail`
(or an indirect call, whose target set is not statically fixed).  Lifting
this needs the pre-monomorphization name carried into MIR.

### Memory

The pad is a stack object; the caught message is a fresh heap copy that
**leaks by design** (it is an ordinary produced `str` that may outlive the
try, like every string the backend cannot prove dead).  The scheduler
itself stays leak-clean across a caught failure: ASan with leak checking on
reports only `mx__dup` allocations.

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
