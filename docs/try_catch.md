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

The native backend has no try/catch yet (it demotes `try_scope` to a
placeholder); when it grows one, the catch binding must be this same plain
text.
