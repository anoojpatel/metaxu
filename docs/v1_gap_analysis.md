# Metaxu v1 Gap Analysis (working document)

Audit of `src/metaxu/compiler/` against the documented v1 language
(`docs/type_system.md`, `docs/ownership_and_borrowing.md`, `docs/effects/`,
`docs/compiler_roadmap.md`), as of commit `7ddfa00`. This drives the v1
completion work; it will be updated as gaps close.

## Status: closed (this branch)

All workstreams below landed on this branch, each verified by source-level
tests (282 passing) and the example gates (`scripts/run_examples.py`:
19/19 through parse -> desugar -> freeze -> infer -> HIR -> MIR -> CLIF,
15/19 executing correctly at `--stage run`, with `test_borrow_check.mx` /
`test_type_error.mx` required to be REJECTED with borrow/type diagnostics —
they are negative fixtures).

Also landed since: trait method dispatch (implement blocks desugar to
mangled functions; runtime dispatch on the receiver's type, user impls win
over builtins) and a minimal runtime library (mutable Vec, immutable
vector[T,N] with element-wise ops, indexing/slicing, sqrt/sin/cos,
to_string/len). The remaining non-executing examples need real FFI/threads
(05, effect_mapping), a SimdOp handler + const-generic N at runtime (06),
and try/catch lowering (04).

Update (2026-08-13): all of those now execute — the run gate is 19/19.
effect_mapping.mx was the last: its `with EFFECT_*` clauses (effect ops
mapped onto named runtime primitives) now compile to
`__effect_runtime$Effect$op` thunks, and the interpreter carries shims for
EFFECT_MUTEX_CREATE/LOCK/UNLOCK and EFFECT_SPAWN/JOIN — originally under
a single-threaded execution model (spawn ran the child to completion at
spawn time).

Update (2026-08-18): the single-threaded model is GONE — both engines now
implement REAL OS threads behind those shims (`docs/threads_runtime.md`
is the spec). Interpreter: spawn starts the closure on its own
`threading.Thread` immediately; join blocks and returns the result
exactly once (double join errors loudly, matching pthread_join); mutexes
are ERRORCHECK-style owner-tracked wrappers over `threading.Lock` — lock
held by ANOTHER thread blocks, self-relock is the loud deadlock error,
unlock-not-held errors; a child's catchable failure surfaces at join,
catchably. Each spawned thread owns its OWN effect scope stack (children
do not inherit the spawner's in-scope handlers; handle-body threads adopt
their creator's context), while in-scope handlers on the performing
thread still override runtime mappings (effects stay virtualizable — a
`spawn(f)` handler case now parses correctly too; it used to collide with
the `spawn` keyword and silently never match). Native: metaxu_threads.c
(pthreads, PTHREAD_MUTEX_ERRORCHECK, immortal handles, `-pthread` on
every link) with ALL effects-scheduler globals `_Thread_local`; mapped
ops no longer demote — unscoped performs call their
`__effect_runtime$E$op` thunk directly, scoped ones route through
mx_perform_or_default with the thunk as fallback. effect_mapping.mx went
from 0 defines / 7 placeholders to 7 / 0; differentials (counter, join
value, error messages) match the interpreter byte-for-byte and run clean
under -fsanitize=thread and ASan (detect_leaks=0: handles are immortal
by design). Error messages are shared byte-for-byte across engines.
Spawn-boundary modes are now partially ENFORCED at compile time
(`compiler/spawn_capture_check.py`): a closure passed to a `with
EFFECT_SPAWN` op must not capture `@local` values or active `@mut`
borrows (kinds `locality-spawn-capture`/`borrow-spawn-capture`, hard
BorrowCheckErrors). Locality PROPAGATES per Rule B (2026-08-18,
`docs/ownership_and_borrowing.md` § "Locality follows the data"):
unannotated bindings and assignments inherit the initializer's
locality with provenance-chain diagnostics, and the only escape is the
explicit `let @global g = v` — a checked coercion allowed exactly when
mode-crossing evidence (scalar type) certifies it, kind
`locality-escape` otherwise. See `docs/threads_runtime.md` § Modes for
the enforced subset and what remains untraced.

Update (2026-08-19): contention is now enforced DYNAMICALLY per
`docs/contention_as_permission.md` (design B — the effects-native
answer to OxCaml's contention axis). A Vec captured by a closure
crossing a REAL spawn is marked contended at the crossing (struct
fields recursed by static layout, STOP at Vec elements — the
vec-of-vecs hole is test-pinned, identically on both engines);
mutating a contended Vec with no runtime mutex held raises catchably
with byte-identical wording, while reads stay free; permission is a
per-logical-thread held-mutex counter bumped by the mutex runtime
(`_ThreadCtx.write_permit` / `_Thread_local mx__tls_write_permit`).
Natively the marking walk is emitted inside the `EFFECT_SPAWN` runtime
thunk — which executes exactly when the spawn is real, on both the
direct-call and the mx_perform_or_default fallback routes — so
handler-virtualized spawns never mark, and unenumerable capture kinds
demote the thunk with a reason. `unsafe { }` spawn escapes therefore
land on a loud runtime net instead of nothing (the old locks-deleted
TSan race experiment now fails deterministically with the
contended-write error). Costs are measured, not asserted — alignment-controlled: the
uncrossed mutator pays ~+7.5% on a worst-case pure-mutation microloop
(~+0.23 ns/guarded call; earlier unaligned readings of +15.8% and even
-17% were shown to be code-layout artifacts — see the spec's Measured
section) —
and `tests/test_contention.py` pins semantics, differentials, TSan
and non-vacuity.

Direction update (2026-08-13): the project targets LLVM for AOT native
compilation (near-C, no GC; modes decide memory). Increment 1 is on this
branch: codegen_llvm.py emits a verifier-clean LLVM module for the direct
subset (scalars, control flow, calls, strings/print, local structs as
stack allocas); llvm_run.py compiles with clang -O2 and executes;
differential tests pin native results == interpreter results. try/catch
is implemented per docs/try_catch.md (example 04 executes; run gate
16/19). Increment 2 (ownership memory model) is also on this branch:
structs cross call boundaries (params as ptr + callee byval-copy, returns
sret-style via a leading result-slot pointer), @global structs live on the
heap (entry-block malloc, GEP on the heap pointer, freed on every ret path
-- sound because MIR value semantics never lets the storage pointer escape
a frame; ASan-verified differential tests prove no leak/double-free), and
print takes multiple arguments (space-joined printf, matching the
interpreter). The `let @global x = S {...}` annotation now actually
reaches MIR's alloc_struct (hir.py previously dropped string-token mode
annotations -- the classic silently-degraded-mode seam). Increment 3
(variants + closures) is also on this branch: enums compile to tagged
unions (%enum.E = { i64 tag, [N x i64] payload }; variant names map to
dense module-wide integer tags so compiled pattern tests compare integers,
never strings; payload slot kinds unify per (enum, slot) with an explicit
no-coercion rule -- heterogeneous slots demote rather than silently
promoting an int store to the unified kind), and closures compile to
{ ptr fn, ptr env } pairs over per-lambda stack env structs (direct
locally-bound calls work, closures pass DOWN as arguments, aggregate
captures copy whole into the env; everything that would dangle the stack
env -- returning a closure, storing one in a field/payload, capturing one
in a closure, creating one in a loop -- demotes with the reason recorded).
Statement-position if/match results that copy-merge arms of different
kinds are recognized as provably dead and elided instead of poisoning kind
inference. Deferred LLVM increments, always as reasoned placeholders:
heap-boxed enum payloads (recursive enums like linked lists), heap-env
closures (escaping upward), nested struct fields, borrow-informed copy
elision for @const params, vec/string runtime, effect CPS at the LLVM
level.

Native try/catch (2026-08-14): `try_scope` was the last construct with no
LLVM lowering at all. It now emits `mx_try` over per-site body/catch thunks
(`metaxu_effects.c`: setjmp landing pads whose chain is per-fiber, so they
compose with the ucontext coroutine scheduler in both directions — a `try`
inside a handle body survives a perform/resume round trip, and a failure on
a body coroutine escapes to its owner stack as the interpreter's
`("error", exc)` message does). Catchable native failures are raised with
the interpreter's exact wording so the catch binding is byte-identical; the
one asymmetry, `match_fail`, was closed on 2026-08-18: the
pre-monomorphization name rides through MIR (`MirFunc.origin_name`), so a
specialized clone raises the interpreter's exact message and the former
try-extent demotion walk is deleted.
See docs/try_catch.md § "Native lowering".

Native Vec element boxes (2026-08-14): a native Vec slot is one 8-byte
word, so any struct/enum element demoted the containing function. Struct
and enum aggregates now occupy a slot as a pointer to an immortal
write-once ELEMENT BOX (the same machinery as effect-boundary and enum
payload boxes): the write mallocs a fresh copy, the read copies back out,
so both edges keep MIR's value semantics while the vec keeps mx_vec
identity semantics. Closures stay excluded (an env pointer may aim at a
frame the vec outlives) and fixed vectors keep the word-kinds-only rule
(an mx_fvec block feeds the SIMD/arith paths, which read words as
numbers).

### What actually blocks native coverage (measured 2026-08-14)

`scripts/native_census.py` counts real `define`s, demoted functions and
demotion reasons with instance detail normalized away, so reasons group
into CAUSES. Counting instances has misled this branch twice; the census
exists to stop that.

On `examples/app/main.mx` (39 defines / 125 demoted), 99 of the 125 are
root demotions rather than cascade ("calls a function that is itself a
placeholder"), and the smallest carry exactly one reason:
`irreconcilable value kinds`. The dominant root cause is a single
mechanism:

> **Monomorphization does not specialize a function whose polymorphism
> flows through a closure-typed parameter.** `std.throw.catch_` has four
> call sites at four different return types and exactly ONE MIR function
> — no clones — because the pass keys on concrete argument signatures and
> a lambda argument offers none. The backend then joins all four return
> kinds into `conflict` and demotes.

Six lines reproduce it, and isolate it from every other variable:

```
fn apply(f) { f() }
fn main() -> int {
    let a = apply(fn() -> 1);
    let b = apply(fn() -> 2);      # both int  -> 5 defines, 0 placeholders
    ...
}
```

Change the second lambda to `fn() -> "s"` and `apply` demotes with
`irreconcilable value kinds`, taking its callers with it.

**Implemented (2026-08-14): per-call-site cloning.** monomorphize.py now
rewrites a call site passing a lambda LITERAL to a known non-generic
function into a clone of that function unique to the site (`apply$ho1`).
The clone body is byte-identical, so behavior cannot change — the
rejected alternative, inferring the lambda's return type, could
miscompile — while the backend's kind cells see one lambda per clone.
Callees with a single call site are left alone (nothing joins), recursive
calls keep the original, `__`-prefixed names never clone (their spelling
is their dispatch), and dead originals are erased by a reference scan.
The six-line repro above now emits both callers cleanly, and three shapes
that demoted BY CONTRACT at shared sites (@mut aggregate write-back
through an indirect call, struct-vs-enum kind disagreement, closure pairs
as indirect arguments) emit and match the interpreter when routed through
literal sites. Var-routed lambdas still share the original and keep the
word-uniform/demotion machinery.

Result on app/main.mx: 39 → 41 real defines, reasons 482 → 465. The next
blocker in the chain is now visible per-site instead of joined: clones of
`catch_`/`run_state` inherit `conflict` from `run_program`'s own return
kinds — the chase continues one level deeper.

This is worth stating because the previous two hypotheses were both
wrong, and both were wrong the same way — counting instances instead of
finding causes. Vec-of-aggregate elements looked like the biggest lever
(175 + 41 + 38 downstream reason instances sat under it); boxing them
removed exactly 6 reason instances and left the demoted-function count
unchanged at 148. Before that, 18 "irreconcilable kinds" reasons in
example 06 turned out to be one trait-dispatched call site.

Roadmap items completed on this branch beyond the v1 criteria:
- Item 5 (deep field-mode validation): transitive @global/@local ownership
  rules enforced as kind="deep-locality" diagnostics.
- Item 7 (CLIF direct codegen): real Cranelift IR text for the direct
  subset (multi-block CFG, stack-slot slots, typed signatures, calls);
  functions outside the subset emit declaration-only placeholders; a
  structural validator covers all emitted output.
- Item 6 (selective CPS): suspending functions in the i64 direct subset
  emit defunctionalized frame layouts (cps_frames.py), %run_<f> br_table
  state machines parking at performs via the runtime stub ABI
  (enqueue/sched_read), and %resume_<f>_<k> shims. Not yet done: frame
  chaining across suspending calls, CLIF-level effect dispatch (stays in
  the interpreter), f64 CPS bodies.

Highlights beyond the original list, found by an adversarial review of the
branch diff and fixed with regression tests (`test_parsed_source_semantics.py`,
`test_effect_continuations.py`, `test_closures_mir.py`,
`test_generic_angle_lexing.py`):

- Parsed match arms carry expression nodes as patterns; they now convert to
  real patterns instead of silently degrading to wildcards.
- Non-exhaustive matches are compile-time errors (`test_exhaustiveness.py`,
  kind `type-nonexhaustive-match`), not runtime `match_fail` traps: enum
  matches (declared enums plus builtin Option/Result) must cover every
  variant or have a wildcard/binding arm; bool matches need true+false or a
  catch-all; int/string/float literal arms always need a catch-all. The
  coverage rule is deliberately shallow — a ctor arm covers its variant only
  when its subpatterns are all irrefutable, so `Some(1)` alone does not
  cover `Some` while `Some(1) | Some(n)` is covered by the binding arm
  (literal-completeness refinement is not analyzed). Matches whose
  scrutinee type the patterns cannot determine, or containing opaque
  pattern forms, are not checked (no false positives). An arm after a
  wildcard/binding arm gets a warning-level unreachable-arm advisory on the
  -1 diagnostics channel.
- `Some(x)`/`None` in argument position were silently dropped at HIR.
- Mode annotations survive freezing, so `@local` escape is actually rejected.
- Borrow/move state is per-function; moves no longer poison other functions.
- `&mut` in argument position is an exclusive borrow released after the call,
  per `ownership_and_borrowing.md` — not an ownership transfer.
- Lambda closures capture by slot name and let-bound closures are callable.
- Captured scalars a closure/handler arm ASSIGNS to are shared cells
  (MIR `cell_wrap`): the mutation writes back to the enclosing binding
  instead of silently vanishing in a by-value env copy.
- `v[i] = x` is a real store (`__index_set`/`__index_store`): Vec mutates in
  place, a fixed vector in an assignable place gets a value-semantics update
  written back, and unsupported targets are loud errors, never no-ops.
- Module-level `let` bindings are constants initialized before the entry
  point (synthesized `__module_init`, one global constant namespace with
  loud cross-module collisions); `%` is a real modulo operator; `op() -> ()`
  handler arms lower as abort-with-unit arms instead of being dropped.
  (Regression tests: `tests/test_silent_seams.py`.)
- `f(a < b, c > d)` lexes as comparisons (generic-angle follow-set check).
- `resume(v)` returns the value of the WHOLE delimited handle body (the
  interpreter parks the body on its own thread per handle scope), so
  non-tail resumes with performs in called functions are correct.

Known remaining gaps (documented, not v1-blocking):

- `codegen_clif.py` remains a stub; the MIR interpreter is the executable
  backend.
- Generics are parametrically checked (`test_generics.py`): explicit
  instantiations (`Stack<Int>{...}`, `Full<Int>(x)`, `identity<Int>(x)`)
  substitute type args into declared field/payload/param types and enforce
  them for values of known type, feeding var-typed values into the
  constraint-graph conflict detection; omitted type args are inferred
  CALL-SITE-LOCALLY (let-polymorphism lite: `identity(1)` and
  `identity("s")` coexist); `where T: Trait` / `fn f<T: Trait>` bounds are
  enforced against the impl registry for resolved instantiations, naming
  the missing impl. Since first writing, four caveats have closed:
  substitution INSIDE type applications (a field/param declared `Vec[T]` /
  `Pair[T]` / nested `Pair[Pair[T]]` checks against the substituted
  application — base constructors must match, and argument positions
  recurse where the value's own type args are known, e.g. an explicit
  `Pair<String>{...}`; unknown element types keep base-name-only checking,
  no false positives); bracket-form explicit instantiations
  (`Full[Int](x)` parses as an index-call — a desugar pass,
  `BracketCtorCallDesugarPass`, recognizes the shape when the base is a
  known generic variant/function and every index is a type display, and
  rewrites it to the angle-form FunctionCall, so checking, HIR lowering
  and monomorphization all treat both spellings identically; value
  indexes like `arr[i](x)` are untouched); impl-block where clauses
  (carried through trait-impl desugaring onto the mangled `__impl$` fns
  and enforced at coherence-load time against the impl registry where
  decidable — i.e. constraints over concrete types like
  `implement Show for P where Q: Eq`; conditional impls,
  `implement Show for Pair[T] where T: Show`, depend on per-instantiation
  type args the base-name registry cannot see and stay permissive with
  runtime dispatch enforcing); and module-qualified generic calls
  (`mod.f<Int>(x)` — type args survive the module system's rename into
  the frozen QualifiedFunctionCall payload and the HIR call, so they are
  checked and monomorphized like plain calls). Still not covered (left to
  inference/runtime): method-call type args, variance, higher-kinded
  params, associated types.
- An optional monomorphization pass (`compiler/monomorphize.py`, HIR->HIR,
  pipeline flag `monomorphize=`, default off) clones generic functions per
  concrete instantiation (`identity$Int`), rewrites call sites (including
  transitively inside clones), and erases fully-specialized originals —
  never synthesized `__`-names, which trait dispatch reaches dynamically.
  Interpreter results are pinned identical with the pass on and off;
  the specialized names in MIR are groundwork for native codegen.
- Traits/impls dictionary desugaring and deep field-mode validation remain
  at their pre-branch level.

Closed since first writing: multi-argument effect ops bind all handler-case
parameters, and a handler case performing its own effect routes to the next
enclosing handler (handler bodies evaluate outside their own delimitation)
instead of deadlocking.

### Found by `examples/app` (the whole-language application, 2026-08-14)

`examples/app/` is a five-module interpreter for a small expression
language (lexer -> parser -> evaluator, ~680 lines of Metaxu). It is the
first artifact that uses the whole language at once — modules, generics
with `where` bounds, a trait with three impls, a custom effect with three
handlers plus `std.throw`/`std.state`, tuples, a recursive AST enum, six
`std.*` modules, and `@mut`/`@local`/`@global`. Writing it exposed three
front-end bugs, all fixed on this branch:

1. **Exhaustiveness rejected genuinely exhaustive nested patterns.**
   A variant counted as covered only when some arm named its ctor with
   all-irrefutable subpatterns, so `match o { Some(TInt(n)) => ..,
   Some(TName(s)) => .., Some(TOp(s)) => .., None => .. }` was rejected
   with "missing variants Some" — naming a variant that was right there.
   Coverage is now the textbook specialization recursion
   (`frozen_constraint_emitter._matrix_exhaustive`), which handles nesting
   to any depth and multi-field variants (`Pair(A, A) | Pair(A, B) |
   Pair(B, A)` is still one combination short), and answers "not covered"
   for everything it cannot prove — literal columns, unknown enums, mixed
   ctor/literal columns — so nothing that compiled before stops compiling.
   Tests: `test_exhaustiveness.py`.
2. **`@local` did nothing for tail-expression returns.** The locality
   escape check fired only on `return x;`, while `fn f() { let @local c =
   ..; c }` — the idiomatic spelling, and the one the whole stdlib uses —
   handed the local to the caller silently. The function's tail
   expression is now checked exactly like a return.
3. **Argument-count mismatches were not a compile error.** `take(1)` for
   `fn take(a, b)` compiled and died at run time with "Unbound variable
   'b' in 'take' (bad lowering or use-after-drop)", blaming the compiler
   for the caller's mistake. Calls to a known signature now check arity
   (skipping names shadowed by a local binding, whose arity is its own).
   Tests for 2 and 3: `test_silent_seams.py`.

4. **Recursion depth: ~45 user frames, and a host traceback at the
   bottom.** A calc program whose evaluation nested deeper than ~45 levels
   (a 50-term left-associative sum was enough) exited with a raw Python
   `RecursionError` — no location, no function name, the host language
   leaking through the boundary. Fixed on this branch; the ceiling, the
   numbers behind it and the remaining interpreter/native divergence are
   the section below.

## Recursion depth (interpreter/native divergence)

The MIR interpreter recurses on the host Python stack, and a Metaxu call
frame costs SEVERAL Python frames — `_call_func` -> `_run_blocks` ->
`_run_ops` -> `_eval_rhs` -> the next `_call_func`. Measured on this tree:

| shape | Python frames per Metaxu frame | depth at the stock 1000 limit |
|---|---|---|
| bare `f(n) -> f(n - 1)` | ~4 | 247 |
| `examples/app`'s evaluator (match + closure + `perform` + std helper per AST node) | ~24 | ~45 |

`compiler/recursion.py` now installs a 100_000-frame budget for the extent
of one entry-point call and puts the host's limit back afterwards (scoped,
because this package is a library — pytest, the LSP server and
`scripts/run_examples.py` all import it). That buys ~24_700 frames of the
cheap shape and ~4_100 nodes of the expensive one, both measured, and the
same budget covers the compile-time phases, which walk the AST recursively
too (~200 nested binary operators used to overflow during compilation, and
PLY reported it as `ParseError: maximum recursion depth exceeded` —
blaming a syntax error that was not there).

Raising `sys.setrecursionlimit` does not protect the C stack, so the number
is not free-floating. It rests on two CPython properties, both pinned by
subprocess tests that assert the child's exit status (a stack smash is a
negative returncode no `except` clause could hide): CPython >= 3.11 keeps
Python frames on the heap and pushes no C frame for a Python-to-Python
call (a 20_000-frame Metaxu recursion completes on a thread given a 128 KiB
stack), and CPython >= 3.12 guards genuinely C-recursive work — nested
`repr`, comparison, deallocation — with a SEPARATE limit `setrecursionlimit`
does not move. `handle` bodies, which run on their own threads, are started
with a 16 MiB stack (`mir_interp._start_with_stack_size`) so the effect path
is no more fragile than the main thread.

**The divergence.** The ceiling is an interpreter property only: the native
backend uses the real machine stack and has no equivalent limit (nor any
diagnostic — it would fault). Exceeding the interpreter's ceiling is
`mir_interp.RecursionLimitExceeded`, an `InterpError` naming the innermost
Metaxu function, and it is deliberately NOT catchable by `try`/`catch`
(docs/try_catch.md): a catch arm would run with the stack still exhausted,
and the native backend has no recoverable equivalent to diverge from.
Tests: `test_recursion_depth.py`.

Not planned for v1: a trampolined / explicit-stack interpreter, which would
remove the ceiling entirely by holding the Metaxu frame stack in a heap list
instead of the Python stack. It is a rewrite of the interpreter's core, not
a refactor — every `_eval_rhs` recursion, the delimited-continuation
machinery (which currently IS a parked Python stack on a thread) and
`try_scope` would all have to become explicit states — and it would buy
depth this budget already provides at ~1/100th of the cost. Revisit if a
program needs more than ~20_000 frames, or if the effect scheduler is
rebuilt for other reasons.

## Effects pump: tail-resume trampoline (2026-08-19)

The handler pump used to be RECURSIVE per event on both engines: a case's
`resume(v)` parked the handler side inside the resume frame inside the
case frame, so each stream element left a (case frame + resume frame)
pair on the scope's owner stack for the whole stream. Natively that was
~250 B/element against the 1 MiB coroutine stacks — a
`sum(map(filter(iota(n), ...), ...))` pipeline segfaulted between 2,000
and 5,000 elements — plus one never-freed `mx_k` (~1 KB of ucontext) per
element; the interpreter hit its 100k-frame recursion budget between
10,000 and 20,000 elements. Fixed by a codegen-assisted trampoline:
`compiler/effect_tail.py` marks resumes in TAIL position (the case's
value IS resume's value — the shape of every std.stream/std.state/
std.log arm except `fold`'s), the native runtime's owner-side event loop
(`mx__pump_events` in metaxu_effects.c, where the equivalence argument
against the recursive scheme is written down) performs those switches
from a constant frame and frees each consumed continuation record, and
the interpreter's `_pump_scope` loops on a `_TailResume` unwind the same
way — BOTH engines consult the same analysis, so they trampoline the
same sites. `std.stream`'s `sum`/`product`/`count` moved from foldr to
tail-shaped accumulator arms as part of this. Measured after the fix:
the pipeline runs 1,000,000 elements natively at flat stack/heap and
200,000+ in the interpreter, with identical outputs. What remains O(n)
by SEMANTICS, not implementation: genuine foldr (`fold`'s
`f(x, resume(()))` does work after the resume — its pending
applications are per-element frames on the owner stack; a bare
`fold(iota(n), ...)` measured native 10k ok / 30k segfault on the 8 MiB
main stack and interpreter 12k ok / 20k RecursionLimitExceeded under
the 100k-frame budget, i.e. a ceiling of 1-3 x 10^4 foldr DISPATCHES —
elements that reach the fold, so a filter in front buys
proportionally more), and any other arm that computes after resuming.
Tests:
`test_effect_tail_resume.py` (analysis, 1M native stack stability,
interpreter 20k, non-tail differentials, single-shot on a consumed-then-
tail-resumed continuation, ASan).

## Loudly-unsupported surface constructs (HIR triage)

Update (2026-08-14): HIR lowering used to end in `return None` for any AST
node class it did not recognize, and every caller skipped a None result — so
the construct simply VANISHED and the program compiled to something that
quietly did less than it said. Eight instances of that one bug were found by
accident, one at a time (if-let, while-let, `unsafe { }`, `@mut e`, list
literals, `for`, `e as T`, early `return`, struct-field initializers).

The fallback is now loud and every AST node class is triaged into exactly one
bucket in `AST_NODE_TRIAGE` / `PATTERN_TRIAGE` (`compiler/hir.py`), each with
a one-line reason; `tests/test_hir_coverage.py` fails if a newly added node
class is left unclassified. The authoritative list lives in that table — this
is the summary of what is deliberately NOT supported and errors loudly:

`TupleLiteral` moved OUT of the unsupported buckets (in both tables) when
tuples landed: `(a, b)` / `(a, b, c)` are values, `(A, B)` is a type, and
`let (a, b) = p;`, `match p { (x, y) => .. }` and `for (k, v) in ps { .. }`
destructure. A tuple **is an anonymous struct** — `alloc_struct "__tuple2"
{ _0of2, _1of2 }` plus a `field_get` per element — so MIR gained no op, the
interpreter gained no value class and native codegen inherited the struct
path unchanged. The field name repeats the arity because inference has no
tuple type: `_0of2` does not exist on a `__tuple3`, which is what makes a
mismatched destructuring a loud error rather than a silent prefix bind.
See `std/README.md` gap 8 and `tests/test_tuples.py` for the full contract,
including the two shapes that demote natively with a reason (two different
tuple types of one arity in a module; a tuple nested directly inside a
same-arity tuple).

- **GPU**: `to_device(x)`, `from_device(x)`, kernel annotations — no runtime.
- **Comptime**: `comptime { }` blocks, `comptime fn`, compile-time values,
  `typeof`-style type reflection, compile-time matching on types — compile-time
  evaluation is not implemented. (`comptime fn` is rejected explicitly rather
  than silently compiled as an ordinary run-time function.)
- **Threads**: the bare `spawn(f())` EXPRESSION form is REMOVED from the
  language (2026-08-18): it duplicated the effect route without being
  virtualizable, had no semantics behind it, and its keyword caused a
  real parse bug (a `spawn(f)` handler case could never match). `spawn`
  is an ordinary identifier now; threads go through
  `perform Thread.spawn(..)` and calling an undefined `spawn(..)` gets
  a compile error pointing there.
  Real threads ARE available through the effect-mapped route
  (`perform Thread.spawn(...)` with `with EFFECT_SPAWN`, backed by OS
  threads on both engines — see `docs/threads_runtime.md`); only the
  bare keyword expression remains unimplemented.
- **Raw pointers**: pointer dereference has no HIR/MIR representation.
- **Uncalled generic instantiation**: `let f = ident<int>;` — a type-applied
  function has no value representation; call it directly (`ident<int>(x)`).
- **Exhaustiveness blind spot (frozen-AST lossiness)**: the exhaustiveness
  checker walks the frozen AST, which drops match-ARM bodies — so a
  non-exhaustive match NESTED inside another match's arm passes the gate
  and fails at run time (`match failure in 'f': no pattern matched`,
  catchable by `try`). Same lossiness family that moved name resolution to
  the mutable AST (docs/name_resolution.md); the fix is the same shape.
  Top-level matches are fully checked (enum variants must be exhaustive;
  int/string literal matches require a wildcard).
- **Bare comprehensions**: `f(e for x in it)` — only the vector-literal form
  `vector[T,N](e for x in it)` has a value representation.
- **1-tuples**: `(e)` is parenthesized grouping and `(e,)` is a syntax
  error, so `let (a) = e` and a `()` *pattern* are rejected by name rather
  than given a `__tuple1` layout. (Tuples of two or more elements are
  supported — see below.)
- **`vector[T,N]` in value position** — it is a TYPE; the value forms are
  `vector[T,N]()`, `vector[T,N](e, ...)` and `vector[T,N].filled(e)`.
- **Pattern forms**: list patterns (`[]`, `[x, ...xs]`),
  struct patterns, range patterns (`1..5`), guards, arbitrary expressions,
  matching on the structure of a function, and matching against a field's or
  an indexed value. Each of these used to degrade to a match-anything
  wildcard, which silently made the arm win for every value and every later
  arm dead code. `examples/02_effects_and_handlers.mx`'s `map` was written
  with list patterns and therefore always returned an empty list; it is now a
  loop.

`examples/06_vector_operations.mx`'s `with_simd` used the inline
`handle SUBJECT { perform Op(p) => body }` form, which had no lowering at all
— the handler vanished, `f` was never called, and the function returned unit.
That form is now lowered for real; its arms answer `None` (the capability's
"no vectorized form available" answer, which is what the file's own comments
promise), because selecting an intrinsic per element function needs the
comptime type matching and function-structure patterns listed above.

## Token and grammar reachability (the layer above the HIR triage)

Update (2026-08-14): the same treatment was applied one layer up, to tokens
and grammar productions — the layer where this project's worst bug lived
(`!` was never a lexer token, and `t_error` logged-and-skipped, so `!e`
compiled as `e`). Full write-up in `docs/token_reachability.md`; the short
version:

- Every token in `Lexer.tokens` is triaged in `lexer.TOKEN_TRIAGE` into
  `GRAMMAR` (97), `CONTEXTUAL` (3 — `once`/`separate`/`many`, reachable as
  `@once` etc.) or `RESERVED_ONLY` (2 — `use`, `kernel`, which now answer
  with a route instead of `Syntax error at 'use'`).
  `tests/test_token_coverage.py` recomputes the `GRAMMAR` bucket from PLY's
  live production table, so the table cannot rot in either direction.
- Nine tokens had no production at all. `impl` is now an accepted spelling
  of `implement` (which is what the docs use); `box`, `option` and `async`
  were reserved with no feature anywhere and are back in the user's
  identifier namespace; `use` and `kernel` stay reserved with guidance;
  `once`/`separate`/`many` were already reachable through the `@` rewrite.
- Grammar reachability is clean: no unreachable or unreferenced
  nonterminals, no orphaned productions, and zero parser conflicts — now
  pinned by tests.
- Silent lexer paths closed: numeric forms the language does not have
  (`1e10` ran as `1`), integer literals outside i64 (interpreter/native
  divergence), unknown mode annotations (`@moot` was silently dropped), the
  import-list keyword rewrite firing in ordinary argument lists, the
  generic-argument scan's silent 80-token cliff, unterminated strings
  reported as an illegal `"`, and `LexError` diagnostics excerpting the
  previously parsed file.

## Name resolution (the last member of that family)

Update (2026-08-14): an undefined variable or callee is now a compile-time
`TypeCheckError` rather than a dropped expression — `undefined_thing; 42`
compiled, ran and answered 42, and `helpr()` (a typo) died only at run time.
This is what made the lexer bug above *silent*: `1e10` split into `1` and an
identifier `e10` in statement position, where an unused undefined name
vanished. `compiler/name_resolution.py` runs over the mutable post-desugar
AST (the frozen AST drops match-arm and for-loop bodies, and cannot tell a
pattern binder from a variable read), enumerates its in-scope categories in
`docs/name_resolution.md`, and scopes out only what runtime trait dispatch
must answer. Zero false positives across the 19 gate files, every
`std/*.mx`, and the ~1,340 programs the suite compiles; it found two real
defects in shipped code, both fixed at the source.

## Headline findings

- 18 of 19 example programs (`examples/*.mx` + root `test_*.mx`) fail at the
  parser. Only `hello.mx` compiles. The grammar accepts a different language
  than the docs describe (effects require `=` and `fn` prefixes, modes are
  postfix, `@local`/`@global`/`@const` cannot lex because those words are
  reserved keywords, `perform` parses as a no-op, `while` has a token but no
  rule, match only supports `Some`/`None` and its one rule has an off-by-one
  that makes the `Some` arm unreachable).
- The 88 passing tests overwhelmingly verify hand-constructed IR, not compiled
  source. Several test files are tautologies (0-assert tests, `try/except:
  pass` wrappers, `assert 1 in tables.types`).
- MIR lowering: nested `if` produces an infinite CFG loop (labels computed
  before arms are lowered); `match` evaluates all arms and returns arm 2
  unconditionally; `resume` compiles to the identity function (only
  tail-resumptive handlers accidentally work); handler functions get colliding
  global names; handlers cannot close over environment.
- Borrow checking: the frozen-AST borrow checker implements the spec, but the
  constraint emitter hardcodes every binding/parameter as `("shared",
  "global")`, so no locality/borrow/linearity rule can ever fire; borrow
  errors never fail compilation.
- Effect classes: any effect marks a function suspending; the documented
  stack-vs-suspend distinction is recorded but never used.
- `codegen_clif.py` is a 21-line fixed-text stub; the MIR interpreter is the
  only executable backend.
- Two front doors: `run_pipeline_from_source` desugars,
  `build_context_from_source` does not.

## Workstreams (in flight on `subagent/*` branches, merged into this branch
after adversarial review)

1. `subagent/parser-v1` — grammar/lexer: doc-faithful syntax for mode
   annotations (prefix `@local/@global/@const/@mut/@owned/@once`), effect
   declarations `effect E<T> { op(...) -> T }`, `perform E.op(...)` producing
   a real `PerformEffect` node, general patterns, enums with type params,
   `while`, semicolon statement termination, `vector`/pointer types,
   multi-imports, borrow/move expressions reachable, correct error columns.
2. `subagent/mir-controlflow-v1` — HIR/MIR: block-label allocation fix
   (nested ifs), real pattern compilation to decision trees, enum/variant
   runtime representation, while/for lowering, interpreter strictness
   (unbound names raise instead of evaluating to their own name string).
3. `subagent/typing-borrow-v1` — real mode/locality data into the borrow
   checker, wiring of region/reference/linearity/exclave APIs, structured
   borrow errors that fail compilation, stack-vs-suspend effect classing,
   desugarer traversal fix, single pipeline front door.
4. Effects/continuations (queued behind 2): capture real single-shot
   continuations at perform sites, handler closures, per-handler unique
   frames, stack/suspend gating at runtime.
5. Traits/impls dictionary desugaring, deep field-mode validation, CLIF
   codegen: after the above land.

## v1 acceptance criteria

- Every `examples/*.mx` and root `test_*.mx` parses; each either runs
  end-to-end under the MIR interpreter with correct semantics or fails with a
  meaningful, documented diagnostic (e.g. `test_type_error.mx` must fail
  type checking — that is its purpose).
- Non-tail `resume`, abort (no resume), and resume-then-continue handler
  shapes all produce correct values.
- Negative tests: locality escape, double exclusive borrow, use-after-move,
  once-used-twice, local-across-suspend each rejected with structured errors.
- Full pytest suite green with the tautological tests replaced by real ones.
