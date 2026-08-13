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
EFFECT_MUTEX_CREATE/LOCK/UNLOCK and EFFECT_SPAWN/JOIN under a documented
single-threaded execution model: spawn runs the child closure to
completion at spawn time (one legal schedule of real thread semantics),
join returns its stored result once, and mutexes are exact — locking a
locked mutex is a deadlock and errors loudly, as do unlock-of-unlocked,
double join, and any mapped symbol without a shim. In-scope handlers
still override runtime mappings (effects stay virtualizable). Real OS
threads remain out of scope for the interpreter; the LLVM backend still
has no effect-op runtime (see below).

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
  the missing impl. Not covered (left to inference/runtime): substitution
  inside type applications (`Vec[T]`), qualified/method call type args,
  bracket-form constructor calls (`Full[Int](x)` parses as an index-call),
  impl where-clauses, variance, higher-kinded params, associated types.
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
