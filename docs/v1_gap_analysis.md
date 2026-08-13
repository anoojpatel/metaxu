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

Highlights beyond the original list, found by an adversarial review of the
branch diff and fixed with regression tests (`test_parsed_source_semantics.py`,
`test_effect_continuations.py`, `test_closures_mir.py`,
`test_generic_angle_lexing.py`):

- Parsed match arms carry expression nodes as patterns; they now convert to
  real patterns instead of silently degrading to wildcards.
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
- Struct-literal field type checking covers literal fields against known
  primitives (incl. substituted type params); full inference-driven field
  checking is future work.
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
