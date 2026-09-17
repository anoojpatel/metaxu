# Type inference: improvement plan

Companion to book chapter 18, which describes the checker as it is:
a flat solver plus a class-constraint conflict detector on the compile
path, and a full SimpleSub biunifier (`metaxu.simplesub.Biunifier`)
that computes principal types on request but does not decide what
compiles. This document is what to change, in priority order, with the
algorithm for each change and the tests that pin it.

## Priorities

1. **Close the call-edge gap.** `apply(fn(s) -> s + "!")` against
   `fn(int) -> int` compiles and fails at run time (a Python `TypeError`
   escapes the interpreter). The conflict detector unions variables
   along unify edges only; a lambda argument's parameter is reached
   through a call edge. Fold call edges into detection, parameter-wise
   and result-wise, and turn the interpreter's host `TypeError` into an
   `InterpError` so nothing leaks even when a gap remains.
2. **Let-polymorphism on the compile path.** `let same = fn(x) -> x`
   used at Int and String compiles today for the wrong reason (the two
   uses never meet); its principal type is `Int ∨ String`. Once part 1
   lands, that program would be *rejected*, so generalization must land
   with it or immediately after. The biunifier already has levels,
   extrusion, `generalize` and `instantiate`; the compile path needs an
   equivalent that keeps the single solve (below).
3. **Conflict diagnostics with two locations.** The detector reports
   the last conflicting node; it should show where each requirement
   came from, with an excerpt for each. All the information exists on
   the class constraints.
4. **Promote the biunifier.** The engine reports every clash the flat
   solver does plus the ones behind call edges, and its principal types
   are exact on expression chains. Making it the compile-path decision
   maker is the principled end state; the risk is the coarse
   statement-flow edges the emitter also uses for effect propagation,
   which over-approximate. Sharpen those first (emitter work), then
   flip behind a flag with the whole suite and the book run both ways.

## 1. Call edges in conflict detection

For each `call(callee, args, result)` where the callee's type is a
function: union `args[i]` with `param_types[i]` and `result` with
`return_type`, for the purposes of class-conflict detection only (the
flat solver's unification is unchanged). For a callee that is still a
variable at that point, skip; the biunifier's `callee <: (args) ->
result` handles it and can be consulted for an advisory. Pin with
`test_inference_cases.py::test_lambda_argument_body_mismatch_is_rejected`
and keep `test_biunification.py` byte-identical.

Interpreter: wrap host `TypeError`/`ZeroDivisionError` at the binary
operator dispatch into `InterpError` with the operator and operand
kinds named, so a checker gap is a loud, catchable runtime error.

## 2. Let-polymorphism by constraint replay

The emitter buffers constraints and solves once, so HM-style
generalize-after-solve does not fit. `compiler/generalize.py` records
the constraints a lambda emits and the variable ids allocated during
its walk, and instantiates the recording under a fresh substitution at
each use. Outer bindings have earlier ids and stay shared (captures
remain monomorphic). Instances alias the original so `once` counting
and effect propagation see one lambda. Core tests
(`test_generalize_core.py`) pass on the engine; the two emitter hooks
are:

- `LetBinding` whose value child is a `LambdaExpression`: wrap the
  child's walk in `facade.record_scheme()` and store the scheme by
  name (value restriction: lambdas only).
- `Variable` use whose binding has a scheme: `facade.instantiate` and
  unify the use with the instance instead of the original.

Acceptance: `same(1); same("a")` still compiles after part 1, and the
biunifier's principal types for the two call results become `Int` and
`String` respectively (today both are `Int ∨ String`).

## 3. Conflict diagnostics with provenance

Each `class` constraint carries the node id that emitted it. Report
the merge point as the headline and one located excerpt per
requirement ("Int comes from here:", "String comes from here:"). Pin
the full text in chapter 6 (it pins only the fragment today).

## 4. Promoting the biunifier

Prerequisite: replace the statement-flow edges (every statement flows
into its block's type) with edges that carry only what effect
propagation needs, so principal types on statement-heavy functions
stop over-approximating. Then: behind a flag, take hard conflicts from
`Biunifier.errors` instead of `_detect_class_conflicts`; require the
suite, both example gates, and the book to pass byte-identically with
the flag on; flip the default; delete the flat detector.

## Not planned

Union and intersection types in surface syntax; nominal struct
extension; structural width subtyping for records. Chapter 18 lists
them as design-document sketches, and nothing above depends on them.
