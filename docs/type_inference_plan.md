# Type inference: improvement plan

Companion to book chapter 18, which describes the checker as it is:
a flat solver plus a class-constraint conflict detector on the compile
path, and a full SimpleSub biunifier (`metaxu.simplesub.Biunifier`)
that computes principal types on request but does not decide what
compiles. This document is what to change, in priority order, with the
algorithm for each change and the tests that pin it.

## Priorities

1. **Close the call-edge gap.** DONE. `apply(fn(s) -> s + "!")`
   against `fn(int) -> int` is rejected at compile time; see below for
   what landed.
2. **Let-polymorphism on the compile path.** DONE for `let`-bound
   lambdas (the value restriction), by constraint replay; see below.
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

## 1. Call edges in conflict detection (landed)

Four pieces, all pinned in `test_inference_cases.py`:

- **Declared function types reach the graph.** The frozen payload used
  to render `fn(int) -> int` through `str()` as
  `((TypeReferenceat 0x..) -> ...)`; `mutaxu_ast._type_display` now
  renders it as `fn(int) -> int`, and the emitter builds a *shell* for
  such a parameter (a function CompactType with a fresh variable in
  every position, classed Int/String/Bool/Float where the display says
  so, nested function types recursively) and unifies the parameter's
  variable with it.
- **A lambda's type is its function type from the start.** Every node
  gets its variable before the walk, and a lambda's parent read that
  variable as the lambda's type before the lambda's handler replaced it
  with the function type, so a lambda passed as an argument or bound by
  `let` was represented by its *return* variable.
  `_preallocate_lambda_types` now allocates the function type up
  front; the pre-allocated variable becomes the return type.
- **The detector unifies function types structurally and folds call
  edges.** `_detect_class_conflicts` keeps one function type per
  union-find component; two function types in one component are
  merged parameter-wise and result-wise, and every
  `call(callee, args, result)` whose callee's component holds a
  function type joins each argument to its parameter and the result to
  the return type, to a fixpoint. Callees in `facade.nonfoldable` are
  skipped: named functions with a bare parameter or type parameters,
  which serve every call through one shared type (`fn same(x)` used at
  Int and String must keep compiling, and does).
- **Interpreter backstop.** `_eval_binop` turns a host `TypeError` into
  `InterpError("binary operator '+' cannot be applied to Int and
  String")`, catchable by `try`.

`test_biunification.py` is unchanged and green.

## 2. Let-polymorphism by constraint replay (landed)

The emitter buffers constraints and solves once, so HM-style
generalize-after-solve does not fit. `compiler/generalize.py` records
the constraints a lambda's walk emits and instantiates the recording
under a fresh substitution at each use. Because node variables are
pre-allocated, the generic set is the variables of the lambda's
*subtree* (plus anything allocated during the walk); a captured outer
binding is not in the subtree and stays shared, which is exactly the
monomorphism a capture needs (`addk("a")` with `let k = 1` is
rejected). Instances alias the original so `once` counting and effect
propagation see one lambda. The emitter hooks:

- `LetBinding` whose only child is a `LambdaExpression` and whose
  binding is immutable (`uniqueness == "shared"`; the value
  restriction, a `let mut` lambda keeps one type): the child's walk is
  wrapped in `facade.record_scheme()` and the scheme is stored in a
  scope table parallel to the binding scopes, so shadowing a name
  without a lambda drops the scheme.
- `FunctionCall` and `Variable` whose name has a scheme: the callee /
  the use is `facade.instantiate(scheme, node_id)` instead of the
  original.

Acceptance as landed: `same(1); same("a")` compiles, the lambda's
principal type is `'a -> 'a` (it was `(Int ∧ String) -> (Int ∨
String)`), and the instances' types are fresh per use. The two call
*results* are not pinned: the flat solver unifies along flow edges and
fuses every statement of `main` into one representative, so their
principal types are the chain's, which is the part-4 prerequisite.

A note on why the biunifier still reads through the flat solver's
links: an attempt to decouple it (constrain on the stream with `unify`
classes collapsed by union-find, no `find()`) rendered simple programs
as `μt. μu. u ∧ ... -> ('a ∨ Int)`, because every `a <: b`, `b <: a`
pair and every statement-flow edge becomes a cycle the coalescer turns
into a μ-binder. The engine's simplifier does not remove those; the
paper's co-occurrence analysis would. That is part 4's work, not a
prerequisite for parts 1 and 2.

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
