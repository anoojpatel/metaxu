# Type inference: improvement plan

Companion to book chapter 18, which describes the checker as it is.
This document is what to change, in priority order, with the algorithm
for each change and the tests that pin it. Everything here is designed
for the checker's actual shape: a walk over the frozen tree that emits
tagged constraints into a buffer, then one solve.

## Priorities

1. **Let-polymorphism for lambda values.** `let same = fn(x) -> x;`
   is usable at one type today (chapter 18 pins the rejection). Fix by
   generalizing at `let` when the right-hand side is syntactically a
   lambda, and instantiating at every use. This is the most
   user-visible gap and the cheapest to close.
2. **Conflict diagnostics with two locations.** "one value is required
   to be Int and String" should say where each requirement came from,
   with an excerpt for each. All the information exists (class
   constraints carry node ids); only the report is missing.
3. **A real subtyping core.** Today `subtype(a, b)` is solved as
   equality (a variable's upper bound is its union-find link), which is
   sound but leaves no room for effect rows, width subtyping, or
   covariant reads of `Vec[T]`. Replace with biunification over lower
   and upper bound sets, with class requirements propagated along
   bounds. No visible change on day one; the foundation for the rest.
4. **Constructor variance from definitions.** Record inferred variance
   on user constructors so `Pair[T]` is covariant when `T` appears only
   in positive positions. Depends on 3.

## 1. Let-polymorphism by constraint replay

### Why not textbook generalization

SimpleSub and HM generalize eagerly: solve the right-hand side, then
quantify the variables that don't escape. Our emitter never solves
mid-walk; it buffers everything and solves once. Rather than rewrite
the emitter around eager unification, generalize by *recording* the
constraints a lambda emits and *replaying* them, under a fresh
substitution, at each use. This keeps one solve, keeps the borrow and
mode checker's single walk untouched, and is local to three places in
the emitter.

### Value restriction

Only a `LetBinding` whose value child is a `LambdaExpression` gets a
scheme. Calls, struct literals, and everything else stay monomorphic.
(A named `fn` is already instantiated per call in the v1 emitter; if
it isn't, the same recorder applies to `FunctionDeclaration`.)

### Recording

Wrap the lambda walk:

    with facade.record_scheme() as rec:
        ...existing LambdaExpression handling...
    schemes[binding_name] = rec.finish(fn_compact)

`record_scheme` snapshots the fresh-id counter on entry and exit.
Every constraint emitted between the two is appended to the recording.
The *generic set* `G` is every variable id allocated inside the window.
Types bound outside the lambda have smaller ids and are therefore
never in `G`; a captured outer binding stays shared across all
instantiations, which is exactly the monomorphism a capture needs.

### Instantiation

At a `Variable` use whose binding has a scheme (call position or
first-class use alike), instead of `add_unify(node_ty, binding_ty)`:

    inst = facade.instantiate(scheme, use_node_id)
    facade.add_unify(node_ty, inst.fn_type)

`instantiate` builds a substitution `σ: G -> fresh vars`, deep-copies
the fn type under `σ` (function, constructor, recursive nodes; vars in
`G` map through `σ`, other vars are shared), and replays the recorded
`unify`, `subtype`, and `class` constraints under `σ`, skipping any
that mention no variable in `G` (they are already in the graph).
`function`, `linearity`, `capture`, and `effect` constraints are not
replayed; they describe the lambda itself. Instead the copy is
registered as an alias of the original, and the checker resolves
aliases before counting calls (so a `once` lambda instantiated twice
and called once each is still two calls of one `once` value) and
before propagating effects.

Replayed constraints carry `instance_of=use_node_id` so a conflict
inside an instance can blame the use site as well as the body.

### Tests

End-to-end, in `test_inference_cases.py` (harness style, xfail until
this lands):

- identity lambda used at Int and String prints both;
- a lambda that captures an outer Int and is applied to a String is
  still rejected ("Int and String");
- first-class use (`let g = same;`) instantiates too;
- a `once` lambda bound with `let` and called twice is still rejected.

Core, in `test_generalize_core.py` (runs on any tree): recording
captures constraints and the generic set; instantiation shares
non-generic variables; two instances resolve to two different
primitives with no solver error; without instantiation the same
program records a unify failure.

## 2. Conflict diagnostics with provenance

The detector already reads, per representative, the set of class
requirements. Each `class` constraint carries the node id that emitted
it. Report:

    <mem>:3:11: one value is required to be Int and String
        let n = 1 + "a";
                ^
    Int comes from here:
        let n = 1 + "a";
                ^
    String comes from here:
        let n = 1 + "a";
                    ^

Rules: the headline location is the merge point (the operator, the
`if`, the call); each requirement's location is the node id on its
class constraint, resolved through the frozen span table; when an
instance replayed the constraint, add "in this use of `same`:" with
the `instance_of` location. Pin the full text in the book (chapter 6
pins only the fragment today).

## 3. Biunification core

Replace `unify(var, t)` (sets `upper_bound = t`, treated as equality)
with `constrain(lhs, rhs)`:

- `var <: t`: add `t` to `var.upper`; for each `l` in `var.lower`,
  `constrain(l, t)`.
- `t <: var`: add `t` to `var.lower`; for each `u` in `var.upper`,
  `constrain(t, u)`.
- function vs function: parameters flipped, result forward.
- constructor vs constructor: same name and arity; arguments under
  the constructor's recorded variance (invariant means both
  directions).
- primitive vs primitive: names must match.
- a `(lhs, rhs)` pair already in the cache is skipped (cycles).

Class requirements attach to variables and propagate along bounds in
both directions; the conflict detector reads them off the bound
closure instead of the union-find representative. Keep `unify` for the
undirected edges (`let` binding and use, operator operands) as
`constrain` in both directions.

Migration: land behind a flag, run the whole suite and the book both
ways, then flip. Expected behavioral change: none at the language
surface; chapter 6's diagnostics must come out byte-identical, which
is the acceptance test.

## 4. Constructor variance

`VarianceInferencer` already records positions and infers a variance
per type parameter; `get_constructor_variances` just never sees the
result because nothing writes `variances` onto the constructor. Wire
`finalize_type_definition` into struct and enum declaration handling,
store the list on the `TypeConstructor`, and `constrain` will use it.
A `Pair[T]` with `T` only in fields becomes covariant; a struct with a
`@mut` field of type `T` stays invariant (a write is a negative
position). Needs 3 to mean anything.

## Not planned

Union and intersection types in surface syntax; nominal struct
extension; structural width subtyping for records. Chapter 18 lists
them as design-document sketches, and nothing above depends on them.
