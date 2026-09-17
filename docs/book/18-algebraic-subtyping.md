# Algebraic subtyping

Chapter 6 showed what the checker says. This chapter is how it decides.
Metaxu's inference descends from Dolan's algebraic subtyping and
Parreaux's SimpleSub, the simplified algorithm for it: types are not
guessed and compared, they are collected as flows between the places a
value is produced and the places it is consumed, and a program is
rejected when some single value would have to be two incompatible
things. Two engines implement that idea in the compiler, and this
chapter is precise about which one does what. The *flat solver* is on
the compile path: it decides what compiles. The *biunifier* is a full
SimpleSub engine that computes principal types on request; it is
advisory today, and the last sections say exactly what that means.

## Every expression is a type variable

The checker does not work on source text. It works on the frozen
syntax tree, and the first thing it does is hand every node in that
tree its own fresh type variable. A literal, a `let`, an `if`, a call,
a whole block: each gets one. Nothing is known about any of them yet;
knowledge arrives as constraints.

```metaxu
fn pick(flag: bool) -> int {
    if flag { 40 } else { 2 }
}

fn main() -> int {
    print(pick(true) + pick(false));
    0
}
```
```output
42
```

Walk `pick` the way the checker does. `40` and `2` are variables that
each carry the requirement Int. Both arms flow into the `if` node; the
`if` is the function body's tail, so it flows into the function's
return type; the return type is declared `int`. Four variables, three
flows, one declared anchor, and everything resolves.

## Flows have a direction

Two kinds of edge come out of the walk. A *flow*, written `a <: b`,
says a value produced at `a` is consumed at `b`: each child expression
flows into the expression that contains it, a block's tail flows into
the block, a returned expression flows into its function's return
type. A *unify* edge says two variables must be the same type, with no
direction: a variable use and its binding, the two sides of `+`, an
assignment's target and its value, the arms of an `if` with each
other. A third kind, the *call* edge, connects a callee's type to the
argument and result types at one call site.

Direction is what makes this subtyping rather than plain unification.
Positions that produce values are *positive*; positions that consume
them are *negative*. A function's return type is positive and its
parameters are negative, and going under a parameter flips the sign of
everything inside it. When a function is handed a function, the
argument flows *into* the lambda's parameter:

```metaxu
fn apply(f: fn(int) -> int) -> int {
    f(20)
}

fn main() -> int {
    print(apply(fn(s) -> s * 2));
    0
}
```
```output
40
```

`f`'s parameter is a negative position, so `20` reaches the lambda's
`s` through the call edge inside `apply`, and `s * 2` is an Int. One
honest limit belongs right here: change the body to `s + "!"` and the
compile path accepts the program, because call edges are not merged
by the conflict detector described next; the mismatch surfaces only
when the interpreter tries to add an Int to a String. The biunifier
sees the clash statically. Closing that gap on the compile path is
tracked in `docs/type_inference_plan.md`.

## The algebra of requirements

Alongside the edges, the walk attaches *class constraints* to
variables. These are the algebra in algebraic subtyping: not types,
but requirements a type must satisfy, and they compose by set union as
variables are merged.

- A literal contributes its primitive: Int, Float, String, Bool, or
  Unit.
- `+ - * / %` require Number of their result; `< <= > >=` require Ord
  of their operands; `== !=` require Eq. Every comparison's result is
  Bool. The bitwise operators require Int of everything they touch.
- The condition of an `if` or `while` must be Bool. A `while` is Unit.
- A struct literal carries Struct and `Struct:Name`; each field
  carries `Field:name`; reading `x.f` requires HasField of `x`.
- Parameters carry Param plus any declared mode, as `Mode:once` and
  the like; lambdas and named functions carry Callable; calls carry
  CallLinearity; borrows, moves, and `exclave` each leave their own
  mark for the mode checker.
- A function with a `performs` clause carries Effectful.

After the walk, the conflict detector takes the four mutually
exclusive literal classes (Int, String, Bool, Float), unions variables
along *unify* edges only, and reads the classes of each component
together. Two different primitives on one component is the rejection
chapter 6 pinned as "one value is required to be Int and String". Flow
edges are deliberately not merged (they are directional, and every
statement flows into its block), and neither are call edges; that is
the source of the limit noted above. The diagnostic points at the
last of the conflicting values in source order, the one that made the
conflict apparent:

```metaxu error
fn main() -> int {
    let small = 1 < "a";
    print(small);
    0
}
```
```output
Int and String
```

The two operands are unified, `1` brings Int, `"a"` brings String, and
the merged variable cannot be both. Ord never even gets consulted; the
primitive clash is found first.

## The flat solver

The solver on the compile path is a unifier that records a bound. A
type variable holds a pair of bounds, lower and upper; when a variable
is unified with anything, that thing becomes its upper bound, and
finding a variable's representative means following upper bounds to
the end, compressing the path as it goes. The bound *is* the
union-find link. Beyond variables the cases are structural:
primitives unify when their names match; constructors when their
names and arities match and their arguments unify under the composed
variance; function types with parameters contravariant and results
covariant; effect types when their operation sets match. An occurs
check refuses to build an infinite type.

Variance composes by a small table: invariant absorbs everything,
equal signs give covariant, different signs give contravariant. Type
constructors carry a variance per parameter; unless the definition's
analysis recorded one, the default is invariant, which is why a
`Vec[int]` is exactly a `Vec[int]` and not a subtype of anything
wider.

A failed constraint does not stop the solve. It is recorded and the
walk continues, so one mistake cannot hide the others behind it.

Recursive types are handled by unfolding, once, per constraint: the
recursive occurrence is replaced by one copy of its definition with
fresh parameters, which is enough to check any single use without
looping.

```metaxu
enum Chain {
    End,
    Link(int, Chain)
}

fn total(c: Chain) -> int {
    match c {
        Chain::End => 0
        Chain::Link(v, rest) => v + total(rest)
    }
}

fn main() -> int {
    print(total(Chain::Link(20, Chain::Link(22, Chain::End))));
    0
}
```
```output
42
```

## Variance is inferred, never written

A type parameter's variance comes from where it appears. The checker
records every position a parameter occupies in its definition as
positive, negative, or neutral: only positive positions makes it
covariant, only negative makes it contravariant, both or neither makes
it invariant. There is no annotation syntax for this. Older design
notes sketch `+T` and `-T`; the parser does not accept them, and you
do not need them.

Declarations introduce parameters in angle brackets and applications
supply them in square brackets: `struct Pair<T>` declares, `Pair[int]`
applies. Generic functions instantiate per call, with the parameter
unified against whatever flows in:

```metaxu
struct Pair<T> { a: T, b: T }

fn flip<T>(p: Pair[T]) -> Pair[T] {
    Pair { a: p.b, b: p.a }
}

fn main() -> int {
    let f = flip(Pair { a: 1, b: 41 });
    print(f.a - f.b);
    let w = flip(Pair { a: "x", b: "y" });
    print(w.a + w.b);
    0
}
```
```output
40
yx
```

## Where generalization stands

A lambda bound with `let` and used at two types compiles today:

```metaxu
fn main() -> int {
    let same = fn(x) -> x;
    print(same(1));
    print(same("a"));
    0
}
```
```output
1
a
```

It compiles for a reason worth understanding, not because `same` was
generalized. Each call connects to the lambda through a call edge, and
call edges are not merged by the conflict detector, so Int and String
never meet in one component. The lambda has a single type variable for
its whole life; ask the biunifier for `same`'s principal type and it
answers `Int ∨ String`, and for the lambda as a function,
`(Int ∧ String) -> (Int ∨ String)`: one function that has absorbed
both uses rather than two instantiations of a polymorphic one. That is
the precise form of chapter 6's caveat on principal types. Named
generic functions such as `flip<T>` above are instantiated per call
by the emitter; `let`-bound lambdas are not, yet. The biunifier
already has the machinery, levels with extrusion, `generalize` and
`instantiate`, exercised at the engine level; putting it on the
compile path is the first item in the inference plan.

## The biunifier

The second engine is the SimpleSub algorithm proper. `constrain(a, b)`
records `a <: b` by decomposing structurally, keeps *lists* of lower
and upper bounds per variable in side tables (never touching the flat
solver's union-find link), propagates each new bound against the
bounds already present, and terminates on cyclic constraint graphs
through a processed-pair cache. A variable constrained against a type
from an inner scope is *extruded*: copied down to the outer level with
fresh variables linked back, so nothing leaks. Coalescing then turns
the solved graph into a principal type: positive variables union with
their lower bounds, negative ones intersect with their upper bounds,
re-entering a variable yields a `μ` binder, and the paper's
simplification passes remove polar-only variables and flatten.

Everything the compile path emits is fed to it: unify edges as both
directions, flow edges as one, the literal classes as bounds, and call
edges as `callee <: (args) -> result`. Its primitive lattice is flat
(no `Int <: Float`), so it reports exactly the primitive clashes the
flat solver would, plus the ones that hide behind call edges. Querying
it never changes what compiles; the tests pin that the MIR of a
program is byte-identical whether or not its principal types were
asked for.

## Effects and linearity ride the same graph

The flow graph carries more than types. When a function declares
`performs Ask`, the effect name is attached to its function type. A
call propagates the callee's effects to the call's result; a flow edge
propagates them from the producing side to the consuming side; a
function whose return type carries an effect carries it too. A
function type that ends up with a suspend-class effect on it is marked
as one that may suspend, and that mark is what later stages use to
decide which functions need a continuation frame. The propagation is
why the checker knows `doubled` suspends even though it never performs
anything itself:

```metaxu
effect Ask {
    ask() -> int
}

fn answer() -> int performs Ask {
    perform Ask.ask()
}

fn doubled() -> int performs Ask {
    answer() * 2
}

fn main() -> int {
    let n = handle Ask with {
        ask() -> resume(21)
    } in {
        doubled()
    };
    print(n);
    0
}
```
```output
42
```

Linearity travels the same way. Every function type carries `once`,
`separate`, or `many`. A lambda that captures something mutably is
forced to at least `separate`; a lambda declared `many` with a mutable
capture is rejected; a `once` callable invoked twice is rejected.
Chapter 10 is where those rules are stated from the user's side; here
they are just more requirements on the same variables, checked in the
same pass, from the same walk.

## What is built, and what is only sketched

On the compile path, load-bearing on every build: a fresh variable per
node; flow, unify, and call edges; the class-constraint algebra with
its conflict detector over unify components; the flat solver with an
occurs check and single-step unfolding of recursive types; variance
inferred for type parameters; effect and linearity propagation over
the solved graph.

Built and tested at the engine level, advisory on the path: the
biunifier, with extrusion, levels, generalization, instantiation,
coalescence, and simplification into principal types.

Not built: generalization of `let`-bound lambdas on the compile path;
merging of call edges into conflict detection (the `s + "!"` gap); a
type-class solver with functional dependencies (present in the source,
never called); union and intersection types in surface syntax; the
older design document's nominal struct extension, structural width
subtyping for records and variants, and interfaces. `Number`, `Ord`,
and `Eq` are requirements a primitive can satisfy, not supertypes: Int
and Float are distinct, and there is no numeric widening.

If you want to see the machinery run, chapter 17 shows how to hold the
pipeline open from Python; `ctx.tables.facade.principal_type_of(node)`
is the biunifier's answer for any node, and the tables it builds are
the variables and constraints this chapter described.
