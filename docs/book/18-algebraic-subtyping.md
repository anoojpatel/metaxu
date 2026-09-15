# Algebraic subtyping

Chapter 6 showed what the checker says. This chapter is how it decides.
Metaxu's inference descends from Dolan's algebraic subtyping and
Parreaux's SimpleSub, the simplified algorithm for it: types are not
guessed and compared, they are collected as flows between the places a
value is produced and the places it is consumed, and a program is
rejected only when some single value would have to be two incompatible
things. This chapter names the pieces of that machinery as they exist
in the compiler today, and is explicit about which parts of the
algebraic-subtyping story Metaxu does not yet do.

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
other.

Direction is what makes this subtyping rather than plain unification.
Positions that produce values are *positive*; positions that consume
them are *negative*. A function's return type is positive and its
parameters are negative, and going under a parameter flips the sign of
everything inside it. The checker tracks this as a polarity on each
constraint, and the solver treats a positive constraint as covariant,
a negative one as contravariant, and a directionless one as invariant.

You feel the direction when a function is handed a function:

```metaxu error
fn apply(f: fn(int) -> int) -> int {
    f(20)
}

fn main() -> int {
    print(apply(fn(s) -> s + "!"));
    0
}
```
```output
Int and String
```

`f`'s parameter is a negative position, so `20` flows *into* the
lambda's `s`. The lambda's body then adds a string to `s`, which
requires `s` to be String. One variable, two requirements, and the
diagnostic names both. Nothing about `apply` is wrong; the conflict is
in the value that would have to pass through it.

## The algebra of requirements

Alongside the edges, the walk attaches *class constraints* to
variables. These are the algebra in algebraic subtyping: not types,
but requirements a type must satisfy, and they compose by set union as
variables are merged.

- A literal contributes its primitive: Int, Float, String, Bool, or
  Unit.
- `+ - * /` require Number of their result; `< <= > >=` require Ord
  of their operands; `== !=` require Eq. Every comparison's result is
  Bool.
- The condition of an `if` or `while` must be Bool. A `while` is Unit.
- A struct literal carries Struct and `Struct:Name`; each field
  carries `Field:name`; reading `x.f` requires HasField of `x`.
- Parameters carry Param plus any declared mode, as `Mode:once` and
  the like; lambdas and named functions carry Callable; calls carry
  CallLinearity; borrows, moves, and `exclave` each leave their own
  mark for the mode checker.
- A function with a `performs` clause carries Effectful.

After the edges are solved, every variable has a representative, and
the classes of everything merged into that representative are read
together. Two different primitives on one representative is the
conflict chapter 6 pinned as "one value is required to be Int and
String". A comparison shows the same mechanism through Ord:

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

## The solver

The solver is a unifier that records bounds. A type variable holds a
pair of bounds, lower and upper; when a variable is unified with
anything, that thing becomes its upper bound, and finding a variable's
representative means following upper bounds to the end, compressing
the path as it goes. The bound *is* the union-find link. Beyond
variables the cases are structural: primitives unify when their names
match; constructors when their names and arities match and their
arguments unify under the composed variance; function types with
parameters contravariant and results covariant; effect types when
their operation sets match, each operation's parameters contravariant
and its result covariant. An occurs check refuses to build an infinite
type.

Variance composes by a small table: invariant absorbs everything,
equal signs give covariant, different signs give contravariant. Type
constructors carry a variance per parameter; unless the definition's
analysis recorded one, the default is invariant, which is why a
`Vec[int]` is exactly a `Vec[int]` and not a subtype of anything
wider.

A failed constraint does not stop the solve. It is recorded and the
walk continues, so one mistake cannot hide the others behind it; the
diagnostics you get are the whole list.

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

Generic functions instantiate the same way generic structs do, per
call, with the parameter unified against whatever flows in:

```metaxu
struct Pair[T] { a: T, b: T }

fn flip[T](p: Pair[T]) -> Pair[T] {
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

One thing generalization does *not* reach today is `let`. A named
`fn` is instantiated fresh at each call, but a lambda bound with `let`
has one type variable for its whole life: use it at two types and the
two uses collide.

```metaxu error
fn main() -> int {
    let same = fn(x) -> x;
    print(same(1));
    print(same("a"));
    0
}
```
```output
Int and String
```

The generalization step that would turn `same` into a type scheme
exists in the solver's code and is not yet wired into the walk. Name
the function with `fn` and it is polymorphic; bind it with `let` and
it is one type. That is also the honest form of chapter 6's caveat on
principal types: the inferred type is sound, and it is the most
general one the *current* walk can produce, which is narrower than
what algebraic subtyping promises.

## Effects and linearity ride the same graph

The flow graph carries more than types. When a function declares
`performs Ask`, the effect name is attached to its function type. A
call propagates the callee's effects to the call's result; a flow edge
propagates them from the producing side to the consuming side; a
function whose return type carries an effect carries it too. Any
function type that ends up with effects on it is marked as one that
may suspend, and that mark is what later stages use to decide which
functions need a continuation frame. The propagation is why the
checker knows `doubled` suspends even though it never performs
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

Built, and load-bearing on every compile: a fresh variable per node;
directed flow edges and undirected unify edges; polarity tracked per
constraint and composed through function and constructor positions;
variance inferred for type parameters; the class-constraint algebra
with its conflict detector; bounds-recording unification with an
occurs check; single-step unfolding of recursive types; effect and
linearity propagation over the solved graph.

Present in the source but not on the path: a type-class solver with
functional dependencies (improvement and instance resolution to a
fixpoint) that nothing calls yet; union and intersection type nodes
that no rule constructs, so the solver never produces MLsub-style
`Int | String` or polar type simplification; and the `let`
generalization above. The older design document also describes
nominal struct extension (`struct Dog: Animal`), structural width
subtyping for records and variants, and interfaces. None of those are
parsed or checked today. `Number`, `Ord`, and `Eq` are requirements
a primitive can satisfy, not supertypes: Int and Float are distinct,
and there is no numeric widening.

If you want to see the machinery run, chapter 17 shows how to hold the
pipeline open from Python; the tables it builds are the variables and
constraints this chapter described.
