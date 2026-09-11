# Types and inference

Metaxu is statically typed, and most programs barely show it. You
annotate function signatures; the checker works out everything else
from how values flow. The engine is SimpleSub-style subtype inference:
every expression contributes constraints, the solver connects them,
and a program is rejected only when some value would have to be two
incompatible things at once. That framing shapes the diagnostics, so
this chapter spends most of its time on rejections. They are where the
type system earns its keep.

## What you never have to write

`let` bindings carry no types. Neither do lambda parameters. The
checker reads both off the uses.

```metaxu
fn main() -> int {
    let base = 40;
    let bump = fn(x) -> x + 2;
    print(bump(base));
    0
}
```
```output
42
```

`base` is an int because `40` is one. `bump` gets the type
`fn(int) -> int` because its body adds `2` to the parameter. The one
place annotation is required is the signature of a named function:
signatures are the contract between functions, and inference fills in
the insides.

## Inference flows through structures

An empty `Vec.new()` has an unknown element type. The first `push`
pins it, and every later use is checked against that choice.

```metaxu
fn main() -> int {
    let @mut nums = Vec.new();
    nums.push(20);
    nums.push(22);
    print(nums[0] + nums[1]);
    0
}
```
```output
42
```

Generic structs work the same way. `Pair[T]` says nothing about `T`;
each literal decides for itself.

```metaxu
struct Pair[T] { a: T, b: T }

fn main() -> int {
    let ints = Pair { a: 20, b: 22 };
    let words = Pair { a: "me", b: "taxu" };
    print(ints.a + ints.b);
    print(words.a + words.b);
    0
}
```
```output
42
metaxu
```

One declaration, two instantiations, zero type arguments written: the
checker infers `Pair[int]` and `Pair[string]` from the fields.

## When values disagree

A conflict diagnostic names the two demands that clashed. Adding a
string to an int asks one value to satisfy both:

```metaxu error
fn main() -> int {
    print(1 + "a");
    0
}
```
```output
one value is required to be Int and String
```

Chapter 1 showed this program and pinned only the headline; this is
the sentence underneath. Notice it is not "expected int, found
string". The checker doesn't privilege one side of the flow: both
uses are real, they cannot both hold, and the message says so.

A condition must be a bool. Testing an int directly is a conflict,
not a truthiness coercion:

```metaxu error
fn main() -> int {
    let n = 3;
    let parity = if n { 1 } else { 0 };
    print(parity);
    0
}
```
```output
Bool and Int
```

Both arms of an `if` expression produce the same value, so they must
agree. Disagreeing arms are the same conflict in different clothes:

```metaxu error
fn main() -> int {
    let label = if 1 < 2 { "small" } else { 0 };
    print(label);
    0
}
```
```output
Int and String
```

The value in conflict is the result of the `if` itself: one arm
demands String, the other Int, and `label` can't be both.

## Declared types are enforced

Inference never overrides an annotation you wrote. A parameter
declared `int` is an int at every call site, and an argument that
disagrees is rejected there:

```metaxu error
fn half(n: int) -> int {
    n / 2
}

fn main() -> int {
    print(half("ten"));
    0
}
```
```output
Int and String
```

Callers are checked against what you declared, not against what the
body happens to tolerate.

## Generics are checked at instantiation

A generic struct's concrete fields still mean what they say. A
literal that fills one with the wrong type is caught even though the
struct as a whole is generic:

```metaxu error
struct Tagged[T] { tag: string, value: T }

fn main() -> int {
    let t = Tagged { tag: 7, value: true };
    print(t.value);
    0
}
```
```output
type mismatch for field 'tag'
```

Written type arguments are counted. `Box` takes one parameter, and
handing it two is an arity error, not a silent truncation:

```metaxu error
struct Box[T] { value: T }

fn open(b: Box[int, string]) -> int {
    0
}

fn main() -> int {
    print(open(Box { value: 3 }));
    0
}
```
```output
type argument
```

And within one instantiation, a type parameter is one type. `Pair[T]`
uses `T` for both fields, so a literal that supplies an int and a
string demands two different answers for the same `T`:

```metaxu error
struct Pair[T] { a: T, b: T }

fn main() -> int {
    let p = Pair { a: 1, b: "two" };
    print(p.a);
    0
}
```
```output
conflicting instantiations of type parameter T
```

If the fields should vary independently, that's a two-parameter
struct, `Pair[A, B]`.

## Bounds come from where clauses

A generic function can require capabilities of its type parameter
with a `where` clause. Chapter 7 covers traits; the inference-side
story is that the bound is checked per instantiation, at each call:

```metaxu error
trait Show {
    fn show(self) -> string;
}

fn describe[T](x: T) -> string where T: Show {
    x.show()
}

fn main() -> int {
    print(describe(3));
    0
}
```
```output
missing `implement Show for Int`
```

The call instantiates `T` to Int, the bound asks for `Show`, no such
block exists, and the diagnostic names the exact block that would fix
it. Without the where clause the same call would typecheck and fail
at runtime instead; chapter 7 shows that variant and why you'd
usually prefer this one.

## Names must exist

Not every rejection is a conflict. A name the checker never saw is
caught in the same pass, before anything runs:

```metaxu error
fn main() -> int {
    let total = 40 + 2;
    print(totl);
    0
}
```
```output
undefined variable 'totl'
```

Where a close spelling exists the diagnostic suggests it, the same
way it does for misspelled function names.

## Principal types, with a caveat

SimpleSub's design promise is principal types: every accepted program
has one most general type, and inference finds it. Metaxu's
implementation treats that as a tracked goal rather than a guarantee.
Guaranteed: accepted programs are sound at the inferred types, and
disagreements surface as the conflicts shown above. Advisory: that
the inferred type is always the most general one possible;
`compiler_roadmap.md` tracks the gap. You'll rarely notice it, and
when you do, the fix is ordinary: write the annotation you meant, and
the checker will hold everyone to it.
