# Types and inference

Metaxu's checker is built on SimpleSub, an inference algorithm in the
algebraic-subtyping family. The practical consequence: you write types
on `fn` signatures when you want a checked boundary, and almost nowhere
else. `let` bindings never need annotations, lambda parameters usually
don't, and whole stdlib modules (`std/option.mx`, `std/sort.mx`) leave
most parameters bare.

```metaxu
fn largest(a: int, b: int) -> int {
    if a > b { a } else { b }
}

fn main() -> int {
    let x = 40;
    let name = "answer";
    let bigger = largest(x + 2, 2);
    print(name + ": " + bigger.to_string());
    0
}
```
```output
answer: 42
```

`x` is an int because `40` is, and because `largest` demands one.
Nothing about that needed saying twice.

## How the checker decides

Every expression starts as an inference variable. Uses add constraints:
a literal classes its variable as Int or String or Bool or Float, an
operator ties its operands together, a call ties arguments to
parameters, both arms of an `if` flow into the same result. The checker
then looks for a variable that has been forced into two classes at
once. That is a compile error, reported at the value that made the
conflict apparent:

```metaxu error
fn main() -> int {
    let n = 1 + "a";
    0
}
```
```output
one value is required to be Int and String
```

A condition must be a bool. There is no truthiness:

```metaxu error
fn main() -> int {
    if 1 { print("yes") } else { print("no") };
    0
}
```
```output
one value is required to be Bool and Int
```

And an `if` used as a value must produce one type, whichever arm runs:

```metaxu error
fn main() -> int {
    let x = if true { 1 } else { "one" };
    0
}
```
```output
one value is required to be Int and String
```

Annotations you do write are enforced the same way. Declaring
`fn add(a: int, b: int)` turns every call site into a checked boundary:

```metaxu error
fn add(a: int, b: int) -> int { a + b }

fn main() -> int {
    let x = add(1, "two");
    0
}
```
```output
call of add: argument has type String, expected Int
```

## Generics

Type declarations introduce parameters with angle brackets; uses apply
them with angle brackets or square brackets (`Pair<int>` and
`Pair[int]` mean the same thing). A generic function gets a fresh
instantiation at every call site, so one program can use it at several
types:

```metaxu
fn identity<T>(x: T) -> T { x }

struct Pair<T> { x: T, y: T }

fn main() -> int {
    let n = identity(41);
    let s = identity("forty-one");
    let ints = Pair<int> { x: 3, y: 4 };
    let words = Pair { x: "left", y: "right" };
    print(s);
    print(n + 1);
    print(ints.x + ints.y);
    print(words.x);
    0
}
```
```output
forty-one
42
7
left
```

`Pair { x: "left", y: "right" }` shows inference at work on a struct:
no type arguments, `T` comes from the fields.

Explicit instantiations are checked, not trusted. Naming the type
argument and then contradicting it is an error:

```metaxu error
struct Pair<T> { x: T, y: T }

fn main() -> int {
    let p = Pair<int> { x: 1, y: "nope" };
    0
}
```
```output
field 'y' of Pair
```

So is passing the wrong number of type arguments:

```metaxu error
fn identity<T>(x: T) -> T { x }

fn main() -> int {
    let i = identity<int, string>(1);
    0
}
```
```output
wrong number of type arguments for identity
```

When you write no type arguments at all, the arguments themselves must
agree on what `T` is:

```metaxu error
fn first<T>(a: T, b: T) -> T { a }

fn main() -> int {
    let x = first(1, "s");
    0
}
```
```output
conflicting instantiations of type parameter T
```

## Trait bounds

A `where` clause (or the inline form `fn f<T: Show>`) restricts a type
parameter to types with an impl. Chapter 7 covers traits themselves;
what matters here is that bounds are checked at the call site, whenever
the instantiation is statically known:

```metaxu
trait Show { fn show(self) -> string }

struct Point { x: int, y: int }

implement Show for Point {
    fn show(self) -> string {
        "(" + self.x.to_string() + ", " + self.y.to_string() + ")"
    }
}

fn describe<T>(x: T) -> string where T: Show { x.show() }

fn main() -> int {
    print(describe(Point { x: 1, y: 2 }));
    0
}
```
```output
(1, 2)
```

Violating a bound names the missing impl, which is also the fix:

```metaxu error
trait Show { fn show(self) -> string }

fn describe<T>(x: T) -> string where T: Show { x.show() }

fn main() -> int {
    print(describe(1));
    0
}
```
```output
missing `implement Show for Int`
```

## What stays permissive, and the honest status

Instantiation checking fires where types are statically known.
A value whose type the checker cannot determine (say, the result of
another generic call it could not resolve) passes through unchecked
rather than producing a false positive; runtime dispatch still enforces
it. Unannotated parameters are checked by how they're used, not by a
declaration they don't have. The design bias is stated in the compiler
docs and worth repeating: a clear error where the checker is sure,
permissiveness where it is not, and never a silently wrong answer.

One more status note. A full biunification engine with
principal-type coalescing (union and intersection types, the SimpleSub
simplification passes) exists in the compiler (`simplesub.py`), but its
results are advisory: nothing in HIR or codegen consumes them yet, and
the hard diagnostics you saw above all come from the constraint-graph
conflict detector, which is deliberately coarser and points at source
locations well. Principal types on statement-heavy functions
over-approximate. If you ask the pipeline what the principal type of an
expression chain is you get a real answer; the language does not yet
act on it. Chapter 18 opens that machinery: the flow graph, the
class-constraint algebra, the flat solver that decides what compiles,
and the biunifier that computes those principal types.

Undefined names are part of the same compile-time contract. A name that
resolves to nothing is an error, never a silently dropped expression:

```metaxu error
fn main() -> int {
    print(total);
    0
}
```
```output
undefined variable 'total'
```
