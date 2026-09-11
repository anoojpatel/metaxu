# Structs and enums

Two ways to build a type: a struct holds several values at once (all
the fields, every time), an enum holds one of several alternatives
(exactly one variant at a time). Product and sum. Everything else in
Metaxu's data modeling, including `Option` and the trees in this
chapter, is these two plus generics.

## Structs

Declare the fields with their types; construct with a literal that
names every field; read with `.`:

```metaxu
struct Point { x: int, y: int }

fn main() -> int {
    let p = Point { x: 3, y: 4 };
    print(p.x + p.y);
    0
}
```
```output
7
```

Field types are checked at the literal, not discovered at runtime:

```metaxu error
struct Point { x: int, y: int }

fn main() -> int {
    let p = Point { x: "three", y: 4 };
    0
}
```
```output
type mismatch for field 'x' of Point
```

One naming caution: `from` is a reserved word (it belongs to import
syntax), so it can't be a field name. A `Segment` goes `a` to `b`, or
`start` to `stop`, not `from` to `to`.

## Mutating fields

Field assignment needs a mutable binding, and it reaches through
nested structs: a dotted path on the left of `=` works to any depth.

```metaxu
struct Point { x: int, y: int }
struct Segment { a: Point, b: Point }

fn main() -> int {
    let mut s = Segment { a: Point { x: 0, y: 0 }, b: Point { x: 4, y: 0 } };
    s.b.y = 3;
    print(s.b.x + s.b.y);
    0
}
```
```output
7
```

Fields can also carry modes in the declaration, like
`@const name: string` or `@mut n: int`. A `@const` field rejects
assignment at compile time no matter how mutable the binding is.
That's chapter 10's territory; here it's enough to know the syntax
exists.

## Structs in and out of functions

Functions take and return structs like any other value. Building a
new struct from an old one is the plain way to express an update:

```metaxu
struct Point { x: int, y: int }

fn shift(p: Point, dx: int) -> Point {
    Point { x: p.x + dx, y: p.y }
}

fn main() -> int {
    let p = Point { x: 1, y: 2 };
    let q = shift(p, 9);
    print(q.x);
    print(q.y);
    0
}
```
```output
10
2
```

## Generic structs

A type parameter in square brackets makes the struct generic. The
literal doesn't repeat the type; the checker infers it from the
fields, and both fields have to agree on what `T` is (chapter 6 shows
the rejection when they don't).

```metaxu
struct Pair[T] { a: T, b: T }

fn main() -> int {
    let ints = Pair { a: 1, b: 2 };
    let words = Pair { a: "left", b: "right" };
    print(ints.a + ints.b);
    print(words.b);
    0
}
```
```output
3
right
```

## Tuples

For a quick aggregate that doesn't deserve a name, a tuple groups
values positionally. A `let` with a tuple pattern takes it apart
again:

```metaxu
fn min_max(a: int, b: int) -> (int, int) {
    if a < b { (a, b) } else { (b, a) }
}

fn main() -> int {
    let (lo, hi) = min_max(9, 4);
    print(lo);
    print(hi);
    0
}
```
```output
4
9
```

## Enums

An enum lists its variants. A variant can be bare, like `Dot`, or
carry a payload, like `Circle(int)`. Construction is qualified with
`::`, and a payload-less variant is written bare, no parentheses,
in both construction and pattern position. Consuming an enum means
matching on it; chapter 4 is all about `match`, so this example uses
only the basics.

```metaxu
enum Shape {
    Dot,
    Circle(int),
    Rect(int, int)
}

fn area_ish(s: Shape) -> int {
    match s {
        Shape::Dot => 0
        Shape::Circle(r) => 3 * r * r
        Shape::Rect(w, h) => w * h
    }
}

fn main() -> int {
    print(area_ish(Shape::Dot));
    print(area_ish(Shape::Circle(2)));
    print(area_ish(Shape::Rect(2, 5)));
    0
}
```
```output
0
12
10
```

The value knows which variant it is; there's no way to read a
`Circle`'s radius without going through a pattern that proves you
have a `Circle`. That proof obligation is the point of sum types.

## Recursive generic enums

Variants can hold the enum's own type, which is how linked structures
are declared. No pointers, no lifetimes, just the type mentioning
itself:

```metaxu
enum Tree[T] {
    Leaf,
    Node(Tree[T], T, Tree[T])
}

fn tree_sum(t: Tree[int]) -> int {
    match t {
        Tree::Leaf => 0
        Tree::Node(l, v, r) => tree_sum(l) + v + tree_sum(r)
    }
}

fn main() -> int {
    let t = Tree::Node(
        Tree::Node(Tree::Leaf, 1, Tree::Leaf),
        2,
        Tree::Node(Tree::Leaf, 3, Tree::Leaf));
    print(tree_sum(t));
    0
}
```
```output
6
```

## Option is just an enum

The standard library's `Option` is an enum with variants `Some(T)`
and `None`, and its variants are in scope unqualified. Printing one
shows the qualified form.

```metaxu
fn describe(o: Option[int]) -> string {
    match o {
        Some(v) => "got " + v.to_string()
        None => "nothing"
    }
}

fn main() -> int {
    print(describe(Some(21)));
    print(describe(None));
    print(Some(3));
    0
}
```
```output
got 21
nothing
Option::Some(3)
```

`Option` replaces every "might not be there" convention: no null, no
sentinel values. Chapter 9 tours the combinators in `std.option` that
make it pleasant at scale. First, though, patterns deserve a full
chapter.
