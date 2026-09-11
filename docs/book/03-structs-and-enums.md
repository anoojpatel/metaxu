# Structs and enums

Metaxu has two ways to build a type: a struct holds all of its fields
at once, an enum holds exactly one of its variants at a time. Between
them and `match` (next chapter) you can model most data without ever
reaching for a null.

## Structs

A struct declares named, typed fields. Construction spells out every
field; access is `value.field`:

```metaxu
struct Point {
    x: int,
    y: int
}

fn manhattan(p: Point) -> int {
    let ax = if p.x < 0 { 0 - p.x } else { p.x };
    let ay = if p.y < 0 { 0 - p.y } else { p.y };
    ax + ay
}

fn main() -> int {
    let p = Point { x: 3, y: -4 };
    print(p.x);
    print(manhattan(p));
    0
}
```
```output
3
7
```

Field order in the literal doesn't matter; leaving a field out or
giving it the wrong type does. The checker names the field:

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

## Mutating fields

Bind with `@mut` and assign through the name. Structs nest, and a
field access chain reads or writes at any depth:

```metaxu
struct Point { x: int, y: int }

struct Segment {
    head: Point,
    tail: Point
}

fn main() -> int {
    let @mut s = Segment {
        head: Point { x: 0, y: 0 },
        tail: Point { x: 5, y: 5 }
    };
    s.tail.x = 10;
    print(s.tail.x);
    print(s.head.y);
    0
}
```
```output
10
0
```

## Field modes

A field can carry its own mode annotation, the same `@`-words that
appear on bindings and parameters:

```metaxu
struct Account {
    @const id: int,
    @mut balance: int
}

fn deposit(acct: @mut Account, amount: int) {
    acct.balance = acct.balance + amount;
}

fn main() -> int {
    let @mut a = Account { id: 7, balance: 100 };
    deposit(a, 50);
    print(a.balance);
    print(a.id);
    0
}
```
```output
150
7
```

`@const` marks a field callers shouldn't write, `@mut` one they may,
and `@local` pins a field to the stack, which is what stops a
heap-bound (`@global`) value from smuggling stack references out of
their scope. The rules that enforce all this, including checking
locality through nested fields, are chapter 10's subject; for now it's
enough that the annotations live in the type, so every function that
touches an `Account` sees the same contract.

## Generic structs

Type parameters go in angle brackets. You rarely write them at the
construction site; the field values pin them down:

```metaxu
struct Pair<T> {
    first: T,
    second: T
}

fn main() -> int {
    let words = Pair { first: "fore", second: "aft" };
    let nums = Pair { first: 1, second: 2 };
    print(words.first + words.second);
    print(nums.first + nums.second);
    0
}
```
```output
foreaft
3
```

Each use instantiates its own `Pair`: `Pair<string>` and `Pair<int>`
are unrelated types, and mixing them up is a compile error.

## Tuples

When the pair doesn't deserve a name, a tuple is a struct without one.
Destructure it with `let` or match on it:

```metaxu
fn divmod(a: int, b: int) -> (int, int) {
    (a / b, a % b)
}

fn main() -> int {
    let (q, r) = divmod(17, 5);
    print(q);
    print(r);
    0
}
```
```output
3
2
```

## Enums

An enum lists its variants. A variant can carry a payload of named,
typed fields, or nothing. You construct one by calling the variant like
a function (bare, or qualified as `Shape::Circle` when you want the
enum name in view), and take it apart with `match`:

```metaxu
enum Shape {
    Circle(radius: int),
    Rect(w: int, h: int),
    Empty
}

fn area(s: Shape) -> int {
    match s {
        Circle(r) => 3 * r * r,
        Rect(w, h) => w * h,
        Empty => 0
    }
}

fn main() -> int {
    print(area(Shape::Circle(2)));
    print(area(Rect(3, 4)));
    print(area(Empty));
    0
}
```
```output
12
12
0
```

The payload is only reachable through a pattern. There's no
`s.radius`: a `Shape` might be a `Rect`, and the checker won't let you
assume otherwise.

## Enums that carry other enums

Variants nest, and enums take type parameters just like structs. Both
at once gives you recursive data:

```metaxu
enum Tree<T> {
    Leaf(value: T),
    Node(left: Tree<T>, right: Tree<T>)
}

fn total(t: Tree<int>) -> int {
    match t {
        Leaf(v) => v,
        Node(l, r) => total(l) + total(r)
    }
}

fn main() -> int {
    let t = Node(Leaf(1), Node(Leaf(2), Leaf(3)));
    print(total(t));
    0
}
```
```output
6
```

## Option

The one-of-two-shapes pattern you'll write most is "a value, or
nothing", so Metaxu builds it in: `Option` with variants `Some(x)` and
`None`. It's an ordinary enum, just predeclared; you could define your
own in three lines, and everything in `std/option.mx` is written
against it with plain `match`:

```metaxu
fn find_first_even(a: int, b: int, c: int) -> Option {
    if a % 2 == 0 { return Some(a); }
    if b % 2 == 0 { return Some(b); }
    if c % 2 == 0 { return Some(c); }
    None
}

fn main() -> int {
    match find_first_even(3, 5, 8) {
        Some(n) => print(f"found {n}"),
        None => print("none even")
    }
    0
}
```
```output
found 8
```

`Result` (variants `Ok`/`Err`) is predeclared the same way; chapter 9
covers both properly, along with `std.option` and `std.result`.

Next: everything `match` can do.
