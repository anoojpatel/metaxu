# Pattern matching

`match` takes a value and a list of arms. Each arm pairs a pattern
with a body; the first pattern that fits runs its body, and the whole
`match` is an expression whose value is that body's value. The
compiler checks that the arms cover every possible value, so "I
forgot a case" is a compile error, not a 2am page.

## Matching an enum

Payload-less variants match under their qualified name, bare, no
parentheses. Arms are separated by newlines, same as handler arms in
chapter 8.

```metaxu
enum Color { Red, Green, Blue }

fn name(c: Color) -> string {
    match c {
        Color::Red => "red"
        Color::Green => "green"
        Color::Blue => "blue"
    }
}

fn main() -> int {
    print(name(Color::Green));
    0
}
```
```output
green
```

## Literals and binders

Patterns aren't only variants. A literal pattern matches one exact
value; a bare name is a binder, which matches anything and binds it
for the arm's body. A binder at the end is the usual catch-all, and
`_` is the binder you use when you don't need the value.

```metaxu
fn describe(n: int) -> string {
    match n {
        0 => "zero"
        1 => "one"
        other => "many: " + other.to_string()
    }
}

fn main() -> int {
    print(describe(0));
    print(describe(1));
    print(describe(7));
    0
}
```
```output
zero
one
many: 7
```

Order matters: `other` above the literals would win every time. The
compiler doesn't reorder arms for you; it only insists the set of
them is complete.

## Exhaustiveness

Completeness is checked against the type. Leave out a variant and the
diagnostic names what's missing:

```metaxu error
enum Color { Red, Green, Blue }

fn main() -> int {
    let c = Color::Red;
    let n = match c {
        Color::Red => 1
        Color::Green => 2
    };
    print(n);
    0
}
```
```output
missing variants Blue
```

The check is shallow where literals are involved. Matching
`Some(1)` handles one particular `Some`, not the variant; as far as
the checker is concerned, `Some` itself is still uncovered:

```metaxu error
fn main() -> int {
    let o = Some(1);
    let n = match o {
        Some(1) => 1
        None => 0
    };
    print(n);
    0
}
```
```output
missing variants Some
```

The fix is an arm that covers the rest of the variant. Nested
patterns compose however you need: a literal inside a constructor, a
binder inside a constructor, in any order, first fit wins.

```metaxu
fn main() -> int {
    let o = Some(1);
    let n = match o {
        Some(1) => 100
        Some(v) => v
        None => 0
    };
    print(n);
    0
}
```
```output
100
```

## Tuples and nesting

Tuple patterns take positional aggregates apart, and everything nests.
Int components still need a catch-all, since no list of int literals
is ever complete.

```metaxu
fn main() -> int {
    let p = (3, 0);
    let spot = match p {
        (0, 0) => "origin"
        (x, 0) => "x axis at " + x.to_string()
        _ => "elsewhere"
    };
    print(spot);
    0
}
```
```output
x axis at 3
```

## Guards

A guard is an extra condition on an arm: `pattern if cond => body`.
The arm is taken only when the pattern fits and the condition holds;
a failing guard falls through to the arms below, in order. Binders
from the pattern are in scope in the guard.

```metaxu
fn classify(n: int) -> string {
    match n {
        v if v > 100 => "big"
        v if v > 10 => "medium"
        v if v > 0 => "small"
        _ => "non-positive"
    }
}

fn main() -> int {
    print(classify(500));
    print(classify(42));
    print(classify(3));
    print(classify(0));
    0
}
```
```output
big
medium
small
non-positive
```

Guarded arms do not count toward exhaustiveness. The compiler can't
know whether `v > 100` will hold, so the unguarded arms below must
still cover the type on their own; delete the `_` arm above and the
program is rejected. One restriction: handler arms (chapter 8) do not
take guards. `pattern if cond =>` is a `match` form only.

## if let and while let

When you care about exactly one pattern, a full `match` is ceremony.
`if let` runs its block when the pattern fits, and `while let` loops
as long as it keeps fitting, rebinding the pattern's names on each
pass.

```metaxu
fn step(n: int) -> Option[int] {
    if n > 0 { Some(n - 1) } else { None }
}

fn main() -> int {
    if let Some(v) = step(10) {
        print(v);
    };
    let mut n = 3;
    while let Some(m) = step(n) {
        print(m);
        n = m;
    }
    0
}
```
```output
9
2
1
0
```

The `while let` stops the moment `step` answers `None`. Chapter 9
leans on these forms constantly, because `Option` and `Result` are
where one-pattern questions come from.
