# Pattern matching

`match` takes a value apart. Each arm is a pattern, `=>`, and an
expression; the first pattern that fits chooses the arm, and whatever
names the pattern binds are in scope on its right-hand side.

```metaxu
enum Command {
    Move(dx: int, dy: int),
    Say(text: string),
    Quit
}

fn describe(c: Command) -> string {
    match c {
        Move(dx, dy) => f"move by {dx},{dy}",
        Say(text) => "say " + text,
        Quit => "quit"
    }
}

fn main() -> int {
    print(describe(Move(3, -1)));
    print(describe(Say("hello")));
    print(describe(Quit));
    0
}
```
```output
move by 3,-1
say hello
quit
```

Arms are separated by commas. Variant patterns can also be qualified
(`Command::Move(dx, dy)`) when the bare name is ambiguous or you just
want the enum in view.

## match is an expression

Like `if`/`else`, a `match` in value position produces the chosen
arm's value. This is the usual way to write a function that maps an
enum to something:

```metaxu
enum Coin { Penny, Nickel, Dime, Quarter }

fn cents(c: Coin) -> int {
    match c {
        Penny => 1,
        Nickel => 5,
        Dime => 10,
        Quarter => 25
    }
}

fn main() -> int {
    let pocket = cents(Quarter) + cents(Dime) + cents(Penny);
    print(pocket);
    0
}
```
```output
36
```

## Literals, binders, and the wildcard

A pattern can be a literal (matches that exact value), a bare name (a
binder: matches anything and binds it), or `_` (matches anything,
binds nothing). Arms are tried top to bottom, so specific cases go
first:

```metaxu
fn ordinal(n: int) -> string {
    match n {
        1 => "first",
        2 => "second",
        3 => "third",
        other => other.to_string() + "th"
    }
}

fn main() -> int {
    print(ordinal(1));
    print(ordinal(3));
    print(ordinal(9));
    match 42 {
        0 => print("zero"),
        _ => print("something else")
    }
    0
}
```
```output
first
third
9th
something else
```

There are no pattern guards (`Some(n) if n > 2 =>` doesn't parse).
Bind the value and put the `if` inside the arm instead.

## Nested patterns

Patterns compose: a variant pattern's arguments are themselves
patterns, so one arm can reach arbitrarily deep. Tuples destructure
the same way:

```metaxu
fn depth_two(o: Option) -> string {
    match o {
        Some(Some(n)) => f"value {n}",
        Some(None) => "an empty inner",
        None => "nothing at all"
    }
}

fn main() -> int {
    print(depth_two(Some(Some(7))));
    print(depth_two(Some(None)));
    print(depth_two(None));
    match (2, 5) {
        (x, y) => print(x * y)
    }
    0
}
```
```output
value 7
an empty inner
nothing at all
10
```

## Exhaustiveness

A `match` must cover every shape its scrutinee can take, and the
checker proves it at compile time. Miss a variant and the program
doesn't run; the diagnostic names what's missing:

```metaxu error
enum Color { Red, Green, Blue }

fn main() -> int {
    let c = Red;
    match c {
        Red => print("red"),
        Green => print("green")
    }
    0
}
```
```output
non-exhaustive match: missing variants Blue
```

The proof is per-variant and deliberately shallow. An arm covers its
variant only when its subpatterns are irrefutable (binders or `_`), so
`Some(1)` alone doesn't cover `Some`:

```metaxu error
fn pick(o: Option) -> int {
    match o {
        Some(1) => 1,
        None => 0
    }
}

fn main() -> int {
    print(pick(Some(2)));
    0
}
```
```output
missing variants Some
```

Add `Some(n) => ...` after the literal arm and it compiles: the binder
arm covers the variant, and the literal arm still wins for `1`. A
final `_` or binder arm makes any match exhaustive, which is also the
only way to match on ints or strings, since no list of literals can
ever cover them.

`bool` counts as a two-variant enum here: match `true` without `false`
and you get the same rejection.

## if let and while let

When you only care about one shape, a full `match` is noise. `if let`
runs its block when the pattern fits and binds its names there;
`while let` repeats as long as the pattern keeps fitting:

```metaxu
fn main() -> int {
    let found = Some(4);
    if let Some(n) = found {
        print(f"got {n}");
    }

    let @mut countdown = Some(3);
    while let Some(n) = countdown {
        print(n);
        countdown = if n > 1 { Some(n - 1) } else { None };
    }
    0
}
```
```output
got 4
3
2
1
```

An else-less `if let` is a statement, unit-valued like any else-less
`if`, so the block runs for its effects only.

Patterns are how every Option-returning function in `std/` gets
consumed, and `std/option.mx` wraps the recurring matches
(`unwrap_or`, `map`, `is_some`) so you don't write them twice. Chapter
9 tours those. Next, though: collections.
