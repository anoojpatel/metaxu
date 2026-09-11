# Functions and values

A Metaxu program is a set of `fn` declarations. Each parameter carries
a type, the return type follows `->`, and the body's last expression
is what the function returns.

```metaxu
fn add(a: int, b: int) -> int {
    a + b
}

fn shout(word: string) {
    print(word + "!");
}

fn main() -> int {
    print(add(19, 23));
    shout("hey");
    0
}
```
```output
42
hey!
```

`shout` has no `->` clause, so it returns unit. Statements end with
`;`; the trailing expression of a block doesn't, and dropping the
semicolon is what makes it the block's value. Put one after it and the
block becomes unit.

`return` exists for leaving early:

```metaxu
fn classify(n: int) -> string {
    if n < 0 {
        return "negative";
    }
    if n == 0 {
        return "zero";
    }
    "positive"
}

fn main() -> int {
    print(classify(-5));
    print(classify(0));
    print(classify(9));
    0
}
```
```output
negative
zero
positive
```

Those two else-less `if`s are statements: an `if` without an `else` is
always unit-valued. The arm runs for its effects (here, the `return`),
but its value is discarded.

## let, @mut, and shadowing

`let` binds a name. Add `@mut` when you intend to reassign the binding
or mutate through it; that's the convention everywhere in `std/`, and
the mode system (chapter 10) is where it grows teeth. `mut` without the
`@` is accepted as the same annotation. A second `let` with the same
name shadows the first, which is how you "change" a value you'd rather
keep immutable:

```metaxu
fn main() -> int {
    let @mut total = 0;
    total = total + 10;
    total = total + 5;
    print(total);

    let x = 100;
    let x = x + 1;   # a new binding, shadowing the old one
    print(x);
    0
}
```
```output
15
101
```

Type annotations on `let` are optional; inference (chapter 6) fills
them in. When you want one, it goes after the name: `let y: float =
1.5;`.

## if and else as expressions

With both arms present, `if`/`else` is an expression, and each arm's
value is the block's last expression:

```metaxu
fn main() -> int {
    let n = 7;
    let size = if n > 10 { "big" } else { "small" };
    print(size);

    if n < 0 {
        print("neg");
    } else if n == 0 {
        print("zero");
    } else {
        print("pos");
    }
    0
}
```
```output
small
pos
```

Braces are mandatory, and there is no freestanding `{ ... }`
expression: blocks get their value from being function bodies or
`if`/`else`/`match` arms.

## while

`while` repeats a block. The condition must be `bool`; the checker
rejects a bare integer there, the same way it rejects `1 + "a"`:

```metaxu
fn main() -> int {
    let @mut i = 0;
    let @mut sum = 0;
    while i < 5 {
        sum = sum + i;
        i = i + 1
    }
    print(sum);
    0
}
```
```output
10
```

(`for x in ...` also exists; chapter 5 covers it alongside the
collections you'd iterate.)

## Numbers, strings, and to_string

`int` and `float` are distinct types with the usual operators. `/` on
ints is integer division; `%` is remainder. Strings compare with `==`
and order lexicographically with `<`. `+` concatenates strings, and
only strings: gluing a number on takes `.to_string()`, or an f-string,
which interpolates any expression between braces:

```metaxu
fn main() -> int {
    print(7 / 2);
    print(7 % 2);
    print(2.5 + 0.5);
    print("score: " + (7 * 6).to_string());
    let name = "Metaxu";
    let n = 5;
    print(f"hello {name}, {n} squared is {n * n}");
    0
}
```
```output
3
1
3.0
score: 42
hello Metaxu, 5 squared is 25
```

Comparisons (`==`, `!=`, `<`, `<=`, `>`, `>=`) produce `bool`, and
`&&`, `||`, `!` combine them; you've seen them in every condition so
far.

## Functions as values

`fn` without a name is a lambda. The body is either a single expression
after `->`, or a braced block; parameter types may be omitted and
inferred. Lambdas close over the bindings around them, and a function
type is written `fn(int) -> int`:

```metaxu
fn apply_twice(f: fn(int) -> int, v: int) -> int {
    f(f(v))
}

fn main() -> int {
    let square = fn(x: int) -> x * x;
    print(square(6));

    let offset = 10;
    let shifted = fn(x) -> x + offset;   # captures offset
    print(shifted(5));

    let clamped = fn(x: int) {
        if x > 100 { 100 } else { x }
    };
    print(apply_twice(clamped, 7));
    0
}
```
```output
36
15
7
```

There is no `|x| ...` form; `fn(x) -> ...` is the lambda syntax.
`std/sort.mx` is written entirely in this style: `sort_by(v, fn(a, b)
-> a < b)`.

## What a call does to its arguments

An unmarked parameter gives the function its own view of the value;
writes to it don't reach the caller. Mark the parameter `@mut` and they
do:

```metaxu
struct Box { value: int }

fn keeps(b: Box) {
    b.value = 99;
}

fn bumps(b: @mut Box, by: int) {
    b.value = b.value + by;
}

fn main() -> int {
    let @mut bx = Box { value: 1 };
    keeps(bx);
    print(bx.value);
    bumps(bx, 41);
    print(bx.value);
    0
}
```
```output
1
42
```

Nothing extra is written at the call site; the parameter's mode is the
contract, and the borrow checker holds both sides to it. (`Vec` behaves
differently, on purpose. Chapter 5.)

## Names must exist

An undefined name is a compile error, not a runtime one. The checker
resolves every variable and callee before anything runs:

```metaxu error
fn main() -> int {
    let total = subtotal + tax;
    0
}
```
```output
undefined variable 'subtotal'
```

Next: putting values together into your own types.
