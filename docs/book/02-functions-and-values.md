# Functions and values

Chapter 1 ran a program. This chapter is the expression language it
was written in: functions, blocks and their values, what a name may
do after you bind it, and functions passed around as values.

## Declaring functions

A function names its parameters with types and declares its return
type. The body is a block, and the block's last expression is the
function's value. No `return` needed on the happy path.

```metaxu
fn add(a: int, b: int) -> int {
    a + b
}

fn main() -> int {
    print(add(3, 4));
    0
}
```
```output
7
```

Statements inside a block end with `;`; the tail expression doesn't,
and dropping the semicolon is what marks it as the tail. That's why
`main` ends with a bare `0` in every example in this book.

## Early return

`return` exists for leaving before the tail. Here the else-less `if`
is a statement (its value is unit), so it takes a `;` like any other
expression used as a statement; the function's real value is the tail
`n` below it.

```metaxu
fn magnitude(n: int) -> int {
    if n < 0 {
        return 0 - n;
    };
    n
}

fn main() -> int {
    print(magnitude(0 - 7));
    print(magnitude(4));
    0
}
```
```output
7
4
```

One thing you might expect from Rust that Metaxu does not have: a
freestanding block expression. `let z = { let a = 1; a + 1 };` is a
parse error. Blocks are valued in exactly three places: function
bodies, `if`/`else` arms, and `match` arms.

## `let` is immutable

`let` binds a name once. Assigning to it again is not a runtime
mistake, it's a program the compiler refuses:

```metaxu error
fn main() -> int {
    let count = 1;
    count = 2;
    0
}
```
```output
cannot assign twice to immutable binding
```

The full diagnostic names the binding and tells you the fix: declare
it mutable with `let mut`. A `mut` binding may be reassigned freely.
Separately, a fresh `let` of the same name is always fine; that's
shadowing, a new binding that hides the old one rather than changing
it.

```metaxu
fn main() -> int {
    let x = 1;
    let x = x + 10;      # shadowing: a new x, computed from the old
    print(x);

    let mut total = 0;   # mutation: one binding, reassigned
    total = total + 5;
    total = total + 7;
    print(total);
    0
}
```
```output
11
12
```

There is a third form, `let @mut`, which makes the binding mutable
and also claims the `@mut` mode on the value itself, the exclusive
right to mutate it that the borrow checker tracks. Chapter 10 covers
modes; until then, the idiom to recognize is
`let @mut v = Vec.new();` for collections you intend to grow.

## `if` is an expression

`if cond { } else { }` produces a value, so it goes anywhere an
expression goes, including the right side of a `let`. Comparisons
produce bools, and bools print as `1` and `0`: native code erases
bools to integers, and the interpreter formats them the same way so
the two engines agree byte for byte.

```metaxu
fn main() -> int {
    let n = 7;
    let label = if n > 5 { "big" } else { "small" };
    print(label);
    print(n > 5);
    print(n > 50);
    0
}
```
```output
big
1
0
```

An `if` without an `else` is unit, so it can't be a value; use it as
a statement. Both arms of a valued `if` must agree on their type,
which chapter 6 has more to say about.

## `while`

`while` repeats a block while its condition holds. This also shows
`print` with two arguments: it joins them with a single space.

```metaxu
fn main() -> int {
    let mut i = 1;
    let mut sum = 0;
    while i < 5 {
        sum = sum + i;
        i = i + 1;
    }
    print("sum", sum);
    0
}
```
```output
sum 10
```

`for` loops iterate collections, so they wait for chapter 5.

## Numbers and strings

Ints and floats are separate types with the usual arithmetic. Floats
always print with their point (`5.0`, not `5`). Strings concatenate
with `+`, but only with other strings: gluing a number on takes
`.to_string()`. For anything beyond a single `+`, an f-string
interpolates a bound name in place.

```metaxu
fn main() -> int {
    print(7 * 6);
    print(2.5 * 2.0);
    let price = 3;
    print("price: " + price.to_string());
    let name = "Metaxu";
    print(f"hello from {name}");
    0
}
```
```output
42
5.0
price: 3
hello from Metaxu
```

Chapter 5 covers the rest of the string surface: length, indexing,
ordering.

## Functions are values

A lambda is written `fn(x: int) -> expr`, or with a block body and a
declared return type, `fn(x: int) -> int { ... }`. There is no `|x|`
form for lambdas that take arguments; the one pipe spelling in the
language is the zero-argument closure `|| { ... }` that `Thread.spawn`
takes (chapter 11). Lambda types read the same way the expressions do:
`fn(int) -> int` is a function from int to int, and a parameter of
that type accepts named functions and lambdas alike. Lambdas capture
the bindings around them.

```metaxu
fn apply_twice(f: fn(int) -> int, x: int) -> int {
    f(f(x))
}

fn main() -> int {
    let double = fn(x: int) -> x * 2;
    print(double(21));
    print(apply_twice(double, 5));

    let step = 10;
    let add_step = fn(n: int) -> int {
        n + step
    };
    print(add_step(32));
    0
}
```
```output
42
20
42
```

Chapter 8 builds a lot on this: handlers receive functions, streams
are functions performing effects.

## Arguments are copies

A scalar argument arrives in the callee as its own value. The
function can compute with it, but nothing it does reaches the
caller's binding (parameters are immutable, same as plain `let`).

```metaxu
fn bump(n: int) -> int {
    n + 1
}

fn main() -> int {
    let n = 41;
    print(bump(n));
    print(n);
    0
}
```
```output
42
41
```

Collections behave differently: a `Vec` is a handle, and two handles
can share one buffer. That distinction gets its own section in
chapter 5, and the rules for who may mutate what through which handle
are chapter 10's subject.
