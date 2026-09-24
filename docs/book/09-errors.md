# Errors

Metaxu splits errors into three stories. Compile-time errors (types,
borrows, undefined names) stop the build; chapters 6 and 10 cover
them. Runtime failures can be contained with `try`/`catch`. And for
errors that are part of a function's contract, the library gives you
`Option`, `Result`, and three effects built on handlers.

## try/catch

`try { body } catch e { handler }` is an expression. If the body
completes, its value is the result and the catch never runs. If a
runtime failure occurs anywhere in the body's dynamic extent, including
inside functions it calls, the rest of the body is abandoned and the
handler's value is the result instead:

```metaxu
fn main() -> int {
    let fine = try { 41 + 1 } catch e { 0 };
    let broken = try {
        let v = Vec.new();
        v.push(10);
        v[3]
    } catch e {
        print("caught: " + e);
        -1
    };
    print(fine);
    print(broken);
    0
}
```
```output
caught: index out of bounds: 3 (length 1)
42
-1
```

`e` binds the failure's message: the plain text as raised, with no
file paths and no compiler context attached, so the same program
catches the same string on every machine and both backends
(`docs/try_catch.md` pins this byte for byte). Nesting behaves the way
handlers do: the innermost try catches, and a failure raised inside a
catch block propagates outward rather than being caught by its own
try.

## What raises, and what catch sees

The catchable failures are: builtin contract violations (indexing out
of bounds, `pop` on an empty Vec, `Tile.get` outside the tile's shape,
a shift count outside 0..63); a `perform` with no handler installed; a
match that no pattern accepts at runtime; a trait method call whose
receiver type has no impl (chapter 7 catches one); a failed file or
process operation from `std.fs` and friends (chapter 12); and your
own `raise(message)`, which fails with exactly that text:

```metaxu
fn parse_port(text: string) -> int {
    let @mut value = 0;
    let @mut i = 0;
    while i < len(text) {
        let d = "0123456789".find(text[i]);
        if d < 0 { raise("parse_port: not a number: " + text) } else { () };
        value = value * 10 + d;
        i = i + 1
    }
    if value > 65535 { raise("parse_port: out of range: " + text) } else { () };
    value
}

fn main() -> int {
    print(parse_port("8080"));
    print(try { parse_port("80x") } catch e { print(e); 0 - 1 });
    print(try { parse_port("99999") } catch e { print(e); 0 - 1 });
    0
}
```
```output
8080
parse_port: not a number: 80x
-1
parse_port: out of range: 99999
-1
```

The unhandled-`perform` case defines the division of labor between
`try` and `handle`:

```metaxu
effect Parser { parse(input: string) -> int }

fn main() -> int {
    let r = try {
        perform Parser.parse("123")
    } catch e {
        print("caught: " + e);
        0
    };
    print(r);
    0
}
```
```output
caught: No handler for effect 'Parser'
0
```

Catching this failure is dynamic recovery, not a handler: the `try`
does not discharge the effect statically, and it cannot resume the
computation. When a handler is installed, it wins, and the try has
nothing to catch:

```metaxu
effect Ask { ask() -> int }

fn main() -> int {
    let v = handle Ask with {
        ask() -> resume(5)
    } in {
        try { perform Ask.ask() + 1 } catch e { -1 }
    };
    print(v);
    0
}
```
```output
6
```

Not everything is catchable, on purpose. An `assert` failure, integer
division by zero, and resuming a continuation twice are fatal on both
backends. Recursion-budget exhaustion also passes through `catch`: a
catch arm would run with the stack still at its ceiling. And
compile-time errors are simply not runtime events; wrapping a type
error in `try` does not turn it into one:

```metaxu error
fn main() -> int {
    let r = try { 1 + "a" } catch e { 0 };
    r
}
```
```output
one value is required to be Int and String
```

## Option and Result

For failures a caller is expected to handle, don't raise: return
`Option` (`Some`/`None`) or `Result` (`Ok`/`Err`). Both are builtins;
`std.option` and `std.result` add the combinators:

```metaxu
import std.option as opt;
import std.result as res;
from std.parse import parse_int;

fn half(n: int) {
    if n % 2 == 0 { Some(n / 2) } else { None }
}

fn main() -> int {
    let a = opt.and_then(parse_int("84"), fn(n: int) -> half(n));
    print(opt.unwrap_or(a, -1));

    let r = opt.ok_or(parse_int("4x"), "not a number");
    match r {
        Ok(v) => print(v),
        Err(e) => print("error: " + e)
    };
    print(opt.unwrap_or(res.ok(r), 0));
    0
}
```
```output
42
error: not a number
0
```

`parse_int` answers an Option, `and_then` chains the next fallible
step, `ok_or` converts to a Result with an error of your choosing, and
`res.ok` goes back. All of them are plain matches.

## The effectful error story

`std.fail`, `std.throw`, and `std.early_return` (designs ported from
Ante's stdlib) treat failure as an algebraic effect: the failing code
performs an operation, and a handler at the boundary decides what
failure means. These handlers abort rather than resume.

`Fail` carries no payload. `try_opt` runs a computation and answers an
Option; a multi-step computation aborts at the first failure with no
per-step matching:

```metaxu
from std.fail import Fail, try_opt;
from std.parse import parse_int_or_fail;

fn sum_fields(a: string, b: string) performs Fail -> int {
    parse_int_or_fail(a) + parse_int_or_fail(b)
}

fn main() -> int {
    match try_opt(fn() -> int { sum_fields("12", "30") }) {
        Some(v) => print(v),
        None => print("malformed")
    };
    match try_opt(fn() -> int { sum_fields("12", "3O") }) {
        Some(v) => print(v),
        None => print("malformed")
    };
    0
}
```
```output
42
malformed
```

`Throw` is `Fail` with an error value. `catch_` turns a throwing
computation into a Result (the underscore is there because `catch` is
a reserved word):

```metaxu
from std.throw import Throw, catch_, throw_if;

fn withdraw(balance: int, amount: int) performs Throw -> int {
    throw_if(amount > balance, "insufficient funds");
    balance - amount
}

fn main() -> int {
    match catch_(fn() -> int { withdraw(100, 30) }) {
        Ok(left) => print("left: " + left.to_string()),
        Err(e) => print("error: " + e)
    };
    match catch_(fn() -> int { withdraw(100, 300) }) {
        Ok(left) => print(left),
        Err(e) => print("error: " + e)
    };
    0
}
```
```output
left: 70
error: insufficient funds
```

`EarlyReturn` is the same mechanism used for control flow rather than
failure: multi-level early exit without a dedicated statement.

```metaxu
from std.early_return import EarlyReturn, with_early_return, return_if;

fn first_negative_index(v: Vec) -> int {
    with_early_return(fn() -> int {
        let @mut i = 0;
        while i < len(v) {
            return_if(v[i] < 0, i);
            i = i + 1
        }
        -1
    })
}

fn main() -> int {
    let @mut v = Vec.new();
    v.push(5);
    v.push(-3);
    v.push(8);
    print(first_negative_index(v));
    0
}
```
```output
1
```

Which to reach for: `Option`/`Result` when the caller should decide at
every step; `Fail`/`Throw` when a whole computation shares one failure
boundary; `try`/`catch` when the failure is a violated contract you
didn't plan for and you want the program to survive it anyway.
