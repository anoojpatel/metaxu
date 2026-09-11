# Errors

Metaxu splits errors into two worlds. Compile-time rejections, the
kind chapters 6 and 10 pinned, stop the program from existing.
Runtime raises happen while a program runs, and `try`/`catch` handles
them. Underneath, a raise is chapter 8's abort pattern with a name,
which is why the rules here will feel like handler rules.

## try is an expression

`try { }` evaluates its block. If nothing raises, the block's value
is the result. If something raises, the rest of the block is
abandoned and the `catch` arm's value is the result instead, with the
error bound to the name after `catch`.

```metaxu
fn main() -> int {
    let ok = try { 40 + 2 } catch e { 0 };
    print(ok);

    let @mut v = Vec.new();
    v.push(1);
    let bad = try {
        print("reading");
        let x = v[10];
        print("never printed");
        x
    } catch e {
        0 - 1
    };
    print(bad);
    0
}
```
```output
42
reading
-1
```

The abort is real: `"never printed"` isn't. Work before the raise
happened; work after it never will.

## What raises

The builtin operations raise on the failures they can't answer:
indexing out of bounds, popping an empty Vec, dispatching a method
with no implementation (chapter 7). The error is a value; printing
it shows the message.

```metaxu
fn main() -> int {
    let @mut v = Vec.new();
    v.push(7);
    print(v.pop());
    let r = try {
        v.pop()
    } catch e {
        print(e);
        0
    };
    print(r);
    0
}
```
```output
7
pop: Vec is empty
0
```

The first `pop` succeeds. The second finds nothing, raises, and the
catch arm both inspects the error and supplies a stand-in. The
message wording is part of the language: the native backend produces
the same bytes.

## Unhandled effects raise

Performing an operation with no handler in scope and no default is a
runtime error too, and `try` catches it:

```metaxu
effect Parser {
    parse(s: string) -> int
}

fn main() -> int {
    let r = try {
        perform Parser.parse("42")
    } catch e {
        print(e);
        0 - 1
    };
    print(r);
    0
}
```
```output
No handler for effect 'Parser'
-1
```

This is the safety net under chapter 8's contract: if no handler is
there when the perform executes, the failure is loud, named, and
catchable.

## Handlers come first

`try` only sees raises that nothing else claimed. If a handler for
the effect is installed inside the `try`, the handler answers and the
catch arm never runs:

```metaxu
effect Parser { parse(s: string) -> int }

fn main() -> int {
    let r = try {
        handle Parser with {
            parse(s) -> resume(s.len())
        } in {
            perform Parser.parse("hello")
        }
    } catch e {
        print("caught");
        0
    };
    print(r);
    0
}
```
```output
5
```

Same perform as before, but this time a handler is nearer. Think of
`try` as the outermost, least specific handler: it catches whatever
falls all the way through.

## Catchable and fatal

What `try` can catch, in one place:

* out-of-bounds indexing, and `pop` on an empty Vec (chapter 5);
* method dispatch with no implementation (chapter 7);
* a `perform` with no handler and no default (above);
* explicit raises from `std.fail` and `std.throw` (below);
* a contended cross-thread write (chapter 11);
* violations of the strict simulated heap in `unsafe` code
  (chapter 16).

What it can never catch: compile errors. A rejection happens before
the program exists, so wrapping the offending code in `try` changes
nothing:

```metaxu error
fn main() -> int {
    let r = try {
        1 + "a"
    } catch e {
        0
    };
    print(r);
    0
}
```
```output
Int and String
```

The conflict is reported exactly as in chapter 6. `try` is a runtime
construct; it has no opinion about programs that never start.

## Option and Result

Not every absence deserves a raise. `Option` says "maybe no value"
in the type, and `Result` says "value or error", and both are
ordinary enums you can match. `std.option` and `std.result` add the
combinators so the matches don't pile up.

```metaxu
from std.option import map, unwrap_or;

fn main() -> int {
    let some = Some(20);
    let doubled = map(some, fn(x) -> x * 2 + 2);
    print(unwrap_or(doubled, 0));
    print(unwrap_or(None, 7));
    0
}
```
```output
42
7
```

`map` transforms the value if there is one and passes `None` through
untouched; `unwrap_or` ends the chain with a default. Uniform call
syntax (chapter 7) lets the same functions chain as methods. The
working rule: use `Option` and `Result` when absence or failure is an
expected answer the caller should handle, and raises when it's a bug
or a broken environment.

## fail, throw, early_return

Chapter 8 built an exception out of a plain abort handler.
`std.fail` is that construction, packaged: `fail(msg)` raises with
your message, and `try` catches it like any builtin raise.

```metaxu
from std.fail import fail;

fn checked_div(a: int, b: int) -> int {
    if b == 0 {
        fail("divide by zero");
        0
    } else {
        a / b
    }
}

fn main() -> int {
    print(checked_div(84, 2));
    let r = try {
        checked_div(1, 0)
    } catch e {
        print(e);
        0 - 1
    };
    print(r);
    0
}
```
```output
42
divide by zero
-1
```

`std.throw` is the same idea carrying an arbitrary value instead of
a message, with its own catching wrapper for typed payloads.

`std.early_return` aims the abort somewhere gentler: at the end of
the current computation, not an error handler. `with_return` runs a
block; `early_return(v)` anywhere inside finishes it with `v`.

```metaxu
from std.early_return import early_return, with_return;

fn main() -> int {
    let r = with_return(fn() -> int {
        print("searching");
        early_return(3);
        print("kept looking");
        99
    });
    print(r);
    0
}
```
```output
searching
3
```

No error exists here; `"kept looking"` is skipped because the
computation already has its answer. All three helpers are small
handlers over the abort pattern; reading their source in `std/` is a
good exercise after chapter 8.
