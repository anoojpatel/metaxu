# The standard library

`std/` is written in Metaxu. The modules lean on the language's own
mechanisms, effects especially, so reading them is a second pass over
chapters 8 and 9 with working code. This chapter tours the everyday
modules, one small program each; `from std.vec import map;` is the
import form used throughout, and chapter 15 covers the module system
itself. Four modules get their own chapters and only a pointer here:
streams (8), sync (11), gpu (13), and the fail family (9).

## The prelude

Some names need no import: `print`, `Vec`, `Option` with `Some` and
`None`, `to_string`, `len`, the operators. They're the vocabulary
every chapter so far has been using.

```metaxu
fn main() -> int {
    let words = "one two";
    print(words.len());
    let n = 3;
    print(n.to_string() + "!");
    print(Some(n));
    0
}
```
```output
7
3!
Option::Some(3)
```

The last line is the default rendering for enum values: type,
variant, payload. Stable across both engines, like all printing.

## std.option and std.result

Combinators for the two workhorse enums, so a pipeline doesn't need
a `match` at every step.

```metaxu
from std.option import unwrap_or;
from std.result import to_option;

fn main() -> int {
    print(unwrap_or(Some(41), 0) + 1);
    print(unwrap_or(None, 7));
    print(unwrap_or(to_option(Ok(20)), 0) * 2);
    print(unwrap_or(to_option(Err("no parse")), 0 - 1));
    0
}
```
```output
42
7
40
-1
```

`to_option` forgets the error, which is often what the edge of a
pipeline wants. Chapter 9 discusses when these types are the right
tool and when a raise is.

## std.iter

Iteration helpers that work with `for`. The one to know is `zip`,
which pairs two collections positionally into tuples:

```metaxu
from std.iter import zip;

fn main() -> int {
    let names = ["ada", "alan"];
    let years = [1815, 1912];
    for pair in zip(names, years) {
        match pair {
            (name, year) => print(name, year)
        }
    }
    0
}
```
```output
ada 1815
alan 1912
```

The tuple pattern takes each pair apart; `print` with two arguments
joins them with a space. `zip` stops at the shorter input.

## std.vec

Transformations over `Vec` as plain functions. Each returns a fresh
Vec (or a scalar) and leaves its input alone.

```metaxu
from std.vec import map, filter, sum;

fn main() -> int {
    let v = [1, 2, 3, 4];
    print(sum(filter(v, fn(x) -> x > 2)));
    let scaled = map(v, fn(x) -> x * 10);
    print(scaled[0]);
    print(sum(scaled));
    0
}
```
```output
7
10
100
```

For pipelines too large to materialize, `std.stream` does the same
shapes element at a time; the pointer section below says where.

## std.sort

Sorting and searching. `sort` returns an ordered copy;
`binary_search` finds a value's index in an ordered Vec, as an
`Option`:

```metaxu
from std.sort import sort, binary_search;

fn main() -> int {
    let s = sort([9, 3, 7]);
    print(s[0], s[1], s[2]);
    match binary_search(s, 7) {
        Some(i) => print(i)
        None => print("absent")
    };
    match binary_search(s, 8) {
        Some(i) => print(i)
        None => print("absent")
    };
    0
}
```
```output
3 7 9
1
absent
```

`binary_search` on an unsorted Vec is allowed to answer nonsense;
that's what the name promises and no more. Sort first.

## std.map

A key-value map. `insert` adds; `get` answers with an `Option`, so a
missing key is a value, not a raise:

```metaxu
from std.map import Map;

fn main() -> int {
    let @mut ages = Map.new();
    ages.insert("ada", 36);
    ages.insert("alan", 41);
    print(ages.len());
    match ages.get("ada") {
        Some(a) => print(a)
        None => print("absent")
    };
    match ages.get("grace") {
        Some(a) => print(a)
        None => print("absent")
    };
    0
}
```
```output
2
36
absent
```

Like `Vec`, a `Map` has identity semantics: two bindings to one map
alias one table, and the thread rules of chapter 11 apply to it the
same way.

## std.string

Chapter 5 covered what strings do on their own: `+`, `len`, indexing,
ordering. `std.string` adds the take-apart and put-together helpers:

```metaxu
from std.string import split, join, to_upper;

fn main() -> int {
    let parts = split("effects,modes,tiles", ",");
    print(parts.len());
    print(parts[1]);
    print(join(parts, " + "));
    print(to_upper("metaxu"));
    0
}
```
```output
3
modes
effects + modes + tiles
METAXU
```

`split` yields a `Vec` of strings, so everything from `std.vec` and
`std.sort` applies to its result directly.

## std.parse

Text to numbers, with failure as data: "not a number" is an expected
answer for input you didn't write, so `parse_int` returns an
`Option` rather than raising.

```metaxu
from std.parse import parse_int;

fn main() -> int {
    match parse_int("42") {
        Some(n) => print(n)
        None => print("not a number")
    };
    match parse_int("4x2") {
        Some(n) => print(n)
        None => print("not a number")
    };
    0
}
```
```output
42
not a number
```

Pair it with `unwrap_or` from `std.option` when a default is the
right recovery.

## std.math

Arithmetic beyond the operators: `abs`, `min`, `max`, integer `pow`,
and float functions that follow the float formatting rules chapter 2
pinned.

```metaxu
from std.math import abs, min, max, pow;

fn main() -> int {
    print(abs(0 - 5));
    print(min(3, 4), max(3, 4));
    print(pow(2, 10));
    0
}
```
```output
5
3 4
1024
```

## std.random

A seeded pseudorandom generator. The seed fixes the stream: same
seed, same draws, run after run, on both engines. This example pins
the properties rather than the constants; the module's own tests pin
the exact streams.

```metaxu
from std.random import Rng;

fn main() -> int {
    let @mut a = Rng.seed(2024);
    let @mut b = Rng.seed(2024);
    let x = a.range(0, 100);
    print(x == b.range(0, 100));
    print(x >= 0);
    print(x < 100);
    0
}
```
```output
1
1
1
```

Two generators from one seed agree draw for draw, and `range(lo, hi)`
stays in its half-open bounds. (Bools print as `1` and `0`, matching
the native representation.) Determinism by default keeps tests
reproducible; code that wants real entropy seeds from outside.

## std.state

Chapter 8 built a counter from operations with a handler holding the
cell. `std.state` is that construction packaged: `with_state(initial,
f)` runs `f` under a state handler, and `get`/`put` are the
operations.

```metaxu
from std.state import get, put, with_state;

fn main() -> int {
    let r = with_state(10, fn() -> int {
        put(get() * 4);
        put(get() + 2);
        get()
    });
    print(r);
    0
}
```
```output
42
```

The block never touches a mutable binding; the mutation lives in the
handler, and a test can swap in a recording handler without changing
the block.

## std.log

Leveled logging with a fixed line format, so grep works:

```metaxu
from std.log import info, warn;

fn main() -> int {
    info("starting up");
    warn("low disk");
    print("done");
    0
}
```
```output
[info] starting up
[warn] low disk
done
```

Logging is an effect underneath, which is the useful part: a handler
above your code can redirect, filter, or silence it without a
configuration system.

## std.test

Assertion helpers and a runner. A test is a name and a thunk
returning a bool; `run_tests` runs a Vec of them and returns how many
passed, leaving the reporting format to you:

```metaxu
from std.test import run_tests;

fn main() -> int {
    let suite = [
        ("addition", fn() -> bool { 1 + 1 == 2 }),
        ("ordering", fn() -> bool { "a" < "b" }),
        ("length", fn() -> bool { "four".len() == 4 })
    ];
    let passed = run_tests(suite);
    print(passed.to_string() + " of 3 passed");
    0
}
```
```output
3 of 3 passed
```

`std.test` is for testing Metaxu programs in Metaxu; the compiler's
own suite is pytest, and the book's examples run under the harness
the README describes.

## The modules with chapters of their own

`std.stream` builds lazy pipelines (`iota`, `map`, `filter`, `sum`,
`for_`) from a single `Emit` effect: no intermediate collections,
elements flowing one at a time through suspending handlers. Chapter 8
walks the mechanism.

`std.sync` is the sanctioned answer to shared mutation across
threads: `protect` pairs a value with its own mutex, and
`read`/`write`/`update` are the only doors in. Chapter 11 shows four
threads driving one counter to exactly a thousand.

`std.gpu` exposes `Tile` values and `Gpu.launch`, an effect with a
CPU default so kernel code runs anywhere, plus a Metal handler for
real hardware. Chapter 13.

`std.fail`, `std.throw`, and `std.early_return` are the abort
pattern packaged three ways, worked through at the end of chapter 9.

That's the tour. The habit the library rewards most: when a module
surprises you, open `std/` and read it. It's short, it's Metaxu, and
after chapter 8 none of it is magic.
