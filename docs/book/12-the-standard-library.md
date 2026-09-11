# The standard library

`std/` is written in Metaxu, modeled on Ante's stdlib (the closest
relative language: algebraic effects plus ownership), and it doubles
as the compiler's toughest test suite; several silent-degradation bugs
were found only by writing these modules. `import std.foo` resolves to
`std/foo.mx`; `from std.foo import x` brings names into scope, and
`import std.foo as f` gives you qualified calls. This chapter tours
each module with running code.

## std.prelude

The everyday names in one import. Only unambiguous names are
re-exported; `map`, `filter` and `sum` exist in several modules and
stay in them.

```metaxu
from std.prelude import unwrap_or, clamp, parse_int;

fn main() -> int {
    print(clamp(99, 0, 10));
    print(unwrap_or(parse_int("41"), 0) + 1);
    0
}
```
```output
10
42
```

## std.option and std.result

Combinators over the builtin `Some`/`None` and `Ok`/`Err`: plain
matches, no effects. Chapter 9 shows them chained; the basics:

```metaxu
import std.option as opt;
import std.result as res;

fn main() -> int {
    let a = opt.map(Some(20), fn(x: int) -> x * 2);
    print(opt.unwrap_or(a, 0));
    print(opt.unwrap_or(None, 7));
    let r = res.map_err(Err("no"), fn(m: string) -> "error: " + m);
    match r { Ok(v) => print(v), Err(e) => print(e) };
    0
}
```
```output
40
7
error: no
```

## std.fail, std.throw, std.early_return

Failure as effects: a computation performs `fail()` or `throw(e)`, a
boundary handler (`try_opt`, `catch_`, `with_early_return`) decides
what that means, and the abort semantics skip everything after the
failing perform. Chapter 9 has the worked examples.

## std.stream

A stream is a thunk performing the `Emit` effect once per element;
consumers are handlers, transformers re-emit, so
`sum(filter(map(iota(n), f), p))` chains lazily with no intermediate
collections. Chapter 8 builds this module up from the effect system.

## std.iter

The adapters `std.stream` defers because they emit tuples:
`enumerate`, `zip`, `take_while`, `windows`, `chunks` and friends.
They are ordinary stream transformers and compose with `std.stream`'s
consumers.

```metaxu
from std.stream import iota, emit_vec, collect;
from std.iter import zip;
from std.vec import of3;

fn main() -> int {
    let pairs = collect(zip(iota(10), emit_vec(of3("a", "b", "c"))));
    for (n, s) in pairs {
        print(s, n)
    }
    0
}
```
```output
a 0
b 1
c 2
```

One documented caveat: `zip` realizes its second stream into a Vec
before pulling the first, because single-shot delimited continuations
cannot step two producers in lockstep. Pass the finite stream second.

## std.vec

Eager combinators over the builtin `Vec`: Vec in, fresh Vec out, the
argument never mutated. The lazy equivalents live in `std.stream`.

```metaxu
from std.vec import range_vec, map, filter, sum;

fn main() -> int {
    let squares = map(range_vec(1, 6), fn(x: int) -> x * x);
    print(sum(squares));
    let odds = filter(squares, fn(x: int) -> x % 2 == 1);
    print(len(odds));
    0
}
```
```output
55
3
```

## std.sort

Stable merge sort, following `std.vec`'s contract. Comparators are a
strict less-than predicate, not a three-way `cmp`; stability is what
lets chained `sort_by_key` calls order by several keys.

```metaxu
from std.sort import sort, binary_search;
from std.vec import of3;

fn main() -> int {
    let s = sort(of3(31, 4, 15));
    print(s[0], s[1], s[2]);
    match binary_search(s, 15) {
        Some(i) => print("found at", i),
        None => print("absent")
    };
    0
}
```
```output
4 15 31
found at 1
```

## std.map

An association-list `Map`. Every operation is O(n), stated up front,
and the API is shaped so a real hash map can replace the
representation without changing callers. `put` overwrites, `get`
answers an Option, and the map mutates in place.

```metaxu
import std.map as map;

fn main() -> int {
    let @mut m = map.empty();
    map.put(m, "ada", 1815);
    map.put(m, "grace", 1906);
    map.put(m, "ada", 1816);
    print(map.size(m));
    match map.get(m, "ada") { Some(y) => print(y), None => print("absent") };
    print(map.get_or(m, "alan", 0));
    0
}
```
```output
2
1816
0
```

## std.string

Helpers over builtin strings. Metaxu has no char type; `s[i]` yields a
one-character string, and this module's API is built on that.

```metaxu
from std.string import repeat, reverse, join;
from std.vec import of3;

fn main() -> int {
    print(repeat("ab", 3));
    print(reverse("stressed"));
    print(join(of3("a", "b", "c"), ", "));
    0
}
```
```output
ababab
desserts
a, b, c
```

## std.parse

Text into values, with the library's two standard answer shapes:
`parse_int` answers an Option, `parse_int_or_fail` performs `Fail` so
a multi-field parse aborts at the first bad field (chapter 9 shows
that composition). Floats, radix prefixes and overflow detection are
documented gaps, not silent misbehavior.

```metaxu
from std.parse import parse_int, trim, split_on;

fn main() -> int {
    match parse_int(trim("  42 ")) { Some(n) => print(n), None => print("bad") };
    match parse_int("4x2") { Some(n) => print(n), None => print("bad") };
    print(len(split_on("3,14,15", ",")));
    0
}
```
```output
42
bad
3
```

## std.math

Constants are real module-level bindings, read as plain names.
`sqrt`/`sin`/`cos` wrap the builtin methods; `abs`/`min`/`max`/`clamp`
take unannotated parameters and work on ints and floats.

```metaxu
from std.math import pi, abs, sqrt, powi;

fn main() -> int {
    print(pi);
    print(abs(-7));
    print(sqrt(9));
    print(powi(2, 10));
    0
}
```
```output
3.141592653589793
7
3.0
1024
```

## std.random

Randomness as an effect: `next()` has no default handler, so the
source is always an explicit choice. `with_seed` installs a xorshift64
generator (same sequence on every backend); `with_sequence` scripts
the draws outright. There is deliberately no real-entropy fallback:
performing `Random.next` unhandled is a loud error, never a silently
constant "random" number.

```metaxu
from std.random import Random, with_seed, next_below;

fn main() -> int {
    with_seed(7, fn() -> () {
        print(next_below(100), next_below(100), next_below(100));
        ()
    });
    0
}
```
```output
91 76 57
```

## std.state

The classic State effect with the classic three runners: `eval_state`
answers the result, `exec_state` the final state, `run_state` both.
The state lives in a cell shared by the handler's arms, which is how
the runners read it back after the block completes.

```metaxu
from std.state import State, run_state, modify, increment;

fn step() performs State -> () {
    increment(5);
    modify(fn(s: int) -> s * 2);
    ()
}

fn main() -> int {
    let r = run_state(10, fn() -> () { step() });
    print(r.state);
    0
}
```
```output
30
```

## std.log

The sink is a handler choice, not a global logger. Unhandled, the
effect's declared defaults print with level prefixes; `quietly`
silences, `with_collected_logs` hands you the lines as a Vec,
`with_min_level` filters by re-performing into the next enclosing
handler, so filters and collectors compose.

```metaxu
from std.log import Log, log_info, log_warn, with_collected_logs;

fn work() -> () {
    log_info("starting");
    log_warn("low disk");
    ()
}

fn main() -> int {
    work();                 # unhandled: the defaults print
    let lines = with_collected_logs(fn() -> () { work() });
    print("collected", len(lines));
    0
}
```
```output
[info] starting
[warn] low disk
collected 2
```

## std.test

Assertions report through a `Report` effect instead of aborting, so a
suite runs to the end and a runner tallies. Importing the module's
`assert_eq` shadows the aborting builtin of the same name on purpose
(chapter 7's plain-call precedence rule). The report's failure count
doubles as a process exit code.

```metaxu
from std.test import assert_eq, run_tests;

fn main() -> int {
    let report = run_tests("suite", fn() -> () {
        assert_eq(1 + 1, 2, "arithmetic");
        assert_eq(len("abc"), 3, "len");
        ()
    });
    print(report.total, "run,", report.failed, "failed");
    report.failed
}
```
```output
2 run, 0 failed
```

## std.sync and std.gpu

`std.sync` is thread-safe sharing: `protect(v)` builds a `Protected`
value the spawn checker recognizes as separate by construction, and
`with_lock`/`read`/`write`/`update` are the access idioms. Chapter 11
covers it with the Thread and Mutex effects. `std.gpu` backs the
`Tile` kernels and the `Gpu` effect; chapter 13 covers tiles, kernels,
and the Metal path.

A last practical note: a `std.*` name with no file under `std/`
(`std.simd`, `std.matrix`) still imports; nothing is rewritten, and
calls fall through to interpreter builtins or fail loudly at run time.
