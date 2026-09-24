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

## std.semver

The first piece of the package manager written in Metaxu: versions,
version requirements in the Cargo dialect, and ranges closed under
intersection, union and complement, which is what a version solver
needs. It is a line-for-line port of glade's Python module, and a
test runs both over the same few hundred cases and compares every
answer (`docs/glade_in_metaxu.md`).

```metaxu
from std.semver import parse_version, parse_requirement, range_contains,
    range_intersect, range_to_string;

fn main() -> int {
    match parse_requirement("^1.2") {
        None => print("bad requirement"),
        Some(r) => {
            print(range_to_string(r));
            match parse_version("1.5.0") {
                None => print("bad version"),
                Some(v) => print(range_contains(r, v))
            };
            match parse_version("2.0.0-rc.1") {
                None => print("bad version"),
                Some(v) => print(range_contains(r, v))
            };
            match parse_requirement(">=1.5, <3") {
                None => print("bad requirement"),
                Some(other) => print(range_to_string(range_intersect(r, other)))
            }
        }
    };
    0
}
```
```output
>=1.2.0, <2.0.0
1
0
>=1.5.0, <2.0.0
```

`1.5.0` is inside `^1.2`; `2.0.0-rc.1` is not, because a prerelease
only satisfies a requirement that names a prerelease of the same
version. Booleans print as `1` and `0`.

## std.solve

The version solver, PubGrub, ported from glade's Python. It takes the
dependency graph as data: every version of every package with its
requirements, which is what a registry index lists. Here the newest
`foo` wants a `bar` the project rules out, so the solver settles on
the older `foo`; had nothing fit, `out.explanation` would hold the
reason as sentences.

```metaxu
from std.semver import version, parse_requirement, range_any, version_to_string;
from std.solve import graph_new, graph_add, dep, solve;

fn req(name: string, text: string) -> Dep {
    match parse_requirement(text) { None => dep(name, range_any()), Some(r) => dep(name, r) }
}

fn main() -> int {
    let @mut g = graph_new();
    let @mut foo10 = Vec.new();
    foo10.push(req("bar", "^1"));
    graph_add(g, "foo", version(1, 0, 0), foo10);
    let @mut foo11 = Vec.new();
    foo11.push(req("bar", "^2"));
    graph_add(g, "foo", version(1, 1, 0), foo11);
    graph_add(g, "bar", version(1, 0, 0), Vec.new());
    graph_add(g, "bar", version(2, 0, 0), Vec.new());
    let @mut root = Vec.new();
    root.push(req("foo", "^1"));
    root.push(req("bar", "^1"));
    graph_add(g, "app", version(0, 1, 0), root);
    let out = solve(g, "app", version(0, 1, 0), Vec.new(), Vec.new());
    print(out.ok);
    let @mut i = 0;
    while i < len(out.names) {
        print(out.names[i] + " " + version_to_string(out.versions[i]));
        i = i + 1
    }
    0
}
```
```output
1
bar 1.0.0
foo 1.0.0
```

## std.hex and std.sha256

Bytes are a `Vec` of ints from 0 to 255 (chapter 5). `std.hex` renders
them as lowercase hex and parses hex back, and `std.sha256` is SHA-256
written in Metaxu, tested against Python's `hashlib` on both engines.
The package manager hashes every vendored tree with it, and a native
binary that does so needs no crypto library.

```metaxu
from std.hex import to_hex, from_hex;
from std.sha256 import sha256_hex, sha256_string;

fn main() -> int {
    print(sha256_string(""));
    match from_hex("616263") {
        None => print("bad hex"),
        Some(bytes) => {
            print(bytes.from_bytes());
            print(sha256_hex(bytes) == sha256_string("abc"));
            print(to_hex(bytes))
        }
    };
    match from_hex("6x") { None => print("bad hex"), Some(b) => print(len(b)) };
    0
}
```
```output
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
abc
1
616263
bad hex
```

## std.path, std.json and std.toml

Three text formats the package manager lives on, each written in
Metaxu and each tested against the Python module that defines it:
`std.path` follows `posixpath` (join, dirname, basename, normalize,
relpath), `std.json` writes what `json.dumps` writes, and `std.toml`
reads the subset of TOML that manifests and lockfiles use (tables,
arrays of tables, dotted and quoted keys, inline tables, strings,
integers, booleans, arrays) and writes it back in the same layout.
The values are ordinary enums, `Json` and `Toml`, so a document is
something you pattern match on.

```metaxu
from std.path import join, normalize, relpath;
from std.json import Json, member, to_json;
from std.toml import Toml, parse, to_toml, get_table, get_str;

fn main() -> int {
    print(normalize(join("pkgs/geom", "../util/./src")));
    print(relpath("/work/app/vendor/geom", "/work/app"));
    let @mut fields = Vec.new();
    fields.push(member("name", JStr("app")));
    fields.push(member("deps", JArr(Vec.new())));
    print(to_json(JObj(fields)));
    match parse("[package]\nname = \"app\"\n\n[dependencies]\ngeom = { git = \"https://x/geom\", rev = \"v1\" }\n") {
        Err(e) => print(e),
        Ok(doc) => {
            match get_table(doc, "package") {
                None => print("no package table"),
                Some(pkg) => match get_str(pkg, "name") {
                    None => print("no name"),
                    Some(name) => print(name)
                }
            };
            print(to_toml(doc))
        }
    };
    match parse("version = 1.5\n") { Err(e) => print(e), Ok(doc) => print("parsed") };
    0
}
```
```output
pkgs/util/src
vendor/geom
{"name":"app","deps":[]}
app
[package]
name = "app"

[dependencies]
geom = { git = "https://x/geom", rev = "v1" }

line 1: floats are not supported
```

## std.fs, std.process, std.env and std.io

Files, other programs, the environment and the error stream are
effects. Each operation has a runtime mapping on both engines (POSIX
calls natively, Python's `os` and `subprocess` on the interpreter), and
a failure is an ordinary catchable error whose text is the same on
both: `read: mx.toml: No such file or directory`. Because they are
effects, a handler in scope replaces the machine. This program writes
and reads a manifest and runs `git`, and nothing touches the disk or
starts a process: two handlers answer everything.

```metaxu
from std.fs import Fs, read_text, write_text, exists;
from std.process import Process, output_of;

fn index_of(names: Vec, name: string) -> int {
    let @mut i = 0;
    let @mut found = 0 - 1;
    while i < len(names) && found < 0 {
        if names[i] == name { found = i } else { () };
        i = i + 1
    }
    found
}

fn main() -> int {
    let @mut names = Vec.new();
    let @mut bodies = Vec.new();
    handle Fs with {
        write_bytes(path, bytes) -> {
            let k = index_of(names, path);
            if k < 0 { names.push(path); bodies.push(bytes) } else { bodies[k] = bytes };
            resume(())
        },
        read_bytes(path) -> {
            let k = index_of(names, path);
            if k < 0 { raise("virtual read: " + path + " was never written") } else { resume(bodies[k]) }
        },
        exists(path) -> resume(index_of(names, path) >= 0)
    } in {
        write_text("mx.toml", "[package]\nname = \"app\"");
        print(read_text("mx.toml"));
        print(exists("mx.lock"));
        if exists("mx.lock") { print(read_text("mx.lock")) } else { print("no lock yet") }
    };
    let @mut git = Vec.new();
    git.push("git");
    git.push("rev-parse");
    git.push("HEAD");
    handle Process with {
        run(argv, cwd) -> resume(1),
        status(h) -> resume(0),
        stdout(h) -> resume("0123abcd\n"),
        stderr(h) -> resume("")
    } in {
        match output_of(git, "") { Ok(t) => print("fake git said " + t.trim()), Err(e) => print(e) }
    };
    0
}
```
```output
[package]
name = "app"
0
no lock yet
fake git said 0123abcd
```

Without the handlers the same calls reach the real filesystem and the
real `git`. One thing to know when writing such a handler: a failure
raised inside a handler arm surfaces at the `handle` expression, not
inside a `try` that sits in the handled body (the arm runs in the
handler's frame), so a virtual filesystem reports a missing file
through `exists` or a Result rather than by raising. `std.env.args()` gives the command line (`metaxuc run
prog.mx -- a b`, or a native binary's own arguments), `std.env.lookup`
an environment variable as an Option, and `std.io.eprintln` writes to
stderr. `docs/io_runtime.md` is the contract for all four.

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
