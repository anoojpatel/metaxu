# The Metaxu Book

This is the language documentation: every part of Metaxu, explained
with programs that run. It is written in the same discipline as the
compiler itself. Each ```` ```metaxu ```` block in these pages is
extracted by `test_book_examples.py` and executed through the real
pipeline on every test run, and its printed output is pinned. If the
language changes, the book breaks loudly.

## Contents

1. [Getting started](01-getting-started.md) — install, run a program,
   what the pipeline does with it.
2. [Functions and values](02-functions-and-values.md) — `fn`, `let`,
   mutability, blocks as expressions, control flow.
3. [Structs and enums](03-structs-and-enums.md) — product and sum
   types, generics, construction and field access.
4. [Pattern matching](04-pattern-matching.md) — `match`, binders,
   literals, exhaustiveness.
5. [Vectors and collections](05-vectors-and-collections.md) — `Vec`,
   fixed-size `vector[T, N]`, strings, slicing.
6. [Types and inference](06-types-and-inference.md) — what you never
   have to write, what the checker rejects, and why.
7. [Traits](07-traits.md) — `trait`, `implements`, dispatch on the
   runtime type.
8. [Effects](08-effects.md) — `perform`, `handle`, `resume`; defaults;
   streams and generators built from them.
9. [Errors](09-errors.md) — `try`/`catch`, `std.option`, `std.result`.
10. [Modes and memory](10-modes-and-memory.md) — locality, mutability,
    linearity; the borrow checker; no lifetimes anywhere.
11. [Threads and sharing](11-threads-and-sharing.md) — `Thread` and
    `Mutex` effects, inferred thread-safety, `std.sync`.
12. [The standard library](12-the-standard-library.md) — a tour of
    `std/`, module by module.
13. [GPU tiles](13-gpu-tiles.md) — `Tile`, kernels, the `Gpu` effect,
    Metal.
14. [Compiling to native](14-compiling-to-native.md) — clang, the
    differential contract, measured performance.
15. [Modules and imports](15-modules-and-imports.md) — `import`,
    `export`, visibility, multi-file programs, name precedence.
16. [Unsafe and FFI](16-unsafe-and-ffi.md) — `unsafe`, extern C,
    the strict simulated heap, the native boundary.
17. [Inside the compiler](17-inside-the-compiler.md) — the pipeline
    stage by stage, the house rules, how to poke at it.
18. [Algebraic subtyping](18-algebraic-subtyping.md) — how inference
    decides: flows, polarity, the class-constraint algebra, the
    solver, and what is built versus sketched.

## How the examples work

A plain `metaxu` fence is a complete program. The `output` fence after
it is the program's exact stdout; the test harness runs the program and
compares byte for byte.

An `error` example is a program the compiler must reject; its `output`
fence holds a fragment of the diagnostic. Rejections are part of the
language and get pinned like everything else.

A `norun` fence is display-only (used for multi-file layouts the
single-source harness can't run). The harness caps how many of these
the book may contain, so the escape hatch stays narrow.

## Influences

The shape of this book borrows from the
[Rust book](https://doc.rust-lang.org/book/), the
[Swift guide](https://docs.swift.org/swift-book/),
[Ante's documentation](https://ante-lang.org), and Jane Street's
[OxCaml](https://oxcaml.org) writing on modes. The ideas in chapter 10
owe the most to the last two.
