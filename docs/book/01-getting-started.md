# Getting started

Metaxu is a systems language with a Python front end, an LLVM native
backend, and a strict reference interpreter that defines what every
program means. You need Python 3.11+, clang, and
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/anoojpatel/metaxu.git
cd metaxu
uv sync --all-groups
```

## The first program

A program declares `fn main() -> int`. Its return value is the process
exit code. `print` writes a line to stdout.

```metaxu
fn main() -> int {
    let who = "Metaxu";
    print("hello, " + who);
    0
}
```
```output
hello, Metaxu
```

(A bare `.mx` file of top-level statements also runs as a script:
`examples/hello.mx` is one line of `print`, and the compiler makes
those statements the body of `main`. A file gets one or the other; a
top-level statement next to a declared `main` is a compile error, not
something that silently runs first or not at all. Everything in this
book uses `main`.)

Two things to notice already. Statements end with `;`, but the last
expression of a block doesn't need one, and its value *is* the block's
value: that final `0` is `main`'s return. And `+` concatenates strings
when both sides are strings. There's no implicit conversion; gluing a
number onto a string takes `.to_string()`, which chapter 2 covers.

## Running it

Save the program as `hello.mx`. The `metaxuc` command installed by
`uv sync` is the front door to the compiler; inside the repository,
call it through `uv run`.

```bash
uv run metaxuc run hello.mx        # interpret it
uv run metaxuc build hello.mx      # compile it to ./hello
./hello
```

`run` executes the program in the interpreter, and the process exits
with whatever `main` returned. `build` emits LLVM IR, hands it to clang
at `-O2`, links the C runtime in `src/metaxu/runtime/native/`, and
writes an executable named after the file (`-o` picks another path,
`--keep-ir` also leaves the `.ll` next to it). Both print
`hello, Metaxu`, and both exit 0.

Two more subcommands are worth knowing on day one. `check` runs the
parser, the type checker and the borrow checker and stops there, which
is the fast loop while you write. `emit` prints one intermediate form:

```bash
uv run metaxuc check hello.mx
uv run metaxuc emit hello.mx --stage mir     # ast, hir, mir, clif or llvm
```

Every subcommand runs the same strict front end as the test suite, so a
program that `check` accepts is a program the compiler accepts.

The native path's output must match the interpreter byte for byte:
stdout, exit code, error messages, float formatting. Differential tests
enforce that contract, and it's the reason this book can promise its
examples behave the same compiled as interpreted. Chapter 14 shows the
native entry points directly.

The repository's own examples run through a separate script, which is
also the merge gate: every file under `examples/` must compile, and
every one with an entry point must run.

```bash
uv run python scripts/run_examples.py --stage run   # interpret examples/
uv run python scripts/run_examples.py               # full pipeline gate
```

## Comments

Both `#` and `//` start a line comment. The standard library uses `#`;
several examples use `//`. Pick one per file and stay with it.

```metaxu
# a comment
// also a comment
fn main() -> int {
    print(41 + 1);  # prints 42
    0
}
```
```output
42
```

## What the pipeline does with your file

```
source -> parser -> module resolution -> desugar -> typecheck
       -> (SimpleSub inference + borrow/mode checking)
       -> HIR -> MIR -> interpreter          (the semantics)
                     -> LLVM -> clang        (native, differentially tested)
                     -> MSL -> Metal via MLX (GPU tile kernels)
```

Undefined names, type conflicts, and mode violations are compile
errors, not runtime surprises. The checker would rather stop you than
guess:

```metaxu error
fn main() -> int {
    print(1 + "a");
    0
}
```
```output
type check failed
```

The rest of the book walks the language from the bottom up. Chapters 2
through 7 are the core language; 8 through 11 are what makes Metaxu
distinctive (effects, modes, inferred thread-safety); 12 through 14
tour the library, the GPU story, and the native backend.
