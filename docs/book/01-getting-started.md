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
`examples/hello.mx` is one line of `print`. But everything in this
book uses `main`, because that's what the native backend compiles.)

Two things to notice already. Statements end with `;`, but the last
expression of a block doesn't need one, and its value *is* the block's
value: that final `0` is `main`'s return. And `+` concatenates strings
when both sides are strings. There's no implicit conversion; gluing a
number onto a string takes `.to_string()`, which chapter 2 covers.

## Running it

Two engines run the same program. The example runner drives both; drop
your file in `examples/` or point the pipeline at it from Python:

```bash
uv run python scripts/run_examples.py --stage run   # interpret examples/
uv run python scripts/run_examples.py               # full pipeline gate
```

The native path emits LLVM IR, hands it to clang at `-O2`, and links
the C runtime in `src/metaxu/runtime/native/`. Its output must match
the interpreter byte for byte: stdout, exit code, error messages,
float formatting. Differential tests enforce that contract, and it's
the reason this book can promise its examples behave the same compiled
as interpreted. Chapter 14 shows the native entry points directly.

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
