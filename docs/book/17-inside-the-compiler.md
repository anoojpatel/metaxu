# Inside the compiler

The last chapter of this book is about the machine that ran every
example in it. You don't need any of this to write Metaxu. You need
it the day a diagnostic confuses you, or you want to add a stage, or
you're deciding whether to trust a language whose compiler is written
in Python. Here is the whole pipeline, one pass at a time, and the
three house rules that keep it honest.

```
source -> parser -> module resolution -> desugar -> typecheck
       -> (SimpleSub inference + borrow/mode checking)
       -> HIR -> MIR -> interpreter          (the semantics)
                     -> LLVM -> clang        (native, differentially tested)
                     -> MSL -> Metal via MLX (GPU tile kernels)
```

Take one program along for the ride:

```metaxu
fn double(x: int) -> int {
    x * 2
}

fn main() -> int {
    print(double(21));
    0
}
```
```output
42
```

## Parser

The front end is an LALR grammar with a lexer and parser generated in
Python. The grammar's ambiguity budget is pinned by a test: exactly
163 shift/reduce conflicts, each one documented at the point in the
test file where its resolution is argued for. Add a production that
introduces a conflict and the count fails until you've written down
why shifting is right.

Some surface syntax never survives the parser. Match guards are the
clearest case: `pattern if cond => body` is rewritten during parsing
into nested matches, with the scrutinee bound once through an
immediately applied lambda so its effects can't run twice. By the
time any later stage looks at the tree, guards don't exist. The
generated binding gets a `__`-prefixed name, which is why chapter 15
reserves that prefix for the compiler.

## Module resolution

Imports resolve next: `std` paths, sibling files, in-file `module`
blocks, export lists (chapter 15). This stage runs before any type
is inferred, so a private import or a reserved name dies here with a
`ModuleError` or `ReservedNameError` and the type checker never sees
the program.

## Desugar

The desugarer flattens conveniences into the core language:
qualified calls get canonical names, f-strings become concatenation,
loops and early exits take their handler-based forms. Trait
coherence is checked here too. Two `implement` blocks for the same
trait and type raise a typed error (chapter 7 pins the fragment
"more than one implement block") before dispatch tables are built.

## Typecheck

Two checkers share this stage. Inference is SimpleSub-based:
subtyping constraints flow through the program in both polarities,
and a conflict is reported as the two requirements that collided.
The borrow and mode checker of chapter 10 runs alongside it, walking
the same tree and accumulating structured errors, each tagged with a
kind such as `immutable-rebind` or `const-field-write`.

Diagnostics carry a location and an excerpt. For a file the location
starts with its path; a program handed to the pipeline as a string is
`<mem>`. Misspell a function and the checker names its nearest
neighbor:

```metaxu error
fn helper() -> int { 3 }

fn main() -> int {
    helpr();
    0
}
```
```output
<mem>:4:5: undefined function 'helpr'; did you mean 'helper'?
  4 |     helpr();
    |     ^~~~~~~
```

A type conflict reads as the two facts that can't both hold. The caret
sits on the value that made the conflict apparent, the last one the
checker saw; here that is the string, not the `1`:

```metaxu error
fn main() -> int {
    let n = 1 + "a";
    print(n);
    0
}
```
```output
<mem>:2:17: type mismatch: one value is required to be Int and String
  2 |     let n = 1 + "a";
    |                 ^~~
```

Those two blocks are harness-pinned like every other rejection in
this book: the exact location, message, and caret excerpt are what
the compiler prints today, and the test suite fails if any of it
drifts.

## HIR and MIR

The checked tree lowers to HIR, a smaller expression language with
scoping resolved, and then to MIR, the shared instruction form every
back end consumes. MIR is where effect handlers become explicit
continuation plumbing and where the emitter uses effect classes
(chapter 8) to decide which functions need coroutine machinery.

## The interpreter is the spec

`mir_interp` executes MIR directly, and its behavior is the
definition of the language. When a question about semantics comes
up, the answer is whatever the interpreter does, written down and
pinned by a test. This is also where small decisions become law: the
interpreter formats booleans as `1` and `0` because native code
erases booleans to integers, and one formatting had to be the
observable behavior of both engines. The strict heap of chapter 16
lives here too.

## Native and Metal

`codegen_llvm` emits LLVM IR from MIR, clang compiles it at `-O2`,
and the result links against the C runtime in
`src/metaxu/runtime/native/`. Chapter 14 covers the entry points and
the measured cost. Tile kernels take the third exit: MIR to MSL,
dispatched on Metal through the MLX shim of chapter 13.

## The house rules

The interpreter is the spec. Back ends conform to it; it conforms to
nothing but its tests. A disagreement between engines is by
definition a back end bug or a spec change.

Native must match byte for byte. Stdout, exit codes, error messages,
float formatting: the differential suite compares them all, and a
mismatch fails the gate even when both outputs look plausible.

Demotion over wrong code. When the LLVM back end meets a construct
it can't yet prove it compiles correctly, it demotes: the program is
refused for native compilation with a reasoned placeholder naming
the construct. A smaller language compiled right beats a bigger one
compiled wrong, and the demotion list shrinks release by release.

## The gates

The rules are enforced by machinery, not by review culture. The unit
suite runs a couple of thousand tests, most of them small programs
with pinned output on one engine or both. On top of it sit two
gates driven by `scripts/run_examples.py`: every program in
`examples/` must interpret to its pinned output, and every program
the native back end claims must compile and match. A change lands
when the suite and both gates are green.

The book is part of the same machinery.
`test_book_examples.py` extracts every `metaxu` fence from these
chapters, runs it through the real pipeline, and compares stdout
against the `output` fence byte for byte; `error` fences must be
rejected with a diagnostic containing their pinned fragment. The
harness also counts `norun` fences against a hard budget, so
display-only examples stay the rare exception. Documentation that
can't drift from the implementation was the cheapest correctness
tool this project bought.

## Poking at the pipeline

Every stage's output is inspectable from the command line:

```bash
uv run metaxuc emit prog.mx --stage ast    # the frozen syntax tree, as JSON
uv run metaxuc emit prog.mx --stage hir
uv run metaxuc emit prog.mx --stage mir
uv run metaxuc emit prog.mx --stage llvm   # what `metaxuc build` hands to clang
```

The whole pipeline is also a Python library. `run_pipeline_ctx` runs a
source string through the front end and hands back a context object
holding each intermediate form:

```python
from metaxu.compiler.pipeline import run_pipeline_ctx

ctx = run_pipeline_ctx("""
fn main() -> int {
    print(21 * 2);
    0
}
""")
print(ctx.mir.dump())
```

```
fn main() -> int
bb0:
    %0 = mul 21, 2
    %1 = call print(%0)
    ret 0
```

The dump above is abridged, and this fence is display-only: the MIR's
exact shape shifts as the compiler does, and pinning it would turn
every optimization into a book edit. The same context object exposes
the parse tree, the typed tree, and the HIR; the test helpers the
book harness uses (`interp_run` and friends in
`test_codegen_llvm.py`) are thin wrappers over it.

That's the machine. Fourteen chapters of examples ran through it to
get onto these pages, and they'll run through it again the next time
anyone touches the compiler. If one of them ever stops matching,
the book breaks loudly, which is exactly the arrangement.
