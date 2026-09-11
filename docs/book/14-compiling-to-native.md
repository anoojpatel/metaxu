# Compiling to native

The interpreter defines what every Metaxu program means. The native
backend's job is to reproduce that meaning exactly: it lowers MIR to
LLVM IR, hands the IR to clang at `-O2`, and links the C runtime in
`src/metaxu/runtime/native/`. This chapter is about the contract
between the two engines, what the backend refuses to guess at, and
what the compiled code costs.

## Two ways in

The example runner drives both engines over `examples/`:

```bash
uv run python scripts/run_examples.py --stage run   # interpret
uv run python scripts/run_examples.py               # full gate: also compile and diff
```

From Python, the entry points are direct:

```bash
uv run python -c "
from metaxu.compiler.codegen_llvm import emit_llvm_from_source, llvm_run

src = open('examples/hello.mx').read()
ir = emit_llvm_from_source(src)   # LLVM IR, as text
print(ir.splitlines()[0])
llvm_run(src)                     # clang -O2, link the runtime, execute
"
```

`emit_llvm_from_source` gives you the IR to read; `llvm_run` takes
the program all the way to a process and runs it.

## The differential contract

Native output must match the interpreter byte for byte: stdout, exit
code, error messages, float formatting. Not "equivalent". Identical.
The differential tests compare the bytes, and every plain fence in
this book runs under that regime.

Float formatting is where such contracts usually die, so it's pinned
directly:

```metaxu
fn main() -> int {
    print(6.0 * 2.0);
    print(1.0 / 100000.0);
    0
}
```
```output
12.0
1e-05
```

Both engines format `12.0` with its trailing zero and flip to
exponent form at the same magnitude. The formatter is one algorithm
implemented twice and tested against itself.

Error messages are under the same contract. A runtime raise carries
the same text compiled as interpreted, which is why a book example
can pin one:

```metaxu
fn main() -> int {
    let @mut v = Vec.new();
    try {
        v.pop();
        print("popped")
    } catch e {
        print(e)
    };
    0
}
```
```output
pop: Vec is empty
```

## Demotion

The backend does not compile every program, and that is a feature.
When the emitter cannot prove it can match the interpreter, it
demotes the construct: what you get is a reasoned placeholder that
names what it couldn't prove, never code that runs and prints
something almost right. Printing a whole `Vec` is the standing
example. The interpreter has a repr for any element type; the native
runtime doesn't yet, so a native `print(v)` demotes honestly instead
of diverging. Demotions shrink as the runtime grows, but the rule
they enforce is permanent: the two engines never disagree silently.

## Microbenchmarks against C

The suite in `benchmarks/` times five small kernels against
hand-written C twins, both sides through the same clang at `-O2`:

| kernel     | native / C |
|------------|------------|
| fib        | 1.02x      |
| mandelbrot | 1.00x      |
| orbit      | 1.03x      |
| nsieve     | 1.09x      |
| par_sum    | 1.06x      |

Read those ratios as "in the same neighborhood as the C twin",
nothing finer. These are microbenchmarks, and microbenchmarks reward
whoever wrote them: five tiny kernels that fit in cache, with C
twins that are straightforward ports rather than expert-tuned C. A
determined person could tilt either column. Nothing in the set
exercises heavy allocation, large working sets, or the shape of a
real program. The table exists to pin regressions and keep
performance in view while the language grows; it's a starting point
for performance work, not a scoreboard against C.

What makes the ratios meaningful at all is the methodology. Both
columns get the same clang and flags, function alignment is pinned
so code layout can't masquerade as a result, the two binaries run in
alternating order, each number is the median of 15 runs, and a run
only counts after an output-equality gate: the twins must print the
same bytes before anyone times them.

Two costs the table doesn't show. Bounds-checked `Vec` loops land
around 1.3x to 1.7x of the unchecked C equivalent, because the
backend does not elide bounds checks yet. And crossing a handler, a
`perform` that suspends to a handler arm and resumes, costs about
280ns compiled. Ordinary calls, arithmetic, and runtime-bound
effects like `Thread.spawn` don't pay it; it's the price of the
general mechanism from chapter 8, paid only where you use it.

## Sanitizers and leaks

The differential suite runs under AddressSanitizer, scoped to the
compiled program and the C runtime: heap misuse in generated code
fails CI. Leak checking is scoped more carefully, because the
runtime leaks by design in one place. Immortal allocations, interned
strings, module constants, and the effect tables live for the whole
process and are never freed; freeing them at exit would be work the
OS is about to do anyway. Those are suppressed by name. Everything
else must be clean, so a missing free in a lowering shows up as a
red build rather than a slow rumor.

## The recursion budget

Deep recursion is bounded on purpose. Both engines count frames and
raise `RecursionLimitExceeded` at 100000, instead of the interpreter
dying of the host's RecursionError while native code overflows the
stack and takes a signal. One budget, one error, same wording, and
because it's an ordinary raise you can `try` around a deliberately
deep call. Loops don't pay anything for this; only call depth is
counted.

The next chapter leaves single files behind: modules, imports, and
what `export` means across them.
