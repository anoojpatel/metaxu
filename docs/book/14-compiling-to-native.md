# Compiling to native

Everything so far ran on the reference interpreter. The native path
takes the same MIR, emits an LLVM IR module as text, hands it to clang
at `-O2`, and links the C runtime objects in
`src/metaxu/runtime/native/` (every binary links `-pthread`). The API
is two calls:

```bash
uv run python - <<'EOF'
from metaxu.compiler.pipeline import emit_llvm_from_source
from metaxu.compiler.llvm_run import compile_and_run

ir = emit_llvm_from_source(open("hello.mx").read())
exit_code, stdout = compile_and_run(ir, "main")
print(stdout, end="")
EOF
```

and the example runner drives the whole `examples/` directory through
it:

```bash
uv run python scripts/run_examples.py               # pipeline gate
uv run python scripts/run_examples.py --stage run   # execution gate
```

`emit_llvm_from_source` raises the same `TypeCheckError` and
`BorrowCheckError` diagnostics as the interpreter path; nothing gets
into codegen that the strict front end rejected.

## The differential contract

The native backend's output must match the interpreter byte for byte:
stdout, exit codes, error messages, and float formatting. The test
suite enforces this with differential assertions that run every
program both ways and compare.

Float formatting is where such contracts usually leak. The interpreter
prints floats as Python reprs (shortest round-trip form), and native
`print` routes through a C implementation of the same algorithm,
because an early tile test caught native `%g` printing `12` where the
interpreter printed `12.0`:

```metaxu
fn main() -> int {
    print(12.0);
    print(0.1);
    print(1.0 / 3.0);
    print(0.00001);
    0
}
```
```output
12.0
0.1
0.3333333333333333
1e-05
```

Error messages are under the same contract. This program prints the
identical string interpreted or compiled, which is what makes error
handling portable across engines:

```metaxu
fn main() -> int {
    let @mut v = Vec.new();
    let msg = try { v.pop(); "popped" } catch e { e };
    print(msg);
    0
}
```
```output
pop: Vec is empty
```

The book's own harness runs every example in these pages through the
interpreter; the differential suite is what extends each claim to the
binary.

## Demotion: never wrong code

The backend doesn't compile every construct yet. Its rule for the gaps
is the house rule: anything it can't prove it lowers correctly becomes
a reasoned placeholder, a comment in the emitted IR naming the
function and the reason, never silently wrong code. A tile whose shape
isn't statically resolvable, for example, demotes with
`not statically resolvable`; a spawn whose capture kinds codegen can't
enumerate demotes with a reason string about contention marks. Calling
a demoted function is a loud runtime abort that names it. The
differential tests assert zero placeholders on everything claimed
supported, so "supported" is a tested word, not a hopeful one.

## Measured performance

The numbers, from `benchmarks/suite/` (this container, clang `-O2`):

| benchmark | stresses | ratio vs C |
| --- | --- | --- |
| fib(35) | recursion, calls | 1.02x |
| mandelbrot 800² | float ALU, branches | 0.99x |
| orbit 20M steps | struct-per-step rebuild | 0.99x |
| nsieve 3x2M | Vec indexing | 1.09x |
| par_sum 40M | OS threads, join values | 1.04x |

Bounds-checked Vec loops in general sit around 1.3 to 1.7x C, the
price of checked indexing (null test, bounds test, contended-write
guard on every access), after inline fast paths removed the old
call-per-index overhead. An effect handler crossing costs about
280 ns, measured by pushing 200k elements through a three-handler
stream pipeline.

The methodology matters as much as the digits. Each benchmark has a
hand-written C twin, and output equality is asserted before a single
timing is taken. Function alignment is pinned (`-falign-functions=64`)
because the contention benchmarks proved that unaligned builds swing
±30% on linker placement luck, enough to invert a comparison.
Runs alternate between the two sides so machine drift hits both
equally, and medians are reported. Trust the paired ratios, not the
absolute milliseconds, which move between runs on a shared container.

Two of the rows are the point of the table. `orbit` rebuilds an
immutable four-field struct 20 million times, value semantics with no
mutation in sight, and costs the same as C mutating doubles in place:
the optimizer dissolves the abstraction completely. And `nsieve`'s 9%
is a chosen trade, not an aspiration to zero: out-of-bounds is a loud
error and unlocked writes to crossed vecs are caught, in production
builds.

## Memory claims and how they're checked

Native memory is mode-based: stack allocation by default, `@global`
values on the heap with real `malloc`/`free`, write-once boxes for
values that cross effect boundaries. Some of those boxes are immortal
by design: thread handles, mutexes, and spawn-crossing environments
are never freed, which is what makes double-join a flag check instead
of a use-after-free and lets handles be copied freely as opaque words.

ASan runs are scoped to that documented contract. Where frees are
claimed, tests run under `-fsanitize=address` with full leak checking.
Where leak-by-design applies (thread programs and their immortal
handles), they run with `detect_leaks=0`, checking for corruption but
not for the leaks the contract already admits. Thread programs also
run under `-fsanitize=thread`; the mutex-protected paths are
TSan-clean, and TSan non-vacuity was verified by deleting the locks
and watching it report.

## The recursion budget

One divergence between host and language is papered over deliberately.
A Metaxu frame costs about 4 Python frames in the interpreter (up to
~24 for effect-and-closure-heavy code), so deep recursion would hit
Python's own `RecursionError` at an arbitrary, host-dependent depth.
Instead, every entry point (the interpreter and the compile-time
phases both) installs a 100k-frame budget and restores the host limit
after. Overrunning it is `RecursionLimitExceeded`: a fatal runtime
error that `try` does not catch, or a `CompileError` when a
compile-time phase recurses too deep. You get a Metaxu-named
diagnostic either way, never a raw host traceback.
