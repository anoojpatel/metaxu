# Emission diagnostics

Eight tiny paired programs, each isolating ONE code shape the LLVM
backend emits, raced against a hand-written C twin doing identical work
(same clang, same `-O2`, same `-falign-functions=64`, output equality
asserted before timing, alternating run order, medians). Where
`benchmarks/suite/` answers "how fast are real programs", each pair here
answers "what does THIS emission pattern cost" — so a win or a
regression in the emitted IR moves one row instead of blending into a
whole-program number.

    uv run python benchmarks/diagnostics/run_diagnostics.py

## Measured (2026-08-19, this container, clang 18 -O2, ratio = metaxu/C)

| pattern | isolates | pre-inlining | now |
| --- | --- | --- | --- |
| vecread  | unit-stride Vec reads (sum 5M x4)  | 2.57x | **~1.5x** |
| vecwrite | unit-stride Vec writes (fill 5M x4)| 2.86x | **~1.7x** |
| vecpush  | append-heavy growth (push 5M x4)   | 1.68x | **~1.3x** |
| modloop  | pure integer ALU loop              | 1.00x | 1.00x |
| closure  | indirect calls through a lambda    | 0.91x | ~0.88x |
| enum     | construct+match 3-case enum in loop| 0.99x | 1.00x |
| string   | int -> string -> len               | 1.44x | ~1.45x |
| struct   | by-value struct rebuild in loop    | 1.00x | 1.00x |

(Absolute times on this shared 4-CPU container drift between runs;
the paired, interleaved ratio is the stable statistic. closure BELOW
1.0 is real: pinned closures beat C's opaque function pointers.)

## What moved the Vec rows

The pre-inlining columns were the motivation for the inline Vec fast
paths in `codegen_llvm.py`: every `v[i]` was an opaque runtime call,
which blocks register caching, LICM and strength reduction around the
loop. The fix landed in three pieces, each measured (see
`run_diagnostics.py`'s docstring for the per-step numbers):

1. **Inline hot path** against the mx_vec header (len @0, cap @8,
   data @16, contended @24), with every miss branching to the runtime so
   diagnostics stay byte-identical.
2. **noreturn fail terminators** for get/set/pop/len misses (which are
   always failures — push's miss includes normal growth and still calls
   the full op). A maybe-returning call in the loop clobbers every
   header load; `noreturn` + `memory(read, inaccessiblemem: readwrite)`
   is an honest contract that lets LICM hoist.
3. **TBAA tags** separating header words from element words (always
   distinct allocations by the runtime's contract), so element stores
   don't pin the hoisted len/data loads. clang then unrolls 4x and
   batches the bounds checks.

The residual ~1.3–1.7x on the Vec rows is the checked-indexing price —
null test + bounds check (+ contended-write guard on mutation, see
`docs/contention_as_permission.md`) per element vs C's raw pointer
arithmetic. That is a chosen trade, not overhead to engineer away in
emission: closing it means eliminating checks (iterator fusion, unsafe
indexing), which is a language-design question.

`string`'s ~1.45x is allocation semantics (every `to_string` allocates;
C reuses a stack buffer) — a separate, known gap.

## Bugs this harness caught

Racing the inline emission end-to-end surfaced two real defects the
all-call emission had been hiding — see the commit that landed the fast
paths: a set-oob message divergence from the interpreter, and a
try-body capture soundness hole (read-before-rebind captures dropped
from the site env) that had only survived by undefined behavior.
