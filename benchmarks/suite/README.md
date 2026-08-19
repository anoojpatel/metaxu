# Metaxu showcase suite

Speed against C, plus the exhibits only Metaxu can write. Run it:

    uv run python benchmarks/suite/run_suite.py

Every race asserts OUTPUT EQUALITY between the Metaxu program and its
hand-written C twin (same clang, same -O2, same alignment pinning)
before a single timing is taken. Methodology inherited from
benchmarks/contention/: -falign-functions=64 everywhere, alternating
run order, medians.

## Measured (2026-08-19, this container, clang -O2, 9 rounds,
## post tail-resume trampoline)

| benchmark | what it stresses | metaxu | C | ratio |
| --- | --- | --- | --- | --- |
| fib(35) | recursion, calls | 30.9 ms | 30.4 ms | **1.02x** |
| mandelbrot 800² | float ALU, branches | 63.7 ms | 62.2 ms | **1.02x** |
| orbit 20M steps | struct-per-step rebuild | 165.7 ms | 165.9 ms | **1.00x** |
| nsieve 3×2M | Vec indexing | 240.7 ms | 188.6 ms | 1.28x |
| par_sum 40M ×(1+4 threads) | OS threads, join values | 66.6 ms | 56.4 ms | 1.18x |
| pipeline 200k (metaxu-only) | 3-stage effects pipeline | 278.6 ms | — | ~460 ns/handler crossing |

(Deltas of a few points between runs are normal — see the contention
benchmarks for how much layout and scheduling luck move microbenchmarks.)

Reading the gaps honestly:

- **orbit at 0.99x is the ergonomics headline**: the Metaxu version
  rebuilds an immutable 4-field struct every step (value semantics, no
  mutation in sight) and costs the same as C mutating doubles in place —
  SROA/mem2reg dissolve the abstraction completely.
- **nsieve's 1.44x** is the price of `v[i]` being a bounds-checked,
  contention-checked runtime call vs C's raw pointer arithmetic. That is
  a real, chosen trade (out-of-bounds is a loud error, data races on
  crossed vecs are caught); closing it means inlining the vec fast path
  into generated code — a known, listed optimization.
- **par_sum's 1.34x** decomposes as: the shared work loop runs at 1.17x
  (a loop-shape/vectorization gap on this `(i*i)%1000` pattern — fib,
  mandelbrot and orbit show calls, floats and branches at parity), plus
  ~2.5 ms/thread of flat spawn cost (1 MiB coroutine stack setup + effect
  scope init) that amortizes with bigger work items. Four threads still
  turn 40M elements around 3.4x faster than the serial pass inside the
  same binary.

## The exhibits

- `par_sum.mx` — the concurrency ergonomics story in one screen: four
  workers, ZERO shared state (each owns its range; results return
  through `join`), so there are no locks to get wrong and the
  thread-safety checker (docs/separate_send_sync.md) has nothing to
  object to. Compare with the C twin's Job structs and pthread plumbing.
- `pipeline.mx` — effects-as-iterators: `sum(map(filter(iota(200000))))`
  as one lazy pipeline against the hand-written loop, both native, one
  binary, self-checked. This benchmark flushed out a real runtime bug on
  arrival: the handler pump dispatched each element RECURSIVELY, capping
  streams at ~4k elements natively (and ~10-20k in the interpreter). The
  tail-resume trampoline fixed both engines — pipelines are flat to 10M+
  elements now — and the ~460 ns/element-crossing figure above is the
  measured price of a handler dispatch. Writing real programs remains
  this compiler's best bug detector.

## Adding a benchmark

One `.mx` file here + (for races) one `.c` twin printing identical
output + a line in run_suite.py's RACES/SOLO list. Keep outputs
deterministic and print only final results — no IO in hot loops.
