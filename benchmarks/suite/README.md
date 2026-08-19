# Metaxu showcase suite

Speed against C, plus the exhibits only Metaxu can write. Run it:

    uv run python benchmarks/suite/run_suite.py

Every race asserts OUTPUT EQUALITY between the Metaxu program and its
hand-written C twin (same clang, same -O2, same alignment pinning)
before a single timing is taken. Methodology inherited from
benchmarks/contention/: -falign-functions=64 everywhere, alternating
run order, medians.

## Measured (2026-08-19, this container, clang -O2, 9 rounds)

| benchmark | what it stresses | metaxu | C | ratio |
| --- | --- | --- | --- | --- |
| fib(35) | recursion, calls | 30.6 ms | 30.8 ms | **0.99x** |
| mandelbrot 800² | float ALU, branches | 62.9 ms | 61.7 ms | **1.02x** |
| orbit 20M steps | struct-per-step rebuild | 166.0 ms | 167.0 ms | **0.99x** |
| nsieve 3×2M | Vec indexing | 230.9 ms | 159.8 ms | 1.44x |
| par_sum 40M ×(1+4 threads) | OS threads, join values | 72.4 ms | 53.8 ms | 1.34x |

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
- `pipeline.mx` — effects-as-iterators: `sum(map(filter(iota(n))))` as
  one lazy pipeline against the hand-written loop, both native, one
  binary, self-checked. (Temporarily sized pending the tail-resume
  trampoline fix this benchmark itself flushed out — writing real
  programs remains this compiler's best bug detector.)

## Adding a benchmark

One `.mx` file here + (for races) one `.c` twin printing identical
output + a line in run_suite.py's RACES/SOLO list. Keep outputs
deterministic and print only final results — no IO in hot loops.
