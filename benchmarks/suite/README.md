# Metaxu showcase suite

Speed against C, plus the exhibits only Metaxu can write. Run it:

    uv run python benchmarks/suite/run_suite.py

Every race asserts OUTPUT EQUALITY between the Metaxu program and its
hand-written C twin (same clang, same -O2, same alignment pinning)
before a single timing is taken. Methodology inherited from
benchmarks/contention/: -falign-functions=64 everywhere, alternating
run order, medians.

## Measured (2026-08-19, this container, clang -O2, 15 rounds,
## post tail-resume trampoline AND inline Vec fast paths)

| benchmark | what it stresses | metaxu | C | ratio |
| --- | --- | --- | --- | --- |
| fib(35) | recursion, calls | 30.6 ms | 30.0 ms | **1.02x** |
| mandelbrot 800² | float ALU, branches | 49.0 ms | 49.6 ms | **0.99x** |
| orbit 20M steps | struct-per-step rebuild | 164.8 ms | 166.3 ms | **0.99x** |
| nsieve 3×2M | Vec indexing | 82.8 ms | 75.9 ms | **1.09x** |
| par_sum 40M ×(1+4 threads) | OS threads, join values | 59.2 ms | 56.8 ms | **1.04x** |
| pipeline 200k (metaxu-only) | 3-stage effects pipeline | 166.1 ms | — | ~830 ns/element through 3 handlers |

(Absolute ms move between runs on this shared container — an earlier
table was taken under load and read 2.5x slower on the C SIDE of nsieve
— so trust the paired, interleaved ratios, not the absolute columns.)

Reading the results honestly:

- **orbit at 0.99x is the ergonomics headline**: the Metaxu version
  rebuilds an immutable 4-field struct every step (value semantics, no
  mutation in sight) and costs the same as C mutating doubles in place —
  SROA/mem2reg dissolve the abstraction completely.
- **nsieve at 1.09x** used to sit at ~1.4x when every `v[i]` was an
  opaque runtime call. The inline Vec fast paths (see
  benchmarks/diagnostics/) close most of that; the remaining ~9% is
  the checked-indexing price (null + bounds + contended-write guard) —
  a chosen trade: out-of-bounds is a loud error and data races on
  crossed vecs are caught.
- **par_sum at ~1.0x**: the shared work loop reached parity once its
  Vec accesses inlined; what remains is ~2.5 ms/thread of flat spawn
  cost (1 MiB coroutine stack setup + effect scope init) that amortizes
  with bigger work items. Four threads still turn 40M elements around
  3.4x faster than the serial pass inside the same binary.

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
  elements now — and the figure above works out to ~280 ns per handler
  crossing (each element crosses filter, map and sum), the measured
  price of a handler dispatch. Writing real programs remains this
  compiler's best bug detector.

## Adding a benchmark

One `.mx` file here + (for races) one `.c` twin printing identical
output + a line in run_suite.py's RACES/SOLO list. Keep outputs
deterministic and print only final results — no IO in hot loops.
