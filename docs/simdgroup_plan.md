# Stage 2 execution-model mapping: simdgroups and threadgroup memory

Design memo, no implementation. How Metaxu's per-pid kernel model maps
onto Metal threadgroups and simdgroups so that `simdgroup_float8x8`
matrix ops and threadgroup-memory tiling (Stage 2 in docs/gpu_tiles.md)
become usable. Grounded in how the code works today: `emit_msl.py`,
`std/gpu.mx`, `metal_launch.py`.

## Where we start

The model today (Stage 1, all landed):

* One launch instance is one device thread. `std/gpu.mx` defines
  `Gpu.launch(n, f)` with the reference handler `run_grid`: instances
  run sequentially in pid order on the CPU. `metal_launch.py` dispatches
  `grid=(n, 1, 1)`, `threadgroup=(min(n, 32), 1, 1)`, and the emitted
  body derives `pid` from `thread_position_in_grid.x`. The threadgroup
  size is a throughput knob, not semantics: no two instances ever
  communicate.
* Tiles are thread-private. `emit_msl.py` declares every tile as a
  plain stack array (`long/float/half name[N]`) and emits whole-tile
  loops for every op inside a single thread.
* The execution contract is snapshot reads plus mask-merged writes:
  kernels read `<b>_in` (the launch-entry snapshot), write `<b>_out`
  with a written-mask `<b>_wm`, and the host merges. Cross-instance
  read-after-write within one launch is outside the contract.
* The C++ shim (`MslKernel.cpp_wrapper`, compiled with
  `-ffp-contract=off`) runs the same body sequentially over the grid
  and is the bit-exact leg; the mlx leg is tolerance-compared for
  floats because Metal compiles fast-math.

One load-bearing property of the current subset: every scalar an
emitted kernel branches on is deterministic in `pid` and the launch
snapshot. Int scalars come from `pid`, integer literals, arithmetic on
those, `rows`/`cols` constants, and `Tile.sum` of an int tile (so
buffer contents can steer control flow); float scalars are rejected in
binops. Nothing in the model is random or timing-dependent, and no
scalar mentions a lane. That fact does the heavy lifting below.

## (a) The mismatch

`simdgroup_float8x8` ops execute cooperatively: the 32 threads of one
SIMD group each hold a fragment of the 8x8 matrix, and
`simdgroup_load` / `simdgroup_multiply_accumulate` /
`simdgroup_store` are collective calls that every thread of the group
must reach together. Threadgroup-memory tiling is the same shape:
threads of a threadgroup stage tiles into `threadgroup` storage,
`threadgroup_barrier`, then read each other's staged data.

Our instances are the opposite: each one owns whole tiles privately and
never synchronizes with a neighbor. Mapping one instance to one thread
leaves a simdgroup op with no group to cooperate across, and leaves
threadgroup memory with no sharers. Something has to give: either an
instance becomes a group of threads, or the language surface grows a
notion of groups.

## (b) Two mapping options

### Option A: one instance = one simdgroup (emitter widens across lanes)

Keep the language surface exactly as it is: `Gpu.launch(n, f)`, one pid
per output tile-block. Change only the lowering: an instance occupies
32 device threads (one simdgroup), `pid` derives from the simdgroup
index rather than the thread index, and the emitter distributes each
tile across the group's lanes instead of one thread's stack.

* Control flow: uniform by construction. All 32 lanes compute the
  same `pid`, read the same snapshot, and no scalar depends on a lane
  id (the subset property above), so as long as the emitter keeps
  scalars replicated (every lane computes them identically, and a
  lane-partial reduction like `Tile.sum` is reduced then broadcast
  before it becomes a scalar) every lane takes the same path through
  the switch-machine. No divergence analysis, no reconvergence
  machinery.
* Tile ops: `Tile.dot` on 8x8 f32/f16 tiles lowers to
  `simdgroup_load` + `simdgroup_multiply_accumulate` +
  `simdgroup_store`; other shapes and ops lower to per-lane strided
  loops over a lane-distributed array (lane k owns elements
  `k, k+32, ...`), with `simdgroup_barrier(mem_flags::mem_threadgroup)`
  only where a distributed value is re-read after redistribution.
  Threadgroup memory becomes the natural home for staged tiles and is
  invisible to the user.
* `std/gpu.mx`: unchanged. Kernels do not change. This is the point.
* `emit_msl.py`: the large cost lives here. It grows a second lowering
  mode (per-simdgroup) beside the current per-thread mode, a
  distributed-vs-replicated classification for declarations, and the
  simdgroup op selection for eligible shapes. The switch-machine CFG
  survives unchanged because control flow is uniform.
* C++ shim: stays a sequential, bit-exact reference and does not
  simulate lanes. It emulates the SEMANTICS of each collective op,
  which are exactly the whole-tile semantics the interpreter already
  pins: a simdgroup 8x8 multiply-accumulate is emulated as the
  existing whole-tile dot loop with the pinned rounding order. The
  shim body can therefore remain today's per-thread body verbatim.
* `metal_launch.py`: dispatch becomes `grid=(32 * n, 1, 1)`,
  `threadgroup=(32, 1, 1)` (or a multiple, several simdgroups per
  threadgroup, later). Buffer plumbing is untouched.
* Harness: only the same grid arithmetic; buffers, masks, merge, and
  the tolerance policy are untouched.
* Execution contract: textually unchanged for the user. Snapshot
  reads, mask-merged writes, no cross-INSTANCE communication.
  Intra-simdgroup cooperation is an implementation detail below the
  contract line.
* Numeric honesty: `simdgroup_multiply_accumulate` does not promise
  our pinned round-product-then-round-accumulate order (it fuses).
  The mlx leg is already tolerance-compared under fast-math, so the
  policy does not change; the shim and interpreter remain the
  bit-exact pair. A kernel that needs bit-exact device results keeps
  the current per-thread lowering (an emitter mode flag).

### Option B: cooperative-launch surface (the grid is over tile-blocks, groups are explicit)

Grow the surface: something like `Gpu.launch_group(blocks, lanes, f)`
where `f` takes `(block, lane)`, plus group operations (a barrier, and
group-shared tiles staged in threadgroup memory). The emitter maps a
block to a threadgroup and a lane to a thread, and the "memory-space
locality states" left open in Stage 1 become load-bearing: a shared
tile is a different mode from a private one, checked like the other
modal axes.

* `std/gpu.mx`: new effect ops and a new kernel signature. Every
  kernel that wants simdgroup speed is rewritten against the group
  API. Kernels can now express things Option A cannot (cross-tile
  data reuse inside a block, software pipelining staged by hand).
* `emit_msl.py`: emits `threadgroup` allocations, real barriers, and
  per-lane index math from `thread_position_in_threadgroup`. It must
  reject barrier calls under lane-divergent control flow, which means
  the uniform-by-construction property no longer covers us: lane ids
  legitimately drive branches, so the emitter needs an actual
  uniformity analysis with loud rejections.
* Interpreter / CPU handler: this is the heavy cost. The CPU handler
  must stay the semantics of record, so it must DEFINE barrier
  semantics: run lanes of a block in lane order in phases, each lane
  advancing to the next barrier, error loudly if lanes disagree on
  which barrier they reached (divergent-barrier is UB on the device,
  so the reference must make it an error, not a behavior). That is a
  new scheduler inside `run_grid`'s replacement, plus shared-tile
  state between lanes, all pinned by tests before any MSL exists.
* C++ shim: must emulate the same phased execution to stay bit-exact
  with the new reference, so it stops being "the same body run in a
  loop" and becomes a small lane scheduler too.
* `metal_launch.py` / harness: block/lane grid math, threadgroup
  memory sizing, and a new failure mode (exceeding threadgroup memory
  limits) surfaced loudly.
* Execution contract: grows a second tier. Cross-block stays snapshot
  plus merge; intra-block gains barrier-ordered shared-memory
  visibility. More power, more contract to hold.

## (c) Reference semantics per option

* Option A: the interpreter is untouched and stays the semantics of
  record as-is. `run_grid` already defines what every tile op means;
  simdgroup execution is an implementation of those same ops. Nothing
  new to pin except the (already existing) device-leg tolerance.
* Option B: the reference must grow lane scheduling, phased barriers,
  divergent-barrier errors, and shared-tile state, all in
  `mir_interp` first, differentially mirrored by the shim. The
  semantics of record gets materially larger before the first fast
  kernel runs.

## (d) Recommendation

Option A, and only then judge whether Option B's expressiveness is
still needed. Option A keeps kernels, the contract, and the reference
interpreter fixed, and it spends its complexity budget in exactly one
file (`emit_msl.py`), which is where our differential harness is
strongest.

Smallest honest first increment, in two steps:

1. Grid plumbing only: add the per-simdgroup dispatch mode to
   `metal_launch.py` and the harness (`grid=(32 * n, 1, 1)`,
   `threadgroup=(32, 1, 1)`, pid from the simdgroup index; lanes 1..31
   idle). Same body, same outputs, verifiable on a Mac against the
   unchanged reference. No numeric change anywhere.
2. Widen one op: `Tile.dot` of 8x8 f32 tiles (then f16) through
   `simdgroup_float8x8`, lane-distributed loads for its operands,
   everything else still replicated. Device leg tolerance-compared as
   today; shim and interpreter untouched and still bit-exact with
   each other. Measure; expand op coverage only where the win shows.

## Status: Option A, both increments, landed

`emit_msl.py` has the second lowering. It is chosen automatically when
a kernel contains an 8x8 f32 or f16 `Tile.dot`; `emit_msl_kernel(...,
simdgroup=...)` and `METAXU_METAL_LOWERING=thread|simdgroup|auto` (read
by the `run_metal` handler) force either way. What the per-simdgroup
body does, exactly:

* `pid` is `thread_position_in_grid.x / 32`, the simdgroup index; the
  lane id is kept only for guards. Dispatch is `grid = 32 * n`,
  `threadgroup = 32` (`MslKernel.grid`), in the mlx engine and the
  generated Mac harness alike.
* Scalars stay per-thread and replicated: every lane computes them
  identically, `Tile.sum` reduces the published tile on every lane, so
  the switch-machine's control flow is uniform and every lane reaches
  every collective and every barrier.
* Tiles are `threadgroup` arrays. Lane 0 does the elementwise work
  (fills, loads, add/mul/scale, conversions, transposes, copies) and a
  `threadgroup_barrier(mem_threadgroup)` publishes each written tile;
  device stores are lane 0's too. Lanes 1 to 31 are idle for those ops
  in this first increment; distributing them is the next step and
  needs a shim that emulates lanes, which is why it was not folded in.
* An eligible dot is the collective: `simdgroup_load` of both operands
  from the threadgroup tiles, `simdgroup_multiply_accumulate` into a
  zero-filled `simdgroup_float8x8` / `simdgroup_half8x8`,
  `simdgroup_store` back to the result tile, then a barrier. Dots of
  other shapes, and all int dots, keep the lane-0 loop.
* The C++ shim compiles the same body: `threadgroup` is defined away,
  `namespace metal` carries `mem_flags`, a no-op barrier, the 8x8
  matrix type and the three collectives, emulated with the
  interpreter's pinned order (round the product, then the accumulate).
  The shim runs lane 0 of each instance, which in this lowering is all
  the non-collective work. Interpreter and shim remain the bit-exact
  pair; the mlx leg keeps its float tolerance, which is where the
  device's fused multiply-accumulate lands.

Pinned in `test_msl_emitter.py` (16x16 f32 and f16 matmuls through
two 8x8 k-blocks, against the interpreter and against an independent
Python ground truth; the whole kernel corpus forced through the new
lowering; body and harness structure) and `test_metal_launch.py`
(handler parity under the default, and under both overrides).

Not done: lane-distributed elementwise loops, several simdgroups per
threadgroup, threadgroup-memory tiling across instances (that is
Option B territory), and measurement on a Mac, which the generated
harness is for.
