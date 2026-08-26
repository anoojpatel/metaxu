# GPU tiles: the portable core and the Apple-first backend

Status: **Stage 0 landed** (portable core, no GPU dependency) —
`Tile.*` dotted statics on both engines, `tile:<elem>:<R>x<C>` kinds,
the compile-time shape checker (`tile_shape_check.py`,
`type-tile-shape`), mx_tile_* native runtime, and the differential
gate (`tests/test_tiles.py`: 19 tests — interpreter reference,
compile-time rejections, byte-identical native output including tile
reprs and catchable bounds raises).  Landing it also forced a real
parity fix: native `print(f64)` used `%g` ("12" for 12.0) where the
interpreter prints Python repr ("12.0") — print now routes through
`mx_f64_to_str` everywhere.  This document is the design of record for
writing Triton/Gluon-class GPU kernels in Metaxu, targeting MLX/Metal
first.

## The strategy in one paragraph

Kernels are written at the **tile level** — programs over
`Tile[T, R, C]` values, not per-thread code — with the compiler owning
how a tile is distributed across hardware. The tile IR, its interpreter
(the semantics reference), shape checking, and later the layout system
are **target-neutral and live in this repo's Python compiler**, exactly
like the rest of the pipeline. Backends are thin text emitters invoked
per the house pattern (emit source text, shell out to a vendor
compiler, differentially test against the interpreter): Metal Shading
Language first — bound into MLX via `mx.fast.metal_kernel`, because
MLX/Apple is the deployment target and no MLIR path reaches
`simdgroup_matrix`-class Metal code — and, when NVIDIA/AMD matter,
textual TritonGPU-dialect MLIR consumed by Triton's own pipeline. We
deliberately adopt the **TritonGPU layout algebra as our vocabulary**
(blocked / slice / dot-operand / shared-swizzled) so that second
emitter is a pretty-printer, not a compiler. We do NOT build on MLIR as
infrastructure: it cannot reach the Apple target, it would break the
pure-Python + text-emission build shape, and the interpreter-first
methodology requires our own tile semantics regardless.

## Why Metaxu is already most of the way there

The Triton/Gluon design splits — inferred behavior by default, explicit
typed control as the escape hatch — instantiate patterns this compiler
already has three of:

| GPU need | existing Metaxu machinery |
| --- | --- |
| memory spaces (device / threadgroup / thread) | the locality axis, extended to a 3-level containment lattice; the borrow checker's "no `@local` ref stored into `@global`" rule generalizes verbatim |
| kernel purity (no IO/spawn/alloc inside kernels) | effect classes (`stack`/`suspend`) gain a `gpu` class; the emitter already checks classes |
| kernel launch + scheduling seam | an effect (`perform Gpu.launch(...)`) with the handler owning the MLX stream; the tile interpreter handles the same effect on CPU |
| host/device buffer safety | separateness + contention: a buffer passed to a launch is a crossed value — the spawn-capture checker at a new boundary |
| layouts (which lane holds which element) | a new mode axis: conversion is semantically identity (the mode criterion), inferred Rule-B style with explicit costed `convert_layout` crossings |
| lane-parallel mutation soundness | a distributed layout IS a static disjointness proof — separateness at lane granularity |

## Mode-axis interactions (decided)

Exactly four places where axes meet; everything else is orthogonal by
construction:

1. **Space is locality.** `@thread ⊂ @threadgroup ⊂ @device` is a
   containment lattice on the existing locality axis (not a new axis).
   References may only be stored outward-contained; escape is always an
   explicit copy (`tile.store`), never a reference crossing.
2. **Layout depends on space.** The family of inhabitable layouts is
   indexed by the space: distributed (blocked / fragment) in registers,
   swizzled shared layouts in threadgroup memory, strided in device
   memory. Ops that move a tile between spaces necessarily assign a
   fresh layout from the target family — "a swizzled register tile" is
   unrepresentable.
3. **Layout licenses `@mut`.** A distributed layout partitions elements
   across lanes, so lane-parallel mutation through it is statically
   race-free — no new mutability rules needed.
4. **`@threadgroup @mut` is barrier-phased.** v1: user code never holds
   it — shared tiles are written only through tile ops, which own
   barrier placement (the `Protected` pattern). v2 (if needed): a
   phase-permit discipline, the contention-as-permission design with
   "barrier since last write" as the permit.

Neither new axis adds subtyping: space and layout are **coercion axes**
(equality constraints + explicit costed conversion ops), so SimpleSub's
lattice does not grow — layout inference is unification over layout
variables plus a conversion-insertion post-pass.

## Layouts (decided vocabulary; implementation is Stage 2)

A layout is a compile-time map from tile coordinates to hardware slots.
Four families, following TritonGPU:

- **strided** (`@device`): shape + strides — what an MLX array is.
- **blocked** (`@thread`): elements-per-lane × lane arrangement (Apple:
  32 lanes/simdgroup) × simdgroup arrangement + iteration order.
- **fragment** (`@thread`): the hardware-defined `simdgroup_matrix`
  8×8 distribution demanded by `tile.dot`.
- **shared+swizzle** (`@threadgroup`): row-major + XOR swizzle chosen
  for bank-conflict freedom against the layouts touching it.

Inference: every tile value gets a layout variable; elementwise ops
equate, transpose/broadcast transform structurally, `dot` (hard) and
`load` coalescing (soft) anchor concrete layouts; conflicts insert
`convert_layout` on the cheapest edge, with the real cost order
metadata-only < in-lane permute < cross-lane shuffle <
threadgroup-memory round trip (+barrier). Provenance-chain diagnostics
(as built for locality) explain every inserted conversion. Statically
checkable: divisibility of tile dims by the distribution product,
per-lane register budget, swizzle bank-conflict freedom.

## Staging

- **Stage 0 — portable core (in progress, no GPU anywhere):**
  `Tile[T, R, C]` with the shape STATIC IN THE TYPE (the key delta from
  today's `vector[T,N]`, whose length is runtime data), a tight op set
  (from_vec/to_vec, zeros, arange, elementwise, scalar broadcast, dot,
  sum, transpose, get), deterministic interpreter semantics as the
  reference, compile-time shape errors, native lowering differentially
  tested. Element types start at `float`(f64)/`int` — the tile
  machinery is width-agnostic and f32/f16 land as their own increment
  (Stage 1 needs them; the CPU reference does not).
- **Stage 1 — the kernel seam (1a/1b/1c landed):**
  * **1a (landed):** the buffer <-> tile boundary — strict
    `Tile.load`/`Tile.store` (catchable range raises) plus the masked
    kernel-side forms `Tile.load_or` (out-of-range reads `other`) and
    `Tile.store_clipped` (out-of-range writes nothing), semantics
    pinned in the interpreter first; stores take the contended-write
    guard like every Vec mutator (differential pins the raise).
  * **1b (landed):** `std/gpu.mx` — `effect Gpu { launch(n, f) =
    run_grid(n, f) }`, the launch as PURE LIBRARY: default = the
    reference semantics (sequential, pid-ordered CPU grid), handlers =
    backends/virtualization.  Zero compiler changes were needed; tiled
    matmul / vecadd / ragged-grid kernels are differentially tested on
    both engines, pid order and handler interposition pinned.
  * **1c (landed):** the first MSL emitter (`emit_msl.py`) for the
    documented kernel subset (masked forms only, INT tiles only until
    f32, literal shapes, full CFG via a switch-machine since MSL has no
    goto).  The emitted body is deliberately C++-compatible, so the
    container tests compile it with clang++ and race it against the
    interpreter END TO END; `scripts/emit_metal_harness.py` generates a
    self-checking Mac harness (`mx.fast.metal_kernel` binding,
    written-mask merge for clipped stores, interpreter-computed
    expected values baked in, exit 0 on match).  Execution contract:
    snapshot reads + mask-merged writes — cross-instance
    read-after-write within one launch is outside the contract (racy
    on real GPUs; the sequential reference would hide it).
  * **Measured CPU cost of the tile abstraction (2026-08-26):** the
    tilemm diagnostic (8x8-tiled 128^3 int matmul through `Gpu.launch`)
    runs ~4x plain C loops natively, vs ~1.3x for the scalar-loop
    metaxu version.  Expected and accepted for now: every tile op is an
    opaque runtime call that ALLOCATES a fresh block (functional
    semantics), and the launch adds a per-instance closure call.  The
    known CPU wins — op fusion, arena/reuse allocation for provably
    non-escaping tiles, the Vec-fast-path inline treatment for tile
    ops — are deliberately deferred: the tile abstraction's performance
    layer is Metal + Stage 2 layouts, and optimizing the CPU path first
    would optimize the reference instead of the product.  The number is
    tracked in benchmarks/diagnostics (tilemm row) so it cannot rot
    silently.
  * **Still open in Stage 1:** f32/f16 scalars (prerequisite for float
    kernels on Metal — no f64 there), the `gpu` effect class (becomes
    load-bearing when the MLX handler dispatches real launches), and
    memory-space locality states (vacuous until threadgroup memory
    arrives in Stage 2).
- **Stage 2 — fast:** inferred layouts + `convert_layout`,
  `simdgroup_matrix` dot, threadgroup-memory tiling, software
  pipelining where it pays on Apple.
- **Stage 3 — expert surface:** Gluon-style explicit layout
  annotations, the autotuner (the benchmark harness's paired-run
  methodology as a per-kernel search with a shape-keyed cache),
  TritonGPU-text emitter for NVIDIA/AMD.
- **Stage 4 — distribution (multi-GPU / multi-node):** collectives as
  sharding coercions + `Dist.*` effects + backend handlers (loopback →
  MLX distributed → NCCL).  See "Stage 4: distribution" below — mostly
  LIBRARY code on top of the language, with two compiler carve-outs.

## Stage 0 design (implementation notes)

- **Type:** `Tile[T, R, C]` parses through the existing type-application
  grammar (no new keyword; `Tile` is a reserved type constructor the
  checker recognizes — unlike `vector`, no lexer token is needed).
  R and C are integer literals in v1; const-generic parameters
  (`const N: int`, already parseable in generic headers) get wired to
  monomorphization when size-polymorphic functions are actually needed
  — Triton practice (constexpr block sizes per instantiation) says
  literals carry a long way.
- **Kind:** `tile:<elem>:<R>x<C>` rides the existing string-kind scheme
  (like `vector:` / `struct:`), so the static shape reaches every
  consistency check and the backend for free.
- **Interpreter value:** a dedicated immutable `MxTile` (elements
  row-major, rows, cols) — deliberately NOT MxVector, so tiles and
  vectors cannot silently mix (the historical silent-seam bug class).
- **Ops (v1):** `Tile.zeros[T,R,C]()`, `Tile.arange[R,C]()` (row-major
  iota, int), `Tile.from_vec[T,R,C](v)` (length-checked, loud),
  `t.to_vec()`, `t + u`, `t * u` (elementwise, same shape enforced at
  compile time), `t * s` / `t + s` (scalar broadcast), `t.dot(u)`
  ((R,K)×(K,C) with K agreement enforced at compile time), `t.sum()`
  (full reduction to scalar), `t.t()` (transpose), `t.get(i, j)`
  (bounds-checked read, loud). Tiles are immutable values: every op
  produces a fresh tile (functional semantics now; layouts make
  in-place lane updates sound later, and immutability keeps Stage 0
  honest instead of pre-committing a mutation story).
- **Shape errors are compile-time `TypeCheckError`s** (elementwise
  mismatch, dot inner-dim disagreement, from_vec length vs R*C when the
  argument length is statically known — runtime-checked otherwise).
- **Native lowering:** `mx_tile_*` C runtime functions (metaxu_rt.c)
  over the same word-block representation as fvec; plain row-major C
  loops (clang auto-vectorizes these; measured SIMD work belongs to
  Stage 2 where layouts exist to direct it). Blocks are write-once and
  leak by design, like fvec — documented, ASan-scoped accordingly.
- **Differential gate:** every tile op tested through parsed source on
  both engines, byte-identical output, per the house conventions.

## Stage 4: distribution (collectives, NCCL, and friends)

Design of record for multi-GPU / multi-node, written down early because
two decisions must be locked before Stage 1 so nothing needs redesign
later.  Nothing in Stages 0-2 blocks on any of this.

### The fourth regime

Distribution extends the concurrency table with one more row, same
shape as the others — a static map saying who owns which elements, plus
explicit costed operations for redistributing:

| boundary | disjointness proof | communication primitive |
| --- | --- | --- |
| lanes in a simdgroup | layout (static partition) | register shuffle |
| threads in a threadgroup | barrier phasing | threadgroup memory |
| host ↔ device | separateness at `Gpu.launch` | buffer transfer |
| rank ↔ rank | sharding (static partition over a device mesh) | collectives |

A layout maps tile coordinates to lanes; a **sharding maps global
tensor coordinates to ranks over a mesh**.  Same algebra, bigger
machines.

### Collectives are sharding coercions (the GSPMD insight)

There is no "all-reduce API" as a primitive concept.  A distributed
tensor type `DTensor[T, M, N, mesh, spec]` says, per dimension: sharded
over a mesh axis, replicated, or PARTIAL (each rank holds an unreduced
addend — the state right after a local matmul of sharded operands).
Every collective is then a typed conversion between specs:

- partial → replicated = **all-reduce**
- sharded → replicated = **all-gather**
- partial → sharded = **reduce-scatter**
- sharded on axis i → sharded on axis j = **all-to-all**

`Dist.convert(t, new_spec)` is the whole blessed surface: before/after
types are static (shape divisibility by mesh extents checked at
compile time by the same const-shape machinery as tile shapes), the
communication cost is visible in the program as a conversion, and the
provenance diagnostics explain every one ("this all-gather exists
because the matmul at line 30 needs B replicated along mesh axis x").
Rank-explicit `send`/`recv` exist underneath as expert effect ops; the
conversion surface is primary because it is the level where
deadlock-freedom is checkable.

### Effects are the mechanism, handlers are the backends

`Dist.*` is an effect, which buys four things:

1. **NCCL is a handler, not a language feature.**  The same program
   runs under a handler lowering to `ncclAllReduce` on CUDA streams,
   one calling MLX's `mx.distributed` (all-sum over MPI or the ring
   backend — the Apple-first path, real today for Mac clusters over
   Thunderbolt), an MPI handler, or the **loopback handler** simulating
   N ranks as threads in one process — the deterministic single-machine
   reference every backend is differentially tested against, no
   cluster required.
2. **Communicator lifecycle is capability-shaped**: the handle lives in
   the handler's scope (`handle Dist with nccl(comm) in { ... }`); no
   collective outside an installed handler, no global communicator
   state, structural init/teardown ordering.
3. **Async overlap and bucketing are handler policy**: a handler may
   coalesce pending performs into one fused collective (DDP-style
   gradient bucketing) and assign streams — without the user program
   changing.  A nicer factoring than callback hooks.
4. **Effect rows make communication visible in signatures**: a function
   performing `Dist.*` says so in its type.

### Safety analyses (the compiler carve-outs)

Two checks are compiler work, in the style of the existing bounded
checkers (inferred property + provenance, zero false positives):

- **Deadlock via uniformity**: the classic NCCL hang is rank-divergent
  control flow around a collective.  `Dist.rank()` is non-uniform; a
  collective inside a branch whose condition is transitively
  rank-dependent is flagged — the Rule-B/separateness pattern's next
  instance, and a check raw NCCL users get only from discipline.
- **In-flight buffer safety**: a buffer handed to an async conversion
  is marked until its completion event; writes before that are the
  raise — the contention-as-permission design's fourth appearance.

### Locked now (so Stage 1+ stays compatible)

1. **The mesh rides the same const-generic machinery as tile shapes.**
2. **One distribution algebra**: sharding specs use the same
   blocked-distribution vocabulary as layouts, so "distribution over
   hardware" exists ONCE, instantiated at lane, threadgroup and mesh
   scale.

### Language-level vs compiler-level (the split, stated)

Distribution is mostly a LIBRARY built on top of the language — which
is the house philosophy (std-powered, like `Protected`): the `Dist`
effect and its ops are an ordinary effect declaration; `DTensor` is
std code wrapping tiles + a spec; every backend (loopback, MLX
distributed, MPI, NCCL) is a handler plus runtime shims; bucketing/
overlap policy is handler code.  The compiler owns exactly: (a) the
two safety analyses above, (b) the const-generic/mesh machinery it
already owns for tiles, and (c) — only if/when we add GSPMD-style
AUTO-sharding, where the compiler chooses specs and inserts
conversions itself — the propagation pass.  v1 keeps conversions
explicit precisely so that stays optional.

### Honesty about determinism

Real all-reduce is not bit-deterministic across backends (NCCL's
ring/tree orders float additions by topology).  The loopback reference
pins a deterministic order (exact, testable semantics); integer
collectives are exact everywhere; float differentials against real
backends get DOCUMENTED TOLERANCES instead of byte-equality.  This is
the first place the byte-identical discipline meets a genuinely
nondeterministic substrate, and the contract is stated up front rather
than discovered.

## What we are NOT doing (and why)

- **Not building on MLIR:** no path to `simdgroup_matrix`-class Metal;
  a C++ dependency against a pure-Python text-emitting toolchain; the
  reference interpreter must exist regardless. MLIR remains an
  *emission target* (textual TritonGPU) for the NVIDIA/AMD stage.
- **Not full dependent types:** literal shapes now, const-generic
  naturals with equality/divisibility constraints at instantiation
  later — checked at monomorphization time, never SMT.
- **Not a mutation story for tiles yet:** functional semantics until
  layouts exist to make lane-local mutation provably sound.
- **Not auto-vectorization heroics in Stage 0:** the CPU lowering is
  simple correct loops; performance work waits for the layer (layouts)
  that can direct it.
