# GPU tiles: the portable core and the Apple-first backend

Status: **Stage 0 in progress** (portable core, no GPU dependency).
This document is the design of record for writing Triton/Gluon-class
GPU kernels in Metaxu, targeting MLX/Metal first.

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
- **Stage 1 — the kernel seam:** `kernel fn` (gpu effect class),
  `Gpu.launch` effect + handlers (CPU tile-interpreter handler first),
  memory-space locality states, masked load/store (kernels cannot
  raise; masked IO is the honest analog of loud bounds checks — the
  semantics "masked-out lane reads `other`, writes nothing" is pinned
  in the interpreter first), f32/f16 scalars, then naive-but-correct
  MSL emission bound via `mx.fast.metal_kernel`, differential vs the
  tile interpreter.
- **Stage 2 — fast:** inferred layouts + `convert_layout`,
  `simdgroup_matrix` dot, threadgroup-memory tiling, software
  pipelining where it pays on Apple.
- **Stage 3 — expert surface:** Gluon-style explicit layout
  annotations, the autotuner (the benchmark harness's paired-run
  methodology as a per-kernel search with a shape-keyed cache),
  TritonGPU-text emitter for NVIDIA/AMD.

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
