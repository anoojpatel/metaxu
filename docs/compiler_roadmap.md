# Metaxu Compiler Pipeline Roadmap

This document tracks high-level goals, status, and pointers across the new Python compiler pipeline embedded in `src/metaxu/compiler/` and the Rust runtime in `src/metaxu/runtime/`. It consolidates the original plan, ongoing work, and upcoming features (borrow checking, CPS, trait dictionaries, struct mode validation).

## Repository Pointers

- Compiler package: `src/metaxu/compiler/`
  - Frozen AST: `mutaxu_ast.py`
  - Types/TyEnv: `types.py`
  - Inferencer adapters: `infer_tables.py`
  - Constraints + MPTC+FD solver: `constraints.py`
  - Trait registry + impls: `impl_registry.py`
  - HIR definitions + builder: `hir.py`
  - MIR IR: `mir.py`
  - Lowering (HIR → MIR, ANF + selective CPS): `lower_hir_to_mir.py`
  - CLIF emitter: `codegen_clif.py`
  - Borrow analysis (drop planning): `borrow_analysis.py`
  - Pipeline runner: `pipeline.py`
  - Tests and goldens: `src/metaxu/compiler/tests/`
- Runtime (Rust): `src/metaxu/runtime/`
  - Cargo: `src/metaxu/runtime/Cargo.toml`
  - Stubs: `src/metaxu/runtime/src/lib.rs`
- Existing language core (parser, type checker, SimpleSub):
  - Parser: `src/metaxu/parser.py`
  - AST types: `src/metaxu/metaxu_ast.py`
  - Type checker: `src/metaxu/type_checker.py`
  - SimpleSub inferencer: `src/metaxu/simplesub.py`
  - Type defs (CompactType, unify): `src/metaxu/type_defs.py`
- Ownership/borrowing docs: `docs/ownership_and_borrowing.md`

## Current Status Snapshot

- Completed
  - Scaffolded compiler modules and runtime stubs
  - FD solver (improvement + instance resolution)
  - CLIF stub emitter
  - Minimal HIR with op annotations; HIRBuilder integration stubs
  - ANF lowering for: Literal, Var, Let, Call, BinOp; If → multi-block CFG
  - Minimal drop planning (end-of-function heuristic)
  - Golden tests: sample1 (pipeline skeleton), sample2 (binop MIR)

- In Progress
  - HIR build over real AST (desugarings deferred)
  - MIR ANF coverage expansion
  - Golden test suite growth
  - Better drop planning from borrow analysis

- Pending (Highlights)
  - Trait dictionary desugaring; assoc type concretization; coherence checks
  - Borrow-analysis v1 with region stack, aliasing, locality, effect-safety
  - Selective CPS for suspending functions; scheduler integration
  - Struct/enum field mode validation (deep ownership rules)
  - End-to-end goldens (Iterator/next_or, suspending read_u32)

## Roadmap Details

### 1) Constraints & Traits (MPTC+FD)
- Files: `constraints.py`, `impl_registry.py`, `hir.py`
- Tasks
  - FD improvement & instance resolution (done)
  - Coherence checks at impl load (consistency/coverage) (pending)
  - Trait dictionary desugaring in HIRBuilder: method `x.m(y)` → `m(dict, x, y)` (pending)
  - Assoc type desugaring via solver: `I::Item` → concrete type post-solve (pending)

### 2) HIR (Typed High-Level IR)
- Files: `hir.py`, `mutaxu_ast.py`, `infer_tables.py`
- Tasks
  - Freeze AST and build HIR with types/effects/suspends (in progress)
  - Ops supported: Literal, Var, Call, Let, Block, BinOp, If (done)
  - Add Lambda/Closure HIR with captures + modes (pending)
  - Add dictionary params to functions (pending)

### 3) MIR (ANF/SSA-lite) & Lowering Passes
- Files: `mir.py`, `lower_hir_to_mir.py`
- Tasks
  - ANF lowering for basics (done)
  - Control-flow lowering: If/Else (done), While/For/Match (pending)
  - Drop insertion from DropPlan (end-only heuristic → per-block) (pending)

### 4) Borrow Checking & Drop Planning
- Files: `borrow_analysis.py`, `hir.py`, docs: `docs/ownership_and_borrowing.md`
- Goals
  - Region stack across HIR blocks; def-use and last-use approximation
  - Aliasing rules: shared `&` vs exclusive `&mut`; overlap checks
  - Move semantics (owned values); use-after-move diagnostics
  - Locality rules: locals cannot escape; allow `exclave` promotion modeling
  - Effect safety: forbid locals in effect ops/handlers; allow globals
  - Produce per-block `DropPlan` for MIR lowering

### 5) Struct/Enum Field Mode Validation (Deep Ownership)
- Files: `hir.py` (mode annotations), `type_defs.py`, analysis pass TBD
- Rules (from docs)
  - A `@global` must not contain (transitively) any `@local` field
  - Global containers cannot store locals
  - Local containers may store references to globals
  - Deep conversions (`to_global`/`to_local`) must be recursive
- Tasks
  - Extract field modes from AST/types when building HIR/types
  - Validate nested ownership and produce diagnostics
  - Add tests (e.g., MixedTree, GlobalContainer) to exercise rules

### 6) CPS (Selective; Suspensions)
- Files: `lower_hir_to_mir.py` (CPS phase TBD), `codegen_clif.py`, runtime stubs
- Inputs
  - Suspensions from side tables (effects) and call graph
- Tasks
  - Mark suspending functions; keep non-suspending functions direct
  - Defunctionalize: generate `Frame` struct layouts, `enum State`
  - Emit `run_<fn>` with `br_table` on state, and `resume_*` shims
  - Insert `sched_read`, `enqueue` calls at park/wake sites
  - CLIF for CPS: frame loads/stores using layout tables

### 7) CLIF Codegen
- Files: `codegen_clif.py`
- Tasks
  - Direct SSA functions with multi-blocks (pending expansion)
  - CPS functions with `br_table` and frame layout helpers (pending)

### 8) Golden Tests & Examples
- Files: `src/metaxu/compiler/tests/`
- Targets
  - sample1: pipeline skeleton (done)
  - sample2: binop MIR (done)
  - If-lowering multi-block (pending)
  - Iterator + next_or (traits/assoc types) (pending)
  - read_u32 (suspending + CPS) (pending)
  - Borrow/mode validation and errors (pending)

## How to Run

- Minimal MIR golden tests
```bash
python -m pytest -q src/metaxu/compiler/tests/test_golden_mir_binop.py
```

- Full golden suite (in progress)
```bash
python -m pytest -q src/metaxu/compiler/tests
```

- End-to-end pipeline from source (parsing + type checking + pipeline)
  - API: `metaxu.compiler.pipeline.run_pipeline_from_source(source, file_path)`

## Immediate Next Actions

- Add If-lowering golden test
- Implement borrow-analysis v1 with region/aliasing checks → per-block DropPlan
- Integrate selective CPS on a small example (read_u32) and extend CLIF for CPS
- Implement trait dictionary desugaring + assoc types; add Iterator/next_or golden

---

This roadmap will be kept up to date as we implement features. For background and rules on modes/borrowing, see `docs/ownership_and_borrowing.md`. For trait solving and FDs, see `src/metaxu/compiler/constraints.py` and impl registry in `impl_registry.py`.
