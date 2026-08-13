# Metaxu Compiler Pipeline Roadmap

This document tracks high-level goals, status, and pointers across the new Python compiler pipeline embedded in `src/metaxu/compiler/` and the Rust runtime in `src/metaxu/runtime/`. It consolidates the original plan, ongoing work, and upcoming features (borrow checking, CPS, trait dictionaries, struct mode validation).

## Repository Pointers

- Compiler package: `src/metaxu/compiler/`
  - Frozen AST: `mutaxu_ast.py`
  - Frozen AST borrow checker: `frozen_borrow_checker.py` (new)
  - Frozen constraint emitter: `frozen_constraint_emitter.py`
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
  - Frozen AST borrow checker with full spec implementation (shared/unique/exclusive borrows, locality, linearity, reference conflicts, exclave)
  - Frozen constraint emitter with borrow checker integration
  - Borrow checker spec document (frozen_borrow_spec.md)
  - Old BorrowChecker in type_checker.py deprecated

- Completed since (see `docs/v1_gap_analysis.md` for full detail)
  - HIR build over real parsed AST: patterns (ctor/literal/negative/bare
    variant), enums, while loops, assignments, closures with captures,
    structs/field access, index/slice/vector expressions, effects
  - MIR: real multi-block control flow, decision-tree pattern compilation,
    strict interpreter (unbound names error), runtime library (Vec,
    vector[T,N], math builtins), trait dispatch on runtime types
  - Effects: delimited single-shot continuations (deep handlers, aborts,
    nested scopes, handler self-performs, multi-arg ops); stack/suspend
    class enforcement
  - Trait impl desugaring (implement blocks -> mangled functions with
    runtime dispatch; user impls win over builtins)
  - Struct field mode validation (deep @global/@local ownership rules,
    transitive, with structured deep-locality diagnostics)
  - Selective CPS: frame layouts (cps_frames.py), %run_<f> br_table state
    machines parking via enqueue/sched_read, %resume_<f>_<k> shims
  - CLIF direct codegen: real multi-block Cranelift IR text for the direct
    subset; declaration-only placeholders elsewhere; structural validator
  - Type enforcement: TypeCheckError for struct-field literal mismatches
    and constraint-graph class conflicts (1 + "a", non-bool conditions)
  - Example gates: 19/19 pipeline (2 negative fixtures rejected),
    15/19 executing, pinned in pytest with golden outputs

- Completed since (verification round)
  - Coherence checks: duplicate implement blocks for the same
    (trait, type, method) raise CoherenceError at desugar time
  - Unhandled-effect advisory: a perform neither lexically handled nor
    covered by the enclosing performs clause gets a checker diagnostic
  - Three adversarial-review rounds over the branch (20+ execution-
    confirmed bugs found and fixed, each pinned by a regression test)

- Pending (Highlights — each needs a design decision or major ABI work)
  - Incremental module compilation: per-module codegen units with
    signature-hash cache keys and linkonce monomorphization (see section 9)
  - Assoc type concretization
  - Frame chaining across suspending calls (needs a parked-return protocol
    and frame allocator in the runtime ABI); CLIF-level effect dispatch
  - FFI/threads runtime (unblocks 05_unsafe_and_ffi, effect_mapping)
  - try/catch semantics (undefined in docs; needs a design decision)
  - Full biunification with principal-type coalescing

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
  - **Proper match lowering with pattern compilation and branching (HIGH PRIORITY)** - This will determine how we make progress lowering down and dealing with branching for match expressions
  - Drop insertion from DropPlan (end-only heuristic → per-block) (pending)

### 4) Borrow Checking & Drop Planning
- Files: `frozen_borrow_checker.py`, `frozen_borrow_spec.md`, `borrow_analysis.py`, `hir.py`, docs: `docs/ownership_and_borrowing.md`
- Status
  - Frozen AST borrow checker complete (shared/unique/exclusive borrows, locality, linearity, reference conflicts, exclave)
  - Integrated with frozen constraint emitter
  - Spec documented in frozen_borrow_spec.md
- Goals (HIR-level)
  - Region stack across HIR blocks; def-use and last-use approximation
  - Aliasing rules: shared `&` vs exclusive `&mut`; overlap checks
  - Move semantics (owned values); use-after-move diagnostics
  - Locality rules: locals cannot escape; allow `exclave` promotion modeling
  - Effect safety: forbid locals in effect ops/handlers; allow globals
  - Produce per-block `DropPlan` for MIR lowering from frozen AST borrow checker results

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
- Files: `cps_frames.py` (frame layouts), `codegen_clif.py`, runtime stubs
- Inputs
  - Suspensions from side tables (effects) and call graph
- Tasks
  - Mark suspending functions; keep non-suspending functions direct (done: `MirFunc.suspending` flag, plus any function containing perform/resume/handle_scope ops)
  - Defunctionalize: generate `Frame` struct layouts, `enum State` (done: `cps_frames.compute_frame_layouts` — state/result discriminant slots plus live-across-suspension variables; emitted as `; frame %f: ...` comment tables in the CLIF)
  - Emit `run_<fn>` with `br_table` on state, and `resume_*` shims (done for suspending functions in the i64 direct subset; others stay honest placeholders)
  - Insert `sched_read`, `enqueue` calls at park/wake sites (done: performs park via `enqueue(frame)`, read-shaped ops via `sched_read(fd, buf, len, k, frame)` with the resume shim as `k`; general effect dispatch stays in the interpreter — CLIF has the mechanical park/wake shape only)
  - CLIF for CPS: frame loads/stores using layout tables (done at park sites / resume prologues; frame chaining across calls to other suspending functions pending)

### 7) CLIF Codegen
- Files: `codegen_clif.py`
- Tasks
  - Direct SSA functions with multi-blocks (pending expansion)
  - CPS functions with `br_table` and frame layout helpers (done; see 6)

### 8) Golden Tests & Examples
- Files: `src/metaxu/compiler/tests/`
- Targets
  - sample1: pipeline skeleton (done)
  - sample2: binop MIR (done)
  - If-lowering multi-block (pending)
  - Iterator + next_or (traits/assoc types) (pending)
  - read_u32 (suspending + CPS) (pending)
  - Borrow/mode validation and errors (pending)

### 9) Incremental Module Compilation (pending — own phase)
- Files: `module_loader.py`, `pipeline.py`, `codegen_llvm.py`, `llvm_run.py`
- Today: whole-program only. The module loader merges every imported file
  into one Program before analysis; one inference run, one LLVM module,
  one clang invocation. Function-level namespacing (dotted renames like
  `math.vector.dot`) keeps modules apart, but nothing is cached between
  compilations except the C runtime objects (mtime-cached .o files).
- Two whole-module analyses currently block per-module codegen and must be
  cut at module boundaries first:
  - Kind inference: the emitter's kind fixpoint flows callee kinds from
    all callers module-wide. At module boundaries this must be replaced by
    DECLARED signatures — which implies a language rule that public
    (exported) functions require type annotations. Advisory until then.
  - Monomorphization: clones are reachability-driven over the whole call
    graph. Assign each clone to the INSTANTIATING module's codegen unit
    with linkonce/weak linkage so duplicate specializations merge at link
    time (the Rust model).
- Plan (in order):
  1. Codegen units: each module emits its own .ll -> .o, cached under a
     key = hash(module source, compiler version, and the SIGNATURES of
     everything it imports). Signature hashing is the crux: a body-only
     edit must not invalidate importers. The loader already computes the
     dependency DAG, so unit partitioning and cache keys live there.
  2. Stable ABI: the dotted-name mangling (module system) plus `$`
     specialization suffixes (monomorphize) are already deterministic;
     freeze them as the object-level ABI and document.
  3. Link step: clang links the cached per-module .o set + runtime
     objects; only dirtied units re-emit ("last-minute linking").
  4. Cross-module inlining/opt: accept the loss initially (LLVM LTO can
     recover it later via -flto on the cached bitcode instead of .o).
- Interpreter path stays whole-program (it is the semantics reference and
  compilation speed there is not a bottleneck).

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

- **Implement proper match lowering with pattern compilation and branching (HIGH PRIORITY)** - This will determine how we make progress lowering down and dealing with branching for match expressions
- Add comprehensive borrow checker tests (UNIQUE vs EXCLUSIVE semantics, locality checking, global-to-local reference prevention, exclave)
- Connect frozen AST borrow checker to full type checking pipeline
- Integrate frozen AST borrow checker with HIR-level DropPlan generation
- Add If-lowering golden test
- Integrate selective CPS on a small example (read_u32) and extend CLIF for CPS
- Implement trait dictionary desugaring + assoc types; add Iterator/next_or golden

---

This roadmap will be kept up to date as we implement features. For background and rules on modes/borrowing, see `docs/ownership_and_borrowing.md`. For trait solving and FDs, see `src/metaxu/compiler/constraints.py` and impl registry in `impl_registry.py`.
