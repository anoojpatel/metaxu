# Metaxu development guide

Metaxu is a systems language (Python front end, Cranelift-targeted backend)
with algebraic effects, mode-based memory management (`@local`/`@global`,
`@mut`/`@const`, `once`/`separate`/`many`), and SimpleSub-style inference.

## Running tests and gates

```bash
uv sync --all-groups                      # one-time setup
uv run python -m pytest src/metaxu/compiler/tests -q   # full suite
uv run python scripts/run_examples.py                  # pipeline gate (19 files)
uv run python scripts/run_examples.py --stage run      # execution gate
uv run python scripts/run_examples.py --stage parse    # parse-only
```

The example gates are also pinned inside pytest
(`src/metaxu/compiler/tests/test_example_gates.py`), including golden
outputs for examples whose docs promise specific values. Two root-level
files are negative fixtures: `test_borrow_check.mx` must be rejected with a
borrow diagnostic and `test_type_error.mx` with a type diagnostic.

## Pipeline shape

```
source -> Parser (PLY, src/metaxu/parser.py; shared instance via
          compiler/shared_parser.py — construction costs ~430ms, reuse it)
       -> module resolution (compiler/module_loader.py: multi-file imports,
          visibility, std/ stdlib resolution, dotted-name namespacing)
       -> desugar passes (compiler/desugar.py: trait impls -> mangled fns,
          bracket-form generic ctors)
       -> freeze (compiler/mutaxu_ast.py: immutable AST + JSON goldens)
       -> infer (compiler/infer_tables.py -> simplesub_adapter.py
                 + frozen_constraint_emitter.py + frozen_borrow_checker.py:
                 generics instantiation checking, exhaustiveness, borrow/modes)
       -> HIR (compiler/hir.py: typed, patterns, effects, trait dispatch;
               optional monomorphize.py pass)
       -> MIR (compiler/lower_hir_to_mir.py: ANF, multi-block CFG)
       -> backends:
          - compiler/mir_interp.py (semantics REFERENCE; execution gate)
          - compiler/codegen_llvm.py + llvm_run.py (native: clang -O2 +
            runtime/native/*.o — differentially tested vs the interpreter)
          - compiler/codegen_clif.py + cps_frames.py (Cranelift text; lags LLVM)
```

`compiler/pipeline.py` is the entry point: `run_pipeline_from_source`
raises `BorrowCheckError` / `TypeCheckError` in strict mode;
`emit_llvm_from_source` + `llvm_run.compile_and_run` produce and execute
native binaries. The interpreter provides delimited single-shot effect
continuations, trait dispatch on runtime types, Vec/vector runtime, a
bounds-checked simulated C heap for FFI, and strict name resolution. The
LLVM backend mirrors it with mode-based memory (stack default, @global
malloc+free, write-once boxes), tagged-union enums, {fn,env} closures with
indirect calls, a ucontext coroutine scheduler for effects, and real SIMD
for statically-sized vectors — anything unprovable demotes to a reasoned
placeholder, never wrong code. The standard library lives in `std/*.mx`
(Ante-modeled, effect-based idioms) and resolves via `import std.foo`.

## Where enforcement lives

- Borrow/mode/locality/linearity: `frozen_borrow_checker.py` driven by
  `frozen_constraint_emitter.py` (deep field-mode rules included).
- Type classes and conflicts: `frozen_constraint_checker.py` plus the
  constraint-graph conflict detection in `simplesub_adapter.py`
  (`1 + "a"` and non-bool conditions fail compilation).
- Effects: classes (`stack`/`suspend`) checked in the emitter;
  runtime semantics in `mir_interp.py`; CPS state machines for
  suspending functions in `codegen_clif.py`.

## Current status and remaining work

See `docs/v1_gap_analysis.md` for the authoritative status: what is
implemented, what is intentionally out of scope (FFI/threads runtime,
try/catch semantics, CLIF-level effect dispatch, frame chaining across
suspending calls, full biunification), and why. `docs/compiler_roadmap.md`
tracks the phase-by-phase plan.

## Conventions

- Tests must go through parsed source (parse -> ... -> interpreter), not
  hand-built HIR/MIR fixtures: the historical bug pattern here was seams
  where constructs silently degraded (patterns to wildcards, dropped
  modes) while fixture-based tests stayed green.
- The interpreter and checkers are strict: prefer a clear error over a
  silent no-op fallback. Do not add lenient fallbacks to make a test pass.
- New compiler behavior needs a regression test in
  `src/metaxu/compiler/tests/`, and the example gates must stay green
  (19/19 pipeline AND 19/19 run; do not regress either).
- Native lowering claims need differential tests (native stdout/exit ==
  interpreter) and, for memory claims, ASan runs scoped to the documented
  contract (full leak-check where frees are claimed; detect_leaks=0 where
  leak-by-design applies).
- Writing real library code in std/ is the best compiler stress test:
  several silent-degradation bugs were only found that way.
