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
source -> Parser (PLY, src/metaxu/parser.py; token disambiguation in lexer.py)
       -> desugar passes (compiler/desugar.py: trait impls -> mangled fns)
       -> freeze (compiler/mutaxu_ast.py: immutable AST + JSON goldens)
       -> infer (compiler/infer_tables.py -> simplesub_adapter.py
                 + frozen_constraint_emitter.py + frozen_borrow_checker.py)
       -> HIR (compiler/hir.py: typed, patterns, effects, trait dispatch)
       -> MIR (compiler/lower_hir_to_mir.py: ANF, multi-block CFG)
       -> CLIF text (compiler/codegen_clif.py) + CPS frames (cps_frames.py)
```

`compiler/pipeline.py` is the entry point: `run_pipeline_from_source`
raises `BorrowCheckError` / `TypeCheckError` in strict mode.
`compiler/mir_interp.py` executes MIR directly (the execution gate's
engine): delimited single-shot effect continuations, trait dispatch on
runtime types, Vec/vector runtime, strict name resolution.

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
  (19/19 pipeline; do not regress the run-stage count).
