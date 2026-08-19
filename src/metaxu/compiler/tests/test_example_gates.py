"""Pin the example gates inside pytest.

`scripts/run_examples.py` is the merge gate, but living outside pytest it
can silently drift. This file pins the exact same facts:
- all example/root .mx files compile through the full pipeline (with the
  two negative fixtures REJECTED with the right diagnostic class), and
- the files known to execute keep executing (with printed output pinned
  for the ones whose docs promise specific values).
"""
from __future__ import annotations

import glob
import os

import pytest

from metaxu.compiler.pipeline import (
    BorrowCheckError,
    TypeCheckError,
    build_context_from_source,
    run_pipeline_from_source,
)
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import MirInterpreter, UNIT

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

NEGATIVE = {
    "test_borrow_check.mx": BorrowCheckError,
    "test_type_error.mx": TypeCheckError,
}

# Files that must EXECUTE (entry found and runs without error).
MUST_RUN = [
    "examples/app/main.mx",
    "examples/01_modes_and_references.mx",
    "examples/02_effects_and_handlers.mx",
    "examples/03_modules_and_imports.mx",
    "examples/04_advanced_types.mx",
    "examples/05_unsafe_and_ffi.mx",
    "examples/06_vector_operations.mx",
    "examples/10_traits_and_structs.mx",
    "examples/effect_mapping.mx",
    "examples/effects.mx",
    "examples/hello.mx",
    "examples/linked_list.mx",
    "examples/ownership.mx",
    "test_locality_heap.mx",
    "test_locality_local.mx",
    "test_locality_ref.mx",
    "test_operations.mx",
]


def all_targets() -> list[str]:
    # examples/app/ is a multi-file application: only its ENTRY file is a
    # target (its sibling modules are reached through imports, and compiling
    # one on its own would just be the same code with no main).
    return sorted(
        os.path.relpath(p, REPO_ROOT)
        for p in glob.glob(os.path.join(REPO_ROOT, "examples", "*.mx"))
        + glob.glob(os.path.join(REPO_ROOT, "examples", "app", "main.mx"))
        + glob.glob(os.path.join(REPO_ROOT, "test_*.mx"))
    )


def execute(rel_path: str):
    # file_path matters: multi-file imports resolve relative to the entry
    # file's directory.
    path = os.path.join(REPO_ROOT, rel_path)
    source = open(path).read()
    ctx = build_context_from_source(source, file_path=path)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    prints: list[str] = []
    interp.register_builtin(
        "print", lambda *a: (prints.append(" ".join(str(x) for x in a)), UNIT)[1]
    )
    entry = "main" if "main" in interp._funcs else "example"
    return interp.call(entry, []), prints


@pytest.mark.parametrize("rel_path", all_targets())
def test_pipeline_gate(rel_path):
    path = os.path.join(REPO_ROOT, rel_path)
    source = open(path).read()
    expected = NEGATIVE.get(os.path.basename(rel_path))
    if expected is not None:
        with pytest.raises(expected):
            run_pipeline_from_source(source, file_path=path)
    else:
        run_pipeline_from_source(source, file_path=path)


@pytest.mark.parametrize("rel_path", MUST_RUN)
def test_run_gate(rel_path):
    execute(rel_path)  # must not raise


def test_effects_example_output():
    _, prints = execute("examples/02_effects_and_handlers.mx")
    assert prints == ["Current value: 0", "New value: 1"]


def test_effects_fstring_example_output():
    """effects.mx logs f"Counter value: {x}"; with f-string interpolation
    real (parse-time desugar to concat + to_string) the printed line is the
    interpolated value, not the literal braces text.  x = State.get() + 1
    with get resuming 0, so the value is 1."""
    _, prints = execute("examples/effects.mx")
    assert prints == ["Starting counter", "Counter value: 1", "Done"]


def test_traits_example_output():
    """The example's own comment promises size 2 -> pops 17."""
    _, prints = execute("examples/10_traits_and_structs.mx")
    assert any("17" in line for line in prints)


def test_modules_example_output():
    """Dot product of (1,2)·(3,4) = 11."""
    _, prints = execute("examples/03_modules_and_imports.mx")
    assert any("11" in line for line in prints)


def test_unsafe_ffi_example_output(monkeypatch, tmp_path):
    """05_unsafe_and_ffi.mx: the Buffer path (malloc/memcpy/free over the
    simulated C heap) runs silently; File.open("test.txt") hits the real
    fopen shim, which returns null when the file does not exist, so the
    match takes the Err arm and prints exactly the error line.  cwd is
    pinned to an empty tmp dir so a stray test.txt cannot flip the arm."""
    monkeypatch.chdir(tmp_path)
    result, prints = execute("examples/05_unsafe_and_ffi.mx")
    assert prints == ["Error: Failed to open file"]


# ---------------------------------------------------------------------------
# Showcase suite (benchmarks/suite/): programs must stay compile-clean
# ---------------------------------------------------------------------------
# The suite races C twins, so a program that silently starts demoting
# would still "run" under the interpreter but the native binary would
# lose functions. Pin: strict pipeline passes AND zero native
# placeholders for every suite program.

def test_showcase_suite_programs_compile_native_clean():
    import glob as _glob
    from metaxu.compiler.pipeline import emit_llvm_from_source
    suite = sorted(_glob.glob(os.path.join(REPO_ROOT, "benchmarks",
                                           "suite", "*.mx")))
    assert len(suite) >= 6, suite
    allowed_placeholders = {
        # std.stream's unused drivers demote honestly; the pipeline's
        # EXECUTED chain is native (it links and runs). Everything else
        # must be completely clean.
        "pipeline.mx",
    }
    for path in suite:
        ir = emit_llvm_from_source(open(path).read(), file_path=path)
        if os.path.basename(path) not in allowed_placeholders:
            assert "placeholder -- unsupported" not in ir, path


def test_emission_diagnostics_programs_compile_native_clean():
    # Same rot-prevention for benchmarks/diagnostics: each pair isolates
    # one emission pattern, so a demotion would silently turn its row
    # into a measurement of the placeholder, not the pattern.
    import glob as _glob
    from metaxu.compiler.pipeline import emit_llvm_from_source
    diags = sorted(_glob.glob(os.path.join(REPO_ROOT, "benchmarks",
                                           "diagnostics", "*.mx")))
    assert len(diags) >= 8, diags
    for path in diags:
        ir = emit_llvm_from_source(open(path).read(), file_path=path)
        assert "placeholder -- unsupported" not in ir, path
