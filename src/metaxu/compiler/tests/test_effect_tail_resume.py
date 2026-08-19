"""Tail-resume trampoline regression tests (compiler/effect_tail.py,
runtime/native/metaxu_effects.c THE PUMP MODEL, mir_interp._pump_scope).

The bug this pins: the handler pump used to be RECURSIVE per event on both
engines — natively each stream element left a (case frame + resume frame)
pair on the scope's owner stack (~250 B/element against the 1 MiB fiber
stacks: a std.stream pipeline segfaulted between 2,000 and 5,000
elements), and every perform leaked its mx_k (~1 KB ucontext) until scope
teardown; the interpreter hit its 100k-frame recursion budget between
10,000 and 20,000 elements.  The fix: a resume in TAIL position (the
case's value IS resume's value, nothing after it — the shape of every
std.stream/std.state/std.log arm except fold's) is handed back to the
scope's event pump, which switches into the body from a constant frame
and frees the consumed continuation record.  Non-tail resumes keep the
general recursive path — that recursion IS foldr's semantics.

All tests go through parsed source, per the project convention.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from metaxu.compiler.codegen_llvm import emit_llvm
from metaxu.compiler.effect_tail import (
    handler_case_fns, program_tail_resume_ids, tail_resume_ids)
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.llvm_run import compile_and_run
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import (
    MirInterpreter, RecursionLimitExceeded, UNIT)
from metaxu.compiler.pipeline import (
    build_context_from_source, run_pipeline_ctx)

needs_clang = pytest.mark.skipif(
    shutil.which("clang") is None, reason="clang is not installed")


def _asan_available() -> bool:
    if shutil.which("clang") is None:
        return False
    import os
    import tempfile
    with tempfile.TemporaryDirectory(prefix="metaxu_asan_probe_") as d:
        c = os.path.join(d, "t.c")
        with open(c, "w") as fh:
            fh.write("int main(void){return 0;}\n")
        proc = subprocess.run(
            ["clang", "-fsanitize=address", c, "-o", os.path.join(d, "t")],
            capture_output=True, text=True)
        return proc.returncode == 0


needs_asan = pytest.mark.skipif(
    not _asan_available(),
    reason="clang ASan runtime not available (compile probe failed)")


def mir_from_source(source: str, monomorphize: bool = False):
    ctx = build_context_from_source(source)
    run_pipeline_ctx(ctx)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    if monomorphize:
        from metaxu.compiler.monomorphize import (
            collect_signatures, monomorphize_hir)
        hir = monomorphize_hir(hir, collect_signatures(ctx.id_map))
    return lower_hir_to_mir(hir)


def llvm_from_source(source: str) -> str:
    # Through monomorphization, like pipeline.emit_llvm_from_source.
    return emit_llvm(mir_from_source(source, monomorphize=True))


def interp_run(source: str, entry: str = "main"):
    """Run through the MIR interpreter (unmonomorphized MIR — the
    semantics reference); returns (result, stdout-equivalent)."""
    interp = MirInterpreter()
    interp.load(mir_from_source(source))
    out: list[str] = []

    def _print(*args):
        out.append(" ".join(str(a) for a in args))
        return UNIT

    interp.register_builtin("print", _print)
    interp.register_builtin("println", _print)
    result = interp.call(entry, [])
    return result, "".join(line + "\n" for line in out)


def assert_native_matches_interp(source: str, tmp_path, entry: str = "main",
                                 clang_args: tuple = ()):
    result, expected_out = interp_run(source, entry)
    ir = llvm_from_source(source)
    exit_code, stdout = compile_and_run(ir, entry, workdir=str(tmp_path),
                                        clang_args=clang_args)
    assert stdout == expected_out
    if result is not UNIT and isinstance(result, (bool, int)):
        assert exit_code == int(result) % 256
    return ir

# ---------------------------------------------------------------------------
# The analysis: which arms are tail, straight from std.stream's real MIR
# ---------------------------------------------------------------------------

_STREAM_PIPELINE = """
from std.stream import iota, sum, map, filter, fold;
fn main() -> int {
    let piped = sum(map(filter(iota(200), fn(x: int) -> x % 3 == 0),
                        fn(x: int) -> x * x));
    let folded = fold(iota(10), 0, fn(x: int, acc: int) -> x + acc);
    print(piped);
    print(folded);
    0
}
"""


def _tail_map():
    funcs = mir_from_source(_STREAM_PIPELINE)
    by_name = {f.name: f for f in funcs}
    cases = handler_case_fns(funcs)
    return by_name, cases


def test_stream_tail_arms_are_marked():
    by_name, cases = _tail_map()
    # iter (sum's engine), map, filter: `resume(())` ends the arm — tail.
    for stem in ("iter", "map", "filter"):
        (name,) = [n for n in cases if f"std.stream.{stem}$" in n]
        assert tail_resume_ids(by_name[name]), \
            f"{name} should have a tail-marked resume"


def test_fold_arm_is_not_marked():
    # fold: `f(x, resume(()))` does work AFTER the resume — the pending
    # f-applications are foldr's semantics.  Must stay on the general path.
    by_name, cases = _tail_map()
    (name,) = [n for n in cases if "std.stream.fold$" in n]
    assert tail_resume_ids(by_name[name]) == frozenset()


def test_non_case_functions_are_never_marked():
    # program_tail_resume_ids only marks resumes inside handler-case
    # functions; the ids it returns must all belong to case fns.
    funcs = mir_from_source(_STREAM_PIPELINE)
    cases = handler_case_fns(funcs)
    marked = program_tail_resume_ids(funcs)
    owning = set()
    for f in funcs:
        for b in f.blocks:
            for op in b.ops:
                if id(op) in marked:
                    owning.add(f.name)
    assert owning and owning <= cases


# ---------------------------------------------------------------------------
# Interpreter: constant pump depth (this exact run hit the 100k-frame
# recursion budget before the trampoline: the ceiling was 10k-20k elements)
# ---------------------------------------------------------------------------

_PIPELINE_20K = """
from std.stream import iota, sum, map, filter;
fn main() -> int {
    let piped = sum(map(filter(iota(20000), fn(x: int) -> x % 3 == 0),
                        fn(x: int) -> x * x));
    print(piped);
    0
}
"""


def test_interp_pipeline_20k_runs_flat():
    result, out = interp_run(_PIPELINE_20K)
    assert result == 0
    assert out == "888822218889\n"  # sum(x*x for x in range(20000) if x%3==0)


# ---------------------------------------------------------------------------
# Native: the reproducer past the old ~4k cliff, differentially checked
# ---------------------------------------------------------------------------

_PIPELINE_5K = """
from std.stream import iota, sum, map, filter;
fn main() -> int {
    let piped = sum(map(filter(iota(5000), fn(x: int) -> x % 3 == 0),
                        fn(x: int) -> x * x));
    print(piped);
    0
}
"""

_PIPELINE_1M = """
from std.stream import iota, sum, map, filter;
fn main() -> int {
    let piped = sum(map(filter(iota(1000000), fn(x: int) -> x % 3 == 0),
                        fn(x: int) -> x * x));
    print(piped);
    0
}
"""


@needs_clang
def test_native_pipeline_5k_past_the_old_cliff(tmp_path):
    # Segfaulted (exit -11) before the trampoline; differential vs interp.
    assert_native_matches_interp(_PIPELINE_5K, tmp_path)


@needs_clang
def test_native_pipeline_1m_stack_stable(tmp_path):
    # 1,000,000 elements through filter+map+sum: O(1) handler stack per
    # element and per-element continuation reclamation, or this dies long
    # before the end (the old cost was ~250 B stack + ~1 KB heap per
    # element).  Interpreter cross-check at this size would take minutes,
    # so the value is pinned (== sum(x*x for x in range(10**6) if x%3==0),
    # independently computed; the interpreter agrees at 200k — see the
    # measurement notes in docs/v1_gap_analysis.md).
    ir = llvm_from_source(_PIPELINE_1M)
    exit_code, stdout = compile_and_run(ir, "main", workdir=str(tmp_path))
    assert exit_code == 0
    assert stdout == "111111277777611111\n"


# ---------------------------------------------------------------------------
# Non-tail resumes still work (the general path is untouched)
# ---------------------------------------------------------------------------

_NON_TAIL_FOLD = """
from std.stream import iota, fold;
fn main() -> int {
    // foldr: 0+(1+(2+...(99+5))) — the arm f(x, resume(())) is NOT tail.
    let total = fold(iota(100), 5, fn(x: int, acc: int) -> x + acc);
    print(total);
    0
}
"""

_NON_TAIL_POST_RESUME = """
effect Count { tick() -> int }
fn main() -> int {
    // `let r = resume(...); r + 1` — work after the resume: general path.
    let n = handle Count with {
        tick() -> { let r = resume(7); r + 1 }
    } in {
        perform Count.tick() * 10
    };
    print(n);
    0
}
"""

_EARLY_STOP_TAKE = """
from std.stream import iota, sum, take;
fn main() -> int {
    // take's arm resumes tail-ly while seen < n and ABORTS (returns
    // without resuming) at the boundary: both paths in one arm.
    let total = sum(take(iota(1000000), 10));
    print(total);
    0
}
"""


@needs_clang
def test_native_fold_non_tail_differential(tmp_path):
    assert_native_matches_interp(_NON_TAIL_FOLD, tmp_path)


@needs_clang
def test_native_post_resume_work_differential(tmp_path):
    assert_native_matches_interp(_NON_TAIL_POST_RESUME, tmp_path)


@needs_clang
def test_native_take_early_stop_differential(tmp_path):
    assert_native_matches_interp(_EARLY_STOP_TAKE, tmp_path)


def test_interp_fold_still_recurses_per_element():
    # Honest cost pin: foldr's pending applications ARE per-element frames.
    # A million-element fold must exhaust the interpreter's recursion
    # budget as a CLEAN RecursionLimitExceeded (never a host RecursionError
    # and never a silent wrong answer).  If this ever passes, someone made
    # fold iterative — which would be a semantics change to audit, not a
    # free win.
    src = """
from std.stream import iota, fold;
fn main() -> int {
    fold(iota(1000000), 0, fn(x: int, acc: int) -> x + acc)
}
"""
    with pytest.raises(RecursionLimitExceeded):
        interp_run(src)


# ---------------------------------------------------------------------------
# Single-shot: a double resume whose SECOND resume is in tail position
# must die with the interpreter's message, not touch freed memory
# ---------------------------------------------------------------------------

_DOUBLE_RESUME_TAIL_SECOND = """
effect Ask { ask() -> int }
fn main() -> int {
    handle Ask with {
        ask() -> {
            let first = resume(1);
            resume(2)
        }
    } in {
        perform Ask.ask()
    }
}
"""


@needs_clang
def test_double_resume_tail_second_is_single_shot_fatal(tmp_path):
    # The first (general) resume consumes k but deliberately does NOT free
    # it, so the second (tail-position) resume's `used` check reads live
    # memory and aborts with the exact single-shot message.
    ir = llvm_from_source(_DOUBLE_RESUME_TAIL_SECOND)
    assert "call i64 @mx_resume_tail" in ir  # the second resume IS tail
    exit_code, _ = compile_and_run(ir, "main", workdir=str(tmp_path))
    assert exit_code != 0  # abort(), never a clean exit
    # Re-run the built binary to capture stderr (compile_and_run drops it).
    run = subprocess.run([str(tmp_path / "prog.bin")], capture_output=True,
                         text=True, timeout=60)
    assert run.returncode != 0
    assert "Continuation already consumed (single-shot violation)" in run.stderr


def test_interp_double_resume_tail_second_is_single_shot_error():
    with pytest.raises(RuntimeError) as ei:
        interp_run(_DOUBLE_RESUME_TAIL_SECOND)
    assert "single-shot violation" in str(ei.value)


# ---------------------------------------------------------------------------
# ASan: the new per-event continuation frees introduce no UAF/double-free
# ---------------------------------------------------------------------------

@needs_asan
def test_native_pipeline_asan_no_uaf(tmp_path):
    # detect_leaks=0 per the leak-by-design contract (heap closure envs);
    # exit 0 under ASan proves the pump's mx__free_k introduces no
    # use-after-free and no double-free across ~1667 tail-freed
    # continuations per scope (5000 elements, three scopes).
    result, expected_out = interp_run(_PIPELINE_5K)
    assert result == 0
    ir = llvm_from_source(_PIPELINE_5K)
    exit_code, stdout = compile_and_run(
        ir, "main", workdir=str(tmp_path),
        clang_args=("-fsanitize=address",),
        run_env={"ASAN_OPTIONS": "detect_leaks=0"})
    assert stdout == expected_out
    assert exit_code == 0
