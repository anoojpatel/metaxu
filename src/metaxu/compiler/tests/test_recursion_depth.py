"""How deep a Metaxu program may recurse, and how it fails when it can't.

Background. A Metaxu call frame is not a Python call frame: the interpreter
spends `_call_func` -> `_run_blocks` -> `_run_ops` -> `_eval_rhs` -> the next
`_call_func` per Metaxu call. Measured on this tree that is ~4 Python frames
for a bare `f(n) -> f(n - 1)` recursion and ~24 for a realistic structural
one (examples/app's evaluator, where a node costs a match, a closure call, a
`perform` and a std helper). At CPython's stock 1000-frame limit that put the
ceiling at 247 bare frames — and about 45 levels of the real evaluator, which
is where users hit it, with a raw host `RecursionError` traceback rather than
a Metaxu diagnostic.

What is pinned here:

* depth — a parsed-source program recursing 500 and 5_000 deep produces the
  right answer, on the plain path AND inside a `handle` body (which the
  interpreter runs on its own thread, so it has its own stack);
* clean failure — overrunning the ceiling raises `RecursionLimitExceeded`
  (an `InterpError`), matched on its MESSAGE, never a `RecursionError`, and
  the conversion itself does not overflow;
* no segfault — `sys.setrecursionlimit` does not protect the C stack, so
  raising it carelessly turns a clean exception into a crash. The subprocess
  tests run the interpreter just under, at and far past the ceiling and
  assert a clean exit status: a segfault shows up as a negative returncode,
  which no `except` clause in the child could hide;
* no global leak — the process-wide recursion limit is exactly what it was
  before the program ran.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from metaxu.compiler import recursion
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import (InterpError, MirInterpreter,
                                        RecursionLimitExceeded)
from metaxu.compiler.pipeline import build_context_from_source, run_pipeline_from_source
from metaxu.errors import CompileError


def run(source: str, fn: str, args: list):
    """parse -> desugar -> freeze -> infer -> HIR -> MIR -> interpret."""
    ctx = build_context_from_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    return interp.call(fn, args)


# A plain self-call: the cheapest shape, ~4 Python frames per Metaxu frame.
PLAIN = """
fn countdown(n: int) -> int {
    if n <= 0 { 0 } else { countdown(n - 1) + 1 }
}
"""

# Structural recursion over an enum, the shape a tree walk / evaluator has.
TREE = """
enum Tree { Leaf(v: int), Node(l: Tree, r: Tree) }

fn build(n: int) -> Tree {
    if n <= 0 { Leaf(1) } else { Node(build(n - 1), Leaf(1)) }
}

fn total(t: Tree) -> int {
    match t {
        Leaf(v) => v,
        Node(l, r) => total(l) + total(r)
    }
}

fn deep_sum(n: int) -> int { total(build(n)) }
"""

# The same recursion INSIDE a handle body. `handle_scope` runs the body on
# its own thread, so this exercises a different stack from the plain path.
IN_HANDLE = """
effect Counter { fn tick() -> int }

fn deep(n: int) -> int performs Counter {
    if n <= 0 { perform Counter.tick() } else { deep(n - 1) + 1 }
}

fn counted(n: int) -> int {
    handle Counter with {
        tick() -> resume(1)
    } in {
        deep(n)
    }
}
"""

# No base case: must hit the ceiling, not run forever and not crash.
RUNAWAY = """
fn forever(n: int) -> int { forever(n + 1) }
"""

RUNAWAY_IN_HANDLE = """
effect Counter { fn tick() -> int }

fn forever(n: int) -> int performs Counter { forever(n + 1) }

fn counted(n: int) -> int {
    handle Counter with {
        tick() -> resume(1)
    } in {
        forever(n)
    }
}
"""

# A runaway inside `try`: recursion exhaustion is interpreter resource
# exhaustion, not a program failure, so `catch` must NOT swallow it.
RUNAWAY_IN_TRY = """
fn forever(n: int) -> int { forever(n + 1) }

fn guarded(n: int) -> int {
    try { forever(n) } catch e { 0 - 1 }
}
"""


# ---------------------------------------------------------------------------
# Depth that actually works
# ---------------------------------------------------------------------------

def test_plain_recursion_500_deep():
    """The depth the old 1000-frame limit could not reach (it stopped at 247)."""
    assert run(PLAIN, "countdown", [500]) == 500


def test_plain_recursion_5000_deep():
    """Well past the 1_000-Metaxu-frame target.

    Also a live check on the frame-cost assumption: 5_000 Metaxu frames only
    fit under RECURSION_LIMIT if the interpreter still costs well under 20
    Python frames per Metaxu frame. If a refactor doubles that cost, this
    fails instead of silently halving every user's usable depth.
    """
    assert run(PLAIN, "countdown", [5000]) == 5000


def test_structural_recursion_500_deep():
    """A tree walk, the shape real programs (parsers, evaluators) recurse in."""
    assert run(TREE, "deep_sum", [500]) == 501


def test_recursion_inside_handle_body_500_deep():
    """The effect-scheduler path: the body runs on its own thread."""
    assert run(IN_HANDLE, "counted", [500]) == 501


def test_recursion_inside_handle_body_5000_deep():
    assert run(IN_HANDLE, "counted", [5000]) == 5001


# ---------------------------------------------------------------------------
# Failing cleanly at the ceiling
# ---------------------------------------------------------------------------

def _assert_clean_recursion_diagnostic(exc: RecursionLimitExceeded,
                                       innermost: str) -> None:
    # Matched on the MESSAGE, not the type alone: an InterpError whose text
    # was a host repr would still be a leaked Python failure.
    assert "recursion limit exceeded" in exc.message
    assert str(recursion.RECURSION_LIMIT) in exc.message
    # The innermost Metaxu function, via the usual `locate` note.
    assert f"while calling {innermost!r}" in exc.message
    # No host-language wreckage anywhere in the rendered diagnostic.
    rendered = str(exc)
    assert "Traceback" not in rendered
    assert "RecursionError" not in rendered
    assert "maximum recursion depth" not in rendered
    # And the host exception it replaced is suppressed, not chained: `raise
    # ... from None` keeps a 100_000-frame "During handling of the above
    # exception" Python traceback out of every report that renders it.
    assert exc.__cause__ is None
    assert exc.__suppress_context__ is True


def test_runaway_recursion_raises_metaxu_diagnostic():
    with pytest.raises(RecursionLimitExceeded) as excinfo:
        run(RUNAWAY, "forever", [0])
    _assert_clean_recursion_diagnostic(excinfo.value, "forever")


def test_recursion_limit_error_is_an_interp_error():
    """So every existing `except InterpError` reporting path keeps working."""
    assert issubclass(RecursionLimitExceeded, InterpError)


def test_runaway_recursion_never_leaks_recursion_error():
    """RecursionError is a RuntimeError, not an InterpError: if the
    conversion regressed, this would fail with the host exception."""
    try:
        run(RUNAWAY, "forever", [0])
    except RecursionLimitExceeded:
        pass
    except RecursionError:  # pragma: no cover - the regression this guards
        pytest.fail("host RecursionError leaked across the language boundary")


def test_runaway_recursion_inside_handle_body_is_clean():
    """The scheduler-thread path converts too, and keeps the innermost name."""
    with pytest.raises(RecursionLimitExceeded) as excinfo:
        run(RUNAWAY_IN_HANDLE, "counted", [0])
    _assert_clean_recursion_diagnostic(excinfo.value, "forever")


def test_try_catch_does_not_swallow_recursion_exhaustion():
    """A catch arm would run with the stack still at the ceiling; recovering
    onto an exhausted stack is not something the language offers (and the
    native backend has no recoverable equivalent). See docs/try_catch.md."""
    with pytest.raises(RecursionLimitExceeded):
        run(RUNAWAY_IN_TRY, "guarded", [0])


def test_recursion_limit_is_restored_after_the_program_runs():
    """The budget is scoped to the call: importing the interpreter must not
    silently rewrite a process-global knob for the whole host process."""
    before = sys.getrecursionlimit()
    assert run(PLAIN, "countdown", [500]) == 500
    assert sys.getrecursionlimit() == before
    with pytest.raises(RecursionLimitExceeded):
        run(RUNAWAY, "forever", [0])
    # Including the slack granted while reporting the overflow.
    assert sys.getrecursionlimit() == before


# ---------------------------------------------------------------------------
# The ceiling is not a segfault (subprocess: a crash cannot be caught)
# ---------------------------------------------------------------------------

_CHILD = textwrap.dedent("""
    import sys
    from metaxu.compiler.hir import HIRBuilder
    from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
    from metaxu.compiler.mir_interp import MirInterpreter, RecursionLimitExceeded
    from metaxu.compiler.pipeline import build_context_from_source

    source, fn, depth = sys.argv[1], sys.argv[2], int(sys.argv[3])
    ctx = build_context_from_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    try:
        print("RESULT", interp.call(fn, [depth]))
    except RecursionLimitExceeded as exc:
        print("CEILING", exc.message[:60])
""")


def _child(source: str, fn: str, depth: int) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", _CHILD, source, fn, str(depth)],
                          capture_output=True, text=True, timeout=300)


# ~4 Python frames per Metaxu frame on the plain path, so the ceiling lands
# near RECURSION_LIMIT / 4. These bracket it: the first must finish, the
# second must be refused. Both must exit 0 — a stack smash would be a
# negative returncode (-SIGSEGV) that the child could not intercept.
_JUST_UNDER = recursion.RECURSION_LIMIT // 5
_WELL_OVER = recursion.RECURSION_LIMIT


@pytest.mark.parametrize("source,fn,depth,expected", [
    (PLAIN, "countdown", _JUST_UNDER, "RESULT"),
    (PLAIN, "countdown", _WELL_OVER, "CEILING"),
    (IN_HANDLE, "counted", _JUST_UNDER, "RESULT"),
    (IN_HANDLE, "counted", _WELL_OVER, "CEILING"),
    (RUNAWAY, "forever", 0, "CEILING"),
])
def test_near_and_past_the_ceiling_never_crashes(source, fn, depth, expected):
    proc = _child(source, fn, depth)
    assert proc.returncode == 0, (
        f"interpreter exited {proc.returncode} (negative = fatal signal, i.e. "
        f"the raised recursion limit overflowed the C stack)\n{proc.stderr[-2000:]}")
    assert proc.stdout.startswith(expected), (proc.stdout, proc.stderr[-2000:])
    assert "Traceback" not in proc.stderr


# ---------------------------------------------------------------------------
# The compile-time twin: a deeply NESTED source expression
# ---------------------------------------------------------------------------

def _nested_sum(terms: int) -> str:
    return "fn f() -> int { " + " + ".join(["1"] * terms) + " }"


def test_deeply_nested_source_expression_compiles():
    """Every front-end phase walks the AST recursively, so nesting is depth
    here too: ~200 terms used to overflow during compilation."""
    run_pipeline_from_source(_nested_sum(1000))
    assert run(_nested_sum(1000), "f", []) == 1000


def test_source_nested_past_the_compiler_ceiling_is_a_metaxu_diagnostic():
    """Not a host RecursionError, and not the PLY-mediated `ParseError:
    maximum recursion depth exceeded` that used to blame a syntax error."""
    with pytest.raises(CompileError) as excinfo:
        run_pipeline_from_source(_nested_sum(30000))
    rendered = str(excinfo.value)
    assert "nests deeper than the compiler can walk" in rendered
    assert "maximum recursion depth" not in rendered
    assert "Traceback" not in rendered
