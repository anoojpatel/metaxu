"""The recursion budget the Python-hosted phases run under.

Metaxu's front end and its MIR interpreter are both written as ordinary
recursive Python, so a deep Metaxu program — a nested source expression at
compile time, a structural recursion at run time — is a deep *Python* call
stack.  With CPython's stock 1000-frame limit that ceiling lands
embarrassingly low, and it announces itself as a host `RecursionError`
traceback rather than as a Metaxu diagnostic.

WHY A METAXU FRAME COSTS SEVERAL PYTHON FRAMES.  The interpreter spends
`_call_func` -> `_run_blocks` -> `_run_ops` -> `_eval_rhs` -> next
`_call_func` per Metaxu call.  Measured on this tree:

  * ~4 Python frames per Metaxu frame for a bare `f(n) -> f(n - 1)`
    recursion (247 Metaxu frames at the stock 1000 limit);
  * ~24 Python frames per level for a realistic structural recursion —
    examples/app's evaluator, where each AST node costs a `match`, a
    closure call, a `perform` and a std helper — which is where the ~45
    levels that users actually hit came from (45 * 24 ~ 1000).

`RECURSION_LIMIT` is therefore chosen against the EXPENSIVE shape: at
100_000 Python frames the examples/app evaluator reaches ~4_100 nested
nodes (measured) and a plain recursion ~24_700 (measured), both far past
the 1_000-Metaxu-frame target.

WHY RAISING IT IS NOT A SEGFAULT WAITING TO HAPPEN.  `sys.setrecursionlimit`
does not protect the C stack, so the number cannot be picked blind.  Two
CPython properties make this one safe, and both are pinned by tests that
run in a SUBPROCESS and assert the exit code (a segfault shows up there as
a negative return code, which no `except` clause could hide):

  * CPython >= 3.11 allocates Python frames on the heap in data-stack
    chunks and does not push C frames for a Python-to-Python call, so
    interpreter recursion costs heap, not C stack.  (Measured: a
    20_000-frame Metaxu recursion completes on a thread given a 128 KiB
    stack.)
  * CPython >= 3.12 guards genuinely C-recursive work — nested `repr`,
    comparison, deallocation — with a SEPARATE C-recursion limit that
    `setrecursionlimit` does not move.  Overrunning it raises
    `RecursionError` rather than smashing the stack.

The budget is installed SCOPED, around an entry point, and put back
afterwards: this package is a library (pytest, the LSP server and
`scripts/run_examples.py` all import it), and permanently rewriting a
process-global knob at import time would change the behaviour of code that
never asked to run a Metaxu program.  The guard is re-entrant and never
lowers a limit an embedder deliberately raised.
"""
from __future__ import annotations

import functools
import sys
import threading
from contextlib import contextmanager

#: Python frames allowed while a Metaxu phase runs. See the module docstring
#: for the measurement and the safety argument.
RECURSION_LIMIT = 100_000

#: Extra frames granted *after* the ceiling is hit, so that turning the
#: overflow into a diagnostic — building the message, walking the unwind
#: handlers, tearing down handle scopes — cannot itself overflow. Granted
#: idempotently, and discarded when the outermost budget exits.
RECURSION_SLACK = 2_000

_LOCK = threading.RLock()
_DEPTH = 0
_SAVED_LIMIT: int | None = None
_CEILING: int | None = None


def current_ceiling() -> int:
    """The limit the active budget installed, or the process limit if none."""
    return _CEILING if _CEILING is not None else sys.getrecursionlimit()


@contextmanager
def recursion_budget(limit: int = RECURSION_LIMIT):
    """Run the body with at least `limit` Python frames available.

    Re-entrant; the host's limit is restored (slack included) when the
    outermost budget exits.
    """
    global _DEPTH, _SAVED_LIMIT, _CEILING
    with _LOCK:
        _DEPTH += 1
        outermost = _DEPTH == 1
        if outermost:
            _SAVED_LIMIT = sys.getrecursionlimit()
            _CEILING = max(_SAVED_LIMIT, limit)
            sys.setrecursionlimit(_CEILING)
    try:
        yield
    finally:
        with _LOCK:
            _DEPTH -= 1
            if _DEPTH == 0:
                if _SAVED_LIMIT is not None:
                    sys.setrecursionlimit(_SAVED_LIMIT)
                _SAVED_LIMIT = None
                _CEILING = None


def grant_slack() -> None:
    """Make room for the code that reports an overflow.

    Idempotent: the target is measured from the installed ceiling, not from
    the current limit, so repeated calls (one per thread that overflowed,
    say) cannot ratchet the limit upwards without bound.
    """
    target = current_ceiling() + RECURSION_SLACK
    if sys.getrecursionlimit() < target:
        sys.setrecursionlimit(target)


def compiler_phase(fn):
    """Run a compile-time phase under the budget, reporting an overrun as a
    Metaxu diagnostic.

    Applied to the front-end front doors (`pipeline.build_context_from_source`
    and friends, `HIRBuilder.build`, `lower_hir_to_mir`) so that a source file
    nesting deeper than the compiler can walk gets a `CompileError` saying so,
    rather than a bare host `RecursionError` — or, when PLY caught the
    overflow mid-parse, a `ParseError: maximum recursion depth exceeded`
    blaming a syntax error that did not exist. `mir_interp` has the run-time
    twin (`RecursionLimitExceeded`), which additionally names the innermost
    Metaxu function, something no compile-time phase can do generically.

    Nesting is free: the budget is re-entrant, and once an inner phase has
    converted the overflow the outer phases see a `CompileError`.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with recursion_budget():
            try:
                return fn(*args, **kwargs)
            except RecursionError:
                # Slack FIRST: the stack is at the ceiling right now, and
                # building/formatting the diagnostic needs frames of its own.
                grant_slack()
                from metaxu.errors import CompileError
                raise CompileError(
                    message=(
                        "the program nests deeper than the compiler can walk "
                        f"({RECURSION_LIMIT} nested Python frames). This is a "
                        "compiler limit, not a syntax error: flatten the "
                        "deepest expression, or split it across several "
                        "statements."),
                    error_type="RecursionLimit",
                ) from None
    return wrapper
