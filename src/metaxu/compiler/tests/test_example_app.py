"""Golden output for examples/app — the multi-file calc application.

examples/app is the repo's one program that exercises the whole language at
once: five modules with import/export and visibility, a generic container
and two generic functions with `where` bounds, a trait with three impls
dispatched on the receiver's runtime type, a custom effect with three
different handlers, two stdlib effects (std.throw, std.state), tuples and
destructuring, exhaustive pattern matching over a recursive AST enum, six
std modules and @mut/@local/@global modes.

The pipeline and run gates already require it to compile and execute (see
test_example_gates.MUST_RUN); this file pins WHAT it prints, so a silent
semantic regression anywhere along that path — a dropped handler arm, a
mis-parsed precedence, an effect that stops threading state — shows up as a
diff instead of a still-green run.
"""
from __future__ import annotations

import os

from metaxu.compiler.pipeline import build_context_from_source
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import MirInterpreter, UNIT

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
APP_MAIN = os.path.join(REPO_ROOT, "examples", "app", "main.mx")

EXPECTED_OUTPUT = [
    '== arithmetic ==',
    '   | let radius = 7; let area = 3 * radius * radius; print area;',
    '   147',
    '   (7 steps)',
    '== precedence and unary minus ==',
    '   | print 2 + 3 * 4 - -6; print (2 + 3) * 4;',
    '   20',
    '   20',
    '   (13 steps)',
    '== conditionals and builtins ==',
    '   | let n = 17; print if n % 2 == 1 then max(n, 100) else min(n, 100); print abs(0 - n);',
    '   100',
    '   17',
    '   (14 steps)',
    '== comparison chains ==',
    '   | let a = 3; let b = 4; print a < b; print a * a + b * b == 25; print (a < b) != (b < a);',
    '   true',
    '   true',
    '   true',
    '   (21 steps)',
    '== multi-line source ==',
    '   | let a = 2;',
    '   | \tlet b = a * a;',
    '   | print b + a;',
    '   6',
    '   (7 steps)',
    '== unbound name ==',
    '   | print 1 + missing;',
    "   ! eval error: unbound name 'missing' (statement at offset 0)",
    '   (3 steps)',
    '== type error ==',
    '   | print if 1 then 2 else 3;',
    '   ! eval error: a condition must be a bool but got int in (if 1 then 2 else 3) (statement at offset 0)',
    '   (2 steps)',
    '== division by zero ==',
    '   | let d = 0; print 10 / d;',
    '   ! eval error: division by zero in (10 / d) (statement at offset 11)',
    '   (4 steps)',
    '== syntax error ==',
    '   | let x = (1 + 2; print x;',
    "   ! parse error: expected ')' but found ';' at offset 14",
    '   (0 steps)',
    '== parse tree ==',
    '   | let t = 1 + 2 * 3; print if t > 6 then abs(0 - t) else t;',
    '   let t = (1 + (2 * 3)) at offset 0',
    '   print (if (t > 6) then abs((0 - t)) else t) at offset 19',
    '== evaluation trace ==',
    '     2 => 2',
    '       3 => 3',
    '       4 => 4',
    '     (3 * 4) => 12',
    '   (2 + (3 * 4)) => 14',
    '== step budget ==',
    '   ! step budget of 3 exhausted at (3 * 4) (statement at offset 0)',
    'programs that failed: 4',
]


def run_app() -> tuple[object, list[str]]:
    source = open(APP_MAIN).read()
    ctx = build_context_from_source(source, file_path=APP_MAIN)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    prints: list[str] = []
    interp.register_builtin(
        "print", lambda *a: (prints.append(" ".join(str(x) for x in a)), UNIT)[1])
    return interp.call("main", []), prints


def test_app_golden_output():
    result, prints = run_app()
    assert prints == EXPECTED_OUTPUT
    # main() answers the number of demo programs that reported an error:
    # the four deliberate failure cases.
    assert result == 4


def test_app_error_paths_are_reported_not_crashes():
    """Every failing demo comes back as a message line, never as an escaped
    exception: the Throw handler in app.main is what turns a lexer, parser
    or evaluator failure into output."""
    _, prints = run_app()
    failures = [line for line in prints if line.strip().startswith("!")]
    assert len(failures) == 5          # 4 demos + the step-budget cut-off
    assert all("error" in line or "budget" in line for line in failures)


def test_app_step_counts_come_from_the_state_effect():
    """std.state threads the evaluator's node counter: a program that
    fails early is charged fewer steps than one that completes."""
    _, prints = run_app()
    steps = [line for line in prints if line.strip().endswith("steps)")]
    assert steps[0] == "   (7 steps)"          # arithmetic, runs to the end
    assert steps[-1] == "   (0 steps)"         # syntax error: never evaluated
