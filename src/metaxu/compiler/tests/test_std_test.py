"""std.test: assertions with reporting as an effect."""
from __future__ import annotations

import os

from metaxu.compiler.pipeline import build_context_from_source
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import MirInterpreter, UNIT

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def run_main(source: str):
    path = os.path.join(REPO_ROOT, "__std_test_probe__.mx")
    ctx = build_context_from_source(source, file_path=path)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    prints: list[str] = []
    interp.register_builtin(
        "print", lambda *a: (prints.append(" ".join(str(x) for x in a)), UNIT)[1]
    )
    return interp.call("main", []), prints


def test_defaults_print_ok_and_fail_lines():
    _, prints = run_main("""
from std.test import Report, check;

fn main() -> int {
    check(1 + 1 == 2, "math works");
    check(1 == 2, "math broken");
    0
}
""")
    assert prints == ["ok math works", "FAIL math broken"]


def test_run_suite_counts_failures():
    result, prints = run_main("""
from std.test import check, check_eq, run_suite;

fn my_tests() -> () {
    check(true, "a");
    check_eq(2 + 2, 4, "b");
    check_eq(2 + 2, 5, "c");
}

fn main() -> int {
    run_suite("arith", fn() -> my_tests())
}
""")
    assert result == 1  # one failure -> exit-code style return
    assert any("expected 5, got 4" in p for p in prints)
    assert any("arith" in p and "2 passed" in p and "1 failed" in p for p in prints)


def test_all_green_suite_returns_zero():
    result, prints = run_main("""
from std.test import check, run_suite;

fn my_tests() -> () {
    check(true, "x");
    check(true, "y");
}

fn main() -> int {
    run_suite("green", fn() -> my_tests())
}
""")
    assert result == 0
    assert any("2 passed," in p and "0 failed" in p for p in prints)
