"""std.log: logging as an effect — defaults print, handlers override."""
from __future__ import annotations

import os

from metaxu.compiler.pipeline import build_context_from_source
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import MirInterpreter, UNIT

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def run_main(source: str):
    path = os.path.join(REPO_ROOT, "__std_log_probe__.mx")
    ctx = build_context_from_source(source, file_path=path)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    prints: list[str] = []
    interp.register_builtin(
        "print", lambda *a: (prints.append(" ".join(str(x) for x in a)), UNIT)[1]
    )
    return interp.call("main", []), prints


def test_default_handlers_print_with_level_prefix():
    _, prints = run_main("""
from std.log import Log;

fn main() -> int {
    perform Log.info("starting");
    perform Log.warn("low disk");
    0
}
""")
    assert prints == ["[info] starting", "[warn] low disk"]


def test_quietly_suppresses_all_logging():
    _, prints = run_main("""
from std.log import Log, quietly;

fn chatty() -> int {
    perform Log.info("noise");
    perform Log.warn("more noise");
    7
}

fn main() -> int {
    quietly(fn() -> chatty())
}
""")
    assert prints == []


def test_collect_logs_captures_instead_of_printing():
    result, prints = run_main("""
from std.log import Log, collect_logs;

fn work() -> int {
    perform Log.info("step one");
    perform Log.warn("step two");
    5
}

fn main() -> int {
    let sink = Vec<string>::new();
    let r = collect_logs(sink, fn() -> work());
    print(sink.len());
    print(sink[0]);
    r
}
""")
    assert result == 5
    assert prints == ["2", "[info] step one"]
