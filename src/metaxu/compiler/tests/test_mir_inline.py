"""MIR inlining (compiler/mir_inline.py): flattening helper calls into a
kernel must not change what the interpreter computes.  Every case runs a
program twice on the reference engine, once as lowered and once with one
function's calls inlined, and compares the output; the refusals (recursion,
write-back parameters) are pinned by message.
"""
from __future__ import annotations

import pytest

from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_inline import InlineError, inline_calls, rename_locals
from metaxu.compiler.mir_interp import UNIT, MirInterpreter
from metaxu.compiler.pipeline import build_context_from_source, run_pipeline_ctx


def _funcs(src: str):
    ctx = build_context_from_source(src)
    run_pipeline_ctx(ctx)
    return lower_hir_to_mir(HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root))


def _run(funcs) -> list:
    it = MirInterpreter()
    it.load(funcs)
    out: list = []
    it.register_builtin("print", lambda *a: (out.append(" ".join(str(x) for x in a)), UNIT)[1])
    it.call("main", [])
    return out


def _check_same(src: str, target: str) -> None:
    funcs = _funcs(src)
    by = {f.name: f for f in funcs}
    flat = inline_calls(by[target], by)
    # no call to a module function survives
    for b in flat.blocks:
        for op in b.ops:
            if op[0] == "let" and op[2][0] == "call":
                assert op[2][1] not in by, op
    before = _run(funcs)
    after = _run([flat if f.name == target else f for f in funcs])
    assert before == after


def test_tile_helpers_with_lambdas_and_captures():
    _check_same("""
fn scale_by(t, k: int) -> Tile { Tile.map(t, fn(x: int) -> x * k) }
fn row_total(t) -> Tile { Tile.reduce_rows(t, 0, fn(a: int, b: int) -> a + b) }
fn big(n: int) -> int { if n > 2 { n * 100 } else { n } }
fn pipeline(pid: int, v: Vec, out: Vec) -> () {
    let t = Tile.load_or(v, pid * 4, 2, 2, 0);
    let r = row_total(scale_by(t, big(pid + 1)));
    Tile.store_clipped(out, pid * 2, r);
    ()
}
fn main() -> int {
    let @mut v = Vec.new(); let @mut out = Vec.new();
    let @mut i = 0;
    while i < 12 { v.push(i + 1); i = i + 1 };
    i = 0;
    while i < 6 { out.push(0); i = i + 1 };
    let @mut p = 0;
    while p < 3 { pipeline(p, v, out); p = p + 1 };
    print(out);
    0
}
""", "pipeline")


def test_nested_helpers_branches_and_loops():
    _check_same("""
fn sq(x: int) -> int { x * x }
fn sum_to(n: int) -> int { let @mut s = 0; let @mut i = 0; while i < n { s = s + sq(i); i = i + 1 }; s }
fn pick(a: int, b: int) -> int { if a > b { sum_to(a) } else { sum_to(b) + 1 } }
fn work(n: int) -> int { pick(n, 3) + pick(2, n) }
fn main() -> int { print(work(1)); print(work(5)); 0 }
""", "work")


def test_unit_returning_helper_and_name_collisions():
    # The helper's temps collide with the caller's (both lower `c1`, `b2`...)
    _check_same("""
fn bump(v: Vec, i: int) -> () { v[i] = v[i] + 1; () }
fn twice(v: Vec, i: int) -> () { bump(v, i); bump(v, i); () }
fn main() -> int {
    let @mut v = Vec.new(); v.push(10); v.push(20);
    twice(v, 1); twice(v, 0);
    print(v);
    0
}
""", "twice")


def test_recursion_is_refused():
    funcs = _funcs("""
fn down(n: int) -> int { if n == 0 { 0 } else { 1 + down(n - 1) } }
fn main() -> int { print(down(3)); 0 }
""")
    by = {f.name: f for f in funcs}
    with pytest.raises(InlineError) as ei:
        inline_calls(by["down"], by)
    assert "recursive call to 'down'" in str(ei.value)


def test_write_back_parameters_are_refused():
    funcs = _funcs("""
struct P { x: int }
fn poke(p: @mut P) -> () { p.x = p.x + 1; () }
fn use_it() -> int { let @mut p = P { x: 1 }; poke(p); p.x }
fn main() -> int { print(use_it()); 0 }
""")
    by = {f.name: f for f in funcs}
    with pytest.raises(InlineError) as ei:
        inline_calls(by["use_it"], by)
    assert "write-back parameters" in str(ei.value)


def test_rename_locals_keeps_callee_and_capture_names():
    funcs = _funcs("""
fn scale_by(t, k: int) -> Tile { Tile.map(t, fn(x: int) -> x * k) }
fn main() -> int { print(scale_by(Tile.arange(1, 3), 3)); 0 }
""")
    by = {f.name: f for f in funcs}
    renamed = rename_locals(by["scale_by"], lambda n: n + "_R")
    ops = [op for b in renamed.blocks for op in b.ops]
    closure = [op for op in ops if op[0] == "let" and op[2][0] == "make_closure"][0]
    assert closure[2][1] == "scale_by$lambda1"           # callee name untouched
    assert closure[3] == (("k", "k_R"),)                  # lambda side kept, value renamed
    assert renamed.param_names() == ("t_R", "k_R")
    out = _run([renamed if f.name == "scale_by" else f for f in funcs])
    assert out == ["tile[1x3](0, 3, 6)"]
