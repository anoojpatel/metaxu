"""Standard library (std/*.mx) tests.

Every test goes through the real pipeline: parsed source -> module
resolution (which loads std/ files from the repo's stdlib root) ->
desugar -> freeze -> infer (strict) -> HIR -> MIR -> interpreter.

Also pinned here:
- `std.*` names with no file under std/ (std.simd, std.matrix,
  std.geometry) keep the external-placeholder behavior examples rely on;
- METAXU_STD_PATH overrides the stdlib root;
- the handler sub-function naming fix in lower_hir_to_mir (two functions
  handling the same effect+op used to collide in MIR's flat namespace).
"""
from __future__ import annotations

import os

import pytest

from metaxu.errors import CompileError
from metaxu.compiler.pipeline import build_context_from_source, run_pipeline_ctx
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import MirInterpreter, UNIT


def run_main(source: str, file_path: str = "<mem>"):
    """Full strict pipeline, then execute main(); returns (result, prints)."""
    ctx = build_context_from_source(source, file_path=file_path)
    run_pipeline_ctx(ctx)   # strict: raises on type/borrow errors
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    prints: list[str] = []
    interp.register_builtin(
        "print", lambda *a: (prints.append(" ".join(str(x) for x in a)), UNIT)[1])
    return interp.call("main", []), prints


# ----------------------------------------------------------------------
# std.fail
# ----------------------------------------------------------------------

def test_fail_try_opt_some_and_none():
    result, _ = run_main("""
from std.fail import Fail, try_opt, fail_if;

fn checked_div(a: int, b: int) performs Fail -> int {
    fail_if(b == 0);
    a / b
}

fn main() -> int {
    let good = try_opt(fn() -> int { checked_div(84, 2) });
    let bad = try_opt(fn() -> int { checked_div(1, 0) });
    let a = match good { Some(v) => v, None => 0 - 1 };
    let b = match bad { Some(v) => 0 - 1, None => 100 };
    a + b
}
""")
    assert result == 142


def test_fail_on_fail_default_and_ok_if():
    result, _ = run_main("""
from std.fail import Fail, on_fail, on_fail_else, ok_if, ignore_fail;

fn main() -> int {
    let a = on_fail(fn() -> int { ok_if(false); 1 }, 7);
    let b = on_fail(fn() -> int { ok_if(true); 30 }, 0 - 1);
    let c = on_fail_else(fn() -> int { ok_if(false); 1 }, fn() -> 5);
    ignore_fail(fn() -> int { ok_if(false); 1 });
    a + b + c
}
""")
    assert result == 42


# ----------------------------------------------------------------------
# std.throw
# ----------------------------------------------------------------------

def test_throw_catch_ok_and_err():
    result, _ = run_main("""
from std.throw import Throw, catch_, throw_if;

fn risky(n: int) performs Throw -> int {
    throw_if(n < 0, "negative");
    n * 2
}

fn main() -> int {
    let ok = catch_(fn() -> int { risky(21) });
    let bad = catch_(fn() -> int { risky(0 - 1) });
    let a = match ok { Ok(v) => v, Err(e) => 0 - 1 };
    let b = match bad { Ok(v) => 0 - 1, Err(e) => len(e) };
    a + b
}
""")
    assert result == 42 + len("negative")


def test_throw_catch_or_and_or_else():
    result, _ = run_main("""
from std.throw import Throw, catch_or, catch_or_else;

fn main() -> int {
    let a = catch_or(fn() -> int { perform Throw.throw("boom"); 1 }, 30);
    let b = catch_or(fn() -> int { 10 }, 0 - 1);
    let c = catch_or_else(fn() -> int { perform Throw.throw("xx"); 1 },
                          fn(e) -> len(e));
    a + b + c
}
""")
    assert result == 42


def test_throw_map_err_rethrows_to_outer_handler():
    result, _ = run_main("""
from std.throw import Throw, catch_, map_err;

fn main() -> int {
    let r = catch_(fn() -> int {
        map_err(fn() -> int { perform Throw.throw("bad"); 1 },
                fn(e) -> e + "!!")
    });
    match r { Ok(v) => 0 - 1, Err(e) => len(e) }
}
""")
    assert result == len("bad!!")


def test_throw_unwrap_err_composes_with_fail():
    result, _ = run_main("""
from std.throw import Throw, unwrap_err;
from std.fail import Fail, try_opt;

fn main() -> int {
    # throwing computation: unwrap_err returns the error
    let e = try_opt(fn() { unwrap_err(fn() -> int { perform Throw.throw(9); 0 }) });
    # clean computation: unwrap_err fails, try_opt answers None
    let clean = try_opt(fn() { unwrap_err(fn() -> int { 5 }) });
    let a = match e { Some(v) => v, None => 0 - 1 };
    let b = match clean { Some(v) => 0 - 1, None => 33 };
    a + b
}
""")
    assert result == 42


# ----------------------------------------------------------------------
# std.early_return
# ----------------------------------------------------------------------

def test_early_return_aborts_and_passes_value():
    result, prints = run_main("""
from std.early_return import EarlyReturn, with_early_return, return_if;

fn main() -> int {
    let a = with_early_return(fn() -> int {
        return_if(true, 40);
        print("unreachable");
        0 - 100
    });
    let b = with_early_return(fn() -> int {
        return_if(false, 0);
        2
    });
    a + b
}
""")
    assert result == 42
    assert prints == []


# ----------------------------------------------------------------------
# std.option / std.result
# ----------------------------------------------------------------------

def test_option_combinators():
    result, _ = run_main("""
from std.option import map, and_then, filter, or_else, unwrap_or, unwrap_or_else, is_some, is_none, contains, ok_or, flatten;

fn main() -> int {
    let a = unwrap_or(map(Some(20), fn(x) -> x * 2), 0);
    let b = unwrap_or(map(None, fn(x) -> x * 2), 1);
    let c = unwrap_or(and_then(Some(10), fn(x) -> if x > 5 { Some(x) } else { None }), 0);
    let d = unwrap_or(filter(Some(3), fn(x) -> x > 5), 100);
    let e = unwrap_or(or_else(None, Some(4)), 0);
    let f = unwrap_or_else(None, fn() -> 5);
    let g = if is_some(Some(1)) { 1 } else { 0 };
    let h = if is_none(None) { 1 } else { 0 };
    let i = if contains(Some(9), 9) { 1 } else { 0 };
    let j = match ok_or(None, "gone") { Ok(v) => 0, Err(x) => len(x) };
    let k = unwrap_or(flatten(Some(Some(11))), 0);
    a + b + c + d + e + f + g + h + i + j + k
}
""")
    assert result == 40 + 1 + 10 + 100 + 4 + 5 + 1 + 1 + 1 + 4 + 11


def test_result_combinators():
    result, _ = run_main("""
from std.result import map, map_err, and_then, unwrap_or, unwrap_or_else, is_ok, is_err, ok, err;
from std.option import unwrap_or as opt_unwrap_or;

fn main() -> int {
    let a = unwrap_or(map(Ok(20), fn(x) -> x + 1), 0);
    let b = match map_err(Err("bad"), fn(e) -> e + "!") { Ok(v) => 0, Err(e) => len(e) };
    let c = unwrap_or(and_then(Ok(5), fn(x) -> Ok(x * 2)), 0);
    let d = unwrap_or(Err("nope"), 3);
    let e = unwrap_or_else(Err("xx"), fn(er) -> len(er));
    let f = if is_ok(Ok(1)) { 1 } else { 0 };
    let g = if is_err(Err(1)) { 1 } else { 0 };
    let h = opt_unwrap_or(ok(Ok(7)), 0);
    let i = opt_unwrap_or(err(Err(8)), 0);
    a + b + c + d + e + f + g + h + i
}
""")
    assert result == 21 + 4 + 10 + 3 + 2 + 1 + 1 + 7 + 8


# ----------------------------------------------------------------------
# std.stream
# ----------------------------------------------------------------------

def test_stream_consumers_and_transformers():
    result, _ = run_main("""
from std.stream import Emit, iota, emit_vec, fold, sum, product, count, collect, all_of, any_of, find, map, filter, take, skip, chain;
from std.option import unwrap_or;

fn main() -> int {
    let s1 = sum(iota(5));
    let p = product(map(iota(3), fn(x) -> x + 1));
    let c = count(filter(iota(10), fn(x) -> x > 6));
    let v = collect(take(iota(100), 3));
    let sk = sum(skip(iota(5), 3));
    let ch = count(chain(iota(2), iota(3)));
    let al = if all_of(iota(4), fn(x) -> x < 10) { 1 } else { 0 };
    let an = if any_of(iota(4), fn(x) -> x > 2) { 1 } else { 0 };
    let fd = unwrap_or(find(iota(10), fn(x) -> x > 4), 0 - 1);
    let @mut src = Vec.new();
    src.push(30);
    src.push(12);
    let ev = sum(emit_vec(src));
    s1 + p + c + len(v) + sk + ch + al + an + fd + ev
}
""")
    assert result == 10 + 6 + 3 + 3 + 7 + 5 + 1 + 1 + 5 + 42


def test_stream_iter_applies_in_order():
    result, prints = run_main("""
from std.stream import Emit, iota, iter;

fn main() -> int {
    iter(iota(3), fn(x) { print(x) });
    0
}
""")
    assert result == 0
    assert prints == ["0", "1", "2"]


def test_stream_for_with_break():
    """break_ is an abort-style handler per iteration: the loop stops,
    elements before the break are kept."""
    result, _ = run_main("""
from std.stream import Emit, Loop, iota, for_;

fn main() -> int {
    let @mut acc = Vec.new();
    for_(iota(10), fn(x) {
        if x > 2 { perform Loop.break_() } else { () };
        acc.push(x)
    });
    len(acc) * 100 + acc[0] + acc[1] + acc[2]
}
""")
    assert result == 303


def test_stream_for_with_continue():
    """continue_ aborts only the current iteration's delimited body."""
    result, _ = run_main("""
from std.stream import Emit, Loop, iota, for_;

fn main() -> int {
    let @mut evens = Vec.new();
    for_(iota(6), fn(x) {
        if x - (x / 2) * 2 == 1 { perform Loop.continue_() } else { () };
        evens.push(x)
    });
    len(evens) * 10 + evens[0] + evens[1] + evens[2]
}
""")
    assert result == 30 + 0 + 2 + 4


def test_stream_fold_is_foldr():
    """fold chains through resume(): f(x1, f(x2, ... f(xN, init)))."""
    result, _ = run_main("""
from std.stream import Emit, iota, fold;

fn main() -> int {
    # non-commutative op makes the fold direction observable:
    # foldr over 0,1,2 with f(x, acc) = acc * 10 + x = ((0*10+2)*10+1)*10+0
    fold(iota(3), 0, fn(x, acc) -> acc * 10 + x)
}
""")
    assert result == 210


# ----------------------------------------------------------------------
# std.math
# ----------------------------------------------------------------------

def test_math_helpers():
    result, _ = run_main("""
from std.math import pi, tau, e, abs, min, max, clamp, sign, sqrt, sin, cos, powi;

fn main() -> int {
    let a = abs(0 - 5) + abs(5);
    let fl = abs(0.0 - 2.5);
    let b = min(3, 4) + max(3, 4);
    let c = clamp(100, 0, 10) + clamp(0 - 5, 0, 10) + clamp(7, 0, 10);
    let d = sign(0 - 9) + sign(0) + sign(9);
    let p = powi(2, 10);
    let consts = if pi > 3.14 { if tau > 6.28 { if e > 2.71 { 1 } else { 0 } } else { 0 } } else { 0 };
    let s = if sqrt(4.0) > 1.99 { 1 } else { 0 };
    let fs = if fl > 2.49 { 1 } else { 0 };
    let trig = if sin(0.0) < 0.001 { if cos(0.0) > 0.999 { 1 } else { 0 } } else { 0 };
    a + b + c + d + p + consts + s + fs + trig
}
""")
    assert result == 10 + 7 + 17 + 0 + 1024 + 1 + 1 + 1 + 1


# ----------------------------------------------------------------------
# std.vec
# ----------------------------------------------------------------------

def test_vec_helpers():
    result, _ = run_main("""
from std.vec import of3, range_vec, sum, product, contains, index_of, first, last, is_empty, map, filter, reverse, concat, max_of, min_of;
from std.option import unwrap_or;

fn main() -> int {
    let v = of3(1, 2, 3);
    let a = sum(v) + product(v);
    let b = sum(range_vec(0, 5));
    let c = if contains(v, 2) { 1 } else { 0 };
    let c2 = if contains(v, 9) { 0 } else { 1 };
    let d = unwrap_or(index_of(v, 3), 0 - 1);
    let d2 = unwrap_or(index_of(v, 9), 0 - 1);
    let e = unwrap_or(first(v), 0 - 1) + unwrap_or(last(v), 0 - 1);
    let f = if is_empty(Vec.new()) { 1 } else { 0 };
    let g = sum(map(v, fn(x) -> x * 10));
    let h = sum(filter(range_vec(0, 5), fn(x) -> x > 2));
    let i = unwrap_or(first(reverse(v)), 0 - 1);
    let j = len(concat(v, range_vec(0, 5)));
    let k = unwrap_or(max_of(range_vec(0, 5)), 0 - 1) + unwrap_or(min_of(v), 0 - 1);
    a + b + c + c2 + d + d2 + e + f + g + h + i + j + k
}
""")
    assert result == 12 + 10 + 1 + 1 + 2 + (-1) + 4 + 1 + 60 + 7 + 3 + 8 + 5


# ----------------------------------------------------------------------
# std.string
# ----------------------------------------------------------------------

def test_string_helpers():
    result, _ = run_main("""
from std.string import is_empty, eq, concat, repeat, join, char_at, contains_char, count_char, index_of_char, starts_with, ends_with, reverse;
from std.option import unwrap_or;

fn main() -> int {
    let a = if is_empty("") { 1 } else { 0 };
    let b = if eq(concat("ab", "cd"), "abcd") { 1 } else { 0 };
    let c = len(repeat("xy", 3));
    let @mut parts = Vec.new();
    parts.push("a");
    parts.push("b");
    parts.push("c");
    let d = len(join(parts, "--"));
    let e = if char_at("hello", 1) == "e" { 1 } else { 0 };
    let f = if contains_char("hello", "l") { 1 } else { 0 };
    let g = count_char("banana", "a");
    let h = unwrap_or(index_of_char("hello", "l"), 0 - 1);
    let i = if starts_with("hello", "he") { 1 } else { 0 };
    let j = if starts_with("he", "hello") { 0 } else { 1 };
    let k = if ends_with("hello", "llo") { 1 } else { 0 };
    let m = if eq(reverse("abc"), "cba") { 1 } else { 0 };
    a + b + c + d + e + f + g + h + i + j + k + m
}
""")
    assert result == 1 + 1 + 6 + 7 + 1 + 1 + 3 + 2 + 1 + 1 + 1 + 1


# ----------------------------------------------------------------------
# std.map
# ----------------------------------------------------------------------

def test_map_put_get_remove():
    result, _ = run_main("""
from std.map import Map, empty, size, is_empty, contains_key, get, get_or, put, remove, keys, values;
from std.option import unwrap_or;

fn main() -> int {
    let m = empty();
    let a = if is_empty(m) { 1 } else { 0 };
    put(m, "one", 1);
    put(m, "two", 2);
    put(m, "three", 3);
    let b = size(m);
    put(m, "two", 22);                       # overwrite keeps size stable
    let c = size(m);
    let d = unwrap_or(get(m, "two"), 0 - 1);
    let e = unwrap_or(get(m, "missing"), 0 - 1);
    let f = get_or(m, "one", 0 - 1);
    let g = if contains_key(m, "three") { 1 } else { 0 };
    let h = unwrap_or(remove(m, "one"), 0 - 1);
    let h2 = unwrap_or(remove(m, "one"), 0 - 100);
    let i = size(m);
    let j = len(keys(m)) + len(values(m));
    a + b + c + d + e + f + g + h + h2 + i + j
}
""")
    assert result == 1 + 3 + 3 + 22 + (-1) + 1 + 1 + 1 + (-100) + 2 + 4


def test_map_int_keys():
    result, _ = run_main("""
from std.map import Map, empty, put, get;
from std.option import unwrap_or;

fn main() -> int {
    let m = empty();
    put(m, 10, 100);
    put(m, 20, 200);
    unwrap_or(get(m, 10), 0 - 1) + unwrap_or(get(m, 99), 0 - 1)
}
""")
    assert result == 99


# ----------------------------------------------------------------------
# std.prelude (re-export chains)
# ----------------------------------------------------------------------

def test_prelude_reexports_work():
    result, _ = run_main("""
from std.prelude import Fail, try_opt, fail_if, catch_, throw_if, with_early_return, return_if, iota, fold, unwrap_or, is_some, abs, clamp, pi;
from std.throw import Throw;
from std.early_return import EarlyReturn;
from std.stream import Emit;

fn main() -> int {
    let a = unwrap_or(try_opt(fn() -> int { fail_if(false); 10 }), 0);
    let b = match catch_(fn() -> int { throw_if(true, "e"); 0 }) { Ok(v) => 0, Err(x) => len(x) };
    let c = with_early_return(fn() -> int { return_if(true, 5); 0 });
    let d = fold(iota(4), 0, fn(x, acc) -> x + acc);
    let f = if is_some(Some(1)) { 1 } else { 0 };
    let g = abs(0 - 2) + clamp(50, 0, 9);
    let h = if pi > 3.0 { 1 } else { 0 };
    a + b + c + d + f + g + h
}
""")
    assert result == 10 + 1 + 5 + 6 + 1 + 11 + 1


def test_prelude_qualified_reexport_reference():
    """A re-exported symbol is reachable through the re-exporting module's
    qualified name (prelude.try_opt resolves to std.fail.try_opt)."""
    result, _ = run_main("""
import std.prelude;
from std.fail import Fail;

fn main() -> int {
    let r = std.prelude.try_opt(fn() -> int { perform Fail.fail(); 1 });
    match r { Some(v) => v, None => 42 }
}
""")
    assert result == 42


# ----------------------------------------------------------------------
# Loader behavior: placeholders, private helpers, env override
# ----------------------------------------------------------------------

def test_unresolved_std_names_stay_placeholders():
    """std.* names with no file under std/ keep compiling as external
    placeholders (examples 03/06 import std.simd/std.matrix/std.geometry)."""
    result, _ = run_main("""
import std.matrix as mat;
import std.simd;
from std.does_not_exist import whatever;

fn main() -> int { 3 }
""")
    assert result == 3


def test_example_03_still_compiles_and_runs():
    """Example 03 imports std.matrix + std.geometry (placeholders) and must
    keep its documented behavior with the real stdlib present."""
    repo_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..", ".."))
    src = open(os.path.join(repo_root,
                            "examples/03_modules_and_imports.mx")).read()
    result, prints = run_main(src)
    assert any("11" in line for line in prints)


def test_std_math_import_resolves_to_real_module():
    """`from std.math import sqrt` now binds the real stdlib function."""
    result, _ = run_main("""
from std.math import sqrt;

fn main() -> int {
    if sqrt(9.0) > 2.99 { 1 } else { 0 }
}
""")
    assert result == 1


def test_private_std_helper_not_importable():
    with pytest.raises(CompileError) as exc:
        build_context_from_source("""
from std.vec import is_none_priv;

fn main() -> int { 0 }
""")
    assert "is_none_priv" in str(exc.value)


def test_metaxu_std_path_env_override(tmp_path, monkeypatch):
    """METAXU_STD_PATH points module resolution at an alternate stdlib."""
    (tmp_path / "custom.mx").write_text("""
fn answer() -> int { 41 }
""")
    monkeypatch.setenv("METAXU_STD_PATH", str(tmp_path))
    result, _ = run_main("""
from std.custom import answer;

fn main() -> int { answer() + 1 }
""")
    assert result == 42


def test_metaxu_std_path_missing_dir_falls_back_to_placeholder(monkeypatch):
    monkeypatch.setenv("METAXU_STD_PATH", "/nonexistent/std/dir")
    result, _ = run_main("""
import std.fail;

fn main() -> int { 5 }
""")
    assert result == 5


# ----------------------------------------------------------------------
# Regression: handler sub-function naming is per-function
# ----------------------------------------------------------------------

def test_two_functions_handling_same_effect_do_not_collide():
    """Two functions installing different handlers for the same effect+op
    lower to distinct MIR sub-functions. Before the fix both lowered to
    __handler_Ask_ask_hs1 (flat namespace): whichever loaded last silently
    won and the other function ran the wrong handler."""
    result, _ = run_main("""
effect Ask {
    ask() -> int
}

fn with_seven(f: fn() -> int) -> int {
    handle Ask with {
        ask() -> resume(7)
    } in {
        f()
    }
}

fn with_thousand(f: fn() -> int) -> int {
    handle Ask with {
        ask() -> resume(1000)
    } in {
        f()
    }
}

fn main() -> int {
    with_seven(fn() -> int { perform Ask.ask() })
        + with_thousand(fn() -> int { perform Ask.ask() })
}
""")
    assert result == 1007
