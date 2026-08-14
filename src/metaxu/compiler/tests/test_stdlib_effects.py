"""Standard library round 2: the effect-shaped modules (std/README.md).

`std.state`, `std.log`, `std.random`, `std.parse`, `std.test` and
`std.iter` are the modules that are *about* Metaxu's effect system rather
than ported from Ante, so every test here exercises the handler behaviour
that makes them worth shipping: state actually threading through resume,
a collector handler answering the lines a computation logged, a seeded
generator being reproducible across two runs, and stream adapters that
stop pulling their source.

Every test goes through the real pipeline (parsed source -> module
resolution, which loads std/*.mx from the repo's stdlib root -> desugar
-> freeze -> STRICT infer/borrow-check -> HIR -> MIR -> interpreter).
`std.state`/`std.log`/`std.test`'s first cut is covered by
test_std_state.py / test_std_log.py / test_std_test.py; this file covers
the round-2 surface and the modules added with it.
"""
from __future__ import annotations

import pytest

from metaxu.compiler.pipeline import build_context_from_source, run_pipeline_ctx
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import InterpError, MirInterpreter, UNIT


def compile_source(source: str, file_path: str = "<mem>") -> MirInterpreter:
    ctx = build_context_from_source(source, file_path=file_path)
    run_pipeline_ctx(ctx)   # strict: raises on type/borrow errors
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    return interp


def run_main(source: str, file_path: str = "<mem>"):
    """Full strict pipeline, then execute main(); returns (result, prints)."""
    interp = compile_source(source, file_path=file_path)
    prints: list[str] = []
    interp.register_builtin(
        "print", lambda *a: (prints.append(" ".join(str(x) for x in a)), UNIT)[1])
    return interp.call("main", []), prints


# ======================================================================
# std.state
# ======================================================================

def test_eval_state_answers_the_computation_result():
    result, _ = run_main("""
from std.state import State, eval_state;

fn body() -> int {
    perform State.put(perform State.get() * 2);
    perform State.get() + 2
}

fn main() -> int {
    eval_state(20, fn() -> body())
}
""")
    assert result == 42


def test_exec_state_answers_the_final_state():
    """The runner reads its own capture cell after the handled block ends —
    that is what makes execState expressible without a second effect."""
    result, _ = run_main("""
from std.state import State, exec_state;

fn body() -> string {
    perform State.put(perform State.get() + 1);
    perform State.put(perform State.get() * 10);
    "ignored"
}

fn main() -> int {
    exec_state(4, fn() -> body())
}
""")
    assert result == 50


def test_run_state_answers_both_halves():
    result, prints = run_main("""
from std.state import State, StateResult, run_state;

fn body() -> string {
    perform State.put(perform State.get() + 5);
    "done"
}

fn main() -> int {
    let r = run_state(37, fn() -> body());
    print("value", r.value);
    r.state
}
""")
    assert result == 42
    assert prints == ["value done"]


def test_state_threads_through_called_functions():
    """Deep handler: get/put performed two calls below the handler still
    reach it, and each sees the previous one's write."""
    result, _ = run_main("""
from std.state import State, eval_state, modify, gets, update, increment;

fn inner() -> () {
    modify(fn(s: int) -> s * 3)
}

fn outer() -> int {
    inner();
    increment(4);
    let doubled = update(fn(s: int) -> s + 1);
    gets(fn(s: int) -> s + doubled)
}

fn main() -> int {
    eval_state(2, fn() -> outer())
}
""")
    # 2 -> *3 = 6 -> +4 = 10 -> update -> 11 (doubled = 11) -> 11 + 11
    assert result == 22


def test_state_without_a_handler_is_loud():
    interp = compile_source("""
from std.state import State;

fn main() -> int {
    perform State.get()
}
""")
    with pytest.raises(InterpError, match="No handler for effect 'State'"):
        interp.call("main", [])


# ======================================================================
# std.log
# ======================================================================

def test_with_collected_logs_answers_the_lines():
    result, prints = run_main("""
from std.log import Log, log_debug, log_info, log_warn, log_error, with_collected_logs;

fn body() -> int {
    log_debug("d");
    log_info("i");
    log_warn("w");
    log_error("e");
    0
}

fn main() -> int {
    let lines = with_collected_logs(fn() -> body());
    print(lines[0]);
    print(lines[1]);
    print(lines[2]);
    print(lines[3]);
    len(lines)
}
""")
    assert result == 4
    # nothing printed by the logging itself -- the handler took the lines
    assert prints == ["[debug] d", "[info] i", "[warn] w", "[error] e"]


def test_run_collected_answers_result_and_lines():
    result, prints = run_main("""
from std.log import Log, log_info, run_collected;

fn body() -> int {
    log_info("hello");
    7
}

fn main() -> int {
    let r = run_collected(fn() -> body());
    print("value", r.value, "lines", len(r.lines), r.lines[0]);
    r.value
}
""")
    assert result == 7
    assert prints == ["value 7 lines 1 [info] hello"]


def test_unhandled_log_uses_the_stdout_defaults():
    _, prints = run_main("""
from std.log import Log, log_info, log_warn;

fn main() -> int {
    log_info("up");
    log_warn("careful");
    0
}
""")
    assert prints == ["[info] up", "[warn] careful"]


def test_quietly_suppresses_everything():
    result, prints = run_main("""
from std.log import Log, log_info, log_error, quietly;

fn body() -> int {
    log_info("noise");
    log_error("more noise");
    5
}

fn main() -> int {
    quietly(fn() -> body())
}
""")
    assert result == 5
    assert prints == []


def test_with_min_level_reperforms_to_the_enclosing_handler():
    """A handler arm evaluates OUTSIDE its own delimitation, so the arm's
    `perform Log.warn(...)` routes to the next enclosing Log handler —
    which is what makes a filter compose with a collector."""
    result, prints = run_main("""
from std.log import Log, log_debug, log_info, log_warn, log_error,
                    with_collected_logs, with_min_level, level_warn;

fn body() -> int {
    log_debug("d");
    log_info("i");
    log_warn("w");
    log_error("e");
    0
}

fn filtered() -> int {
    with_min_level(level_warn, fn() -> body())
}

fn main() -> int {
    let lines = with_collected_logs(fn() -> filtered());
    print(lines[0]);
    print(lines[1]);
    len(lines)
}
""")
    assert result == 2
    assert prints == ["[warn] w", "[error] e"]


def test_innermost_log_handler_wins():
    result, prints = run_main("""
from std.log import Log, log_info, with_collected_logs, quietly;

fn inner() -> int {
    log_info("swallowed");
    0
}

fn body() -> int {
    log_info("kept");
    quietly(fn() -> inner());
    0
}

fn main() -> int {
    let lines = with_collected_logs(fn() -> body());
    print(lines[0]);
    len(lines)
}
""")
    assert result == 1
    assert prints == ["[info] kept"]


# ======================================================================
# std.random
# ======================================================================

_DRAWS = """
from std.random import Random, with_seed, with_sequence, next_below, next_range,
                       next_bool, choose, take_random, shuffle;
from std.option import unwrap_or;

fn show(v: Vec) -> string {
    let @mut s = "";
    let @mut i = 0;
    while i < len(v) {
        s = s + to_string(v[i]) + ",";
        i = i + 1
    }
    s
}
"""


def test_with_seed_is_reproducible_and_seed_dependent():
    result, prints = run_main(_DRAWS + """
fn draws() -> Vec {
    take_random(6, 100)
}

fn main() -> int {
    let a = show(with_seed(42, fn() -> draws()));
    let b = show(with_seed(42, fn() -> draws()));
    let c = show(with_seed(43, fn() -> draws()));
    print("same", a == b);
    print("differs", a == c);
    print(a);
    0
}
""")
    assert prints[0] == "same True"
    assert prints[1] == "differs False"
    # six draws, all in range, and not all equal (a constant "generator"
    # would pass the reproducibility test above)
    draws = [int(x) for x in prints[2].split(",") if x]
    assert len(draws) == 6
    assert all(0 <= d < 100 for d in draws)
    assert len(set(draws)) > 1


def test_next_range_and_next_below_stay_in_bounds():
    result, _ = run_main(_DRAWS + """
fn body() -> int {
    let @mut ok = 0;
    let @mut i = 0;
    while i < 50 {
        let a = next_below(7);
        let b = next_range(10, 20);
        if a >= 0 && a < 7 && b >= 10 && b < 20 { ok = ok + 1 } else { () };
        i = i + 1
    }
    ok
}

fn main() -> int {
    with_seed(1234, fn() -> body())
}
""")
    assert result == 50


def test_next_below_zero_is_zero_not_a_division_error():
    result, _ = run_main(_DRAWS + """
fn main() -> int {
    with_seed(1, fn() -> next_below(0))
}
""")
    assert result == 0


def test_with_sequence_replays_scripted_draws_and_cycles():
    result, prints = run_main(_DRAWS + """
from std.vec import of3;

fn body() -> int {
    let a = perform Random.next();
    let b = perform Random.next();
    let c = perform Random.next();
    let d = perform Random.next();
    print("draws", a, b, c, d);
    0
}

fn main() -> int {
    with_sequence(of3(3, 1, 4), fn() -> body())
}
""")
    assert prints == ["draws 3 1 4 3"]


def test_shuffle_is_a_permutation_and_leaves_the_source_alone():
    result, prints = run_main(_DRAWS + """
from std.vec import range_vec, sum;

fn body(v: Vec) -> Vec {
    shuffle(v)
}

fn main() -> int {
    let v = range_vec(0, 8);
    let s = with_seed(99, fn() -> body(v));
    print("sizes", len(v), len(s));
    print("sums", sum(v), sum(s));
    print("source", show(v));
    print("shuffled", show(s));
    0
}
""")
    assert prints[0] == "sizes 8 8"
    assert prints[1] == "sums 28 28"
    assert prints[2] == "source 0,1,2,3,4,5,6,7,"      # untouched
    assert prints[3] != prints[2].replace("source", "shuffled")
    assert sorted(prints[3].split()[1].split(",")[:-1]) == \
        sorted("0 1 2 3 4 5 6 7".split())


def test_choose_answers_none_for_an_empty_vec():
    result, _ = run_main(_DRAWS + """
fn body() -> int {
    let empty = Vec.new();
    let a = unwrap_or(choose(empty), 0 - 1);
    let full = Vec.new();
    full.push(9);
    let b = unwrap_or(choose(full), 0 - 1);
    a + b
}

fn main() -> int {
    with_seed(5, fn() -> body())
}
""")
    assert result == 8    # -1 (None) + 9


def test_xorshift_draws_are_spread_across_buckets():
    """Distribution sanity for the xorshift64 generator (it replaced an LCG
    once the language grew `^`/`<<`/`>>`).

    160 draws below 4 must land in all four buckets, none of them wildly
    over-represented. A stuck or short-period generator fails this; so
    would an LCG whose LOW bits were used (the reason the old one had to
    concatenate two high-bit slices). The draw count is kept modest
    because every draw is a delimited `perform`."""
    result, prints = run_main(_DRAWS + """
fn body() -> Vec {
    let @mut counts = Vec.new();
    let @mut k = 0;
    while k < 4 {
        counts.push(0);
        k = k + 1
    }
    let @mut i = 0;
    while i < 160 {
        let d = next_below(4);
        counts[d] = counts[d] + 1;
        i = i + 1
    }
    counts
}

fn main() -> int {
    print(show(with_seed(7, fn() -> body())));
    0
}
""")
    counts = [int(x) for x in prints[0].split(",") if x]
    assert len(counts) == 4
    assert sum(counts) == 160
    # Expected 40 per bucket; a very loose band that still excludes a
    # degenerate generator.
    assert all(15 <= c <= 75 for c in counts), counts


def test_neighbouring_seeds_produce_different_first_draws():
    """Small, adjacent seeds must not produce correlated streams — the
    weakness `stir()` (three discarded steps) exists to remove."""
    result, prints = run_main(_DRAWS + """
fn first_three() -> Vec {
    take_random(3, 1000)
}

fn main() -> int {
    print(show(with_seed(0, fn() -> first_three())));
    print(show(with_seed(1, fn() -> first_three())));
    print(show(with_seed(2, fn() -> first_three())));
    0
}
""")
    streams = [p for p in prints]
    assert len(set(streams)) == 3, streams


def test_random_without_a_handler_is_loud():
    """No entropy shim exists, so Random declares NO default: performing it
    unhandled must fail loudly rather than answer a constant."""
    interp = compile_source("""
from std.random import Random, next_below;

fn main() -> int {
    next_below(10)
}
""")
    with pytest.raises(InterpError, match="No handler for effect 'Random'"):
        interp.call("main", [])


# ======================================================================
# std.parse
# ======================================================================

def test_parse_int_accepts_signs_and_rejects_junk():
    result, prints = run_main("""
from std.parse import parse_int, parse_int_or;
from std.option import unwrap_or, is_none;

fn main() -> int {
    print("good", unwrap_or(parse_int("123"), 0 - 1));
    print("neg", unwrap_or(parse_int("-45"), 0 - 1));
    print("plus", unwrap_or(parse_int("+9"), 0 - 1));
    print("zero", unwrap_or(parse_int("0"), 0 - 1));
    print("bad",
          is_none(parse_int("12a")),
          is_none(parse_int("")),
          is_none(parse_int("-")),
          is_none(parse_int(" 1")),
          is_none(parse_int("1 ")));
    parse_int_or("nope", 77)
}
""")
    assert result == 77
    assert prints == [
        "good 123", "neg -45", "plus 9", "zero 0",
        "bad True True True True True",
    ]


def test_digit_and_space_helpers():
    result, prints = run_main(r"""
from std.parse import digit_value, is_digit, is_space;

fn main() -> int {
    print("digits", digit_value("0"), digit_value("7"), digit_value("9"));
    print("nondigit", digit_value("x"), is_digit("3"), is_digit("-"));
    print("space", is_space(" "), is_space("\t"), is_space("\n"), is_space("a"));
    digit_value("4")
}
""")
    assert result == 4
    assert prints == [
        "digits 0 7 9",
        "nondigit -1 True False",
        "space True True True False",
    ]


def test_parse_int_or_fail_composes_with_std_fail():
    result, prints = run_main("""
from std.parse import parse_int_or_fail;
from std.fail import Fail, try_opt, on_fail;
from std.option import unwrap_or, is_none;

fn main() -> int {
    print("ok", unwrap_or(try_opt(fn() -> parse_int_or_fail("55")), 0 - 1));
    print("bad", is_none(try_opt(fn() -> parse_int_or_fail("oops"))));
    on_fail(fn() -> parse_int_or_fail("x"), 13)
}
""")
    assert result == 13
    assert prints == ["ok 55", "bad True"]


def test_trim_and_split_on():
    result, prints = run_main(r"""
from std.parse import trim, split_on;

fn show(v: Vec) -> string {
    let @mut s = "";
    let @mut i = 0;
    while i < len(v) {
        s = s + "<" + v[i] + ">";
        i = i + 1
    }
    s
}

fn main() -> int {
    print("trim", "[" + trim("  hi there \n") + "]");
    print("trim-empty", "[" + trim("   ") + "]");
    print("split", show(split_on("a,b,,c", ",")));
    print("split-none", show(split_on("abc", ",")));
    len(split_on("a,b,,c", ","))
}
""")
    assert result == 4
    assert prints == [
        "trim [hi there]",
        "trim-empty []",
        "split <a><b><><c>",
        "split-none <abc>",
    ]


def test_parse_int_vec_aborts_at_the_first_bad_field():
    result, prints = run_main("""
from std.parse import parse_int_vec;
from std.fail import Fail, on_fail;
from std.vec import sum;

fn main() -> int {
    let good = on_fail(fn() -> parse_int_vec("1, 2, 3", ","), Vec.new());
    print("good", len(good), sum(good));
    let bad = on_fail(fn() -> parse_int_vec("1, x, 3", ","), Vec.new());
    print("bad", len(bad));
    sum(good)
}
""")
    assert result == 6
    assert prints == ["good 3 6", "bad 0"]


def test_parse_bool():
    result, _ = run_main("""
from std.parse import parse_bool;
from std.option import unwrap_or, is_none;

fn main() -> int {
    let a = if unwrap_or(parse_bool("true"), false) { 1 } else { 0 };
    let b = if unwrap_or(parse_bool("false"), true) { 0 } else { 10 };
    let c = if is_none(parse_bool("maybe")) { 100 } else { 0 };
    a + b + c
}
""")
    assert result == 111


# ======================================================================
# std.test
# ======================================================================

def test_assertions_tally_instead_of_aborting():
    """The interpreter's `assert_eq` builtin aborts on the first mismatch;
    std.test's shadows it (user functions win in plain-call position) and
    reports through the Report effect, so the suite runs to the end."""
    result, prints = run_main("""
from std.test import Report, TestReport, assert_true, assert_false, assert_eq,
                     assert_ne, run_tests;

fn suite() -> () {
    assert_true(1 + 1 == 2, "a");
    assert_false(1 == 2, "b");
    assert_eq(6 * 7, 41, "c");
    assert_eq("x" + "y", "xy", "d");
    assert_ne(1, 1, "e");
}

fn main() -> int {
    let r = run_tests("demo", fn() -> suite());
    print("total", r.total, "failed", r.failed);
    print(r.failures[0]);
    print(r.failures[1]);
    r.failed
}
""")
    assert result == 2
    assert prints == [
        "total 5 failed 2",
        "c: expected 41, got 42",
        "e: expected anything but 1",
    ]


def test_collect_failures_is_empty_for_a_green_suite():
    result, _ = run_main("""
from std.test import Report, assert_true, assert_eq, collect_failures;

fn suite() -> () {
    assert_true(true, "a");
    assert_eq(2, 2, "b");
}

fn main() -> int {
    len(collect_failures(fn() -> suite()))
}
""")
    assert result == 0


def test_assert_eq_shadows_the_builtin_rather_than_aborting():
    """The builtin would raise AssertionError here; the module's version
    must merely report, so `main` returns normally."""
    result, prints = run_main("""
from std.test import Report, assert_eq;

fn main() -> int {
    assert_eq(1, 2, "mismatch");
    99
}
""")
    assert result == 99
    assert prints == ["FAIL mismatch: expected 2, got 1"]


# ======================================================================
# std.iter
# ======================================================================

_ITER_HELPERS = """
from std.stream import Emit, iota, emit_vec, collect, sum, count;
from std.iter import Pair, pair, enumerate, zip, zip_with, take_while,
                     drop_while, step_by, windows, chunks;
from std.vec import of3;

fn show_ints(v: Vec) -> string {
    let @mut s = "";
    let @mut i = 0;
    while i < len(v) {
        s = s + to_string(v[i]) + ",";
        i = i + 1
    }
    s
}

fn show_pairs(v: Vec) -> string {
    let @mut s = "";
    let @mut i = 0;
    while i < len(v) {
        s = s + "(" + to_string(v[i].first) + " " + to_string(v[i].second) + ")";
        i = i + 1
    }
    s
}

fn show_vecs(v: Vec) -> string {
    let @mut s = "";
    let @mut i = 0;
    while i < len(v) {
        s = s + "[" + show_ints(v[i]) + "]";
        i = i + 1
    }
    s
}
"""


def test_enumerate_pairs_index_with_element():
    _, prints = run_main(_ITER_HELPERS + """
fn main() -> int {
    print(show_pairs(collect(enumerate(emit_vec(of3(10, 20, 30))))));
    0
}
""")
    assert prints == ["(0 10)(1 20)(2 30)"]


def test_zip_stops_at_the_shorter_stream():
    _, prints = run_main(_ITER_HELPERS + """
fn main() -> int {
    print(show_pairs(collect(zip(iota(5), emit_vec(of3(7, 8, 9))))));
    print(show_pairs(collect(zip(emit_vec(of3(7, 8, 9)), iota(2)))));
    0
}
""")
    assert prints == ["(0 7)(1 8)(2 9)", "(7 0)(8 1)"]


def test_zip_with_applies_the_combiner():
    _, prints = run_main(_ITER_HELPERS + """
fn main() -> int {
    print(show_ints(collect(zip_with(iota(5), emit_vec(of3(7, 8, 9)),
                                     fn(a, b) -> a * b))));
    0
}
""")
    assert prints == ["0,8,18,"]


def test_take_while_stops_pulling_the_source():
    """Not a filter: the adapter declines to resume, so the source stops —
    which is why this terminates against a 1000-element producer."""
    result, prints = run_main(_ITER_HELPERS + """
fn main() -> int {
    print(show_ints(collect(take_while(iota(10), fn(x: int) -> x < 4))));
    sum(take_while(iota(1000), fn(x: int) -> x < 5))
}
""")
    assert result == 0 + 1 + 2 + 3 + 4
    assert prints == ["0,1,2,3,"]


def test_drop_while_drops_only_the_leading_run():
    _, prints = run_main(_ITER_HELPERS + """
fn main() -> int {
    print(show_ints(collect(drop_while(iota(6), fn(x: int) -> x < 3))));
    print(show_ints(collect(drop_while(emit_vec(of3(1, 9, 1)),
                                       fn(x: int) -> x < 5))));
    0
}
""")
    assert prints == ["3,4,5,", "9,1,"]


def test_step_by_and_windows_and_chunks():
    _, prints = run_main(_ITER_HELPERS + """
fn main() -> int {
    print(show_ints(collect(step_by(iota(10), 3))));
    print(show_vecs(collect(windows(iota(5), 3))));
    print(show_vecs(collect(chunks(iota(7), 3))));
    print(show_vecs(collect(windows(iota(2), 3))));
    0
}
""")
    assert prints == [
        "0,3,6,9,",
        "[0,1,2,][1,2,3,][2,3,4,]",
        "[0,1,2,][3,4,5,][6,]",
        "",                       # stream shorter than the window
    ]


def test_prelude_reexports_the_round2_names():
    result, prints = run_main("""
from std.prelude import State, eval_state, exec_state, modify,
                        Log, log_info, with_collected_logs,
                        parse_int_or, trim;

fn body() -> int {
    modify(fn(s: int) -> s + 1);
    log_info("bumped");
    perform State.get()
}

fn main() -> int {
    let lines = with_collected_logs(fn() -> eval_state(41, fn() -> body()));
    print("lines", len(lines), lines[0]);
    print("parse", parse_int_or("12", 0), "[" + trim(" x ") + "]");
    exec_state(41, fn() -> body())
}
""")
    assert result == 42
    assert prints[0] == "lines 1 [info] bumped"
    assert prints[1] == "parse 12 [x]"


def test_adapters_chain_with_std_stream_consumers():
    result, _ = run_main(_ITER_HELPERS + """
from std.stream import map, filter;

fn main() -> int {
    sum(map(take_while(drop_while(iota(100), fn(x: int) -> x < 5),
                       fn(x: int) -> x < 10),
            fn(x: int) -> x * 2))
}
""")
    assert result == 2 * (5 + 6 + 7 + 8 + 9)
