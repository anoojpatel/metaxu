"""Regression tests for the silent-wrong-behavior seams the stdlib work
exposed (std/README.md "Language gaps" items 3, 4, 5, 6 and 7).

The repo convention is work-or-loud: every construct here either does what
the source says all the way through parse -> infer -> HIR -> MIR ->
interpreter, or fails with a clear diagnostic. Each fix is tested in both
directions — the working path AND the loud path for receivers/targets that
cannot support it.

Covered:
- `v[i] = x` index assignment stores through to the shared Vec; a fixed
  vector[T,N] in an assignable place gets a value-semantics update
  written back; immutable receivers with no place error loudly;
- module-level `let` bindings are real constants, initialized before the
  entry point, visible in-module and across module boundaries;
- a handler arm with a unit-literal body (`op() -> ()`) participates as
  an abort-style arm instead of being silently dropped; non-empty tuple
  literals are a loud error;
- assignment to a captured scalar inside a closure / handler arm writes
  back to the enclosing binding (shared-cell captures);
- `%` is a real modulo operator through the whole pipeline;
- zip comprehensions (`f(a, b) for (a, b) in (xs, ys)`) iterate in
  lockstep instead of silently dropping the vector literal.
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


# ----------------------------------------------------------------------
# Gap 7: `v[i] = x` was a silent no-op
# ----------------------------------------------------------------------

def test_vec_index_assignment_stores():
    result, _ = run_main("""
fn main() -> int {
    let v = Vec.new();
    v.push(10);
    v.push(20);
    v.push(30);
    v[1] = 99;
    v[0] = v[2];
    v[0] + v[1] + v[2]
}
""")
    assert result == 30 + 99 + 30


def test_vec_index_assignment_through_function_argument():
    """The store mutates the one shared MxVec (identity semantics)."""
    result, _ = run_main("""
fn zero_first(v: Vec) -> () {
    v[0] = 0
}

fn main() -> int {
    let v = Vec.new();
    v.push(7);
    zero_first(v);
    v[0]
}
""")
    assert result == 0


def test_nested_vec_index_assignment():
    result, _ = run_main("""
fn main() -> int {
    let row = Vec.new();
    row.push(1);
    row.push(2);
    let m = Vec.new();
    m.push(row);
    m[0][1] = 42;
    row[1]
}
""")
    assert result == 42


def test_index_assignment_out_of_bounds_is_loud():
    interp = compile_source("""
fn main() -> int {
    let v = Vec.new();
    v.push(1);
    v[3] = 9;
    0
}
""")
    with pytest.raises(InterpError, match="out of bounds"):
        interp.call("main", [])


def test_fixed_vector_index_assignment_is_a_value_update():
    """`v[0] = 9` on a fixed vector[T,N] rebinds the variable with a
    functionally updated vector (value semantics, like struct fields)."""
    result, _ = run_main("""
fn main() -> int {
    let @mut v = vector[int, 3](1, 2, 3);
    v[0] = 9;
    v[0] + v[1] + v[2]
}
""")
    assert result == 9 + 2 + 3


def test_struct_field_vector_index_assignment_writes_back():
    """The ownership.mx shape: buf.data[0] = 42 through a @mut param."""
    result, _ = run_main("""
struct Buffer {
    data: vector[Int, 3]
}

fn process(buf: @mut Buffer) {
    buf.data[0] = 42
}

fn main() -> int {
    let buf = Buffer { data: vector[int, 3](1, 2, 3) };
    process(buf);
    buf.data[0]
}
""")
    assert result == 42


def test_index_assignment_into_immutable_temporary_is_loud():
    """A fixed vector reached through another index is a temporary — no
    place to write back to, so the store must error, not vanish."""
    interp = compile_source("""
fn main() -> int {
    let row = vector[int, 2](1, 2);
    let m = Vec.new();
    m.push(row);
    m[0][1] = 9;
    0
}
""")
    with pytest.raises(InterpError, match="immutable vector"):
        interp.call("main", [])


def test_index_assignment_into_string_is_loud():
    interp = compile_source("""
fn main() -> int {
    let s = "abc";
    s[0] = "z";
    0
}
""")
    with pytest.raises(InterpError, match="string"):
        interp.call("main", [])


def test_slice_assignment_is_loud_at_compile_time():
    with pytest.raises(NotImplementedError, match="slice"):
        compile_source("""
fn main() -> int {
    let v = Vec.new();
    v[0:1] = 9;
    0
}
""")


def test_map_remove_uses_in_place_stores():
    """std.map's remove now shifts in place; behavior stays correct."""
    result, _ = run_main("""
from std.map import Map, empty, put, remove, get_or, size, contains_key;

fn main() -> int {
    let m = empty();
    put(m, 1, 10);
    put(m, 2, 20);
    put(m, 3, 30);
    let r = match remove(m, 2) { Some(v) => v, None => 0 - 1 };
    let gone = if contains_key(m, 2) { 0 } else { 1 };
    let kept = get_or(m, 1, 0 - 1) + get_or(m, 3, 0 - 1);
    r + gone + kept + size(m)
}
""")
    assert result == 20 + 1 + 40 + 2


# ----------------------------------------------------------------------
# Gap 5: module-level `let` read back as unit, silently
# ----------------------------------------------------------------------

def test_module_level_let_is_a_real_constant():
    result, _ = run_main("""
let ANSWER = 42;

fn main() -> int {
    ANSWER
}
""")
    assert result == 42


def test_module_level_let_initializers_run_in_order():
    result, _ = run_main("""
let BASE = 10;
let SCALED = BASE * 5;

fn main() -> int {
    SCALED + BASE
}
""")
    assert result == 60


def test_module_level_let_visible_in_every_function():
    result, _ = run_main("""
let K = 7;

fn scaled(x: int) -> int { x * K }

fn main() -> int {
    scaled(3) + K
}
""")
    assert result == 28


def test_std_math_constants_are_real_values():
    result, _ = run_main("""
from std.math import pi, tau, e;

fn main() -> int {
    let a = if pi > 3.14 { 1 } else { 0 };
    let b = if tau > 6.28 { 1 } else { 0 };
    let c = if e > 2.71 { 1 } else { 0 };
    let d = if tau < 6.29 { 1 } else { 0 };
    a + b + c + d
}
""")
    assert result == 4


def test_module_constant_collision_is_loud():
    from metaxu.errors import CompileError
    import os
    repo_root = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..", ".."))
    with pytest.raises(CompileError, match="constant"):
        # both this file and std.math declare `pi` -> one global namespace
        compile_source("""
from std.math import pi;

let pi = 3.0;

fn main() -> int { 0 }
""")


# ----------------------------------------------------------------------
# Gap 3: `op() -> ()` handler arms silently dropped
# ----------------------------------------------------------------------

def test_unit_literal_handler_arm_is_kept_as_abort_arm():
    """`op() -> ()` returns without resuming: abort with unit — the arm
    must exist (it used to vanish, leaving the op unhandled)."""
    result, _ = run_main("""
effect Stop {
    fn stop() -> ();
    fn go(x: int) -> int;
}

fn main() -> int {
    let r = handle Stop with {
        stop() -> (),
        go(x) -> { resume(x + 1) }
    } in {
        let a = perform Stop.go(10);
        perform Stop.stop();
        a + 1000
    };
    # stop() aborts the handle with unit, so r is unit, not 1011
    let flag = if r == () { 5 } else { 6 };
    flag
}
""")
    assert result == 5


def test_unit_literal_arm_comma_separated_keeps_following_arm():
    """The arm AFTER a `-> ()` arm must also survive."""
    result, _ = run_main("""
effect Pair {
    fn first() -> ();
    fn second() -> int;
}

fn main() -> int {
    handle Pair with {
        first() -> (),
        second() -> { resume(21) }
    } in {
        perform Pair.second() * 2
    }
}
""")
    assert result == 42


def test_unit_literal_expression_is_unit():
    result, _ = run_main("""
fn nothing() -> () { () }

fn main() -> int {
    let u = nothing();
    if u == () { 9 } else { 0 }
}
""")
    assert result == 9


def test_nonempty_tuple_literal_is_loud():
    with pytest.raises(NotImplementedError, match="tuple"):
        compile_source("""
fn main() -> int {
    let t = (1, 2);
    0
}
""")


# ----------------------------------------------------------------------
# Gap 4: scalar captures were by-value; mutation silently lost
# ----------------------------------------------------------------------

def test_closure_mutation_of_captured_scalar_writes_back():
    result, _ = run_main("""
fn main() -> int {
    let @mut x = 1;
    let bump = fn() -> () { x = x + 5 };
    bump();
    bump();
    x
}
""")
    assert result == 11


def test_closure_counter_keeps_state_across_calls():
    result, _ = run_main("""
fn make_step() -> int {
    let @mut n = 0;
    let step = fn() -> int { n = n + 1; n };
    step();
    step();
    step()
}

fn main() -> int {
    make_step()
}
""")
    assert result == 3


def test_enclosing_scope_and_closure_alias_the_same_binding():
    """Writes made OUTSIDE the closure are visible inside it too."""
    result, _ = run_main("""
fn main() -> int {
    let @mut x = 1;
    let read = fn() -> int { x = x + 0; x };
    let a = read();
    x = 40;
    let b = read();
    a + b + x
}
""")
    assert result == 1 + 40 + 40


def test_write_only_capture_is_captured_and_writes_back():
    """A body that only ASSIGNS an enclosing variable (never reads it)
    still captures it: the write must land in the enclosing binding."""
    result, _ = run_main("""
struct P { x: int }

fn main() -> int {
    let @mut p = P { x: 1 };
    let set = fn() -> () { p = P { x: 9 } };
    set();
    p.x
}
""")
    assert result == 9


def test_escaped_closure_keeps_mutable_state():
    result, _ = run_main("""
fn make() -> fn() -> int {
    let @mut n = 100;
    fn() -> int { n = n + 1; n }
}

fn main() -> int {
    let c = make();
    c();
    c()
}
""")
    assert result == 102


def test_handler_arm_mutation_of_captured_scalar_writes_back():
    """The stdlib's Vec-as-cell workaround is no longer needed: a handler
    arm assigning a captured scalar is visible after the handle."""
    result, _ = run_main("""
effect Tick {
    fn tick() -> int;
}

fn main() -> int {
    let @mut count = 0;
    let r = handle Tick with {
        tick() -> { count = count + 1; resume(count) }
    } in {
        perform Tick.tick();
        perform Tick.tick();
        perform Tick.tick()
    };
    count * 100 + r
}
""")
    assert result == 3 * 100 + 3


def test_effect_body_mutation_of_captured_scalar_writes_back():
    result, _ = run_main("""
effect Noop {
    fn poke() -> int;
}

fn main() -> int {
    let @mut acc = 0;
    handle Noop with {
        poke() -> { resume(1) }
    } in {
        acc = acc + perform Noop.poke();
        acc = acc + 10;
        0
    };
    acc
}
""")
    assert result == 11


# ----------------------------------------------------------------------
# Gap 6: `%` modulo operator
# ----------------------------------------------------------------------

def test_modulo_operator_full_pipeline():
    result, _ = run_main("""
fn main() -> int {
    let a = 7 % 3;
    let b = 10 % 2;
    let c = 9 % 5 * 100;
    a + b + c
}
""")
    assert result == 1 + 0 + 400


def test_modulo_has_multiplicative_precedence():
    result, _ = run_main("""
fn main() -> int {
    1 + 7 % 3
}
""")
    assert result == 2


def test_modulo_parity_in_condition():
    result, _ = run_main("""
fn is_even(n: int) -> bool {
    n % 2 == 0
}

fn main() -> int {
    let a = if is_even(4) { 1 } else { 0 };
    let b = if is_even(5) { 0 } else { 1 };
    a + b
}
""")
    assert result == 2


def test_modulo_type_conflict_is_loud():
    from metaxu.errors import CompileError
    with pytest.raises(Exception, match="[Tt]ype"):
        compile_source("""
fn main() -> int {
    1 % "a"
}
""")


# ----------------------------------------------------------------------
# Zip comprehensions (exposed while making tuple literals loud)
# ----------------------------------------------------------------------

def test_zip_comprehension_iterates_in_lockstep():
    result, _ = run_main("""
fn main() -> int {
    let xs = vector[int, 3](1, 2, 3);
    let ys = vector[int, 3](10, 20, 30);
    let sums = vector[int, 3](a + b for (a, b) in (xs, ys));
    sums[0] + sums[1] + sums[2]
}
""")
    assert result == 11 + 22 + 33


def test_zip_comprehension_length_mismatch_is_loud():
    interp = compile_source("""
fn main() -> int {
    let xs = vector[int, 2](1, 2);
    let ys = vector[int, 3](10, 20, 30);
    let sums = vector[int, 2](a + b for (a, b) in (xs, ys));
    sums[0]
}
""")
    with pytest.raises(InterpError, match="length"):
        interp.call("main", [])
