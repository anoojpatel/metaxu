"""Tuples: `(a, b)` literals, types, and destructuring.

std/README.md item 8 recorded "no tuple destructuring — `let (a, b) = p` is
a parse error" as the gap that kept `std.iter` on a hand-rolled
`struct Pair<A, B>` and kept `enumerate`/`zip` out of `std.stream`. This
file pins the fix, through parsed source only (the repo convention).

WHAT A TUPLE IS. An anonymous struct: `(a, b)` lowers to
`alloc_struct "__tuple2" { _0of2: a, _1of2: b }` and destructuring is a
`field_get` per element (`compiler/hir.py`, TUPLE_STRUCT_PREFIX). MIR gains
no op, the interpreter gains no value class, and native codegen inherits
the entire struct path — layout, GEPs, byval params, sret returns, field
kinds — so tuples lower natively with no backend feature of their own.

WHY THE FIELD NAME REPEATS THE ARITY (`_0of2`, not `_0`). Metaxu's
inference has no tuple type, so nothing upstream can reject
`let (a, b) = triple`. With plain `_0`/`_1` names that program would
silently bind a PREFIX of a 3-tuple — a positional pattern quietly meaning
something other than what it says. Because the arity is in the name,
`_0of2` simply does not exist on a `__tuple3`, so the mismatch is a loud
error with no new MIR op, builtin or runtime check.

THE 1-TUPLE RULE. There are none. `(e)` is parenthesized grouping (it
always was), `(e,)` is a syntax error, and `()` is the unit value — so a
tuple is two or more elements, and every shape that would need a 1-tuple
is rejected by name rather than given a `__tuple1` layout.

NATIVE. Differential tests prove interpreter == clang-compiled native for
construction, element reads, `let`/`match`/`for` destructuring, and tuples
crossing function boundaries. The two shapes that cannot lower demote with
a reason and never miscompile: a module that builds two DIFFERENT tuple
types of the same arity (they share one native layout), and a tuple nested
directly inside a tuple of the same arity (`__tuple2` would inline itself).
"""
from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from metaxu.compiler.hir import (
    AST_NODE_TRIAGE, HIRBuilder, LOWERED, PATTERN_TRIAGE, TUPLE_STRUCT_PREFIX,
    UnsupportedConstruct, is_tuple_struct, tuple_field_name,
    tuple_struct_name,
)
from metaxu.compiler.llvm_run import compile_and_run
from metaxu.compiler.codegen_llvm import emit_llvm
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import InterpError, MirInterpreter, UNIT
from metaxu.compiler.pipeline import (
    TypeCheckError, build_context_from_source, run_pipeline_ctx,
)
from metaxu import parser as parser_mod
from metaxu.errors import CompileError

needs_clang = pytest.mark.skipif(
    shutil.which("clang") is None, reason="clang is not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_main(source: str, entry: str = "main"):
    """Full strict pipeline, then execute `entry`; returns (result, prints)."""
    ctx = build_context_from_source(source, file_path="<mem>")
    run_pipeline_ctx(ctx)          # strict: raises on type/borrow errors
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    prints: list[str] = []

    def _print(*args):
        prints.append(" ".join(str(a) for a in args))
        return UNIT

    interp.register_builtin("print", _print)
    interp.register_builtin("println", _print)
    return interp.call(entry, []), prints


def mir_from_source(source: str):
    ctx = build_context_from_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    return lower_hir_to_mir(hir)


def mir_text(source: str) -> str:
    return "\n".join(str(op) for f in mir_from_source(source)
                     for b in f.blocks for op in b.ops)


def assert_native_matches_interp(source: str, tmp_path, entry: str = "main",
                                 clang_args: tuple[str, ...] = ()):
    """The differential assertion: clang-compiled result == interpreter."""
    result, printed = run_main(source, entry)
    expected_out = "".join(line + "\n" for line in printed)
    ir = emit_llvm(mir_from_source(source))
    exit_code, stdout = compile_and_run(ir, entry, workdir=str(tmp_path),
                                        clang_args=clang_args)
    assert stdout == expected_out
    if result is not UNIT and isinstance(result, (bool, int)):
        assert exit_code == int(result) % 256
    return ir


_ASAN_PROBE: list[bool] = []


def asan_available() -> bool:
    """True when clang can link -fsanitize=address (ASan runtime installed)."""
    if not _ASAN_PROBE:
        if shutil.which("clang") is None:
            _ASAN_PROBE.append(False)
        else:
            import os
            import tempfile
            with tempfile.TemporaryDirectory(prefix="metaxu_asan_probe_") as d:
                c = os.path.join(d, "t.c")
                with open(c, "w") as fh:
                    fh.write("int main(void){return 0;}\n")
                proc = subprocess.run(
                    ["clang", "-fsanitize=address", c,
                     "-o", os.path.join(d, "t")],
                    capture_output=True, text=True)
                _ASAN_PROBE.append(proc.returncode == 0)
    return _ASAN_PROBE[0]


needs_asan = pytest.mark.skipif(
    not asan_available(),
    reason="clang ASan runtime not available (compile probe failed)")


def native_placeholder_reasons(source: str) -> str:
    """The reasons codegen gives for functions it refuses to emit."""
    ir = emit_llvm(mir_from_source(source))
    return "\n".join(l for l in ir.splitlines() if "reason:" in l)


# ===========================================================================
# 1. Representation
# ===========================================================================

def test_a_tuple_is_an_anonymous_struct_with_arity_bearing_fields():
    """The whole design in one assertion: no new MIR op, an `alloc_struct`
    whose name carries the arity and whose FIELD NAMES carry it too."""
    text = mir_text("fn main() -> int { let (a, b) = (1, 2); a + b }")
    assert "('alloc_struct', '__tuple2', 'local')" in text
    assert "('_0of2', " in text and "('_1of2', " in text
    assert "('field_get', '_0of2')" in text
    assert "('field_get', '_1of2')" in text
    # Nothing tuple-specific reached MIR: every instruction kind here is one
    # that already existed for ordinary structs.
    kinds = {op[2][0] for f in mir_from_source(
        "fn main() -> int { let (a, b) = (1, 2); a + b }")
        for b in f.blocks for op in b.ops
        if op[0] == "let" and len(op) == 4}
    assert kinds <= {"const", "alloc_struct", "field_get", "copy", "binop"}


def test_arity_is_in_the_struct_name_and_in_every_field_name():
    assert tuple_struct_name(3) == "__tuple3"
    assert tuple_field_name(0, 3) == "_0of3"
    assert tuple_field_name(2, 3) == "_2of3"
    assert is_tuple_struct("__tuple2") and not is_tuple_struct("Pair")
    # A 2-tuple's element name is NOT a 3-tuple's element name, which is
    # exactly what makes an arity mismatch loud instead of a prefix bind.
    assert tuple_field_name(0, 2) != tuple_field_name(0, 3)


def test_the_parser_and_hir_agree_on_the_tuple_spelling():
    """`parser.py` spells the prefix and the field name itself rather than
    importing the HIR builder; the two must not drift."""
    assert parser_mod.TUPLE_TYPE_PREFIX == TUPLE_STRUCT_PREFIX
    for arity in (2, 3, 7):
        for i in range(arity):
            assert parser_mod.tuple_field_name(i, arity) == \
                tuple_field_name(i, arity)


def test_triage_tables_list_tuples_as_lowered():
    """`TupleLiteral` was UNSUPPORTED in both tables; implementing it means
    moving both entries, and test_hir_coverage holds the tables to the live
    AST classes."""
    assert AST_NODE_TRIAGE["TupleLiteral"][0] == LOWERED
    assert PATTERN_TRIAGE["TupleLiteral"][0] == LOWERED


# ===========================================================================
# 2. Literals, and the 1-tuple rule
# ===========================================================================

def test_pair_and_triple_literals_round_trip():
    assert run_main("fn main() -> int { let (a, b) = (1, 2); a + b }")[0] == 3
    assert run_main(
        "fn main() -> int { let (a, b, c) = (1, 2, 3); a + b + c }")[0] == 6


def test_a_parenthesized_expression_is_grouping_not_a_one_tuple():
    """The grammar hazard: `(e)` has always been grouping, and stays so."""
    assert run_main("fn main() -> int { let x = (1); x + 1 }")[0] == 2
    assert run_main("fn main() -> int { (2 + 3) * 4 }")[0] == 20


def test_a_trailing_comma_does_not_make_a_one_tuple():
    """Rust spells the 1-tuple `(a,)`. Metaxu does not have one, so rather
    than accepting the spelling and inventing a `__tuple1` layout nothing
    else understands, it stays a syntax error."""
    with pytest.raises(CompileError, match="Syntax error"):
        run_main("fn main() -> int { let x = (1,); 0 }")


def test_the_empty_tuple_is_still_unit():
    """Pre-existing behaviour that tuples must not disturb: `()` is the unit
    VALUE, not a 0-tuple (std/README.md gap 3)."""
    result, _ = run_main("""
fn nothing() -> () { () }
fn main() -> int { let u = nothing(); if u == () { 9 } else { 0 } }
""")
    assert result == 9


def test_tuples_nest():
    assert run_main("""
fn main() -> int {
    let p = ((1, 2), 3);
    let (q, r) = p;
    let (a, b) = q;
    a + b + r
}
""")[0] == 6


def test_tuple_elements_keep_their_own_types():
    _, prints = run_main("""
fn main() -> int {
    let (s, n) = ("hi", 7);
    print(s);
    print(n);
    0
}
""")
    assert prints == ["hi", "7"]


def test_a_tuple_prints_like_a_tuple():
    """The runtime value is an MxStruct, but it must not READ like one."""
    assert run_main("fn main() -> int { print((1, 2)); 0 }")[1] == ["(1, 2)"]
    assert run_main("fn main() -> int { print((1, 2, 3)); 0 }")[1] == \
        ["(1, 2, 3)"]


def test_tuples_compare_structurally():
    assert run_main(
        "fn main() -> int { if (1, 2) == (1, 2) { 1 } else { 0 } }")[0] == 1
    assert run_main(
        "fn main() -> int { if (1, 2) == (1, 3) { 1 } else { 0 } }")[0] == 0


def test_a_tuple_literal_element_that_cannot_resolve_is_loud():
    """A dropped element would build a SHORT tuple, which is the silent
    degradation StructInstantiation already refuses."""
    with pytest.raises(TypeCheckError, match="undefined variable 'zzz'"):
        run_main("fn main() -> int { let (a, b) = (1, zzz); a }")


# ===========================================================================
# 3. `let` destructuring
# ===========================================================================

def test_let_destructuring_binds_positionally():
    assert run_main("""
fn main() -> int {
    let (first, second) = (10, 3);
    first - second
}
""")[0] == 7


def test_let_destructuring_carries_the_binding_mode():
    """`let @mut (a, b) = ..` makes BOTH names mutable, not just the temp."""
    assert run_main("""
fn main() -> int {
    let @mut (a, b) = (1, 2);
    a = a + 5;
    a + b
}
""")[0] == 8


def test_let_destructuring_of_a_call_result():
    assert run_main("""
fn split(x: int) -> (int, int) { (x, x * 2) }
fn main() -> int { let (a, b) = split(3); a + b }
""")[0] == 9


def test_a_tuple_type_annotation_parses_and_checks():
    """`(A, B)` in type position names the same anonymous struct `(a, b)`
    builds; `(A)` is grouping there too."""
    assert run_main("""
fn total(p: (int, int)) -> int { let (a, b) = p; a + b }
fn main() -> int { total((4, 5)) }
""")[0] == 9


def test_let_destructuring_of_one_name_is_loud():
    """`let (a) = e` would need a 1-tuple. There is none, so say so instead
    of quietly treating the parentheses as grouping and binding the whole
    value to `a`."""
    with pytest.raises(CompileError, match="1-element tuple, which does not"):
        run_main("fn main() -> int { let (a) = (1, 2); 0 }")


def test_repeating_a_name_in_a_destructuring_is_loud():
    """Both elements would bind `a` and the LAST would win — silently
    discarding an element the program named."""
    with pytest.raises(CompileError, match="duplicate name"):
        run_main("fn main() -> int { let (a, a) = (1, 2); a }")


def test_the_destructuring_temporary_is_deterministic():
    """The temp is named from the SOURCE FILE and OFFSET, not a counter: one
    Parser instance is shared across files (`compiler/shared_parser.py`), so
    a counter would make the generated name depend on how many files had
    been parsed before — i.e. on test ordering, which the frozen-AST and MIR
    goldens would see."""
    src = "fn main() -> int { let (a, b) = (1, 2); a + b }"
    assert mir_text(src) == mir_text(src)
    assert re.search(r"__tuple_\w+_at\d+", mir_text(src))


def test_module_level_destructuring_in_two_modules_does_not_collide(tmp_path):
    """The temp's FILE half is not decoration. A module-level
    `let (a, b) = ..` is hoisted into `__module_init` and published as a
    module CONSTANT, and module constants share one global namespace — so
    an offset-only name made two modules destructuring at the same column a
    loud "constant '__tuple_at4' is declared in both module 'ma' and module
    'mb'" on perfectly good code."""
    (tmp_path / "ma.mx").write_text("let (p, q) = (1, 2);\nexport { p, q };\n")
    (tmp_path / "mb.mx").write_text("let (r, s) = (30, 40);\nexport { r, s };\n")
    main = tmp_path / "main.mx"
    main.write_text("""
from ma import p, q;
from mb import r, s;

fn main() -> int { p + q + r + s }
""")
    ctx = build_context_from_source(main.read_text(), file_path=str(main))
    run_pipeline_ctx(ctx)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    assert interp.call("main", []) == 73


# ===========================================================================
# 4. Arity is exact, in both directions
# ===========================================================================

def test_destructuring_more_elements_than_the_tuple_has_is_loud():
    with pytest.raises(InterpError, match="tuple arity mismatch"):
        run_main("fn main() -> int { let (a, b, c) = (1, 2); a }")


def test_destructuring_fewer_elements_than_the_tuple_has_is_loud():
    """THE reason element names carry their arity. A prefix bind here would
    be a positional pattern silently meaning something other than what it
    says, and Metaxu's inference has no tuple type to catch it earlier."""
    with pytest.raises(InterpError, match="tuple arity mismatch"):
        run_main("fn main() -> int { let (a, b) = (1, 2, 3); a + b }")


def test_a_match_arm_of_the_wrong_arity_is_loud():
    with pytest.raises(InterpError, match="tuple arity mismatch"):
        run_main("""
fn main() -> int {
    let p = (1, 2, 3);
    match p { (a, b) => a + b }
}
""")


def test_the_arity_error_names_both_arities():
    with pytest.raises(InterpError) as exc:
        run_main("fn main() -> int { let (a, b) = (1, 2, 3); a }")
    msg = str(exc.value)
    assert "2-element tuple pattern" in msg and "3-element tuple" in msg


# ===========================================================================
# 5. Match patterns
# ===========================================================================

def test_a_tuple_pattern_destructures_a_match_scrutinee():
    assert run_main("""
fn main() -> int {
    let p = (3, 4);
    match p { (x, y) => x * y }
}
""")[0] == 12


def test_tuple_subpatterns_are_refutable():
    """The tuple SHAPE carries no test, but what is inside it does — so the
    arms really are tried in order and a literal element can decline."""
    src = """
fn classify(p) -> string {
    match p {
        (0, y) => "zero-first",
        (x, 0) => "zero-second",
        (x, y) => "neither"
    }
}
fn main() -> int {
    print(classify((0, 5)));
    print(classify((5, 0)));
    print(classify((5, 5)));
    0
}
"""
    assert run_main(src)[1] == ["zero-first", "zero-second", "neither"]


def test_tuple_patterns_nest_with_constructor_patterns():
    src = """
fn go(p) -> int {
    match p {
        (Some(x), y) => x + y,
        (None, y) => y
    }
}
fn main() -> int { go((Some(3), 4)) + go((None, 10)) }
"""
    assert run_main(src)[0] == 17


def test_a_tuple_pattern_binder_is_visible_in_the_arm_body():
    """`name_resolution.bind_pattern` has to know tuple patterns bind, or
    every element name reads as an undefined variable."""
    assert run_main("""
fn main() -> int {
    match (6, 7) { (a, b) => a * b }
}
""")[0] == 42


def test_an_undefined_name_in_a_tuple_arm_body_is_still_caught():
    """The anti-false-positive direction has a matching direction: making
    binders visible must not make the resolver blind."""
    with pytest.raises(TypeCheckError, match="undefined variable 'nope'"):
        run_main("""
fn main() -> int {
    match (1, 2) { (a, b) => a + nope }
}
""")


def test_the_unit_pattern_is_rejected_rather_than_matching_everything():
    """`()` has no elements to read, so a `()` arm would match EVERY value —
    a silent catch-all, which is what PATTERN_TRIAGE exists to prevent."""
    with pytest.raises(UnsupportedConstruct,
                       match="a tuple pattern needs two or more"):
        run_main("fn main() -> int { let x = 5; match x { () => 1 } }")


def test_tuple_patterns_work_in_if_let_and_while_let():
    assert run_main("""
fn main() -> int {
    let p = (1, 2);
    if let (a, b) = p { a + b } else { 0 }
}
""")[0] == 3


# ===========================================================================
# 6. `for` destructuring
# ===========================================================================

def test_for_destructuring_over_a_vec_of_tuples():
    assert run_main("""
fn main() -> int {
    let @mut pairs = Vec.new();
    pairs.push((1, 2));
    pairs.push((3, 4));
    let @mut total = 0;
    for (a, b) in pairs { total = total + a * b }
    total
}
""")[0] == 14


def test_for_destructuring_of_one_name_is_loud():
    with pytest.raises(CompileError, match="1-element tuple, which does not"):
        run_main("""
fn main() -> int {
    let @mut v = Vec.new();
    v.push((1, 2));
    for (a) in v { () }
    0
}
""")


def test_the_zip_comprehension_form_still_means_lockstep_iteration():
    """`(xs, ys)` as a comprehension ITERABLE is the pre-existing zip form
    (a tuple of sequences), not a tuple value to destructure. Tuples must
    not have quietly taken that spelling over."""
    assert run_main("""
fn main() -> int {
    let xs = vector[int, 2](1, 2);
    let ys = vector[int, 2](3, 4);
    let z = vector[int, 2](a + b for (a, b) in (xs, ys));
    z[0] + z[1]
}
""")[0] == 10


def test_a_shorthand_lambda_of_two_names_is_still_two_parameters():
    """`(a, b) -> e` is a TWO-PARAMETER lambda, and always has been
    (`Parser._params_from_expr`). Tuples do not repurpose that spelling into
    a destructuring parameter — which is why lambda parameters are the one
    position this feature deliberately leaves alone."""
    assert run_main("""
fn apply2(f) -> int { f(3, 4) }
fn main() -> int { apply2((a, b) -> a * b) }
""")[0] == 12


# ===========================================================================
# 7. Native
# ===========================================================================

@needs_clang
def test_native_construction_and_element_reads_match(tmp_path):
    ir = assert_native_matches_interp("""
fn main() -> int {
    let (a, b) = (7, 5);
    print(a);
    print(b);
    a - b
}
""", tmp_path)
    # Riding the struct path: a real named LLVM aggregate and GEPs, not a
    # bespoke tuple representation.
    assert "%struct.__tuple2" in ir
    assert "getelementptr inbounds %struct.__tuple2" in ir


@needs_clang
def test_native_match_destructuring_matches(tmp_path):
    assert_native_matches_interp("""
fn main() -> int {
    let p = (3, 4);
    match p { (x, y) => { print(x); print(y); x * y } }
}
""", tmp_path)


@needs_clang
def test_native_tuples_cross_function_boundaries(tmp_path):
    assert_native_matches_interp("""
fn make(x: int) -> (int, int) { (x, x * 2) }
fn total(p: (int, int)) -> int { let (a, b) = p; a + b }
fn main() -> int {
    let t = make(4);
    let n = total(t);
    print(n);
    n
}
""", tmp_path)


@needs_clang
def test_native_three_tuples_and_string_elements_match(tmp_path):
    assert_native_matches_interp("""
fn main() -> int {
    let (a, b, c) = (1, 2, 3);
    let (s, t) = ("hi", "yo");
    print(s);
    print(t);
    print(c);
    a + b + c
}
""", tmp_path)


@needs_asan
def test_a_global_tuple_is_heap_allocated_and_freed(tmp_path):
    """Modes reach tuples because a tuple IS a struct: `@global` puts it on
    the heap through the same entry-block malloc + free-on-every-ret-path
    the struct path already had.  Full leak-check (no detect_leaks=0): the
    contract here is that every byte is freed."""
    ir = assert_native_matches_interp("""
fn main() -> int {
    let @global (a, b) = (11, 4);
    print(a);
    print(b);
    print(a - b);
    0
}
""", tmp_path, clang_args=("-fsanitize=address",))
    assert re.search(r"call ptr @malloc\(i64 \d+\).*__tuple2", ir)
    assert re.search(r"call void @free\(ptr %hv\.\w+\)\s*; @global struct", ir)


def test_two_different_tuple_types_of_one_arity_demote_with_a_reason():
    """`__tuple2` is EVERY 2-tuple in the module, so `(1, 2)` and
    `(1.0, 2.5)` share one native layout. Joining their field kinds would
    retype the int literal as a double (i64 is the kind lattice's bottom) —
    wrong code, not a demotion — so the tuple struct is marked bad instead
    and every function touching it falls back to the interpreter."""
    src = """
fn main() -> int {
    let (a, b) = (1, 2);
    let (c, d) = (1.0, 2.5);
    print(a);
    print(d);
    a
}
"""
    assert run_main(src)[0] == 1          # the interpreter is unaffected
    reasons = native_placeholder_reasons(src)
    assert "conflicting element representations" in reasons


def test_a_tuple_nested_in_a_same_arity_tuple_demotes_with_a_reason():
    """The other cost of naming tuple structs by arity alone: `((1, 2), 3)`
    makes `__tuple2` a field of `__tuple2`, which has no finite inline
    layout. Loud demotion, correct interpretation."""
    src = """
fn main() -> int {
    let p = ((1, 2), 3);
    let (q, r) = p;
    let (a, b) = q;
    a + b + r
}
"""
    assert run_main(src)[0] == 6
    assert "recursively inlined layout" in native_placeholder_reasons(src)


# ===========================================================================
# 8. The std.iter payoff
# ===========================================================================

def test_std_iter_enumerate_emits_real_tuples():
    """`std.iter` carried a hand-rolled `struct Pair<A, B>` and callers
    wrote `p.first` / `p.second`. Both are gone: the adapters emit tuples
    and consumers destructure."""
    result, _ = run_main("""
from std.stream import emit_vec, collect;
from std.iter import enumerate;
from std.vec import of3;

fn main() -> int {
    let ps = collect(enumerate(emit_vec(of3(10, 20, 30))));
    let @mut total = 0;
    for (i, x) in ps { total = total + i * x }
    total
}
""")
    assert result == 0 * 10 + 1 * 20 + 2 * 30


def test_std_iter_zip_pairs_are_matchable():
    result, _ = run_main("""
from std.stream import iota, emit_vec, collect;
from std.iter import zip;
from std.vec import of3;

fn main() -> int {
    let ps = collect(zip(iota(3), emit_vec(of3(7, 8, 9))));
    let @mut total = 0;
    let @mut i = 0;
    while i < len(ps) {
        total = total + match ps[i] { (a, b) => a * b };
        i = i + 1
    }
    total
}
""")
    assert result == 0 * 7 + 1 * 8 + 2 * 9


def test_std_iter_no_longer_exports_pair():
    """The struct is really gone, not merely unused: importing it fails."""
    with pytest.raises(CompileError, match="no symbol 'Pair'"):
        run_main("""
from std.iter import Pair;
fn main() -> int { 0 }
""")
