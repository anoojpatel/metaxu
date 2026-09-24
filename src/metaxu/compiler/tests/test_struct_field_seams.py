"""Regression tests for the struct-field seams behind the last
`examples/collections.mx` native demotion.

The demotion read

    ; function @mx_push: placeholder -- unsupported for direct LLVM emission
    ;   reason: struct 'List' has no field 'data'

while the source clearly declared `data: vector[T,N]`.  The message blamed
the declaration for something else entirely:

  * `struct List<T> { data: vector[T,N] }` never declares `N`.  A vector
    size must be an integer literal or a const generic parameter
    (`struct List<T, const N: int>`, the spelling used by
    examples/06_vector_operations.mx), so the program was ill-formed —
    and accepted.
  * `List { data: vector[T,N], len: 0 }` writes a TYPE where a value
    belongs (the value forms are `vector[T,N]()`, `vector[T,N](e, ...)`,
    `vector[T,N].filled(e)`).  That expression had no HIR lowering, so
    HIR's StructInstantiation case dropped the whole `data:` assignment;
    MIR's alloc_struct then listed only `len`, native codegen's struct
    table (built from alloc_struct sites) recorded List as a one-field
    struct, and `push`'s `field_get 'data'` "proved" the declaration wrong.

Work-or-loud, both directions:
  * the undeclared size parameter is rejected EARLY (TypeCheckError from
    the frozen constraint emitter) with a message that names the
    parameter, the field and the fix — including through a mode-annotated
    field type (`unique vector[Int,N]`), whose display used to be the
    useless `"ModeTypeAnnotationat 0x7f..."`;
  * the declared `const N: int` form compiles, keeps BOTH fields all the
    way to MIR, and `push` emits natively;
  * a type in value position is a loud HIR error instead of a vanishing
    field;
  * list literals (`[]`, `[a, b]`, `[x, ...xs]`) — which had NO lowering
    at all and therefore vanished the same way — build real Vecs.
"""
from __future__ import annotations

import os

import pytest

from metaxu.compiler.pipeline import (
    TypeCheckError,
    build_context_from_source,
    emit_llvm_from_source,
    run_pipeline_ctx,
)
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.mir import dump_mir
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import InterpError, MirInterpreter, UNIT


def compile_source(source: str) -> MirInterpreter:
    ctx = build_context_from_source(source)
    run_pipeline_ctx(ctx)  # strict: raises on type/borrow errors
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    return interp


def run_main(source: str):
    interp = compile_source(source)
    prints: list[str] = []
    interp.register_builtin(
        "print", lambda *a: (prints.append(" ".join(str(x) for x in a)), UNIT)[1])
    return interp.call("main", []), prints


def mir_text(source: str) -> str:
    ctx = build_context_from_source(source)
    run_pipeline_ctx(ctx)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    return dump_mir(lower_hir_to_mir(hir))


# ----------------------------------------------------------------------
# Loud: a vector size that names no declared parameter
# ----------------------------------------------------------------------

def test_undeclared_vector_size_param_is_rejected():
    with pytest.raises(TypeCheckError) as exc:
        run_pipeline_ctx(build_context_from_source("""
struct List<T> {
    data: vector[T,N]
    len: Int
}
"""))
    msg = str(exc.value)
    assert "undeclared type parameter 'N'" in msg
    assert "'data'" in msg and "'List'" in msg
    # The old, misleading message must not come back for this shape.
    assert "has no field" not in msg


def test_undeclared_vector_size_param_names_the_fix():
    with pytest.raises(TypeCheckError) as exc:
        run_pipeline_ctx(build_context_from_source("""
struct List<T> {
    data: vector[T,N]
}
"""))
    assert "struct List<T, const N: int>" in str(exc.value)


def test_undeclared_vector_size_param_through_mode_annotation():
    """`unique vector[Int,N]` froze to the repr "ModeTypeAnnotationat 0x..",
    which no field-level check could read (examples/ownership.mx hid the
    same ill-formed field behind it)."""
    with pytest.raises(TypeCheckError) as exc:
        run_pipeline_ctx(build_context_from_source("""
struct Buffer {
    data: unique vector[Int,N]
}
"""))
    assert "undeclared type parameter 'N'" in str(exc.value)
    assert "ModeTypeAnnotation" not in str(exc.value)


def test_undeclared_inner_vector_size_param_is_rejected():
    with pytest.raises(TypeCheckError) as exc:
        run_pipeline_ctx(build_context_from_source("""
struct Matrix<T, const N: int> {
    rows: vector[vector[T,M],N]
}
"""))
    assert "undeclared type parameter 'M'" in str(exc.value)


# ----------------------------------------------------------------------
# Works: declared parameters and literal sizes
# ----------------------------------------------------------------------

def test_declared_const_generic_size_is_accepted():
    text = mir_text("""
struct List<T, const N: int> {
    data: vector[T,N]
    len: Int
}

fn push_item<T, const N: int>(list: @mut List[T,N], item: T) {
    list.data[list.len] = item
    list.len = list.len + 1
}
""")
    assert "field_get" in text  # push_item lowered, no vanished field


def test_literal_and_nested_declared_sizes_are_accepted():
    mir_text("""
struct Matrix<T, const M: int, const N: int> {
    rows: vector[vector[T,N],M]
    fixed: vector[Int,3]
}
""")


def test_struct_type_params_used_as_sizes_survive_instantiation():
    """The whole point: both declared fields reach alloc_struct."""
    text = mir_text("""
struct List<T, const N: int> {
    data: vector[T,N]
    len: Int
}

fn main() -> int {
    let l = List { data: vector[int,4](1, 2, 3, 4), len: 4 }
    return l.len
}
""")
    assert "('alloc_struct', 'List', 'local'), (('data'" in text
    assert "('len'" in text


# ----------------------------------------------------------------------
# Loud: a type written where a value belongs
# ----------------------------------------------------------------------

def test_bare_vector_type_in_value_position_is_loud():
    with pytest.raises(NotImplementedError) as exc:
        compile_source("""
struct List<T, const N: int> {
    data: vector[T,N]
    len: Int
}

fn empty<T, const N: int>() -> List[T,N] {
    List { data: vector[T,N], len: 0 }
}
""")
    msg = str(exc.value)
    assert "is a TYPE, not a value" in msg
    assert "vector[T, N]()" in msg


def test_type_valued_field_never_yields_a_short_struct():
    """The failure mode being fixed: a field whose initializer cannot be
    lowered must stop the build, because a struct that silently loses a
    field reaches BOTH engines and misattributes the error to the
    declaration ("struct 'Pair' has no field 'b'")."""
    with pytest.raises(NotImplementedError) as exc:
        compile_source("""
struct Pair {
    a: Int
    b: Int
}

fn make() -> Pair {
    Pair { a: 1, b: vector[Int,2] }
}
""")
    assert "has no field" not in str(exc.value)
    assert "vector[Int, 2] is a TYPE, not a value" in str(exc.value)


# ----------------------------------------------------------------------
# List literals: the same vanishing-field seam, one node over
# ----------------------------------------------------------------------

def test_empty_list_literal_field_survives():
    result, prints = run_main("""
struct Queue<T> {
    items: Vec[T]
    capacity: int
}

fn main() {
    let q = Queue { items: [], capacity: 10 }
    print(len(q.items))
    print(q.capacity)
}
""")
    assert prints == ["0", "10"]


def test_list_literal_elements_reach_the_interpreter():
    _, prints = run_main("""
fn main() {
    let xs = [1, 2, 3];
    print(len(xs));
    print(xs[0]);
    print(xs[2]);
}
""")
    assert prints == ["3", "1", "3"]


def test_list_literal_spread_concatenates():
    _, prints = run_main("""
fn main() {
    let xs = [2, 3];
    let ys = [1, ...xs, 4];
    print(len(ys));
    print(ys[0]);
    print(ys[1]);
    print(ys[3]);
}
""")
    assert prints == ["4", "1", "2", "4"]


def test_list_literal_spread_copies_instead_of_aliasing():
    """`[...xs]` is a fresh list: Vec has identity semantics, so returning
    `xs` itself would make the copy observable through the original."""
    _, prints = run_main("""
fn main() {
    let xs = [1, 2];
    let ys = [...xs];
    ys.push(3);
    print(len(xs));
    print(len(ys));
}
""")
    assert prints == ["2", "3"]


def test_spreading_a_non_list_is_loud():
    interp = compile_source("""
fn main() {
    let n = 7;
    let ys = [1, ...n];
    print(len(ys));
}
""")
    with pytest.raises(InterpError) as exc:
        interp.call("main", [])
    assert "cannot spread" in str(exc.value)


# ----------------------------------------------------------------------
# The example itself
# ----------------------------------------------------------------------

COLLECTIONS_MX = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "..", "examples", "collections.mx")


def test_collections_example_emits_push_natively():
    """The example itself: List keeps both fields natively and `push` is a
    real definition.  `empty` still demotes, but for a reason that is TRUE
    (`vector[T,N]()` cannot be zero-initialized for a generic element type
    — the interpreter's __vec_zeros rejects it too), not for a field the
    declaration does have."""
    ir = emit_llvm_from_source(open(COLLECTIONS_MX).read())
    assert "%struct.List = type { ptr, i64 }  ; data, len" in ir
    # `push` ends in an assignment, so it returns unit (i64 0).
    assert "define i64 @mx_push(" in ir
    assert "has no field" not in ir
    assert ir.count("placeholder -- unsupported") == 1
    assert "@mx_empty: placeholder" in ir
    assert "__vec_zeros base type is not a constant" in ir
