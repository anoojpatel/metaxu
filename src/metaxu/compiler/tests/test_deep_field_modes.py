"""Deep ownership validation of struct field modes (roadmap item 5).

Rules under test (docs/ownership_and_borrowing.md):
- A @global value must not contain (transitively) any @local field.
- Global containers cannot store locals.
- Local containers MAY store references to globals.

Every test goes through the real front end via run_pipeline_from_source.
"""
from __future__ import annotations

import pathlib

import pytest

from metaxu.compiler.pipeline import run_pipeline_from_source
from metaxu.compiler.frozen_borrow_checker import BorrowCheckError


# ---------------------------------------------------------------------------
# Rule: a @global value must not contain (transitively) any @local field
# ---------------------------------------------------------------------------

def test_global_struct_with_direct_local_field_rejected():
    src = """
struct Scratch {
    @local temp: int,
    value: int
}

fn main() {
    let @global g = Scratch { temp: 1, value: 2 }
}
"""
    with pytest.raises(BorrowCheckError, match="deep|@local field"):
        run_pipeline_from_source(src)


def test_global_struct_with_transitively_local_field_rejected():
    src = """
struct Inner {
    @local temp: int
}

struct Outer {
    inner: Inner
}

fn main() {
    let @global g = Outer { inner: Inner { temp: 1 } }
}
"""
    with pytest.raises(BorrowCheckError, match="Outer.inner -> Inner.temp"):
        run_pipeline_from_source(src)


def test_transitive_error_is_structured_deep_locality():
    src = """
struct Inner {
    @local temp: int
}

struct Outer {
    inner: Inner
}

fn main() {
    let @global g = Outer { inner: Inner { temp: 1 } }
}
"""
    with pytest.raises(BorrowCheckError) as excinfo:
        run_pipeline_from_source(src)
    kinds = {e.kind for e in excinfo.value.errors}
    assert "deep-locality" in kinds


# ---------------------------------------------------------------------------
# Rule: global containers cannot store locals
# ---------------------------------------------------------------------------

def test_assigning_local_into_global_container_field_rejected():
    src = """
struct Box {
    value: int
}

fn main() {
    let @local x = 5
    let @global g = Box { value: 0 }
    g.value = x
}
"""
    with pytest.raises(BorrowCheckError, match="global containers cannot store locals"):
        run_pipeline_from_source(src)


def test_global_initializer_storing_local_rejected():
    src = """
struct Box {
    value: int
}

fn main() {
    let @local x = 5
    let @global g = Box { value: x }
}
"""
    with pytest.raises(BorrowCheckError, match="cannot store @local value 'x'"):
        run_pipeline_from_source(src)


# ---------------------------------------------------------------------------
# Rule: local containers MAY store references to globals
# ---------------------------------------------------------------------------

def test_local_container_holding_reference_to_global_accepted():
    src = """
struct Holder {
    r: int
}

fn main() {
    let @global g = 7
    let @local h = Holder { r: g }
}
"""
    run_pipeline_from_source(src)  # must not raise


def test_local_container_holding_borrow_of_global_accepted():
    src = """
struct Holder {
    r: int
}

fn main() {
    let @global g = 7
    let @local h = Holder { r: &g }
}
"""
    run_pipeline_from_source(src)  # must not raise


# ---------------------------------------------------------------------------
# Plain local structs with @local fields stay legal
# ---------------------------------------------------------------------------

def test_local_struct_with_local_fields_accepted():
    src = """
struct Scratch {
    @local temp: int,
    value: int
}

fn main() {
    let @local s = Scratch { temp: 1, value: 2 }
    let s2 = Scratch { temp: 3, value: 4 }
}
"""
    run_pipeline_from_source(src)  # must not raise


def test_assigning_local_into_local_container_accepted():
    src = """
struct Box {
    value: int
}

fn main() {
    let @local x = 5
    let @local b = Box { value: 0 }
    b.value = x
}
"""
    run_pipeline_from_source(src)  # must not raise


def test_counter_example_stays_accepted():
    """The Counter struct from examples/01 has a @local field but is bound
    locally (structs default local), so it must keep compiling."""
    example = pathlib.Path(__file__).resolve().parents[4] / "examples" / "01_modes_and_references.mx"
    run_pipeline_from_source(example.read_text())  # must not raise


# ---------------------------------------------------------------------------
# Transitivity through enum variant payload types
# ---------------------------------------------------------------------------

def test_global_struct_with_local_field_through_enum_rejected():
    src = """
struct Inner {
    @local temp: int
}

enum Wrap {
    Some(inner: Inner),
    None
}

struct Outer {
    wrapped: Wrap
}

fn main() {
    let @global g = Outer { wrapped: Wrap.None }
}
"""
    with pytest.raises(BorrowCheckError, match="deep|@local field"):
        run_pipeline_from_source(src)


# ---------------------------------------------------------------------------
# Rule: a field declared @const cannot be assigned, through let bindings,
# parameters, and nested paths alike (mut and plain fields stay writable)
# ---------------------------------------------------------------------------

def test_const_field_write_through_let_binding_rejected():
    src = """
struct P { @const name: string, age: int }

fn main() -> int {
    let mut p = P { name: "a", age: 1 };
    p.name = "b";
    0
}
"""
    with pytest.raises(BorrowCheckError, match="cannot assign to @const field 'name' of P"):
        run_pipeline_from_source(src)


def test_const_field_write_through_parameter_rejected():
    src = """
struct P { @const name: string, age: int }

fn rename(p: P) -> int {
    p.name = "b";
    0
}

fn main() -> int {
    rename(P { name: "a", age: 1 })
}
"""
    with pytest.raises(BorrowCheckError, match="cannot assign to @const field 'name' of P"):
        run_pipeline_from_source(src)


def test_const_field_write_through_nested_path_rejected():
    src = """
struct Inner { @const id: int, n: int }
struct Outer { inner: Inner }

fn main() -> int {
    let mut o = Outer { inner: Inner { id: 1, n: 2 } };
    o.inner.id = 9;
    0
}
"""
    with pytest.raises(BorrowCheckError, match="cannot assign to @const field 'id' of Inner"):
        run_pipeline_from_source(src)


def test_mut_and_plain_fields_stay_writable():
    src = """
struct P { @const name: string, @mut hits: int, age: int }

fn main() -> int {
    let mut p = P { name: "a", hits: 0, age: 1 };
    p.hits = 5;
    p.age = 2;
    0
}
"""
    run_pipeline_from_source(src)


def test_const_check_follows_the_binding_not_the_name():
    # P has a @const `name`; a Q-typed binding called `p` writes its own
    # plain `name` freely (zero false positives from name collisions).
    src = """
struct P { @const name: string }
struct Q { name: string }

fn main() -> int {
    let mut p = Q { name: "a" };
    p.name = "b";
    0
}
"""
    run_pipeline_from_source(src)
