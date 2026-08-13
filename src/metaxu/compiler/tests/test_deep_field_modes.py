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
