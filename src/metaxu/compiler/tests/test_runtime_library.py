"""Tests for the minimal runtime library: Vec, index/slice expressions,
math builtin methods, and fixed-size vector literals.

Every test goes through the real front end: parsed source -> desugar ->
freeze -> infer -> HIR -> MIR -> interpreter.

Runtime representation choices (documented in mir_interp):
- Vec<T> is a MUTABLE runtime object (MxVec wraps a Python list with identity
  semantics), so `self.elements.push(x)` inside a method is visible to the
  caller even though structs themselves have value semantics.
- vector[T, N] is an immutable value (MxVector wraps a tuple); element-wise
  arithmetic and slicing produce new vectors, and a scalar operand broadcasts.
"""
from __future__ import annotations

import math

import pytest

from metaxu.compiler.pipeline import build_context_from_source
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import (
    InterpError,
    MirInterpreter,
    MxVec,
    MxVector,
    UNIT,
)


def build_interp(source: str) -> MirInterpreter:
    ctx = build_context_from_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    return interp


def call(source: str, fn: str = "main", args: list | None = None):
    return build_interp(source).call(fn, args or [])


# ---------------------------------------------------------------------------
# Vec: new / push / pop / len / index
# ---------------------------------------------------------------------------

def test_vec_new_push_pop():
    result = call('''
        fn main() -> int {
            let v = Vec<Int>::new();
            v.push(42);
            v.push(17);
            v.pop()
        }
    ''')
    assert result == 17


def test_vec_len_and_index():
    result = call('''
        fn main() -> int {
            let v = Vec<Int>::new();
            v.push(10);
            v.push(20);
            v.push(30);
            v[1] + v.len()
        }
    ''')
    assert result == 23


def test_vec_new_returns_empty_mxvec():
    result = call('fn main() { Vec<Int>::new() }')
    assert isinstance(result, MxVec)
    assert len(result) == 0


def test_vec_is_mutable_through_method_calls():
    """Pushes inside &mut self methods are visible to the caller: the Vec is
    one shared mutable object (list identity), unlike value-semantics structs."""
    result = call('''
        struct Stack {
            elements: Vec<Int>,
        }

        implement Stack {
            fn push_twice(self, a: int, b: int) {
                self.elements.push(a);
                self.elements.push(b);
            }
        }

        fn main() -> int {
            let s = Stack { elements: Vec<Int>::new() };
            s.push_twice(1, 2);
            s.elements.len()
        }
    ''')
    assert result == 2


def test_index_expression_on_vec_field_through_mut_self_method():
    result = call('''
        struct Stack { elements: Vec<Int>, size: Int }

        implement Stack {
            fn peek_at(self, i: int) -> int {
                return self.elements[i];
            }
        }

        fn main() -> int {
            let s = Stack { elements: Vec<Int>::new(), size: 0 };
            s.elements.push(7);
            s.elements.push(9);
            s.peek_at(1)
        }
    ''')
    assert result == 9


def test_vec_pop_empty_is_a_clear_error():
    with pytest.raises(InterpError, match="pop: Vec is empty"):
        call('''
            fn main() -> int {
                let v = Vec<Int>::new();
                v.pop()
            }
        ''')


def test_vec_push_on_non_vec_is_a_clear_error():
    with pytest.raises(InterpError, match="push: expected a Vec receiver"):
        call('''
            fn main() {
                let x = 5;
                x.push(1)
            }
        ''')


def test_vec_index_out_of_bounds_is_a_clear_error():
    with pytest.raises(InterpError, match="index out of bounds"):
        call('''
            fn main() -> int {
                let v = Vec<Int>::new();
                v.push(1);
                v[3]
            }
        ''')


# ---------------------------------------------------------------------------
# Math builtin methods on numbers
# ---------------------------------------------------------------------------

def test_math_sqrt_method_on_float():
    assert call('fn main() -> float { let x = 9.0; x.sqrt() }') == 3.0


def test_math_sin_cos_methods_on_float():
    result = call('''
        fn main() -> float {
            let angle = 0.5;
            angle.sin() * angle.sin() + angle.cos() * angle.cos()
        }
    ''')
    assert result == pytest.approx(1.0)


def test_math_method_on_computed_receiver():
    """`(v.x*v.x + v.y*v.y).sqrt()` — method call on a parenthesized expr."""
    result = call('''
        struct V { x: float, y: float }
        fn main() -> float {
            let v = V { x: 3.0, y: 4.0 };
            (v.x * v.x + v.y * v.y).sqrt()
        }
    ''')
    assert result == 5.0


def test_math_method_on_non_number_is_a_clear_error():
    with pytest.raises(InterpError, match="sqrt: expected a number"):
        call('fn main() { let s = "hi"; s.sqrt() }')


def test_math_sqrt_domain_error_is_clear():
    with pytest.raises(InterpError, match="sqrt: domain error"):
        call('fn main() -> float { let x = 0.0 - 4.0; x.sqrt() }')


def test_user_trait_impl_wins_over_math_builtin():
    """A user impl providing `sqrt` for its own type beats the builtin; the
    builtin still serves receivers without an impl."""
    result = call('''
        struct Boxed { v: float }

        implement Boxed {
            fn sqrt(self) -> float { 123.0 }
        }

        fn main() -> float {
            let b = Boxed { v: 4.0 };
            let x = 16.0;
            b.sqrt() + x.sqrt()
        }
    ''')
    assert result == 127.0


# ---------------------------------------------------------------------------
# Fixed-size vector literals and operations
# ---------------------------------------------------------------------------

def test_vector_literal_and_element_access():
    result = call('''
        fn main() -> int {
            let v = vector[int, 3](1, 2, 3);
            v[0] + v[2]
        }
    ''')
    assert result == 4


def test_vector_literal_value():
    result = call('fn main() { vector[int, 3](1, 2, 3) }')
    assert result == MxVector(elements=(1, 2, 3))


def test_vector_literal_wrong_arity_is_a_clear_error():
    with pytest.raises(InterpError, match="2 elements for a vector of size 3"):
        call('fn main() { vector[int, 3](1, 2) }')


def test_vector_elementwise_add_and_mul():
    result = call('''
        fn main() {
            let a = vector[float, 3](1.0, 2.0, 3.0);
            let b = vector[float, 3](4.0, 5.0, 6.0);
            a + b * a
        }
    ''')
    assert result == MxVector(elements=(5.0, 12.0, 21.0))


def test_vector_scalar_broadcast():
    result = call('''
        fn main() {
            let a = vector[float, 2](1.5, 2.5);
            a * 2.0
        }
    ''')
    assert result == MxVector(elements=(3.0, 5.0))


def test_vector_size_mismatch_is_a_clear_error():
    src = '''
        fn main() {
            let a = vector[int, 2](1, 2);
            let b = vector[int, 3](1, 2, 3);
            a + b
        }
    '''
    with pytest.raises(InterpError, match="vector size mismatch"):
        call(src)


def test_vector_zeros():
    result = call('fn main() { vector[float, 4]() }')
    assert result == MxVector(elements=(0.0, 0.0, 0.0, 0.0))


def test_vector_zeros_int():
    result = call('fn main() { vector[int, 2]() }')
    assert result == MxVector(elements=(0, 0))


def test_vector_filled():
    result = call('fn main() { vector[float, 4].filled(1.0) }')
    assert result == MxVector(elements=(1.0, 1.0, 1.0, 1.0))


def test_vector_comprehension_over_range():
    result = call('''
        fn main() {
            vector[float, 4](x * 2.0 for x in 0..4)
        }
    ''')
    assert result == MxVector(elements=(0.0, 2.0, 4.0, 6.0))


def test_vector_comprehension_captures_enclosing_local():
    result = call('''
        fn main() {
            let scale = 10;
            vector[int, 3](x * scale for x in 0..3)
        }
    ''')
    assert result == MxVector(elements=(0, 10, 20))


def test_vector_comprehension_wrong_size_is_a_clear_error():
    with pytest.raises(InterpError, match="3 elements for a vector of size 4"):
        call('fn main() { vector[int, 4](x for x in 0..3) }')


def test_vector_slice_start_stop():
    result = call('''
        fn main() {
            let v = vector[int, 4](10, 20, 30, 40);
            v[1:3]
        }
    ''')
    assert result == MxVector(elements=(20, 30))


def test_vector_slice_step_and_reverse():
    interp = build_interp('''
        fn every_other() {
            let v = vector[int, 4](1, 2, 3, 4);
            v[::2]
        }
        fn reversed() {
            let v = vector[int, 4](1, 2, 3, 4);
            v[::-1]
        }
    ''')
    assert interp.call("every_other", []) == MxVector(elements=(1, 3))
    assert interp.call("reversed", []) == MxVector(elements=(4, 3, 2, 1))


def test_vector_len_builtin():
    assert call('fn main() -> int { let v = vector[int, 3](7, 8, 9); v.len() }') == 3


def test_indexing_non_indexable_is_a_clear_error():
    with pytest.raises(InterpError, match="cannot index"):
        call('fn main() { let x = 5; x[0] }')


def test_vector_to_string():
    result = call('''
        fn main() -> string {
            let v = vector[int, 2](1, 2);
            v.to_string()
        }
    ''')
    assert result == "vector[1, 2]"
