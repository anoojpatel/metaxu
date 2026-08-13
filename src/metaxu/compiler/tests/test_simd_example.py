"""Tests for the mechanisms behind examples/06_vector_operations.mx.

The example is capability-style SIMD: every operation performs
SimdOp.try_vectorize / try_horizontal to ASK whether a vectorized form is
available, and its match takes the scalar-fallback branch on None. Three
language/runtime mechanisms make that executable, each pinned here through
parsed source (never hand-built HIR/MIR):

1. Default effect handlers: `op(params) -> T = expr;` in an effect
   declaration makes an unhandled perform evaluate `expr` (the capability
   answers its default when nobody provides it). Installed handlers always
   win; effects without a default still fail loudly.
2. Const-generic receiver sizes: in `implement<..., const N> ... for
   vector[T,N]`, N binds to the receiver's runtime length at method entry
   (M/N to dims 0/1 for a matrix receiver), so scalar fallbacks like
   `for i in 1..N` see the real size.
3. For loops: `for x in iterable { ... }` lowers to the existing While
   machinery over __index_get; unsupported shapes fail loudly.

Plus supporting fixes surfaced by the example: `e as T` casts lower (numeric
conversion at runtime, pass-through otherwise), lambda names are qualified
per enclosing function (two `(a,b) -> ...` lambdas in different methods used
to collide and silently swap bodies), and a flat vector passed to a
matrix-annotated parameter (vector[vector[T,N],M]) is promoted to an Mx1
column so `mat.matmul(vec)` — the example's matrix-vector multiplication —
indexes strictly.
"""
from __future__ import annotations

import math
import os

import pytest

from metaxu.compiler.pipeline import build_context_from_source
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import InterpError, MirInterpreter, MxVector, UNIT

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def run_main(source: str):
    ctx = build_context_from_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    prints: list[str] = []
    interp.register_builtin(
        "print", lambda *a: (prints.append(" ".join(str(x) for x in a)), UNIT)[1])
    return interp.call("main", []), prints


# ---------------------------------------------------------------------------
# 1. Default effect handlers
# ---------------------------------------------------------------------------

def test_unhandled_perform_answers_declared_default():
    result, _ = run_main("""
        effect Cap {
            ask(x: int) -> int = 42;
        }
        fn main() -> int {
            perform Cap.ask(1)
        }
    """)
    assert result == 42


def test_default_expression_reads_the_op_arguments():
    result, _ = run_main("""
        effect Cap {
            ask(x: int) -> int = x + 1;
        }
        fn main() -> int {
            perform Cap.ask(41)
        }
    """)
    assert result == 42


def test_option_default_takes_scalar_branch_like_simd():
    # The exact SimdOp shape: an Option-returning capability defaulting to
    # None, whose caller matches and falls back.
    result, _ = run_main("""
        effect Cap {
            try_fast(x: int) -> Option<int> = None;
        }
        fn main() -> int {
            match perform Cap.try_fast(5) {
                Some(v) -> v,
                None -> 7
            }
        }
    """)
    assert result == 7


def test_installed_handler_overrides_default():
    result, _ = run_main("""
        effect Cap {
            ask(x: int) -> int = 0;
        }
        fn main() -> int {
            handle Cap with {
                ask(x) -> resume(x * 2)
            } in {
                perform Cap.ask(21)
            }
        }
    """)
    assert result == 42


def test_effect_without_default_still_fails_loudly():
    with pytest.raises(InterpError, match="No handler for effect"):
        run_main("""
            effect Cap {
                ask(x: int) -> int;
            }
            fn main() -> int {
                perform Cap.ask(1)
            }
        """)


# ---------------------------------------------------------------------------
# 2. For loops
# ---------------------------------------------------------------------------

def test_for_over_range():
    result, _ = run_main("""
        fn main() -> int {
            let mut s = 0;
            for i in 0..5 {
                s = s + i;
            }
            s
        }
    """)
    assert result == 10


def test_for_over_vector_elements():
    result, _ = run_main("""
        fn main() -> float {
            let v = vector[float,3](1.5, 2.5, 3.0);
            let mut s = 0.0;
            for x in v {
                s = s + x;
            }
            s
        }
    """)
    assert result == 7.0


# ---------------------------------------------------------------------------
# 3. Const-generic N bound to the receiver's runtime size
# ---------------------------------------------------------------------------

def test_const_generic_size_binds_to_receiver_length():
    result, _ = run_main("""
        implement<const N: int> vector[int,N] {
            fn size(self) -> int { N }
        }
        fn main() -> int {
            let v = vector[int,3](7, 8, 9);
            v.size()
        }
    """)
    assert result == 3


def test_scalar_fallback_loop_over_n_like_reduce():
    # The example's reduce fallback shape: result = self[0], then
    # `for i in 1..N`.
    result, _ = run_main("""
        implement<T, const N: int> vector[T,N] {
            fn total(self) -> int {
                let mut result = self[0];
                for i in 1..N {
                    result = result + self[i];
                }
                result
            }
        }
        fn main() -> int {
            let v = vector[int,4](1, 2, 3, 4);
            v.total()
        }
    """)
    assert result == 10


def test_matrix_receiver_binds_both_dims():
    result, _ = run_main("""
        implement<T, const M: int, const N: int> vector[vector[T,N],M] {
            fn dims(self) -> int { M * 10 + N }
        }
        fn main() -> int {
            let m = vector[vector[int,3],2](
                vector[int,3](1, 2, 3),
                vector[int,3](4, 5, 6)
            );
            m.dims()
        }
    """)
    assert result == 23


# ---------------------------------------------------------------------------
# Supporting fixes surfaced by the example
# ---------------------------------------------------------------------------

def test_cast_to_float_converts():
    # mean()'s shape: int size cast to float for the division.
    result, _ = run_main("""
        fn main() -> float {
            10 / 4 as float
        }
    """)
    assert result == 2.5


def test_lambda_names_do_not_collide_across_methods():
    # sum uses (a,b)->a+b, prod uses (a,b)->a*b. With per-function "lambdaN"
    # names these collided in the flat MIR namespace and sum silently
    # multiplied.
    result, _ = run_main("""
        implement<const N: int> vector[int,N] {
            fn combine_add(self) -> int { self.walk((a, b) -> a + b) }
            fn combine_mul(self) -> int { self.walk((a, b) -> a * b) }
            fn walk(self, f: fn(int,int) -> int) -> int {
                let mut r = self[0];
                for i in 1..N {
                    r = f(r, self[i]);
                }
                r
            }
        }
        fn main() -> int {
            let v = vector[int,3](2, 3, 4);
            v.combine_add() * 100 + v.combine_mul()
        }
    """)
    assert result == 9 * 100 + 24


def test_flat_vector_promotes_to_column_for_matrix_param():
    result, _ = run_main("""
        fn ncols(m: vector[vector[float,3],3]) -> int {
            len(m[0])
        }
        fn main() -> int {
            let v = vector[float,3](1.0, 2.0, 3.0);
            ncols(v)
        }
    """)
    assert result == 1


# ---------------------------------------------------------------------------
# The example end-to-end
# ---------------------------------------------------------------------------

def execute_example():
    src = open(os.path.join(REPO_ROOT, "examples", "06_vector_operations.mx")).read()
    return run_main(src)


def test_example_06_executes():
    result, _ = execute_example()
    assert result is UNIT


def test_example_06_computed_outputs():
    """Values that follow from the example's own data: v1 = (1,2,3,4) and
    v2 = (x*2 for x in 0..4) = (0,2,4,6), so v1+v2 = (1,4,7,10), v1.v2 = 40
    and rms = sqrt((1+4+9+16)/4) = sqrt(7.5)."""
    _, prints = execute_example()
    assert "Sum: vector[1.0, 4.0, 7.0, 10.0]" in prints
    assert "Dot product: 40.0" in prints
    rms_lines = [p for p in prints if p.startswith("RMS: ")]
    assert len(rms_lines) == 1
    assert math.isclose(float(rms_lines[0][len("RMS: "):]), math.sqrt(7.5))
