"""End-to-end tests for lambda closures: capture, binding, and calling.

Pins down two defects found by adversarial review:

1. Lambda lowering captured values keyed by SOURCE name while the lambda body
   was compiled against the enclosing env's SLOT names (source `x` lives in
   slot `x_2`), so the body's lookup missed the closure env at runtime
   (InterpError "Unbound variable 'x_2'").
2. Calling a let-bound closure (`let g = fn(y) ...; g(2)`) lowered to a plain
   ("call", "g") whose callee only resolved against named MirFuncs, never
   against a local bound to an MxClosure (InterpError "Unknown callee: 'g'").

Every test goes through the real front end:
parse -> desugar -> freeze -> infer -> HIR -> MIR -> interpreter.
"""
from __future__ import annotations

from metaxu.compiler.pipeline import build_context_from_source
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import MirInterpreter


def call(source: str, fn: str, args: list):
    ctx = build_context_from_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    return interp.call(fn, args)


# ---------------------------------------------------------------------------
# Capture-and-call: the exact scenario from the review
# ---------------------------------------------------------------------------

def test_closure_captures_let_bound_var_and_is_callable():
    src = """
fn main() -> int {
    let x = 5;
    let g = fn(y: int) -> x + y;
    g(2)
}
"""
    assert call(src, "main", []) == 7


# ---------------------------------------------------------------------------
# Two captures
# ---------------------------------------------------------------------------

def test_closure_with_two_captures():
    src = """
fn main() -> int {
    let a = 10;
    let b = 3;
    let h = fn(y: int) -> a - b + y;
    h(1)
}
"""
    assert call(src, "main", []) == 8


# ---------------------------------------------------------------------------
# Closure passed as an argument and called through the parameter name
# ---------------------------------------------------------------------------

def test_closure_passed_as_argument_and_called():
    src = """
fn apply(f: fn(int) -> int, v: int) -> int {
    f(v)
}

fn main() -> int {
    let x = 100;
    let g = fn(y: int) -> x + y;
    apply(g, 23)
}
"""
    assert call(src, "main", []) == 123


# ---------------------------------------------------------------------------
# No-capture lambda (regression: plain closures must keep working)
# ---------------------------------------------------------------------------

def test_no_capture_lambda_still_works():
    src = """
fn main() -> int {
    let d = fn(y: int) -> y * 2;
    d(21)
}
"""
    assert call(src, "main", []) == 42


# ---------------------------------------------------------------------------
# Direct function calls and builtins keep resolving as before
# ---------------------------------------------------------------------------

def test_direct_function_call_still_resolves_by_name():
    src = """
fn add_one(n: int) -> int {
    n + 1
}

fn main() -> int {
    add_one(41)
}
"""
    assert call(src, "main", []) == 42


# ---------------------------------------------------------------------------
# Lambda body forms (grammar): all four spellings, and the shapes they must
# NOT capture (struct-literal bodies, typed returns)
# ---------------------------------------------------------------------------

def test_arrow_block_lambda_body():
    """`fn(x) -> { stmts; tail }` — arrow with a block body; the block's tail
    expression is the result, exactly like a function body."""
    src = """
fn main() -> int {
    let f = fn(x: int) -> { let y = x * 2; y + 1 };
    f(20)
}
"""
    assert call(src, "main", []) == 41


def test_arrow_expression_lambda_body_still_works():
    src = "fn main() -> int { let f = fn(x: int) -> x + 1; f(41) }"
    assert call(src, "main", []) == 42


def test_bare_block_lambda_body_still_works():
    src = "fn main() -> int { let f = fn(x: int) { x + 5 }; f(37) }"
    assert call(src, "main", []) == 42


def test_struct_literal_lambda_body_not_parsed_as_block():
    """`-> P { x: v }` is a struct literal (IDENTIFIER LBRACE_STRUCT), not an
    arrow-block body — the new production must not swallow it."""
    src = """
struct P { x: int }

fn main() -> int {
    let f = fn(v: int) -> P { x: v };
    f(42).x
}
"""
    assert call(src, "main", []) == 42


def test_typed_return_with_block_body_still_works():
    src = "fn main() -> int { let f = fn(x: int) -> int { x * 2 }; f(21) }"
    assert call(src, "main", []) == 42
