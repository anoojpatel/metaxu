"""Real positive/negative tests for the frozen AST borrow checker.

Covers:
- Locality (@local) escape via return -> error; exclave promotes legally
- Unannotated code stays permissive (defaults shared/global)
- Borrow conflicts (shared vs unique, double exclusive)
- Move invalidation (use-after-move, move-after-move, borrow-after-move)
- Linearity (once-value invoked twice)
- Region nesting
- Reference tracking (global holding a reference to a local)
- Structured BorrowError data (kind, variable, node_id, message)
- Pipeline enforcement (BorrowCheckError raised in strict mode)

Where the surface syntax cannot yet express a construct (mode annotations on
let-bindings, Borrow*/Move/Exclave nodes), tests hand-build frozen ASTs.
"""

import itertools

import pytest

import metaxu.metaxu_ast as fast
from metaxu.compiler.frozen_borrow_checker import BorrowCheckError, BorrowError
from metaxu.compiler.infer_tables import build_tables_from_frozen_via_simplesub
from metaxu.compiler.mutaxu_ast import AstNode, Span, build_frozen_ast_with_map
from metaxu.compiler.pipeline import run_pipeline, run_pipeline_from_source

SPAN = Span(file="<test>", start=0, end=0)


def make_node_factory():
    """Return a node() helper allocating unique node_ids for one tree."""
    counter = itertools.count(1)

    def node(kind, value=None, *children):
        return AstNode(
            node_id=next(counter), kind=kind, children=tuple(children), span=SPAN, value=value
        )

    return node


def borrow_errors_of(frozen):
    tables = build_tables_from_frozen_via_simplesub(frozen)
    return list(tables.constraints.get(-2, []))


# ---------------------------------------------------------------------------
# Locality (@local) enforcement
# ---------------------------------------------------------------------------

def test_local_variable_cannot_escape_via_return():
    """A @local binding returned from its function escapes its region -> error."""
    node = make_node_factory()
    fn = node("FunctionDeclaration", {"name": "f", "params": []},
        node("Block", None,
            node("LetBinding", {"name": "x", "mode": "local"}, node("Literal", 1)),
            node("ReturnStatement", None, node("Variable", {"name": "x"}))))

    errors = borrow_errors_of(fn)

    assert any(e.kind == "locality-escape" and e.variable == "x" for e in errors)


def test_local_parameter_cannot_escape_via_return():
    """A @local parameter returned to the caller escapes -> error."""
    node = make_node_factory()
    fn = node("FunctionDeclaration", {"name": "f", "params": ["p"]},
        node("Parameter", {"name": "p", "mode": "local"}),
        node("ReturnStatement", None, node("Variable", {"name": "p"})))

    errors = borrow_errors_of(fn)

    assert any(e.kind == "locality-escape" and e.variable == "p" for e in errors)


def test_unannotated_binding_returns_freely():
    """Unannotated code defaults to global/shared and stays permissive."""
    node = make_node_factory()
    fn = node("FunctionDeclaration", {"name": "f", "params": []},
        node("Block", None,
            node("LetBinding", {"name": "x"}, node("Literal", 1)),
            node("ReturnStatement", None, node("Variable", {"name": "x"}))))

    assert borrow_errors_of(fn) == []


def test_global_annotated_binding_returns_freely():
    """An explicit @global binding can escape its region."""
    node = make_node_factory()
    fn = node("FunctionDeclaration", {"name": "f", "params": []},
        node("Block", None,
            node("LetBinding", {"name": "x", "mode": "global"}, node("Literal", 1)),
            node("ReturnStatement", None, node("Variable", {"name": "x"}))))

    assert borrow_errors_of(fn) == []


def test_exclave_promotes_local_legally():
    """Returning `exclave x` copies the local to the caller's frame -> no error."""
    node = make_node_factory()
    fn = node("FunctionDeclaration", {"name": "f", "params": []},
        node("Block", None,
            node("LetBinding", {"name": "x", "mode": "local"}, node("Literal", 1)),
            node("ReturnStatement", None,
                node("ExclaveExpression", None, node("Variable", {"name": "x"})))))

    assert borrow_errors_of(fn) == []


def test_exclave_of_moved_value_is_error():
    """Exclave has nothing to copy after the value has been moved out."""
    node = make_node_factory()
    fn = node("FunctionDeclaration", {"name": "f", "params": []},
        node("Block", None,
            node("LetBinding", {"name": "x", "mode": "local"}, node("Literal", 1)),
            node("Move", {"variable": "x"}),
            node("ReturnStatement", None,
                node("ExclaveExpression", None, node("Variable", {"name": "x"})))))

    errors = borrow_errors_of(fn)

    assert any(e.kind == "use-after-move" and e.variable == "x" for e in errors)


def test_region_nesting_ok():
    """Locals used inside their own (possibly nested) regions are fine."""
    node = make_node_factory()
    fn = node("FunctionDeclaration", {"name": "f", "params": []},
        node("Block", None,
            node("LetBinding", {"name": "outer", "mode": "local"}, node("Literal", 1)),
            node("Block", None,
                node("LetBinding", {"name": "inner", "mode": "local"}, node("Literal", 2)),
                # Using an outer local inside a nested region is not an escape.
                node("FunctionCall", {"name": "g"},
                    node("Variable", {"name": "outer"}),
                    node("Variable", {"name": "inner"})))))

    errors = borrow_errors_of(fn)

    assert not [e for e in errors if e.kind == "locality-escape"]


# ---------------------------------------------------------------------------
# Suspend effects vs @local values
# ---------------------------------------------------------------------------

def _effect_program(node, effect_class, arg_mode):
    return node("Block", None,
        node("EffectDeclaration", {"name": "E", "effect_class": effect_class}),
        node("FunctionDeclaration", {"name": "eff_fn", "params": ["x"], "performs": ["E"]},
            node("Parameter", {"name": "x", "mode": None}),
            node("ReturnStatement", None, node("Variable", {"name": "x"}))),
        node("FunctionDeclaration", {"name": "caller", "params": []},
            node("Block", None,
                node("LetBinding", {"name": "v", "mode": arg_mode}, node("Literal", 42)),
                node("FunctionCall", {"name": "eff_fn"}, node("Variable", {"name": "v"})))))


def test_local_passed_to_suspend_effect_is_error():
    node = make_node_factory()
    prog = _effect_program(node, "suspend", "local")

    errors = borrow_errors_of(prog)

    assert any(e.kind == "suspend-local" and e.variable == "v" for e in errors)


def test_local_passed_to_stack_effect_is_allowed():
    node = make_node_factory()
    prog = _effect_program(node, "stack", "local")

    assert borrow_errors_of(prog) == []


def test_global_passed_to_suspend_effect_is_allowed():
    node = make_node_factory()
    prog = _effect_program(node, "suspend", "global")

    assert borrow_errors_of(prog) == []


# ---------------------------------------------------------------------------
# Borrow conflicts and move invalidation (via the real freezing path)
# ---------------------------------------------------------------------------

def test_two_exclusive_borrows_conflict():
    node = make_node_factory()
    prog = node("Block", None,
        node("LetBinding", {"name": "x"}, node("Literal", 1)),
        node("BorrowExclusive", {"variable": "x"}),
        node("BorrowExclusive", {"variable": "x"}))

    errors = borrow_errors_of(prog)

    assert any(e.kind == "borrow-conflict" and e.variable == "x" for e in errors)


def test_unique_borrow_while_shared_borrowed_conflicts():
    parsed = fast.Block([
        fast.LetStatement([fast.LetBinding("x", fast.Literal(1))]),
        fast.BorrowShared("x"),
        fast.BorrowUnique("x"),
    ])
    frozen, _ = build_frozen_ast_with_map(parsed)

    errors = borrow_errors_of(frozen)

    assert any(e.kind == "borrow-conflict" and e.variable == "x" for e in errors)


def test_use_after_move_is_error():
    parsed = fast.Block([
        fast.LetStatement([fast.LetBinding("x", fast.Literal(1))]),
        fast.Move("x"),
        fast.Variable("x"),
    ])
    frozen, _ = build_frozen_ast_with_map(parsed)

    errors = borrow_errors_of(frozen)

    assert any(e.kind == "use-after-move" and e.variable == "x" for e in errors)


def test_move_after_move_is_error():
    parsed = fast.Block([
        fast.LetStatement([fast.LetBinding("x", fast.Literal(1))]),
        fast.Move("x"),
        fast.Move("x"),
    ])
    frozen, _ = build_frozen_ast_with_map(parsed)

    errors = borrow_errors_of(frozen)

    assert any(e.kind == "move-after-move" and e.variable == "x" for e in errors)


def test_borrow_after_move_is_error():
    parsed = fast.Block([
        fast.LetStatement([fast.LetBinding("x", fast.Literal(1))]),
        fast.Move("x"),
        fast.BorrowShared("x"),
    ])
    frozen, _ = build_frozen_ast_with_map(parsed)

    errors = borrow_errors_of(frozen)

    assert any(e.kind == "borrow-after-move" and e.variable == "x" for e in errors)


def test_single_move_of_live_value_is_ok():
    parsed = fast.Block([
        fast.LetStatement([fast.LetBinding("x", fast.Literal(1))]),
        fast.Move("x"),
    ])
    frozen, _ = build_frozen_ast_with_map(parsed)

    assert borrow_errors_of(frozen) == []


# ---------------------------------------------------------------------------
# Linearity
# ---------------------------------------------------------------------------

def test_once_value_used_twice_is_error():
    node = make_node_factory()
    prog = node("Block", None,
        node("LetBinding", {"name": "f"},
            node("LambdaExpression", {"params": [], "captures": {}, "linearity": "once"},
                node("Literal", 1))),
        node("FunctionCall", {"name": "f"}),
        node("FunctionCall", {"name": "f"}))

    errors = borrow_errors_of(prog)

    assert any(e.kind == "linearity" and e.variable == "f" for e in errors)


def test_once_value_used_once_is_ok():
    node = make_node_factory()
    prog = node("Block", None,
        node("LetBinding", {"name": "f"},
            node("LambdaExpression", {"params": [], "captures": {}, "linearity": "once"},
                node("Literal", 1))),
        node("FunctionCall", {"name": "f"}))

    assert borrow_errors_of(prog) == []


def test_many_value_used_twice_is_ok():
    node = make_node_factory()
    prog = node("Block", None,
        node("LetBinding", {"name": "f"},
            node("LambdaExpression", {"params": [], "captures": {}, "linearity": "many"},
                node("Literal", 1))),
        node("FunctionCall", {"name": "f"}),
        node("FunctionCall", {"name": "f"}))

    assert borrow_errors_of(prog) == []


# ---------------------------------------------------------------------------
# Reference tracking
# ---------------------------------------------------------------------------

def test_global_cannot_hold_reference_to_local():
    node = make_node_factory()
    fn = node("FunctionDeclaration", {"name": "f", "params": []},
        node("Block", None,
            node("LetBinding", {"name": "l", "mode": "local"}, node("Literal", 1)),
            node("LetBinding", {"name": "g", "mode": "global"},
                node("BorrowShared", {"variable": "l"}))))

    errors = borrow_errors_of(fn)

    assert any(e.kind == "dangling-reference" and e.variable == "l" for e in errors)


def test_local_can_hold_reference_to_global():
    node = make_node_factory()
    fn = node("FunctionDeclaration", {"name": "f", "params": []},
        node("Block", None,
            node("LetBinding", {"name": "g", "mode": "global"}, node("Literal", 1)),
            node("LetBinding", {"name": "l", "mode": "local"},
                node("BorrowShared", {"variable": "g"}))))

    errors = borrow_errors_of(fn)

    assert not [e for e in errors if e.kind == "dangling-reference"]


# ---------------------------------------------------------------------------
# Structured error data
# ---------------------------------------------------------------------------

def test_borrow_errors_are_structured():
    parsed = fast.Block([
        fast.LetStatement([fast.LetBinding("x", fast.Literal(1))]),
        fast.Move("x"),
        fast.Variable("x"),
    ])
    frozen, _ = build_frozen_ast_with_map(parsed)

    errors = borrow_errors_of(frozen)

    assert errors, "expected at least one borrow error"
    err = errors[0]
    assert isinstance(err, BorrowError)
    assert err.kind == "use-after-move"
    assert err.variable == "x"
    assert isinstance(err.node_id, int) and err.node_id > 0
    assert str(err) == err.message
    assert "moved" in str(err)


# ---------------------------------------------------------------------------
# Pipeline enforcement
# ---------------------------------------------------------------------------

def _bad_program_tables():
    parsed = fast.Block([
        fast.LetStatement([fast.LetBinding("x", fast.Literal(1))]),
        fast.Move("x"),
        fast.Variable("x"),
    ])
    frozen, id_map = build_frozen_ast_with_map(parsed)
    tables = build_tables_from_frozen_via_simplesub(frozen)
    return frozen, id_map, tables


def test_pipeline_raises_borrow_check_error_on_bad_program():
    frozen, id_map, tables = _bad_program_tables()

    with pytest.raises(BorrowCheckError) as excinfo:
        run_pipeline(frozen, tables, id_map=id_map)

    errors = excinfo.value.errors
    assert any(e.kind == "use-after-move" and e.variable == "x" for e in errors)


def test_pipeline_strict_false_opts_out_of_enforcement():
    frozen, id_map, tables = _bad_program_tables()

    hir_txt, mir_txt, clif_txt = run_pipeline(frozen, tables, id_map=id_map, strict=False)

    assert isinstance(hir_txt, str) and hir_txt
    assert isinstance(mir_txt, str)
    assert isinstance(clif_txt, str)


def test_pipeline_does_not_raise_on_hello_world():
    ast_json, hir_txt, mir_txt, clif_txt = run_pipeline_from_source('print("Hello from Metaxu!")')

    assert isinstance(hir_txt, str) and hir_txt
    assert isinstance(mir_txt, str) and mir_txt
