"""Tests for effect-class handling (stack vs suspend) and effect safety.

Per docs/effects/continuation_design.md, V1 has exactly two effect classes:
- stack:   continuation cannot escape the current dynamic extent; behaves
           like a plain call and does NOT mark the function as suspending.
- suspend: continuation may escape and resume later; the function suspends
           and @local values may not cross the suspension point.
Effects with no declared class are conservatively treated as suspend-class.
"""

import itertools

import metaxu.metaxu_ast as fast
from metaxu.compiler.infer_tables import build_tables_from_frozen_via_simplesub
from metaxu.compiler.mutaxu_ast import AstNode, Span, build_frozen_ast_with_map

SPAN = Span(file="<test>", start=0, end=0)


def make_node_factory():
    counter = itertools.count(1)

    def node(kind, value=None, *children):
        return AstNode(
            node_id=next(counter), kind=kind, children=tuple(children), span=SPAN, value=value
        )

    return node


def test_effect_class_is_frozen_into_ast():
    """EffectDeclaration.effect_class survives freezing into the value payload."""
    effect_decl = fast.EffectDeclaration(
        "Async", [], fast.EffectOperation("yield", [], None), effect_class="suspend"
    )
    frozen, _ = build_frozen_ast_with_map(fast.Block([effect_decl]))

    decl = next(n for n in _walk(frozen) if n.kind == "EffectDeclaration")
    assert decl.value == {"name": "Async", "effect_class": "suspend"}


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


def _program_with_effect(node, effect_class, declare_effect=True):
    """A function performing effect E, with E optionally declared with a class."""
    children = []
    if declare_effect:
        children.append(node("EffectDeclaration", {"name": "E", "effect_class": effect_class}))
    fn = node("FunctionDeclaration", {"name": "eff_fn", "params": ["x"], "performs": ["E"]},
        node("Parameter", {"name": "x", "mode": None}),
        node("ReturnStatement", None, node("Variable", {"name": "x"})))
    children.append(fn)
    return node("Block", None, *children), fn


def test_suspend_class_effect_marks_function_suspending():
    node = make_node_factory()
    prog, fn = _program_with_effect(node, "suspend")

    tables = build_tables_from_frozen_via_simplesub(prog)

    assert fn.node_id in tables.suspends
    assert "E" in tables.effects.get(fn.node_id, set())


def test_stack_class_effect_does_not_mark_function_suspending():
    node = make_node_factory()
    prog, fn = _program_with_effect(node, "stack")

    tables = build_tables_from_frozen_via_simplesub(prog)

    assert fn.node_id not in tables.suspends
    # The effect itself is still tracked; only the suspend classification changes.
    assert "E" in tables.effects.get(fn.node_id, set())


def test_undeclared_effect_class_defaults_to_suspend():
    """Effects with no declared class are conservatively suspend-class."""
    node = make_node_factory()
    prog, fn = _program_with_effect(node, None, declare_effect=False)

    tables = build_tables_from_frozen_via_simplesub(prog)

    assert fn.node_id in tables.suspends


def test_local_passed_to_suspend_effect_is_caught():
    """A @local value crossing a suspension point is an error."""
    node = make_node_factory()
    prog = node("Block", None,
        node("EffectDeclaration", {"name": "Async", "effect_class": "suspend"}),
        node("FunctionDeclaration", {"name": "suspend_fn", "params": ["x"], "performs": ["Async"]},
            node("Parameter", {"name": "x", "mode": None}),
            node("ReturnStatement", None, node("Variable", {"name": "x"}))),
        node("FunctionDeclaration", {"name": "test_escape", "params": []},
            node("Block", None,
                node("LetBinding", {"name": "local_v", "mode": "local"}, node("Literal", 42)),
                node("FunctionCall", {"name": "suspend_fn"},
                    node("Variable", {"name": "local_v"})))))

    tables = build_tables_from_frozen_via_simplesub(prog)
    errors = list(tables.constraints.get(-2, []))

    assert any(e.kind == "suspend-local" and e.variable == "local_v" for e in errors)


def test_local_passed_to_stack_effect_is_allowed():
    """Stack effects behave like plain calls: locals may be passed."""
    node = make_node_factory()
    prog = node("Block", None,
        node("EffectDeclaration", {"name": "Console", "effect_class": "stack"}),
        node("FunctionDeclaration", {"name": "stack_fn", "params": ["x"], "performs": ["Console"]},
            node("Parameter", {"name": "x", "mode": None}),
            node("ReturnStatement", None, node("Variable", {"name": "x"}))),
        node("FunctionDeclaration", {"name": "test_stack", "params": []},
            node("Block", None,
                node("LetBinding", {"name": "local_v", "mode": "local"}, node("Literal", 42)),
                node("FunctionCall", {"name": "stack_fn"},
                    node("Variable", {"name": "local_v"})))))

    tables = build_tables_from_frozen_via_simplesub(prog)
    errors = list(tables.constraints.get(-2, []))

    assert errors == []
