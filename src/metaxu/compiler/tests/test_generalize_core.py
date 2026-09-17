"""Core tests for batch-mode let-generalization (compiler/generalize.py).

These drive the facade directly, the way the emitter would around a
lambda's walk, so they run on any tree: no parser, no pipeline.
"""
from __future__ import annotations

from metaxu.type_defs import CompactType, TypeConstructor, next_id
from metaxu.compiler.simplesub_adapter import SimpleSubFacade
from metaxu.compiler.generalize import Scheme, instantiate


def prim(name: str) -> CompactType:
    return CompactType(id=next_id(), kind="primitive", name=name)


def identity_lambda(facade: SimpleSubFacade):
    """Emit what the emitter emits for `fn(x) -> x` inside a recording:
    a parameter variable, a body use unified with it, the use flowing
    into the return type."""
    with facade.record_scheme() as rec:
        x = CompactType.fresh_var()
        ret = CompactType.fresh_var()
        use = CompactType.fresh_var()
        facade.add_class_constraint("Param", [x], 1)
        facade.add_unify(use, x)
        facade.add_subtype(use, ret)
        fn = CompactType(id=next_id(), kind="function", param_types=[x],
                         return_type=ret, linearity="many")
        facade.add_function_type(fn, [x], ret, "many", 2)
    return rec.finish(fn), fn


def test_recorder_captures_generic_ids_and_constraints():
    facade = SimpleSubFacade({})
    outer = CompactType.fresh_var()          # allocated before the window
    scheme, fn = identity_lambda(facade)
    assert fn.param_types[0].id in scheme.generic_ids
    assert fn.return_type.id in scheme.generic_ids
    assert outer.id not in scheme.generic_ids
    tags = [c[0] for c in scheme.constraints]
    assert tags == ["class", "unify", "subtype", "function"]


def test_two_instances_resolve_to_two_primitives():
    facade = SimpleSubFacade({})
    facade.surface_solver_errors = True
    scheme, fn = identity_lambda(facade)

    inst_int = instantiate(scheme, facade, use_node_id=10)
    inst_str = instantiate(scheme, facade, use_node_id=11)
    assert inst_int is not inst_str
    assert inst_int.param_types[0] is not fn.param_types[0]

    facade.add_unify(inst_int.param_types[0], prim("Int"))
    facade.add_unify(inst_str.param_types[0], prim("String"))
    facade.solve()

    assert inst_int.return_type.find().name == "Int"
    assert inst_str.return_type.find().name == "String"
    assert fn.return_type.find().kind == "var", "the original stays generic"
    assert not [e for e in facade.errors if "Cannot unify" in e]


def test_without_instantiation_the_same_uses_conflict():
    facade = SimpleSubFacade({})
    facade.surface_solver_errors = True
    scheme, fn = identity_lambda(facade)

    # monomorphic use: both call sites talk to the one parameter
    facade.add_unify(fn.param_types[0], prim("Int"))
    facade.add_unify(fn.param_types[0], prim("String"))
    facade.solve()

    assert any("Cannot unify" in e for e in facade.errors)


def test_captured_outer_variable_is_shared_across_instances():
    facade = SimpleSubFacade({})
    facade.surface_solver_errors = True
    k = CompactType.fresh_var()              # `let k = ...` outside the lambda
    with facade.record_scheme() as rec:
        x = CompactType.fresh_var()
        ret = CompactType.fresh_var()
        facade.add_unify(x, k)               # body: x + k
        facade.add_subtype(x, ret)
        fn = CompactType(id=next_id(), kind="function", param_types=[x],
                         return_type=ret, linearity="many")
    scheme = rec.finish(fn)
    assert k.id not in scheme.generic_ids

    a = instantiate(scheme, facade)
    b = instantiate(scheme, facade)
    facade.add_unify(a.param_types[0], prim("Int"))
    facade.add_unify(b.param_types[0], prim("String"))
    facade.solve()

    # k took Int through the first instance, so the second collides
    assert k.find().name == "Int"
    assert any("Cannot unify" in e for e in facade.errors)


def test_instances_alias_the_original_for_call_counting():
    facade = SimpleSubFacade({})
    with facade.record_scheme() as rec:
        x = CompactType.fresh_var()
        ret = CompactType.fresh_var()
        fn = CompactType(id=next_id(), kind="function", param_types=[x],
                         return_type=ret, linearity="once")
        facade.add_function_type(fn, [x], ret, "once", 3)
        facade.add_linearity(fn, "once", 3)
    scheme = rec.finish(fn)

    a = instantiate(scheme, facade)
    b = instantiate(scheme, facade)
    facade.add_call(a, [prim("Int")], CompactType.fresh_var(), 20)
    facade.add_call(b, [prim("Int")], CompactType.fresh_var(), 21)
    facade.solve()

    assert any("Once callable invoked more than once" in e
               for e in facade.errors)


def test_constructor_types_are_copied_structurally():
    facade = SimpleSubFacade({})
    with facade.record_scheme() as rec:
        t = CompactType.fresh_var()
        box = CompactType(id=next_id(), kind="constructor",
                          constructor=TypeConstructor("Box", 1),
                          type_args=[t], name="Box")
        ret = CompactType.fresh_var()
        facade.add_unify(ret, box)
        fn = CompactType(id=next_id(), kind="function", param_types=[t],
                         return_type=ret, linearity="many")
    scheme = rec.finish(fn)

    inst = instantiate(scheme, facade)
    facade.add_unify(inst.param_types[0], prim("Int"))
    facade.solve()

    resolved = inst.return_type.find()
    assert resolved.kind == "constructor" and resolved.name == "Box"
    assert resolved.type_args[0].find().name == "Int"
    assert t.find().kind == "var"
