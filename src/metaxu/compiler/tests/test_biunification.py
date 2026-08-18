"""Tests for the biunification (algebraic subtyping) engine.

Engine-level tests drive metaxu.simplesub.Biunifier directly: they cover
the constraint-propagation core (bound recording, transitivity,
contravariance, termination on recursive graphs), let-polymorphism
(levels, extrusion, instantiation) and coalescence/simplification into
principal types.

Pipeline-level tests go through parsed source (repo convention): the
SimpleSubFacade exposes principal types lazily via `principal_type_of`,
and — the hard behavioral constraint — programs that failed to compile
before biunification still fail with the same diagnostic kinds, and
querying principal types never changes what compiles.
"""

import pytest

from metaxu.simplesub import Biunifier, BiunifyError, PFun, PVar
from metaxu.type_defs import CompactType, next_id

from metaxu.compiler.pipeline import build_context_from_source, run_pipeline_ctx
from metaxu.compiler.frozen_borrow_checker import TypeCheckError


def prim(name: str) -> CompactType:
    return CompactType(id=next_id(), kind='primitive', name=name)


def fn(params, ret) -> CompactType:
    return CompactType(id=next_id(), kind='function',
                       param_types=list(params), return_type=ret)


def nodes_by_kind(ctx):
    out = []

    def walk(n):
        out.append(n)
        for c in n.children:
            walk(c)

    walk(ctx.frozen_root)
    return out


def named(nodes, kind, name):
    for n in nodes:
        if n.kind == kind and isinstance(n.value, dict) and n.value.get("name") == name:
            return n
    raise AssertionError(f"no {kind} named {name}")


# ---------------------------------------------------------------------------
# Engine level
# ---------------------------------------------------------------------------

def test_identity_coalesces_to_polymorphic_arrow():
    b = Biunifier()
    p, r = b.fresh_var(), b.fresh_var()
    b.constrain(p, r)  # the parameter flows to the return
    ident = fn([p], r)
    assert b.principal_type(ident) == "'a -> 'a"
    assert b.errors == []


def test_twice_applied_param_produces_right_bounds():
    """A parameter used as Int and also returned: the engine must record
    Int as an UPPER bound of the parameter (usage requirement), keep the
    flow to the result, and coalesce to (Int ∧ 'a) -> 'a."""
    b = Biunifier()
    x, out = b.fresh_var(), b.fresh_var()
    b.constrain(x, prim('Int'))   # x is used where an Int is required
    b.constrain(x, out)           # x is also returned
    g = fn([x], out)

    uppers = [u.find() for u in b.info(x).upper]
    assert any(u.kind == 'primitive' and u.name == 'Int' for u in uppers), \
        "Int must be recorded as an upper bound of the parameter"
    assert b.principal_type(g) == "(Int ∧ 'a) -> 'a"
    assert b.errors == []


def test_multiple_bounds_per_var():
    """TypeBounds can hold at most one upper/lower bound; the engine's side
    tables must hold several."""
    b = Biunifier()
    v = b.fresh_var()
    b.constrain(v, prim('Int'))
    b.constrain(v, prim('String'))
    assert len(b.info(v).upper) == 2
    # Negative position: intersection of the upper bounds.
    rendered = b.principal_type(v, positive=False)
    assert rendered == "Int ∧ String"
    # No lower bounds exist, so no Int/String clash is reported.
    assert b.errors == []


def test_lower_bound_propagates_through_new_upper_bound():
    """The biunification core: when a var already has a lower bound and
    gains an upper bound, the lower bound is constrained against it."""
    b = Biunifier()
    v = b.fresh_var()
    b.constrain(prim('Int'), v)      # Int <: v
    b.constrain(v, prim('String'))   # v <: String  ->  Int <: String fires
    assert len(b.errors) == 1
    err = b.errors[0]
    assert err.kind == "type-conflict"
    assert "Int" in err.message and "String" in err.message


def test_transitivity_through_var_chain():
    """Int <: a, a <: b, b <: String must surface the Int/String clash even
    though no single constraint mentions both primitives."""
    b = Biunifier()
    a, v = b.fresh_var(), b.fresh_var()
    b.constrain(prim('Int'), a)
    b.constrain(a, v)
    b.constrain(v, prim('String'))
    assert any("Int" in e.message and "String" in e.message for e in b.errors)


def test_branch_join_union_collapses_when_arms_agree():
    b = Biunifier()
    res = b.fresh_var()
    b.constrain(prim('Int'), res)
    b.constrain(prim('Int'), res)
    assert b.principal_type(res) == "Int"


def test_branch_join_union_when_arms_differ():
    b = Biunifier()
    res = b.fresh_var()
    b.constrain(prim('Int'), res)
    b.constrain(prim('String'), res)
    assert b.principal_type(res) == "Int ∨ String"
    assert b.errors == []  # a union is fine; only the class checker forbids it


def test_contravariance_parameter_position_flips():
    """callee <: (Int) -> out decomposes contravariantly: Int becomes a
    LOWER bound of the callee's parameter, and the callee's return flows
    into out as a lower bound."""
    b = Biunifier()
    p, r = b.fresh_var(), b.fresh_var()
    callee = fn([p], r)
    out = b.fresh_var()
    b.constrain(callee, fn([prim('Int')], out))

    lowers = [t.find() for t in b.info(p).lower]
    assert any(t.kind == 'primitive' and t.name == 'Int' for t in lowers), \
        "argument type must arrive as a LOWER bound of the parameter (contravariance)"
    assert not b.info(p).upper, "contravariance must not add an upper bound here"
    # The covariant return flows the other way: `out` becomes an upper
    # bound of the callee's return var (var-on-the-left recording).
    r_uppers = [t.find() for t in b.info(r).upper]
    assert any(t.kind == 'var' and t.id == out.find().id for t in r_uppers)


def test_function_arity_mismatch_reports_conflict():
    b = Biunifier()
    b.constrain(fn([prim('Int')], prim('Int')),
                fn([prim('Int'), prim('Int')], prim('Int')))
    assert len(b.errors) == 1
    assert b.errors[0].kind == "type-conflict"


def test_primitive_lattice_is_flat():
    """No Int <: Float — widening the lattice would change which programs
    compile."""
    b = Biunifier()
    b.constrain(prim('Int'), prim('Float'))
    assert len(b.errors) == 1
    assert "Float" in b.errors[0].message and "Int" in b.errors[0].message


def test_recursive_constraint_terminates_and_yields_mu_type():
    b = Biunifier()
    f = b.fresh_var()
    b.constrain(fn([prim('Int')], f), f)  # (Int -> f) <: f
    assert b.principal_type(f) == "μt. Int -> t"
    assert b.errors == []


def test_mutually_recursive_constraints_terminate():
    b = Biunifier()
    f, g = b.fresh_var(), b.fresh_var()
    b.constrain(fn([prim('Int')], g), f)
    b.constrain(fn([prim('Int')], f), g)
    b.constrain(f, g)
    b.constrain(g, f)
    rendered = b.principal_type(f)  # must not hang or blow the stack
    assert isinstance(rendered, str) and rendered


def test_extrusion_does_not_leak_inner_vars():
    """Constraining an outer-level var against a type containing an
    inner-level var must extrude: the recorded bound contains only vars at
    the outer level, freshly created and linked back to the original."""
    b = Biunifier()
    x = b.fresh_var()             # level 0
    b.enter_level()
    y = b.fresh_var()             # level 1
    b.constrain(x, fn([y], y))
    b.exit_level()

    assert len(b.info(x).upper) == 1
    bound = b.info(x).upper[0]
    bound_vars = [t.find() for t in (bound.param_types or []) + [bound.return_type]]
    for t in bound_vars:
        assert t.kind == 'var'
        assert t.id != y.id, "the inner var itself must not appear in the bound"
        assert b.info(t).level == 0, "extruded copies must live at the outer level"
    # ... and the copies stay linked to y so information keeps flowing.
    linked = {t.find().id
              for t in b.info(y).lower + b.info(y).upper}
    assert linked & {t.id for t in bound_vars}


def test_let_polymorphism_instantiates_per_use_site():
    b = Biunifier()
    b.enter_level()
    p, r = b.fresh_var(), b.fresh_var()
    b.constrain(p, r)
    ident = fn([p], r)
    b.exit_level()
    scheme = b.generalize(ident)

    i1, res1 = b.instantiate(scheme), b.fresh_var()
    b.constrain(i1, fn([prim('Int')], res1))
    i2, res2 = b.instantiate(scheme), b.fresh_var()
    b.constrain(i2, fn([prim('String')], res2))

    assert b.principal_type(res1) == "Int"
    assert b.principal_type(res2) == "String"
    # The scheme body itself stays unpolluted by either use site.
    assert b.principal_type(ident) == "'a -> 'a"
    assert b.errors == []


def test_error_shape_matches_type_conflict_reporting():
    b = Biunifier()
    v = b.fresh_var()
    b.constrain(prim('Int'), v)
    b.constrain(v, prim('String'), node_id=42)
    err = b.errors[0]
    assert isinstance(err, BiunifyError)
    assert err.kind == "type-conflict"
    assert err.node_id == 42
    assert err.message.startswith("type mismatch: one value is required to be")


# ---------------------------------------------------------------------------
# Pipeline level (through parsed source)
# ---------------------------------------------------------------------------

def test_pipeline_principal_types_for_concrete_program():
    src = (
        "fn add_one(x: Int) -> Int {\n"
        "    return x + 1\n"
        "}\n"
        "\n"
        "fn main() -> Int {\n"
        "    let y = add_one(41)\n"
        "    return y\n"
        "}\n"
    )
    ctx = build_context_from_source(src)
    facade = ctx.tables.facade
    nodes = nodes_by_kind(ctx)

    assert facade.principal_type_of(
        named(nodes, "FunctionDeclaration", "add_one").node_id) == "Int -> Int"
    assert facade.principal_type_of(
        named(nodes, "FunctionDeclaration", "main").node_id) == "() -> Int"
    assert facade.principal_type_of(
        named(nodes, "LetBinding", "y").node_id) == "Int"
    assert facade.biunify_errors() == []


def test_pipeline_polymorphic_function_used_at_two_types():
    """The pipeline's constraint stream is monomorphic (one shared type per
    function, no generalization), so a function used at Int and String
    infers a union/intersection type rather than per-site instantiation —
    and, crucially, still compiles exactly as before."""
    src = (
        "fn identity(x) {\n"
        "    return x\n"
        "}\n"
        "\n"
        "fn main() -> Int {\n"
        "    let a = identity(1)\n"
        "    let s = identity(\"hello\")\n"
        "    return a\n"
        "}\n"
    )
    ctx = build_context_from_source(src)
    run_pipeline_ctx(ctx)  # must still compile
    facade = ctx.tables.facade
    nodes = nodes_by_kind(ctx)

    ident_ty = facade.principal_type_of(
        named(nodes, "FunctionDeclaration", "identity").node_id)
    # Both use-site types flow into the single shared function type.
    assert ident_ty == "(Int ∧ String) -> (Int ∨ String)"
    a_ty = facade.principal_type_of(named(nodes, "LetBinding", "a").node_id)
    assert a_ty == "Int ∨ String"


def test_pipeline_recursive_function_coalesces_without_hanging():
    src = (
        "fn count(n: Int) -> Int {\n"
        "    return count(n - 1)\n"
        "}\n"
        "fn main() -> Int {\n"
        "    return 0\n"
        "}\n"
    )
    ctx = build_context_from_source(src)
    facade = ctx.tables.facade
    nodes = nodes_by_kind(ctx)
    rendered = facade.principal_type_of(
        named(nodes, "FunctionDeclaration", "count").node_id)
    assert isinstance(rendered, str) and "->" in rendered


def test_pipeline_int_plus_string_still_fails_with_type_conflict():
    src = 'fn main() -> Int {\n    let x = 1 + "a"\n    return 0\n}\n'
    ctx = build_context_from_source(src)
    with pytest.raises(TypeCheckError) as exc:
        run_pipeline_ctx(ctx)
    assert "type mismatch" in str(exc.value)
    # The biunifier independently sees the same clash (advisory).
    assert any("Int" in e.message and "String" in e.message
               for e in ctx.tables.facade.biunify_errors())


def test_pipeline_non_bool_condition_still_fails():
    src = 'fn main() -> Int {\n    if 1 { return 2 }\n    return 0\n}\n'
    ctx = build_context_from_source(src)
    with pytest.raises(TypeCheckError) as exc:
        run_pipeline_ctx(ctx)
    assert "type mismatch" in str(exc.value)


def test_principal_type_queries_do_not_change_compilation():
    """Querying principal types is advisory: the MIR produced after
    querying must be identical to the MIR of an untouched context."""
    src = (
        "fn main() -> Int {\n"
        "    let y = 1 + 2\n"
        "    return y\n"
        "}\n"
    )
    ctx_plain = build_context_from_source(src)
    _, _, mir_plain, _ = run_pipeline_ctx(ctx_plain)

    ctx_queried = build_context_from_source(src)
    ctx_queried.tables.facade.principal_types()  # force the whole table
    ctx_queried.tables.facade.biunify_errors()
    _, _, mir_queried, _ = run_pipeline_ctx(ctx_queried)

    assert mir_plain == mir_queried
