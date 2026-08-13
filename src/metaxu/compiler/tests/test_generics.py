"""Parametric generics: instantiation checking, call-site-local inference,
trait-bound (where-clause) checking, and the optional monomorphization pass.

All tests go through parsed source (parse -> desugar -> freeze -> infer ->
HIR -> MIR -> interpreter), per the repo convention: no hand-built fixtures.

What the type system promises (docs/type_system.md):
- generic type declarations with `<>`, applications with `[]`/`<>`;
- explicit type parameters ("help catch type errors earlier") and implicit
  ones ("inferred from usage");
- bounds/constraints on type parameters (`where T: Trait` / `fn f<T: Trait>`).

Also covered (previously documented caveats, now closed — see
docs/v1_gap_analysis.md):
- substitution inside type applications (`Vec[T]`, nested `Pair[Pair[T]]`)
  for values whose element/argument types are known, base-name checking
  when they are not;
- bracket-form explicit instantiations (`Full[Int](x)`, `identity[Int](x)`)
  routed through the same checking (and lowering) as the angle form;
- impl-block where clauses, enforced at coherence-load time where decidable
  (constraints over concrete types); conditional impls
  (`implement Show for Pair[T] where T: Show`) stay permissive;
- module-qualified generic calls (`mod.f<Int>(x)`) checked and
  monomorphized like plain ones.

What is deliberately NOT covered here (see docs/v1_gap_analysis.md): variance
annotations, higher-kinded type parameters, and associated types.
"""
from __future__ import annotations

import pytest

from metaxu.compiler.pipeline import (
    build_context_from_source,
    run_pipeline_from_source,
)
from metaxu.compiler.frozen_borrow_checker import TypeCheckError
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir import dump_mir
from metaxu.compiler.mir_interp import MirInterpreter
from metaxu.compiler.monomorphize import collect_signatures, monomorphize_hir


def compile_ok(source: str) -> None:
    run_pipeline_from_source(source)


def reject(source: str, needle: str) -> None:
    with pytest.raises(TypeCheckError) as exc:
        run_pipeline_from_source(source)
    assert needle in str(exc.value), (
        f"expected a diagnostic containing {needle!r}, got: {exc.value}"
    )


def run_main(source: str, monomorphize: bool = False, entry: str = "main"):
    """Compile and interpret, optionally through the monomorphization pass."""
    ctx = build_context_from_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    if monomorphize:
        hir = monomorphize_hir(hir, collect_signatures(ctx.id_map))
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    return interp.call(entry, [])


def mir_text(source: str, monomorphize: bool = False) -> str:
    ctx = build_context_from_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    if monomorphize:
        hir = monomorphize_hir(hir, collect_signatures(ctx.id_map))
    return dump_mir(lower_hir_to_mir(hir))


# ---------------------------------------------------------------------------
# Instantiation checking: generic structs
# ---------------------------------------------------------------------------

GENERIC_STRUCT = """
struct Pair<T> { x: T, y: T }
"""


def test_struct_explicit_instantiation_ok():
    compile_ok(GENERIC_STRUCT + """
fn main() -> int {
    let p = Pair<Int> { x: 1, y: 2 }
    return p.x
}
""")


def test_struct_explicit_instantiation_wrong_literal_rejected():
    reject(GENERIC_STRUCT + """
fn main() -> int {
    let p = Pair<Int> { x: 1, y: "nope" }
    return 0
}
""", "field 'y' of Pair")


def test_struct_explicit_instantiation_string_ok():
    compile_ok(GENERIC_STRUCT + """
fn main() -> int {
    let p = Pair<String> { x: "a", y: "b" }
    return 0
}
""")


def test_struct_known_typed_variable_rejected():
    # Non-literal field values whose type is KNOWN (a string-classed local)
    # are checked through the constraint-graph conflict machinery.
    reject(GENERIC_STRUCT + """
fn main() -> int {
    let s = "hello"
    let p = Pair<Int> { x: 1, y: s }
    return 0
}
""", "Int and String")


def test_struct_type_arg_arity_rejected():
    reject(GENERIC_STRUCT + """
fn main() -> int {
    let p = Pair<Int, String> { x: 1, y: 2 }
    return 0
}
""", "wrong number of type arguments for Pair")


def test_struct_field_expecting_other_struct_rejected():
    reject("""
struct Inner { v: int }
struct Other { w: int }
struct Holder<T> { name: String, inner: Inner }
fn main() -> int {
    let h = Holder<Int> { name: "h", inner: Other { w: 1 } }
    return 0
}
""", "field 'inner' of Holder")


def test_struct_nested_struct_value_ok():
    compile_ok("""
struct Inner { v: int }
struct Holder<T> { name: String, inner: Inner }
fn main() -> int {
    let h = Holder<Int> { name: "h", inner: Inner { v: 1 } }
    return h.inner.v
}
""")


# ---------------------------------------------------------------------------
# Instantiation checking: generic enums
# ---------------------------------------------------------------------------

GENERIC_ENUM = """
enum Box<T> {
    Full(T),
    Empty
}
"""


def test_enum_explicit_instantiation_ok():
    compile_ok(GENERIC_ENUM + """
fn main() -> int {
    let b = Full<Int>(3)
    return 0
}
""")


def test_enum_explicit_instantiation_wrong_payload_rejected():
    reject(GENERIC_ENUM + """
fn main() -> int {
    let b = Full<Int>("nope")
    return 0
}
""", "Box.Full")


def test_enum_known_typed_variable_payload_rejected():
    reject(GENERIC_ENUM + """
fn main() -> int {
    let s = "hello"
    let b = Full<Int>(s)
    return 0
}
""", "Int and String")


# ---------------------------------------------------------------------------
# Instantiation checking: generic function calls with explicit type args
# ---------------------------------------------------------------------------

IDENTITY = """
fn identity<T>(x: T) -> T { return x }
"""


def test_fn_explicit_instantiation_ok():
    assert run_main(IDENTITY + """
fn main() -> int {
    let i = identity<Int>(41)
    return i + 1
}
""") == 42


def test_fn_explicit_instantiation_wrong_arg_rejected():
    reject(IDENTITY + """
fn main() -> int {
    let i = identity<Int>("s")
    return 0
}
""", "call of identity")


def test_fn_explicit_instantiation_known_var_rejected():
    reject(IDENTITY + """
fn main() -> int {
    let s = "hello"
    let i = identity<Int>(s)
    return 0
}
""", "Int and String")


def test_fn_type_arg_arity_rejected():
    reject(IDENTITY + """
fn main() -> int {
    let i = identity<Int, String>(1)
    return 0
}
""", "wrong number of type arguments for identity")


def test_type_args_on_non_generic_fn_rejected():
    reject("""
fn f(x: int) -> int { return x }
fn main() -> int { return f<Int>(1) }
""", "non-generic function f")


def test_mono_fn_declared_param_type_enforced():
    # Non-generic declared parameter types are checked for known-typed args
    # through the same machinery.
    reject("""
fn f(x: int) -> int { return x }
fn main() -> int { return f("s") }
""", "call of f")


# ---------------------------------------------------------------------------
# Inference of instantiations: call-site-local (let-polymorphism lite)
# ---------------------------------------------------------------------------

def test_inferred_instantiations_coexist_per_call_site():
    # identity(1) and identity("s") in one program: each call site gets a
    # fresh instantiation, nothing is shared across call sites.
    assert run_main(IDENTITY + """
fn main() -> int {
    let i = identity(1)
    let s = identity("str")
    return i
}
""") == 1


PAIR = """
fn pair<T>(a: T, b: T) -> T { return a }
"""


def test_inferred_conflicting_instantiation_rejected():
    reject(PAIR + """
fn main() -> int {
    let x = pair(1, "s")
    return 0
}
""", "conflicting instantiations of type parameter T")


def test_inferred_conflict_through_variables_rejected():
    # Var-typed args participate via call-site-local unification feeding the
    # conflict-detection machinery.
    reject(PAIR + """
fn main() -> int {
    let a = 1
    let b = "s"
    let x = pair(a, b)
    return 0
}
""", "Int and String")


def test_inferred_matching_instantiation_ok():
    assert run_main(PAIR + """
fn main() -> int {
    let x = pair(20, 22)
    let y = pair("a", "b")
    return x + 22
}
""") == 42


def test_generic_result_feeds_conflict_detection():
    # identity<String>(...) constrains the call's result var: adding an Int
    # to it must fail.
    reject(IDENTITY + """
fn main() -> int {
    let s = identity<String>("a")
    return s + 1
}
""", "Int and String")


# ---------------------------------------------------------------------------
# Substitution inside type applications (Vec[T], Pair[T], nested)
# ---------------------------------------------------------------------------

NESTED_APP = """
struct Pair<T> { x: T, y: T }
struct Holder<T> { inner: Pair[T] }
"""


def test_type_application_field_wrong_instantiation_rejected():
    # Field declared Pair[T] in Holder<Int> checks against Pair[Int]:
    # a Pair<String> value is a structural mismatch.
    reject(NESTED_APP + """
fn main() -> int {
    let h = Holder<Int> { inner: Pair<String> { x: "a", y: "b" } }
    return 0
}
""", "field 'inner' of Holder: expected Pair[Int], got Pair[String]")


def test_type_application_field_right_instantiation_ok():
    assert run_main(NESTED_APP + """
fn main() -> int {
    let h = Holder<Int> { inner: Pair<Int> { x: 40, y: 2 } }
    return h.inner.x + h.inner.y
}
""") == 42


def test_type_application_field_base_name_mismatch_rejected():
    # Base constructors must match even when the value carries no type args.
    reject(NESTED_APP + """
fn main() -> int {
    let h = Holder<Int> { inner: 3 }
    return 0
}
""", "field 'inner' of Holder: expected Pair[Int], got Int")


def test_type_application_field_unknown_value_permissive():
    # A value whose type is not statically known stays unchecked
    # (no false positives; inference/runtime still applies).
    compile_ok(NESTED_APP + """
fn mk<U>(v: U) -> U { return v }
fn main() -> int {
    let p = mk(1)
    let h = Holder<Int> { inner: p }
    return 0
}
""")


def test_type_application_field_args_unknown_base_checked():
    # Value of known base but unknown args (`Pair {...}` inferred):
    # base-name check passes, argument positions stay permissive.
    compile_ok(NESTED_APP + """
fn main() -> int {
    let h = Holder<Int> { inner: Pair { x: 1, y: 2 } }
    return 0
}
""")


def test_type_application_vec_field_wrong_value_rejected():
    reject("""
struct H<T> { items: Vec[T] }
fn main() -> int {
    let h = H<Int> { items: 3 }
    return 0
}
""", "field 'items' of H: expected Vec[Int], got Int")


def test_type_application_vec_field_unknown_stays_permissive():
    # Vec.new() has no statically known element type: base-name-only
    # checking, current permissive behavior preserved.
    compile_ok("""
struct H<T> { items: Vec[T] }
fn main() -> int {
    let v = Vec.new()
    v.push(1)
    let h = H<Int> { items: v }
    return 0
}
""")


def test_type_application_param_wrong_rejected():
    reject("""
struct Pair<T> { x: T, y: T }
fn f<T>(p: Pair[T]) -> int { return 0 }
fn main() -> int {
    return f<Int>(Pair<String> { x: "a", y: "b" })
}
""", "call of f: argument has type Pair[String], expected Pair[Int]")


def test_type_application_param_right_ok():
    compile_ok("""
struct Pair<T> { x: T, y: T }
fn f<T>(p: Pair[T]) -> int { return 0 }
fn main() -> int {
    return f<Int>(Pair<Int> { x: 1, y: 2 })
}
""")


def test_type_application_deep_nesting_checked():
    # Two levels down: Wrap<Int> declares p: Pair[Pair[T]]; the value's
    # explicit Pair<Pair[String]> instantiation conflicts at depth 2.
    reject("""
struct Pair<T> { x: T, y: T }
struct Wrap<T> { p: Pair[Pair[T]] }
fn main() -> int {
    let w = Wrap<Int> { p: Pair<Pair[String]> {
        x: Pair<String> { x: "a", y: "b" },
        y: Pair<String> { x: "c", y: "d" } } }
    return 0
}
""", "field 'p' of Wrap: expected Pair[Pair[Int]], got Pair[Pair[String]]")


def test_type_application_enum_payload_checked():
    # Variant payload declared as a type application substitutes too.
    reject("""
struct Pair<T> { x: T, y: T }
enum Carton<T> {
    Boxed(Pair[T]),
    Missing
}
fn main() -> int {
    let c = Boxed<Int>(Pair<String> { x: "a", y: "b" })
    return 0
}
""", "Carton.Boxed: payload has type Pair[String], expected Pair[Int]")


# ---------------------------------------------------------------------------
# Bracket-form explicit instantiations: Full[Int](x) == Full<Int>(x)
# ---------------------------------------------------------------------------

def test_bracket_enum_ctor_runs():
    assert run_main(GENERIC_ENUM + """
fn main() -> int {
    let b = Full[Int](3)
    return match b { Full(v) => v, Empty => 0 }
}
""") == 3


def test_bracket_enum_ctor_wrong_payload_rejected():
    reject(GENERIC_ENUM + """
fn main() -> int {
    let b = Full[Int]("nope")
    return 0
}
""", "Box.Full")


def test_bracket_enum_ctor_known_var_rejected():
    reject(GENERIC_ENUM + """
fn main() -> int {
    let s = "hello"
    let b = Full[Int](s)
    return 0
}
""", "Int and String")


def test_bracket_fn_call_runs():
    assert run_main(IDENTITY + """
fn main() -> int {
    let i = identity[Int](41)
    return i + 1
}
""") == 42


def test_bracket_fn_call_wrong_arg_rejected():
    reject(IDENTITY + """
fn main() -> int {
    let i = identity[Int]("s")
    return 0
}
""", "call of identity")


def test_bracket_fn_call_monomorphizes():
    txt = mir_text(IDENTITY + """
fn main() -> int {
    let i = identity[Int](41)
    return i + 1
}
""", monomorphize=True)
    assert "identity$Int" in txt


def test_bracket_non_type_index_left_alone():
    # `identity[n](1)` with n a plain local is NOT an instantiation: the
    # shape stays an ordinary indexed call (unknown stays permissive; no
    # rewrite, no false instantiation checking).
    compile_ok(IDENTITY + """
fn main() -> int {
    let n = 0
    let i = identity[n](1)
    return 0
}
""")


# ---------------------------------------------------------------------------
# Impl-block where clauses (coherence-load time, where decidable)
# ---------------------------------------------------------------------------

IMPL_WHERE = """
trait Show { fn show(self) -> String }
trait Eq2 { fn eq2(self) -> Bool }
struct P { v: int }
struct Q { w: int }
"""


def test_impl_where_concrete_violation_rejected():
    # `where Q: Eq2` with no `implement Eq2 for Q` anywhere is decidable
    # as soon as all impls are loaded: rejected, naming the missing impl.
    reject(IMPL_WHERE + """
implement Show for P where Q: Eq2 {
    fn show(self) -> String { return "p" }
}
fn main() -> int { return 0 }
""", "missing `implement Eq2 for Q`")


def test_impl_where_concrete_satisfied_ok():
    compile_ok(IMPL_WHERE + """
implement Eq2 for Q { fn eq2(self) -> Bool { return true } }
implement Show for P where Q: Eq2 {
    fn show(self) -> String { return "p" }
}
fn main() -> int { return 0 }
""")


def test_impl_where_self_constraint_checked():
    # The impl's own target type in its where clause is concrete too.
    reject(IMPL_WHERE + """
implement Show for P where P: Eq2 {
    fn show(self) -> String { return "p" }
}
fn main() -> int { return 0 }
""", "missing `implement Eq2 for P`")


def test_impl_where_conditional_stays_permissive():
    # `implement Show for Pair[T] where T: Show` depends on each
    # instantiation's type arguments, which the base-name registry cannot
    # decide: stays permissive (runtime dispatch enforces).
    compile_ok(IMPL_WHERE + """
struct Pair<T> { x: T, y: T }
implement Show for Pair[T] where T: Show {
    fn show(self) -> String { return "pair" }
}
fn main() -> int { return 0 }
""")


# ---------------------------------------------------------------------------
# Module-qualified generic calls: mod.f<Int>(x)
# ---------------------------------------------------------------------------

QUAL_LIB = "fn ident<T>(x: T) -> T { return x }\n"


def _write_tree(tmp_path, files: dict[str, str]) -> str:
    for rel, src in files.items():
        (tmp_path / rel).write_text(src)
    return str(tmp_path / "main.mx")


def test_module_qualified_generic_call_wrong_rejected(tmp_path):
    root = _write_tree(tmp_path, {
        "mathmod.mx": QUAL_LIB,
        "main.mx": """
import mathmod;
fn main() -> int {
    let a = mathmod.ident<Int>("s")
    return 0
}
""",
    })
    with pytest.raises(TypeCheckError) as exc:
        run_pipeline_from_source((tmp_path / "main.mx").read_text(),
                                 file_path=root)
    assert "call of mathmod.ident" in str(exc.value)


def test_module_qualified_generic_call_right_runs(tmp_path):
    root = _write_tree(tmp_path, {
        "mathmod.mx": QUAL_LIB,
        "main.mx": """
import mathmod;
fn main() -> int {
    let a = mathmod.ident<Int>(41)
    return a + 1
}
""",
    })
    ctx = build_context_from_source((tmp_path / "main.mx").read_text(),
                                    file_path=root)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    assert interp.call("main", []) == 42


def test_module_qualified_generic_call_monomorphizes(tmp_path):
    root = _write_tree(tmp_path, {
        "mathmod.mx": QUAL_LIB,
        "main.mx": """
import mathmod;
fn main() -> int {
    let a = mathmod.ident<Int>(41)
    return a + 1
}
""",
    })
    ctx = build_context_from_source((tmp_path / "main.mx").read_text(),
                                    file_path=root)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    hir = monomorphize_hir(hir, collect_signatures(ctx.id_map))
    txt = dump_mir(lower_hir_to_mir(hir))
    assert "mathmod.ident$Int" in txt
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    assert interp.call("main", []) == 42


def test_module_qualified_where_clause_checked(tmp_path):
    # Trait bounds on a module function are enforced across the module
    # boundary, exactly like plain calls.
    root = _write_tree(tmp_path, {
        "showlib.mx": """
trait Show { fn show(self) -> String }
fn describe<T>(x: T) -> T where T: Show { return x }
""",
        "main.mx": """
import showlib;
fn main() -> int {
    let x = showlib.describe<Int>(1)
    return 0
}
""",
    })
    with pytest.raises(TypeCheckError) as exc:
        run_pipeline_from_source((tmp_path / "main.mx").read_text(),
                                 file_path=root)
    assert "implement Show for Int" in str(exc.value)


# ---------------------------------------------------------------------------
# Trait-bound (where-clause) checking
# ---------------------------------------------------------------------------

SHOWABLE = """
trait Show {
    fn show(self) -> String
}

struct P { v: int }

implement Show for P {
    fn show(self) -> String { return "p" }
}

fn describe<T>(x: T) -> T where T: Show { return x }
"""


def test_where_clause_satisfied_by_impl():
    compile_ok(SHOWABLE + """
fn main() -> int {
    let p = P { v: 1 }
    let q = describe(p)
    return 0
}
""")


def test_where_clause_violation_rejected_names_missing_impl():
    reject(SHOWABLE + """
fn main() -> int {
    let x = describe(1)
    return 0
}
""", "implement Show for Int")


def test_where_clause_checked_with_explicit_type_args():
    reject(SHOWABLE + """
fn main() -> int {
    let x = describe<Int>(1)
    return 0
}
""", "implement Show for Int")


def test_inline_bound_syntax_checked():
    # `fn f<T: Trait>` inline bounds merge into the same where checking.
    reject("""
trait Show { fn show(self) -> String }
struct P { v: int }
implement Show for P { fn show(self) -> String { return "p" } }
fn describe2<T: Show>(x: T) -> T { return x }
fn main() -> int {
    let x = describe2("s")
    return 0
}
""", "implement Show for String")


def test_where_clause_unresolved_param_not_flagged():
    # A call whose instantiation cannot be resolved statically stays
    # unchecked (runtime dispatch still enforces).
    compile_ok(SHOWABLE + """
fn passthrough<U>(y: U) -> U { return describe(y) }
fn main() -> int { return 0 }
""")


# ---------------------------------------------------------------------------
# Monomorphization pass (optional, default off)
# ---------------------------------------------------------------------------

MONO_PROGRAM = IDENTITY + """
fn wrap<T>(x: T) -> T { return identity(x) }

fn main() -> int {
    let i = identity(1)
    let s = identity("str")
    let e = identity<Int>(40)
    let w = wrap(1)
    return i + e + w
}
"""


def test_monomorphize_identical_results():
    assert run_main(MONO_PROGRAM, monomorphize=False) == 42
    assert run_main(MONO_PROGRAM, monomorphize=True) == 42


def test_monomorphize_specialized_names_in_mir():
    txt = mir_text(MONO_PROGRAM, monomorphize=True)
    assert "identity$Int" in txt
    assert "identity$String" in txt
    # Transitive specialization through a generic wrapper.
    assert "wrap$Int" in txt
    # Fully specialized originals are erased.
    assert "func identity suspending" not in txt
    assert "func wrap suspending" not in txt


def test_monomorphize_off_keeps_generics():
    txt = mir_text(MONO_PROGRAM, monomorphize=False)
    assert "identity$Int" not in txt
    assert "func identity suspending" in txt


def test_monomorphize_pipeline_flag():
    # The pipeline-level flag threads through run_pipeline_from_source.
    _ast, _hir, mir_plain, _c = run_pipeline_from_source(MONO_PROGRAM)
    _ast, _hir, mir_mono, _c = run_pipeline_from_source(
        MONO_PROGRAM, monomorphize=True)
    assert "identity$Int" not in mir_plain
    assert "identity$Int" in mir_mono


def test_monomorphize_unresolvable_keeps_generic_original():
    src = IDENTITY + """
fn relay<U>(y: U) -> U { return identity(y) }

fn main() -> int {
    let a = identity(2)
    return a
}
"""
    txt = mir_text(src, monomorphize=True)
    # identity(2) specializes...
    assert "identity$Int" in txt
    # ...but identity(y) inside relay is unresolvable, so the generic
    # original must survive (relay itself was never instantiated).
    assert "func identity suspending" in txt
    assert "func relay suspending" in txt
    assert run_main(src, monomorphize=True) == 2


def test_monomorphize_struct_and_variant_arg_instantiation():
    src = """
struct P { v: int }
enum Wrap { W(int) }
fn first<T>(x: T, y: T) -> T { return x }
fn main() -> int {
    let a = first(P { v: 1 }, P { v: 2 })
    let b = first(W(1), W(2))
    return a.v
}
"""
    txt = mir_text(src, monomorphize=True)
    assert "first$P" in txt
    assert "first$Wrap" in txt
    assert run_main(src, monomorphize=True) == 1


def test_monomorphize_recursion_terminates():
    src = """
fn count<T>(x: T, n: int) -> int {
    if n <= 0 { return 0 }
    return count(x, n - 1) + 1
}
fn main() -> int { return count("s", 3) }
"""
    # Recursive call has x: T with T -> String, same instantiation: must
    # resolve to the in-flight clone, not loop.
    txt = mir_text(src, monomorphize=True)
    assert "count$String" in txt
    assert run_main(src, monomorphize=True) == 3
    assert run_main(src, monomorphize=False) == 3


def test_monomorphize_preserves_trait_dispatch_impls():
    # Regression: generic __impl$ functions are reached dynamically by the
    # interpreter's trait dispatch; the pass must never erase them.
    src = """
trait Doubler {
    fn double(self) -> int
}
struct N { v: int }
implement Doubler for N {
    fn double(self) -> int { return self.v * 2 }
}
fn main() -> int {
    let n = N { v: 21 }
    return n.double()
}
"""
    assert run_main(src, monomorphize=False) == 42
    assert run_main(src, monomorphize=True) == 42
