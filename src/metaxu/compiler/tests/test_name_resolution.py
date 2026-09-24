"""Undefined names are compile-time errors (compiler/name_resolution.py).

The bug this closes: a name that resolved to nothing was *dropped*.

    fn main() -> int { undefined_thing; 42 }   compiled, ran, answered 42
    fn main() -> int { undefined_thing }       compiled, ran, answered ()
    fn main() -> int { helpr(); 7 }            compiled; died only at RUN time

It is the same family as the lexer audit's `let x = 1e10; x`, which answered
`1` precisely *because* the stray `e10` became a discarded identifier
(docs/token_reachability.md).

Two halves, and the second one is the important one:

* the REJECTION tests below pin that an undefined variable or callee is now
  a `TypeCheckError` with a source location (and a "did you mean" hint when
  a near name exists);
* `TestNoFalsePositives` is the anti-false-positive battery — one test per
  in-scope category enumerated in docs/name_resolution.md.  A false positive
  breaks working code, so every category the language really provides has a
  program here that must keep compiling.  `TestCorpus` runs the same
  guarantee over the 19 gate files and every `std/*.mx`.

Everything goes through parsed source (never hand-built HIR/MIR), per the
project convention.
"""
from __future__ import annotations

import glob
import os

import pytest

from metaxu.compiler.frozen_borrow_checker import TypeCheckError
from metaxu.compiler.name_resolution import UNRESOLVED_NAME_KIND
from metaxu.compiler.pipeline import build_context_from_source, run_pipeline_from_source

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def unresolved(source: str, file_path: str = "<mem>") -> list:
    """The name-resolution diagnostics for `source` (never raises)."""
    ctx = build_context_from_source(source, file_path=file_path)
    return [e for e in ctx.tables.constraints.get(-2, ())
            if getattr(e, "kind", "") == UNRESOLVED_NAME_KIND]


def compiles_clean(source: str, file_path: str = "<mem>") -> None:
    """Assert `source` produces no name-resolution diagnostic at all."""
    errs = unresolved(source, file_path=file_path)
    assert not errs, "false positive(s): " + "; ".join(str(e) for e in errs)


def rejected(source: str) -> TypeCheckError:
    with pytest.raises(TypeCheckError) as exc:
        run_pipeline_from_source(source)
    return exc.value


# ---------------------------------------------------------------------------
# The three programs that motivated the work
# ---------------------------------------------------------------------------

def test_undefined_name_in_statement_position_is_rejected():
    """It used to compile, run and answer 42: a name in statement position
    lowered to no MIR at all, so the undefined name simply vanished."""
    err = rejected("fn main() -> int { undefined_thing; 42 }")
    assert "undefined variable 'undefined_thing'" in str(err)


def test_undefined_name_in_tail_position_is_rejected():
    """It used to compile and answer `()` — the function's declared `int`
    return type notwithstanding."""
    err = rejected("fn main() -> int { undefined_thing }")
    assert "undefined variable 'undefined_thing'" in str(err)


def test_typo_callee_is_rejected_at_compile_time_with_a_suggestion():
    """`helpr()` used to reach the interpreter and die there with
    `Unknown callee`. It is a compile-time error now, and the suggestion
    names the function that was meant."""
    err = rejected("fn helper() -> int { 1 }\n"
                   "fn main() -> int { helpr(); 7 }")
    assert "undefined function 'helpr'" in str(err)
    assert "did you mean 'helper'" in str(err)


# ---------------------------------------------------------------------------
# Diagnostics: location, excerpt, kind
# ---------------------------------------------------------------------------

def test_diagnostic_carries_file_line_and_column():
    errs = unresolved("fn main() -> int {\n"
                      "    let a = 1;\n"
                      "    a + missing\n"
                      "}")
    assert len(errs) == 1
    loc = errs[0].location
    assert loc is not None
    assert loc.line == 3
    assert "<mem>:3:" in str(errs[0])


def test_diagnostic_renders_a_source_excerpt_with_a_caret():
    err = rejected("fn main() -> int { missing_name }")
    text = str(err)
    assert "fn main() -> int { missing_name }" in text
    assert "^" in text


def test_diagnostic_kind_is_a_type_error_so_the_pipeline_promotes_it():
    errs = unresolved("fn main() -> int { nope }")
    assert [e.kind for e in errs] == [UNRESOLVED_NAME_KIND]
    assert errs[0].kind.startswith("type-")   # what run_pipeline keys on


def test_suggestion_is_omitted_when_nothing_is_close():
    errs = unresolved("fn main() -> int { qqqqzzzz }")
    assert "did you mean" not in str(errs[0])


def test_suggestion_can_come_from_a_local_binding():
    errs = unresolved("fn main() -> int { let counter = 1; countr }")
    assert "did you mean 'counter'" in str(errs[0])


# ---------------------------------------------------------------------------
# Rejection: every position a name can appear in
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,source", [
    ("statement", "fn main() -> int { nope; 0 }"),
    ("tail", "fn main() -> int { nope }"),
    ("argument", "fn g(x: int) -> int { x }\nfn main() -> int { g(nope) }"),
    ("let initializer", "fn main() -> int { let a = nope; a }"),
    ("binary operand", "fn main() -> int { 1 + nope }"),
    ("if condition", "fn main() -> int { if nope { 1 } else { 2 } }"),
    ("while condition", "fn main() -> int { while nope { }; 0 }"),
    ("return expression", "fn main() -> int { return nope; }"),
    ("match arm body", "fn main() -> int { match 1 { 1 -> nope, _ -> 0 } }"),
    ("for body", "fn main() -> int { for i in 0..3 { nope }; 0 }"),
    ("lambda body",
     "fn main() -> int { let f = fn(x: int) -> int { x + nope }; f(1) }"),
    ("nested function body",
     "fn main() -> int { fn inner() -> int { nope } inner() }"),
    ("catch body", "fn main() -> int { try { 1 } catch e { nope } }"),
    ("struct field value",
     "struct S { a: int }\nfn main() -> int { let s = S { a: nope }; s.a }"),
    ("field-access base", "fn main() -> int { nope.field }"),
    ("index base", "fn main() -> int { nope[0] }"),
    ("assignment target", "fn main() -> int { nope = 5; 0 }"),
    ("field-assignment base", "fn main() -> int { nope.f = 5; 0 }"),
    ("index-assignment base", "fn main() -> int { nope[0] = 5; 0 }"),
    ("assignment value",
     "fn main() -> int { let mut a = 1; a = nope; a }"),
])
def test_undefined_variable_is_rejected_everywhere(label, source):
    assert unresolved(source), f"{label}: undefined name was accepted"


def test_match_arm_bodies_are_checked_even_though_the_frozen_ast_drops_them():
    """The frozen AST freezes MatchExpression arms into a payload descriptor
    and does not carry the arm BODIES as children at all — which is why this
    pass runs over the mutable post-desugar AST that HIR itself lowers."""
    errs = unresolved("fn main() -> int { match 1 { 1 -> nope, _ -> 0 } }")
    assert errs and errs[0].variable == "nope"


def test_for_bodies_are_checked_even_though_the_frozen_ast_drops_them():
    """`ForStatement` freezes its body to an empty node of kind `list`."""
    errs = unresolved("fn main() -> int { for i in 0..2 { nope }; 0 }")
    assert errs and errs[0].variable == "nope"


def test_a_pattern_binding_is_not_in_scope_outside_its_arm():
    errs = unresolved("""
        fn main() -> int {
            let o = Some(1);
            match o { Some(x) -> x, None -> 0 };
            x
        }
    """)
    assert errs and errs[0].variable == "x"


def test_a_local_is_not_in_scope_before_its_own_let():
    errs = unresolved("fn main() -> int { later; let later = 1; later }")
    assert errs and errs[0].variable == "later"


def test_a_local_of_another_function_is_not_in_scope():
    errs = unresolved("""
        fn one() -> int { let secret = 1; secret }
        fn main() -> int { secret }
    """)
    assert errs and errs[0].variable == "secret"


def test_undefined_callee_is_a_compile_error_not_a_runtime_one():
    """The interpreter's `Unknown callee` stays as defence in depth, but the
    front end must not let the program get that far."""
    err = rejected("fn main() -> int { totally_unknown(1) }")
    assert "undefined function 'totally_unknown'" in str(err)


def test_undefined_dotted_callee_is_rejected():
    errs = unresolved("""
        struct S { a: int }
        fn main() -> int { S.no_such_static(1) }
    """)
    assert errs and errs[0].variable == "S.no_such_static"


# ---------------------------------------------------------------------------
# The anti-false-positive battery: one test per in-scope category
# ---------------------------------------------------------------------------

class TestNoFalsePositives:
    """Every category in docs/name_resolution.md § "What is in scope"."""

    def test_locals_parameters_and_shadowing(self):
        compiles_clean("""
            fn f(a: int, b: int) -> int { let c = a + b; let c = c * 2; c }
            fn main() -> int { f(1, 2) }
        """)

    def test_for_loop_variables_including_nesting(self):
        compiles_clean("""
            fn main() -> int {
                let mut t = 0;
                for i in 0..3 { for j in 0..2 { t = t + i + j; } }
                t
            }
        """)

    def test_match_if_let_and_while_let_pattern_bindings(self):
        compiles_clean("""
            enum Tree { Leaf(v: int), Node(l: Tree, r: Tree) }
            fn sum(t: Tree) -> int {
                match t { Leaf(v) -> v, Node(l, r) -> sum(l) + sum(r) }
            }
            fn main() -> int {
                let o = Some(3);
                let a = if let Some(x) = o { x } else { 0 };
                let r = Ok(7);
                let b = match r { Ok(v) -> v, Err(e) -> 0 };
                a + b + sum(Leaf(1))
            }
        """)

    def test_mode_annotated_pattern_bindings(self):
        """`Ok(@mut v)` parses its payload as a BorrowUnique; at pattern
        level the annotation is static and the pattern just binds `v`."""
        compiles_clean("""
            fn main() -> int {
                let r = Ok(5);
                match r { Ok(@mut v) -> v, Err(e) -> 0 }
            }
        """)

    def test_catch_binding(self):
        compiles_clean("fn main() -> int { try { 1 } catch e { 2 } }")

    def test_handler_arm_parameters_and_resume(self):
        compiles_clean("""
            effect St { get() -> int; put(v: int) -> int; }
            fn body() -> int performs St { perform St.get() + perform St.put(2) }
            fn main() -> int {
                handle body() {
                    perform St.get(  ) => { resume(40) }
                    perform St.put(v) => { resume(v) }
                }
            }
        """)

    def test_module_constants_forward_references_and_nested_functions(self):
        compiles_clean("""
            let PI = 3;
            fn main() -> int { later() + PI }
            fn later() -> int {
                fn inner(x: int) -> int { x + 1 }
                inner(1)
            }
        """)

    def test_builtins_in_call_and_method_position(self):
        compiles_clean("""
            fn main() -> int {
                let v = [1, 2, 3];
                push(v, 4);
                let n = len(v);
                print("n");
                assert(n > 0);
                n
            }
        """)

    def test_effect_operations_called_unqualified(self):
        compiles_clean("""
            effect Emit { emit(x: int) -> int = x; }
            fn main() -> int { emit(1) }
        """)

    def test_effect_operation_defaults_read_their_own_parameters(self):
        compiles_clean("""
            effect Cap { ask(x: int) -> int = x + 1; }
            fn main() -> int { perform Cap.ask(41) }
        """)

    def test_trait_methods_impl_methods_and_static_calls(self):
        compiles_clean("""
            trait Show { fn show(self) -> str; }
            struct P { x: int }
            implement Show for P { fn show(self) -> str { "p" } }
            implement P { fn make(v: int) -> P { P { x: v } } }
            fn main() -> str { let p = P.make(1); p.show() }
        """)

    def test_enum_variant_constructors_including_the_builtin_ones(self):
        compiles_clean("""
            enum Color { Red, Green }
            fn main() -> int {
                let a = Red;
                let b = Color.Green;
                let c = Some(1);
                let d = None;
                let e = Ok(1);
                let f = Err(2);
                0
            }
        """)

    def test_type_names_in_value_position(self):
        compiles_clean("""
            fn main() -> int {
                let v = Vec.new();
                push(v, 1);
                let w = vector[float,3](1.0, 2.0, 3.0);
                len(v)
            }
        """)

    def test_extern_ffi_declarations(self):
        compiles_clean("""
            extern "C" {
                fn malloc(size: uint) -> *void;
                fn free(ptr: *void);
                type FILE;
            }
            fn main() -> int {
                unsafe { let p = malloc(8); free(p); }
                0
            }
        """)

    def test_the_null_pointer_literal(self):
        compiles_clean("""
            extern "C" { fn malloc(size: uint) -> *void; }
            fn main() -> int {
                let p = malloc(4);
                if p == null { 1 } else { 0 }
            }
        """)

    def test_lambda_parameters_and_captures(self):
        compiles_clean("""
            fn main() -> int {
                let base = 10;
                let add = fn(x: int) -> int { x + base };
                let twice = fn(g: fn(int) -> int, y: int) -> int { g(g(y)) };
                twice(add, 1)
            }
        """)

    def test_type_parameters_and_const_generics_in_value_position(self):
        """An impl's type parameters are in scope as VALUES inside its
        methods: a const generic is bound to the receiver's runtime
        dimension at method entry, and a plain parameter is what `type_of`
        takes."""
        compiles_clean("""
            trait Sized2<T, const N: int> { fn size(self) -> int; }
            implement<T, const N: int> Sized2<T,N> for vector[T,N] {
                fn size(self) -> int {
                    let mut t = 0;
                    for i in 0..N { t = t + 1; }
                    t
                }
            }
            fn main() -> int {
                let v = vector[float,3](1.0, 2.0, 3.0);
                v.size()
            }
        """)

    def test_comprehension_targets(self):
        compiles_clean("""
            fn main() -> int {
                let w = vector[float,4](x * 2.0 for x in 0..4);
                0
            }
        """)

    def test_struct_field_initializers_and_reads(self):
        compiles_clean("""
            struct S { a: int, b: int }
            fn main() -> int {
                let k = 2;
                let s = S { a: k, b: k + 1 };
                s.a + s.b
            }
        """)

    def test_names_imported_from_a_real_stdlib_module(self):
        compiles_clean("""
            from std.math import sqrt;
            import std.math;
            fn main() -> float { sqrt(4.0) + std.math.abs(0.0 - 1.0) }
        """)

    def test_names_imported_from_an_unresolved_std_placeholder(self):
        """`std.simd` has no file under the stdlib root, so the loader takes
        the external-placeholder path: nothing is rewritten and the bare
        imported names are all the program has. They are still in scope."""
        compiles_clean("""
            from std.simd import SimdOp, SimdRegister;
            effect S2 { go() -> int = 1; }
            fn main() -> int { go() }
        """)

    def test_compiler_reserved_double_underscore_names_are_never_reported(self):
        """Indexing lowers to `__index_get`; the whole `__` namespace belongs
        to the compiler (module_loader.check_reserved_names) and is exempt."""
        compiles_clean("fn main() -> int { let v = [1,2]; v[0] }")

    def test_receiver_calls_on_locals_and_fields(self):
        compiles_clean("""
            struct Box { items: int }
            fn main() -> int {
                let v = [1, 2];
                let n = v.len();
                let s = n.to_string();
                n
            }
        """)


# ---------------------------------------------------------------------------
# The corpus: the gate files and the whole standard library
# ---------------------------------------------------------------------------

def _corpus() -> list[str]:
    return (sorted(glob.glob(os.path.join(REPO_ROOT, "examples", "*.mx")))
            + sorted(glob.glob(os.path.join(os.path.dirname(__file__),
                                            "fixtures", "test_*.mx")))
            + sorted(glob.glob(os.path.join(REPO_ROOT, "std", "*.mx"))))


class TestImportsAreModuleScoped:
    """An import binds a name in its own module only. A file that imports
    a module does not inherit that module's imports; before the resolver
    walked each module body with its own imports, `main` below compiled
    clean and died at run time with `Unknown callee: 'trim'`."""

    # (`parse_int_or` rather than `trim`: `trim` became a builtin, so a
    # bare `trim(...)` resolves everywhere by design.)
    def _project(self, tmp_path):
        (tmp_path / "util.mx").write_text(
            "export { helper };\nfrom std.parse import parse_int_or;\n"
            "fn helper(s: string) -> int { parse_int_or(s, 0) }\n")
        return tmp_path / "main.mx"

    def test_a_dependencys_import_is_not_in_the_importers_scope(self, tmp_path):
        main = self._project(tmp_path)
        src = ("import util;\n\nfn main() -> int {\n"
               "    print(parse_int_or(\"1\", 0));\n    0\n}\n")
        main.write_text(src)
        errs = unresolved(src, file_path=str(main))
        assert len(errs) == 1 and "undefined function 'parse_int_or'" in str(errs[0])

    def test_the_dependency_itself_and_an_own_import_are_clean(self, tmp_path):
        main = self._project(tmp_path)
        src = ("import util;\nfrom std.parse import parse_int_or;\n\n"
               "fn main() -> int {\n"
               "    print(util.helper(\"1\") + parse_int_or(\"2\", 0));\n    0\n}\n")
        main.write_text(src)
        compiles_clean(src, file_path=str(main))

    def test_a_nested_module_block_sees_the_files_imports(self):
        compiles_clean("""
            from std.parse import parse_int_or;
            module inner {
                export { tidy };
                fn tidy(s: string) -> int { parse_int_or(s, 0) }
            }
            fn main() -> int { print(inner.tidy("7")); 0 }
        """)


class TestCorpus:
    """Not one false positive anywhere in the shipped corpus."""

    @pytest.mark.parametrize("path", _corpus(),
                             ids=lambda p: os.path.basename(p))
    def test_file_resolves_every_name(self, path):
        source = open(path).read()
        errs = unresolved(source, file_path=path)
        assert not errs, ("name resolution flagged shipped code: "
                          + "; ".join(str(e) for e in errs))

    def test_the_corpus_is_the_size_the_gates_expect(self):
        """A guard on the guard: if the corpus shrinks, the parametrization
        above silently checks less."""
        examples = glob.glob(os.path.join(REPO_ROOT, "examples", "*.mx"))
        fixtures = glob.glob(os.path.join(os.path.dirname(__file__),
                                          "fixtures", "test_*.mx"))
        stdlib = glob.glob(os.path.join(REPO_ROOT, "std", "*.mx"))
        assert len(examples) + len(fixtures) == 19   # the pipeline/run gates
        assert len(stdlib) >= 17                  # std/*.mx


# ---------------------------------------------------------------------------
# Drift guards on the "language provides it" sets
# ---------------------------------------------------------------------------

def test_dotted_builtin_names_match_the_interpreters_registry():
    """`DOTTED_BUILTIN_NAMES` exempts callees like `Vec.new` that no user
    declaration backs. If the runtime grows or loses one, the exemption list
    must move with it — otherwise the check either misses a typo or rejects
    a working program."""
    from metaxu.compiler.mir_interp import MirInterpreter
    from metaxu.compiler.name_resolution import DOTTED_BUILTIN_NAMES

    registry = MirInterpreter()._builtins
    assert set(DOTTED_BUILTIN_NAMES) == {n for n in registry if "." in n}


def test_extra_builtin_names_are_names_a_backend_really_knows():
    """`type_of` has no interpreter shim (calling it is still a loud runtime
    error) but both native backends list it as a runtime name, so it is a
    known NAME, not an unresolved one."""
    from metaxu.compiler.codegen_clif import _RUNTIME_NAMES as CLIF_NAMES
    from metaxu.compiler.codegen_llvm import _RUNTIME_NAMES as LLVM_NAMES
    from metaxu.compiler.name_resolution import EXTRA_BUILTIN_NAMES

    assert "type_of" in EXTRA_BUILTIN_NAMES
    assert "type_of" in LLVM_NAMES and "type_of" in CLIF_NAMES


def test_calling_type_of_is_still_loud_at_run_time():
    """Exempting the NAME is not the same as providing the function."""
    from metaxu.compiler.hir import HIRBuilder
    from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
    from metaxu.compiler.mir_interp import InterpError, MirInterpreter

    source = "fn main() -> int { type_of(1); 0 }"
    compiles_clean(source)
    ctx = build_context_from_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    with pytest.raises(InterpError, match="Unknown callee"):
        interp.call("main", [])


def test_builtin_variants_are_the_ones_hir_synthesizes():
    """`Some`/`None`/`Ok`/`Err` construct even with no user enum declaring
    them, so they must be in scope with no declaration."""
    from metaxu.compiler.name_resolution import BUILTIN_VARIANTS

    assert BUILTIN_VARIANTS == frozenset({"Some", "None", "Ok", "Err"})
    compiles_clean("fn main() -> int { let a = Some(1); let b = Ok(2); 0 }")


# ---------------------------------------------------------------------------
# Defence in depth: the runtime check stays
# ---------------------------------------------------------------------------

def _run_with_resolution_disabled(monkeypatch, source: str):
    """Compile and run `source` with the name-resolution pass switched off.

    This is the defence-in-depth question: *if* the compile-time check ever
    missed something, would the engine still be loud? Disabling the pass is
    the only honest way to ask it — and it keeps the test on parsed source
    rather than a hand-built MIR fixture."""
    import metaxu.compiler.name_resolution as nr
    from metaxu.compiler.hir import HIRBuilder
    from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
    from metaxu.compiler.mir_interp import MirInterpreter, UNIT

    monkeypatch.setattr(nr, "check_names", lambda root, file_path=None: [])
    ctx = build_context_from_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    interp.register_builtin("print", lambda *a: UNIT)
    return interp.call("main", [])


def test_the_interpreter_still_rejects_an_unknown_callee(monkeypatch):
    from metaxu.compiler.mir_interp import InterpError

    with pytest.raises(InterpError, match="Unknown callee"):
        _run_with_resolution_disabled(
            monkeypatch, "fn main() -> int { no_such_fn(1) }")


def test_the_interpreter_still_rejects_an_unbound_variable(monkeypatch):
    from metaxu.compiler.mir_interp import InterpError

    with pytest.raises(InterpError, match="Unbound variable"):
        _run_with_resolution_disabled(
            monkeypatch, "fn main() -> int { let a = nowhere; a }")


def test_an_unbound_name_in_statement_position_is_loud_at_run_time_too(monkeypatch):
    """The statement-position fix, from the runtime side: with the
    compile-time check off, `nowhere; 42` used to answer 42. The forced read
    makes it raise instead."""
    from metaxu.compiler.mir_interp import InterpError

    with pytest.raises(InterpError, match="Unbound variable"):
        _run_with_resolution_disabled(
            monkeypatch, "fn main() -> int { nowhere; 42 }")


def test_a_statement_position_name_read_emits_a_real_mir_read():
    """Statement position must not swallow. Every other expression form
    emits its own instruction; a bare name read emitted NOTHING, which is
    the mechanism by which `undefined_thing; 42` became `ret 42`. The read
    is now forced, so the runtime guard above can still fire."""
    from metaxu.compiler.hir import HIRBuilder
    from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
    from metaxu.compiler.mir import dump_mir

    ctx = build_context_from_source("fn main() -> int { let x = 1; x; 42 }")
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    mir = dump_mir(lower_hir_to_mir(hir))
    # the `x` slot is read by a copy, not silently dropped
    assert "('copy',), ('x_" in mir


@pytest.mark.parametrize("source,keeps", [
    ("fn g() -> int { 1 }\nfn main() -> int { g(); 42 }", "('call', 'g')"),
    ("fn main() -> int { let v = [1,2]; v[0]; 42 }", "__index_get"),
    ("fn main() -> int { let x = 1; x / 0; 42 }", "('binop', '/')"),
    ("struct S { a: int }\n"
     "fn main() -> int { let s = S { a: 1 }; s.a; 42 }", "field_get"),
    ("fn main() -> int { let x = 1; x as float; 42 }", "__cast"),
])
def test_statement_position_keeps_every_evaluating_form(source, keeps):
    """The survey behind the statement-position finding: nothing whose
    evaluation is observable is discarded when its value is unused."""
    from metaxu.compiler.hir import HIRBuilder
    from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
    from metaxu.compiler.mir import dump_mir

    ctx = build_context_from_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    assert keeps in dump_mir(lower_hir_to_mir(hir))
