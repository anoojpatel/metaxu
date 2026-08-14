"""Adversarial review round 7: four fixes that did not cover their own case.

Three of the four share one shape — a guard written for a specific defect
that left an adjacent spelling of the SAME defect unguarded:

1. `name_resolution.visit_function` EXTENDED the enclosing scope stack, so a
   nested `fn` resolved the outer function's locals.  The pass exists to turn
   "dies at run time with `Unbound variable`" into a compile error, and this
   was that exact failure walking straight through it.  `hir.HIRBuilder.build`
   hoists every FunctionDeclaration — at any depth — into the flat MIR
   namespace as its own `HFun` bound to its own parameters only, so isolation
   is what the runtime actually does.  Lambdas are the one construct that
   really captures (MIR `make_closure`), and they must keep doing so.
2. `lower_hir_to_mir`'s statement-position read was gated on `i != n - 1`, so
   a bare-name TAIL emitted no MIR at all and `mir_interp`'s `ret` fallback
   answered with the previous op's value.
3. `lexer._INT_LITERAL_MAX` was `2**63` so the most negative i64 stayed
   writable — which let the BARE positive `9223372036854775808` through, the
   interpreter/native divergence the bound was added to prevent.
4. The lexer's generic disambiguation guarded the `<<` OPENER but not the
   `>>` CLOSER, so `a < b >> c` was claimed as `a<b>` plus a stray `>`.

Everything is exercised through parsed source (the repo convention).
"""
from __future__ import annotations

import shutil

import pytest

from metaxu.compiler.codegen_llvm import emit_llvm
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.llvm_run import compile_and_run
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir import dump_mir
from metaxu.compiler.mir_interp import InterpError, MirInterpreter, UNIT
from metaxu.compiler.name_resolution import UNRESOLVED_NAME_KIND
from metaxu.compiler.pipeline import (
    TypeCheckError, build_context_from_source, run_pipeline_from_source,
)
from metaxu.errors import CompileError
from metaxu.lexer import Lexer

needs_clang = pytest.mark.skipif(
    shutil.which("clang") is None, reason="clang is not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unresolved(source: str) -> list:
    ctx = build_context_from_source(source, file_path="<mem>")
    return [e for e in ctx.tables.constraints.get(-2, ())
            if getattr(e, "kind", "") == UNRESOLVED_NAME_KIND]


def compiles_clean(source: str) -> None:
    errs = unresolved(source)
    assert not errs, "false positive(s): " + "; ".join(str(e) for e in errs)


def rejected(source: str) -> TypeCheckError:
    with pytest.raises(TypeCheckError) as exc:
        run_pipeline_from_source(source)
    return exc.value


def mir_from_source(source: str):
    ctx = build_context_from_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    return lower_hir_to_mir(hir)


def interp_run(source: str, entry: str = "main"):
    """Compile and run WITHOUT the strict front end, returning (value, stdout)."""
    interp = MirInterpreter()
    interp.load(mir_from_source(source))
    out: list[str] = []

    def _print(*args):
        out.append(" ".join(str(a) for a in args))
        return UNIT

    interp.register_builtin("print", _print)
    interp.register_builtin("println", _print)
    return interp.call(entry, []), "".join(line + "\n" for line in out)


def run_main(source: str, entry: str = "main"):
    """Full strict pipeline, then execute."""
    run_pipeline_from_source(source)
    return interp_run(source, entry)[0]


def lex_types(source: str) -> list[str]:
    lx = Lexer()
    lx.input(source)
    out: list[str] = []
    while True:
        tok = lx.token()
        if tok is None:
            return out
        out.append(tok.type)


def run_with_resolution_disabled(monkeypatch, source: str, entry: str = "main"):
    """Run with the name-resolution pass switched off.

    Defence in depth: if the compile-time check ever misses, the engine must
    still be LOUD rather than answer something.  Switching the pass off is the
    only honest way to ask, and it keeps the test on parsed source.
    """
    import metaxu.compiler.name_resolution as nr

    monkeypatch.setattr(nr, "check_names", lambda root, file_path=None: [])
    return interp_run(source, entry)[0]


# ---------------------------------------------------------------------------
# 1. A nested `fn` does not see the enclosing function's locals
# ---------------------------------------------------------------------------

NESTED_READS_OUTER = (
    "fn main() -> int {\n"
    "    let secret = 5;\n"
    "    fn inner() -> int { secret + 1 }\n"
    "    inner()\n"
    "}")


def test_a_nested_fn_cannot_read_the_enclosing_functions_local():
    """The false negative: this compiled clean and died at RUN time with
    `Unbound variable 'secret'` — precisely what the pass exists to
    prevent."""
    err = rejected(NESTED_READS_OUTER)
    assert "undefined variable 'secret'" in str(err)


def test_the_runtime_is_still_loud_when_the_check_is_off(monkeypatch):
    """Defence in depth: hoisting really does leave `secret` unbound."""
    with pytest.raises(InterpError, match="Unbound variable 'secret'"):
        run_with_resolution_disabled(monkeypatch, NESTED_READS_OUTER)


def test_a_nested_fn_still_sees_every_program_wide_name():
    """Isolation must drop the enclosing LOCALS and nothing else: module
    functions, module constants, enum variants and types are all hoisted or
    global, so they stay visible."""
    source = (
        "let LIMIT = 10;\n"
        "enum Color { Red, Green }\n"
        "fn helper(a: int) -> int { a + 1 }\n"
        "fn main() -> int {\n"
        "    fn inner(p: int) -> int {\n"
        "        let q = p + LIMIT;\n"
        "        let c = Color.Red;\n"
        "        helper(q)\n"
        "    }\n"
        "    inner(1)\n"
        "}")
    compiles_clean(source)
    assert run_main(source) == 12


def test_a_nested_fn_sees_its_own_params_and_locals():
    source = ("fn main() -> int {\n"
              "    fn inner(p: int) -> int { let q = p * 2; q + p }\n"
              "    inner(3)\n"
              "}")
    compiles_clean(source)
    assert run_main(source) == 9


def test_a_lambda_still_captures_the_enclosing_local():
    """The anti-over-fix: lambdas DO close over the environment (MIR
    `make_closure`), so `visit_lambda` must keep extending the scope."""
    source = ("fn main() -> int {\n"
              "    let secret = 5;\n"
              "    let f = fn(x: int) -> int { x + secret };\n"
              "    f(1)\n"
              "}")
    compiles_clean(source)
    assert run_main(source) == 6
    assert "make_closure" in dump_mir(mir_from_source(source))


def test_a_lambda_nested_in_a_lambda_still_captures():
    source = ("fn main() -> int {\n"
              "    let secret = 5;\n"
              "    let f = fn(x: int) -> int {\n"
              "        let g = fn(y: int) -> int { y + secret };\n"
              "        g(x)\n"
              "    };\n"
              "    f(1)\n"
              "}")
    compiles_clean(source)
    assert run_main(source) == 6


def test_an_impl_method_still_sees_its_blocks_type_parameters():
    """`implement` methods are visited at function depth 0, so isolation must
    not strip the block's type parameters — hir binds a const generic to the
    receiver's runtime dimension at method entry."""
    source = (
        "trait Summable { fn total(self) -> int; }\n"
        "implement<T, const N: int> Summable for vector[T,N] {\n"
        "    fn total(self) -> int {\n"
        "        let mut acc = 0;\n"
        "        for i in 0..N { acc = acc + 1 }\n"
        "        acc\n"
        "    }\n"
        "}\n"
        "fn main() -> int { 0 }")
    compiles_clean(source)


# --- the same gap elsewhere in the file: effect-operation defaults ---------

EFFECT_DEFAULT_READS_OUTER = (
    "fn main() -> int {\n"
    "    let secret = 5;\n"
    "    effect Loc {\n"
    "        get() -> int = secret;\n"
    "    }\n"
    "    perform Loc.get()\n"
    "}")


def test_an_effect_default_inside_a_function_cannot_read_outer_locals():
    """The identical gap one branch away: `hir.build` compiles an operation's
    default into a standalone `__effect_default$Eff$op` taking exactly the
    operation's parameters, so an `effect` declared inside a function body had
    the same false negative — clean compile, then `Unbound variable 'secret'
    in '__effect_default$Loc$get'`."""
    err = rejected(EFFECT_DEFAULT_READS_OUTER)
    assert "undefined variable 'secret'" in str(err)


def test_the_effect_default_runtime_is_loud_too(monkeypatch):
    with pytest.raises(InterpError, match="Unbound variable 'secret'"):
        run_with_resolution_disabled(monkeypatch, EFFECT_DEFAULT_READS_OUTER)


def test_a_top_level_effect_default_still_resolves_its_own_parameters():
    source = ("effect Log {\n"
              "    twice(x: int) -> int = x + x;\n"
              "}\n"
              "fn main() -> int { perform Log.twice(4) }")
    compiles_clean(source)
    assert run_main(source) == 8


# ---------------------------------------------------------------------------
# 2. A bare-name TAIL is read too
# ---------------------------------------------------------------------------

def test_a_bare_name_tail_emits_a_real_read():
    """`lower_expr("Var")` answers a slot name and emits nothing; the caller
    of a function body is `ret <slot>`, and `mir_interp`'s ret used to fall
    back to the previous op's value when that slot was absent.  The forced
    copy is what makes the fallback unreachable from source."""
    mir = dump_mir(mir_from_source(
        "fn main() -> int { let x = 1; x }"))
    # two copies: the `let` binding and the forced tail read
    assert mir.count("('copy',)") == 2


def test_an_unbound_tail_name_is_loud_instead_of_answering_unit(monkeypatch):
    """With the compile-time check off, `fn inner() -> int { nowhere }` used
    to answer `()` — a silent wrong value out of a function declared `int`."""
    source = ("fn inner() -> int { nowhere }\n"
              "fn main() -> int { inner() }")
    with pytest.raises(InterpError, match="Unbound variable 'nowhere'"):
        run_with_resolution_disabled(monkeypatch, source)


def test_a_statement_position_name_read_is_still_forced(monkeypatch):
    """The half that already worked stays working."""
    with pytest.raises(InterpError, match="Unbound variable"):
        run_with_resolution_disabled(
            monkeypatch, "fn main() -> int { nowhere; 42 }")


def test_a_resumed_continuation_still_returns_its_value():
    """`ret`'s fallback is now scoped to resumed continuation frames, which is
    the only case it was ever documented for.  A handler that resumes must
    still produce the resumed value."""
    source = ("effect Ask { get() -> int; }\n"
              "fn body() -> int { let v = perform Ask.get(); v + 1 }\n"
              "fn main() -> int {\n"
              "    handle body() {\n"
              "        perform Ask.get() => resume(41)\n"
              "    }\n"
              "}")
    assert run_main(source) == 42


# ---------------------------------------------------------------------------
# 3. `9223372036854775808` is only the magnitude of the most negative i64
# ---------------------------------------------------------------------------

def test_the_bare_two_to_the_63_literal_is_rejected():
    """The off-by-one: `value > 2**63` let exactly 2**63 through, so the
    interpreter answered the exact bignum while `emit_llvm` demoted the
    function with "outside i64 range" — the divergence the guard exists to
    prevent."""
    with pytest.raises(CompileError) as exc:
        build_context_from_source("fn main() -> int { 9223372036854775808 }")
    assert "out of range for a 64-bit int" in str(exc.value)


def test_the_most_negative_literal_stays_writable():
    """The other half of the contract the old bound was protecting."""
    assert run_main("fn main() -> int { -9223372036854775808 }") \
        == -9223372036854775808


def test_the_boundary_literals_keep_their_old_verdicts():
    assert run_main("fn main() -> int { 9223372036854775807 }") \
        == 9223372036854775807
    with pytest.raises(CompileError):
        build_context_from_source("fn main() -> int { 9223372036854775809 }")


@pytest.mark.parametrize("source", [
    # binary minus: the literal itself is still out of range
    "fn main() -> int { let x = 1; x - 9223372036854775808 }",
    # doubled unary minus is +2**63, which does not fit either
    "fn main() -> int { --9223372036854775808 }",
])
def test_two_to_the_63_is_rejected_anywhere_but_under_a_unary_minus(source):
    with pytest.raises(CompileError) as exc:
        build_context_from_source(source)
    assert "out of range for a 64-bit int" in str(exc.value)


def test_the_most_negative_literal_is_one_in_range_constant():
    """Folding the sign into the literal is what removes the divergence: the
    backends see -2**63 (an i64), not `neg(const 2**63)`.  Before the fix
    emit_llvm demoted the whole function over the unnegated magnitude."""
    ir = emit_llvm(mir_from_source("fn main() -> int { -9223372036854775808 }"))
    assert "outside i64 range" not in ir
    assert "ret i64 -9223372036854775808" in ir


@needs_clang
def test_native_matches_the_interpreter_for_the_most_negative_literal(tmp_path):
    source = ("fn main() -> int {\n"
              "    let x = -9223372036854775808;\n"
              "    print(x);\n"
              "    0\n"
              "}")
    result, expected_out = interp_run(source)
    ir = emit_llvm(mir_from_source(source))
    exit_code, stdout = compile_and_run(ir, "main", workdir=str(tmp_path))
    assert stdout == expected_out
    assert exit_code == int(result) % 256


@pytest.mark.parametrize("source,want", [
    # after an OPERATOR the minus is unary, so the fold must still happen —
    # note `>>` is still two GREATERs when Pass 0 runs
    ("fn main() -> int { let a = 0;\n"
     "    if a > -9223372036854775808 { 1 } else { 0 } }", 1),
    ("fn main() -> int { let a = 4;\n"
     "    if a >> 1 > -9223372036854775808 { 1 } else { 0 } }", 1),
    ("fn g(p: int) -> int { p }\n"
     "fn main() -> int { g(-9223372036854775808) }", -(2 ** 63)),
    ("fn main() -> int { return -9223372036854775808; }", -(2 ** 63)),
])
def test_the_unary_reading_is_recognised_in_every_operand_position(source, want):
    assert run_main(source) == want


def test_ordinary_negative_literals_are_untouched():
    """Only the 2**63 magnitude folds; `-5` stays MINUS + NUMBER."""
    assert lex_types("-5")[:2] == ["MINUS", "NUMBER"]
    assert run_main("fn main() -> int { -5 }") == -5
    assert run_main("fn main() -> int { let x = 8; x - 5 }") == 3


# ---------------------------------------------------------------------------
# 4. `>>` is a shift, even where a generic list could have closed
# ---------------------------------------------------------------------------

def test_a_shift_after_a_comparison_is_not_a_generic_argument_list():
    """`a < b >> c` lexed as `a<b>` plus a stray `>`, and the program failed
    with an unrelated "uncalled generic instantiation".  Pass B guarded the
    `<<` opener but not the `>>` closer."""
    assert lex_types("a < b >> c") == [
        "IDENTIFIER", "LESS", "IDENTIFIER", "SHR", "IDENTIFIER"]


def test_a_shift_in_an_argument_list_is_not_a_generic_argument_list():
    assert lex_types("g(a < b, c >> d)") == [
        "IDENTIFIER", "LPAREN", "IDENTIFIER", "LESS", "IDENTIFIER", "COMMA",
        "IDENTIFIER", "SHR", "IDENTIFIER", "RPAREN"]


def test_the_documented_precedence_is_what_runs():
    """Shifts bind tighter than comparisons, so this is `a < (b >> c)`:
    `1 < (8 >> 1)` is `1 < 4`, which is true."""
    assert run_main("fn main() -> int {\n"
                    "    let a = 1; let b = 8; let c = 1;\n"
                    "    if a < b >> c { 1 } else { 0 }\n"
                    "}") == 1


def test_a_shift_argument_still_reaches_the_callee():
    source = ("fn g(p: bool, q: int) -> int { q }\n"
              "fn main() -> int {\n"
              "    let a = 1; let b = 2; let c = 8; let d = 1;\n"
              "    g(a < b, c >> d)\n"
              "}")
    assert run_main(source) == 4


def test_nested_generics_still_close_on_the_doubled_bracket():
    """The case the old behaviour existed for.  Nesting is counted with
    `depth`, so the outermost bracket closes on the SECOND `>` and the new
    guard never fires for it."""
    types = lex_types("fn f(q: Pair<Pair<int>>) -> int { 0 }")
    assert "SHR" not in types
    assert types.count("RGENERIC") == 2

    types = lex_types("let m: Map<String, Vec<Int>> = x;")
    assert "SHR" not in types
    assert types.count("LGENERIC") == types.count("RGENERIC") == 2


def test_a_nested_generic_type_still_compiles():
    source = ("struct Pair<A, B> { x: A, y: B }\n"
              "fn take(p: Pair<int, Pair<int, int>>) -> int { 0 }\n"
              "fn main() -> int { 0 }")
    compiles_clean(source)
    assert run_main(source) == 0


def test_the_opener_guard_is_untouched():
    """`a << b >> (c)` has balanced angle counts and a `(` follow — the shape
    Pass B's scan accepts — and stays two shifts."""
    assert lex_types("a << b >> (c)")[:5] == [
        "IDENTIFIER", "SHL", "IDENTIFIER", "SHR", "LPAREN"]


def test_a_spaced_pair_of_angle_brackets_is_still_not_a_shift():
    assert lex_types("a > > b") == [
        "IDENTIFIER", "GREATER", "GREATER", "IDENTIFIER"]


def test_a_real_generic_call_still_lexes_as_a_generic():
    assert lex_types("identity<Int>(3)") == [
        "IDENTIFIER", "LGENERIC", "IDENTIFIER", "RGENERIC", "LPAREN",
        "NUMBER", "RPAREN"]
