"""Bitwise operators: `&`, `|`, `^`, `~`, `<<`, `>>`.

std/README.md item 13 recorded "no bitwise operators" as the gap keeping
`std/random.mx` on an LCG instead of a xorshift. This file pins the fix in
both directions, through parsed source only (the repo convention):

* SPELLING. The C/Rust spelling with Rust's precedence — shifts bind
  tighter than `&`, `&` tighter than `^`, `^` tighter than `|`, and all of
  them tighter than the comparisons, so `flags & MASK == 0` is
  `(flags & MASK) == 0` rather than C's famous mis-grouping.
  `^` and `~` were illegal characters, `|` in expression position was a
  syntax error, and `<<`/`>>` are synthesized by the lexer's Pass D from
  ADJACENT angle brackets Pass B did not claim for a generic argument
  list — so `Vec<Vec<int>>` keeps working and `a > > b` stays an error.
* SEMANTICS. i64 two's complement on both engines: `>>` is ARITHMETIC
  (sign-extending, LLVM `ashr`), `<<` WRAPS into i64 instead of growing a
  Python bignum, and a shift count outside 0..63 is a LOUD error rather
  than LLVM poison / an unbounded Python shift.
* TYPING. Int-only. A Float or String operand is a compile error through
  the existing literal-class conflict detection, exactly like `1 + "a"`.
* NATIVE. Differential tests prove interpreter == clang-compiled native,
  including negative operands and the xorshift `std.random` now uses.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.llvm_run import compile_and_run
from metaxu.compiler.codegen_llvm import emit_llvm
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import InterpError, MirInterpreter, UNIT
from metaxu.compiler.pipeline import (
    TypeCheckError, build_context_from_source, run_pipeline_ctx,
)
from metaxu.compiler.shared_parser import shared_parser
from metaxu.errors import CompileError
from metaxu.lexer import Lexer

needs_clang = pytest.mark.skipif(
    shutil.which("clang") is None, reason="clang is not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_main(source: str, entry: str = "main"):
    """Full strict pipeline, then execute `entry`; returns (result, prints)."""
    ctx = build_context_from_source(source, file_path="<mem>")
    run_pipeline_ctx(ctx)          # strict: raises on type/borrow errors
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    prints: list[str] = []
    interp.register_builtin(
        "print", lambda *a: (prints.append(" ".join(str(x) for x in a)), UNIT)[1])
    return interp.call(entry, []), prints


def lex_types(source: str) -> list[str]:
    lx = Lexer()
    lx.input(source)
    out: list[str] = []
    while True:
        tok = lx.token()
        if tok is None:
            return out
        out.append(tok.type)


def mir_from_source(source: str):
    ctx = build_context_from_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    return lower_hir_to_mir(hir)


def interp_run(source: str, entry: str = "main"):
    interp = MirInterpreter()
    interp.load(mir_from_source(source))
    out: list[str] = []

    def _print(*args):
        out.append(" ".join(str(a) for a in args))
        return UNIT

    interp.register_builtin("print", _print)
    interp.register_builtin("println", _print)
    return interp.call(entry, []), "".join(line + "\n" for line in out)


def assert_native_matches_interp(source: str, tmp_path, entry: str = "main"):
    """The differential assertion: clang-compiled result == interpreter."""
    result, expected_out = interp_run(source, entry)
    ir = emit_llvm(mir_from_source(source))
    exit_code, stdout = compile_and_run(ir, entry, workdir=str(tmp_path))
    assert stdout == expected_out
    if result is not UNIT and isinstance(result, (bool, int)):
        assert exit_code == int(result) % 256
    return ir


# ---------------------------------------------------------------------------
# 1. Lexing and spelling
# ---------------------------------------------------------------------------

def test_caret_and_tilde_are_tokens():
    """Both characters used to reach t_error, which is a loud LexError."""
    assert lex_types("a ^ b") == ["IDENTIFIER", "CARET", "IDENTIFIER"]
    assert lex_types("~a") == ["TILDE", "IDENTIFIER"]


def test_shifts_are_synthesized_from_adjacent_angle_brackets():
    assert lex_types("a << b") == ["IDENTIFIER", "SHL", "IDENTIFIER"]
    assert lex_types("a >> b") == ["IDENTIFIER", "SHR", "IDENTIFIER"]


def test_a_spaced_pair_of_angle_brackets_is_not_a_shift():
    """Adjacency is required, so the spelling of a shift is exactly `<<`.

    `a > > b` keeps two GREATER tokens and stays the syntax error it has
    always been — it must not quietly become a shift."""
    assert lex_types("a > > b") == ["IDENTIFIER", "GREATER", "GREATER",
                                    "IDENTIFIER"]
    with pytest.raises(CompileError):
        shared_parser().parse("fn main() -> int { let a = 1; a > > 2 }")


def test_nested_generics_still_close_with_two_angle_brackets():
    """The reason `>>` is NOT a lexer regex: it would have eaten these.

    Pass B retags a type-argument list's brackets before Pass D looks for
    adjacent pairs, so `Vec<Vec<int>>` never sees an SHR."""
    types = lex_types("fn f(q: Pair<Pair<int>>) -> int { 0 }")
    assert "SHR" not in types
    assert types.count("RGENERIC") == 2
    shared_parser().parse(
        "struct P<A> { x: A } fn f(q: P<P<int>>) -> int { 0 }")


def test_a_generic_list_never_opens_with_a_doubled_angle_bracket():
    """`a << b >> (c)` has balanced angle counts and a `(` follow, which is
    exactly the shape Pass B's generic scan accepts. Refusing to START a
    scan at an adjacent `<<` is what keeps it a shift expression."""
    types = lex_types("a << b >> (c)")
    assert types[:5] == ["IDENTIFIER", "SHL", "IDENTIFIER", "SHR", "LPAREN"]


# ---------------------------------------------------------------------------
# 2. Interpreter semantics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expr,want", [
    ("12 & 10", 8),
    ("12 | 10", 14),
    ("12 ^ 10", 6),
    ("~5", -6),
    ("~0", -1),
    ("1 << 10", 1024),
    ("1024 >> 3", 128),
    ("0 & 7", 0),
    ("255 & 15", 15),
])
def test_bitwise_on_positive_ints(expr, want):
    result, _ = run_main(f"fn main() -> int {{ {expr} }}")
    assert result == want


@pytest.mark.parametrize("expr,want", [
    # Negative operands: two's complement, and `>>` sign-extends (ashr).
    ("(0 - 1) & 255", 255),
    ("(0 - 8) >> 1", -4),
    ("(0 - 1) >> 63", -1),
    ("(0 - 1) ^ (0 - 1)", 0),
    ("(0 - 2) | 1", -1),
    ("~(0 - 1)", 0),
])
def test_bitwise_on_negative_ints(expr, want):
    result, _ = run_main(f"fn main() -> int {{ {expr} }}")
    assert result == want


def test_shift_left_wraps_into_i64_instead_of_growing_a_bignum():
    """Python's `<<` is unbounded; i64 wraps. Without the wrap the
    interpreter and native code would disagree — and a xorshift generator
    would diverge on its very first step."""
    result, _ = run_main("fn main() -> int { 1 << 63 }")
    assert result == -(2 ** 63)
    result, _ = run_main("fn main() -> int { let x = 1 << 62; x << 1 }")
    assert result == -(2 ** 63)


@pytest.mark.parametrize("expr,want", [
    # Rust precedence: shifts > & > ^ > | > comparisons.
    ("1 | 2 & 3", 3),          # 1 | (2 & 3) == 1 | 2 == 3
    ("1 ^ 3 & 1", 0),          # 1 ^ (3 & 1) == 1 ^ 1
    ("1 << 2 + 1", 8),         # 1 << (2 + 1)
    ("2 * 3 & 5", 4),          # (2 * 3) & 5 == 6 & 5
])
def test_precedence_between_bitwise_levels(expr, want):
    result, _ = run_main(f"fn main() -> int {{ {expr} }}")
    assert result == want


def test_bitwise_binds_tighter_than_comparison():
    """Rust's grouping, not C's: `6 & 3 == 2` is `(6 & 3) == 2` (true).
    Under C's precedence it would be `6 & (3 == 2)`."""
    result, _ = run_main("fn main() -> bool { 6 & 3 == 2 }")
    assert result is True


def test_parentheses_and_variables():
    result, _ = run_main(
        "fn main() -> int { let mask = 255; let v = 4095; (v >> 4) & mask }")
    assert result == 255


def test_bitwise_through_a_function_and_a_loop():
    """popcount, the classic `n & (n - 1)` loop — exercises the operators
    inside a real function body rather than a constant fold."""
    src = """
    fn popcount(n: int) -> int {
        let @mut x = n;
        let @mut c = 0;
        while x != 0 {
            x = x & (x - 1);
            c = c + 1
        }
        c
    }
    fn main() -> int { popcount(255) + popcount(1024) }
    """
    result, _ = run_main(src)
    assert result == 9


# ---------------------------------------------------------------------------
# 3. Loud failures (never a silent fallback)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expr", ["1 & 2.5", "2.5 | 1", "1 ^ 0.5"])
def test_a_float_operand_is_a_compile_error(expr):
    with pytest.raises(TypeCheckError) as exc:
        run_main(f"fn main() -> int {{ {expr} }}")
    assert "Float" in str(exc.value) and "Int" in str(exc.value)


@pytest.mark.parametrize("expr", ['1 | "a"', '"a" & 1', '"a" << 1'])
def test_a_string_operand_is_a_compile_error(expr):
    with pytest.raises(TypeCheckError) as exc:
        run_main(f"fn main() -> int {{ {expr} }}")
    assert "String" in str(exc.value) and "Int" in str(exc.value)


def test_complement_of_a_float_is_a_compile_error():
    with pytest.raises(TypeCheckError) as exc:
        run_main("fn main() -> int { ~1.5 }")
    assert "Float" in str(exc.value) and "Int" in str(exc.value)


@pytest.mark.parametrize("expr", ["1 << 64", "1 << 100", "1 >> 64"])
def test_a_shift_count_at_or_past_the_width_is_loud(expr):
    with pytest.raises(InterpError) as exc:
        run_main(f"fn main() -> int {{ {expr} }}")
    assert "out of range" in str(exc.value)


def test_a_negative_shift_count_is_loud():
    with pytest.raises(InterpError) as exc:
        run_main("fn main() -> int { 1 << (0 - 1) }")
    assert "out of range" in str(exc.value)


def test_a_dynamic_out_of_range_shift_count_is_loud():
    """The guard is a runtime check, not a constant-folding trick."""
    src = """
    fn shift(v: int, n: int) -> int { v << n }
    fn main() -> int { shift(1, 70) }
    """
    with pytest.raises(InterpError) as exc:
        run_main(src)
    assert "out of range" in str(exc.value)


# ---------------------------------------------------------------------------
# 4. Native differential (interpreter == clang-compiled native)
# ---------------------------------------------------------------------------

@needs_clang
def test_native_matches_interpreter_on_positive_operands(tmp_path):
    src = """
    fn main() -> int {
        print(12 & 10);
        print(12 | 10);
        print(12 ^ 10);
        print(~5);
        print(1 << 10);
        print(1024 >> 3);
        0
    }
    """
    ir = assert_native_matches_interp(src, tmp_path)
    assert "and i64" in ir and "or i64" in ir and "xor i64" in ir
    assert "shl i64" in ir and "ashr i64" in ir


@needs_clang
def test_native_matches_interpreter_on_negative_operands(tmp_path):
    """`>>` must be ARITHMETIC natively (ashr), or the two engines disagree
    on every negative value."""
    src = """
    fn main() -> int {
        print((0 - 1) & 255);
        print((0 - 8) >> 1);
        print((0 - 1) >> 63);
        print((0 - 2) | 1);
        print(~(0 - 1));
        print((0 - 1) ^ (0 - 1));
        0
    }
    """
    assert_native_matches_interp(src, tmp_path)


@needs_clang
def test_native_matches_interpreter_on_shift_left_wraparound(tmp_path):
    src = """
    fn main() -> int {
        print(1 << 63);
        print(3 << 62);
        print((1 << 62) << 1);
        0
    }
    """
    assert_native_matches_interp(src, tmp_path)


@needs_clang
def test_native_matches_interpreter_on_a_xorshift_step(tmp_path):
    """The generator std/random.mx now uses, run natively and interpreted."""
    src = """
    fn step(s: int) -> int {
        let a = s ^ (s << 13);
        let b = a ^ ((a >> 7) & 144115188075855871);
        b ^ (b << 17)
    }
    fn main() -> int {
        let @mut s = 88172645463325252;
        let @mut i = 0;
        while i < 8 {
            s = step(s);
            print(s);
            i = i + 1
        }
        0
    }
    """
    assert_native_matches_interp(src, tmp_path)


@needs_clang
def test_native_popcount_matches_interpreter(tmp_path):
    src = """
    fn popcount(n: int) -> int {
        let @mut x = n;
        let @mut c = 0;
        while x != 0 {
            x = x & (x - 1);
            c = c + 1
        }
        c
    }
    fn main() -> int { popcount(255) + popcount(1024) }
    """
    assert_native_matches_interp(src, tmp_path)


@needs_clang
def test_native_out_of_range_shift_aborts_like_the_interpreter(tmp_path):
    """LLVM's `shl` is POISON past the bit width, which would make the same
    program answer differently on the two engines. @mx_shift_check turns it
    into the interpreter's loud failure."""
    src = """
    fn shift(v: int, n: int) -> int { v << n }
    fn main() -> int { shift(1, 70) }
    """
    with pytest.raises(InterpError):
        run_main(src)
    ir = emit_llvm(mir_from_source(src))
    assert "@mx_shift_check" in ir
    exit_code, _stdout = compile_and_run(ir, "main", workdir=str(tmp_path))
    assert exit_code != 0, "an out-of-range shift must not run to completion"


@needs_clang
def test_native_shift_check_message_names_the_count(tmp_path):
    src = """
    fn shift(v: int, n: int) -> int { v >> n }
    fn main() -> int { shift(1, 99) }
    """
    ir = emit_llvm(mir_from_source(src))
    compile_and_run(ir, "main", workdir=str(tmp_path))   # builds prog.bin
    proc = subprocess.run([str(tmp_path / "prog.bin")],
                          capture_output=True, text=True)
    assert proc.returncode != 0
    assert "shift amount 99 out of range" in proc.stderr
