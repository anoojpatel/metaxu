"""F-string interpolation: parse-time desugaring to concat + to_string.

`f"a {x} b"` desugars IN THE PARSER (Parser._desugar_fstring,
src/metaxu/parser.py) into `"a " + to_string(x) + " b"`: literal segments
stay string Literals, `{expr}` segments are parsed as ordinary Metaxu
expressions by a dedicated cached Parser and wrapped in the `to_string`
builtin (identity on strings).  `{{`/`}}` escape to literal braces.  Empty
`{}`, unbalanced braces, unparsable segments, and non-expression segments
are clear CompileErrors — never a silent literal fallback.

Because the desugared form is plain `+` and `to_string` — both native since
LLVM increment 5 — the interpreter and native engines agree with no codegen
change; the differential test at the bottom pins that.

All tests go through parsed source, per the project convention.
"""
from __future__ import annotations

import shutil

import pytest

from metaxu.compiler.codegen_llvm import emit_llvm
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.llvm_run import compile_and_run
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import UNIT, MirInterpreter
from metaxu.compiler.pipeline import build_context_from_source
from metaxu.errors import CompileError

needs_clang = pytest.mark.skipif(
    shutil.which("clang") is None, reason="clang is not installed")


def call(source: str, fn: str = "main"):
    ctx = build_context_from_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    return interp.call(fn, [])


# ---------------------------------------------------------------------------
# Interpolation semantics (interpreter, parsed source)
# ---------------------------------------------------------------------------

def test_literal_only_fstring_is_plain_string():
    assert call('fn main() -> string { f"just text" }') == "just text"


def test_single_int_expr():
    assert call('''
        fn main() -> string {
            let x = 42;
            f"Counter value: {x}"
        }
    ''') == "Counter value: 42"


def test_multiple_exprs():
    assert call('''
        fn main() -> string {
            let a = 1;
            let b = 2;
            f"a={a} b={b}"
        }
    ''') == "a=1 b=2"


def test_arithmetic_inside_braces():
    assert call('''
        fn main() -> string {
            let a = 2;
            let b = 3;
            f"{a} * {b} = {a * b}"
        }
    ''') == "2 * 3 = 6"


def test_string_variable_segment():
    # to_string of a string is identity in both engines.
    assert call('''
        fn main() -> string {
            let s = "world";
            f"hello {s}"
        }
    ''') == "hello world"


def test_float_expr():
    assert call('''
        fn main() -> string {
            let x = 1.5;
            f"float: {x}"
        }
    ''') == "float: 1.5"


def test_expr_only_fstring():
    # No literal segments at all: the result is just to_string(expr).
    assert call('fn main() -> string { let x = 7; f"{x}" }') == "7"


def test_double_brace_escapes():
    assert call('fn main() -> string { f"braces {{x}} stay" }') == "braces {x} stay"
    assert call('''
        fn main() -> string {
            let x = 1;
            f"{{{x}}}"
        }
    ''') == "{1}"


def test_empty_fstring():
    assert call('fn main() -> string { f"" }') == ""


# ---------------------------------------------------------------------------
# Errors: clear CompileError, never silent literal fallback
# ---------------------------------------------------------------------------

def test_empty_braces_is_compile_error():
    with pytest.raises(CompileError, match="empty expression"):
        call('fn main() -> string { f"value: {}" }')


def test_whitespace_only_braces_is_compile_error():
    with pytest.raises(CompileError, match="empty expression"):
        call('fn main() -> string { f"value: {  }" }')


def test_unparsable_segment_is_compile_error_naming_segment():
    with pytest.raises(CompileError, match=r"cannot parse expression segment '\{1 \+\}'"):
        call('fn main() -> string { f"bad {1 +} seg" }')


def test_unterminated_brace_is_compile_error():
    with pytest.raises(CompileError, match="unterminated"):
        call('fn main() -> string { f"oops {x" }')


def test_lone_close_brace_is_compile_error():
    with pytest.raises(CompileError, match="single '}'"):
        call('fn main() -> string { f"oops } here" }')


def test_statement_segment_is_compile_error():
    with pytest.raises(CompileError, match="not a single expression"):
        call('fn main() -> string { f"bad {let x = 1} seg" }')


# ---------------------------------------------------------------------------
# Native differential: both engines see the same desugared concat chain
# ---------------------------------------------------------------------------

@needs_clang
def test_native_fstring_matches_interpreter(tmp_path):
    source = '''
fn main() -> int {
    let x = 42;
    print(f"Counter value: {x}");
    let a = 2;
    let b = 3;
    print(f"{a} * {b} = {a * b}");
    let s = "str";
    print(f"a {s} and {{escaped}} braces");
    0
}
'''
    ctx = build_context_from_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    mir = lower_hir_to_mir(hir)

    interp = MirInterpreter()
    interp.load(mir)
    lines: list[str] = []
    interp.register_builtin(
        "print", lambda *a: (lines.append(" ".join(str(x) for x in a)), UNIT)[1])
    interp.call("main", [])
    expected = "".join(line + "\n" for line in lines)

    ir = emit_llvm(mir)
    exit_code, stdout = compile_and_run(ir, "main", workdir=str(tmp_path))
    assert exit_code == 0
    assert stdout == expected
    assert stdout == ("Counter value: 42\n"
                      "2 * 3 = 6\n"
                      "a str and {escaped} braces\n")
