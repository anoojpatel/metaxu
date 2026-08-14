"""Adversarial review round 6: four confirmed findings, pinned.

Each test reproduces the reported symptom through parsed source (never a
hand-built HIR/MIR fixture) and asserts the fixed behaviour.

1. f-string segment parsing registered its synthetic wrapper source under
   the ENCLOSING file's path, so every later diagnostic for that file lost
   its caret excerpt (or quoted `fn __fstring_expr__() { x }` as line 1).
2. `InterpError.locate` rewrote `args[0]`, so the string a user's
   `catch e` binds gained " [in function 'boom' declared at /abs/path:2:1]"
   — a language-visible value change that also leaked host paths.
3. Nested `fn` declarations were invisible to the module resolver, so a
   nested `fn len` had its own call sites rewritten to `__builtin$len`:
   the same source returned 99 without an import and raised
   "len: unsupported receiver type 'Int'" with one.
4. F-string interpolation synthesized a BARE `to_string(...)` call, so a
   user `fn to_string` captured every `f"{x}"` — contradicting the
   "compiler-synthesized builtin calls cannot be rebound" invariant of
   docs/name_precedence.md.

The last test is the systematic form of finding 4: a static scan of the
front end for synthesized calls to builtin names that lack the marker.
"""
from __future__ import annotations

import ast as pyast
import pathlib
import shutil

import pytest

from metaxu.compiler.hir import (BUILTIN_CALL_PREFIX, BUILTIN_FUNCTION_NAMES,
                                 HIRBuilder)
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import InterpError, MirInterpreter, UNIT
from metaxu.compiler.pipeline import (build_context_from_source,
                                      run_pipeline_from_source)
from metaxu.compiler.shared_parser import shared_parser
from metaxu.errors import get_source_text
import metaxu.metaxu_ast as fast
import metaxu.parser as parser_mod

_SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]   # .../src


def run(source: str, fn: str = "main", args: list | None = None,
        file_path: str = "<mem>"):
    """parse -> ... -> interpreter (the project's mandated test path)."""
    ctx = build_context_from_source(source, file_path=file_path)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    interp.register_builtin("print", lambda *a: UNIT)
    return interp.call(fn, args or [])


def mir_callees(source: str) -> list[str]:
    ctx = build_context_from_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    names: list[str] = []
    for f in lower_hir_to_mir(hir):
        for b in f.blocks:
            for op in b.ops:
                if (op[0] == "let" and isinstance(op[2], tuple)
                        and op[2][0] == "call"):
                    names.append(op[2][1])
    return names


# ======================================================================
# Finding 1: the f-string segment parse must not clobber the file's
# registered source text.
# ======================================================================

FSTRING_DIAG_SRC = '''fn main() -> int {
    let n = 5
    let msg = f"n is {n}"
    let bad = 1 + "oops"
    return 0
}
'''


def test_fstring_does_not_clobber_the_files_registered_source():
    """Parsing a file with an f-string leaves that file's source registered."""
    shared_parser().parse(FSTRING_DIAG_SRC, file_path="round6_fstr.mx")
    assert get_source_text("round6_fstr.mx") == FSTRING_DIAG_SRC


def test_gate_example_with_an_fstring_keeps_its_registered_source():
    """The originally reported reproduction: examples/effects.mx contains an
    f-string, and parsing it used to leave `fn __fstring_expr__() { x }`
    registered as that file's source."""
    path = _SRC_ROOT.parent / "examples" / "effects.mx"
    assert path.is_file()
    text = path.read_text()
    assert 'f"' in text, "reproduction relies on effects.mx having an f-string"
    shared_parser().parse(text, file_path=str(path))
    assert get_source_text(str(path)) == text


def test_diagnostic_after_an_fstring_excerpts_the_real_source_line():
    """A diagnostic elsewhere in a file containing an f-string shows the
    REAL line, not the synthetic `fn __fstring_expr__() { n }` wrapper."""
    with pytest.raises(Exception) as ei:
        run_pipeline_from_source(FSTRING_DIAG_SRC, file_path="round6_diag.mx")
    msg = str(ei.value)
    assert "round6_diag.mx:4:" in msg, msg
    assert '  4 |     let bad = 1 + "oops"' in msg, msg
    assert "__fstring_expr__" not in msg, msg


def test_fstring_segment_source_is_registered_under_a_synthetic_key():
    """The wrapper IS still registered (segment diagnostics keep excerpts) —
    just never under a real file's path."""
    shared_parser().parse(FSTRING_DIAG_SRC, file_path="round6_key.mx")
    wrapper = get_source_text("<fstring in round6_key.mx>")
    assert wrapper is not None and wrapper.startswith("fn __fstring_expr__()")
    assert get_source_text("round6_key.mx") == FSTRING_DIAG_SRC


# ======================================================================
# Finding 2: the value `catch` binds is the plain message; the compiler's
# function/location context lives in the DIAGNOSTIC only.
# ======================================================================

CATCH_SRC = """
effect Parser { parse(input: string) -> int }

fn boom() -> int {
    perform Parser.parse("x")
}

fn main() -> string {
    try {
        boom();
        "ok"
    } catch e {
        e
    }
}
"""


def test_caught_value_is_the_plain_message():
    caught = run(CATCH_SRC, file_path="/abs/host/path/round6.mx")
    assert caught == "No handler for effect 'Parser'"
    # The three things that must never reach a user value:
    assert "in function" not in caught
    assert "declared at" not in caught
    assert "/abs/host/path" not in caught


def test_caught_value_does_not_depend_on_the_host_path():
    """Same program, two compilation paths -> identical caught value."""
    a = run(CATCH_SRC, file_path="/one/place/round6.mx")
    b = run(CATCH_SRC, file_path="/somewhere/else/entirely/round6.mx")
    assert a == b


def test_developer_diagnostic_still_names_the_function():
    """An UNCAUGHT failure still reports the innermost function and its
    declaration site — that context moved out of the value, not away."""
    with pytest.raises(InterpError) as ei:
        run(CATCH_SRC.replace("""    try {
        boom();
        "ok"
    } catch e {
        e
    }""", "    boom(); \"ok\""), file_path="round6_uncaught.mx")
    exc = ei.value
    assert "in function 'boom'" in str(exc)
    assert "round6_uncaught.mx:" in str(exc)
    # ...while `.message` stays the plain, language-visible text.
    assert exc.message == "No handler for effect 'Parser'"
    # and `args` was not rewritten behind the raiser's back
    assert exc.args[0] == exc.message


# ======================================================================
# Finding 3: nested `fn` declarations are module-provided names.
# ======================================================================

NESTED_LEN_BODY = """
fn main() -> int {
    fn len(n: int) -> int {
        return 99
    }
    return len(1)
}
"""


def test_nested_fn_len_behaves_the_same_with_and_without_an_import():
    without = run(NESTED_LEN_BODY)
    with_import = run("import std.math\n" + NESTED_LEN_BODY)
    assert without == 99
    assert with_import == 99


def test_nested_fn_call_is_not_rewritten_to_the_builtin():
    callees = mir_callees("import std.math\n" + NESTED_LEN_BODY)
    assert "len" in callees
    assert f"{BUILTIN_CALL_PREFIX}len" not in callees


def test_nested_fn_does_not_capture_method_position():
    """A nested `fn len` still must not hijack `v.len()` — method position
    resolves impl -> builtin -> plain fn (docs/name_precedence.md §2)."""
    assert run("""
import std.math

fn main() -> int {
    fn len(n: int) -> int {
        return 99
    }
    let v = Vec<int>::new();
    v.push(1);
    v.push(2);
    return v.len() + len(0)
}
""") == 101


def test_unqualified_builtin_in_an_imported_module_still_binds_lexically():
    """The nested-function exemption must not re-open the hole section 3 of
    docs/name_precedence.md closed: an entry `fn len` still does not capture
    std/vec.mx's own bare `len(...)` calls."""
    callees = mir_callees("""
import std.vec

fn len(n: int) -> int { return 99 }

fn main() -> int {
    return len(1)
}
""")
    assert "len" in callees                       # the entry program's own call
    assert f"{BUILTIN_CALL_PREFIX}len" in callees  # std/vec.mx's bare len(...)


# ======================================================================
# Finding 4: f-string interpolation is a compiler-synthesized builtin call.
# ======================================================================

USER_TO_STRING = """
fn to_string(x: int) -> string {
    return "USER"
}

fn main() -> string {
    let x = 5
    return f"{x}"
}

fn direct() -> string {
    return to_string(7)
}
"""


def test_user_to_string_does_not_capture_fstring_interpolation():
    assert run(USER_TO_STRING) == "5"


def test_user_to_string_is_still_directly_callable():
    assert run(USER_TO_STRING, fn="direct") == "USER"


def test_fstring_synthesizes_a_marked_call():
    module = shared_parser().parse('fn f() -> string { return f"{1}" }',
                                   file_path="round6_marker.mx")
    seen: list = []

    def walk(node):
        if not isinstance(node, fast.Node) or any(n is node for n in seen):
            return
        seen.append(node)
        for attr, value in vars(node).items():
            if attr in ("parent", "scope", "location"):
                continue
            for item in (value if isinstance(value, (list, tuple)) else [value]):
                walk(item)

    walk(module)
    names = [n.name for n in seen if isinstance(n, fast.FunctionCall)]
    assert f"{BUILTIN_CALL_PREFIX}to_string" in names
    assert "to_string" not in names


@pytest.mark.skipif(shutil.which("clang") is None,
                    reason="clang is not installed")
def test_user_to_string_and_fstring_agree_natively(tmp_path):
    """The marker must mean the same thing on the native backend: the
    f-string prints the builtin's rendering, the direct call the user's."""
    from metaxu.compiler.tests.test_codegen_llvm import \
        assert_native_matches_interp
    assert_native_matches_interp("""
fn to_string(x: int) -> string {
    return "USER"
}

fn main() -> int {
    let x = 5;
    print(f"{x}");
    print(to_string(7));
    return 0
}
""", tmp_path)


def test_parser_marker_matches_the_hir_marker():
    """The parser spells the marker itself (it must not import the HIR
    builder); pin the two constants equal so they cannot drift."""
    assert parser_mod.BUILTIN_CALL_PREFIX == BUILTIN_CALL_PREFIX


# ======================================================================
# Systematic audit: every compiler-SYNTHESIZED call to a builtin name must
# carry the marker.  Finding 4 was the last unmarked one; this scan is what
# keeps the invariant from having to be rediscovered case by case.
# ======================================================================

# Front-end files that build call nodes out of thin air: the parser
# (f-strings, print), the desugar passes, HIR lowering (unary ops, for-loop
# bounds, list/vector literals, index/slice/range/zip/cast intrinsics) and
# MIR lowering.
_SYNTHESIS_SITES = (
    "metaxu/parser.py",
    "metaxu/compiler/desugar.py",
    "metaxu/compiler/hir.py",
    "metaxu/compiler/lower_hir_to_mir.py",
    "metaxu/compiler/module_loader.py",
)

# Constructors and local helpers whose FIRST positional argument (or
# `name=`/`method=` keyword) is a callee name.  `mk_call` is hir.py's
# local list-literal helper; the intrinsic-coverage test below fails if
# synthesis ever moves to a helper this set does not know about.
_CALL_CTORS = frozenset({"FunctionCall", "QualifiedFunctionCall", "MethodCall",
                         "mk_call"})
_CALLEE_KWARGS = frozenset({"callee", "name", "method"})


def _synthesized_callee_literals(path: pathlib.Path):
    """Every string LITERAL used as a callee name in `path`, with its line."""
    tree = pyast.parse(path.read_text(), filename=str(path))
    out: list[tuple[int, str]] = []
    for node in pyast.walk(tree):
        if not isinstance(node, pyast.Call):
            continue
        func = node.func
        fname = (func.attr if isinstance(func, pyast.Attribute)
                 else func.id if isinstance(func, pyast.Name) else None)
        if fname in _CALL_CTORS and node.args:
            first = node.args[0]
            if isinstance(first, pyast.Constant) and isinstance(first.value, str):
                out.append((node.lineno, first.value))
        for kw in node.keywords:
            if (kw.arg in _CALLEE_KWARGS
                    and isinstance(kw.value, pyast.Constant)
                    and isinstance(kw.value.value, str)):
                if fname in _CALL_CTORS or kw.arg == "callee":
                    out.append((node.lineno, kw.value.value))
    return out


def test_no_synthesized_call_targets_an_unmarked_builtin_name():
    """A synthesized call to a builtin name must be spelled
    `BUILTIN_CALL_PREFIX + name` (or be a reserved `__`-prefixed intrinsic).

    A bare literal there means user code of that name silently captures a
    piece of surface syntax — exactly finding 4.
    """
    offenders: list[str] = []
    for rel in _SYNTHESIS_SITES:
        path = _SRC_ROOT / rel
        assert path.is_file(), f"audit target missing: {path}"
        for lineno, literal in _synthesized_callee_literals(path):
            if literal.startswith("__"):
                continue            # reserved compiler namespace
            if literal in BUILTIN_FUNCTION_NAMES:
                offenders.append(f"{rel}:{lineno}: bare callee {literal!r}")
    assert not offenders, (
        "compiler-synthesized calls to builtin names must carry the "
        f"{BUILTIN_CALL_PREFIX!r} marker (docs/name_precedence.md):\n  "
        + "\n  ".join(offenders))


def test_the_audit_scan_actually_sees_synthesized_callees():
    """Guard against the scan silently matching nothing (a green test that
    checks nothing is worse than no test).

    The reserved intrinsics of docs/name_precedence.md §4 are all spelled as
    string literals at their synthesis sites; if a new call-building helper
    appears that `_CALL_CTORS`/`_CALLEE_KWARGS` does not know about, some of
    them stop being visible and this fails — which is the signal that the
    marker audit above has gone blind.
    """
    literals: set[str] = set()
    for rel in _SYNTHESIS_SITES:
        literals |= {lit for _, lit in
                     _synthesized_callee_literals(_SRC_ROOT / rel)}
    assert {"__index_get", "__index_set", "__index_store", "__slice_get",
            "__range", "__zip", "__cast",
            "__vec_lit", "__vec_dim", "__vec_zeros", "__vec_filled",
            "__vec_comprehension",
            "__list_lit", "__list_concat"} <= literals
