"""std.json (Metaxu) against Python's json.dumps, the oracle.

A generated program builds a tree of every value shape and prints it
compact and pretty; Python renders the same tree with
separators=(",", ":") and with indent=2, both ensure_ascii=False; the
transcripts must match, then the native binary must match the
interpreter.
"""
from __future__ import annotations

import json
import shutil

import pytest

from metaxu.compiler.tests.test_codegen_llvm import interp_run, llvm_from_source

needs_clang = pytest.mark.skipif(shutil.which("clang") is None, reason="clang is not installed")

TREE = {
    "name": "glade",
    "n": -12,
    "zero": 0,
    "yes": True,
    "no": False,
    "nothing": None,
    "quote": 'say "hi"',
    "slash": "a\\b/c",
    "controls": "tab\tnl\ncr\rbs\bff\fbell\x07nul-free\x1f",
    "unicode": "héllo wörld ✓",
    "empty_obj": {},
    "empty_arr": [],
    "list": [1, "two", None, [3, [4]], {"k": "v"}],
    "nested": {"a": {"b": {"c": [True]}}},
}


def _mx_literal(s: str) -> str:
    out = '"'
    for ch in s:
        if ch == '"':
            out += '\\"'
        elif ch == "\\":
            out += "\\\\"
        elif ch == "\n":
            out += "\\n"
        elif ch == "\t":
            out += "\\t"
        elif ch == "\r":
            out += "\\r"
        elif ord(ch) < 32:
            # Metaxu string literals have no \x escape: build the control
            # character from its byte at run time.
            out += '" + byte_str(%d) + "' % ord(ch)
        else:
            out += ch
    return out + '"'


def _build(v, counter: list[int]) -> tuple[list[str], str]:
    """(statements, expression) constructing `v` as a Json value."""
    if v is None:
        return [], "JNull"
    if isinstance(v, bool):
        return [], "JBool(true)" if v else "JBool(false)"
    if isinstance(v, int):
        return [], f"JInt({v})" if v >= 0 else f"JInt(0 - {-v})"
    if isinstance(v, str):
        return [], f"JStr({_mx_literal(v)})"
    counter[0] += 1
    name = f"v{counter[0]}"
    stmts = [f"    let @mut {name} = Vec.new();"]
    if isinstance(v, list):
        for item in v:
            s, e = _build(item, counter)
            stmts += s
            stmts.append(f"    {name}.push({e});")
        return stmts, f"JArr({name})"
    for k, item in v.items():
        s, e = _build(item, counter)
        stmts += s
        stmts.append(f"    {name}.push(member({_mx_literal(k)}, {e}));")
    return stmts, f"JObj({name})"


def program() -> str:
    stmts, expr = _build(TREE, [0])
    lines = ["from std.json import Json, member, to_json, to_json_pretty;",
             "fn byte_str(b: int) -> string { let @mut v = Vec.new(); v.push(b); v.from_bytes() }",
             "fn main() -> int"]
    lines[-1] += " {"
    lines += stmts
    lines.append(f"    let tree = {expr};")
    lines.append("    print(to_json(tree));")
    lines.append("    print(to_json_pretty(tree));")
    lines.append('    print(to_json(JStr("")));')
    lines.append("    print(to_json(JArr(Vec.new())));")
    lines.append("    0")
    lines.append("}")
    return "\n".join(lines) + "\n"


def oracle() -> str:
    return "\n".join([
        json.dumps(TREE, separators=(",", ":"), ensure_ascii=False),
        json.dumps(TREE, indent=2, ensure_ascii=False),
        '""',
        "[]",
    ]) + "\n"


@pytest.fixture(scope="module")
def transcript() -> str:
    _result, out = interp_run(program())
    return out


def test_interpreter_matches_json_dumps(transcript):
    assert transcript == oracle()


def test_native_is_placeholder_free():
    assert "placeholder -- unsupported" not in llvm_from_source(program())


@needs_clang
def test_native_prints_what_the_interpreter_printed(tmp_path, transcript):
    from metaxu.compiler.llvm_run import compile_and_run
    exit_code, stdout = compile_and_run(llvm_from_source(program()), "main",
                                        workdir=str(tmp_path), timeout=300)
    assert exit_code == 0
    assert stdout == transcript
