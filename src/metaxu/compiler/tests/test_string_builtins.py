"""The linear string builtins against Python's str methods, the oracle.

`split`, `find`, `replace`, `trim` and `join` are builtin methods on
strings (and `join` on a Vec of strings), Python-backed in the
interpreter and `mx_str_*` in the C runtime natively. One generated
Metaxu program prints an answer per case; the same cases go through
Python; the transcripts must match line for line, and the native
backend then has to print exactly what the interpreter printed,
catchable diagnostics included.
"""
from __future__ import annotations

import random
import shutil

import pytest

from metaxu.compiler.tests.test_codegen_llvm import interp_run, llvm_from_source

needs_clang = pytest.mark.skipif(shutil.which("clang") is None, reason="clang is not installed")

ALPHABET = "ab, -=\t"  # no \n or \r: the transcript is line-split and
                        # the native run goes through text-mode newline translation
SEPARATORS = [",", ", ", "-", "=", " = ", "ab", "b", "--", " "]


def _mx_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\t", "\\t").replace("\n", "\\n") + '"'


def _cases(seed: int = 11):
    rng = random.Random(seed)
    strings = ["", "a", "abc", " a b ", "a,b,,c", ",", ",,", "a--b--", "\t x \t", "  ",
               "name = metaxu", "mississippi"]
    for _ in range(80):
        n = rng.randint(0, 9)
        strings.append("".join(rng.choice(ALPHABET) for _ in range(n)))
    return strings


def program() -> str:
    lines = ["fn main() -> int {"]
    for s in _cases():
        ms = _mx_str(s)
        lines.append(f"    print({ms}.trim());")
        for sep in SEPARATORS:
            msep = _mx_str(sep)
            lines.append(f"    print({ms}.find({msep}));")
            lines.append(f"    print(len({ms}.split({msep})));")
            lines.append(f"    print({ms}.split({msep}).join(\"|\"));")
            lines.append(f"    print({ms}.replace({msep}, \"<>\"));")
    # plain-call forms and the two catchable diagnostics
    lines.append('    print(find("hello", "l"));')
    lines.append('    print(join(split("x-y-z", "-"), "+"));')
    lines.append('    print(trim("  t  ") + replace("aXa", "X", "Y"));')
    lines.append('    print(try { let p = "a".split(""); "no error" } catch e { e });')
    lines.append('    print(try { "a".replace("", "b") } catch e { e });')
    lines.append("    0")
    lines.append("}")
    return "\n".join(lines) + "\n"


def oracle() -> list[str]:
    out: list[str] = []
    for s in _cases():
        out.append(s.strip(" \t\n\r"))
        for sep in SEPARATORS:
            out.append(str(s.find(sep)))
            parts = s.split(sep)
            out.append(str(len(parts)))
            out.append("|".join(parts))
            out.append(s.replace(sep, "<>"))
    out.append(str("hello".find("l")))
    out.append("+".join("x-y-z".split("-")))
    out.append("t" + "aYa")
    out.append("split: empty separator")
    out.append("replace: empty pattern")
    return out


@pytest.fixture(scope="module")
def transcript() -> list[str]:
    _result, out = interp_run(program())
    return out.rstrip("\n").split("\n")


def test_interpreter_matches_python(transcript):
    assert transcript == oracle()


def test_type_errors_are_loud():
    for expr, msg in [
        ('(5).trim()', "trim: expected a string receiver, got 'Int'"),
        ('"a".split(3)', "split: expected a string separator, got 'Int'"),
        ('"a".find(1.5)', "find: expected a string argument, got 'Float'"),
        ('"a".replace("a", 1)', "replace: expected a string replacement, got 'Int'"),
        ('"a".join(",")', "join: expected a Vec receiver, got 'String'"),
        ('[1, 2].join(",")', "join: element 0 is not a string, got 'Int'"),
    ]:
        _res, out = interp_run(f"""
fn main() -> int {{
    print(try {{ let r = {expr}; "no error" }} catch e {{ e }});
    0
}}
""")
        assert out.strip() == msg, expr


def test_user_function_wins_plain_but_not_in_method_position():
    # docs/name_precedence.md: a plain call prefers the user function; the
    # method form is always the builtin.
    _res, out = interp_run("""
fn trim(s: string) -> string { "user" }
fn main() -> int {
    print(trim("  x  "));
    print("  x  ".trim());
    0
}
""")
    assert out == "user\nx\n"


def test_native_lowers_every_builtin_without_placeholders():
    ir = llvm_from_source(program())
    assert "placeholder -- unsupported" not in ir
    for sym in ("mx_str_find", "mx_str_split", "mx_str_replace",
                "mx_str_trim", "mx_str_join"):
        assert f"@{sym}(" in ir


@needs_clang
def test_native_prints_what_the_interpreter_printed(tmp_path, transcript):
    from metaxu.compiler.llvm_run import compile_and_run
    exit_code, stdout = compile_and_run(llvm_from_source(program()), "main",
                                        workdir=str(tmp_path), timeout=300)
    assert exit_code == 0
    assert stdout.rstrip("\n").split("\n") == transcript


def test_std_parse_and_std_string_delegate(tmp_path):
    # The Metaxu wrappers keep their names and now accept multi-character
    # separators; the std helpers stay usable from the interpreter and,
    # under clang, natively.
    src = """
from std.parse import trim, split_on;
from std.string import join;
fn main() -> int {
    let parts = split_on("k => v => w", " => ");
    print(len(parts));
    print(join(parts, "/"));
    print(trim("\\t padded \\n"));
    print(len(split_on("", ",")));
    0
}
"""
    _res, out = interp_run(src)
    assert out == "3\nk/v/w\npadded\n1\n"
    if shutil.which("clang"):
        from metaxu.compiler.llvm_run import compile_and_run
        exit_code, stdout = compile_and_run(
            llvm_from_source(src), "main", workdir=str(tmp_path))
        assert exit_code == 0 and stdout == out


# --- bytes: to_bytes / from_bytes ---------------------------------------------

_BYTES_SRC = """
fn main() -> int {
    let b = "héllo".to_bytes();
    print(len(b));
    print(b[0]);
    print(b[1]);
    print(b.from_bytes());
    print(from_bytes(to_bytes("abc")));
    let @mut bad = Vec.new();
    bad.push(104);
    bad.push(255);
    print(try { bad.from_bytes() } catch e { e });
    let @mut big = Vec.new();
    big.push(300);
    print(try { big.from_bytes() } catch e { e });
    let @mut nul = Vec.new();
    nul.push(65);
    nul.push(0);
    print(try { nul.from_bytes() } catch e { e });
    let @mut trunc = Vec.new();
    trunc.push(97);
    trunc.push(226);
    trunc.push(130);
    print(try { trunc.from_bytes() } catch e { e });
    let @mut sur = Vec.new();
    sur.push(237);
    sur.push(160);
    sur.push(128);
    print(try { sur.from_bytes() } catch e { e });
    let empty = Vec.new();
    print(len(empty.from_bytes()));
    0
}
"""

_BYTES_OUT = [
    "6", "104", "195", "héllo", "abc",
    "from_bytes: invalid UTF-8 at byte 1",
    "from_bytes: element 0 is not a byte (0..255): 300",
    "from_bytes: element 1 is NUL (a string cannot hold NUL)",
    "from_bytes: invalid UTF-8 at byte 1",
    "from_bytes: invalid UTF-8 at byte 0",
    "0",
]


def test_bytes_round_trip_and_diagnostics_on_the_interpreter():
    # The UTF-8 bytes of a string as a Vec of ints and back; a non-byte, a
    # NUL and each malformed UTF-8 shape (truncated sequence, surrogate)
    # raise, and the byte index matches CPython's decoder.
    _res, out = interp_run(_BYTES_SRC)
    assert out.rstrip("\n").split("\n") == _BYTES_OUT


@needs_clang
def test_native_bytes_match_the_interpreter(tmp_path):
    from metaxu.compiler.llvm_run import compile_and_run
    ir = llvm_from_source(_BYTES_SRC)
    assert "placeholder -- unsupported" not in ir
    exit_code, stdout = compile_and_run(ir, "main", workdir=str(tmp_path))
    assert exit_code == 0
    assert stdout.rstrip("\n").split("\n") == _BYTES_OUT
