"""std.path (Metaxu) against Python's posixpath, the oracle.

One generated program prints join, dirname, basename, normalize,
is_absolute and relpath over a table of paths that covers the corner
cases (empty, dots, repeated and leading slashes, escaping `..`); the
transcript must match posixpath line for line on the interpreter and
then natively.  relpath is only compared for pairs that are both
absolute or both relative (the module's stated limit: no working
directory to resolve against).
"""
from __future__ import annotations

import posixpath
import shutil

import pytest

from metaxu.compiler.tests.test_codegen_llvm import interp_run, llvm_from_source

needs_clang = pytest.mark.skipif(shutil.which("clang") is None, reason="clang is not installed")

PATHS = ["", ".", "..", "/", "//", "///", "a", "a/", "/a", "//a", "///a/b", "a/b", "a/b/",
         "a//b", "./a//b/", "a/./b", "a/b/../c", "a/..", "/../a", "../../a", "a/b/c/../../..",
         "a/b/c/../../../..", "/a/b/", ".hidden/..", "x/y/z", "/x/y/z", "a/b/c", "a/c"]
JOINS = [("a", "b"), ("a/", "b"), ("/a", "b"), ("a", "/b"), ("", "b"), ("a", ""), ("", ""),
         ("a/b", "c/d"), ("/", "a"), ("a", ".")]


def _rel_pairs():
    # Both absolute or both relative (the module's limit), and neither side
    # escaping upward: posixpath resolves a relative `..` against the real
    # working directory, which std.path has no access to, so those answers
    # legitimately differ.
    pairs = []
    for p in PATHS:
        for s in PATHS:
            if not p or not s or posixpath.isabs(p) != posixpath.isabs(s):
                continue
            if any(posixpath.normpath(x).startswith("..") for x in (p, s)):
                continue
            pairs.append((p, s))
    return pairs


def program() -> str:
    lines = ["from std.path import join, dirname, basename, normalize, relpath, is_absolute;",
             "fn main() -> int {"]
    for a, b in JOINS:
        lines.append(f'    print(join("{a}", "{b}") + "|");')
    for p in PATHS:
        lines.append(f'    print(normalize("{p}") + "|" + dirname("{p}") + "|" + basename("{p}") + "|" + is_absolute("{p}").to_string());')
    for p, s in _rel_pairs():
        lines.append(f'    print(relpath("{p}", "{s}"));')
    lines.append("    0")
    lines.append("}")
    return "\n".join(lines) + "\n"


def oracle() -> list[str]:
    out = []
    for a, b in JOINS:
        out.append(posixpath.join(a, b) + "|")
    for p in PATHS:
        out.append("|".join([posixpath.normpath(p), posixpath.dirname(p), posixpath.basename(p),
                             "1" if posixpath.isabs(p) else "0"]))
    for p, s in _rel_pairs():
        out.append(posixpath.relpath(p, s))
    return out


@pytest.fixture(scope="module")
def transcript() -> list[str]:
    _result, out = interp_run(program())
    return out.rstrip("\n").split("\n")


def test_interpreter_matches_posixpath(transcript):
    assert transcript == oracle()


def test_relpath_refuses_mixed_paths():
    # An assertion failure is not a catchable error: the program stops.
    with pytest.raises(AssertionError, match="both paths must be absolute or both relative"):
        interp_run("""
from std.path import relpath;
fn main() -> int {
    print(relpath("/a", "b"));
    0
}
""")


def test_native_is_placeholder_free():
    assert "placeholder -- unsupported" not in llvm_from_source(program())


@needs_clang
def test_native_prints_what_the_interpreter_printed(tmp_path, transcript):
    from metaxu.compiler.llvm_run import compile_and_run
    exit_code, stdout = compile_and_run(llvm_from_source(program()), "main",
                                        workdir=str(tmp_path), timeout=300)
    assert exit_code == 0
    assert stdout.rstrip("\n").split("\n") == transcript
