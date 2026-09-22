"""`metaxuc`, the command-line front door (compiler/cli.py).

The CLI is a thin layer over the same pipeline functions the rest of the
suite uses, so these tests pin the layer itself: argument handling, exit
codes, where output goes, and that a rejected program is rejected with
the same diagnostic the library raises.  Every test goes through source
files on disk, as a user would.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

from metaxu.compiler.cli import main

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

HELLO = """
fn main() -> int {
    let who = "Metaxu";
    print("hello, " + who);
    0
}
"""

SEVEN = """
fn main() -> int {
    print("seven");
    7
}
"""

TYPE_ERROR = """
fn main() -> int {
    print(1 + "a");
    0
}
"""

needs_clang = pytest.mark.skipif(shutil.which("clang") is None,
                                 reason="clang is not installed")


def _write(tmp_path, name, src):
    p = tmp_path / name
    p.write_text(src)
    return str(p)


def test_run_prints_and_returns_mains_value(tmp_path, capsys):
    assert main(["run", _write(tmp_path, "hello.mx", HELLO)]) == 0
    assert capsys.readouterr().out == "hello, Metaxu\n"
    assert main(["run", _write(tmp_path, "seven.mx", SEVEN)]) == 7
    assert capsys.readouterr().out == "seven\n"


def test_run_script_file_without_main(tmp_path, capsys):
    """A bare file of statements gets a synthesized main (chapter 1)."""
    path = _write(tmp_path, "script.mx", 'print("top level");\n')
    assert main(["run", path]) == 0
    assert capsys.readouterr().out == "top level\n"


def test_script_statements_next_to_main_are_refused(tmp_path, capsys):
    path = _write(tmp_path, "both.mx",
                  'print("script");\nfn main() -> int { print("main"); 0 }\n')
    assert main(["run", path]) == 1
    err = capsys.readouterr().err
    assert "both.mx:1" in err and "fn main" in err
    assert capsys.readouterr().out == ""


def test_run_resolves_imports_relative_to_the_file(tmp_path, capsys):
    (tmp_path / "util.mx").write_text(
        "export { twice };\nfn twice(x: int) -> int { x * 2 }\n")
    path = _write(tmp_path, "main.mx",
                  "from util import twice;\n"
                  "fn main() -> int { print(twice(21)); 0 }\n")
    assert main(["run", path]) == 0
    assert capsys.readouterr().out == "42\n"


def test_type_error_is_reported_on_stderr_with_exit_1(tmp_path, capsys):
    path = _write(tmp_path, "bad.mx", TYPE_ERROR)
    assert main(["run", path]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("TypeCheckError:")
    assert "bad.mx:3:15" in captured.err       # file:line:column
    assert "print(1 + \"a\")" in captured.err  # the source excerpt


def test_check_accepts_and_rejects_like_the_pipeline(tmp_path, capsys):
    good = _write(tmp_path, "hello.mx", HELLO)
    assert main(["check", good]) == 0
    assert capsys.readouterr().out == f"{good}: ok\n"
    borrow = os.path.join(FIXTURES, "test_borrow_check.mx")
    assert main(["check", borrow]) == 1
    assert capsys.readouterr().err.startswith("BorrowCheckError:")


def test_missing_file_is_a_user_error(tmp_path, capsys):
    assert main(["run", str(tmp_path / "nope.mx")]) == 1
    assert "cannot read" in capsys.readouterr().err


def test_no_entry_point_is_a_user_error(tmp_path, capsys):
    path = _write(tmp_path, "lib.mx", "fn helper(x: int) -> int { x }\n")
    assert main(["run", path]) == 1
    assert "no `fn main() -> int`" in capsys.readouterr().err


@pytest.mark.parametrize("stage,marker", [
    ("ast", '"'),                 # JSON
    ("hir", "main"),
    ("mir", "func main"),
    ("clif", "function"),
    ("llvm", "define"),
])
def test_emit_prints_each_stage(tmp_path, capsys, stage, marker):
    path = _write(tmp_path, "hello.mx", HELLO)
    assert main(["emit", path, "--stage", stage]) == 0
    out = capsys.readouterr().out
    assert marker in out and out.endswith("\n")


@needs_clang
def test_build_produces_a_binary_matching_the_interpreter(tmp_path):
    path = _write(tmp_path, "seven.mx", SEVEN)
    out = str(tmp_path / "bin" / "seven")          # parent dir is created
    assert main(["build", path, "-o", out, "--keep-ir"]) == 0
    assert os.path.exists(out) and os.path.exists(out + ".ll")
    proc = subprocess.run([out], capture_output=True, text=True)
    assert proc.returncode == 7
    assert proc.stdout == "seven\n"


@needs_clang
def test_build_default_output_is_the_stem_in_cwd(tmp_path, monkeypatch, capsys):
    path = _write(tmp_path, "hello.mx", HELLO)
    monkeypatch.chdir(tmp_path)
    assert main(["build", path]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "hello")
    assert os.path.exists(tmp_path / "hello")
    assert not os.path.exists(tmp_path / "hello.ll")   # no --keep-ir


def test_console_script_is_installed():
    """`uv sync` installs metaxuc (pyproject [project.scripts])."""
    proc = subprocess.run([sys.executable, "-m", "metaxu.compiler.cli", "--help"],
                          capture_output=True, text=True)
    assert proc.returncode == 0
    for sub in ("run", "build", "check", "emit"):
        assert sub in proc.stdout
