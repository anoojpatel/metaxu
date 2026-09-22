"""What `uv tool install` / `uvx` get: the wheel and the installed layout.

The compiler runs from two layouts. In a checkout the standard library is
`std/` at the repository root; in an installed package it is `metaxu/std/`,
copied into the wheel by pyproject's force-include. These tests pin the
wheel's contents (the library present, developer build products absent,
both console scripts registered) and the loader's fallback to the
installed location. Building the wheel needs `uv` on PATH; the test skips
without it rather than pretending.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import zipfile

import pytest

from metaxu.compiler import module_loader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

needs_uv = pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")


@pytest.fixture(scope="module")
def wheel_names(tmp_path_factory):
    if shutil.which("uv") is None:
        pytest.skip("uv is not installed")
    out = tmp_path_factory.mktemp("dist")
    proc = subprocess.run(["uv", "build", "--wheel", "-o", str(out)],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    wheels = [p for p in os.listdir(out) if p.endswith(".whl")]
    assert len(wheels) == 1, wheels
    with zipfile.ZipFile(os.path.join(out, wheels[0])) as zf:
        return set(zf.namelist()), {n: zf.read(n) for n in zf.namelist()
                                    if n.endswith("entry_points.txt")}


@needs_uv
def test_wheel_carries_the_standard_library(wheel_names):
    names, _ = wheel_names
    shipped = {n for n in names if n.startswith("metaxu/std/") and n.endswith(".mx")}
    in_repo = {f"metaxu/std/{f}" for f in os.listdir(os.path.join(REPO_ROOT, "std"))
               if f.endswith(".mx")}
    assert shipped == in_repo


@needs_uv
def test_wheel_carries_the_native_runtime_sources_only(wheel_names):
    names, _ = wheel_names
    assert "metaxu/runtime/native/metaxu_rt.c" in names
    assert "metaxu/runtime/native/build.py" in names
    stray = [n for n in names if "_build/" in n or "__pycache__" in n
             or "/tests/" in n or n.endswith(("parsetab.py", "parser.out"))]
    assert stray == []


@needs_uv
def test_wheel_registers_both_console_scripts(wheel_names):
    _, entry_points = wheel_names
    (text,) = [v.decode() for v in entry_points.values()]
    assert "metaxuc = metaxu.compiler.cli:main" in text
    assert "metaxu = metaxu.metaxu:main" in text


def test_stdlib_dir_prefers_the_checkout_then_the_installed_copy(monkeypatch):
    monkeypatch.delenv("METAXU_STD_PATH", raising=False)
    here = os.path.dirname(os.path.abspath(module_loader.__file__))
    checkout = os.path.normpath(os.path.join(here, "..", "..", "..", "std"))
    installed = os.path.normpath(os.path.join(here, "..", "std"))
    assert module_loader._stdlib_dir() == checkout   # this IS a checkout

    real_isdir = os.path.isdir
    monkeypatch.setattr(module_loader.os.path, "isdir",
                        lambda p: p == installed or (p != checkout and real_isdir(p)))
    assert module_loader._stdlib_dir() == installed

    monkeypatch.setattr(module_loader.os.path, "isdir", lambda p: False)
    assert module_loader._stdlib_dir() is None


def test_env_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("METAXU_STD_PATH", str(tmp_path))
    assert module_loader._stdlib_dir() == str(tmp_path)
    monkeypatch.setenv("METAXU_STD_PATH", str(tmp_path / "missing"))
    assert module_loader._stdlib_dir() is None
