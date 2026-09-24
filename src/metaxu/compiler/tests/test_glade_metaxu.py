"""glade in Metaxu against glade in Python: the parity test.

`glade/*.mx` is the package manager written in Metaxu
(docs/glade_in_metaxu.md, step 4). Both implementations run the same
command sequences over the registry fixture of `test_glade.py` (a
directory index plus git repositories with version tags), each in its own
project directory and cache, and must agree byte for byte on the exit
code, stdout, stderr, the manifest, the lockfile and the vendored trees.
Only the project path differs between the two runs, so it is folded to a
marker before comparing.

The Metaxu glade runs on the MIR interpreter here (compiled once per
module); the native build is exercised by `test_native_glade_matches`
when clang is available.
"""
from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import UNIT, MirInterpreter
from metaxu.compiler.pipeline import build_context_from_source
from metaxu.glade.cli import main as python_glade
from metaxu.compiler.tests.test_glade import Registry, git, run_main

GLADE_MAIN = Path(__file__).resolve().parents[4] / "glade" / "main.mx"

needs_clang = pytest.mark.skipif(shutil.which("clang") is None, reason="clang is not installed")


@pytest.fixture(scope="module")
def glade_mir():
    ctx = build_context_from_source(GLADE_MAIN.read_text(), file_path=str(GLADE_MAIN))
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    return lower_hir_to_mir(hir)


@pytest.fixture
def registry(tmp_path):
    reg = Registry(tmp_path / "index")
    reg.publish("util", "0.1.0", {"src/lib.mx": "export { clamp };\n"
                                    "fn clamp(x: int, lo: int, hi: int) -> int {\n"
                                    "    if x < lo { lo } else if x > hi { hi } else { x }\n}\n"})
    reg.publish("geom", "0.1.0", {"src/lib.mx": "export { area };\n"
                                    "fn area(w: int, h: int) -> int { w * h }\n"})
    reg.publish("geom", "0.2.0", {"src/lib.mx": "export { area, boxed };\nfrom util import clamp;\n"
                                    "fn area(w: int, h: int) -> int { w * h }\n"
                                    "fn boxed(w: int, h: int) -> int { clamp(area(w, h), 0, 50) }\n"},
                deps={"util": "^0.1"})
    return reg


class Impl:
    """One implementation with its own project directory and cache."""

    def __init__(self, name: str, base: Path, registry: Registry):
        self.name = name
        self.proj = base / f"app_{name}"
        self.cache = base / f"cache_{name}"
        self.cache.mkdir()
        self.registry = registry
        self.transcript: list[tuple[int, str, str]] = []

    def env(self) -> dict[str, str]:
        return {**os.environ, "XDG_CACHE_HOME": str(self.cache), "HOME": str(self.cache)}

    def fold(self, text: str) -> str:
        return text.replace(str(self.proj), "<proj>")

    def state(self) -> dict[str, str | None]:
        out: dict[str, str | None] = {}
        for name in ("mx.toml", "mx.lock", "main.mx"):
            p = self.proj / name
            out[name] = p.read_text() if p.exists() else None
        modules = self.proj / "mx_modules"
        if modules.is_dir():
            for p in sorted(modules.rglob("*")):
                if p.is_file():
                    out["mx_modules/" + p.relative_to(modules).as_posix()] = p.read_text()
        return out


class PythonImpl(Impl):
    def run(self, *args: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        saved = dict(os.environ)
        os.environ.update(self.env())
        try:
            with redirect_stdout(out), redirect_stderr(err):
                try:
                    code = python_glade(list(args))
                except SystemExit as e:      # argparse's usage errors
                    code = int(e.code or 0)
        finally:
            os.environ.clear()
            os.environ.update(saved)
        return code, out.getvalue(), err.getvalue()


class InterpImpl(Impl):
    def __init__(self, base: Path, registry: Registry, mir):
        super().__init__("mx", base, registry)
        self.mir = mir

    def run(self, *args: str) -> tuple[int, str, str]:
        interp = MirInterpreter()
        interp.load(self.mir)
        interp.program_args = list(args)
        out, err = io.StringIO(), io.StringIO()
        saved = dict(os.environ)
        os.environ.update(self.env())
        try:
            with redirect_stdout(out), redirect_stderr(err):
                result = interp.call("main", [])
        finally:
            os.environ.clear()
            os.environ.update(saved)
        code = 0 if result is UNIT else int(result)
        return code, out.getvalue(), err.getvalue()


class NativeImpl(Impl):
    def __init__(self, base: Path, registry: Registry, binary: str):
        super().__init__("native", base, registry)
        self.binary = binary

    def run(self, *args: str) -> tuple[int, str, str]:
        proc = subprocess.run([self.binary, *args], capture_output=True, text=True,
                              env=self.env(), cwd=str(self.proj.parent))
        return proc.returncode, proc.stdout, proc.stderr


def step(impls: list[Impl], *args: str, expect: int | None = None,
         same_text: bool = True) -> str:
    """Run one command on every implementation; every visible result must
    agree with the Python one. Returns the (folded) stdout.

    `same_text=False` compares the exit code and the project state only:
    argparse words its usage errors its own way (`glade: error: the
    following arguments are required: name`), and mirroring argparse is
    not the point."""
    results = []
    for impl in impls:
        argv = [a.replace("$PROJ", str(impl.proj)).replace("$REG", str(impl.registry.root))
                for a in args]
        code, out, err = impl.run(*argv)
        results.append((impl, code, impl.fold(out), impl.fold(err)))
    ref_impl, ref_code, ref_out, ref_err = results[0]
    if expect is not None:
        assert ref_code == expect, (args, ref_out, ref_err)
    for impl, code, out, err in results[1:]:
        label = f"{impl.name} vs {ref_impl.name} on {' '.join(args)}"
        report = (f"{label}\n--- {ref_impl.name}: exit {ref_code}\n{ref_out}{ref_err}"
                  f"--- {impl.name}: exit {code}\n{out}{err}")
        if same_text:
            assert (code, out, err) == (ref_code, ref_out, ref_err), report
        else:
            assert code == ref_code, report
            assert err.startswith("glade: "), report
        assert impl.state() == ref_impl.state(), label
    return ref_out


def lock_names(lock_text: str) -> list[str]:
    return re.findall(r'^name = "([^"]*)"$', lock_text, flags=re.M)


def project_steps(impls: list[Impl]) -> None:
    step(impls, "init", "$PROJ", "--name", "app", expect=0)
    for impl in impls:
        (impl.proj / "main.mx").write_text(
            "import geom;\n\nfn main() -> int {\n    print(geom.area(3, 4));\n    0\n}\n")


def G(*args: str) -> tuple[str, ...]:
    return ("--project", "$PROJ", "--registry", "$REG", *args)


@pytest.fixture
def impls(tmp_path, registry, glade_mir):
    return [PythonImpl("py", tmp_path, registry), InterpImpl(tmp_path, registry, glade_mir)]


def test_add_sync_tree_check_search_paths(impls):
    project_steps(impls)
    out = step(impls, *G("add", "geom"), expect=0)
    assert "added geom = '^0.2.0'" in out
    py = impls[0]
    assert (py.proj / "mx.lock").read_text().startswith("version = 2\n")
    assert (py.proj / "mx_modules" / "util" / "src" / "lib.mx").is_file()
    assert step(impls, *G("check"), expect=0) == ""
    tree = step(impls, *G("tree"), expect=0)
    assert "geom 0.2.0 (^0.2.0)" in tree and "util 0.1.0 (^0.1)" in tree
    paths = step(impls, *G("paths"), expect=0)
    assert '"geom": "<proj>/mx_modules/geom"' in paths
    assert "geom 0.2.0  test package geom" in step(impls, *G("search", "ge"), expect=0)
    assert "no packages matching 'zzz'" in step(impls, *G("search", "zzz"), expect=0)
    assert step(impls, *G("sync"), expect=0) == "locked 2 package(s)\n"
    # The compiler reads what the Metaxu glade vendored.
    for impl in impls:
        assert run_main(impl.proj) == ["12"]
        (impl.proj / "main.mx").write_text(
            "import geom;\n\nfn main() -> int {\n    print(geom.boxed(9, 9));\n    0\n}\n")
        assert run_main(impl.proj) == ["50"]
    # Drift is reported the same way, and `check` exits 1 for it.
    for impl in impls:
        (impl.proj / "mx_modules" / "util" / "src" / "lib.mx").write_text("# edited\n")
    drift = step(impls, *G("check"), expect=1)
    assert "util" in drift
    step(impls, *G("sync"), expect=0)
    assert step(impls, *G("check"), expect=0) == ""


def test_older_line_and_update(impls, registry):
    project_steps(impls)
    step(impls, *G("add", "geom", "^0.1"), expect=0)
    lock = impls[0].state()["mx.lock"]
    assert 'version = "0.1.0"' in lock and "util" not in lock
    registry.publish("geom", "0.1.1", {"src/lib.mx": "export { area };\n"
                                         "fn area(w: int, h: int) -> int { w * h + 0 }\n"})
    step(impls, *G("sync"), expect=0)
    assert 'version = "0.1.0"' in impls[0].state()["mx.lock"]
    assert step(impls, *G("update"), expect=0).endswith("geom 0.1.0 -> 0.1.1\n")
    assert 'version = "0.1.1"' in impls[0].state()["mx.lock"]
    assert step(impls, *G("update"), expect=0) == "nothing to update\n"
    assert step(impls, *G("update", "geom"), expect=0) == "nothing to update\n"


def test_unsatisfiable_and_conflicting_requirements(impls, registry):
    registry.publish("other", "1.0.0", {"src/lib.mx": "export { x };\nfn x() -> int { 1 }\n"},
                     deps={"util": "^0.5"})
    registry.publish("util", "0.5.0", {"src/lib.mx": "export { clamp };\n"
                                         "fn clamp(x: int, lo: int, hi: int) -> int { x }\n"})
    project_steps(impls)
    step(impls, *G("add", "geom", "^0.2"), expect=0)
    step(impls, *G("add", "util", "^0.9"), expect=2)
    step(impls, *G("add", "other"), expect=2)
    step(impls, *G("add", "geom", "not a requirement"), expect=2)


def test_unknown_package_and_moved_tag(impls, registry):
    project_steps(impls)
    step(impls, *G("add", "geometry"), expect=2)
    repo = registry.repos["geom"]
    (repo / "src" / "lib.mx").write_text("export { area };\nfn area(w: int, h: int) -> int { 0 }\n")
    git("commit", "-q", "-am", "sneaky", cwd=repo)
    git("tag", "-f", "v0.2.0", cwd=repo)
    step(impls, *G("add", "geom", "^0.2"), expect=2)


def test_git_and_path_dependencies_and_remove(impls, registry, tmp_path):
    project_steps(impls)
    local = tmp_path / "local"
    (local / "src").mkdir(parents=True)
    (local / "mx.toml").write_text('[package]\nname = "local"\nversion = "0.3.0"\n\n'
                                   '[dependencies]\ngeom = "^0.2"\n')
    (local / "src" / "lib.mx").write_text("export { twice };\nimport geom;\n"
                                          "fn twice(w: int) -> int { geom.area(w, 2) }\n")
    step(impls, *G("add", "local", "--path", "../local"), expect=0)
    step(impls, *G("add", "geom_git", "--git", f"file://{registry.repos['geom']}",
                   "--rev", "v0.1.0"), expect=0)
    lock = impls[0].state()["mx.lock"]
    assert 'source = "path+../local"' in lock and 'rev = "v0.1.0"' in lock
    tree = step(impls, *G("tree"), expect=0)
    assert "local" in tree and "geom_git" in tree
    for impl in impls:
        (impl.proj / "main.mx").write_text(
            "import local;\n\nfn main() -> int {\n    print(local.twice(5));\n    0\n}\n")
        assert run_main(impl.proj) == ["10"]
    step(impls, *G("remove", "local"), expect=0)
    assert lock_names(impls[0].state()["mx.lock"]) == ["geom_git"]
    assert lock_names(impls[1].state()["mx.lock"]) == ["geom_git"]
    step(impls, *G("remove", "nosuch"), expect=2)
    step(impls, *G("add", "geom_git", "--git", "file:///nowhere/at/all.git", "--rev", "v1"), expect=2)


def test_usage_errors_agree(impls):
    project_steps(impls)
    # argparse's own wording is not mirrored: exit code 2 and no change.
    step(impls, expect=2, same_text=False)
    step(impls, "--project", "$PROJ", "frobnicate", expect=2, same_text=False)
    step(impls, "--bogus", "sync", expect=2, same_text=False)
    step(impls, *G("add"), expect=2, same_text=False)
    step(impls, *G("remove"), expect=2, same_text=False)
    step(impls, *G("search"), expect=2, same_text=False)
    # glade's own diagnostics are compared word for word.
    step(impls, "--project", "$PROJ/missing", "sync", expect=2)
    step(impls, *G("add", "geom", "--git", "file:///x"), expect=2)
    step(impls, *G("add", "geom", "--path", "../nowhere"), expect=2)


def test_every_reached_function_compiles_natively():
    # What still demotes is unreached from `main`: std.toml's writer (glade
    # writes its files line by line), std.string.is_empty and std.fail's
    # higher-order handlers, all imported by modules glade uses.
    from metaxu.compiler.pipeline import emit_llvm_from_source
    llvm = emit_llvm_from_source(GLADE_MAIN.read_text(), file_path=str(GLADE_MAIN))
    demoted = set(re.findall(r"^; function @(\S+): placeholder", llvm, flags=re.M))
    assert "mx_main" not in demoted
    assert all(name.startswith(("mx_std_toml_", "mx_std_string_is_empty", "mx_std_fail_",
                                "mx___handle_body_Fail_std_fail_", "mx___handler_Fail_fail_std_fail_"))
               for name in demoted), sorted(demoted)


# --- the console script -----------------------------------------------------

def test_launcher_finds_the_sources_and_runs_on_the_interpreter(tmp_path, monkeypatch, capsys):
    from metaxu.glade import launch
    assert launch.sources_dir() == GLADE_MAIN.parent
    monkeypatch.setenv("GLADE_IMPL", "interp")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    proj = tmp_path / "app"
    assert launch.main(["init", str(proj), "--name", "app"]) == 0
    assert launch.main(["--project", str(proj), "tree"]) == 0
    assert launch.main(["--project", str(proj), "frobnicate"]) == 2
    out, err = capsys.readouterr()
    assert out == f"wrote {proj / 'mx.toml'}\napp 0.1.0\n"
    assert err == "glade: unknown command 'frobnicate'\n"
    monkeypatch.setenv("GLADE_IMPL", "python")
    assert launch.main(["--project", str(proj), "tree"]) == 0
    assert capsys.readouterr().out == "app 0.1.0\n"


@needs_clang
def test_native_glade_matches_and_the_launcher_caches_the_binary(tmp_path, registry, monkeypatch, capsys):
    from metaxu.glade import launch
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    binary = launch.native_binary(launch.sources_dir())
    assert binary is not None and binary.is_file()
    assert "building the native glade" in capsys.readouterr().err
    built_at = binary.stat().st_mtime_ns
    # The launcher's default path finds the cached binary and runs it.
    proj = tmp_path / "launched"
    assert launch.main(["init", str(proj), "--name", "launched"]) == 0
    assert capsys.readouterr().err == ""
    assert (proj / "mx.toml").is_file()
    assert binary.stat().st_mtime_ns == built_at
    # And it agrees with the Python glade on a full scenario.
    impls = [PythonImpl("py", tmp_path, registry), NativeImpl(tmp_path, registry, str(binary))]
    project_steps(impls)
    step(impls, *G("add", "geom"), expect=0)
    step(impls, *G("check"), expect=0)
    step(impls, *G("tree"), expect=0)
    step(impls, *G("paths"), expect=0)
    step(impls, *G("search", "ge"), expect=0)
    assert step(impls, *G("update"), expect=0) == "nothing to update\n"
    step(impls, *G("remove", "geom"), expect=0)
    step(impls, *G("add", "geometry"), expect=2)
    step(impls, *G("add", "geom", "not a requirement"), expect=2)
    # A failed `add` leaves the requirement in the manifest on both sides,
    # so this stays last: every later sync would fail the same way.
    step(impls, *G("add", "util", "^0.9"), expect=2)
