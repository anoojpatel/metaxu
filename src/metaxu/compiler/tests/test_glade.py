"""glade end to end: a local registry, real git repositories, the compiler.

The registry is a directory in the index layout (docs/glade.md); the
packages are git repositories with version tags. `glade add` picks a
version, `sync` vendors it, the lock records it, and a program that
imports the package compiles through the real module resolver.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from metaxu.compiler.pipeline import build_context_from_source
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import UNIT, MirInterpreter
from metaxu.packages import PackageError, read_lock
from metaxu.glade.cli import main as glade
from metaxu.glade.project import Project, read_manifest


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
                          cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


class Registry:
    """A directory registry plus the package repositories it points at."""

    def __init__(self, root: Path):
        self.root = root
        (root / "packages").mkdir(parents=True)
        self.repos: dict[str, Path] = {}
        self.entries: dict[str, list[str]] = {}

    def publish(self, name: str, version: str, files: dict[str, str],
                deps: dict[str, str] | None = None) -> str:
        repo = self.repos.get(name)
        if repo is None:
            repo = self.root / f"{name}.git"
            (repo / "src").mkdir(parents=True)
            git("init", "-q", cwd=repo)
            self.repos[name] = repo
        dep_lines = "".join(f'{d} = "{r}"\n' for d, r in (deps or {}).items())
        (repo / "mx.toml").write_text(
            f'[package]\nname = "{name}"\nversion = "{version}"\n\n[dependencies]\n{dep_lines}')
        for rel, body in files.items():
            (repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (repo / rel).write_text(body)
        git("add", ".", cwd=repo)
        git("commit", "-q", "-m", f"{name} {version}", cwd=repo)
        git("tag", f"v{version}", cwd=repo)
        commit = git("rev-parse", "HEAD", cwd=repo)
        entry = self.entries.setdefault(name, [])
        dep_toml = ", ".join(f'{d} = "{r}"' for d, r in (deps or {}).items())
        entry.append(f'[[versions]]\nversion = "{version}"\ncommit = "{commit}"\n'
                     f"dependencies = {{ {dep_toml} }}\n")
        (self.root / "packages" / f"{name}.toml").write_text(
            f'name = "{name}"\ngit = "file://{repo}"\ndescription = "test package {name}"\n\n'
            + "\n".join(entry))
        return commit


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


def project(tmp_path, registry: Registry) -> Path:
    proj = tmp_path / "app"
    assert glade(["init", str(proj), "--name", "app"]) == 0
    (proj / "main.mx").write_text(
        "import geom;\n\nfn main() -> int {\n    print(geom.area(3, 4));\n    0\n}\n")
    return proj


def run_main(proj: Path) -> list[str]:
    path = proj / "main.mx"
    ctx = build_context_from_source(path.read_text(), file_path=str(path))
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)))
    out: list[str] = []
    interp.register_builtin("print", lambda *a: (out.append(" ".join(str(x) for x in a)), UNIT)[1])
    interp.call("main", [])
    return out


def T(proj: Path, reg: Registry, *args: str) -> int:
    return glade(["--project", str(proj), "--registry", str(reg.root), *args])


def test_add_picks_newest_syncs_transitively_and_the_compiler_uses_it(tmp_path, registry, capsys):
    proj = project(tmp_path, registry)
    assert T(proj, registry, "add", "geom") == 0
    out = capsys.readouterr().out
    assert "added geom = '^0.2.0'" in out
    assert read_manifest(proj).deps["geom"].requirement_text == "^0.2.0"

    lock = read_lock(proj)
    assert lock["geom"].version == "0.2.0" and lock["geom"].rev == "v0.2.0"
    assert lock["util"].version == "0.1.0"                      # geom's dependency came along
    assert lock["geom"].source == f"registry+{registry.root}"
    assert (proj / "mx_modules" / "geom" / "src" / "lib.mx").is_file()
    assert not (proj / "mx_modules" / "geom" / ".git").exists()
    assert (proj / "mx.lock").read_text().startswith("version = 2\n")

    assert run_main(proj) == ["12"]
    (proj / "main.mx").write_text(
        "import geom;\n\nfn main() -> int {\n    print(geom.boxed(9, 9));\n    0\n}\n")
    assert run_main(proj) == ["50"]                             # util reached through geom

    assert T(proj, registry, "check") == 0
    assert T(proj, registry, "tree") == 0
    tree = capsys.readouterr().out
    assert "geom 0.2.0 (^0.2.0)" in tree and "util 0.1.0 (^0.1)" in tree


def test_requirement_selects_an_older_line_and_update_moves_within_it(tmp_path, registry, capsys):
    proj = project(tmp_path, registry)
    assert T(proj, registry, "add", "geom", "^0.1") == 0
    assert read_lock(proj)["geom"].version == "0.1.0"
    assert "util" not in read_lock(proj)                       # 0.1.0 has no dependencies

    registry.publish("geom", "0.1.1", {"src/lib.mx": "export { area };\n"
                                         "fn area(w: int, h: int) -> int { w * h + 0 }\n"})
    assert T(proj, registry, "sync") == 0
    assert read_lock(proj)["geom"].version == "0.1.0"          # sync keeps the lock
    assert T(proj, registry, "update") == 0
    assert "geom 0.1.0 -> 0.1.1" in capsys.readouterr().out
    assert read_lock(proj)["geom"].version == "0.1.1"          # update moves, within ^0.1
    assert T(proj, registry, "update") == 0
    assert "nothing to update" in capsys.readouterr().out


def test_unsatisfiable_requirement_is_explained(tmp_path, registry, capsys):
    proj = project(tmp_path, registry)
    assert T(proj, registry, "add", "geom", "^0.2") == 0
    assert T(proj, registry, "add", "util", "^0.9") == 2
    err = capsys.readouterr().err
    assert "version solving failed" in err
    assert "no versions of util match >=0.9.0, <0.10.0" in err


def test_conflict_between_two_requesters_names_both(tmp_path, registry, capsys):
    registry.publish("other", "1.0.0", {"src/lib.mx": "export { x };\nfn x() -> int { 1 }\n"},
                     deps={"util": "^0.5"})
    registry.publish("util", "0.5.0", {"src/lib.mx": "export { clamp };\n"
                                         "fn clamp(x: int, lo: int, hi: int) -> int { x }\n"})
    proj = project(tmp_path, registry)
    assert T(proj, registry, "add", "geom", "^0.2") == 0      # geom 0.2 wants util ^0.1
    assert T(proj, registry, "add", "other") == 2              # other wants util ^0.5
    err = capsys.readouterr().err
    assert "geom" in err and "other 1.0.0 depends on util >=0.5.0, <0.6.0" in err


def test_unknown_package_and_moved_tag_are_refused(tmp_path, registry, capsys):
    proj = project(tmp_path, registry)
    assert T(proj, registry, "add", "geometry") == 2
    assert "no package 'geometry'" in capsys.readouterr().err and True
    # the index pins a commit; moving the tag afterwards is detected
    repo = registry.repos["geom"]
    (repo / "src" / "lib.mx").write_text("export { area };\nfn area(w: int, h: int) -> int { 0 }\n")
    git("commit", "-q", "-am", "sneaky", cwd=repo)
    git("tag", "-f", "v0.2.0", cwd=repo)
    assert T(proj, registry, "add", "geom", "^0.2") == 2
    assert "tag v0.2.0 is at" in capsys.readouterr().err


def test_git_and_path_dependencies_still_work_alongside_registry_ones(tmp_path, registry):
    proj = project(tmp_path, registry)
    local = tmp_path / "local"
    (local / "src").mkdir(parents=True)
    (local / "mx.toml").write_text('[package]\nname = "local"\nversion = "0.3.0"\n\n'
                                   '[dependencies]\ngeom = "^0.2"\n')
    (local / "src" / "lib.mx").write_text("export { twice };\nimport geom;\n"
                                          "fn twice(w: int) -> int { geom.area(w, 2) }\n")
    assert T(proj, registry, "add", "local", "--path", "../local") == 0
    assert T(proj, registry, "add", "geom_git", "--git", f"file://{registry.repos['geom']}",
             "--rev", "v0.1.0") == 0
    lock = read_lock(proj)
    assert lock["local"].source == "path+../local" and lock["local"].version is None
    assert lock["geom_git"].rev == "v0.1.0" and lock["geom_git"].version is None
    assert lock["geom"].version == "0.2.0"                      # pulled in by the path dep
    (proj / "main.mx").write_text(
        "import local;\n\nfn main() -> int {\n    print(local.twice(5));\n    0\n}\n")
    assert run_main(proj) == ["10"]
    assert T(proj, registry, "remove", "local") == 0
    assert "local" not in read_lock(proj) and "geom" not in read_lock(proj)


def test_search_lists_the_index(tmp_path, registry, capsys):
    proj = project(tmp_path, registry)
    assert T(proj, registry, "search", "ge") == 0
    assert "geom 0.2.0  test package geom" in capsys.readouterr().out


def test_project_api_prefers_locked_versions(tmp_path, registry):
    proj = project(tmp_path, registry)
    assert T(proj, registry, "add", "geom", "^0.1") == 0
    registry.publish("geom", "0.1.5", {"src/lib.mx": "export { area };\n"
                                         "fn area(w: int, h: int) -> int { w * h }\n"})
    p = Project(proj, registry_override=str(registry.root))
    p._collect_pinned(read_lock(proj))
    from metaxu.glade.semver import Version
    assert p.resolve({"geom": Version.parse("0.1.0")})["geom"] == Version.parse("0.1.0")
    assert p.resolve({})["geom"] == Version.parse("0.1.5")
