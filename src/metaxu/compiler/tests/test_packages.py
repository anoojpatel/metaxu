"""mxpkg against real local git repositories (docs/packages.md)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from metaxu import packages
from metaxu.packages import (PackageError, check, package_roots, read_lock,
                             sync, tree, tree_hash)


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


def make_repo(root: Path, name: str, files: dict[str, str],
              manifest_deps: str = "", tag: str = "v1") -> Path:
    repo = root / f"{name}-src"
    (repo / "src").mkdir(parents=True)
    (repo / "mx.toml").write_text(
        f'[package]\nname = "{name}"\nversion = "0.1.0"\n\n'
        f"[dependencies]\n{manifest_deps}")
    for rel, body in files.items():
        (repo / rel).write_text(body)
    git("init", "-q", cwd=repo)
    git("-c", "user.email=t@t", "-c", "user.name=t", "add", ".", cwd=repo)
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
        "-m", "init", cwd=repo)
    git("tag", tag, cwd=repo)
    return repo


def make_project(root: Path, deps: str) -> Path:
    proj = root / "app"
    proj.mkdir()
    (proj / "mx.toml").write_text(
        '[package]\nname = "app"\nversion = "0.1.0"\n\n'
        f"[dependencies]\n{deps}")
    (proj / "main.mx").write_text("fn main() -> int { 0 }\n")
    return proj


def test_sync_vendors_git_dep_and_writes_lock(tmp_path):
    geom = make_repo(tmp_path, "geom", {"src/lib.mx": "export area;\n"})
    proj = make_project(tmp_path, f'geom = {{ git = "file://{geom}", rev = "v1" }}\n')

    locked = sync(proj)

    vendored = proj / "mx_modules" / "geom"
    assert (vendored / "src" / "lib.mx").read_text() == "export area;\n"
    assert not (vendored / ".git").exists()
    assert locked["geom"].commit == git("rev-parse", "HEAD", cwd=geom)
    assert locked["geom"].hash == tree_hash(vendored)
    assert read_lock(proj)["geom"].rev == "v1"
    assert package_roots(proj) == {"geom": vendored}
    assert check(proj) == []


def test_transitive_dependencies_flatten_into_one_table(tmp_path):
    util = make_repo(tmp_path, "util", {"src/lib.mx": "export clamp;\n"})
    geom = make_repo(tmp_path, "geom", {"src/lib.mx": "export area;\n"},
                     manifest_deps=f'util = {{ git = "file://{util}", rev = "v1" }}\n')
    proj = make_project(tmp_path, f'geom = {{ git = "file://{geom}", rev = "v1" }}\n')

    locked = sync(proj)

    assert set(locked) == {"geom", "util"}
    assert (proj / "mx_modules" / "util" / "src" / "lib.mx").is_file()
    out = tree(proj)
    assert "geom git+file://" in out and "  util git+file://" in out


def test_conflicting_revisions_are_an_error_naming_both(tmp_path):
    util = make_repo(tmp_path, "util", {"src/lib.mx": "export clamp;\n"})
    (util / "src" / "lib.mx").write_text("export clamp, wrap;\n")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "v2", cwd=util)
    git("tag", "v2", cwd=util)
    geom = make_repo(tmp_path, "geom", {"src/lib.mx": "export area;\n"},
                     manifest_deps=f'util = {{ git = "file://{util}", rev = "v1" }}\n')
    proj = make_project(
        tmp_path,
        f'geom = {{ git = "file://{geom}", rev = "v1" }}\n'
        f'util = {{ git = "file://{util}", rev = "v2" }}\n')

    with pytest.raises(PackageError) as ei:
        sync(proj)
    msg = str(ei.value)
    assert "conflicting requirements for 'util'" in msg
    assert "app wants" in msg and "geom wants" in msg


def test_check_detects_hand_edits_and_missing_trees(tmp_path):
    geom = make_repo(tmp_path, "geom", {"src/lib.mx": "export area;\n"})
    proj = make_project(tmp_path, f'geom = {{ git = "file://{geom}", rev = "v1" }}\n')
    sync(proj)

    (proj / "mx_modules" / "geom" / "src" / "lib.mx").write_text("export area, hack;\n")
    problems = check(proj)
    assert len(problems) == 1 and "geom" in problems[0] and "differs" in problems[0]

    # sync notices the drift and refetches
    sync(proj)
    assert check(proj) == []


def test_path_dependencies_are_used_in_place(tmp_path):
    lib = tmp_path / "util"
    (lib / "src").mkdir(parents=True)
    (lib / "mx.toml").write_text('[package]\nname = "util"\nversion = "0.1.0"\n')
    (lib / "src" / "lib.mx").write_text("export clamp;\n")
    proj = make_project(tmp_path, 'util = { path = "../util" }\n')

    locked = sync(proj)

    assert locked["util"].source == "path+../util"
    assert not (proj / "mx_modules" / "util").exists()
    assert package_roots(proj) == {"util": lib.resolve()}


def test_std_cannot_be_a_dependency(tmp_path):
    proj = make_project(tmp_path, 'std = { path = "../std" }\n')
    with pytest.raises(PackageError, match="reserved"):
        sync(proj)


def test_git_dependency_needs_a_rev(tmp_path):
    proj = make_project(tmp_path, 'geom = { git = "file:///nowhere" }\n')
    with pytest.raises(PackageError, match="needs a rev"):
        sync(proj)


def test_add_edits_manifest_and_stale_vendor_is_removed(tmp_path):
    geom = make_repo(tmp_path, "geom", {"src/lib.mx": "export area;\n"})
    proj = make_project(tmp_path, "")
    packages.add(proj, "geom", git=f"file://{geom}", rev="v1")
    sync(proj)
    assert (proj / "mx_modules" / "geom").is_dir()

    # drop it from the manifest: sync removes the vendored tree
    (proj / "mx.toml").write_text('[package]\nname = "app"\nversion = "0.1.0"\n')
    sync(proj)
    assert not (proj / "mx_modules" / "geom").exists()
    assert read_lock(proj) == {}


def test_cli_paths_prints_json(tmp_path, capsys):
    geom = make_repo(tmp_path, "geom", {"src/lib.mx": "export area;\n"})
    proj = make_project(tmp_path, f'geom = {{ git = "file://{geom}", rev = "v1" }}\n')
    assert packages.main(["--project", str(proj), "sync"]) == 0
    assert packages.main(["--project", str(proj), "paths"]) == 0
    out = capsys.readouterr().out
    assert '"geom"' in out and "mx_modules/geom" in out
    assert packages.main(["--project", str(proj), "check"]) == 0
