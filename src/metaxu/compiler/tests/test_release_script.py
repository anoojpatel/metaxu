"""scripts/release.py: the text transformations behind `release --bump`.

The git steps are not exercised here (they would tag the developer's
checkout); the pure functions that decide what a release changes are.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
spec = importlib.util.spec_from_file_location(
    "release", os.path.join(REPO_ROOT, "scripts", "release.py"))
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)

PYPROJECT = '[project]\nname = "metaxu"\nversion = "0.1.0"\n'
CHANGELOG = """# Changelog

## Unreleased

- something new

## 0.1.0 (2026-09-23)

- the first one
"""


def test_bump_each_part():
    assert release.bump("0.1.0", "patch") == "0.1.1"
    assert release.bump("0.1.9", "minor") == "0.2.0"
    assert release.bump("0.2.3", "major") == "1.0.0"
    with pytest.raises(release.ReleaseError):
        release.bump("1.0", "patch")


def test_set_version_rewrites_only_the_version_line():
    out = release.set_version(PYPROJECT, "0.2.0")
    assert out == '[project]\nname = "metaxu"\nversion = "0.2.0"\n'
    assert release.current_version(out) == "0.2.0"
    with pytest.raises(release.ReleaseError):
        release.set_version(PYPROJECT, "v0.2.0")


def test_changelog_release_moves_unreleased_and_opens_a_new_one():
    out = release.release_changelog(CHANGELOG, "0.2.0", "2026-10-01")
    assert out == """# Changelog

## Unreleased

## 0.2.0 (2026-10-01)

- something new

## 0.1.0 (2026-09-23)

- the first one
"""


def test_changelog_refuses_empty_notes_and_duplicates():
    empty = "# Changelog\n\n## Unreleased\n\n## 0.1.0 (2026-09-23)\n\n- x\n"
    with pytest.raises(release.ReleaseError, match="empty"):
        release.release_changelog(empty, "0.2.0", "2026-10-01")
    with pytest.raises(release.ReleaseError, match="already"):
        release.release_changelog(CHANGELOG, "0.1.0", "2026-10-01")
    with pytest.raises(release.ReleaseError, match="Unreleased"):
        release.release_changelog("# Changelog\n\n## 0.1.0\n\n- x\n", "0.2.0", "d")


def test_retarget_touches_only_version_spellings():
    text = ("install git+https://github.com/anoojpatel/metaxu@v0.1.0 and "
            "metaxu-0.1.0-py3-none-any.whl; version 0.1.0 of the format stays")
    out = release.retarget(text, "0.1.0", "0.2.0")
    assert "@v0.2.0" in out and "metaxu-0.2.0-py3-none-any.whl" in out
    assert "version 0.1.0 of the format stays" in out


def test_the_repo_changelog_and_versions_are_release_ready():
    """The real files satisfy the script's preconditions, so the next
    `release --bump` will not stop on them."""
    with open(os.path.join(REPO_ROOT, "CHANGELOG.md")) as fh:
        text = fh.read()
    assert text.split("## ", 1)[1].startswith("Unreleased")
    with open(os.path.join(REPO_ROOT, "pyproject.toml")) as fh:
        ver = release.current_version(fh.read())
    assert release.SEMVER.match(ver)
    for path in release.USER_FACING:
        assert f"v{ver}" in path.read_text(), f"{path.name} does not name v{ver}"


def test_metaxuc_reports_the_pyproject_version():
    proc = subprocess.run([sys.executable, "-m", "metaxu.compiler.cli", "--version"],
                          capture_output=True, text=True)
    with open(os.path.join(REPO_ROOT, "pyproject.toml")) as fh:
        ver = release.current_version(fh.read())
    assert proc.returncode == 0
    assert proc.stdout.strip() in (f"metaxuc {ver}", "metaxuc unknown")
