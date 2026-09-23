"""Cut a release in one command.

    uv run python scripts/release.py 0.2.0          # an explicit version
    uv run python scripts/release.py --bump minor   # 0.1.0 -> 0.2.0
    uv run python scripts/release.py --bump patch --no-push

What it does, in order, stopping at the first problem:

1. Refuses a dirty working tree or a branch other than main (override
   with --branch).
2. Sets `version` in pyproject.toml.
3. In CHANGELOG.md, renames the `## Unreleased` section to
   `## <version> (<date>)` and opens a fresh, empty Unreleased section
   above it. An empty Unreleased section is refused: a release with no
   notes is a mistake.
4. Replaces the previous version everywhere it is spelled out for users
   (README.md, the book's first chapter, the website's Download button
   and install line) and rebuilds the book site.
5. Commits "Release v<version>", creates the annotated tag v<version>,
   and pushes the branch and the tag. The tag push starts
   .github/workflows/release.yml, which runs the suite and both gates,
   builds the wheel and sdist, and publishes the GitHub Release with the
   changelog section as its notes.

Versions are SemVer (MAJOR.MINOR.PATCH, optional -prerelease); tags are
the version with a `v` in front. Steps 2 to 4 are pure functions of
text so tests can pin them without git.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = REPO / "pyproject.toml"
CHANGELOG = REPO / "CHANGELOG.md"
# Files that spell the current version out for users, and therefore
# must move with it. Each occurrence of `v<old>` and of the wheel file
# name is rewritten; nothing else in these files is touched.
USER_FACING = [
    REPO / "README.md",
    REPO / "docs" / "book" / "01-getting-started.md",
    REPO / "website" / "index.html",
]

SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-([0-9A-Za-z.-]+))?$")


class ReleaseError(Exception):
    pass


# --- pure text transformations -------------------------------------------

def current_version(pyproject_text: str) -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject_text, re.M)
    if not m:
        raise ReleaseError("pyproject.toml has no [project] version line")
    return m.group(1)


def bump(version: str, part: str) -> str:
    m = SEMVER.match(version)
    if not m:
        raise ReleaseError(f"current version {version!r} is not SemVer")
    major, minor, patch = (int(m.group(i)) for i in (1, 2, 3))
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ReleaseError(f"unknown bump {part!r} (major, minor or patch)")


def set_version(pyproject_text: str, version: str) -> str:
    if not SEMVER.match(version):
        raise ReleaseError(f"{version!r} is not a SemVer version (MAJOR.MINOR.PATCH)")
    new, n = re.subn(r'^version\s*=\s*"[^"]+"', f'version = "{version}"',
                     pyproject_text, count=1, flags=re.M)
    if n != 1:
        raise ReleaseError("pyproject.toml has no [project] version line")
    return new


def release_changelog(text: str, version: str, date: str) -> str:
    """Rename `## Unreleased` to the version and open a new empty one."""
    lines = text.split("\n")
    heads = [i for i, l in enumerate(lines) if l.startswith("## ")]
    if not heads or lines[heads[0]].strip() != "## Unreleased":
        raise ReleaseError("CHANGELOG.md must start its sections with `## Unreleased`")
    start = heads[0]
    end = heads[1] if len(heads) > 1 else len(lines)
    body = [l for l in lines[start + 1:end] if l.strip()]
    if not body:
        raise ReleaseError("the Unreleased section of CHANGELOG.md is empty; "
                           "write the notes before cutting the release")
    for i in heads[1:]:
        if lines[i].split()[1] == version:
            raise ReleaseError(f"CHANGELOG.md already has a section for {version}")
    lines[start:start + 1] = ["## Unreleased", "", f"## {version} ({date})"]
    return "\n".join(lines)


def retarget(text: str, old: str, new: str) -> str:
    """Move every user-facing spelling of the old version to the new."""
    text = text.replace(f"v{old}", f"v{new}")
    return text.replace(f"metaxu-{old}-py3-none-any.whl", f"metaxu-{new}-py3-none-any.whl")


# --- the command ------------------------------------------------------------

def _git(*args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ReleaseError(f"git {' '.join(args)} failed:\n{proc.stderr.strip()}")
    return proc.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="cut a release: bump, changelog, commit, tag, push")
    ap.add_argument("version", nargs="?", help="the new version, e.g. 0.2.0")
    ap.add_argument("--bump", choices=["major", "minor", "patch"],
                    help="derive the new version from the current one")
    ap.add_argument("--branch", default="main", help="branch releases are cut from")
    ap.add_argument("--no-push", action="store_true", help="commit and tag, do not push")
    ap.add_argument("--date", default=_dt.date.today().isoformat(), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if bool(args.version) == bool(args.bump):
        ap.error("give a version or --bump, not both and not neither")
    try:
        pyproject = PYPROJECT.read_text()
        old = current_version(pyproject)
        new = args.version or bump(old, args.bump)
        if new == old:
            raise ReleaseError(f"{new} is already the current version")
        if _git("status", "--porcelain"):
            raise ReleaseError("the working tree is not clean; commit or stash first")
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        if branch != args.branch:
            raise ReleaseError(f"on branch {branch!r}; releases are cut from "
                               f"{args.branch!r} (or pass --branch)")
        if f"v{new}" in _git("tag", "--list", f"v{new}"):
            raise ReleaseError(f"tag v{new} already exists")

        PYPROJECT.write_text(set_version(pyproject, new))
        CHANGELOG.write_text(release_changelog(CHANGELOG.read_text(), new, args.date))
        for path in USER_FACING:
            path.write_text(retarget(path.read_text(), old, new))
        subprocess.run([sys.executable, str(REPO / "scripts" / "build_book_site.py")],
                       cwd=REPO, check=True, capture_output=True)
        _git("add", "-A")
        _git("commit", "-q", "-m", f"Release v{new}")
        _git("tag", "-a", f"v{new}", "-m", f"Metaxu {new}")
        print(f"released v{new} (was {old}): commit {_git('rev-parse', '--short', 'HEAD')}, tag v{new}")
        if args.no_push:
            print(f"not pushed; run: git push origin {branch} v{new}")
            return 0
        _git("push", "origin", branch, f"v{new}")
        print(f"pushed {branch} and v{new}; the Release workflow publishes "
              f"https://github.com/anoojpatel/metaxu/releases/tag/v{new}")
        return 0
    except ReleaseError as e:
        print(f"release: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
