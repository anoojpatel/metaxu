"""Metaxu packages: manifest, lockfile, vendoring (docs/packages.md).

The compiler never fetches. This module resolves a project's
dependencies transitively, vendors git dependencies into mx_modules/,
writes mx.lock, and answers the one question the module resolver asks:
which directory is the root of the package called `name`?
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST = "mx.toml"
LOCK = "mx.lock"
VENDOR = "mx_modules"
LOCK_VERSION = 1


class PackageError(Exception):
    pass


@dataclass(frozen=True)
class Dep:
    name: str
    git: str | None = None
    rev: str | None = None
    path: str | None = None
    version: str | None = None      # a registry requirement; resolved by glade (metaxu.glade)

    @property
    def source(self) -> str:
        if self.git:
            return f"git+{self.git}"
        if self.path:
            return f"path+{self.path}"
        return "registry"


@dataclass
class Locked:
    name: str
    source: str
    hash: str
    rev: str | None = None
    commit: str | None = None
    version: str | None = None      # registry packages (glade): the resolved version


LOCK_VERSIONS_READ = {1, 2}         # 2 adds the optional `version` field


@dataclass
class Manifest:
    name: str
    version: str
    deps: dict[str, Dep] = field(default_factory=dict)
    public: list[str] = field(default_factory=list)


# --- manifest -----------------------------------------------------------

def read_manifest(root: Path) -> Manifest:
    path = root / MANIFEST
    if not path.is_file():
        raise PackageError(f"no {MANIFEST} in {root}")
    data = tomllib.loads(path.read_text())
    pkg = data.get("package") or {}
    if "name" not in pkg:
        raise PackageError(f"{path}: [package] needs a name")
    deps: dict[str, Dep] = {}
    for name, spec in (data.get("dependencies") or {}).items():
        if isinstance(spec, str):
            deps[name] = Dep(name, version=spec)          # `geom = "^0.2"`
            continue
        if not isinstance(spec, dict):
            raise PackageError(f"{path}: dependency '{name}' must be a string or a table")
        if "git" in spec:
            if "rev" not in spec:
                raise PackageError(
                    f"{path}: git dependency '{name}' needs a rev "
                    "(a tag, branch, or commit)")
            deps[name] = Dep(name, git=spec["git"], rev=str(spec["rev"]))
        elif "path" in spec:
            deps[name] = Dep(name, path=spec["path"])
        elif "version" in spec:
            deps[name] = Dep(name, version=str(spec["version"]))
        else:
            raise PackageError(
                f"{path}: dependency '{name}' needs a version requirement, "
                "`git` + `rev`, or `path`")
    return Manifest(name=pkg["name"], version=str(pkg.get("version", "0.0.0")),
                    deps=deps, public=list(pkg.get("public", [])))


def _toml_str(s: str) -> str:
    return json.dumps(s)  # a JSON string is a valid basic TOML string


def write_manifest(root: Path, m: Manifest) -> None:
    lines = ["[package]", f"name = {_toml_str(m.name)}",
             f"version = {_toml_str(m.version)}"]
    if m.public:
        lines.append("public = [" + ", ".join(_toml_str(p) for p in m.public) + "]")
    lines += ["", "[dependencies]"]
    for name in sorted(m.deps):
        d = m.deps[name]
        if d.git:
            lines.append(f"{name} = {{ git = {_toml_str(d.git)}, rev = {_toml_str(d.rev)} }}")
        elif d.path:
            lines.append(f"{name} = {{ path = {_toml_str(d.path)} }}")
        else:
            lines.append(f"{name} = {_toml_str(d.version or '*')}")
    (root / MANIFEST).write_text("\n".join(lines) + "\n")


# --- lockfile -----------------------------------------------------------

def read_lock(root: Path) -> dict[str, Locked]:
    path = root / LOCK
    if not path.is_file():
        return {}
    data = tomllib.loads(path.read_text())
    if data.get("version") not in LOCK_VERSIONS_READ:
        raise PackageError(f"{path}: unsupported lock version {data.get('version')}")
    out: dict[str, Locked] = {}
    for p in data.get("package") or []:
        out[p["name"]] = Locked(name=p["name"], source=p["source"], hash=p["hash"],
                                rev=p.get("rev"), commit=p.get("commit"),
                                version=p.get("version"))
    return out


def write_lock(root: Path, locked: dict[str, Locked]) -> None:
    # a lock stays at version 1 until it needs the field version 2 adds
    lock_version = 2 if any(p.version is not None for p in locked.values()) else LOCK_VERSION
    lines = [f"version = {lock_version}"]
    for name in sorted(locked):
        p = locked[name]
        lines += ["", "[[package]]", f"name = {_toml_str(p.name)}",
                  f"source = {_toml_str(p.source)}"]
        if p.version is not None:
            lines.append(f"version = {_toml_str(p.version)}")
        if p.rev is not None:
            lines.append(f"rev = {_toml_str(p.rev)}")
        if p.commit is not None:
            lines.append(f"commit = {_toml_str(p.commit)}")
        lines.append(f"hash = {_toml_str(p.hash)}")
    (root / LOCK).write_text("\n".join(lines) + "\n")


# --- hashing and fetching -----------------------------------------------

def tree_hash(root: Path) -> str:
    """SHA-256 over sorted relative paths and contents, .git excluded."""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if ".git" in p.parts or not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        h.update(rel.encode() + b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return "sha256:" + h.hexdigest()


def _git(*args: str, cwd: Path | None = None) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PackageError(f"git {' '.join(args)} failed:\n{proc.stderr.strip()}")
    return proc.stdout.strip()


def fetch_git(dep: Dep, dest: Path) -> str:
    """Clone `dep.git` at `dep.rev` into `dest` (replacing what is
    there), strip .git, return the commit hash."""
    assert dep.git and dep.rev
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _git("clone", "--quiet", dep.git, str(dest))
    _git("checkout", "--quiet", dep.rev, cwd=dest)
    commit = _git("rev-parse", "HEAD", cwd=dest)
    shutil.rmtree(dest / ".git", ignore_errors=True)
    return commit


# --- resolution ---------------------------------------------------------

def dep_root(project: Path, dep: Dep, requester_root: Path) -> Path:
    if dep.path:
        return (requester_root / dep.path).resolve()
    return project / VENDOR / dep.name


def sync(project: Path, log=lambda s: None) -> dict[str, Locked]:
    """Resolve transitively, fetch what the lock lacks, vendor, write the lock."""
    project = project.resolve()
    manifest = read_manifest(project)
    old = read_lock(project)
    locked: dict[str, Locked] = {}
    requested: dict[str, tuple[Dep, str, Path]] = {}   # name -> (dep, requester, root)
    queue: list[tuple[Dep, Path, str]] = [
        (d, project, manifest.name) for d in manifest.deps.values()]
    while queue:
        dep, req_root, requester = queue.pop(0)
        if dep.name == "std":
            raise PackageError(f"{requester}: 'std' is reserved and cannot be a dependency")
        if dep.version is not None:
            raise PackageError(
                f"{requester}: '{dep.name}' is a registry dependency ({dep.version}); "
                "resolving it takes the solver: run `glade sync` (metaxu.glade)")
        root = dep_root(project, dep, req_root)
        prior = requested.get(dep.name)
        if prior is not None:
            prior_dep, prior_requester, prior_root = prior
            same = ((prior_dep.git == dep.git and prior_dep.rev == dep.rev)
                    if (dep.git or prior_dep.git) else prior_root == root)
            if not same:
                raise PackageError(
                    f"conflicting requirements for '{dep.name}': "
                    f"{prior_requester} wants {prior_dep.source}@{prior_dep.rev}, "
                    f"{requester} wants {dep.source}@{dep.rev}; "
                    "pin one revision in the root manifest")
            continue
        requested[dep.name] = (dep, requester, root)
        # A path dependency's manifest spelling is relative to whoever
        # requested it (geom's `../util`); the lock records it relative
        # to the PROJECT, which is the only base the compiler has.
        source = dep.source if dep.git else (
            "path+" + Path(os.path.relpath(root, project)).as_posix())
        if dep.git:
            have = old.get(dep.name)
            fresh = (have is None or have.source != dep.source
                     or have.rev != dep.rev or not root.is_dir()
                     or tree_hash(root) != have.hash)
            if fresh:
                log(f"fetch {dep.name} {dep.git}@{dep.rev}")
                commit = fetch_git(dep, root)
            else:
                commit = have.commit
            locked[dep.name] = Locked(dep.name, source, tree_hash(root),
                                      rev=dep.rev, commit=commit)
        else:
            if not root.is_dir():
                raise PackageError(f"{requester}: path dependency '{dep.name}' "
                                   f"not found at {root}")
            locked[dep.name] = Locked(dep.name, source, tree_hash(root))
        if (root / MANIFEST).is_file():
            sub = read_manifest(root)
            for d in sub.deps.values():
                queue.append((d, root, sub.name))
    # drop vendored dirs no longer required
    vendor = project / VENDOR
    if vendor.is_dir():
        for child in vendor.iterdir():
            if child.is_dir() and child.name not in locked:
                log(f"remove {child.name} (no longer required)")
                shutil.rmtree(child)
    write_lock(project, locked)
    return locked


def check(project: Path) -> list[str]:
    """Hash drift between the lock and what is on disk; empty means clean."""
    project = project.resolve()
    locked = read_lock(project)
    manifest = read_manifest(project)
    problems: list[str] = []
    for name, entry in locked.items():
        if entry.source.startswith("path+"):
            root = (project / entry.source[len("path+"):]).resolve()
        else:
            root = project / VENDOR / name
        if not root.is_dir():
            problems.append(f"{name}: missing at {root}; run `glade sync`")
            continue
        actual = tree_hash(root)
        if actual != entry.hash:
            problems.append(f"{name}: tree hash {actual} differs from lock {entry.hash}")
    for name in manifest.deps:
        if name not in locked:
            problems.append(f"{name}: in {MANIFEST} but not in {LOCK}; run `glade sync`")
    return problems


def package_roots(project: Path) -> dict[str, Path]:
    """The table the module resolver consumes: name -> root directory,
    from the lock (never the manifest) plus manifest paths for
    path dependencies, which are used in place."""
    project = project.resolve()
    locked = read_lock(project)
    roots: dict[str, Path] = {}
    for name, entry in locked.items():
        if entry.source.startswith("path+"):
            rel = entry.source[len("path+"):]
            roots[name] = (project / rel).resolve()
        else:
            roots[name] = project / VENDOR / name
    return roots


def add(project: Path, name: str, git: str | None = None,
        rev: str | None = None, path: str | None = None) -> None:
    project = project.resolve()
    m = read_manifest(project)
    if git:
        if not rev:
            raise PackageError("--git needs --rev")
        m.deps[name] = Dep(name, git=git, rev=rev)
    elif path:
        m.deps[name] = Dep(name, path=path)
    else:
        raise PackageError("add needs --git URL --rev REV or --path DIR")
    write_manifest(project, m)


def tree(project: Path) -> str:
    project = project.resolve()
    m = read_manifest(project)
    locked = read_lock(project)
    lines = [f"{m.name} {m.version}"]

    def walk(root: Path, deps: dict[str, Dep], indent: str, seen: set[str]) -> None:
        for name in sorted(deps):
            d = deps[name]
            entry = locked.get(name)
            where = (f"{d.source}@{d.rev} {entry.commit[:10]}"
                     if entry and entry.commit else d.source)
            marker = " (*)" if name in seen else ""
            lines.append(f"{indent}{name} {where}{marker}")
            if name in seen:
                continue
            seen.add(name)
            sub_root = dep_root(project, d, root)
            if (sub_root / MANIFEST).is_file():
                walk(sub_root, read_manifest(sub_root).deps, indent + "  ", seen)

    walk(project, m.deps, "  ", set())
    return "\n".join(lines)


# --- CLI ------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="mxpkg (legacy; use glade)", description=__doc__.splitlines()[0])
    ap.add_argument("--project", default=".", help="project root (default: .)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_add = sub.add_parser("add", help="add a dependency to mx.toml")
    p_add.add_argument("name")
    p_add.add_argument("--git")
    p_add.add_argument("--rev")
    p_add.add_argument("--path")
    sub.add_parser("sync", help="fetch, vendor, and write mx.lock")
    sub.add_parser("check", help="verify vendored trees against mx.lock")
    sub.add_parser("tree", help="print the dependency tree")
    sub.add_parser("paths", help="print name -> root as JSON")
    args = ap.parse_args(argv)
    project = Path(args.project)
    try:
        if args.cmd == "add":
            add(project, args.name, git=args.git, rev=args.rev, path=args.path)
        elif args.cmd == "sync":
            sync(project, log=lambda s: print(s))
            print(f"locked {len(read_lock(project))} package(s)")
        elif args.cmd == "check":
            problems = check(project)
            for p in problems:
                print(p)
            return 1 if problems else 0
        elif args.cmd == "tree":
            print(tree(project))
        elif args.cmd == "paths":
            print(json.dumps({k: str(v) for k, v in package_roots(project).items()},
                             indent=2, sort_keys=True))
    except PackageError as e:
        print(f"mxpkg: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
