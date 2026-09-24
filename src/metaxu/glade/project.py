"""A project: its manifest, its lock, resolution and vendoring.

`mx.toml` grows a third kind of dependency next to `git` and `path`:
a version requirement against the registry.

    [package]
    name = "app"
    version = "0.1.0"

    [dependencies]
    geom = "^0.2"                                  # from the registry
    fast = { version = "~1.4", registry = "https://github.com/me/index" }
    util = { git = "https://github.com/x/util", rev = "v0.3.0" }
    local = { path = "../local" }

    [glade]
    registry = "https://github.com/anoojpatel/glade-index"   # the default

Resolution hands the whole graph to PubGrub at once. Registry packages
offer every version the index lists; git and path packages offer
exactly one version (the one in their own manifest), so a range
elsewhere in the graph that excludes it is a reported conflict, never a
silent second copy. The result is one version per name, which is what
the compiler's flat `mx_modules/` layout needs.

`mx.lock` records the outcome. It is the compiler's only input
(`metaxu.packages.package_roots`); the manifest is the tool's.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from metaxu.packages import (LOCK, MANIFEST, VENDOR, Locked, PackageError,
                             _git, _toml_str, fetch_git, read_lock,
                             tree_hash, write_lock)
from metaxu.packages import Dep as PinnedDep

from .index import DEFAULT_REGISTRY, Registry
from .pubgrub import NoSolution, Solver
from .semver import Range, Version, VersionError, parse_requirement

import tomllib


@dataclass(frozen=True)
class Spec:
    """One line of [dependencies]."""
    name: str
    requirement: Range | None = None      # registry
    requirement_text: str = ""
    registry: str | None = None           # a registry other than the project's
    git: str | None = None
    rev: str | None = None
    path: str | None = None

    @property
    def kind(self) -> str:
        return "git" if self.git else "path" if self.path else "registry"

    def pinned(self) -> PinnedDep:
        return PinnedDep(self.name, git=self.git, rev=self.rev, path=self.path)


@dataclass
class Manifest:
    name: str
    version: str = "0.0.0"
    deps: dict[str, Spec] = field(default_factory=dict)
    public: list[str] = field(default_factory=list)
    registry: str | None = None


def read_manifest(root: Path) -> Manifest:
    path = root / MANIFEST
    if not path.is_file():
        raise PackageError(f"no {MANIFEST} in {root}")
    data = tomllib.loads(path.read_text())
    pkg = data.get("package") or {}
    if "name" not in pkg:
        raise PackageError(f"{path}: [package] needs a name")
    m = Manifest(name=str(pkg["name"]), version=str(pkg.get("version", "0.0.0")),
                 public=list(pkg.get("public", [])),
                 registry=(data.get("glade") or {}).get("registry"))
    for name, spec in (data.get("dependencies") or {}).items():
        m.deps[name] = _parse_spec(path, str(name), spec)
    return m


def _parse_spec(path: Path, name: str, spec) -> Spec:
    if isinstance(spec, str):
        return Spec(name, requirement=_req(path, name, spec), requirement_text=spec)
    if not isinstance(spec, dict):
        raise PackageError(f"{path}: dependency '{name}' must be a string or a table")
    if "git" in spec:
        if "rev" not in spec:
            raise PackageError(f"{path}: git dependency '{name}' needs a rev (tag, branch or commit)")
        return Spec(name, git=str(spec["git"]), rev=str(spec["rev"]))
    if "path" in spec:
        return Spec(name, path=str(spec["path"]))
    if "version" in spec:
        text = str(spec["version"])
        return Spec(name, requirement=_req(path, name, text), requirement_text=text,
                    registry=spec.get("registry"))
    raise PackageError(f"{path}: dependency '{name}' needs a version requirement, "
                       "`git` + `rev`, or `path`")


def _req(path: Path, name: str, text: str) -> Range:
    try:
        return parse_requirement(text)
    except VersionError as e:
        raise PackageError(f"{path}: dependency '{name}': {e}") from None


def write_manifest(root: Path, m: Manifest) -> None:
    lines = ["[package]", f"name = {_toml_str(m.name)}", f"version = {_toml_str(m.version)}"]
    if m.public:
        lines.append("public = [" + ", ".join(_toml_str(p) for p in m.public) + "]")
    lines += ["", "[dependencies]"]
    for name in sorted(m.deps):
        d = m.deps[name]
        if d.git:
            lines.append(f"{name} = {{ git = {_toml_str(d.git)}, rev = {_toml_str(d.rev or '')} }}")
        elif d.path:
            lines.append(f"{name} = {{ path = {_toml_str(d.path)} }}")
        elif d.registry:
            lines.append(f"{name} = {{ version = {_toml_str(d.requirement_text)}, "
                         f"registry = {_toml_str(d.registry)} }}")
        else:
            lines.append(f"{name} = {_toml_str(d.requirement_text)}")
    if m.registry:
        lines += ["", "[glade]", f"registry = {_toml_str(m.registry)}"]
    (root / MANIFEST).write_text("\n".join(lines) + "\n")


# --- resolution ---------------------------------------------------------------

@dataclass
class Pick:
    name: str
    version: Version
    spec: Spec                 # how it was requested (kind decides the source)
    root: Path                 # where its tree lives (vendor dir or the path)
    registry: Registry | None = None


class Project:
    def __init__(self, root: Path, registry_override: str | None = None,
                 cache: Path | None = None, log=lambda s: None):
        self.root = root.resolve()
        self.log = log
        self.manifest = read_manifest(self.root)
        source = (registry_override or self.manifest.registry
                  or os.environ.get("GLADE_REGISTRY") or DEFAULT_REGISTRY)
        self.cache = cache
        self.registries: dict[str, Registry] = {}
        self.default_registry = self._registry(source)
        self._pinned: dict[str, tuple[Spec, Path, Manifest | None]] = {}
        self._registry_of: dict[str, Registry] = {}

    def _registry(self, source: str) -> Registry:
        if source not in self.registries:
            self.registries[source] = Registry(source, cache=self.cache)
        return self.registries[source]

    # pinned (git/path) packages: fetched before solving, because their
    # manifests are the only place their own dependencies are written
    def _place_pinned(self, spec: Spec, requester_root: Path, old: dict[str, Locked]) -> Path:
        if spec.path:
            root = (requester_root / spec.path).resolve()
            if not root.is_dir():
                raise PackageError(f"path dependency '{spec.name}' not found at {root}")
            return root
        root = self.root / VENDOR / spec.name
        have = old.get(spec.name)
        fresh = (have is None or have.source != f"git+{spec.git}" or have.rev != spec.rev
                 or not root.is_dir() or tree_hash(root) != have.hash)
        if fresh:
            self.log(f"fetch {spec.name} {spec.git}@{spec.rev}")
            fetch_git(spec.pinned(), root)
        return root

    def _collect_pinned(self, old: dict[str, Locked]) -> None:
        queue: list[tuple[Spec, Path, str]] = [
            (s, self.root, self.manifest.name) for s in self.manifest.deps.values()
            if s.kind != "registry"]
        while queue:
            spec, req_root, requester = queue.pop(0)
            if spec.name == "std":
                raise PackageError(f"{requester}: 'std' is reserved and cannot be a dependency")
            root = self._place_pinned(spec, req_root, old)
            prior = self._pinned.get(spec.name)
            if prior is not None:
                p_spec, p_root, _ = prior
                same = (p_spec.git == spec.git and p_spec.rev == spec.rev) if spec.git or p_spec.git else p_root == root
                if not same:
                    raise PackageError(
                        f"conflicting requirements for '{spec.name}': one requester wants "
                        f"{p_spec.pinned().source}@{p_spec.rev}, {requester} wants "
                        f"{spec.pinned().source}@{spec.rev}; pin one in the root manifest")
                continue
            sub = read_manifest(root) if (root / MANIFEST).is_file() else None
            self._pinned[spec.name] = (spec, root, sub)
            if sub is not None:
                queue.extend((s, root, sub.name) for s in sub.deps.values() if s.kind != "registry")

    def _registry_for(self, spec: Spec) -> Registry:
        return self._registry(spec.registry) if spec.registry else self.default_registry

    def resolve(self, prefer: dict[str, Version] | None = None) -> dict[str, Version]:
        """Run the solver; `prefer` (usually the lock) breaks ties first."""
        prefer = prefer or {}
        project = self
        root_name = self.manifest.name
        root_version = _version_or_zero(self.manifest.version)

        class P:
            def versions(inner, package: str) -> list[Version]:
                if package == root_name:
                    return [root_version]
                if package in project._pinned:
                    _spec, _root, sub = project._pinned[package]
                    return [_version_or_zero(sub.version) if sub else Version(0, 0, 0)]
                reg = project._registry_of.get(package, project.default_registry)
                entry = reg.entry(package)
                if entry is None:
                    return []
                vs = sorted(entry.versions, reverse=True)
                if package in prefer and prefer[package] in vs:
                    vs.remove(prefer[package])
                    vs.insert(0, prefer[package])
                return vs

            def dependencies(inner, package: str, version: Version) -> dict[str, Range] | None:
                if package == root_name:
                    return project._deps_of(project.manifest)
                if package in project._pinned:
                    _spec, _root, sub = project._pinned[package]
                    return project._deps_of(sub) if sub else {}
                reg = project._registry_of.get(package, project.default_registry)
                entry = reg.entry(package)
                if entry is None or version not in entry.versions:
                    return None
                return dict(entry.versions[version].dependencies)

        try:
            picked = Solver(P()).solve(root_name, root_version)
        except NoSolution as e:
            raise PackageError("version solving failed:\n" + str(e)) from None
        picked.pop(root_name, None)
        return picked

    def _deps_of(self, m: Manifest) -> dict[str, Range]:
        out: dict[str, Range] = {}
        for spec in m.deps.values():
            if spec.kind == "registry":
                assert spec.requirement is not None
                out[spec.name] = spec.requirement
                if spec.registry:
                    self._registry_of[spec.name] = self._registry(spec.registry)
            else:
                # a pinned package has one version; the range is exactly it
                _s, _root, sub = self._pinned[spec.name]
                out[spec.name] = Range.exact(_version_or_zero(sub.version) if sub else Version(0, 0, 0))
        return out

    # --- the commands ---------------------------------------------------------

    def sync(self, update: set[str] | None = None, refresh: bool = True) -> dict[str, Locked]:
        """Resolve, fetch what the lock lacks, vendor, write the lock.

        `update` names packages whose locked version should not be
        preferred (`glade update`); None keeps every locked version that
        still satisfies the requirements."""
        old = read_lock(self.root)
        self._collect_pinned(old)
        needs_registry = any(s.kind == "registry" for s in self._all_specs())
        if needs_registry and refresh:
            for reg in self.registries.values():
                reg.refresh(self.log)
        if needs_registry and not self.default_registry.available():
            raise PackageError(f"registry {self.default_registry.source} has no "
                               f"{'packages/'} directory (is it an index?)")
        prefer = {n: Version.parse(e.version) for n, e in old.items()
                  if e.version and (update is None or n not in update)}
        picked = self.resolve(prefer)

        locked: dict[str, Locked] = {}
        for name, version in sorted(picked.items()):
            if name in self._pinned:
                spec, root, _sub = self._pinned[name]
                if spec.path:
                    source = "path+" + Path(os.path.relpath(root, self.root)).as_posix()
                    locked[name] = Locked(name, source, tree_hash(root))
                else:
                    have = old.get(name)
                    commit = have.commit if have and have.rev == spec.rev else _git("rev-parse", "HEAD", cwd=root) if (root / ".git").is_dir() else (have.commit if have else None)
                    locked[name] = Locked(name, f"git+{spec.git}", tree_hash(root),
                                          rev=spec.rev, commit=commit)
                continue
            reg = self._registry_of.get(name, self.default_registry)
            entry = reg.entry(name)
            assert entry is not None
            ve = entry.versions[version]
            root = self.root / VENDOR / name
            have = old.get(name)
            source = f"registry+{reg.source}"
            fresh = (have is None or have.source != source or have.version != str(version)
                     or not root.is_dir() or tree_hash(root) != have.hash)
            if fresh:
                self.log(f"fetch {name} {version} ({entry.git}@{ve.tag})")
                commit = reg.fetch(entry, version, root)
            else:
                commit = have.commit
            locked[name] = Locked(name, source, tree_hash(root), rev=ve.tag,
                                  commit=commit, version=str(version))
        vendor = self.root / VENDOR
        if vendor.is_dir():
            for child in sorted(vendor.iterdir()):
                if child.is_dir() and child.name not in locked:
                    self.log(f"remove {child.name} (no longer required)")
                    shutil.rmtree(child)
        write_lock(self.root, locked)
        return locked

    def _all_specs(self) -> list[Spec]:
        out = list(self.manifest.deps.values())
        for _spec, _root, sub in self._pinned.values():
            if sub is not None:
                out.extend(sub.deps.values())
        return out

    def add(self, name: str, requirement: str | None = None, *, git: str | None = None,
            rev: str | None = None, path: str | None = None, registry: str | None = None) -> Spec:
        if name == "std":
            raise PackageError("'std' is the standard library, not a dependency")
        if git:
            if not rev:
                raise PackageError("--git needs --rev (a tag, branch or commit)")
            spec = Spec(name, git=git, rev=rev)
        elif path:
            spec = Spec(name, path=path)
        else:
            reg = self._registry(registry) if registry else self.default_registry
            reg.refresh(self.log)
            entry = reg.entry(name)
            if entry is None:
                near = [n for n in reg.names() if name in n or n in name]
                hint = f"; did you mean {', '.join(near)}?" if near else ""
                raise PackageError(f"no package '{name}' in {reg.source}{hint}")
            if requirement is None or requirement == "*":
                newest = max(v for v in entry.versions if not v.pre) if any(
                    not v.pre for v in entry.versions) else max(entry.versions)
                requirement = f"^{newest}"
            spec = Spec(name, requirement=parse_requirement(requirement),
                        requirement_text=requirement, registry=registry)
        self.manifest.deps[name] = spec
        write_manifest(self.root, self.manifest)
        return spec

    def remove(self, name: str) -> None:
        if name not in self.manifest.deps:
            raise PackageError(f"'{name}' is not a dependency in {MANIFEST}")
        del self.manifest.deps[name]
        write_manifest(self.root, self.manifest)

    def tree(self) -> str:
        locked = read_lock(self.root)
        lines = [f"{self.manifest.name} {self.manifest.version}"]

        def manifest_at(root: Path) -> Manifest | None:
            return read_manifest(root) if (root / MANIFEST).is_file() else None

        def walk(m: Manifest, m_root: Path, indent: str, seen: set[str]) -> None:
            for name in sorted(m.deps):
                spec = m.deps[name]
                entry = locked.get(name)
                if entry is None:
                    where = "(not locked; run `glade sync`)"
                elif spec.kind == "registry":
                    where = f"{entry.version} ({spec.requirement_text})"
                elif spec.kind == "git":
                    where = f"{entry.source}@{entry.rev} {(entry.commit or '')[:10]}".rstrip()
                else:
                    where = entry.source
                marker = " (*)" if name in seen else ""
                lines.append(f"{indent}{name} {where}{marker}")
                if name in seen or entry is None:
                    continue
                seen.add(name)
                if spec.path:
                    sub_root = (m_root / spec.path).resolve()
                else:
                    sub_root = self.root / VENDOR / name
                sub = manifest_at(sub_root)
                if sub is not None:
                    walk(sub, sub_root, indent + "  ", seen)

        walk(self.manifest, self.root, "  ", set())
        return "\n".join(lines)


def _version_or_zero(text: str) -> Version:
    try:
        return Version.parse(text)
    except VersionError:
        return Version(0, 0, 0)


def init(root: Path, name: str | None = None) -> Path:
    root = root.resolve()
    if (root / MANIFEST).exists():
        raise PackageError(f"{root / MANIFEST} already exists")
    root.mkdir(parents=True, exist_ok=True)
    write_manifest(root, Manifest(name=name or root.name, version="0.1.0"))
    if not (root / "main.mx").exists() and not (root / "src").exists():
        (root / "main.mx").write_text('fn main() -> int {\n    print("hello");\n    0\n}\n')
    return root / MANIFEST
