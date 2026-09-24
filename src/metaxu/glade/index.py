"""The registry: a git repository of package descriptions.

There is no registry server. A registry is a git repository (or a
directory, for tests and private mirrors) with one file per package:

    packages/geom.toml

    name = "geom"
    git = "https://github.com/someone/geom"
    description = "shapes and areas"

    [[versions]]
    version = "0.2.0"
    tag = "v0.2.0"                      # default: v<version>
    commit = "3f2c9a1e..."              # optional: pins the tag's commit
    dependencies = { util = "^0.1" }

    [[versions]]
    version = "0.1.0"

Publishing a version is a pull request that adds a `[[versions]]`
entry, the way Homebrew formulae or the early crates.io index worked.
Every version's dependencies are in the index, so resolution reads one
small repository and never clones a package until it has been chosen.

The index is cloned once into the cache (`$GLADE_CACHE`, default
`~/.cache/glade`) and refreshed with `git fetch` on `glade sync`; a build
with a warm cache and a complete `mx_modules/` needs no network.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from metaxu.packages import PackageError

from .semver import Range, Version, VersionError, parse_requirement

PACKAGES_DIR = "packages"
DEFAULT_REGISTRY = "https://github.com/anoojpatel/glade-index"


class RegistryError(PackageError):
    pass


@dataclass(frozen=True)
class VersionEntry:
    version: Version
    tag: str
    commit: str | None
    dependencies: dict[str, Range]


@dataclass
class IndexEntry:
    name: str
    git: str
    description: str = ""
    versions: dict[Version, VersionEntry] = field(default_factory=dict)


def cache_dir() -> Path:
    env = os.environ.get("GLADE_CACHE")
    if env:
        return Path(env)
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "glade"


def _git(*args: str, cwd: Path | None = None) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RegistryError(f"git {' '.join(args)} failed:\n{proc.stderr.strip()}")
    return proc.stdout.strip()


def parse_entry(text: str, where: str) -> IndexEntry:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise RegistryError(f"{where}: {e}") from None
    for key in ("name", "git"):
        if key not in data:
            raise RegistryError(f"{where}: missing `{key}`")
    entry = IndexEntry(name=str(data["name"]), git=str(data["git"]),
                       description=str(data.get("description", "")))
    for v in data.get("versions") or []:
        try:
            ver = Version.parse(str(v["version"]))
        except (KeyError, VersionError) as e:
            raise RegistryError(f"{where}: bad version entry: {e}") from None
        deps: dict[str, Range] = {}
        for dep, req in (v.get("dependencies") or {}).items():
            try:
                deps[str(dep)] = parse_requirement(str(req))
            except VersionError as e:
                raise RegistryError(f"{where}: {entry.name} {ver}: dependency {dep}: {e}") from None
        if ver in entry.versions:
            raise RegistryError(f"{where}: version {ver} listed twice")
        entry.versions[ver] = VersionEntry(
            version=ver, tag=str(v.get("tag") or f"v{ver}"),
            commit=(str(v["commit"]) if v.get("commit") else None), dependencies=deps)
    return entry


class Registry:
    """One index, addressed by git URL or local directory."""

    def __init__(self, source: str, cache: Path | None = None):
        self.source = source
        self._entries: dict[str, IndexEntry | None] = {}
        local = Path(source).expanduser()
        if local.is_dir():
            self.root = local.resolve()
            self.remote = False
        else:
            # sha256, which the Metaxu glade (glade/index.mx) computes too,
            # so both implementations share one cache layout
            key = hashlib.sha256(source.encode()).hexdigest()[:16]
            self.root = (cache or cache_dir()) / "index" / key
            self.remote = True

    def refresh(self, log=lambda s: None) -> None:
        """Clone or fast-forward the index. A local directory needs nothing."""
        if not self.remote:
            return
        if (self.root / ".git").is_dir():
            log(f"update index {self.source}")
            _git("fetch", "--quiet", "--depth", "1", "origin", cwd=self.root)
            _git("reset", "--quiet", "--hard", "origin/HEAD", cwd=self.root)
        else:
            log(f"clone index {self.source}")
            self.root.parent.mkdir(parents=True, exist_ok=True)
            _git("clone", "--quiet", "--depth", "1", self.source, str(self.root))

    def available(self) -> bool:
        return (self.root / PACKAGES_DIR).is_dir()

    def entry(self, name: str) -> IndexEntry | None:
        if name not in self._entries:
            path = self.root / PACKAGES_DIR / f"{name}.toml"
            self._entries[name] = (parse_entry(path.read_text(), str(path))
                                   if path.is_file() else None)
        return self._entries[name]

    def names(self) -> list[str]:
        d = self.root / PACKAGES_DIR
        return sorted(p.stem for p in d.glob("*.toml")) if d.is_dir() else []

    def fetch(self, entry: IndexEntry, version: Version, dest: Path) -> str:
        """Shallow-clone the package's tag into `dest` (replacing it),
        check the commit when the index pins one, strip .git, return the
        commit hash."""
        ve = entry.versions[version]
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--quiet", "--depth", "1", "--branch", ve.tag, entry.git, str(dest))
        commit = _git("rev-parse", "HEAD", cwd=dest)
        if ve.commit and not commit.startswith(ve.commit):
            shutil.rmtree(dest)
            raise RegistryError(
                f"{entry.name} {version}: tag {ve.tag} is at {commit[:12]} but the "
                f"index pins {ve.commit[:12]}; the tag moved, refusing to use it")
        shutil.rmtree(dest / ".git", ignore_errors=True)
        return commit
