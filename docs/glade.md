# glade: the Metaxu package manager

`glade` fetches other people's Metaxu code, picks versions that fit
together, and writes down exactly what it picked so the compiler and
everyone else's checkout build the same thing. A glade is a clearing
in a wood, the open ground where paths from different directions meet
and you can see what is there; packages from many repositories meet in
one lockfile the same way.

The design borrows one thing from each tool it resembles:

- from Cargo and Bundler, the shape: a manifest of requirements, a
  lockfile of decisions, one version of each package per build;
- from uv and pub, the solver: PubGrub, which propagates every choice,
  learns from each conflict, and explains failures as sentences;
- from Go modules and Homebrew, the registry: a git repository, so
  publishing is a pull request and mirroring is a clone;
- from pip and uv, the surface: `add`, `remove`, `sync`, `update`,
  `tree`, and nothing you have to learn before your first import.

`docs/packages.md` describes the on-disk layout the compiler reads
(`mx.lock`, `mx_modules/`, `src/lib.mx` facades). Nothing there
changed. This document covers what `glade` adds on top.

## The manifest

`mx.toml` has three kinds of dependency. Two existed before; the first
is new.

```toml
[package]
name = "app"
version = "0.1.0"

[dependencies]
geom = "^0.2"                                              # from the registry
fast = { version = "~1.4", registry = "https://github.com/me/index" }
util = { git = "https://github.com/x/util", rev = "v0.3.0" }
local = { path = "../local" }

[glade]
registry = "https://github.com/anoojpatel/glade-index"   # the default; omit it
```

A bare string is a version requirement against the registry. The
dialect is Cargo's:

| requirement | means |
|---|---|
| `^1.2.3` or `1.2.3` | `>=1.2.3, <2.0.0` (for `0.x`, the minor is the compatibility line: `^0.2.3` is `<0.3.0`) |
| `~1.2.3` | `>=1.2.3, <1.3.0` |
| `1.2.*` | `>=1.2.0, <1.3.0` |
| `=1.2.3` | exactly that |
| `>=1.2, <2` | comparators, comma means and |
| `*` | anything |

Prerelease versions (`2.0.0-rc.1`) never satisfy a requirement unless
the requirement names a prerelease of the same version, so `^1` will
not hand you a release candidate.

## The registry

A registry is a git repository, or a directory, with one file per
package under `packages/`:

```toml
# packages/geom.toml
name = "geom"
git = "https://github.com/someone/geom"
description = "shapes and areas"

[[versions]]
version = "0.2.0"
tag = "v0.2.0"                        # default: v<version>
commit = "3f2c9a1e..."                # optional; refuses a tag that later moved
dependencies = { util = "^0.1" }

[[versions]]
version = "0.1.0"
```

That file is the whole publishing story. To release `geom 0.3.0`, tag
the commit `v0.3.0` in geom's repository, then open a pull request to
the index that adds a `[[versions]]` entry. Whoever maintains the index
reviews the entry like any other change. There is no account, no
upload, no server to keep alive, and a company can run a private
registry by pointing `[glade] registry` at a repository of its own.

The index lists every version's dependencies, so resolution needs only
the index. `glade` clones it once into `~/.cache/glade/index/` (or
`$GLADE_CACHE`) and fetches updates on `sync`; the packages themselves
are cloned only after a version is chosen, shallow and at the tag, and
vendored into `mx_modules/<name>/` with `.git` removed. `glade sync
--offline` skips the refresh, and a project with a warm cache and a
complete `mx_modules/` builds with the network off, as before.

## Resolution

`glade sync` reads the manifest, fetches any git dependencies it has not
seen (their manifests are the only place their dependencies live),
reads path dependencies in place, and hands the whole graph to the
solver:

- a registry package offers every version the index lists;
- a git or path package offers exactly one version, the one in its own
  `mx.toml`, so a requirement elsewhere that excludes it is a reported
  conflict rather than a silent second copy;
- versions already in `mx.lock` are tried first, so a sync with no
  manifest change picks the same versions again;
- `glade update` drops that preference and moves each locked version as
  far forward as its requirements allow.

The solver is PubGrub (`src/metaxu/glade/pubgrub.py`), written from the
algorithm's published description. It treats resolution as
satisfiability: a decision is a version choice, a dependency is a
clause, and a conflict is analysed to find the earliest choice that
caused it, which is where the solver jumps back to, having recorded
what it learned. Two consequences matter in practice. A graph with
fifty versions of one package that all conflict with a neighbour is
rejected after one look, because one learned fact covers all fifty.
And when there is no solution, the message is the chain of facts that
proves it:

```
Because other 1.0.0 depends on util >=0.5.0, <0.6.0 and geom >=0.2.0, <0.3.0
depends on util >=0.1.0, <0.2.0, other 1.0.0 and geom >=0.2.0, <0.3.0 are
incompatible.
And because app depends on geom >=0.2.0, <0.3.0, other 1.0.0 is forbidden.
```

The result is one version per name, which is what the compiler's flat
`mx_modules/` needs, and it is written to `mx.lock`:

```toml
version = 2

[[package]]
name = "geom"
source = "registry+https://github.com/anoojpatel/glade-index"
version = "0.2.0"
rev = "v0.2.0"
commit = "3f2c9a1e..."
hash = "sha256:9b1d..."
```

Lock version 2 adds the `version` field. A project with only git and
path dependencies keeps writing version 1, byte for byte what it wrote
before, and the compiler reads both.

## Commands

```
glade init [DIR] [--name NAME]       start a project (mx.toml, main.mx)
glade add NAME [REQ]                 add a registry dependency; REQ defaults to ^newest
glade add NAME --git URL --rev REV   add a git dependency
glade add NAME --path DIR            add a path dependency
glade remove NAME
glade sync [--offline]               resolve, fetch, vendor, write mx.lock
glade update [NAME ...]              re-resolve, letting locked versions move forward
glade tree                           the dependency tree with versions and requirements
glade check                          verify mx_modules/ against mx.lock; nonzero on drift (CI)
glade search TEXT                    packages in the registry whose name contains TEXT
glade paths                          name -> root as JSON, the table the compiler reads
```

`add` and `remove` run `sync` afterwards, so the lock and `mx_modules/`
never disagree with the manifest you just edited. `uv sync` installs
`glade` next to `metaxuc`; `scripts/mxpkg.py`, the old name, forwards to
it.

## What is deliberately not here

No version-range solving across registries with different opinions of
a name: a name means one package per project. No build scripts, no
features, no binary artifacts, no yanking (remove the index entry and
open a pull request). No hosted index yet: `DEFAULT_REGISTRY` in
`src/metaxu/glade/index.py` names the repository the Metaxu index will
live in, and until it exists a project sets `[glade] registry` to a
repository or directory of its own; every test runs against a
directory.

## Written in Metaxu

The `glade` command runs the package manager written in Metaxu:
`glade/*.mx` at the repository root, a Metaxu package of its own over
`std.semver`, `std.solve`, `std.toml`, `std.sha256`, `std.path`,
`std.json` and the IO effects. The Python implementation in
`src/metaxu/glade/` stays as the reference, the way the interpreter is
the reference for the native backends: `test_glade_metaxu.py` runs both
over the scenarios of `test_glade.py` (a directory registry, git
repositories with tags, path and git dependencies, conflicts, a moved
tag, drift) and requires the same exit code, stdout, stderr, manifest,
lockfile and vendored trees, byte for byte. Only argparse's own usage
wording is not mirrored.

`metaxu.glade.launch` picks the engine. With `clang` installed it
compiles the program once through the LLVM backend into
`$XDG_CACHE_HOME/glade/bin/glade-<key>` (the key hashes the glade and
standard library sources and the compiler; the first run after a change
prints `building the native glade` and takes about a minute) and every
later command runs that binary. Without clang it runs on the MIR
interpreter, compiling the sources each time, a few seconds per
command. `GLADE_IMPL=native`, `interp` or `python` forces an engine;
`GLADE_SOURCES` points at another copy of the sources. The wheel ships
`glade/` as `metaxu/glade/mx/`, next to the standard library.

`docs/glade_in_metaxu.md` is the record of how it got here: what the
language and standard library needed, in what order, and what each
step found in the compiler.

## Where the code is

| file | what |
|---|---|
| `glade/main.mx` | the command: option parsing, dispatch, `glade: <why>` and exit 2 |
| `glade/manifest.mx`, `glade/lock.mx` | `mx.toml` and `mx.lock` in and out, in the Python layout byte for byte |
| `glade/index.mx` | the registry: index format, the cache, fetching a version, the moved-tag check |
| `glade/project.mx` | resolution over `std.solve`, `sync`, `add`, `remove`, `update`, `tree`, `check`, `paths`, `init` |
| `glade/hash.mx`, `glade/git.mx` | the tree hash over `std.sha256`; running git |
| `src/metaxu/glade/launch.py` | the console script: native binary cache, interpreter fallback, `GLADE_IMPL` |
| `src/metaxu/glade/semver.py`, `pubgrub.py`, `index.py`, `project.py`, `cli.py` | the Python reference implementation |
| `src/metaxu/packages.py` | the lock and vendor layout the compiler reads |
| `test_glade_solver.py`, `test_glade.py` | the solver's scenarios; the reference against a local registry and real git repositories |
| `test_glade_metaxu.py` | the parity test, on the interpreter and natively |
