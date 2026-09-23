# Packages: pulling in other Metaxu code

Chapter 15 of the book describes the module system as it stands: files
are modules, `module` blocks nest them, `export` lists are the
interface, `std` resolves from the compiler's own tree. What it has no
answer for is code that lives in another repository. This document is
that answer, sized to what the language needs now.

The tool is now `glade` (`docs/glade.md`), which adds version requirements,
a git-hosted registry and a PubGrub solver on top of the layout
described here. Everything below about `mx.lock`, `mx_modules/` and the
resolver hook still holds; where this page says `mxpkg`, read `glade`.

## Principles

- **The compiler never touches the network.** Fetching is a separate
  tool that runs before a build and leaves files on disk. A build with
  a complete `mx_modules/` directory works offline, on a plane, in CI
  with the network off.
- **Pinned, not ranged.** A dependency is a git URL plus a revision, or
  a local path. There is no version-range solver. Two requesters that
  disagree on a revision are an error naming both, resolved by a
  person. This is deliberately less than Cargo; it is also nothing
  that can surprise you.
- **One name, one root.** A dependency's name becomes a top-level
  module path. `import geom.shapes;` means `<geom root>/src/shapes.mx`
  and nothing else. A dependency that collides with a sibling file's
  name is rejected, not resolved by precedence.
- **The lockfile is the build input.** `mx.toml` says what you want;
  `mx.lock` says what you got, down to a commit and a tree hash. The
  compiler reads the lock, never the manifest.

## Files

`mx.toml`, at the project root:

```toml
[package]
name = "app"
version = "0.1.0"

[dependencies]
geom = { git = "https://github.com/someone/geom", rev = "v0.3.0" }
util = { path = "../util" }
```

`mx.lock`, written by the tool, committed with the project:

```toml
version = 1

[[package]]
name = "geom"
source = "git+https://github.com/someone/geom"
rev = "v0.3.0"
commit = "3f2c9a1e..."
hash = "sha256:9b1d..."

[[package]]
name = "util"
source = "path+../util"
hash = "sha256:77e0..."
```

`mx_modules/<name>/`: the vendored checkout of each git dependency,
at the locked commit, with its `.git` directory removed. Path
dependencies are not copied; they are used in place, which is what you
want while developing two packages side by side.

A dependency is itself a package: it may carry its own `mx.toml` with
further dependencies, which are resolved transitively into the same
flat `mx_modules/` (one root per name, hence the conflict rule above).

## Layout of a package

```
geom/
    mx.toml
    src/
        lib.mx        # what `import geom;` means
        shapes.mx     # what `import geom.shapes;` means
        internal.mx   # reachable only from inside the package
```

`src/lib.mx` is the package's facade; its `export` list is the
package's public interface. Other files under `src/` are importable by
dotted path from outside only if `lib.mx` re-exports them or the
package's manifest lists them under `[package] public = [...]`. The
default is closed, matching the file-level rule in chapter 15 (a file
that exports nothing is unusable, not accidentally open).

## Resolution

Module resolution already runs right after the parser. It gains one
lookup: a table `name -> root directory` built from `mx.lock` and
`mx_modules/`. For an import path whose head is `p`:

1. `std` is reserved and resolves as today.
2. A sibling file `p.mx` next to the importing file wins, as today.
3. Otherwise `p` must be a locked dependency; the remainder of the path
   walks under its `src/`.
4. A head that is both a sibling file and a dependency is an error
   ("ambiguous module 'p': sibling file and dependency").

Inside a dependency, its own imports resolve against *its* siblings
and *its* locked dependencies (the flat table is shared, so a name
means the same package everywhere in one build).

## The tool

`mxpkg` is a small Python program (`src/metaxu/packages.py`, CLI in
`scripts/mxpkg.py`) with five verbs:

- `mxpkg add <name> --git <url> --rev <rev>` / `--path <dir>`: edit
  `mx.toml`.
- `mxpkg sync`: resolve transitively, fetch what the lock lacks or what
  changed, vendor into `mx_modules/`, write `mx.lock`.
- `mxpkg check`: recompute tree hashes and compare with the lock;
  nonzero exit on drift. Meant for CI.
- `mxpkg tree`: print the dependency tree with sources and commits.
- `mxpkg paths`: print the `name -> root` table as JSON. This is the
  seam the compiler consumes; anything that can produce this JSON can
  stand in for the tool.

Fetching uses the `git` executable (clone, checkout of the pinned
revision, then `.git` removed). Hashes are SHA-256 over sorted relative
paths and file contents of the vendored tree, so `check` catches a
hand edit inside `mx_modules/`.

## What this does not do, yet

No registry, no semantic-version ranges, no solver, no build scripts,
no per-dependency feature flags, no binary artifacts. Each is a
separate decision for later, and none is needed to share a library
between two repositories today. The one likely next step is a
`[replace]` table for overriding a transitive dependency with a local
path while fixing it, which fits the current design without changing
the lock format.

## Status

Built and pinned:

- The tool (`mxpkg add/sync/check/tree/paths`), tested against local
  git repositories (`test_packages.py`).
- The resolver hook (`compiler/module_loader.py`). The resolver finds
  the nearest `mx.lock` at or above the root file, reads the
  name-to-root table from it, and resolves an import whose head is a
  locked dependency from that package's `src/`: the bare name is
  `src/lib.mx`, `name.a.b` is `src/a/b.mx` and must be listed under
  `[package] public` when imported from outside the package. A head
  that is both a sibling file and a dependency is rejected as
  ambiguous. Inside a dependency, a bare import is qualified with the
  package name (`import shapes;` in geom means `geom.shapes`), so a
  package's files never collide with the project's. Pinned in
  `test_package_resolution.py` (path dependencies, no git needed) and
  by the example gate's `examples/pkg_app/`.

Not built: everything under "What this does not do, yet".
