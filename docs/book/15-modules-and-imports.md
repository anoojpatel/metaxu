# Modules and imports

A Metaxu program starts as one file with a `main`. It stops being one
file the moment you want to reuse something, and the module system is
deliberately small: files are modules, `module` blocks make modules
inside a file, `export` says what leaves, and two import forms bring
names in. There is no visibility ladder, no re-export algebra, no
crate graph. What a module doesn't export, you can't touch.

## Importing from the standard library

The form you'll write most is `from`-import, which binds member names
directly into your file:

```metaxu
from std.stream import iota, sum;

fn main() -> int {
    print(sum(iota(5)));
    0
}
```
```output
10
```

`iota(5)` streams 0 through 4 and `sum` folds them; chapter 8 built
both out of the `Emit` effect. Note that `from` is a reserved word
because of this statement. You can't name a variable `from`, and the
parse error if you try is blunt.

The other form is `import std.stream;`, which declares the dependency
without binding any member names. Calls stay fully qualified.

## Qualified paths need the module declared

A fully qualified path such as `std.stream.sum` resolves only once the
file has said `import std.stream;`. Without that line the resolver
reports `undefined function 'std.stream.sum'`; with it, every member
is reachable by its full name and nothing lands in your namespace:

```metaxu
import std.stream;

fn main() -> int {
    print(std.stream.sum(std.stream.iota(100)));
    0
}
```
```output
4950
```

Use this form when a module contributes one or two calls and you want
the reader to see where each came from without scrolling up. Once a
name appears more than twice, `from`-import it.

## Modules inside a file

A `module` block creates a named namespace anywhere in a file. Its
`export { ... }` list is the entire public interface; everything else
in the block is private to it.

```metaxu
module geometry {
    export { area, perimeter };

    fn area(w: int, h: int) -> int {
        w * h
    }

    fn perimeter(w: int, h: int) -> int {
        2 * (w + h)
    }

    # not exported: only code inside this block can call it
    fn shave(x: int) -> int {
        x - 1
    }
}

fn main() -> int {
    print(geometry.area(3, 4));
    print(geometry.perimeter(3, 4));
    0
}
```
```output
12
14
```

The dotted call `geometry.area(3, 4)` works from `main` with no
import, same as `std` paths: modules in the same file are already in
scope under their own names. Top-level `let` bindings inside a module
block are module constants; they export like functions and are
immutable, with the same diagnostic chapter 2 pins for rebinding.

## What visibility rejects

Pull `shave` out of `geometry` and the resolver stops you:

```
module geometry {
    export { area };

    fn area(w: int, h: int) -> int { w * h }
    fn shave(x: int) -> int { x - 1 }
}

from geometry import shave;

fn main() -> int {
    print(shave(2));
    0
}
```

```
<mem>:8:1: ModuleError: cannot import private symbol 'shave' from module 'geometry' (imported by module 'main')
  8 | from geometry import shave;
    | ^~~~~~~~~~~~~~~~~~~~~~~~~~

Notes:
  - 'shave' is not exported by 'geometry'
```

Calling `geometry.shave(2)` directly is rejected the same way, at the
call site: "symbol 'shave' of module 'geometry' is private (referenced
from module 'main' as 'geometry.shave')". One honesty note about the
fences above: they are plain fences, not `error` fences. The book's
harness only pins compile rejections that arrive as `TypeCheckError`
or `BorrowCheckError`, and this one is a `ModuleError`, a plain
`CompileError` from the resolver. The diagnostic is quoted verbatim
from running the exact program shown, but the harness doesn't
re-check it on every run the way it does the typed rejections.

## Multi-file programs

A directory of `.mx` files is a set of modules named by their file
stems. `examples/app/` in the repository is the small worked example
the pipeline gate builds:

```
examples/app/
    main.mx        # the entry point: owns fn main
    geometry.mx    # exports area
    util.mx        # exports clamp
```

`main.mx` opens by naming what it needs. Sibling files import by
module name; the standard library imports look exactly like they do
in a single file:

```metaxu norun
# examples/app/main.mx (import head)
import geometry;
from util import clamp;
from std.stream import iota, sum;

fn main() -> int {
    print(geometry.area(clamp(sum(iota(10)), 1, 6), 2));
    0
}
```

This is a display-only fence (the packages section below has the
other one): the harness extracts single-source programs, and a
multi-file layout has no single source to extract. The program itself
runs through the example gate, which is where its behavior is pinned.

Each file's `export` works as in a `module` block, with one default
worth knowing: a file with no `export` statement exports nothing, so
a library file that forgets its exports is unusable rather than
accidentally wide open.

## Name precedence

When a plain call could mean two things, the resolver prefers the
nearest definition. Your own top-level `fn len` beats the builtin
`len` in call position:

```metaxu
fn len(s: string) -> int {
    999
}

fn main() -> int {
    print(len("hello"));
    print("hello".len());
    0
}
```
```output
999
5
```

The method spelling still reaches the builtin, because method
dispatch resolves through the receiver (chapter 7's UFCS rules), not
through the plain-call namespace. The precedence order for a plain
name is: local and file-level definitions, then imported names, then
builtins. Shadowing a builtin is legal and silent, which is a reason
to keep import lists short enough to read.

## The `__` prefix is not yours

Names beginning with two underscores belong to the compiler. The
desugarer manufactures bindings like `__guard_scrut_at12` when it
rewrites match guards, emits calls to symbols like `__trait$` and
`__module_init`, and the borrow checker deliberately skips
`__`-prefixed names so those generated lowerings never trip a false
positive. That bargain only holds if user code can't claim the prefix
for a function, since a user function of that name would silently
redirect the compiler's own calls. The module loader refuses such a
declaration up front:

```
fn __helper() -> int {
    1
}

fn main() -> int {
    print(__helper());
    0
}
```

```
<mem>:1:1: ReservedNameError: function name '__helper' is reserved: names beginning with '__' belong to the compiler
  1 | fn __helper() -> int {
    | ^
```

The rule is about *functions*: a `let __tmp = 1;` is accepted today,
because a local binding cannot be mistaken for a compiler symbol.
Don't lean on that; the prefix is reserved in spirit everywhere. Same
caveat as the visibility rejection above: `ReservedNameError` is a
plain `CompileError`, so this is a quoted diagnostic in a plain fence
rather than a harness-pinned `error` block.

## Packages: code from another repository

Everything above is one project. Code that lives elsewhere comes in
as a *package*. The compiler never touches the network. A separate
tool, `tap`, fetches dependencies before a build and leaves files on
disk; a build with a complete `mx_modules/` directory works offline.
`docs/packages.md` describes the layout on disk and `docs/tap.md` the
tool.

A project declares what it wants in `mx.toml`. A dependency is a
version requirement against the registry, a git URL plus a revision,
or a local path:

```toml
[package]
name = "pkg_app"
version = "0.1.0"

[dependencies]
geom = { path = "deps/geom" }
```

```toml
[dependencies]
shapes = "^0.2"                                           # from the registry
util = { git = "https://github.com/x/util", rev = "v0.3.0" }
```

Requirements use Cargo's spelling: `^0.2` means any `0.2.x`, `~1.4`
means any `1.4.x`, `>=1, <3` is what it says, and a bare `1.2.3` is
`^1.2.3`. The registry is a git repository with one small file per
package listing its versions and their dependencies; publishing a
version is a pull request that adds a line. `tap add shapes` looks the
newest version up and writes the requirement for you.

`tap sync` hands the whole graph to a version solver (PubGrub, the
algorithm behind Cargo and uv), which picks one version per name that
satisfies every requirement, or explains in plain sentences why none
can. Git and path dependencies take part with exactly one version each,
so a requirement that excludes them is a reported conflict. Locked
versions are kept until you ask `tap update` to move them. The tool
then fetches whatever the lock lacks (git and registry packages are
vendored into `mx_modules/<name>/` with their `.git` removed; path
dependencies are used in place) and writes `mx.lock`, which records
each package's source, version and a hash of its tree. The compiler
reads the lock, never the manifest. This is the lock of
`examples/pkg_app/` in the repository, whose `geom` package itself
depends on a `util` package:

```toml
version = 1

[[package]]
name = "geom"
source = "path+deps/geom"
hash = "sha256:626013fffbbdb0e268c9b61fbe34d2d37ced1e72da29141b23bab3589c33475d"

[[package]]
name = "util"
source = "path+deps/util"
hash = "sha256:edd308f57cc009399b1917e2927b661c78d30ee27f599969deab4e2c50084c33"
```

The lock is flat: one root per name, so a name means the same package
everywhere in one build, and `util` is reachable from the application
even though only `geom` asked for it. `tap tree` shows who asked:

```text
pkg_app 0.1.0
  geom path+deps/geom
    util path+../util
```

A package is a directory with a manifest and a `src/`. Its
`src/lib.mx` is the facade, what `import geom;` means; every other
file under `src/` is private to the package unless the manifest lists
it under `[package] public`:

```text
examples/pkg_app/
    mx.toml
    mx.lock
    main.mx
    deps/geom/
        mx.toml            # public = ["shapes"], depends on util
        src/lib.mx         # `import geom;`
        src/shapes.mx      # `import geom.shapes;` (public)
        src/internal.mx    # geom's own files only
    deps/util/
        mx.toml
        src/lib.mx         # `import util;`
```

The entry file imports a package exactly like a module, because to the
resolver it is one. This is the example's `main.mx`, and the example
gate runs it:

```metaxu norun
import geom;
from geom.shapes import name;
import util;

fn main() -> int {
    print(geom.area(3, 4));
    print(name(2, 2));
    print(util.clamp(geom.area(9, 9), 0, 50));
    print(geom.describe(2, 5));
    0
}
```

It prints `12`, `square`, `50`, and `rectangle of area 10`. Inside
`geom`, `lib.mx` says `import shapes;` and `from internal import
twice;`; those bare names are qualified with the package name by the
resolver, so they mean `geom.shapes` and `geom.internal` even if the
application has a `shapes.mx` of its own.

Resolution of an import whose head is `p` goes: `std` is reserved; a
module declared in the file; a sibling file `p.mx` next to the root
file; a locked dependency named `p`. A head that is both a sibling
file and a dependency is not resolved by precedence, it is rejected.
The three rejections a package can produce, quoted from running them
against the example:

```text
ModuleError: module 'geom.internal' is not public in package 'geom' (imported from module 'main')

Notes:
  - a package exposes src/lib.mx and the modules listed under [package] public in its mx.toml
  - import the facade (`import geom;`) or ask the package to list "internal" as public
```

```text
ModuleError: ambiguous module 'geom': both a sibling file and the dependency 'geom' provide it (imported from module 'main')

Notes:
  - sibling file: examples/pkg_app/geom.mx
  - dependency root: examples/pkg_app/deps/geom
  - rename the file or the dependency; a name is never resolved by precedence
```

```text
ModuleError: module 'nothing' not found (imported from module 'main')

Notes:
  - looked for examples/pkg_app/nothing.mx
  - module paths resolve relative to the root file's directory: examples/pkg_app
  - 'nothing' is not a locked dependency either (examples/pkg_app/mx.lock lists: geom, util)
```

`tap check` recomputes the tree hashes and exits nonzero on drift, so
it catches a hand edit inside `mx_modules/`; that is the command a CI
job should run. There are no build scripts, no feature flags and no
binary artifacts; each is a separate decision for later, and you need
none of them to share a library between two repositories today.

## Where this leaves you

Modules here are a naming discipline, not a compilation-unit design.
One file per concern, an `export` list you can read in one glance,
`from`-imports for the names you use often, qualified paths for the
ones you don't, and a lockfile when the code comes from somewhere
else. Chapter 12 tours what `std` itself exports; chapter 17 shows
where module resolution sits in the pipeline, which is early, right
after the parser and before any type is inferred.
