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
without binding any member names. Calls stay fully qualified. It earns
its keep in multi-file programs, where it's how one of your own files
names another; for `std` modules it's mostly documentation, because of
what the next section shows.

## Qualified paths need no import

A fully qualified path under `std` resolves with no import statement
anywhere in the file. The module loads the first time the resolver
walks the path:

```metaxu
fn main() -> int {
    print(std.stream.sum(std.stream.iota(100)));
    0
}
```
```output
4950
```

This is why one-off uses of a library function don't force an import
header on a short script. Once you use a name more than twice,
`from`-import it; the qualified spelling is for the reader who wants
to know where a name came from without scrolling up.

## Modules inside a file

A `module` block creates a named namespace anywhere in a file. Its
`export` list is the entire public interface; everything else in the
block is private to it.

```metaxu
module geometry {
    export area, perimeter;

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
    export area;

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
ModuleError: <mem>:8:1: cannot import private name 'shave' from module 'geometry'
```

Calling `geometry.shave(2)` directly is rejected the same way, at the
call site. One honesty note about the fences above: they are plain
fences, not `error` fences. The book's harness only pins compile
rejections that arrive as `TypeCheckError` or `BorrowCheckError`, and
this one is a `ModuleError`, a plain `CompileError` from the resolver.
The diagnostic is quoted verbatim from running the exact program
shown, but the harness doesn't re-check it on every run the way it
does the typed rejections.

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

This is the book's one display-only fence in this chapter: the
harness extracts single-source programs, and a multi-file layout has
no single source to extract. The program itself runs through the
example gate, which is where its behavior is pinned.

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
rewrites match guards, and the borrow checker deliberately skips
`__`-prefixed names so those generated lowerings never trip a false
positive. That bargain only holds if user code can't claim the
prefix, so declaring one is rejected up front:

```
fn main() -> int {
    let __tmp = 1;
    print(__tmp);
    0
}
```

```
ReservedNameError: <mem>:2:9: '__tmp': names beginning with '__' are reserved for compiler-generated bindings
```

Same caveat as the visibility rejection above: `ReservedNameError` is
a plain `CompileError`, so this is a quoted diagnostic in a plain
fence rather than a harness-pinned `error` block.

## Where this leaves you

Modules here are a naming discipline, not a compilation-unit design.
One file per concern, an `export` list you can read in one glance,
`from`-imports for the names you use often, qualified paths for the
ones you don't. Chapter 12 tours what `std` itself exports; chapter
17 shows where module resolution sits in the pipeline, which is
early, right after the parser and before any type is inferred.
