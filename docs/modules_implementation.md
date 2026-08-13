# Module system: implemented semantics

This documents what the module system actually does as of the multi-file
implementation in `src/metaxu/compiler/module_loader.py`. The surface
grammar (`module`, `import`, `from ... import`, `export`, `public import`,
`visibility { ... }`) predates this work; the semantics below follow
`examples/03_modules_and_imports.mx` and the grammar where they speak, and
choose the boring standard thing (Rust/OCaml/Python-flavored) where they
are silent. Every choice is marked **[doc]** (forced by an existing
example/grammar) or **[chosen]** (our decision).

## Compilation model

Module resolution is a front-end pass that runs immediately after parsing
and before desugaring/freezing, inside `build_context_from_source`
(`compiler/pipeline.py`). It produces a single merged `Program`; every
later stage (inference, borrow checking, HIR, MIR, the interpreter, both
codegens) is module-unaware and sees only the merged program.

- **[chosen]** A program with no `module` blocks and no imports skips the
  pass entirely and compiles byte-for-byte as before (single-file
  backward compatibility).
- `run_pipeline_from_source(source, file_path=...)` and
  `build_context_from_source(source, file_path=...)` accept the on-disk
  path of the root file; file imports resolve relative to it.

## What a module is

- **[doc]** `module a.b { ... }` declares module `a.b`. Declared module
  paths are absolute (a nested `module` block's dotted name is its full
  path, not relative to the enclosing block).
- **[chosen]** Every source file is itself a module: the file's top-level
  scope. The *root* file's top-level scope is the **entry module**
  (conventionally `main`). A file loaded to satisfy `import a.b` *is*
  module `a.b`; its top-level declarations belong to `a.b`, and any
  `module` blocks inside it register at their own absolute paths.
- **[chosen]** In-file `module` blocks and file modules follow identical
  namespacing/visibility rules.

## Import forms

- **[doc]** `import a.b` — binds the *last* path component (`b`) as a
  module alias in the importing scope (`b.f(x)`), per
  `tests/golden/test_modules.mx` (`import math; ... math.add(x, y)`).
  `import a.b as m` binds `m` instead.
- **[doc]** `from a.b import x, y as z` — binds `x` and `z` directly.
- **[doc]** Relative from-imports, per example 03 (`module math.transform
  { from ..vector import ... }` reaching `math.vector`): `.p` (level 1)
  resolves inside the current module (`current.p`), `..p` (level 2) in the
  parent (`parent_of_current.p`), `...p` (level 3) one level higher.
  Escaping above the module root is a `ModuleError`.
- **[chosen]** Fully qualified references (`a.b.f(x)`) work without an
  import, exactly like Rust's full crate paths; visibility is still
  enforced.
- **[doc]** `public import` / `public from ... import` re-export the
  imported names: they become importable *from* the re-exporting module.
- `use` is a reserved keyword but has no grammar production; there is no
  `use` statement (unchanged).

## File mapping and loading

- **[chosen]** Module path -> file path: `import a.b` loads `a/b.mx`
  resolved against a single search root, the **directory of the root
  file** being compiled. (One root, not per-importer resolution, so a
  module path names the same file no matter who imports it.)
- **[chosen]** In-program module declarations take precedence: a path that
  is already declared in an in-file `module` block never hits the disk.
- Missing module file -> `CompileError` (`error_type="ModuleError"`)
  naming the module, the importer, and the path that was tried.
- Compiling from memory (`file_path="<mem>"`, the default) with a file
  import -> `ModuleError` explaining there is no on-disk location.
- **Cycles**: the import graph (in-file and cross-file edges alike) must
  be acyclic; a cycle raises `ModuleError` with the cycle chain
  (`a -> b -> a`). Self-imports count.
- **[chosen]** The `std.*` namespace is reserved for the (not yet
  implemented) standard library. `import std.x` / `from std.x import y`
  resolve to an *external placeholder*: the import succeeds (examples 03
  and 06 import `std.matrix`, `std.simd`, `std.math`, ...), no symbol or
  visibility checking happens, and references through it are left
  untouched — imported names like `sqrt` fall through to interpreter
  builtins, and calls like `io.println(...)` fail at run time with an
  unknown-callee error if nothing provides them.

## Namespacing

- **Functions** are namespaced. Each module's top-level `fn f` gets the
  final symbol name `<module.path>.f` (e.g. `math.vector.dot`); the MIR
  function table and interpreter use these dotted names directly.
  **[chosen]** The entry module's functions keep their plain names, so
  `main` remains the entry point and single-file programs are unchanged.
- References are rewritten during resolution, per the scope rule:
  **own module's declarations first, then import bindings**; anything
  else is left untouched (locals, builtins like `print`).
  - unqualified `f(x)`: resolves to the defining module's final name.
  - qualified `m.f(x)` / `a.b.f(x)`: the longest leading prefix that
    names a module (via alias or absolute path) is resolved; the call is
    rewritten to the final symbol name. A dotted call whose head is *not*
    a module (e.g. `receiver.method(...)`) is untouched.
  - `m.Type.method(x)`: the module qualifier is stripped
    (types are global, see below); `Type.method` dispatches as usual.
- **Types, traits and effects are NOT namespaced.** They merge into one
  global namespace. Declaring the same type/trait/effect name in two
  different modules is a `ModuleError` (collision), so the merge can
  never silently pun two definitions. This keeps struct instantiation,
  pattern matching, trait dispatch, and effect handling module-unaware.

## Visibility

Effective visibility of a top-level symbol in a module:

1. an explicit `visibility { name: public|private|protected }` entry wins
   (**[chosen]** `protected` is treated as private for cross-module
   access);
2. else, if the module has an `export { ... }` list, only the listed
   names are public (**[doc]**: example 03's `math.vector` exports
   `{Vector2, dot, scale}` and its `magnitude` is commented "Private
   helper function (not exported)");
3. else **everything top-level is public** (**[doc]**: example 03's
   `math.transform` has no export list and `main` imports `rotate` from
   it; golden `test_modules.mx` likewise).

Enforcement (all `ModuleError`s raised at compile time, naming the symbol
and both modules):

- `from m import x` where `x` is private -> error; where `x` does not
  exist -> error.
- Qualified reference `m.x(...)` where `x` is private (from outside `m`)
  or does not exist -> error.
- New grammar: `export { name, ... }` is now also accepted as a
  *file-level* statement (files are modules too); it parses to
  `ExportDeclaration` and feeds rule 2.

## Interpreter / checkers

No interpreter or borrow-checker changes were needed: module-qualified
functions are ordinary MIR functions with dotted names, the borrow
checker runs per function regardless of name, and traits/effects work
cross-module because declarations merge into the global namespace before
inference runs.

## Known caveats (deliberate scope limits)

- Unqualified references only resolve through declarations and imports;
  a name that resolves to nothing is left for the interpreter, which
  fails at run time with `Unknown callee` rather than at compile time
  (it may legitimately be a builtin or a local closure).
- Types/traits/effects being global means a *private* type is still
  nameable unqualified from another module without an import (visibility
  for types is enforced at import sites and qualified references only).
- Module functions are not first-class through qualified names
  (`let f = math.vector.dot` does not resolve); import the name and use
  it unqualified instead.
- A local variable that shadows a module alias/name can be hijacked by
  qualified-call resolution (`import math; let math = ...; math.x()`).
  Avoid shadowing module names.
- `perform m.E.op(...)` qualified effect performs are not supported;
  import the effect name (`from m import E`) and use `perform E.op(...)`.
- Module-level `let` bindings are not module symbols (functions, structs,
  enums, traits, effects only).
