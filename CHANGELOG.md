# Changelog

Write notes under `Unreleased` as changes land. `scripts/release.py`
turns that section into the next version's section and tags it; the
release workflow (`.github/workflows/release.yml`) refuses a version
with no section and uses it as the release notes.

## Unreleased

- `glade`, the package manager (`docs/glade.md`): version requirements in
  `mx.toml` (`geom = "^0.2"`), a registry that is a git repository of
  package descriptions, a PubGrub solver with explained conflicts, and
  `add`, `remove`, `sync`, `update`, `tree`, `check`, `search`. Lock
  version 2 adds the resolved `version`; git- and path-only projects
  keep writing version 1. `mxpkg` forwards to `glade`.
- `std.semver` and `std.solve`: versions, requirements, ranges and the
  PubGrub solver in Metaxu, the first pieces of glade written in the
  language, each differentially tested against its Python module
  (`docs/glade_in_metaxu.md` tracks the rest).
- `metaxuc --version`.
- Native backend: `s[i]` and `s[a:b:c]` on strings lower to the C
  runtime; nested enum payloads keep their own kinds (an
  `Option[Option[int]]` no longer conflicts with every other `Option` in
  the module); kind-polymorphic helpers such as `is_none(o: Option)` are
  cloned per call-site kinds, which also fixes a helper reached with
  `Vec` of int and `Vec` of float printing `10.0` for `10`. `std.semver`
  and `std.solve` now compile and run natively.
- String builtins `split`, `find`, `replace`, `trim` and `join`, linear
  on both engines (C-backed natively); `std.parse.trim`, `split_on` and
  `std.string.join` now delegate to them, and `split_on` accepts a
  separator of any length.
- `to_bytes` and `from_bytes`: a string's UTF-8 bytes as a `Vec` of
  ints and back, with the same diagnostics on both engines.
- `std.hex` and `std.sha256`, pure Metaxu, tested against hashlib.
- `std.path` (posixpath semantics), `std.json` (a writer matching
  `json.dumps`) and `std.toml` (the manifest subset, with a writer in
  glade's layout), each tested against its Python oracle on both
  engines. Recursive enums held in `Vec`s now compile natively.
- `std.fs`, `std.process`, `std.env` and `std.io`: files, processes,
  the environment and stderr as effects with runtime mappings on both
  engines (`docs/io_runtime.md`); `std.env.args()` answers the command
  line (`metaxuc run file.mx -- a b`; native binaries take it from
  `argv`). A handler in scope virtualizes any of them.
- An assignment statement evaluates to `()`. A `-> ()` function ending
  in `s.f = v` used to return the struct. A `let` does too: a block
  ending in `let a = f()` used to evaluate to `a`.
- glade is written in Metaxu (`glade/*.mx`, `docs/glade.md`). The
  `glade` command now runs that program: compiled once into a cached
  native binary when clang is installed, on the interpreter otherwise;
  `GLADE_IMPL=python` runs the Python implementation, which stays as the
  reference. `test_glade_metaxu.py` runs both over the same registries
  and requires identical exit codes, output, error text, lockfiles and
  vendored trees. The parity test also fixed two things in the Python
  glade: a malformed requirement on the command line was a traceback,
  and stale packages were removed in an unspecified order.
- Native backend: the ordering comparisons on strings (`<`, `<=`, `>`,
  `>=`) lower to `mx_str_cmp` (code point order, the interpreter's)
  instead of demoting the function.
- `std.process`'s record of a finished program is `Completed` (it was
  `Outcome`, which `std.solve` also declares).
- `scripts/release.py`: one command to cut a release.

## 0.1.0 (2026-09-23)

The first tagged version: the v1 compiler end to end.

- `metaxuc`, one command with `run` (interpreter), `build` (native
  executable through LLVM and clang), `check` and `emit`; installable
  with `uv tool install`.
- The language: algebraic effects with handlers and delimited
  continuations, `try`/`catch`, pattern matching with exhaustiveness
  checking, traits, generics with SimpleSub-style inference, modes
  (`@local`/`@global`, `@mut`/`@const`, `once`/`separate`/`many`) in
  place of lifetimes, inferred thread safety, modules, and packages with
  a lockfile (`mxpkg`).
- Three engines that agree byte for byte on every tested program: the
  reference interpreter, the LLVM native backend with a small C runtime,
  and the Metal path for tile kernels (with a bit-exact shim off Mac).
- A standard library in Metaxu (`std.stream`, `std.sync`, `std.gpu`,
  `std.state`, `std.option`, `std.result`, and more).
- The Metaxu Book, eighteen chapters whose examples run as tests.
- Known limits: one target triple per host, no incremental builds, and
  the Cranelift backend lags LLVM. `docs/v1_gap_analysis.md` has the
  full list.
