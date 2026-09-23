# Changelog

Each version has a section here. The release workflow
(`.github/workflows/release.yml`) refuses to publish a tag whose version
has no section, and uses the section as the release notes.

## 0.1.0

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
