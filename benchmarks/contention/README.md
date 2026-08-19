# Contention-guard benchmarks

The four-way comparison behind the design-B-vs-C verdict in
`docs/contention_as_permission.md` § "The B-vs-C performance verdict".

    uv run python benchmarks/contention/run_bench.py

Builds three runtimes from the CURRENT tree (shipped permit guard, a
no-guard layout control, and the design-C freeze variant via
`freeze-variant.patch`), links each against the same compiled IR for the
two benchmark programs, and reports alignment-controlled, rotation-
interleaved medians. Read run_bench.py's docstring before changing the
methodology — both of its controls exist because their absence produced
convincingly wrong numbers (including a NEGATIVE cost for added code).

Expected shape of results (exact numbers vary a few points run to run):
permit ≈ +7–11% on the pure-mutation microloop vs the no-guard control,
freeze ≈ 0%, lock/unlock deltas ≈ +0–7% (~1 ns/pair against ~18 ns
pthread ERRORCHECK pairs). If `freeze-variant.patch` stops applying, the
guard changed: regenerate the patch and re-run rather than deleting the
variant.

**Historical caveat (post inline Vec fast paths):** generated code now
carries its own copy of the write guard inline (codegen_llvm emits the
contended-flag test + permit read directly; see
benchmarks/diagnostics/), so swapping the RUNTIME's guard via these
patches no longer removes the guard from the hot path — the mutation
rows measure the runtime's cold/full ops only. The B-vs-C decision this
harness supported is closed; the post-inline guard cost is measured and
recorded in docs/contention_as_permission.md § "Post-inlining update"
(~+1.3 ns per write, guard-vs-no-guard on the inline store loop).
