# Recovery kit (container reset, 2026-09-11)

The remote container was rebuilt from a fresh GitHub clone. The GitHub
remote never held this session's work (pushes 403 by design; transfer
was via bundles), so everything local was lost. This kit holds what the
session still had verbatim in context.

## What the USER holds (the restore base)
- metaxu-v1-branch.bundle, tip 258af2d ("website: EB Garamond wordmark
  + three small gradients") — the last bundle sent. Contains ALL
  compiler work (f32/f16 tiles, MSL emitter, Metal handler, simdgroup
  plan, vortex website) through that commit.

## What this kit holds (work AFTER 258af2d, never bundled)
- website/index.html + website/roadmap.html: recovered byte-faithful
  from the published artifacts (already placed in the repo checkout).
  These are the SIMPLIFIED landing + MILESTONES roadmap.
- docs-book/: the book harness contract + chapters 01, 08, 10 and the
  README (final versions, all were green under the harness).
- tests/test_book_examples.py: the doctest harness.
- scripts/build_book_site.py: markdown -> website/book generator.
- FIX_PLAN.md: the six compiler fixes with their verbatim patch
  scripts, in the order they were applied and verified (suite was
  2148 passed + 1 skip after all six).

## Lost and needing regeneration (after base restore)
- test_match_guards.py content is IN FIX_PLAN.md. The const-field and
  rebinding regression tests are in FIX_PLAN.md too.

## Reconstructed as UNVERIFIED drafts (2026-09-11, post-reset)
Book chapters 02-07, 09, 11-17 were reconstructed in docs/book/ from
the pre-reset agents' completion reports (recovered from the session
transcript) plus a ground-truth brief. The compiler in this checkout
predates the language, so NONE of their examples have been executed.

Verification debt, to be paid on the restored 258af2d tree, in order:
1. Apply FIX_PLAN.md (all six fixes; the drafts assume the post-fix
   semantics: bools print 1/0, bare `Shape::Dot`, match guards,
   immutable plain `let`, @const field writes rejected, coherence as
   a pinnable error).
2. Run `pytest src/metaxu/compiler/tests/test_book_examples.py`.
   Expect failures where a draft invented a pin; fix the pins (or the
   example) until green. Chapters 01, 08, 10 and README were carried
   verbatim from green versions and should pass unchanged.
3. Rebuild website/book/ with scripts/build_book_site.py.
