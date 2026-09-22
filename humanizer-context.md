# Voice context for the humanizer skill

The humanizer skill installed at `.claude/skills/humanizer/` is
blader's (github.com/blader/humanizer, the 25 patterns from
Wikipedia's "Signs of AI writing"). It takes a writing sample and
house rules from the user; this file is that input for the Metaxu
Book (`docs/book/*.md`) and the website copy. Hand it to the skill
along with the file to edit, so a pass tightens the prose without
flattening it.

## Who is writing, and to whom

One person who built the compiler, writing for a programmer who knows
a systems language and is deciding whether to learn this one. Register:
technical, plain, direct. "You" is fine; "we" is not (there is no
team voice). No marketing, no cheerleading, no apologising.

## Rules the repository already enforces

- No em dashes in prose (`scripts/lint_book.py` fails on them). Pinned
  program output may contain them, because the compiler prints them.
- Comments inside Metaxu code use `#`, never `//`.
- Every ```` ```metaxu ```` example is executed by the test harness and
  its ```` ```output ```` fence is compared byte for byte. Never change
  a line inside a fence; never change a quoted diagnostic.
- Performance figures are microbenchmarks. Say so, with the caveats,
  every time they appear.

## What counts as this book's voice (protect it)

- Naming the actual diagnostic text, file path, function, or number
  instead of describing it.
- Honesty notes: "what is built versus sketched", "the harness does not
  re-check this one", "this is the one display-only fence". These are
  the point, not hedging.
- Short cross-references by chapter number ("chapter 10 covers modes").
- Plain repetition of the exact technical term (`Vec`, `@mut`, `once`).
  Do not vary it for elegance.
- Dry asides and the occasional fragment. One-sentence paragraphs when
  the sentence earns it.

## Banned or suspect

- Tier 1 AI vocabulary (delve, tapestry, testament, multifaceted, realm,
  interplay, leverage, underscore as a verb, "it's worth noting").
- "Not only X but Y", forced triads of abstract nouns, "In other
  words" restatements, paragraph-closing recaps, "Whether you..."
  endings, "There are several ways to" openers.
- Passive constructions that hide who acts when the actor is the
  compiler, the checker, the interpreter, or you.
- Any claim about the compiler that is not already in the text or the
  code. The skill may cut and sharpen; it may not invent.
