# Contributing to Metaxu

This page covers how to set up a working copy, what to run before you
open a pull request, and the rules a change has to follow. The README's
Development section has the repository layout.

## Setup

You need Python 3.11 or newer, clang (the native backend compiles
through it), and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/anoojpatel/metaxu.git
cd metaxu
uv sync --all-groups
```

## Before you open a pull request

Run all three. A pull request is ready when they pass.

```bash
uv run python -m pytest src/metaxu/compiler/tests -q   # the test suite, about three minutes
uv run python scripts/run_examples.py                  # every example program compiles
uv run python scripts/run_examples.py --stage run      # every example program runs
```

If you touched the book (`docs/book/`), also run the prose lint:

```bash
uv run python scripts/lint_book.py
```

The book's code examples are part of the test suite, so the first
command already checks that they still print what the text says.

## Writing a test

Tests live in `src/metaxu/compiler/tests/`. A test is a Metaxu program
as a string (or a file under `tests/fixtures/`) that goes through the
whole pipeline, then an assertion about the result: the printed output,
the returned value, or the error the compiler raised. `test_book_examples.py`
and `test_example_gates.py` are short and show the shape.

Do not construct HIR or MIR by hand in a test. The bugs this project has
had were in the seams between stages, where a construct was silently
turned into something weaker, and a hand-built fixture starts after the
seam.

A test for a compile error checks the error type and a fragment of the
message. Every diagnostic carries a file, line and column, and
`docs/diagnostics_locations.md` describes how that location travels from
the parser to the message.

## What a change has to satisfy

- New compiler behavior comes with a regression test in the same
  commit.
- Both example gates stay at 21 of 21.
- When a program is wrong, the compiler or the interpreter rejects it
  with a message. Do not add a fallback that lets it run so a test
  passes.
- The interpreter (`mir_interp.py`) is the reference for what a program
  means. A change to the LLVM backend needs a test that runs the program
  natively and compares stdout and the exit code with the interpreter.
- A change that alters what a book example prints updates the chapter in
  the same commit.
- The standard library in `std/` is written in Metaxu. Writing real code
  there is the best stress test the compiler has, and several bugs were
  only found that way, so extending `std/` is a welcome contribution.

## Documentation

Design notes live in `docs/`. Each covers one area (type inference,
modules, packages, GPU tiles, diagnostics) and says what is implemented
and what is not. When a change moves one of those lines, update the
note. `docs/v1_gap_analysis.md` is the summary of what works today.

The book in `docs/book/` is the user-facing reference. Its prose is
linted: no em or en dashes, and code fences must name a language the
harness knows. `scripts/build_book_site.py` renders it into `website/book/`.

## The website

`website/` holds metaxulang.org. The pages are static and deploy through
`.github/workflows/pages.yml` on a push to `main` that touches the site
or the book. `website/README.md` explains the build and the domain
setup.

## Branches and commits

Work on a branch and open a pull request against `main`. Write the
commit subject in the imperative and keep it under about 70 characters.
Use the body to say why the change was made and what you checked,
because the diff already shows what changed.

## Questions

Open an issue. There is no chat channel yet.
