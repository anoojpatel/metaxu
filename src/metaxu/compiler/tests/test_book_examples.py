"""The Metaxu Book's examples, executed (docs/book/).

Every ```metaxu fence in the book is a claim about the language, and
claims need tests: this harness extracts each one and runs it through
the real pipeline (parse -> check -> interpret), exactly like every
other test in this suite.  A book example cannot rot without failing CI.

The fence contract (docs/book/README.md restates it for authors):

  * ```metaxu               a complete program; must run and exit 0.
                            If the NEXT fence is ```output, stdout must
                            match it byte for byte (trailing-newline
                            normalized).  A bare block with no output
                            fence is an authoring error: pin the output
                            or mark the block.
  * ```metaxu error         must be REJECTED at compile time; the
                            following ```output fence holds a fragment
                            the diagnostic must contain.
  * ```metaxu norun         shown, not executed (fragments, multi-file
                            examples).  Use sparingly; the point of the
                            book is that its examples run.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from metaxu.compiler.frozen_borrow_checker import (BorrowCheckError,
                                                   TypeCheckError)

from metaxu.compiler.tests.test_codegen_llvm import interp_run

BOOK = Path(__file__).resolve().parents[4] / "docs" / "book"

_FENCE = re.compile(
    r"^```metaxu([^\n`]*)\n(.*?)^```[ \t]*$", re.M | re.S)
_OUTPUT = re.compile(r"\A(?:\s*\n)*```output[^\n]*\n(.*?)^```[ \t]*$",
                     re.M | re.S)


def _examples():
    """Yield (id, kind, source, expected) for every metaxu fence."""
    assert BOOK.is_dir(), f"missing {BOOK}"
    found = False
    for md in sorted(BOOK.glob("*.md")):
        text = md.read_text()
        for i, m in enumerate(_FENCE.finditer(text), start=1):
            found = True
            flags = m.group(1).split()
            src = m.group(2)
            line = text[:m.start()].count("\n") + 1
            eid = f"{md.name}:{line}"
            tail = text[m.end():]
            out = _OUTPUT.match(tail)
            expected = out.group(1) if out else None
            if "norun" in flags:
                yield eid, "norun", src, None
            elif "error" in flags:
                yield eid, "error", src, expected
            else:
                yield eid, "run", src, expected
    assert found, "no ```metaxu examples found in docs/book"


_CASES = list(_examples())


@pytest.mark.parametrize("eid,kind,src,expected", _CASES,
                         ids=[c[0] for c in _CASES])
def test_book_example(eid, kind, src, expected):
    if kind == "norun":
        pytest.skip("norun block (display only)")
    if kind == "error":
        assert expected is not None, (
            f"{eid}: error blocks need an ```output fence holding a "
            "fragment of the expected diagnostic")
        with pytest.raises((TypeCheckError, BorrowCheckError)) as ei:
            interp_run(src)
        frag = expected.strip()
        assert frag and frag in str(ei.value), (
            f"{eid}: diagnostic does not contain the pinned fragment\n"
            f"--- pinned ---\n{frag}\n--- actual ---\n{ei.value}")
        return
    assert expected is not None, (
        f"{eid}: pin the output with an ```output fence (or mark the "
        "block norun)")
    result, out = interp_run(src)
    assert result == 0, f"{eid}: exited {result}\n{out}"
    assert out.rstrip("\n") == expected.rstrip("\n"), (
        f"{eid}: output mismatch\n--- expected ---\n{expected}"
        f"\n--- actual ---\n{out}")


def test_book_has_substance():
    """The book exists and its chapters carry executed examples."""
    runnable = [c for c in _CASES if c[1] == "run"]
    norun = [c for c in _CASES if c[1] == "norun"]
    assert len(runnable) >= 40, len(runnable)
    # norun is a narrow escape hatch, not a lifestyle
    assert len(norun) <= len(runnable) // 4, (len(norun), len(runnable))
