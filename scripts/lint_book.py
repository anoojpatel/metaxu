"""Structural lint for the Metaxu Book (docs/book/*.md), no compiler needed.

test_book_examples.py executes every example; this script checks the
things the harness assumes BEFORE execution, so a chapter can't be
malformed in ways that make a green run meaningless:

- every ```metaxu fence (run or error) is followed directly by an
  ```output fence; norun fences are not
- runnable programs declare `fn main() -> int` and their last
  expression is a bare `0`
- error programs declare `fn main() -> int` too (the harness compiles
  the whole program)
- fences are balanced and info strings are only metaxu / metaxu error /
  metaxu norun / output / bash / python / text
- no tabs inside fences (the interpreter's excerpts assume spaces)
- output fences are non-empty for run and error examples
- prose contains no em-dashes (pinned program output may)

    uv run python scripts/lint_book.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent / "docs" / "book"
ALLOWED_INFO = {"metaxu", "output", "bash", "python", "text", ""}
ALLOWED_FLAGS = {"error", "norun"}


def lint(path: Path) -> list[str]:
    problems: list[str] = []
    lines = path.read_text().split("\n")
    fences: list[tuple[int, str, list[str], list[str]]] = []
    i = 0
    prose_lines: list[tuple[int, str]] = []
    while i < len(lines):
        m = re.match(r"^```([^`]*)$", lines[i])
        if not m:
            prose_lines.append((i + 1, lines[i]))
            i += 1
            continue
        parts = m.group(1).split()
        info, flags = (parts[0] if parts else ""), parts[1:]
        start = i + 1
        body: list[str] = []
        i += 1
        while i < len(lines) and not lines[i].startswith("```"):
            body.append(lines[i])
            i += 1
        if i >= len(lines):
            problems.append(f"{start}: unclosed fence ```{m.group(1)}")
            break
        if info not in ALLOWED_INFO:
            problems.append(f"{start}: unknown fence info '{info}'")
        if info != "metaxu" and flags:
            problems.append(f"{start}: flags on a non-metaxu fence: {flags}")
        for f in flags:
            if f not in ALLOWED_FLAGS:
                problems.append(f"{start}: unknown metaxu flag '{f}'")
        if any("\t" in b for b in body):
            problems.append(f"{start}: tab character inside fence")
        fences.append((start, info, flags, body))
        i += 1

    for idx, (start, info, flags, body) in enumerate(fences):
        if info != "metaxu":
            continue
        kind = ("norun" if "norun" in flags else
                "error" if "error" in flags else "run")
        src = "\n".join(body)
        nxt = fences[idx + 1] if idx + 1 < len(fences) else None
        if kind == "norun":
            if nxt and nxt[1] == "output" and nxt[0] == start + len(body) + 2:
                problems.append(f"{start}: norun fence followed by output")
            continue
        if not (nxt and nxt[1] == "output"
                and nxt[0] == start + len(body) + 2):
            problems.append(f"{start}: {kind} example lacks a directly "
                            "following ```output fence")
        elif not "\n".join(nxt[3]).strip():
            problems.append(f"{start}: empty output fence for {kind} example")
        if not re.search(r"^fn main\(\) -> int \{", src, re.M):
            problems.append(f"{start}: {kind} example has no "
                            "`fn main() -> int {`")
        if kind == "run":
            tail = [b for b in body if b.strip()]
            # the program's last two meaningful lines are `    0` and `}`
            if len(tail) < 2 or tail[-1].strip() != "}" or \
                    tail[-2].strip() != "0":
                problems.append(f"{start}: run example does not end in a "
                                "bare `0` tail before the closing brace")

    for n, line in prose_lines:
        if "—" in line and path.name not in ("README.md",
                                                 "01-getting-started.md"):
            problems.append(f"{n}: em-dash in prose")
    return problems


def main() -> int:
    bad = 0
    for path in sorted(BOOK.glob("*.md")):
        for p in lint(path):
            print(f"{path.name}:{p}")
            bad += 1
    total = len(list(BOOK.glob("*.md")))
    print(f"{total} files, {bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
