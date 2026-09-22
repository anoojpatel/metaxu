"""Assemble the deployable static site for metaxulang.org.

The pages under website/ are authored to Claude Artifact conventions:
they start at `<title>` with no `<!DOCTYPE>`/`<html>`/`<head>`/`<body>`
shell, because the artifact publisher adds one.  A real web host needs
the shell, so this script builds a deploy tree instead of publishing
website/ as-is:

  1. rebuild the book pages from docs/book/*.md (build_book_site.main),
     so the deployed book can never lag the Markdown;
  2. copy website/ into the output directory, skipping design
     scratch files and the README;
  3. wrap every .html page that lacks a doctype in a minimal shell
     (lang, charset, viewport, description);
  4. write CNAME (custom domain), .nojekyll (no Jekyll pass on GitHub
     Pages) and a small 404.html.

Usage: python scripts/build_site.py [--out _site] [--domain metaxulang.org]

Stdlib only, so the Pages workflow needs nothing beyond Python.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "website"
DOMAIN = "metaxulang.org"
DESCRIPTION = ("Metaxu: a systems language with algebraic effects, "
               "mode-based memory safety and tile-level GPU kernels, "
               "with a compiler written to be read.")
SKIP_DIRS = {"design"}
SKIP_FILES = {"README.md"}

SHELL_HEAD = ('<!DOCTYPE html>\n<html lang="en">\n<head>\n'
              '<meta charset="utf-8">\n'
              '<meta name="viewport" content="width=device-width,'
              'initial-scale=1">\n'
              '<meta name="description" content="{description}">\n')
SHELL_TAIL = "\n</body>\n</html>\n"

NOT_FOUND = """<title>Not found · Metaxu</title>
<link rel="stylesheet" href="/book/style.css">
<main style="max-width:40em;margin:12vh auto;padding:0 16px">
  <h1>Not found</h1>
  <p>There is no page at this address.
  <a href="/">Back to the front page</a> or
  <a href="/book/index.html">open the book</a>.</p>
</main>
"""


def wrap(html_text: str) -> str:
    """Prepend the document shell to an artifact-style page.

    The shell opens <head> without closing it: <title>, <link> and
    <style> that follow are head content, and the first body element
    (<header>, <nav>, <main>) closes head and opens body implicitly,
    exactly as the HTML parser specifies.  Pages that already carry a
    doctype are returned unchanged.
    """
    if html_text.lstrip()[:15].lower().startswith("<!doctype"):
        return html_text
    return SHELL_HEAD.format(description=DESCRIPTION) + html_text + SHELL_TAIL


def build(out: Path, domain: str, rebuild_book: bool = True) -> int:
    if rebuild_book:
        sys.path.insert(0, str(REPO / "scripts"))
        import build_book_site  # noqa: WPS433 (script import by design)
        build_book_site.main()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    pages = 0
    for path in sorted(SRC.rglob("*")):
        rel = path.relative_to(SRC)
        if path.is_dir() or rel.parts[0] in SKIP_DIRS or path.name in SKIP_FILES:
            continue
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".html":
            dest.write_text(wrap(path.read_text()))
            pages += 1
        else:
            shutil.copy2(path, dest)
    (out / "404.html").write_text(wrap(NOT_FOUND))
    (out / ".nojekyll").write_text("")
    if domain:
        (out / "CNAME").write_text(domain + "\n")
    print(f"assembled {pages} pages into {out} for {domain or '(no domain)'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, default=REPO / "_site")
    ap.add_argument("--domain", default=DOMAIN,
                    help="custom domain for CNAME ('' to omit)")
    ap.add_argument("--no-book", action="store_true",
                    help="skip regenerating website/book from docs/book")
    args = ap.parse_args(argv)
    return build(args.out, args.domain, rebuild_book=not args.no_book)


if __name__ == "__main__":
    sys.exit(main())
