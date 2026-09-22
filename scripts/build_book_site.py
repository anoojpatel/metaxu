"""Render the Metaxu Book (docs/book/*.md) as website/book/*.html.

The book's source of truth stays the markdown (whose examples run in
CI); this generator renders it in the website's design system. Fonts
and the color tokens are extracted from website/index.html at build
time so the pages can't drift from the landing page's look.

    uv run python scripts/build_book_site.py

Outputs: website/book/index.html (from README.md), one page per
chapter, style.css (fonts + tokens + book layout), book.js (the same
regex highlighter the landing page uses, for ```metaxu fences).

The converter handles exactly the markdown this book uses: #/##/###
headings, paragraphs, fenced code with info strings, `inline code`
(and ````quoted-fence```` runs), links, **bold**, *emphasis*, bullet
and numbered lists, and pipe tables. Anything else renders as a plain
paragraph rather than guessing.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BOOK = REPO / "docs" / "book"
OUT = REPO / "website" / "book"
LANDING = REPO / "website" / "index.html"

BOOK_CSS = """
/* ---------- book layout (shared tokens above) ---------- */
* { box-sizing: border-box; }
body {
  margin: 0; background-color: var(--bg);
  background-image: linear-gradient(180deg, rgba(242, 239, 227, .8),
                                    rgba(242, 239, 227, 0) 420px);
  background-repeat: no-repeat;
  color: var(--fg);
  font-family: "Helvetica Neue", Helvetica, "Neue Haas Grotesk", Arial, sans-serif;
  font-size: 16px; line-height: 1.65;
  -webkit-font-smoothing: antialiased;
}
.mono, code, pre { font-family: 'Hasklig', 'Source Code Pro', ui-monospace, Menlo, monospace; }
a { color: var(--fg); text-decoration: underline; text-decoration-color: var(--dim); text-underline-offset: 3px; }
a:hover { color: var(--acc); text-decoration-color: var(--acc); }
.top { border-bottom: 1px solid var(--line); }
.top .inner {
  max-width: 780px; margin: 0 auto; padding: 16px 24px;
  display: flex; justify-content: space-between; align-items: baseline;
  gap: 16px; flex-wrap: wrap;
}
.brand { font-family: 'Space Grotesk', 'Helvetica Neue', Arial, sans-serif; font-weight: 700;
  letter-spacing: .08em; text-transform: uppercase; font-size: 15px;
  text-decoration: none; }
.topnav { display: flex; gap: 18px; flex-wrap: wrap; }
.topnav a { font-family: 'Hasklig', ui-monospace, monospace; font-size: 12.5px;
  color: var(--mut); text-decoration: none; letter-spacing: .04em; }
.topnav a:hover { color: var(--acc); }
main { max-width: 780px; margin: 0 auto; padding: 40px 24px 80px; }
h1, h2, h3 { font-family: 'EB Garamond', Georgia, serif; font-weight: 700;
  letter-spacing: 0; }
h1 { font-size: clamp(34px, 5vw, 46px);
  line-height: 1.1; margin: 8px 0 20px; text-wrap: balance; }
h2 { font-size: 27px; line-height: 1.2; margin: 44px 0 12px; }
h3 { font-size: 21px; margin: 32px 0 10px; }
p { max-width: 70ch; }
li { max-width: 66ch; margin-bottom: 6px; }
code {
  font-size: .9em; background: var(--panel); border: 1px solid var(--line);
  padding: 1px 5px; border-radius: 2px; font-feature-settings: "calt" 1;
}
pre {
  background: var(--panel); border: 1px solid var(--line);
  padding: 16px 18px; overflow-x: auto; font-size: 13.5px; line-height: 1.6;
  margin: 0;
}
pre code { background: none; border: none; padding: 0; font-size: inherit; }
.fence { margin: 20px 0; }
.fence + .fence { margin-top: 10px; }
.fence .label {
  font-family: 'Hasklig', ui-monospace, monospace; font-size: 11px;
  letter-spacing: .08em; color: var(--dim); text-transform: uppercase;
  border: 1px solid var(--line); border-bottom: none;
  display: inline-block; padding: 3px 10px; background: var(--panel2);
}
.fence.output .label { color: var(--acc-dim); }
.fence.error .label { color: #a05a2c; }
.fence.output pre { border-left: 2px solid var(--acc-dim); }
table { border-collapse: collapse; margin: 20px 0; width: 100%; }
th, td { border: 1px solid var(--line); padding: 8px 12px; text-align: left;
  font-size: 14.5px; }
th { font-family: 'Hasklig', ui-monospace, monospace; font-size: 11.5px;
  text-transform: uppercase; letter-spacing: .08em; color: var(--dim); }
td code { white-space: nowrap; }
.tablewrap { overflow-x: auto; }
.pager { display: flex; justify-content: space-between; gap: 16px;
  margin-top: 56px; border-top: 1px solid var(--line); padding-top: 18px; }
.pager a { font-family: 'Hasklig', ui-monospace, monospace; font-size: 13px;
  text-decoration: none; color: var(--mut); }
.pager a:hover { color: var(--acc); }
.tok-k { color: var(--acc); } .tok-t { color: #3e5c74; }
.tok-s { color: #5c7a3f; } .tok-n { color: #7a5c36; }
.tok-c { color: #7d8a80; } .tok-m { color: #4e7a8a; }
"""

BOOK_JS = r"""
(function () {
  var RE = new RegExp(
    '(\\/\\/[^\\n]*|#[^\\n]*)' +
    '|("(?:[^"\\\\]|\\\\.)*")' +
    '|(\\b(?:fn|let|mut|while|for|perform|effect|handle|resume|match|from|import|module|export|try|catch|if|else|in|with|performs|struct|enum|trait|implement|implements|return|unsafe|extern|exclave|where|type)\\b)' +
    '|(@[a-z_]+)' +
    '|\\b(\\d+(?:\\.\\d+)?)\\b' +
    '|\\b([A-Z][A-Za-z0-9_]*)\\b',
    'g');
  function esc(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  document.querySelectorAll('code.mx').forEach(function (el) {
    var src = el.textContent, out = '', last = 0, m;
    RE.lastIndex = 0;
    while ((m = RE.exec(src)) !== null) {
      out += esc(src.slice(last, m.index));
      var cls = m[1] ? 'tok-c' : m[2] ? 'tok-s' : m[3] ? 'tok-k'
              : m[4] ? 'tok-m' : m[5] ? 'tok-n' : 'tok-t';
      out += '<span class="' + cls + '">' + esc(m[0]) + '</span>';
      last = m.index + m[0].length;
    }
    out += esc(src.slice(last));
    el.innerHTML = out;
  });
})();
"""


def extract_shared_style() -> str:
    """Fonts + color tokens from the landing page, verbatim."""
    landing = LANDING.read_text()
    faces = re.findall(r"@font-face \{.*?\n\}", landing, re.S)
    root = re.search(r":root \{.*?\n\}", landing, re.S)
    assert faces and root, "landing page style blocks not found"
    return "\n".join(faces) + "\n\n" + root.group(0) + "\n" + BOOK_CSS


def inline_md(text: str) -> str:
    """Inline markdown -> HTML (code spans first, so nothing inside
    them is touched)."""
    parts = re.split(r"(````.*?````|`[^`]*`)", text)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # a code span
            inner = part[4:-4] if part.startswith("````") else part[1:-1]
            out.append(f"<code>{html.escape(inner.strip())}</code>")
            continue
        seg = html.escape(part)
        seg = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                     lambda m: '<a href="%s">%s</a>' % (
                         m.group(2).replace(".md", ".html")
                         if not m.group(2).startswith("http")
                         else m.group(2), m.group(1)), seg)
        seg = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", seg)
        seg = re.sub(r"(?<![\w*])\*([^\s*][^*]*?)\*(?![\w*])",
                     r"<em>\1</em>", seg)
        out.append(seg)
    return "".join(out)


def convert(md: str) -> tuple[str, str]:
    """Markdown -> (title, body HTML)."""
    lines = md.split("\n")
    out: list[str] = []
    title = "Metaxu Book"
    i = 0
    para: list[str] = []
    in_list: str | None = None

    def flush_para():
        nonlocal para
        if para:
            out.append(f"<p>{inline_md(' '.join(para))}</p>")
            para = []

    def close_list():
        nonlocal in_list
        if in_list:
            out.append(f"</{in_list}>")
            in_list = None

    while i < len(lines):
        line = lines[i]
        # A fence opener is exactly three backticks plus an info string
        # that may hold flags after a space ("metaxu error"); a fourth
        # backtick (an ````-quoted run) is not an opener.
        fence = re.match(r"^```([^`]*)$", line)
        if fence:
            flush_para(); close_list()
            parts = fence.group(1).split()
            info = parts[0] if parts else ""
            extra = parts[1:]
            block: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i]); i += 1
            code = html.escape("\n".join(block))
            if info == "metaxu" or info.startswith("metaxu"):
                kind = ("error" if "error" in extra else
                        "norun" if "norun" in extra else "run")
                label = {"run": "metaxu", "error": "metaxu · rejected",
                         "norun": "metaxu · display only"}[kind]
                cls = " error" if kind == "error" else ""
                out.append(f'<div class="fence{cls}"><span class="label">'
                           f'{label}</span><pre><code class="mx">{code}'
                           f'</code></pre></div>')
            elif info == "output":
                out.append('<div class="fence output"><span class="label">'
                           f'output</span><pre><code>{code}</code></pre>'
                           '</div>')
            else:
                label = info or "text"
                out.append(f'<div class="fence"><span class="label">{label}'
                           f'</span><pre><code>{code}</code></pre></div>')
            i += 1
            continue
        h = re.match(r"^(#{1,4}) (.*)$", line)
        if h:
            flush_para(); close_list()
            level = len(h.group(1))
            text = h.group(2)
            if level == 1:
                title = text
            out.append(f"<h{level}>{inline_md(text)}</h{level}>")
            i += 1
            continue
        if line.startswith("|"):
            flush_para(); close_list()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            out.append('<div class="tablewrap"><table>')
            for r, cells in enumerate(rows):
                tag = "th" if r == 0 else "td"
                out.append("<tr>" + "".join(
                    f"<{tag}>{inline_md(c)}</{tag}>" for c in cells)
                    + "</tr>")
            out.append("</table></div>")
            continue
        m = re.match(r"^(\s*)([-*]|\d+\.) (.*)$", line)
        if m:
            flush_para()
            want = "ol" if m.group(2)[0].isdigit() else "ul"
            if in_list != want:
                close_list()
                out.append(f"<{want}>")
                in_list = want
            item = [m.group(3)]
            i += 1
            while i < len(lines) and re.match(r"^\s{2,}\S", lines[i]) \
                    and not re.match(r"^\s*([-*]|\d+\.) ", lines[i]):
                item.append(lines[i].strip()); i += 1
            out.append(f"<li>{inline_md(' '.join(item))}</li>")
            continue
        if not line.strip():
            flush_para(); close_list()
            i += 1
            continue
        para.append(line.strip())
        i += 1
    flush_para(); close_list()
    return title, "\n".join(out)


def page(title: str, body: str, prev_: tuple | None,
         next_: tuple | None, is_index: bool) -> str:
    pager = ""
    if not is_index:
        left = (f'<a href="{prev_[0]}">&#8592; {html.escape(prev_[1])}</a>'
                if prev_ else '<a href="index.html">&#8592; Contents</a>')
        right = (f'<a href="{next_[0]}">{html.escape(next_[1])} &#8594;</a>'
                 if next_ else '<a href="index.html">Contents</a>')
        pager = f'<nav class="pager">{left}{right}</nav>'
    return f"""<title>{html.escape(title)}</title>
<link rel="stylesheet" href="style.css">
<header class="top">
  <div class="inner">
    <a class="brand" href="../index.html">Metaxu</a>
    <nav class="topnav">
      <a href="index.html">the book</a>
      <a href="../roadmap.html">roadmap</a>
      <a href="https://github.com/anoojpatel/metaxu">github &#8599;</a>
    </nav>
  </div>
</header>
<main>
{body}
{pager}
</main>
<script src="book.js"></script>
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "style.css").write_text(extract_shared_style())
    (OUT / "book.js").write_text(BOOK_JS)
    chapters = sorted(p for p in BOOK.glob("*.md") if p.name != "README.md")
    titles = {}
    for ch in chapters:
        first = ch.read_text().split("\n", 1)[0]
        titles[ch.name] = first.lstrip("# ").strip()
    for idx, ch in enumerate(chapters):
        title, body = convert(ch.read_text())
        prev_ = None
        if idx > 0:
            pn = chapters[idx - 1].name
            prev_ = (pn.replace(".md", ".html"), titles[pn])
        next_ = None
        if idx + 1 < len(chapters):
            nn = chapters[idx + 1].name
            next_ = (nn.replace(".md", ".html"), titles[nn])
        html_name = ch.name.replace(".md", ".html")
        (OUT / html_name).write_text(
            page(f"{title} · The Metaxu Book", body, prev_, next_, False))
    title, body = convert((BOOK / "README.md").read_text())
    (OUT / "index.html").write_text(page("The Metaxu Book", body,
                                         None, None, True))
    print(f"built {len(chapters) + 1} pages into {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
