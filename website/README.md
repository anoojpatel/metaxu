# Metaxu website

The site is two self-contained static pages: `index.html` (the landing/docs page) and `roadmap.html` (the staged roadmap), each with fonts base64-embedded and all CSS/JS/SVG inline, zero network requests at view time. They cross-link relatively, so deploy them side by side.
It is authored to Claude Artifact conventions — it starts at `<title>` with no `<!DOCTYPE>`/`<html>`/`<head>`/`<body>` wrapper, because the artifact publisher adds that shell.
To deploy on GitHub Pages (or any static host), either serve it as-is (browsers tolerate the missing wrapper) or prepend the trivial shell: `<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>` … `</body></html>`.
Every number on the page comes from README.md, benchmarks/suite/README.md, and docs/gpu_tiles.md — update those first, then mirror here.
