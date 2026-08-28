# Metaxu website

`index.html` is the whole site: one self-contained static page (fonts base64-embedded, all CSS/JS/SVG inline, zero network requests at view time).
It is authored to Claude Artifact conventions — it starts at `<title>` with no `<!DOCTYPE>`/`<html>`/`<head>`/`<body>` wrapper, because the artifact publisher adds that shell.
To deploy on GitHub Pages (or any static host), either serve it as-is (browsers tolerate the missing wrapper) or prepend the trivial shell: `<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>` … `</body></html>`.
Every number on the page comes from README.md, benchmarks/suite/README.md, and docs/gpu_tiles.md — update those first, then mirror here.
