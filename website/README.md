# Metaxu website

The site is static: `index.html` (the landing page), `roadmap.html`
(the staged roadmap) and `book/` (the Metaxu Book, rendered from
`docs/book/*.md` by `scripts/build_book_site.py`). Fonts are base64
embedded and all CSS/JS/SVG is inline, so a page makes zero network
requests at view time. Pages cross-link relatively, so deploy them side
by side.

The pages are authored to Claude Artifact conventions: each starts at
`<title>` with no `<!DOCTYPE>`/`<html>`/`<head>`/`<body>` shell, because
the artifact publisher adds that shell. `scripts/build_site.py` adds the
same shell for a real host (see below). Every number on the landing page
comes from `README.md`, `benchmarks/suite/README.md` and
`docs/gpu_tiles.md`; update those first, then mirror here.

## Deploying to metaxulang.org

`.github/workflows/pages.yml` publishes the site to GitHub Pages on
every push to `main` or `v1-compiler` that touches `website/`,
`docs/book/` or the build scripts (and on manual dispatch). The build
step is `python scripts/build_site.py --out _site`, which:

1. rebuilds `website/book/` from `docs/book/*.md`, so the deployed book
   never lags the Markdown;
2. copies `website/` into `_site/`, skipping `design/` and this README;
3. wraps every page that lacks a doctype in a minimal shell (`lang`,
   charset, viewport, description);
4. writes `CNAME` (`metaxulang.org`), `.nojekyll` and a `404.html`.

Run the same command locally and serve `_site/` with any static server
to preview exactly what will be deployed. `_site/` is git-ignored.

### One-time repository settings

In the GitHub repository, Settings, Pages:

- Build and deployment, Source: **GitHub Actions** (not "Deploy from a
  branch"; the workflow uploads its own artifact).
- Custom domain: `metaxulang.org`, then Save. GitHub checks the DNS
  below and issues a Let's Encrypt certificate once it resolves; that
  can take up to an hour after the records propagate.
- Tick **Enforce HTTPS** once the certificate is issued.

Optionally verify the domain under the account's Settings, Pages,
"Verified domains", which stops anyone else's Pages site from claiming
it if the CNAME is ever removed.

### DNS records at the registrar

| host | type  | value                     |
|------|-------|---------------------------|
| `@`  | A     | `185.199.108.153`         |
| `@`  | A     | `185.199.109.153`         |
| `@`  | A     | `185.199.110.153`         |
| `@`  | A     | `185.199.111.153`         |
| `@`  | AAAA  | `2606:50c0:8000::153`     |
| `@`  | AAAA  | `2606:50c0:8001::153`     |
| `@`  | AAAA  | `2606:50c0:8002::153`     |
| `@`  | AAAA  | `2606:50c0:8003::153`     |
| `www`| CNAME | `anoojpatel.github.io.`   |

Remove any existing A/AAAA/CNAME records on the apex first (registrar
parking pages usually leave one behind). With `www` pointed at the
Pages host, GitHub redirects `www.metaxulang.org` to the apex.

Check propagation with `dig +short metaxulang.org A` (should list the
four addresses) and `dig +short www.metaxulang.org CNAME`.
