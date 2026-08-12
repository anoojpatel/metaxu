"""Every example program and repo-root .mx test file must parse.

Parsing is the hard requirement here: each file must go through the PLY
front end without a ParseError.  As of the parser overhaul all of these
files also pass the *full* pipeline (parse -> desugar -> freeze -> infer ->
HIR -> MIR -> CLIF), which is checked by a second, separate test so a
regression in a later stage is distinguishable from a parse regression.

There are currently no known-bad full-pipeline cases; if a later change
breaks a specific file downstream of the parser, move that file into
KNOWN_BAD_PIPELINE below (with a comment) rather than weakening the parse
assertion.
"""

from pathlib import Path

import pytest

from metaxu.parser import Parser
from metaxu.compiler.pipeline import run_pipeline_from_source

REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent

# Files whose full pipeline is known to fail (parse must still succeed).
# Currently empty: all example files pass the full pipeline.
KNOWN_BAD_PIPELINE: set[str] = set()


def _collect_mx_files():
    files = sorted((REPO_ROOT / "examples").glob("*.mx"))
    files += sorted(REPO_ROOT.glob("test_*.mx"))
    assert files, f"no .mx files found under {REPO_ROOT}"
    return files


MX_FILES = _collect_mx_files()


@pytest.mark.parametrize("path", MX_FILES, ids=lambda p: p.name)
def test_example_parses(path):
    """Each example/root .mx file must parse without a ParseError."""
    source = path.read_text()
    parser = Parser()  # fresh parser per file (module-name registry is stateful)
    module = parser.parse(source, file_path=str(path))
    assert module is not None


@pytest.mark.parametrize("path", MX_FILES, ids=lambda p: p.name)
def test_example_full_pipeline(path):
    """Each example/root .mx file should run the full pipeline to CLIF."""
    if path.name in KNOWN_BAD_PIPELINE:
        pytest.skip("known-bad full-pipeline case (parse-only guaranteed)")
    source = path.read_text()
    ast_json, hir_txt, mir_txt, clif_txt = run_pipeline_from_source(source)
    assert ast_json
