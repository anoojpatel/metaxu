from __future__ import annotations

import pytest

from metaxu.compiler.pipeline import build_context_from_source, run_pipeline_ctx


def test_pipeline_context_minimal():
    # Minimal program; parser should accept an empty main body
    src = """
    fn main() {
    }
    """
    ctx = build_context_from_source(src, file_path="<test>")
    ast_json, hir_txt, mir_txt, clif_txt = run_pipeline_ctx(ctx)

    assert isinstance(ast_json, str) and ast_json
    assert isinstance(hir_txt, str) and hir_txt
    assert isinstance(mir_txt, str) and mir_txt
    assert isinstance(clif_txt, str) and clif_txt
