from __future__ import annotations

from pathlib import Path

from metaxu.compiler.mutaxu_ast import AstNode, Span, dump_ast_json
from metaxu.compiler.infer_tables import InferSideTables
from metaxu.compiler.pipeline import run_pipeline


def build_min_ast() -> AstNode:
    return AstNode(node_id=1, kind="Fn", children=tuple(), span=Span(file="<mem>", start=0, end=0))


def build_tables() -> InferSideTables:
    return InferSideTables(
        types={1: "Unit"},
        effects={},
        suspends=set(),
        symbols={1: "main"},
        constraints={},
        tyenv=type("TyEnvStub", (), {"apply": lambda self, t: t})(),
    )


def test_pipeline_golden(tmp_path: Path) -> None:
    root = build_min_ast()
    tables = build_tables()
    # Dump AST json
    ast_json = dump_ast_json(root)

    base = Path(__file__).parent / "golden"
    hir_txt, mir_txt, clif_txt = run_pipeline(root, tables)

    assert ast_json.strip() == (base / "sample1.ast.json").read_text().strip()
    assert hir_txt.strip() == (base / "sample1.hir.txt").read_text().strip()
    assert mir_txt.strip() == (base / "sample1.mir.txt").read_text().strip()
    assert clif_txt.strip() == (base / "sample1.clif.txt").read_text().strip()
