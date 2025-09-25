from __future__ import annotations

from .mutaxu_ast import AstNode, Span, dump_ast_json, build_frozen_ast_with_map
from .infer_tables import InferSideTables, build_from_ast_and_typechecker
from .hir import HIRBuilder, dump_hir
from .lower_hir_to_mir import lower_hir_to_mir
from .mir import dump_mir
from .codegen_clif import emit_clif


def run_pipeline(ast_root: AstNode, tables: InferSideTables, id_map: dict[int, object] | None = None) -> tuple[str, str, str]:
    """Run the minimal pipeline and return (hir_txt, mir_txt, clif_txt)."""
    hir_funcs = HIRBuilder(tables, id_map=id_map).build(ast_root)
    hir_txt = dump_hir(hir_funcs)
    mir_funcs = lower_hir_to_mir(hir_funcs)
    mir_txt = dump_mir(mir_funcs)
    clif_txt = emit_clif(mir_funcs)
    return hir_txt, mir_txt, clif_txt


def run_pipeline_from_source(source: str, file_path: str = "<mem>") -> tuple[str, str, str, str]:
    """Parse with the real parser, type-check with TypeChecker/SimpleSub, then run the pipeline.

    Returns (ast_json, hir_txt, mir_txt, clif_txt).
    """
    from metaxu.parser import Parser
    from metaxu.type_checker import TypeChecker
    import metaxu.metaxu_ast as fast

    parser = Parser()
    module = parser.parse(source, file_path=file_path)
    # Wrap in Program for TypeChecker entry point
    program = fast.Program([module]) if not isinstance(module, fast.Program) else module

    tc = TypeChecker()
    tc.check(program)

    frozen_root, id_map = build_frozen_ast_with_map(program)
    tables = build_from_ast_and_typechecker(id_map, tc)

    ast_json = dump_ast_json(frozen_root)
    hir_txt, mir_txt, clif_txt = run_pipeline(frozen_root, tables, id_map=id_map)
    return ast_json, hir_txt, mir_txt, clif_txt
