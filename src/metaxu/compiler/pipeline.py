from __future__ import annotations

from dataclasses import dataclass

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
    # Get borrow errors from frozen borrow checker (stored under key -2)
    borrow_errors = tables.constraints.get(-2, [])
    mir_funcs = lower_hir_to_mir(hir_funcs, borrow_errors)
    mir_txt = dump_mir(mir_funcs)
    clif_txt = emit_clif(mir_funcs)
    return hir_txt, mir_txt, clif_txt


def run_pipeline_from_source(source: str) -> tuple[str, str, str, str]:
    """Parse with the real parser, type-check with frozen AST borrow checker, then run the pipeline.

    Returns (ast_json, hir_txt, mir_txt, clif_txt).
    """
    from metaxu.parser import Parser
    import metaxu.metaxu_ast as fast
    from .infer_tables import build_tables_from_frozen_via_simplesub
    from .desugar import run_default_desugaring, DesugarContext

    parser = Parser()
    program = parser.parse(source)
    
    # Run desugaring passes on mutable AST before freezing
    desugar_ctx = DesugarContext(source=source)
    desugared_program = run_default_desugaring(program, desugar_ctx)
    
    # Build frozen AST from desugared program
    frozen_root, id_map = build_frozen_ast_with_map(desugared_program)
    
    # Build tables from frozen AST
    tables = build_tables_from_frozen_via_simplesub(frozen_root)

    ast_json = dump_ast_json(frozen_root)
    hir_txt, mir_txt, clif_txt = run_pipeline(frozen_root, tables, id_map=id_map)
    return ast_json, hir_txt, mir_txt, clif_txt


@dataclass(slots=True)
class PhaseContext:
    """Aggregates immutable analysis products for downstream phases.

    Keep the original AST unmodified; carry all derived info here.
    """
    source: str | None
    file_path: str | None
    program: object
    frozen_root: AstNode
    id_map: dict[int, object]
    tables: InferSideTables
    type_checker: object | None  # Optional since we use frozen AST borrow checker


def build_context_from_source(source: str, file_path: str = "<mem>") -> PhaseContext:
    """Build a PhaseContext by parsing + type checking with frozen AST borrow checker without mutating AST."""
    from metaxu.parser import Parser
    import metaxu.metaxu_ast as fast
    from .infer_tables import build_tables_from_frozen_via_simplesub

    parser = Parser()
    module = parser.parse(source, file_path=file_path)
    program = fast.Program([module]) if not isinstance(module, fast.Program) else module

    frozen_root, id_map = build_frozen_ast_with_map(program)
    tables = build_tables_from_frozen_via_simplesub(frozen_root)
    return PhaseContext(
        source=source,
        file_path=file_path,
        program=program,
        frozen_root=frozen_root,
        id_map=id_map,
        tables=tables,
        type_checker=None,  # Not using old TypeChecker anymore
    )


def run_pipeline_ctx(ctx: PhaseContext) -> tuple[str, str, str, str]:
    """Run the pipeline using a prebuilt PhaseContext.

    Returns (ast_json, hir_txt, mir_txt, clif_txt).
    """
    ast_json = dump_ast_json(ctx.frozen_root)
    hir_txt, mir_txt, clif_txt = run_pipeline(ctx.frozen_root, ctx.tables, id_map=ctx.id_map)
    return ast_json, hir_txt, mir_txt, clif_txt
