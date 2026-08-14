from __future__ import annotations

from dataclasses import dataclass

from .mutaxu_ast import AstNode, Span, dump_ast_json, build_frozen_ast_with_map
from .infer_tables import InferSideTables
from .frozen_borrow_checker import BorrowCheckError, TypeCheckError
from .hir import HIRBuilder, dump_hir
from .lower_hir_to_mir import lower_hir_to_mir
from .mir import dump_mir
from .codegen_clif import emit_clif


def run_pipeline(
    ast_root: AstNode,
    tables: InferSideTables,
    id_map: dict[int, object] | None = None,
    strict: bool = True,
    monomorphize: bool = False,
) -> tuple[str, str, str]:
    """Run the minimal pipeline and return (hir_txt, mir_txt, clif_txt).

    When `strict` is True (the default) and the borrow checker reported any
    errors, a BorrowCheckError carrying the structured error list is raised
    instead of silently compiling the broken program. Pass strict=False to
    lower anyway (e.g. for diagnostics tooling).

    When `monomorphize` is True, generic functions are cloned per concrete
    instantiation after HIR construction (see compiler/monomorphize.py);
    default off — behavior is identical either way, the specialized names
    are groundwork for native codegen.
    """
    # Structured diagnostics from the frozen checker are stored under key -2.
    # Type errors (kind "type-*") surface as TypeCheckError; the rest are
    # borrow/locality/effect errors and surface as BorrowCheckError.
    all_errors = list(tables.constraints.get(-2, []))
    type_errors = [e for e in all_errors if getattr(e, "kind", "").startswith("type-")]
    borrow_errors = [e for e in all_errors if e not in type_errors]
    # Inference-level diagnostics live under key -1. Most are advisory
    # ("Unresolved callee ..."), but constraint-graph class conflicts are
    # hard type errors (structured: kind == "type-conflict").
    type_errors += [e for e in tables.constraints.get(-1, ())
                    if getattr(e, "kind", "") == "type-conflict"]
    if strict and type_errors:
        raise TypeCheckError(type_errors)
    if strict and borrow_errors:
        raise BorrowCheckError(borrow_errors)
    hir_funcs = HIRBuilder(tables, id_map=id_map).build(ast_root)
    if monomorphize:
        from .monomorphize import monomorphize_hir, collect_signatures
        hir_funcs = monomorphize_hir(hir_funcs, collect_signatures(id_map or {}))
    hir_txt = dump_hir(hir_funcs)
    mir_funcs = lower_hir_to_mir(hir_funcs, borrow_errors)
    mir_txt = dump_mir(mir_funcs)
    clif_txt = emit_clif(mir_funcs)
    return hir_txt, mir_txt, clif_txt


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
    """Build a PhaseContext by running the full front-end phase sequence.

    This is the single front door for source input. Phases:
      1. parse                       -> mutable AST
      2. freeze + tables (prelim)    -> trait/type info for desugaring
      3. desugar (with tables)       -> mutable AST (trait dictionaries etc.)
      4. freeze (final)              -> frozen AST + id_map
      5. tables (final)              -> InferSideTables incl. borrow errors

    The preliminary freeze/tables round exists so desugaring passes that need
    analysis results (e.g. TraitDictionaryDesugarPass needs trait_impls) get a
    populated DesugarContext instead of tables=None.
    """
    import metaxu.metaxu_ast as fast
    from .infer_tables import build_tables_from_frozen_via_simplesub
    from .desugar import run_default_desugaring, DesugarContext
    from .module_loader import resolve_modules
    from .shared_parser import shared_parser

    module = shared_parser().parse(source, file_path=file_path)
    program = fast.Program([module]) if not isinstance(module, fast.Program) else module

    # Module resolution: load imported files, enforce visibility, namespace
    # module functions, rewrite references. Programs that use no module
    # constructs are returned untouched (single-file backward compatibility).
    program = resolve_modules(program, file_path=file_path)

    # Preliminary analysis over the un-desugared program so desugaring passes
    # can consult traits/trait_impls/types.
    prelim_frozen, _prelim_id_map = build_frozen_ast_with_map(program)
    prelim_tables = build_tables_from_frozen_via_simplesub(prelim_frozen)

    desugar_ctx = DesugarContext(
        source=source,
        file_path=file_path,
        traits=dict(prelim_tables.traits or {}),
        trait_impls=dict(prelim_tables.trait_impls or {}),
        tables=prelim_tables,
    )
    program = run_default_desugaring(program, desugar_ctx)

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


def run_pipeline_ctx(ctx: PhaseContext, strict: bool = True,
                     monomorphize: bool = False) -> tuple[str, str, str, str]:
    """Run the pipeline using a prebuilt PhaseContext.

    Returns (ast_json, hir_txt, mir_txt, clif_txt). Raises BorrowCheckError
    when strict (default) and the program failed borrow checking.
    """
    ast_json = dump_ast_json(ctx.frozen_root)
    hir_txt, mir_txt, clif_txt = run_pipeline(
        ctx.frozen_root, ctx.tables, id_map=ctx.id_map, strict=strict,
        monomorphize=monomorphize)
    return ast_json, hir_txt, mir_txt, clif_txt


def emit_llvm_from_source(source: str, strict: bool = True,
                          file_path: str = "<mem>") -> str:
    """Parse, check, lower to MIR and emit an LLVM IR module (text).

    Separate entry point from run_pipeline_from_source (whose CLIF-returning
    signature is unchanged).  Raises BorrowCheckError / TypeCheckError in
    strict mode exactly like the main pipeline.  Pass `file_path` when the
    source lives on disk so multi-file imports resolve relative to its
    directory (exactly like run_pipeline_from_source).
    """
    from .codegen_llvm import emit_llvm

    ctx = build_context_from_source(source, file_path=file_path)
    run_pipeline_ctx(ctx, strict=strict)  # strict type/borrow gate
    hir_funcs = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    mir_funcs = lower_hir_to_mir(hir_funcs)
    return emit_llvm(mir_funcs)


def run_pipeline_from_source(source: str, strict: bool = True,
                             file_path: str = "<mem>",
                             monomorphize: bool = False) -> tuple[str, str, str, str]:
    """Parse, desugar, type/borrow check, and run the pipeline from source.

    Thin wrapper over build_context_from_source + run_pipeline_ctx so that
    both front doors run the identical phase sequence. Pass `file_path` when
    the source lives on disk so multi-file imports can resolve relative to
    its directory.

    Returns (ast_json, hir_txt, mir_txt, clif_txt). Raises BorrowCheckError
    when strict (default) and the program failed borrow checking.
    """
    ctx = build_context_from_source(source, file_path=file_path)
    return run_pipeline_ctx(ctx, strict=strict, monomorphize=monomorphize)
