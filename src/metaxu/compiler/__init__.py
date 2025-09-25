"""Compiler pipeline package for Metaxu.

Submodules:
- mutaxu_ast: immutable AST mirror with node_id/span and JSON dumping utilities
- types: thin wrappers around Ty, EffectSet, TyEnv and unification interface
- infer_tables: adapters that read the existing SimpleSub side tables
- constraints: ClassConstraints, InstanceHeads, FD registry and solver
- impl_registry: registry for trait dictionaries and instance heads
- hir: typed High-level IR and builder (AST → HIR)
- mir: MIR (ANF-direct + CPS) data structures
- lower_hir_to_mir: lowering passes from HIR to MIR
- codegen_clif: Cranelift IR string emitter

Python 3.11+
"""

from . import mutaxu_ast as ast
from . import types
from . import infer_tables
from . import constraints
from . import impl_registry
from . import hir
from . import mir
from . import lower_hir_to_mir
from . import codegen_clif

__all__ = [
    "ast",
    "types",
    "infer_tables",
    "constraints",
    "impl_registry",
    "hir",
    "mir",
    "lower_hir_to_mir",
    "codegen_clif",
]
