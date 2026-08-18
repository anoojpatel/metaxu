from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Dict, Tuple

try:
    from metaxu.type_defs import CompactType as _CompactType
except Exception:  # pragma: no cover
    _CompactType = None  # type: ignore[assignment]


@dataclass(slots=True)
class InferSideTables:
    """Thin adapters around the existing SimpleSub side tables.

    Expect inputs (likely from src/metaxu/simplesub.py / type_checker) to be
    passed in during pipeline construction. These tables are treated as
    read-only views for the HIR builder and later stages.
    """

    types: Mapping[int, Any]
    effects: Mapping[int, Any]
    suspends: set[int]
    symbols: Mapping[int, Any]
    constraints: Mapping[int, Sequence[Any]]
    tyenv: Any
    traits: Mapping[int, Any] = None  # Trait definitions by node_id
    trait_impls: Mapping[str, Any] = None  # Trait implementations by type name
    # The SimpleSubFacade that produced these tables (or None for tables
    # built by hand). Kept so downstream consumers/tooling can ask for
    # principal types (facade.principal_type_of(node_id)) -- a lazy,
    # advisory query that never affects what compiles.
    facade: Any = None

    def ty_of(self, node_id: int) -> Any:
        return self.types[node_id]

    def effects_of(self, node_id: int) -> Any:
        return self.effects.get(node_id)

    def suspends_node(self, node_id: int) -> bool:
        return node_id in self.suspends

    def sym_of(self, node_id: int) -> Any:
        return self.symbols.get(node_id)

    def constraints_of(self, node_id: int) -> Sequence[Any]:
        return self.constraints.get(node_id, ())

    def apply_tyenv(self, ty: Any) -> Any:
        # Delegate to the provided TyEnv if it has apply
        if hasattr(self.tyenv, "apply"):
            return self.tyenv.apply(ty)  # type: ignore[attr-defined]
        return ty


def _make_tyenv_from_simplesub() -> Any:
    class _TyEnv:
        def apply(self, ty: Any) -> Any:  # noqa: D401
            try:
                if _CompactType is not None and isinstance(ty, _CompactType):
                    return ty.find()  # type: ignore[attr-defined]
            except Exception:
                pass
            return ty
    return _TyEnv()


def build_tables_from_frozen_via_simplesub(frozen_root: Any) -> InferSideTables:
    """Build inference side tables from frozen AST via SimpleSub constraint emission.

    This function:
    1. Allocates CompactType variables for each frozen node
    2. Walks the frozen AST emitting constraints to SimpleSub
    3. Runs constraint solving
    4. Maps results back to node IDs
    """
    # Allocate CompactType variables for each frozen node
    def walk(n: Any, acc: Dict[int, Any]) -> None:
        if n.node_id not in acc:
            acc[n.node_id] = _CompactType.fresh_var()  # type: ignore[union-attr]
        for c in n.children:
            walk(c, acc)

    types: Dict[int, Any] = {}
    walk(frozen_root, types)

    # Build a facade over SimpleSub that accepts CompactTypes and constraints
    from .simplesub_adapter import SimpleSubFacade  # type: ignore
    from .frozen_constraint_emitter import emit_constraints  # type: ignore
    ss = SimpleSubFacade(types)
    _, borrow_errors = emit_constraints(frozen_root, types, ss)
    ss.solve()

    # Resolve the node ids the checkers recorded into source positions, once,
    # from the frozen AST that owns the spans. Every consumer of these
    # diagnostics (strict-mode exceptions, tooling reading the tables) then
    # sees "file:line:column: message".
    from .frozen_borrow_checker import locate_errors  # type: ignore
    locate_errors(borrow_errors, frozen_root)
    locate_errors(ss.errors, frozen_root)

    constraints: Dict[int, Any] = {0: ss.constraints}
    if ss.errors:
        constraints[-1] = tuple(ss.errors)

    # Add borrow checking errors
    if borrow_errors:
        constraints[-2] = tuple(borrow_errors)

    effects: Dict[int, set[str]] = {}
    suspends: set[int] = set()
    traits: Dict[int, Any] = {}
    trait_impls: Dict[str, Any] = {}
    
    # Walk frozen AST to collect trait definitions and implementations
    def collect_traits(n: Any) -> None:
        if n.kind == "TraitDefinition":
            traits[n.node_id] = n
        elif n.kind == "TraitImplementation":
            # Collect trait implementation info
            struct_name = n.value.get("struct_name") if n.value else None
            trait_name = n.value.get("trait_name") if n.value else None
            if struct_name and trait_name:
                trait_impls[f"{struct_name}:{trait_name}"] = n
        for c in n.children:
            collect_traits(c)
    
    collect_traits(frozen_root)
    
    if ss.effect_info is not None:
        # Map effects from types dict (return types)
        for fn_ty, effs in ss.effect_info.effects_by_type.items():
            for node_id, ty in types.items():
                if ty is fn_ty:
                    effects[node_id] = effs
        for fn_ty in ss.effect_info.suspends:
            for node_id, ty in types.items():
                if ty is fn_ty:
                    suspends.add(node_id)
        
        # Also map effects from function_types dict (CompactType function types)
        # Use equality check in addition to identity check for propagated effects
        if hasattr(ss, 'function_types') and ss.function_types:
            for fn_ty, effs in ss.effect_info.effects_by_type.items():
                for node_id, ty in ss.function_types.items():
                    if ty is fn_ty or ty == fn_ty:
                        effects[node_id] = effs
            for fn_ty in ss.effect_info.suspends:
                for node_id, ty in ss.function_types.items():
                    if ty is fn_ty or ty == fn_ty:
                        suspends.add(node_id)

    return InferSideTables(
        types=types,
        effects=effects,
        suspends=suspends,
        symbols={},
        constraints=constraints,
        tyenv=_make_tyenv_from_simplesub(),
        traits=traits,
        trait_impls=trait_impls,
        facade=ss,
    )
