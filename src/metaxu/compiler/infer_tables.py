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


def build_from_ast_and_typechecker(id_map: Dict[int, Any], type_checker: Any) -> InferSideTables:
    """Construct side tables from the existing AST and SimpleSub.

    Arguments:
    - id_map: mapping from frozen node_id to original AST node
    - type_checker: instance of src/metaxu/type_checker.TypeChecker after check_program
    """
    types: Dict[int, Any] = {}
    effects: Dict[int, Any] = {}
    suspends: set[int] = set()
    symbols: Dict[int, Any] = {}
    constraints: Dict[int, Sequence[Any]] = {}

    inferencer = getattr(type_checker, 'type_inferencer', None)

    for nid, node in id_map.items():
        ty = getattr(node, 'type_var', None)
        if ty is None:
            ty = getattr(node, 'inferred_type', None)
        if ty is not None:
            types[nid] = ty
        # TODO: integrate real effects/suspends when available

    # We don't have per-node constraints; keep a placeholder under 0
    if inferencer is not None:
        constraints[0] = list(getattr(inferencer, 'constraints', []) or [])
    tyenv = _make_tyenv_from_simplesub()

    return InferSideTables(
        types=types,
        effects=effects,
        suspends=suspends,
        symbols=symbols,
        constraints=constraints,
        tyenv=tyenv,
    )
