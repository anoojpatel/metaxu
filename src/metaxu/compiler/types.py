from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

try:
    # Integrate with existing SimpleSub CompactType/unify when available
    from metaxu.type_defs import CompactType as _CompactType, unify as _ss_unify
except Exception:  # pragma: no cover - optional integration at import time
    _CompactType = None  # type: ignore[assignment]
    def _ss_unify(a: Any, b: Any, variance: str = 'invariant') -> bool:  # type: ignore[no-redef]
        return a == b


class Ty(Protocol):
    """Protocol for types from the existing inferencer.

    We purposefully keep this abstract and rely on TyEnv.apply to rewrite
    types after solving constraints.
    """

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        ...


@dataclass(slots=True)
class EffectSet:
    effects: frozenset[str]

    def __or__(self, other: "EffectSet") -> "EffectSet":
        return EffectSet(self.effects | other.effects)

    def __contains__(self, name: str) -> bool:
        return name in self.effects

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"EffectSet({sorted(self.effects)})"


@dataclass(slots=True)
class TyEnv:
    """Type substitution environment produced by SimpleSub.

    This holds the final substitution mapping. apply() should map a type to its
    representative using the substitution; default is identity.
    """

    subst: dict[Any, Any]

    def apply(self, ty: Ty) -> Ty:
        # If this looks like a SimpleSub CompactType, use its representative
        try:
            if _CompactType is not None and isinstance(ty, _CompactType):
                return ty.find()  # type: ignore[attr-defined]
        except Exception:
            pass
        return self.subst.get(ty, ty)


def unify(a: Ty, b: Ty, env: TyEnv) -> TyEnv:
    """Unify two types, returning an updated TyEnv.

    This is a thin interface that should call into the existing unifier.
    For now it's a stub so that the constraints solver can be wired.
    """
    # Delegate to SimpleSub's unify for CompactTypes if available.
    try:
        if _CompactType is not None and isinstance(a, _CompactType) and isinstance(b, _CompactType):
            _ss_unify(a, b, 'invariant')
            return env
    except Exception:
        pass
    if a == b:
        return env
    env.subst.setdefault(a, b)
    env.subst.setdefault(b, a)
    return env
