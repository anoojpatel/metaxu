from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

from .hir import HFun, HExpr


@dataclass(slots=True)
class DropPlan:
    """A simple plan describing which locals to drop at function end per HIR func.

    This is a placeholder for a richer borrow checker that would:
    - Track borrows (&, &mut) across uses
    - Enforce non-aliasing rules for &mut
    - Determine exact last-use points for each local
    For now we approximate: locals bound in Let inside the top-level body Block
    are dropped before function return if their type appears to need dropping.
    """

    drop_at_end: List[str]


def _type_needs_drop(ty: object) -> bool:
    """Heuristic: consult a needs_drop() method or property when available."""
    try:
        if hasattr(ty, "needs_drop"):
            nd = ty.needs_drop  # type: ignore[attr-defined]
            if callable(nd):
                return bool(nd())
            return bool(nd)
    except Exception:
        pass
    return False


def _collect_let_bindings(e: HExpr, out: List[tuple[str, object]]) -> None:
    if e.op == "Let" and e.bindings:
        for (name, sub) in e.bindings:
            out.append((name, sub.ty))
    # Recurse into children
    if e.operands:
        for ch in e.operands:
            _collect_let_bindings(ch, out)


def plan_drops(funcs: List[HFun]) -> Dict[str, DropPlan]:
    plans: Dict[str, DropPlan] = {}
    for f in funcs:
        lets: List[tuple[str, object]] = []
        _collect_let_bindings(f.body, lets)
        drop_candidates: List[str] = []
        for (name, ty) in lets:
            if _type_needs_drop(ty):
                drop_candidates.append(str(name))
        plans[str(f.sym)] = DropPlan(drop_at_end=drop_candidates)
    return plans
