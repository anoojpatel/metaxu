from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .types import Ty, TyEnv, unify


@dataclass(frozen=True, slots=True)
class ClassConstraint:
    name: str
    params: tuple[Ty, ...]

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        ps = ", ".join(map(repr, self.params))
        return f"{self.name}({ps})"


@dataclass(frozen=True, slots=True)
class InstanceHead:
    name: str
    pats: tuple[Any, ...]  # TyPattern like Ty with vars
    context: tuple[ClassConstraint, ...]


@dataclass(slots=True)
class FDMap:
    """Functional dependency map: name -> (det_indices, detd_indices)."""

    fds: Mapping[str, tuple[tuple[int, ...], tuple[int, ...]]]

    def get(self, name: str) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        return self.fds.get(name)


def improve_fds(constraints: list[ClassConstraint], fd_map: FDMap, tyenv: TyEnv) -> bool:
    """CHR-style improvement: for any pair C s … t1 and C s … t2 with same det positions unify t1 ≡ t2.

    Returns True if tyenv changed (approx), signaling another round may help.
    """
    changed = False
    by_name: dict[str, list[ClassConstraint]] = {}
    for c in constraints:
        by_name.setdefault(c.name, []).append(c)
    for name, group in by_name.items():
        fd = fd_map.get(name)
        if not fd:
            continue
        det, detd = fd
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                ci, cj = group[i], group[j]
                if all(ci.params[k] == cj.params[k] for k in det):
                    for kk in detd:
                        before = dict(tyenv.subst)
                        tyenv = unify(ci.params[kk], cj.params[kk], tyenv)
                        if tyenv.subst != before:
                            changed = True
    return changed


def _match_instance(constraint: ClassConstraint, head: InstanceHead) -> tuple[dict[Any, Any]] | None:
    """Attempt to match constraint against instance head patterns; return MGU mapping if success.

    Here we treat Ty as structural and assume simple equality or var capture; a
    real implementation should share the SimpleSub matcher/unifier.
    """
    if constraint.name != head.name:
        return None
    if len(constraint.params) != len(head.pats):
        return None
    subst: dict[Any, Any] = {}
    for arg, pat in zip(constraint.params, head.pats):
        if isinstance(pat, str) and pat.startswith("?"):
            # treat pattern like a variable name: ?X
            subst.setdefault(pat, arg)
        elif pat != arg:
            return None
    return subst


def resolve_instances(constraints: list[ClassConstraint], instances: Sequence[InstanceHead], tyenv: TyEnv) -> bool:
    """Resolve constraints to instance heads; replace with context under MGU.

    Mutates constraints list in-place. Returns True if constraints changed.
    """
    changed = False
    i = 0
    while i < len(constraints):
        c = constraints[i]
        for inst in instances:
            m = _match_instance(c, inst)
            if m is None:
                continue
            # Apply instance context
            ctx = list(inst.context)
            # Simple capture: replace variables in context using m
            def subst_cc(cc: ClassConstraint) -> ClassConstraint:
                new_params: list[Ty] = []
                for p in cc.params:
                    new_params.append(m.get(p, p))
                return ClassConstraint(cc.name, tuple(new_params))

            new_ctx = [subst_cc(cc) for cc in ctx]
            # Replace c by new_ctx
            constraints.pop(i)
            for cc in reversed(new_ctx):
                constraints.insert(i, cc)
            changed = True
            break
        else:
            i += 1
    return changed


def solve_until_fixpoint(constraints: list[ClassConstraint], fd_map: FDMap, instances: Sequence[InstanceHead], tyenv: TyEnv) -> TyEnv:
    """Run improvement + instance resolution to a fixpoint."""
    changed = True
    while changed:
        changed = False
        if improve_fds(constraints, fd_map, tyenv):
            changed = True
        if resolve_instances(constraints, instances, tyenv):
            changed = True
    return tyenv
