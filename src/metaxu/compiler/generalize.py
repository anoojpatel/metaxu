"""Batch-mode let-generalization by constraint replay.

The constraint emitter buffers everything and solves once, so it cannot
generalize the HM way (solve the right-hand side, quantify what does
not escape). Instead a lambda's walk is *recorded*: the constraints it
emits and the type variables allocated meanwhile. A use of the bound
name *instantiates* the recording under a fresh substitution for those
variables and replays it. Variables allocated before the window (a
captured outer binding) are never substituted, which is exactly the
monomorphism a capture needs. See docs/type_inference_plan.md, part 1.

Only ``unify``, ``subtype`` and ``class`` constraints are replayed.
``function``, ``linearity``, ``capture`` and ``effect`` describe the
lambda itself; the instance is registered as an *alias* of the
original so the checker counts calls and propagates effects against
the one lambda that exists at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from metaxu import type_defs
from metaxu.type_defs import CompactType, next_id

REPLAYED_TAGS = ("unify", "subtype", "class")


def _current_id() -> int:
    return type_defs._next_id


def _vars_in(ty: Any, acc: set[int]) -> None:
    """Collect the ids of every type variable reachable in ``ty``
    (structurally: no ``find``, so a pre-solve recording is stable)."""
    if not isinstance(ty, CompactType):
        return
    if ty.kind == "var":
        acc.add(ty.id)
        return
    for child in (ty.param_types or []):
        _vars_in(child, acc)
    if ty.return_type is not None:
        _vars_in(ty.return_type, acc)
    for child in (ty.type_args or []):
        _vars_in(child, acc)


def _operands(constraint: tuple) -> Iterable[Any]:
    tag = constraint[0]
    if tag == "unify":
        return (constraint[1], constraint[2])
    if tag == "subtype":
        return (constraint[1], constraint[2])
    if tag == "class":
        return tuple(constraint[2])
    return ()


@dataclass(frozen=True)
class Scheme:
    """A generalized lambda: its function type, the variable ids that
    are generic in it, and the constraints to replay per instance."""

    fn_type: CompactType
    generic_ids: frozenset[int]
    constraints: tuple[tuple, ...]

    def mentions_generic(self, ty: Any) -> bool:
        acc: set[int] = set()
        _vars_in(ty, acc)
        return bool(acc & self.generic_ids)


class SchemeRecorder:
    """Context manager around a lambda's walk. On exit it knows which
    constraints the walk emitted and which ids it allocated."""

    def __init__(self, facade: Any) -> None:
        self.facade = facade
        self.start_len = 0
        self.end_len = 0
        self.start_id = 0
        self.end_id = 0

    def __enter__(self) -> "SchemeRecorder":
        self.start_len = len(self.facade._constraints)
        self.start_id = _current_id()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.end_len = len(self.facade._constraints)
        self.end_id = _current_id()

    def finish(self, fn_type: CompactType,
               extra_generic: Iterable[int] = ()) -> Scheme:
        """``extra_generic``: variable ids that belong to the lambda even
        though they were allocated before the window. The pipeline
        pre-allocates one variable per frozen node before the walk, so a
        lambda's parameter and body variables are older than the window;
        the emitter passes the ids of every node in the lambda's subtree.
        A captured outer binding is never in that set, so it stays
        shared across instances."""
        recorded = tuple(self.facade._constraints[self.start_len:self.end_len])
        seen: set[int] = set()
        _vars_in(fn_type, seen)
        for c in recorded:
            for op in _operands(c):
                _vars_in(op, seen)
        generic = frozenset(i for i in seen
                            if self.start_id < i <= self.end_id)
        generic = generic | frozenset(extra_generic)
        return Scheme(fn_type=fn_type, generic_ids=generic,
                      constraints=recorded)


def instantiate(scheme: Scheme, facade: Any,
                use_node_id: int | None = None) -> CompactType:
    """A fresh copy of the scheme's function type, with the recorded
    constraints replayed under the same substitution. Registers the
    copy as an alias of the original on the facade."""
    sigma: dict[int, CompactType] = {}

    def sub(ty: Any) -> Any:
        if not isinstance(ty, CompactType):
            return ty
        if ty.kind == "var":
            if ty.id in scheme.generic_ids:
                fresh = sigma.get(ty.id)
                if fresh is None:
                    fresh = CompactType.fresh_var()
                    sigma[ty.id] = fresh
                return fresh
            return ty
        # Structured types are copied once per instance (memoized by id),
        # so the instance's function type and every replayed mention of
        # it are the same object.
        done = sigma.get(ty.id)
        if done is not None:
            return done
        if ty.kind == "function":
            copy = CompactType(
                id=next_id(), kind="function",
                param_types=[sub(p) for p in (ty.param_types or [])],
                return_type=(sub(ty.return_type)
                             if ty.return_type is not None else None),
                linearity=ty.linearity)
            sigma[ty.id] = copy
            return copy
        if ty.kind in ("constructor", "recursive"):
            copy = CompactType(
                id=next_id(), kind=ty.kind,
                constructor=ty.constructor,
                recursive_ref=ty.recursive_ref,
                name=ty.name,
                type_args=[sub(a) for a in (ty.type_args or [])])
            sigma[ty.id] = copy
            return copy
        return ty

    inst = sub(scheme.fn_type)
    for c in scheme.constraints:
        tag = c[0]
        if tag not in REPLAYED_TAGS:
            continue
        if not any(scheme.mentions_generic(op) for op in _operands(c)):
            continue  # already in the graph; nothing generic to rename
        if tag == "unify":
            _, a, b, variance = c
            facade.add_unify(sub(a), sub(b), variance)
        elif tag == "subtype":
            _, a, b = c
            facade.add_subtype(sub(a), sub(b))
        else:
            _, cls, args, node_id = c
            facade.add_class_constraint(cls, [sub(x) for x in args],
                                        node_id)
    aliases = getattr(facade, "aliases", None)
    if aliases is not None:
        aliases[inst.id] = scheme.fn_type
    return inst
