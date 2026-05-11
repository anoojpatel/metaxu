from __future__ import annotations

from typing import Any, Sequence
from dataclasses import dataclass, field


_ALLOWED_LINEARITIES = {"once", "separate", "many"}
_ALLOWED_CAPTURE_MODES = {"borrow", "borrow_mut", "move"}


@dataclass
class EffectInfo:
    effects_by_type: dict[Any, set[str]] = field(default_factory=dict)
    suspends: set[Any] = field(default_factory=set)


def check_constraints(constraints: Sequence[tuple], function_types: dict = None) -> tuple[list[str], EffectInfo]:
    errors: list[str] = []
    linearity_by_type: dict[Any, str] = {}
    call_counts: dict[Any, int] = {}
    effect_info = EffectInfo()
    fn_return_ty: dict[Any, Any] = {}
    subtype_edges: list[tuple[Any, Any]] = []
    if function_types is None:
        function_types = {}

    # Process all constraints
    for constraint in constraints:
        tag = constraint[0] if constraint else None
        if tag == "unresolved":
            _, kind, name, node_id = constraint
            errors.append(f"Unresolved {kind} '{name}' at node {node_id}")
        elif tag == "function":
            _, fn_ty, _param_tys, ret_ty, linearity, node_id = constraint
            if linearity not in _ALLOWED_LINEARITIES:
                errors.append(f"Invalid function linearity '{linearity}' at node {node_id}")
            linearity_by_type[fn_ty] = linearity
            fn_return_ty[fn_ty] = ret_ty
        elif tag == "linearity":
            _, fn_ty, linearity, node_id = constraint
            if linearity not in _ALLOWED_LINEARITIES:
                errors.append(f"Invalid linearity '{linearity}' at node {node_id}")
            linearity_by_type[fn_ty] = linearity
        elif tag == "capture":
            _, fn_ty, captured_name, _captured_ty, mode, node_id = constraint
            if mode not in _ALLOWED_CAPTURE_MODES:
                errors.append(f"Invalid capture mode '{mode}' for '{captured_name}' at node {node_id}")
            if mode == "borrow_mut" and linearity_by_type.get(fn_ty) == "many":
                errors.append(f"Mutable capture '{captured_name}' requires separate or once linearity at node {node_id}")
        elif tag == "effect":
            _, fn_ty, effect_name, _node_id = constraint
            effect_info.effects_by_type.setdefault(fn_ty, set()).add(effect_name)
        elif tag == "call":
            _, callee_ty, _arg_tys, result_ty, node_id = constraint
            call_counts[callee_ty] = call_counts.get(callee_ty, 0) + 1
            if linearity_by_type.get(callee_ty) == "once" and call_counts[callee_ty] > 1:
                errors.append(f"Once callable invoked more than once at node {node_id}")
            if callee_ty in effect_info.effects_by_type:
                for effect in effect_info.effects_by_type[callee_ty]:
                    effect_info.effects_by_type.setdefault(result_ty, set()).add(effect)
        elif tag == "subtype":
            _, subtype, supertype = constraint
            subtype_edges.append((subtype, supertype))

    # Propagate effects along subtype edges
    changed = True
    while changed:
        changed = False
        for sub, sup in subtype_edges:
            if sub in effect_info.effects_by_type:
                for effect in effect_info.effects_by_type[sub]:
                    if effect not in effect_info.effects_by_type.get(sup, set()):
                        effect_info.effects_by_type.setdefault(sup, set()).add(effect)
                        changed = True

    # NOW propagate effects from return types to function types (after callee-to-result propagation)
    for fn_ty, ret_ty in fn_return_ty.items():
        if ret_ty in effect_info.effects_by_type:
            for effect in effect_info.effects_by_type[ret_ty]:
                effect_info.effects_by_type.setdefault(fn_ty, set()).add(effect)

    # Also propagate from function types to their return types (inverse direction)
    for fn_ty, ret_ty in fn_return_ty.items():
        if fn_ty in effect_info.effects_by_type:
            for effect in effect_info.effects_by_type[fn_ty]:
                if effect not in effect_info.effects_by_type.get(ret_ty, set()):
                    effect_info.effects_by_type.setdefault(ret_ty, set()).add(effect)

    for fn_ty, effects in effect_info.effects_by_type.items():
        if effects:
            effect_info.suspends.add(fn_ty)

    return errors, effect_info
