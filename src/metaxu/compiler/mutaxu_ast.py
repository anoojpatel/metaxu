from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Dict
import json
import metaxu.metaxu_ast as fast


@dataclass(frozen=True, slots=True)
class Span:
    file: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class AstNode:
    """Frozen, minimal AST mirror for the HIR builder stage.

    This mirrors the existing parsed AST (already available in src/metaxu),
    but provides an immutable view with the essential fields needed for
    subsequent passes. The actual construction will wrap/copy nodes from the
    existing parser output.
    """

    node_id: int
    kind: str
    children: tuple[AstNode, ...]
    span: Span
    # Optional data for leaf nodes or annotations
    value: Any | None = None

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "span": {
                "file": self.span.file,
                "start": self.span.start,
                "end": self.span.end,
            },
            "value": self.value,
            "children": [c.to_json_obj() for c in self.children],
        }


def dump_ast_json(root: AstNode) -> str:
    """Pretty JSON for golden tests."""
    return json.dumps(root.to_json_obj(), indent=2, sort_keys=True)


def _span_of(node: Any) -> Span:
    loc = getattr(node, 'location', None)
    if loc is None:
        # Fallbacks from parser where file and positions may be missing
        return Span(file=getattr(node, 'source_file', '<unknown>'), start=0, end=0)
    # Location may be a simple object with file/line/column; normalize to a range
    file = getattr(loc, 'file', getattr(node, 'source_file', '<unknown>'))
    start = getattr(loc, 'column', 0) or 0
    end = start
    return Span(file=file, start=start, end=end)


def _value_of(node: Any) -> Any | None:
    if isinstance(node, fast.Literal):
        return getattr(node, "value", None)
    if isinstance(node, fast.Variable):
        return {"name": getattr(node, "name", None)}
    if isinstance(node, fast.BinaryOperation):
        return {"operator": getattr(node, "operator", None)}
    if isinstance(node, fast.ComparisonExpression):
        return {"operator": getattr(node, "operator", None)}
    if isinstance(node, fast.LetBinding):
        return {
            "name": getattr(node, "identifier", None),
            "mode": _mode_value(getattr(node, "mode", None)),
        }
    if isinstance(node, fast.Parameter):
        return {
            "name": getattr(node, "name", None),
            "mode": _mode_value(getattr(node, "mode", None)),
        }
    if isinstance(node, fast.FunctionDeclaration):
        return {
            "name": getattr(node, "name", None),
            "params": [getattr(p, "name", None) for p in getattr(node, "params", [])],
            "performs": [_effect_name(e) for e in getattr(node, "performs", []) or []],
            # Generic signature info (for parametric instantiation checking):
            # declared type parameters, per-parameter declared type displays
            # (None for unannotated params), the declared return type display,
            # and where-clause / inline-bound constraints [{param, trait, kind}].
            "type_params": [
                _type_param_name(tp) for tp in getattr(node, "type_params", None) or []
            ],
            "param_types": [
                _type_display(getattr(p, "type_annotation", None))
                for p in getattr(node, "params", [])
            ],
            "return_type": _safe_type_display(getattr(node, "return_type", None)),
            "where": _where_constraints(node),
        }
    if isinstance(node, fast.EffectDeclaration):
        return {
            "name": getattr(node, "name", None),
            "effect_class": getattr(node, "effect_class", None),
        }
    if isinstance(node, fast.HandleEffect):
        return {
            "effect_name": getattr(node, "effect_name", None),
        }
    if isinstance(node, fast.PerformEffect):
        name = getattr(node, "effect_name", None)
        return {"effect_name": name if isinstance(name, str) or name is None else str(name)}
    if isinstance(node, fast.Resume):
        return {
            "value": getattr(node, "value", None),
        }
    if isinstance(node, fast.FunctionCall):
        name = getattr(node, "name", None)
        payload: dict[str, Any] = {
            "name": name if isinstance(name, str) or name is None else str(name)
        }
        # Explicit instantiation type args (`identity<Int>(x)`, `Full<Int>(3)`).
        type_args = getattr(node, "type_args", None) or []
        if type_args:
            payload["type_args"] = [_type_display(a) for a in type_args]
        return payload
    if isinstance(node, fast.Assignment):
        # The assignment target may be a complex expression (field access,
        # indexing); stringify it so the frozen AST stays JSON serializable.
        name = getattr(node, "name", None)
        return {"name": name if isinstance(name, str) or name is None else str(name)}
    if isinstance(node, fast.LambdaExpression):
        return {
            "params": [getattr(p, "name", None) for p in getattr(node, "params", [])],
            "captures": dict(getattr(node, "capture_modes", {}) or {}),
            "linearity": _mode_value(getattr(node, "linearity", None)),
        }
    if isinstance(node, fast.StructDefinition):
        return {
            "name": getattr(node, "name", None),
            "type_params": [
                _type_display(tp) for tp in getattr(node, "type_params", None) or []
            ],
            "fields": [
                {
                    "name": getattr(f, "name", None),
                    "type": _type_display(getattr(f, "type_info", None)),
                    # Per-field mode annotations (@mut/@const/@local ...) are
                    # attached by the parser as `field.modes`; carry them so
                    # deep ownership validation can see declared field modes.
                    "mode": _mode_value(getattr(f, "modes", None)),
                }
                for f in getattr(node, "fields", None) or []
            ],
        }
    if isinstance(node, fast.EnumDefinition):
        return {
            "name": getattr(node, "name", None),
            "type_params": [
                _type_param_name(tp) for tp in getattr(node, "type_params", None) or []
            ],
            "variants": [
                {
                    "name": getattr(v, "name", None),
                    "fields": [
                        {"name": fname, "type": _type_display(ftype)}
                        for (fname, ftype) in (getattr(v, "fields", None) or [])
                    ],
                }
                for v in getattr(node, "variants", None) or []
            ],
        }
    if isinstance(node, fast.StructInstantiation):
        struct_name = getattr(node, "struct_name", None)
        return {
            "name": str(struct_name) if struct_name is not None else None,
            "type_args": [
                _type_display(a) for a in getattr(node, "type_args", None) or []
            ],
        }
    if isinstance(node, fast.StructField):
        return {"name": getattr(node, "name", None)}
    if isinstance(node, fast.Implementation):
        # Pre-desugar impl registry info: `implement Trait for Type`.
        # (Post-desugar the same info lives in the mangled __impl$ names.)
        return {
            "trait": _safe_type_display(getattr(node, "interface_name", None)),
            "type": _safe_type_display(getattr(node, "type_name", None)),
        }
    if isinstance(node, fast.FieldAccess):
        return {"fields": tuple(getattr(node, "fields", ()) or ())}
    if isinstance(node, fast.QualifiedFunctionCall):
        return {"name": ".".join(getattr(node, "parts", ()) or ())}
    if isinstance(node, fast.BorrowShared):
        return {"variable": getattr(node, "variable", None)}
    if isinstance(node, fast.BorrowUnique):
        return {"variable": getattr(node, "variable", None)}
    if isinstance(node, fast.Move):
        return {"variable": getattr(node, "variable", None)}
    if isinstance(node, fast.ExclaveExpression):
        return {"expression": getattr(node, "expression", None)}
    return None


def _type_display(t: Any) -> str | None:
    """Best-effort display name for a surface type expression.

    Handles TypeReference/TypeParameter/Variable (all carry .name) and
    TypeApplication (constructor + args). Used so the frozen AST keeps
    enough declared-type information for field-level type checking.
    """
    if t is None:
        return None
    if isinstance(t, str):
        return t
    name = getattr(t, "name", None)
    if isinstance(name, str):
        return name
    ctor = getattr(t, "type_constructor", None)
    if ctor is not None:
        base = ctor if isinstance(ctor, str) else _type_display(ctor)
        args = [_type_display(a) or "?" for a in getattr(t, "type_args", None) or []]
        return f"{base}[{', '.join(args)}]" if args else base
    return str(t)


def _safe_type_display(t: Any) -> str | None:
    """_type_display that also tolerates bare Python classes (e.g. the parser
    stores the NoneType *class* as the default return type)."""
    if t is None:
        return None
    if isinstance(t, type):
        return getattr(t, "__name__", None)
    return _type_display(t)


def _type_param_name(tp: Any) -> str | None:
    """Name of a declared type parameter (TypeParameter or bare string)."""
    if isinstance(tp, str):
        return tp
    name = getattr(tp, "name", None)
    return name if isinstance(name, str) else None


def _flatten_bounds(bound: Any) -> list[str]:
    """Flatten a type bound expression into trait-name displays.

    Handles a single bound (`T: Display`), compound bounds
    (`T: Display + Ord`), and legacy list shapes.
    """
    if bound is None:
        return []
    if isinstance(bound, (list, tuple)):
        out: list[str] = []
        for b in bound:
            out.extend(_flatten_bounds(b))
        return out
    left = getattr(bound, "left", None)
    right = getattr(bound, "right", None)
    if bound.__class__.__name__ == "CompoundTypeBound" and (left is not None or right is not None):
        return _flatten_bounds(left) + _flatten_bounds(right)
    disp = _safe_type_display(bound)
    return [disp] if disp else []


def _where_constraints(node: Any) -> list[dict[str, str]]:
    """Collect trait-bound constraints for a generic declaration.

    Merges inline bounds on type parameters (`fn f<T: Trait>`) with the
    where clause (`fn f<T>(..) -> R where T: Trait`). Each entry is
    {"param": <type param name>, "trait": <trait name>, "kind": <kind>}.
    """
    out: list[dict[str, str]] = []
    for tp in getattr(node, "type_params", None) or []:
        pname = _type_param_name(tp)
        if pname is None:
            continue
        for trait in _flatten_bounds(getattr(tp, "bounds", None)):
            out.append({"param": pname, "trait": trait, "kind": "bound"})
    where = getattr(node, "where_clause", None)
    for c in getattr(where, "constraints", None) or []:
        pname = _safe_type_display(getattr(c, "type_param", None))
        for trait in _flatten_bounds(getattr(c, "bound_type", None)):
            if pname:
                out.append({
                    "param": pname,
                    "trait": trait,
                    "kind": str(getattr(c, "kind", "subtype") or "subtype"),
                })
    return out


def _mode_value(mode: Any) -> Any | None:
    """Normalize surface mode annotations to plain strings for the payload.

    The parser produces ModeAnnotation nodes (wrapping a Uniqueness/Locality/
    LinearityMode carrying .mode) and sometimes lists of them; downstream
    (_split_mode in the constraint emitter) understands strings and lists of
    strings.
    """
    if mode is None:
        return None
    if isinstance(mode, str):
        return mode
    if isinstance(mode, (list, tuple)):
        flat: list[str] = []
        for m in mode:
            v = _mode_value(m)
            if isinstance(v, list):
                flat.extend(v)
            elif v is not None:
                flat.append(v)
        return flat or None
    inner = getattr(mode, "mode_type", None)
    if inner is not None:
        return _mode_value(inner)
    value = getattr(mode, "mode", None)
    return value if isinstance(value, str) else None


def _effect_name(effect: Any) -> str:
    name = getattr(effect, "effect_name", None)
    if name is None:
        name = getattr(effect, "name", None)
    if name is None:
        name = effect
    return str(name)


def build_frozen_ast_with_map(parsed_root: Any) -> Tuple[AstNode, Dict[int, Any]]:
    """Convert the existing AST into a frozen representation and return a mapping
    from frozen node_id to original node object.

    We rely on metaxu.metaxu_ast.Node.children for traversal.
    """
    next_id = 1
    id_map: Dict[int, Any] = {}

    def is_ast_node(value: Any) -> bool:
        return hasattr(value, "children") or isinstance(value, fast.Node)

    def go(n: Any) -> AstNode:
        nonlocal next_id
        nid = next_id
        next_id += 1
        id_map[nid] = n
        kind = n.__class__.__name__
        # Children: prefer explicit children list if present
        kids = []
        for attr in ("params",):
            for c in getattr(n, attr, []) or []:
                if c is not None:
                    kids.append(go(c))
        if hasattr(n, 'children') and isinstance(n.children, list):
            for c in n.children:
                if c is not None:
                    kids.append(go(c))
        for attr in ("expression", "body", "value", "base"):
            c = getattr(n, attr, None)
            if c is not None and c not in getattr(n, 'children', []):
                if isinstance(c, list):
                    for item in c:
                        if item is not None and item not in getattr(n, 'children', []) and is_ast_node(item):
                            kids.append(go(item))
                else:
                    if is_ast_node(c):
                        kids.append(go(c))
        span = _span_of(n)
        return AstNode(node_id=nid, kind=kind, children=tuple(kids), span=span, value=_value_of(n))

    root = go(parsed_root)
    return root, id_map


def build_frozen_ast(parsed_root: Any) -> AstNode:
    root, _ = build_frozen_ast_with_map(parsed_root)
    return root
