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
        return {"name": getattr(node, "identifier", None)}
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
        }
    if isinstance(node, fast.FunctionCall):
        return {"name": getattr(node, "name", None)}
    if isinstance(node, fast.Assignment):
        return {"name": getattr(node, "name", None)}
    if isinstance(node, fast.LambdaExpression):
        return {
            "params": [getattr(p, "name", None) for p in getattr(node, "params", [])],
            "captures": dict(getattr(node, "capture_modes", {}) or {}),
            "linearity": _mode_value(getattr(node, "linearity", None)),
        }
    if isinstance(node, fast.StructInstantiation):
        struct_name = getattr(node, "struct_name", None)
        return {"name": str(struct_name) if struct_name is not None else None}
    if isinstance(node, fast.StructField):
        return {"name": getattr(node, "name", None)}
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


def _mode_value(mode: Any) -> Any | None:
    if mode is None:
        return None
    return getattr(mode, "mode", mode if isinstance(mode, str) else None)


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
