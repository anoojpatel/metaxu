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


def build_frozen_ast_with_map(parsed_root: Any) -> Tuple[AstNode, Dict[int, Any]]:
    """Convert the existing AST into a frozen representation and return a mapping
    from frozen node_id to original node object.

    We rely on metaxu.metaxu_ast.Node.children for traversal.
    """
    next_id = 1
    id_map: Dict[int, Any] = {}

    def go(n: Any) -> AstNode:
        nonlocal next_id
        nid = next_id
        next_id += 1
        id_map[nid] = n
        kind = n.__class__.__name__
        # Children: prefer explicit children list if present
        kids = []
        if hasattr(n, 'children') and isinstance(n.children, list):
            for c in n.children:
                if c is not None:
                    kids.append(go(c))
        span = _span_of(n)
        return AstNode(node_id=nid, kind=kind, children=tuple(kids), span=span, value=None)

    root = go(parsed_root)
    return root, id_map


def build_frozen_ast(parsed_root: Any) -> AstNode:
    root, _ = build_frozen_ast_with_map(parsed_root)
    return root
