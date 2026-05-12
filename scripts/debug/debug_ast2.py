"""Inspect raw parsed AST for if expression via deep traversal."""
import logging
logging.disable(logging.CRITICAL)

from metaxu.parser import Parser
import metaxu.metaxu_ast as ast

src = "fn sign(x: Int) -> Int { if x > 0 { 1 } else { 0 } }"
parser = Parser()
program = parser.parse(src)

def show(node, indent=0, visited=None):
    if node is None:
        return
    if visited is None:
        visited = set()
    obj_id = id(node)
    if obj_id in visited:
        return
    visited.add(obj_id)
    pad = "  " * indent
    name = type(node).__name__
    # Show key attrs (skip large/recursive ones)
    key_attrs = {}
    for k, v in vars(node).items():
        if k.startswith('_') or k in ('node_id', 'scope', 'parent', 'type_info'):
            continue
        if isinstance(v, (str, int, float, bool, type(None))):
            key_attrs[k] = v
        elif isinstance(v, list) and len(v) <= 3:
            key_attrs[k] = f"[{len(v)} items]"
    print(f"{pad}{name}  {key_attrs}")

    # Recurse into children, body, statements, params, condition, branches
    for attr in ('children', 'body', 'statements', 'params', 'bindings',
                 'condition', 'then_branch', 'else_branch', 'then_body', 'else_body',
                 'expression', 'arguments', 'left', 'right', 'initializer'):
        val = getattr(node, attr, None)
        if val is None:
            continue
        if isinstance(val, list):
            for item in val:
                if hasattr(item, '__dict__'):
                    show(item, indent + 1, visited)
        elif hasattr(val, '__dict__'):
            show(val, indent + 1, visited)

show(program)
