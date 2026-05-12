"""Inspect the raw parsed AST for an if expression."""
import logging
logging.disable(logging.CRITICAL)

from metaxu.parser import Parser
import metaxu.metaxu_ast as ast

src = "fn sign(x: Int) -> Int { if x > 0 { 1 } else { 0 } }"
parser = Parser()
program = parser.parse(src)

def show(node, indent=0):
    if node is None:
        return
    pad = "  " * indent
    name = type(node).__name__
    attrs = {k: v for k, v in vars(node).items()
             if not k.startswith('_') and k not in ('children', 'node_id', 'scope', 'parent')}
    print(f"{pad}{name}  {attrs}")
    for child in getattr(node, 'children', []):
        if child is not None and hasattr(child, '__class__'):
            show(child, indent + 1)

show(program)
