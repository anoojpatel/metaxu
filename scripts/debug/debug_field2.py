"""Debug: how does the parser represent p.x and does it end up in id_map?"""
import logging
logging.disable(logging.CRITICAL)

from metaxu.parser import Parser
import metaxu.metaxu_ast as fast
from metaxu.compiler.mutaxu_ast import build_frozen_ast_with_map

src = """
struct Point {
    x: Int,
    y: Int
}

fn get_x(p: Point) -> Int {
    p.x
}
"""

parser = Parser()
program = parser.parse(src)

# Deep walk raw AST
def walk(node, indent=0):
    if node is None or isinstance(node, str):
        return
    pad = "  " * indent
    name = type(node).__name__
    key = {}
    for k in ('name','value','operator','fields','field_name'):
        v = getattr(node, k, None)
        if v is not None:
            key[k] = v
    print(f"{pad}{name} {key}")
    for c in getattr(node, 'children', []):
        if c is not None and not isinstance(c, str):
            walk(c, indent+1)
    for attr in ('body', 'statements', 'expression', 'base'):
        v = getattr(node, attr, None)
        if v is not None and v not in getattr(node, 'children', []):
            if isinstance(v, list):
                for item in v:
                    if item is not None and not isinstance(item, str):
                        walk(item, indent+1)
            elif hasattr(v, '__dict__'):
                walk(v, indent+1)

print("=== Raw AST ===")
walk(program)

print("\n=== id_map types ===")
frozen, id_map = build_frozen_ast_with_map(program)
for nid, orig in sorted(id_map.items()):
    print(f"  {nid}: {type(orig).__name__}  children_count={len(getattr(orig,'children',[]))}")
