"""Debug: what does p.x parse to and where does it go in FunctionDeclaration?"""
import logging
logging.disable(logging.CRITICAL)

from metaxu.parser import Parser
import metaxu.metaxu_ast as fast

src = "fn get_x(p: Point) -> Int { p.x }"
parser = Parser()
program = parser.parse(src)

# Find the FunctionDeclaration
def find_all(node, cls, visited=None):
    if visited is None: visited = set()
    if id(node) in visited: return []
    visited.add(id(node))
    results = []
    if isinstance(node, cls):
        results.append(node)
    for c in getattr(node, 'children', []):
        if c is not None and not isinstance(c, str):
            results.extend(find_all(c, cls, visited))
    for attr in ('body', 'statements', 'expression', 'base', 'left', 'right',
                 'condition', 'then_branch', 'else_branch'):
        v = getattr(node, attr, None)
        if v is not None and not isinstance(v, str):
            if isinstance(v, list):
                for item in v:
                    if item is not None and not isinstance(item, str):
                        results.extend(find_all(item, cls, visited))
            elif hasattr(v, '__dict__') and v not in getattr(node,'children',[]):
                results.extend(find_all(v, cls, visited))
    return results

fns = find_all(program, fast.FunctionDeclaration)
for fn in fns:
    print(f"FunctionDeclaration: {fn.name}")
    print(f"  body type: {type(fn.body).__name__}")
    if isinstance(fn.body, list):
        for i, stmt in enumerate(fn.body):
            print(f"  body[{i}]: {type(stmt).__name__}  attrs={list(vars(stmt).keys())}")
            for attr in vars(stmt):
                if not attr.startswith('_') and attr not in ('parent','scope','location','children','node_id'):
                    v = getattr(stmt, attr)
                    vr = repr(v) if not hasattr(v,'__dict__') else type(v).__name__
                    print(f"    .{attr} = {type(v).__name__}({vr})")
