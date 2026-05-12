"""Check what the parser produces for perform/handle."""
import logging
logging.disable(logging.CRITICAL)

from metaxu.parser import Parser
import metaxu.metaxu_ast as fast

def walk(node, indent=0, visited=None):
    if visited is None: visited = set()
    if node is None or isinstance(node, str) or id(node) in visited:
        return
    visited.add(id(node))
    pad = "  " * indent
    name = type(node).__name__
    key = {k: getattr(node, k) for k in ('name','op_name','param_name','effect_name',
                                           'arguments','parts','value','operator')
           if hasattr(node, k) and not callable(getattr(node, k))}
    print(f"{pad}{name}  {key}")
    children = list(getattr(node, 'children', []))
    for attr in ('body','statements','expression','base','continuation',
                 'handler','arguments','left','right','condition',
                 'then_branch','else_branch'):
        v = getattr(node, attr, None)
        if v is None or isinstance(v, str): continue
        if isinstance(v, list):
            for item in v:
                if item not in children and item is not None and not isinstance(item, str):
                    children.append(item)
        elif hasattr(v, '__dict__') and v not in children:
            children.append(v)
    for c in children:
        if c is not None and not isinstance(c, str):
            walk(c, indent+1, visited)

parser = Parser()

print("=== perform emit(x) ===")
prog = parser.parse("effect Log = {\n    fn emit(v: Int) -> Int\n}\nfn f(x: Int) -> Int {\n    perform emit(x)\n}")
walk(prog)

print("\n=== perform Log.emit(x) ===")
prog2 = parser.parse("effect Log = {\n    fn emit(v: Int) -> Int\n}\nfn f(x: Int) -> Int {\n    perform Log.emit(x)\n}")
walk(prog2)

print("\n=== handle Log with { emit(v) -> v } in g(x) ===")
prog3 = parser.parse("""effect Log = {
    fn emit(v: Int) -> Int
}
fn g(x: Int) -> Int { perform emit(x) }
fn f(x: Int) -> Int {
    handle Log with {
        emit(v) -> v
    } in g(x)
}""")
walk(prog3)
