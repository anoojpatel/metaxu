"""Deeper inspection: what does _from_orig_expr see for the IfExpression?"""
import logging
logging.disable(logging.CRITICAL)

from metaxu.compiler.pipeline import build_context_from_source
from metaxu.compiler.hir import HIRBuilder
import metaxu.metaxu_ast as fast

src = "fn sign(x: Int) -> Int { if x > 0 { 1 } else { 0 } }"
ctx = build_context_from_source(src)

# Show all id_map entries
print("=== id_map ===")
for nid, orig in sorted(ctx.id_map.items()):
    print(f"  {nid}: {type(orig).__name__}  attrs={[k for k in vars(orig) if not k.startswith('_')]}")

# Check what the FunctionDeclaration body is
fn = next((v for v in ctx.id_map.values() if isinstance(v, fast.FunctionDeclaration)), None)
print(f"\n=== FunctionDeclaration body ===")
print(f"  type: {type(fn.body).__name__}")
print(f"  children: {[type(c).__name__ for c in fn.body.children if c is not None]}")

ifexpr = next((v for v in ctx.id_map.values() if isinstance(v, fast.IfExpression)), None)
print(f"\n=== IfExpression ===")
print(f"  condition type: {type(ifexpr.condition).__name__}")
print(f"  then_branch type: {type(ifexpr.then_branch).__name__}, value={getattr(ifexpr.then_branch, 'value', '?')}")
print(f"  else_branch type: {type(ifexpr.else_branch).__name__}, value={getattr(ifexpr.else_branch, 'value', '?')}")
print(f"  children: {[(type(c).__name__, getattr(c,'value','?')) for c in ifexpr.children]}")

# Check if ComparisonExpression is in id_map
comp = next((v for v in ctx.id_map.values() if type(v).__name__ == 'ComparisonExpression'), None)
print(f"\n=== ComparisonExpression ===")
if comp:
    print(f"  type: {type(comp).__name__}")
    print(f"  attrs: {vars(comp)}")
    print(f"  children: {[type(c).__name__ for c in comp.children]}")
else:
    print("  NOT FOUND in id_map")

# Also show the FunctionDeclaration body type more carefully
print(f"\n=== fn.body detailed ===")
body = fn.body
print(f"  body type: {type(body).__name__}")
if hasattr(body, 'statements'):
    print(f"  statements: {[type(s).__name__ for s in body.statements]}")
if hasattr(body, 'children'):
    print(f"  children: {[type(c).__name__ for c in body.children]}")
