"""Debug field access lowering."""
import logging
logging.disable(logging.CRITICAL)

from metaxu.compiler.pipeline import build_context_from_source
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir import dump_mir

src = """
struct Point {
    x: Int,
    y: Int
}

fn get_x(p: Point) -> Int {
    p.x
}
"""

ctx = build_context_from_source(src)
hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)

def show(h, indent=0):
    if h is None: return
    pad = "  " * indent
    extras = []
    for attr in ('literal','var_name','binop','struct_name','field_name','callee'):
        v = getattr(h, attr, None)
        if v is not None: extras.append(f"{attr}={v!r}")
    print(f"{pad}{h.op}  {' '.join(extras)}")
    for child in [h.cond, h.left, h.right, h.base, h.lambda_body, h.field_val]:
        if child is not None: show(child, indent+1)
    for ops in [h.then_ops, h.else_ops, h.operands]:
        for c in (ops or []): show(c, indent+1)
    for name, c in (h.bindings or []): show(c, indent+2)

print("=== HIR ===")
for f in hir:
    print(f"func {f.sym}")
    show(f.body, 1)

print("\n=== MIR ===")
mir = lower_hir_to_mir(hir)
print(dump_mir(mir))

# Also check what FieldAccess looks like in id_map
import metaxu.metaxu_ast as fast
print("=== FieldAccess nodes in id_map ===")
for nid, orig in ctx.id_map.items():
    if isinstance(orig, fast.FieldAccess):
        print(f"  id={nid} base={type(getattr(orig,'base',None) or getattr(orig,'expression',None)).__name__} fields={getattr(orig,'fields',None)}")
