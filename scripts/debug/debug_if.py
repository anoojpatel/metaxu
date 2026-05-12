"""Debug: trace HIR and MIR for if/else programs."""
import logging
logging.disable(logging.CRITICAL)

from metaxu.compiler.pipeline import build_context_from_source
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir import dump_mir
from metaxu.compiler.mir_interp import MirInterpreter

src = "fn sign(x: Int) -> Int { if x > 0 { 1 } else { 0 } }"
ctx = build_context_from_source(src)
hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)


def show_hexpr(h, indent=0):
    if h is None:
        return
    pad = "  " * indent
    extras = []
    if h.literal is not None:
        extras.append(f"lit={h.literal!r}")
    if h.var_name:
        extras.append(f"var={h.var_name!r}")
    if h.binop:
        extras.append(f"binop={h.binop!r}")
    if h.struct_name:
        extras.append(f"struct={h.struct_name!r}")
    if h.field_name:
        extras.append(f"field={h.field_name!r}")
    print(f"{pad}{h.op}  ty={h.ty!r}  {' '.join(extras)}")
    for child in [h.cond, h.left, h.right, h.base, h.lambda_body, h.field_val]:
        if child is not None:
            show_hexpr(child, indent + 1)
    for ops in [h.then_ops, h.else_ops, h.operands]:
        for c in (ops or []):
            show_hexpr(c, indent + 1)
    for name, c in (h.bindings or []):
        print(f"{pad}  bind {name!r}:")
        show_hexpr(c, indent + 2)
    for name, c in (h.fields or []):
        print(f"{pad}  field {name!r}:")
        show_hexpr(c, indent + 2)


print("=== HIR ===")
for f in hir:
    print(f"func {f.sym}")
    show_hexpr(f.body, 1)

print()
print("=== MIR ===")
mir = lower_hir_to_mir(hir)
print(dump_mir(mir))

print()
print("=== INTERP ===")
interp = MirInterpreter()
interp.load(mir)
found = [f for f in mir if "sign" in f.name]
if found:
    for arg in [5, 0, -3]:
        r = interp.call(found[0].name, [arg])
        print(f"  sign({arg}) = {r!r}")
