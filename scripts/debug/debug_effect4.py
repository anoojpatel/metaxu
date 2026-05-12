"""Trace the full effect pipeline for perform+handle."""
import logging
logging.disable(logging.CRITICAL)

from metaxu.compiler.pipeline import build_context_from_source
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir import dump_mir
from metaxu.compiler.mir_interp import MirInterpreter
import metaxu.metaxu_ast as fast

src = """
effect Log = {
    fn emit(v: Int) -> Int
}

fn logged_add(a: Int, b: Int) -> Int {
    let result = a + b
    perform emit(result)
}

fn main(a: Int, b: Int) -> Int {
    handle Log with {
        emit(v) -> v
    } in logged_add(a, b)
}
"""

ctx = build_context_from_source(src)

print("=== id_map types ===")
type_counts = {}
for nid, orig in ctx.id_map.items():
    t = type(orig).__name__
    type_counts[t] = type_counts.get(t, 0) + 1
for t, c in sorted(type_counts.items()):
    print(f"  {t}: {c}")

print("\n=== PerformEffect / HandleEffect nodes ===")
for nid, orig in sorted(ctx.id_map.items()):
    if isinstance(orig, (fast.PerformEffect, fast.HandleEffect, fast.HandleCase)):
        print(f"  {nid}: {type(orig).__name__}  attrs={[k for k in vars(orig) if not k.startswith('_') and k not in ('parent','scope','location','children','node_id')]}")

print("\n=== FunctionCall nodes (possible disguised performs) ===")
for nid, orig in sorted(ctx.id_map.items()):
    if isinstance(orig, fast.FunctionCall):
        print(f"  {nid}: FunctionCall  name={orig.name!r}  args={len(getattr(orig,'arguments',[]) or [])}")

hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)

def show(h, indent=0):
    if h is None: return
    pad = "  " * indent
    extras = []
    for attr in ('literal','var_name','binop','struct_name','field_name','callee','effect_op','handle_effect'):
        v = getattr(h, attr, None)
        if v is not None: extras.append(f"{attr}={v!r}")
    print(f"{pad}{h.op}  {' '.join(extras)}")
    for child in [h.cond, h.left, h.right, h.base, h.lambda_body, h.field_val, h.handle_body]:
        if child is not None: show(child, indent+1)
    for ops in [h.then_ops, h.else_ops, h.operands, h.perform_args]:
        for c in (ops or []): show(c, indent+1)
    for name, c in (h.bindings or []): show(c, indent+2)
    if h.handle_cases:
        for (op, param, body) in h.handle_cases:
            print(f"{pad}  handle_case {op!r}({param!r}):")
            show(body, indent+3)

print("\n=== HIR ===")
for f in hir:
    print(f"func {f.sym}")
    show(f.body, 1)

print("\n=== MIR ===")
mir = lower_hir_to_mir(hir)
print(dump_mir(mir))
