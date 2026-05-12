"""Probe: compile .mx source -> MIR -> interpreter."""
import logging
logging.disable(logging.CRITICAL)

from metaxu.compiler.pipeline import build_context_from_source
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir import dump_mir
from metaxu.compiler.mir_interp import MirInterpreter


def run(label, src, fn, args, expected=None):
    try:
        ctx = build_context_from_source(src)
        hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
        mir = lower_hir_to_mir(hir)
        found = [f for f in mir if fn in f.name]
        if not found:
            print(f"SKIP  {label}: no fn '{fn}' in {[f.name for f in mir]}")
            return
        interp = MirInterpreter()
        interp.load(mir)
        result = interp.call(found[0].name, args)
        ok = "" if expected is None else ("  OK" if result == expected else f"  MISMATCH expected={expected!r}")
        print(f"OK    {label}: {fn}{args} = {result!r}{ok}")
        print("      MIR:")
        for line in dump_mir(mir).splitlines():
            print(f"        {line}")
    except Exception as e:
        print(f"ERR   {label}: {e}")


# --- arithmetic ---
run("double",    "fn double(x: Int) -> Int { x + x }",                          "double", [21], 42)
run("add",       "fn add(a: Int, b: Int) -> Int { a + b }",                      "add",    [10, 32], 42)
run("mul",       "fn mul(a: Int, b: Int) -> Int { a * b }",                      "mul",    [6, 7], 42)

# --- let binding (newline separated, no semicolons) ---
run("let_x",
    "fn f(n: Int) -> Int {\nlet x = n + 1\nx + x\n}",
    "f", [20], 42)

# --- if/else (single expression in each branch) ---
run("if_pos",
    "fn sign(x: Int) -> Int { if x > 0 { 1 } else { 0 } }",
    "sign", [5], 1)

run("if_zero",
    "fn sign(x: Int) -> Int { if x > 0 { 1 } else { 0 } }",
    "sign", [0], 0)

# --- multi-function call ---
run("call_helper",
    "fn add(a: Int, b: Int) -> Int { a + b }\nfn double(x: Int) -> Int { add(x, x) }",
    "double", [21], 42)

# --- recursion ---
run("fact",
    "fn fact(n: Int) -> Int {\nif n < 2 { 1 } else { n * fact(n - 1) }\n}",
    "fact", [5], 120)
