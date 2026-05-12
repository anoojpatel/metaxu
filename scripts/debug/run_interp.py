"""Quick smoke-runner: compile .mx source -> MIR -> interpreter, print results."""
import sys
import traceback
import logging
logging.disable(logging.CRITICAL)

from metaxu.compiler.pipeline import build_context_from_source
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir import dump_mir
from metaxu.compiler.mir_interp import MirInterpreter


def compile_and_run(label: str, src: str, fn_name: str, args: list, expected=None):
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    print(f"  source: {src.strip()[:100]}")
    try:
        ctx = build_context_from_source(src)
        hir_funcs = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
        mir_funcs = lower_hir_to_mir(hir_funcs)

        print("\n  --- MIR ---")
        for line in dump_mir(mir_funcs).splitlines():
            print(f"  {line}")

        found = [f for f in mir_funcs if fn_name in f.name]
        if not found:
            print(f"\n  [SKIP] no function matching '{fn_name}' in [{', '.join(f.name for f in mir_funcs)}]")
            return

        interp = MirInterpreter()
        interp.load(mir_funcs)
        result = interp.call(found[0].name, args)
        status = ""
        if expected is not None:
            status = "  OK" if result == expected else f"  MISMATCH (expected {expected!r})"
        print(f"\n  RESULT  {fn_name}({', '.join(repr(a) for a in args)}) = {result!r}{status}")
    except Exception as e:
        print(f"\n  ERROR: {e}")
        traceback.print_exc()


# ── Basic arithmetic ──────────────────────────────────────────────────────────
compile_and_run(
    "double(21)",
    "fn double(x: Int) -> Int { x + x }",
    "double", [21], expected=42
)

compile_and_run(
    "add(10, 32)",
    "fn add(a: Int, b: Int) -> Int { a + b }",
    "add", [10, 32], expected=42
)

compile_and_run(
    "mul(6, 7)",
    "fn mul(a: Int, b: Int) -> Int { a * b }",
    "mul", [6, 7], expected=42
)

# ── Let binding ───────────────────────────────────────────────────────────────
compile_and_run(
    "let_binding: let x = n+1; x+x",
    "fn compute(n: Int) -> Int { let x = n + 1; x + x }",
    "compute", [20], expected=42
)

# ── If / else branch ──────────────────────────────────────────────────────────
compile_and_run(
    "if branch (positive)",
    """
fn sign(x: Int) -> Int {
    if x > 0 {
        1
    } else {
        0
    }
}
""",
    "sign", [5], expected=1
)

compile_and_run(
    "if branch (zero)",
    """
fn sign(x: Int) -> Int {
    if x > 0 {
        1
    } else {
        0
    }
}
""",
    "sign", [0], expected=0
)

# ── Recursive call ────────────────────────────────────────────────────────────
compile_and_run(
    "factorial(5) via recursion",
    """
fn fact(n: Int) -> Int {
    if n < 2 {
        1
    } else {
        n * fact(n - 1)
    }
}
""",
    "fact", [5], expected=120
)

# ── Multi-function call ───────────────────────────────────────────────────────
compile_and_run(
    "double via helper call",
    """
fn add(a: Int, b: Int) -> Int { a + b }
fn double(x: Int) -> Int { add(x, x) }
""",
    "double", [21], expected=42
)
