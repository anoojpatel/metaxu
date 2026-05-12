"""Run increasingly complex Metaxu programs through the full pipeline.

Syntax reference (from parser.py):
  struct Foo { a: Type, b: Type }           -- comma-separated fields with COLON
  Foo { a = expr, b = expr }                -- instantiation uses EQUALS
  effect Name = { fn op(p: T) -> R ... }    -- effect decl uses EQUALS before brace
  perform fn op(arg) -> RetTy               -- perform uses `fn` keyword
  handle TypeExpr with { op(v) -> expr } in expr
"""
import logging
logging.disable(logging.CRITICAL)
import traceback

from metaxu.compiler.pipeline import build_context_from_source
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir import dump_mir
from metaxu.compiler.mir_interp import MirInterpreter, MxStruct, MxClosure, UNIT


def run(label, src, fn, args, expected=None):
    print(f"\n{'='*64}")
    print(f"  {label}")
    print(f"  call: {fn}({', '.join(repr(a) for a in args)})")
    print(f"{'='*64}")
    try:
        ctx = build_context_from_source(src)
        hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
        mir = lower_hir_to_mir(hir)
        print(dump_mir(mir))
        found = [f for f in mir if fn in f.name]
        if not found:
            print(f"  [SKIP] no fn '{fn}' — available: {[f.name for f in mir]}")
            return
        interp = MirInterpreter()
        interp.load(mir)
        result = interp.call(found[0].name, args)
        if expected is not None:
            status = "OK" if result == expected else f"MISMATCH (expected {expected!r})"
        else:
            status = "ran"
        print(f"\n  RESULT = {result!r}  [{status}]")
    except Exception as e:
        print(f"\n  ERROR: {e}")
        traceback.print_exc()


# ── 1. Struct instantiation + field access ────────────────────────────────────
# struct fields: comma-separated with COLON; instantiation: EQUALS
run("struct Point: create and read field x",
"""
struct Point {
    x: Int,
    y: Int
}

fn make_point(a: Int, b: Int) -> Point {
    Point { x = a, y = b }
}

fn get_x(p: Point) -> Int {
    p.x
}

fn main(a: Int, b: Int) -> Int {
    let p = make_point(a, b)
    get_x(p)
}
""", "main", [10, 20], expected=10)


# ── 2. Struct field arithmetic + value update ─────────────────────────────────
run("struct Counter: increment returns new struct",
"""
struct Counter {
    value: Int
}

fn increment(c: Counter) -> Counter {
    Counter { value = c.value + 1 }
}

fn main(n: Int) -> Int {
    let c = Counter { value = n }
    let c2 = increment(c)
    c2.value
}
""", "main", [41], expected=42)


# ── 3. Struct with two fields used in arithmetic ──────────────────────────────
run("struct Config: two fields, arithmetic on both",
"""
struct Config {
    limit: Int,
    scale: Int
}

fn compute(cfg: Config, n: Int) -> Int {
    n * cfg.scale + cfg.limit
}

fn main(n: Int) -> Int {
    let cfg = Config { limit = 2, scale = 10 }
    compute(cfg, n)
}
""", "main", [4], expected=42)


# ── 4. Nested struct field access ─────────────────────────────────────────────
run("struct Outer holds struct Inner: deep field access",
"""
struct Inner {
    v: Int
}

struct Outer {
    inner: Inner,
    extra: Int
}

fn sum_outer(o: Outer) -> Int {
    o.inner.v + o.extra
}

fn main(a: Int, b: Int) -> Int {
    let i = Inner { v = a }
    let o = Outer { inner = i, extra = b }
    sum_outer(o)
}
""", "main", [20, 22], expected=42)


# ── 5. Multi-function call chain ──────────────────────────────────────────────
run("call chain: triple(n) = double(double(n)) - n",
"""
fn double(x: Int) -> Int { x + x }
fn triple(x: Int) -> Int { double(x) + x }

fn main(n: Int) -> Int {
    triple(n)
}
""", "main", [14], expected=42)


# ── 6. Recursive fibonacci ────────────────────────────────────────────────────
run("recursive fib(9) = 34",
"""
fn fib(n: Int) -> Int {
    if n < 2 {
        n
    } else {
        fib(n - 1) + fib(n - 2)
    }
}

fn main(n: Int) -> Int {
    fib(n)
}
""", "main", [9], expected=34)


# ── 7. Struct + recursion ─────────────────────────────────────────────────────
run("struct wraps fib result",
"""
struct Result {
    value: Int
}

fn fib(n: Int) -> Int {
    if n < 2 {
        n
    } else {
        fib(n - 1) + fib(n - 2)
    }
}

fn main(n: Int) -> Int {
    let r = Result { value = fib(n) }
    r.value
}
""", "main", [9], expected=34)


# ── 8. Stack effect: perform + handle ────────────────────────────────────────
# Syntax: effect Name = { fn op(p: T) -> R }
# perform fn op(arg) -> RetTy
# handle TypeExpr with { op(v) -> expr } in expr
run("stack effect Log: perform + handle passes value through",
"""
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
""", "main", [19, 23], expected=42)


# ── 9. Struct + stack effect ──────────────────────────────────────────────────
run("struct Pair + effect: emit sum of fields",
"""
struct Pair {
    x: Int,
    y: Int
}

effect Emit = {
    fn emit(v: Int) -> Int
}

fn sum_pair(p: Pair) -> Int {
    let s = p.x + p.y
    perform emit(s)
}

fn main(a: Int, b: Int) -> Int {
    let p = Pair { x = a, y = b }
    handle Emit with {
        emit(v) -> v
    } in sum_pair(p)
}
""", "main", [19, 23], expected=42)
