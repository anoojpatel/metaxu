"""Probe which effect/perform/handle syntax variants parse correctly."""
import logging
logging.disable(logging.CRITICAL)

from metaxu.compiler.pipeline import build_context_from_source

def probe(label, src):
    try:
        ctx = build_context_from_source(src)
        print(f"  OK    {label}")
    except Exception as e:
        msg = str(e)
        # Extract just the key error
        for line in msg.splitlines():
            if 'Syntax error' in line or 'ParseError at <mem>' in line:
                print(f"  FAIL  {label}  —  {line.strip()}")
                return
        print(f"  FAIL  {label}  —  {msg[:120]}")

print("=== effect declaration syntax ===")
probe("effect Name = { fn op(v: Int) -> Int }",
      "effect Log = {\n    fn emit(v: Int) -> Int\n}")

probe("effect Name : stack = { fn op(v: Int) -> Int }",
      "effect Log : stack = {\n    fn emit(v: Int) -> Int\n}")

probe("effect Name = { fn op(v) -> Int }  (no type on param)",
      "effect Log = {\n    fn emit(v) -> Int\n}")

print("\n=== perform syntax ===")
probe("perform fn emit(result) -> Int  (typed)",
      "effect Log = {\n    fn emit(v: Int) -> Int\n}\nfn f(x: Int) -> Int {\n    perform fn emit(x) -> Int\n}")

probe("perform Log.emit(x)  (qualified)",
      "effect Log = {\n    fn emit(v: Int) -> Int\n}\nfn f(x: Int) -> Int {\n    perform Log.emit(x)\n}")

probe("perform emit(x)  (bare name)",
      "effect Log = {\n    fn emit(v: Int) -> Int\n}\nfn f(x: Int) -> Int {\n    perform emit(x)\n}")

print("\n=== handle syntax ===")
probe("handle Log with { emit(v) -> v } in expr",
      "effect Log = {\n    fn emit(v: Int) -> Int\n}\nfn f(x: Int) -> Int {\n    handle Log with {\n        emit(v) -> v\n    } in x\n}")

probe("handle Log with { emit(v) -> v } in fn_call",
      "effect Log = {\n    fn emit(v: Int) -> Int\n}\nfn g(x: Int) -> Int { perform fn emit(x) -> Int }\nfn f(x: Int) -> Int {\n    handle Log with {\n        emit(v) -> v\n    } in g(x)\n}")

print("\n=== performs clause on function ===")
probe("fn f() -> Int performs Log { ... }",
      "effect Log = {\n    fn emit(v: Int) -> Int\n}\nfn f(x: Int) -> Int performs Log {\n    perform fn emit(x) -> Int\n}")
