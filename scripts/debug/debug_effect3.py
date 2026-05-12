"""Narrow down which handle-in-body variants parse."""
import logging
logging.disable(logging.CRITICAL)
from metaxu.compiler.pipeline import build_context_from_source

def probe(label, src):
    try:
        ctx = build_context_from_source(src)
        print(f"  OK    {label}")
    except Exception as e:
        msg = str(e)
        for line in msg.splitlines():
            if 'Syntax error' in line or 'ParseError at <mem>' in line:
                print(f"  FAIL  {label}  —  {line.strip()}")
                return
        print(f"  FAIL  {label}  —  {msg[:100]}")

# The issue is `g(x)` as the `in expr` part
# Parser grammar: handle_expression : HANDLE type_expression WITH LBRACE handle_cases RBRACE IN expression
# So `expression` after IN should include function calls

probe("handle in variable",
"""effect Log = {
    fn emit(v: Int) -> Int
}
fn f(x: Int) -> Int {
    handle Log with {
        emit(v) -> v
    } in x
}""")

probe("handle in add",
"""effect Log = {
    fn emit(v: Int) -> Int
}
fn f(x: Int) -> Int {
    handle Log with {
        emit(v) -> v
    } in x + 1
}""")

probe("handle in fn call (1 arg)",
"""effect Log = {
    fn emit(v: Int) -> Int
}
fn g(x: Int) -> Int { x }
fn f(x: Int) -> Int {
    handle Log with {
        emit(v) -> v
    } in g(x)
}""")

probe("handle in fn call with perform inside",
"""effect Log = {
    fn emit(v: Int) -> Int
}
fn g(x: Int) -> Int { perform emit(x) }
fn f(x: Int) -> Int {
    handle Log with {
        emit(v) -> v
    } in g(x)
}""")

probe("handle inline: perform in continuation body directly",
"""effect Log = {
    fn emit(v: Int) -> Int
}
fn f(x: Int) -> Int {
    handle Log with {
        emit(v) -> v
    } in perform emit(x)
}""")

probe("handle in let then field",
"""effect Log = {
    fn emit(v: Int) -> Int
}
fn f(x: Int) -> Int {
    handle Log with {
        emit(v) -> v
    } in {
        let r = perform emit(x)
        r
    }
}""")
