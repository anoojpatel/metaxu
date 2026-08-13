"""Tests for the direct-subset CLIF emitter (codegen_clif.py).

Covers:
- golden-ish structural tests: constant function, arithmetic, if/else with a
  join, while loop with a back-edge, calls between functions, float math
- a regex-based structural validator run over the CLIF emitted for ALL
  example programs (we cannot run cranelift here, so the validator checks
  def-before-use, block references, one-terminator-per-block, and that every
  reachable exit is a return/trap)
- effectful/struct/closure functions must become comment-only placeholders,
  never bogus bodies
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from metaxu.compiler.mir import MirBlock, MirFunc
from metaxu.compiler.codegen_clif import emit_clif

REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_func(name: str, blocks: list[MirBlock], suspending: bool = False) -> MirFunc:
    return MirFunc(name=name, ty_sig=None, blocks=blocks, suspending=suspending)


def block(ops: list[tuple], term: tuple) -> MirBlock:
    return MirBlock(ops=ops, term=term)


# ---------------------------------------------------------------------------
# Structural CLIF validator (regex-based; see module docstring)
# ---------------------------------------------------------------------------

_FUNC_HEADER_RE = re.compile(r"^function %(\w+)\(([^)]*)\) -> (\w+) \{$")
_BLOCK_HEADER_RE = re.compile(r"^block(\d+)(?:\(([^)]*)\))?:$")
_RESULT_RE = re.compile(r"^(v\d+) = ")
_VALUE_RE = re.compile(r"\bv\d+\b")
_BLOCK_REF_RE = re.compile(r"\bblock(\d+)\b")
_TERMINATORS = ("jump", "brif", "return", "trap")


def split_functions(clif: str) -> list[list[str]]:
    """Return the emitted (non-placeholder) function bodies as line lists."""
    funcs: list[list[str]] = []
    cur: list[str] | None = None
    for line in clif.splitlines():
        if _FUNC_HEADER_RE.match(line):
            assert cur is None, "nested function"
            cur = [line]
        elif cur is not None:
            cur.append(line)
            if line == "}":
                funcs.append(cur)
                cur = None
    assert cur is None, "unterminated function"
    return funcs


def validate_function(lines: list[str]) -> None:
    """Assert the structural CLIF invariants for one emitted function."""
    header = _FUNC_HEADER_RE.match(lines[0])
    assert header, f"bad function header: {lines[0]!r}"
    assert lines[-1] == "}"

    defined: set[str] = set()
    declared_slots: set[str] = set()
    declared_sigs: set[str] = set()
    declared_fns: set[str] = set()
    blocks: dict[int, list[str]] = {}
    cur_block: int | None = None
    order: list[int] = []

    for raw in lines[1:-1]:
        line = raw.split(";")[0].rstrip()  # strip trailing comments
        if not line.strip():
            continue
        m = _BLOCK_HEADER_RE.match(line)
        if m:
            bi = int(m.group(1))
            assert bi not in blocks, f"duplicate block{bi}"
            blocks[bi] = []
            order.append(bi)
            cur_block = bi
            for arg in filter(None, (a.strip() for a in (m.group(2) or "").split(","))):
                vname, _, vty = arg.partition(":")
                assert vty.strip() in ("i64", "f64"), f"bad block arg {arg!r}"
                defined.add(vname.strip())
            continue
        inst = line.strip()
        if cur_block is None:
            # Preamble entry: stack slot, signature, or function declaration.
            if inst.startswith("ss"):
                name = inst.split(" ", 1)[0]
                assert re.fullmatch(r"ss\d+ = explicit_slot \d+", inst), inst
                declared_slots.add(name)
            elif inst.startswith("sig"):
                declared_sigs.add(inst.split(" ", 1)[0])
            elif inst.startswith("fn"):
                name, _, rest = inst.partition(" = ")
                declared_fns.add(name)
                sig = rest.split()[-1]
                assert sig in declared_sigs, f"fn decl uses undeclared {sig}: {inst}"
            else:
                raise AssertionError(f"unexpected preamble line: {inst!r}")
            continue

        # (a) every used value is defined before use
        rhs = inst
        rm = _RESULT_RE.match(inst)
        if rm:
            rhs = inst[rm.end():]
        for v in _VALUE_RE.findall(rhs):
            assert v in defined, f"use of {v} before definition in: {inst!r}"
        if rm:
            assert rm.group(1) not in defined, f"redefinition of {rm.group(1)}"
            defined.add(rm.group(1))
        # referenced stack slots / fn refs must be declared
        for ss in re.findall(r"\bss\d+\b", inst):
            assert ss in declared_slots, f"undeclared stack slot {ss} in {inst!r}"
        if inst.startswith("call ") or " = call " in inst:
            fnref = re.search(r"\bfn\d+\b", inst)
            assert fnref and fnref.group(0) in declared_fns, f"undeclared fn in {inst!r}"
        blocks[cur_block].append(inst)

    assert blocks, "function has no blocks"
    assert order[0] == 0, "entry block must be block0"

    # (b) every referenced block exists; (c) exactly one terminator, last
    succs: dict[int, list[int]] = {}
    exits: dict[int, str] = {}
    for bi, insts in blocks.items():
        assert insts, f"block{bi} is empty (no terminator)"
        terms = [i for i in insts if i.split()[0] in _TERMINATORS]
        assert len(terms) == 1, f"block{bi} has {len(terms)} terminators: {terms}"
        assert insts[-1] == terms[0], f"block{bi} terminator is not last"
        opcode = terms[0].split()[0]
        targets = [int(t) for t in _BLOCK_REF_RE.findall(terms[0])]
        for t in targets:
            assert t in blocks, f"block{bi} jumps to missing block{t}"
        succs[bi] = targets
        exits[bi] = opcode

    # (d) return/trap present on all exit paths: walk reachable blocks
    seen: set[int] = set()
    stack = [0]
    while stack:
        bi = stack.pop()
        if bi in seen:
            continue
        seen.add(bi)
        stack.extend(succs[bi])
    reachable_exits = [exits[bi] for bi in seen if not succs[bi]]
    assert reachable_exits, "no reachable exit block"
    assert all(e in ("return", "trap") for e in reachable_exits)


def validate_module(clif: str) -> int:
    """Validate every emitted function; return how many there were."""
    funcs = split_functions(clif)
    for lines in funcs:
        validate_function(lines)
    return len(funcs)


def count_placeholders(clif: str) -> int:
    return len(re.findall(r"placeholder -- unsupported", clif))


# ---------------------------------------------------------------------------
# Golden-ish structural tests on hand-built MIR
# ---------------------------------------------------------------------------

def test_constant_function():
    f = make_func("answer", [
        block([("params", ()), ("let", "c1", ("const", 42), ())], ("ret", "c1")),
    ])
    clif = emit_clif([f])
    assert "function %answer() -> i64 {" in clif
    assert "v0 = iconst.i64 42" in clif
    assert "return v0" in clif
    validate_module(clif)


def test_arithmetic_and_comparison():
    f = make_func("arith", [
        block([
            ("params", ("a", "b")),
            ("let", "s", ("binop", "+"), ("a", "b")),
            ("let", "p", ("binop", "*"), ("s", "b")),
            ("let", "c", ("binop", "<"), ("p", "a")),
        ], ("ret", "c")),
    ])
    clif = emit_clif([f])
    assert "function %arith(i64, i64) -> i64 {" in clif
    assert "block0(v0: i64, v1: i64):" in clif
    assert re.search(r"v2 = iadd v0, v1", clif)
    assert re.search(r"v3 = imul v2, v1", clif)
    assert re.search(r"v4 = icmp slt v3, v0", clif)
    assert re.search(r"v5 = uextend\.i64 v4", clif)
    assert "return v5" in clif
    validate_module(clif)


def test_float_arithmetic():
    f = make_func("fmath", [
        block([
            ("params", ("x",)),
            ("let", "c1", ("const", 2.0), ()),
            ("let", "m", ("binop", "*"), ("x", "c1")),
        ], ("ret", "m")),
    ])
    clif = emit_clif([f])
    assert "function %fmath(f64) -> f64 {" in clif
    assert "f64const 0x1.0000000000000p+1" in clif
    assert re.search(r"fmul v0, v1", clif)
    validate_module(clif)


def test_if_else_two_blocks_and_join():
    # Mirrors lower_hir_to_mir's If layout: br_if -> then/else, both copy
    # into a shared result slot and jump to the join.
    f = make_func("choose", [
        block([
            ("params", ("x",)),
            ("let", "c1", ("const", 5), ()),
            ("let", "t", ("binop", "<"), ("x", "c1")),
        ], ("br_if", "t", 1, 2)),
        block([
            ("let", "c2", ("const", 10), ()),
            ("let", "res", ("copy",), ("c2",)),
        ], ("br", 3)),
        block([
            ("let", "c3", ("const", 20), ()),
            ("let", "res", ("copy",), ("c3",)),
        ], ("br", 3)),
        block([], ("ret", "res")),
    ])
    clif = emit_clif([f])
    # multiply-assigned result slot -> stack slot; brif with both targets
    assert re.search(r"ss0 = explicit_slot 8", clif)
    assert re.search(r"brif v\d+, block1, block2", clif)
    assert "block3:" in clif
    assert clif.count("stack_store") == 2  # one per arm
    assert re.search(r"v\d+ = stack_load\.i64 ss0", clif)
    validate_module(clif)


def test_while_loop_back_edge():
    # Header/body/exit with a back-edge body -> header (mirrors lowering).
    f = make_func("count", [
        block([
            ("params", ("n",)),
            ("let", "c0", ("const", 0), ()),
            ("let", "i", ("copy",), ("c0",)),
        ], ("br", 1)),
        block([
            ("let", "t", ("binop", "<"), ("i", "n")),
        ], ("br_if", "t", 2, 3)),
        block([
            ("let", "c1", ("const", 1), ()),
            ("let", "s", ("binop", "+"), ("i", "c1")),
            ("let", "i", ("copy",), ("s",)),
        ], ("br", 1)),
        block([], ("ret", "i")),
    ])
    clif = emit_clif([f])
    assert "jump block1" in clif  # includes the back-edge from block2
    assert clif.count("jump block1") == 2
    # loop variable i is multiply-assigned -> stack slot loads in the header
    assert re.search(r"block1:\n    v\d+ = stack_load\.i64 ss0", clif)
    validate_module(clif)


def test_call_between_functions():
    callee = make_func("add1", [
        block([
            ("params", ("x",)),
            ("let", "c1", ("const", 1), ()),
            ("let", "s", ("binop", "+"), ("x", "c1")),
        ], ("ret", "s")),
    ])
    caller = make_func("main", [
        block([
            ("params", ()),
            ("let", "c7", ("const", 7), ()),
            ("let", "r", ("call", "add1"), ("c7",)),
        ], ("ret", "r")),
    ])
    clif = emit_clif([callee, caller])
    assert "function %add1(i64) -> i64 {" in clif
    assert "sig0 = (i64) -> i64" in clif
    assert "fn0 = %add1 sig0" in clif
    assert re.search(r"v\d+ = call fn0\(v\d+\)", clif)
    assert validate_module(clif) == 2


def test_call_float_signature_propagates_to_caller():
    # Callee's f64 signature is picked up by the module-wide fixpoint, so the
    # caller's int constant argument is emitted as an f64 constant.
    callee = make_func("halve", [
        block([
            ("params", ("x",)),
            ("let", "c", ("const", 0.5), ()),
            ("let", "m", ("binop", "*"), ("x", "c")),
        ], ("ret", "m")),
    ])
    caller = make_func("main", [
        block([
            ("params", ()),
            ("let", "c8", ("const", 8), ()),
            ("let", "r", ("call", "halve"), ("c8",)),
        ], ("ret", "r")),
    ])
    clif = emit_clif([callee, caller])
    assert "function %halve(f64) -> f64 {" in clif
    assert "function %main() -> f64 {" in clif
    assert "sig0 = (f64) -> f64" in clif
    assert re.search(r"v0 = f64const 0x1\.0000000000000p\+3", clif)  # 8 -> 8.0
    validate_module(clif)


def test_extern_call_declared():
    f = make_func("main", [
        block([
            ("params", ()),
            ("let", "c1", ("const", 3), ()),
            ("let", "r", ("call", "print"), ("c1",)),
        ], ("ret", "r")),
    ])
    clif = emit_clif([f])
    assert "fn0 = %print sig0" in clif
    validate_module(clif)


def test_select_and_unit():
    f = make_func("pick", [
        block([
            ("params", ("c",)),
            ("let", "a", ("const", 1), ()),
            ("let", "b", ("const", 2), ()),
            ("let", "u", ("const_ty", "Unit"), ()),
            ("let", "r", ("select",), ("c", "a", "b")),
        ], ("ret", "r")),
    ])
    clif = emit_clif([f])
    assert re.search(r"v\d+ = select v0, v\d+, v\d+", clif)
    assert "iconst.i64 0" in clif  # unit constant
    validate_module(clif)


def test_match_fail_traps():
    f = make_func("m", [
        block([("params", ()), ("match_fail", "no pattern matched")], ("br", 1)),
        block([("let", "u", ("const_ty", "Unit"), ())], ("ret", "u")),
    ])
    clif = emit_clif([f])
    assert "trap user0" in clif
    validate_module(clif)


# ---------------------------------------------------------------------------
# Placeholder behavior for non-direct functions
# ---------------------------------------------------------------------------

def test_effectful_function_is_placeholder_not_bogus_body():
    f = make_func("worker", [
        block([
            ("params", ("x",)),
            ("perform", "pv1", "State", "get", (), 1, "pv1"),
        ], ("br", 1)),
        block([], ("ret", "pv1")),
    ], suspending=True)
    clif = emit_clif([f])
    assert "function %worker(" not in clif  # no emitted body at all
    assert "placeholder -- unsupported" in clif
    assert "uses effects (perform)" in clif
    assert "; declare %worker(i64) -> i64" in clif
    # every non-empty line of the placeholder is a comment
    chunk = [c for c in clif.split("\n\n") if "worker" in c][0]
    assert all(line.startswith(";") for line in chunk.splitlines() if line.strip())


def test_handle_scope_and_resume_are_placeholders():
    handler = make_func("__handler_State_get_hs1", [
        block([
            ("params", ("v", "__k")),
            ("let", "rv", ("resume",), ("__k", "v")),
        ], ("ret", "rv")),
    ])
    outer = make_func("outer", [
        block([
            ("params", ()),
            ("let", "h", ("handle_scope", "__body", "State", ()), ()),
        ], ("ret", "h")),
    ])
    clif = emit_clif([handler, outer])
    assert count_placeholders(clif) == 2
    assert "uses effects (resume)" in clif
    assert "uses effects (handle_scope)" in clif
    assert split_functions(clif) == []


def test_struct_variant_closure_string_are_placeholders():
    fs = [
        make_func("s", [block([
            ("params", ()),
            ("let", "s1", ("alloc_struct", "Point", "local"), (("x", "a"),)),
        ], ("ret", "s1"))]),
        make_func("v", [block([
            ("params", ()),
            ("let", "v1", ("make_variant", "Opt", "Some"), ("a",)),
        ], ("ret", "v1"))]),
        make_func("c", [block([
            ("params", ()),
            ("let", "c1", ("make_closure", "lambda1", ("x",)), ()),
        ], ("ret", "c1"))]),
        make_func("st", [block([
            ("params", ()),
            ("let", "c1", ("const", "hello"), ()),
        ], ("ret", "c1"))]),
    ]
    clif = emit_clif(fs)
    assert count_placeholders(clif) == 4
    assert "uses structs (alloc_struct)" in clif
    assert "uses variants (make_variant)" in clif
    assert "uses closures (make_closure)" in clif
    assert "uses strings (string constant)" in clif
    assert split_functions(clif) == []


def test_runtime_builtin_call_is_placeholder():
    f = make_func("m", [
        block([
            ("params", ()),
            ("let", "r", ("call", "__vec_lit"), ()),
        ], ("ret", "r")),
    ])
    clif = emit_clif([f])
    assert count_placeholders(clif) == 1
    assert "__vec_lit" in clif


# ---------------------------------------------------------------------------
# All-examples validation
# ---------------------------------------------------------------------------

# Negative fixtures never reach codegen (the pipeline rejects them).
_REJECTED = {"test_borrow_check.mx", "test_type_error.mx"}


def _example_files():
    files = sorted((REPO_ROOT / "examples").glob("*.mx"))
    files += sorted(REPO_ROOT.glob("test_*.mx"))
    return [f for f in files if f.name not in _REJECTED]


@pytest.mark.parametrize("path", _example_files(), ids=lambda p: p.name)
def test_all_examples_emit_structurally_valid_clif(path):
    from metaxu.compiler.pipeline import run_pipeline_from_source

    _ast, _hir, _mir, clif = run_pipeline_from_source(path.read_text())
    emitted = validate_module(clif)
    placeholders = count_placeholders(clif)
    assert emitted + placeholders > 0
    # every placeholder must carry at least one reason line
    for chunk in clif.split("\n\n"):
        if "placeholder -- unsupported" in chunk:
            assert ";   reason: " in chunk, f"placeholder without reason:\n{chunk}"
