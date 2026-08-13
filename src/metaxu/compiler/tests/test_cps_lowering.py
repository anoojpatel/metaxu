"""Tests for selective CPS lowering of suspending functions (roadmap item 6).

Covers:
- cps_frames.compute_frame_layouts: frame slots for variables live across a
  perform (and across calls to other suspending functions), state/result
  discriminant slots, per-point live sets, and the negative cases (a value
  not live across a suspension gets no slot; the perform's own destination
  is delivered via the result slot, not a live-across slot).
- codegen_clif CPS emission for a real parsed program with an effect:
  %run_<f> with br_table dispatch, frame stores before the park site,
  enqueue/sched_read extern declarations matching the runtime ABI in
  src/metaxu/runtime/src/lib.rs, %resume_<f>_<k> shims with return_call.
- read-shaped performs park through sched_read with the resume shim's
  address as the continuation argument.
- all emitted CPS functions pass the structural validator shared with
  test_codegen_clif.py.
"""
from __future__ import annotations

import re

from metaxu.compiler.mir import MirBlock, MirFunc
from metaxu.compiler.codegen_clif import emit_clif
from metaxu.compiler.cps_frames import (
    RESULT_OFFSET, STATE_OFFSET, compute_frame_layouts, is_suspending, var_offset)

from test_codegen_clif import split_functions, validate_module


def make_func(name: str, blocks: list[MirBlock], suspending: bool = False) -> MirFunc:
    return MirFunc(name=name, ty_sig=None, blocks=blocks, suspending=suspending)


def block(ops: list[tuple], term: tuple) -> MirBlock:
    return MirBlock(ops=ops, term=term)


def compile_clif(source: str) -> str:
    from metaxu.compiler.pipeline import run_pipeline_from_source

    _ast, _hir, _mir, clif = run_pipeline_from_source(source)
    return clif


# ---------------------------------------------------------------------------
# compute_frame_layouts
# ---------------------------------------------------------------------------

def _worker_two_vars() -> MirFunc:
    # a is defined before the perform and used after it -> live across.
    # b is defined before the perform but only used before it -> NOT live.
    return make_func("worker", [
        block([
            ("params", ()),
            ("let", "a", ("const", 5), ()),
            ("let", "b", ("const", 6), ()),
            ("let", "c", ("binop", "+"), ("b", "b")),
            ("perform", "pv1", "Ask", "ask", ("c",), 1, "pv1"),
        ], ("br", 1)),
        block([
            ("let", "s", ("binop", "+"), ("a", "pv1")),
        ], ("ret", "s")),
    ], suspending=True)


def test_live_across_var_gets_frame_slot_dead_var_does_not():
    layouts = compute_frame_layouts([_worker_two_vars()])
    assert set(layouts) == {"worker"}
    lay = layouts["worker"]
    assert lay["state_offset"] == STATE_OFFSET == 0
    assert lay["result_offset"] == RESULT_OFFSET == 8
    assert "a" in lay["vars"]          # live across the perform -> frame slot
    assert "b" not in lay["vars"]      # used only before the perform -> none
    assert "c" not in lay["vars"]      # perform argument, consumed at the park
    assert lay["live_across"] == ["a"]
    assert lay["size"] == 24           # state + result + a (no params)
    assert var_offset(lay, "a") == 16

    [point] = lay["suspend_points"]
    assert point["state"] == 1
    assert point["effect"] == "Ask" and point["op"] == "ask"
    assert point["resume_block"] == 1
    assert point["dst"] == "pv1"
    assert point["live"] == ["a"]


def test_perform_dst_is_not_live_across_its_own_perform():
    # pv1 is delivered through the result slot; it must not get a
    # live-across slot from its own suspension point.
    f = make_func("w", [
        block([
            ("params", ()),
            ("perform", "pv1", "Ask", "ask", (), 1, "pv1"),
        ], ("br", 1)),
        block([], ("ret", "pv1")),
    ], suspending=True)
    lay = compute_frame_layouts([f])["w"]
    assert lay["vars"] == {}
    assert lay["suspend_points"][0]["live"] == []
    assert lay["size"] == 16


def test_dst_of_earlier_perform_live_across_later_perform():
    # let a = perform; let b = perform; a + b  -> a is live across point 2.
    f = make_func("w", [
        block([
            ("params", ()),
            ("perform", "pv1", "Ask", "ask", (), 1, "pv1"),
        ], ("br", 1)),
        block([
            ("let", "a", ("copy",), ("pv1",)),
            ("perform", "pv2", "Ask", "ask", (), 2, "pv2"),
        ], ("br", 2)),
        block([
            ("let", "s", ("binop", "+"), ("a", "pv2")),
        ], ("ret", "s")),
    ], suspending=True)
    lay = compute_frame_layouts([f])["w"]
    p1, p2 = lay["suspend_points"]
    assert p1["live"] == []
    assert p2["live"] == ["a"]
    assert "a" in lay["vars"] and "pv2" not in lay["vars"] and "s" not in lay["vars"]


def test_params_have_frame_slots_and_live_across_param_is_not_duplicated():
    # Params always get frame slots (state 0 reads them from the frame);
    # a param that is also live across a perform keeps its single slot.
    f = make_func("w", [
        block([
            ("params", ("x", "y")),
            ("perform", "pv1", "Ask", "ask", (), 1, "pv1"),
        ], ("br", 1)),
        block([
            ("let", "s", ("binop", "+"), ("x", "pv1")),
        ], ("ret", "s")),
    ], suspending=True)
    lay = compute_frame_layouts([f])["w"]
    assert lay["params"] == {"x": 16, "y": 24}
    assert "x" in lay["live_across"] and "y" not in lay["live_across"]
    assert "x" not in lay["vars"]  # no duplicate slot
    assert lay["size"] == 32


def test_var_live_across_call_to_suspending_function_gets_slot():
    helper = make_func("helper", [
        block([
            ("params", ()),
            ("perform", "pv1", "Ask", "ask", (), 1, "pv1"),
        ], ("br", 1)),
        block([], ("ret", "pv1")),
    ], suspending=True)
    # outer is itself suspending (contains a perform) and keeps `a` live
    # across the suspending call to helper.
    outer = make_func("outer", [
        block([
            ("params", ()),
            ("perform", "pv0", "Ask", "ask", (), 1, "pv0"),
        ], ("br", 1)),
        block([
            ("let", "a", ("copy",), ("pv0",)),
            ("let", "r", ("call", "helper"), ()),
            ("let", "s", ("binop", "+"), ("a", "r")),
        ], ("ret", "s")),
    ], suspending=True)
    layouts = compute_frame_layouts([helper, outer])
    lay = layouts["outer"]
    [call] = lay["suspending_calls"]
    assert call["callee"] == "helper"
    assert "a" in call["live"]
    assert "a" in lay["vars"]          # framed because of the suspending call
    assert "r" not in call["live"]     # defined BY the call, not before it


def test_only_suspending_functions_get_layouts():
    direct = make_func("plain", [
        block([("params", ()), ("let", "c", ("const", 1), ())], ("ret", "c")),
    ])
    flagged = make_func("f1", [
        block([("params", ()), ("let", "c", ("const", 1), ())], ("ret", "c")),
    ], suspending=True)
    with_resume = make_func("h", [
        block([
            ("params", ("v", "__k")),
            ("let", "rv", ("resume",), ("__k", "v")),
        ], ("ret", "rv")),
    ])
    assert not is_suspending(direct)
    assert is_suspending(flagged)      # flag alone marks it
    assert is_suspending(with_resume)  # effect op alone marks it
    layouts = compute_frame_layouts([direct, flagged, with_resume])
    assert set(layouts) == {"f1", "h"}


# ---------------------------------------------------------------------------
# CLIF output for a parsed effectful program
# ---------------------------------------------------------------------------

_TWO_ASKS = """
effect Ask {
    ask() -> int
}

fn main() -> int {
    handle Ask with {
        ask() -> resume(7)
    } in {
        let a = perform Ask.ask();
        let b = perform Ask.ask();
        a + b
    }
}
"""


def test_parsed_effect_program_emits_cps_shape():
    clif = compile_clif(_TWO_ASKS)

    # The handle body sub-function performs twice -> CPS with two states.
    run_funcs = [ln for ln in clif.splitlines()
                 if ln.startswith("function %run_")]
    assert len(run_funcs) == 1
    m = re.search(r"function %run_(\w+)\(i64\) -> i64 \{", clif)
    assert m, "no %run_ function emitted"
    name = m.group(1)

    # Frame layout comment table with state/result and the live-across var.
    frame = re.search(rf"; frame %{name}: size=(\d+), (.+)", clif)
    assert frame, "no frame comment table"
    assert "[0]=state:i64" in frame.group(2)
    assert "[8]=result:i64" in frame.group(2)
    assert re.search(r"\[16\]=\w+:i64", frame.group(2))  # `a` is live across

    # Dispatch: br_table on the loaded state with entry + 2 resume targets.
    assert re.search(r"v\d+ = load\.i64 v0\b.*\n.*ireduce\.i32", clif)
    assert re.search(r"br_table v\d+, block\d+, \[block\d+, block\d+, block\d+\]", clif)

    # Park sites: live-var frame stores BEFORE the enqueue call, then the
    # state store, then enqueue(frame) and the parked return.
    park2 = re.search(
        r"(?: {4}v(\d+) = stack_load\.i64 ss\d+\n"
        r" {4}store v\1, v0\+\d+  ; save \w+\n)"
        r"(?:.*\n)*?"
        r" {4}store v\d+, v0  ; state = 2\n"
        r" {4}call fn\d+\(v0\)\n"
        r" {4}v\d+ = iconst\.i64 0\n"
        r" {4}return v\d+  ; parked",
        clif)
    assert park2, "no frame-store-then-enqueue park shape for point 2"

    # Runtime externs declared with the ABI signatures from lib.rs:
    #   enqueue(frame: *mut u8)  ->  (i64)
    #   sched_read(fd, buf, len, k, frame)  ->  (i64, i64, i64, i64, i64)
    enq = re.search(r"sig(\d+) = \(i64\)\n {4}fn(\d+) = %enqueue sig\1", clif)
    assert enq, "enqueue extern not declared with (i64) signature"
    srd = re.search(
        r"sig(\d+) = \(i64, i64, i64, i64, i64\)\n {4}fn(\d+) = %sched_read sig\1",
        clif)
    assert srd, "sched_read extern not declared with (i64 x5) signature"
    # enqueue is actually called with the frame pointer
    assert re.search(rf"call fn{enq.group(2)}\(v0\)", clif)

    # Resume shims: one per suspension point, storing into the result slot
    # and tail-calling the run function.
    assert f"function %resume_{name}_1(i64, i64) -> i64 {{" in clif
    assert f"function %resume_{name}_2(i64, i64) -> i64 {{" in clif
    assert clif.count("store v1, v0+8  ; frame.result = value") == 2
    assert clif.count(f"fn0 = %run_{name} sig0") == 2
    assert clif.count("return_call fn0(v0)") == 2

    # Resume prologues restore the live var and deliver the result slot.
    assert re.search(
        r"; state 2: resume after Ask\.ask\n"
        r" {4}v\d+ = load\.i64 v0\+16  ; restore \w+\n"
        r" {4}stack_store v\d+, ss\d+\n"
        r" {4}v\d+ = load\.i64 v0\+8  ; resumed value",
        clif)

    # Everything emitted (run + 2 shims) passes the structural validator.
    assert validate_module(clif) == 3


def test_single_ask_program_matches_effect_continuation_shape():
    # The Ask/resume(7) shape from test_effect_continuations.py.
    clif = compile_clif("""
effect Ask {
    ask() -> int
}

fn main() -> int {
    handle Ask with {
        ask() -> resume(7)
    } in {
        perform Ask.ask() + 1
    }
}
""")
    assert re.search(r"function %run_\w+\(i64\) -> i64 \{", clif)
    assert re.search(r"br_table v\d+, block\d+, \[block\d+, block\d+\]", clif)
    assert re.search(r"function %resume_\w+_1\(i64, i64\) -> i64 \{", clif)
    # handler (resume) and main (handle_scope) stay placeholders: effect
    # dispatch is interpreter territory, only the mechanical shape is CLIF.
    assert "uses effects (resume)" in clif
    assert "uses effects (handle_scope)" in clif
    validate_module(clif)


def test_deep_handler_program_helper_gets_cps_body():
    # helper() performs Ask -> suspending flag from lowering; its body is in
    # the subset so it gets a real CPS body. The handle body contains no
    # effect op itself, so per the marking rule it stays DIRECT: it calls
    # %helper as an extern-style entry (suspension marking is not transitive
    # through the call graph at the CLIF level).
    clif = compile_clif("""
effect Ask {
    ask() -> int
}

fn helper() performs Ask -> int {
    let x = perform Ask.ask();
    x * 10
}

fn main() -> int {
    handle Ask with {
        ask() -> resume(4)
    } in {
        helper() + 2
    }
}
""")
    assert "function %run_helper(i64) -> i64 {" in clif
    assert "function %resume_helper_1(i64, i64) -> i64 {" in clif
    # the direct handle body calls %helper as an extern entry point
    assert re.search(r"fn\d+ = %helper sig\d+", clif)
    validate_module(clif)


# ---------------------------------------------------------------------------
# sched_read wiring for read-shaped performs
# ---------------------------------------------------------------------------

def test_read_perform_parks_through_sched_read_with_shim_address():
    f = make_func("reader", [
        block([
            ("params", ("fd",)),
            ("perform", "pv1", "Io", "read", ("fd",), 1, "pv1"),
        ], ("br", 1)),
        block([], ("ret", "pv1")),
    ], suspending=True)
    clif = emit_clif([f])
    assert "function %run_reader(i64) -> i64 {" in clif
    # continuation = the resume shim's address, frame as last arg
    shim_decl = re.search(r"(fn\d+) = %resume_reader_1 sig", clif)
    assert shim_decl, "resume shim not declared for func_addr"
    assert re.search(rf"v\d+ = func_addr\.i64 {shim_decl.group(1)}", clif)
    srd_decl = re.search(r"(fn\d+) = %sched_read sig", clif)
    assert srd_decl
    # sched_read(fd, buf, len, k, frame): fd from the perform arg, padded 0s
    assert re.search(
        rf"call {srd_decl.group(1)}\(v\d+, v\d+, v\d+, v\d+, v0\)", clif)
    assert "function %resume_reader_1(i64, i64) -> i64 {" in clif
    assert validate_module(clif) == 2


def test_non_read_perform_parks_through_enqueue():
    f = make_func("getter", [
        block([
            ("params", ()),
            ("perform", "pv1", "State", "get", (), 1, "pv1"),
        ], ("br", 1)),
        block([], ("ret", "pv1")),
    ], suspending=True)
    clif = emit_clif([f])
    enq_decl = re.search(r"(fn\d+) = %enqueue sig", clif)
    assert enq_decl
    assert re.search(rf"call {enq_decl.group(1)}\(v0\)", clif)
    assert "func_addr" not in clif  # no sched_read continuation needed
    validate_module(clif)


# ---------------------------------------------------------------------------
# Structural properties of emitted CPS functions
# ---------------------------------------------------------------------------

def test_cps_function_with_control_flow_validates():
    # perform inside a loop: the loop counter is live across the suspension
    # and the resume block branches back into the loop header.
    f = make_func("loopy", [
        block([
            ("params", ("n",)),
            ("let", "c0", ("const", 0), ()),
            ("let", "i", ("copy",), ("c0",)),
            ("let", "acc", ("copy",), ("c0",)),
        ], ("br", 1)),
        block([
            ("let", "t", ("binop", "<"), ("i", "n")),
        ], ("br_if", "t", 2, 4)),
        block([
            ("perform", "pv", "Ask", "ask", (), 3, "pv"),
        ], ("br", 3)),
        block([
            ("let", "acc", ("binop", "+"), ("acc", "pv")),
            ("let", "c1", ("const", 1), ()),
            ("let", "i", ("binop", "+"), ("i", "c1")),
        ], ("br", 1)),
        block([], ("ret", "acc")),
    ], suspending=True)
    layouts = compute_frame_layouts([f])
    lay = layouts["loopy"]
    # i, n and acc are all live across the in-loop suspension
    assert {"i", "acc"} <= set(lay["vars"])
    assert "n" in lay["params"]
    [point] = lay["suspend_points"]
    assert set(point["live"]) >= {"acc", "i", "n"}

    clif = emit_clif([f])
    assert "function %run_loopy(i64) -> i64 {" in clif
    # park saves each live var into its frame slot before enqueue
    for lv in ("acc", "i"):
        assert re.search(rf"store v\d+, v0\+{var_offset(lay, lv)}  ; save {lv}", clif)
    assert validate_module(clif) == 2


def test_every_emitted_function_has_single_terminator_blocks():
    # Sanity: split_functions/validate_function reject double terminators;
    # run this over a CPS module to make sure park replacement removed the
    # original `br` of perform blocks instead of stacking a second one.
    clif = compile_clif(_TWO_ASKS)
    for lines in split_functions(clif):
        joined = "\n".join(lines)
        assert "jump" not in joined or "return v" in joined or "return_call" in joined
    validate_module(clif)
