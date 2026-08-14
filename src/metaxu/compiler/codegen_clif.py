"""Cranelift IR (CLIF) text emitter for direct (non-suspending) functions.

Emits real, structurally valid Cranelift textual IR for MIR functions whose
ops fall entirely in the DIRECT subset:

  params, const (int/bool/float/None), const_ty Unit, copy, binop
  (int arithmetic + comparisons, float arithmetic + comparisons, logical
  and/or on i64), select, direct call (to other emitted functions or to
  declared externals), drop (no-op comment), match_fail (trap), and the
  br / br_if / ret terminators.

SUSPENDING functions (the ``MirFunc.suspending`` flag, or any function
containing perform/resume/handle_scope ops) are lowered selectively to CPS
at the CLIF level when their other ops stay inside the direct subset above
restricted to i64 (no floats) and their calls avoid runtime builtins and
other suspending functions.  For each such function we emit, using the
frame layouts from ``cps_frames.compute_frame_layouts``:

  * a ``; frame %f: size=NN, [0]=state:i64, [8]=result:i64, ...`` comment
    table describing the defunctionalized frame;
  * ``%run_<f>(i64) -> i64`` taking the frame pointer: its entry loads the
    state discriminant and ``br_table``s to one block per resume point
    (state 0 = function entry, reading params from the frame; state k =
    after the k-th perform, restoring that point's live variables and the
    frame's result slot).  Each perform is a PARK site: the segment stores
    the live variables and the next state into the frame, calls the runtime
    (``enqueue(frame)``, or ``sched_read(fd, buf, len, k, frame)`` with the
    resume shim's address for ops named ``read``), and returns 0 (parked).
    Final segments return the function's value.
  * one ``%resume_<f>_<k>(i64, i64) -> i64`` shim per resume point that
    stores the resumed value into the frame's result slot and tail-calls
    (``return_call``) ``%run_<f>``.

Honest scope note: general effect dispatch — locating the matching handler,
single-shot continuation bookkeeping, handler aborts — stays in the MIR
interpreter.  The CLIF-level CPS here is the roadmap's mechanical park/wake
shape (state machine + frame traffic + scheduler calls), not full handler
dispatch; the performed effect/op names appear only as comments at the park
sites.

Everything else — resume/handle_scope ops themselves, closures
(make_closure / indirect calls), structs, variants, vectors and string ops
(including calls into the vec/string/trait runtime builtins), f64 values
inside suspending functions — is emitted as a clearly marked, comment-only
placeholder carrying the reasons plus an extern-style declaration, never as
silently wrong code.

Type model (documented convention):
  * ints, bools and unit are all ``i64``; unit is the constant 0.
  * floats are ``f64``.
  * icmp/fcmp produce a narrow flag value (i8 in modern CLIF); every
    comparison result is immediately ``uextend``-ed to i64 so that booleans
    are uniformly i64 values.  ``brif`` takes the i64 condition directly
    (non-zero = taken), which CLIF permits for any integer type.
  * float constants are printed in hexadecimal float notation
    (Python ``float.hex()``), the form Cranelift's reader parses.

Per-function signatures are inferred: every value defaults to i64 and is
promoted to f64 by a monotone fixpoint over float constants, copies, binops,
selects, calls to known-float externals (sqrt/sin/cos) and calls to other
emitted functions (whose signatures are themselves iterated to a module-wide
fixpoint).  If promotion produces an inconsistency (e.g. an f64 flowing into
a comparison-result slot or across a call boundary with an i64 signature)
the function is demoted to a placeholder rather than emitted wrong.

MIR is not SSA: lowering re-assigns result variables (if/match result slots,
loop counters).  Rather than threading block arguments, every MIR variable
that is multiply-assigned — or single-assigned outside the entry block and
used in another block — gets an explicit 8-byte stack slot with
stack_store/stack_load; the remaining variables map 1:1 to SSA values.
This is the "stack slots" option: simpler than SSA-argument threading and
always correct with respect to dominance.

Public API: ``emit_clif(mir_funcs) -> str`` (used by pipeline.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .hir import BUILTIN_CALL_PREFIX
from .mir import MirFunc
from .cps_frames import (RESULT_OFFSET, STATE_OFFSET, compute_frame_layouts,
                         is_suspending, var_offset)

I64 = "i64"
F64 = "f64"

# Binop tables ---------------------------------------------------------------

_ARITH_INT = {"+": "iadd", "-": "isub", "*": "imul", "/": "sdiv", "%": "srem"}
_ARITH_FLT = {"+": "fadd", "-": "fsub", "*": "fmul", "/": "fdiv"}
_CMP_INT = {"==": "eq", "!=": "ne", "<": "slt", "<=": "sle", ">": "sgt", ">=": "sge"}
_CMP_FLT = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}
_LOGIC = {"&&": "band", "||": "bor", "and": "band", "or": "bor"}

_SUPPORTED_BINOPS = set(_ARITH_INT) | set(_CMP_INT) | set(_LOGIC)

# Externals with known float signatures.
_MATH_EXTERNS: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "sqrt": ((F64,), F64),
    "sin": ((F64,), F64),
    "cos": ((F64,), F64),
}

# Callees implemented by the vec/string/trait runtime: calling them means the
# function manipulates runtime objects (vectors, strings, trait dictionaries)
# that the direct subset cannot represent as i64/f64.
_RUNTIME_PREFIXES = ("__vec_", "__index_", "__slice_", "__range", "__trait$", "__static$")
_RUNTIME_NAMES = {"to_string", "int_to_str", "type_of", "len", "push", "pop", "Vec.new"}

_HEADER = (
    "; CLIF emitted by metaxu codegen_clif (direct subset + selective CPS)\n"
    "; conventions: ints/bools/unit -> i64 (unit = 0); floats -> f64;\n"
    ";   icmp/fcmp results are uextend-ed to i64; brif takes the i64 condition.\n"
    "; suspending functions in the i64 direct subset are CPS-lowered:\n"
    ";   %run_<f>(frame) br_tables on frame.state, parks at performs via\n"
    ";   enqueue/sched_read, and %resume_<f>_<k> shims deliver resumed values;\n"
    ";   effect dispatch itself stays in the interpreter (mechanical shape only).\n"
    "; functions outside these subsets appear as comment-only placeholders."
)


def _is_runtime_builtin(name: str) -> bool:
    return name in _RUNTIME_NAMES or any(name.startswith(p) for p in _RUNTIME_PREFIXES)


def _sanitize(name: str) -> str:
    """Restrict a symbol to CLIF external-name characters [A-Za-z0-9_]."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def _fmt_f64(x: float) -> str:
    if x != x:
        return "NaN"
    if x == float("inf"):
        return "Inf"
    if x == float("-inf"):
        return "-Inf"
    return float(x).hex()


class _Unsupported(Exception):
    """Raised during emission when a function turns out non-direct."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# Per-function analysis
# ---------------------------------------------------------------------------

@dataclass
class _Info:
    f: MirFunc
    params: Tuple[str, ...] = ()
    reasons: List[str] = field(default_factory=list)
    def_count: Dict[str, int] = field(default_factory=dict)
    def_block: Dict[str, int] = field(default_factory=dict)
    use_blocks: Dict[str, Set[int]] = field(default_factory=dict)
    ret_vars: List[str] = field(default_factory=list)
    calls: List[Tuple[int, str, str, Tuple[str, ...]]] = field(default_factory=list)
    slots: List[str] = field(default_factory=list)
    suspending: bool = False

    def add_reason(self, r: str) -> None:
        if r not in self.reasons:
            self.reasons.append(r)


def _analyze(f: MirFunc, module_names: Set[str]) -> _Info:
    info = _Info(f=f)
    try:
        _analyze_inner(info, module_names)
    except Exception as exc:  # defensive: malformed MIR must never crash codegen
        info.add_reason(f"analysis error: {type(exc).__name__}: {exc}")
    return info


def _analyze_inner(info: _Info, module_names: Set[str]) -> None:
    f = info.f
    if not f.blocks:
        info.add_reason("function has no blocks")
        return
    if f.blocks[0].ops and f.blocks[0].ops[0][0] == "params":
        info.params = tuple(f.blocks[0].ops[0][1])
    info.suspending = is_suspending(f)

    for p in info.params:
        info.def_count[p] = 1
        info.def_block[p] = 0

    def add_use(name: str, bi: int) -> None:
        info.use_blocks.setdefault(name, set()).add(bi)

    def add_def(name: str, bi: int) -> None:
        info.def_count[name] = info.def_count.get(name, 0) + 1
        info.def_block.setdefault(name, bi)

    n_blocks = len(f.blocks)
    for bi, b in enumerate(f.blocks):
        for op in b.ops:
            kind = op[0]
            if kind == "params":
                if bi != 0:
                    info.add_reason("params op outside entry block")
                continue
            if kind == "perform":
                # ("perform", dst, effect, op, args, resume_bb, dst) — a CPS
                # suspension point, not a reason by itself: the driver routes
                # suspending functions to CPS emission (or a placeholder if
                # the rest of the function is outside the CPS subset).
                if len(op) == 7:
                    for a in op[4]:
                        add_use(a, bi)
                    add_def(op[1], bi)
                    if not (isinstance(op[5], int) and 0 <= op[5] < n_blocks):
                        info.add_reason(f"perform resume block bb{op[5]} out of range")
                    if op is not b.ops[-1]:
                        info.add_reason("perform is not the last op of its block")
                else:
                    info.add_reason(f"malformed perform op (arity {len(op)})")
                continue
            if kind == "drop":
                continue  # no-op for i64/f64 values; emitted as a comment
            if kind == "match_fail":
                continue  # emitted as a trap
            if kind != "let" or len(op) != 4:
                info.add_reason(f"unsupported op {kind!r}")
                continue
            _, dst, rhs, args = op
            rk = rhs[0]
            if rk == "const":
                v = rhs[1]
                if isinstance(v, str):
                    info.add_reason("uses strings (string constant)")
                elif v is not None and not isinstance(v, (bool, int, float)):
                    info.add_reason(f"unsupported constant type {type(v).__name__}")
                add_def(dst, bi)
            elif rk == "const_ty":
                if rhs[1] != "Unit":
                    info.add_reason("opaque typed constant (lowering fallback)")
                add_def(dst, bi)
            elif rk == "copy":
                add_use(args[0], bi)
                add_def(dst, bi)
            elif rk == "binop":
                if rhs[1] not in _SUPPORTED_BINOPS:
                    info.add_reason(f"unsupported binop {rhs[1]!r}")
                for a in args:
                    add_use(a, bi)
                add_def(dst, bi)
            elif rk == "select":
                for a in args:
                    add_use(a, bi)
                add_def(dst, bi)
            elif rk == "call":
                callee = rhs[1]
                info.calls.append((bi, dst, callee, tuple(args)))
                for a in args:
                    add_use(a, bi)
                add_def(dst, bi)
            elif rk in ("resume", "handle_scope"):
                info.add_reason(f"uses effects ({rk})")
            elif rk == "make_closure":
                info.add_reason("uses closures (make_closure)")
            elif rk in ("alloc_struct", "field_get", "field_set"):
                info.add_reason(f"uses structs ({rk})")
            elif rk in ("make_variant", "variant_tag", "variant_field"):
                info.add_reason(f"uses variants ({rk})")
            else:
                info.add_reason(f"unsupported op {rk!r}")

        t = b.term
        if t[0] == "br":
            if not (0 <= t[1] < n_blocks):
                info.add_reason(f"branch target bb{t[1]} out of range")
        elif t[0] == "br_if":
            add_use(t[1], bi)
            for tgt in (t[2], t[3]):
                if not (0 <= tgt < n_blocks):
                    info.add_reason(f"branch target bb{tgt} out of range")
        elif t[0] == "ret":
            add_use(t[1], bi)
            info.ret_vars.append(t[1])
        elif t[0] == "unreachable":
            pass  # emitted as trap
        else:
            info.add_reason(f"unsupported terminator {t[0]!r}")

    # Calls: only direct calls to emitted functions or plain externals.
    for (_bi, _dst, callee, _args) in info.calls:
        if callee in info.def_count:
            info.add_reason(f"indirect call through local {callee!r} (closures)")
        elif _is_runtime_builtin(_clif_callee(callee)):
            info.add_reason(
                f"calls runtime builtin {_clif_callee(callee)!r} "
                "(vec/string/trait)")

    # Every used name must be defined somewhere in the function.  Names with
    # no local definition are captured-environment references (handler/lambda
    # sub-functions receive enclosing locals via the effect/closure runtime).
    for name in info.use_blocks:
        if name not in info.def_count:
            info.add_reason(
                f"references {name!r} with no local definition (captured environment)")

    # Storage classes: SSA value vs explicit stack slot.
    slots: List[str] = []
    for name, cnt in info.def_count.items():
        if cnt > 1:
            slots.append(name)
            continue
        db = info.def_block.get(name, 0)
        uses = info.use_blocks.get(name, set())
        if db != 0 and any(u != db for u in uses):
            # Defined in a non-entry block and used elsewhere: dominance is
            # not guaranteed, so spill to a stack slot (always correct).
            slots.append(name)
    info.slots = sorted(slots)


# ---------------------------------------------------------------------------
# Type inference (i64 by default, monotone promotion to f64)
# ---------------------------------------------------------------------------

def _clif_callee(callee: str) -> str:
    """External symbol name for a call op's callee.

    A method-position builtin call carries the ``__builtin$`` marker the
    front end adds so plain calls can prefer a same-named user function
    (NAME PRECEDENCE, docs/name_precedence.md).  CLIF has no builtin
    runtime -- these are plain externals -- so the marker is stripped for
    the symbol name, and the callee never resolves to a module function
    that happens to share the bare name."""
    if callee.startswith(BUILTIN_CALL_PREFIX):
        return callee[len(BUILTIN_CALL_PREFIX):]
    return callee


def _callee_sig(callee: str, sigs: Dict[str, Tuple[Tuple[str, ...], str]],
                ) -> Optional[Tuple[Tuple[str, ...], str]]:
    if callee.startswith(BUILTIN_CALL_PREFIX):
        # Builtin marker: an external, never the module function that may
        # share the bare name.
        return _MATH_EXTERNS.get(_clif_callee(callee))
    if callee in sigs:
        return sigs[callee]
    return _MATH_EXTERNS.get(callee)


def _infer_f64(info: _Info, sigs: Dict[str, Tuple[Tuple[str, ...], str]]) -> Set[str]:
    f64s: Set[str] = set()

    def mark(name: str) -> bool:
        if name in f64s:
            return False
        f64s.add(name)
        return True

    def unify(names: Sequence[str]) -> bool:
        if any(n in f64s for n in names):
            changed = False
            for n in names:
                changed = mark(n) or changed
            return changed
        return False

    changed = True
    while changed:
        changed = False
        for b in info.f.blocks:
            for op in b.ops:
                if op[0] != "let":
                    continue
                _, dst, rhs, args = op
                rk = rhs[0]
                if rk == "const":
                    if isinstance(rhs[1], float) and not isinstance(rhs[1], bool):
                        changed = mark(dst) or changed
                elif rk == "copy":
                    changed = unify((dst, args[0])) or changed
                elif rk == "binop":
                    o = rhs[1]
                    if o in _CMP_INT:
                        changed = unify(args) or changed  # dst stays i64
                    elif o in _LOGIC:
                        pass  # i64-only
                    else:
                        changed = unify((dst, *args)) or changed
                elif rk == "select":
                    if len(args) == 3:
                        changed = unify((dst, args[1], args[2])) or changed
                elif rk == "call":
                    sig = _callee_sig(rhs[1], sigs)
                    if sig is not None:
                        ptys, rty = sig
                        if len(ptys) == len(args):
                            for a, pt in zip(args, ptys):
                                if pt == F64:
                                    changed = mark(a) or changed
                            if rty == F64:
                                changed = mark(dst) or changed
        if len(info.ret_vars) > 1:
            changed = unify(info.ret_vars) or changed
    return f64s


def _sig_of(info: _Info, f64s: Set[str]) -> Tuple[Tuple[str, ...], str]:
    ptys = tuple(F64 if p in f64s else I64 for p in info.params)
    rty = F64 if any(r in f64s for r in info.ret_vars) else I64
    return (ptys, rty)


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

def _check_consistency(info: _Info, f64s: Set[str],
                       sigs: Dict[str, Tuple[Tuple[str, ...], str]]) -> List[str]:
    """Detect i64/f64 conflicts that monotone promotion cannot repair."""
    probs: List[str] = []

    def ty(n: str) -> str:
        return F64 if n in f64s else I64

    for b in info.f.blocks:
        for op in b.ops:
            if op[0] != "let":
                continue
            _, dst, rhs, args = op
            rk = rhs[0]
            if rk == "binop":
                o = rhs[1]
                if o in _CMP_INT and dst in f64s:
                    probs.append(f"comparison result {dst!r} promoted to f64")
                if o in _LOGIC and (dst in f64s or any(a in f64s for a in args)):
                    probs.append(f"logical binop {o!r} applied to f64 values")
            elif rk == "select" and len(args) == 3 and args[0] in f64s:
                probs.append(f"select condition {args[0]!r} is f64")
            elif rk == "call":
                sig = _callee_sig(rhs[1], sigs)
                if sig is not None:
                    ptys, rty = sig
                    if len(ptys) != len(args):
                        probs.append(
                            f"call to {rhs[1]!r} with {len(args)} args, expects {len(ptys)}")
                    else:
                        for a, pt in zip(args, ptys):
                            if ty(a) != pt:
                                probs.append(
                                    f"call to {rhs[1]!r}: arg {a!r} is {ty(a)}, expects {pt}")
                        if ty(dst) != rty:
                            probs.append(
                                f"call to {rhs[1]!r}: result {dst!r} is {ty(dst)}, returns {rty}")
        if b.term[0] == "br_if" and b.term[1] in f64s:
            probs.append(f"br_if condition {b.term[1]!r} is f64")
    return probs


def _emit_placeholder(info: _Info, sig: Tuple[Tuple[str, ...], str]) -> str:
    sym = _sanitize(info.f.name)
    ptys, rty = sig
    lines = [f"; function %{sym}: placeholder -- unsupported for direct CLIF emission"]
    for r in info.reasons:
        lines.append(f";   reason: {r}")
    lines.append(f"; declare %{sym}({', '.join(ptys)}) -> {rty}")
    return "\n".join(lines)


def _emit_direct(info: _Info, f64s: Set[str],
                 sigs: Dict[str, Tuple[Tuple[str, ...], str]]) -> str:
    """Emit one direct function. Raises _Unsupported on internal surprises."""
    f = info.f

    def ty(n: str) -> str:
        return F64 if n in f64s else I64

    counter = 0

    def fresh() -> str:
        nonlocal counter
        v = f"v{counter}"
        counter += 1
        return v

    valmap: Dict[str, str] = {}
    slotmap: Dict[str, str] = {n: f"ss{i}" for i, n in enumerate(info.slots)}

    # Function declarations for calls: (symbol, sig) -> fnK
    fn_map: Dict[Tuple[str, Tuple[Tuple[str, ...], str]], str] = {}

    def declare(callee: str, sig: Tuple[Tuple[str, ...], str]) -> str:
        key = (_sanitize(callee), sig)
        if key not in fn_map:
            fn_map[key] = f"fn{len(fn_map)}"
        return fn_map[key]

    def use(name: str, lines: List[str]) -> str:
        if name in slotmap:
            v = fresh()
            lines.append(f"    {v} = stack_load.{ty(name)} {slotmap[name]}")
            return v
        try:
            return valmap[name]
        except KeyError:
            raise _Unsupported(f"use of {name!r} before its definition")

    def setval(name: str, v: str, lines: List[str]) -> None:
        if name in slotmap:
            lines.append(f"    stack_store {v}, {slotmap[name]}")
        else:
            valmap[name] = v

    def emit_const(name: str, value: Any, lines: List[str]) -> None:
        v = fresh()
        if ty(name) == F64:
            if value is None or isinstance(value, bool):
                value = float(bool(value)) if isinstance(value, bool) else 0.0
            lines.append(f"    {v} = f64const {_fmt_f64(float(value))}")
        else:
            if value is None:
                iv = 0
            elif isinstance(value, bool):
                iv = int(value)
            elif isinstance(value, float):
                iv = int(value)  # unreachable in practice: floats are promoted
            else:
                iv = int(value)
            lines.append(f"    {v} = iconst.i64 {iv}")
        setval(name, v, lines)

    body: List[str] = []
    for bi, b in enumerate(f.blocks):
        lines: List[str] = []
        if bi == 0:
            pvals = []
            for p in info.params:
                pv = fresh()
                pvals.append(f"{pv}: {ty(p)}")
                if p in slotmap:
                    lines.append(f"    stack_store {pv}, {slotmap[p]}")
                else:
                    valmap[p] = pv
            args = f"({', '.join(pvals)})" if pvals else ""
            header = f"block0{args}:"
        else:
            header = f"block{bi}:"

        terminated = False
        for op in b.ops:
            kind = op[0]
            if kind == "params":
                continue
            if kind == "drop":
                lines.append(f"    ; drop {op[1]}")
                continue
            if kind == "match_fail":
                lines.append(f"    trap user0  ; match_fail: {op[1]}")
                terminated = True
                break
            # kind == "let" (analysis guarantees this)
            _, dst, rhs, opargs = op
            rk = rhs[0]
            if rk == "const":
                emit_const(dst, rhs[1], lines)
            elif rk == "const_ty":
                emit_const(dst, None, lines)  # Unit -> 0
            elif rk == "copy":
                src = use(opargs[0], lines)
                setval(dst, src, lines)
            elif rk == "binop":
                o = rhs[1]
                l = use(opargs[0], lines)
                r = use(opargs[1], lines)
                is_flt = ty(opargs[0]) == F64
                if o in _CMP_INT:  # comparison (int or float operands)
                    c = fresh()
                    if is_flt:
                        lines.append(f"    {c} = fcmp {_CMP_FLT[o]} {l}, {r}")
                    else:
                        lines.append(f"    {c} = icmp {_CMP_INT[o]} {l}, {r}")
                    v = fresh()
                    lines.append(f"    {v} = uextend.i64 {c}")
                    setval(dst, v, lines)
                elif o in _LOGIC:
                    v = fresh()
                    lines.append(f"    {v} = {_LOGIC[o]} {l}, {r}")
                    setval(dst, v, lines)
                else:
                    v = fresh()
                    mnem = _ARITH_FLT.get(o) if is_flt else _ARITH_INT.get(o)
                    if mnem is None:
                        raise _Unsupported(f"binop {o!r} on {ty(opargs[0])}")
                    lines.append(f"    {v} = {mnem} {l}, {r}")
                    setval(dst, v, lines)
            elif rk == "select":
                c = use(opargs[0], lines)
                t = use(opargs[1], lines)
                e = use(opargs[2], lines)
                v = fresh()
                lines.append(f"    {v} = select {c}, {t}, {e}")
                setval(dst, v, lines)
            elif rk == "call":
                callee = rhs[1]
                sig = _callee_sig(callee, sigs)
                if sig is None:  # plain external: signature from observed types
                    sig = (tuple(ty(a) for a in opargs), ty(dst))
                fnref = declare(_clif_callee(callee), sig)
                avals = [use(a, lines) for a in opargs]
                v = fresh()
                lines.append(f"    {v} = call {fnref}({', '.join(avals)})")
                setval(dst, v, lines)
            else:  # unreachable given analysis
                raise _Unsupported(f"op {rk!r} slipped past analysis")

        if not terminated:
            t = b.term
            if t[0] == "br":
                lines.append(f"    jump block{t[1]}")
            elif t[0] == "br_if":
                c = use(t[1], lines)
                lines.append(f"    brif {c}, block{t[2]}, block{t[3]}")
            elif t[0] == "ret":
                rv = use(t[1], lines)
                lines.append(f"    return {rv}")
            else:  # ("unreachable",) placeholder terminator
                lines.append("    trap unreachable")

        body.append(header)
        body.extend(lines)

    # Assemble: header, preamble (stack slots, sigs/fns), blocks.
    ptys, rty = _sig_of(info, f64s)
    sym = _sanitize(f.name)
    out: List[str] = [f"function %{sym}({', '.join(ptys)}) -> {rty} {{"]
    for name in info.slots:
        out.append(f"    {slotmap[name]} = explicit_slot 8  ; {name}")
    for ((csym, csig), fnref) in fn_map.items():
        k = fnref[2:]
        cptys, crty = csig
        out.append(f"    sig{k} = ({', '.join(cptys)}) -> {crty}")
        out.append(f"    {fnref} = %{csym} sig{k}")
    if len(out) > 1:
        out.append("")
    out.extend(body)
    out.append("}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Selective CPS emission for suspending functions
# ---------------------------------------------------------------------------

# Runtime ABI (src/metaxu/runtime/src/lib.rs):
#   pub extern "C" fn enqueue(frame: *mut u8)
#   pub extern "C" fn sched_read(fd: i64, buf: *mut u8, len: usize,
#                                k: extern "C" fn(*mut u8, usize), frame: *mut u8)
# Pointers, usize and function pointers are all i64 words in CLIF.
_ENQUEUE_SIG: Tuple[Tuple[str, ...], Optional[str]] = ((I64,), None)
_SCHED_READ_SIG: Tuple[Tuple[str, ...], Optional[str]] = ((I64, I64, I64, I64, I64), None)


def _cps_blockers(info: _Info, suspending_names: Set[str],
                  sigs: Dict[str, Tuple[Tuple[str, ...], str]]) -> List[str]:
    """Reasons a reason-free suspending function still cannot get a CPS body."""
    reasons: List[str] = []
    f64s = _infer_f64(info, sigs)
    if f64s:
        sample = ", ".join(sorted(f64s)[:3])
        reasons.append(f"CPS subset is i64-only (f64 values: {sample})")
    for (_bi, _dst, callee, _args) in info.calls:
        if callee in suspending_names:
            reasons.append(
                f"calls suspending function {callee!r} "
                "(frame chaining across suspending calls not implemented)")
    if not f64s:
        for p in _check_consistency(info, set(), sigs):
            reasons.append(f"type conflict: {p}")
    return reasons


def _frame_comment(layout: Dict[str, Any]) -> str:
    sym = _sanitize(layout["name"])
    entries = [(STATE_OFFSET, "state"), (RESULT_OFFSET, "result")]
    entries += [(off, name) for name, off in layout["params"].items()]
    entries += [(off, name) for name, off in layout["vars"].items()]
    entries.sort()
    cells = ", ".join(f"[{off}]={name}:i64" for off, name in entries)
    return f"; frame %{sym}: size={layout['size']}, {cells}"


def _emit_cps(info: _Info, layout: Dict[str, Any],
              sigs: Dict[str, Tuple[Tuple[str, ...], str]]) -> str:
    """Emit %run_<f> + %resume_<f>_<k> shims for one suspending function.

    All values are i64.  Every MIR variable lives in an explicit stack slot
    (the function has multiple entry points, so SSA dominance cannot be
    assumed); params and live-across variables additionally have frame slots
    (see cps_frames) that are written at park sites and read back by the
    entry/resume prologues.
    """
    f = info.f
    sym = _sanitize(f.name)
    run_sym = f"run_{sym}"
    points = layout["suspend_points"]
    point_at = {(p["block"], p["op_index"]): p for p in points}
    n = len(f.blocks)
    m = len(points)

    # Block numbering inside %run_<f>:
    #   0                = dispatch (br_table on frame.state)
    #   1                = state-0 prologue (load params from the frame)
    #   2 + bi           = original MIR block bi
    #   2 + n + (k - 1)  = state-k prologue (restore live vars + result)
    #   2 + n + m        = invalid-state trap
    def mapped(bi: int) -> int:
        return 2 + bi

    def resume_block(k: int) -> int:
        return 2 + n + (k - 1)

    trap_block = 2 + n + m

    counter = 0

    def fresh() -> str:
        nonlocal counter
        v = f"v{counter}"
        counter += 1
        return v

    # Every variable gets a stack slot (uniform storage, always valid).
    all_vars = sorted(info.def_count)
    slotmap = {name: f"ss{i}" for i, name in enumerate(all_vars)}

    # Function declarations. enqueue/sched_read are always declared (the
    # scheduler ABI); resume shims and direct callees on demand.
    fn_map: Dict[Tuple[str, Tuple[Tuple[str, ...], Optional[str]]], str] = {}

    def declare(callee: str, sig: Tuple[Tuple[str, ...], Optional[str]]) -> str:
        key = (_sanitize(callee), sig)
        if key not in fn_map:
            fn_map[key] = f"fn{len(fn_map)}"
        return fn_map[key]

    enqueue_fn = declare("enqueue", _ENQUEUE_SIG)
    sched_read_fn = declare("sched_read", _SCHED_READ_SIG)

    frame = "v0"  # block0 parameter: the frame pointer

    def frame_ref(off: int) -> str:
        return frame if off == 0 else f"{frame}+{off}"

    def use(name: str, lines: List[str]) -> str:
        if name not in slotmap:
            raise _Unsupported(f"use of {name!r} with no storage")
        v = fresh()
        lines.append(f"    {v} = stack_load.i64 {slotmap[name]}")
        return v

    def setval(name: str, v: str, lines: List[str]) -> None:
        lines.append(f"    stack_store {v}, {slotmap[name]}")

    body: List[str] = []

    def add_block(bid: int, lines: List[str], comment: str = "") -> None:
        body.append(f"block{bid}:" + (f"  ; {comment}" if comment else ""))
        body.extend(lines)

    # -- dispatch -----------------------------------------------------------
    fv = fresh()  # v0: the frame-pointer block argument
    assert fv == frame
    lines: List[str] = []
    state = fresh()
    lines.append(f"    {state} = load.i64 {frame_ref(STATE_OFFSET)}  ; state")
    idx = fresh()
    lines.append(f"    {idx} = ireduce.i32 {state}")
    table = ", ".join(["block1"] + [f"block{resume_block(k)}" for k in range(1, m + 1)])
    lines.append(f"    br_table {idx}, block{trap_block}, [{table}]")
    body.append(f"block0({frame}: i64):  ; dispatch on frame.state")
    body.extend(lines)

    # -- state-0 prologue: params from frame -------------------------------
    lines = []
    for p, off in layout["params"].items():
        v = fresh()
        lines.append(f"    {v} = load.i64 {frame_ref(off)}  ; param {p}")
        setval(p, v, lines)
    lines.append(f"    jump block{mapped(0)}")
    add_block(1, lines, "state 0: function entry")

    # -- original blocks ----------------------------------------------------
    for bi, b in enumerate(f.blocks):
        lines = []
        parked = False
        for oi, op in enumerate(b.ops):
            kind = op[0]
            if kind == "params":
                continue
            if kind == "drop":
                lines.append(f"    ; drop {op[1]}")
                continue
            if kind == "match_fail":
                lines.append(f"    trap user0  ; match_fail: {op[1]}")
                parked = True  # block is terminated
                break
            if kind == "perform":
                point = point_at.get((bi, oi))
                if point is None:
                    raise _Unsupported("perform op missing from frame layout")
                k = point["state"]
                eff_desc = f"{point['effect']}.{point['op']}({', '.join(point['args'])})"
                lines.append(f"    ; park: perform {eff_desc} -> suspend point {k}")
                for lv in point["live"]:
                    v = use(lv, lines)
                    lines.append(f"    store {v}, {frame_ref(var_offset(layout, lv))}  ; save {lv}")
                sv = fresh()
                lines.append(f"    {sv} = iconst.i64 {k}")
                lines.append(f"    store {sv}, {frame_ref(STATE_OFFSET)}  ; state = {k}")
                if point["op"] == "read":
                    # sched_read(fd, buf, len, k, frame): wire the perform's
                    # args positionally (missing ones are 0) and pass the
                    # resume shim as the continuation k.
                    shim_fn = declare(f"resume_{sym}_{k}", ((I64, I64), I64))
                    kv = fresh()
                    lines.append(f"    {kv} = func_addr.i64 {shim_fn}")
                    argv: List[str] = []
                    for ai in range(3):
                        if ai < len(point["args"]):
                            argv.append(use(point["args"][ai], lines))
                        else:
                            z = fresh()
                            lines.append(f"    {z} = iconst.i64 0")
                            argv.append(z)
                    lines.append(
                        f"    call {sched_read_fn}({argv[0]}, {argv[1]}, {argv[2]}, {kv}, {frame})")
                else:
                    lines.append(f"    call {enqueue_fn}({frame})")
                z = fresh()
                lines.append(f"    {z} = iconst.i64 0")
                lines.append(f"    return {z}  ; parked")
                parked = True
                break
            # kind == "let" (analysis guarantees this)
            _, dst, rhs, opargs = op
            rk = rhs[0]
            if rk in ("const", "const_ty"):
                value = rhs[1] if rk == "const" else None
                if value is None:
                    iv = 0
                elif isinstance(value, bool):
                    iv = int(value)
                else:
                    iv = int(value)
                v = fresh()
                lines.append(f"    {v} = iconst.i64 {iv}")
                setval(dst, v, lines)
            elif rk == "copy":
                v = use(opargs[0], lines)
                setval(dst, v, lines)
            elif rk == "binop":
                o = rhs[1]
                l = use(opargs[0], lines)
                r = use(opargs[1], lines)
                if o in _CMP_INT:
                    c = fresh()
                    lines.append(f"    {c} = icmp {_CMP_INT[o]} {l}, {r}")
                    v = fresh()
                    lines.append(f"    {v} = uextend.i64 {c}")
                    setval(dst, v, lines)
                elif o in _LOGIC:
                    v = fresh()
                    lines.append(f"    {v} = {_LOGIC[o]} {l}, {r}")
                    setval(dst, v, lines)
                else:
                    mnem = _ARITH_INT.get(o)
                    if mnem is None:
                        raise _Unsupported(f"binop {o!r} in CPS function")
                    v = fresh()
                    lines.append(f"    {v} = {mnem} {l}, {r}")
                    setval(dst, v, lines)
            elif rk == "select":
                c = use(opargs[0], lines)
                t = use(opargs[1], lines)
                e = use(opargs[2], lines)
                v = fresh()
                lines.append(f"    {v} = select {c}, {t}, {e}")
                setval(dst, v, lines)
            elif rk == "call":
                callee = rhs[1]
                csig = _callee_sig(callee, sigs)
                sig: Tuple[Tuple[str, ...], Optional[str]]
                if csig is None:
                    sig = (tuple(I64 for _ in opargs), I64)
                else:
                    sig = csig
                fnref = declare(_clif_callee(callee), sig)
                avals = [use(a, lines) for a in opargs]
                v = fresh()
                lines.append(f"    {v} = call {fnref}({', '.join(avals)})")
                setval(dst, v, lines)
            else:
                raise _Unsupported(f"op {rk!r} in CPS function")

        if not parked:
            t = b.term
            if t[0] == "br":
                lines.append(f"    jump block{mapped(t[1])}")
            elif t[0] == "br_if":
                c = use(t[1], lines)
                lines.append(f"    brif {c}, block{mapped(t[2])}, block{mapped(t[3])}")
            elif t[0] == "ret":
                rv = use(t[1], lines)
                lines.append(f"    return {rv}")
            else:
                lines.append("    trap unreachable")
        add_block(mapped(bi), lines, f"mir bb{bi}")

    # -- resume prologues ---------------------------------------------------
    for point in points:
        k = point["state"]
        lines = []
        for lv in point["live"]:
            v = fresh()
            lines.append(f"    {v} = load.i64 {frame_ref(var_offset(layout, lv))}  ; restore {lv}")
            setval(lv, v, lines)
        v = fresh()
        lines.append(f"    {v} = load.i64 {frame_ref(RESULT_OFFSET)}  ; resumed value")
        setval(point["dst"], v, lines)
        lines.append(f"    jump block{mapped(point['resume_block'])}")
        add_block(resume_block(k),
                  lines, f"state {k}: resume after {point['effect']}.{point['op']}")

    # -- invalid-state trap -------------------------------------------------
    add_block(trap_block, ["    trap user1  ; invalid frame state"])

    # -- assemble %run_<f> --------------------------------------------------
    out: List[str] = [_frame_comment(layout), f"function %{run_sym}(i64) -> i64 {{"]
    for name in all_vars:
        out.append(f"    {slotmap[name]} = explicit_slot 8  ; {name}")
    for ((csym, csig), fnref) in fn_map.items():
        kk = fnref[2:]
        cptys, crty = csig
        arrow = f" -> {crty}" if crty is not None else ""
        out.append(f"    sig{kk} = ({', '.join(cptys)}){arrow}")
        out.append(f"    {fnref} = %{csym} sig{kk}")
    out.append("")
    out.extend(body)
    out.append("}")
    chunks = ["\n".join(out)]

    # -- %resume_<f>_<k> shims ---------------------------------------------
    for point in points:
        k = point["state"]
        shim = [
            f"function %resume_{sym}_{k}(i64, i64) -> i64 {{",
            "    sig0 = (i64) -> i64",
            f"    fn0 = %{run_sym} sig0",
            "",
            "block0(v0: i64, v1: i64):  ; (frame, resumed value)",
            f"    store v1, v0+{RESULT_OFFSET}  ; frame.result = value",
            f"    ; frame.state is already {k} (stored at the park site)",
            "    return_call fn0(v0)",
            "}",
        ]
        chunks.append("\n".join(shim))
    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Module driver
# ---------------------------------------------------------------------------

def emit_clif(funcs: Sequence[MirFunc]) -> str:
    """Emit CLIF text for a MIR module.

    Direct functions get full bodies; suspending functions in the CPS
    subset get %run_/%resume_ state machines (see module docstring);
    everything else gets a comment-only placeholder with its reasons and
    an extern-style declaration.
    """
    module_names = {f.name for f in funcs}
    layouts = compute_frame_layouts(funcs)
    suspending_names = set(layouts)
    infos = [_analyze(f, module_names) for f in funcs]

    # Module-wide signature fixpoint: functions may call later functions whose
    # float-ness is only known after their own inference round.
    sigs: Dict[str, Tuple[Tuple[str, ...], str]] = {}
    f64_sets: Dict[str, Set[str]] = {}
    for info in infos:
        sigs[info.f.name] = (tuple(I64 for _ in info.params), I64)
    for _round in range(6):
        changed = False
        for info in infos:
            if info.reasons or info.suspending:
                continue  # suspending functions keep their i64 default sig
            f64s = _infer_f64(info, sigs)
            f64_sets[info.f.name] = f64s
            sig = _sig_of(info, f64s)
            if sigs[info.f.name] != sig:
                sigs[info.f.name] = sig
                changed = True
        if not changed:
            break

    chunks: List[str] = [_HEADER]
    for info in infos:
        if info.suspending:
            cps_reasons = list(info.reasons)
            if not cps_reasons:
                cps_reasons = _cps_blockers(info, suspending_names, sigs)
            if not cps_reasons:
                try:
                    chunks.append(_emit_cps(info, layouts[info.f.name], sigs))
                    continue
                except _Unsupported as exc:
                    cps_reasons = [exc.reason]
                except Exception as exc:  # never crash the pipeline
                    cps_reasons = [f"emission error: {type(exc).__name__}: {exc}"]
            info.add_reason("suspending function outside the CPS-emittable subset")
            for r in cps_reasons:
                info.add_reason(r)
            chunks.append(_emit_placeholder(info, sigs[info.f.name]))
            continue
        if not info.reasons:
            f64s = f64_sets.get(info.f.name, set())
            probs = _check_consistency(info, f64s, sigs)
            if probs:
                for p in probs:
                    info.add_reason(f"type conflict: {p}")
            else:
                try:
                    chunks.append(_emit_direct(info, f64s, sigs))
                    continue
                except _Unsupported as exc:
                    info.add_reason(exc.reason)
                except Exception as exc:  # never crash the pipeline
                    info.add_reason(f"emission error: {type(exc).__name__}: {exc}")
        chunks.append(_emit_placeholder(info, sigs[info.f.name]))
    return "\n\n".join(chunks)
