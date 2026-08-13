"""Cranelift IR (CLIF) text emitter for direct (non-suspending) functions.

Emits real, structurally valid Cranelift textual IR for MIR functions whose
ops fall entirely in the DIRECT subset:

  params, const (int/bool/float/None), const_ty Unit, copy, binop
  (int arithmetic + comparisons, float arithmetic + comparisons, logical
  and/or on i64), select, direct call (to other emitted functions or to
  declared externals), drop (no-op comment), match_fail (trap), and the
  br / br_if / ret terminators.

Functions outside that subset — effects (perform/resume/handle_scope),
closures (make_closure / indirect calls), structs, variants, vectors and
string ops (including calls into the vec/string/trait runtime builtins),
or suspending functions awaiting CPS lowering — are emitted as a clearly
marked, comment-only placeholder carrying the reasons plus an extern-style
declaration, never as silently wrong code.

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

from .mir import MirFunc

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
    "; CLIF emitted by metaxu codegen_clif (direct subset)\n"
    "; conventions: ints/bools/unit -> i64 (unit = 0); floats -> f64;\n"
    ";   icmp/fcmp results are uextend-ed to i64; brif takes the i64 condition.\n"
    "; functions outside the direct subset appear as comment-only placeholders."
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
    if f.suspending:
        info.add_reason("suspending function (CPS lowering pending)")

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
                info.add_reason("uses effects (perform)")
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
        elif _is_runtime_builtin(callee):
            info.add_reason(f"calls runtime builtin {callee!r} (vec/string/trait)")

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

def _callee_sig(callee: str, sigs: Dict[str, Tuple[Tuple[str, ...], str]],
                ) -> Optional[Tuple[Tuple[str, ...], str]]:
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
                fnref = declare(callee, sig)
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
# Module driver
# ---------------------------------------------------------------------------

def emit_clif(funcs: Sequence[MirFunc]) -> str:
    """Emit CLIF text for a MIR module.

    Direct functions get full bodies; everything else gets a comment-only
    placeholder with its reasons and an extern-style declaration.
    """
    module_names = {f.name for f in funcs}
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
            if info.reasons:
                continue
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
