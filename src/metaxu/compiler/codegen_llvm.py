"""LLVM IR text emitter for direct (non-suspending) MIR functions.

Emits a single, self-contained LLVM module (textual IR, compilable with
``clang x.ll``) for MIR functions whose ops fall entirely in the DIRECT
subset:

  params, const (int/bool/float/string/None), const_ty Unit, copy, binop
  (int arithmetic + comparisons, float arithmetic + comparisons, logical
  and/or), select, direct calls (to other emitted functions or the small
  builtin set below), drop (no-op comment), match_fail (call @abort +
  unreachable), local struct alloc/field ops, and the br / br_if / ret /
  unreachable terminators.

Increment 3 adds enum variants (make_variant / variant_tag / variant_field
as tagged unions) and stack-environment closures (make_closure + direct
locally-bound closure calls); see the dedicated sections below.

Everything else — suspending functions (perform/resume/handle_scope: the
CPS lowering lives in codegen_clif for now), try_scope, escaping closures,
vector/string runtime builtins — is emitted as a clearly marked,
comment-only placeholder carrying the reasons, never as silently wrong
code.  Functions that call a placeholder function are themselves demoted
(the module must link), with an explicit reason.

Type model (documented conventions):
  * ints, bools and unit are all ``i64``; unit is the constant 0.
  * floats are ``double`` (printed as IEEE-754 bit patterns, ``0x...``).
  * strings are ``ptr`` values pointing at private unnamed_addr constant
    NUL-terminated byte arrays; they are immutable and never freed.
  * icmp/fcmp produce i1, immediately ``zext``-ed to i64 so booleans are
    uniformly i64.  ``br_if`` compares its i64 condition against 0.
  * logical &&/|| normalize both operands with ``icmp ne 0`` before
    and/or (truthiness semantics, matching the MIR interpreter, not
    bitwise-and like a naive lowering).
  * int / and % use ``sdiv``/``srem`` (C truncating semantics).  The MIR
    interpreter uses Python floor semantics; these agree for non-negative
    operands.  Division by zero is UB natively (the interpreter raises).
  * LOCAL (default / @local) structs: a named ``%struct.T`` per struct
    type, one entry-block ``alloca`` per struct-typed MIR variable,
    ``getelementptr`` + load/store for fields.  MIR struct ops have value
    semantics (``field_set`` yields an updated copy), so every def of a
    struct variable stores a whole aggregate into that variable's own
    storage — no aliasing, and SROA/mem2reg scalarize it at -O2.  Zero
    memory management for locals: the frame is the allocation.
  * @GLOBAL structs live on the heap: a struct variable defined by an
    ``alloc_struct`` with locality "global" gets its storage from an
    entry-block ``call ptr @malloc(i64 <8 * nfields>)`` instead of an
    alloca (every scalar field kind — i64/double/ptr — is 8 bytes, so the
    size and the GEP layout are exact), field access GEPs the heap
    pointer, and every ``ret`` path frees the block with
    ``call void @free(ptr ...)``.  Freeing at function exit is provably
    sound here because MIR struct values have pure value semantics: every
    cross-frame transfer below (parameter, return) moves the *aggregate*
    by copy, never the storage pointer, so a callee's heap block can never
    be reached after the callee returns.  Should a future op let a raw
    struct pointer escape, that op must demote or suppress the free: a
    leak is safe, a dangling pointer is not (unproven lifetimes leak by
    design in this increment).  abort()/unreachable paths do not free
    (the process is dying).  A variable whose defs mix @local and @global
    allocations is uniformly heap-backed (storage location is unobservable
    under value semantics; the conservative cost is one malloc+free).
    This agrees with borrow_analysis.plan_drops, whose only drop point
    today is function exit (drop_at_end -> MIR ``drop`` ops); the frees do
    not depend on its needs_drop heuristic, only on the non-escape
    guarantee above.
  * struct values CROSS CALL BOUNDARIES by pointer, preserving MIR value
    semantics at both edges:
      - struct parameters are passed as ``ptr`` and the callee immediately
        copies the aggregate into its own storage in the entry prelude
        (byval-copy).  A later borrow-informed increment can elide that
        copy for @const/read-only params once the borrow checker's results
        are threaded into codegen.
      - struct returns are sret-style (the ONE convention used
        everywhere): the caller passes its result variable's storage as a
        leading ``ptr %agg.ret`` argument, the callee copies the returned
        aggregate into it and returns ``void``.  Small-struct returns as
        first-class LLVM aggregates were considered and rejected to keep
        one uniform path.  No ABI ``sret`` attribute is needed: all such
        calls are module-internal.
    Nested struct fields (a struct kind inside a struct field cell) are
    still demoted — a later increment.
  * ``print``/``println`` of a single value routes by operand type to
    @metaxu_print_i64 / @metaxu_print_f64 / @metaxu_print_str, small
    helpers defined in this module on top of a declared @printf
    ("%lld\n" / "%g\n" / "%s\n").  Multi-argument print calls one @printf
    with a per-call format string joining the per-kind directives with
    single spaces ("%lld %s\n" etc.), matching the interpreter's
    ``print(*args)`` (sep=" ").  Note: float and bool formatting can
    differ from the Python interpreter's str() ("%g" vs repr; bools are
    kind-erased to i64 and print as 1/0, not "True"/"False"); differential
    tests should print ints/strings or compare via comparisons.
  * every metaxu function symbol is prefixed ``mx_`` (and sanitized to
    [A-Za-z0-9_]) so user functions named main/printf/abs cannot collide
    with libc; the native entry wrapper lives in llvm_run.py.

ENUM VARIANTS (increment 3) are tagged unions:
  * each enum E gets ``%enum.E = type { i64, [N x i64] }`` — an integer tag
    plus payload slots sized to the LARGEST variant of E (every scalar
    field kind — i64/double/ptr — is 8 bytes in this backend, so [N x i64]
    is exact storage; slot loads/stores use the slot's inferred scalar
    type through the opaque payload pointer).  ``make_variant`` with an
    empty enum name (front-end gap) uses the shared ``%enum.anon`` type.
  * variant names map to dense integer tags at emission time via ONE
    module-wide table: every variant name occurring in a ``make_variant``
    op or a compiled-pattern tag comparison is collected, sorted, and
    numbered from 0.  Equal names get equal tags and distinct names get
    distinct tags, which preserves the interpreter's string-equality
    semantics for tag tests (cross-enum tag comparisons cannot occur in
    well-typed MIR).  The mapping is documented in a module comment.
  * compiled match arms compare ``variant_tag`` results against CONST
    STRING variant names (see lower_hir_to_mir.compile_pattern).  A const
    string whose every use is an ==/!= against a variant_tag result is a
    "tag literal" and is emitted as its integer tag — the string never
    reaches native code.  A tag-vs-string comparison that does not fit
    that shape demotes via kind conflict rather than guessing.
  * payload slot kinds are unified per (enum, slot index) across ALL
    variants, and deliberately one-way (stores accumulate into the cell,
    reads take the cell's kind): unlike function params, the type system
    allows two variants of one enum to put genuinely different types in
    the same slot, so a store whose final kind differs from the cell kind
    demotes ("heterogeneous payload slot") instead of coercing.
  * variant values are aggregates with the same value semantics as
    structs: entry-block alloca per variant-kinded variable, aggregate
    copies, ptr + callee byval-copy across calls, sret-style returns.
    There is no @global form (make_variant carries no locality); a payload
    slot holding another aggregate (recursive enums like a linked list, or
    struct payloads) demotes — a heap-boxed-payload increment lifts that.

CLOSURES (increment 3) are fn-pointer + stack-environment pairs:
  * a closure value is ``%mx.closure = type { ptr, ptr }`` — the lambda's
    function pointer and a pointer to an environment struct.  Each lambda
    L with captures gets ``%env.L = type { ... }`` holding its captured
    values IN CAPTURE-LIST ORDER (kinds inferred like struct fields;
    aggregate captures are copied in whole — capturing another closure
    demotes).
  * ``make_closure`` gets one entry-block env alloca per op site, stores
    the captured values (eager, by value — matching the interpreter) and
    then the {fn, env} pair into the closure variable's own pair alloca.
    A make_closure inside a CFG cycle demotes: re-executing the site would
    overwrite the shared env storage while previously created pair copies
    may still alias it.
  * the lambda is emitted with a leading ``ptr %cl.env`` parameter (after
    ``%agg.ret`` when sret) and re-loads every capture from the env struct
    in its prelude; its other free names still demote.
  * a call whose callee is a local variable of closure kind loads fn+env
    from the pair and calls ``fn(env, args...)`` — typed with the lambda's
    signature, which kind inference pins because closure kinds
    (``closure:L``) propagate like any other kind; two different lambdas
    reaching one call site is a kind conflict and demotes.
  * closures may be passed DOWN as call arguments (ptr + byval pair copy;
    the env outlives the callee because the creating frame is still
    live).  Everything that could make the env dangle demotes honestly:
    returning a closure, storing one in a struct field or variant payload,
    capturing one in another closure.  A heap-env increment lifts these.

Per-function value kinds (i64 / f64 / str / struct:T / enum:E /
closure:L) are inferred exactly
in the spirit of codegen_clif's i64->f64 promotion: every value defaults
to i64 and is promoted by a monotone fixpoint (module-wide over function
signatures and struct field kinds).  Irreconcilable kinds ("conflict")
demote the function rather than emitting wrong code.

MIR is not SSA: lowering re-assigns result variables (if/match result
slots, loop counters).  Mirroring codegen_clif's classification, every
non-struct MIR variable that is multiply-assigned — or defined outside the
entry block and used in another block — gets an entry-block ``alloca``
with load/store around uses; single-assignment values (and everything
defined in the entry block, which dominates all blocks) map to SSA
registers.  mem2reg removes the allocas at -O2.

Public API: ``emit_llvm(mir_funcs) -> str`` and ``mangle(name) -> str``.
"""

from __future__ import annotations

import re
import struct as _structmod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .mir import MirFunc
from .cps_frames import is_suspending

# Value kinds -----------------------------------------------------------------

I64 = "i64"
F64 = "f64"
STR = "str"
CONFLICT = "conflict"
_STRUCT_PREFIX = "struct:"
_ENUM_PREFIX = "enum:"
_CLOSURE_PREFIX = "closure:"

_LLTY = {I64: "i64", F64: "double", STR: "ptr"}
_SCALARS = (I64, F64, STR)

# The uniform closure value: { fn pointer, env pointer }.
_CLOSURE_PAIR_TY = "%mx.closure"

# Binop tables ----------------------------------------------------------------

_ARITH_INT = {"+": "add", "-": "sub", "*": "mul", "/": "sdiv", "%": "srem"}
_ARITH_FLT = {"+": "fadd", "-": "fsub", "*": "fmul", "/": "fdiv", "%": "frem"}
_CMP_INT = {"==": "eq", "!=": "ne", "<": "slt", "<=": "sle", ">": "sgt", ">=": "sge"}
_CMP_FLT = {"==": "oeq", "!=": "one", "<": "olt", "<=": "ole", ">": "ogt", ">=": "oge"}
_LOGIC = {"&&": "and", "||": "or", "and": "and", "or": "or"}
_SUPPORTED_BINOPS = set(_ARITH_INT) | set(_CMP_INT) | set(_LOGIC)

# Builtins --------------------------------------------------------------------

_PRINT_BUILTINS = {"print", "println"}
_MATH_EXTERNS = {"sqrt", "sin", "cos"}  # double -> double libc functions
_INLINE_BUILTINS = {"neg", "not"}

# Callees implemented by the vec/string/trait interpreter runtime.
_RUNTIME_PREFIXES = ("__vec_", "__index_", "__slice_", "__range", "__trait$", "__static$")
_RUNTIME_NAMES = {"to_string", "int_to_str", "type_of", "len", "push", "pop",
                  "Vec.new", "assert_eq"}

_I64_MIN, _I64_MAX = -(2 ** 63), 2 ** 63 - 1

_HEADER = (
    "; LLVM IR emitted by metaxu codegen_llvm (direct subset)\n"
    "; conventions: ints/bools/unit -> i64 (unit = 0); floats -> double;\n"
    ";   strings -> ptr to private constant byte arrays; cmp results zext to i64;\n"
    ";   &&/|| normalize operands with icmp ne 0 (truthiness, not bitwise);\n"
    ";   / and % are sdiv/srem (trunc toward zero; interpreter floors);\n"
    ";   local structs -> %struct.T entry allocas + GEP (value semantics,\n"
    ";   whole-aggregate copies; zero heap management -- the frame owns them);\n"
    ";   @global structs -> entry-block malloc(8 * nfields) + GEP on the heap\n"
    ";   pointer, freed on every ret path: sound because value semantics means\n"
    ";   the storage pointer never escapes the frame (aggregates cross frames\n"
    ";   by copy).  Any storage whose lifetime cannot be proven leaks by\n"
    ";   design rather than risking a double-free/use-after-free;\n"
    ";   struct params pass as ptr + callee byval-copy into own storage\n"
    ";   (a borrow-informed increment can elide the copy for @const params);\n"
    ";   struct returns are sret-style: caller passes its result slot as a\n"
    ";   leading ptr %agg.ret arg, callee copies the aggregate in, rets void;\n"
    ";   print/println route by operand type to @metaxu_print_{i64,f64,str};\n"
    ";   multi-arg print joins per-kind printf directives with spaces;\n"
    ";   metaxu symbols are prefixed mx_ to stay clear of libc names;\n"
    ";   enums -> %enum.E = { i64 tag, [N x i64] payload } tagged unions,\n"
    ";   variant names mapped to dense integer tags (module-wide table in a\n"
    ";   comment below); pattern tag tests compare integers, never strings;\n"
    ";   closures -> %mx.closure = { ptr fn, ptr env } pairs over per-lambda\n"
    ";   stack %env.L structs; lambdas take env as a leading param.\n"
    "; functions outside the subset appear as comment-only placeholders."
)


def _llparam(kind: str) -> str:
    """The LLVM parameter/return-slot type for a value kind (aggregates -> ptr)."""
    if _is_agg(kind):
        return "ptr"
    return _LLTY.get(kind, "i64")


def _sanitize(name: str) -> str:
    """Restrict a symbol to [A-Za-z0-9_]."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def mangle(name: str) -> str:
    """The module-local LLVM symbol for a metaxu function name."""
    return "mx_" + _sanitize(name)


def _is_runtime_builtin(name: str) -> bool:
    return name in _RUNTIME_NAMES or any(name.startswith(p) for p in _RUNTIME_PREFIXES)


def _is_struct(kind: str) -> bool:
    return kind.startswith(_STRUCT_PREFIX)


def _struct_name(kind: str) -> str:
    return kind[len(_STRUCT_PREFIX):]


def _is_enum(kind: str) -> bool:
    return kind.startswith(_ENUM_PREFIX)


def _enum_name(kind: str) -> str:
    return kind[len(_ENUM_PREFIX):]


def _is_closure(kind: str) -> bool:
    return kind.startswith(_CLOSURE_PREFIX)


def _closure_lambda(kind: str) -> str:
    return kind[len(_CLOSURE_PREFIX):]


def _is_agg(kind: str) -> bool:
    """Aggregate kinds: stored in own allocas, cross calls by pointer."""
    return _is_struct(kind) or _is_enum(kind) or _is_closure(kind)


def _enum_llname(ename: str) -> str:
    """The %enum type name for an enum ('' -> the shared anon type)."""
    return "%enum." + (_sanitize(ename) or "anon")


def _agg_ty(kind: str) -> str:
    """The LLVM named type of an aggregate kind."""
    if _is_struct(kind):
        return f"%struct.{_sanitize(_struct_name(kind))}"
    if _is_enum(kind):
        return _enum_llname(_enum_name(kind))
    return _CLOSURE_PAIR_TY


def _join(a: str, b: str) -> str:
    """Kind lattice: i64 is bottom; f64/str/struct:T are incomparable tops."""
    if a == b:
        return a
    if a == I64:
        return b
    if b == I64:
        return a
    return CONFLICT


def _fmt_f64(x: float) -> str:
    """IEEE-754 bit-pattern hex literal, the unambiguous LLVM float syntax."""
    return "0x%016X" % _structmod.unpack("<Q", _structmod.pack("<d", float(x)))[0]


def _escape_bytes(data: bytes) -> str:
    out: List[str] = []
    for b in data:
        if 0x20 <= b < 0x7F and b not in (0x22, 0x5C):  # printable, not " or \
            out.append(chr(b))
        else:
            out.append("\\%02X" % b)
    return "".join(out)


class _Unsupported(Exception):
    """Raised during emission when a function turns out non-direct."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# Per-function analysis (mirrors codegen_clif._analyze)
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
    calls: List[Tuple[str, str, Tuple[str, ...]]] = field(default_factory=list)
    slots: List[str] = field(default_factory=list)
    suspending: bool = False
    # Variables defined by an @global alloc_struct: heap-backed storage.
    global_alloc_vars: Set[str] = field(default_factory=set)
    # Const-string variables that only feed tag comparisons: var -> variant name.
    tag_consts: Dict[str, str] = field(default_factory=dict)
    # make_closure sites: (dst, lambda name, capture names in order).
    closure_defs: List[Tuple[str, str, Tuple[str, ...]]] = field(default_factory=list)
    # Calls through a local variable (closure calls): (dst, callee var, args).
    closure_calls: List[Tuple[str, str, Tuple[str, ...]]] = field(default_factory=list)
    # Capture names this function receives through its env (it is a lambda).
    env_captures: Tuple[str, ...] = ()
    is_lambda: bool = False
    # Results of copy/select ops that are provably never observed (see
    # _dead_results): excluded from kind unification, emitted as comments.
    dead_results: Set[str] = field(default_factory=set)

    def add_reason(self, r: str) -> None:
        if r not in self.reasons:
            self.reasons.append(r)


def _find_tag_consts(f: MirFunc) -> Dict[str, str]:
    """Const-string vars whose EVERY use is an ==/!= against a variant_tag
    result (the compiled-pattern shape): they lower to integer tags."""
    tag_vars: Set[str] = set()
    str_consts: Dict[str, str] = {}
    use_count: Dict[str, int] = {}
    tag_cmp_count: Dict[str, int] = {}

    def count_use(n: str) -> None:
        use_count[n] = use_count.get(n, 0) + 1

    for b in f.blocks:
        for op in b.ops:
            if op[0] == "let" and len(op) == 4:
                _, dst, rhs, args = op
                if rhs[0] == "variant_tag":
                    tag_vars.add(dst)
                elif rhs[0] == "const" and isinstance(rhs[1], str) \
                        and not isinstance(rhs[1], bool):
                    str_consts[dst] = rhs[1]
                if rhs[0] == "alloc_struct" or rhs[0] == "make_closure":
                    for (_n, v) in args:
                        count_use(v)
                else:
                    for a in args:
                        count_use(a)
            elif op[0] == "perform":
                for a in op[4]:
                    count_use(a)
        t = b.term
        if t[0] == "br_if":
            count_use(t[1])
        elif t[0] == "ret":
            count_use(t[1])
    for b in f.blocks:
        for op in b.ops:
            if op[0] != "let" or len(op) != 4 or op[2][0] != "binop":
                continue
            if op[2][1] not in ("==", "!="):
                continue
            a0, a1 = op[3]
            if a0 in tag_vars and a1 in str_consts:
                tag_cmp_count[a1] = tag_cmp_count.get(a1, 0) + 1
            elif a1 in tag_vars and a0 in str_consts:
                tag_cmp_count[a0] = tag_cmp_count.get(a0, 0) + 1
    return {v: s for v, s in str_consts.items()
            if tag_cmp_count.get(v, 0) > 0
            and tag_cmp_count[v] == use_count.get(v, 0)}


def _dead_results(f: MirFunc) -> Set[str]:
    """Variables whose value is provably never observed: every use is as an
    operand of a copy/select whose own destination is dead (transitively),
    and they never reach a terminator or any other op.

    Statement-position if/match expressions lower to exactly this shape: a
    result variable copy-merged from arms of DIFFERENT kinds (e.g. one arm
    yields unit/i64, the other an enum).  Unifying kinds through those dead
    copies would poison live variables into aggregate kinds, so dead
    copy/select ops are excluded from inference and emitted as comments —
    which is sound precisely because nothing ever reads their result."""
    soft_uses: Dict[str, List[str]] = {}   # var -> dsts of copy/select users
    hard_used: Set[str] = set()
    defs: Set[str] = set()
    for b in f.blocks:
        for op in b.ops:
            if op[0] == "let" and len(op) == 4:
                _, dst, rhs, args = op
                defs.add(dst)
                rk = rhs[0]
                if rk in ("copy", "select"):
                    for a in args:
                        soft_uses.setdefault(a, []).append(dst)
                elif rk in ("alloc_struct", "make_closure"):
                    for pair in args:
                        hard_used.add(pair[1])
                elif rk == "call":
                    hard_used.add(rhs[1])  # closure callee var (if any)
                    hard_used.update(args)
                else:
                    hard_used.update(args)
            elif op[0] == "perform":
                hard_used.update(op[4])
        t = b.term
        if t[0] in ("br_if", "ret"):
            hard_used.add(t[1])
    dead: Set[str] = set()
    changed = True
    while changed:
        changed = False
        for v in defs:
            if v in dead or v in hard_used:
                continue
            if all(d in dead for d in soft_uses.get(v, [])):
                dead.add(v)
                changed = True
    return dead


def _blocks_in_cycles(f: MirFunc) -> Set[int]:
    """Block indices that lie on a CFG cycle (loop bodies/headers)."""
    n = len(f.blocks)
    succs: List[List[int]] = []
    for b in f.blocks:
        t = b.term
        if t[0] == "br":
            succs.append([t[1]] if 0 <= t[1] < n else [])
        elif t[0] == "br_if":
            succs.append([x for x in (t[2], t[3]) if 0 <= x < n])
        else:
            succs.append([])
    reach: List[Set[int]] = []
    for i in range(n):
        seen: Set[int] = set()
        stack = list(succs[i])
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(succs[x])
        reach.append(seen)
    return {i for i in range(n) if i in reach[i]}


def _analyze(f: MirFunc, module_names: Set[str], closures: "_ClosureTable") -> _Info:
    info = _Info(f=f)
    try:
        _analyze_inner(info, module_names, closures)
    except Exception as exc:  # defensive: malformed MIR must never crash codegen
        info.add_reason(f"analysis error: {type(exc).__name__}: {exc}")
    return info


def _analyze_inner(info: _Info, module_names: Set[str],
                   closures: "_ClosureTable") -> None:
    f = info.f
    if not f.blocks:
        info.add_reason("function has no blocks")
        return
    if f.blocks[0].ops and f.blocks[0].ops[0][0] == "params":
        info.params = tuple(f.blocks[0].ops[0][1])
    info.suspending = is_suspending(f)
    if info.suspending:
        info.add_reason(
            "suspending function (LLVM CPS lowering not implemented; "
            "effects run in the interpreter / CLIF CPS)")
    info.tag_consts = _find_tag_consts(f)
    info.dead_results = _dead_results(f)
    cycle_blocks = _blocks_in_cycles(f)

    # A make_closure target receives its captures through the env struct:
    # they are entry-defined names, exactly like parameters.
    if f.name in closures.targets:
        info.is_lambda = True
        info.env_captures = closures.targets[f.name]
        if f.name in closures.bad:
            info.add_reason(closures.bad[f.name])

    for p in info.params:
        info.def_count[p] = 1
        info.def_block[p] = 0
    for c in info.env_captures:
        if c not in info.def_count:
            info.def_count[c] = 1
            info.def_block[c] = 0

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
                # Suspension point; the function is already demoted above.
                continue
            if kind == "drop":
                continue  # emitted as a comment
            if kind == "match_fail":
                continue  # emitted as @abort + unreachable
            if kind != "let" or len(op) != 4:
                info.add_reason(f"unsupported op {kind!r}")
                continue
            _, dst, rhs, args = op
            rk = rhs[0]
            if rk == "const":
                v = rhs[1]
                if isinstance(v, bool) or v is None:
                    pass
                elif isinstance(v, int):
                    if not (_I64_MIN <= v <= _I64_MAX):
                        info.add_reason(f"integer constant {v} outside i64 range")
                elif isinstance(v, (float, str)):
                    pass
                else:
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
                info.calls.append((dst, callee, tuple(args)))
                for a in args:
                    add_use(a, bi)
                add_def(dst, bi)
            elif rk == "alloc_struct":
                locality = rhs[2] if len(rhs) > 2 else "local"
                if locality == "global":
                    info.global_alloc_vars.add(dst)
                elif locality != "local":
                    info.add_reason(
                        f"unknown locality {locality!r} for struct {rhs[1]!r}")
                for (_fname, fval) in args:
                    add_use(fval, bi)
                add_def(dst, bi)
            elif rk == "field_get":
                add_use(args[0], bi)
                add_def(dst, bi)
            elif rk == "field_set":
                add_use(args[0], bi)
                add_use(args[1], bi)
                add_def(dst, bi)
            elif rk == "make_variant":
                for a in args:
                    add_use(a, bi)
                add_def(dst, bi)
            elif rk in ("variant_tag", "variant_field"):
                add_use(args[0], bi)
                add_def(dst, bi)
            elif rk == "make_closure":
                lname = rhs[1]
                if lname not in module_names:
                    info.add_reason(
                        f"make_closure of unknown function {lname!r}")
                if bi in cycle_blocks:
                    info.add_reason(
                        "make_closure inside a loop (re-executing the site "
                        "would overwrite the shared stack env while earlier "
                        "pair copies may still alias it)")
                for (_cn, vn) in args:
                    add_use(vn, bi)
                add_def(dst, bi)
                info.closure_defs.append(
                    (dst, lname, tuple(cn for (cn, _vn) in args)))
            elif rk in ("resume", "handle_scope"):
                info.add_reason(f"uses effects ({rk})")
            elif rk == "try_scope":
                info.add_reason("uses try/catch (try_scope)")
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
            pass
        else:
            info.add_reason(f"unsupported terminator {t[0]!r}")

    # Calls: partition into closure calls (callee is a local variable — the
    # interpreter's shadowing order: locals first) and direct calls, which
    # must hit module functions or the supported builtins.
    direct_calls: List[Tuple[str, str, Tuple[str, ...]]] = []
    for (dst, callee, cargs) in info.calls:
        if callee in info.def_count:
            info.closure_calls.append((dst, callee, cargs))
        elif callee in closures.targets:
            # Lambdas are only callable through their closure value: a direct
            # call would skip the env parameter.
            info.add_reason(
                f"direct call to lambda {callee!r} (callable only through "
                "its closure value)")
        elif callee in module_names:
            direct_calls.append((dst, callee, cargs))
        elif callee in _PRINT_BUILTINS or callee in _MATH_EXTERNS or callee in _INLINE_BUILTINS:
            direct_calls.append((dst, callee, cargs))
        elif _is_runtime_builtin(callee):
            info.add_reason(f"calls runtime builtin {callee!r} (vec/string/trait)")
        else:
            info.add_reason(f"unknown external callee {callee!r} (cannot link natively)")
    info.calls = direct_calls

    # Every used name must be defined somewhere in the function.
    for name in info.use_blocks:
        if name not in info.def_count:
            info.add_reason(
                f"references {name!r} with no local definition (captured environment)")


# ---------------------------------------------------------------------------
# Module-wide struct table
# ---------------------------------------------------------------------------

@dataclass
class _StructTable:
    fields: Dict[str, Tuple[str, ...]] = field(default_factory=dict)  # name -> field order
    bad: Dict[str, str] = field(default_factory=dict)                 # name -> reason
    kinds: Dict[Tuple[str, str], str] = field(default_factory=dict)   # (name, field) -> kind

    def field_kind(self, sname: str, fname: str) -> str:
        return self.kinds.get((sname, fname), I64)

    def mark_field(self, sname: str, fname: str, kind: str) -> bool:
        key = (sname, fname)
        cur = self.kinds.get(key, I64)
        nk = _join(cur, kind)
        if nk != cur:
            self.kinds[key] = nk
            return True
        return False


def _build_struct_table(funcs: Sequence[MirFunc]) -> _StructTable:
    table = _StructTable()
    for f in funcs:
        for b in f.blocks:
            for op in b.ops:
                if op[0] != "let" or len(op) != 4 or op[2][0] != "alloc_struct":
                    continue
                sname = op[2][1]
                names = tuple(fn for (fn, _fv) in op[3])
                if sname not in table.fields:
                    table.fields[sname] = names
                elif set(table.fields[sname]) != set(names):
                    table.bad.setdefault(
                        sname,
                        f"inconsistent field sets for struct {sname!r}: "
                        f"{sorted(table.fields[sname])} vs {sorted(names)}")
    return table


# ---------------------------------------------------------------------------
# Module-wide variant (tagged-union) table
# ---------------------------------------------------------------------------

@dataclass
class _VariantTable:
    # Module-wide dense integer tag per variant name (make_variant names and
    # pattern tag literals, sorted then numbered): string equality on tags in
    # the interpreter maps exactly to integer equality natively.
    tags: Dict[str, int] = field(default_factory=dict)
    # enum name ('' = anon) -> payload slot count (max over its variants).
    payload_max: Dict[str, int] = field(default_factory=dict)
    # (enum name, slot index) -> scalar kind.  One-way cells: stores
    # accumulate in, reads take the cell kind; a store whose final kind
    # differs from the cell demotes instead of coercing (two variants of one
    # enum may legally put different types in the same slot).
    cells: Dict[Tuple[str, int], str] = field(default_factory=dict)
    # enum name -> variant names seen in make_variant ops (for the doc line).
    variants_of: Dict[str, Set[str]] = field(default_factory=dict)

    def cell_kind(self, ename: str, idx: int) -> str:
        return self.cells.get((ename, idx), I64)

    def mark_cell(self, ename: str, idx: int, kind: str) -> bool:
        key = (ename, idx)
        cur = self.cells.get(key, I64)
        nk = _join(cur, kind)
        if nk != cur:
            self.cells[key] = nk
            return True
        return False


def _build_variant_table(funcs: Sequence[MirFunc],
                         infos: Sequence[_Info]) -> _VariantTable:
    table = _VariantTable()
    names: Set[str] = set()
    for f in funcs:
        for b in f.blocks:
            for op in b.ops:
                if op[0] != "let" or len(op) != 4 or op[2][0] != "make_variant":
                    continue
                ename, vname = op[2][1], op[2][2]
                names.add(vname)
                table.variants_of.setdefault(ename, set()).add(vname)
                table.payload_max[ename] = max(
                    table.payload_max.get(ename, 0), len(op[3]))
    for info in infos:
        names.update(info.tag_consts.values())
    table.tags = {n: i for i, n in enumerate(sorted(names))}
    return table


# ---------------------------------------------------------------------------
# Module-wide closure table (make_closure targets and their captures)
# ---------------------------------------------------------------------------

@dataclass
class _ClosureTable:
    # lambda name -> capture names, in env-struct field order.
    targets: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    bad: Dict[str, str] = field(default_factory=dict)
    # (lambda name, capture name) -> kind (two-way cells, like parameters:
    # the creator's captured value and the lambda's uses are one value).
    cells: Dict[Tuple[str, str], str] = field(default_factory=dict)
    # lambda name -> arity declared at the make_closure site.
    arity: Dict[str, int] = field(default_factory=dict)

    def cell_kind(self, lname: str, cap: str) -> str:
        return self.cells.get((lname, cap), I64)

    def mark_cell(self, lname: str, cap: str, kind: str) -> bool:
        key = (lname, cap)
        cur = self.cells.get(key, I64)
        nk = _join(cur, kind)
        if nk != cur:
            self.cells[key] = nk
            return True
        return False


def _build_closure_table(funcs: Sequence[MirFunc]) -> _ClosureTable:
    table = _ClosureTable()
    for f in funcs:
        for b in f.blocks:
            for op in b.ops:
                if op[0] != "let" or len(op) != 4 or op[2][0] != "make_closure":
                    continue
                lname = op[2][1]
                caps = tuple(cn for (cn, _vn) in op[3])
                arity = len(op[2][2]) if len(op[2]) > 2 else 0
                if lname not in table.targets:
                    table.targets[lname] = caps
                    table.arity[lname] = arity
                elif table.targets[lname] != caps or table.arity[lname] != arity:
                    table.bad.setdefault(
                        lname,
                        f"lambda {lname!r} created with inconsistent capture "
                        "lists across make_closure sites")
    return table


# ---------------------------------------------------------------------------
# Kind inference (i64 by default, monotone promotion; module fixpoint)
# ---------------------------------------------------------------------------

@dataclass
class _Sig:
    params: List[str]
    ret: str = I64


def _infer_kinds(info: _Info, sigs: Dict[str, _Sig], structs: _StructTable,
                 variants: _VariantTable, closures: _ClosureTable,
                 ) -> Tuple[Dict[str, str], bool]:
    """One inner fixpoint over a function.  Returns (kinds, global_changed)
    where global_changed reports promotions written into shared cells
    (struct fields, enum payload slots, closure captures) so the module
    driver keeps iterating."""
    kinds: Dict[str, str] = {}
    global_changed = False

    def get(n: str) -> str:
        return kinds.get(n, I64)

    def mark(n: str, k: str) -> bool:
        nk = _join(get(n), k)
        if nk != get(n):
            kinds[n] = nk
            return True
        return False

    def unify(names: Sequence[str]) -> bool:
        k = I64
        for n in names:
            k = _join(k, get(n))
        changed = False
        for n in names:
            changed = mark(n, k) or changed
        return changed

    fname = info.f.name
    own_sig = sigs.get(fname)

    changed = True
    while changed:
        changed = False
        if own_sig is not None and len(own_sig.params) == len(info.params):
            for p, pk in zip(info.params, own_sig.params):
                changed = mark(p, pk) or changed
            for r in info.ret_vars:
                changed = mark(r, own_sig.ret) or changed
        if len(info.ret_vars) > 1:
            changed = unify(info.ret_vars) or changed
        # Lambda captures behave like parameters: two-way join between the
        # creator-side cell and the local uses.
        if info.is_lambda:
            for cap in info.env_captures:
                nk = _join(closures.cell_kind(fname, cap), get(cap))
                if closures.mark_cell(fname, cap, nk):
                    changed = global_changed = True
                changed = mark(cap, nk) or changed
        for b in info.f.blocks:
            for op in b.ops:
                if op[0] != "let" or len(op) != 4:
                    continue
                _, dst, rhs, args = op
                rk = rhs[0]
                if rk == "const":
                    v = rhs[1]
                    if isinstance(v, bool) or v is None:
                        pass
                    elif isinstance(v, float):
                        changed = mark(dst, F64) or changed
                    elif isinstance(v, str):
                        if dst not in info.tag_consts:
                            changed = mark(dst, STR) or changed
                        # tag literals stay i64: they lower to integer tags
                elif rk == "copy":
                    if dst not in info.dead_results:
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
                    if len(args) == 3 and dst not in info.dead_results:
                        changed = unify((dst, args[1], args[2])) or changed
                elif rk == "call":
                    callee = rhs[1]
                    if callee in info.def_count:
                        # Closure call: types flow through the lambda's sig
                        # once the callee variable's closure kind is known.
                        ck = get(callee)
                        sig = sigs.get(_closure_lambda(ck)) if _is_closure(ck) else None
                        if sig is not None and len(sig.params) == len(args):
                            for a, pk in zip(args, sig.params):
                                changed = mark(a, pk) or changed
                            changed = mark(dst, sig.ret) or changed
                    elif callee in _MATH_EXTERNS:
                        for a in args:
                            changed = mark(a, F64) or changed
                        changed = mark(dst, F64) or changed
                    elif callee == "neg":
                        if len(args) == 1:
                            changed = unify((dst, args[0])) or changed
                    elif callee in _PRINT_BUILTINS or callee == "not":
                        pass  # dst is unit/bool -> i64
                    else:
                        sig = sigs.get(callee)
                        if sig is not None and len(sig.params) == len(args):
                            for a, pk in zip(args, sig.params):
                                changed = mark(a, pk) or changed
                            changed = mark(dst, sig.ret) or changed
                elif rk == "make_variant":
                    changed = mark(dst, _ENUM_PREFIX + rhs[1]) or changed
                    # One-way: store kinds accumulate into the payload cell;
                    # the post-fixpoint check demotes on store/cell mismatch
                    # rather than coercing the store.
                    for i, a in enumerate(args):
                        if variants.mark_cell(rhs[1], i, get(a)):
                            changed = global_changed = True
                elif rk == "variant_tag":
                    pass  # dst is the integer tag: i64 (the default)
                elif rk == "variant_field":
                    bk = get(args[0])
                    if _is_enum(bk):
                        changed = mark(
                            dst, variants.cell_kind(_enum_name(bk), rhs[1])) or changed
                elif rk == "make_closure":
                    changed = mark(dst, _CLOSURE_PREFIX + rhs[1]) or changed
                    for (cn, vn) in args:
                        nk = _join(closures.cell_kind(rhs[1], cn), get(vn))
                        if closures.mark_cell(rhs[1], cn, nk):
                            changed = global_changed = True
                        changed = mark(vn, nk) or changed
                elif rk == "alloc_struct":
                    sname = rhs[1]
                    changed = mark(dst, _STRUCT_PREFIX + sname) or changed
                    for (fn_, fv) in args:
                        fk = structs.field_kind(sname, fn_)
                        nk = _join(fk, get(fv))
                        if structs.mark_field(sname, fn_, nk):
                            changed = global_changed = True
                        changed = mark(fv, nk) or changed
                elif rk == "field_get":
                    bk = get(args[0])
                    if _is_struct(bk):
                        sname = _struct_name(bk)
                        fk = structs.field_kind(sname, rhs[1])
                        nk = _join(fk, get(dst))
                        if structs.mark_field(sname, rhs[1], nk):
                            changed = global_changed = True
                        changed = mark(dst, nk) or changed
                elif rk == "field_set":
                    changed = unify((dst, args[0])) or changed
                    bk = get(args[0])
                    if _is_struct(bk):
                        sname = _struct_name(bk)
                        fk = structs.field_kind(sname, rhs[1])
                        nk = _join(fk, get(args[1]))
                        if structs.mark_field(sname, rhs[1], nk):
                            changed = global_changed = True
                        changed = mark(args[1], nk) or changed
    return kinds, global_changed


# ---------------------------------------------------------------------------
# Consistency checking (post-fixpoint; anything wrong demotes the function)
# ---------------------------------------------------------------------------

def _check_consistency(info: _Info, kinds: Dict[str, str], sigs: Dict[str, _Sig],
                       structs: _StructTable, variants: _VariantTable,
                       closures: _ClosureTable,
                       module_names: Set[str]) -> List[str]:
    probs: List[str] = []

    def ty(n: str) -> str:
        return kinds.get(n, I64)

    for name in sorted(set(info.def_count) | set(info.use_blocks)):
        if ty(name) == CONFLICT:
            probs.append(f"irreconcilable value kinds for {name!r}")

    # Struct-kinded params and returns are supported: params pass as ptr with
    # a callee byval-copy, returns are sret-style (see module docstring).

    for b in info.f.blocks:
        for op in b.ops:
            if op[0] != "let" or len(op) != 4:
                continue
            _, dst, rhs, args = op
            rk = rhs[0]
            if rk == "binop":
                o = rhs[1]
                if o in _CMP_INT:
                    if ty(dst) not in (I64,):
                        probs.append(f"comparison result {dst!r} promoted to {ty(dst)}")
                    if ty(args[0]) not in (I64, F64) or ty(args[1]) not in (I64, F64):
                        probs.append(f"comparison {o!r} on non-numeric operands")
                elif o in _LOGIC:
                    if any(ty(x) != I64 for x in (dst, *args)):
                        probs.append(f"logical binop {o!r} on non-i64 values")
                else:
                    if ty(dst) not in (I64, F64):
                        probs.append(f"arithmetic {o!r} on non-numeric kind {ty(dst)}")
            elif rk == "select" and len(args) == 3:
                if dst in info.dead_results:
                    continue  # emitted as a comment, never observed
                if ty(args[0]) != I64:
                    probs.append(f"select condition {args[0]!r} is {ty(args[0])}")
                if _is_agg(ty(dst)):
                    probs.append("select over aggregate values")
            elif rk == "call":
                callee = rhs[1]
                if callee in info.def_count:
                    # Closure call: the callee variable must be pinned to one
                    # statically-known lambda whose signature matches.
                    ck = ty(callee)
                    if not _is_closure(ck):
                        probs.append(
                            f"call through local {callee!r} that is not a "
                            "statically-known closure")
                        continue
                    lname = _closure_lambda(ck)
                    sig = sigs.get(lname)
                    if sig is None or lname not in module_names:
                        probs.append(f"closure call to unknown lambda {lname!r}")
                        continue
                    if len(sig.params) != len(args):
                        probs.append(
                            f"closure call to {lname!r} with wrong arity")
                        continue
                    for a, pk in zip(args, sig.params):
                        if ty(a) != pk:
                            probs.append(
                                f"closure call to {lname!r}: arg {a!r} is "
                                f"{ty(a)}, expects {pk}")
                    if ty(dst) != sig.ret:
                        probs.append(
                            f"closure call to {lname!r}: result {dst!r} is "
                            f"{ty(dst)}, returns {sig.ret}")
                elif callee in _PRINT_BUILTINS:
                    for a in args:
                        if ty(a) not in (I64, F64, STR):
                            probs.append(f"print of unsupported kind {ty(a)}")
                elif callee in _MATH_EXTERNS or callee in _INLINE_BUILTINS:
                    pass  # kinds pinned during inference
                elif callee in module_names:
                    sig = sigs.get(callee)
                    if sig is None or len(sig.params) != len(args):
                        probs.append(f"call to {callee!r} with wrong arity")
                        continue
                    for a, pk in zip(args, sig.params):
                        if ty(a) != pk:
                            probs.append(
                                f"call to {callee!r}: arg {a!r} is {ty(a)}, expects {pk}")
                    if ty(dst) != sig.ret:
                        probs.append(
                            f"call to {callee!r}: result {dst!r} is {ty(dst)}, "
                            f"returns {sig.ret}")
            elif rk == "alloc_struct":
                # locality "local" -> frame alloca; "global" -> heap malloc
                # with free at function exit (unknown localities were already
                # demoted during analysis).
                sname = rhs[1]
                if sname in structs.bad:
                    probs.append(structs.bad[sname])
                elif set(fn_ for (fn_, _fv) in args) != set(structs.fields.get(sname, ())):
                    probs.append(f"alloc_struct field mismatch for {sname!r}")
            elif rk in ("field_get", "field_set"):
                bk = ty(args[0])
                if not _is_struct(bk):
                    probs.append(
                        f"cannot determine struct type of {args[0]!r} for {rk}")
                else:
                    sname = _struct_name(bk)
                    if sname in structs.bad:
                        probs.append(structs.bad[sname])
                    elif rhs[1] not in structs.fields.get(sname, ()):
                        probs.append(f"struct {sname!r} has no field {rhs[1]!r}")
            elif rk == "make_variant":
                ename = rhs[1]
                for i, a in enumerate(args):
                    ck = variants.cell_kind(ename, i)
                    if ty(a) != ck and ck in _SCALARS:
                        probs.append(
                            f"heterogeneous payload slot {i} of enum "
                            f"{ename or 'anon'!r}: variant {rhs[2]!r} stores "
                            f"{ty(a)}, the unified slot is {ck} (no coercion "
                            "through tagged-union storage)")
            elif rk == "variant_tag":
                if not _is_enum(ty(args[0])):
                    probs.append(
                        f"cannot determine enum type of {args[0]!r} for variant_tag")
                if ty(dst) != I64:
                    probs.append(f"variant tag {dst!r} promoted to {ty(dst)}")
            elif rk == "variant_field":
                bk = ty(args[0])
                if not _is_enum(bk):
                    probs.append(
                        f"cannot determine enum type of {args[0]!r} for variant_field")
                else:
                    ename = _enum_name(bk)
                    idx = rhs[1]
                    # Patterns can name a payload slot no constructor fills
                    # (dead arm); grow the union so the GEP stays in bounds.
                    variants.payload_max[ename] = max(
                        variants.payload_max.get(ename, 0), idx + 1)
                    ck = variants.cell_kind(ename, idx)
                    if ty(dst) != ck:
                        probs.append(
                            f"variant_field {idx} of enum {ename or 'anon'!r}: "
                            f"result {dst!r} is {ty(dst)}, slot is {ck}")
            elif rk == "make_closure":
                lname = rhs[1]
                for (cn, _vn) in args:
                    ck = closures.cell_kind(lname, cn)
                    if _is_closure(ck):
                        probs.append(
                            f"lambda {lname!r} captures closure {cn!r} "
                            "(closure-in-closure envs are a later increment)")
                    elif ck == CONFLICT:
                        probs.append(
                            f"lambda {lname!r} capture {cn!r} has conflicting kinds")
        if b.term[0] == "br_if" and ty(b.term[1]) != I64:
            probs.append(f"br_if condition {b.term[1]!r} is {ty(b.term[1])}")
        if b.term[0] == "ret" and _is_closure(ty(b.term[1])):
            # The pair's env pointer aims into this frame: returning it would
            # dangle.  A heap-env increment can lift this.
            probs.append(
                "returns a closure (its stack env would dangle; heap-env "
                "closures are a later increment)")
        # ret of a struct/enum value is fine: sret-style, the aggregate is
        # copied into the caller-provided %agg.ret slot (never a raw frame
        # pointer).

    # Struct field / enum payload / capture kinds must be scalar.
    used_structs = {_struct_name(k) for k in kinds.values() if _is_struct(k)}
    for sname in sorted(used_structs):
        for fn_ in structs.fields.get(sname, ()):
            fk = structs.field_kind(sname, fn_)
            if _is_agg(fk):
                probs.append(
                    f"struct {sname!r} field {fn_!r} holds an aggregate "
                    f"({fk}) (nested aggregate fields are a later increment)")
            elif fk == CONFLICT:
                probs.append(f"struct {sname!r} field {fn_!r} has conflicting kinds")
    used_enums = {_enum_name(k) for k in kinds.values() if _is_enum(k)}
    for ename in sorted(used_enums):
        for i in range(variants.payload_max.get(ename, 0)):
            ck = variants.cell_kind(ename, i)
            if _is_agg(ck):
                probs.append(
                    f"enum {ename or 'anon'!r} payload slot {i} holds an "
                    f"aggregate ({ck}) (heap-boxed payloads — e.g. recursive "
                    "enums — are a later increment)")
            elif ck == CONFLICT:
                probs.append(
                    f"enum {ename or 'anon'!r} payload slot {i} has "
                    "conflicting kinds")
    # Captures this function loads from its own env must be liftable too.
    if info.is_lambda:
        for cap in info.env_captures:
            ck = closures.cell_kind(info.f.name, cap)
            if _is_closure(ck):
                probs.append(
                    f"capture {cap!r} is itself a closure (closure-in-closure "
                    "envs are a later increment)")
            elif ck == CONFLICT:
                probs.append(f"capture {cap!r} has conflicting kinds")
    return probs


def _compute_slots(info: _Info, kinds: Dict[str, str]) -> List[str]:
    """Non-aggregate variables needing an alloca (mirrors codegen_clif)."""
    slots: List[str] = []
    for name, cnt in info.def_count.items():
        if _is_agg(kinds.get(name, I64)):
            continue  # aggregate vars always get their own storage alloca
        if cnt > 1:
            slots.append(name)
            continue
        db = info.def_block.get(name, 0)
        uses = info.use_blocks.get(name, set())
        if db != 0 and any(u != db for u in uses):
            slots.append(name)
    return sorted(slots)


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

class _ModuleState:
    """Shared cross-function emission state: string pool + runtime needs."""

    def __init__(self) -> None:
        self.strings: Dict[str, str] = {}  # content -> global name
        self.print_helpers: Set[str] = set()  # subset of {"i64","f64","str"}
        self.math_used: Set[str] = set()
        self.uses_abort = False
        self.uses_malloc = False   # @global structs: malloc/free declares
        self.uses_printf = False   # direct variadic printf (multi-arg print)
        self.used_enums: Set[str] = set()      # enum names needing %enum types
        self.uses_closure_pair = False         # %mx.closure type needed
        # lambda name -> ((capture name, kind), ...) for %env.L emission
        self.env_types: Dict[str, Tuple[Tuple[str, str], ...]] = {}

    def intern_string(self, content: str) -> str:
        if content not in self.strings:
            self.strings[content] = f"@.str.{len(self.strings)}"
        return self.strings[content]


def _emit_placeholder(info: _Info, sig: _Sig) -> str:
    sym = mangle(info.f.name)
    ptys = ", ".join(_llparam(k) for k in sig.params)
    rty = "void (sret ptr)" if _is_agg(sig.ret) else _LLTY.get(sig.ret, "i64")
    lines = [f"; function @{sym}: placeholder -- unsupported for direct LLVM emission"]
    for r in info.reasons:
        lines.append(f";   reason: {r}")
    lines.append(f"; declare @{sym}({ptys}) -> {rty}")
    return "\n".join(lines)


def _emit_function(info: _Info, kinds: Dict[str, str], sigs: Dict[str, _Sig],
                   structs: _StructTable, variants: _VariantTable,
                   closures: _ClosureTable, mod: _ModuleState,
                   emitted_names: Set[str]) -> str:
    f = info.f
    sig = sigs[f.name]

    def kind(n: str) -> str:
        return kinds.get(n, I64)

    def llty(n: str) -> str:
        return _LLTY.get(kind(n), "i64")

    def env_fields(lname: str) -> Tuple[Tuple[str, str], ...]:
        """The env struct layout of a lambda: (capture, kind) in list order."""
        return tuple((cn, closures.cell_kind(lname, cn))
                     for cn in closures.targets.get(lname, ()))

    slots = _compute_slots(info, kinds)
    slotset = set(slots)
    # Aggregate variables (struct / enum / closure kinds): each owns storage.
    agg_vars = sorted(n for n in info.def_count if _is_agg(kind(n)))
    aggset = set(agg_vars)
    # Heap-backed struct variables (@global alloc_struct defs): storage is an
    # entry-block malloc'd block instead of an alloca, freed on every ret.
    heap_vars = sorted(n for n in agg_vars if n in info.global_alloc_vars)
    heapset = set(heap_vars)
    sret = _is_agg(sig.ret)

    counter = 0

    def fresh() -> str:
        nonlocal counter
        v = f"%t{counter}"
        counter += 1
        return v

    valmap: Dict[str, str] = {}

    def slot_ref(n: str) -> str:
        return f"%slot.{_sanitize(n)}"

    def struct_ref(n: str) -> str:
        """The storage pointer for an aggregate variable (alloca or heap block)."""
        if n in heapset:
            return f"%hv.{_sanitize(n)}"
        return f"%sv.{_sanitize(n)}"

    def use(name: str, lines: List[str]) -> str:
        if name in aggset:
            return struct_ref(name)
        if name in slotset:
            v = fresh()
            lines.append(f"  {v} = load {llty(name)}, ptr {slot_ref(name)}")
            return v
        try:
            return valmap[name]
        except KeyError:
            raise _Unsupported(f"use of {name!r} before its definition")

    def setval(name: str, v: str, lines: List[str]) -> None:
        if name in aggset:
            raise _Unsupported(f"scalar assignment to aggregate variable {name!r}")
        if name in slotset:
            lines.append(f"  store {llty(name)} {v}, ptr {slot_ref(name)}")
        else:
            valmap[name] = v

    def scalar_const(name: str, value: Any) -> str:
        k = kind(name)
        if k == F64:
            if value is None:
                value = 0.0
            if isinstance(value, bool):
                value = float(value)
            return _fmt_f64(float(value))
        if k == STR:
            if isinstance(value, str):
                return mod.intern_string(value)
            raise _Unsupported(f"non-string constant for string value {name!r}")
        if value is None:
            return "0"
        if isinstance(value, bool):
            return str(int(value))
        if isinstance(value, float):
            return str(int(value))
        return str(int(value))

    def gep(sname: str, base_ptr: str, fname: str, lines: List[str]) -> str:
        idx = structs.fields[sname].index(fname)
        v = fresh()
        lines.append(
            f"  {v} = getelementptr inbounds %struct.{_sanitize(sname)}, "
            f"ptr {base_ptr}, i32 0, i32 {idx}")
        return v

    def agg_copy(tyname: str, src_ptr: str, dst_ptr: str, lines: List[str]) -> None:
        v = fresh()
        lines.append(f"  {v} = load {tyname}, ptr {src_ptr}")
        lines.append(f"  store {tyname} {v}, ptr {dst_ptr}")

    def enum_gep(ename: str, base_ptr: str, lines: List[str], *,
                 payload: Optional[int] = None) -> str:
        """GEP to the tag (payload=None) or a payload slot of a tagged union."""
        v = fresh()
        if payload is None:
            lines.append(
                f"  {v} = getelementptr inbounds {_enum_llname(ename)}, "
                f"ptr {base_ptr}, i32 0, i32 0")
        else:
            lines.append(
                f"  {v} = getelementptr inbounds {_enum_llname(ename)}, "
                f"ptr {base_ptr}, i32 0, i32 1, i32 {payload}")
        return v

    def emit_frees(lines: List[str]) -> None:
        """Free every heap-backed @global struct block (called on ret paths).

        Sound because the malloc unconditionally happens in the entry block
        (exactly once per invocation) and value semantics guarantees the
        storage pointer never escapes this frame (see module docstring)."""
        for n in heap_vars:
            lines.append(f"  call void @free(ptr {struct_ref(n)})"
                         f"  ; @global struct {n}: end of frame")

    # Params are visible from the entry block on: SSA args directly, spilled
    # params through their slot (the store happens in the entry prelude);
    # aggregate params through their own storage (byval-copied in the
    # prelude).  A lambda's captures are entry-defined the same way: scalar
    # captures load into %cap.* registers (or their slot), aggregate captures
    # copy into their own storage.
    for p in info.params:
        if p not in slotset and p not in aggset:
            valmap[p] = f"%a.{_sanitize(p)}"
    if info.is_lambda:
        for c in info.env_captures:
            if c not in slotset and c not in aggset:
                valmap[c] = f"%cap.{_sanitize(c)}"

    # One env alloca per make_closure site, named and created in the entry
    # block (storage must dominate every use; the site itself may sit in a
    # conditional block).
    env_allocas: Dict[int, str] = {}
    env_entry: List[str] = []
    env_seq = 0
    for b in f.blocks:
        for op in b.ops:
            if op[0] == "let" and len(op) == 4 and op[2][0] == "make_closure":
                lname = op[2][1]
                name = f"%env.site{env_seq}.{_sanitize(op[1])}"
                env_seq += 1
                env_allocas[id(op)] = name
                env_entry.append(
                    f"  {name} = alloca %env.{_sanitize(lname)}"
                    f"  ; closure env for {op[1]} -> {lname}")

    body: List[str] = []
    for bi, b in enumerate(f.blocks):
        lines: List[str] = []
        terminated = False
        for op in b.ops:
            opk = op[0]
            if opk == "params":
                continue
            if opk == "drop":
                if op[1] in heapset:
                    lines.append(f"  ; drop {op[1]} (@global struct: freed on ret paths)")
                else:
                    lines.append(f"  ; drop {op[1]} (scalar/local: frame-owned, no-op)")
                continue
            if opk == "match_fail":
                mod.uses_abort = True
                lines.append(f"  call void @abort()  ; match_fail: {op[1]}")
                lines.append("  unreachable")
                terminated = True
                break
            # opk == "let" (analysis guarantees this)
            _, dst, rhs, opargs = op
            rk = rhs[0]
            if rk == "const" and dst in info.tag_consts:
                # Pattern tag literal: the variant-name string lowers to its
                # module-wide integer tag; the string never reaches native
                # code.
                vname = info.tag_consts[dst]
                tagv = variants.tags.get(vname)
                if tagv is None:
                    raise _Unsupported(f"no tag assigned for variant {vname!r}")
                lines.append(f"  ; tag literal: {vname!r} -> {tagv}")
                setval(dst, str(tagv), lines)
            elif rk in ("const", "const_ty"):
                value = rhs[1] if rk == "const" else None
                setval(dst, scalar_const(dst, value), lines)
            elif rk == "copy":
                if dst in info.dead_results:
                    lines.append(f"  ; dead copy elided: {dst} (never observed)")
                    continue
                if kind(opargs[0]) != kind(dst):
                    raise _Unsupported(
                        f"copy between kinds {kind(opargs[0])} -> {kind(dst)}")
                if dst in aggset:
                    src = use(opargs[0], lines)
                    agg_copy(_agg_ty(kind(dst)), src, struct_ref(dst), lines)
                else:
                    setval(dst, use(opargs[0], lines), lines)
            elif rk == "binop":
                o = rhs[1]
                l = use(opargs[0], lines)
                r = use(opargs[1], lines)
                is_flt = kind(opargs[0]) == F64
                if o in _CMP_INT:
                    c = fresh()
                    if is_flt:
                        lines.append(f"  {c} = fcmp {_CMP_FLT[o]} double {l}, {r}")
                    else:
                        lines.append(f"  {c} = icmp {_CMP_INT[o]} i64 {l}, {r}")
                    v = fresh()
                    lines.append(f"  {v} = zext i1 {c} to i64")
                    setval(dst, v, lines)
                elif o in _LOGIC:
                    lb, rb, v = fresh(), fresh(), fresh()
                    lines.append(f"  {lb} = icmp ne i64 {l}, 0")
                    lines.append(f"  {rb} = icmp ne i64 {r}, 0")
                    lines.append(f"  {v} = {_LOGIC[o]} i1 {lb}, {rb}")
                    z = fresh()
                    lines.append(f"  {z} = zext i1 {v} to i64")
                    setval(dst, z, lines)
                else:
                    mnem = _ARITH_FLT[o] if is_flt else _ARITH_INT[o]
                    v = fresh()
                    ty = "double" if is_flt else "i64"
                    lines.append(f"  {v} = {mnem} {ty} {l}, {r}")
                    setval(dst, v, lines)
            elif rk == "select":
                if dst in info.dead_results:
                    lines.append(f"  ; dead select elided: {dst} (never observed)")
                    continue
                c = use(opargs[0], lines)
                t = use(opargs[1], lines)
                e = use(opargs[2], lines)
                cb = fresh()
                lines.append(f"  {cb} = icmp ne i64 {c}, 0")
                v = fresh()
                ty = llty(dst)
                lines.append(f"  {v} = select i1 {cb}, {ty} {t}, {ty} {e}")
                setval(dst, v, lines)
            elif rk == "call":
                callee = rhs[1]
                # Locals shadow builtins (the interpreter's resolution
                # order), so the closure-call check comes first.
                if callee in info.def_count:
                    # Closure call: load {fn, env} from the pair and call
                    # fn(env, args...), typed with the pinned lambda's sig.
                    ck = kind(callee)
                    if not _is_closure(ck):
                        raise _Unsupported(
                            f"call through local {callee!r} that is not a "
                            "statically-known closure")
                    lname = _closure_lambda(ck)
                    if lname not in emitted_names:
                        raise _Unsupported(
                            f"closure call to non-emitted lambda {lname!r}")
                    csig = sigs[lname]
                    base = use(callee, lines)
                    fpp, envpp = fresh(), fresh()
                    lines.append(
                        f"  {fpp} = getelementptr inbounds {_CLOSURE_PAIR_TY}, "
                        f"ptr {base}, i32 0, i32 0")
                    fnv = fresh()
                    lines.append(f"  {fnv} = load ptr, ptr {fpp}")
                    lines.append(
                        f"  {envpp} = getelementptr inbounds {_CLOSURE_PAIR_TY}, "
                        f"ptr {base}, i32 0, i32 1")
                    envv = fresh()
                    lines.append(f"  {envv} = load ptr, ptr {envpp}")
                    avals = [f"ptr {envv}"]
                    for a, pk in zip(opargs, csig.params):
                        avals.append(f"{_llparam(pk)} {use(a, lines)}")
                    if _is_agg(csig.ret):
                        if dst not in aggset:
                            raise _Unsupported(
                                f"call result {dst!r} not aggregate-kinded for "
                                f"sret closure call to {lname!r}")
                        avals.insert(0, f"ptr {struct_ref(dst)}")
                        lines.append(f"  call void {fnv}({', '.join(avals)})")
                    else:
                        v = fresh()
                        rty = _LLTY[csig.ret]
                        lines.append(
                            f"  {v} = call {rty} {fnv}({', '.join(avals)})")
                        setval(dst, v, lines)
                elif callee in _PRINT_BUILTINS:
                    if len(opargs) == 1:
                        a = use(opargs[0], lines)
                        k = kind(opargs[0])
                        mod.print_helpers.add(k)
                        hn = {I64: "metaxu_print_i64", F64: "metaxu_print_f64",
                              STR: "metaxu_print_str"}[k]
                        lines.append(f"  call void @{hn}({_LLTY[k]} {a})")
                    else:
                        # 0 or 2+ args: one printf with space-joined per-kind
                        # directives, matching the interpreter's print(*args).
                        fmt = " ".join(
                            {I64: "%lld", F64: "%g", STR: "%s"}[kind(a)]
                            for a in opargs) + "\n"
                        avals = [f"{_LLTY[kind(a)]} {use(a, lines)}"
                                 for a in opargs]
                        g = mod.intern_string(fmt)
                        mod.uses_printf = True
                        r = fresh()
                        call_args = ", ".join([f"ptr {g}"] + avals)
                        lines.append(
                            f"  {r} = call i32 (ptr, ...) @printf({call_args})")
                    setval(dst, "0", lines)  # unit
                elif callee == "neg":
                    a = use(opargs[0], lines)
                    v = fresh()
                    if kind(dst) == F64:
                        lines.append(f"  {v} = fneg double {a}")
                    else:
                        lines.append(f"  {v} = sub i64 0, {a}")
                    setval(dst, v, lines)
                elif callee == "not":
                    a = use(opargs[0], lines)
                    c, v = fresh(), fresh()
                    lines.append(f"  {c} = icmp eq i64 {a}, 0")
                    lines.append(f"  {v} = zext i1 {c} to i64")
                    setval(dst, v, lines)
                elif callee in _MATH_EXTERNS:
                    mod.math_used.add(callee)
                    a = use(opargs[0], lines)
                    v = fresh()
                    lines.append(f"  {v} = call double @{callee}(double {a})")
                    setval(dst, v, lines)
                else:
                    if callee not in emitted_names:
                        raise _Unsupported(f"call to non-emitted function {callee!r}")
                    csig = sigs[callee]
                    avals = []
                    for a, pk in zip(opargs, csig.params):
                        # aggregate args pass their storage pointer; the callee
                        # byval-copies the aggregate in its entry prelude.
                        avals.append(f"{_llparam(pk)} {use(a, lines)}")
                    if _is_agg(csig.ret):
                        # sret-style: dst's own storage is the result slot.
                        if dst not in aggset:
                            raise _Unsupported(
                                f"call result {dst!r} not aggregate-kinded for "
                                f"sret call to {callee!r}")
                        avals.insert(0, f"ptr {struct_ref(dst)}")
                        lines.append(
                            f"  call void @{mangle(callee)}({', '.join(avals)})")
                    else:
                        v = fresh()
                        rty = _LLTY[csig.ret]
                        lines.append(
                            f"  {v} = call {rty} @{mangle(callee)}({', '.join(avals)})")
                        setval(dst, v, lines)
            elif rk == "alloc_struct":
                sname = rhs[1]
                if dst not in aggset:
                    raise _Unsupported(f"alloc_struct result {dst!r} not struct-kinded")
                for (fn_, fv) in opargs:
                    p = gep(sname, struct_ref(dst), fn_, lines)
                    fk = structs.field_kind(sname, fn_)
                    lines.append(f"  store {_LLTY[fk]} {use(fv, lines)}, ptr {p}")
            elif rk == "field_get":
                sname = _struct_name(kind(opargs[0]))
                base = use(opargs[0], lines)
                p = gep(sname, base, rhs[1], lines)
                fk = structs.field_kind(sname, rhs[1])
                v = fresh()
                lines.append(f"  {v} = load {_LLTY[fk]}, ptr {p}")
                setval(dst, v, lines)
            elif rk == "field_set":
                # Value semantics: dst = copy of base with one field updated.
                sname = _struct_name(kind(opargs[0]))
                if dst not in aggset:
                    raise _Unsupported(f"field_set result {dst!r} not struct-kinded")
                base = use(opargs[0], lines)
                agg_copy(f"%struct.{_sanitize(sname)}", base, struct_ref(dst), lines)
                p = gep(sname, struct_ref(dst), rhs[1], lines)
                fk = structs.field_kind(sname, rhs[1])
                lines.append(f"  store {_LLTY[fk]} {use(opargs[1], lines)}, ptr {p}")
            elif rk == "make_variant":
                # Tagged union: store the integer tag, then the payload slots.
                ename, vname = rhs[1], rhs[2]
                if dst not in aggset or not _is_enum(kind(dst)):
                    raise _Unsupported(
                        f"make_variant result {dst!r} not enum-kinded")
                tagv = variants.tags.get(vname)
                if tagv is None:
                    raise _Unsupported(f"no tag assigned for variant {vname!r}")
                mod.used_enums.add(ename)
                p = enum_gep(ename, struct_ref(dst), lines)
                lines.append(
                    f"  store i64 {tagv}, ptr {p}  ; tag {vname}={tagv}")
                for i, fv in enumerate(opargs):
                    ck = variants.cell_kind(ename, i)
                    p = enum_gep(ename, struct_ref(dst), lines, payload=i)
                    lines.append(
                        f"  store {_LLTY[ck]} {use(fv, lines)}, ptr {p}")
            elif rk == "variant_tag":
                bk = kind(opargs[0])
                if not _is_enum(bk):
                    raise _Unsupported(
                        f"variant_tag of non-enum value {opargs[0]!r}")
                base = use(opargs[0], lines)
                p = enum_gep(_enum_name(bk), base, lines)
                v = fresh()
                lines.append(f"  {v} = load i64, ptr {p}")
                setval(dst, v, lines)
            elif rk == "variant_field":
                bk = kind(opargs[0])
                if not _is_enum(bk):
                    raise _Unsupported(
                        f"variant_field of non-enum value {opargs[0]!r}")
                ename = _enum_name(bk)
                ck = variants.cell_kind(ename, rhs[1])
                base = use(opargs[0], lines)
                p = enum_gep(ename, base, lines, payload=rhs[1])
                v = fresh()
                lines.append(f"  {v} = load {_LLTY[ck]}, ptr {p}")
                setval(dst, v, lines)
            elif rk == "make_closure":
                # Fill this site's env struct with the captured values, then
                # store the {fn, env} pair into the closure variable.
                lname = rhs[1]
                if dst not in aggset or not _is_closure(kind(dst)):
                    raise _Unsupported(
                        f"make_closure result {dst!r} not closure-kinded")
                if lname not in emitted_names:
                    raise _Unsupported(
                        f"make_closure of non-emitted lambda {lname!r}")
                envp = env_allocas.get(id(op))
                if envp is None:  # unreachable: prescan covers every site
                    raise _Unsupported("make_closure site missing env storage")
                fields = env_fields(lname)
                mod.env_types[lname] = fields
                mod.uses_closure_pair = True
                ety = f"%env.{_sanitize(lname)}"
                for i, ((cn, ck), (_cn2, vn)) in enumerate(zip(fields, opargs)):
                    p = fresh()
                    lines.append(
                        f"  {p} = getelementptr inbounds {ety}, ptr {envp}, "
                        f"i32 0, i32 {i}")
                    if _is_agg(ck):
                        # aggregate capture: copy the whole value into the env
                        agg_copy(_agg_ty(ck), use(vn, lines), p, lines)
                    else:
                        lines.append(
                            f"  store {_LLTY[ck]} {use(vn, lines)}, ptr {p}")
                p0, p1 = fresh(), fresh()
                lines.append(
                    f"  {p0} = getelementptr inbounds {_CLOSURE_PAIR_TY}, "
                    f"ptr {struct_ref(dst)}, i32 0, i32 0")
                lines.append(f"  store ptr @{mangle(lname)}, ptr {p0}")
                lines.append(
                    f"  {p1} = getelementptr inbounds {_CLOSURE_PAIR_TY}, "
                    f"ptr {struct_ref(dst)}, i32 0, i32 1")
                lines.append(f"  store ptr {envp}, ptr {p1}")
            else:  # unreachable given analysis
                raise _Unsupported(f"op {rk!r} slipped past analysis")

        if not terminated:
            t = b.term
            if t[0] == "br":
                lines.append(f"  br label %bb{t[1]}")
            elif t[0] == "br_if":
                c = use(t[1], lines)
                cb = fresh()
                lines.append(f"  {cb} = icmp ne i64 {c}, 0")
                lines.append(f"  br i1 {cb}, label %bb{t[2]}, label %bb{t[3]}")
            elif t[0] == "ret":
                if sret:
                    if kind(t[1]) != sig.ret:
                        raise _Unsupported(
                            f"return value {t[1]!r} is {kind(t[1])}, "
                            f"function returns {sig.ret}")
                    # Copy the aggregate into the caller's slot BEFORE any
                    # frees (the returned value may live in a heap block).
                    src = use(t[1], lines)
                    agg_copy(_agg_ty(sig.ret), src, "%agg.ret", lines)
                    emit_frees(lines)
                    lines.append("  ret void")
                else:
                    rv = use(t[1], lines)
                    emit_frees(lines)
                    lines.append(f"  ret {_LLTY[sig.ret]} {rv}")
            else:  # ("unreachable",) placeholder terminator (no frees: dead end)
                lines.append("  unreachable")
        body.append(f"bb{bi}:")
        body.extend(lines)

    # Assemble: define header, entry block (allocas + heap mallocs + param
    # spills/byval-copies + capture loads), blocks.  A struct/enum return
    # prepends the caller's result slot as a leading `ptr %agg.ret`
    # parameter (sret-style); a lambda takes its env struct as a leading
    # `ptr %cl.env` parameter (after %agg.ret when both are present).
    pdecls = []
    if sret:
        pdecls.append("ptr %agg.ret")
    if info.is_lambda:
        pdecls.append("ptr %cl.env")
    for p, pk in zip(info.params, sig.params):
        pdecls.append(f"{_llparam(pk)} %a.{_sanitize(p)}")
    rty = "void" if sret else _LLTY[sig.ret]
    out = [f"define {rty} @{mangle(f.name)}({', '.join(pdecls)}) {{"]
    entry: List[str] = []
    for n in slots:
        entry.append(f"  {slot_ref(n)} = alloca {llty(n)}  ; mir slot: {n}")
    for n in agg_vars:
        if n in heapset:
            continue  # heap-backed: malloc'd below instead of an alloca
        entry.append(
            f"  {struct_ref(n)} = alloca {_agg_ty(kind(n))}  ; aggregate: {n}")
    entry.extend(env_entry)
    for n in heap_vars:
        sname = _struct_name(kind(n))
        size = 8 * len(structs.fields.get(sname, ()))  # all field kinds are 8 bytes
        mod.uses_malloc = True
        entry.append(
            f"  {struct_ref(n)} = call ptr @malloc(i64 {size})"
            f"  ; @global struct {n}: {sname}, freed on ret paths")
    for p in info.params:
        if p in aggset:
            # byval-copy: the caller passed a pointer to ITS storage; copy the
            # aggregate into this frame's own storage to preserve MIR value
            # semantics (a borrow-informed increment can elide this for
            # @const params).
            agg_copy(_agg_ty(kind(p)), f"%a.{_sanitize(p)}",
                     struct_ref(p), entry)
        elif p in slotset:
            entry.append(f"  store {llty(p)} %a.{_sanitize(p)}, ptr {slot_ref(p)}")
    if info.is_lambda:
        # Reload every capture from the env struct (creator stored them at
        # the make_closure site, eagerly, by value).
        fields = env_fields(f.name)
        mod.env_types[f.name] = fields
        ety = f"%env.{_sanitize(f.name)}"
        for i, (cn, ck) in enumerate(fields):
            p = f"%capp.{_sanitize(cn)}"
            entry.append(
                f"  {p} = getelementptr inbounds {ety}, ptr %cl.env, "
                f"i32 0, i32 {i}")
            if cn in aggset:
                agg_copy(_agg_ty(ck), p, struct_ref(cn), entry)
            elif cn in slotset:
                v = f"%capv.{_sanitize(cn)}"
                entry.append(f"  {v} = load {_LLTY[ck]}, ptr {p}")
                entry.append(f"  store {_LLTY[ck]} {v}, ptr {slot_ref(cn)}")
            else:
                entry.append(
                    f"  %cap.{_sanitize(cn)} = load {_LLTY[ck]}, ptr {p}")
    entry.append("  br label %bb0")
    out.append("entry:")
    out.extend(entry)
    out.extend(body)
    out.append("}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Runtime support text (string pool, printf helpers, declares, struct types)
# ---------------------------------------------------------------------------

_PRINT_FMTS = {
    "i64": ("@.fmt.i64", "%lld\n"),
    "f64": ("@.fmt.f64", "%g\n"),
    "str": ("@.fmt.str", "%s\n"),
}
_PRINT_ARG = {"i64": "i64", "f64": "double", "str": "ptr"}


def _string_global(name: str, content: str) -> str:
    data = content.encode("utf-8") + b"\x00"
    return (f"{name} = private unnamed_addr constant "
            f"[{len(data)} x i8] c\"{_escape_bytes(data)}\"")


def _emit_runtime(mod: _ModuleState) -> List[str]:
    chunks: List[str] = []
    if mod.strings:
        chunks.append("\n".join(
            _string_global(gname, content)
            for content, gname in sorted(mod.strings.items(), key=lambda kv: kv[1])))
    decls: List[str] = []
    if mod.print_helpers or mod.uses_printf:
        decls.append("declare i32 @printf(ptr, ...)")
    if mod.uses_abort:
        decls.append("declare void @abort() noreturn")
    if mod.uses_malloc:
        decls.append("declare noalias ptr @malloc(i64)")
        decls.append("declare void @free(ptr)")
    for name in sorted(mod.math_used):
        decls.append(f"declare double @{name}(double)")
    if decls:
        chunks.append("\n".join(decls))
    if mod.print_helpers:
        fmts = "\n".join(_string_global(g, f)
                         for k, (g, f) in _PRINT_FMTS.items() if k in mod.print_helpers)
        chunks.append(fmts)
        for k in sorted(mod.print_helpers):
            g, _ = _PRINT_FMTS[k]
            aty = _PRINT_ARG[k]
            chunks.append("\n".join([
                f"define internal void @metaxu_print_{k}({aty} %x) {{",
                "entry:",
                f"  %r = call i32 (ptr, ...) @printf(ptr {g}, {aty} %x)",
                "  ret void",
                "}",
            ]))
    return chunks


def _emit_struct_types(structs: _StructTable, used: Set[str]) -> Optional[str]:
    lines: List[str] = []
    for sname in sorted(used):
        if sname in structs.bad:
            continue
        ftys = ", ".join(_LLTY.get(structs.field_kind(sname, fn_), "i64")
                         for fn_ in structs.fields.get(sname, ()))
        fields_desc = ", ".join(structs.fields.get(sname, ()))
        lines.append(f"%struct.{_sanitize(sname)} = type {{ {ftys} }}  ; {fields_desc}")
    return "\n".join(lines) if lines else None


def _emit_enum_types(variants: _VariantTable, used: Set[str]) -> Optional[str]:
    """%enum.E tagged-union types plus the documented tag mapping."""
    lines: List[str] = []
    if used and variants.tags:
        pairs = ", ".join(f"{n}={i}" for n, i in sorted(
            variants.tags.items(), key=lambda kv: kv[1]))
        lines.append(f"; variant tag mapping (module-wide, dense): {pairs}")
    for ename in sorted(used):
        n = variants.payload_max.get(ename, 0)
        vnames = sorted(variants.variants_of.get(ename, ()))
        tag_doc = ", ".join(
            f"{v}={variants.tags[v]}" for v in vnames if v in variants.tags)
        lines.append(
            f"{_enum_llname(ename)} = type {{ i64, [{n} x i64] }}"
            f"  ; tag + {n} payload slots" + (f"; tags: {tag_doc}" if tag_doc else ""))
    return "\n".join(lines) if lines else None


def _emit_closure_types(mod: _ModuleState) -> Optional[str]:
    """%mx.closure pair + per-lambda %env.L structs (env field order is the
    make_closure capture-list order)."""
    lines: List[str] = []
    if mod.uses_closure_pair:
        lines.append(f"{_CLOSURE_PAIR_TY} = type {{ ptr, ptr }}  ; {{ fn, env }}")
    for lname in sorted(mod.env_types):
        fields = mod.env_types[lname]
        ftys = ", ".join(
            _agg_ty(k) if _is_agg(k) else _LLTY.get(k, "i64")
            for (_cn, k) in fields)
        desc = ", ".join(cn for (cn, _k) in fields)
        body = f"{{ {ftys} }}" if ftys else "{}"
        lines.append(
            f"%env.{_sanitize(lname)} = type {body}  ; captures: {desc or '(none)'}")
    return "\n".join(lines) if lines else None


# ---------------------------------------------------------------------------
# Module driver
# ---------------------------------------------------------------------------

def emit_llvm(funcs: Sequence[MirFunc]) -> str:
    """Emit one LLVM IR module (text) for a MIR module.

    Direct functions get full definitions; everything else gets a
    comment-only placeholder carrying its reasons (see module docstring).
    """
    module_names = {f.name for f in funcs}
    structs = _build_struct_table(funcs)
    closures = _build_closure_table(funcs)
    infos = [_analyze(f, module_names, closures) for f in funcs]
    variants = _build_variant_table(funcs, infos)

    def dep_names(info: _Info, kinds: Dict[str, str]) -> Set[str]:
        """Module functions this one references and cannot link without:
        direct callees, make_closure targets, and resolved closure callees."""
        deps = {callee for (_d, callee, _a) in info.calls
                if callee in module_names}
        deps |= {lname for (_d, lname, _c) in info.closure_defs
                 if lname in module_names}
        for (_d, cvar, _a) in info.closure_calls:
            ck = kinds.get(cvar, I64)
            if _is_closure(ck) and _closure_lambda(ck) in module_names:
                deps.add(_closure_lambda(ck))
        return deps

    # Duplicate MIR function names (e.g. lambda counters restarting per
    # enclosing function) would produce colliding symbols and ambiguous
    # direct calls: demote every function carrying a duplicated name.
    seen: Dict[str, int] = {}
    for f in funcs:
        seen[f.name] = seen.get(f.name, 0) + 1
    for info in infos:
        if seen[info.f.name] > 1:
            info.add_reason(
                f"duplicate function name {info.f.name!r} in module (ambiguous symbol)")

    # Module-wide kind/signature fixpoint (params/ret and struct field cells
    # promoted monotonically; callers and callees feed each other).
    sigs: Dict[str, _Sig] = {
        info.f.name: _Sig(params=[I64] * len(info.params)) for info in infos}
    kind_sets: Dict[str, Dict[str, str]] = {}
    candidates = [info for info in infos if not info.reasons]
    for _round in range(12):
        changed = False
        for info in candidates:
            kinds, cell_changed = _infer_kinds(info, sigs, structs, variants, closures)
            changed = changed or cell_changed
            if kind_sets.get(info.f.name) != kinds:
                kind_sets[info.f.name] = kinds
                changed = True
            own = sigs[info.f.name]
            for i, p in enumerate(info.params):
                nk = _join(own.params[i], kinds.get(p, I64))
                if nk != own.params[i]:
                    own.params[i] = nk
                    changed = True
            for r in info.ret_vars:
                nk = _join(own.ret, kinds.get(r, I64))
                if nk != own.ret:
                    own.ret = nk
                    changed = True
            resolved_calls = list(info.calls)
            for (dst, cvar, args) in info.closure_calls:
                ck = kinds.get(cvar, I64)
                if _is_closure(ck):
                    resolved_calls.append((dst, _closure_lambda(ck), args))
            for (dst, callee, args) in resolved_calls:
                csig = sigs.get(callee)
                if csig is None or len(csig.params) != len(args):
                    continue
                for i, a in enumerate(args):
                    nk = _join(csig.params[i], kinds.get(a, I64))
                    if nk != csig.params[i]:
                        csig.params[i] = nk
                        changed = True
                nk = _join(csig.ret, kinds.get(dst, I64))
                if nk != csig.ret:
                    csig.ret = nk
                    changed = True
        if not changed:
            break

    # Post-fixpoint consistency; anything wrong becomes a placeholder reason.
    for info in candidates:
        kinds = kind_sets.get(info.f.name, {})
        for p in _check_consistency(info, kinds, sigs, structs, variants,
                                    closures, module_names):
            info.add_reason(p)

    # A function referencing a placeholder cannot link: cascade demotion
    # (direct calls, make_closure fn-pointer targets, closure calls).
    emitted = {info.f.name for info in infos if not info.reasons}
    while True:
        demoted = False
        for info in infos:
            if info.reasons or info.f.name not in emitted:
                continue
            for dep in sorted(dep_names(info, kind_sets.get(info.f.name, {}))):
                if dep not in emitted:
                    info.add_reason(
                        f"calls function {dep!r} that is itself a placeholder")
                    emitted.discard(info.f.name)
                    demoted = True
                    break
        if not demoted:
            break

    # Emission (a late _Unsupported also demotes, then cascades once more).
    mod = _ModuleState()
    used_structs: Set[str] = set()
    emitted_chunks: Dict[str, str] = {}
    progress = True
    while progress:
        progress = False
        for info in infos:
            name = info.f.name
            if name not in emitted or name in emitted_chunks:
                continue
            kinds = kind_sets.get(name, {})
            try:
                chunk = _emit_function(info, kinds, sigs, structs, variants,
                                       closures, mod, emitted)
            except _Unsupported as exc:
                info.add_reason(exc.reason)
            except Exception as exc:  # never crash the pipeline
                info.add_reason(f"emission error: {type(exc).__name__}: {exc}")
            else:
                emitted_chunks[name] = chunk
                used_structs |= {_struct_name(k) for k in kinds.values() if _is_struct(k)}
                mod.used_enums |= {_enum_name(k) for k in kinds.values()
                                   if _is_enum(k)}
                if any(_is_closure(k) for k in kinds.values()):
                    mod.uses_closure_pair = True
                continue
            # Demotion during emission: drop this function and everything
            # already emitted that references it, then redo the affected ones.
            emitted.discard(name)
            for other in infos:
                if other.f.name in emitted and name in dep_names(
                        other, kind_sets.get(other.f.name, {})):
                    other.add_reason(
                        f"calls function {name!r} that is itself a placeholder")
                    emitted.discard(other.f.name)
                    emitted_chunks.pop(other.f.name, None)
            progress = True

    chunks: List[str] = [_HEADER]
    st = _emit_struct_types(structs, used_structs)
    if st:
        chunks.append(st)
    et = _emit_enum_types(variants, mod.used_enums)
    if et:
        chunks.append(et)
    ct = _emit_closure_types(mod)
    if ct:
        chunks.append(ct)
    chunks.extend(_emit_runtime(mod))
    for info in infos:
        chunk = emitted_chunks.get(info.f.name)
        if chunk is not None:
            chunks.append(chunk)
        else:
            chunks.append(_emit_placeholder(info, sigs[info.f.name]))
    return "\n\n".join(chunks)
