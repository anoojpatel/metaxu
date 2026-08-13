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

Increment 4 adds recursive/nested aggregates: enum payload slots holding
aggregates are HEAP-BOXED (making recursive enums like linked lists and
trees representable), struct fields holding structs/enums are INLINED in
the parent layout, and closures that escape (returned, or created in a
loop) get HEAP environments.  See the dedicated sections below — and note
the prominently documented free strategy: boxes and heap envs LEAK BY
DESIGN this increment.

Increment 5 links the NATIVE RUNTIME (src/metaxu/runtime/native/
metaxu_rt.c, built+linked by llvm_run) and lowers the vec/string builtins
to it:
  * Vec values are a new scalar-like kind family ``vec:ELEM``: an opaque
    ``mx_vec*`` pointer (8 bytes, stored like any ptr in struct fields,
    enum payload slots and closure envs) with IDENTITY semantics — every
    copy is a shallow pointer copy aliasing the one shared vector, exactly
    the interpreter's MxVec.  ELEM is the unified element kind; elements
    travel as opaque 8-byte words (f64 bitcast, str/vec ptrtoint) through
    mx_vec_push/mx_vec_pop/mx_vec_get.  ``Vec.new``->mx_vec_new,
    ``push``/``pop``->mx_vec_push/pop, ``__index_get``->mx_vec_get,
    ``len``->mx_vec_len (or mx_str_len on a string receiver).
  * ``to_string``/``int_to_str`` -> mx_i64_to_str / mx_f64_to_str /
    identity on a string.  CAVEAT (same kind-erasure divergence as print):
    bools and unit are erased to i64, so ``true.to_string()`` yields "1"
    natively where the interpreter says "True"; tests stringify ints,
    floats and strings.
  * string ``+`` -> mx_str_concat and ``==``/``!=`` -> mx_str_eq; concat
    and to_string results are fresh malloc'd strings that LEAK BY DESIGN
    (the box/heap-env contract; ordering comparisons on strings demote).
  * ``__trait$m`` calls resolve STATICALLY against the receiver's inferred
    kind, mirroring the interpreter's dispatch order (impl for the
    receiver's type name -> builtin -> plain function); an i64 receiver is
    kind-erased (int/bool/unit), so it resolves only when no impl exists
    for any of those type names.  Unresolvable dispatch demotes.
  * FREE STRATEGY for vecs: mx_vec_free at frame exit ONLY for vecs
    proven non-escaping by _provably_dead_vecs (created unconditionally in
    the entry block; never returned, stored, captured, or passed anywhere
    except as the receiver of the non-retaining vec builtins).  Everything
    else leaks by design — identity sharing makes any other free
    potentially a double-free.
  * sqrt/sin/cos remain plain libm externs (documented choice, see
    llvm_run) — no wrappers, no intrinsics.

Increment 6 replaces the per-(enum, slot) payload kind model with
PER-VARIANT, PER-VALUE payload typing so one enum can hold different
payload types in the same slot index — across variants (Leaf(int) |
Fork(Tree, Tree)) and across generic instantiations of one variant
(Some(3) and Some(node) in one module, linked_list.mx's reality).  See the
REFINED ENUM KINDS section below.  The MIR ``variant_field`` op now
carries the pattern's ctor name as a third element (lower_hir_to_mir;
interpreter and CLIF ignore it) so the backend knows WHICH variant's slot
a read refers to — chosen over recovering the variant from the dominating
tag test because compile_pattern interleaves nested sub-pattern blocks
between the tag branch and later slot reads, making a dominator trace
fragile exactly where it matters (nested ctor patterns).

Everything else — suspending functions (perform/resume/handle_scope: the
CPS lowering lives in codegen_clif for now), try_scope,
fixed-size vector literals/comprehensions/slices — is emitted as a
clearly marked, comment-only placeholder carrying the reasons, never as
silently wrong code.  Functions that call a placeholder function are
themselves demoted (the module must link), with an explicit reason.

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
        are threaded into codegen.  COPY-OUT (interpreter write-back
        parity): a struct param the callee REBINDS anywhere is copied back
        through the caller's pointer on every ret path, matching
        mir_interp._write_back_struct_args (`self.field = ...` methods
        mutate the caller's binding); lambdas never copy out because the
        interpreter's closure-call path performs no write-back.
      - struct returns are sret-style (the ONE convention used
        everywhere): the caller passes its result variable's storage as a
        leading ``ptr %agg.ret`` argument, the callee copies the returned
        aggregate into it and returns ``void``.  Small-struct returns as
        first-class LLVM aggregates were considered and rejected to keep
        one uniform path.  No ABI ``sret`` attribute is needed: all such
        calls are module-internal.
    NESTED AGGREGATE FIELDS (increment 4): a struct field whose kind is
    itself a struct or enum is laid out INLINE in the parent %struct.T
    (the field's LLVM type is the nested %struct.X / %enum.E), fully
    stack-based: alloc_struct copies the aggregate value into the field
    region, field_get copies it out into the destination's own storage,
    field_set copies the whole parent then overwrites the region — plain
    value semantics with recursive GEPs, no heap.  Struct sizes are
    computed recursively (every leaf cell is 8 bytes; enum payload slots
    are 8 bytes each — aggregates there are boxed pointers, see below), so
    @global malloc sizes stay exact.  A struct-in-struct cycle with no
    intervening enum box has no finite layout and demotes (such a value
    could never be constructed anyway).  A field holding a closure still
    demotes: the pair's env pointer may aim at a stack frame the struct
    could outlive.
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
  * variant values are aggregates with the same value semantics as
    structs: entry-block alloca per variant-kinded variable, aggregate
    copies, ptr + callee byval-copy across calls, sret-style returns.
    There is no @global form (make_variant carries no locality).

REFINED ENUM KINDS (increment 6) type payload slots per variant and per
VALUE, replacing the old module-wide per-(enum, slot) cells:
  * an enum value's kind carries a payload REFINEMENT recording the actual
    representation of every variant any flow into it can construct:
    ``enum:Option{None:;Some:struct:Node}`` (canonical form: variants
    sorted, slots comma-separated).  make_variant refines its destination
    with its own arg kinds; refinements join pointwise per (variant, slot)
    along every dataflow edge (copies, params/returns, struct fields,
    captures) — the same monotone fixpoint as all other kinds.  Slot kinds
    inside a refinement are NAME-ONLY (a nested enum appears as ``enum:F``
    with no braces), which keeps refinement strings finite for recursive
    enums.
  * ``variant_field`` reads its variant's slot kind out of the BASE
    VALUE's refinement (the op names its variant, see above).  Different
    variants — or different instantiations of one variant in different
    values — can now disagree about a slot index; each value knows its own
    representation.  Reading a variant absent from the refinement is a
    dead arm (the refinement lists every constructible variant, so the
    guarding tag test can never pass): scalar results emit a typed zero,
    never a bogus aggregate read.
  * SOUNDNESS: every dataflow edge except enum-in-enum nesting unifies
    kinds two-way, so a reader's refinement is exactly the join over its
    writers; the post-fixpoint make_variant check demotes any site whose
    stored kind differs from that join ("heterogeneous payload slot ...
    no coercion"), which now fires only for genuinely MERGED mixed flows
    (e.g. one variable holding both Some(3) and Some(node)), not for
    disjoint uses.
  * enum-in-enum nesting is the one boundary where a value's refinement is
    stripped (slot kinds are name-only): extraction therefore assumes the
    CANONICAL representation — module-wide per-(enum, variant, slot) cells
    joining every make_variant store — and demotes when any store
    disagrees with the join (`mixed` cells: the nested enum's slot
    representation is instantiation-dependent and was lost at the boxing
    boundary).
  * the union layout is unchanged and instantiation-independent:
    ``%enum.E = { i64 tag, [N x i64] }`` with N the max payload arity over
    all variants; every slot is an 8-byte cell (scalars inline, aggregates
    as boxed pointers), so all refinements of one enum share one LLVM
    type.  Boxing decisions are per (variant, slot) refinement kind.  The
    emitted type comment documents each variant's canonical slot kinds and
    flags mixed slots.

BOXED AGGREGATE PAYLOADS (increment 4; since increment 6 decided per
variant and per value, see REFINED ENUM KINDS below): a payload slot whose
kind
is itself a struct or enum stores a HEAP POINTER to a boxed copy of the
aggregate (the 8-byte slot holds the ptr), which makes recursive enums
(linked lists, trees) representable with a finite layout:
  * ``make_variant`` mallocs the box (exact recursive size) and copies the
    aggregate value in; ``variant_field`` loads the pointer and copies the
    aggregate out into the destination's own storage — value semantics are
    preserved at both edges.
  * boxes are WRITE-ONCE: the only store through a box pointer is the fill
    at the make_variant site.  Aggregate copies (variable defs, byval
    params, sret returns, env captures, boxing itself) copy the pointer
    shallowly, so boxes are freely shared — which is observationally
    equivalent to the interpreter's value semantics precisely because no
    native code path ever mutates a filled box (MIR has no payload-set op,
    and field_set copies its base aggregate wholesale).
  * FREE STRATEGY (the documented, prominently stated choice): boxes LEAK
    BY DESIGN this increment.  Shallow sharing means box ownership is not
    unique, so any per-value free would need deep copies at every
    aggregate copy to avoid double-frees; instead no box is ever freed.
    This is provably sound (a leak can never be a use-after-free or
    double-free); ASan tests for box programs therefore assert with
    ``detect_leaks=0`` — they prove no UAF/double-free, not leak-freedom.
    The @global struct malloc/free protocol is UNCHANGED and remains fully
    leak-clean; struct blocks are freed at frame exit while any boxes
    referenced from their field regions are simply leaked.
  * a payload slot holding a closure still demotes (its env pointer may
    aim at a stack frame the boxed value could outlive).

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
    live).

HEAP CLOSURE ENVIRONMENTS (increment 4): a lambda is marked heap-env when
its closure could outlive (or alias across re-executions of) the creating
site:
  * RETURNED closures: after the module kind fixpoint, any function whose
    return kind is ``closure:L`` marks L heap-env.  Kind propagation makes
    this complete for the emitted subset: a pair that reaches a caller
    reaches it through some emitted function's return (storing a pair in
    a struct field / enum payload / another env still demotes the storing
    function, so no other upward path exists).
  * LOOPED sites: a make_closure inside a CFG cycle marks L heap-env —
    each execution mallocs a FRESH env, so earlier pair copies never alias
    the new one (this replaces the increment-3 demotion).
  * a heap env is malloc'd at the make_closure site and NEVER FREED — the
    same leak-by-design contract as payload boxes (an immortal env can
    never dangle; that is what makes the escape analysis need only be
    conservative, never precise).  Non-escaping lambdas keep their
    zero-cost stack envs.
  * still demoted honestly: storing a closure in a struct field or variant
    payload, capturing a closure in another closure.

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
from .desugar import parse_impl_method_name
from .hir import TRAIT_CALL_PREFIX

# Value kinds -----------------------------------------------------------------

I64 = "i64"
F64 = "f64"
STR = "str"
CONFLICT = "conflict"
_STRUCT_PREFIX = "struct:"
_ENUM_PREFIX = "enum:"
_CLOSURE_PREFIX = "closure:"
# A growable Vec value: an opaque `mx_vec*` heap pointer (8 bytes) with
# IDENTITY semantics, parameterized by its unified element kind
# ("vec:i64" / "vec:f64" / "vec:str" / "vec:vec:i64" ...).  Copies are
# shallow pointer copies, exactly matching the interpreter's MxVec (every
# copy aliases the one shared vector).  Elements are opaque 8-byte words in
# the native runtime; the element kind decides the bitcast at push/pop/get.
_VEC_PREFIX = "vec:"

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

# Vec/string builtins now lowered to the NATIVE runtime (metaxu_rt.c, linked
# by llvm_run): these mirror the interpreter's builtins exactly.  Like the
# interpreter's resolution order ("builtins first" for plain calls), these
# names win over same-named module functions; local closure variables still
# shadow them.
#   Vec.new      -> mx_vec_new          push  -> mx_vec_push
#   pop          -> mx_vec_pop          len   -> mx_vec_len / mx_str_len
#   __index_get  -> mx_vec_get
#   to_string / int_to_str -> mx_i64_to_str / mx_f64_to_str / identity(str)
_NATIVE_RT_CALLS = {"Vec.new", "push", "pop", "len", "to_string",
                    "int_to_str", "__index_get"}

# Interpreter builtins that trait dispatch can fall back to when no user
# impl matches the receiver type (mir_interp._dispatch_trait_call step 2).
_TRAIT_BUILTIN_FALLBACK = {"to_string", "int_to_str", "len", "push", "pop",
                           "sqrt", "sin", "cos"}

# Callees still implemented only by the interpreter runtime (demote).
_RUNTIME_PREFIXES = ("__vec_", "__index_", "__slice_", "__range", "__static$")
_RUNTIME_NAMES = {"type_of", "assert_eq"}

# Native runtime symbol signatures (metaxu_rt.h ABI): name -> (ret, params).
_RT_SIGS = {
    "mx_vec_new": ("ptr", ()),
    "mx_vec_push": ("void", ("ptr", "i64")),
    "mx_vec_pop": ("i64", ("ptr",)),
    "mx_vec_len": ("i64", ("ptr",)),
    "mx_vec_get": ("i64", ("ptr", "i64")),
    "mx_vec_set": ("void", ("ptr", "i64", "i64")),
    "mx_vec_free": ("void", ("ptr",)),
    "mx_str_concat": ("ptr", ("ptr", "ptr")),
    "mx_str_len": ("i64", ("ptr",)),
    "mx_i64_to_str": ("ptr", ("i64",)),
    "mx_f64_to_str": ("ptr", ("double",)),
    "mx_str_eq": ("i64", ("ptr", "ptr")),
}

_I64_MIN, _I64_MAX = -(2 ** 63), 2 ** 63 - 1

_HEADER = (
    "; LLVM IR emitted by metaxu codegen_llvm (direct subset)\n"
    "; conventions: ints/bools/unit -> i64 (unit = 0); floats -> double;\n"
    ";   strings -> ptr to private constant byte arrays; cmp results zext to i64;\n"
    ";   &&/|| normalize operands with icmp ne 0 (truthiness, not bitwise);\n"
    ";   / and % are sdiv/srem (trunc toward zero; interpreter floors);\n"
    ";   local structs -> %struct.T entry allocas + GEP (value semantics,\n"
    ";   whole-aggregate copies; zero heap management -- the frame owns them);\n"
    ";   @global structs -> entry-block malloc(recursive layout size) + GEP\n"
    ";   on the heap\n"
    ";   pointer, freed on every ret path: sound because value semantics means\n"
    ";   the storage pointer never escapes the frame (aggregates cross frames\n"
    ";   by copy).  Any storage whose lifetime cannot be proven leaks by\n"
    ";   design rather than risking a double-free/use-after-free;\n"
    ";   struct params pass as ptr + callee byval-copy into own storage\n"
    ";   (a borrow-informed increment can elide the copy for @const params);\n"
    ";   rebound struct params copy OUT through the caller's pointer on ret\n"
    ";   (interpreter write-back parity; lambdas never copy out);\n"
    ";   struct returns are sret-style: caller passes its result slot as a\n"
    ";   leading ptr %agg.ret arg, callee copies the aggregate in, rets void;\n"
    ";   print/println route by operand type to @metaxu_print_{i64,f64,str};\n"
    ";   multi-arg print joins per-kind printf directives with spaces;\n"
    ";   metaxu symbols are prefixed mx_ to stay clear of libc names;\n"
    ";   enums -> %enum.E = { i64 tag, [N x i64] payload } tagged unions,\n"
    ";   variant names mapped to dense integer tags (module-wide table in a\n"
    ";   comment below); pattern tag tests compare integers, never strings;\n"
    ";   payload slots are typed PER VARIANT and PER VALUE (each enum\n"
    ";   value's kind carries a refinement recording its constructible\n"
    ";   variants' slot representations), so one slot index may hold\n"
    ";   different types in different variants or instantiations; merged\n"
    ";   flows that mix representations inside one value still demote;\n"
    ";   closures -> %mx.closure = { ptr fn, ptr env } pairs over per-lambda\n"
    ";   stack %env.L structs; lambdas take env as a leading param;\n"
    ";   escaping closures (returned / created in a loop) malloc their env\n"
    ";   at the site instead (heap env, leaked by design: immortal envs\n"
    ";   cannot dangle);\n"
    ";   struct fields holding structs/enums are laid out INLINE in the\n"
    ";   parent type (recursive GEPs, still stack-based, value semantics);\n"
    ";   enum payload slots holding aggregates store a HEAP POINTER to a\n"
    ";   write-once boxed copy (make_variant boxes in, variant_field copies\n"
    ";   out).  FREE STRATEGY: payload boxes and heap closure envs LEAK BY\n"
    ";   DESIGN (never freed) -- shallow pointer sharing makes ownership\n"
    ";   non-unique, and a leak is provably sound where a free is not.\n"
    ";   @global struct blocks are still freed at frame exit as before;\n"
    ";   Vec values -> opaque mx_vec* pointers (native runtime metaxu_rt.c,\n"
    ";   linked by llvm_run) with IDENTITY semantics (shallow ptr copies,\n"
    ";   matching the interpreter's MxVec); elements are opaque 8-byte\n"
    ";   words (f64 bitcast, str/vec ptrtoint) through mx_vec_push/pop/get;\n"
    ";   len routes by kind to mx_vec_len/mx_str_len; to_string routes to\n"
    ";   mx_i64_to_str/mx_f64_to_str/identity; string + is mx_str_concat,\n"
    ";   ==/!= is mx_str_eq.  Concat/to_string results leak by design; a\n"
    ";   Vec is mx_vec_free'd at frame exit only when provably\n"
    ";   non-escaping, otherwise it leaks by design too;\n"
    ";   __trait$ method calls are resolved statically against the\n"
    ";   receiver's inferred kind (impl fn -> builtin -> plain fn),\n"
    ";   mirroring the interpreter's runtime dispatch;\n"
    "; functions outside the subset appear as comment-only placeholders."
)


def _llparam(kind: str) -> str:
    """The LLVM parameter/return-slot type for a value kind (aggregates -> ptr)."""
    if _is_agg(kind) or _is_vec(kind):
        return "ptr"
    return _LLTY.get(kind, "i64")


def _llscalar(kind: str) -> str:
    """The LLVM type of a non-aggregate (register-sized) value kind.
    Vec values are opaque `mx_vec*` pointers."""
    if _is_vec(kind):
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
    """The enum's name, with any payload refinement suffix stripped
    ('enum:Option{None:;Some:i64}' -> 'Option')."""
    base = kind[len(_ENUM_PREFIX):]
    brace = base.find("{")
    return base if brace < 0 else base[:brace]


def _strip_refinement(kind: str) -> str:
    """The name-only form of a kind, as stored in module-wide variant cells
    and inside refinement strings: refined enum kinds drop their refinement;
    every other kind is unchanged.  Keeping nested enum references name-only
    is what keeps refinement strings finite for recursive enums."""
    if _is_enum(kind):
        return _ENUM_PREFIX + _enum_name(kind)
    return kind


def _enum_refinement(kind: str) -> Optional[Dict[str, Tuple[str, ...]]]:
    """Parse an enum kind's per-variant payload refinement, or None when the
    kind is name-only.  Format (canonical, variants sorted):
    'enum:E{VarA:kind0,kind1;VarB:}' — slot kinds are themselves name-only
    (no nested braces), so plain ;/,-splitting is exact."""
    if not _is_enum(kind):
        return None
    base = kind[len(_ENUM_PREFIX):]
    brace = base.find("{")
    if brace < 0:
        return None
    body = base[brace + 1:-1]
    ref: Dict[str, Tuple[str, ...]] = {}
    if not body:
        return ref
    for part in body.split(";"):
        vname, _, slots = part.partition(":")
        ref[vname] = tuple(s for s in slots.split(",") if s)
    return ref


def _format_enum_kind(ename: str, ref: Dict[str, Tuple[str, ...]]) -> str:
    """The canonical refined enum kind string (variants sorted by name), so
    kind-string equality is representation equality."""
    body = ";".join(f"{v}:{','.join(ref[v])}" for v in sorted(ref))
    return f"{_ENUM_PREFIX}{ename}{{{body}}}"


def _is_closure(kind: str) -> bool:
    return kind.startswith(_CLOSURE_PREFIX)


def _closure_lambda(kind: str) -> str:
    return kind[len(_CLOSURE_PREFIX):]


def _is_vec(kind: str) -> bool:
    return kind.startswith(_VEC_PREFIX)


def _vec_elem(kind: str) -> str:
    """The element kind of a vec kind ('vec:f64' -> 'f64')."""
    return kind[len(_VEC_PREFIX):]


def _vec_of(elem: str) -> str:
    return _VEC_PREFIX + elem


def _is_word_kind(kind: str) -> bool:
    """Kinds storable as an opaque 8-byte word in a Vec element slot."""
    return kind in (I64, F64, STR) or _is_vec(kind)


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


def _llcell(kind: str) -> str:
    """The LLVM type of an INLINE storage cell for a kind: scalars map via
    _llscalar (vec -> ptr), aggregates inline their named type (struct
    fields, env fields)."""
    return _agg_ty(kind) if _is_agg(kind) else _llscalar(kind)


def _join(a: str, b: str) -> str:
    """Kind lattice: i64 is bottom; f64/str/struct:T are incomparable tops.
    Vec kinds join pointwise on their element kind (vec:i64 is the vec
    bottom: a fresh Vec.new before any push).  Refined enum kinds of the
    same enum join their refinements pointwise per (variant, slot): the
    refinement records the value's actual payload REPRESENTATION, so a
    per-slot conflict (or a variant-arity mismatch) conflicts the whole
    kind.  A name-only enum kind meeting a refined one is a CONFLICT, not a
    bottom: name-only enum kinds never arise as value kinds in well-formed
    flows (make_variant and variant_field always produce refined kinds), so
    treating one as 'no information' could let two representations alias."""
    if a == b:
        return a
    if a == I64:
        return b
    if b == I64:
        return a
    if _is_vec(a) and _is_vec(b):
        e = _join(_vec_elem(a), _vec_elem(b))
        return CONFLICT if e == CONFLICT else _vec_of(e)
    if _is_enum(a) and _is_enum(b):
        ename = _enum_name(a)
        if ename != _enum_name(b):
            return CONFLICT
        ra, rb = _enum_refinement(a), _enum_refinement(b)
        if ra is None or rb is None:
            return CONFLICT
        merged: Dict[str, Tuple[str, ...]] = dict(ra)
        for v, slots in rb.items():
            cur = merged.get(v)
            if cur is None:
                merged[v] = slots
                continue
            if len(cur) != len(slots):
                return CONFLICT
            js = tuple(_join(x, y) for x, y in zip(cur, slots))
            if CONFLICT in js:
                return CONFLICT
            merged[v] = js
        return _format_enum_kind(ename, merged)
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
    # Runtime-dispatched trait method calls: (dst, method name, args) —
    # statically resolved against the receiver's inferred kind.
    trait_calls: List[Tuple[str, str, Tuple[str, ...]]] = field(default_factory=list)
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
                    # Re-executing a stack-env site would overwrite storage
                    # earlier pair copies may still alias: give this lambda a
                    # fresh heap env per execution instead (leaked by design).
                    closures.heap_env.add(lname)
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
    # interpreter's shadowing order: locals first), trait-dispatched calls
    # (resolved statically against the receiver kind later), native runtime
    # builtins (which, like the interpreter's "builtins first" order, win
    # over same-named module functions), and direct calls, which must hit
    # module functions or the supported builtins.
    direct_calls: List[Tuple[str, str, Tuple[str, ...]]] = []
    for (dst, callee, cargs) in info.calls:
        if callee in info.def_count:
            info.closure_calls.append((dst, callee, cargs))
        elif callee.startswith(TRAIT_CALL_PREFIX):
            method = callee[len(TRAIT_CALL_PREFIX):]
            if not cargs:
                info.add_reason(
                    f"trait method call {method!r} with no receiver")
            else:
                info.trait_calls.append((dst, method, cargs))
        elif callee in _NATIVE_RT_CALLS:
            direct_calls.append((dst, callee, cargs))
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
    # (enum name, variant name, slot index) -> NAME-ONLY kind: the join of
    # every make_variant store into that variant's slot module-wide.  These
    # cells are the CANONICAL representation used when a value crosses a
    # nesting boundary (an enum boxed inside another enum's payload slot
    # loses its per-value refinement); they are only sound to read through
    # when no store disagrees with the join (see `mixed`).
    cells: Dict[Tuple[str, str, int], str] = field(default_factory=dict)
    # enum name -> variant names seen in make_variant ops.
    variants_of: Dict[str, Set[str]] = field(default_factory=dict)
    # (enum name, variant name) -> payload arity from make_variant sites.
    arity: Dict[Tuple[str, str], int] = field(default_factory=dict)
    # enum name -> reason (inconsistent construction arity etc.).
    bad: Dict[str, str] = field(default_factory=dict)
    # (enum, variant, slot) cells where some post-fixpoint make_variant
    # store kind differs from the joined cell kind: the slot's native
    # representation is per-value (instantiation-dependent), so reading it
    # through the canonical cells would guess.  Filled by the module driver
    # after the kind fixpoint.
    mixed: Set[Tuple[str, str, int]] = field(default_factory=set)

    def cell_kind(self, ename: str, vname: str, idx: int) -> str:
        return self.cells.get((ename, vname, idx), I64)

    def mark_cell(self, ename: str, vname: str, idx: int, kind: str) -> bool:
        key = (ename, vname, idx)
        cur = self.cells.get(key, I64)
        nk = _join(cur, kind)
        if nk != cur:
            self.cells[key] = nk
            return True
        return False

    def canon(self, ename: str) -> Dict[str, Tuple[str, ...]]:
        """The canonical refinement of an enum: every constructed variant,
        each slot at its module-wide cell kind.  Monotone over the fixpoint
        (variants_of/arity are fixed at table build; cells only promote)."""
        ref: Dict[str, Tuple[str, ...]] = {}
        for v in self.variants_of.get(ename, ()):
            n = self.arity.get((ename, v), 0)
            ref[v] = tuple(self.cell_kind(ename, v, i) for i in range(n))
        return ref

    def canon_kind(self, ename: str) -> str:
        return _format_enum_kind(ename, self.canon(ename))

    def mixed_slots_of(self, ename: str) -> List[Tuple[str, str, int]]:
        return sorted(k for k in self.mixed if k[0] == ename)


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
                key = (ename, vname)
                if key not in table.arity:
                    table.arity[key] = len(op[3])
                elif table.arity[key] != len(op[3]):
                    table.bad.setdefault(
                        ename,
                        f"variant {vname!r} of enum {ename or 'anon'!r} "
                        f"constructed with inconsistent payload arities "
                        f"({table.arity[key]} vs {len(op[3])})")
    for info in infos:
        names.update(info.tag_consts.values())
    table.tags = {n: i for i, n in enumerate(sorted(names))}
    return table


# ---------------------------------------------------------------------------
# Recursive size computation (nested layouts + boxed payload slots)
# ---------------------------------------------------------------------------

def _kind_size(kind: str, structs: "_StructTable", variants: "_VariantTable",
               _seen: Tuple[str, ...] = ()) -> Optional[int]:
    """Byte size of a value kind's storage, or None for an infinite layout.

    Scalars are 8 bytes; a closure pair is 16; an enum is 8 (tag) + 8 per
    payload slot (aggregate slots hold an 8-byte box POINTER, which is what
    breaks recursion); a struct is the sum of its inline field sizes.  Only
    struct-in-struct chains recurse, so a cycle there (no intervening enum
    box) has no finite layout and returns None."""
    if _is_struct(kind):
        if kind in _seen:
            return None
        sname = _struct_name(kind)
        total = 0
        for fn_ in structs.fields.get(sname, ()):
            fs = _kind_size(structs.field_kind(sname, fn_), structs, variants,
                            _seen + (kind,))
            if fs is None:
                return None
            total += fs
        return total
    if _is_enum(kind):
        return 8 + 8 * variants.payload_max.get(_enum_name(kind), 0)
    if _is_closure(kind):
        return 16
    return 8  # i64 / f64 / str-ptr (and the CONFLICT sentinel, never emitted)


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
    # Lambdas whose envs are malloc'd at the site (leaked by design) because
    # their closures may escape: returned anywhere in the module, or created
    # inside a CFG cycle.  Membership is conservative and only ever ADDS heap
    # allocation — a heap env is always sound (it can never dangle).
    heap_env: Set[str] = field(default_factory=set)

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
# Module-wide trait-impl table and static trait-call resolution
# ---------------------------------------------------------------------------
#
# `recv.m(args)` lowers to a `__trait$m` call; the interpreter dispatches on
# the RECEIVER'S RUNTIME TYPE NAME (mir_interp._dispatch_trait_call):
#   1. __impl$Trait$Type$m for the receiver's type (exact, then
#      case-insensitive);  2. builtin m;  3. plain function m;  4. error.
# Kind inference gives us the receiver's kind statically, so the same
# resolution runs at compile time: struct:T/enum:E map to type name T/E,
# vec to "Vec", str to "String", f64 to "Float".  A receiver whose kind is
# i64 is AMBIGUOUS (ints, bools and unit are all kind-erased to i64), so an
# i64 receiver resolves to a builtin only when NO impl exists for any of
# Int/Bool/Unit — otherwise the native dispatch could pick a different
# target than the interpreter and the function demotes instead.

@dataclass
class _TraitTable:
    # method -> type name -> {trait name: impl function name}
    by_method: Dict[str, Dict[str, Dict[str, str]]] = field(default_factory=dict)


def _build_trait_table(module_names: Set[str]) -> _TraitTable:
    table = _TraitTable()
    for name in module_names:
        parsed = parse_impl_method_name(name)
        if parsed is None:
            continue
        trait_name, type_name, method = parsed
        table.by_method.setdefault(method, {}).setdefault(
            type_name, {})[trait_name] = name
    return table


# Scalar type names the interpreter reports for kind-erased i64 receivers.
_I64_RUNTIME_TYPE_NAMES = {"int", "bool", "unit"}


def _resolve_trait_call(method: str, recv_kind: str, traits: _TraitTable,
                        module_names: Set[str], *, assume_final: bool,
                        ) -> Tuple[str, Optional[str]]:
    """Statically resolve a `__trait$method` call for a receiver kind.

    Returns one of:
      ("func", fname)     -- call the module function fname directly
      ("builtin", name)   -- lower as the interpreter builtin `name`
      ("pending", None)   -- receiver kind still i64/bottom; try again later
                             (only when not assume_final)
      ("demote", reason)  -- cannot be resolved soundly; demote the function
    The mapping mirrors mir_interp._dispatch_trait_call exactly; anything it
    would decide differently at runtime returns "demote", never a guess.
    """
    by_type = traits.by_method.get(method, {})

    def impl_for(tyname: str) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
        """(trait->fn map, problem).  Exact name first, then case-insensitive
        (the interpreter's order); two distinct case-folded keys demote."""
        tm = by_type.get(tyname)
        if tm is not None:
            return tm, None
        low = tyname.lower()
        matches = [tm for k, tm in by_type.items() if k.lower() == low]
        if len(matches) > 1:
            return None, (f"trait method {method!r}: multiple case-folded "
                          f"impl types match receiver type {tyname!r}")
        return (matches[0] if matches else None), None

    def from_impl(tyname: str) -> Optional[Tuple[str, Optional[str]]]:
        tm, prob = impl_for(tyname)
        if prob is not None:
            return ("demote", prob)
        if tm is None:
            return None
        if len(tm) > 1:
            opts = ", ".join(sorted(tm))
            return ("demote",
                    f"ambiguous trait method {method!r} on type {tyname!r} "
                    f"(implemented by traits: {opts})")
        return ("func", next(iter(tm.values())))

    def plain_fn_fallback(reason: str) -> Tuple[str, Optional[str]]:
        # Interpreter step 3: a plain user function of the same name — but
        # ONLY when the method is not a builtin (builtins win at step 2).
        if method not in _TRAIT_BUILTIN_FALLBACK and method in module_names:
            return ("func", method)
        return ("demote", reason)

    if recv_kind == CONFLICT:
        return ("demote", f"trait method {method!r} receiver has conflicting kinds")

    if _is_struct(recv_kind) or _is_enum(recv_kind):
        tyname = _struct_name(recv_kind) if _is_struct(recv_kind) \
            else _enum_name(recv_kind)
        hit = from_impl(tyname)
        if hit is not None:
            return hit
        # Builtin fallback on an aggregate receiver: to_string would use
        # Python str() of the struct/variant, len/push/pop would raise —
        # neither is representable natively.
        if method in _TRAIT_BUILTIN_FALLBACK:
            return ("demote",
                    f"trait method {method!r} on {recv_kind} falls back to the "
                    "interpreter builtin (not representable natively)")
        return plain_fn_fallback(
            f"trait method {method!r} has no impl for type {tyname!r}")

    if _is_vec(recv_kind):
        hit = from_impl("Vec")
        if hit is not None:
            return hit
        if method in ("push", "pop", "len"):
            return ("builtin", method)
        if method in ("to_string", "int_to_str"):
            return ("demote",
                    "to_string of a Vec (interpreter renders 'Vec[...]'; no "
                    "native equivalent)")
        return plain_fn_fallback(
            f"trait method {method!r} on a Vec receiver has no native lowering")

    if recv_kind == STR:
        hit = from_impl("String")
        if hit is not None:
            return hit
        if method == "len":
            return ("builtin", "len")
        if method in ("to_string", "int_to_str"):
            return ("builtin", "to_string")
        return plain_fn_fallback(
            f"trait method {method!r} on a string receiver has no native lowering")

    if recv_kind == F64:
        hit = from_impl("Float")
        if hit is not None:
            return hit
        if method in _MATH_EXTERNS:
            return ("builtin", method)
        if method in ("to_string", "int_to_str"):
            return ("builtin", "to_string")
        return plain_fn_fallback(
            f"trait method {method!r} on a float receiver has no native lowering")

    if _is_closure(recv_kind):
        return ("demote", f"trait method {method!r} on a closure receiver")

    # recv_kind == I64: bottom (still unresolved) OR genuinely int/bool/unit.
    if not assume_final:
        return ("pending", None)
    erased = [t for t in by_type if t.lower() in _I64_RUNTIME_TYPE_NAMES]
    if erased:
        return ("demote",
                f"trait method {method!r} has impls for kind-erased scalar "
                f"receiver types ({', '.join(sorted(erased))}); i64 receivers "
                "cannot be dispatched statically")
    if method in _MATH_EXTERNS:
        return ("builtin", method)
    if method in ("to_string", "int_to_str"):
        return ("builtin", "to_string")
    if method in ("push", "pop", "len"):
        return ("demote",
                f"trait method {method!r} on a receiver of kind i64 "
                "(interpreter would reject a non-Vec receiver)")
    return plain_fn_fallback(
        f"trait method {method!r} on a receiver of kind i64 has no native lowering")


# ---------------------------------------------------------------------------
# Kind inference (i64 by default, monotone promotion; module fixpoint)
# ---------------------------------------------------------------------------

@dataclass
class _Sig:
    params: List[str]
    ret: str = I64


def _infer_kinds(info: _Info, sigs: Dict[str, _Sig], structs: _StructTable,
                 variants: _VariantTable, closures: _ClosureTable,
                 traits: _TraitTable, module_names: Set[str],
                 assume_final: bool = False,
                 ) -> Tuple[Dict[str, str], bool]:
    """One inner fixpoint over a function.  Returns (kinds, global_changed)
    where global_changed reports promotions written into shared cells
    (struct fields, enum payload slots, closure captures) so the module
    driver keeps iterating.

    ``assume_final``: trait-call receivers still at the i64 bottom are
    treated as genuinely-int receivers (the driver sets this only after the
    unassuming fixpoint has converged, so nothing else can promote them)."""
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

    # Field name -> owning struct, when the module has EXACTLY one struct
    # with that field: a field_get/field_set receiver still at the i64
    # bottom can then only be that struct (any other value would be an
    # interpreter error), so its kind is pinned.  Ambiguous names stay
    # unresolved and demote via the existing consistency check.
    field_owner: Dict[str, Optional[str]] = {}
    for sname_, fields_ in structs.fields.items():
        for fn_ in fields_:
            field_owner[fn_] = None if fn_ in field_owner else sname_

    def apply_builtin(name: str, dst: str, args: Tuple[str, ...],
                      plain_call: bool) -> bool:
        """Kind constraints of a native runtime builtin call.  For PLAIN
        calls (interpreter resolution: builtins first) receiver kinds are
        pinned eagerly; for trait-resolved calls the receiver is already
        known to be a vec/str (resolution is kind-driven)."""
        ch = False
        if name == "Vec.new":
            ch = mark(dst, _vec_of(I64)) or ch
        elif name in ("push", "pop", "__index_get"):
            if not args:
                return ch
            if plain_call:
                ch = mark(args[0], _vec_of(I64)) or ch
            rk = get(args[0])
            if _is_vec(rk):
                # Two-way element unification: pushed values and read
                # elements are one type per vec.  (push's own dst is unit.)
                if name == "push":
                    other = args[1] if len(args) == 2 else None
                else:
                    other = dst
                if other is not None:
                    nk = _join(_vec_elem(rk), get(other))
                    if nk != CONFLICT:
                        ch = mark(args[0], _vec_of(nk)) or ch
                        ch = mark(other, nk) or ch
                    else:
                        ch = mark(args[0], CONFLICT) or ch
        elif name == "len":
            pass  # receiver may be vec or str; dst stays i64
        elif name in ("to_string", "int_to_str"):
            ch = mark(dst, STR) or ch
        elif name in _MATH_EXTERNS:
            for a in args:
                ch = mark(a, F64) or ch
            ch = mark(dst, F64) or ch
        return ch

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
                    elif callee.startswith(TRAIT_CALL_PREFIX):
                        method = callee[len(TRAIT_CALL_PREFIX):]
                        if args:
                            res, target = _resolve_trait_call(
                                method, get(args[0]), traits, module_names,
                                assume_final=assume_final)
                            if res == "func":
                                sig = sigs.get(target)
                                if sig is not None and len(sig.params) == len(args):
                                    for a, pk in zip(args, sig.params):
                                        changed = mark(a, pk) or changed
                                    changed = mark(dst, sig.ret) or changed
                            elif res == "builtin":
                                changed = apply_builtin(
                                    target, dst, args, plain_call=False) or changed
                    elif callee in _NATIVE_RT_CALLS:
                        # Interpreter resolution order: builtins first, so
                        # these win over same-named module functions.
                        changed = apply_builtin(
                            callee, dst, args, plain_call=True) or changed
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
                    ename, vname = rhs[1], rhs[2]
                    # The dst kind carries this site's refinement: the actual
                    # per-slot representation of the value built here (nested
                    # enum payloads are recorded name-only; their contents go
                    # through the canonical module cells instead).
                    site_ref = {vname: tuple(_strip_refinement(get(a))
                                             for a in args)}
                    changed = mark(
                        dst, _format_enum_kind(ename, site_ref)) or changed
                    # One-way: store kinds also accumulate into the
                    # module-wide per-(variant, slot) cells backing canon();
                    # the driver marks cells `mixed` post-fixpoint when a
                    # store disagrees with the join.
                    for i, a in enumerate(args):
                        if variants.mark_cell(ename, vname, i,
                                              _strip_refinement(get(a))):
                            changed = global_changed = True
                elif rk == "variant_tag":
                    pass  # dst is the integer tag: i64 (the default)
                elif rk == "variant_field":
                    bk = get(args[0])
                    vname = rhs[2] if len(rhs) > 2 else None
                    if _is_enum(bk) and vname is not None:
                        ref = _enum_refinement(bk) or {}
                        slots = ref.get(vname)
                        if slots is not None and rhs[1] < len(slots):
                            sk = slots[rhs[1]]
                            if _is_enum(sk):
                                # Nested enum extraction crosses a boxing
                                # boundary: the boxed value's own refinement
                                # was stripped, so the result assumes the
                                # canonical module-wide representation (the
                                # consistency check demotes if any store
                                # disagrees with it).
                                sk = variants.canon_kind(_enum_name(sk))
                            changed = mark(dst, sk) or changed
                        # variant absent from the refinement: the arm is
                        # dead (no flow constructs it); dst stays at bottom.
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
                    if bk == I64 and field_owner.get(rhs[1]):
                        changed = mark(
                            args[0],
                            _STRUCT_PREFIX + field_owner[rhs[1]]) or changed
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
                    if bk == I64 and field_owner.get(rhs[1]):
                        changed = mark(
                            args[0],
                            _STRUCT_PREFIX + field_owner[rhs[1]]) or changed
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
                       closures: _ClosureTable, traits: _TraitTable,
                       module_names: Set[str]) -> List[str]:
    probs: List[str] = []

    def ty(n: str) -> str:
        return kinds.get(n, I64)

    def check_builtin(name: str, dst: str, args: Tuple[str, ...]) -> None:
        """Validate a native-runtime builtin call's final kinds."""
        if name == "Vec.new":
            if args:
                probs.append("Vec.new with arguments")
            elif not _is_vec(ty(dst)):
                probs.append(
                    f"Vec.new result {dst!r} has kind {ty(dst)}, not a Vec")
        elif name == "push":
            if len(args) != 2:
                probs.append(f"push with {len(args)} arguments (expects 2)")
            elif not _is_vec(ty(args[0])):
                probs.append(
                    f"push receiver {args[0]!r} has kind {ty(args[0])}, not a Vec")
            elif not _is_word_kind(_vec_elem(ty(args[0]))):
                probs.append(
                    f"Vec of {_vec_elem(ty(args[0]))} elements (only 8-byte "
                    "word kinds fit native Vec slots)")
            elif ty(args[1]) != _vec_elem(ty(args[0])):
                probs.append(
                    f"push of {ty(args[1])} into a Vec of "
                    f"{_vec_elem(ty(args[0]))}")
        elif name in ("pop", "__index_get"):
            want = 1 if name == "pop" else 2
            if len(args) != want:
                probs.append(f"{name} with {len(args)} arguments (expects {want})")
            elif not _is_vec(ty(args[0])):
                probs.append(
                    f"{name} receiver {args[0]!r} has kind {ty(args[0])}, "
                    "not a Vec (string/vector indexing stays interpreted)")
            elif name == "__index_get" and ty(args[1]) != I64:
                probs.append(f"__index_get index {args[1]!r} is {ty(args[1])}")
            elif not _is_word_kind(_vec_elem(ty(args[0]))):
                probs.append(
                    f"Vec of {_vec_elem(ty(args[0]))} elements (only 8-byte "
                    "word kinds fit native Vec slots)")
            elif ty(dst) != _vec_elem(ty(args[0])):
                probs.append(
                    f"{name} result {dst!r} is {ty(dst)}, Vec elements are "
                    f"{_vec_elem(ty(args[0]))}")
        elif name == "len":
            if len(args) != 1:
                probs.append(f"len with {len(args)} arguments (expects 1)")
            elif not (_is_vec(ty(args[0])) or ty(args[0]) == STR):
                probs.append(
                    f"len receiver {args[0]!r} has kind {ty(args[0])} "
                    "(only Vec and string lower natively)")
            elif ty(dst) != I64:
                probs.append(f"len result {dst!r} promoted to {ty(dst)}")
        elif name in ("to_string", "int_to_str"):
            if len(args) != 1:
                probs.append(f"{name} with {len(args)} arguments (expects 1)")
            elif ty(args[0]) not in (I64, F64, STR):
                probs.append(
                    f"{name} of kind {ty(args[0])} (only i64/f64/str lower "
                    "to mx_i64_to_str/mx_f64_to_str/identity)")
            elif ty(dst) != STR:
                probs.append(f"{name} result {dst!r} is {ty(dst)}, not str")
        elif name in _MATH_EXTERNS:
            pass  # kinds pinned to f64 during inference

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
            if rk in ("const", "const_ty"):
                if _is_agg(ty(dst)) and dst not in info.dead_results:
                    probs.append(
                        f"constant {dst!r} promoted to aggregate kind "
                        f"{ty(dst)} (no scalar-to-aggregate coercion)")
                elif _is_vec(ty(dst)) and dst not in info.dead_results \
                        and not (rk == "const" and rhs[1] is None) \
                        and rk != "const_ty":
                    probs.append(
                        f"constant {dst!r} promoted to Vec kind {ty(dst)} "
                        "(no literal Vec values)")
            elif rk == "binop":
                o = rhs[1]
                if any(_is_vec(ty(x)) for x in (dst, *args)):
                    # Vec ==/!= is item-wise in the interpreter but would be
                    # pointer identity natively; no vec arithmetic exists.
                    probs.append(
                        f"binop {o!r} on Vec values (interpreter compares "
                        "contents; native pointers cannot)")
                elif o in _CMP_INT:
                    if ty(dst) not in (I64,):
                        probs.append(f"comparison result {dst!r} promoted to {ty(dst)}")
                    elif ty(args[0]) == STR and ty(args[1]) == STR:
                        # ==/!= on strings -> mx_str_eq (content equality,
                        # exactly the interpreter's).  Ordering comparisons
                        # on strings stay demoted.
                        if o not in ("==", "!="):
                            probs.append(
                                f"string ordering comparison {o!r} (only "
                                "==/!= lower to mx_str_eq)")
                    elif ty(args[0]) not in (I64, F64) or ty(args[1]) not in (I64, F64):
                        probs.append(f"comparison {o!r} on non-numeric operands")
                elif o in _LOGIC:
                    if any(ty(x) != I64 for x in (dst, *args)):
                        probs.append(f"logical binop {o!r} on non-i64 values")
                else:
                    if ty(dst) == STR:
                        # str + str -> mx_str_concat (fresh malloc'd string,
                        # leaked by design like boxes/heap envs).
                        if o != "+" or ty(args[0]) != STR or ty(args[1]) != STR:
                            probs.append(
                                f"string arithmetic {o!r} (only + on two "
                                "strings lowers to mx_str_concat)")
                    elif ty(dst) not in (I64, F64):
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
                elif callee.startswith(TRAIT_CALL_PREFIX):
                    method = callee[len(TRAIT_CALL_PREFIX):]
                    if not args:
                        probs.append(f"trait method call {method!r} with no receiver")
                        continue
                    res, target = _resolve_trait_call(
                        method, ty(args[0]), traits, module_names,
                        assume_final=True)
                    if res == "builtin":
                        check_builtin(target, dst, args)
                    elif res == "func":
                        sig = sigs.get(target)
                        if sig is None or target not in module_names:
                            probs.append(
                                f"trait method {method!r} resolves to unknown "
                                f"function {target!r}")
                        elif len(sig.params) != len(args):
                            probs.append(
                                f"trait method {method!r} -> {target!r} with "
                                "wrong arity")
                        else:
                            for a, pk in zip(args, sig.params):
                                if ty(a) != pk:
                                    probs.append(
                                        f"trait call {method!r} -> {target!r}: "
                                        f"arg {a!r} is {ty(a)}, expects {pk}")
                            if ty(dst) != sig.ret:
                                probs.append(
                                    f"trait call {method!r} -> {target!r}: "
                                    f"result {dst!r} is {ty(dst)}, returns {sig.ret}")
                    else:  # demote (or a pending that survived assume_final)
                        probs.append(target or
                                     f"trait method {method!r} cannot be resolved")
                elif callee in _NATIVE_RT_CALLS:
                    check_builtin(callee, dst, args)
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
                ename, vname = rhs[1], rhs[2]
                if ename in variants.bad:
                    probs.append(variants.bad[ename])
                    continue
                dk = ty(dst)
                ref = _enum_refinement(dk) if _is_enum(dk) else None
                if ref is None or vname not in ref \
                        or len(ref[vname]) != len(args):
                    if dst in info.dead_results:
                        continue
                    probs.append(
                        f"make_variant {vname!r} of enum {ename or 'anon'!r}: "
                        f"destination kind {dk} lacks the variant's refinement")
                    continue
                for i, a in enumerate(args):
                    sk = ref[vname][i]
                    store_k = _strip_refinement(ty(a))
                    if sk == CONFLICT or store_k == CONFLICT:
                        probs.append(
                            f"payload slot {i} of enum {ename or 'anon'!r} "
                            f"variant {vname!r} has conflicting kinds")
                    elif store_k != sk:
                        probs.append(
                            f"heterogeneous payload slot {i} of enum "
                            f"{ename or 'anon'!r}: variant {vname!r} stores "
                            f"{store_k} where merged flows require {sk} (no "
                            "coercion through tagged-union storage)")
                    elif _is_closure(sk):
                        probs.append(
                            f"enum {ename or 'anon'!r} payload slot {i} holds "
                            f"a closure ({sk}) (its env pointer may outlive "
                            "the creating frame)")
                    elif _is_agg(sk) and _kind_size(sk, structs, variants) is None:
                        probs.append(
                            f"enum {ename or 'anon'!r} payload slot {i} boxes "
                            f"a value of {sk} whose layout is infinite")
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
                    continue
                ename = _enum_name(bk)
                idx = rhs[1]
                vname = rhs[2] if len(rhs) > 2 else None
                # Patterns can name a payload slot no constructor fills
                # (dead arm); grow the union so the GEP stays in bounds.
                variants.payload_max[ename] = max(
                    variants.payload_max.get(ename, 0), idx + 1)
                if ename in variants.bad:
                    probs.append(variants.bad[ename])
                    continue
                if vname is None:
                    probs.append(
                        f"variant_field {idx} of enum {ename or 'anon'!r} "
                        "carries no variant name (legacy MIR shape; payload "
                        "slot kinds are per-variant)")
                    continue
                ref = _enum_refinement(bk)
                if ref is None:
                    probs.append(
                        f"variant_field on enum value {args[0]!r} whose kind "
                        f"{bk} has no payload refinement")
                    continue
                if vname not in ref or idx >= len(ref[vname]):
                    # Dead arm: the refinement lists every variant any flow
                    # into this value can construct, so this variant's tag
                    # test can never pass here.  A scalar-shaped result is
                    # emitted as a typed zero (never executed); an aggregate
                    # result would need storage semantics we refuse to fake.
                    if _is_agg(ty(dst)) and dst not in info.dead_results:
                        probs.append(
                            f"variant_field {idx} of enum {ename or 'anon'!r} "
                            f"variant {vname!r} reads an unconstructed "
                            f"variant into aggregate kind {ty(dst)}")
                    continue
                sk = ref[vname][idx]
                if _is_enum(sk):
                    # Nested extraction reads through the canonical cells:
                    # sound only when every store into the nested enum agrees
                    # with them (otherwise the representation is per-value
                    # and was lost at the boxing boundary).
                    nested = _enum_name(sk)
                    mixed = variants.mixed_slots_of(nested)
                    if mixed:
                        descr = ", ".join(
                            f"{v}[{i}]" for (_e, v, i) in mixed)
                        probs.append(
                            f"variant_field {idx} of enum {ename or 'anon'!r} "
                            f"variant {vname!r} extracts nested enum "
                            f"{nested or 'anon'!r} whose payload slots "
                            f"({descr}) have instantiation-dependent "
                            "representations (lost at the boxing boundary)")
                        continue
                    sk = variants.canon_kind(nested)
                if sk == CONFLICT:
                    probs.append(
                        f"variant_field {idx} of enum {ename or 'anon'!r} "
                        f"variant {vname!r} has a conflicting slot kind")
                elif _is_closure(sk):
                    probs.append(
                        f"enum {ename or 'anon'!r} payload slot {idx} holds "
                        f"a closure ({sk}) (its env pointer may outlive the "
                        "creating frame)")
                elif ty(dst) != sk:
                    probs.append(
                        f"variant_field {idx} of enum {ename or 'anon'!r} "
                        f"variant {vname!r}: result {dst!r} is {ty(dst)}, "
                        f"slot is {sk}")
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
        if b.term[0] == "ret" and _is_closure(ty(b.term[1])) \
                and _closure_lambda(ty(b.term[1])) not in closures.heap_env:
            # Defensive: the module driver marks every returned lambda
            # heap-env from its sig before this check runs, so this only
            # fires if that invariant is ever broken — a stack env crossing
            # a return would dangle.
            probs.append(
                f"returns closure of lambda "
                f"{_closure_lambda(ty(b.term[1]))!r} not marked heap-env "
                "(stack env would dangle)")
        # ret of a struct/enum value is fine: sret-style, the aggregate is
        # copied into the caller-provided %agg.ret slot (never a raw frame
        # pointer).  ret of a heap-env closure is fine: the pair is copied
        # sret-style and its env pointer aims at an immortal heap block.

    # Struct fields inline nested structs/enums (must have a finite layout);
    # enum payload slots box nested structs/enums; closures in either place
    # still demote (their env pointer may aim at a dying stack frame).
    used_structs = {_struct_name(k) for k in kinds.values() if _is_struct(k)}
    for sname in sorted(used_structs):
        if _kind_size(_STRUCT_PREFIX + sname, structs, variants) is None:
            probs.append(
                f"struct {sname!r} has a recursively inlined layout (a "
                "struct-in-struct cycle with no intervening enum box has no "
                "finite size)")
        for fn_ in structs.fields.get(sname, ()):
            fk = structs.field_kind(sname, fn_)
            if _is_closure(fk):
                probs.append(
                    f"struct {sname!r} field {fn_!r} holds a closure "
                    f"({fk}) (its env pointer may outlive the creating frame)")
            elif fk == CONFLICT:
                probs.append(f"struct {sname!r} field {fn_!r} has conflicting kinds")
    # Enum payload slot validity (closures, conflicts, infinite layouts) is
    # checked per-site above: at make_variant against the destination's
    # refinement and at variant_field against the base's refinement / the
    # canonical cells — module-wide cells no longer demote functions that
    # only ever touch well-refined values of the enum.  Refined enum kinds
    # reaching THIS function through its own values still need every slot of
    # every refinement to be emittable (a refined kind can arrive through a
    # signature without any local variant op).
    for k in sorted(set(kinds.values())):
        ref = _enum_refinement(k)
        if not ref:
            continue
        ename = _enum_name(k)
        for v in sorted(ref):
            for i, sk in enumerate(ref[v]):
                if _is_closure(sk):
                    probs.append(
                        f"enum {ename or 'anon'!r} payload slot {i} holds a "
                        f"closure ({sk}) (its env pointer may outlive the "
                        "creating frame)")
                elif sk == CONFLICT:
                    probs.append(
                        f"enum {ename or 'anon'!r} variant {v!r} payload "
                        f"slot {i} has conflicting kinds")
                elif _is_agg(sk) and _kind_size(sk, structs, variants) is None:
                    probs.append(
                        f"enum {ename or 'anon'!r} variant {v!r} payload "
                        f"slot {i} boxes a value of {sk} whose layout is "
                        "infinite")
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
# Provably-dead Vec analysis (which Vec.new results may be freed at exit)
# ---------------------------------------------------------------------------

# Native runtime calls that only READ or MUTATE a Vec through its receiver
# argument without retaining the pointer (metaxu_rt.c stores no receiver).
_VEC_SAFE_RECEIVER_BUILTINS = {"push", "pop", "len", "__index_get"}


def _provably_dead_vecs(f: MirFunc, kinds: Dict[str, str],
                        builtin_of) -> List[str]:
    """Vec.new result variables whose vector provably never escapes the
    frame, so `mx_vec_free` at every ret path is sound.

    The argument mirrors the @global-struct free proof: the malloc
    (mx_vec_new) happens unconditionally in the ENTRY block, and the
    pointer never leaves the frame — it is not returned, not stored in any
    aggregate (struct field / enum payload / closure env / another Vec),
    not captured, and not passed to any call except as the RECEIVER of the
    non-retaining native vec builtins.  Aliases created by `copy` are
    tracked (they hold the same pointer and are freed zero times — only the
    defining variable is freed once).  Anything not provable simply leaks
    by design (a leak is sound; a bad free is not), so this analysis bails
    conservatively: selects, re-definitions, cyclic entry blocks, or any
    unrecognized use disqualify the vec.

    ``builtin_of(callee, args)`` names the native builtin a call lowers to
    (None for anything else, including trait calls resolved to functions).
    """
    if not f.blocks:
        return []
    if 0 in _blocks_in_cycles(f):
        return []  # a re-executed entry block would double-free

    entry_ops = f.blocks[0].ops
    candidates = [op[1] for op in entry_ops
                  if op[0] == "let" and len(op) == 4
                  and op[2][0] == "call" and op[2][1] == "Vec.new"
                  and _is_vec(kinds.get(op[1], I64))]
    if not candidates:
        return []

    # def map: var -> list of (rhs, args)
    defs: Dict[str, List[Tuple[tuple, tuple]]] = {}
    for b in f.blocks:
        for op in b.ops:
            if op[0] == "let" and len(op) == 4:
                defs.setdefault(op[1], []).append((op[2], op[3]))

    freed: List[str] = []
    for site in candidates:
        # Alias group closure over plain copies.
        group: Set[str] = {site}
        changed = True
        while changed:
            changed = False
            for b in f.blocks:
                for op in b.ops:
                    if op[0] == "let" and len(op) == 4 and op[2][0] == "copy" \
                            and op[3] and op[3][0] in group \
                            and op[1] not in group:
                        group.add(op[1])
                        changed = True
        # Every group member's every def must be the site's Vec.new (for the
        # site itself, exactly once) or a copy from within the group.
        ok = True
        for m in group:
            for (rhs, dargs) in defs.get(m, []):
                if m == site and rhs[0] == "call" and rhs[1] == "Vec.new":
                    continue
                if rhs[0] == "copy" and dargs and dargs[0] in group:
                    continue
                ok = False
        if len([1 for (rhs, _a) in defs.get(site, [])
                if rhs[0] == "call" and rhs[1] == "Vec.new"]) != 1:
            ok = False
        # Every use of every member must be a whitelisted, non-escaping one.
        if ok:
            for b in f.blocks:
                for op in b.ops:
                    if not ok:
                        break
                    if op[0] == "drop":
                        continue
                    if op[0] == "match_fail" or op[0] == "params":
                        continue
                    if op[0] == "perform":
                        if any(a in group for a in op[4]):
                            ok = False
                        continue
                    if op[0] != "let" or len(op) != 4:
                        # unknown op shape: bail if we cannot see its uses
                        ok = False
                        continue
                    _, dst, rhs, oargs = op
                    rk = rhs[0]
                    if rk == "copy":
                        # copies from the group were folded into the group;
                        # a group member copied into a non-member cannot
                        # happen (closure above), so nothing to check.
                        continue
                    if rk == "call":
                        callee = rhs[1]
                        bname = builtin_of(callee, oargs)
                        if bname in _VEC_SAFE_RECEIVER_BUILTINS:
                            # receiver-only use is safe; a group member in
                            # any VALUE position escapes (stored in the vec)
                            if any(a in group for a in oargs[1:]):
                                ok = False
                            continue
                        if any(a in group for a in oargs):
                            ok = False
                        continue
                    if rk == "alloc_struct" or rk == "make_closure":
                        if any(v in group for (_n, v) in oargs):
                            ok = False
                        continue
                    # binop / select / field ops / variants / everything else:
                    # any appearance of a group member disqualifies.
                    if any(a in group for a in oargs):
                        ok = False
                t = b.term
                if t[0] in ("br_if", "ret") and t[1] in group:
                    ok = False
        if ok:
            freed.append(site)
    return freed


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
        self.runtime_syms: Set[str] = set()  # mx_* native runtime declares
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
    rty = "void (sret ptr)" if _is_agg(sig.ret) else _llscalar(sig.ret)
    lines = [f"; function @{sym}: placeholder -- unsupported for direct LLVM emission"]
    for r in info.reasons:
        lines.append(f";   reason: {r}")
    lines.append(f"; declare @{sym}({ptys}) -> {rty}")
    return "\n".join(lines)


def _emit_function(info: _Info, kinds: Dict[str, str], sigs: Dict[str, _Sig],
                   structs: _StructTable, variants: _VariantTable,
                   closures: _ClosureTable, traits: _TraitTable,
                   module_names: Set[str], mod: _ModuleState,
                   emitted_names: Set[str]) -> str:
    f = info.f
    sig = sigs[f.name]

    def kind(n: str) -> str:
        return kinds.get(n, I64)

    def llty(n: str) -> str:
        return _llscalar(kind(n))

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
        if _is_vec(k):
            if value is None:
                return "null"  # an uninitialized Vec slot (const None)
            raise _Unsupported(f"non-None constant for Vec value {name!r}")
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

    def builtin_of(callee: str, cargs: Tuple[str, ...]) -> Optional[str]:
        """The native builtin name a call op lowers to, or None (used both
        by the vec-escape analysis and nowhere else; mirrors the emission
        dispatch order below: locals shadow, then trait resolution, then
        plain builtins)."""
        if callee in info.def_count:
            return None  # closure call through a local
        if callee.startswith(TRAIT_CALL_PREFIX):
            if not cargs:
                return None
            res, target = _resolve_trait_call(
                callee[len(TRAIT_CALL_PREFIX):], kind(cargs[0]), traits,
                module_names, assume_final=True)
            return target if res == "builtin" else None
        if callee in _NATIVE_RT_CALLS:
            return callee
        return None

    # Vecs provably dead at frame exit (see _provably_dead_vecs): freed on
    # every ret path.  All other Vec.new results LEAK BY DESIGN — identity
    # semantics means the pointer may be shared anywhere it escaped to, so
    # no free can be proven unique (same contract as boxes/heap envs).
    vec_free_vars = _provably_dead_vecs(f, kinds, builtin_of)
    vec_free_set = set(vec_free_vars)

    # COPY-IN/COPY-OUT struct params: the interpreter WRITES BACK a struct
    # argument when the callee rebinds the parameter (mir_interp.
    # _write_back_struct_args — `self.field = ...` methods mutate the
    # caller's binding).  Natively the caller already passes its storage
    # pointer, so the callee copies the final param value back through it on
    # every ret path.  Statically "rebinds anywhere" over-approximates the
    # interpreter's per-execution identity test, but a not-taken rebind path
    # writes back the unchanged aggregate — observationally a no-op.
    # Closure calls get NO write-back in the interpreter, so lambdas never
    # copy out.  Only struct kinds write back (enums/closures never do).
    writeback_params = [] if info.is_lambda else [
        p for p in info.params
        if _is_struct(kinds.get(p, I64)) and info.def_count.get(p, 0) > 1]

    def emit_writebacks(lines: List[str]) -> None:
        for p in writeback_params:
            agg_copy(_agg_ty(kinds.get(p, I64)), struct_ref(p),
                     f"%a.{_sanitize(p)}", lines)
            lines.append(f"  ; ^ copy-out: rebound struct param {p} "
                         "written back to the caller")

    def emit_frees(lines: List[str]) -> None:
        """Free every heap-backed @global struct block and every provably
        frame-local Vec (called on ret paths).

        Sound because the mallocs unconditionally happen in the entry block
        (exactly once per invocation) and the pointers provably never
        escape this frame (value semantics for structs; the vec escape
        analysis for Vecs — see module docstring)."""
        for n in heap_vars:
            lines.append(f"  call void @free(ptr {struct_ref(n)})"
                         f"  ; @global struct {n}: end of frame")
        for n in vec_free_vars:
            mod.runtime_syms.add("mx_vec_free")
            lines.append(f"  call void @mx_vec_free(ptr {use(n, lines)})"
                         f"  ; local Vec {n}: provably non-escaping")

    def to_word(k: str, v: str, lines: List[str]) -> str:
        """Reinterpret a value of word kind k as the opaque i64 element word
        the native Vec ABI stores (the runtime never inspects elements)."""
        if k == I64:
            return v
        t = fresh()
        if k == F64:
            lines.append(f"  {t} = bitcast double {v} to i64")
        else:  # str / vec pointers
            lines.append(f"  {t} = ptrtoint ptr {v} to i64")
        return t

    def from_word(k: str, v: str, lines: List[str]) -> str:
        """Inverse of to_word: element word back to its typed value."""
        if k == I64:
            return v
        t = fresh()
        if k == F64:
            lines.append(f"  {t} = bitcast i64 {v} to double")
        else:
            lines.append(f"  {t} = inttoptr i64 {v} to ptr")
        return t

    def emit_direct_call(dst: str, callee: str, opargs: Tuple[str, ...],
                         lines: List[str]) -> None:
        """A direct call to another emitted module function (also the target
        of a statically-resolved trait call)."""
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
            rty = _llscalar(csig.ret)
            lines.append(
                f"  {v} = call {rty} @{mangle(callee)}({', '.join(avals)})")
            setval(dst, v, lines)

    def emit_rt_builtin(name: str, dst: str, opargs: Tuple[str, ...],
                        lines: List[str]) -> None:
        """Lower an interpreter builtin to its native runtime call
        (metaxu_rt.c, linked by llvm_run).  The consistency check already
        validated arities and kinds; anything off here is a hard error."""
        if name == "Vec.new":
            mod.runtime_syms.add("mx_vec_new")
            v = fresh()
            note = ("freed on ret paths (provably non-escaping)"
                    if dst in vec_free_set else "leaks by design (may escape)")
            lines.append(f"  {v} = call ptr @mx_vec_new()  ; Vec.new: {note}")
            setval(dst, v, lines)
        elif name == "push":
            recv = use(opargs[0], lines)
            elem = _vec_elem(kind(opargs[0]))
            w = to_word(elem, use(opargs[1], lines), lines)
            mod.runtime_syms.add("mx_vec_push")
            lines.append(f"  call void @mx_vec_push(ptr {recv}, i64 {w})")
            setval(dst, "0", lines)  # unit
        elif name == "pop":
            recv = use(opargs[0], lines)
            elem = _vec_elem(kind(opargs[0]))
            mod.runtime_syms.add("mx_vec_pop")
            w = fresh()
            lines.append(f"  {w} = call i64 @mx_vec_pop(ptr {recv})")
            setval(dst, from_word(elem, w, lines), lines)
        elif name == "__index_get":
            recv = use(opargs[0], lines)
            elem = _vec_elem(kind(opargs[0]))
            idx = use(opargs[1], lines)
            mod.runtime_syms.add("mx_vec_get")
            w = fresh()
            lines.append(f"  {w} = call i64 @mx_vec_get(ptr {recv}, i64 {idx})")
            setval(dst, from_word(elem, w, lines), lines)
        elif name == "len":
            recv = use(opargs[0], lines)
            sym = "mx_vec_len" if _is_vec(kind(opargs[0])) else "mx_str_len"
            mod.runtime_syms.add(sym)
            v = fresh()
            lines.append(f"  {v} = call i64 @{sym}(ptr {recv})")
            setval(dst, v, lines)
        elif name in ("to_string", "int_to_str"):
            k = kind(opargs[0])
            a = use(opargs[0], lines)
            if k == STR:
                setval(dst, a, lines)  # to_string of a string is identity
            else:
                sym = "mx_f64_to_str" if k == F64 else "mx_i64_to_str"
                mod.runtime_syms.add(sym)
                v = fresh()
                lines.append(
                    f"  {v} = call ptr @{sym}({_llscalar(k)} {a})"
                    "  ; fresh malloc'd string (leaks by design)")
                setval(dst, v, lines)
        elif name in _MATH_EXTERNS:
            mod.math_used.add(name)
            a = use(opargs[0], lines)
            v = fresh()
            lines.append(f"  {v} = call double @{name}(double {a})")
            setval(dst, v, lines)
        else:  # unreachable given resolution + consistency
            raise _Unsupported(f"builtin {name!r} has no native lowering")

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

    # One env alloca per stack-env make_closure site, named and created in
    # the entry block (storage must dominate every use; the site itself may
    # sit in a conditional block).  Heap-env lambdas malloc a FRESH env at
    # the site instead (each execution gets its own immortal block).
    env_allocas: Dict[int, str] = {}
    env_entry: List[str] = []
    env_seq = 0
    for b in f.blocks:
        for op in b.ops:
            if op[0] == "let" and len(op) == 4 and op[2][0] == "make_closure":
                lname = op[2][1]
                if lname in closures.heap_env:
                    continue
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
                is_str = kind(opargs[0]) == STR and kind(opargs[1]) == STR
                if is_str and o == "+":
                    # String concatenation -> fresh malloc'd string from the
                    # native runtime; never freed (leaks by design, same
                    # contract as boxes/heap envs this increment).
                    mod.runtime_syms.add("mx_str_concat")
                    v = fresh()
                    lines.append(
                        f"  {v} = call ptr @mx_str_concat(ptr {l}, ptr {r})"
                        "  ; leaks by design")
                    setval(dst, v, lines)
                elif is_str and o in ("==", "!="):
                    # Content equality via mx_str_eq (returns 0/1), exactly
                    # the interpreter's string comparison.
                    mod.runtime_syms.add("mx_str_eq")
                    e = fresh()
                    lines.append(f"  {e} = call i64 @mx_str_eq(ptr {l}, ptr {r})")
                    if o == "==":
                        setval(dst, e, lines)
                    else:
                        c, v = fresh(), fresh()
                        lines.append(f"  {c} = icmp eq i64 {e}, 0")
                        lines.append(f"  {v} = zext i1 {c} to i64")
                        setval(dst, v, lines)
                elif o in _CMP_INT:
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
                        rty = _llscalar(csig.ret)
                        lines.append(
                            f"  {v} = call {rty} {fnv}({', '.join(avals)})")
                        setval(dst, v, lines)
                elif callee.startswith(TRAIT_CALL_PREFIX):
                    # Statically-resolved trait dispatch (kind-driven; the
                    # consistency check validated the resolution).
                    method = callee[len(TRAIT_CALL_PREFIX):]
                    if not opargs:
                        raise _Unsupported(
                            f"trait method call {method!r} with no receiver")
                    res, target = _resolve_trait_call(
                        method, kind(opargs[0]), traits, module_names,
                        assume_final=True)
                    if res == "builtin":
                        emit_rt_builtin(target, dst, opargs, lines)
                    elif res == "func":
                        emit_direct_call(dst, target, opargs, lines)
                    else:
                        raise _Unsupported(
                            target or f"unresolved trait method call {method!r}")
                elif callee in _NATIVE_RT_CALLS:
                    emit_rt_builtin(callee, dst, opargs, lines)
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
                    emit_direct_call(dst, callee, opargs, lines)
            elif rk == "alloc_struct":
                sname = rhs[1]
                if dst not in aggset:
                    raise _Unsupported(f"alloc_struct result {dst!r} not struct-kinded")
                for (fn_, fv) in opargs:
                    p = gep(sname, struct_ref(dst), fn_, lines)
                    fk = structs.field_kind(sname, fn_)
                    if _is_agg(fk):
                        # nested aggregate field: copy the value into the
                        # inline field region (value semantics, no heap)
                        agg_copy(_agg_ty(fk), use(fv, lines), p, lines)
                    else:
                        lines.append(
                            f"  store {_llscalar(fk)} {use(fv, lines)}, ptr {p}")
            elif rk == "field_get":
                sname = _struct_name(kind(opargs[0]))
                base = use(opargs[0], lines)
                p = gep(sname, base, rhs[1], lines)
                fk = structs.field_kind(sname, rhs[1])
                if _is_agg(fk):
                    if dst not in aggset:
                        raise _Unsupported(
                            f"field_get result {dst!r} not aggregate-kinded "
                            f"for nested field {rhs[1]!r}")
                    agg_copy(_agg_ty(fk), p, struct_ref(dst), lines)
                else:
                    v = fresh()
                    lines.append(f"  {v} = load {_llscalar(fk)}, ptr {p}")
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
                if _is_agg(fk):
                    agg_copy(_agg_ty(fk), use(opargs[1], lines), p, lines)
                else:
                    lines.append(
                        f"  store {_llscalar(fk)} {use(opargs[1], lines)}, ptr {p}")
            elif rk == "make_variant":
                # Tagged union: store the integer tag, then the payload slots
                # at THIS variant's refined slot kinds (different variants —
                # and different instantiations of one variant — may disagree
                # about what a slot index holds).
                ename, vname = rhs[1], rhs[2]
                if dst not in aggset or not _is_enum(kind(dst)):
                    raise _Unsupported(
                        f"make_variant result {dst!r} not enum-kinded")
                tagv = variants.tags.get(vname)
                if tagv is None:
                    raise _Unsupported(f"no tag assigned for variant {vname!r}")
                vref = _enum_refinement(kind(dst)) or {}
                vslots = vref.get(vname)
                if vslots is None or len(vslots) != len(opargs):
                    raise _Unsupported(
                        f"make_variant {vname!r} destination kind lacks the "
                        "variant's refinement")
                mod.used_enums.add(ename)
                p = enum_gep(ename, struct_ref(dst), lines)
                lines.append(
                    f"  store i64 {tagv}, ptr {p}  ; tag {vname}={tagv}")
                for i, fv in enumerate(opargs):
                    ck = vslots[i]
                    p = enum_gep(ename, struct_ref(dst), lines, payload=i)
                    if _is_agg(ck):
                        # Boxed aggregate payload: malloc a write-once box,
                        # copy the aggregate in, store the POINTER in the
                        # 8-byte slot.  Never freed (leak by design: shallow
                        # pair/aggregate copies share box pointers, so no
                        # free can be proven unique — see module docstring).
                        size = _kind_size(ck, structs, variants)
                        if size is None:
                            raise _Unsupported(
                                f"boxed payload of {ck} has infinite layout")
                        mod.uses_malloc = True
                        box = fresh()
                        lines.append(
                            f"  {box} = call ptr @malloc(i64 {max(size, 8)})"
                            f"  ; boxed {ck} payload (leaks by design)")
                        agg_copy(_agg_ty(ck), use(fv, lines), box, lines)
                        lines.append(f"  store ptr {box}, ptr {p}")
                    else:
                        lines.append(
                            f"  store {_llscalar(ck)} {use(fv, lines)}, ptr {p}")
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
                vname = rhs[2] if len(rhs) > 2 else None
                vref = _enum_refinement(bk)
                if vname is None or vref is None:
                    raise _Unsupported(
                        f"variant_field {rhs[1]} of enum {ename or 'anon'!r} "
                        "without a variant refinement (legacy MIR shape)")
                if vname not in vref or rhs[1] >= len(vref[vname]):
                    # Dead arm: no flow into this value constructs vname (the
                    # refinement lists every constructible variant), so the
                    # guarding tag test can never pass and this read never
                    # executes.  Emit a typed zero for scalar results;
                    # aggregate results keep their (never-read) storage.
                    if dst in aggset:
                        lines.append(
                            f"  ; dead arm: variant {vname!r} never "
                            f"constructed for this value; {dst} left "
                            "uninitialized (unreachable)")
                    else:
                        zk = kind(dst)
                        if zk == F64:
                            zv = _fmt_f64(0.0)
                        elif zk == STR:
                            zv = mod.intern_string("")
                        elif _is_vec(zk):
                            zv = "null"
                        else:
                            zv = "0"
                        lines.append(
                            f"  ; dead arm: variant {vname!r} never "
                            "constructed for this value (unreachable)")
                        setval(dst, zv, lines)
                    continue
                ck = vref[vname][rhs[1]]
                if _is_enum(ck):
                    # Nested extraction: the boxed value carries the
                    # canonical module-wide representation (consistency
                    # demoted any mixed slots), and kind(dst) is the
                    # canonical refined kind of the same enum.
                    ck = kind(dst) if _is_enum(kind(dst)) else ck
                base = use(opargs[0], lines)
                p = enum_gep(ename, base, lines, payload=rhs[1])
                if _is_agg(ck):
                    # Boxed payload: load the box pointer, copy the aggregate
                    # out into the destination's own storage (value
                    # semantics; the box itself stays untouched and shared).
                    if dst not in aggset:
                        raise _Unsupported(
                            f"variant_field result {dst!r} not "
                            f"aggregate-kinded for boxed slot {rhs[1]}")
                    box = fresh()
                    lines.append(f"  {box} = load ptr, ptr {p}")
                    agg_copy(_agg_ty(ck), box, struct_ref(dst), lines)
                else:
                    v = fresh()
                    lines.append(f"  {v} = load {_llscalar(ck)}, ptr {p}")
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
                fields = env_fields(lname)
                mod.env_types[lname] = fields
                mod.uses_closure_pair = True
                ety = f"%env.{_sanitize(lname)}"
                if lname in closures.heap_env:
                    # Escaping (or looped) closure: malloc a fresh env at
                    # the site, never freed (leak by design — an immortal
                    # env can never dangle, see module docstring).
                    esize = 0
                    for (_cn, ck) in fields:
                        s = _kind_size(ck, structs, variants)
                        if s is None:
                            raise _Unsupported(
                                f"env capture of {ck} has infinite layout")
                        esize += s
                    mod.uses_malloc = True
                    envp = fresh()
                    lines.append(
                        f"  {envp} = call ptr @malloc(i64 {max(esize, 8)})"
                        f"  ; heap env for {dst} -> {lname} (leaks by design)")
                else:
                    envp = env_allocas.get(id(op))
                    if envp is None:  # unreachable: prescan covers every site
                        raise _Unsupported("make_closure site missing env storage")
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
                            f"  store {_llscalar(ck)} {use(vn, lines)}, ptr {p}")
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
                    # Write back rebound struct params FIRST: if the caller
                    # aliased its result slot with an argument (dst == arg),
                    # the interpreter's order makes the RESULT win, so the
                    # sret copy must come after the copy-outs.  Then copy the
                    # aggregate into the caller's slot BEFORE any frees (the
                    # returned value may live in a heap block).
                    emit_writebacks(lines)
                    src = use(t[1], lines)
                    agg_copy(_agg_ty(sig.ret), src, "%agg.ret", lines)
                    emit_frees(lines)
                    lines.append("  ret void")
                else:
                    rv = use(t[1], lines)
                    emit_writebacks(lines)
                    emit_frees(lines)
                    lines.append(f"  ret {_llscalar(sig.ret)} {rv}")
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
    rty = "void" if sret else _llscalar(sig.ret)
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
        # Recursive layout size: leaf cells are 8 bytes, nested aggregates
        # inline, boxed payload slots are 8-byte pointers.
        size = _kind_size(kind(n), structs, variants)
        if size is None:  # unreachable: consistency demoted infinite layouts
            raise _Unsupported(f"@global struct {sname!r} has infinite layout")
        mod.uses_malloc = True
        entry.append(
            f"  {struct_ref(n)} = call ptr @malloc(i64 {max(size, 8)})"
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
                entry.append(f"  {v} = load {_llscalar(ck)}, ptr {p}")
                entry.append(f"  store {_llscalar(ck)} {v}, ptr {slot_ref(cn)}")
            else:
                entry.append(
                    f"  %cap.{_sanitize(cn)} = load {_llscalar(ck)}, ptr {p}")
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
    # Native metaxu runtime symbols (metaxu_rt.c, linked by llvm_run).
    for name in sorted(mod.runtime_syms):
        rt, params = _RT_SIGS[name]
        decls.append(f"declare {rt} @{name}({', '.join(params)})")
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
        # Nested aggregate fields inline their named type (LLVM permits
        # forward references between named types, so order is free).
        ftys = ", ".join(_llcell(structs.field_kind(sname, fn_))
                         for fn_ in structs.fields.get(sname, ()))
        fields_desc = ", ".join(structs.fields.get(sname, ()))
        lines.append(f"%struct.{_sanitize(sname)} = type {{ {ftys} }}  ; {fields_desc}")
    return "\n".join(lines) if lines else None


def _emit_enum_types(variants: _VariantTable, used: Set[str]) -> Optional[str]:
    """%enum.E tagged-union types plus the documented tag mapping and the
    per-variant payload slot kinds (canonical module-wide cells; slots whose
    stores disagree across instantiations are flagged 'mixed per value')."""
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
        for v in vnames:
            arity = variants.arity.get((ename, v), 0)
            slot_docs = []
            for i in range(arity):
                ck = variants.cell_kind(ename, v, i)
                doc = f"boxed {ck}" if _is_agg(ck) else ck
                if (ename, v, i) in variants.mixed:
                    doc += " (mixed per value)"
                slot_docs.append(doc)
            lines.append(
                f";   variant {v}({', '.join(slot_docs)})")
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
            _agg_ty(k) if _is_agg(k) else _llscalar(k)
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
    traits = _build_trait_table(module_names)
    infos = [_analyze(f, module_names, closures) for f in funcs]
    variants = _build_variant_table(funcs, infos)

    def dep_names(info: _Info, kinds: Dict[str, str]) -> Set[str]:
        """Module functions this one references and cannot link without:
        direct callees, make_closure targets, resolved closure callees, and
        statically-resolved trait-call targets.  Native runtime builtin
        names never count, even when a module function shares the name (the
        builtin wins, mirroring the interpreter's resolution order)."""
        deps = {callee for (_d, callee, _a) in info.calls
                if callee in module_names and callee not in _NATIVE_RT_CALLS}
        deps |= {lname for (_d, lname, _c) in info.closure_defs
                 if lname in module_names}
        for (_d, cvar, _a) in info.closure_calls:
            ck = kinds.get(cvar, I64)
            if _is_closure(ck) and _closure_lambda(ck) in module_names:
                deps.add(_closure_lambda(ck))
        for (_d, method, targs) in info.trait_calls:
            if not targs:
                continue
            res, target = _resolve_trait_call(
                method, kinds.get(targs[0], I64), traits, module_names,
                assume_final=True)
            if res == "func" and target in module_names:
                deps.add(target)
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
    # promoted monotonically; callers and callees feed each other).  Two
    # phases: the first never assumes anything about trait-call receivers
    # still at the i64 bottom; once it converges, the second treats those
    # receivers as genuinely int (nothing else can promote them anymore)
    # and keeps iterating so the late resolutions' kinds propagate.
    sigs: Dict[str, _Sig] = {
        info.f.name: _Sig(params=[I64] * len(info.params)) for info in infos}
    kind_sets: Dict[str, Dict[str, str]] = {}
    candidates = [info for info in infos if not info.reasons]
    for assume_final in (False, True):
        for _round in range(12):
            changed = False
            for info in candidates:
                kinds, cell_changed = _infer_kinds(
                    info, sigs, structs, variants, closures, traits,
                    module_names, assume_final=assume_final)
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
                resolved_calls = [
                    (dst, callee, args) for (dst, callee, args) in info.calls
                    if callee not in _NATIVE_RT_CALLS]
                for (dst, cvar, args) in info.closure_calls:
                    ck = kinds.get(cvar, I64)
                    if _is_closure(ck):
                        resolved_calls.append((dst, _closure_lambda(ck), args))
                for (dst, method, targs) in info.trait_calls:
                    if not targs:
                        continue
                    res, target = _resolve_trait_call(
                        method, kinds.get(targs[0], I64), traits, module_names,
                        assume_final=assume_final)
                    if res == "func":
                        resolved_calls.append((dst, target, targs))
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

    # A lambda whose closure is RETURNED by any function needs a heap env:
    # the pair crosses the creating frame's boundary, so a stack env would
    # dangle.  sig.ret is the join of every ret-var kind across the module
    # fixpoint, so this is complete for the emitted subset (storing a pair
    # in a field/payload/env demotes the storing function instead).  Over-
    # marking is sound — a heap env only leaks, it can never dangle.
    for sig in sigs.values():
        if _is_closure(sig.ret):
            closures.heap_env.add(_closure_lambda(sig.ret))

    # Mark module-wide variant cells whose stores disagree post-fixpoint:
    # the joined cell kind is then NOT the representation every writer used
    # (multi-instantiation slots like a generic Option holding ints in one
    # use and structs in another), so nested extraction through the
    # canonical cells must demote rather than guess.  Per-value refinements
    # are unaffected — each value still knows its own representation.
    for info in candidates:
        kinds = kind_sets.get(info.f.name, {})
        for b in info.f.blocks:
            for op in b.ops:
                if op[0] != "let" or len(op) != 4 or op[2][0] != "make_variant":
                    continue
                ename, vname = op[2][1], op[2][2]
                for i, a in enumerate(op[3]):
                    if _strip_refinement(kinds.get(a, I64)) != \
                            variants.cell_kind(ename, vname, i):
                        variants.mixed.add((ename, vname, i))

    # Post-fixpoint consistency; anything wrong becomes a placeholder reason.
    for info in candidates:
        kinds = kind_sets.get(info.f.name, {})
        for p in _check_consistency(info, kinds, sigs, structs, variants,
                                    closures, traits, module_names):
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
                                       closures, traits, module_names, mod,
                                       emitted)
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

    # Close the used type sets over nested references: inline struct fields
    # and boxed enum payload slots name %struct/%enum types that may never
    # appear as a local variable kind, and env structs may inline aggregates.
    for fields in mod.env_types.values():
        for (_cn, k) in fields:
            if _is_struct(k):
                used_structs.add(_struct_name(k))
            elif _is_enum(k):
                mod.used_enums.add(_enum_name(k))
    while True:
        more_structs: Set[str] = set()
        more_enums: Set[str] = set()
        for sname in used_structs:
            for fn_ in structs.fields.get(sname, ()):
                fk = structs.field_kind(sname, fn_)
                # A referenced type with an infinite layout is never emitted
                # (every function touching it was demoted): do not pull it.
                if _kind_size(fk, structs, variants) is None:
                    continue
                if _is_struct(fk):
                    more_structs.add(_struct_name(fk))
                elif _is_enum(fk):
                    more_enums.add(_enum_name(fk))
        for ename in mod.used_enums:
            for (e2, _v2, _i2), ck in variants.cells.items():
                if e2 != ename:
                    continue
                if _kind_size(ck, structs, variants) is None:
                    continue
                if _is_struct(ck):
                    more_structs.add(_struct_name(ck))
                elif _is_enum(ck):
                    more_enums.add(_enum_name(ck))
        if more_structs <= used_structs and more_enums <= mod.used_enums:
            break
        used_structs |= more_structs
        mod.used_enums |= more_enums

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
