"""Tests for the LLVM IR emitter (codegen_llvm.py) and native run harness.

Three layers:

1. Structural tests on the emitted IR text (no clang needed): define lines,
   block labels, alloca slots, GEPs, printf declaration/helpers, and honest
   comment-only placeholders for everything outside the direct subset.
2. Native differential tests (skipped when clang is absent): parsed-source
   programs are compiled through the REAL front end to MIR, emitted as LLVM,
   compiled with clang -O2, executed, and the native (exit code, stdout) is
   asserted equal to what the MIR interpreter computes for the same program.
   Exit codes only preserve the low 8 bits, so value results are compared
   modulo 256 and anything larger goes through print/stdout.
3. All 17 accepted example files run through emit_llvm: it must never crash,
   every MIR function must appear either as a real define or as an explicit
   placeholder comment carrying at least one reason, and (when opt is
   available) every module must pass LLVM's own IR verifier.

Increment 2 (ownership memory model) adds: struct params byval-copied
through a ptr, sret-style struct returns, @global structs malloc'd on the
heap and freed on every ret path, and multi-arg print.  The heap tests run
the native binary under clang -fsanitize=address when the ASan runtime is
present (exit 0 == no leak, no double-free, no use-after-free); when the
runtime is missing they are skipped, not silently weakened.

Increment 3 (variants + closures) adds: enums as %enum.E tagged unions
(integer tag + [N x i64] payload slots; variant names mapped to dense
module-wide integer tags so pattern tag tests compare integers, never
strings), and closures as {fn ptr, env ptr} pairs over per-lambda stack env
structs (direct locally-bound calls and closures passed DOWN as arguments).

Increment 4 (heap boxing for recursive data) adds: enum payload slots
holding aggregates store a heap POINTER to a write-once boxed copy
(make_variant boxes in, variant_field copies out — recursive enums like
linked lists become representable), struct fields holding structs/enums are
INLINED in the parent layout (recursive GEPs, no heap), and closures that
escape (returned anywhere in the module, or created in a loop) malloc their
env at the site.  FREE STRATEGY under test: payload boxes and heap closure
envs LEAK BY DESIGN (shallow copies share the pointers, so no free is
provably unique); ASan tests for box/heap-env programs therefore run with
ASAN_OPTIONS=detect_leaks=0 and prove no use-after-free / no double-free
only.  The @global struct malloc/free protocol is unchanged and its ASan
tests still prove full leak-freedom.  Still demoted honestly: closures
stored in struct fields / enum payloads / captured in other closures, and
heterogeneous payload slots.

Increment 5 (native vec/string runtime) adds: the C runtime metaxu_rt.o is
linked into every native binary; Vec.new/push/pop/len/__index_get lower to
mx_vec_* (a Vec is an opaque mx_vec* with IDENTITY semantics — shallow
pointer copies alias one shared vector, elements travel as opaque 8-byte
words), to_string routes by kind to mx_i64_to_str / mx_f64_to_str /
identity, string + is mx_str_concat and string ==/!= is mx_str_eq, and
__trait$ method calls resolve STATICALLY against the receiver's inferred
kind (impl fn for the type name, else builtin, else plain fn — mirroring
mir_interp's dispatch order).  FREE STRATEGY under test: a Vec provably
confined to its frame is mx_vec_free'd on every ret path and its programs
are FULLY leak-checked under ASan; escaping vecs (and all concat/to_string
results) leak by design, so their ASan tests use detect_leaks=0 and prove
no-UAF/no-double-free only.  Known kind-erasure caveat (same as print):
bools/unit erase to i64, so to_string of a bool natively yields "1"/"0",
not "True"/"False" — differential sources stringify ints/floats/strings.

Increment 6 (per-variant payload slot typing) adds: enum payload slots are
typed per VARIANT and per VALUE — each enum value's kind carries a
refinement of its constructible variants' slot representations, so
Leaf(int) | Fork(Tree, Tree) and generic instantiation mixes like Some(3)
vs Some(node) in one module (linked_list.mx's reality) now emit; the MIR
variant_field op carries its pattern's ctor name so reads know which
variant's slot they touch.  Still demoted honestly: ONE value merging two
representations of the same variant (heterogeneous ... no coercion),
nested enums extracted through a boxing boundary when their slots are
instantiation-dependent, and legacy two-element variant_field ops.

Increment 7 (native algebraic effects) adds: handle_scope / perform /
resume lower to the C effects runtime metaxu_effects.c (ucontext
coroutines; linked into every native binary next to metaxu_rt.o) with the
interpreter's exact semantics — deep handlers (resume returns the WHOLE
delimited body's value), single-shot continuations, abort when a case
returns without resuming, dynamic innermost-first routing with busy
scopes skipped (handler self-performs route outward).  Handle sites get a
shared `%henv.<site>` env struct + body-thunk/dispatcher shims; boundary
values travel as opaque 8-byte words with kinds unified through
module-wide per-op-name cells.  Suspending functions no longer demote.
The differential catalogue below replays test_effect_continuations.py's
shapes natively (8/42/10/99/103/300/42-nested/224/101/2/42-args) and runs
examples 02 and effects.mx end-to-end.  Pure-effect programs are FULLY
leak-checked under ASan (the runtime frees stacks/scopes/continuations on
completion and abort); programs that also concat strings keep
detect_leaks=0 per the leak-by-design contract.  Still demoted honestly:
aggregates (closures included) crossing the effect boundary, resume
outside its own handler case, same-named ops with conflicting kinds.

Increment 8 (memory reclamation + copy elision) adds:
  * OWNED STRINGS: a produced string (concat / to_string result — always a
    fresh malloc, metaxu_rt never returns an input) whose every use is
    non-retaining (concat operand, ==/!=, print, len) is freed at each
    redefinition (a concat LOOP no longer grows memory) and at frame exit,
    via a null-initialized shadow slot; literal defs record null (interned
    constants are NEVER freed — provenance is static).  Programs in this
    class run under FULL ASan leak checking.  Anything retained — returned,
    stored, passed to a call, aliased by to_string-of-str, crossing an
    effect boundary — still leaks by design.
  * UNIQUE BOXES: an entry-block make_variant whose enum value (closed
    over intra-frame copies) is only ever variant_tag/variant_field-read —
    never returned/passed/stored/captured/re-boxed — solely owns its
    payload boxes; they are freed on every ret path, FULL-leak-checked.
    Shared boxes (anything passed to a call, e.g. every recursive list
    traversal) keep the detect_leaks=0 contract.
  * COPY ELISION (a): an aggregate param never rebound and never reaching a
    callee write-back position skips the entry byval copy and reads the
    caller's storage through the passed pointer.  (b): a variant_field
    result that is only read becomes a BOX VIEW (a pointer into the
    write-once box) instead of an aggregate copy; copies of views alias the
    same box.  `; elide-copy:` comments pin both structurally.  The cases
    that MUST NOT elide stay pinned: rebound params (byval + write-back)
    and params passed onward to rebinding callees.

Increment 9 (native FFI) adds: the `rawptr` kind (raw C pointers, 8-byte
scalars; `null` -> the ptr null constant; ==/!= -> pointer icmp; ordering,
print, Vec storage and effect-boundary crossings demote honestly), extern
C calls to the REAL libc symbols the interpreter shims over its simulated
heap (malloc/free/memcpy/realloc/fopen/fclose — fclose declared with its C
i32 return and sext'd), as_ptr (identity on strings; a fresh
mx_vec_as_bytes byte snapshot on vecs — never a view into the vec's word
buffer, whose 8-byte-element layout would be wrong bytes), inline
ptr_read/ptr_write, __vec_lit -> mx_vec_new + pushes (identity semantics
stand in for the interpreter's immutable MxVector; indistinguishable for
accepted programs since MxVector supports no mutation), compile-time
__static$Type$method resolution (the interpreter's _dispatch_static_call
order), and assert -> inline branch-to-abort.  The interpreter's simulated
heap rejects overruns/UAF/double-frees as errors; natively those programs
are real UB — the same strict-error-vs-UB contract as division by zero —
so the ASan differentials below pin the ACCEPTED side: the malloc/
ptr_write/ptr_read/free round trip and memcpy-from-string run FULLY
leak-checked (real allocator, program-managed frees), while vec snapshots
and example 05's Ok-arm File box keep the detect_leaks=0 leak-by-design
contract.  05_unsafe_and_ffi.mx emits with ZERO placeholders and runs
natively end-to-end with the cwd pinned (both fopen outcomes).

Increment 12 (silent-seam constructs) adds: index assignment
(`__index_store` -> mx_vec_set in place on Vecs — the aliasing result
keeps provably-local vecs freeable — and the new mx_fvec_set_copy
FUNCTIONAL update on vector[T,N] places, copying the write-once block so
other shares never observe the write; `__index_set` is Vec-only, with
immutable receivers demoted at compile time carrying the interpreter's
error), mutable-capture cells (`cell_wrap` -> one-word malloc'd cells,
reads/writes through the cell pointer, closure AND handle-scope envs
capture the POINTER — closure counters, escaping-counter state retention
and handler-frame counters like std.stream take/skip's `seen` all run
natively; captures not provably after the wrap demote, since the
interpreter froze a value copy there), module constants (`__module_init`
-> @mx_g_<name> internal globals stored by the emitted initializer,
which llvm_run's entry wrapper calls first; readers load with no local
binding, parameter shadowing stays local, flow-sensitive assignment
shadowing demotes, and a module whose initializer demoted refuses to run
natively at all), and zip comprehensions (`__zip` results are VIRTUAL —
restricted to comprehension-iterable uses — and the site drives the new
mx_fvec_zip_map through a two-word thunk; length mismatches abort with
the interpreter's message).  Cells and fvec blocks leak by design
(detect_leaks=0); index-stored local Vecs and scalar-global programs run
FULLY leak-checked.

Increment 14 (aggregates across the effect boundary) adds: structs, enums
and closure pairs as perform arguments, resume values, handler-case /
body results and handle-scope results cross as BOUNDARY BOXES — a fresh
malloc'd write-once copy whose pointer is the 8-byte word (the enum
payload box contract at the effect boundary); receivers copy out per
their kind (case params byval-copy straight from the box), the per-site
shims malloc the box for aggregate body/case results and call sret-style
into it, and every member lambda of a boundary-crossing closure kind is
forced heap-env.  Boxes are immortal (leak by design, detect_leaks=0
proves no-UAF/no-double-free with aggregates in flight across
park/resume); std.stream's `find` (Option-valued handle result) emits
and runs natively both paths.  Still demoted honestly: konts, rawptr
words, conflicting same-named-op cells (aggregate or scalar — the
coarseness rule is unchanged), infinite layouts, and performs of ops
with a `with SYMBOL` C-runtime mapping (effect_mapping.mx's EFFECT_*
threads/mutex primitives have no native runtime; the interpreter routes
unscoped performs to them, so mx_perform's abort would diverge — demote,
never guess).
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from metaxu.compiler.codegen_llvm import emit_fvec_reduce, emit_llvm, mangle
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.llvm_run import LlvmRunError, compile_and_run
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir import MirBlock, MirFunc
from metaxu.compiler.mir_interp import UNIT, MirInterpreter
from metaxu.compiler.pipeline import build_context_from_source

REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent

needs_clang = pytest.mark.skipif(
    shutil.which("clang") is None, reason="clang is not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_func(name: str, blocks: list[MirBlock], suspending: bool = False) -> MirFunc:
    return MirFunc(name=name, ty_sig=None, blocks=blocks, suspending=suspending)


def block(ops: list[tuple], term: tuple) -> MirBlock:
    return MirBlock(ops=ops, term=term)


def mir_from_source(source: str, monomorphize: bool = False) -> list[MirFunc]:
    ctx = build_context_from_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    if monomorphize:
        from metaxu.compiler.monomorphize import collect_signatures, monomorphize_hir
        hir = monomorphize_hir(hir, collect_signatures(ctx.id_map))
    return lower_hir_to_mir(hir)


def llvm_from_source(source: str, monomorphize: bool = True) -> str:
    """Emit LLVM the way pipeline.emit_llvm_from_source does — through the
    monomorphization pass. `interp_run` deliberately stays on the
    UNSPECIALIZED MIR, so every differential in this file compares native
    monomorphized code against the unmonomorphized semantics reference."""
    return emit_llvm(mir_from_source(source, monomorphize=monomorphize))


def interp_run(source: str, entry: str = "main"):
    """Run through the MIR interpreter; returns (result, stdout-equivalent)."""
    interp = MirInterpreter()
    interp.load(mir_from_source(source))
    out: list[str] = []

    def _print(*args):
        out.append(" ".join(str(a) for a in args))
        return UNIT

    interp.register_builtin("print", _print)
    interp.register_builtin("println", _print)
    result = interp.call(entry, [])
    return result, "".join(line + "\n" for line in out)


def count_placeholders(ir: str) -> int:
    return len(re.findall(r"placeholder -- unsupported", ir))


def assert_native_matches_interp(source: str, tmp_path, entry: str = "main",
                                 clang_args: tuple[str, ...] = ()):
    """The differential assertion: clang-compiled result == interpreter."""
    result, expected_out = interp_run(source, entry)
    ir = llvm_from_source(source)
    exit_code, stdout = compile_and_run(ir, entry, workdir=str(tmp_path),
                                        clang_args=clang_args)
    assert stdout == expected_out
    if result is not UNIT and isinstance(result, (bool, int)):
        assert exit_code == int(result) % 256
    return ir


_ASAN_PROBE: list[bool] = []


def asan_available() -> bool:
    """True when clang can link -fsanitize=address (ASan runtime installed)."""
    if not _ASAN_PROBE:
        if shutil.which("clang") is None:
            _ASAN_PROBE.append(False)
        else:
            import tempfile, os
            with tempfile.TemporaryDirectory(prefix="metaxu_asan_probe_") as d:
                c = os.path.join(d, "t.c")
                with open(c, "w") as fh:
                    fh.write("int main(void){return 0;}\n")
                proc = subprocess.run(
                    ["clang", "-fsanitize=address", c, "-o", os.path.join(d, "t")],
                    capture_output=True, text=True)
                _ASAN_PROBE.append(proc.returncode == 0)
    return _ASAN_PROBE[0]


needs_asan = pytest.mark.skipif(
    not asan_available(),
    reason="clang ASan runtime not available (compile probe failed)")


def assert_native_matches_interp_asan(source: str, tmp_path, entry: str = "main"):
    """Differential + ASan: exit 0 under ASan proves the emitted frees sound
    (no leak, no double-free, no use-after-free) for programs whose @global
    structs are all freed.  Only call under @needs_asan."""
    # ASan replaces the exit code on error, so require interpreter results
    # that are observed via stdout plus a 0 exit.
    result, _ = interp_run(source, entry)
    assert result in (UNIT, 0), "ASan differential sources must exit 0"
    return assert_native_matches_interp(
        source, tmp_path, entry, clang_args=("-fsanitize=address",))


# ---------------------------------------------------------------------------
# Structural tests (no clang required)
# ---------------------------------------------------------------------------

def test_constant_function_define():
    ir = llvm_from_source("fn answer() -> int { 42 }")
    assert "define i64 @mx_answer()" in ir
    assert re.search(r"ret i64", ir)
    assert count_placeholders(ir) == 0


def test_arithmetic_and_comparison():
    ir = llvm_from_source("fn f(a: int, b: int) -> bool { (a + b) * b < a }")
    assert "define i64 @mx_f(i64 %a.a, i64 %a.b)" in ir
    assert re.search(r"%t\d+ = add i64", ir)
    assert re.search(r"%t\d+ = mul i64", ir)
    assert re.search(r"%t\d+ = icmp slt i64", ir)
    assert re.search(r"%t\d+ = zext i1 %t\d+ to i64", ir)


def test_if_else_uses_alloca_slot_and_conditional_branch():
    ir = llvm_from_source(
        "fn choose(x: int) -> int { if x < 5 { 10 } else { 20 } }")
    # the shared if-result variable is multiply-assigned -> alloca slot
    assert re.search(r"%slot\.\w+ = alloca i64", ir)
    assert re.search(r"br i1 %t\d+, label %bb1, label %bb2", ir)
    assert "bb3:" in ir  # join block
    assert ir.count("store i64") >= 2  # one store per arm


def test_while_loop_back_edge_and_slots():
    ir = llvm_from_source("""
fn count(n: int) -> int {
    let mut i = 0;
    while i < n { i = i + 1; }
    i
}
""")
    assert ir.count("br label %bb1") == 2  # entry->header and the back-edge
    assert re.search(r"%slot\.i_\d+ = alloca i64", ir)
    assert re.search(r"load i64, ptr %slot\.i_\d+", ir)


def test_print_string_declares_printf_and_helper():
    ir = llvm_from_source('fn main() { print("Hello from Metaxu!") }')
    assert "declare i32 @printf(ptr, ...)" in ir
    assert "define internal void @metaxu_print_str(ptr %x)" in ir
    assert re.search(
        r'@\.str\.0 = private unnamed_addr constant \[19 x i8\] '
        r'c"Hello from Metaxu!\\00"', ir)
    assert "call void @metaxu_print_str(ptr @.str.0)" in ir


def test_print_int_routes_to_i64_helper():
    ir = llvm_from_source("fn main() { print(41 + 1) }")
    assert "define internal void @metaxu_print_i64(i64 %x)" in ir
    assert re.search(r"call void @metaxu_print_i64\(i64 %t\d+\)", ir)
    assert "%lld" in ir


def test_float_math_and_print_helper():
    ir = llvm_from_source("fn main() { print(1.5 * 2.0) }")
    assert re.search(r"%t\d+ = fmul double", ir)
    assert "define internal void @metaxu_print_f64(double %x)" in ir
    # float constants are IEEE-754 bit patterns
    assert re.search(r"0x[0-9A-F]{16}", ir)


def test_float_signature_propagates_through_module_fixpoint():
    ir = llvm_from_source("""
fn halve(x: float) -> float { x * 0.5 }
fn main() -> float { halve(8.0) }
""")
    assert "define double @mx_halve(double %a.x" in ir
    assert "define double @mx_main()" in ir
    assert re.search(r"call double @mx_halve\(double", ir)


def test_float_comparison_uses_fcmp():
    ir = llvm_from_source("fn f(x: float) -> bool { x > 6.9 }")
    assert re.search(r"fcmp ogt double", ir)


def test_local_struct_gets_type_alloca_and_gep():
    ir = llvm_from_source("""
struct Point { x: int, y: int }
fn main() -> int {
    let p = Point { x: 3, y: 4 };
    p.x + p.y
}
""")
    assert "%struct.Point = type { i64, i64 }" in ir
    assert re.search(r"%sv\.\w+ = alloca %struct\.Point", ir)
    assert re.search(
        r"getelementptr inbounds %struct\.Point, ptr %sv\.\w+, i32 0, i32 0", ir)
    assert re.search(
        r"getelementptr inbounds %struct\.Point, ptr %sv\.\w+, i32 0, i32 1", ir)
    # no heap traffic for local structs: the frame owns the memory
    assert "call ptr @malloc" not in ir
    assert "call void @free" not in ir


def test_struct_field_assignment_value_semantics():
    ir = llvm_from_source("""
struct Counter { n: int }
fn main() -> int {
    let mut c = Counter { n: 0 };
    c.n = c.n + 2;
    c.n
}
""")
    # field_set copies the aggregate, then stores the field
    assert re.search(r"%t\d+ = load %struct\.Counter, ptr", ir)
    assert re.search(r"store %struct\.Counter %t\d+, ptr", ir)
    assert count_placeholders(ir) == 0


def test_nested_calls_declare_nothing_extra():
    ir = llvm_from_source("""
fn add(a: int, b: int) -> int { a + b }
fn twice(x: int) -> int { add(x, x) }
fn main() -> int { twice(21) }
""")
    assert "define i64 @mx_add(i64 %a.a, i64 %a.b)" in ir
    assert re.search(r"call i64 @mx_add\(i64 .+, i64 .+\)", ir)
    assert count_placeholders(ir) == 0


def test_logic_binop_is_truthiness_not_bitwise():
    # `2 && 1` must be true (1), not `2 & 1` (0): hand-built MIR since the
    # surface grammar has no logical operator tokens yet.
    f = make_func("f", [
        block([
            ("params", ("a", "b")),
            ("let", "r", ("binop", "&&"), ("a", "b")),
        ], ("ret", "r")),
    ])
    ir = emit_llvm([f])
    assert re.search(r"icmp ne i64 %a\.a, 0", ir)
    assert re.search(r"icmp ne i64 %a\.b, 0", ir)
    assert re.search(r"%t\d+ = and i1 ", ir)
    assert "and i64" not in ir


def test_match_fail_raises_the_interpreter_message():
    """match_fail lowers to mx_raise with the interpreter's exact wording
    (catchable by an enclosing try; fatal with no pad in flight, matching
    an uncaught InterpError)."""
    f = make_func("m", [
        block([("params", ()), ("match_fail", "no pattern matched")], ("br", 1)),
        block([("let", "u", ("const_ty", "Unit"), ())], ("ret", "u")),
    ])
    ir = emit_llvm([f])
    assert "call void @mx_raise(ptr" in ir
    assert "match failure in 'm': no pattern matched" in ir
    assert "unreachable" in ir


def test_match_fail_in_a_mono_clone_names_the_origin():
    """A monomorphization clone's match_fail names the PRE-specialization
    function (MirFunc.origin_name), binding the same string the
    unspecialized interpreter binds."""
    f = make_func("classify$Int", [
        block([("params", ()), ("match_fail", "no pattern matched")], ("br", 1)),
        block([("let", "u", ("const_ty", "Unit"), ())], ("ret", "u")),
    ])
    f.origin_name = "classify"
    ir = emit_llvm([f])
    assert "match failure in 'classify': no pattern matched" in ir
    assert "match failure in 'classify$Int'" not in ir


# ---------------------------------------------------------------------------
# Placeholder honesty
# ---------------------------------------------------------------------------

def test_suspending_function_emits_mx_perform():
    # Increment 7: suspending functions are no longer demoted — a perform
    # lowers to mx_perform against the native effects runtime (the
    # coroutine stack is the continuation; no CPS transform needed).
    f = make_func("worker", [
        block([
            ("params", ("x",)),
            ("perform", "pv1", "State", "get", (), 1, "pv1"),
        ], ("br", 1)),
        block([], ("ret", "pv1")),
    ], suspending=True)
    ir = emit_llvm([f])
    assert count_placeholders(ir) == 0
    assert "define i64 @mx_worker(i64 %a.x)" in ir
    assert "declare i64 @mx_perform(ptr, ptr, ptr, i64)" in ir
    assert "call i64 @mx_perform(" in ir
    # the perform scratch array is materialized in the entry block
    assert "%perform.args = alloca [8 x i64]" in ir


def test_unknown_locality_is_placeholder():
    # "local" allocas and "global" mallocs are lowered; anything else must
    # demote rather than guess a storage class.
    f = make_func("boxer", [
        block([
            ("params", ("n",)),
            ("let", "b1", ("alloc_struct", "Box", "region"), (("val", "n"),)),
            ("let", "v", ("field_get", "val"), ("b1",)),
        ], ("ret", "v")),
    ])
    ir = emit_llvm([f])
    assert count_placeholders(ir) == 1
    assert "unknown locality 'region'" in ir


# ---------------------------------------------------------------------------
# Increment 2 structural tests: struct calls/returns and @global heap structs
# ---------------------------------------------------------------------------

def test_let_global_annotation_reaches_mir_alloc_struct():
    # Seam regression: `let @global x = S {...}` must reach MIR as an
    # alloc_struct with locality "global" (hir.py used to drop string-token
    # mode annotations, silently degrading @global to a local alloc).
    funcs = mir_from_source("""
struct Point { x: int, y: int }
fn main() -> int {
    let @global g = Point { x: 1, y: 2 };
    let l = Point { x: 3, y: 4 };
    g.x + l.y
}
""")
    localities = [op[2][2]
                  for f in funcs for b in f.blocks for op in b.ops
                  if op[0] == "let" and op[2][0] == "alloc_struct"]
    assert localities.count("global") == 1
    assert localities.count("local") == 1

def test_struct_param_passes_ptr_readonly_callee_elides_byval_copy():
    # Increment 8: a read-only struct param (never rebound, never reaching a
    # write-back position) skips the entry byval copy and reads the caller's
    # aggregate through the passed pointer.  The caller side is unchanged.
    ir = llvm_from_source("""
struct Point { x: int, y: int }
fn getx(p: Point) -> int { p.x }
fn main() -> int {
    let p = Point { x: 3, y: 4 };
    getx(p)
}
""")
    assert count_placeholders(ir) == 0
    # callee: struct param arrives as ptr...
    assert "define i64 @mx_getx(ptr %a.p)" in ir
    # ...and is read THROUGH that pointer: no byval copy, no own storage
    assert "; elide-copy: param p reads through the caller's pointer" in ir
    assert not re.search(r"load %struct\.Point, ptr %a\.p", ir)
    assert not re.search(r"%sv\.p = alloca", ir)
    assert re.search(
        r"getelementptr inbounds %struct\.Point, ptr %a\.p", ir)
    # caller passes the storage pointer of its struct variable
    assert re.search(r"call i64 @mx_getx\(ptr %sv\.\w+\)", ir)


def test_rebound_mut_struct_param_keeps_byval_copy_and_write_back():
    # The case that MUST NOT elide: a rebound @mut struct param still
    # byval-copies on entry and copies back out through the caller's pointer
    # on ret (interpreter write-back parity).
    ir = llvm_from_source("""
struct Counter { n: int }
fn bump(c: @mut Counter) -> int {
    c.n = c.n + 1;
    c.n
}
fn main() -> int {
    let c = Counter { n: 10 };
    bump(c) + c.n
}
""")
    assert count_placeholders(ir) == 0
    bump = ir[ir.index("define i64 @mx_bump"):]
    bump = bump[:bump.index("\n}") + 2]
    assert "; elide-copy: param" not in bump
    assert re.search(r"load %struct\.Counter, ptr %a\.c", bump)  # byval in
    assert "copy-out: rebound struct param" in bump               # write-back


def test_rebound_plain_struct_param_never_writes_back():
    # Value semantics for plain params (round 5 finding 1): a NON-@mut
    # struct param the callee mutates keeps its byval copy but is NOT
    # copied back out — the caller's binding stays untouched, in both
    # engines (interpreter parity is pinned in test_round5_regressions).
    ir = llvm_from_source("""
struct Counter { n: int }
fn bump(c: Counter) -> int {
    c.n = c.n + 1;
    c.n
}
fn main() -> int {
    let c = Counter { n: 10 };
    bump(c) + c.n
}
""")
    assert count_placeholders(ir) == 0
    bump = ir[ir.index("define i64 @mx_bump"):]
    bump = bump[:bump.index("\n}") + 2]
    assert re.search(r"load %struct\.Counter, ptr %a\.c", bump)  # byval in
    assert "copy-out: rebound struct param" not in bump          # no write-back


def test_param_passed_to_rebinding_callee_keeps_byval_copy():
    # relay never rebinds c itself, but passes it to bump (@mut receiver),
    # which writes back through the pointer it is given.  relay must keep
    # its own copy so bump's write-back mutates RELAY's binding
    # (interpreter parity), never main's storage.
    ir = llvm_from_source("""
struct Counter { n: int }
fn bump(c: @mut Counter) -> int {
    c.n = c.n + 1;
    c.n
}
fn relay(c: Counter) -> int { bump(c) + c.n }
fn main() -> int {
    let c = Counter { n: 5 };
    relay(c)
}
""")
    assert count_placeholders(ir) == 0
    relay = ir[ir.index("define i64 @mx_relay"):]
    relay = relay[:relay.index("\n}") + 2]
    assert "; elide-copy: param" not in relay
    assert re.search(r"load %struct\.Counter, ptr %a\.c", relay)


def test_struct_return_is_sret_style():
    ir = llvm_from_source("""
struct Point { x: int, y: int }
fn mk() -> Point { Point { x: 1, y: 2 } }
fn main() -> int { let p = mk(); p.x }
""")
    assert count_placeholders(ir) == 0
    # callee: leading result-slot pointer, void return, aggregate copy out
    assert "define void @mx_mk(ptr %agg.ret)" in ir
    assert re.search(r"store %struct\.Point %t\d+, ptr %agg\.ret", ir)
    assert re.search(r"^  ret void", ir, re.M)
    # caller passes its own struct variable's storage as the result slot
    assert re.search(r"call void @mx_mk\(ptr %sv\.\w+\)", ir)


def test_global_struct_mallocs_in_entry_and_frees_on_ret():
    ir = llvm_from_source("""
struct Point { x: int, y: int }
fn main() -> int {
    let @global g = Point { x: 3, y: 4 };
    g.x + g.y
}
""")
    assert count_placeholders(ir) == 0
    assert "declare noalias ptr @malloc(i64)" in ir
    assert "declare void @free(ptr)" in ir
    # 2 fields x 8 bytes, malloc'd in the entry prelude, GEP on the heap ptr
    assert re.search(r"%hv\.\w+ = call ptr @malloc\(i64 16\)", ir)
    assert re.search(r"getelementptr inbounds %struct\.Point, ptr %hv\.\w+", ir)
    # every ret path frees the block before returning
    body = ir[ir.index("define i64 @mx_main"):]
    assert re.search(r"call void @free\(ptr %hv\.\w+\)", body)
    assert body.index("call void @free") < body.index("ret i64")


def test_local_structs_stay_on_the_stack_next_to_global_ones():
    ir = llvm_from_source("""
struct Point { x: int, y: int }
fn main() -> int {
    let a = Point { x: 1, y: 2 };
    let @global b = Point { x: 10, y: 20 };
    a.x + b.y
}
""")
    assert count_placeholders(ir) == 0
    assert re.search(r"%sv\.\w+ = alloca %struct\.Point", ir)  # local
    assert re.search(r"%hv\.\w+ = call ptr @malloc\(i64 16\)", ir)  # @global
    assert ir.count("call ptr @malloc") == 1  # only the @global one


def test_multi_arg_print_joins_with_spaces():
    ir = llvm_from_source('fn main() { print(1 + 1, "and", 3) }')
    assert count_placeholders(ir) == 0
    # one printf with a space-joined format string, matching print(*args)
    assert re.search(
        r'@\.str\.\d+ = private unnamed_addr constant \[\d+ x i8\] '
        r'c"%lld %s %lld\\0A\\00"', ir)
    assert re.search(
        r"call i32 \(ptr, \.\.\.\) @printf\(ptr @\.str\.\d+, "
        r"i64 %t\d+, ptr @\.str\.\d+, i64 3\)", ir)


def test_unknown_closure_target_is_placeholder():
    # A make_closure of a function that is not in the module has no fn
    # pointer to take: demote honestly.
    fs = [
        make_func("c", [block([
            ("params", ()),
            ("let", "c1", ("make_closure", "lambda1", ("x",)), ()),
        ], ("ret", "c1"))]),
    ]
    ir = emit_llvm(fs)
    assert "make_closure of unknown function 'lambda1'" in ir


def test_closure_in_loop_gets_fresh_heap_env_per_iteration():
    # Increment 4: a make_closure inside a CFG cycle no longer demotes — the
    # site mallocs a FRESH env each execution (leaked by design), so earlier
    # pair copies can never alias the new one.  Hand-built MIR variant; the
    # parsed-source loop-lambda path is pinned by
    # test_loop_lambda_captures_compile (the front-end capture gap is fixed).
    fs = [
        make_func("looper", [
            block([("params", ("n",)),
                   ("let", "i0", ("const", 0), ()),
                   ("let", "i", ("copy",), ("i0",))], ("br", 1)),
            block([("let", "b", ("binop", "<"), ("i", "n"))],
                  ("br_if", "b", 2, 3)),
            block([
                ("let", "c2", ("make_closure", "inner", ("x",)), (("k", "i"),)),
                ("let", "r", ("call", "c2"), ("i",)),
                ("let", "c1", ("const", 1), ()),
                ("let", "i2", ("binop", "+"), ("i", "c1")),
                ("let", "i", ("copy",), ("i2",)),
            ], ("br", 1)),
            block([], ("ret", "i")),
        ]),
        make_func("inner", [block([
            ("params", ("x",)),
            ("let", "s", ("binop", "+"), ("x", "k")),
        ], ("ret", "s"))]),
    ]
    ir = emit_llvm(fs)
    assert count_placeholders(ir) == 0
    # env malloc'd at the site (inside the loop body), not alloca'd in entry
    assert re.search(r"%t\d+ = call ptr @malloc\(i64 8\)  ; heap env", ir)
    assert "leaks by design" in ir
    assert not re.search(r"alloca %env\.inner", ir)


def test_returned_closure_gets_heap_env_and_emits():
    # Increment 4: `let f = fn(y) -> x + y; f` escaping upward is sound now —
    # the lambda is marked heap-env (detected from make_adder's return kind)
    # and its env is malloc'd at the site, never freed (leak by design), so
    # the returned pair's env pointer can never dangle.
    ir = llvm_from_source("""
fn make_adder(x: int) -> fn(int) -> int {
    let f = fn(y: int) -> x + y;
    f
}
fn main() -> int {
    let add2 = make_adder(2);
    add2(40)
}
""")
    assert count_placeholders(ir) == 0
    # the closure return is sret-style: pair copied into the caller's slot
    assert "define void @mx_make_adder(ptr %agg.ret, i64 %a.x)" in ir
    assert re.search(r"call ptr @malloc\(i64 8\)  ; heap env for \w+ -> [\w$]*lambda\d+", ir)
    # no free of the env anywhere: it leaks by design
    assert "call void @free" not in ir


def test_unknown_external_and_runtime_builtin_are_placeholders():
    fs = [
        make_func("a", [block([
            ("params", ()),
            ("let", "r", ("call", "mystery_ffi"), ()),
        ], ("ret", "r"))]),
        make_func("b", [block([
            ("params", ()),
            ("let", "r", ("call", "type_of"), ()),
        ], ("ret", "r"))]),
    ]
    ir = emit_llvm(fs)
    assert count_placeholders(ir) == 2
    assert "unknown external callee 'mystery_ffi'" in ir
    assert "calls runtime builtin 'type_of'" in ir


def test_caller_of_placeholder_is_demoted_for_linkability():
    fs = [
        make_func("bad", [block([
            ("params", ()),
            ("let", "r", ("call", "type_of"), ()),
        ], ("ret", "r"))]),
        make_func("good_but_calls_bad", [block([
            ("params", ()),
            ("let", "r", ("call", "bad"), ()),
        ], ("ret", "r"))]),
    ]
    ir = emit_llvm(fs)
    assert count_placeholders(ir) == 2
    assert "calls function 'bad' that is itself a placeholder" in ir


def test_llvm_run_refuses_placeholder_entry():
    f = make_func("main", [
        block([
            ("params", ()),
            ("let", "r", ("call", "type_of"), ()),
        ], ("ret", "r")),
    ])
    ir = emit_llvm([f])
    with pytest.raises(LlvmRunError, match="not natively runnable"):
        compile_and_run(ir, "main")


# ---------------------------------------------------------------------------
# All-examples emission: total, honest coverage
# ---------------------------------------------------------------------------

_REJECTED = {"test_borrow_check.mx", "test_type_error.mx"}


def _example_files():
    files = sorted((REPO_ROOT / "examples").glob("*.mx"))
    files += sorted(REPO_ROOT.glob("test_*.mx"))
    return [f for f in files if f.name not in _REJECTED]


@pytest.mark.parametrize("path", _example_files(), ids=lambda p: p.name)
def test_all_examples_emit_defines_or_explicit_placeholders(path):
    funcs = mir_from_source(path.read_text())
    ir = emit_llvm(funcs)  # must never crash
    defines = set(re.findall(r"^define (?:i64|double|ptr|void) @(\w+)\(", ir, re.M))
    placeholders = set(re.findall(
        r"^; function @(\w+): placeholder -- unsupported", ir, re.M))
    for f in funcs:
        sym = mangle(f.name)
        assert sym in defines or sym in placeholders, (
            f"function {f.name!r} is neither a define nor a placeholder")
        assert not (sym in defines and sym in placeholders), (
            f"function {f.name!r} is both a define and a placeholder")
    # every placeholder chunk carries at least one reason line
    for chunk in ir.split("\n\n"):
        if "placeholder -- unsupported" in chunk:
            assert ";   reason: " in chunk, f"placeholder without reason:\n{chunk}"


@pytest.mark.skipif(shutil.which("opt") is None, reason="LLVM opt not installed")
@pytest.mark.parametrize("path", _example_files(), ids=lambda p: p.name)
def test_all_examples_pass_llvm_verifier(path, tmp_path):
    ir = llvm_from_source(path.read_text())
    ll = tmp_path / "mod.ll"
    ll.write_text(ir)
    proc = subprocess.run(
        ["opt", "-passes=verify", "-disable-output", str(ll)],
        capture_output=True, text=True)
    assert proc.returncode == 0, f"LLVM verifier rejected module:\n{proc.stderr}"


# ---------------------------------------------------------------------------
# Native differential tests: clang-compiled binaries vs. the MIR interpreter
# ---------------------------------------------------------------------------

@needs_clang
def test_native_arithmetic_exit_code(tmp_path):
    assert_native_matches_interp("fn main() -> int { 7 + 3 * 5 - 4 }", tmp_path)


@needs_clang
def test_native_if_else_branches(tmp_path):
    assert_native_matches_interp("""
fn pick(x: int) -> int { if x < 5 { 10 } else { 20 } }
fn main() -> int { print(pick(3)); print(pick(7)); pick(3) + pick(7) }
""", tmp_path)


@needs_clang
def test_native_while_loop_accumulator(tmp_path):
    assert_native_matches_interp("""
fn main() -> int {
    let mut i = 0;
    let mut acc = 0;
    while i < 10 { i = i + 1; acc = acc + i; }
    print(acc);
    acc
}
""", tmp_path)


@needs_clang
def test_native_nested_calls(tmp_path):
    assert_native_matches_interp("""
fn add(a: int, b: int) -> int { a + b }
fn twice(x: int) -> int { add(x, x) }
fn main() -> int { print(twice(add(10, 11))); twice(add(10, 11)) }
""", tmp_path)


@needs_clang
def test_native_float_math_via_comparison_branches(tmp_path):
    # Float results are observed through comparisons and integer prints so
    # printf formatting differences ("%g" vs Python str) cannot cause a
    # spurious mismatch.
    assert_native_matches_interp("""
fn main() -> int {
    let x = 3.5 * 2.0;
    if x > 6.9 { print(1); } else { print(0); }
    let y = 1.5 + 2.25;
    if y == 3.75 { print(100); } else { print(200); }
    if 1.0 / 4.0 < 0.3 { print(11); } else { print(22); }
    0
}
""", tmp_path)


@needs_clang
def test_native_comparisons_as_branch_conditions(tmp_path):
    assert_native_matches_interp("""
fn main() -> int {
    let mut hits = 0;
    if 3 < 4 { hits = hits + 1; }
    if 4 <= 4 { hits = hits + 1; }
    if 5 > 4 { hits = hits + 1; }
    if 5 >= 6 { hits = hits + 1; }
    if 7 == 7 { hits = hits + 1; }
    if 7 != 7 { hits = hits + 1; }
    print(hits);
    hits
}
""", tmp_path)


@needs_clang
def test_native_integer_division(tmp_path):
    # Non-negative operands: sdiv agrees with the interpreter's floor //.
    assert_native_matches_interp(
        "fn main() -> int { print(17 / 5); print(100 / 4 / 5); 0 }", tmp_path)


@needs_clang
def test_native_print_int_stdout(tmp_path):
    assert_native_matches_interp(
        "fn main() { print(42); print(0 - 7); print(1000000) }", tmp_path)


@needs_clang
def test_native_print_string_stdout(tmp_path):
    assert_native_matches_interp(
        (REPO_ROOT / "examples" / "hello.mx").read_text(), tmp_path)


@needs_clang
def test_native_local_struct_program(tmp_path):
    ir = assert_native_matches_interp("""
struct Point { x: int, y: int }
fn dist2(a: int, b: int) -> int { a * a + b * b }
fn main() -> int {
    let p = Point { x: 3, y: 4 };
    let d = dist2(p.x, p.y);
    print(d);
    d
}
""", tmp_path)
    assert "%struct.Point = type { i64, i64 }" in ir


@needs_clang
def test_native_struct_field_assignment_in_loop(tmp_path):
    assert_native_matches_interp("""
struct Counter { n: int }
fn main() -> int {
    let mut c = Counter { n: 0 };
    let mut i = 0;
    while i < 5 { c.n = c.n + 2; i = i + 1; }
    print(c.n);
    c.n
}
""", tmp_path)


@needs_clang
def test_native_result_larger_than_exit_code_range_via_stdout(tmp_path):
    # 1000 does not fit in an exit code; the differential goes through
    # stdout, and the exit-code assertion is modulo 256 by convention.
    assert_native_matches_interp(
        "fn main() -> int { print(999 + 1); 999 + 1 }", tmp_path)


# ---------------------------------------------------------------------------
# Increment 2 native differentials: value semantics across calls, sret
# returns, @global heap structs (ASan-proven frees), multi-arg print
# ---------------------------------------------------------------------------

@needs_clang
def test_native_struct_param_mutation_does_not_leak_to_caller(tmp_path):
    # The callee byval-copies its struct parameter, so mutating the local
    # copy must not affect the caller's struct (MIR value semantics).
    assert_native_matches_interp("""
struct Box { v: int }
fn bump(b: Box) -> int {
    let mut c = b;
    c.v = c.v + 100;
    c.v
}
fn main() -> int {
    let bx = Box { v: 7 };
    let r = bump(bx);
    print(r);
    print(bx.v);
    bx.v
}
""", tmp_path)


@needs_clang
def test_native_struct_returned_from_function(tmp_path):
    ir = assert_native_matches_interp("""
struct Point { x: int, y: int }
fn mk(a: int, b: int) -> Point { Point { x: a * 2, y: b + 5 } }
fn main() -> int {
    let p = mk(10, 20);
    print(p.x);
    print(p.y);
    p.x + p.y
}
""", tmp_path)
    assert "define void @mx_mk(ptr %agg.ret, i64 %a.a, i64 %a.b)" in ir


@needs_clang
def test_native_struct_through_call_chain(tmp_path):
    # A struct crossing two frames: passed in, updated (value semantics),
    # and returned back out sret-style.
    assert_native_matches_interp("""
struct Point { x: int, y: int }
fn shift(p: Point, dx: int) -> Point {
    Point { x: p.x + dx, y: p.y }
}
fn main() -> int {
    let a = Point { x: 3, y: 4 };
    let b = shift(a, 10);
    print(a.x, b.x, b.y);
    a.x + b.x + b.y
}
""", tmp_path)


@needs_clang
@needs_asan
def test_native_global_struct_alloc_use_free_under_asan(tmp_path):
    # Exit 0 under -fsanitize=address proves the malloc/free pairing sound:
    # no leak (LeakSanitizer), no double-free, no use-after-free.
    ir = assert_native_matches_interp_asan("""
struct Point { x: int, y: int }
fn main() -> int {
    let @global g = Point { x: 30, y: 12 };
    print(g.x + g.y);
    0
}
""", tmp_path)
    assert re.search(r"call ptr @malloc\(i64 16\)", ir)
    assert re.search(r"call void @free\(ptr %hv\.\w+\)", ir)


@needs_clang
@needs_asan
def test_native_mixed_local_and_global_structs_under_asan(tmp_path):
    # A function mixing frame-owned and heap-owned structs, with the heap
    # value crossing a call boundary by copy (the callee must never retain
    # or free the caller's heap block).
    ir = assert_native_matches_interp_asan("""
struct Pair { a: int, b: int }
fn total(p: Pair) -> int { p.a + p.b }
fn main() -> int {
    let stackp = Pair { a: 1, b: 2 };
    let @global heapp = Pair { a: 10, b: 20 };
    print(total(stackp));
    print(total(heapp));
    print(stackp.a + heapp.b);
    0
}
""", tmp_path)
    assert ir.count("call ptr @malloc") == 1
    # exactly one free (in main); the callee frees nothing it did not malloc
    assert ir.count("call void @free") == 1


@needs_clang
@needs_asan
def test_native_global_struct_in_loop_under_asan(tmp_path):
    # The @global alloc site executes per iteration in MIR, but the variable
    # has a single storage block (value semantics: each def overwrites it
    # wholesale), so one malloc + one free per invocation is sound and
    # ASan-clean.
    assert_native_matches_interp_asan("""
struct Acc { n: int }
fn main() -> int {
    let mut total = 0;
    let mut i = 0;
    while i < 4 {
        let @global a = Acc { n: i * 10 };
        total = total + a.n;
        i = i + 1;
    }
    print(total);
    0
}
""", tmp_path)


@needs_clang
def test_native_multi_arg_print_matches_interpreter_join(tmp_path):
    # Interpreter print(*args) joins with a single space; the native printf
    # format string must match exactly.
    assert_native_matches_interp("""
fn main() -> int {
    print(1, 2, 3);
    print("x", 42);
    print(7 * 6, "is the answer");
    0
}
""", tmp_path)


# ---------------------------------------------------------------------------
# Increment 3 structural tests: enum tagged unions
# ---------------------------------------------------------------------------

_OPTION_MATCH_SRC = """
enum Option { Some(int), None }
fn unwrap_or(o: Option, d: int) -> int {
    match o { Some(x) -> x, None -> d }
}
fn main() -> int {
    let a = Some(5);
    let b = None;
    print(unwrap_or(a, 0));
    print(unwrap_or(b, 7));
    unwrap_or(a, 0)
}
"""


def test_enum_tagged_union_type_and_documented_tags():
    ir = llvm_from_source(_OPTION_MATCH_SRC)
    assert count_placeholders(ir) == 0
    # tagged union: integer tag + payload slots sized to the largest variant
    assert "%enum.Option = type { i64, [1 x i64] }" in ir
    # the module documents the dense variant-name -> integer mapping
    assert "; variant tag mapping (module-wide, dense): None=0, Some=1" in ir
    # make_variant stores the integer tag (with a doc comment)...
    assert re.search(r"store i64 1, ptr %t\d+  ; tag Some=1", ir)
    assert re.search(r"store i64 0, ptr %t\d+  ; tag None=0", ir)
    # ...and the payload through a GEP into the slot array
    assert re.search(
        r"getelementptr inbounds %enum\.Option, ptr %sv\.\w+, i32 0, i32 1, i32 0", ir)


def test_enum_pattern_tag_test_compares_integers_not_strings():
    ir = llvm_from_source(_OPTION_MATCH_SRC)
    # the compiled pattern's variant-name strings lower to integer tags:
    # no string constant for a variant name reaches the module
    assert '"Some' not in ir and '"None' not in ir
    assert "; tag literal: 'Some' -> 1" in ir
    assert re.search(r"icmp eq i64 %t\d+, 1", ir)  # tag == Some
    # enum values cross the call boundary as ptr; unwrap_or never rebinds
    # its param, so (increment 8) the byval copy is elided and the tag/
    # payload reads go through the caller's pointer directly
    assert "define i64 @mx_unwrap_or(ptr %a.o, i64 %a.d)" in ir
    assert "; elide-copy: param o reads through the caller's pointer" in ir
    assert re.search(r"getelementptr inbounds %enum\.Option, ptr %a\.o", ir)


def test_enum_returned_from_function_is_sret_style():
    ir = llvm_from_source("""
enum Option { Some(int), None }
fn mk(n: int) -> Option { if n > 0 { Some(n) } else { None } }
fn main() -> int { let o = mk(3); match o { Some(x) -> x, None -> 0 } }
""")
    assert count_placeholders(ir) == 0
    assert "define void @mx_mk(ptr %agg.ret, i64 %a.n)" in ir
    assert re.search(r"store %enum\.Option %t\d+, ptr %agg\.ret", ir)
    assert re.search(r"call void @mx_mk\(ptr %sv\.\w+, i64 3\)", ir)


def test_recursive_enum_payload_is_heap_boxed():
    # Increment 4: a linked-list-style enum lives in flat 8-byte payload
    # slots because the recursive slot holds a heap POINTER to a boxed copy.
    # Increment 8 refines the free strategy: the OUTER list value is only
    # ever matched in this frame, so its box is uniquely owned and freed at
    # frame exit; the INNER Cons box (re-boxed as the outer's payload, its
    # pointer shallow-copied into the outer box) still leaks by design.
    ir = llvm_from_source("""
enum IntList { Cons(int, IntList), Nil }
fn main() -> int {
    let l = Cons(1, Cons(2, Nil));
    match l { Cons(h, t) -> h, Nil -> 0 }
}
""")
    assert count_placeholders(ir) == 0
    # finite layout: tag + 2 slots (i64 head, boxed-tail ptr as 8 bytes)
    assert "%enum.IntList = type { i64, [2 x i64] }" in ir
    # make_variant boxes the aggregate payload: malloc(24) = the IntList
    # size.  Inner box: shared (leaks); outer box: unique (freed).
    assert re.search(
        r"call ptr @malloc\(i64 24\)  ; boxed enum:IntList payload "
        r"\(leaks by design\)", ir)
    assert re.search(
        r"call ptr @malloc\(i64 24\)  ; boxed enum:IntList payload "
        r"\(unique: freed at frame exit\)", ir)
    # the box is filled with a whole-aggregate copy, then the ptr stored
    assert re.search(r"store %enum\.IntList %t\d+, ptr %t\d+", ir)
    # variant_field on the boxed slot loads the ptr and copies the value out
    assert re.search(r"%t\d+ = load ptr, ptr %t\d+", ir)
    # exactly the unique box is freed, nothing else
    assert ir.count("call void @free") == 1
    assert "; unique box:" in ir


def test_per_variant_payload_slots_emit_mixed_variant_kinds():
    # Increment 6: payload slot kinds are PER VARIANT — A(int) and B(str)
    # sharing slot 0 no longer demotes (each variant knows its own slot
    # representation), and the IR documents the per-variant kinds.
    ir = llvm_from_source("""
enum Mix { A(int), B(str) }
fn main() -> int {
    let a = A(1);
    let b = B("x");
    0
}
""")
    assert count_placeholders(ir) == 0
    assert ";   variant A(i64)" in ir
    assert ";   variant B(str)" in ir


def test_merged_same_variant_mixed_instantiation_still_demotes():
    # PER-VALUE limits: ONE value (pick's merged result) holding both
    # Some(int) and Some(str) has no single native slot representation, so
    # the writer demotes instead of coercing — the honest boundary of the
    # per-variant model.
    ir = llvm_from_source("""
enum Opt { Some(int), None }
fn pick(n: int) -> Opt { if n > 0 { Some(1) } else { Some("s") } }
fn main() -> int { let x = pick(1); 0 }
""")
    assert count_placeholders(ir) >= 1
    assert "; function @mx_pick: placeholder" in ir
    assert re.search(
        r"heterogeneous payload slot 0 of enum 'Opt': variant 'Some' stores "
        r"(i64|str) where merged flows require (str|i64)", ir)
    assert "no coercion through tagged-union storage" in ir


def test_nested_mixed_enum_extraction_demotes():
    # An enum whose slot representation is instantiation-dependent (W holds
    # i64 in one use, str in another) loses its per-value refinement when
    # boxed inside ANOTHER enum's payload: extraction would have to guess a
    # representation, so the reader demotes with the boxing-boundary reason.
    ir = llvm_from_source("""
enum Inner { W(int), Z }
enum Outer { O(Inner), E }
fn use_int() -> Inner { W(1) }
fn use_str() -> Inner { W("s") }
fn peel(o: Outer) -> int {
    match o { O(x) -> match x { W(y) -> 0, Z -> 1 }, E -> 2 }
}
fn main() -> int {
    let a = O(use_int());
    let b = O(use_str());
    peel(a)
}
""")
    assert "; function @mx_peel: placeholder" in ir
    assert "extracts nested enum 'Inner'" in ir
    assert "instantiation-dependent" in ir
    # the writers themselves stay native: each value knows its own kinds
    for fname in ("use_int", "use_str"):
        assert re.search(rf"^define \S+ @mx_{fname}\(", ir, re.M)


def test_dead_statement_position_match_result_does_not_poison_kinds():
    # Statement-position if/match results are copy-merged from arms of
    # different kinds (unit/i64 vs enum).  Those results are provably dead;
    # the dead copies must be elided rather than unifying a live i64 loop
    # flag with an enum kind.
    ir = llvm_from_source("""
enum State { Go(int), Stop }
fn main() -> int {
    let mut cur = Go(2);
    let mut running = 1;
    while running == 1 {
        match cur {
            Go(x) -> { if x == 0 { running = 0; } else { cur = Go(x - 1); } },
            Stop -> { running = 0; }
        }
    }
    0
}
""")
    assert count_placeholders(ir) == 0
    assert "; dead copy elided:" in ir


# ---------------------------------------------------------------------------
# Increment 3 structural tests: closures
# ---------------------------------------------------------------------------

_CLOSURE_SRC = """
fn main() -> int {
    let x = 10;
    let g = fn(y: int) -> x + y;
    g(5)
}
"""


def test_closure_pair_env_struct_and_leading_env_param():
    ir = llvm_from_source(_CLOSURE_SRC)
    assert count_placeholders(ir) == 0
    # closure value representation: the {fn, env} pair type
    assert "%mx.closure = type { ptr, ptr }" in ir
    # per-lambda env struct holding the captured value
    assert re.search(r"%env\.[\w$]*lambda\d+ = type \{ i64 \}", ir)
    # site: env alloca, capture store, then fn+env stored into the pair
    assert re.search(r"%env\.site0\.[\w$.]+ = alloca %env\.[\w$]*lambda\d+", ir)
    assert re.search(r"store ptr @mx_[\w$]*lambda\d+, ptr %t\d+", ir)
    assert re.search(r"store ptr %env\.site0\.[\w$.]+, ptr %t\d+", ir)
    # the lambda takes env as a leading param and reloads the capture
    assert re.search(r"define i64 @mx_[\w$]*lambda\d+\(ptr %cl\.env, i64 %a\.y\)", ir)
    assert re.search(r"%cap\.\w+ = load i64, ptr %capp\.\w+", ir)


def test_closure_call_loads_fn_and_env_from_pair():
    ir = llvm_from_source(_CLOSURE_SRC)
    # the call goes through the pair: load fn ptr, load env ptr, indirect call
    assert re.search(
        r"getelementptr inbounds %mx\.closure, ptr %sv\.\w+, i32 0, i32 0", ir)
    assert re.search(
        r"getelementptr inbounds %mx\.closure, ptr %sv\.\w+, i32 0, i32 1", ir)
    assert re.search(r"%t\d+ = call i64 %t\d+\(ptr %t\d+, i64 5\)", ir)


_LOOP_LAMBDA_SRC = """
fn main() -> int {
    let mut i = 0;
    let mut s = 0;
    while i < 3 {
        let f = fn(y: int) -> y + i;
        s = s + f(1);
        i = i + 1;
    }
    s
}
"""


def test_loop_lambda_captures_compile():
    # The former front-end gap (empty capture lists for lambdas created in
    # loop bodies) is FIXED: the lambda now captures `i` for real, so the
    # whole program emits with zero placeholders.  The loop-aliasing hazard
    # is handled by per-execution heap envs (the make_closure site sits in
    # a CFG cycle, so each iteration mallocs a fresh env — see
    # test_closure_in_loop_gets_fresh_heap_env_per_iteration).
    ir = llvm_from_source(_LOOP_LAMBDA_SRC)
    assert count_placeholders(ir) == 0
    assert re.search(r"call ptr @malloc\(i64 8\)  ; heap env", ir)
    assert re.search(r"define i64 @mx_[\w$]*lambda\d+\(ptr %cl\.env, i64 %a\.\w+\)", ir)


@needs_clang
def test_native_loop_lambda_captures(tmp_path):
    # Differential for the fixed loop-lambda captures: (1+0)+(1+1)+(1+2)=6.
    assert_native_matches_interp(_LOOP_LAMBDA_SRC, tmp_path)


# ---------------------------------------------------------------------------
# Increment 3 native differentials: enums and closures vs. the interpreter
# ---------------------------------------------------------------------------

@needs_clang
def test_native_enum_match_with_payload_extraction(tmp_path):
    # Some(5) -> 5, None -> default: the canonical Option shape.
    assert_native_matches_interp(_OPTION_MATCH_SRC, tmp_path)


@needs_clang
def test_native_enum_match_multiple_variants_and_literal_patterns(tmp_path):
    # Mixed ctor patterns, a literal subpattern inside a ctor (Add(0)), and a
    # nullary variant, all against the same scrutinee.
    assert_native_matches_interp("""
enum Op { Add(int), Mul(int), Nop }
fn apply(op: Op, base: int) -> int {
    match op {
        Add(0) -> base,
        Add(x) -> base + x,
        Mul(x) -> base * x,
        Nop -> 0 - base
    }
}
fn main() -> int {
    print(apply(Add(0), 10));
    print(apply(Add(5), 10));
    print(apply(Mul(3), 10));
    print(apply(Nop, 10));
    apply(Mul(2), 7)
}
""", tmp_path)


@needs_clang
def test_native_enum_state_machine_while_loop(tmp_path):
    # A while loop driven by re-matching a mutable enum variable each
    # iteration (linked-list-style traversal without aggregate payloads:
    # recursive enum chains demote until payloads can be heap-boxed).
    assert_native_matches_interp("""
enum State { Go(int), Stop }
fn main() -> int {
    let mut cur = Go(5);
    let mut sum = 0;
    let mut running = 1;
    while running == 1 {
        match cur {
            Go(x) -> {
                sum = sum + x;
                if x == 0 { running = 0; } else { cur = Go(x - 1); }
            },
            Stop -> { running = 0; }
        }
    }
    print(sum);
    sum
}
""", tmp_path)


@needs_clang
def test_native_enum_returned_and_str_payload(tmp_path):
    # Enums crossing frames both ways (param + sret return) and a string
    # payload slot.  (Since increment 6 slot kinds are per-variant, so an
    # int-payload variant sharing slot 0 would also be fine — see
    # test_native_mixed_variant_tree_sum.)
    assert_native_matches_interp("""
enum Msg { Text(str), Shout(str, int), Empty }
fn pick(n: int) -> Msg {
    if n == 0 { Empty } else { if n < 0 { Text("negative") } else { Shout("plus", n) } }
}
fn show(m: Msg) -> int {
    match m {
        Text(s) -> { print(s); 1 },
        Shout(s, k) -> { print(s, k); 2 },
        Empty -> 0
    }
}
fn main() -> int {
    print(show(pick(3)));
    print(show(pick(0 - 4)));
    print(show(pick(0)));
    0
}
""", tmp_path)


@needs_clang
def test_native_closure_with_one_capture(tmp_path):
    assert_native_matches_interp("""
fn main() -> int {
    let x = 10;
    let g = fn(y: int) -> x + y;
    print(g(5));
    print(g(0 - 3));
    g(1)
}
""", tmp_path)


@needs_clang
def test_native_closure_with_two_captures(tmp_path):
    assert_native_matches_interp("""
fn main() -> int {
    let a = 100;
    let b = 3;
    let h = fn(y: int) -> a - b + y;
    print(h(2));
    print(h(0));
    h(0 - 1)
}
""", tmp_path)


@needs_clang
def test_native_closure_passed_as_argument(tmp_path):
    # The receiving function calls a closure it did not create: the call
    # loads fn+env from the pair parameter (env still live: it sits in the
    # caller's frame below us).
    ir = assert_native_matches_interp("""
fn apply(f: fn(int) -> int, v: int) -> int { f(v) }
fn twice(f: fn(int) -> int, v: int) -> int { f(f(v)) }
fn main() -> int {
    let x = 10;
    let g = fn(y: int) -> x + y;
    print(apply(g, 7));
    print(twice(g, 7));
    apply(g, 1)
}
""", tmp_path)
    assert "define i64 @mx_apply(ptr %a.f, i64 %a.v)" in ir


@needs_clang
def test_native_closure_capturing_enum_aggregate(tmp_path):
    # Aggregate captures are copied whole into the env struct.
    assert_native_matches_interp("""
enum Option { Some(int), None }
fn unwrap_or(o: Option, d: int) -> int {
    match o { Some(x) -> x, None -> d }
}
fn main() -> int {
    let a = Some(5);
    let g = fn(y: int) -> y + unwrap_or(a, 0);
    print(g(2));
    g(10)
}
""", tmp_path)


@needs_clang
@needs_asan
def test_native_enums_closures_and_global_structs_under_asan(tmp_path):
    # Enums and closures are pure stack storage; mixing them with a @global
    # heap struct must stay ASan-clean (no leak / double-free / UAF).
    ir = assert_native_matches_interp_asan("""
struct Pair { a: int, b: int }
enum Option { Some(int), None }
fn get(o: Option, d: int) -> int { match o { Some(x) -> x, None -> d } }
fn main() -> int {
    let @global p = Pair { a: 30, b: 12 };
    let o = Some(p.a);
    let f = fn(y: int) -> get(o, 0) + y;
    print(f(p.b));
    0
}
""", tmp_path)
    assert ir.count("call ptr @malloc") == 1
    assert ir.count("call void @free") == 1


# ---------------------------------------------------------------------------
# Increment 4 structural tests: boxed payloads, inline nested fields,
# heap closure envs, and the demotions that (honestly) remain
# ---------------------------------------------------------------------------

def assert_native_matches_interp_asan_boxes(source: str, tmp_path,
                                            entry: str = "main"):
    """Differential + ASan for programs whose payload boxes / heap closure
    envs LEAK BY DESIGN: runs with ASAN_OPTIONS=detect_leaks=0, so exit 0
    proves no use-after-free and no double-free (NOT leak-freedom — that is
    exactly the documented free-strategy contract).  Only call under
    @needs_asan."""
    result, expected_out = interp_run(source, entry)
    assert result in (UNIT, 0), "ASan differential sources must exit 0"
    ir = llvm_from_source(source)
    exit_code, stdout = compile_and_run(
        ir, entry, workdir=str(tmp_path),
        clang_args=("-fsanitize=address",),
        run_env={"ASAN_OPTIONS": "detect_leaks=0"})
    assert stdout == expected_out
    assert exit_code == 0
    return ir


_NESTED_STRUCT_SRC = """
struct Inner { a: int, b: int }
struct Outer { inner: Inner, k: int }
fn main() -> int {
    let mut o = Outer { inner: Inner { a: 1, b: 2 }, k: 3 };
    o.k = o.inner.a + 10;
    o.inner = Inner { a: 7, b: 8 };
    print(o.k, o.inner.a, o.inner.b);
    o.k + o.inner.b
}
"""


def test_nested_struct_field_is_inlined_no_heap():
    # Increment 4: struct-in-struct fields inline the nested type in the
    # parent layout — recursive GEPs, pure stack, zero heap traffic.
    ir = llvm_from_source(_NESTED_STRUCT_SRC)
    assert count_placeholders(ir) == 0
    assert "%struct.Inner = type { i64, i64 }" in ir
    assert "%struct.Outer = type { %struct.Inner, i64 }" in ir
    # field_get of the nested field copies the whole inner aggregate out
    assert re.search(r"load %struct\.Inner, ptr %t\d+", ir)
    # no heap anywhere: inline nesting is fully stack-based
    assert "call ptr @malloc" not in ir
    assert "call void @free" not in ir


_LINKED_LIST_SHAPE_SRC = """
struct Node { data: int, next: Option }
fn prepend(list: Option, value: int) -> Option {
    Some(Node { data: value, next: list })
}
fn sum(list: Option) -> int {
    match list { Some(n) -> n.data + sum(n.next), None -> 0 }
}
fn main() -> int {
    let l0 = None;
    let l1 = prepend(l0, 3);
    let l2 = prepend(l1, 2);
    let l3 = prepend(l2, 1);
    print(sum(l3));
    sum(l3)
}
"""


def test_struct_enum_mutual_recursion_boxes_at_the_enum_slot():
    # The linked_list.mx shape: struct Node holds enum Option INLINE, and
    # Option's payload slot boxes struct Node — the box is what makes the
    # mutually recursive layout finite.
    ir = llvm_from_source(_LINKED_LIST_SHAPE_SRC)
    assert count_placeholders(ir) == 0
    # Node inlines the Option enum; Option's slot holds the box ptr as i64
    assert "%struct.Node = type { i64, %enum.Option }" in ir
    assert "%enum.Option = type { i64, [1 x i64] }" in ir
    # boxed payload: malloc of sizeof(Node) = 8 (data) + 16 (inline Option)
    assert re.search(
        r"call ptr @malloc\(i64 24\)  ; boxed struct:Node payload "
        r"\(leaks by design\)", ir)
    assert "call void @free" not in ir  # boxes leak by design


def test_linked_list_example_emits_all_real_bodies():
    # Increment 6: per-value payload refinements let pop_front/remove_next/
    # get/get_mut — whose REAL bodies put both ints (Some(node.data)) and
    # Nodes (Some(next_node)) into Option's slot 0 — emit natively: each
    # Option VALUE knows its own instantiation's representation.  Since the
    # front-end statement-rule fix (else-less if/if-let is unit-valued),
    # main's trailing if-let no longer merges unit with struct:Node, so the
    # whole example is fully native: zero placeholders.
    #
    # The native path monomorphizes (pipeline.emit_llvm_from_source), so
    # push_front — whose T resolves to Int at all three call sites — emits
    # under its specialized name and the fully-specialized original is
    # erased. Every other generic here has a call site the pass cannot
    # resolve (new_list() takes no arguments; get/get_mut/take_node take the
    # list, whose T only appears inside the type application LinkedList[T]),
    # so those keep their generic name.
    ir = llvm_from_source(
        (REPO_ROOT / "examples" / "linked_list.mx").read_text())
    for fname in ("new_list", "push_front$Int", "pop_front", "remove_next",
                  "get", "get_mut", "take_node", "main"):
        assert re.search(rf"^define (?:i64|double|ptr|void) @{mangle(fname)}\(",
                         ir, re.M), f"{fname} did not emit"
    # The erased original must be gone, not silently emitted alongside.
    assert not re.search(r"^define .* @mx_push_front\(", ir, re.M)
    assert count_placeholders(ir) == 0
    # the IR documents the per-value nature of the shared Some slot
    assert ";   variant Some(boxed struct:Node (mixed per value))" in ir


def test_struct_in_struct_cycle_demotes():
    # A struct-in-struct cycle with no intervening enum box has no finite
    # inline layout (hand-built MIR: the surface type system would reject
    # constructing such a value, but codegen must still never diverge).
    fs = [
        make_func("f", [block([
            ("params", ("x",)),
            ("let", "a1", ("alloc_struct", "A", "local"), (("f", "x"),)),
            ("let", "a2", ("alloc_struct", "A", "local"), (("f", "a1"),)),
            ("let", "r", ("field_get", "f"), ("a2",)),
        ], ("ret", "x"))]),
    ]
    ir = emit_llvm(fs)
    assert count_placeholders(ir) == 1
    assert "recursively inlined layout" in ir
    # the impossible type must not be emitted either
    assert "%struct.A = type" not in ir


def test_legacy_variant_field_without_ctor_name_demotes():
    # Hand-built MIR using the pre-increment-6 two-element variant_field op
    # (no ctor name): payload slot kinds are per-variant now, so the read
    # cannot be attributed to a variant and demotes honestly instead of
    # guessing (the interpreter still accepts the legacy shape).
    fs = [
        make_func("f", [block([
            ("params", ()),
            ("let", "c", ("const", 1), ()),
            ("let", "v", ("make_variant", "E", "Only"), ("c",)),
            ("let", "x", ("variant_field", 0), ("v",)),
        ], ("ret", "x"))]),
    ]
    ir = emit_llvm(fs)
    assert count_placeholders(ir) == 1
    assert "carries no variant name (legacy MIR shape" in ir


def test_closure_in_enum_payload_still_demotes():
    # A closure pair stored in a boxed payload could outlive its stack env
    # (and env-escape analysis does not chase payload flow): demote honestly.
    ir = llvm_from_source("""
enum Holder { Fn(fn(int) -> int), Nothing }
fn main() -> int {
    let x = 1;
    let g = fn(y: int) -> x + y;
    let h = Fn(g);
    0
}
""")
    assert count_placeholders(ir) >= 1
    assert "payload slot 0 holds a closure" in ir


def test_mixed_scalar_aggregate_variants_emit_per_variant():
    # Increment 6 lifts the old per-slot caveat: Leaf(int) | Fork(Tree,
    # Tree) puts i64 and a boxed aggregate in the SAME slot index, but each
    # variant's slots are typed separately, so this now emits — with the
    # per-variant kinds documented on the enum type.
    ir = llvm_from_source("""
enum Tree { Leaf(int), Fork(Tree, Tree) }
fn main() -> int {
    let t = Fork(Leaf(1), Leaf(2));
    0
}
""")
    assert count_placeholders(ir) == 0
    assert "%enum.Tree = type { i64, [2 x i64] }" in ir
    assert ";   variant Leaf(i64)" in ir
    assert ";   variant Fork(boxed enum:Tree, boxed enum:Tree)" in ir


# ---------------------------------------------------------------------------
# Increment 4 native differentials: recursive data and escaping closures
# ---------------------------------------------------------------------------

_CONS_SUM_SRC = """
enum IntList { Cons(int, IntList), Nil }
fn sum(l: IntList) -> int {
    match l { Cons(h, t) -> h + sum(t), Nil -> 0 }
}
fn main() -> int {
    let l = Cons(1, Cons(2, Cons(3, Nil)));
    print(sum(l));
    sum(l)
}
"""


@needs_clang
def test_native_linked_list_three_node_sum(tmp_path):
    # The increment's target program: build a 3-node list, sum it natively,
    # match the interpreter.
    assert_native_matches_interp(_CONS_SUM_SRC, tmp_path)


@needs_clang
def test_native_tree_nested_enum_two_levels(tmp_path):
    # A tree-shaped enum nested two levels deep, recursively summed.  Slots
    # stay homogeneous: slot 0 is always int, slots 1-2 always Tree.
    assert_native_matches_interp("""
enum Tree { Branch(int, Tree, Tree), Empty }
fn total(t: Tree) -> int {
    match t {
        Branch(v, l, r) -> v + total(l) + total(r),
        Empty -> 0
    }
}
fn main() -> int {
    let t = Branch(1, Branch(2, Empty, Branch(4, Empty, Empty)), Branch(3, Empty, Empty));
    print(total(t));
    total(t)
}
""", tmp_path)


# The increment-6 target shapes: per-variant slot typing (Leaf/Fork) and
# per-value instantiation typing (one Some holding ints AND nodes).

_MIXED_TREE_SUM_SRC = """
enum Tree { Leaf(int), Fork(Tree, Tree) }
fn total(t: Tree) -> int {
    match t { Leaf(n) -> n, Fork(l, r) -> total(l) + total(r) }
}
fn main() -> int {
    let t = Fork(Fork(Leaf(1), Leaf(2)), Fork(Leaf(3), Fork(Leaf(4), Leaf(5))));
    print(total(t));
    total(t)
}
"""

_POP_FRONT_SUM_SRC = """
struct Node { data: int, next: Option }
struct List { head: Option }
fn push_front(list: @mut List, value: int) {
    let new_node = Node { data: value, next: list.head };
    list.head = Some(new_node);
}
fn pop_front(list: @mut List) -> Option {
    if let Some(head) = list.head {
        list.head = head.next;
        return Some(head.data)
    } else {
        return None
    }
}
fn sum(list: List) -> int {
    let mut cur = list.head;
    let mut acc = 0;
    let mut going = 1;
    while going == 1 {
        if let Some(node) = cur {
            acc = acc + node.data;
            cur = node.next;
        } else {
            going = 0;
        }
    }
    acc
}
fn main() -> int {
    let @mut l = List { head: None };
    push_front(l, 3);
    push_front(l, 2);
    push_front(l, 1);
    let popped = pop_front(l);
    if let Some(v) = popped {
        print(v);
    } else {
        print(0 - 1);
    }
    print(sum(l));
    sum(l)
}
"""


@needs_clang
def test_native_mixed_variant_tree_sum(tmp_path):
    # Leaf(int) | Fork(Tree, Tree): slot 0 is i64 for Leaf and a boxed Tree
    # for Fork — the previous increments' caveat, now summed natively.
    ir = assert_native_matches_interp(_MIXED_TREE_SUM_SRC, tmp_path)
    assert ";   variant Fork(boxed enum:Tree, boxed enum:Tree)" in ir


@needs_clang
def test_native_pop_front_mixed_instantiation_list(tmp_path):
    # linked_list.mx's exact reality end-to-end: ONE generic Some variant
    # holds Node inside the list and int out of pop_front.  Build a list,
    # pop the head, sum the rest — natively, matching the interpreter.
    ir = assert_native_matches_interp(_POP_FRONT_SUM_SRC, tmp_path)
    assert ";   variant Some(boxed struct:Node (mixed per value))" in ir


@needs_clang
@needs_asan
def test_native_mixed_variant_tree_sum_no_uaf_under_asan(tmp_path):
    # ASan (leaks off: boxes leak BY DESIGN) proves the per-variant boxing
    # traffic has no use-after-free / double-free.
    src = _MIXED_TREE_SUM_SRC.replace("total(t)\n}", "total(t) - 15\n}", 1)
    assert_native_matches_interp_asan_boxes(src, tmp_path)


@needs_clang
@needs_asan
def test_native_pop_front_mixed_instantiation_no_uaf_under_asan(tmp_path):
    # Same contract-scoped ASan proof for the per-value mixed Some slot:
    # raw i64 stores and boxed Node stores share the slot index without a
    # single stray free or out-of-bounds box access.
    src = _POP_FRONT_SUM_SRC.replace("sum(l)\n}", "sum(l) - 5\n}", 1)
    assert_native_matches_interp_asan_boxes(src, tmp_path)


@needs_clang
def test_native_struct_with_struct_field_value_semantics(tmp_path):
    # Inline nested fields keep value semantics: copying the outer struct
    # must deep-copy the inline inner region (it is part of the same bytes).
    assert_native_matches_interp("""
struct Inner { a: int, b: int }
struct Outer { inner: Inner, k: int }
fn main() -> int {
    let mut o = Outer { inner: Inner { a: 1, b: 2 }, k: 3 };
    let snapshot = o;
    o.inner = Inner { a: 100, b: 200 };
    o.k = 99;
    print(o.inner.a, o.inner.b, o.k);
    print(snapshot.inner.a, snapshot.inner.b, snapshot.k);
    o.inner.a + snapshot.inner.a
}
""", tmp_path)


@needs_clang
def test_native_nested_struct_access_and_update(tmp_path):
    ir = assert_native_matches_interp(_NESTED_STRUCT_SRC, tmp_path)
    assert "%struct.Outer = type { %struct.Inner, i64 }" in ir


@needs_clang
def test_native_mutually_recursive_struct_enum_list(tmp_path):
    # The linked_list.mx shape end-to-end: struct nodes chained through a
    # boxed Option payload, built by calls and summed by recursive matching.
    assert_native_matches_interp(_LINKED_LIST_SHAPE_SRC, tmp_path)


@needs_clang
def test_native_returned_closure(tmp_path):
    # An escaping closure: created in make_adder, called after make_adder's
    # frame is gone — only sound because the env is heap (leaked by design).
    assert_native_matches_interp("""
fn make_adder(x: int) -> fn(int) -> int {
    let f = fn(y: int) -> x + y;
    f
}
fn main() -> int {
    let add2 = make_adder(2);
    let add10 = make_adder(10);
    print(add2(40));
    print(add10(32));
    add2(40)
}
""", tmp_path)


@needs_clang
@needs_asan
def test_native_boxed_list_no_uaf_under_asan(tmp_path):
    # FREE-STRATEGY PROOF, scoped to what the contract claims: with leak
    # detection off (boxes leak BY DESIGN), ASan exit 0 proves the box
    # traffic has no use-after-free and no double-free.
    ir = assert_native_matches_interp_asan_boxes("""
enum IntList { Cons(int, IntList), Nil }
fn sum(l: IntList) -> int {
    match l { Cons(h, t) -> h + sum(t), Nil -> 0 }
}
fn main() -> int {
    let l = Cons(1, Cons(2, Cons(3, Nil)));
    print(sum(l));
    print(sum(Cons(10, l)));
    0
}
""", tmp_path)
    assert "boxed enum:IntList payload" in ir


@needs_clang
@needs_asan
def test_native_boxes_next_to_freed_global_structs_under_asan(tmp_path):
    # Boxes (leaked) and @global structs (freed at frame exit) coexist: with
    # detect_leaks=0 ASan still catches any double-free or UAF, so exit 0
    # proves the frees only ever hit the @global block, never a box.
    ir = assert_native_matches_interp_asan_boxes("""
struct Pair { a: int, b: int }
enum IntList { Cons(int, IntList), Nil }
fn sum(l: IntList) -> int {
    match l { Cons(h, t) -> h + sum(t), Nil -> 0 }
}
fn main() -> int {
    let @global p = Pair { a: 30, b: 12 };
    let l = Cons(p.a, Cons(p.b, Nil));
    print(sum(l));
    0
}
""", tmp_path)
    # the @global struct still gets its paired free; boxes get none
    assert re.search(r"call void @free\(ptr %hv\.\w+\)", ir)
    assert ir.count("call void @free") == 1


@needs_clang
@needs_asan
def test_native_returned_closure_no_uaf_under_asan(tmp_path):
    # Heap envs leak by design; detect_leaks=0 + exit 0 proves calling the
    # escaped closure never touches freed memory.
    assert_native_matches_interp_asan_boxes("""
fn make_adder(x: int) -> fn(int) -> int {
    let f = fn(y: int) -> x + y;
    f
}
fn main() -> int {
    let add2 = make_adder(2);
    print(add2(40));
    0
}
""", tmp_path)


# ---------------------------------------------------------------------------
# Increment 5: native vec/string runtime (mx_* lowering + linked metaxu_rt)
# ---------------------------------------------------------------------------

_VEC_SUM_SRC = """
fn main() -> int {
    let v = Vec.new();
    let mut i = 0;
    while i < 100 {
        v.push(i);
        i = i + 1;
    }
    let mut sum = 0;
    let mut j = 0;
    let n = v.len();
    while j < n {
        sum = sum + v[j];
        j = j + 1;
    }
    print(sum);
    let mut drained = 0;
    while v.len() > 0 {
        drained = drained + v.pop();
    }
    print(drained);
    0
}
"""


def test_vec_builtins_declare_runtime_symbols():
    # Structural: every used mx_* symbol is declared, the vec is recognized
    # as frame-confined (freed), and main is a real define.
    ir = llvm_from_source(_VEC_SUM_SRC)
    assert "define i64 @mx_main()" in ir
    for decl in ("declare ptr @mx_vec_new()",
                 "declare void @mx_vec_push(ptr, i64)",
                 "declare i64 @mx_vec_pop(ptr)",
                 "declare i64 @mx_vec_len(ptr)",
                 "declare i64 @mx_vec_get(ptr, i64)",
                 "declare void @mx_vec_free(ptr)"):
        assert decl in ir, f"missing runtime declare: {decl}"
    assert "call void @mx_vec_free" in ir
    assert count_placeholders(ir) == 0


def test_string_ops_declare_runtime_symbols():
    ir = llvm_from_source("""
fn main() -> int {
    let s = "a" + to_string(1);
    if s == "a1" { print(s); } else { print("no"); }
    print(len(s));
    0
}
""")
    for decl in ("declare ptr @mx_str_concat(ptr, ptr)",
                 "declare ptr @mx_i64_to_str(i64)",
                 "declare i64 @mx_str_eq(ptr, ptr)",
                 "declare i64 @mx_str_len(ptr)"):
        assert decl in ir, f"missing runtime declare: {decl}"
    assert count_placeholders(ir) == 0


@needs_clang
def test_native_vec_sum_loop(tmp_path):
    # Differential: push 0..99, sum via index reads, drain via pop — the
    # runtime object must link and match the interpreter exactly.
    ir = assert_native_matches_interp(_VEC_SUM_SRC, tmp_path)
    assert "call ptr @mx_vec_new()" in ir


@needs_clang
def test_native_vec_index_access(tmp_path):
    assert_native_matches_interp("""
fn main() -> int {
    let v = Vec.new();
    v.push(10);
    v.push(20);
    v.push(30);
    print(v[0], v[1], v[2]);
    v[2] - v[1] - v[0]
}
""", tmp_path)


@needs_clang
def test_native_string_concat_and_to_string(tmp_path):
    ir = assert_native_matches_interp("""
fn main() -> int {
    let a = "answer: " + to_string(41 + 1);
    print(a);
    print("x" + "y" + "z");
    0
}
""", tmp_path)
    assert "@mx_str_concat" in ir
    assert "@mx_i64_to_str" in ir


@needs_clang
def test_native_string_equality(tmp_path):
    # ==/!= on strings is content equality via mx_str_eq (bool results are
    # printed as ints: the documented bool kind-erasure divergence).
    ir = assert_native_matches_interp("""
fn main() -> int {
    if "abc" == "abc" { print(1); } else { print(0); }
    if "abc" != "abd" { print(3); } else { print(4); }
    if "a" + "b" == "ab" { print(5); } else { print(6); }
    0
}
""", tmp_path)
    assert "@mx_str_eq" in ir


@needs_clang
def test_native_float_math_and_to_string(tmp_path):
    # sqrt/sin/cos are libm externs (-lm, the documented pick) and
    # mx_f64_to_str reproduces Python's str(float) exactly.
    ir = assert_native_matches_interp("""
fn main() -> int {
    print(to_string(sqrt(2.25)));
    print(to_string(sin(0.0)));
    print(to_string(cos(0.0)));
    print("pi-ish: " + to_string(3.14159));
    0
}
""", tmp_path)
    assert "@mx_f64_to_str" in ir
    assert "declare double @sqrt(double)" in ir


@needs_clang
def test_native_vec_float_elements(tmp_path):
    # f64 elements round-trip through the opaque i64 word slots (bitcast).
    assert_native_matches_interp("""
fn main() -> int {
    let v = Vec.new();
    v.push(1.5);
    v.push(2.25);
    print(to_string(v[0] + v.pop()));
    0
}
""", tmp_path)


@needs_clang
def test_native_vec_string_elements(tmp_path):
    # str elements round-trip through the word slots (ptrtoint/inttoptr).
    assert_native_matches_interp("""
fn main() -> int {
    let v = Vec.new();
    v.push("hello");
    v.push("world");
    print(v[0] + " " + v[1]);
    0
}
""", tmp_path)


@needs_clang
def test_native_vec_returned_identity(tmp_path):
    # A returned Vec is the SAME vector (identity semantics: the pointer
    # crosses the frame shallowly) — and therefore is never freed.
    ir = assert_native_matches_interp("""
fn make() -> Vec<Int> {
    let v = Vec.new();
    v.push(1);
    v
}
fn main() -> int {
    let v = make();
    v.push(2);
    print(v.len(), v[0], v[1]);
    0
}
""", tmp_path)
    assert "call void @mx_vec_free" not in ir


@needs_clang
def test_native_vec_struct_enum_mix(tmp_path):
    # Vec + struct + enum interplay: a vec field is a shared pointer inside
    # a byval-copied struct (mutations visible through copies, exactly like
    # the interpreter's MxVec-in-MxStruct), popped values box into enum
    # payloads, and match dispatches on the tag.
    assert_native_matches_interp("""
struct Sensor { readings: Vec<Int>, id: int }
enum Reading { Got(int), Empty }
fn record(s: Sensor, x: int) -> int {
    s.readings.push(x);
    s.readings.len()
}
fn last(s: Sensor) -> Reading {
    if s.readings.len() == 0 { return Empty; }
    Got(s.readings.pop())
}
fn main() -> int {
    let s = Sensor { readings: Vec.new(), id: 7 };
    print(record(s, 10));
    print(record(s, 32));
    match last(s) {
        Got(x) -> print(x),
        Empty -> print(0 - 1),
    }
    0
}
""", tmp_path)


@needs_clang
def test_native_struct_param_write_back(tmp_path):
    # Interpreter write-back parity (mir_interp._write_back_struct_args): a
    # callee that rebinds its @mut struct param (`c.n = ...`) mutates the
    # CALLER's binding — natively a copy-out through the caller's pointer
    # on every ret path.  11, 12, then c.n itself is 12.
    ir = assert_native_matches_interp("""
struct Counter { n: int }
fn bump(c: @mut Counter) -> int {
    c.n = c.n + 1;
    c.n
}
fn main() -> int {
    let c = Counter { n: 10 };
    print(bump(c));
    print(bump(c));
    print(c.n);
    0
}
""", tmp_path)
    assert "copy-out: rebound struct param" in ir


@needs_clang
def test_native_trait_dispatch_static_resolution(tmp_path):
    # examples/10 end-to-end: __trait$ calls on a struct receiver resolve
    # to the impl functions, on the Vec field to mx_vec_*; the whole module
    # emits with zero placeholders and matches the interpreter.
    src = (REPO_ROOT / "examples" / "10_traits_and_structs.mx").read_text()
    _result, expected = interp_run(src, "example")
    ir = llvm_from_source(src)
    assert count_placeholders(ir) == 0
    code, out = compile_and_run(ir, "example", workdir=str(tmp_path))
    assert out == expected
    assert code == 0


@needs_clang
def test_native_example_03_modules(tmp_path):
    # examples/03 end-to-end: float math methods + to_string + concat; the
    # module emits fully (including Python-style float formatting).
    src = (REPO_ROOT / "examples" / "03_modules_and_imports.mx").read_text()
    _result, expected = interp_run(src, "main")
    ir = llvm_from_source(src)
    assert count_placeholders(ir) == 0
    code, out = compile_and_run(ir, "main", workdir=str(tmp_path))
    assert out == expected
    assert code == 0


@needs_clang
@needs_asan
def test_native_vec_freed_fully_leak_checked_under_asan(tmp_path):
    # THE FREE-CONTRACT PROOF, strong form: a vec provably confined to its
    # frame is mx_vec_free'd on ret, so this runs under FULL ASan leak
    # checking (no detect_leaks=0) — exit 0 proves leak-freedom AND
    # no-UAF/no-double-free for the whole vec traffic.
    ir = assert_native_matches_interp_asan(_VEC_SUM_SRC, tmp_path)
    assert "call void @mx_vec_free" in ir


@needs_clang
@needs_asan
def test_native_escaping_vec_no_uaf_under_asan(tmp_path):
    # A vec stored in a struct field escapes its creating frame: it is
    # never freed (leaks BY DESIGN — identity sharing makes any free
    # potentially double), so leak detection is off; exit 0 proves the
    # shared-pointer traffic has no use-after-free and no double-free.
    ir = assert_native_matches_interp_asan_boxes("""
struct Holder { items: Vec<Int>, id: int }
fn fill(h: Holder) -> int {
    h.items.push(4);
    h.items.push(2);
    h.items.len()
}
fn main() -> int {
    let h = Holder { items: Vec.new(), id: 1 };
    print(fill(h));
    print(h.items.pop());
    0
}
""", tmp_path)
    assert "leaks by design (may escape)" in ir
    assert "call void @mx_vec_free" not in ir


# ---------------------------------------------------------------------------
# Increment 7: native algebraic effects
# ---------------------------------------------------------------------------

_FX_ROUNDTRIP = """
effect Ask { ask() -> int }
fn main() -> int {
    handle Ask with { ask() -> resume(7) } in { perform Ask.ask() + 1 }
}
"""


def test_handle_scope_emits_runtime_call_and_shims():
    ir = llvm_from_source(_FX_ROUNDTRIP)
    assert count_placeholders(ir) == 0
    # runtime declares
    assert "declare i64 @mx_handle(ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64)" in ir
    assert "declare i64 @mx_perform(ptr, ptr, ptr, i64)" in ir
    assert "declare i64 @mx_resume(ptr, i64)" in ir
    # per-site artifacts: env type, op-name/arity tables, thunk + dispatcher
    assert re.search(r"%henv\.\w+ = type \{", ir)
    assert re.search(r"@mxfx\.ops\.\w+ = private unnamed_addr constant "
                     r"\[1 x ptr\]", ir)
    assert re.search(r"@mxfx\.np\.\w+ = private unnamed_addr constant "
                     r"\[1 x i64\] \[i64 1\]", ir)
    assert re.search(r"define internal i64 @mxfx\.body\.\w+\(ptr %env\)", ir)
    assert re.search(r"define internal i64 @mxfx\.disp\.\w+"
                     r"\(ptr %env, i64 %op, ptr %args, ptr %k\)", ir)
    # the dispatcher documents its dense op index order
    assert "op index order ['ask']" in ir
    # subfunctions take the leading env param; the case gets __k as a ptr
    assert re.search(r"define i64 @mx___handler_Ask_ask_\w+"
                     r"\(ptr %cl\.env, i64 %a\._, ptr %a\.__k\)", ir)
    assert "call i64 @mx_resume(ptr %a.__k, i64 7)" in ir


def test_resume_outside_its_handler_case_demotes():
    # A resume op whose continuation is not the containing case's own __k
    # param would pump the scope from a foreign stack: demote honestly.
    f = make_func("rogue", [
        block([
            ("params", ("v",)),
            ("let", "r", ("resume",), ("__k", "v")),
        ], ("ret", "r")),
    ])
    ir = emit_llvm([f])
    assert count_placeholders(ir) == 1
    assert "resume outside its own handler case" in ir


_FX_CLOSURE_ARG_SRC = """
effect Apply { app(f: fn(int) -> int) -> int }
fn main() -> int {
    handle Apply with { app(f) -> resume(f(2)) } in {
        let double = fn(x: int) -> int { x * 2 };
        perform Apply.app(double)
    }
}
"""


def test_closure_crossing_effect_boundary_boxes():
    # Increment 14: performing with a closure argument crosses as a boxed
    # {fn, env} pair word.  The member lambda is forced heap-env (the
    # boxed pair may outlive the creating frame; an immortal env can
    # never dangle), the sender boxes the 16-byte pair, and the
    # dispatcher hands the case fn the box pointer to byval-copy from.
    ir = llvm_from_source(_FX_CLOSURE_ARG_SRC)
    assert count_placeholders(ir) == 0
    assert "cannot cross the effect boundary" not in ir
    assert re.search(
        r"call ptr @malloc\(i64 16\)"
        r"  ; boundary box: closure:\S+ \(write-once, leaks by design\)", ir)
    assert re.search(r"heap env for \w+ -> \S+ \(leaks by design\)", ir)
    assert re.search(
        r"inttoptr i64 %c0\.a0w to ptr"
        r"  ; boundary box: case param closure:", ir)


@needs_clang
def test_native_effect_resume_value_becomes_perform_value(tmp_path):
    assert_native_matches_interp(_FX_ROUNDTRIP, tmp_path)


@needs_clang
def test_native_effect_deep_handler_across_calls_42(tmp_path):
    assert_native_matches_interp("""
effect Ask { ask() -> int }
fn helper() performs Ask -> int { let x = perform Ask.ask(); x * 10 }
fn main() -> int {
    handle Ask with { ask() -> resume(4) } in { helper() + 2 }
}
""", tmp_path)


@needs_clang
def test_native_effect_multiple_performs(tmp_path):
    assert_native_matches_interp("""
effect Ask { ask() -> int }
fn main() -> int {
    handle Ask with { ask() -> resume(5) } in {
        let a = perform Ask.ask();
        let b = perform Ask.ask();
        a + b
    }
}
""", tmp_path)


@needs_clang
def test_native_effect_abort_skips_rest_99(tmp_path):
    # The handler declines to resume: 99 is the handle value and the print
    # after the perform never runs (native stdout must equal interpreter's).
    assert_native_matches_interp("""
effect Fail { fail() -> int }
fn main() -> int {
    handle Fail with { fail() -> 99 } in {
        let x = perform Fail.fail();
        print("unreachable");
        x + 1
    }
}
""", tmp_path)


@needs_clang
def test_native_effect_abort_inside_called_function(tmp_path):
    assert_native_matches_interp("""
effect Fail { fail() -> int }
fn helper() performs Fail -> int {
    let x = perform Fail.fail();
    print("unreachable-helper");
    x
}
fn main() -> int {
    handle Fail with { fail() -> 7 } in {
        let y = helper();
        print("unreachable-main");
        y
    }
}
""", tmp_path)


@needs_clang
def test_native_effect_post_resume_handler_code_103(tmp_path):
    assert_native_matches_interp("""
effect Ask { ask() -> int }
fn main() -> int {
    handle Ask with {
        ask() -> { let rest = resume(1); rest + 100 }
    } in { perform Ask.ask() + 2 }
}
""", tmp_path)


@needs_clang
def test_native_effect_resume_returns_whole_body_300(tmp_path):
    assert_native_matches_interp("""
effect Ask { ask() -> int }
fn helper() performs Ask -> int { perform Ask.ask() }
fn main() -> int {
    handle Ask with {
        ask() -> { let rest = resume(1); rest * 100 }
    } in { helper() + 2 }
}
""", tmp_path)


@needs_clang
def test_native_effect_nested_two_effects_42(tmp_path):
    assert_native_matches_interp("""
effect State { get() -> int }
effect Logger { log(message: string) -> Unit }
fn body() performs State, Logger -> int {
    let v = perform State.get();
    perform Logger.log("got it");
    v + 1
}
fn main() -> int {
    handle State with { get() -> resume(41) } in {
        handle Logger with {
            log(message) -> { print(message); resume(()) }
        } in { body() }
    }
}
""", tmp_path)


@needs_clang
def test_native_effect_nested_post_resume_224(tmp_path):
    assert_native_matches_interp("""
effect State { get() -> int }
effect Logger { log(message: string) -> Unit }
fn body() performs State, Logger -> int {
    let v = perform State.get();
    perform Logger.log("hi");
    v + 1
}
fn main() -> int {
    handle State with {
        get() -> { let rest = resume(10); rest * 2 }
    } in {
        handle Logger with {
            log(message) -> { let r = resume(()); r + 1 }
        } in { body() + 100 }
    }
}
""", tmp_path)


@needs_clang
def test_native_effect_handler_self_perform_routes_outward(tmp_path):
    assert_native_matches_interp("""
effect Ask { ask() -> int }
fn main() -> int {
    handle Ask with { ask() -> resume(100) } in {
        handle Ask with {
            ask() -> { let outer = perform Ask.ask(); resume(outer + 1) }
        } in { perform Ask.ask() }
    }
}
""", tmp_path)


@needs_clang
def test_native_effect_rearming_reaches_inner_handler(tmp_path):
    assert_native_matches_interp("""
effect Ask { ask() -> int }
fn main() -> int {
    handle Ask with { ask() -> resume(1000) } in {
        handle Ask with { ask() -> resume(1) } in {
            perform Ask.ask() + perform Ask.ask()
        }
    }
}
""", tmp_path)


@needs_clang
def test_native_effect_multi_argument_op(tmp_path):
    assert_native_matches_interp("""
effect Math { add(a: int, b: int) -> int }
fn main() -> int {
    handle Math with { add(a, b) -> resume(a + b) } in {
        perform Math.add(40, 2)
    }
}
""", tmp_path)


@needs_clang
def test_native_effect_string_crosses_boundary(tmp_path):
    # op-name kind cells: the resume value is a string, so the perform's
    # result decodes as a str word; concat + print must match.
    assert_native_matches_interp("""
effect Ask { name() -> string }
fn main() -> int {
    handle Ask with { name() -> resume("world") } in {
        let s = perform Ask.name();
        print("hello " + s);
        0
    }
}
""", tmp_path)


@needs_clang
def test_native_effect_float_crosses_boundary(tmp_path):
    # f64 boundary words bitcast losslessly (compared via branches, not
    # printed: %g formatting diverges from the interpreter).
    assert_native_matches_interp("""
effect M { pi() -> float }
fn main() -> int {
    handle M with { pi() -> resume(3.5) } in {
        let x = perform M.pi() + 0.25;
        if x == 3.75 { 1 } else { 0 }
    }
}
""", tmp_path)


@needs_clang
def test_native_effect_handle_inside_loop(tmp_path):
    # The site env alloca is refilled per iteration; each mx_handle
    # completes within its iteration, so a stack env stays safe.
    assert_native_matches_interp("""
effect Ask { ask() -> int }
fn main() -> int {
    let mut total = 0;
    let mut i = 0;
    while i < 3 {
        let bonus = i * 10;
        let r = handle Ask with { ask() -> resume(bonus) } in {
            perform Ask.ask() + 1
        };
        total = total + r;
        i = i + 1;
    };
    print(total);
    total
}
""", tmp_path)


@needs_clang
def test_native_effect_captures_through_shared_env(tmp_path):
    # Body AND case read enclosing locals through the site's shared env.
    assert_native_matches_interp("""
effect Ask { ask() -> int }
fn main() -> int {
    let base = 30;
    let inc = 4;
    handle Ask with { ask() -> resume(base + inc) } in {
        perform Ask.ask() + base
    }
}
""", tmp_path)


@needs_clang
def test_native_effect_vec_identity_across_boundary(tmp_path):
    # A Vec captured into the handler case keeps identity semantics: the
    # case pushes into the same vector main reads afterwards, while the
    # performs come from a called function.
    assert_native_matches_interp("""
effect Sink { emit(x: int) -> Unit }
fn pump(n: int) performs Sink {
    let mut i = 0;
    while i < n {
        perform Sink.emit(i * i);
        i = i + 1;
    }
}
fn main() -> int {
    let v = Vec.new();
    handle Sink with { emit(x) -> { v.push(x); resume(()) } } in {
        pump(4)
    };
    print(v.len());
    print(v[3]);
    0
}
""", tmp_path)


@needs_clang
def test_native_example_02_effects_and_handlers(tmp_path):
    # The flagship effects example runs natively end-to-end: nested
    # State/Logger handles, performs in a called function, to_string +
    # string concat crossing the boundary.  stdout must equal the
    # interpreter's exactly.
    src = (REPO_ROOT / "examples" / "02_effects_and_handlers.mx").read_text()
    ir = assert_native_matches_interp(src, tmp_path)
    # only the generic list-pattern `map` helper demotes (unbound generic
    # env — unrelated to effects); everything effectful is native
    assert count_placeholders(ir) == 1
    assert "; function @mx_map: placeholder" in ir


@needs_clang
def test_native_example_effects_mx(tmp_path):
    # effects.mx is fully native: zero placeholders.
    src = (REPO_ROOT / "examples" / "effects.mx").read_text()
    ir = assert_native_matches_interp(src, tmp_path)
    assert count_placeholders(ir) == 0


@needs_asan
def test_native_effects_fully_leak_checked_under_asan(tmp_path):
    # Pure-int effects: the machinery itself (coroutine stacks, scope
    # records, continuations) must be LEAK-CLEAN — full leak checking on,
    # exercising nested scopes, post-resume code and an abort.
    assert_native_matches_interp_asan("""
effect State { get() -> int }
effect Fail { fail() -> int }
fn body() performs State, Fail -> int {
    let v = perform State.get();
    let w = perform Fail.fail();
    print(v + w);
    0
}
fn main() -> int {
    let a = handle State with {
        get() -> { let rest = resume(20); rest }
    } in {
        handle Fail with { fail() -> 5 } in { body() }
    };
    print(a);
    0
}
""", tmp_path)


@needs_asan
def test_native_effects_with_string_concat_asan_no_uaf(tmp_path):
    # Example-02-shaped traffic under ASan: concat/to_string results leak
    # by design, so detect_leaks=0 — this proves no UAF / no double-free
    # across coroutine switches (fiber annotations active).
    src = (REPO_ROOT / "examples" / "02_effects_and_handlers.mx").read_text()
    result, expected_out = interp_run(src)
    ir = llvm_from_source(src)
    exit_code, stdout = compile_and_run(
        ir, "main", workdir=str(tmp_path),
        clang_args=("-fsanitize=address",),
        run_env={"ASAN_OPTIONS": "detect_leaks=0"})
    assert exit_code == 0
    assert stdout == expected_out


# ---------------------------------------------------------------------------
# Increment 8: memory reclamation (owned strings, unique boxes) + copy elision
# ---------------------------------------------------------------------------

_STR_LOOP_SRC = """
fn main() -> int {
    let mut s = "";
    let mut i = 0;
    while i < 50 {
        s = s + "ab";
        i = i + 1;
    }
    print(len(s));
    0
}
"""


def test_owned_string_loop_frees_previous_value_each_iteration():
    # The loop accumulator `s` is an OWNED string: each redefinition frees
    # the previous concat result through the shadow slot, the initial ""
    # literal records null (never freed), and frame exit frees the last.
    ir = llvm_from_source(_STR_LOOP_SRC)
    assert count_placeholders(ir) == 0
    assert "declare void @mx_str_free(ptr)" in ir
    assert re.search(r"%strown\.\w+ = alloca ptr  ; owned string shadow", ir)
    assert "previous value freed" in ir
    assert re.search(r"store ptr null, ptr %strown\.\w+"
                     r"  ; owned string \w+: literal \(never freed\)", ir)
    assert "freed at frame exit" in ir
    # the concat itself is documented as owned, not leaked
    assert re.search(r"@mx_str_concat\(.*\)  ; owned \(freed when dead\)", ir)


@needs_clang
def test_native_owned_string_loop(tmp_path):
    assert_native_matches_interp(_STR_LOOP_SRC, tmp_path)


@needs_clang
@needs_asan
def test_native_owned_string_loop_fully_leak_checked_under_asan(tmp_path):
    # THE INCREMENT-8 STRING PROOF: 50 concats, every fresh malloc freed —
    # FULL leak checking (no detect_leaks=0), so exit 0 proves the loop no
    # longer grows memory AND that no free ever hit a live or literal
    # string (no UAF / double-free / bad-free of rodata).
    ir = assert_native_matches_interp_asan(_STR_LOOP_SRC, tmp_path)
    assert "call void @mx_str_free" in ir


@needs_clang
@needs_asan
def test_native_owned_string_chain_fully_leak_checked_under_asan(tmp_path):
    # Straight-line concat/to_string chains: producer temps feeding concat
    # operands and print/len consumers are all owned and freed at frame
    # exit.  (This is the shape test_native_string_concat_and_to_string
    # runs without ASan; it now sustains FULL leak checking.)
    assert_native_matches_interp_asan("""
fn main() -> int {
    let a = "answer: " + to_string(41 + 1);
    print(a);
    print("x" + "y" + "z");
    print(len("a" + "bc"));
    0
}
""", tmp_path)


@needs_clang
@needs_asan
def test_native_owned_string_mixed_literal_produced_paths_asan(tmp_path):
    # Provenance across branches: some defs re-produce (concat), some reset
    # to a literal.  The shadow must free exactly the produced values and
    # never the interned literals, on every interleaving — full leak check.
    assert_native_matches_interp_asan("""
fn main() -> int {
    let mut s = "start";
    let mut even = 1;
    let mut i = 0;
    while i < 10 {
        if even == 1 { s = s + "e"; even = 0; } else { s = "odd"; even = 1; }
        i = i + 1;
    }
    print(s);
    print(len(s));
    let mut d = "x";
    let mut j = 0;
    while j < 4 { d = d + d; j = j + 1; }
    print(len(d));
    0
}
""", tmp_path)


def test_returned_concat_string_still_leaks_by_design():
    # A concat result that RETURNS has no ownership proof: no shadow, no
    # free — the leak-by-design contract is unchanged for escaping strings.
    ir = llvm_from_source("""
fn shout(s: str) -> str { s + "!" }
fn main() -> int { print(shout("hi")); 0 }
""")
    assert count_placeholders(ir) == 0
    assert "mx_str_free" not in ir
    assert re.search(r"@mx_str_concat\(.*\)  ; leaks by design", ir)


_UNIQUE_BOX_SRC = """
struct Pair { a: int, b: int }
enum Wrap { W(Pair), E }
fn main() -> int {
    let w = W(Pair { a: 30, b: 12 });
    match w { W(p) -> print(p.a + p.b), E -> print(0) }
    0
}
"""


def test_unique_box_freed_at_frame_exit_and_boxview_elides_extraction():
    # `w` is only ever matched in its own frame: its payload box is uniquely
    # owned and freed on the ret path.  The extracted `p` is only read, so
    # it becomes a BOX VIEW (GEP reads through the box pointer) instead of
    # an aggregate copied out.
    ir = llvm_from_source(_UNIQUE_BOX_SRC)
    assert count_placeholders(ir) == 0
    assert re.search(
        r"call ptr @malloc\(i64 16\)  ; boxed struct:Pair payload "
        r"\(unique: freed at frame exit\)", ir)
    assert re.search(r"call void @free\(ptr %t\d+\)  ; unique box:", ir)
    assert "; elide-copy: variant_field" in ir
    # the view really skips the copy: no %struct.Pair load feeds a %sv slot
    assert not re.search(r"store %struct\.Pair %t\d+, ptr %sv\.", ir)


@needs_clang
@needs_asan
def test_native_unique_box_fully_leak_checked_under_asan(tmp_path):
    # THE INCREMENT-8 BOX PROOF: full leak checking — exit 0 proves the
    # unique box is freed exactly once, after its last (box-view) read.
    ir = assert_native_matches_interp_asan(_UNIQUE_BOX_SRC, tmp_path)
    assert "; unique box:" in ir


def test_shared_box_still_leaks_by_design():
    # The list is PASSED to sum(): a callee could retain its copy (e.g. in
    # a heap closure env), so uniqueness is unprovable and every box stays
    # leaked — the honest boundary of the conservative class.
    ir = llvm_from_source(_CONS_SUM_SRC)
    assert count_placeholders(ir) == 0
    assert "unique: freed at frame exit" not in ir
    assert "call void @free" not in ir


def test_examples_elision_census_does_not_regress():
    # Elided copies across the accepted examples, pinned via the
    # `; elide-copy:` IR markers: 11 param byval copies + 11 variant_field
    # extraction copies (linked_list.mx dominates) at increment 8.
    total = 0
    for path in _example_files():
        ir = llvm_from_source(path.read_text())
        total += len(re.findall(r"; elide-copy:", ir))
    assert total >= 22


def test_ffi_example_emits_fully_native():
    # 05_unsafe_and_ffi.mx (increment 9): every function — the four impl
    # methods over extern C calls AND main (vector literal + __static$
    # dispatch + trait calls) — emits as a real define.  Zero placeholders,
    # real C declares with C signatures, rawptr null compares, and the
    # vec byte-snapshot accessor.
    ir = llvm_from_source(
        (REPO_ROOT / "examples" / "05_unsafe_and_ffi.mx").read_text())
    assert count_placeholders(ir) == 0
    assert "define void @mx___impl__Buffer_new(" in ir
    assert "define i64 @mx_main()" in ir
    assert "declare noalias ptr @malloc(i64)" in ir
    assert "declare void @free(ptr)" in ir
    assert "declare ptr @memcpy(ptr, ptr, i64)" in ir
    assert "declare noalias ptr @fopen(ptr, ptr)" in ir
    assert "declare i32 @fclose(ptr)" in ir       # C int, not i64
    assert re.search(r"sext i32 %t\d+ to i64", ir)  # fclose result widened
    # null pointer literal + pointer identity comparison
    assert re.search(r"icmp (eq|ne) ptr %t\d+, null", ir)
    assert "call ptr @mx_fvec_as_bytes(ptr" in ir  # vector.as_ptr() snapshot
    assert "; vector literal" in ir  # __vec_lit -> immutable mx_fvec block
    assert ir.count("call void @mx_fvec_init") == 5
    # __static$ calls resolved at compile time: no call instruction or
    # demotion reason references a __static symbol (the module header
    # comment legitimately documents the mechanism).
    assert not re.search(r"call .*__static", ir)
    assert not re.search(r"reason:.*__static", ir)


def test_examples_define_census_does_not_regress():
    # Aggregate emission census across all accepted examples: the number of
    # real defines must not regress below the increment-6 level (increment 3
    # emitted 18; increment 4's boxing/inlining reached 30; the native
    # vec/string runtime + static trait dispatch lifted 01/03/06/10/
    # collections to 45 — then the front-end if-let/early-return fixes gave
    # linked_list.mx its REAL pop_front/remove_next/get/get_mut bodies
    # (heterogeneous Option slots: 4 honest demotions + main) and
    # test_operations.mx a real `assert` call (1 more), landing at 39;
    # increment 6's per-variant/per-value payload typing un-demoted those
    # four linked_list bodies, landing at 43; the front-end statement-rule
    # fix (else-less if is unit) gave linked_list its main (44); increment
    # 7's native effects lifted every suspending function that stays in
    # word kinds — all of 02_effects_and_handlers (except the generic
    # `map` helper) and effects.mx, plus effectful helpers elsewhere —
    # landing at 61.  Increment 8 (reclamation + copy elision) changes
    # memory behavior only, never coverage: still 61.  The FFI front-end
    # fix REMOVED 4 fake defines: 05_unsafe_and_ffi.mx's Buffer.new/
    # Buffer.free/File.open/File.close previously lowered to empty
    # `ret unit` bodies because the HIR builder silently dropped unsafe
    # blocks; with their real bodies restored they demote honestly
    # (extern C callees malloc/free/fopen/fclose/as_ptr have no native
    # lowering yet — the placeholders name the exact unlinked symbol),
    # landing at 57.  Increment 9 (native FFI) won those back and more:
    # the rawptr kind + real extern C calls + as_ptr/ptr_read/ptr_write
    # lift all five 05_unsafe_and_ffi.mx impl fns, __vec_lit + __static$
    # resolution lift its main plus ownership.mx's main, and the native
    # assert lifts test_locality_heap.mx / test_operations.mx mains —
    # landing at 69 (04_advanced_types' main stays demoted: its next
    # blocker behind __vec_lit is the try/catch-shaped Result flow).
    # Increment 10 (native fixed-vector runtime) lifts 06's transpose,
    # transpose$lambda8 and static_assert; its two __effect_default fns —
    # previously dead trivial defines — now demote honestly (they are
    # actually CALLED via default-resolved performs, and their `f`
    # parameter joins conflicting closure kinds), landing at 70.  06's
    # remaining demotions are all higher-order: map/reduce/zip receive
    # DIFFERENT lambdas at one call site (indirect closure calls are not
    # a feature yet), never vector builtins.  (The parser-sharing round
    # later landed at a 77 baseline.)  Increment 12 (silent-seam
    # constructs) lifts ownership.mx entirely (process + main: the fvec
    # functional index update through a struct-field place), landing at
    # 79; collections.mx's push now demotes on its List<T> field-table
    # gap instead of __index_store, and 06's zip demotions are all
    # closure-kind conflicts, never __zip itself.  Increment 13 (indirect
    # closure calls) lifts 06's higher-order shapes whose lambdas agree on
    # kinds — reduce/sum/product/dot/norm-style drivers over same-kind
    # lambdas emit through the word-uniform ABI — landing at 88; 06's map
    # keeps demoting honestly on GENUINE polymorphism (one `map` receives
    # f64-valued AND str-valued lambdas, so its parameter kinds conflict
    # for real).  Increment 14 (boundary boxes) REMOVED one fake define:
    # effect_mapping.mx's main$lambda3 previously emitted with raw
    # mx_perform calls for the runtime-mapped Mutex.lock/unlock ops —
    # native code that would ABORT where the interpreter's EFFECT_*
    # primitives succeed (unobservable only because the demoted main
    # never called it).  Runtime-mapped performs now demote explicitly
    # (never wrong code), landing at 87; no example gains defines from
    # the boxing itself (their remaining demotions are other gaps —
    # std.stream's find, the increment's real win, is census'd in
    # test_std_stream_full_surface_census).
    #
    # The HIR-triage round REMOVED 6 defines that only existed because the
    # front end had silently deleted a handler — the same shape as the FFI
    # round above, which removed 4 defines once unsafe blocks stopped being
    # dropped.  examples/06_vector_operations.mx's `with_simd` is written with
    # the inline `handle SUBJECT { perform Op(p) => body }` form, and that node
    # had NO HIR lowering: the whole handler vanished, `f` was never called,
    # and SimdOp's try_vectorize/try_horizontal could only ever be answered by
    # their declared defaults.  Native codegen lowers that static case, so
    # reduce/fold/sum/mean/dot/norm/normalize/matmul emitted.  With the handler
    # real, those ops have a declared default AND a handle scope, so answering
    # a perform is a runtime choice — "dynamic default routing has no native
    # lowering" — and every function reached through them demotes explicitly
    # rather than emitting code for the wrong route.  Net 87 -> 81 (06 loses
    # 11 and gains the two handler-arm functions plus three main lambdas that
    # are now reachable).
    #
    # Increment 15 lowers that runtime choice instead of refusing it: a
    # perform of an op with BOTH a declared default and a handle scope emits
    # mx_perform_or_default, which runs the runtime's own innermost-non-busy
    # scope lookup and calls the op's default thunk only where plain
    # mx_perform would have aborted — the interpreter's precedence, made
    # native.  06_vector_operations.mx wins back reduce/sum/mean/dot/norm/
    # normalize/matmul (+ matmul's two lambdas, sum/prod's fold lambdas and
    # three main lambdas): 26 -> 33 defines, landing the census at 88.  06's
    # residue is now genuine polymorphism (one `map` takes f64-valued AND
    # str-valued lambdas) and fold's `type_of` / free type variables — never
    # effect routing.
    #
    # Increment 16 (aggregates in indirect closure-call signatures) leaves
    # this census UNCHANGED at 88, and that is the honest result: not one
    # of the 19 examples passes a struct or an enum through an indirect
    # closure call.  06's remaining higher-order demotions are the genuine
    # polymorphism above (conflicting f64/str kinds), which no ABI change
    # can fix.  The increment's coverage win is census'd where it actually
    # lands — test_std_stream_aggregate_element_census, where std.stream's
    # map/filter over a struct/enum element type go from demoted to
    # emitted-and-natively-differential.
    #
    # Increment 19 (native try/catch over mx_try landing pads) removes the
    # last wholesale demotion reason, "uses try/catch (try_scope)".  Only
    # ONE example writes a try: 04_advanced_types.mx, whose `try_parse`,
    # its try body and its catch body all emit now — 1 -> 3 defines, and
    # the census 89 -> 91.  04's `main` still demotes on __list_lit and its
    # ToString impl on the heterogeneous Result receiver, both unrelated
    # gaps, so 04 gains defines without becoming natively runnable.
    total_defines = 0
    for path in _example_files():
        ir = llvm_from_source(path.read_text())
        total_defines += len(re.findall(
            r"^define (?:i64|double|ptr|void) @mx_\w+\(", ir, re.M))
    assert total_defines >= 91


# ---------------------------------------------------------------------------
# Increment 9: native FFI — rawptr kind, extern C calls, as_ptr/ptr_read/
# ptr_write, __vec_lit, __static$ resolution, assert
# ---------------------------------------------------------------------------

_FFI_ROUNDTRIP_SRC = """
extern "C" {
    fn malloc(size: uint) -> *void;
    fn free(ptr: *void);
}

fn main() -> int {
    let p = malloc(8);
    ptr_write(p, 0, 65);
    ptr_write(p, 1, 66);
    let a = ptr_read(p, 0);
    let b = ptr_read(p, 1);
    free(p);
    print(a + b);
    0
}
"""

_MEMCPY_STR_SRC = """
extern "C" {
    fn malloc(size: uint) -> *void;
    fn free(ptr: *void);
    fn memcpy(dest: *void, src: *void, n: uint) -> *void;
}

fn main() -> int {
    let d = malloc(2);
    memcpy(d, "AB".as_ptr(), 2);
    let x = ptr_read(d, 0) + ptr_read(d, 1);
    free(d);
    print(x);
    0
}
"""

_VEC_SNAPSHOT_SRC = """
extern "C" {
    fn malloc(size: uint) -> *void;
    fn free(ptr: *void);
    fn memcpy(dest: *void, src: *void, n: uint) -> *void;
}

fn main() -> int {
    let data = vector[int,3](65, 66, 67);
    let d = malloc(3);
    memcpy(d, data.as_ptr(), 3);
    let x = ptr_read(d, 2);
    free(d);
    print(x);
    0
}
"""


def test_extern_malloc_free_emit_real_c_calls():
    ir = llvm_from_source(_FFI_ROUNDTRIP_SRC)
    assert count_placeholders(ir) == 0
    assert "declare noalias ptr @malloc(i64)" in ir
    assert "declare void @free(ptr)" in ir
    assert re.search(r"%t\d+ = call ptr @malloc\(i64 8\)  ; extern C", ir)
    assert re.search(r"call void @free\(ptr %\w+\)  ; extern C free", ir)
    # ptr_read / ptr_write inline as byte loads/stores through i8 GEPs
    assert re.search(r"getelementptr inbounds i8, ptr %\w+, i64", ir)
    assert re.search(r"%t\d+ = trunc i64 \d+ to i8", ir)
    assert re.search(r"%t\d+ = zext i8 %t\d+ to i64", ir)


def test_null_literal_and_pointer_identity_compare():
    ir = llvm_from_source("""
extern "C" {
    fn malloc(size: uint) -> *void;
    fn free(ptr: *void);
}

fn main() -> int {
    let p = malloc(4);
    let ok = p != null;
    free(p);
    if ok { 1 } else { 0 }
}
""")
    assert count_placeholders(ir) == 0
    # null lowers to the ptr null constant; ==/!= is pointer icmp
    assert re.search(r"icmp ne ptr %t\d+, null", ir)


def test_vector_literal_lowers_to_native_vec():
    # Increment 10: a fixed-vector literal is an immutable mx_fvec block,
    # filled in place before the pointer is ever shared (write-once).
    ir = llvm_from_source(
        "fn main() -> int { let v = vector[int,3](7, 8, 9); v[0] + v[2] }")
    assert count_placeholders(ir) == 0
    assert "call ptr @mx_fvec_new(i64 3)  ; vector literal" in ir
    assert ir.count("call void @mx_fvec_init") == 3
    assert "call i64 @mx_fvec_get" in ir
    assert "call void @mx_vec_push" not in ir  # no growable-Vec stand-in


def test_vector_literal_leaks_by_design_never_freed():
    # Fixed-vector blocks are shallow-shared (immutability makes that
    # sound) so ownership is never unique: no block is ever freed.
    ir = llvm_from_source(
        "fn main() -> int { let v = vector[int,2](4, 5); v[0] + v[1] }")
    assert "immutable block, leaks by design" in ir
    assert "call void @mx_vec_free" not in ir
    assert "mx_fvec_free" not in ir  # no such symbol exists


def test_static_method_call_resolves_at_compile_time():
    ir = llvm_from_source("""
struct Point { x: int, y: int }

implement Point {
    fn origin() -> Point { Point { x: 0, y: 0 } }
}

fn main() -> int {
    let p = Point.origin();
    p.x
}
""")
    assert count_placeholders(ir) == 0
    # resolved to a direct call of the impl fn; no call instruction or
    # demotion reason references a __static symbol
    assert not re.search(r"call .*__static", ir)
    assert not re.search(r"reason:.*__static", ir)
    assert re.search(r"call void @mx___impl__Point_origin\(ptr", ir)


def test_static_call_without_impl_demotes_honestly():
    # A __static$ call whose (type, method) pair has no impl and no dotted
    # fallback demotes with the interpreter's failure named (hand-built
    # MIR: the front end only produces __static$ for real dotted calls).
    f = make_func("caller", [block([
        ("params", ()),
        ("let", "r", ("call", "__static$Widget$origin"), ()),
    ], ("ret", "r"))])
    ir = emit_llvm([f])
    assert count_placeholders(ir) == 1
    assert ("static method 'origin' on type 'Widget' has no impl and no "
            "native dotted fallback") in ir


def test_assert_lowers_to_branch_abort():
    ir = llvm_from_source("fn main() -> int { assert(1 < 2); 7 }")
    assert count_placeholders(ir) == 0
    assert re.search(r"br i1 %t\d+, label %assert\.ok\.\w+, "
                     r"label %assert\.fail\.\w+", ir)
    assert "call void @abort()  ; assert failed" in ir


def test_as_ptr_identity_on_string_no_snapshot():
    # Native strings already ARE NUL-terminated byte pointers, so
    # str.as_ptr() emits no call at all — identity, marked by a comment.
    ir = llvm_from_source("""
fn main() -> int {
    let p = "abc".as_ptr();
    if p != null { 1 } else { 0 }
}
""")
    assert count_placeholders(ir) == 0
    assert "; as_ptr: identity on a native" in ir
    assert "call ptr @mx_vec_as_bytes" not in ir  # no snapshot for strings


def test_rawptr_demotions_stay_honest():
    # print of a raw pointer: the interpreter renders an MxPtr repr no
    # native code can reproduce — demote, never guess.
    ir = llvm_from_source("""
extern "C" {
    fn malloc(size: uint) -> *void;
}

fn main() -> int {
    let p = malloc(4);
    print(p);
    0
}
""")
    assert "print of unsupported kind rawptr" in ir
    # pointer ordering: only ==/!= lower (identity); < is not comparable
    ir2 = llvm_from_source("""
extern "C" {
    fn malloc(size: uint) -> *void;
}

fn main() -> int {
    let a = malloc(4);
    let b = malloc(4);
    if a < b { 1 } else { 0 }
}
""")
    assert "pointer ordering comparison '<'" in ir2
    # rawptr Vec elements: not a word kind (a stored pointer would escape
    # every ownership analysis) — demote.
    ir3 = llvm_from_source("""
extern "C" {
    fn malloc(size: uint) -> *void;
}

fn main() -> int {
    let v = Vec.new();
    v.push(malloc(4));
    0
}
""")
    assert "Vec of rawptr elements" in ir3


@needs_asan
def test_native_malloc_ptr_write_read_free_roundtrip_asan_full(tmp_path):
    # REAL malloc round trip: write bytes through ptr_write, read them
    # back, free — under FULL ASan leak checking (nothing here leaks:
    # as_ptr is not involved and the buffer is freed by the program).
    ir = assert_native_matches_interp_asan(_FFI_ROUNDTRIP_SRC, tmp_path)
    assert re.search(r"call ptr @malloc\(i64 8\)  ; extern C", ir)


@needs_asan
def test_native_memcpy_from_string_asan_full(tmp_path):
    # memcpy out of a string's as_ptr (identity — zero allocation) into a
    # real malloc'd buffer; fully leak-checked (the program frees its own
    # allocation and no snapshot exists).
    assert_native_matches_interp_asan(_MEMCPY_STR_SRC, tmp_path)


@needs_asan
def test_native_vec_as_ptr_snapshot_asan_no_uaf(tmp_path):
    # vec.as_ptr() is a fresh byte snapshot (mx_vec_as_bytes) that LEAKS
    # BY DESIGN, so detect_leaks=0: this proves no UAF/double-free — in
    # particular that freeing the program's own buffer and the vec's
    # frame-exit mx_vec_free never touch the snapshot.
    result, expected_out = interp_run(_VEC_SNAPSHOT_SRC)
    assert result in (UNIT, 0)
    ir = llvm_from_source(_VEC_SNAPSHOT_SRC)
    assert "call ptr @mx_fvec_as_bytes(ptr" in ir
    exit_code, stdout = compile_and_run(
        ir, "main", workdir=str(tmp_path),
        clang_args=("-fsanitize=address",),
        run_env={"ASAN_OPTIONS": "detect_leaks=0"})
    assert exit_code == 0
    assert stdout == expected_out


@needs_asan
def test_native_vector_literal_differential_asan(tmp_path):
    # Fixed-vector blocks leak BY DESIGN (immutable, shallow-shared), so
    # ASan proves no UAF/double-free only: detect_leaks=0 (the sharing
    # contract, same as boxes/heap envs).
    src = """
fn main() -> int {
    let v = vector[int,4](3, 5, 7, 11);
    print(v[0] + v[3]);
    print(v.len());
    0
}
"""
    result, expected_out = interp_run(src)
    assert result in (UNIT, 0)
    ir = llvm_from_source(src)
    exit_code, stdout = compile_and_run(
        ir, "main", workdir=str(tmp_path),
        clang_args=("-fsanitize=address",),
        run_env={"ASAN_OPTIONS": "detect_leaks=0"})
    assert exit_code == 0
    assert stdout == expected_out


@needs_asan
def test_native_assert_passing_differential(tmp_path):
    assert_native_matches_interp_asan("""
fn main() -> int {
    let x = 6 * 7;
    assert(x == 42);
    print(x);
    0
}
""", tmp_path)


def _interp_run_in_cwd(source: str, cwd: str):
    """interp_run with the interpreter's fopen paths pinned to cwd."""
    import os
    old = os.getcwd()
    os.chdir(cwd)
    try:
        return interp_run(source)
    finally:
        os.chdir(old)


@needs_asan
@pytest.mark.parametrize("have_file", [False, True],
                         ids=["fopen-fails", "fopen-succeeds"])
def test_native_ffi_example_05_differential(tmp_path, have_file):
    # 05_unsafe_and_ffi.mx end-to-end with the working directory pinned:
    # without test.txt fopen returns null and the Err arm prints; with it
    # the Ok arm boxes a File struct and fcloses the real stream.  The Ok
    # box and the vec byte snapshot leak by design -> detect_leaks=0
    # (proves no UAF / no double-free around real malloc/free/fopen).
    src = (REPO_ROOT / "examples" / "05_unsafe_and_ffi.mx").read_text()
    rundir = tmp_path / "cwd"
    rundir.mkdir()
    if have_file:
        (rundir / "test.txt").write_text("hello\n")
    result, expected_out = _interp_run_in_cwd(src, str(rundir))
    assert result in (UNIT, 0)
    ir = llvm_from_source(src)
    exit_code, stdout = compile_and_run(
        ir, "main", workdir=str(tmp_path),
        clang_args=("-fsanitize=address",),
        run_env={"ASAN_OPTIONS": "detect_leaks=0"},
        run_cwd=str(rundir))
    assert exit_code == 0
    assert stdout == expected_out


@needs_clang
def test_native_static_dispatch_differential(tmp_path):
    assert_native_matches_interp("""
struct Counter { n: int }

implement Counter {
    fn make(start: int) -> Counter { Counter { n: start } }
}

fn main() -> int {
    let c = Counter.make(41);
    print(c.n + 1);
    c.n + 1
}
""", tmp_path)


# ---------------------------------------------------------------------------
# Increment 10: fixed-vector runtime natively — immutable mx_fvec blocks,
# element-wise arithmetic w/ broadcast, zeros/filled/dim, slices, ranges,
# comprehensions, __cast, promote_matrix, vector printing, effect-op
# defaults as direct calls
# ---------------------------------------------------------------------------

_ELEMENTWISE_SRC = """
fn main() -> int {
    let a = vector[float,4](1.0, 2.0, 3.0, 4.0);
    let b = vector[float,4](10.0, 20.0, 30.0, 40.0);
    print(a);
    print((a + b).to_string());
    print((a * b).to_string());
    print((b - a).to_string());
    print((b / a).to_string());
    print((a * 2.0).to_string());
    print((1.0 + a).to_string());
    let iv = vector[int,3](7, 8, 9);
    print((iv / 2).to_string());
    0
}
"""

_ZEROS_FILLED_DIM_SRC = """
implement<T, const N: int> vector[T,N] {
    fn size(self) -> int { N }
}

fn main() -> int {
    let z = vector[float,4]();
    let o = vector[float,3].filled(2.5);
    print(z.to_string());
    print(o.to_string());
    print(z.size());
    print(o.len());
    0
}
"""

_SLICES_SRC = """
fn main() -> int {
    let v = vector[int,5](10, 20, 30, 40, 50);
    print(v[1:3].to_string());
    print(v[::-1].to_string());
    print(v[::2].to_string());
    print(v.to_string());
    0
}
"""

_COMPREHENSION_SRC = """
fn main() -> int {
    let k = 2.5;
    let v = vector[float,4](x * k for x in 0..4);
    let w = vector[float,4](e + 1.0 for e in v);
    print(v.to_string());
    print(w.to_string());
    0
}
"""

_MATRIX_SRC = """
fn main() -> int {
    let m = vector[vector[float,2],2](
        vector[float,2](1.0, 2.0),
        vector[float,2](3.0, 4.0)
    );
    print(m[1].to_string());
    print(m[0][1].to_string());
    print((m + m).to_string());
    print((m * 2.0).to_string());
    print(m.to_string());
    0
}
"""

_PROMOTE_SRC = """
fn pick(m: vector[vector[float,1],2]) -> float {
    m[1][0]
}

fn main() -> int {
    print(pick(vector[float,2](5.0, 6.0)).to_string());
    0
}
"""

_EFFECT_DEFAULT_SRC = """
effect Cap {
    ask(x: int) -> int = 7;
}

fn main() -> int {
    print(perform Cap.ask(41));
    0
}
"""

# Increment 15 — DYNAMIC default routing.  `ask` declares a default AND is
# listed by a handle scope, so ONE perform site resolves two different ways
# depending on the handler stack at the moment it runs: inside the handle
# the scope answers (2), outside it the declared default does (7).
_DYNAMIC_DEFAULT_SRC = """
effect Cap {
    ask(x: int) -> int = 7;
}

fn inner() -> int performs Cap {
    perform Cap.ask(1)
}

fn main() -> int {
    let handled = handle Cap with { ask(x) -> resume(x + 1) } in { inner() };
    let bare = inner();
    print(handled);
    print(bare);
    0
}
"""

# (a) the default is taken because NO handler is installed anywhere.
_DYNAMIC_DEFAULT_ONLY_SRC = """
effect Cap {
    ask(x: int) -> int = x + 100;
}

fn inner(x: int) -> int performs Cap {
    perform Cap.ask(x)
}

fn wrapped(x: int) -> int {
    handle Cap with { ask(v) -> resume(v * 2) } in { x }
}

fn main() -> int {
    print(inner(1));
    print(wrapped(9));
    0
}
"""

# (d) a default's result flows into subsequent computation, through an
# AGGREGATE boundary (the enum crosses as a write-once boundary box on both
# routes) — exactly example 06's `Option<...>` capability shape.
_DYNAMIC_DEFAULT_AGG_SRC = """
enum Answer {
    Nothing,
    Value(v: int)
}

effect Cap {
    probe(x: int) -> Answer = Nothing;
}

fn ask(x: int) -> int performs Cap {
    match perform Cap.probe(x) {
        Value(v) -> v,
        Nothing -> x * 3
    }
}

fn main() -> int {
    let bare = ask(4);
    let handled = handle Cap with { probe(v) -> resume(Value(v + 1000)) }
        in { ask(4) };
    print(bare);
    print(handled);
    print(bare + handled);
    0
}
"""

# (c) the same op resolving differently at two sites in ONE dynamic extent:
# the handler case's own perform routes OUTWARD past its busy scope (the
# interpreter's _find_mir_frame busy skip == mx__find_scope's), finds no
# other scope, and lands on the default.
_DYNAMIC_DEFAULT_BUSY_SRC = """
effect Cap {
    ask(x: int) -> int = x * 10;
}

fn inner(x: int) -> int performs Cap {
    perform Cap.ask(x)
}

fn nested(x: int) -> int {
    handle Cap with {
        ask(v) -> {
            let outer = perform Cap.ask(v + 1);
            resume(outer + 1)
        }
    } in { inner(x) }
}

fn main() -> int {
    print(nested(2));
    0
}
"""

# Two dynamically-routed ops of one effect, answered by DIFFERENT nested
# scopes (and by their defaults where no scope lists them) — the outward
# walk past a scope that handles the OTHER op, with a str result crossing
# the boundary and flowing into a concat.
_DYNAMIC_DEFAULT_NESTED_SRC = """
effect Cfg {
    level(x: int) -> int = x + 1;
    label(x: int) -> string = "default";
}

fn describe(x: int) -> string performs Cfg {
    let n = perform Cfg.level(x);
    let s = perform Cfg.label(n);
    s + "/" + n.to_string()
}

fn boosted(x: int) -> string {
    handle Cfg with { level(v) -> resume(v * 100) } in { describe(x) }
}

fn named(x: int) -> string {
    handle Cfg with { label(v) -> resume("named") } in { describe(x) }
}

fn both(x: int) -> string {
    handle Cfg with { level(v) -> resume(v * 2) } in { named(x) }
}

fn main() -> int {
    print(describe(1));
    print(boosted(1));
    print(named(1));
    print(both(1));
    print(describe(5));
    0
}
"""

# A default that ITSELF performs: it runs on the performing stack, so its
# perform sees the same scope stack and parks the same fiber.  `base` is
# handled in the second call and defaulted in the first, while `ask` is
# defaulted in both.
_DYNAMIC_DEFAULT_PERFORMS_SRC = """
effect Cap {
    ask(x: int) -> int = perform Cap.base(x) + 5;
    base(x: int) -> int = x;
}

fn inner(x: int) -> int performs Cap {
    perform Cap.ask(x) * 2
}

fn main() -> int {
    let bare = inner(3);
    let handled = handle Cap with { base(v) -> resume(v * 100) }
        in { inner(3) };
    print(bare);
    print(handled);
    0
}
"""


def test_elementwise_binop_lowers_to_simd_or_mx_fvec_binop():
    # Increment 11: the float shapes in this program all have statically
    # known lengths, so they emit INLINE vector IR; the int division keeps
    # the mx_fvec_binop C loop (division-by-zero abort semantics).
    ir = llvm_from_source(_ELEMENTWISE_SRC)
    assert count_placeholders(ir) == 0
    assert "fadd <4 x double>" in ir
    assert "; scalar broadcast splat" in ir
    assert re.search(r"@mx_fvec_binop\(i64 3, i64 0, i64 0, i64 1,", ir)
    # vectors print through the runtime repr, freed right after
    assert "call ptr @mx_fvec_to_str" in ir
    assert "; print never retains the repr" in ir


def test_zeros_filled_and_dim_lower_natively():
    ir = llvm_from_source(_ZEROS_FILLED_DIM_SRC)
    assert count_placeholders(ir) == 0
    assert "; zero-filled vector" in ir
    assert "call ptr @mx_fvec_filled(i64" in ir
    # `size` binds const N via __vec_dim 0 -> mx_fvec_len
    assert "define i64 @mx___impl__vector_size(" in ir
    assert "call i64 @mx_fvec_len(ptr" in ir


def test_slice_lowers_to_fresh_copy_with_static_none_mask():
    ir = llvm_from_source(_SLICES_SRC)
    assert count_placeholders(ir) == 0
    # v[1:3]: start+stop given, step omitted -> mask 3
    assert re.search(r"@mx_fvec_slice\(ptr %t\d+, i64 %?t?\d+, "
                     r"i64 %?t?\d+, i64 0, i64 3\)", ir)
    # v[::-1] / v[::2]: only step given -> mask 4
    assert len(re.findall(r"i64 4\)  ; honest copy", ir)) == 2


def test_comprehension_emits_per_site_thunk_and_mx_fvec_map():
    ir = llvm_from_source(_COMPREHENSION_SRC)
    assert count_placeholders(ir) == 0
    assert ir.count("call ptr @mx_fvec_map(ptr") == 2
    assert re.search(
        r"define internal i64 @mx\.vcth\.\d+\(ptr %env, i64 %w\)", ir)
    # the range comprehension's int elements convert to the f64 parameter
    assert "sitofp i64 %w to double" in ir
    # ranges stay confined to iteration shapes
    assert "; range as an int vector" in ir


def test_cast_lowers_to_sitofp_fptosi():
    ir = llvm_from_source("""
fn main() -> int {
    let f = 3 as float;
    let i = (f * 1.5) as int;
    print(i);
    0
}
""")
    assert count_placeholders(ir) == 0
    assert "sitofp i64" in ir and "; `as float`" in ir
    assert "fptosi double" in ir and "; `as int`" in ir


def test_matrix_literals_and_nested_binops_emit():
    ir = llvm_from_source(_MATRIX_SRC)
    assert count_placeholders(ir) == 0
    # nested element-wise ops carry depth 1
    assert re.search(r"@mx_fvec_binop\(i64 0, i64 1, i64 1, i64 0,", ir)
    assert re.search(r"@mx_fvec_binop\(i64 2, i64 1, i64 1, i64 1,", ir)
    # matrix repr recurses (depth 1 in mx_fvec_to_str)
    assert re.search(r"@mx_fvec_to_str\(ptr %t\d+, i64 1, i64 1\)", ir)


def test_promote_matrix_rebinds_flat_vector_param_at_entry():
    ir = llvm_from_source(_PROMOTE_SRC)
    assert count_placeholders(ir) == 0
    assert "call ptr @mx_fvec_promote(ptr" in ir
    assert "; promote_matrix: flat vector:f64 -> Mx1 matrix" in ir


def test_push_on_fixed_vector_demotes_honestly():
    # The interpreter rejects push on an MxVector; the immutable native
    # block must never be mutated either — demote, never emit a push.
    ir = llvm_from_source("""
fn main() -> int {
    let v = vector[int,2](1, 2);
    v.push(3);
    0
}
""")
    assert count_placeholders(ir) == 1
    assert "not a Vec (fixed vectors are immutable)" in ir


def test_effect_default_perform_lowers_to_direct_call():
    # No handle_scope in the module lists `ask`, so no scope can ever
    # intercept it: the perform IS a direct call of the declared default.
    ir = llvm_from_source(_EFFECT_DEFAULT_SRC)
    assert count_placeholders(ir) == 0
    assert "-> declared default" in ir
    assert "call i64 @mx___effect_default_Cap_ask(i64" in ir
    assert "call i64 @mx_perform" not in ir


def test_effect_default_with_handle_scope_routes_dynamically():
    # Increment 15: `ask` has a default AND appears in a handle scope, so
    # which one answers a given perform is decided AT THE PERFORM by the
    # runtime's scope stack.  One perform site, one call:
    # mx_perform_or_default runs the same innermost-non-busy lookup as
    # mx_perform and calls the op's default thunk only where mx_perform
    # would have aborted.
    ir = llvm_from_source(_DYNAMIC_DEFAULT_SRC)
    assert count_placeholders(ir) == 0
    # the new call shape, with the per-op thunk pointer and a null env
    assert re.search(
        r"call i64 @mx_perform_or_default\(ptr @\.str\.\d+, ptr @\.str\.\d+, "
        r"ptr %perform\.args, i64 1, "
        r"ptr @mxfx\.dflt\.__effect_default_Cap_ask, ptr null\)", ir)
    assert "in-scope handler, else the declared default" in ir
    assert ("declare i64 @mx_perform_or_default(ptr, ptr, ptr, i64, ptr, ptr)"
            in ir)
    # the thunk: word decode -> call the compiled default -> word encode
    assert ("define internal i64 @mxfx.dflt.__effect_default_Cap_ask"
            "(ptr %env, ptr %args) {") in ir
    assert "%a0w = load i64, ptr %a0p" in ir  # word straight off the scratch
    assert "call i64 @mx___effect_default_Cap_ask(i64 %a0w)" in ir
    # exactly ONE perform site; the plain mx_perform path is not used here
    assert ir.count("call i64 @mx_perform_or_default(") == 1
    assert not re.search(r"call i64 @mx_perform\(", ir)
    # the static-default direct call is NOT taken (a scope may intercept)
    assert "-> declared default (no handle scope" not in ir


def test_range_value_outside_iteration_demotes():
    # Hand-built MIR: a __range result reaching ret would expose the
    # list-vs-vector repr divergence, so the function demotes.
    fs = [
        make_func("main", [block([
            ("params", ()),
            ("let", "a", ("const", 0), ()),
            ("let", "b", ("const", 3), ()),
            ("let", "r", ("call", "__range"), ("a", "b")),
        ], ("ret", "r"))]),
    ]
    ir = emit_llvm(fs)
    assert count_placeholders(ir) == 1
    assert "used outside the iteration protocol" in ir


def test_vector_operations_example_lifts_transpose_and_static_assert():
    # 06_vector_operations.mx after increment 10: the pure-vector functions
    # (transpose + its comprehension lambdas, static_assert) emit; the
    # higher-order SIMD plumbing stays honestly demoted — map/reduce take
    # DIFFERENT lambdas at one call site (closure-kind conflicts), zip's
    # lambda references an undefined name, fold's generic type_of code has
    # free type variables.
    #
    # The HIR triage round changed WHICH honest reason dominates. `with_simd`
    # uses the inline `handle SUBJECT { perform Op(p) => body }` form, which
    # had no HIR lowering at all: the handler silently vanished, `f` was never
    # called, and SimdOp performs could only ever be answered by their declared
    # defaults — which the native backend does lower. With the handler real,
    # try_vectorize/try_horizontal have BOTH a declared default and a handle
    # scope, so answering a perform is a RUNTIME choice; increment 15 lowers
    # exactly that (mx_perform_or_default), which is what puts reduce/sum/
    # mean/dot/norm/normalize/matmul back on the native path.
    ir = llvm_from_source(
        (REPO_ROOT / "examples" / "06_vector_operations.mx").read_text())
    assert "define ptr @mx___impl__vector_transpose(ptr" in ir
    assert "define ptr @mx___impl__vector_transpose_lambda8(" in ir
    assert "define ptr @mx_static_assert()" in ir
    assert ir.count("call ptr @mx_fvec_map(ptr") >= 2
    # increment 15: SimdOp's ops route dynamically — the handle scope in
    # `with_simd` answers when it is installed, the declared `= None`
    # defaults answer everywhere else.  `try_horizontal` (reduce's perform)
    # is the one that survives to emission; `try_vectorize`'s performers
    # (map/zip) still demote on genuine higher-order polymorphism, so no
    # thunk is kept for it — thunk liveness follows its users.
    assert ("define internal i64 @mxfx.dflt.__effect_default_SimdOp_"
            "try_horizontal(ptr %env, ptr %args) {") in ir
    assert "mxfx.dflt.__effect_default_SimdOp_try_vectorize" not in ir
    assert "has a declared default and also appears in a handle scope" not in ir
    assert ir.count("call i64 @mx_perform_or_default(") == 1
    assert "define double @mx___impl_VectorOps_vector_reduce(" in ir
    for lifted in ("__impl__vector_sum",
                   "__impl__vector_mean", "__impl__vector_dot",
                   "__impl__vector_norm", "__impl__vector_normalize",
                   "__impl__vector_matmul"):
        assert f"; function @mx_{lifted}: placeholder" not in ir
    # the honest residue: genuine higher-order polymorphism, not routing
    assert "; function @mx_main: placeholder" in ir
    assert "irreconcilable value kinds" in ir
    for lifted_builtin in ("__vec_dim", "__vec_zeros", "__vec_filled",
                           "__vec_comprehension", "__slice_get", "__range",
                           "__cast"):
        assert f"calls runtime builtin '{lifted_builtin}'" not in ir


@needs_clang
def test_native_elementwise_differential(tmp_path):
    assert_native_matches_interp(_ELEMENTWISE_SRC, tmp_path)


@needs_clang
def test_native_zeros_filled_dim_differential(tmp_path):
    assert_native_matches_interp(_ZEROS_FILLED_DIM_SRC, tmp_path)


@needs_clang
def test_native_slice_differential(tmp_path):
    assert_native_matches_interp(_SLICES_SRC, tmp_path)


@needs_clang
def test_native_comprehension_with_capture_differential(tmp_path):
    assert_native_matches_interp(_COMPREHENSION_SRC, tmp_path)


@needs_clang
def test_native_cast_roundtrip_differential(tmp_path):
    assert_native_matches_interp("""
fn main() -> int {
    let f = 3 as float;
    let g = f * 1.5;
    let i = g as int;
    print(f.to_string());
    print(g.to_string());
    print(i);
    0
}
""", tmp_path)


@needs_clang
def test_native_matrix_differential(tmp_path):
    assert_native_matches_interp(_MATRIX_SRC, tmp_path)


@needs_clang
def test_native_promote_matrix_differential(tmp_path):
    assert_native_matches_interp(_PROMOTE_SRC, tmp_path)


@needs_clang
def test_native_for_range_differential(tmp_path):
    assert_native_matches_interp("""
fn main() -> int {
    let mut s = 0;
    for i in 2..7 {
        s = s + i;
    }
    print(s);
    0
}
""", tmp_path)


@needs_clang
def test_native_effect_default_differential(tmp_path):
    assert_native_matches_interp(_EFFECT_DEFAULT_SRC, tmp_path)


@needs_asan
def test_native_elementwise_asan_no_uaf(tmp_path):
    # Fixed-vector blocks (and the repr strings to_string produces) LEAK
    # BY DESIGN: detect_leaks=0 proves no UAF/double-free — in particular
    # that printing frees each repr exactly once and that shallow-shared
    # blocks are never freed at all.
    assert_native_matches_interp_asan_boxes(_ELEMENTWISE_SRC, tmp_path)


@needs_asan
def test_native_matrix_asan_no_uaf(tmp_path):
    # Nested matrices shallow-share row blocks across literals, binop
    # results and __index_get reads; ASan (leaks excused by contract)
    # proves no row is ever freed or read after free.
    assert_native_matches_interp_asan_boxes(_MATRIX_SRC, tmp_path)


@needs_asan
def test_native_slice_and_comprehension_asan_no_uaf(tmp_path):
    # Slices are fresh copies and comprehensions fresh blocks: prove the
    # copies are independent allocations (no aliasing UAF) under ASan.
    assert_native_matches_interp_asan_boxes(_SLICES_SRC, tmp_path)
    assert_native_matches_interp_asan_boxes(_COMPREHENSION_SRC, tmp_path)


# ---------------------------------------------------------------------------
# Increment 11: vector arithmetic emits real LLVM SIMD IR
# ---------------------------------------------------------------------------
# Static-length tracking (_fvec_static_lens) + the inline `<N x double>` /
# `<N x i64>` fast path for flat vector binops.  The contract under test:
# NEVER WRONG, ONLY FASTER — every proven shape emits one vector arithmetic
# instruction over loads off the block's word array; every unproven shape
# (dynamic/mismatched lengths, matrices, int division, N > 64) falls back
# to the increment 10 mx_fvec_binop C loop with its abort semantics.
# Differentials prove result equality with the interpreter for every
# emitted shape (clang always compiles at -O2, so each differential below
# is also an -O2 equivalence witness).
#
# PERFORMANCE NOTE (deliberately not a timing assert — wall-clock tests
# flake in CI): the structural asserts pin that static-length float/int
# binops are single vector instructions on `<N x double>` / `<N x i64>`,
# which IS the SIMD register width story; the dynamic-length fallback
# remains a C loop that LLVM may still auto-vectorize with a runtime trip
# count.  Expected behavior: the inline path removes the call + per-element
# loop overhead entirely for small fixed N.

_SIMD_FLOAT_SRC = """
fn main() -> int {
    let a = vector[float,4](1.5, -2.0, 3.25, 4.0);
    let b = vector[float,4](10.0, 20.5, -30.0, 0.5);
    print((a + b).to_string());
    print((a - b).to_string());
    print((a * b).to_string());
    print((b / a).to_string());
    print((a * -2.5).to_string());
    print((100.0 + a).to_string());
    0
}
"""

_SIMD_INT_SRC = """
fn main() -> int {
    let a = vector[int,3](7, -8, 9);
    let b = vector[int,3](-1, 2, 40);
    print((a + b).to_string());
    print((a - b).to_string());
    print((a * b).to_string());
    print((a * -3).to_string());
    print((10 + a).to_string());
    let p = vector[int,3](7, 8, 9);
    print((p / 2).to_string());
    0
}
"""
# NOTE: p (all non-negative) is deliberate for /: the backend's
# C-truncating division convention diverges from the interpreter's floor
# on negative operands (the standing documented divergence, unchanged by
# this increment — / stays on the runtime call anyway; the language has
# no source-level % operator).

_SIMD_PROPAGATION_SRC = """
fn main() -> int {
    let v = vector[float,5](1.0, 2.0, 3.0, 4.0, 5.0);
    let s = v[1:3];
    print((s + s).to_string());
    let z = vector[float,4]();
    let f = vector[float,4].filled(2.5);
    print((z + f).to_string());
    let c = vector[float,4](x * 2.0 for x in 0..4);
    print((c * c).to_string());
    let sum = (c + f) * 2.0;
    print(sum.to_string());
    0
}
"""

_SIMD_JOIN_SRC = """
fn main() -> int {
    let mut v = vector[float,2](1.0, 2.0);
    if 1 == 1 {
        v = vector[float,3](1.0, 2.0, 3.0);
    }
    print((v + v).to_string());
    0
}
"""


def test_static_length_float_binop_emits_vector_ir():
    # THE POINT of increment 11: static-length float vector arithmetic is
    # real SIMD IR — `<4 x double>` loads off the block words (offset 8,
    # the block's actual 8-byte alignment), one vector instruction per op,
    # a store into a fresh result block, and NO runtime binop call left
    # anywhere in the module.
    ir = llvm_from_source(_SIMD_FLOAT_SRC)
    assert count_placeholders(ir) == 0
    for mnem in ("fadd", "fsub", "fmul", "fdiv"):
        assert f"{mnem} <4 x double>" in ir, mnem
    assert re.search(
        r"getelementptr inbounds i8, ptr %t\d+, i64 8", ir)
    assert "load <4 x double>, ptr" in ir and ", align 8" in ir
    assert "store <4 x double>" in ir
    assert "call ptr @mx_fvec_new(i64 4)" in ir
    assert "@mx_fvec_binop(" not in ir


def test_scalar_broadcast_emits_splat_both_orientations():
    # vector*scalar and scalar+vector both splat via insertelement +
    # shufflevector; the vector operand's static length drives the width.
    ir = llvm_from_source(_SIMD_FLOAT_SRC)
    assert "insertelement <4 x double> poison, double" in ir
    assert ("shufflevector <4 x double> %t" in ir
            and "<4 x i32> zeroinitializer" in ir)
    assert ir.count("; scalar broadcast splat") == 2


def test_int_vector_add_sub_mul_inline_div_keeps_runtime_call():
    # Int + - * vectorize; / keeps the mx_fvec_binop C loop even with
    # known lengths — DOCUMENTED CHOICE: the runtime aborts loudly on
    # division by zero where vector sdiv would be UB.
    ir = llvm_from_source(_SIMD_INT_SRC)
    assert count_placeholders(ir) == 0
    for mnem in ("add", "sub", "mul"):
        assert f"{mnem} <3 x i64>" in ir, mnem
    assert "insertelement <3 x i64> poison, i64" in ir
    assert "sdiv <" not in ir and "srem <" not in ir
    # op 3 = /, base 0 = int, mode 1 = vector-scalar
    assert re.search(r"@mx_fvec_binop\(i64 3, i64 0, i64 0, i64 1,", ir)


def test_static_lengths_propagate_through_slices_zeros_filled_comprehensions():
    # const-bounds slice of a known-length vector -> <2 x double>; zeros /
    # filled with const counts -> <4 x double>; comprehension output takes
    # its iterable's length; binop RESULTS keep their operands' length so
    # chains stay on the fast path.
    ir = llvm_from_source(_SIMD_PROPAGATION_SRC)
    assert count_placeholders(ir) == 0
    assert "fadd <2 x double>" in ir     # slice v[1:3]
    assert "fadd <4 x double>" in ir     # zeros + filled
    assert "fmul <4 x double>" in ir     # comprehension result, and chain
    assert "@mx_fvec_binop(" not in ir


def test_dynamic_length_falls_back_to_runtime_call():
    # Parameters have no static length (lengths are per-function facts, not
    # part of the kind that flows through sigs): the always-correct C loop
    # remains the lowering, and no vector-arithmetic IR is emitted.
    ir = llvm_from_source("""
fn add(a: vector[float,4], b: vector[float,4]) -> vector[float,4] {
    a + b
}

fn main() -> int {
    let v = vector[float,4](1.0, 2.0, 3.0, 4.0);
    print(add(v, v).to_string());
    0
}
""")
    assert count_placeholders(ir) == 0
    assert re.search(r"@mx_fvec_binop\(i64 0, i64 1, i64 0, i64 0,", ir)
    assert "fadd <" not in ir


def test_mismatched_static_lengths_fall_back_to_aborting_runtime_call():
    # Both lengths are KNOWN but unequal: the inline path must never be
    # taken (it would skip the length-mismatch abort); the runtime call
    # keeps the interpreter's error semantics (mx_rt_fail at run time).
    ir = llvm_from_source("""
fn main() -> int {
    let a = vector[float,2](1.0, 2.0);
    let b = vector[float,3](1.0, 2.0, 3.0);
    print((a + b).to_string());
    0
}
""")
    assert re.search(r"@mx_fvec_binop\(i64 0, i64 1, i64 0, i64 0,", ir)
    assert "fadd <" not in ir


def test_length_join_across_branch_defs_degrades_to_dynamic():
    # One variable, two defs with different lengths: the per-name join is
    # dynamic, so the binop falls back (a sound join can never pick 2 OR 3).
    ir = llvm_from_source(_SIMD_JOIN_SRC)
    assert count_placeholders(ir) == 0
    assert re.search(r"@mx_fvec_binop\(i64 0, i64 1, i64 0, i64 0,", ir)
    assert "fadd <" not in ir


def test_length_above_cap_falls_back():
    # N = 100 > 64: correct via the C loop (which LLVM can still
    # auto-vectorize), no giant IR vectors.
    ir = llvm_from_source("""
fn main() -> int {
    let a = vector[float,100].filled(1.5);
    let b = a + a;
    print(b.len());
    0
}
""")
    assert count_placeholders(ir) == 0
    assert re.search(r"@mx_fvec_binop\(i64 0, i64 1, i64 0, i64 0,", ir)
    assert "fadd <" not in ir


def test_matrix_binops_keep_runtime_recursion():
    # Nested vectors (depth > 0) stay on mx_fvec_binop: elements are row
    # POINTERS, not lanes — a vector instruction over them would be wrong.
    ir = llvm_from_source(_MATRIX_SRC)
    assert count_placeholders(ir) == 0
    assert re.search(r"@mx_fvec_binop\(i64 0, i64 1, i64 1, i64 0,", ir)
    assert "fadd <" not in ir


def test_vector_ir_module_passes_llvm_verifier(tmp_path):
    if shutil.which("opt") is None:
        pytest.skip("LLVM opt not installed")
    for src in (_SIMD_FLOAT_SRC, _SIMD_INT_SRC, _SIMD_PROPAGATION_SRC):
        ll = tmp_path / "simd.ll"
        ll.write_text(llvm_from_source(src))
        proc = subprocess.run(
            ["opt", "-passes=verify", "-disable-output", str(ll)],
            capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr


@needs_clang
def test_native_simd_float_differential(tmp_path):
    # Differential at -O2 (compile_and_run always passes clang -O2): the
    # inline vector IR must produce the interpreter's exact reprs for
    # + - * / and both broadcast orientations, negatives included.
    ir = assert_native_matches_interp(_SIMD_FLOAT_SRC, tmp_path)
    assert "fadd <4 x double>" in ir  # the fast path was actually on trial


@needs_clang
def test_native_simd_int_differential(tmp_path):
    ir = assert_native_matches_interp(_SIMD_INT_SRC, tmp_path)
    assert "mul <3 x i64>" in ir


@needs_clang
def test_native_simd_propagation_differential(tmp_path):
    ir = assert_native_matches_interp(_SIMD_PROPAGATION_SRC, tmp_path)
    assert "@mx_fvec_binop(" not in ir


@needs_clang
def test_native_simd_length_join_differential(tmp_path):
    assert_native_matches_interp(_SIMD_JOIN_SRC, tmp_path)


@needs_clang
def test_native_simd_inf_nan_differential(tmp_path):
    # inf via overflow and nan via inf - inf — both produced INSIDE the
    # vector fast path (float division by zero can't be the vehicle: the
    # interpreter rejects it, the documented IEEE divergence).  The runtime
    # repr renders "inf"/"-inf"/"nan" exactly like Python's str().
    ir = assert_native_matches_interp("""
fn main() -> int {
    let mut v = vector[float,2](10.0, -10.0);
    let mut i = 0;
    while i < 400 {
        v = v * 10.0;
        i = i + 1;
    }
    let nans = v - v;
    print(v.to_string());
    print(nans.to_string());
    0
}
""", tmp_path)
    assert "fmul <2 x double>" in ir and "fsub <2 x double>" in ir


@needs_asan
def test_native_simd_asan_no_oob_store(tmp_path):
    # The inline path stores `<N x double>` into a fresh mx_fvec_new block
    # of exactly 8 + 8N bytes: ASan proves the vector store stays inside
    # the allocation (blocks leak by design -> detect_leaks=0 contract).
    assert_native_matches_interp_asan_boxes(_SIMD_FLOAT_SRC, tmp_path)
    assert_native_matches_interp_asan_boxes(_SIMD_PROPAGATION_SRC, tmp_path)


# --- Reductions: the emission helper (no MIR shape reaches it yet) --------

def test_reduce_helper_emits_ordered_fadd_reduction():
    decl, body, res = emit_fvec_reduce("%v", 4, "f64", "rd")
    assert decl == ("declare double @llvm.vector.reduce.fadd.v4f64"
                    "(double, <4 x double>)")
    text = "\n".join(body)
    assert "getelementptr inbounds i8, ptr %v, i64 8" in text
    assert "load <4 x double>" in text and "align 8" in text
    # ORDERED reduction (no reassoc), seeded with -0.0 (exact fadd
    # identity) == the interpreter's left-to-right fold, bit for bit.
    assert ("call double @llvm.vector.reduce.fadd.v4f64"
            "(double -0.000000e+00, <4 x double> %rd.v)") in text
    assert res == "%rd.sum"


def test_reduce_helper_int_form_and_bad_inputs():
    decl, body, res = emit_fvec_reduce("%v", 8, "i64", "rd")
    assert decl == "declare i64 @llvm.vector.reduce.add.v8i64(<8 x i64>)"
    assert any("call i64 @llvm.vector.reduce.add.v8i64" in ln
               for ln in body)
    with pytest.raises(ValueError):
        emit_fvec_reduce("%v", 0, "f64", "rd")
    with pytest.raises(ValueError):
        emit_fvec_reduce("%v", 4, "str", "rd")


def _reduce_module(vals, leaf: str) -> str:
    """A synthetic module exercising emit_fvec_reduce end to end: build the
    block with the real runtime (mx_fvec_new/_init), reduce, print."""
    import struct
    n = len(vals)
    if leaf == "f64":
        words = [struct.unpack("<q", struct.pack("<d", float(v)))[0]
                 for v in vals]
    else:
        words = [int(v) for v in vals]
    decl, body, res = emit_fvec_reduce("%v", n, leaf, "rd")
    fmt = "%.17g\\0A\\00" if leaf == "f64" else "%lld\\0A\\00"
    fmtlen = 7 if leaf == "f64" else 6  # strlen + \n + NUL
    lines = [
        "declare ptr @mx_fvec_new(i64)",
        "declare void @mx_fvec_init(ptr, i64, i64)",
        "declare i32 @printf(ptr, ...)",
        decl,
        f'@.fmt = private unnamed_addr constant [{fmtlen} x i8] c"{fmt}"',
        f"define i64 @{mangle('main')}() {{",
        "entry:",
        f"  %v = call ptr @mx_fvec_new(i64 {n})",
    ]
    lines += [f"  call void @mx_fvec_init(ptr %v, i64 {i}, i64 {w})"
              for i, w in enumerate(words)]
    lines += body
    ty = "double" if leaf == "f64" else "i64"
    lines += [
        f"  %p = call i32 (ptr, ...) @printf(ptr @.fmt, {ty} {res})",
        "  ret i64 0",
        "}",
    ]
    return "\n".join(lines) + "\n"


@needs_clang
def test_reduce_helper_float_runs_and_matches_python_fold(tmp_path):
    # Values chosen so ORDER MATTERS: a reassociating reduction (or a tree
    # reduction) would round differently — equality with Python's
    # left-to-right fold proves the ordered semantics, not just the sum.
    vals = [1.0, 1e16, -1e16, 3.5, 0.1, -0.25]
    expected = 0.0
    for v in vals:
        expected = expected + v
    ir = _reduce_module(vals, "f64")
    if shutil.which("opt") is not None:
        ll = tmp_path / "reduce.ll"
        ll.write_text(ir)
        proc = subprocess.run(
            ["opt", "-passes=verify", "-disable-output", str(ll)],
            capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
    exit_code, out = compile_and_run(ir, "main", workdir=str(tmp_path))
    assert exit_code == 0
    assert out == f"{expected:.17g}\n"


@needs_clang
def test_reduce_helper_int_runs_and_matches_python_sum(tmp_path):
    vals = [5, -7, 40, 3, -1, 2**40]
    ir = _reduce_module(vals, "i64")
    exit_code, out = compile_and_run(ir, "main", workdir=str(tmp_path))
    assert exit_code == 0
    assert out == f"{sum(vals)}\n"


# ---------------------------------------------------------------------------
# Increment 12: silent-seam constructs — index assignment, mutable-capture
# cells, module constants, zip comprehensions
# ---------------------------------------------------------------------------

_VEC_INDEX_STORE_SRC = """
fn main() -> int {
    let v = Vec.new();
    v.push(1);
    v.push(2);
    v[0] = 10;
    print(v[0] + v[1]);
    0
}
"""


def test_index_store_on_vec_lowers_to_mx_vec_set_and_stays_freeable():
    ir = llvm_from_source(_VEC_INDEX_STORE_SRC)
    assert count_placeholders(ir) == 0
    assert re.search(r"call void @mx_vec_set\(ptr %t\d+, i64 %?\w+, "
                     r"i64 %?\w+\)", ir)
    # The __index_store result aliases the receiver (same pointer), so the
    # dead-vec analysis still proves the vec frame-local and frees it.
    assert "call void @mx_vec_free" in ir


_FVEC_INDEX_STORE_SRC = """
fn main() -> int {
    let mut v = vector[int, 3](1, 2, 3);
    let w = v;
    v[1] = 20;
    print(v[0] + v[1] + v[2]);
    print(w[1]);
    0
}
"""


def test_index_store_on_fixed_vector_is_functional_copy():
    ir = llvm_from_source(_FVEC_INDEX_STORE_SRC)
    assert count_placeholders(ir) == 0
    # Never a mutation in place: the write-once block is COPIED, the
    # element set in the fresh block, and the place rebound.
    assert "call ptr @mx_fvec_set_copy(ptr" in ir
    assert "functional update" in ir
    assert "mx_fvec_init" in ir  # the literal still fills in place


def test_index_store_immutable_receiver_demotes_at_compile_time():
    ir = llvm_from_source("""
fn main() -> int {
    let s = "abc";
    s[0] = "x";
    1
}
""")
    assert count_placeholders(ir) == 1
    assert "index assignment into a str receiver" in ir
    assert "the interpreter rejects it" in ir


def test_index_set_on_immutable_vector_demotes_with_interpreter_error():
    # `m[0][1] = x` reaches the in-place __index_set form; a fixed-vector
    # inner row is immutable, so it demotes at compile time with the
    # interpreter's own message (never a silent store, never an abort the
    # compiler could have predicted).
    ir = llvm_from_source("""
fn main() -> int {
    let m = vector[vector[int,2],2](vector[int,2](1,2), vector[int,2](3,4));
    m[0][1] = 9;
    0
}
""")
    assert count_placeholders(ir) == 1
    assert "cannot assign into an immutable vector" in ir


def test_mutable_capture_lowers_to_shared_cell():
    ir = llvm_from_source("""
fn main() -> int {
    let mut count = 0;
    let bump = fn() { count = count + 1; };
    bump();
    bump();
    print(count);
    0
}
""")
    assert count_placeholders(ir) == 0
    # One-word heap cell, reads/writes through the pointer, env captures
    # the POINTER (aliasing is the point).
    assert re.search(r"%cellp\.count_\d+ = call ptr @malloc\(i64 8\)", ir)
    assert "; cell write: count_" in ir
    assert "; cell read: count_" in ir
    assert "mutable capture: cell pointer for count_" in ir


def test_pre_wrap_capture_demotes_honestly():
    # `r` captures count BEFORE the wrap (the interpreter freezes a VALUE
    # copy there: r() sees 0 even after w() runs); whole-frame cell
    # backing would show r the live value — demote, never diverge.
    ir = llvm_from_source("""
fn main() -> int {
    let mut count = 0;
    let r = fn() -> int { count };
    let w = fn() { count = count + 1; };
    w();
    print(r());
    0
}
""")
    assert "not provably after its cell_wrap" in ir


def test_module_constants_emit_globals_and_init_stores():
    ir = llvm_from_source("""
let BASE = 40;
let SCALE = 2.5;
let NAME = "metaxu";
fn get() -> int { BASE + 1 }
fn main() -> int {
    print(BASE + get());
    print(NAME);
    if SCALE > 2.0 { print(1) } else { print(0) };
    0
}
""")
    assert count_placeholders(ir) == 0
    assert "@mx_g_BASE = internal global i64 0" in ir
    assert "@mx_g_SCALE = internal global double" in ir
    assert "@mx_g_NAME = internal global ptr null" in ir
    # init is a normal function storing into the globals; readers load.
    assert re.search(r"^define (?:i64|double|ptr) @mx___module_init\(\)",
                     ir, re.M)
    assert "store i64 40, ptr @mx_g_BASE" in ir
    # get() reads the global with no local binding (interpreter fallback).
    assert re.search(r"load i64, ptr @mx_g_BASE", ir)


def test_local_shadow_of_module_constant_demotes():
    # Assignment creates a flow-sensitive local shadow in the interpreter
    # (reads before it see the global): no static storage class matches.
    ir = llvm_from_source("""
let BASE = 7;
fn f() -> int { BASE = 2; BASE }
fn main() -> int { print(f()); print(BASE); 0 }
""")
    assert "shadows the global flow-sensitively" in ir


def test_zip_comprehension_emits_two_word_thunk_and_zip_map():
    ir = llvm_from_source("""
fn main() -> int {
    let xs = vector[int, 3](1, 2, 3);
    let ys = vector[int, 3](10, 20, 30);
    let zs = vector[int, 3](a + b for (a, b) in (xs, ys));
    print(zs[0] + zs[1] + zs[2]);
    0
}
""")
    assert count_placeholders(ir) == 0
    assert re.search(
        r"define internal i64 @mx\.vzth\.\d+\(ptr %env, i64 %wa, i64 %wb\)",
        ir)
    assert re.search(
        r"call ptr @mx_fvec_zip_map\(ptr %t\d+, ptr %t\d+, "
        r"ptr @mx\.vzth\.\d+, ptr %t\d+, i64 3\)", ir)
    # The zip result itself is virtual: no materialized list of tuples.
    assert "virtual pair iterable" in ir


# --- native differentials --------------------------------------------------

@needs_clang
def test_native_index_store_vec_differential(tmp_path):
    assert_native_matches_interp(_VEC_INDEX_STORE_SRC, tmp_path)


@needs_asan
def test_native_index_store_vec_fully_leak_checked_under_asan(tmp_path):
    # The index-stored local vec is still provably frame-local (the store
    # result aliases the receiver) and freed: FULL leak check.
    assert_native_matches_interp_asan(_VEC_INDEX_STORE_SRC, tmp_path)


@needs_clang
def test_native_index_store_fvec_differential_preserves_shares(tmp_path):
    # `w = v` snapshots the pre-update value: the functional update must
    # COPY the block, never mutate it in place (w[1] stays 2).
    assert_native_matches_interp(_FVEC_INDEX_STORE_SRC, tmp_path)


@needs_asan
def test_native_index_store_fvec_asan_no_uaf(tmp_path):
    # fvec blocks (original + functional-update copy) leak by design.
    result, expected_out = interp_run(_FVEC_INDEX_STORE_SRC)
    ir = llvm_from_source(_FVEC_INDEX_STORE_SRC)
    exit_code, stdout = compile_and_run(
        ir, "main", workdir=str(tmp_path),
        clang_args=("-fsanitize=address",),
        run_env={"ASAN_OPTIONS": "detect_leaks=0"})
    assert stdout == expected_out
    assert exit_code == 0


@needs_clang
def test_native_index_store_struct_field_place_differential(tmp_path):
    # ownership.mx's shape: `buf.data[i] = x` through a @mut struct param
    # — the functional fvec update flows through field write-back.
    assert_native_matches_interp("""
struct Buffer { data: vector[int,3] }
fn process(buf: @mut Buffer) {
    buf.data[0] = 42
}
fn main() -> int {
    let buf = Buffer { data: vector[int,3](1, 2, 3) };
    process(buf);
    print(buf.data[0]);
    print(buf.data[1]);
    0
}
""", tmp_path)


@needs_clang
def test_native_nested_vec_index_set_differential(tmp_path):
    # `m[0][1] = x` on a Vec of Vecs: in-place store on the shared inner
    # row (identity semantics observable through the original binding).
    assert_native_matches_interp("""
fn main() -> int {
    let row = Vec.new();
    row.push(1);
    row.push(2);
    let m = Vec.new();
    m.push(row);
    m[0][1] = 9;
    print(row[1]);
    0
}
""", tmp_path)


_COUNTER_SRC = """
fn main() -> int {
    let mut count = 0;
    let bump = fn() { count = count + 1; };
    bump();
    bump();
    bump();
    print(count);
    0
}
"""


@needs_clang
def test_native_mutable_capture_counter_differential(tmp_path):
    assert_native_matches_interp(_COUNTER_SRC, tmp_path)


@needs_asan
def test_native_mutable_capture_counter_asan_no_uaf(tmp_path):
    # Cells leak by design (malloc'd one-word boxes, never freed).
    result, expected_out = interp_run(_COUNTER_SRC)
    ir = llvm_from_source(_COUNTER_SRC)
    exit_code, stdout = compile_and_run(
        ir, "main", workdir=str(tmp_path),
        clang_args=("-fsanitize=address",),
        run_env={"ASAN_OPTIONS": "detect_leaks=0"})
    assert stdout == expected_out
    assert exit_code == 0


@needs_clang
def test_native_escaping_closure_retains_cell_state(tmp_path):
    # The counter closure ESCAPES its creating frame: the cell (like the
    # heap env holding its pointer) must outlive make_counter, and two
    # counters must not share state.
    assert_native_matches_interp("""
fn make_counter() -> fn() -> int {
    let mut n = 0;
    fn() -> int {
        n = n + 1;
        n
    }
}
fn main() -> int {
    let c = make_counter();
    c();
    c();
    print(c());
    let d = make_counter();
    print(d());
    0
}
""", tmp_path)


@needs_clang
def test_native_handler_cell_counter_differential(tmp_path):
    # The std.stream take/skip shape: a handler case mutating a captured
    # scalar across resumes (the scope env carries the cell pointer).
    assert_native_matches_interp("""
effect Emit { emit(x) -> Unit }
fn main() -> int {
    let mut total = 0;
    handle Emit with {
        emit(x) -> { total = total + x; resume(()) }
    } in {
        perform Emit.emit(4);
        perform Emit.emit(5);
        ()
    };
    print(total);
    0
}
""", tmp_path)


_GLOBALS_SRC = """
let BASE = 40;
let SCALE = 2.5;
let NAME = "metaxu";
fn get() -> int { BASE + 1 }
fn main() -> int {
    print(BASE + get());
    print(NAME);
    if SCALE > 2.0 { print(1) } else { print(0) };
    0
}
"""


@needs_clang
def test_native_module_constants_differential(tmp_path):
    # Reads from main AND from another function; the entry wrapper calls
    # @mx___module_init first (the interpreter's _ensure_globals).
    ir = assert_native_matches_interp(_GLOBALS_SRC, tmp_path)
    wrapper = (tmp_path / "prog.ll").read_text()
    assert "call ptr @mx___module_init()" in wrapper


@needs_asan
def test_native_module_constants_fully_leak_checked_under_asan(tmp_path):
    # Scalar globals allocate nothing: FULL leak check.
    assert_native_matches_interp_asan(_GLOBALS_SRC, tmp_path)


@needs_clang
def test_native_module_constant_from_lambda_differential(tmp_path):
    # A lambda's free global name resolves as a global read (no env
    # capture), the interpreter's lookup-fallback order.
    assert_native_matches_interp("""
let K = 5;
fn main() -> int {
    let f = fn() -> int { K * 2 };
    print(f());
    0
}
""", tmp_path)


_ZIP_SRC = """
fn main() -> int {
    let xs = vector[int, 3](1, 2, 3);
    let ys = vector[int, 3](10, 20, 30);
    let zs = vector[int, 3](a + b for (a, b) in (xs, ys));
    print(zs[0] + zs[1] + zs[2]);
    0
}
"""


@needs_clang
def test_native_zip_comprehension_differential(tmp_path):
    assert_native_matches_interp(_ZIP_SRC, tmp_path)


@needs_asan
def test_native_zip_comprehension_asan_no_uaf(tmp_path):
    # fvec blocks (sources + zip_map result) leak by design.
    result, expected_out = interp_run(_ZIP_SRC)
    ir = llvm_from_source(_ZIP_SRC)
    exit_code, stdout = compile_and_run(
        ir, "main", workdir=str(tmp_path),
        clang_args=("-fsanitize=address",),
        run_env={"ASAN_OPTIONS": "detect_leaks=0"})
    assert stdout == expected_out
    assert exit_code == 0


@needs_clang
def test_native_zip_length_mismatch_aborts_loudly(tmp_path):
    # The interpreter raises "zip: sequences have different lengths";
    # mx_fvec_zip_map aborts — loud on both sides, never a short zip.
    src = """
fn main() -> int {
    let xs = vector[int, 2](1, 2);
    let ys = vector[int, 3](10, 20, 30);
    let zs = vector[int, 2](a + b for (a, b) in (xs, ys));
    print(zs[0]);
    0
}
"""
    with pytest.raises(Exception, match="zip"):
        interp_run(src)
    ir = llvm_from_source(src)
    assert count_placeholders(ir) == 0
    exit_code, stdout = compile_and_run(ir, "main", workdir=str(tmp_path))
    assert exit_code != 0  # SIGABRT, no output
    assert stdout == ""


_STD_STREAM_SRC = """
from std.stream import Emit, iota, iter, fold, sum, count, take;

fn main() -> int {
    let s = take(iota(10), 3);
    let total = sum(s);
    let n = count(iota(5));
    print(total);
    print(n);
    total + n
}
"""


def test_std_stream_import_emits_cell_counter_handlers():
    # A program importing std.stream: the cell constructs are LIFTED —
    # no `cell_wrap` demotion reason survives anywhere, and the counter
    # handler cases (take/skip's `seen` — the exact shapes the stdlib
    # documents as relying on shared cells) emit as real defines.  Since
    # increment 13 the USED consumers (fold/sum/count/take + main) emit
    # too: their function-valued parameters go through the word-uniform
    # indirect-call ABI.  Drivers this program never calls (iter, map,
    # filter, ...) still demote honestly — their producer/f parameters
    # stay at the i64 bottom (no closure ever flows in), so a call
    # through them cannot be typed.
    ir = llvm_from_source(_STD_STREAM_SRC)
    # the seam construct itself never demotes anything anymore
    assert "unsupported op 'cell_wrap'" not in ir
    assert not re.search(r"reason:.*cell_wrap", ir)
    assert re.search(
        r"^define i64 @mx___handler_Emit_emit_std_stream_take(?:_ho\d+)?_hs\d+\(",
        ir, re.M)
    assert re.search(
        r"^define i64 @mx___handler_Emit_emit_std_stream_skip(?:_ho\d+)?_hs\d+\(",
        ir, re.M)
    assert re.search(
        r"^define i64 @mx___handler_Loop_break__std_stream_for_(?:_ho\d+)?_hs\d+\(",
        ir, re.M)
    # The whole used pipeline emits: main, fold, sum, count, take, iota,
    # the take thunk, and fold's handler case (which calls `f` indirectly).
    # take and iota return closures, so their defines are sret-style.
    # A `(?:_ho\d+)?` in the symbol accepts monomorphize's higher-order
    # per-call-site clones: a stdlib function whose every call site passes
    # a lambda literal is cloned per site and its dead original erased, so
    # the define exists under the clone name (same body, fresh symbol).
    for sym in ("mx_main", r"mx_std_stream_fold(?:_ho\d+)?",
                r"mx_std_stream_sum(?:_ho\d+)?",
                r"mx_std_stream_count(?:_ho\d+)?",
                r"mx_std_stream_take(?:_ho\d+)?",
                "mx_std_stream_iota",
                r"mx_std_stream_take(?:_ho\d+)?_lambda1",
                r"mx___handler_Emit_emit_std_stream_fold(?:_ho\d+)?_hs1"):
        assert re.search(rf"^define (?:i64|void) @{sym}\(", ir, re.M), sym
    # fold's f is a dynamic closure (sum/product/count lambdas): the case
    # calls it indirectly through the word-uniform ABI.
    assert re.search(r"indirect closure call \(.*sum\$lambda", ir)
    # Unused drivers demote honestly on their bottom-kinded parameters.
    assert re.search(
        r"reason: call through local 'producer' that is not a "
        r"statically-known closure", ir)


def test_ownership_example_now_emits_fully_native():
    # ownership.mx (both fns) was fully demoted on __index_store before
    # increment 12; the fvec functional update lifts it end to end.
    ir = llvm_from_source(
        (REPO_ROOT / "examples" / "ownership.mx").read_text())
    assert count_placeholders(ir) == 0
    assert "call ptr @mx_fvec_set_copy(ptr" in ir


@needs_clang
def test_native_ownership_example_differential(tmp_path):
    src = (REPO_ROOT / "examples" / "ownership.mx").read_text()
    result, expected_out = interp_run(src)
    ir = llvm_from_source(src)
    exit_code, stdout = compile_and_run(ir, "main", workdir=str(tmp_path))
    assert stdout == expected_out


# ---------------------------------------------------------------------------
# Increment 13: indirect closure calls — dynamic closure kinds, the
# word-uniform lambda ABI, env-captured closure pairs, std.stream natively
# ---------------------------------------------------------------------------

_APPLY_TWICE_SRC = """
fn apply(f: fn(int) -> int, v: int) -> int { f(v) }
fn twice(f: fn(int) -> int, v: int) -> int { f(f(v)) }
fn main() -> int {
    let x = 10;
    let g = fn(y: int) -> x + y;
    let h = fn(y: int) -> y * 2;
    print(apply(g, 7));
    print(apply(h, 7));
    print(twice(g, 7));
    print(twice(h, 7));
    apply(g, 1)
}
"""


def test_dynamic_closure_join_instead_of_conflict():
    # Two different lambdas reach apply/twice's `f`: the kinds JOIN to a
    # dynamic member set instead of conflicting, and the whole program
    # emits with zero placeholders.
    ir = llvm_from_source(_APPLY_TWICE_SRC)
    assert count_placeholders(ir) == 0
    assert "irreconcilable value kinds" not in ir
    # the call site is an indirect word-uniform call naming both members
    assert re.search(
        r"%t\d+ = call i64 %t\d+\(ptr %t\d+, i64 [%\w.]+\)"
        r"  ; indirect closure call \(main\$lambda\d+\|main\$lambda\d+\), "
        r"word-uniform ABI", ir)


def test_word_uniform_lambda_signature_and_typed_locals_untouched():
    # Participating lambdas carry the word-uniform ABI marker; a lambda
    # only ever called through its own local binding keeps its typed
    # signature (no ABI marker, direct typed indirect call).
    ir = llvm_from_source(_APPLY_TWICE_SRC)
    assert re.search(
        r"define i64 @mx_main_lambda\d+\(ptr %cl\.env, i64 %a\.y\) "
        r"\{  ; word-uniform lambda ABI", ir)
    local_only = llvm_from_source("""
fn main() -> int {
    let g = fn(y: float) -> float { y * 2.0 };
    print(g(1.5));
    0
}
""")
    assert count_placeholders(local_only) == 0
    assert "word-uniform lambda ABI" not in local_only
    # typed call: double argument straight through the loaded fn pointer
    assert re.search(r"call double %t\d+\(ptr %t\d+, double", local_only)


def test_word_uniform_f64_params_decode_and_encode():
    # An f64 lambda through an indirect site: parameters arrive as i64
    # words and are bitcast-decoded in the prelude; the return value is
    # bitcast-encoded back to a word.
    ir = llvm_from_source("""
fn apply(f: fn(float) -> float, v: float) -> float { f(v) }
fn main() -> int {
    let a = fn(x: float) -> x * 2.0;
    let b = fn(x: float) -> x + 0.5;
    print(apply(a, 1.25));
    print(apply(b, 1.25));
    0
}
""")
    assert count_placeholders(ir) == 0
    assert re.search(
        r"define i64 @mx_main_lambda\d+\(ptr %cl\.env, i64 %aw\.x\)", ir)
    assert re.search(
        r"%a\.x = bitcast i64 %aw\.x to double  ; word-uniform param x: "
        r"f64 decoded", ir)
    assert re.search(r"ret i64 %t\d+  ; word-uniform lambda return "
                     r"\(f64 encoded\)", ir)
    # the site word-encodes the argument and decodes the result
    assert re.search(r"bitcast double %a\.v to i64", ir)


def test_mismatched_arity_lambdas_at_one_site_demote_loudly():
    # A 1-arity and a 2-arity lambda joined at one `f`: no sound indirect
    # call exists (the interpreter's zip-binding would leave the second
    # parameter unbound) — the join conflicts and demotes with reasons.
    ir = llvm_from_source("""
fn pick(c: bool) -> int {
    let mut f = fn(x: int) -> x + 1;
    if c { f = fn(x: int, y: int) -> x + y; } else { () };
    f(3)
}
fn main() -> int { pick(true) }
""")
    assert count_placeholders(ir) >= 1
    assert "irreconcilable value kinds for 'f" in ir


def test_aggregate_args_through_indirect_call_now_box():
    # Increment 16: two lambdas taking a STRUCT parameter reach one site.
    # The struct arg travels as a pointer word and both lambdas take
    # `i64 %aw.p` and copy out.  This replaces the increment-13
    # scalar-only demotion.
    #
    # Increment 17 (work item 2): the pointer word is now the CALLER'S
    # STORAGE, not a fresh box — neither lambda writes through the
    # position (writeback_map) and the call is synchronous on this stack,
    # so the box bought nothing.  The receiving ABI is byte-for-byte the
    # same; only the allocation is gone.
    ir = llvm_from_source("""
struct P { a: int }
fn apply(f: fn(P) -> int, p: P) -> int { f(p) }
fn main() -> int {
    let g = fn(p: P) -> p.a + 1;
    let h = fn(p: P) -> p.a * 2;
    print(apply(g, P { a: 4 }));
    apply(h, P { a: 4 })
}
""")
    assert count_placeholders(ir) == 0
    assert "scalar-only" not in ir
    # NO box for the argument any more.
    assert not re.search(
        r"call ptr @malloc\(i64 \d+\)"
        r"  ; boundary box: struct:P \(write-once, leaks by design\)", ir)
    assert re.search(
        r"ptrtoint ptr %[\w.]+ to i64  ; elide-box: read-only aggregate "
        r"argument \w+ \(struct:P\) passes the caller's storage pointer", ir)
    assert re.search(
        r"call i64 %\w+\(ptr %\w+, i64 %\w+\)"
        r"  ; indirect closure call \(.*\), word-uniform ABI", ir)
    assert len(re.findall(
        r"; word-uniform param p: boundary box struct:P", ir)) == 2


_COMPREHENSION_DYN_SRC = """
fn total(f: fn(int) -> int) -> int {
    let src = vector[int, 3](1, 2, 3);
    let v = vector[int, 3](f(x) for x in src);
    v[0] + v[1] + v[2]
}
fn main() -> int {
    let add = fn(x: int) -> x + 1;
    let mul = fn(x: int) -> x * 10;
    print(total(add));
    print(total(mul));
    0
}
"""
# ^ The lambdas are routed through variables ON PURPOSE: monomorphize's
# per-call-site cloning fires on lambda LITERAL arguments, and this test
# pins the SHARED-site machinery (one `total`, two callers, word-uniform
# indirect f). Literal arguments would split the site into two pinned
# clones and the machinery under test would never engage.


def test_comprehension_body_calling_dynamic_closure_emits():
    # The comprehension BODY is always a per-site synthesized lambda (so
    # the thunk keeps one pinned symbol — source programs cannot make the
    # body variable itself dynamic), but the body may CAPTURE a dynamic f
    # and call it indirectly: the typed thunk path and the word-uniform
    # indirect path compose.
    ir = llvm_from_source(_COMPREHENSION_DYN_SRC)
    assert count_placeholders(ir) == 0
    assert "indirect closure call" in ir
    assert re.search(r"comprehension via \w+\$lambda\d+", ir)


@needs_clang
def test_native_comprehension_body_calls_dynamic_closure(tmp_path):
    assert_native_matches_interp(_COMPREHENSION_DYN_SRC, tmp_path)


def test_env_captured_closures_marked_heap_env_and_emit():
    # std.stream's take shape: the returned thunk CAPTURES `producer` (a
    # closure pair inside another closure's env) and the handle site
    # captures it again.  Everything emits; iota's lambda is heap-env.
    ir = llvm_from_source(_STD_STREAM_SRC)
    # take$lambda1's env inlines a closure pair field
    assert re.search(
        r"%env\.std_stream_take_lambda1 = type \{[^}]*%mx\.closure", ir)
    # the handle-site env of fold holds the dynamic f pair by value
    assert re.search(r"%henv\.\w*fold\w* = type \{[^}]*%mx\.closure", ir)


@needs_clang
def test_native_apply_twice_two_lambdas_one_site(tmp_path):
    # THE INCREMENT-13 SHAPE: two different lambdas (one capture-carrying,
    # one not) through apply's and twice's single call sites.
    assert_native_matches_interp(_APPLY_TWICE_SRC, tmp_path)


@needs_clang
@needs_asan
def test_native_apply_twice_fully_leak_checked_under_asan(tmp_path):
    # Both lambdas keep stack envs (never returned, never env-captured):
    # the indirect-call machinery allocates nothing — FULL leak check.
    assert_native_matches_interp_asan("""
fn apply(f: fn(int) -> int, v: int) -> int { f(v) }
fn main() -> int {
    let x = 10;
    let g = fn(y: int) -> x + y;
    let h = fn(y: int) -> y * 2;
    print(apply(g, 7));
    print(apply(h, 7));
    0
}
""", tmp_path)


@needs_clang
def test_native_f64_lambda_word_roundtrip(tmp_path):
    # f64 params AND f64 returns through the word ABI: bitcast round trips
    # must be exact for every value.
    assert_native_matches_interp("""
fn apply(f: fn(float) -> float, v: float) -> float { f(v) }
fn main() -> int {
    let a = fn(x: float) -> x * 2.0;
    let b = fn(x: float) -> x + 0.5;
    print(apply(a, 1.25));
    print(apply(b, 1.25));
    print(apply(a, 0.0 - 3.25));
    0
}
""", tmp_path)


@needs_clang
def test_native_merged_closure_variable_dynamic_dispatch(tmp_path):
    # One VARIABLE holding different lambdas on different paths: the
    # stored pair decides at runtime which fn pointer runs.
    assert_native_matches_interp("""
fn choose(c: bool) -> int {
    let mut f = fn(x: int) -> x + 1;
    if c { f = fn(x: int) -> x * 10; } else { () };
    f(4)
}
fn main() -> int {
    print(choose(true));
    print(choose(false));
    0
}
""", tmp_path)


@needs_clang
def test_native_capture_carrying_lambdas_through_indirect_site(tmp_path):
    # Both lambdas CARRY CAPTURES: each pair's env pointer must travel
    # with its fn pointer through the one indirect site.
    assert_native_matches_interp("""
fn apply(f: fn(int) -> int, v: int) -> int { f(v) }
fn main() -> int {
    let a = 100;
    let b = 7;
    let add_a = fn(y: int) -> y + a;
    let mul_b = fn(y: int) -> y * b;
    print(apply(add_a, 1));
    print(apply(mul_b, 3));
    0
}
""", tmp_path)


@needs_clang
def test_native_std_stream_end_to_end(tmp_path):
    # THE INCREMENT-13 TARGET: a std.stream program — import, chain
    # take/sum/count drivers over iota — runs natively end to end, with
    # the handler cases calling f through the word-uniform ABI on top of
    # the effects runtime.
    assert_native_matches_interp(_STD_STREAM_SRC, tmp_path, entry="main")


@needs_clang
def test_native_std_stream_chain_and_transformers(tmp_path):
    # chain merges two producers; map/filter re-emit through their own
    # handler scopes with f/pred called indirectly.
    assert_native_matches_interp("""
from std.stream import Emit, iota, sum, map, filter, chain;
fn main() -> int {
    let doubled = map(iota(10), fn(x: int) -> x * 2);
    let big = filter(doubled, fn(x: int) -> x > 5);
    print(sum(big));
    print(sum(chain(iota(4), iota(3))));
    0
}
""", tmp_path)


@needs_clang
def test_native_std_stream_iter_and_collect(tmp_path):
    # iter drives print through an indirect f; collect pushes into a
    # captured Vec (the shared-vector aliasing contract).
    assert_native_matches_interp("""
from std.stream import Emit, iota, iter, collect;
fn main() -> int {
    iter(iota(4), fn(x) { print(x) });
    let v = collect(iota(5));
    print(len(v));
    print(v[0] + v[4]);
    0
}
""", tmp_path)


@needs_clang
def test_native_std_stream_for_with_break(tmp_path):
    # for_'s nested Loop handler: break_ aborts the per-element scope and
    # stops the loop; the body mutates a captured cell across resumes.
    assert_native_matches_interp("""
from std.stream import Emit, Loop, iota, for_;
fn main() -> int {
    let mut acc = 0;
    for_(iota(10), fn(x: int) {
        if x > 4 { perform Loop.break_() } else { () };
        acc = acc + x
    });
    print(acc);
    0
}
""", tmp_path)


_STD_STREAM_FULL_SRC = """
from std.stream import Emit, Loop, iota, emit_vec, iter, for_, fold, sum,
    product, count, collect, all_of, any_of, find, map, filter, take, skip,
    chain;
fn main() -> int {
    match find(iota(9), fn(x: int) -> x > 6) {
        Some(v) => { print(v) },
        None => { print(0 - 1) }
    };
    match find(iota(4), fn(x: int) -> x > 40) {
        Some(v) => { print(v) },
        None => { print(0 - 1) }
    };
    let s = filter(map(chain(iota(6), take(iota(9), 3)), fn(x: int) -> x * 2),
                   fn(x: int) -> x > 4);
    print(sum(s));
    print(product(take(iota(5), 3)));
    print(count(skip(iota(9), 5)));
    iter(iota(3), fn(x) { print(x) });
    let v = collect(iota(4));
    print(len(v));
    print(sum(emit_vec(v)));
    if all_of(iota(4), fn(x: int) -> x < 9) { print(1) } else { print(0) };
    if any_of(iota(4), fn(x: int) -> x > 2) { print(1) } else { print(0) };
    let mut acc = 0;
    for_(iota(10), fn(x: int) {
        if x > 4 { perform Loop.break_() } else { () };
        acc = acc + x
    });
    print(acc);
    0
}
"""


def test_std_stream_full_surface_census():
    # When a program exercises the whole stdlib surface, EVERY std.stream
    # function emits — increment 14's boundary boxes lifted `find` (its
    # Option handle value crosses the effect boundary as a boxed enum:
    # the body thunk and the Some-arm case malloc the box, the owner
    # copies the Option out of it).  ZERO placeholders.
    ir = llvm_from_source(_STD_STREAM_FULL_SRC)
    for fn in ("iota", "emit_range", "emit_vec", "iter", "for_", "fold",
               "sum", "product", "count", "collect", "all_of", "any_of",
               "find", "map", "filter", "take", "skip", "chain"):
        # (?:_ho\d+)? accepts monomorphize's per-call-site clones of
        # higher-order stdlib functions (dead originals are erased).
        assert re.search(
            rf"^define (?:i64|void|ptr|double) @mx_std_stream_{fn}(?:_ho\d+)?\(",
            ir, re.M), fn
    assert count_placeholders(ir) == 0
    # find's handle value is a boxed Option at the boundary.  The producer
    # side boxes it inside the subfunction now (the boundary-word return
    # ABI, increment 17) rather than in the shim, so the shim just
    # forwards the word...
    assert re.search(
        r"call i64 @mx_std_stream_find\$?\w*\(ptr %env\)"
        r"  ; boundary-word body result enum:Option", ir) or re.search(
        r"; boundary-word (?:body|case) result enum:Option", ir)
    assert re.search(
        r"call ptr @malloc\(i64 \d+\)"
        r"  ; boundary box: enum:Option[^\n]*\(write-once, leaks by design\)",
        ir)
    # ...and the RECEIVER now views that immortal write-once box instead
    # of copying the Option out of it (increment 17, work item 1).
    assert re.search(
        r"; elide-copy: \w+ views the producer's write-once boundary box "
        r"\(enum:Option", ir)


@needs_clang
def test_native_std_stream_full_surface_differential(tmp_path):
    # The capstone: chain/map/filter/take/skip feeding sum/product/count/
    # iter/collect/emit_vec/all_of/any_of/for_, all natively, one binary.
    assert_native_matches_interp(_STD_STREAM_FULL_SRC, tmp_path)


@needs_asan
def test_native_std_stream_asan_no_uaf(tmp_path):
    # Heap closure envs, mutable-capture cells and effect scopes: the
    # leak-by-design contract (detect_leaks=0) proves no use-after-free /
    # no double-free across the whole stream pipeline.
    result, expected_out = interp_run(_STD_STREAM_SRC)
    ir = llvm_from_source(_STD_STREAM_SRC)
    exit_code, stdout = compile_and_run(
        ir, "main", workdir=str(tmp_path),
        clang_args=("-fsanitize=address",),
        run_env={"ASAN_OPTIONS": "detect_leaks=0"})
    assert stdout == expected_out
    assert exit_code == int(result) % 256


@needs_clang
def test_native_dynamic_returned_closure(tmp_path):
    # A function returning ONE OF TWO lambdas: the return kind is the
    # dynamic member set (sret pair copy), both lambdas get heap envs,
    # and the caller's calls dispatch on the runtime fn pointer.
    assert_native_matches_interp("""
fn pick(c: bool) -> fn(int) -> int {
    let base = 100;
    if c { fn(x: int) -> x + base } else { fn(x: int) -> x * 2 }
}
fn main() -> int {
    let f = pick(true);
    let g = pick(false);
    print(f(5));
    print(g(5));
    0
}
""", tmp_path)


@needs_clang
def test_native_fold_shape_direct_two_lambdas(tmp_path):
    # The fold/reduce shape without the stdlib: one higher-order driver,
    # two different 2-arity lambdas, called at one nested site.
    assert_native_matches_interp("""
fn fold3(f: fn(int, int) -> int, a: int, b: int, c: int, init: int) -> int {
    f(c, f(b, f(a, init)))
}
fn main() -> int {
    print(fold3(fn(x: int, acc: int) -> x + acc, 1, 2, 3, 0));
    print(fold3(fn(x: int, acc: int) -> x * acc, 1, 2, 3, 1));
    0
}
""", tmp_path)


# ---------------------------------------------------------------------------
# Increment 14: aggregates across the effect boundary (boundary boxes) —
# perform args / resume values / case results / handle results box into a
# fresh write-once malloc'd copy whose pointer is the boundary word;
# receivers copy out per kind.  Boxes are immortal (leak by design:
# detect_leaks=0 proves no-UAF/no-double-free with aggregates in flight
# across park/resume).  Same-named-op conflicts, konts, rawptr and
# C-runtime-mapped ops (effect_mapping.mx) stay honestly demoted.
# ---------------------------------------------------------------------------

_FX_STRUCT_ROUNDTRIP_SRC = """
struct Point { x: int, y: int }
effect Geo { reflect(p: Point) -> Point }
fn flip() performs Geo -> Point {
    let p = Point { x: 3, y: 9 };
    perform Geo.reflect(p)
}
fn main() -> int {
    let q = handle Geo with {
        reflect(p) -> resume(Point { x: p.y, y: p.x })
    } in {
        flip()
    };
    print(q.x);
    print(q.y);
    0
}
"""


def test_struct_across_effect_boundary_boxes_structurally():
    # The perform arg is boxed at the sender (16-byte Point), the case
    # param arrives as the box pointer (byval-copied by the case fn), the
    # resume value is boxed by the case, and the perform result / handle
    # value copy out of their boxes.  Zero placeholders.
    ir = llvm_from_source(_FX_STRUCT_ROUNDTRIP_SRC)
    assert count_placeholders(ir) == 0
    assert "cannot cross the effect boundary" not in ir
    # sender-side boxes: the perform argument and the resume value
    assert len(re.findall(
        r"call ptr @malloc\(i64 16\)"
        r"  ; boundary box: struct:Point \(write-once, leaks by design\)",
        ir)) >= 2
    # receiver side: the perform result / handle value VIEW the producer's
    # immortal write-once box instead of copying out of it (increment 17,
    # work item 1) — the ABI word is unchanged, only the copy is gone.
    assert re.search(
        r"; elide-copy: \w+ views the producer's write-once boundary box "
        r"\(struct:Point\)", ir)
    # dispatcher hands the case fn the box pointer for its byval-copy
    assert re.search(
        r"inttoptr i64 %c0\.a0w to ptr  ; boundary box: case param "
        r"struct:Point", ir)


@needs_clang
def test_native_struct_perform_arg_and_resume_value(tmp_path):
    # THE INCREMENT-14 ROUNDTRIP: a struct crosses perform -> handler
    # (field reads) -> resume -> performer, through a suspending helper.
    assert_native_matches_interp(_FX_STRUCT_ROUNDTRIP_SRC, tmp_path)


@needs_clang
def test_native_effectful_fn_returns_struct_through_handle_scope(tmp_path):
    # A handle-scope RESULT that is a struct: the body returns it (sret
    # into the body thunk's box), an abort-style case can also produce it
    # (sret into the dispatcher's box), and the owner copies it out of
    # whichever box won.  Both completion paths are exercised.
    assert_native_matches_interp("""
struct Acc { total: int, stopped: int }
effect Tick { tick(n: int) -> int }
fn run(limit: int) -> Acc {
    handle Tick with {
        tick(n) -> {
            if n > limit { Acc { total: 0 - n, stopped: 1 } }
            else { resume(n * 10) }
        }
    } in {
        let a = perform Tick.tick(1);
        let b = perform Tick.tick(2);
        let c = perform Tick.tick(3);
        Acc { total: a + b + c, stopped: 0 }
    }
}
fn main() -> int {
    let done = run(5);
    print(done.total);
    print(done.stopped);
    let cut = run(2);
    print(cut.total);
    print(cut.stopped);
    0
}
""", tmp_path)


@needs_clang
def test_native_enum_option_across_boundary_both_paths(tmp_path):
    # An enum as the handle value without the stdlib: the Hit arm
    # (case-result box) and the Miss arm (body-result box) both cross.
    assert_native_matches_interp("""
enum Found { Hit(int), Miss }
effect Probe { probe(x: int) -> Unit }
fn scan(stop: int) -> Found {
    handle Probe with {
        probe(x) -> {
            if x == stop { Hit(x * 100) } else { resume(()) }
        }
    } in {
        perform Probe.probe(1);
        perform Probe.probe(2);
        perform Probe.probe(3);
        Miss
    }
}
fn main() -> int {
    match scan(2) { Hit(v) => { print(v) }, Miss => { print(0 - 1) } };
    match scan(9) { Hit(v) => { print(v) }, Miss => { print(0 - 1) } };
    0
}
""", tmp_path)


@needs_clang
def test_native_struct_resume_result_chains_fold_style(tmp_path):
    # Aggregate RESUME RESULT under deep semantics: resume's value is the
    # WHOLE delimited body's completion (a struct box), which each case
    # copies out, extends and re-boxes through its own sret — the
    # std.stream fold shape with a struct accumulator.
    assert_native_matches_interp("""
struct Acc { v: int }
effect Emit2 { emit2(x: int) -> Unit }
fn main() -> int {
    let r = handle Emit2 with {
        emit2(x) -> {
            let rest = resume(());
            Acc { v: rest.v + x }
        }
    } in {
        perform Emit2.emit2(5);
        perform Emit2.emit2(7);
        Acc { v: 100 }
    };
    print(r.v);
    0
}
""", tmp_path)


@needs_clang
def test_native_closure_as_perform_argument(tmp_path):
    # The flipped increment-7 demotion: a closure crossing as a perform
    # argument (boxed {fn, env} pair, member forced heap-env), applied by
    # the handler and its result resumed back.
    assert_native_matches_interp(_FX_CLOSURE_ARG_SRC, tmp_path)


_FX_STD_FIND_SRC = """
from std.stream import Emit, iota, find;
fn main() -> int {
    let r = find(iota(10), fn(x: int) -> x > 6);
    match r { Some(v) => { print(v) }, None => { print(0 - 1) } };
    let r2 = find(iota(5), fn(x: int) -> x > 40);
    match r2 { Some(v) => { print(v) }, None => { print(0 - 1) } };
    0
}
"""


@needs_clang
def test_native_std_stream_find_both_paths(tmp_path):
    # THE INCREMENT-14 TARGET: std.stream's find — an Option-valued
    # handle result over a real stream — natively, Some AND None paths,
    # matched against the interpreter.
    ir = assert_native_matches_interp(_FX_STD_FIND_SRC, tmp_path)
    assert re.search(r"^define \w+ @mx_std_stream_find(?:_ho\d+)?\(",
                     ir, re.M)


@needs_asan
def test_native_aggregates_across_boundary_asan_no_uaf(tmp_path):
    # Boundary boxes leak by design (immortal, write-once), so
    # detect_leaks=0; ASan proves no use-after-free / no double-free with
    # structs, Options and closure pairs in flight across park/resume —
    # including a double perform roundtrip reusing a received aggregate.
    src = """
from std.stream import Emit, iota, find;
struct Pair { a: int, b: int }
effect Swap { swap(p: Pair) -> Pair }
fn main() -> int {
    match find(iota(10), fn(x: int) -> x > 6) {
        Some(v) => { print(v) }, None => { print(0 - 1) }
    };
    match find(iota(5), fn(x: int) -> x > 40) {
        Some(v) => { print(v) }, None => { print(0 - 1) }
    };
    let q = handle Swap with {
        swap(p) -> resume(Pair { a: p.b, b: p.a })
    } in {
        let s = perform Swap.swap(Pair { a: 1, b: 2 });
        let t = perform Swap.swap(s);
        t
    };
    print(q.a);
    print(q.b);
    0
}
"""
    result, expected_out = interp_run(src)
    ir = llvm_from_source(src)
    exit_code, stdout = compile_and_run(
        ir, "main", workdir=str(tmp_path),
        clang_args=("-fsanitize=address",),
        run_env={"ASAN_OPTIONS": "detect_leaks=0"})
    assert stdout == expected_out
    assert exit_code == int(result) % 256


def test_conflicting_aggregate_kinds_across_same_named_ops_demote():
    # The coarseness rule is unchanged: op cells are keyed by op NAME
    # (routing is dynamic), so two same-named ops carrying different
    # struct kinds join to conflict and demote — for aggregates exactly
    # as for scalars.
    ir = llvm_from_source("""
struct A { x: int }
struct B { y: float }
effect E1 { get() -> A }
effect E2 { get() -> B }
fn main() -> int {
    let a = handle E1 with { get() -> resume(A { x: 1 }) }
            in { perform E1.get() };
    let b = handle E2 with { get() -> resume(B { y: 2.0 }) }
            in { perform E2.get() };
    a.x
}
""")
    assert count_placeholders(ir) >= 1
    assert "effect op 'get' has conflicting result kinds" in ir


def test_effect_mapping_runtime_mapped_ops_emit_thread_primitives():
    # effect_mapping.mx is FULLY native now (docs/threads_runtime.md):
    # its `with EFFECT_*` ops lower through the __effect_runtime$E$op
    # thunks to the pthreads-backed metaxu_threads.c primitives.  No
    # handle scope in the module lists the ops, so every perform is a
    # DIRECT call to its thunk (no boundary, no mx_perform).
    ir = llvm_from_source(
        (REPO_ROOT / "examples" / "effect_mapping.mx").read_text())
    assert count_placeholders(ir) == 0
    assert "call i64 @mx_thread_spawn(ptr" in ir
    assert "call i64 @mx_thread_join(i64" in ir
    assert "call i64 @mx_mutex_create()" in ir
    assert "call i64 @mx_mutex_lock(i64" in ir
    assert "call i64 @mx_mutex_unlock(i64" in ir
    assert "@mx_perform(" not in ir  # all five performs statically routed
    # The spawned lambda is invoked from the C child thread: it must be
    # word-uniform (i64 (ptr env)) with a HEAP env (the child runs after
    # the spawning frame moved on).
    assert re.search(r"define i64 @mx_main_lambda\d+\(ptr %cl\.env\)", ir)
    assert re.search(r"heap env for \w+ -> main\$lambda\d+", ir)


# ---------------------------------------------------------------------------
# examples/collections.mx: the fixed-capacity List, natively
# ---------------------------------------------------------------------------
# The example's `push` used to demote with the misleading "struct 'List' has
# no field 'data'": `struct List<T>` never declared the vector size `N`, and
# the `vector[T,N]` written in value position was dropped by HIR, so
# alloc_struct — the only thing codegen's struct table is built from — listed
# `len` alone.  With `const N: int` declared and the value spelled
# `vector[T,N](...)`, List is a two-field native struct and push emits.
# (`len` is renamed `n` and `push` `push_item` only because the interpreter
# resolves the BUILTIN `push` first; the lowering under test is identical.)

_COLLECTIONS_PUSH_SRC = """
struct List<T, const N: int> {
    data: vector[T,N]
    len: Int
}

fn push_item<T, const N: int>(list: @mut List[T,N], item: T) {
    list.data[list.len] = item
    list.len = list.len + 1
}

fn main() {
    let @mut l = List { data: vector[int,4](0, 0, 0, 0), len: 0 }
    push_item(l, 7)
    push_item(l, 9)
    print(l.data[0])
    print(l.data[1])
    print(l.len)
}
"""


def test_collections_push_emits_with_both_fields():
    ir = llvm_from_source(_COLLECTIONS_PUSH_SRC)
    assert "%struct.List = type { ptr, i64 }  ; data, len" in ir
    assert "define void @mx_push_item(" in ir
    assert "has no field" not in ir
    assert count_placeholders(ir) == 0


@needs_clang
def test_native_collections_push_differential(tmp_path):
    assert_native_matches_interp(_COLLECTIONS_PUSH_SRC, tmp_path)


# ---------------------------------------------------------------------------
# HIR triage round: constructs that previously had NO front-end lowering at
# all (they vanished silently) now reach the native backend. Each is pinned
# differentially — native stdout/exit must equal the interpreter — so a
# construct that is newly *present* cannot be newly *wrong*.
# ---------------------------------------------------------------------------

_TRIAGE_INDIRECT_CALL_SRC = """
fn main() -> int {
    let n = (fn(x: int) -> int { x * 2 })(3);
    print(n);
    n
}
"""


@needs_clang
def test_native_indirect_call_of_lambda_literal_differential(tmp_path):
    """`(fn(x) -> x*2)(3)` is a CallExpression (computed callee); the whole
    call used to vanish in HIR and the expression evaluated to unit."""
    assert_native_matches_interp(_TRIAGE_INDIRECT_CALL_SRC, tmp_path)


_TRIAGE_ADDRESS_OF_SRC = """
struct P { a: int, b: int }

fn main() -> int {
    let p = P { a: 7, b: 9 };
    let q = &p.a;
    print(q);
    q
}
"""


@needs_clang
def test_native_address_of_field_differential(tmp_path):
    """`&x.f` builds an AddressOf (only the bare-name form builds a Borrow*);
    it had no lowering, so the expression silently became nothing."""
    assert_native_matches_interp(_TRIAGE_ADDRESS_OF_SRC, tmp_path)


_TRIAGE_BORROW_EXPR_SRC = """
fn main() -> int {
    let x = 5;
    let y = borrow x;
    print(y);
    y
}
"""


@needs_clang
def test_native_borrow_expression_differential(tmp_path):
    """`borrow x` had no lowering: the binding was dropped entirely."""
    assert_native_matches_interp(_TRIAGE_BORROW_EXPR_SRC, tmp_path)


_TRIAGE_DOTTED_VARIANT_SRC = """
enum Color { Red, Green, Blue }

fn main() -> int {
    let c = Color.Blue;
    let n = match c {
        Color.Red => 1,
        Color.Green => 2,
        Color.Blue => 3
    };
    print(n);
    n
}
"""


@needs_clang
def test_native_dotted_enum_variant_differential(tmp_path):
    """`Color.Blue` in expression position used to build a field read of an
    undefined variable; in pattern position it degraded to a wildcard, so the
    FIRST arm matched every colour."""
    assert_native_matches_interp(_TRIAGE_DOTTED_VARIANT_SRC, tmp_path)


# ---------------------------------------------------------------------------
# Increment 15: dynamic effect-default routing (mx_perform_or_default)
#
# An op with BOTH a declared `= expr` default and a handle scope listing it
# resolves per-perform, from the runtime scope stack.  The interpreter's
# precedence (mir_interp, `perform`) is:
#
#     in-scope non-busy handler frame
#   > `with SYMBOL` runtime mapping (__effect_runtime$E$op)
#   > declared `= expr` default (__effect_default$E$op)
#   > loud "No handler for effect" error
#
# Native mirrors rungs 1, 3 and 4 exactly (mx__perform_impl); rung 2 has no
# native implementation at all, so any op declaring a runtime mapping still
# demotes with that reason — divergence is refused, not papered over.
# ---------------------------------------------------------------------------

def test_dynamic_default_thunk_boxes_aggregate_results():
    # The default's return is an ENUM: it crosses the boundary as a
    # write-once box pointer, exactly like a handler case's aggregate
    # result (the thunk mallocs, sret-fills, returns the pointer word).
    ir = llvm_from_source(_DYNAMIC_DEFAULT_AGG_SRC)
    assert count_placeholders(ir) == 0
    assert re.search(
        r"define internal i64 @mxfx\.dflt\.__effect_default_Cap_probe"
        r"\(ptr %env, ptr %args\) \{", ir)
    assert re.search(
        r"%rbox = call ptr @malloc\(i64 \d+\)"
        r"  ; boundary box: default result enum:Answer", ir)
    assert re.search(
        r"call void @mx___effect_default_Cap_probe\(ptr %rbox, i64 %a0w\)",
        ir)


def test_dynamic_and_static_default_routes_coexist_in_one_module():
    # Only `base` is listed by a handle scope, so ONLY `base` routes
    # dynamically; `ask` (a default with no scope anywhere) keeps the
    # increment-10 static direct call.  The two rungs must not blur.
    ir = llvm_from_source(_DYNAMIC_DEFAULT_PERFORMS_SRC)
    assert count_placeholders(ir) == 0
    assert ("; perform Cap.ask -> declared default (no handle scope in the "
            "module lists it)") in ir
    assert "call i64 @mx___effect_default_Cap_ask(i64 %a.x)" in ir
    assert "mxfx.dflt.__effect_default_Cap_ask" not in ir
    # `ask`'s default body performs `base`, which IS scoped: one thunk, one
    # dynamic call site, inside the default function itself (defaults run
    # on the performing stack, so a perform inside one parks this fiber).
    assert ir.count(
        "define internal i64 @mxfx.dflt.__effect_default_Cap_base(") == 1
    assert ir.count("call i64 @mx_perform_or_default(") == 1
    assert re.search(
        r"define i64 @mx___effect_default_Cap_ask\(i64 %a\.x\) \{"
        r"(?:(?!\n\}).)*mx_perform_or_default", ir, re.S)


def test_dynamic_default_disagreeing_with_its_handler_demotes():
    # The default's signature joins the SAME op-name cells as the handler
    # cases, so a default answering int where the handler resumes a string
    # cannot be reconciled.  The result must be an honest demotion, never a
    # thunk decoding boundary words the wrong way.
    ir = llvm_from_source("""
effect Cap {
    ask(x: int) -> int = 7;
}

fn inner() -> int performs Cap {
    perform Cap.ask(1)
}

fn main() -> int {
    let a = handle Cap with { ask(x) -> resume("nope") } in { inner() };
    let b = inner();
    print(a);
    print(b);
    0
}
""")
    assert count_placeholders(ir) > 0
    # the default was pulled to the op cell's str kind and could not hold
    # its own `7`, so IT demotes and every performer cascades with it
    assert ("; function @mx___effect_default_Cap_ask: placeholder" in ir)
    assert ("calls function '__effect_default$Cap$ask' that is itself a "
            "placeholder") in ir
    # no thunk survives for an op whose routes disagree
    assert "define internal i64 @mxfx.dflt." not in ir


_SCOPED_MAPPED_OP_SRC = """
effect Locky = {
    fn create() -> int with EFFECT_MUTEX_CREATE
}

fn grab() -> int performs Locky {
    perform Locky.create()
}

fn main() -> int {
    let a = handle Locky with { create() -> resume(5) } in { grab() };
    print(a);
    0
}
"""


def test_runtime_mapped_op_in_a_scope_routes_dynamically():
    # The interpreter's precedence puts the `with SYMBOL` runtime mapping
    # BETWEEN the handler frames and the default.  A scope in this module
    # lists the op, so routing is a RUNTIME decision: the perform lowers
    # to mx_perform_or_default with the op's __effect_runtime$E$op thunk
    # as the fallback — a handler in scope wins, and only otherwise does
    # the mapping's primitive run (docs/threads_runtime.md).
    ir = llvm_from_source(_SCOPED_MAPPED_OP_SRC)
    assert count_placeholders(ir) == 0
    assert ir.count("call i64 @mx_perform_or_default(") == 1
    assert "@mxfx.dflt.__effect_runtime_Locky_create" in ir


@needs_clang
def test_native_scoped_mapped_op_handler_wins(tmp_path):
    """Virtualization preserved natively: the in-scope handler intercepts
    the mapped op (prints 5, never creates a real mutex) — differential
    against the interpreter."""
    assert_native_matches_interp(_SCOPED_MAPPED_OP_SRC, tmp_path)


@needs_clang
def test_native_dynamic_default_taken_when_unhandled(tmp_path):
    """(a) No handler installed at the perform -> the declared default
    answers, and (b) a handler installed elsewhere in the same program does
    not change that."""
    assert_native_matches_interp(_DYNAMIC_DEFAULT_ONLY_SRC, tmp_path)


@needs_clang
def test_native_dynamic_default_handler_wins_when_installed(tmp_path):
    """(b)+(c) ONE perform site, two answers: 2 inside the handle, 7
    outside it — the routing decision the whole increment exists for."""
    ir = assert_native_matches_interp(_DYNAMIC_DEFAULT_SRC, tmp_path)
    assert ir.count("call i64 @mx_perform_or_default(") == 1


@needs_clang
def test_native_dynamic_default_busy_scope_falls_through(tmp_path):
    """(c) The handler case's own perform routes outward past its BUSY
    scope and lands on the default — the interpreter's busy skip
    (_find_mir_frame) and the runtime's (mx__find_scope) agreeing."""
    assert_native_matches_interp(_DYNAMIC_DEFAULT_BUSY_SRC, tmp_path)


@needs_clang
def test_native_dynamic_default_result_flows_onward(tmp_path):
    """(d) The default's value flows into subsequent computation, across an
    aggregate boundary box and back out through a match."""
    assert_native_matches_interp(_DYNAMIC_DEFAULT_AGG_SRC, tmp_path)


@needs_clang
def test_native_dynamic_default_nested_scopes_differential(tmp_path):
    """(c)+(d) at scale: two dynamically-routed ops, three nested scopes,
    each perform walking outward past scopes that handle the OTHER op
    before either a handler or a default answers — with str results
    crossing the boundary and flowing into a concat."""
    ir = assert_native_matches_interp(_DYNAMIC_DEFAULT_NESTED_SRC, tmp_path)
    assert count_placeholders(ir) == 0
    assert ir.count("call i64 @mx_perform_or_default(") == 2
    assert ("define internal i64 @mxfx.dflt.__effect_default_Cfg_level"
            "(ptr %env, ptr %args) {") in ir
    assert ("define internal i64 @mxfx.dflt.__effect_default_Cfg_label"
            "(ptr %env, ptr %args) {") in ir
    # the str default's result crosses as a pointer word on both routes
    assert "inttoptr i64 %r to i64" not in ir
    assert "%w = ptrtoint ptr %r to i64" in ir


@needs_clang
def test_native_dynamic_default_performs_from_the_default(tmp_path):
    """The default runs ON THE PERFORMING STACK: its own perform sees the
    same scope stack and, when a scope answers, parks this same fiber."""
    assert_native_matches_interp(_DYNAMIC_DEFAULT_PERFORMS_SRC, tmp_path)


@needs_asan
def test_native_dynamic_default_asan_leak_clean(tmp_path):
    """Scalar-only dynamic-default programs are FULLY leak-checked: the
    default path allocates nothing at all (no scope, no coroutine, no
    continuation), and the scope path frees its machinery as before."""
    assert_native_matches_interp_asan(_DYNAMIC_DEFAULT_BUSY_SRC, tmp_path)
    assert_native_matches_interp_asan(_DYNAMIC_DEFAULT_PERFORMS_SRC, tmp_path)


@needs_asan
def test_native_dynamic_default_aggregate_asan_no_uaf(tmp_path):
    """Boundary boxes leak BY DESIGN (immortal, write-once) on both routes,
    so this one proves no use-after-free / no double-free only — the
    documented aggregate contract."""
    import os
    env_key = "ASAN_OPTIONS"
    old = os.environ.get(env_key)
    os.environ[env_key] = "detect_leaks=0"
    try:
        assert_native_matches_interp(
            _DYNAMIC_DEFAULT_AGG_SRC, tmp_path,
            clang_args=("-fsanitize=address",))
    finally:
        if old is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = old


# ---------------------------------------------------------------------------
# Increment 16: aggregates in indirect closure-call signatures.
#
# The word-uniform indirect ABI (`i64 (ptr env, i64 args...)`, increment 13)
# was scalar-only.  It now carries struct/enum aggregates through the SAME
# write-once boundary boxes increment 14 introduced at the effect boundary:
#   * an aggregate ARGUMENT at an indirect site mallocs a fresh copy and
#     travels as its pointer word; the callee's prelude decodes the word to
#     a ptr and the ordinary aggregate-param convention (byval copy-out, or
#     the elide-copy read-through) is exactly the copy-out contract;
#   * an aggregate RETURN drops sret: the word-uniform lambda mallocs a box,
#     copies its result in and rets the pointer word; the caller copies out
#     into its own storage.
# Boxes are immortal (leak by design), so nothing can dangle across the
# call.  Fast paths are untouched: a lambda pinned at every site keeps its
# TYPED signature (ptr params / sret) and allocates nothing, and scalar
# indirect calls still pass raw words.  Still demoted honestly: closure
# pairs and konts in indirect args/returns, kinds that conflict across the
# member set, aggregates with an infinite layout, and @mut aggregate params
# (their write-back cannot travel back through a box the caller drops).
# ---------------------------------------------------------------------------

_IND_AGG_ROUNDTRIP_SRC = """
struct Point { x: int, y: int }
fn apply(f: fn(Point) -> Point, p: Point) -> Point {
    f(p)
}
fn main() -> int {
    let swap = fn(q: Point) -> Point { Point { x: q.y, y: q.x } };
    let bump = fn(q: Point) -> Point { Point { x: q.x + 10, y: q.y + 20 } };
    let a = apply(swap, Point { x: 1, y: 2 });
    let b = apply(bump, Point { x: 3, y: 4 });
    print(a.x, a.y, b.x, b.y);
    0
}
"""
# ^ Lambdas via variables on purpose: this pins the SHARED-site word-uniform
# ABI (two lambdas joined through one apply). Literal arguments would be
# split into per-site clones by monomorphize and each would pin instead.


def test_indirect_aggregate_param_and_return_box_at_the_site():
    # The target shape: two different lambdas through one indirect site,
    # a struct in AND a struct out.
    ir = llvm_from_source(_IND_AGG_ROUNDTRIP_SRC)
    assert count_placeholders(ir) == 0
    # Both lambdas take the word-uniform ABI: i64 word in, i64 word out,
    # NO sret result slot.
    defs = re.findall(
        r"^define i64 @(mx_main_lambda\d+)\(ptr %cl\.env, i64 %aw\.q\) \{"
        r"  ; word-uniform lambda ABI$", ir, re.M)
    assert len(defs) == 2, ir
    # the lambdas' bodies carry no sret result slot at all
    for body in re.findall(r"^define i64 @mx_main_lambda\d+.*?^\}",
                           ir, re.M | re.S):
        assert "%agg.ret" not in body
    # Caller side (increment 17): the ARGUMENT no longer boxes — it passes
    # our storage pointer, since no member writes through it and the call
    # is synchronous.  The RESULT box (the callee's ret) is unavoidable
    # (the callee's storage dies), but the caller views it instead of
    # copying it out.
    assert re.search(
        r"ptrtoint ptr %[\w.]+ to i64  ; elide-box: read-only aggregate "
        r"argument \w+ \(struct:Point\)", ir)
    # apply() itself allocates nothing: its argument box is gone
    apply_body = re.search(r"^define void @mx_apply\(.*?^\}", ir,
                           re.M | re.S).group(0)
    assert "@malloc" not in apply_body, apply_body
    assert re.search(
        r"call i64 %\w+\(ptr %\w+, i64 %\w+\)"
        r"  ; indirect closure call \(main\$lambda\d+\|main\$lambda\d+\), "
        r"word-uniform ABI", ir)
    assert re.search(
        r"; elide-copy: \w+ views the producer's write-once boundary box "
        r"\(struct:Point\)", ir)
    # exactly one malloc per lambda remains: the returned box
    assert len(re.findall(
        r"call ptr @malloc\(i64 16\)"
        r"  ; boundary box: struct:Point \(write-once, leaks by design\)",
        ir)) == 2
    # Callee side: the param word decodes to a ptr, the return value boxes.
    assert len(re.findall(
        r"; word-uniform param q: boundary box struct:Point", ir)) == 2
    assert len(re.findall(
        r"ret i64 %\w+  ; word-uniform lambda return \(struct:Point encoded\)",
        ir)) == 2


_IND_ENUM_RETURN_SRC = """
enum Opt { Some(int), None }
fn pick(f: fn(int) -> Opt, x: int) -> int {
    match f(x) {
        Opt.Some(v) => v,
        Opt.None => 0 - 1
    }
}
fn main() -> int {
    let g = fn(x: int) -> Opt { if x > 0 { Opt.Some(x * 2) } else { Opt.None } };
    let h = fn(x: int) -> Opt { Opt.Some(x + 100) };
    print(pick(g, 5));
    print(pick(g, 0 - 3));
    print(pick(h, 5));
    0
}
"""


def test_indirect_enum_return_boxes():
    # A lambda returning an ENUM through an indirect call: the tagged
    # union boxes on the way out (the callee's storage dies, so this box
    # is real) and the caller VIEWS the box instead of copying it out
    # (increment 17, work item 1).
    ir = llvm_from_source(_IND_ENUM_RETURN_SRC)
    assert count_placeholders(ir) == 0
    assert re.search(
        r"call ptr @malloc\(i64 \d+\)"
        r"  ; boundary box: enum:Opt\{[^}]*\} \(write-once, leaks by design\)",
        ir)
    assert re.search(
        r"ret i64 %\w+  ; word-uniform lambda return \(enum:Opt", ir)
    assert re.search(
        r"; elide-copy: \w+ views the producer's write-once boundary box "
        r"\(enum:Opt", ir)


_IND_AGG_CAPTURE_SRC = """
struct Pt { x: int, y: int }
fn run(f: fn(Pt) -> Pt, p: Pt) -> Pt { f(p) }
fn main() -> int {
    let base = Pt { x: 100, y: 200 };
    let add = fn(q: Pt) -> Pt { Pt { x: q.x + base.x, y: q.y + base.y } };
    let neg = fn(q: Pt) -> Pt { Pt { x: 0 - q.x, y: 0 - q.y } };
    let a = run(add, Pt { x: 1, y: 2 });
    let b = run(neg, Pt { x: 1, y: 2 });
    print(a.x, a.y, b.x, b.y);
    0
}
"""


def test_indirect_closure_capturing_and_taking_an_aggregate():
    # A closure that CAPTURES an aggregate (env field, inline copy) and
    # TAKES one through the word ABI (boundary box) in the same lambda.
    ir = llvm_from_source(_IND_AGG_CAPTURE_SRC)
    assert count_placeholders(ir) == 0
    # the capture is an inline struct field of the env, not a box
    assert re.search(r"%env\.main_lambda\d+ = type \{ %struct\.Pt \}", ir)
    assert re.search(r"; word-uniform param q: boundary box struct:Pt", ir)


def test_scalar_indirect_call_allocates_nothing():
    # Work item 2: a purely scalar indirect site still passes raw words —
    # no malloc, no boundary box anywhere in the module.
    ir = llvm_from_source("""
fn twice(f: fn(int) -> int, x: int) -> int { f(f(x)) }
fn main() -> int {
    print(twice(fn(v: int) -> v + 1, 3));
    print(twice(fn(v: int) -> v * 3, 3));
    0
}
""")
    assert count_placeholders(ir) == 0
    assert "indirect closure call" in ir
    assert "call ptr @malloc(" not in ir
    assert "; boundary box:" not in ir


def test_pinned_aggregate_lambda_keeps_its_typed_signature():
    # Work item 2: ONE lambda through the site stays pinned, so it keeps
    # the typed closure-call path (ptr param + sret return) and allocates
    # nothing — the increment-13 aggregate fast path is not regressed.
    ir = llvm_from_source("""
struct Point { x: int, y: int }
fn apply(f: fn(Point) -> Point, p: Point) -> Point { f(p) }
fn main() -> int {
    let a = apply(fn(q: Point) -> Point { Point { x: q.y, y: q.x } },
                  Point { x: 1, y: 2 });
    print(a.x, a.y);
    0
}
""")
    assert count_placeholders(ir) == 0
    assert "call ptr @malloc(" not in ir
    assert "; boundary box:" not in ir
    assert "word-uniform lambda ABI" not in ir
    assert "word-uniform param" not in ir
    assert re.search(
        r"^define void @mx_main_lambda\d+\(ptr %agg\.ret, ptr %cl\.env, "
        r"ptr %a\.q\)", ir, re.M)


def test_closure_pair_in_indirect_args_still_demotes():
    # Kind agreement / honest demotion: a closure PAIR travelling through
    # an indirect site has no boundary box (the boxed env pointer's
    # lifetime cannot be vouched for), so the site demotes with a reason.
    ir = llvm_from_source("""
fn drive(h, k) -> int { h(k, 4) }
fn main() -> int {
    let twice = fn(g, x) -> g(g(x));
    let thrice = fn(g, x) -> g(g(g(x)));
    let inc = fn(v: int) -> v + 1;
    let dbl = fn(v: int) -> v * 2;
    print(drive(twice, inc));
    print(drive(thrice, dbl));
    0
}
""")
    # (Every lambda routed through a variable on purpose: a literal at the
    # call site would trigger monomorphize's per-site cloning, each clone
    # would pin its pair, and the shared-site demotion under test would
    # never happen — that improved case is pinned separately below.)
    assert count_placeholders(ir) >= 1
    assert re.search(
        r"reason: indirect closure call through 'h': arg 'k' of kind "
        r"closure:\*\{.*\} has no word encoding and no boundary box", ir)


def test_indirect_members_disagreeing_on_aggregate_kind_demote():
    # Work item 3: one member says the position is a STRUCT, the other an
    # ENUM.  The two-way fixpoint conflicts the kind and the site demotes
    # rather than guessing a representation.
    ir = llvm_from_source("""
struct P { a: int }
enum E { X(int), Y }
fn apply(f, v) { f(v) }
fn main() -> int {
    let fp = fn(p: P) -> p.a + 1;
    let fe = fn(e: E) -> match e { E.X(n) => n * 2, E.Y => 0 };
    print(apply(fp, P { a: 4 }));
    print(apply(fe, E.X(5)));
    0
}
""")
    # (Lambdas via variables on purpose — see the note on the closure-pair
    # test above; the split-clone version of this program emits cleanly and
    # is pinned separately below.)
    assert count_placeholders(ir) >= 1
    assert "reason: irreconcilable value kinds for 'v'" in ir
    assert re.search(
        r"reason: indirect closure call through 'f': arg 'v' of kind "
        r"conflict has no word encoding and no boundary box", ir)


def test_mut_aggregate_param_through_indirect_call_demotes():
    # A @mut aggregate param has WRITE-BACK semantics (the callee copies
    # it out through the caller's pointer).  A boundary box is dropped by
    # the caller, so the write-back would be lost: demote, never emit a
    # silently non-writing call.
    ir = llvm_from_source("""
struct P { a: int }
fn apply(f: fn(@mut P) -> int, @mut p: P) -> int { f(p) }
fn main() -> int {
    let mut p = P { a: 1 };
    let mut q = P { a: 2 };
    let fa = fn(@mut r: P) -> { r.a = r.a + 10; r.a };
    let fm = fn(@mut r: P) -> { r.a = r.a * 10; r.a };
    print(apply(fa, p));
    print(apply(fm, q));
    print(p.a, q.a);
    0
}
""")
    # (Lambdas via variables on purpose: literals would be split into
    # per-site clones, each of which pins its lambda and CAN write back —
    # that improved case is pinned separately below.)
    assert count_placeholders(ir) >= 1
    assert re.search(
        r"reason: indirect closure call through 'f': lambda 'main\$lambda\d+' "
        r"is not word-uniform \(its @mut parameter 'r' of kind struct:P "
        r"writes back through the caller's pointer, which the indirect ABI's "
        r"boundary box cannot carry back\)", ir)


@needs_clang
def test_native_indirect_aggregate_roundtrip(tmp_path):
    assert_native_matches_interp(_IND_AGG_ROUNDTRIP_SRC, tmp_path)


@needs_clang
def test_native_indirect_enum_return(tmp_path):
    assert_native_matches_interp(_IND_ENUM_RETURN_SRC, tmp_path)


@needs_clang
def test_native_indirect_aggregate_capture_and_param(tmp_path):
    assert_native_matches_interp(_IND_AGG_CAPTURE_SRC, tmp_path)


_IND_AGG_LOOP_SRC = """
struct Acc { lo: int, hi: int }
fn step(f: fn(Acc, int) -> Acc, a: Acc, n: int) -> Acc {
    let @mut acc = a;
    let @mut i = 0;
    while i < n {
        acc = f(acc, i);
        i = i + 1
    };
    acc
}
fn main() -> int {
    let up = fn(a: Acc, i: int) -> Acc { Acc { lo: a.lo + i, hi: a.hi } };
    let dn = fn(a: Acc, i: int) -> Acc { Acc { lo: a.lo, hi: a.hi - i } };
    let x = step(up, Acc { lo: 0, hi: 100 }, 5);
    let y = step(dn, Acc { lo: 0, hi: 100 }, 5);
    print(x.lo, x.hi, y.lo, y.hi);
    0
}
"""


@needs_clang
def test_native_indirect_aggregate_in_a_loop(tmp_path):
    # The boxes are per-call and immortal: a loop over an indirect
    # aggregate site allocates a fresh box every iteration and each
    # callee copy-out sees exactly its own.
    assert_native_matches_interp(_IND_AGG_LOOP_SRC, tmp_path)


@needs_asan
def test_native_indirect_aggregate_asan_no_uaf(tmp_path):
    # Indirect-call boundary boxes leak BY DESIGN (immortal, write-once),
    # exactly like the effect-boundary boxes: detect_leaks=0, so exit 0
    # proves no use-after-free and no double-free only.
    assert_native_matches_interp_asan_boxes(_IND_AGG_ROUNDTRIP_SRC, tmp_path)
    assert_native_matches_interp_asan_boxes(_IND_AGG_CAPTURE_SRC, tmp_path)


@needs_asan
def test_native_pinned_aggregate_closure_call_asan_leak_clean(tmp_path):
    # The pinned (typed) aggregate closure path allocates nothing at all,
    # so it stays under the FULL leak check.
    assert_native_matches_interp_asan("""
struct Point { x: int, y: int }
fn apply(f: fn(Point) -> Point, p: Point) -> Point { f(p) }
fn main() -> int {
    let a = apply(fn(q: Point) -> Point { Point { x: q.y, y: q.x } },
                  Point { x: 1, y: 2 });
    print(a.x, a.y);
    0
}
""", tmp_path)


# --- the real-world payoff: std.stream over an aggregate element type -------

_STREAM_STRUCT_ELEM_SRC = """
from std.stream import Emit, map, filter;

struct Item { id: int, w: int }

fn items() -> fn() -> Unit {
    fn() {
        perform Emit.emit(Item { id: 1, w: 10 });
        perform Emit.emit(Item { id: 2, w: 20 });
        perform Emit.emit(Item { id: 3, w: 30 });
        ()
    }
}

fn total_w(s: fn() -> Unit) -> int {
    handle Emit with {
        emit(it) -> it.w + resume(())
    } in {
        s();
        0
    }
}

fn main() -> int {
    # Lambdas via variables on purpose: this pins the SHARED-site
    # word-uniform aggregate ABI through one `map`; literal arguments
    # would be split into per-site clones by monomorphize.
    let keep_heavy = fn(it: Item) -> bool { it.w > 10 };
    let bump = fn(it: Item) -> Item { Item { id: it.id + 100, w: it.w * 2 } };
    let nudge = fn(it: Item) -> Item { Item { id: it.id, w: it.w + 1 } };
    let heavy = filter(items(), keep_heavy);
    let bumped = map(heavy, bump);
    print(total_w(bumped));
    print(total_w(map(items(), nudge)));
    0
}
"""

_STREAM_ENUM_ELEM_SRC = """
from std.stream import Emit, map, filter;

enum Cell { Val(int), Empty }

fn cells() -> fn() -> Unit {
    fn() {
        perform Emit.emit(Cell.Val(1));
        perform Emit.emit(Cell.Empty);
        perform Emit.emit(Cell.Val(3));
        ()
    }
}

fn total(s: fn() -> Unit) -> int {
    handle Emit with {
        emit(c) -> match c {
            Cell.Val(n) => n + resume(()),
            Cell.Empty => resume(())
        }
    } in {
        s();
        0
    }
}

fn main() -> int {
    # Lambdas via variables on purpose (see the struct-element source).
    let tenfold = fn(c: Cell) -> Cell {
        match c { Cell.Val(n) => Cell.Val(n * 10), Cell.Empty => Cell.Empty }
    };
    let is_val = fn(c: Cell) -> bool {
        match c { Cell.Val(n) => true, Cell.Empty => false }
    };
    let ident = fn(c: Cell) -> Cell { c };
    let doubled = map(cells(), tenfold);
    let nonempty = filter(doubled, is_val);
    print(total(nonempty));
    print(total(map(cells(), ident)));
    0
}
"""


def test_std_stream_aggregate_element_census():
    # THE PAYOFF.  std.stream's map/filter over a STRUCT (and an ENUM)
    # element type: the element crosses the Emit boundary as a box
    # (increment 14) and the user lambda `f(x)` is an indirect call taking
    # AND returning that same aggregate (increment 16).  Before this
    # increment map's handler case, map itself, its thunk and `main` all
    # demoted with the scalar-only reason; now the whole consumed pipeline
    # emits.
    #
    # The rest of std.stream still demotes, honestly and for an unrelated
    # reason: the module's Emit element cell is MONOMORPHIC, so pulling in
    # iota/emit_range/sum/product/count (which do int arithmetic on the
    # element) under an aggregate element type conflicts their kinds.
    # That is the generic-instantiation gap, not the ABI.
    for src, elem in ((_STREAM_STRUCT_ELEM_SRC, r"struct:Item"),
                      (_STREAM_ENUM_ELEM_SRC, r"enum:Cell")):
        ir = llvm_from_source(src)
        for sym in ("mx_main", r"mx_std_stream_map(?:_ho\d+)?",
                    r"mx_std_stream_filter(?:_ho\d+)?",
                    r"mx_std_stream_map(?:_ho\d+)?_lambda1",
                    r"mx_std_stream_filter(?:_ho\d+)?_lambda1",
                    r"mx___handler_Emit_emit_std_stream_map(?:_ho\d+)?_hs3",
                    r"mx___handler_Emit_emit_std_stream_filter(?:_ho\d+)?_hs3",
                    r"mx___handle_body_Emit_std_stream_map(?:_ho\d+)?_hs3",
                    r"mx___handle_body_Emit_std_stream_filter(?:_ho\d+)?_hs3"):
            assert re.search(rf"^define (?:i64|void|ptr|double) @{sym}\(",
                             ir, re.M), (sym, elem)
        assert "scalar-only" not in ir
        assert re.search(rf"; word-uniform param \w+: boundary box {elem}", ir)
        assert re.search(
            rf"ret i64 %\w+  ; word-uniform lambda return \({elem}", ir)


@needs_clang
def test_native_std_stream_struct_element_differential(tmp_path):
    assert_native_matches_interp(_STREAM_STRUCT_ELEM_SRC, tmp_path)


@needs_clang
def test_native_std_stream_enum_element_differential(tmp_path):
    assert_native_matches_interp(_STREAM_ENUM_ELEM_SRC, tmp_path)


@needs_asan
def test_native_std_stream_aggregate_element_asan_no_uaf(tmp_path):
    # Effect scopes + heap envs + boundary boxes on both the effect edge
    # and the indirect-call edge: detect_leaks=0 per the leak-by-design
    # contract, so exit 0 proves no use-after-free / no double-free.
    assert_native_matches_interp_asan_boxes(_STREAM_STRUCT_ELEM_SRC, tmp_path)
    assert_native_matches_interp_asan_boxes(_STREAM_ENUM_ELEM_SRC, tmp_path)


def test_indirect_aggregate_ir_passes_llvm_verifier(tmp_path):
    # The boxed-aggregate indirect ABI produces well-formed IR: the
    # `i64 (ptr, i64...)` indirect call type, the inttoptr/ptrtoint word
    # conversions and the box copies all check out under LLVM's verifier.
    if shutil.which("opt") is None:
        pytest.skip("LLVM opt not installed")
    for src in (_IND_AGG_ROUNDTRIP_SRC, _IND_ENUM_RETURN_SRC,
                _IND_AGG_CAPTURE_SRC, _IND_AGG_LOOP_SRC,
                _STREAM_STRUCT_ELEM_SRC, _STREAM_ENUM_ELEM_SRC):
        ll = tmp_path / "ind_agg.ll"
        ll.write_text(llvm_from_source(src))
        proc = subprocess.run(
            ["opt", "-passes=verify", "-disable-output", str(ll)],
            capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# Increment 17: BOUNDARY-BOX TRAFFIC REDUCTION (pure optimization).
#
# Three elisions, each with an explicit safety condition; anything not
# provable keeps boxing.  The differential/ASan tests above are the
# correctness net — these pin the mallocs that must NOT be emitted.
#
#   (1) DOUBLE-BOX ELISION.  A value whose single def receives a boundary
#       word (perform / resume / handle value / word-uniform indirect call
#       result) is a pointer to a box the producer malloc'd, filled once and
#       never frees.  The receiver keeps that pointer (a bbox_view) instead
#       of copying out, and handing it to the NEXT boundary passes the
#       pointer straight through.  Safety: the box is immortal (no emitted
#       path frees a boundary box), write-once (handle-scope subfunctions
#       never copy out through a param pointer, and the word ABI refuses
#       @mut aggregate params), and the receiver is read-only (single def,
#       never at a callee write-back position).  Handler cases and handle
#       bodies now RETURN that word directly (the boundary-word ABI), so
#       the shims' unconditional box disappears too.
#   (2) READ-ONLY AGGREGATE ARGUMENT.  An aggregate argument of an indirect
#       closure call passes the caller's storage pointer when every
#       statically-possible callee is word-uniform and leaves the position
#       out of its write-back set (writeback_map — the same analysis the
#       direct-call elide-copy pass uses).  Safety: nobody writes through
#       the pointer, and an indirect call is synchronous on this stack, so
#       the storage outlives it.
#   (3) LOOP-INVARIANT BOX.  A value with a single block-0 def that no
#       write-back can touch is boxed once at the end of block 0 and reused
#       at every boundary site in the loop.
# ---------------------------------------------------------------------------

_FX17_CHAIN_SRC = """
struct St { a: int, b: int }
effect E { one(s: St) -> St two(s: St) -> St }
fn main() -> int {
    let out = handle E with {
        one(s) -> resume(St { a: s.a + 1, b: s.b }),
        two(s) -> resume(St { a: s.a, b: s.b + 1 })
    } in {
        let x = perform E.one(St { a: 0, b: 0 });
        let y = perform E.two(x);
        let z = perform E.one(y);
        perform E.two(z)
    };
    print(out.a);
    print(out.b);
    0
}
"""


def test_boundary_results_view_the_producers_box_instead_of_copying_out():
    # Elision (1), receive side: every perform result and the handle value
    # keep the producer's pointer; NOTHING copies an aggregate out of a
    # boundary box any more in this program.
    ir = llvm_from_source(_FX17_CHAIN_SRC)
    assert count_placeholders(ir) == 0
    assert len(re.findall(
        r"; elide-copy: \w+ views the producer's write-once boundary box "
        r"\(struct:St\)", ir)) == 7   # 4 performs + 2 resumes + the handle value
    # the old copy-out shape (inttoptr the word, then load/store the
    # aggregate into our own slot) is gone
    assert "to ptr  ; boundary box: struct:St" not in ir


def test_chained_perform_passes_the_received_box_straight_through():
    # Elision (1), send side: `perform E.two(x)` where x came straight out
    # of the previous perform re-uses that box; only the FRESH structs (the
    # initial St and the two resume values) still box.
    ir = llvm_from_source(_FX17_CHAIN_SRC)
    assert len(re.findall(
        r"; elide-box: \w+ already IS an immortal write-once boundary box "
        r"\(struct:St\); its pointer passes through", ir)) == 6
    assert len(re.findall(
        r"call ptr @malloc\(i64 16\)"
        r"  ; boundary box: struct:St \(write-once, leaks by design\)",
        ir)) == 3
    assert len(re.findall(r"call ptr @malloc\(", ir)) == 3


def test_handle_shims_allocate_nothing_for_aggregate_results():
    # Elision (1), the double box itself: the dispatcher used to malloc a
    # box per dispatch and sret-fill it from the case fn, which then boxed
    # again at the next hop.  Cases and bodies return the word directly, so
    # both shims are allocation-free.
    ir = llvm_from_source(_FX17_CHAIN_SRC)
    shims = re.findall(r"^define internal i64 @mxfx\.(?:body|disp)\..*?^\}",
                       ir, re.M | re.S)
    assert shims
    for shim in shims:
        assert "@malloc" not in shim, shim
    assert re.search(r"; boundary-word body result struct:St", ir)
    assert len(re.findall(r"; boundary-word case result struct:St", ir)) == 2
    assert len(re.findall(
        r"^define i64 @mx___handler_E_\w+\(ptr %cl\.env, ptr %a\.s, "
        r"ptr %a\.__k\) \{  ; handle-scope boundary-word ABI$", ir, re.M)) == 2


@needs_clang
def test_native_chained_perform_differential(tmp_path):
    assert_native_matches_interp(_FX17_CHAIN_SRC, tmp_path)


@needs_asan
def test_native_chained_perform_asan_no_uaf(tmp_path):
    # Boundary boxes leak by design, so detect_leaks=0: exit 0 proves the
    # elisions never turned an immortal box into a use-after-free — the
    # failure mode this whole round has to be guarded against, since a
    # passed-through pointer is read on the OTHER coroutine stack.
    assert_native_matches_interp_asan_boxes(_FX17_CHAIN_SRC, tmp_path)


_FX17_PAYLOAD_VIEW_SRC = """
struct Pt { x: int, y: int }
enum Hold { Full(Pt), Empty }
effect Sw { sw(p: Pt) -> Pt }
fn main() -> int {
    let b = Full(Pt { x: 1, y: 2 });
    let r = handle Sw with { sw(p) -> resume(Pt { x: p.y, y: p.x }) } in {
        match b {
            Full(p) => { perform Sw.sw(p) },
            Empty => { Pt { x: 0, y: 0 } }
        }
    };
    print(r.x);
    print(r.y);
    0
}
"""


def test_enum_payload_view_still_boxes_at_the_effect_boundary():
    # THE GUARD on elision (1): a variant_field box view aliases an enum
    # PAYLOAD box, and _unique_box_enums may free one of those at frame
    # exit — so its pointer must never be re-exported across a boundary,
    # where it could outlive this frame.  Only boxes proven immortal pass
    # through; a payload view copies into a fresh box, as before.
    ir = llvm_from_source(_FX17_PAYLOAD_VIEW_SRC)
    assert count_placeholders(ir) == 0
    assert re.search(
        r"; elide-copy: variant_field \w+ reads through the box pointer", ir)
    # the perform argument still allocates a box and copies into it
    assert re.search(
        r"call ptr @malloc\(i64 16\)"
        r"  ; boundary box: struct:Pt \(write-once, leaks by design\)\n"
        r"  %\w+ = load %struct\.Pt, ptr %\w+", ir)


@needs_clang
def test_native_enum_payload_view_boundary_differential(tmp_path):
    assert_native_matches_interp(_FX17_PAYLOAD_VIEW_SRC, tmp_path)


@needs_asan
def test_native_enum_payload_view_boundary_asan_no_uaf(tmp_path):
    # Payload boxes leak by design here (the enum is copied into the
    # handle-site env, so it is not a unique box): detect_leaks=0, exit 0
    # proves no use-after-free of the payload box across the boundary.
    assert_native_matches_interp_asan_boxes(_FX17_PAYLOAD_VIEW_SRC, tmp_path)


_FX17_RO_IND_ARG_SRC = """
struct S { a: int, b: int }
fn bump(s: S) -> int { s.a + s.b }
fn keep(s: S) -> int { s.a }
fn apply(f: fn(S) -> int, s: S) -> int { f(s) }
fn main() -> int {
    let fb = fn(s: S) -> bump(s);
    let fk = fn(s: S) -> keep(s);
    print(apply(fb, S { a: 3, b: 4 }));
    print(apply(fk, S { a: 5, b: 6 }));
    0
}
"""
# ^ Lambdas via variables on purpose: pins the shared-site elision (2)
# machinery, which monomorphize's per-site cloning would otherwise bypass.


def test_readonly_aggregate_indirect_argument_needs_no_box():
    # Elision (2): both members are word-uniform and neither writes back
    # through the position, so the argument travels as a pointer to the
    # caller's own storage.  apply() allocates NOTHING.
    ir = llvm_from_source(_FX17_RO_IND_ARG_SRC)
    assert count_placeholders(ir) == 0
    assert re.search(
        r"ptrtoint ptr %[\w.]+ to i64  ; elide-box: read-only aggregate "
        r"argument \w+ \(struct:S\) passes the caller's storage pointer", ir)
    apply_body = re.search(r"^define i64 @mx_apply\(.*?^\}", ir,
                           re.M | re.S).group(0)
    assert "@malloc" not in apply_body, apply_body
    # the receiving ABI is untouched: still a word decoded to a ptr
    assert len(re.findall(
        r"; word-uniform param s: boundary box struct:S", ir)) == 2


@needs_clang
def test_native_readonly_aggregate_indirect_argument_differential(tmp_path):
    assert_native_matches_interp(_FX17_RO_IND_ARG_SRC, tmp_path)


@needs_asan
def test_native_readonly_aggregate_indirect_argument_asan_no_uaf(tmp_path):
    # Heap closure envs leak by design (detect_leaks=0); exit 0 proves the
    # caller-storage pointer is never read after its frame died — the
    # lifetime claim behind elision (2).
    assert_native_matches_interp_asan_boxes(_FX17_RO_IND_ARG_SRC, tmp_path)


_FX17_LOOP_INVARIANT_SRC = """
struct Cfg { k: int, n: int }
effect Q { ask(c: Cfg) -> int }
fn main() -> int {
    let total = handle Q with { ask(c) -> resume(c.k + c.n) } in {
        let cfg = Cfg { k: 3, n: 4 };
        let mut i = 0;
        let mut s = 0;
        while i < 10 { s = s + perform Q.ask(cfg); i = i + 1; };
        s
    };
    print(total);
    0
}
"""


def test_loop_invariant_boundary_box_is_hoisted_out_of_the_loop():
    # Elision (3): cfg never changes after its block-0 def, so ONE box
    # serves all ten iterations.  The malloc must live in the entry block,
    # not the loop body.
    ir = llvm_from_source(_FX17_LOOP_INVARIANT_SRC)
    assert count_placeholders(ir) == 0
    body = re.search(r"^define i64 @mx___handle_body_Q_main\w*\(.*?^\}",
                     ir, re.M | re.S).group(0)
    assert len(re.findall(r"call ptr @malloc\(", body)) == 1
    blocks = re.split(r"^(\w+):$", body, flags=re.M)
    by_label = dict(zip(blocks[1::2], blocks[2::2]))
    assert "@malloc" in by_label["bb0"], body
    for label, text in by_label.items():
        if label != "bb0":
            assert "@malloc" not in text, (label, body)
    assert re.search(
        r"; loop-invariant boundary box: \w+ \(struct:Cfg\)", body)
    assert re.search(
        r"; elide-box: \w+ \(struct:Cfg\) reuses its loop-invariant box",
        body)


@needs_clang
def test_native_loop_invariant_box_differential(tmp_path):
    assert_native_matches_interp(_FX17_LOOP_INVARIANT_SRC, tmp_path)


@needs_asan
def test_native_loop_invariant_box_asan_no_uaf(tmp_path):
    # The hoisted box is shared by every iteration and by the handler on
    # the other stack: detect_leaks=0 (boundary boxes leak by design),
    # exit 0 proves no use-after-free / double-free from that sharing.
    assert_native_matches_interp_asan_boxes(_FX17_LOOP_INVARIANT_SRC, tmp_path)


_FX17_MUT_LOOP_SRC = """
struct P { a: int, b: int }
effect Ask { ask(p: P) -> P }
fn main() -> int {
    let t = handle Ask with { ask(p) -> resume(P { a: p.a + 1, b: p.b }) } in {
        let mut acc = P { a: 0, b: 5 };
        let mut i = 0;
        while i < 10 { acc = perform Ask.ask(acc); i = i + 1; };
        acc
    };
    print(t.a);
    print(t.b);
    0
}
"""


def test_reassigned_loop_accumulator_keeps_its_per_iteration_box():
    # The NEGATIVE side of elision (3): acc is rebound every iteration, so
    # its box contents change and nothing can be hoisted or shared.  It
    # still mallocs inside the loop — correctness over speed.
    ir = llvm_from_source(_FX17_MUT_LOOP_SRC)
    assert count_placeholders(ir) == 0
    assert "loop-invariant boundary box" not in ir
    body = re.search(r"^define i64 @mx___handle_body_Ask_main\w*\(.*?^\}",
                     ir, re.M | re.S).group(0)
    blocks = re.split(r"^(\w+):$", body, flags=re.M)
    by_label = dict(zip(blocks[1::2], blocks[2::2]))
    assert any("@malloc" in t for lab, t in by_label.items() if lab != "bb0")


@needs_clang
def test_native_reassigned_loop_accumulator_differential(tmp_path):
    assert_native_matches_interp(_FX17_MUT_LOOP_SRC, tmp_path)


@needs_asan
def test_native_reassigned_loop_accumulator_asan_no_uaf(tmp_path):
    assert_native_matches_interp_asan_boxes(_FX17_MUT_LOOP_SRC, tmp_path)


def test_increment17_ir_passes_llvm_verifier(tmp_path):
    # Box views, passed-through pointers, the boundary-word ABI and the
    # hoisted boxes all have to produce well-formed, dominance-correct IR.
    if shutil.which("opt") is None:
        pytest.skip("LLVM opt not installed")
    for src in (_FX17_CHAIN_SRC, _FX17_PAYLOAD_VIEW_SRC, _FX17_RO_IND_ARG_SRC,
                _FX17_LOOP_INVARIANT_SRC, _FX17_MUT_LOOP_SRC,
                _FX_STRUCT_ROUNDTRIP_SRC, _FX_STD_FIND_SRC):
        ll = tmp_path / "inc17.ll"
        ll.write_text(llvm_from_source(src))
        proc = subprocess.run(
            ["opt", "-passes=verify", "-disable-output", str(ll)],
            capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# Increment 18: the native path runs on MONOMORPHIZED HIR
# ---------------------------------------------------------------------------
#
# The backend's value-kind cells are per-function and monomorphic: one
# generic function reached at two different types joins both kinds into
# `conflict`, and the whole function (plus everything calling it) demotes to
# a comment-only placeholder.  pipeline.emit_llvm_from_source now runs
# compiler/monomorphize.py first, so each instantiation is a separate
# function with its own kind cells.  The interpreter path deliberately does
# NOT monomorphize -- it is the semantics reference -- so every differential
# below compares native SPECIALIZED code against UNSPECIALIZED interpreted
# semantics, which is exactly the contract the pass claims.
#
# The pass only rewrites call sites whose instantiation it can RESOLVE.
# Unresolvable sites keep their generic callee and the generic original
# stays loaded, so enabling it never changes what a program computes -- only
# how much of it the backend can emit.

_MONO_TWO_TYPES_SRC = """
fn identity<T>(x: T) -> T { x }

fn main() -> int {
    print(identity("mono"))
    return identity(7)
}
"""

_MONO_LET_SRC = """
fn identity<T>(x: T) -> T { x }

fn main() -> int {
    let a = 7
    let b = "mono-let"
    let c = identity(b)
    print(c)
    return identity(a)
}
"""

_MONO_DECLARED_PARAM_SRC = """
fn identity<T>(x: T) -> T { x }
fn wrap_int(n: int) -> int { identity(n) }
fn wrap_str(s: string) -> string { identity(s) }

fn main() -> int {
    print(wrap_str("mono-param"))
    return wrap_int(9)
}
"""

_MONO_RECURSION_SRC = """
fn count_down<T>(x: T, n: int) -> int {
    if n <= 0 { return 0 }
    return count_down(x, n - 1) + 1
}

fn main() -> int {
    print(count_down("s", 2).to_string())
    return count_down(1, 5)
}
"""

_MONO_STRUCT_SRC = """
struct P { v: int }

fn pick<T>(a: T, b: T) -> T { a }

fn main() -> int {
    let s = pick("x", "y")
    print(s)
    let p = pick(P { v: 5 }, P { v: 6 })
    return pick(p.v, 0)
}
"""


def test_mono_two_instantiations_undemote():
    # WITHOUT the pass, identity's kind cells see both an i64 and a str
    # argument and main demotes.  WITH it, each instantiation is its own
    # function and the module emits completely.
    plain = llvm_from_source(_MONO_TWO_TYPES_SRC, monomorphize=False)
    assert count_placeholders(plain) == 1
    assert "define i64 @mx_main(" not in plain

    ir = llvm_from_source(_MONO_TWO_TYPES_SRC)
    assert count_placeholders(ir) == 0
    assert "define i64 @mx_identity_Int(" in ir
    assert "define ptr @mx_identity_String(" in ir
    assert "define i64 @mx_main(" in ir
    # The fully specialized original is erased, not emitted alongside.
    assert not re.search(r"^define .* @mx_identity\(", ir, re.M)


def test_mono_resolves_through_let_bindings():
    # `let a = 1; identity(a)` is the same instantiation as `identity(1)`;
    # before let-resolution the pass could not see it and the module demoted.
    plain = llvm_from_source(_MONO_LET_SRC, monomorphize=False)
    assert count_placeholders(plain) == 1

    ir = llvm_from_source(_MONO_LET_SRC)
    assert count_placeholders(ir) == 0
    assert "define i64 @mx_identity_Int(" in ir
    assert "define ptr @mx_identity_String(" in ir


def test_mono_resolves_through_declared_primitive_params():
    plain = llvm_from_source(_MONO_DECLARED_PARAM_SRC, monomorphize=False)
    assert count_placeholders(plain) == 1

    ir = llvm_from_source(_MONO_DECLARED_PARAM_SRC)
    assert count_placeholders(ir) == 0
    assert "define i64 @mx_identity_Int(" in ir
    assert "define ptr @mx_identity_String(" in ir


def test_mono_rebound_names_are_not_trusted():
    # A name that is REBOUND (assignment target, lambda parameter, pattern
    # binder) is excluded from the let-environment wholesale: the pass would
    # otherwise have to prove the shadow carries the same type, and picking
    # the wrong clone is a miscompile.  Conservative == still correct: the
    # call simply stays generic.
    src = """
fn identity<T>(x: T) -> T { x }

fn main() -> int {
    let mut a = 1
    a = 2
    return identity(a)
}
"""
    from metaxu.compiler.monomorphize import _rebound_names
    ctx = build_context_from_source(src)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    main = next(f for f in hir if str(f.sym) == "main")
    assert "a" in _rebound_names(main.body)

    ir = llvm_from_source(src)
    assert "mx_identity_Int" not in ir        # stayed generic on purpose
    assert "define i64 @mx_identity(" in ir   # ...and the original survives
    assert count_placeholders(ir) == 0


def test_mono_clone_is_memoized_not_duplicated():
    # Code-growth guard: N call sites at the SAME instantiation must produce
    # exactly ONE clone, not N.
    src = """
fn identity<T>(x: T) -> T { x }

fn main() -> int {
    return identity(1) + identity(2) + identity(3) + identity(4)
}
"""
    ir = llvm_from_source(src)
    assert len(re.findall(r"^define .* @mx_identity_Int\(", ir, re.M)) == 1
    assert count_placeholders(ir) == 0


def test_mono_growth_is_bounded_by_instantiation_count():
    # One generic reached at three types yields exactly three clones and the
    # original is erased: emitted-function growth is +2, not exponential.
    src = """
fn identity<T>(x: T) -> T { x }

fn main() -> int {
    print(identity("s"))
    let f = identity(1.5)
    print(f.to_string())
    return identity(3)
}
"""
    plain = llvm_from_source(src, monomorphize=False)
    ir = llvm_from_source(src)

    def defs(text):
        # user functions only; runtime shims (metaxu_print_str) come and go
        # with which builtins the module ends up lowering natively
        return {n for n in re.findall(r"^define [^@]*@(\w+)\(", text, re.M)
                if n.startswith("mx_")}

    grown = defs(ir) - defs(plain)
    assert grown == {"mx_identity_Int", "mx_identity_String",
                     "mx_identity_Float", "mx_main"}, grown
    assert "mx_identity" not in defs(ir)


def test_mono_polymorphic_recursion_terminates():
    # Polymorphic recursion (the classic non-termination hazard: each level
    # instantiates at a STRICTLY LARGER type).  _known_type reports a struct
    # by its bare name -- Box, not Box[Box[Int]] -- so the instantiation
    # lattice is finite and the memo closes the loop after one clone.
    src = """
struct Box { v: int }

fn depth<T>(x: T, n: int) -> int {
    if n <= 0 { return 0 }
    return depth(Box { v: n }, n - 1) + 1
}

fn main() -> int { return depth(1, 4) }
"""
    ir = llvm_from_source(src)
    assert len(re.findall(r"^define .* @mx_depth_Box\(", ir, re.M)) == 1
    assert len(re.findall(r"^define .* @mx_depth_Int\(", ir, re.M)) == 1


def test_mono_mir_interprets_identically():
    # The pass's own correctness contract, checked on the one corpus example
    # it actually specializes: interpreting monomorphized MIR must give the
    # same answer as interpreting the unspecialized MIR.
    src = (REPO_ROOT / "examples" / "linked_list.mx").read_text()

    def run(mono):
        interp = MirInterpreter()
        interp.load(mir_from_source(src, monomorphize=mono))
        out = []
        interp.register_builtin("print", lambda *a: (out.append(a), UNIT)[1])
        return interp.call("main", []), tuple(out)

    assert run(False) == run(True)


def test_mono_interpreter_path_is_unaffected():
    # Item 1's contract: only the LLVM entry point monomorphizes.  The
    # interpreter front door keeps its default of OFF and its MIR is
    # byte-identical to the unmonomorphized lowering.
    from metaxu.compiler.mir import dump_mir
    from metaxu.compiler.pipeline import (emit_llvm_from_source,
                                          run_pipeline_from_source)
    _a, _h, mir_txt, _c = run_pipeline_from_source(_MONO_TWO_TYPES_SRC)
    assert "identity$Int" not in mir_txt
    assert dump_mir(mir_from_source(_MONO_TWO_TYPES_SRC)) == mir_txt
    assert "mx_identity_Int" in emit_llvm_from_source(_MONO_TWO_TYPES_SRC)


@needs_clang
def test_native_mono_two_instantiations_differential(tmp_path):
    assert_native_matches_interp(_MONO_TWO_TYPES_SRC, tmp_path)


@needs_clang
def test_native_mono_let_differential(tmp_path):
    assert_native_matches_interp(_MONO_LET_SRC, tmp_path)


@needs_clang
def test_native_mono_declared_param_differential(tmp_path):
    assert_native_matches_interp(_MONO_DECLARED_PARAM_SRC, tmp_path)


@needs_clang
def test_native_mono_recursion_differential(tmp_path):
    assert_native_matches_interp(_MONO_RECURSION_SRC, tmp_path)


@needs_clang
def test_native_mono_struct_instantiation_differential(tmp_path):
    assert_native_matches_interp(_MONO_STRUCT_SRC, tmp_path)


@needs_clang
def test_native_mono_linked_list_example_differential(tmp_path):
    # The one corpus example the pass actually specializes (push_front$Int):
    # native monomorphized code must still equal the interpreter running the
    # unspecialized program.
    assert_native_matches_interp(
        (REPO_ROOT / "examples" / "linked_list.mx").read_text(), tmp_path)


_MONO_ASAN_SRC = """
struct P { v: int }

fn pick<T>(a: T, b: T) -> T { a }

fn main() -> int {
    print(pick("asan", "no"))
    let p = pick(P { v: 5 }, P { v: 6 })
    print(p.v.to_string())
    return pick(0, 1)
}
"""


@needs_asan
def test_native_mono_struct_instantiation_asan(tmp_path):
    # Cloned bodies allocate exactly like their originals; ASan over a
    # struct-carrying instantiation proves cloning introduced no
    # double-free and no use-after-free (payload boxes leak by design, so
    # this runs under the documented detect_leaks=0 contract).
    assert_native_matches_interp_asan_boxes(_MONO_ASAN_SRC, tmp_path)


def test_mono_ir_passes_llvm_verifier(tmp_path):
    if shutil.which("opt") is None:
        pytest.skip("LLVM opt not installed")
    for src in (_MONO_TWO_TYPES_SRC, _MONO_LET_SRC, _MONO_DECLARED_PARAM_SRC,
                _MONO_RECURSION_SRC, _MONO_STRUCT_SRC, _MONO_ASAN_SRC):
        ll = tmp_path / "mono.ll"
        ll.write_text(llvm_from_source(src))
        proc = subprocess.run(
            ["opt", "-passes=verify", "-disable-output", str(ll)],
            capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# Increment 19: NATIVE try/catch (mx_try landing pads)
# ---------------------------------------------------------------------------
#
# `try { body } catch e { handler }` was the last construct that demoted
# wholesale ("uses try/catch (try_scope)").  It now lowers to an env fill
# plus mx_try over a per-site body thunk and catch thunk (metaxu_effects.c:
# a setjmp landing pad whose chain is PER-FIBER, so it composes with the
# ucontext coroutine scheduler instead of fighting it).  Three things can
# silently go wrong, and each has tests below:
#
#   1. the VALUE the catch binding receives must be byte-identical to the
#      interpreter's `InterpError.message` (docs/try_catch.md);
#   2. the CONTROL FLOW must match — caught / not caught / nested / across
#      an effect scope in both directions / uncaught;
#   3. the SET of catchable failures must match exactly.  Where it cannot
#      (match_fail: the interpreter's message embeds the MIR function name
#      and monomorphization renames it) the try DEMOTES with that reason,
#      rather than catching more or less than the interpreter would.

_TRY_CAUGHT_SRC = """
effect Fail {
    boom(x: int) -> int
}

fn body() performs Fail -> int {
    perform Fail.boom(1)
}

fn main() -> int {
    let r = try {
        body();
        1
    } catch e {
        print("caught: " + e);
        7
    };
    print(r);
    0
}
"""

_TRY_UNCAUGHT_SRC = """
effect Fail {
    boom(x: int) -> int
}

fn main() -> int {
    print("before");
    perform Fail.boom(1);
    print("after");
    0
}
"""

_TRY_NESTED_SRC = """
effect Fail {
    boom(x: int) -> int
}

fn inner() performs Fail -> int {
    perform Fail.boom(1)
}

fn main() -> int {
    let r = try {
        let a = try {
            inner()
        } catch e1 {
            print("inner: " + e1);
            5
        };
        a + 1
    } catch e2 {
        print("outer: " + e2);
        99
    };
    print(r);
    0
}
"""

_TRY_OVER_EFFECT_SRC = """
effect Fail {
    boom(x: int) -> int
}

fn main() -> int {
    let r = handle Fail with {
        boom(x) -> resume(x + 100)
    } in {
        try {
            let v = perform Fail.boom(2);
            v * 10
        } catch e {
            print("must not happen: " + e);
            0
        }
    };
    print(r);
    0
}
"""

_TRY_NORMAL_SRC = """
fn main() -> int {
    let r = try {
        3 + 4
    } catch e {
        print("must not happen: " + e);
        0
    };
    print(r);
    0
}
"""

# A failure raised INSIDE a handle body (on a coroutine stack) caught by a
# try OUTSIDE the handle: the cross-fiber path (MX_EV_ERROR + re-raise on
# the owner stack), which the interpreter gets from the ("error", exc)
# message its body thread sends to _pump_scope.
_TRY_ACROSS_HANDLE_SRC = """
effect Fail {
    boom(x: int) -> int
}

effect Other {
    ask() -> int
}

fn main() -> int {
    let r = try {
        handle Fail with {
            boom(x) -> resume(x + 1)
        } in {
            let a = perform Fail.boom(1);
            let b = perform Other.ask();
            a + b
        }
    } catch e {
        print("caught: " + e);
        42
    };
    print(r);
    0
}
"""

# The caught value ITSELF, printed verbatim.
_TRY_VALUE_SRC = """
effect Parser {
    parse(input: string) -> int
}

fn boom() performs Parser -> int {
    perform Parser.parse("x")
}

fn main() -> int {
    let msg = try {
        boom();
        "ok"
    } catch e {
        e
    };
    print(msg);
    0
}
"""


# An effect ABORT (handler returns without resuming) unwinding THROUGH a
# try installed inside the handle body.  That is a scope TEARDOWN, not a
# failure: the interpreter's _ScopeAbort is a BaseException its try_scope
# deliberately does not catch, and natively the abort longjmps to the
# scope's own mx_handle without ever consulting a landing pad.
_TRY_ABORT_THROUGH_SRC = """
effect Fail {
    boom(x: int) -> int
}

fn main() -> int {
    let r = handle Fail with {
        boom(x) -> x + 500
    } in {
        try {
            let v = perform Fail.boom(3);
            v * 1000
        } catch e {
            print("must not happen: " + e);
            0
        }
    };
    print(r);
    0
}
"""

# A failure raised BY A CATCH BLOCK propagates outward, never to its own
# try (docs/try_catch.md rule 4): mx_try pops its pad before calling the
# catch thunk, so the enclosing try's pad is the innermost one.
_TRY_CATCH_RAISES_SRC = """
effect A { one() -> int }
effect B { two() -> int }

fn main() -> int {
    let r = try {
        let inner = try {
            perform A.one()
        } catch e1 {
            print("inner: " + e1);
            perform B.two()
        };
        inner + 1
    } catch e2 {
        print("outer: " + e2);
        11
    };
    print(r);
    0
}
"""

# A runtime CONTRACT VIOLATION (metaxu_rt.c), not an effect failure: proves
# the mx_rt_raise conversion reaches the landing pad with the interpreter's
# exact wording.
_TRY_VEC_CONTRACT_SRC = """
fn main() -> int {
    let r = try {
        let v = Vec.new();
        v.push(1);
        v.pop();
        v.pop()
    } catch e {
        print("caught: " + e);
        7
    };
    print(r);
    0
}
"""


def test_try_scope_no_longer_demotes_and_calls_mx_try():
    ir = llvm_from_source(_TRY_CAUGHT_SRC)
    assert "uses try/catch (try_scope)" not in ir
    assert count_placeholders(ir) == 0
    assert "declare i64 @mx_try(ptr, ptr, ptr, ptr)" in ir
    assert re.search(r"call i64 @mx_try\(ptr @mxtc\.body\.", ir)
    # per-site thunks: the body takes only the env; the catch also takes the
    # failure message pointer (a `str`, never copied or reformatted).
    assert re.search(r"define internal i64 @mxtc\.body\.\w+\(ptr %env\)", ir)
    assert re.search(
        r"define internal i64 @mxtc\.catch\.\w+\(ptr %env, ptr %msg\)", ir)


def test_try_catch_parameter_is_a_str_in_the_native_signature():
    ir = llvm_from_source(_TRY_VALUE_SRC)
    # the catch subfunction: env pointer + the message pointer (never i64)
    assert re.search(r"define \w+ @mx___catch_main\w*\(ptr %cl\.env, ptr ", ir)


@needs_clang
def test_try_caught_failure_matches_interpreter(tmp_path):
    assert_native_matches_interp(_TRY_CAUGHT_SRC, tmp_path)


@needs_clang
def test_try_caught_value_is_byte_identical_to_the_interpreter(tmp_path):
    # The interpreter binds InterpError.message; so must the native runtime.
    _result, expected = interp_run(_TRY_VALUE_SRC)
    assert expected == "No handler for effect 'Parser'\n"
    ir = llvm_from_source(_TRY_VALUE_SRC)
    code, stdout = compile_and_run(ir, "main", workdir=str(tmp_path))
    assert stdout == expected
    assert code == 0


@needs_clang
def test_try_that_catches_nothing_matches_interpreter(tmp_path):
    assert_native_matches_interp(_TRY_NORMAL_SRC, tmp_path)


@needs_clang
def test_nested_try_innermost_catches(tmp_path):
    assert_native_matches_interp(_TRY_NESTED_SRC, tmp_path)


@needs_clang
def test_try_body_performing_an_effect_handled_outside_it(tmp_path):
    # The pad is installed ON THE HANDLE BODY'S COROUTINE; the perform parks
    # that fiber, the handler runs on the owner stack, and the resume comes
    # back to a body whose pad is still exactly where it was.
    assert_native_matches_interp(_TRY_OVER_EFFECT_SRC, tmp_path)


@needs_clang
def test_try_catches_a_failure_raised_inside_a_handle_body(tmp_path):
    assert_native_matches_interp(_TRY_ACROSS_HANDLE_SRC, tmp_path)


@needs_clang
def test_failure_in_a_catch_block_escapes_to_the_enclosing_try(tmp_path):
    _result, expected = interp_run(_TRY_CATCH_RAISES_SRC)
    assert expected == ("inner: No handler for effect 'A'\n"
                        "outer: No handler for effect 'B'\n11\n")
    assert_native_matches_interp(_TRY_CATCH_RAISES_SRC, tmp_path)


@needs_clang
def test_effect_abort_unwinds_through_a_try_without_catching(tmp_path):
    # Teardown, not failure: the catch block must never run and the handle
    # value must be the non-resuming case's value (503, not 3000).
    _result, expected = interp_run(_TRY_ABORT_THROUGH_SRC)
    assert expected == "503\n"
    assert_native_matches_interp(_TRY_ABORT_THROUGH_SRC, tmp_path)


@needs_clang
def test_try_catches_a_runtime_contract_violation(tmp_path):
    # metaxu_rt.c's catchable failures reach the pad with the interpreter's
    # exact wording ("pop: Vec is empty").
    _result, expected = interp_run(_TRY_VEC_CONTRACT_SRC)
    assert expected == "caught: pop: Vec is empty\n7\n"
    assert_native_matches_interp(_TRY_VEC_CONTRACT_SRC, tmp_path)


@needs_clang
def test_uncaught_failure_matches_the_interpreter(tmp_path):
    """No try installed: the interpreter raises InterpError and the native
    binary dies -- same message, same stdout prefix, both non-zero."""
    from metaxu.compiler.mir_interp import InterpError

    with pytest.raises(InterpError) as ei:
        interp_run(_TRY_UNCAUGHT_SRC)
    assert ei.value.message == "No handler for effect 'Fail'"

    ir = llvm_from_source(_TRY_UNCAUGHT_SRC)
    code, stdout = compile_and_run(ir, "main", workdir=str(tmp_path))
    assert code != 0                      # abort(), never a silent 0
    assert stdout == "before\n"           # and nothing after the failure
    # the message goes to stderr, unchanged by the catchable-raise plumbing
    proc = subprocess.run([str(tmp_path / "prog.bin")],
                          capture_output=True, text=True, timeout=60)
    assert "No handler for effect 'Fail'" in proc.stderr


@needs_clang
def test_try_ir_passes_llvm_verifier(tmp_path):
    if shutil.which("opt") is None:
        pytest.skip("LLVM opt not installed")
    for src in (_TRY_CAUGHT_SRC, _TRY_NESTED_SRC, _TRY_OVER_EFFECT_SRC,
                _TRY_NORMAL_SRC, _TRY_ACROSS_HANDLE_SRC, _TRY_VALUE_SRC,
                _TRY_ABORT_THROUGH_SRC, _TRY_VEC_CONTRACT_SRC,
                _TRY_CATCH_RAISES_SRC):
        ll = tmp_path / "try.ll"
        ll.write_text(llvm_from_source(src))
        proc = subprocess.run(
            ["opt", "-passes=verify", "-disable-output", str(ll)],
            capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr


@needs_asan
def test_try_that_catches_nothing_is_leak_clean_under_asan(tmp_path):
    # FULL leak checking: a try whose body completes normally allocates
    # nothing at all -- no heap pad state, no message copy.
    assert_native_matches_interp_asan(_TRY_NORMAL_SRC, tmp_path)


@needs_asan
def test_unwinding_try_is_memory_safe_under_asan(tmp_path):
    # detect_leaks=0, the documented contract for values that leak by
    # design: the caught message is an ordinary produced `str` the backend
    # cannot prove dead.  Exit 0 proves what matters here -- the longjmp
    # corrupted no coroutine stack, the scope teardown at the landing pad
    # double-freed nothing, and no continuation record was used after free.
    for i, src in enumerate((_TRY_CAUGHT_SRC, _TRY_NESTED_SRC,
                             _TRY_ACROSS_HANDLE_SRC, _TRY_OVER_EFFECT_SRC,
                             _TRY_ABORT_THROUGH_SRC, _TRY_VEC_CONTRACT_SRC,
                             _TRY_CATCH_RAISES_SRC)):
        d = tmp_path / f"case{i}"
        d.mkdir()
        assert_native_matches_interp_asan_boxes(src, d)


@needs_asan
def test_unwinding_try_frees_the_effect_machinery(tmp_path):
    """The scheduler itself stays leak-clean across a caught failure.

    ASan with LEAK CHECKING ON over the cross-fiber shape: the only
    allocation allowed to survive is the caught message copy (mx__dup), so
    a coroutine stack, a scope record or a continuation record showing up
    in the leak report is a real regression in the teardown path.
    """
    ir = llvm_from_source(_TRY_ACROSS_HANDLE_SRC)
    workdir = tmp_path / "leakcheck"
    workdir.mkdir()
    compile_and_run(ir, "main", workdir=str(workdir),
                    clang_args=("-fsanitize=address",),
                    run_env={"ASAN_OPTIONS": "detect_leaks=0"})
    proc = subprocess.run([str(workdir / "prog.bin")],
                          capture_output=True, text=True, timeout=60)
    err = proc.stderr
    if "LeakSanitizer" not in err:
        return  # nothing leaked at all: strictly better than the contract
    m = re.search(r"leaked in (\d+) allocation\(s\)", err)
    assert m, err
    n_leaks = int(m.group(1))
    # EVERY leaked allocation's #1 frame is the failure-message copy.  A
    # coroutine stack (mx_handle's malloc), a scope record or a continuation
    # record surviving would push this count apart.
    assert err.count("in mx__dup") == n_leaks, err
    assert n_leaks <= 2, err  # one message per raise on this path


_TRY_AROUND_MATCH_SRC = """
enum Color { Red, Green, Blue }

fn classify(c: Color) -> int {
    match c {
        Red -> 1,
        Green -> 2,
        Blue -> 3
    }
}

fn main() -> int {
    let r = try {
        classify(Blue)
    } catch e {
        print(e);
        9
    };
    print(r);
    0
}
"""


def test_try_around_match_emits_and_names_the_origin():
    """The old blanket demotion ("a try whose extent can reach a match_fail
    demotes") is gone: match_fail raises through mx_raise with the
    pre-monomorphization origin name, so the message is byte-identical to
    the interpreter's and the try lowers like any other."""
    ir = llvm_from_source(_TRY_AROUND_MATCH_SRC)
    assert "try/catch demoted:" not in ir
    assert "call i64 @mx_try(" in ir
    # the match_fail block is present and raises with the ORIGINAL name
    assert "match failure in 'classify': no pattern matched" in ir


@needs_clang
def test_native_try_catches_match_failure_like_the_interpreter(tmp_path):
    """Differential for the formerly-demoted shape: a non-exhaustive match
    inside a try, in a GENERIC function the native lane monomorphizes.
    The catch must bind the interpreter's exact message (embedding the
    pre-specialization name), and the surrounding program continues."""
    assert_native_matches_interp("""
enum Color { Red, Green, Blue }

fn pick<T>(c: Color, fallback: T) -> int {
    match c {
        Red -> 1,
        Green -> 2
    }
}

fn main() -> int {
    let caught = try {
        pick(Blue, 7)
    } catch e {
        print(e);
        9
    };
    print(caught);
    let fine = try {
        pick(Red, 7)
    } catch e {
        0 - 1
    };
    print(fine);
    0
}
""", tmp_path)


def test_example_04_gains_its_try_parse_defines():
    # 04_advanced_types.mx's try_parse is the motivating case: a speculative
    # `perform Parser.parse` with no handler installed, whose failure the
    # try turns into `Err(message)`.  Its body/catch subfunctions and
    # try_parse itself now emit; `main` still demotes on __list_lit and
    # to_string on its heterogeneous Result receiver (unrelated gaps).
    ir = llvm_from_source(
        (REPO_ROOT / "examples" / "04_advanced_types.mx").read_text())
    assert "uses try/catch (try_scope)" not in ir
    assert "define void @mx_try_parse(" in ir
    assert re.search(r"define i64 @mx___try_body_try_parse\w*\(", ir)
    assert re.search(r"define i64 @mx___catch_try_parse\w*\(", ir)
    assert re.search(r"call i64 @mx_try\(", ir)


# ---------------------------------------------------------------------------
# Boxed aggregates in Vec slots (increment 20)
#
# A native Vec slot is one 8-byte word. Word kinds sit in it directly;
# struct/enum aggregates sit in it as a pointer to an immortal write-once
# ELEMENT BOX (to_word mallocs a fresh copy on write, vec_elem_into copies
# back out on read). That keeps MIR's value semantics at both edges while
# the vec itself keeps mx_vec identity semantics.
# ---------------------------------------------------------------------------

_VEC_OF_ENUM_SRC = """
enum Value {
    VInt(int),
    VBool(bool)
}

fn main() -> int {
    let mut v = Vec.new();
    v.push(VInt(1));
    v.push(VBool(true));
    let x = v[0];
    let n = match x { VInt(i) => i, VBool(_) => 0 };
    print(n);
    print(v.len());
    return 0;
}
"""


def test_vec_of_enum_needs_no_placeholder(tmp_path):
    """Before element boxes this demoted with 'Vec of enum:Value{...}
    elements (only 8-byte word kinds fit native Vec slots)' — one push was
    enough to lose the whole function."""
    ir = assert_native_matches_interp(_VEC_OF_ENUM_SRC, tmp_path)
    assert count_placeholders(ir) == 0
    assert "element box" in ir


def test_vec_push_boxes_a_fresh_copy_per_push(tmp_path):
    """Value semantics on the WRITE edge: mutating the local after pushing
    must not show through the vec, and pushing the same local twice with a
    mutation in between must yield two distinct elements. A shared (not
    per-push) box would print 99 twice."""
    ir = assert_native_matches_interp("""
struct P { a: int, b: int }

fn main() -> int {
    let mut p = P { a: 1, b: 2 };
    let mut v = Vec.new();
    v.push(p);
    p.a = 99;
    v.push(p);
    print(v[0].a);
    print(v[1].a);
    return 0;
}
""", tmp_path)
    assert count_placeholders(ir) == 0


def test_vec_aggregate_read_write_and_pop_round_trip(tmp_path):
    """index_set replaces a slot with a new box; an element already read
    out is unaffected by that store (read edge copies OUT); pop decodes an
    aggregate; multi-slot enum payloads survive the round trip."""
    ir = assert_native_matches_interp("""
struct P { a: int, b: int }

enum Shape {
    Circle(int),
    Rect(int, int)
}

fn main() -> int {
    let mut v = Vec.new();
    v.push(P { a: 1, b: 2 });
    v.push(P { a: 99, b: 0 });
    let e0 = v[0];
    v[0] = P { a: 7, b: 8 };
    print(v[0].a);
    print(v[0].b);
    print(e0.a);
    let popped = v.pop();
    print(popped.a);
    print(v.len());

    let mut s = Vec.new();
    s.push(Circle(5));
    s.push(Rect(3, 4));
    let mut i = 0;
    let mut total = 0;
    while i < s.len() {
        let area = match s[i] {
            Circle(r) => r * r,
            Rect(w, h) => w * h
        };
        total = total + area;
        i = i + 1;
    }
    print(total);
    return 0;
}
""", tmp_path)
    assert count_placeholders(ir) == 0


def test_vec_keeps_reference_identity_with_boxed_elements(tmp_path):
    """Element boxes must not turn a Vec into a value: a push through one
    name is still visible through an alias, and a nested inner vec pushed
    into an outer one keeps aliasing after the fact."""
    ir = assert_native_matches_interp("""
struct P { a: int, b: int }

fn main() -> int {
    let mut v = Vec.new();
    v.push(P { a: 1, b: 2 });
    let w = v;
    w.push(P { a: 42, b: 0 });
    print(v.len());
    print(v[1].a);

    let mut outer = Vec.new();
    let mut inner = Vec.new();
    inner.push(P { a: 7, b: 7 });
    outer.push(inner);
    inner.push(P { a: 8, b: 8 });
    print(outer[0].len());
    return 0;
}
""", tmp_path)
    assert count_placeholders(ir) == 0


_ESCAPING_PAYLOAD_SRC = """
struct Inner { x: int, y: int }

enum Wrap {
    Boxed(Inner),
    Empty
}

fn build() -> Vec {
    let w = Boxed(Inner { x: 11, y: 22 });
    let mut v = Vec.new();
    v.push(w);
    return v;
}

fn main() -> int {
    let v = build();
    let got = v[0];
    let sum = match got {
        Boxed(inner) => inner.x + inner.y,
        Empty => 0
    };
    print(sum);
    return 0;
}
"""


def test_enum_payload_box_is_not_freed_when_the_enum_reaches_a_vec(tmp_path):
    """_unique_box_enums frees an entry-block make_variant's payload box on
    every ret path when it proves this frame is the sole owner. Pushing the
    enum into a Vec breaks that: the element box holds a COPY of the enum
    carrying the same payload pointer, and here the vec outlives build().

    The disqualification is load-bearing rather than theoretical — letting
    `push` through the use whitelist makes this program print 30 instead of
    33, reading a freed payload box."""
    ir = assert_native_matches_interp(_ESCAPING_PAYLOAD_SRC, tmp_path)
    assert count_placeholders(ir) == 0
    # No unique-box free survives in build(): the push made ownership shared.
    assert "unique box:" not in ir


@needs_asan
def test_vec_element_boxes_are_asan_clean(tmp_path):
    """Element boxes leak by design (immortal, write-once — exactly the
    enum payload / effect boundary contract), so leak detection is off;
    exit 0 proves no use-after-free and no double-free across push, index
    read/write, pop and an escaping vec."""
    assert_native_matches_interp_asan_boxes(_VEC_OF_ENUM_SRC, tmp_path)
    assert_native_matches_interp_asan_boxes(_ESCAPING_PAYLOAD_SRC, tmp_path)


# ---------------------------------------------------------------------------
# Higher-order call-site cloning (monomorphize's second axis) — the
# native payoff. Shapes that DEMOTED when two lambdas joined through one
# shared site now split into per-site clones, each of which pins its one
# lambda and takes the typed path. The shared-site machinery (word-uniform
# ABI, honest demotions) is still pinned above via var-routed lambdas.
# ---------------------------------------------------------------------------

def test_ho_clones_split_two_return_types_and_erase_the_original(tmp_path):
    """fn apply(f) { f() } reached at Int and at String: before cloning
    this joined apply's kinds to conflict and demoted apply plus both
    callers. Now each site's clone pins, and the dead original is erased
    (no define, no placeholder, no symbol at all)."""
    ir = assert_native_matches_interp("""
fn apply(f) { f() }
fn main() -> int {
    let a = apply(fn() -> 1);
    let b = apply(fn() -> "s");
    print(a);
    print(b);
    0
}
""", tmp_path)
    assert count_placeholders(ir) == 0
    assert re.search(r"@mx_apply_ho\d+", ir)
    assert not re.search(r"@mx_apply[^_]", ir)   # original fully gone


def test_ho_clone_mut_aggregate_write_back_works_natively(tmp_path):
    """The @mut-write-back shape that MUST demote through a shared
    word-uniform site (boundary boxes drop the write) works through
    per-site clones: each pins its lambda, passes the caller's real
    pointer, and the write-back lands. p.a and q.a prove it."""
    ir = assert_native_matches_interp("""
struct P { a: int }
fn apply(f: fn(@mut P) -> int, @mut p: P) -> int { f(p) }
fn main() -> int {
    let mut p = P { a: 1 };
    let mut q = P { a: 2 };
    print(apply(fn(@mut r: P) -> { r.a = r.a + 10; r.a }, p));
    print(apply(fn(@mut r: P) -> { r.a = r.a * 10; r.a }, q));
    print(p.a, q.a);
    0
}
""", tmp_path)
    assert count_placeholders(ir) == 0


def test_ho_clone_struct_vs_enum_sites_both_emit(tmp_path):
    """One site passes a struct, the other an enum — the two-way kind
    conflict that demoted the shared apply resolves into two clean
    monomorphic clones."""
    ir = assert_native_matches_interp("""
struct P { a: int }
enum E { X(int), Y }
fn apply(f, v) { f(v) }
fn main() -> int {
    print(apply(fn(p: P) -> p.a + 1, P { a: 4 }));
    print(apply(fn(e: E) -> match e { E.X(n) => n * 2, E.Y => 0 }, E.X(5)));
    0
}
""", tmp_path)
    assert count_placeholders(ir) == 0


def test_ho_clone_closure_pair_argument_emits(tmp_path):
    """A closure PAIR through an indirect position has no boundary box and
    demotes at a shared site; through per-site clones both the h and k
    positions pin and the whole program emits."""
    ir = assert_native_matches_interp("""
fn drive(h, k) -> int { h(k, 4) }
fn main() -> int {
    let twice = fn(g, x) -> g(g(x));
    let thrice = fn(g, x) -> g(g(g(x)));
    print(drive(twice, fn(v: int) -> v + 1));
    print(drive(thrice, fn(v: int) -> v * 2));
    0
}
""", tmp_path)
    assert count_placeholders(ir) == 0


def test_ho_clone_stdlib_catch_shape_emits_per_site(tmp_path):
    """The measured blocker shape: catch_-style effectful wrappers reached
    with different lambda types. Each clone carries its own handler
    sub-functions (per-containing-function naming), so the sites cannot
    collide."""
    ir = assert_native_matches_interp("""
effect Throw {
    throw(msg: string) -> int
}

fn catchy(f) {
    handle Throw with {
        throw(msg) -> 0 - 1
    } in {
        f()
    }
}

fn main() -> int {
    let ok = catchy(fn() -> 40);
    let boom = catchy(fn() -> { let x = perform Throw.throw("no"); x + 99 });
    print(ok);
    print(boom);
    0
}
""", tmp_path)
    assert count_placeholders(ir) == 0
    assert re.search(r"@mx___handler_Throw_throw_catchy_ho\d+_hs\d+", ir)


# ---------------------------------------------------------------------------
# Real OS threads (docs/threads_runtime.md): EFFECT_SPAWN/JOIN + mutexes
# ---------------------------------------------------------------------------
# The interpreter (test_threads.py) is the semantics reference; these
# differentials run the SCHEDULE-INDEPENDENT programs natively — final
# counter values, join results, error messages — never interleavings
# (the spec's determinism policy).  ASan runs use detect_leaks=0: thread
# and mutex handles (and spawn-crossing closure envs/boxes) are IMMORTAL
# by design, so ASan proves no-UAF/no-double-free, not leak-freedom.

_TSAN_PROBE: list[bool] = []


def tsan_available() -> bool:
    """True when clang can link -fsanitize=thread (TSan runtime installed)."""
    if not _TSAN_PROBE:
        if shutil.which("clang") is None:
            _TSAN_PROBE.append(False)
        else:
            import tempfile, os
            with tempfile.TemporaryDirectory(prefix="metaxu_tsan_probe_") as d:
                c = os.path.join(d, "t.c")
                with open(c, "w") as fh:
                    fh.write("int main(void){return 0;}\n")
                proc = subprocess.run(
                    ["clang", "-fsanitize=thread", c,
                     "-o", os.path.join(d, "t")],
                    capture_output=True, text=True)
                _TSAN_PROBE.append(proc.returncode == 0)
    return _TSAN_PROBE[0]


needs_tsan = pytest.mark.skipif(
    not tsan_available(),
    reason="clang TSan runtime not available (compile probe failed)")

_THREAD_EFFECTS_DECL = """
extern type Thread[T];
extern type Mutex;

effect Thread = {
    fn spawn[T](f: fn() -> @global T) -> @global Thread[T] with EFFECT_SPAWN
    fn join[T](thread: @global Thread[T]) -> @global T with EFFECT_JOIN
}

effect Mutex = {
    fn create() -> @global Mutex with EFFECT_MUTEX_CREATE
    fn lock(mutex: @global Mutex) -> () with EFFECT_MUTEX_LOCK
    fn unlock(mutex: @global Mutex) -> () with EFFECT_MUTEX_UNLOCK
}
"""

# N=4 threads x M=250 mutex-guarded increments of ONE Vec slot (Vec has
# identity semantics on both engines, so the captured vec IS shared
# state).  Final value 1000 is schedule-independent.  KEEP THE LOCK/UNLOCK
# LINES: deleting them was the TSan non-vacuity experiment — TSan then
# reports a data race on the slot (mx_vec_get/mx_vec_set from two child
# threads; see the commit message that landed the native thread path) —
# and the test below asserts the MUTEXED program is TSan-clean.
_THREAD_COUNTER_SRC = _THREAD_EFFECTS_DECL + """
fn main() -> int {
    let m = perform Mutex.create();
    let @mut counter = Vec.new();
    counter.push(0);
    let @mut handles = Vec.new();
    let @mut i = 0;
    while i < 4 {
        let t = perform Thread.spawn(|| {
            let @mut j = 0;
            while j < 250 {
                perform Mutex.lock(m);
                counter[0] = counter[0] + 1;
                perform Mutex.unlock(m);
                j = j + 1
            };
            0
        });
        handles.push(t);
        i = i + 1
    };
    let @mut k = 0;
    while k < 4 {
        perform Thread.join(handles[k]);
        k = k + 1
    };
    print(counter[0].to_string());
    0
}
"""

_THREAD_JOIN_VALUE_SRC = _THREAD_EFFECTS_DECL + """
fn main() -> int {
    let x = 40;
    let t = perform Thread.spawn(|| { x + 2 });
    let v = perform Thread.join(t);
    print(v.to_string());
    0
}
"""

# Every catchable thread/mutex error path, in one deterministic program:
# unlock-not-held, self-relock (EDEADLK), double join, and a child
# failure surfacing catchably at join.  The caught messages are
# language-visible values, so native output must be byte-identical to
# the interpreter's.
_THREAD_ERRORS_SRC = _THREAD_EFFECTS_DECL + """
fn main() -> int {
    let m = perform Mutex.create();
    let a = try { perform Mutex.unlock(m); "no error" } catch e { e };
    print(a);
    perform Mutex.lock(m);
    let b = try { perform Mutex.lock(m); "no error" } catch e { e };
    print(b);
    perform Mutex.unlock(m);
    let t = perform Thread.spawn(|| { 5 });
    perform Thread.join(t);
    let c = try { perform Thread.join(t); "no error" } catch e { e };
    print(c);
    let tf = perform Thread.spawn(|| {
        let @mut v = Vec.new();
        v.pop()
    });
    let d = try { perform Thread.join(tf); "no failure" } catch e {
        "caught: " + e
    };
    print(d);
    0
}
"""


@needs_clang
def test_native_thread_counter_matches_interp(tmp_path):
    ir = assert_native_matches_interp(_THREAD_COUNTER_SRC, tmp_path)
    assert count_placeholders(ir) == 0
    assert "call i64 @mx_thread_spawn(ptr" in ir


@needs_clang
def test_native_join_value_matches_interp(tmp_path):
    ir = assert_native_matches_interp(_THREAD_JOIN_VALUE_SRC, tmp_path)
    assert count_placeholders(ir) == 0


@needs_clang
def test_native_thread_error_paths_match_interp(tmp_path):
    """The caught error messages (unlock-not-held, deadlock self-relock,
    double join, child failure at join) are byte-identical across
    engines."""
    ir = assert_native_matches_interp(_THREAD_ERRORS_SRC, tmp_path)
    assert count_placeholders(ir) == 0


@needs_asan
def test_native_thread_counter_asan_no_uaf(tmp_path):
    """ASan (detect_leaks=0: handles/envs are immortal by design): exit 0
    proves no use-after-free / no double-free across the spawn/join and
    mutex traffic."""
    assert_native_matches_interp_asan_boxes(_THREAD_COUNTER_SRC, tmp_path)


@needs_tsan
def test_native_thread_counter_tsan_clean(tmp_path):
    """-fsanitize=thread (runtime objects TSan-instrumented too, see
    llvm_run._runtime_object_paths): a TSan report changes the exit code
    (66) and breaks the stdout/exit differential, so passing == zero
    reports.  Non-vacuity: with the lock/unlock lines deleted from the
    counter source, TSan reports a data race on the Vec slot (verified
    during development; see the landing commit message)."""
    result, expected_out = interp_run(_THREAD_COUNTER_SRC)
    assert result in (UNIT, 0)
    ir = llvm_from_source(_THREAD_COUNTER_SRC)
    exit_code, stdout = compile_and_run(
        ir, "main", workdir=str(tmp_path),
        clang_args=("-fsanitize=thread",))
    assert stdout == expected_out
    assert exit_code == 0
