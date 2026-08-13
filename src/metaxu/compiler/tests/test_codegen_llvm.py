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
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from metaxu.compiler.codegen_llvm import emit_llvm, mangle
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


def mir_from_source(source: str) -> list[MirFunc]:
    ctx = build_context_from_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    return lower_hir_to_mir(hir)


def llvm_from_source(source: str) -> str:
    return emit_llvm(mir_from_source(source))


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


def test_match_fail_calls_abort():
    f = make_func("m", [
        block([("params", ()), ("match_fail", "no pattern matched")], ("br", 1)),
        block([("let", "u", ("const_ty", "Unit"), ())], ("ret", "u")),
    ])
    ir = emit_llvm([f])
    assert "declare void @abort() noreturn" in ir
    assert "call void @abort()" in ir
    assert "unreachable" in ir


# ---------------------------------------------------------------------------
# Placeholder honesty
# ---------------------------------------------------------------------------

def test_suspending_function_is_placeholder():
    f = make_func("worker", [
        block([
            ("params", ("x",)),
            ("perform", "pv1", "State", "get", (), 1, "pv1"),
        ], ("br", 1)),
        block([], ("ret", "pv1")),
    ], suspending=True)
    ir = emit_llvm([f])
    assert count_placeholders(ir) == 1
    assert "suspending function" in ir
    assert "define" not in ir
    # every non-empty line of the placeholder chunk is a comment
    chunk = [c for c in ir.split("\n\n") if "worker" in c][0]
    assert all(line.startswith(";") for line in chunk.splitlines() if line.strip())


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

def test_struct_param_passes_ptr_with_callee_byval_copy():
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
    # ...and is byval-copied into the callee's own storage in the prelude
    assert re.search(r"load %struct\.Point, ptr %a\.p", ir)
    assert re.search(r"store %struct\.Point %t\d+, ptr %sv\.p", ir)
    # caller passes the storage pointer of its struct variable
    assert re.search(r"call i64 @mx_getx\(ptr %sv\.\w+\)", ir)


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


def test_variants_and_closures_are_placeholders():
    fs = [
        make_func("v", [block([
            ("params", ()),
            ("let", "v1", ("make_variant", "Opt", "Some"), ("a",)),
        ], ("ret", "v1"))]),
        make_func("c", [block([
            ("params", ()),
            ("let", "c1", ("make_closure", "lambda1", ("x",)), ()),
        ], ("ret", "c1"))]),
    ]
    ir = emit_llvm(fs)
    assert count_placeholders(ir) == 2
    assert "uses variants (make_variant)" in ir
    assert "uses closures (make_closure)" in ir


def test_unknown_external_and_runtime_builtin_are_placeholders():
    fs = [
        make_func("a", [block([
            ("params", ()),
            ("let", "r", ("call", "mystery_ffi"), ()),
        ], ("ret", "r"))]),
        make_func("b", [block([
            ("params", ()),
            ("let", "r", ("call", "__vec_lit"), ()),
        ], ("ret", "r"))]),
    ]
    ir = emit_llvm(fs)
    assert count_placeholders(ir) == 2
    assert "unknown external callee 'mystery_ffi'" in ir
    assert "calls runtime builtin '__vec_lit'" in ir


def test_caller_of_placeholder_is_demoted_for_linkability():
    fs = [
        make_func("bad", [block([
            ("params", ()),
            ("let", "r", ("call", "__vec_lit"), ()),
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
            ("let", "r", ("call", "__vec_lit"), ()),
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
