"""Regression tests for the ten findings of adversarial review round 5.

Every test goes through parsed source (parse -> ... -> interpreter / LLVM),
per the repo convention. Where a finding changed semantics (struct-param
write-back), both directions are pinned: the value-semantics side (plain
params stay callee-local) and the by-reference side (@mut params and method
receivers still write back), in both engines.
"""
from __future__ import annotations

import shutil

import pytest

from metaxu.errors import CompileError
from metaxu.compiler.pipeline import (build_context_from_source,
                                      emit_llvm_from_source)
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import InterpError, MirInterpreter, UNIT

needs_clang = pytest.mark.skipif(
    shutil.which("clang") is None, reason="clang is not installed")


def build_interp(source: str, file_path: str = "<mem>"):
    ctx = build_context_from_source(source, file_path=file_path)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    return interp


def run_main(source: str, entry: str = "main"):
    interp = build_interp(source)
    prints: list[str] = []
    interp.register_builtin(
        "print", lambda *a: (prints.append(" ".join(str(x) for x in a)), UNIT)[1])
    return interp.call(entry, []), prints


def native_matches_interp(source: str, tmp_path, entry: str = "main"):
    """Differential: clang-compiled result/stdout == interpreter's."""
    from metaxu.compiler.llvm_run import compile_and_run
    interp = build_interp(source)
    out: list[str] = []
    interp.register_builtin(
        "print", lambda *a: (out.append(" ".join(str(x) for x in a)), UNIT)[1])
    result = interp.call(entry, [])
    expected = "".join(line + "\n" for line in out)
    ir = emit_llvm_from_source(source)
    assert "placeholder -- unsupported" not in ir
    code, stdout = compile_and_run(ir, entry, workdir=str(tmp_path))
    assert stdout == expected
    if result is not UNIT and isinstance(result, (bool, int)):
        assert code == int(result) % 256
    return result


# ---------------------------------------------------------------------------
# Finding 1: struct write-back only for @mut params (plain params keep
# value semantics; rebinding stays local to the callee)
# ---------------------------------------------------------------------------

PLAIN_REBIND_SRC = """
struct Point { x: int, y: int }

fn clobber(p: Point) -> int {
    p = Point{x: 0, y: 0};
    p.x
}

fn main() -> int {
    let q = Point{x: 5, y: 6};
    clobber(q);
    q.x
}
"""


def test_plain_param_rebinding_stays_local_to_callee():
    result, _ = run_main(PLAIN_REBIND_SRC)
    assert result == 5


PLAIN_FIELD_SET_SRC = """
struct Counter { n: int }

fn bump(c: Counter) -> int {
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
"""


def test_plain_param_field_set_stays_local_to_callee():
    # Value semantics: both bumps see a fresh copy (11, 11) and the
    # caller's binding is untouched (10).
    _, prints = run_main(PLAIN_FIELD_SET_SRC)
    assert prints == ["11", "11", "10"]


MUT_FIELD_SET_SRC = """
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
"""


def test_mut_param_still_writes_back():
    # By-reference semantics for @mut: 11, 12, and the caller sees 12.
    _, prints = run_main(MUT_FIELD_SET_SRC)
    assert prints == ["11", "12", "12"]


def test_method_receiver_still_writes_back():
    # `self.field = ...` in an impl method mutates the caller's binding
    # (the receiver is by-reference), as example 10's Stack relies on.
    result, _ = run_main("""
struct Box { v: int }

implement Box {
    fn set(self, v: int) {
        self.v = v;
    }
}

fn main() -> int {
    let b = Box { v: 1 };
    b.set(42);
    b.v
}
""")
    assert result == 42


@needs_clang
def test_native_plain_param_value_semantics_matches_interp(tmp_path):
    native_matches_interp(PLAIN_FIELD_SET_SRC, tmp_path)


@needs_clang
def test_native_plain_rebinding_matches_interp(tmp_path):
    assert native_matches_interp(PLAIN_REBIND_SRC, tmp_path) == 5


@needs_clang
def test_native_mut_param_write_back_matches_interp(tmp_path):
    native_matches_interp(MUT_FIELD_SET_SRC, tmp_path)


# ---------------------------------------------------------------------------
# Finding 2: module-constant collection must not hoist lets out of
# module-level lambda bodies / initializer subexpressions
# ---------------------------------------------------------------------------

def test_module_level_lambda_body_let_is_not_hoisted():
    src = """
let DOUBLE = fn(x: int) -> int { let y = x + x; y };

fn main() -> int {
    DOUBLE(7)
}
"""
    result, _ = run_main(src)
    assert result == 14
    # And the lambda-local `y` must not be published as a module constant.
    ctx = build_context_from_source(src)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    init = next(m for m in lower_hir_to_mir(hir) if m.name == "__module_init")
    assert "DOUBLE" in init.globals_decl
    assert "y" not in init.globals_decl


# ---------------------------------------------------------------------------
# Finding 3: unqualified-call rewriting in non-entry modules must respect
# local (let / param) shadowing
# ---------------------------------------------------------------------------

def _run_tree(tmp_path, files: dict[str, str], root: str = "main.mx",
              entry: str = "main"):
    for rel, src in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src)
    root_path = tmp_path / root
    ctx = build_context_from_source(root_path.read_text(),
                                    file_path=str(root_path))
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    return interp.call(entry, [])


def test_local_let_shadows_module_function_in_non_entry_module(tmp_path):
    # In mathlib, use_local's `let add = ...` shadows the module's own
    # `add`: the call must hit the local closure (2+4=6), not the module
    # function (which would give 105).
    assert _run_tree(tmp_path, {
        "mathlib.mx": """
fn add(a: int, b: int) -> int { a + b + 99 }

fn use_local() -> int {
    let add = fn(a: int, b: int) -> int { a + b };
    add(2, 4)
}
""",
        "main.mx": """
from mathlib import use_local;

fn main() -> int { use_local() }
""",
    }) == 6


def test_param_shadows_module_function_in_non_entry_module(tmp_path):
    # A parameter bound to a closure shadows the module's function too.
    assert _run_tree(tmp_path, {
        "mathlib.mx": """
fn add(a: int, b: int) -> int { a + b + 99 }

fn apply(add: fn(int, int) -> int) -> int { add(1, 2) }

fn use_param() -> int {
    apply(fn(a: int, b: int) -> int { a + b })
}
""",
        "main.mx": """
from mathlib import use_param;

fn main() -> int { use_param() }
""",
    }) == 3


def test_unshadowed_call_still_rewrites_to_module_function(tmp_path):
    # The other direction: with no local binding in scope the module's own
    # function is still found (namespaced) from a sibling function.
    assert _run_tree(tmp_path, {
        "mathlib.mx": """
fn add(a: int, b: int) -> int { a + b }

fn twice(x: int) -> int { add(x, x) }
""",
        "main.mx": """
from mathlib import twice;

fn main() -> int { twice(21) }
""",
    }) == 42


# ---------------------------------------------------------------------------
# Finding 4: @mut write-back is uniform across call paths (closure calls
# included)
# ---------------------------------------------------------------------------

CLOSURE_MUT_SRC = """
struct Counter { n: int }

fn bump(c: @mut Counter) {
    c.n = c.n + 1;
}

fn main() -> int {
    let c = Counter { n: 0 };
    bump(c);
    let via = fn(x: @mut Counter) { bump(x) };
    via(c);
    c.n
}
"""


def test_mut_write_back_through_closure_call():
    result, _ = run_main(CLOSURE_MUT_SRC)
    assert result == 2


def test_plain_closure_param_does_not_write_back():
    # A closure param NOT declared @mut keeps value semantics: the inner
    # mutation stays inside the closure frame.
    result, _ = run_main("""
struct Counter { n: int }

fn main() -> int {
    let c = Counter { n: 0 };
    let touch = fn(x: Counter) { x.n = 99; };
    touch(c);
    c.n
}
""")
    assert result == 0


@needs_clang
def test_native_closure_mut_write_back_matches_interp(tmp_path):
    assert native_matches_interp(CLOSURE_MUT_SRC, tmp_path) == 2


# ---------------------------------------------------------------------------
# Finding 5: emit_llvm_from_source accepts file_path (multi-file programs
# emit LLVM from disk)
# ---------------------------------------------------------------------------

def test_emit_llvm_from_source_resolves_imports_via_file_path(tmp_path):
    (tmp_path / "mathlib.mx").write_text(
        "fn add(x: int, y: int) -> int { x + y }\n")
    root = tmp_path / "main.mx"
    root.write_text("""
from mathlib import add;

fn main() -> int { add(40, 2) }
""")
    ir = emit_llvm_from_source(root.read_text(), file_path=str(root))
    assert "mx_mathlib_add" in ir  # the imported module's function is emitted
    # Without file_path the loader cannot resolve the on-disk import.
    with pytest.raises(CompileError):
        emit_llvm_from_source(root.read_text())


# ---------------------------------------------------------------------------
# Finding 6: a failed __module_init re-raises on every call (never a
# silent skip)
# ---------------------------------------------------------------------------

def test_failed_module_init_reraises_on_every_call():
    interp = build_interp("""
let BAD = missing_function();

fn main() -> int { 1 }
""")
    with pytest.raises(InterpError) as first:
        interp.call("main", [])
    with pytest.raises(InterpError) as second:
        interp.call("main", [])
    # Same cached failure, re-raised — not a silent skip that would let
    # main run without its module constants.
    assert str(first.value) == str(second.value)


# ---------------------------------------------------------------------------
# Finding 7: LLVM symbol sanitization is injective (no silent collisions)
# ---------------------------------------------------------------------------

def test_sanitize_is_injective_and_deterministic():
    from metaxu.compiler.codegen_llvm import _sanitize, _sanitize_reset
    _sanitize_reset()
    first = [_sanitize("f.g"), _sanitize("f_g"), _sanitize("f$g")]
    assert len(set(first)) == 3          # injective
    assert first[0] == "f_g"             # first claimant keeps the plain form
    assert all(s == _sanitize(n) for s, n in zip(first, ["f.g", "f_g", "f$g"]))
    _sanitize_reset()
    assert [_sanitize("f.g"), _sanitize("f_g"), _sanitize("f$g")] == first


def test_colliding_module_symbols_emit_distinct_functions(tmp_path):
    # Two module functions whose names differ only in special characters
    # ('m.n.f' vs 'm_n.f' after namespacing) must land on distinct LLVM
    # symbols instead of silently merging.
    for rel, src in {
        "m/n.mx": "fn f() -> int { 1 }\n",
        "m_n.mx": "fn f() -> int { 2 }\n",
        "main.mx": """
from m.n import f;
import m_n;

fn main() -> int { f() + m_n.f() }
""",
    }.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src)
    root = tmp_path / "main.mx"
    ir = emit_llvm_from_source(root.read_text(), file_path=str(root))
    defines = [l for l in ir.splitlines() if l.startswith("define ")]
    names = [l.split("@", 1)[1].split("(", 1)[0] for l in defines]
    assert len(names) == len(set(names)), f"duplicate symbols: {names}"


# ---------------------------------------------------------------------------
# Finding 8: promote_matrix — the DOCUMENTED example-06 semantics
# (mat.matmul(vec) is matrix-vector multiplication) — engine parity
# ---------------------------------------------------------------------------

MATRIX_PROMOTION_SRC = """
fn colsum(m: vector[vector[float,3],3]) -> float {
    m[0][0] + m[1][0] + m[2][0]
}

fn main() -> int {
    let v = vector[float,3](1.0, 2.0, 3.0);
    if colsum(v) == 6.0 { 42 } else { 7 }
}
"""


def test_flat_vector_for_matrix_param_is_column_promotion():
    # examples/06's `let transformed = mat.matmul(vec);  # Matrix-vector
    # multiplication` pins the intent: a flat vector where a matrix is
    # required is the DOCUMENTED Mx1-column embedding, not an error (the
    # promotion is the boundary that makes matmul(vec) well-shaped; the
    # end-to-end matmul is pinned by the example-06 gate and
    # test_simd_example). Each scalar becomes a one-element row: 1+2+3.
    result, _ = run_main(MATRIX_PROMOTION_SRC)
    assert result == 42


@needs_clang
def test_native_matrix_promotion_matches_interp(tmp_path):
    # Both engines implement the same promotion (mir_interp promote_matrix
    # / codegen_llvm mx_fvec_promote): pin the parity differentially.
    native_matches_interp(MATRIX_PROMOTION_SRC, tmp_path)


# ---------------------------------------------------------------------------
# Finding 9: f-string expression segments balance braces
# ---------------------------------------------------------------------------

def test_fstring_expression_with_nested_braces_parses():
    result, _ = run_main("""
fn pick(n: int) -> string {
    f"got {if n > 0 { 1 } else { 2 }}!"
}

fn main() -> string { pick(5) }
""")
    assert result == "got 1!"


def test_fstring_unterminated_brace_still_errors():
    with pytest.raises(CompileError, match="unterminated"):
        build_context_from_source('fn main() -> string { f"broken {x" }')


# ---------------------------------------------------------------------------
# Finding 10: invalid UTF-8 in a C string is a clear InterpError
# ---------------------------------------------------------------------------

def test_invalid_utf8_c_string_is_loud_interp_error():
    src = """
extern "C" {
    type FILE;
    fn fopen(filename: *char, mode: *char) -> *FILE;
}

fn main() -> int {
    unsafe {
        let name = vector[int,2](255, 0).as_ptr();
        let mode = "r".as_ptr();
        fopen(name, mode);
        0
    }
}
"""
    with pytest.raises(InterpError, match="invalid UTF-8 in C string"):
        run_main(src)
