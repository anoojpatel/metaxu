"""Name-resolution precedence: user functions win over builtins.

The rule (docs/name_precedence.md), enforced identically by the MIR
interpreter and the LLVM backend:

* PLAIN call ``f(a, b)``     -- a user module function named ``f`` WINS over
  a builtin of the same name; the builtin is the fallback when no user
  function exists.  Locals bound to closures still shadow both.
* METHOD position ``x.m(a)`` -- a user *impl* of ``m`` for the receiver's
  type wins; otherwise the runtime BUILTIN ``m`` wins; a plain top-level
  ``fn m`` is only the last resort (UFCS).  A top-level function is not a
  method of anything, so it must not hijack ``x.len()`` for every receiver.
* RESERVED -- a user function whose name starts with ``__`` is a loud
  compile error: that namespace holds every symbol the compiler generates
  and emits calls to (``__trait$``/``__static$``/``__impl$``/
  ``__module_init``/``__builtin$`` and the ``__vec_*``/``__index_*``/
  ``__list_*`` intrinsics), so it can never be shadowed.

Every test goes through parsed source (parse -> ... -> interpreter/native),
never hand-built HIR/MIR.
"""
from __future__ import annotations

import shutil

import pytest

from metaxu.errors import CompileError
from metaxu.compiler.hir import BUILTIN_CALL_PREFIX, HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.llvm_run import compile_and_run
from metaxu.compiler.codegen_llvm import emit_llvm
from metaxu.compiler.mir_interp import InterpError, MirInterpreter, UNIT
from metaxu.compiler.pipeline import build_context_from_source, run_pipeline_ctx


needs_clang = pytest.mark.skipif(
    shutil.which("clang") is None, reason="clang is not installed")


def _mir(source: str, *, file_path: str = "<mem>", strict: bool = True):
    ctx = build_context_from_source(source, file_path=file_path)
    if strict:
        run_pipeline_ctx(ctx)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    return lower_hir_to_mir(hir)


def run_main(source: str, entry: str = "main", *, file_path: str = "<mem>",
             strict: bool = True):
    """Full front end -> MIR interpreter; returns (result, printed lines)."""
    interp = MirInterpreter()
    interp.load(_mir(source, file_path=file_path, strict=strict))
    prints: list[str] = []
    interp.register_builtin(
        "print", lambda *a: (prints.append(" ".join(str(x) for x in a)), UNIT)[1])
    interp.register_builtin(
        "println", lambda *a: (prints.append(" ".join(str(x) for x in a)), UNIT)[1])
    return interp.call(entry, []), prints


def assert_native_matches_interp(source: str, tmp_path, entry: str = "main"):
    """Differential: clang-compiled (exit code, stdout) == interpreter."""
    result, prints = run_main(source, entry)
    expected_out = "".join(line + "\n" for line in prints)
    ir = emit_llvm(_mir(source))
    exit_code, stdout = compile_and_run(ir, entry, workdir=str(tmp_path))
    assert stdout == expected_out
    if result is not UNIT and isinstance(result, (bool, int)):
        assert exit_code == int(result) % 256
    return ir


# ---------------------------------------------------------------------------
# 1. A user function shadows the same-named builtin (the motivating bug)
# ---------------------------------------------------------------------------

# examples/collections.mx's shape: a user `push` over a user list type.
# Before the precedence flip this call reached the Vec builtin and died with
# "push: expected a Vec receiver, got 'List'".
USER_PUSH = """
struct List {
    data: Vec,
    count: int
}

fn push(list: List, item: int) -> List {
    List { data: list.data, count: list.count + item }
}

fn main() -> int {
    let l = List { data: Vec.new(), count: 1 };
    let l2 = push(l, 41);
    l2.count
}
"""


def test_user_push_wins_over_vec_builtin():
    result, _ = run_main(USER_PUSH)
    assert result == 42


def test_user_len_and_to_string_win_over_builtins():
    result, prints = run_main("""
fn len(n: int) -> int { n + 100 }

fn to_string(n: int) -> string { "user:" + int_to_str(n) }

fn main() -> int {
    print(to_string(7));
    len(5)
}
""")
    assert result == 105
    assert prints == ["user:7"]


def test_print_statement_always_reaches_the_builtin():
    """`print(...)` is a GRAMMAR PRODUCTION (`print` is a lexer keyword),
    not an ordinary call, so it is bound to the builtin at HIR-build time
    and no module function can capture it."""
    mir = _mir("""
fn main() -> int {
    print(1);
    0
}
""")
    main = next(f for f in mir if f.name == "main")
    callees = [op[2][1] for b in main.blocks for op in b.ops
               if op[0] == "let" and op[2][0] == "call"]
    assert callees == [f"{BUILTIN_CALL_PREFIX}print"]


def test_unary_operators_always_reach_their_builtins():
    """`-x` / `!x` lower to synthesized `neg`/`not` calls; they carry the
    builtin marker so a module function named `neg`/`not` cannot capture
    the operators."""
    mir = _mir("""
fn neg(n: int) -> int { 999 }

fn main() -> int {
    let x = 5;
    0 - 0 + (0 - x)
}
""")
    main = next(f for f in mir if f.name == "main")
    for b in main.blocks:
        for op in b.ops:
            if op[0] == "let" and op[2][0] == "call":
                assert op[2][1].startswith(BUILTIN_CALL_PREFIX), op


def test_builtin_name_table_matches_the_interpreter():
    """BUILTIN_FUNCTION_NAMES (used by the module resolver to bind a
    module's unqualified builtin calls) must stay equal to the
    interpreter's registered bare builtins."""
    from metaxu.compiler.hir import BUILTIN_FUNCTION_NAMES
    registered = {n for n in MirInterpreter()._builtins
                  if "." not in n and not n.startswith("__")}
    assert registered == set(BUILTIN_FUNCTION_NAMES)


@needs_clang
def test_user_shadowing_builtins_native_differential(tmp_path):
    """Same precedence natively: the two engines must not diverge."""
    assert_native_matches_interp("""
fn push(a: int, b: int) -> int { a * 1000 + b }

fn len(n: int) -> int { n + 100 }

fn main() -> int {
    print(push(3, 41));
    print(len(5));
    let v = Vec.new();
    v.push(7);
    v.push(8);
    print(v.len());
    0
}
""", tmp_path)


@needs_clang
def test_shadowed_builtin_emits_a_real_call_to_the_user_function(tmp_path):
    """The native module must DEFINE and CALL @mx_push, not the runtime."""
    ir = assert_native_matches_interp("""
fn push(a: int, b: int) -> int { a + b }

fn main() -> int {
    print(push(1, 2));
    0
}
""", tmp_path)
    assert "define i64 @mx_push(" in ir
    assert "call i64 @mx_push(" in ir


# ---------------------------------------------------------------------------
# 2. The builtin is still reachable when no user function shadows it
# ---------------------------------------------------------------------------

def test_builtin_still_wins_when_no_user_function_exists():
    result, prints = run_main("""
fn main() -> int {
    let v = Vec.new();
    push(v, 1);
    push(v, 2);
    push(v, 3);
    print(to_string(len(v)));
    len(v)
}
""")
    assert result == 3
    assert prints == ["3"]


@needs_clang
def test_builtin_plain_calls_native_differential(tmp_path):
    assert_native_matches_interp("""
fn main() -> int {
    let v = Vec.new();
    push(v, 10);
    push(v, 20);
    print(len(v));
    print(to_string(99));
    len(v)
}
""", tmp_path)


# ---------------------------------------------------------------------------
# 3. Reserved names (__*) are rejected loudly, never silently overridden
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "__index_get",       # compiler intrinsic (indexing)
    "__vec_lit",         # compiler intrinsic (vector literal)
    "__list_lit",        # compiler intrinsic (list literal)
    "__module_init",     # synthesized module-constant initializer
    "__trait",           # dispatch prefix stem
    "__builtin",         # method-position builtin marker stem
    "__anything_at_all",
])
def test_reserved_function_names_are_rejected(name):
    with pytest.raises(CompileError) as exc:
        build_context_from_source(
            f"fn {name}(a: int) -> int {{ a }}\nfn main() -> int {{ 0 }}\n")
    assert exc.value.error_type == "ReservedNameError"
    assert name in str(exc.value)


def test_reserved_name_rejected_inside_an_impl_block():
    with pytest.raises(CompileError) as exc:
        build_context_from_source("""
struct Box { v: int }

trait Peek {
    fn __peek(self) -> int
}

implement Peek for Box {
    fn __peek(self) -> int { self.v }
}

fn main() -> int { 0 }
""")
    assert exc.value.error_type == "ReservedNameError"


def test_reserved_name_rejected_in_an_imported_module(tmp_path, monkeypatch):
    """Imported files are gated too (before module renaming)."""
    (tmp_path / "sneaky.mx").write_text("""
export { helper }

fn __module_init() -> int { 1 }

fn helper() -> int { 2 }
""")
    monkeypatch.setenv("METAXU_STD_PATH", str(tmp_path))
    with pytest.raises(CompileError) as exc:
        build_context_from_source("""
from std.sneaky import helper;

fn main() -> int { helper() }
""")
    assert exc.value.error_type == "ReservedNameError"


def test_ordinary_single_underscore_names_are_fine():
    result, _ = run_main("""
fn _private_helper(a: int) -> int { a + 1 }

fn main() -> int { _private_helper(41) }
""")
    assert result == 42


# ---------------------------------------------------------------------------
# 4. Method position: builtin wins over a plain function, impl wins over both
# ---------------------------------------------------------------------------

def test_method_position_keeps_the_builtin_over_a_plain_function():
    """`v.len()` is the Vec's length even though a plain `fn len` exists.

    Direction 1 of the method-position rule: a top-level function is not a
    method, so it must not hijack every receiver in the program.
    """
    result, _ = run_main("""
fn len(n: int) -> int { n + 100 }

fn main() -> int {
    let v = Vec.new();
    v.push(1);
    v.push(2);
    v.len() * 1000 + len(3)
}
""")
    assert result == 2 * 1000 + 103


def test_method_position_lowers_builtin_methods_behind_the_marker():
    """The MIR seam itself: `v.len()` carries the __builtin$ marker, the
    plain `len(3)` call does not (that is what keeps the two positions
    distinguishable)."""
    mir = _mir("""
fn len(n: int) -> int { n + 100 }

fn main() -> int {
    let v = Vec.new();
    v.push(1);
    v.len() + len(3)
}
""")
    main = next(f for f in mir if f.name == "main")
    callees = [op[2][1] for b in main.blocks for op in b.ops
               if op[0] == "let" and op[2][0] == "call"]
    assert f"{BUILTIN_CALL_PREFIX}len" in callees      # v.len()
    assert f"{BUILTIN_CALL_PREFIX}push" in callees     # v.push(1)
    assert "len" in callees                            # plain len(3)


def test_method_position_user_impl_wins_over_the_builtin():
    """Direction 2: a user impl for the receiver type beats the builtin,
    exactly as trait dispatch has always behaved."""
    result, _ = run_main("""
struct Ruler { marks: int }

trait Measure {
    fn len(self) -> int
}

implement Measure for Ruler {
    fn len(self) -> int { self.marks * 10 }
}

fn main() -> int {
    let r = Ruler { marks: 4 };
    let v = Vec.new();
    v.push(1);
    r.len() + v.len()
}
""")
    assert result == 41


def test_method_position_falls_back_to_a_plain_function_for_non_builtins():
    """UFCS on a COMPUTED receiver is unchanged: a method name that is
    neither a trait method nor a builtin still lowers to a bare call of
    the plain function with the receiver first (so, under the flipped
    plain-call precedence, the user function is what runs)."""
    result, _ = run_main("""
fn identity(n: int) -> int { n }

fn double(n: int) -> int { n * 2 }

fn main() -> int {
    identity(21).double()
}
""")
    assert result == 42


def test_method_position_builtin_error_is_still_loud():
    """A builtin method on a receiver it rejects errors as before (no
    silent fallthrough to a same-named plain function)."""
    with pytest.raises(InterpError) as exc:
        run_main("""
fn pop(n: int) -> int { n }

fn main() -> int {
    let x = 5;
    x.pop()
}
""")
    assert "pop" in str(exc.value)


def test_wrapper_function_named_after_its_own_builtin_method_terminates():
    """std/math.mx's shape: `fn sqrt(x) { x.sqrt() }`.  Under the flipped
    plain-call precedence this would be infinite recursion if method
    position followed the same rule; it does not."""
    result, _ = run_main("""
fn sqrt(x: float) -> float { x.sqrt() }

fn main() -> float { sqrt(9.0) }
""")
    assert result == 3.0


@needs_clang
def test_method_position_native_differential(tmp_path):
    assert_native_matches_interp("""
fn len(n: int) -> int { n + 100 }

fn main() -> int {
    let v = Vec.new();
    v.push(1);
    v.push(2);
    v.push(3);
    print(v.len());
    print(len(3));
    v.len()
}
""", tmp_path)


# ---------------------------------------------------------------------------
# 5. std/ modules keep working (their names are namespaced, and their own
#    builtin-shadowing helpers still resolve correctly)
# ---------------------------------------------------------------------------

def test_std_vec_helpers_still_resolve_with_a_local_shadowing_push():
    """std.vec's helpers push into their own Vecs while the importing
    program defines its own `push` and `len`."""
    result, _ = run_main("""
from std.vec import of3, sum, map;

fn push(a: int, b: int) -> int { a * 100 + b }

fn len(n: int) -> int { n + 1000 }

fn main() -> int {
    let v = of3(1, 2, 3);
    let doubled = map(v, fn(x) -> x * 2);
    sum(doubled) + push(1, 2) + len(0)
}
""")
    assert result == 12 + 102 + 1000


def test_std_math_wrappers_resolve_through_the_module_namespace():
    """std.math exports sqrt/sin/cos, whose names collide with builtins;
    imported they are `std.math.sqrt`, and the wrapper bodies' `x.sqrt()`
    still reaches the builtin."""
    result, _ = run_main("""
from std.math import sqrt, abs, powi;

fn main() -> float {
    sqrt(16.0) + abs(0.0 - 1.0) + powi(2, 3)
}
""")
    assert result == 4.0 + 1.0 + 8


def test_std_math_as_entry_file_does_not_self_recurse(monkeypatch):
    """Compiled AS THE ENTRY FILE, std/math.mx's `sqrt` keeps its bare name
    and shadows the builtin for plain calls — its own `x.sqrt()` body must
    still reach the builtin (otherwise: infinite recursion)."""
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[4]
    source = (repo_root / "std" / "math.mx").read_text()
    interp = MirInterpreter()
    interp.load(_mir(source, file_path=str(repo_root / "std" / "math.mx")))
    assert interp.call("sqrt", [9.0]) == 3.0
    assert interp.call("cos", [0.0]) == 1.0
