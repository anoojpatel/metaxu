"""HIR lowering has no silent "unknown construct" path.

Eight bugs of one identical shape were found here one accident at a time:
`hir.py`'s `_from_orig_expr` returned None for an AST node class it did not
handle, callers skipped the None, and the construct VANISHED — if-let,
while-let, `unsafe { }`, `@mut e`, list literals, `for`, `e as T`, early
`return` and struct-field initializers each compiled to a program that
quietly did less than the source said. `_convert_pattern` had the mirror
version: an unrecognized pattern became a match-anything wildcard, so the arm
matched everything and every later arm was dead code.

This file pins the fix as a property rather than as a list of anecdotes:

1. every AST node class is triaged into EXACTLY ONE bucket
   (`AST_NODE_TRIAGE`), so a newly added node class cannot be forgotten —
   this test fails until it is classified;
2. each bucket raises the right kind of error, never returns None;
3. the constructs implemented during the triage round really work, through
   parsed source into the interpreter;
4. the constructs that are loudly unsupported really are loud;
5. across the whole shipped corpus (examples + std) HIR lowering never drops
   an expression and never degrades a pattern to a wildcard.

All tests go through parsed source per the repo convention.
"""
from __future__ import annotations

import glob
import inspect
import os

import pytest

import metaxu.metaxu_ast as fast
import metaxu.unsafe_ast as uast
import metaxu.extern_ast as east
import metaxu.decorator_ast as dast
from metaxu.compiler.hir import (
    AST_NODE_TRIAGE,
    HIRBuilder,
    HIRCompilerBug,
    LOWERED,
    NOT_AN_EXPRESSION,
    PATTERN_TRIAGE,
    UNSUPPORTED,
    UnsupportedConstruct,
)
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import MirInterpreter, UNIT
from metaxu.compiler.pipeline import build_context_from_source, run_pipeline_ctx

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

AST_MODULES = (fast, uast, east, dast)
_AST_MODULE_NAMES = {m.__name__ for m in AST_MODULES}


def all_ast_node_classes() -> dict[str, type]:
    """Every Node subclass defined by the AST modules, by name.

    Keyed by name because metaxu_ast defines two `WildcardPattern` classes
    (a value-level Pattern and a TypePattern); the later definition shadows
    the former, and HIR matches by class name where that matters.
    """
    out: dict[str, type] = {}
    for mod in AST_MODULES:
        for name, obj in vars(mod).items():
            if (inspect.isclass(obj)
                    and obj.__module__ in _AST_MODULE_NAMES
                    and issubclass(obj, fast.Node)):
                out.setdefault(name, obj)
    return out


def run_main(source: str, file_path: str = "<mem>"):
    """Full strict pipeline, then execute main(); returns (result, prints)."""
    ctx = build_context_from_source(source, file_path=file_path)
    run_pipeline_ctx(ctx)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    prints: list[str] = []
    interp.register_builtin(
        "print", lambda *a: (prints.append(" ".join(str(x) for x in a)), UNIT)[1])
    return interp.call("main", []), prints


def build_hir(source: str, file_path: str = "<mem>"):
    ctx = build_context_from_source(source, file_path=file_path)
    return HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)


# ---------------------------------------------------------------------------
# 1. Triage completeness: exactly one bucket per AST node class
# ---------------------------------------------------------------------------

def test_every_ast_node_class_is_triaged():
    """A newly added AST node class must be classified before it can be used.

    This is the whole point of the table: the old failure mode was a node
    class nobody had thought about reaching lowering and silently vanishing.
    """
    classes = set(all_ast_node_classes())
    missing = sorted(classes - set(AST_NODE_TRIAGE))
    assert not missing, (
        "AST node classes with no triage bucket — add each to "
        f"AST_NODE_TRIAGE in hir.py: {missing}")


def test_triage_table_has_no_stale_entries():
    classes = set(all_ast_node_classes())
    stale = sorted(set(AST_NODE_TRIAGE) - classes)
    assert not stale, f"AST_NODE_TRIAGE names classes that no longer exist: {stale}"


def test_every_triage_entry_has_a_valid_bucket_and_a_reason():
    for name, entry in AST_NODE_TRIAGE.items():
        bucket, reason = entry
        assert bucket in (LOWERED, NOT_AN_EXPRESSION, UNSUPPORTED), (name, bucket)
        assert reason and reason.strip(), f"{name} has an empty reason"


def test_buckets_are_disjoint_and_cover_the_table():
    """'Exactly one bucket' is structural here (a dict maps each name to one
    entry) — this pins that the three buckets are all non-empty and that the
    partition really is a partition."""
    by_bucket: dict[str, list[str]] = {LOWERED: [], NOT_AN_EXPRESSION: [],
                                       UNSUPPORTED: []}
    for name, (bucket, _r) in AST_NODE_TRIAGE.items():
        by_bucket[bucket].append(name)
    assert sum(len(v) for v in by_bucket.values()) == len(AST_NODE_TRIAGE)
    for bucket, names in by_bucket.items():
        assert names, f"bucket {bucket} is empty"
        assert len(names) == len(set(names))


def test_pattern_triage_names_real_classes_with_valid_buckets():
    classes = set(all_ast_node_classes())
    stale = sorted(set(PATTERN_TRIAGE) - classes)
    assert not stale, f"PATTERN_TRIAGE names classes that no longer exist: {stale}"
    for name, (bucket, reason) in PATTERN_TRIAGE.items():
        assert bucket in (LOWERED, UNSUPPORTED), (name, bucket)
        assert reason and reason.strip(), f"{name} has an empty reason"


# ---------------------------------------------------------------------------
# 2. The fallback is loud for every bucket, and never returns None
# ---------------------------------------------------------------------------

class _FakeNode(fast.Node):
    """A node class the triage table has deliberately never heard of."""


def _bare_builder() -> tuple[HIRBuilder, object]:
    ctx = build_context_from_source("fn main() -> int { 0 }", file_path="demo.mx")
    b = HIRBuilder(ctx.tables, id_map=ctx.id_map)
    b.build(ctx.frozen_root)          # populates _root_file and side tables
    return b, ctx.frozen_root


@pytest.mark.parametrize(
    "cls_name",
    sorted(n for n, (b, _r) in AST_NODE_TRIAGE.items() if b == UNSUPPORTED))
def test_unsupported_bucket_raises_a_named_diagnostic(cls_name):
    builder, root = _bare_builder()
    node = object.__new__(all_ast_node_classes()[cls_name])
    with pytest.raises(UnsupportedConstruct) as exc:
        builder._unlowerable(node, root)
    assert cls_name in str(exc.value)
    assert "not supported" in str(exc.value)


@pytest.mark.parametrize(
    "cls_name",
    sorted(n for n, (b, _r) in AST_NODE_TRIAGE.items() if b == NOT_AN_EXPRESSION))
def test_not_an_expression_bucket_raises_a_compiler_bug(cls_name):
    builder, root = _bare_builder()
    node = object.__new__(all_ast_node_classes()[cls_name])
    with pytest.raises(HIRCompilerBug) as exc:
        builder._unlowerable(node, root)
    assert cls_name in str(exc.value)
    assert "not an expression" in str(exc.value)


def test_untriaged_node_class_raises_and_names_the_table():
    builder, root = _bare_builder()
    with pytest.raises(HIRCompilerBug) as exc:
        builder._unlowerable(_FakeNode(), root)
    assert "AST_NODE_TRIAGE" in str(exc.value)


def test_untriaged_pattern_node_raises_and_names_the_table():
    builder, _root = _bare_builder()
    with pytest.raises(HIRCompilerBug) as exc:
        builder._convert_pattern(_FakeNode())
    assert "PATTERN_TRIAGE" in str(exc.value)


# ---------------------------------------------------------------------------
# 3. Constructs implemented during the triage round: they really work
# ---------------------------------------------------------------------------

def test_inline_handle_block_installs_a_real_handler():
    """`handle SUBJECT { perform Op(p) => body }` had NO lowering: the whole
    handler vanished and the expression evaluated to unit (this is how
    examples/06's with_simd ran nothing)."""
    result, _ = run_main("""
effect Log {
    fn emit(msg: int) -> int;
}
fn body() -> int performs Log {
    let a = perform Log.emit(1);
    a + 6
}
fn main() -> int {
    handle body() {
        perform Log.emit(m) => { resume(m * 10) }
    }
}
""")
    assert result == 16


def test_inline_handle_block_multi_arg_arm():
    result, _ = run_main("""
effect Pair {
    fn combine(a: int, b: int) -> int;
}
fn body() -> int performs Pair {
    perform Pair.combine(3, 4)
}
fn main() -> int {
    handle body() {
        perform Pair.combine(x, y) => { resume(x * y) }
    }
}
""")
    assert result == 12


def test_inline_handle_block_rejects_a_non_operation_arm():
    with pytest.raises(UnsupportedConstruct, match="operation pattern"):
        build_hir("""
fn body() -> int { 1 }
fn main() -> int {
    handle body() {
        1 + 2 => 3
    }
}
""")


def test_indirect_call_of_a_lambda_literal():
    """`(fn(x) -> x*2)(3)` is a CallExpression (computed callee); it had no
    lowering, so the call vanished and the expression was unit."""
    assert run_main("""
fn main() -> int {
    (fn(x: int) -> int { x * 2 })(3)
}
""")[0] == 6


def test_indirect_call_of_a_stored_closure():
    assert run_main("""
fn main() -> int {
    let v = Vec.new();
    v.push(fn(x: int) -> int { x + 1 });
    v[0](41)
}
""")[0] == 42


def test_borrow_expression_evaluates_to_the_borrowed_value():
    """`borrow x` had no lowering: `let y = borrow x;` bound nothing."""
    assert run_main("""
fn main() -> int {
    let x = 5;
    let y = borrow x;
    y
}
""")[0] == 5


def test_address_of_a_non_variable_evaluates_to_the_referenced_value():
    """`&x` on a bare name already lowered to the value; `&x.f` built an
    AddressOf, which had no lowering and silently became nothing."""
    assert run_main("""
struct P { a: int }
fn main() -> int {
    let p = P { a: 7 };
    let q = &p.a;
    q
}
""")[0] == 7


@pytest.mark.parametrize("variant,expected",
                         [("Red", 1), ("Green", 2), ("Blue", 3)])
def test_dotted_enum_variant_in_expression_and_pattern(variant, expected):
    """`Color.Red` is a nullary variant in BOTH positions. In expression
    position it used to build a field read of an undefined variable `Color`;
    in pattern position it fell into the wildcard fallback, so the FIRST arm
    matched every colour and the rest were dead code."""
    assert run_main(f"""
enum Color {{ Red, Green, Blue }}
fn main() -> int {{
    let c = Color.{variant};
    match c {{
        Color.Red => 1,
        Color.Green => 2,
        Color.Blue => 3
    }}
}}
""")[0] == expected


def test_ordinary_field_reads_are_untouched_by_the_variant_rule():
    assert run_main("""
struct P { a: int, b: int }
fn main() -> int {
    let p = P { a: 7, b: 9 };
    p.b
}
""")[0] == 9


@pytest.mark.parametrize("decl", [
    "fn helper(x: int) -> int { x + 1 }",
    "struct Inner { a: int }",
    "enum Inner { A, B }",
    "import std.math;",
])
def test_declarations_in_statement_position_lower_to_unit(decl):
    """A declaration inside a block is realized by another pass; as a
    statement it contributes nothing. That must be an EXPLICIT unit, not a
    silently skipped None."""
    assert run_main(f"""
fn main() -> int {{
    {decl}
    3
}}
""")[0] == 3


# ---------------------------------------------------------------------------
# 4. Constructs that are loudly unsupported
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,source,needle", [
    ("comptime block", """
fn main() -> int { comptime { let x = 1; } 0 }
""", "ComptimeBlock"),
    ("comptime fn", """
comptime fn sz<T>() -> int { 1 }
fn main() -> int { 0 }
""", "comptime fn"),
    ("to_device", """
fn main() -> int { let x = 1; to_device(x); 0 }
""", "ToDevice"),
    ("from_device", """
fn main() -> int { let x = 1; from_device(x); 0 }
""", "FromDevice"),
    ("uncalled generic instantiation", """
fn ident<T>(x: T) -> T { x }
fn main() -> int { let f = ident<int>; 0 }
""", "GenericInstance"),
    ("bare comprehension", """
fn dbl(x: int) -> int { x * 2 }
fn main() -> int { let xs = [1,2,3]; len(dbl(y) for y in xs) }
""", "Comprehension"),
])
def test_unsupported_expression_constructs_are_loud(label, source, needle):
    with pytest.raises(UnsupportedConstruct) as exc:
        build_hir(source, file_path="demo.mx")
    assert needle in str(exc.value)


@pytest.mark.parametrize("label,arm,needle", [
    ("list", "[] => 1", "ListLiteral"),
    # `(1, 2) => 1` is a SUPPORTED tuple pattern now (see test_tuples.py);
    # the shapes that remain unsupported are the ones with no tuple to
    # destructure, kept here so they cannot regress to a silent catch-all.
    ("unit tuple", "() => 1", "a tuple pattern needs two or more"),
    ("range", "1..5 => 1", "RangeExpression"),
    ("arithmetic", "1 + 2 => 1", "BinaryOperation"),
    ("lambda", "fn(y) -> y * y => 1", "LambdaExpression"),
    ("unknown constructor", "Weird(a) => 1", "not a known enum variant"),
    ("field value", "p.a => 1", "not a known enum variant"),
])
def test_unsupported_patterns_are_loud(label, arm, needle):
    """Each of these used to degrade to a match-anything wildcard, silently
    making the arm win for every value and every later arm dead."""
    with pytest.raises(UnsupportedConstruct) as exc:
        build_hir(f"""
struct P {{ a: int }}
fn main() -> int {{
    let p = P {{ a: 1 }};
    let x = 3;
    match x {{
        {arm},
        _ => 9
    }}
}}
""", file_path="demo.mx")
    assert needle in str(exc.value)


def test_unsupported_if_let_pattern_names_the_construct():
    with pytest.raises(UnsupportedConstruct, match="if let"):
        build_hir("""
fn main() -> int {
    let v = 3;
    if let x + 1 = v { return 1 }
    return 0
}
""")


def test_unsupported_while_let_pattern_names_the_construct():
    with pytest.raises(UnsupportedConstruct, match="while let"):
        build_hir("""
fn main() -> int {
    let v = 3;
    while let x + 1 = v { return 1 }
    return 0
}
""")


def test_unsupported_unary_operator_is_loud():
    """An unrecognized unary operator used to drop the whole expression.

    `~` is a REAL operator now (bitwise complement, see test_bitwise.py), so
    the loud path is checked with an operator the language does not have."""
    builder, root = _bare_builder()
    node = fast.UnaryOperation("#", fast.Literal(1))
    with pytest.raises(UnsupportedConstruct, match="unary operator"):
        builder._from_orig_expr(node, root)


def test_the_three_real_unary_operators_lower():
    """`-`, `!` and `~` all lower to their builtin calls."""
    builder, root = _bare_builder()
    for op, callee in (("-", "neg"), ("!", "not"), ("~", "bnot")):
        he = builder._from_orig_expr(fast.UnaryOperation(op, fast.Literal(1)),
                                     root)
        assert he is not None and he.callee.endswith(callee)


# ---------------------------------------------------------------------------
# 5. Corpus property: nothing is dropped, nothing degrades to a wildcard
# ---------------------------------------------------------------------------

def _corpus_files() -> list[str]:
    return (sorted(glob.glob(os.path.join(REPO_ROOT, "examples", "*.mx")))
            + sorted(glob.glob(os.path.join(REPO_ROOT, "std", "*.mx"))))


@pytest.mark.parametrize("path", _corpus_files(),
                         ids=lambda p: os.path.basename(p))
def test_corpus_lowers_without_dropping_or_wildcarding(path, monkeypatch):
    """The invariant, checked on real shipped code.

    - `_from_orig_expr` returns None only for a None input (the documented
      contract): anything else would be a construct disappearing;
    - `_convert_pattern` answers a wildcard only for a source wildcard:
      anything else would make an arm match values it must not.
    """
    dropped: list[str] = []
    wildcarded: list[str] = []

    real_expr = HIRBuilder._from_orig_expr
    real_pat = HIRBuilder._convert_pattern

    def watched_expr(self, orig, frozen_ctx):
        result = real_expr(self, orig, frozen_ctx)
        if result is None and orig is not None:
            dropped.append(type(orig).__name__)
        return result

    def watched_pat(self, p):
        result = real_pat(self, p)
        source_wildcard = (
            p is None
            or type(p).__name__ == "WildcardPattern"
            or (isinstance(p, fast.Variable) and getattr(p, "name", None) == "_")
            or (isinstance(p, fast.QualifiedName)
                and [str(x) for x in (getattr(p, "parts", []) or [])] == ["_"]))
        if result.kind == "wildcard" and not source_wildcard:
            wildcarded.append(type(p).__name__)
        return result

    monkeypatch.setattr(HIRBuilder, "_from_orig_expr", watched_expr)
    monkeypatch.setattr(HIRBuilder, "_convert_pattern", watched_pat)

    source = open(path).read()
    ctx = build_context_from_source(source, file_path=path)
    HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)

    assert not dropped, f"{os.path.basename(path)}: silently dropped {sorted(set(dropped))}"
    assert not wildcarded, (
        f"{os.path.basename(path)}: patterns silently degraded to wildcards: "
        f"{sorted(set(wildcarded))}")


# ---------------------------------------------------------------------------
# The removed `spawn(e)` keyword form
# ---------------------------------------------------------------------------
# `spawn` was a lexer keyword with an expression production but no semantics
# behind it (SpawnExpression triaged UNSUPPORTED), and the keyword caused a
# real bug: a `spawn(f)` handler case parsed as a SpawnExpression and could
# never match a perform. The keyword is gone: `spawn` is an ordinary
# identifier, threads go through the Thread effect, and calling an undefined
# `spawn(..)` gets a compile error pointing at the effect route.

def test_bare_spawn_is_an_undefined_function_with_a_routing_hint():
    from metaxu.compiler.frozen_borrow_checker import TypeCheckError
    with pytest.raises(TypeCheckError) as exc:
        ctx = build_context_from_source("""
fn work() -> int { 1 }
fn main() -> int { spawn(work()); 0 }
""")
        run_pipeline_ctx(ctx)
    msg = str(exc.value)
    assert "undefined function 'spawn'" in msg
    assert "perform Thread.spawn" in msg


def test_spawn_is_an_ordinary_identifier_now():
    """The keyword is freed: users can define and call their own `spawn`,
    and a variable may be named spawn."""
    result, _ = run_main("""
fn spawn(n: int) -> int { n * 2 }

fn main() -> int {
    let spawned = spawn(21);
    spawned
}
""")
    assert result == 42
