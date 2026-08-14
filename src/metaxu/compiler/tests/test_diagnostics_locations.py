"""Source locations in diagnostics.

Every loud compile-time diagnostic must name `file:line:column` and show the
offending source line with a caret. These tests pin the locations for a
representative diagnostic from each class, using fixtures written so the
expected line/column is unambiguous (the construct under test appears exactly
once, on a line of its own, well past line 1).

They also pin the two things location plumbing gets wrong most easily:
  * a multi-line file must report the RIGHT line, not always line 1;
  * a node built by a parser helper rather than by a grammar production
    still gets a location (inherited from the enclosing production).
"""
from __future__ import annotations

import pytest

from metaxu.compiler.desugar import CoherenceError
from metaxu.compiler.frozen_borrow_checker import BorrowCheckError, TypeCheckError
from metaxu.compiler.hir import HIRBuilder, UnsupportedConstruct
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import InterpError, MirInterpreter
from metaxu.compiler.mutaxu_ast import build_frozen_ast_with_map, dump_ast_json
from metaxu.compiler.pipeline import build_context_from_source, run_pipeline_from_source
from metaxu.compiler.shared_parser import shared_parser
from metaxu.errors import CompileError, SourceLocation, source_excerpt
import metaxu.metaxu_ast as fast


def write(tmp_path, name: str, source: str) -> str:
    """Write `source` to a real file and return its path (diagnostics quote
    the file, so fixtures live on disk)."""
    path = tmp_path / name
    path.write_text(source)
    return str(path)


def compile_expecting(tmp_path, name: str, source: str, exc_type):
    path = write(tmp_path, name, source)
    with pytest.raises(exc_type) as excinfo:
        run_pipeline_from_source(source, file_path=path)
    return path, str(excinfo.value)


# ---------------------------------------------------------------------------
# The parser attaches locations at all
# ---------------------------------------------------------------------------

def test_parser_attaches_line_and_column_to_every_node():
    src = (
        "fn main() -> int {\n"      # line 1
        "    let x = 1\n"           # line 2, `1` at column 13
        "    let y = x + 2\n"       # line 3
        "    return y\n"            # line 4
        "}\n"
    )
    module = shared_parser().parse(src, file_path="loc.mx")

    seen: list = []

    def walk(node):
        if not isinstance(node, fast.Node) or any(n is node for n in seen):
            return
        seen.append(node)
        for attr, value in vars(node).items():
            if attr in ("parent", "scope", "location"):
                continue
            for item in (value if isinstance(value, (list, tuple)) else [value]):
                walk(item)

    walk(module)
    assert seen, "parse produced no nodes"
    unlocated = [type(n).__name__ for n in seen
                 if not getattr(getattr(n, "location", None), "line", 0)]
    assert unlocated == [], f"nodes without a location: {sorted(set(unlocated))}"
    assert all(n.location.file == "loc.mx" for n in seen)

    # And the positions are real, not a constant: the literal `2` of line 3
    # is at column 17.
    literals = [n for n in seen if isinstance(n, fast.Literal) and n.value == 2]
    assert [(n.location.line, n.location.column) for n in literals] == [(3, 17)]


def test_locations_are_offsets_and_line_columns_that_agree():
    src = "fn f() -> int {\n    return 7\n}\n"
    module = shared_parser().parse(src, file_path="agree.mx")

    lit = None

    def walk(node):
        nonlocal lit
        if not isinstance(node, fast.Node):
            return
        if isinstance(node, fast.Literal) and node.value == 7:
            lit = node
        for attr, value in vars(node).items():
            if attr in ("parent", "scope", "location"):
                continue
            for item in (value if isinstance(value, (list, tuple)) else [value]):
                walk(item)

    walk(module)
    assert lit is not None
    loc = lit.location
    assert (loc.line, loc.column) == (2, 12)
    # offset/end_offset are 0-based character offsets of the same text
    assert src[loc.offset:loc.end_offset] == "7"


# ---------------------------------------------------------------------------
# Compile-time diagnostics carry file:line:column
# ---------------------------------------------------------------------------

UNDECLARED_PARAM = """\
struct Ok { a: int }

struct Buf<T> {
    data: vector[T, N]
}

fn main() -> int { return 0 }
"""


def test_type_check_error_names_the_field_line(tmp_path):
    path, msg = compile_expecting(tmp_path, "buf.mx", UNDECLARED_PARAM, TypeCheckError)
    # `data: vector[T, N]` is line 4, column 5.
    assert f"{path}:4:5:" in msg
    assert "undeclared type parameter 'N'" in msg
    assert "data: vector[T, N]" in msg  # excerpt
    assert "^" in msg                   # caret


NON_EXHAUSTIVE = """\
enum Shape { Circle(r: int), Square(s: int), Dot }

fn f(s: Shape) -> int {
    match s {
        Dot => 0
    }
}
"""


def test_non_exhaustive_match_names_the_match_line(tmp_path):
    path, msg = compile_expecting(tmp_path, "exh.mx", NON_EXHAUSTIVE, TypeCheckError)
    assert f"{path}:4:5:" in msg          # the `match s {` line
    assert "non-exhaustive match" in msg
    assert "match s {" in msg


TYPE_CONFLICT = """\
fn main() -> int {
    let a = 1
    let b = a + "hello"
    return b
}
"""


def test_type_conflict_names_the_offending_value(tmp_path):
    path, msg = compile_expecting(tmp_path, "conflict.mx", TYPE_CONFLICT, TypeCheckError)
    # The String that conflicts with the Int is on line 3, column 17.
    assert f"{path}:3:17:" in msg
    assert "type mismatch" in msg


BORROW_CONFLICT = """\
struct Node { data: int }

fn main() -> int {
    let @mut n = Node { data: 1 }
    let @mut r1 = @mut n
    let @mut r2 = @mut n
    return 0
}
"""


def test_borrow_check_error_names_the_second_borrow(tmp_path):
    path, msg = compile_expecting(tmp_path, "borrow.mx", BORROW_CONFLICT,
                                  BorrowCheckError)
    # The conflicting second borrow is on line 6.
    assert f"{path}:6:" in msg
    assert "borrow check failed" in msg
    assert "^" in msg


COHERENCE = """\
trait Show {
    fn show(self) -> int;
}

struct P { x: int }

implement Show for P {
    fn show(self) -> int { return 1 }
}

implement Show for P {
    fn show(self) -> int { return 2 }
}

fn main() -> int { return 0 }
"""


def test_coherence_error_names_the_duplicate_method(tmp_path):
    path, msg = compile_expecting(tmp_path, "coh.mx", COHERENCE, CoherenceError)
    assert f"{path}:12:5:" in msg      # the second `fn show`
    assert "Conflicting implementations" in msg


MISSING_MODULE = """\
fn helper() -> int { return 1 }

import nosuchmodule.here;

fn main() -> int { return helper() }
"""


def test_module_error_names_the_import_line(tmp_path):
    path, msg = compile_expecting(tmp_path, "mod.mx", MISSING_MODULE, CompileError)
    assert f"{path}:3:1:" in msg
    assert "ModuleError" in msg
    assert "not found" in msg


RESERVED = """\
fn ok() -> int { return 1 }

fn __helper() -> int {
    return 2
}
"""


def test_reserved_name_error_names_the_declaration(tmp_path):
    path, msg = compile_expecting(tmp_path, "reserved.mx", RESERVED, CompileError)
    assert f"{path}:3:1:" in msg
    assert "ReservedNameError" in msg


UNSUPPORTED_PATTERN = """\
fn f(x: int) -> int {
    let y = x + 1
    match y {
        [a, b] => 1,
        _ => 0
    }
}
"""


def test_unsupported_construct_names_the_pattern(tmp_path):
    path, msg = compile_expecting(tmp_path, "pat.mx", UNSUPPORTED_PATTERN,
                                  UnsupportedConstruct)
    assert f"{path}:4:9" in msg        # the `[a, b]` pattern
    assert "unsupported pattern ListLiteral" in msg
    assert "[a, b] => 1," in msg       # excerpt


PARSE_ERROR = """\
fn main() -> int {
    let x = 1
    let = 2
    return x
}
"""


def test_parse_error_names_the_bad_token(tmp_path):
    path = write(tmp_path, "syntax.mx", PARSE_ERROR)
    with pytest.raises(CompileError) as excinfo:
        shared_parser().parse(PARSE_ERROR, file_path=path)
    msg = str(excinfo.value)
    assert f"{path}:3:9:" in msg       # the `=` that has no name before it
    assert "ParseError" in msg
    assert "let = 2" in msg


# ---------------------------------------------------------------------------
# The line reported is the RIGHT line, not line 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("blank_lines", [0, 5, 17])
def test_multi_line_file_reports_the_right_line(tmp_path, blank_lines):
    """The same error, pushed further down the file, must move with it."""
    padding = "".join(f"// filler {i}\n" for i in range(blank_lines))
    source = padding + NON_EXHAUSTIVE
    path, msg = compile_expecting(tmp_path, f"pad{blank_lines}.mx", source,
                                  TypeCheckError)
    expected_line = blank_lines + 4     # the `match s {` line
    assert f"{path}:{expected_line}:5:" in msg
    assert f"  {expected_line} | " in msg   # the excerpt's gutter


def test_two_errors_in_one_file_get_distinct_lines(tmp_path):
    source = """\
enum Shape { Circle(r: int), Square(s: int), Dot }

fn f(s: Shape) -> int {
    match s {
        Dot => 0
    }
}

fn g(s: Shape) -> int {
    match s {
        Circle(r) => r
    }
}
"""
    path, msg = compile_expecting(tmp_path, "two.mx", source, TypeCheckError)
    assert f"{path}:4:5:" in msg
    assert f"{path}:10:5:" in msg


# ---------------------------------------------------------------------------
# Frozen AST spans
# ---------------------------------------------------------------------------

def test_frozen_span_carries_offsets_and_line_columns():
    src = "fn f() -> int {\n    return 7\n}\n"
    module = shared_parser().parse(src, file_path="span.mx")
    frozen, _ = build_frozen_ast_with_map(module)

    def walk(n):
        yield n
        for c in n.children:
            yield from walk(c)

    literals = [n for n in walk(frozen) if n.kind == "Literal"]
    assert literals, "expected a frozen Literal node"
    span = literals[0].span
    assert span.file == "span.mx"
    assert (span.line, span.column) == (2, 12)
    assert src[span.start:span.end] == "7"
    assert span.text() == "span.mx:2:12"


def test_frozen_span_json_shape_includes_positions():
    src = "fn f() -> int {\n    return 7\n}\n"
    module = shared_parser().parse(src, file_path="json.mx")
    frozen, _ = build_frozen_ast_with_map(module)
    payload = dump_ast_json(frozen)
    for key in ('"line"', '"column"', '"end_line"', '"end_column"',
                '"start"', '"end"'):
        assert key in payload


# ---------------------------------------------------------------------------
# `exclave` used to crash the freeze with a JSON serialization TypeError
# ---------------------------------------------------------------------------

EXCLAVE_LITERAL = """\
fn f() -> int {
    let x = 1
    return exclave 5
}

fn main() -> int { return f() }
"""

EXCLAVE_VARIABLE = """\
fn f() -> int {
    let x = 7
    return exclave x
}

fn main() -> int { return f() }
"""


@pytest.mark.parametrize("source,expected", [
    (EXCLAVE_LITERAL, 5),
    (EXCLAVE_VARIABLE, 7),
])
def test_exclave_compiles_and_runs(source, expected):
    """`exclave <expr>` froze its operand INTO the payload, so dumping the
    frozen AST died with "Object of type Literal is not JSON serializable"."""
    ast_json, _hir, _mir, _clif = run_pipeline_from_source(source)
    assert '"ExclaveExpression"' in ast_json

    ctx = build_context_from_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    assert interp.call("f", []) == expected


def test_exclave_payload_is_json_scalar_only():
    module = shared_parser().parse(EXCLAVE_LITERAL, file_path="exclave.mx")
    frozen, _ = build_frozen_ast_with_map(module)

    def walk(n):
        yield n
        for c in n.children:
            yield from walk(c)

    nodes = [n for n in walk(frozen) if n.kind == "ExclaveExpression"]
    assert len(nodes) == 1
    # The operand is a CHILD, never the payload; the payload names a variable
    # or nothing at all.
    assert nodes[0].value == {"expression": None}
    assert any(c.kind == "Literal" for c in nodes[0].children)


def test_unserializable_payload_is_a_located_compile_error():
    """The freeze refuses to build a payload JSON cannot represent, and says
    where — instead of dying much later inside json.dumps."""
    import metaxu.compiler.mutaxu_ast as mast

    node = fast.Literal(1)
    node.location = SourceLocation(file="bad.mx", line=3, column=5)
    original = mast._value_of
    try:
        mast._value_of = lambda n: {"oops": node} if n is node else original(n)
        with pytest.raises(CompileError) as excinfo:
            mast.build_frozen_ast_with_map(node)
    finally:
        mast._value_of = original
    msg = str(excinfo.value)
    assert "bad.mx:3:5" in msg
    assert "FrozenAstError" in msg


# ---------------------------------------------------------------------------
# Run-time errors: function granularity (MIR ops carry no spans)
# ---------------------------------------------------------------------------

OUT_OF_BOUNDS = """\
fn boom(xs: vector[int, 2]) -> int {
    return xs[9]
}

fn main() -> int {
    let v = vector[int, 2](1, 2)
    return boom(v)
}
"""


def test_interp_error_names_the_function_it_happened_in(tmp_path):
    path = write(tmp_path, "rt.mx", OUT_OF_BOUNDS)
    ctx = build_context_from_source(OUT_OF_BOUNDS, file_path=path)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    with pytest.raises(InterpError) as excinfo:
        interp.call("main", [])
    msg = str(excinfo.value)
    assert "index out of bounds" in msg     # original message preserved
    assert "in function 'boom'" in msg      # innermost frame, not `main`
    assert f"{path}:1:1" in msg             # `boom`'s declaration


# ---------------------------------------------------------------------------
# Excerpt rendering
# ---------------------------------------------------------------------------

def test_source_excerpt_underlines_the_range(tmp_path):
    path = write(tmp_path, "excerpt.mx", "fn main() -> int {\n    return 42\n}\n")
    loc = SourceLocation(file=path, line=2, column=12, end_line=2, end_column=14)
    assert source_excerpt(loc) == ("  2 |     return 42\n"
                                   "    |            ^~")


def test_source_excerpt_of_unknown_file_is_none():
    assert source_excerpt(SourceLocation(file="/no/such/file.mx", line=1,
                                         column=1)) is None
