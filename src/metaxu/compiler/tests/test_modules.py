"""Multi-file module system tests (docs/modules_implementation.md).

All positive tests go through parsed source -> full pipeline -> MIR
interpreter (per the repo convention: no hand-built HIR/MIR fixtures).
Multi-file trees are laid out under tmp_path; the root file's directory is
the module search root.
"""
from __future__ import annotations

import pytest

from metaxu.errors import CompileError
from metaxu.compiler.pipeline import build_context_from_source
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import MirInterpreter, UNIT


def compile_tree(tmp_path, files: dict[str, str], root: str = "main.mx"):
    """Write `files` under tmp_path and build a PhaseContext for `root`."""
    for rel, src in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src)
    root_path = tmp_path / root
    return build_context_from_source(root_path.read_text(),
                                     file_path=str(root_path))


def run_tree(tmp_path, files: dict[str, str], root: str = "main.mx",
             entry: str = "main"):
    """Compile a multi-file tree and execute `entry`, returning
    (result, printed_lines, interp)."""
    ctx = compile_tree(tmp_path, files, root)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    prints: list[str] = []
    interp.register_builtin(
        "print", lambda *a: (prints.append(" ".join(str(x) for x in a)), UNIT)[1])
    return interp.call(entry, []), prints, interp


# ----------------------------------------------------------------------
# Cross-file import + call
# ----------------------------------------------------------------------

def test_import_and_call_across_files(tmp_path):
    result, prints, _ = run_tree(tmp_path, {
        "mathlib.mx": """
fn add(x: int, y: int) -> int { x + y }
""",
        "main.mx": """
from mathlib import add;

fn main() -> int {
    let s = add(2, 3);
    print(s);
    s
}
""",
    })
    assert result == 5
    assert prints == ["5"]


def test_qualified_and_unqualified_resolution(tmp_path):
    """`import a.b` binds the last component; full paths work with no
    import; from-imports bind unqualified names — all three name the same
    function."""
    result, _, interp = run_tree(tmp_path, {
        "geo/shapes.mx": """
struct Square { side: int }

fn area(s: Square) -> int { s.side * s.side }
""",
        "main.mx": """
import geo.shapes;
from geo.shapes import area, Square;

fn main() -> int {
    let sq = Square { side: 4 };
    let a = area(sq);              # unqualified via from-import
    let b = shapes.area(sq);       # via `import geo.shapes` binding
    let c = geo.shapes.area(sq);   # fully qualified, no import needed
    a + b + c
}
""",
    })
    assert result == 48
    # the module's function is namespaced under its dotted path
    assert "geo.shapes.area" in interp._funcs
    assert "area" not in interp._funcs


def test_import_alias(tmp_path):
    result, _, _ = run_tree(tmp_path, {
        "veclib.mx": "fn double(x: int) -> int { x * 2 }\n",
        "main.mx": """
import veclib as v;
from veclib import double as twice;

fn main() -> int { v.double(3) + twice(4) }
""",
    })
    assert result == 14


def test_transitive_imports(tmp_path):
    """main -> a -> b: b is loaded transitively and a's call into b
    resolves inside a's own scope."""
    result, _, interp = run_tree(tmp_path, {
        "b.mx": "fn base() -> int { 7 }\n",
        "a.mx": """
from b import base;

fn wrapped() -> int { base() + 1 }
""",
        "main.mx": """
from a import wrapped;

fn main() -> int { wrapped() }
""",
    })
    assert result == 8
    assert "a.wrapped" in interp._funcs and "b.base" in interp._funcs


def test_same_function_name_in_two_modules(tmp_path):
    """Namespacing keeps same-named functions in different modules apart."""
    result, _, _ = run_tree(tmp_path, {
        "one.mx": "fn value() -> int { 1 }\n",
        "two.mx": "fn value() -> int { 2 }\n",
        "main.mx": """
import one;
import two;

fn main() -> int { one.value() * 10 + two.value() }
""",
    })
    assert result == 12


# ----------------------------------------------------------------------
# Visibility
# ----------------------------------------------------------------------

def test_import_private_symbol_errors(tmp_path):
    with pytest.raises(CompileError) as exc:
        compile_tree(tmp_path, {
            "lib.mx": """
export { shown }

fn shown() -> int { 1 }
fn hidden() -> int { 2 }
""",
            "main.mx": """
from lib import hidden;

fn main() -> int { hidden() }
""",
        })
    msg = str(exc.value)
    assert "hidden" in msg and "lib" in msg and "private" in msg


def test_qualified_reference_to_private_symbol_errors(tmp_path):
    with pytest.raises(CompileError) as exc:
        compile_tree(tmp_path, {
            "lib.mx": """
export { shown }

fn shown() -> int { 1 }
fn hidden() -> int { 2 }
""",
            "main.mx": """
import lib;

fn main() -> int { lib.hidden() }
""",
        })
    msg = str(exc.value)
    assert "hidden" in msg and "lib" in msg and "private" in msg


def test_visibility_block_marks_private(tmp_path):
    with pytest.raises(CompileError) as exc:
        compile_tree(tmp_path, {
            "lib.mx": """
visibility { helper: private }

fn helper() -> int { 2 }
""",
            "main.mx": """
from lib import helper;

fn main() -> int { helper() }
""",
        })
    assert "helper" in str(exc.value)


def test_no_export_list_means_public(tmp_path):
    result, _, _ = run_tree(tmp_path, {
        "lib.mx": "fn anything() -> int { 9 }\n",
        "main.mx": """
from lib import anything;

fn main() -> int { anything() }
""",
    })
    assert result == 9


def test_export_list_allows_listed_symbols(tmp_path):
    result, _, _ = run_tree(tmp_path, {
        "lib.mx": """
export { shown }

fn shown() -> int { 1 }
fn hidden() -> int { 2 }
""",
        "main.mx": """
from lib import shown;

fn main() -> int { shown() }
""",
    })
    assert result == 1


# ----------------------------------------------------------------------
# Loader errors
# ----------------------------------------------------------------------

def test_missing_module_errors(tmp_path):
    with pytest.raises(CompileError) as exc:
        compile_tree(tmp_path, {
            "main.mx": """
from nowhere import thing;

fn main() -> int { 0 }
""",
        })
    msg = str(exc.value)
    assert "nowhere" in msg and "not found" in msg


def test_missing_symbol_in_real_module_errors(tmp_path):
    with pytest.raises(CompileError) as exc:
        compile_tree(tmp_path, {
            "lib.mx": "fn real() -> int { 1 }\n",
            "main.mx": """
from lib import imaginary;

fn main() -> int { 0 }
""",
        })
    msg = str(exc.value)
    assert "imaginary" in msg and "no symbol" in msg


def test_import_cycle_errors(tmp_path):
    with pytest.raises(CompileError) as exc:
        compile_tree(tmp_path, {
            "a.mx": "from b import g;\n\nfn f() -> int { g() }\n",
            "b.mx": "from a import f;\n\nfn g() -> int { f() }\n",
            "main.mx": "from a import f;\n\nfn main() -> int { f() }\n",
        })
    msg = str(exc.value)
    assert "cycle" in msg and "a" in msg and "b" in msg


def test_file_import_from_memory_source_errors():
    """No file_path (in-memory compile) + a file import = clear error."""
    with pytest.raises(CompileError) as exc:
        build_context_from_source("""
from disk_only import f;

fn main() -> int { 0 }
""")
    assert "disk_only" in str(exc.value)


def test_type_collision_across_modules_errors(tmp_path):
    with pytest.raises(CompileError) as exc:
        compile_tree(tmp_path, {
            "a.mx": "struct Point { x: int }\n\nfn fa() -> int { 1 }\n",
            "b.mx": "struct Point { y: int }\n\nfn fb() -> int { 2 }\n",
            "main.mx": """
import a;
import b;

fn main() -> int { a.fa() + b.fb() }
""",
        })
    msg = str(exc.value)
    assert "Point" in msg


# ----------------------------------------------------------------------
# Traits and effects across modules
# ----------------------------------------------------------------------

def test_trait_defined_in_one_module_used_in_another(tmp_path):
    result, prints, _ = run_tree(tmp_path, {
        "traits.mx": """
trait Describe {
    fn describe(self) -> string;
}
""",
        "animals.mx": """
from traits import Describe;

struct Dog { name: string }

implement Describe for Dog {
    fn describe(self) -> string { "dog" }
}
""",
        "main.mx": """
from traits import Describe;
from animals import Dog;

fn main() -> int {
    let d = Dog { name: "rex" };
    print(d.describe());
    0
}
""",
    })
    assert result == 0
    assert prints == ["dog"]


def test_effect_defined_in_one_module_used_in_another(tmp_path):
    result, _, _ = run_tree(tmp_path, {
        "fx.mx": """
effect Counter {
    next() -> int
}
""",
        "main.mx": """
from fx import Counter;

fn bump() performs Counter -> int {
    perform Counter.next() + perform Counter.next()
}

fn main() -> int {
    handle Counter with {
        next() -> resume(21)
    } in {
        bump()
    }
}
""",
    })
    assert result == 42


# ----------------------------------------------------------------------
# In-file module blocks behave like file modules
# ----------------------------------------------------------------------

def test_in_file_modules_namespaced_consistently(tmp_path):
    result, _, interp = run_tree(tmp_path, {
        "main.mx": """
module helpers {
    export { triple }

    fn triple(x: int) -> int { x * 3 }

    fn internal(x: int) -> int { x }
}

from helpers import triple;

fn main() -> int { triple(2) + helpers.triple(3) }
""",
    })
    assert result == 15
    assert "helpers.triple" in interp._funcs
    assert "triple" not in interp._funcs


def test_in_file_module_private_symbol_qualified_reference_errors(tmp_path):
    with pytest.raises(CompileError) as exc:
        compile_tree(tmp_path, {
            "main.mx": """
module helpers {
    export { triple }

    fn triple(x: int) -> int { x * 3 }

    fn internal(x: int) -> int { x }
}

fn main() -> int { helpers.internal(1) }
""",
        })
    assert "internal" in str(exc.value)


def test_relative_import_between_sibling_modules(tmp_path):
    """`from ..sibling import f` inside `module pkg.a` reaches `pkg.sibling`
    (example 03's `from ..vector import ...` shape)."""
    result, _, _ = run_tree(tmp_path, {
        "main.mx": """
module pkg.strings {
    fn shout(s: string) -> string { s + "!" }
}

module pkg.caller {
    from ..strings import shout;

    fn go() -> string { shout("hey") }
}

from pkg.caller import go;

fn main() -> int {
    print(go());
    0
}
""",
    })
    assert result == 0


# ----------------------------------------------------------------------
# Backward compatibility / gates
# ----------------------------------------------------------------------

def test_single_file_no_imports_is_untouched():
    """A plain program must bypass module resolution entirely."""
    import metaxu.metaxu_ast as fast
    from metaxu.parser import Parser
    from metaxu.compiler.module_loader import resolve_modules

    src = "fn main() -> int { 41 + 1 }\n"
    module = Parser().parse(src, file_path="<mem>")
    program = fast.Program([module])
    assert resolve_modules(program, "<mem>") is program
    # and the function name is untouched
    fn = program.statements[0].body.statements[0]
    assert fn.name == "main"


def test_std_imports_stay_lenient(tmp_path):
    """`std.*` is a reserved external namespace: importing it succeeds and
    imported names fall through to builtins (examples 03/06 depend on it)."""
    result, _, _ = run_tree(tmp_path, {
        "main.mx": """
import std.matrix as mat;
from std.math import sqrt;

fn main() -> int { 3 }
""",
    })
    assert result == 3


def test_example_03_output_unchanged():
    """Example 03 keeps its documented behavior (dot product 11) while its
    module functions are now properly namespaced."""
    import os
    repo_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..", ".."))
    src = open(os.path.join(repo_root,
                            "examples/03_modules_and_imports.mx")).read()
    ctx = build_context_from_source(src)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    syms = [f.sym for f in hir]
    assert "math.vector.dot" in syms
    assert "math.transform.rotate" in syms
    assert "main" in syms                      # entry stays unqualified
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    prints: list[str] = []
    interp.register_builtin(
        "print", lambda *a: (prints.append(" ".join(str(x) for x in a)), UNIT)[1])
    interp.call("main", [])
    assert any("11" in line for line in prints)


# ----------------------------------------------------------------------
# Borrow checking across modules
# ----------------------------------------------------------------------

def test_borrow_error_inside_imported_module_still_fires(tmp_path):
    """The borrow checker sees imported modules' functions: a double
    unique borrow in a library file fails the whole compile."""
    from metaxu.compiler.pipeline import run_pipeline_ctx, BorrowCheckError

    ctx = compile_tree(tmp_path, {
        "lib.mx": """
fn bad() -> int {
    let @mut x = 1;
    let @mut r1 = @mut x;
    let @mut r2 = @mut x;
    0
}
""",
        "main.mx": """
from lib import bad;

fn main() -> int { bad() }
""",
    })
    with pytest.raises(BorrowCheckError):
        run_pipeline_ctx(ctx)
