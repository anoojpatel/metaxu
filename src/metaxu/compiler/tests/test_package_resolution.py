"""The module resolver's package hook (docs/packages.md, "Resolution").

Every case builds a real project on disk, path dependencies only (no
git, no network), runs `mxpkg sync` to write the lock the compiler
reads, and compiles the entry file through the full pipeline the way
the example gate does.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import UNIT, MirInterpreter, mx_display
from metaxu.compiler.pipeline import build_context_from_source, run_pipeline_ctx
from metaxu.errors import CompileError
from metaxu.packages import sync

GEOM_LIB = """export { area, describe };

import shapes;
from internal import twice;

fn area(w: int, h: int) -> int {
    w * h
}

fn describe(w: int, h: int) -> string {
    shapes.name(w, h) + " of area " + twice(area(w, h) / 2).to_string()
}
"""

GEOM_SHAPES = """export { name };

fn name(w: int, h: int) -> string {
    if w == h { "square" } else { "rectangle" }
}
"""

GEOM_INTERNAL = """export { twice };

fn twice(n: int) -> int {
    n * 2
}
"""

UTIL_LIB = """export { clamp };

fn clamp(x: int, lo: int, hi: int) -> int {
    if x < lo { lo } else { if x > hi { hi } else { x } }
}
"""


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def make_workspace(tmp_path: Path, main_src: str, *, public: str = '"shapes"',
                   extra_project_files: dict[str, str] | None = None) -> Path:
    write(tmp_path / "geom" / "mx.toml",
          '[package]\nname = "geom"\nversion = "0.1.0"\n'
          + (f"public = [{public}]\n" if public else "")
          + '\n[dependencies]\nutil = { path = "../util" }\n')
    write(tmp_path / "geom" / "src" / "lib.mx", GEOM_LIB)
    write(tmp_path / "geom" / "src" / "shapes.mx", GEOM_SHAPES)
    write(tmp_path / "geom" / "src" / "internal.mx", GEOM_INTERNAL)
    write(tmp_path / "util" / "mx.toml",
          '[package]\nname = "util"\nversion = "0.1.0"\n')
    write(tmp_path / "util" / "src" / "lib.mx", UTIL_LIB)
    app = tmp_path / "app"
    write(app / "mx.toml",
          '[package]\nname = "app"\nversion = "0.1.0"\n\n'
          '[dependencies]\ngeom = { path = "../geom" }\n')
    write(app / "main.mx", main_src)
    for rel, text in (extra_project_files or {}).items():
        write(app / rel, text)
    sync(app)
    return app / "main.mx"


def run_file(entry: Path) -> str:
    source = entry.read_text()
    ctx = build_context_from_source(source, file_path=str(entry))
    run_pipeline_ctx(ctx)
    hir_funcs = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir_funcs))
    out: list[str] = []

    def _print(*args):
        out.append(" ".join(str(mx_display(a)) for a in args))
        return UNIT

    interp.register_builtin("print", _print)
    assert interp.call("main", []) == 0
    return "\n".join(out)


def test_import_of_a_dependency_is_its_facade(tmp_path):
    entry = make_workspace(tmp_path, """import geom;

fn main() -> int {
    print(geom.area(3, 4));
    print(geom.describe(2, 2));
    0
}
""")
    assert run_file(entry) == "12\nsquare of area 4"


def test_dependency_siblings_resolve_inside_the_package(tmp_path):
    # geom's lib.mx says `import shapes;` and `from internal import twice;`
    # and the project has its own shapes.mx: the package's bare imports
    # mean the package's files, the project's file is untouched
    entry = make_workspace(tmp_path, """import geom;
import shapes;

fn main() -> int {
    print(shapes.name(1, 2));
    print(geom.describe(1, 2));
    0
}
""", extra_project_files={"shapes.mx": """export { name };

fn name(a: int, b: int) -> string {
    "the app's own shapes"
}
"""})
    assert run_file(entry) == "the app's own shapes\nrectangle of area 2"


def test_public_module_of_a_dependency_is_importable(tmp_path):
    entry = make_workspace(tmp_path, """from geom.shapes import name;

fn main() -> int {
    print(name(5, 5));
    0
}
""")
    assert run_file(entry) == "square"


def test_non_public_module_of_a_dependency_is_rejected(tmp_path):
    entry = make_workspace(tmp_path, """from geom.internal import twice;

fn main() -> int {
    print(twice(4));
    0
}
""")
    with pytest.raises(CompileError) as ei:
        run_file(entry)
    msg = str(ei.value)
    assert "module 'geom.internal' is not public in package 'geom'" in msg
    assert "[package] public" in msg


def test_transitive_dependency_is_one_flat_table(tmp_path):
    # util is geom's dependency, not the app's, and the lock is flat: the
    # app may import it, and it means the same package it means in geom
    entry = make_workspace(tmp_path, """import util;

fn main() -> int {
    print(util.clamp(50, 0, 10));
    0
}
""")
    assert run_file(entry) == "10"


def test_sibling_file_and_dependency_with_one_name_is_ambiguous(tmp_path):
    entry = make_workspace(tmp_path, """import geom;

fn main() -> int {
    print(geom.area(1, 1));
    0
}
""", extra_project_files={"geom.mx": "export { area };\nfn area(a: int, b: int) -> int { 0 }\n"})
    with pytest.raises(CompileError) as ei:
        run_file(entry)
    msg = str(ei.value)
    assert "ambiguous module 'geom'" in msg
    assert "never resolved by precedence" in msg


def test_unknown_module_names_the_locked_dependencies(tmp_path):
    entry = make_workspace(tmp_path, """import nothing;

fn main() -> int {
    0
}
""")
    with pytest.raises(CompileError) as ei:
        run_file(entry)
    msg = str(ei.value)
    assert "module 'nothing' not found" in msg
    assert "not a locked dependency either" in msg
    assert "geom, util" in msg


def test_without_a_lock_resolution_is_unchanged(tmp_path):
    # no mx.lock anywhere above the file: a bare import is a sibling file
    # or nothing, exactly as before packages existed
    write(tmp_path / "main.mx", "import geom;\n\nfn main() -> int { 0 }\n")
    with pytest.raises(CompileError) as ei:
        run_file(tmp_path / "main.mx")
    msg = str(ei.value)
    assert "module 'geom' not found" in msg
    assert "locked dependency" not in msg
