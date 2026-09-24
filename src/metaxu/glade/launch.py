"""The `glade` console script: run the package manager written in Metaxu.

`glade/*.mx` at the repository root (shipped inside the wheel as
`metaxu/glade/mx/`) is the implementation; `metaxu.glade.cli` is the
Python reference it is tested against (`test_glade_metaxu.py`). This
module picks an engine for the Metaxu program and runs it:

- native, when `clang` is installed: the program is compiled once through
  the LLVM backend into `$XDG_CACHE_HOME/glade/bin/glade-<key>` (the key
  hashes the glade and standard library sources and the compiler) and
  every later run executes that binary directly;
- the MIR interpreter otherwise, compiling the sources on each run (a few
  seconds per command).

`GLADE_IMPL` overrides the choice: `native` (fail rather than fall back),
`interp`, or `python` for the reference implementation.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

from .index import cache_dir


def sources_dir() -> Path:
    """Where `main.mx` and its siblings live: the installed copy inside the
    package, else the repository checkout's `glade/`."""
    env = os.environ.get("GLADE_SOURCES")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    for cand in (here / "mx", here.parents[2] / "glade"):
        if (cand / "main.mx").is_file():
            return cand
    raise FileNotFoundError("glade: cannot find the Metaxu sources (glade/main.mx); "
                            "set GLADE_SOURCES to the directory that holds them")


def _std_dir() -> Path | None:
    from metaxu.compiler.module_loader import _stdlib_dir
    found = _stdlib_dir()
    return Path(found) if found else None


def build_key(src: Path) -> str:
    """A digest of everything whose change should rebuild the binary: the
    glade sources, the standard library and the compiler's own code."""
    h = hashlib.sha256()
    roots: list[Path] = [src]
    std = _std_dir()
    if std is not None:
        roots.append(std)
    roots.append(Path(__file__).resolve().parents[1] / "compiler")
    roots.append(Path(__file__).resolve().parents[1] / "runtime" / "native")
    for root in roots:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in (".mx", ".py", ".c", ".h") and "tests" not in p.parts:
                h.update(str(p.relative_to(root)).encode())
                h.update(b"\0")
                h.update(p.read_bytes())
                h.update(b"\0")
    return h.hexdigest()[:16]


def native_binary(src: Path, *, build: bool = True) -> Path | None:
    """The cached native binary for these sources, building it when
    `build` is set and it is missing. None when clang is unavailable."""
    import shutil
    if shutil.which("clang") is None:
        return None
    out = cache_dir() / "bin" / f"glade-{build_key(src)}"
    if out.is_file():
        return out
    if not build:
        return None
    from metaxu.compiler.llvm_run import compile_to_binary
    from metaxu.compiler.pipeline import emit_llvm_from_source
    main_mx = src / "main.mx"
    print("glade: building the native glade (first run after a change)...", file=sys.stderr)
    llvm = emit_llvm_from_source(main_mx.read_text(), file_path=str(main_mx))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".{os.getpid()}.tmp")
    compile_to_binary(llvm, "main", out_path=str(tmp), timeout=900)
    os.replace(tmp, out)
    return out


def run_native(binary: Path, argv: list[str]) -> int:
    sys.stdout.flush()
    sys.stderr.flush()
    return subprocess.call([str(binary), *argv])


def run_interpreter(src: Path, argv: list[str]) -> int:
    from metaxu.compiler.hir import HIRBuilder
    from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
    from metaxu.compiler.mir_interp import UNIT, MirInterpreter
    from metaxu.compiler.pipeline import build_context_from_source
    main_mx = src / "main.mx"
    ctx = build_context_from_source(main_mx.read_text(), file_path=str(main_mx))
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    interp.program_args = list(argv)
    result = interp.call("main", [])
    if result is UNIT or result is None:
        return 0
    return int(result) % 256


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    impl = os.environ.get("GLADE_IMPL", "").strip().lower()
    if impl == "python":
        from .cli import main as python_main
        return python_main(argv)
    src = sources_dir()
    if impl != "interp":
        try:
            binary = native_binary(src)
        except Exception as e:                 # a build failure, not a user error
            if impl == "native":
                raise
            first = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            print(f"glade: native build failed ({first}); running on the interpreter",
                  file=sys.stderr)
            binary = None
        if binary is not None:
            return run_native(binary, argv)
        if impl == "native":
            print("glade: GLADE_IMPL=native but clang is not installed", file=sys.stderr)
            return 2
    return run_interpreter(src, argv)


if __name__ == "__main__":
    sys.exit(main())
