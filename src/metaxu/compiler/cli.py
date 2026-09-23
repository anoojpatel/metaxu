"""``metaxuc``: the command-line front door to the v1 compiler.

    metaxuc run   FILE            interpret FILE; exit code = main's return
    metaxuc build FILE [-o OUT]   compile FILE to a native executable
    metaxuc check FILE            parse, type check and borrow check only
    metaxuc emit  FILE [--stage]  print one stage: ast, hir, mir, clif, llvm

Every subcommand runs the same strict front end as the test suite and
the example gates (``build_context_from_source`` + ``run_pipeline_ctx``),
so a program the tests would reject is rejected here with the same
diagnostic.  ``run`` uses the MIR interpreter, the semantics reference;
``build`` uses the LLVM backend and links the C runtime, exactly as the
differential tests do.  A compile error prints the diagnostic to stderr
and exits 1; a runtime error does the same.  A crash inside the compiler
is left as a traceback on purpose: that is a compiler bug, not a user
error, and hiding it would help nobody.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

from metaxu.errors import CompileError

from .frozen_borrow_checker import BorrowCheckError, TypeCheckError
from .hir import HIRBuilder, HIRLoweringError
from .lower_hir_to_mir import lower_hir_to_mir
from .llvm_run import LlvmRunError, compile_to_binary
from .mir_interp import UNIT, InterpError, MirInterpreter
from .pipeline import (
    build_context_from_source,
    emit_llvm_from_source,
    run_pipeline_ctx,
)

USER_ERRORS = (CompileError, BorrowCheckError, TypeCheckError,
               HIRLoweringError, InterpError, LlvmRunError)
STAGES = ("ast", "hir", "mir", "clif", "llvm")


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        raise CompileError(f"cannot read {path}: {exc.strerror}") from None


def _checked_context(path: str):
    """Front end: parse, resolve, desugar, freeze, infer, borrow check."""
    ctx = build_context_from_source(_read(path), file_path=path)
    run_pipeline_ctx(ctx)  # strict: raises the typed diagnostics
    return ctx


def _pick_entry(interp: MirInterpreter, requested: str | None, path: str) -> str:
    if requested is not None:
        if requested not in interp._funcs:
            raise CompileError(f"{path}: no function named {requested!r}")
        return requested
    if "main" in interp._funcs:
        return "main"
    ex = interp._funcs.get("example")
    if ex is not None and not ex.param_names():
        return "example"
    raise CompileError(
        f"{path}: no `fn main() -> int` to run (pass --entry NAME to run "
        "another zero-argument function)")


def cmd_run(args: argparse.Namespace) -> int:
    ctx = _checked_context(args.file)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    entry = _pick_entry(interp, args.entry, args.file)
    result = interp.call(entry, [])
    if result is UNIT or result is None:
        return 0
    if isinstance(result, bool):
        return int(result)
    if isinstance(result, int):
        return result % 256      # what the OS keeps of a native exit code
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    _checked_context(args.file)
    print(f"{args.file}: ok")
    return 0


def cmd_emit(args: argparse.Namespace) -> int:
    if args.stage == "llvm":
        text = emit_llvm_from_source(_read(args.file), file_path=args.file)
    else:
        ctx = build_context_from_source(_read(args.file), file_path=args.file)
        ast_json, hir_txt, mir_txt, clif_txt = run_pipeline_ctx(ctx)
        text = {"ast": ast_json, "hir": hir_txt, "mir": mir_txt,
                "clif": clif_txt}[args.stage]
    sys.stdout.write(text if text.endswith("\n") else text + "\n")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    ir = emit_llvm_from_source(_read(args.file), file_path=args.file)
    out = args.output
    if out is None:
        stem = os.path.splitext(os.path.basename(args.file))[0]
        out = os.path.join(os.getcwd(), stem)
    entry = args.entry or "main"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    compile_to_binary(ir, entry, out_path=out,
                      ll_path=(out + ".ll") if args.keep_ir else None)
    if not args.keep_ir:
        try:
            os.remove(out + ".ll")
        except OSError:
            pass
    print(out)
    return 0


def _installed_version() -> str:
    """The package version, from the installed metadata (the same
    number pyproject.toml carries and the release tag is made from)."""
    try:
        from importlib.metadata import version
        return version("metaxu")
    except Exception:  # noqa: BLE001 - a source tree run without install
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="metaxuc",
        description="Compile, run and inspect Metaxu programs.")
    ap.add_argument("--version", action="version",
                    version=f"metaxuc {_installed_version()}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="interpret a program")
    p.add_argument("file")
    p.add_argument("--entry", help="function to call (default: main)")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("build", help="compile a program to a native executable")
    p.add_argument("file")
    p.add_argument("-o", "--output", help="output path (default: ./<stem>)")
    p.add_argument("--entry", help="entry function (default: main)")
    p.add_argument("--keep-ir", action="store_true",
                   help="also write the LLVM IR next to the binary as <out>.ll")
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("check", help="type check and borrow check without running")
    p.add_argument("file")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("emit", help="print one intermediate form")
    p.add_argument("file")
    p.add_argument("--stage", choices=STAGES, default="llvm")
    p.set_defaults(fn=cmd_emit)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except USER_ERRORS as exc:
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
