"""Native execution harness for LLVM modules emitted by codegen_llvm.

``compile_and_run(llvm_ir, entry="main")`` writes the module to a temp
directory, appends a tiny C-ABI ``@main`` wrapper, compiles it with
``clang -O2`` and executes the binary, returning ``(exit_code, stdout)``.

Entry-point convention (documented choice): the emitted metaxu entry
function keeps its mangled name (``@mx_main`` etc.) and is never itself the
C ``main``; instead we synthesize::

    define i32 @main() {
      %r = call i64 @mx_<entry>()
      %t = trunc i64 %r to i32
      ret i32 %t
    }

so the native exit code is the entry function's i64 result truncated —
the OS only preserves the low 8 bits, so tests should either keep results
in [0, 256) / compare modulo 256, or (preferred) print values through the
@metaxu_print_* helpers and compare stdout.  An entry returning double or
ptr is called for effect and the process exits 0 (its value has no
meaningful exit-code encoding).

The entry function must be a real zero-argument ``define`` in the module;
if codegen_llvm demoted it to a placeholder comment this raises
``LlvmRunError`` (with the placeholder's reasons when available) rather
than running something else.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile

from .codegen_llvm import mangle

__all__ = ["compile_and_run", "LlvmRunError"]


class LlvmRunError(RuntimeError):
    """The module could not be compiled or the entry point is unusable."""


_WRAPPERS = {
    "i64": (
        "define i32 @main() {{\n"
        "entry:\n"
        "  %r = call i64 @{sym}()\n"
        "  %t = trunc i64 %r to i32\n"
        "  ret i32 %t\n"
        "}}"
    ),
    "double": (
        "define i32 @main() {{\n"
        "entry:\n"
        "  %r = call double @{sym}()\n"
        "  ret i32 0\n"
        "}}"
    ),
    "ptr": (
        "define i32 @main() {{\n"
        "entry:\n"
        "  %r = call ptr @{sym}()\n"
        "  ret i32 0\n"
        "}}"
    ),
}


def _placeholder_reasons(llvm_ir: str, sym: str) -> list[str]:
    reasons: list[str] = []
    in_placeholder = False
    for line in llvm_ir.splitlines():
        if line.startswith(f"; function @{sym}: placeholder"):
            in_placeholder = True
            continue
        if in_placeholder:
            m = re.match(r";\s+reason: (.*)", line)
            if m:
                reasons.append(m.group(1))
            else:
                break
    return reasons


def compile_and_run(llvm_ir: str, entry: str = "main", *,
                    workdir: str | None = None,
                    timeout: float = 60.0,
                    clang_args: tuple[str, ...] = ()) -> tuple[int, str]:
    """Compile ``llvm_ir`` with clang and run it; return (exit_code, stdout).

    ``entry`` is the metaxu function name (unmangled).  ``workdir`` keeps the
    .ll/.bin files for inspection instead of a fresh temp dir.  ``clang_args``
    are appended to the clang invocation (e.g. ("-fsanitize=address",) so the
    differential tests can prove the emitted malloc/free pairs sound).
    """
    sym = mangle(entry)
    m = re.search(
        rf"^define (i64|double|ptr) @{re.escape(sym)}\(\)", llvm_ir, re.M)
    if m is None:
        reasons = _placeholder_reasons(llvm_ir, sym)
        detail = ("; ".join(reasons) if reasons
                  else "no zero-argument define found in the module "
                       "(struct-returning entries have a ptr sret param and "
                       "cannot be an OS entry point)")
        raise LlvmRunError(f"entry {entry!r} (@{sym}) is not natively runnable: {detail}")
    rty = m.group(1)
    full_ir = llvm_ir + "\n\n; native entry wrapper (llvm_run)\n" + _WRAPPERS[rty].format(sym=sym) + "\n"

    if workdir is None:
        workdir = tempfile.mkdtemp(prefix="metaxu_llvm_")
    ll_path = os.path.join(workdir, "prog.ll")
    bin_path = os.path.join(workdir, "prog.bin")
    with open(ll_path, "w") as fh:
        fh.write(full_ir)

    compile_proc = subprocess.run(
        ["clang", "-O2", "-Wno-override-module", ll_path, "-o", bin_path, "-lm",
         *clang_args],
        capture_output=True, text=True, timeout=timeout)
    if compile_proc.returncode != 0:
        raise LlvmRunError(
            f"clang failed (exit {compile_proc.returncode}) on {ll_path}:\n"
            f"{compile_proc.stderr}")

    run_proc = subprocess.run(
        [bin_path], capture_output=True, text=True, timeout=timeout)
    return run_proc.returncode, run_proc.stdout
