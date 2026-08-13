"""Native execution harness for LLVM modules emitted by codegen_llvm.

``compile_and_run(llvm_ir, entry="main")`` writes the module to a temp
directory, appends a tiny C-ABI ``@main`` wrapper, compiles it with
``clang -O2`` and executes the binary, returning ``(exit_code, stdout)``.

The native metaxu runtime (``src/metaxu/runtime/native/metaxu_rt.c``:
mx_vec_* / mx_str_* / mx_*_to_str, plus ``metaxu_effects.c``: mx_handle /
mx_perform / mx_resume — the ucontext coroutine scheduler backing
algebraic effects) is compiled via its cached build recipe
and linked into every binary, so modules emitted with native vec/string
builtin lowerings resolve their ``mx_*`` declares.  When the caller passes
``-fsanitize=address`` in ``clang_args`` the runtime object is rebuilt
ASan-instrumented into a separate cache directory, so ASan differential
tests also check the runtime's own memory traffic.  ``sqrt``/``sin``/
``cos`` stay plain libm externs (``-lm`` below) — the documented choice:
libm already matches the interpreter's ``math.*`` on the tested domain, so
no mx_ wrappers or LLVM intrinsics are introduced.

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

from metaxu.runtime.native.build import DEFAULT_BUILD_DIR, runtime_objects

from .codegen_llvm import mangle

__all__ = ["compile_and_run", "LlvmRunError"]


def _runtime_object_paths(clang_args: tuple[str, ...]) -> list[str]:
    """The native runtime objects to link (metaxu_rt.o + metaxu_effects.o),
    ASan-instrumented when the module itself is being sanitized (separate
    cache dir per flag set)."""
    if any("-fsanitize=address" in a for a in clang_args):
        objs = runtime_objects(
            build_dir=DEFAULT_BUILD_DIR.parent / "_build_asan",
            extra_cflags=("-fsanitize=address", "-fno-omit-frame-pointer"))
    else:
        objs = runtime_objects()
    return [str(o) for o in objs]


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
                    clang_args: tuple[str, ...] = (),
                    run_env: dict[str, str] | None = None) -> tuple[int, str]:
    """Compile ``llvm_ir`` with clang and run it; return (exit_code, stdout).

    ``entry`` is the metaxu function name (unmangled).  ``workdir`` keeps the
    .ll/.bin files for inspection instead of a fresh temp dir.  ``clang_args``
    are appended to the clang invocation (e.g. ("-fsanitize=address",) so the
    differential tests can prove the emitted malloc/free pairs sound).
    ``run_env`` entries are overlaid on the inherited environment for the
    binary's execution — e.g. {"ASAN_OPTIONS": "detect_leaks=0"} for programs
    whose payload boxes / heap closure envs leak BY DESIGN, where ASan should
    prove only no-UAF/no-double-free, not leak-freedom.
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
        ["clang", "-O2", "-Wno-override-module", ll_path,
         *_runtime_object_paths(clang_args), "-o", bin_path, "-lm",
         *clang_args],
        capture_output=True, text=True, timeout=timeout)
    if compile_proc.returncode != 0:
        raise LlvmRunError(
            f"clang failed (exit {compile_proc.returncode}) on {ll_path}:\n"
            f"{compile_proc.stderr}")

    env = None
    if run_env:
        env = dict(os.environ)
        env.update(run_env)
    run_proc = subprocess.run(
        [bin_path], capture_output=True, text=True, timeout=timeout, env=env)
    return run_proc.returncode, run_proc.stdout
