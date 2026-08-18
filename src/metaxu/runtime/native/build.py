"""Build recipe for the native Metaxu runtime (metaxu_rt.c).

Dependency-free: subprocess + clang only.  Usage:

    from metaxu.runtime.native.build import compile_runtime, build_archive
    obj = compile_runtime()          # -> .../native/_build/metaxu_rt.o
    lib = build_archive()            # -> .../native/_build/libmetaxu_rt.a

or from the shell:

    python src/metaxu/runtime/native/build.py [--archive]

Artifacts are cached by mtime: the object is rebuilt only when it is
missing or older than metaxu_rt.c / metaxu_rt.h (or this script).
A failed compile raises RuntimeError with clang's stderr -- no silent
fallbacks, matching the runtime's own error philosophy.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Sequence

NATIVE_DIR = Path(__file__).resolve().parent
RUNTIME_C = NATIVE_DIR / "metaxu_rt.c"
RUNTIME_H = NATIVE_DIR / "metaxu_rt.h"
EFFECTS_C = NATIVE_DIR / "metaxu_effects.c"
EFFECTS_H = NATIVE_DIR / "metaxu_effects.h"
THREADS_C = NATIVE_DIR / "metaxu_threads.c"
THREADS_H = NATIVE_DIR / "metaxu_threads.h"
DEFAULT_BUILD_DIR = NATIVE_DIR / "_build"

# C11, optimized, PIC so the object can also land in shared objects later.
# -pthread everywhere (not just metaxu_threads.c): it defines _REENTRANT
# consistently, and the effects runtime's _Thread_local state is part of
# the threads contract (docs/threads_runtime.md).
CFLAGS: tuple[str, ...] = (
    "-std=c11", "-O2", "-g", "-fPIC", "-Wall", "-Wextra", "-pthread",
)


def _sources_mtime() -> float:
    # metaxu_rt.c includes metaxu_effects.h (mx_raisef: catchable failures),
    # so the effects header is one of its sources too.
    return max(p.stat().st_mtime
               for p in (RUNTIME_C, RUNTIME_H, EFFECTS_H, Path(__file__)))


def _is_fresh(artifact: Path) -> bool:
    return artifact.exists() and artifact.stat().st_mtime >= _sources_mtime()


def _run(cmd: Sequence[str], what: str) -> None:
    proc = subprocess.run(list(cmd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{what} failed (exit {proc.returncode}):\n"
            f"  command: {' '.join(cmd)}\n{proc.stderr}"
        )


def compile_runtime(
    build_dir: Optional[Path] = None,
    clang: str = "clang",
    extra_cflags: Sequence[str] = (),
) -> Path:
    """Compile metaxu_rt.c to metaxu_rt.o and return the object's path.

    Cached by mtime: recompiles only when the artifact is missing or older
    than the runtime sources.  ``extra_cflags`` (e.g. ``-fsanitize=address``)
    are appended after the default CFLAGS; callers passing extra flags
    should also pass a dedicated ``build_dir`` so differently-flagged
    objects never share a cache slot.
    """
    build_dir = Path(build_dir) if build_dir is not None else DEFAULT_BUILD_DIR
    build_dir.mkdir(parents=True, exist_ok=True)
    obj = build_dir / "metaxu_rt.o"
    if _is_fresh(obj):
        return obj
    _run(
        [clang, *CFLAGS, *extra_cflags, "-c", str(RUNTIME_C), "-o", str(obj)],
        "compiling metaxu_rt.c",
    )
    return obj


def _effects_mtime() -> float:
    return max(p.stat().st_mtime for p in (EFFECTS_C, EFFECTS_H, Path(__file__)))


def compile_effects_runtime(
    build_dir: Optional[Path] = None,
    clang: str = "clang",
    extra_cflags: Sequence[str] = (),
) -> Path:
    """Compile metaxu_effects.c (the ucontext coroutine scheduler backing
    handle_scope/perform/resume) to metaxu_effects.o and return its path.

    Same contract as compile_runtime: mtime-cached, extra flags (e.g.
    ``-fsanitize=address``) should come with a dedicated ``build_dir``.
    """
    build_dir = Path(build_dir) if build_dir is not None else DEFAULT_BUILD_DIR
    build_dir.mkdir(parents=True, exist_ok=True)
    obj = build_dir / "metaxu_effects.o"
    if obj.exists() and obj.stat().st_mtime >= _effects_mtime():
        return obj
    _run(
        [clang, *CFLAGS, *extra_cflags, "-c", str(EFFECTS_C), "-o", str(obj)],
        "compiling metaxu_effects.c",
    )
    return obj


def _threads_mtime() -> float:
    # metaxu_threads.c includes metaxu_effects.h (mx_try/mx_raise: the
    # child's landing pad and the catchable errors).
    return max(p.stat().st_mtime
               for p in (THREADS_C, THREADS_H, EFFECTS_H, Path(__file__)))


def compile_threads_runtime(
    build_dir: Optional[Path] = None,
    clang: str = "clang",
    extra_cflags: Sequence[str] = (),
) -> Path:
    """Compile metaxu_threads.c (the pthreads-backed Thread/Mutex effect
    primitives, docs/threads_runtime.md) to metaxu_threads.o and return
    its path.  Same contract as compile_runtime: mtime-cached, extra
    flags should come with a dedicated ``build_dir``."""
    build_dir = Path(build_dir) if build_dir is not None else DEFAULT_BUILD_DIR
    build_dir.mkdir(parents=True, exist_ok=True)
    obj = build_dir / "metaxu_threads.o"
    if obj.exists() and obj.stat().st_mtime >= _threads_mtime():
        return obj
    _run(
        [clang, *CFLAGS, *extra_cflags, "-c", str(THREADS_C), "-o", str(obj)],
        "compiling metaxu_threads.c",
    )
    return obj


def runtime_objects(
    build_dir: Optional[Path] = None,
    clang: str = "clang",
    extra_cflags: Sequence[str] = (),
) -> tuple[Path, Path, Path]:
    """Every native runtime object a metaxu binary links: (metaxu_rt.o,
    metaxu_effects.o, metaxu_threads.o)."""
    return (
        compile_runtime(build_dir=build_dir, clang=clang,
                        extra_cflags=extra_cflags),
        compile_effects_runtime(build_dir=build_dir, clang=clang,
                                extra_cflags=extra_cflags),
        compile_threads_runtime(build_dir=build_dir, clang=clang,
                                extra_cflags=extra_cflags),
    )


def build_archive(
    build_dir: Optional[Path] = None,
    clang: str = "clang",
    ar: str = "ar",
) -> Path:
    """Build libmetaxu_rt.a (via ``ar rcs``; all runtime objects) and
    return its path."""
    obj = compile_runtime(build_dir=build_dir, clang=clang)
    fx = compile_effects_runtime(build_dir=build_dir, clang=clang)
    thr = compile_threads_runtime(build_dir=build_dir, clang=clang)
    lib = obj.parent / "libmetaxu_rt.a"
    if _is_fresh(lib) and lib.stat().st_mtime >= max(
            obj.stat().st_mtime, fx.stat().st_mtime, thr.stat().st_mtime):
        return lib
    _run([ar, "rcs", str(lib), str(obj), str(fx), str(thr)],
         "archiving libmetaxu_rt.a")
    return lib


if __name__ == "__main__":
    import sys

    if "--archive" in sys.argv[1:]:
        print(build_archive())
    else:
        print(compile_runtime())
