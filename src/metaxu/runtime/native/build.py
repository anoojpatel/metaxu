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
DEFAULT_BUILD_DIR = NATIVE_DIR / "_build"

# C11, optimized, PIC so the object can also land in shared objects later.
CFLAGS: tuple[str, ...] = (
    "-std=c11", "-O2", "-g", "-fPIC", "-Wall", "-Wextra",
)


def _sources_mtime() -> float:
    return max(p.stat().st_mtime for p in (RUNTIME_C, RUNTIME_H, Path(__file__)))


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


def build_archive(
    build_dir: Optional[Path] = None,
    clang: str = "clang",
    ar: str = "ar",
) -> Path:
    """Build libmetaxu_rt.a (via ``ar rcs``) and return its path."""
    obj = compile_runtime(build_dir=build_dir, clang=clang)
    lib = obj.parent / "libmetaxu_rt.a"
    if _is_fresh(lib) and lib.stat().st_mtime >= obj.stat().st_mtime:
        return lib
    _run([ar, "rcs", str(lib), str(obj)], "archiving libmetaxu_rt.a")
    return lib


if __name__ == "__main__":
    import sys

    if "--archive" in sys.argv[1:]:
        print(build_archive())
    else:
        print(compile_runtime())
