"""Tests for the native C runtime (src/metaxu/runtime/native/metaxu_rt.c).

Strategy: compile the runtime with the build recipe, then compile-and-run
small C driver programs linked against it, asserting on stdout/stderr/exit
codes.  The interpreter (mir_interp.py) is the reference semantics: Vec is
a growable mutable vector with identity semantics, pop on empty and
out-of-bounds indexing abort loudly (no silent fallbacks), and
mx_f64_to_str must match Python's str(float) formatting -- which the float
tests pin by literally comparing against str() of the same values.

An ASan-compiled variant of the round-trip driver asserts exit 0, proving
mx_vec_free / free() pairing leak-free; it is skipped (not weakened) when
the ASan runtime is missing, same probe as test_codegen_llvm.py.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

NATIVE_DIR = Path(__file__).resolve().parents[2] / "runtime" / "native"

# Load the build recipe by file path: src/metaxu/runtime has no __init__.py,
# so a plain package import would depend on namespace-package resolution.
_spec = importlib.util.spec_from_file_location(
    "metaxu_native_build", NATIVE_DIR / "build.py")
build = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("metaxu_native_build", build)
_spec.loader.exec_module(build)


needs_clang = pytest.mark.skipif(
    shutil.which("clang") is None, reason="clang not on PATH")


_ASAN_PROBE: list[bool] = []


def asan_available() -> bool:
    """True when clang can link -fsanitize=address (ASan runtime installed)."""
    if not _ASAN_PROBE:
        if shutil.which("clang") is None:
            _ASAN_PROBE.append(False)
        else:
            import tempfile, os
            with tempfile.TemporaryDirectory(prefix="metaxu_asan_probe_") as d:
                c = os.path.join(d, "t.c")
                with open(c, "w") as fh:
                    fh.write("int main(void){return 0;}\n")
                proc = subprocess.run(
                    ["clang", "-fsanitize=address", c, "-o", os.path.join(d, "t")],
                    capture_output=True, text=True)
                _ASAN_PROBE.append(proc.returncode == 0)
    return _ASAN_PROBE[0]


needs_asan = pytest.mark.skipif(
    not asan_available(),
    reason="clang ASan runtime not available (compile probe failed)")


def compile_and_run(tmp_path: Path, driver_src: str, *,
                    sanitize: bool = False) -> subprocess.CompletedProcess:
    """Write a C driver, compile it against the runtime, and execute it.

    Normal mode links the cached metaxu_rt.o from the build recipe;
    sanitize mode recompiles the runtime source together with the driver
    under -fsanitize=address so the whole binary is instrumented.

    metaxu_effects.c comes along either way: metaxu_rt.c's CATCHABLE
    contract violations (pop on empty, index out of bounds, ...) are raised
    through mx_raisef, which lives there with the try/catch landing pads.
    With no `try` installed -- as in every driver here -- a raise prints the
    same line and abort()s, so these tests observe exactly what they did
    before the split.

    metaxu_threads.c comes along too since the contention work
    (docs/contention_as_permission.md): the Vec mutators' contended-write
    guard reads the per-thread permit counter (mx__write_permit), which
    lives with the mutex primitives.  The three objects always link as one
    unit in production (build.runtime_objects / llvm_run).
    """
    driver = tmp_path / "driver.c"
    driver.write_text(driver_src)
    exe = tmp_path / "driver"
    cmd = ["clang", "-std=c11", "-Wall", "-pthread", f"-I{NATIVE_DIR}"]
    if sanitize:
        cmd += ["-g", "-fsanitize=address",
                str(driver), str(build.RUNTIME_C), str(build.EFFECTS_C),
                str(build.THREADS_C)]
    else:
        obj = build.compile_runtime()
        cmd += [str(driver), str(obj), str(build.compile_effects_runtime()),
                str(build.compile_threads_runtime())]
    cmd += ["-o", str(exe), "-lm"]
    cc = subprocess.run(cmd, capture_output=True, text=True)
    assert cc.returncode == 0, f"driver compile failed:\n{cc.stderr}"
    return subprocess.run([str(exe)], capture_output=True, text=True)


DRIVER_PRELUDE = """\
#include "metaxu_rt.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
"""


# ---------------------------------------------------------------------------
# Build recipe
# ---------------------------------------------------------------------------

@needs_clang
def test_compile_runtime_is_cached_by_mtime(tmp_path):
    obj1 = build.compile_runtime(build_dir=tmp_path)
    assert obj1.exists() and obj1.suffix == ".o"
    stamp = obj1.stat().st_mtime_ns
    obj2 = build.compile_runtime(build_dir=tmp_path)
    assert obj2 == obj1
    assert obj2.stat().st_mtime_ns == stamp, "fresh artifact must not rebuild"


@needs_clang
def test_build_archive_produces_static_lib(tmp_path):
    lib = build.build_archive(build_dir=tmp_path)
    assert lib.name == "libmetaxu_rt.a" and lib.exists()
    assert lib.read_bytes()[:8] == b"!<arch>\n"


# ---------------------------------------------------------------------------
# Vec semantics
# ---------------------------------------------------------------------------

@needs_clang
def test_vec_push_pop_len_get_set_roundtrip(tmp_path):
    proc = compile_and_run(tmp_path, DRIVER_PRELUDE + """
int main(void) {
    mx_vec *v = mx_vec_new();
    printf("%" PRId64 "\\n", mx_vec_len(v));       /* 0 */
    mx_vec_push(v, 10);
    mx_vec_push(v, 20);
    mx_vec_push(v, 30);
    printf("%" PRId64 "\\n", mx_vec_len(v));       /* 3 */
    printf("%" PRId64 "\\n", mx_vec_get(v, 0));    /* 10 */
    printf("%" PRId64 "\\n", mx_vec_get(v, 2));    /* 30 */
    mx_vec_set(v, 1, -7);
    printf("%" PRId64 "\\n", mx_vec_get(v, 1));    /* -7 */
    printf("%" PRId64 "\\n", mx_vec_pop(v));       /* 30 */
    printf("%" PRId64 "\\n", mx_vec_pop(v));       /* -7 */
    printf("%" PRId64 "\\n", mx_vec_len(v));       /* 1 */
    mx_vec *alias = v;                             /* identity semantics */
    mx_vec_push(alias, 99);
    printf("%" PRId64 "\\n", mx_vec_get(v, 1));    /* 99 */
    mx_vec_free(v);
    return 0;
}
""")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "0\n3\n10\n30\n-7\n30\n-7\n1\n99\n"


@needs_clang
def test_vec_pop_on_empty_aborts_with_message(tmp_path):
    proc = compile_and_run(tmp_path, DRIVER_PRELUDE + """
int main(void) {
    mx_vec *v = mx_vec_new();
    mx_vec_push(v, 1);
    mx_vec_pop(v);
    mx_vec_pop(v);          /* empty: must abort, not return garbage */
    printf("unreachable\\n");
    return 0;
}
""")
    assert proc.returncode != 0, "pop on empty Vec must abort nonzero"
    assert "pop: Vec is empty" in proc.stderr
    assert "unreachable" not in proc.stdout


@needs_clang
def test_vec_get_out_of_bounds_aborts(tmp_path):
    proc = compile_and_run(tmp_path, DRIVER_PRELUDE + """
int main(void) {
    mx_vec *v = mx_vec_new();
    mx_vec_push(v, 1);
    mx_vec_push(v, 2);
    mx_vec_get(v, 2);       /* one past the end */
    printf("unreachable\\n");
    return 0;
}
""")
    assert proc.returncode != 0
    assert "index out of bounds: 2 (length 2)" in proc.stderr
    assert "unreachable" not in proc.stdout


@needs_clang
def test_vec_get_negative_index_aborts(tmp_path):
    proc = compile_and_run(tmp_path, DRIVER_PRELUDE + """
int main(void) {
    mx_vec *v = mx_vec_new();
    mx_vec_push(v, 5);
    mx_vec_get(v, -1);      /* no Python-style wraparound */
    return 0;
}
""")
    assert proc.returncode != 0
    assert "index out of bounds: -1 (length 1)" in proc.stderr


@needs_clang
def test_vec_set_out_of_bounds_aborts(tmp_path):
    proc = compile_and_run(tmp_path, DRIVER_PRELUDE + """
int main(void) {
    mx_vec *v = mx_vec_new();
    mx_vec_set(v, 0, 42);   /* empty vec: set never grows */
    return 0;
}
""")
    assert proc.returncode != 0
    assert "index out of bounds: 0 (length 0)" in proc.stderr


@needs_clang
def test_vec_growth_past_initial_capacity(tmp_path):
    # 1000 pushes forces many reallocations (initial cap 8, doubling).
    proc = compile_and_run(tmp_path, DRIVER_PRELUDE + """
int main(void) {
    mx_vec *v = mx_vec_new();
    for (int64_t i = 0; i < 1000; i++) mx_vec_push(v, i);
    int64_t sum = 0;
    for (int64_t i = 0; i < mx_vec_len(v); i++) sum += mx_vec_get(v, i);
    printf("%" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 "\\n",
           mx_vec_len(v), sum, mx_vec_get(v, 999), mx_vec_pop(v));
    printf("%" PRId64 "\\n", mx_vec_len(v));
    mx_vec_free(v);
    return 0;
}
""")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == f"1000 {sum(range(1000))} 999 999\n999\n"


# ---------------------------------------------------------------------------
# Strings
# ---------------------------------------------------------------------------

@needs_clang
def test_str_concat_len_eq(tmp_path):
    proc = compile_and_run(tmp_path, DRIVER_PRELUDE + """
int main(void) {
    char *hw = mx_str_concat("hello, ", "world");
    printf("%s\\n", hw);
    printf("%" PRId64 "\\n", mx_str_len(hw));            /* 12 */
    printf("%" PRId64 "\\n", mx_str_len(""));            /* 0 */
    char *hw2 = mx_str_concat(mx_str_concat("hello", ", "), "world");
    printf("%" PRId64 "\\n", mx_str_eq(hw, hw2));        /* 1: by contents */
    printf("%" PRId64 "\\n", mx_str_eq(hw, "hello"));    /* 0 */
    printf("%" PRId64 "\\n", mx_str_eq("", ""));         /* 1 */
    char *empty = mx_str_concat("", "");
    printf("[%s]%" PRId64 "\\n", empty, mx_str_len(empty));
    free(hw); free(hw2); free(empty);
    return 0;
}
""")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "hello, world\n12\n0\n1\n0\n1\n[]0\n"


@needs_clang
def test_i64_to_str_formats(tmp_path):
    proc = compile_and_run(tmp_path, DRIVER_PRELUDE + """
int main(void) {
    int64_t cases[] = {0, 7, -7, 42, 9223372036854775807LL,
                       (-9223372036854775807LL - 1)};
    for (int i = 0; i < 6; i++) {
        char *s = mx_i64_to_str(cases[i]);
        printf("%s\\n", s);
        free(s);
    }
    return 0;
}
""")
    assert proc.returncode == 0, proc.stderr
    expected = [str(v) for v in
                [0, 7, -7, 42, 2**63 - 1, -(2**63)]]
    assert proc.stdout == "".join(e + "\n" for e in expected)


# The reference for mx_f64_to_str is Python's str(float); every C literal
# below is bit-identical to the Python float the expectation is computed
# from, so the assertion literally pins "matches str(float)".
F64_CASES = [
    ("3.0", 3.0),
    ("0.5", 0.5),
    ("0.1", 0.1),
    ("100.0", 100.0),                       # fixed notation, not 1e+02
    ("-0.0", -0.0),
    ("2.5", 2.5),
    ("1234.5", 1234.5),
    ("-2.75", -2.75),
    ("1.0/3.0", 1 / 3),                     # 17 significant digits
    ("9999999999999998.0", 9999999999999998.0),  # largest fixed exponent
    ("1e16", 1e16),                         # scientific from 1e16 up
    ("1e-4", 1e-4),                         # smallest fixed exponent
    ("1e-5", 1e-5),                         # scientific below 1e-4
    ("1.5e-5", 1.5e-5),
    ("1e100", 1e100),
    ("-1.7976931348623157e308", -1.7976931348623157e308),
    ("5e-324", 5e-324),                     # min subnormal
]


@needs_clang
def test_f64_to_str_matches_python_str(tmp_path):
    lines = "\n".join(
        f'    {{ char *s = mx_f64_to_str({lit}); printf("%s\\n", s); free(s); }}'
        for lit, _ in F64_CASES)
    proc = compile_and_run(tmp_path, DRIVER_PRELUDE + f"""
int main(void) {{
{lines}
    {{ char *s = mx_f64_to_str(INFINITY); printf("%s\\n", s); free(s); }}
    {{ char *s = mx_f64_to_str(-INFINITY); printf("%s\\n", s); free(s); }}
    {{ char *s = mx_f64_to_str(NAN); printf("%s\\n", s); free(s); }}
    return 0;
}}
""")
    assert proc.returncode == 0, proc.stderr
    expected = [str(v) for _, v in F64_CASES] + ["inf", "-inf", "nan"]
    assert proc.stdout.splitlines() == expected


# ---------------------------------------------------------------------------
# ASan: the ownership rules in the README hold (vec_free frees everything
# the vec owns; str results are plain malloc'd buffers)
# ---------------------------------------------------------------------------

@needs_clang
@needs_asan
def test_roundtrip_is_leak_free_under_asan(tmp_path):
    # Exit 0 under -fsanitize=address proves no leak (LeakSanitizer), no
    # double-free, no use-after-free for a full Vec + string workout in
    # which every owned allocation is released.
    proc = compile_and_run(tmp_path, DRIVER_PRELUDE + """
int main(void) {
    mx_vec *v = mx_vec_new();
    for (int64_t i = 0; i < 100; i++) mx_vec_push(v, i);
    mx_vec_set(v, 50, -1);
    int64_t acc = 0;
    while (mx_vec_len(v) > 0) acc += mx_vec_pop(v);
    printf("%" PRId64 "\\n", acc);
    mx_vec_free(v);
    mx_vec_free(NULL);                       /* documented no-op */

    mx_vec *empty = mx_vec_new();
    mx_vec_free(empty);                      /* free with no data buffer */

    char *a = mx_str_concat("abc", "def");
    char *b = mx_i64_to_str(123456789);
    char *c = mx_f64_to_str(2.5);
    char *d = mx_str_concat(a, b);
    printf("%s %s %s %s %" PRId64 "\\n", a, b, c, d, mx_str_eq(a, d));
    free(a); free(b); free(c); free(d);
    return 0;
}
""", sanitize=True)
    assert proc.returncode == 0, f"ASan reported errors:\n{proc.stderr}"
    expected_acc = sum(range(100)) - 50 - 1
    assert proc.stdout == (
        f"{expected_acc}\nabcdef 123456789 2.5 abcdef123456789 0\n")
