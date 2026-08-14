"""C-driver unit tests for the native effects runtime (metaxu_effects.c).

Each test compiles a small C program directly against the runtime objects
(no LLVM backend involved) and checks the process's stdout / exit status.
The programs replay the shape catalogue of test_effect_continuations.py --
the MIR interpreter's reference semantics -- at the C ABI level:

  - resume's value becomes perform's value (round-trip)
  - deep handlers: resume() returns the WHOLE delimited body's value,
    including across function calls (the 42 and 300 cases)
  - multiple performs reach the same installed handler
  - a handler that returns without resuming aborts the scope: its value
    is the handle result and the suspended post-perform code never runs
  - post-resume handler code transforms the body's completion value (103)
  - nested scopes with two effects, innermost-first routing (224)
  - a handler's own perform routes OUTWARD, past its busy scope (101)
  - deep re-arming: after resume, later performs still reach the inner
    handler (both hit inner => 2)
  - single-shot: double resume aborts with the interpreter's message
  - unhandled performs abort with the interpreter's message
  - argument padding (fewer args than params = UNIT/0) and the arity
    error for MORE args than params
  - mx_perform_or_default (increment 15): the declared-default rung —
    default taken when nothing handles the op, handler beating the
    default when one is installed, ONE site resolving both ways in one
    program, the busy-scope skip landing on the default, a default that
    itself performs (it runs on the performing stack), the env
    pass-through, and NULL default keeping mx_perform's abort

ASan builds re-run the nested/abort shapes with LEAK CHECKING ENABLED:
the effect machinery itself (scope records, coroutine stacks, continuation
records) must be fully freed on completion AND on abort -- the
leak-by-design contract covers value boxes, not the scheduler.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from metaxu.runtime.native.build import DEFAULT_BUILD_DIR, runtime_objects

NATIVE_DIR = Path(__file__).resolve().parents[2] / "runtime" / "native"

needs_clang = pytest.mark.skipif(
    shutil.which("clang") is None, reason="clang is not installed")


def _compile_and_run(tmp_path, c_source: str, asan: bool = False,
                     run_env: dict | None = None):
    """Compile a C driver against the runtime objects; run it.
    Returns (returncode, stdout, stderr)."""
    if asan:
        objs = runtime_objects(
            build_dir=DEFAULT_BUILD_DIR.parent / "_build_asan",
            extra_cflags=("-fsanitize=address", "-fno-omit-frame-pointer"))
        extra = ["-fsanitize=address", "-fno-omit-frame-pointer"]
    else:
        objs = runtime_objects()
        extra = []
    src = tmp_path / "driver.c"
    src.write_text(c_source)
    exe = tmp_path / "driver.bin"
    proc = subprocess.run(
        ["clang", "-std=c11", "-O1", "-g", *extra,
         f"-I{NATIVE_DIR}", str(src), *[str(o) for o in objs],
         "-o", str(exe)],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"driver compile failed:\n{proc.stderr}"
    import os
    env = dict(os.environ)
    if run_env:
        env.update(run_env)
    run = subprocess.run([str(exe)], capture_output=True, text=True,
                         timeout=60, env=env)
    return run.returncode, run.stdout, run.stderr


_PRELUDE = r"""
#include <stdio.h>
#include <stdint.h>
#include "metaxu_effects.h"
"""


@needs_clang
def test_perform_resume_round_trip(tmp_path):
    # resume(7) becomes the perform's value; body computes 7 + 1 = 8.
    code, out, _ = _compile_and_run(tmp_path, _PRELUDE + r"""
static int64_t body(void *env) {
    (void)env;
    return mx_perform("Ask", "ask", NULL, 0) + 1;
}
static int64_t handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op; (void)args;
    return mx_resume(k, 7);
}
int main(void) {
    static const char *ops[] = {"ask"};
    static const int64_t nparams[] = {1};
    int64_t r = mx_handle(body, NULL, handler, NULL, "Ask", ops, nparams, 1);
    printf("%lld\n", (long long)r);
    return 0;
}
""")
    assert code == 0
    assert out == "8\n"


@needs_clang
def test_deep_handler_across_function_calls(tmp_path):
    # helper() performs deep inside a call; resume(4) threads back through
    # the whole call: 4 * 10 + 2 = 42.
    code, out, _ = _compile_and_run(tmp_path, _PRELUDE + r"""
static int64_t helper(void) {
    int64_t x = mx_perform("Ask", "ask", NULL, 0);
    return x * 10;
}
static int64_t body(void *env) { (void)env; return helper() + 2; }
static int64_t handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op; (void)args;
    return mx_resume(k, 4);
}
int main(void) {
    static const char *ops[] = {"ask"};
    static const int64_t nparams[] = {1};
    printf("%lld\n", (long long)mx_handle(
        body, NULL, handler, NULL, "Ask", ops, nparams, 1));
    return 0;
}
""")
    assert code == 0
    assert out == "42\n"


@needs_clang
def test_multiple_performs_same_handler(tmp_path):
    # Deep handlers stay installed: 5 + 5 = 10.
    code, out, _ = _compile_and_run(tmp_path, _PRELUDE + r"""
static int64_t body(void *env) {
    (void)env;
    int64_t a = mx_perform("Ask", "ask", NULL, 0);
    int64_t b = mx_perform("Ask", "ask", NULL, 0);
    return a + b;
}
static int64_t handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op; (void)args;
    return mx_resume(k, 5);
}
int main(void) {
    static const char *ops[] = {"ask"};
    static const int64_t nparams[] = {1};
    printf("%lld\n", (long long)mx_handle(
        body, NULL, handler, NULL, "Ask", ops, nparams, 1));
    return 0;
}
""")
    assert code == 0
    assert out == "10\n"


@needs_clang
def test_abort_when_handler_does_not_resume(tmp_path):
    # Handler returns 99 without resuming: 99 is the handle value and the
    # post-perform body code (the "unreachable" print) never runs.
    code, out, _ = _compile_and_run(tmp_path, _PRELUDE + r"""
static int64_t body(void *env) {
    (void)env;
    int64_t x = mx_perform("Fail", "fail", NULL, 0);
    printf("unreachable\n");
    return x + 1;
}
static int64_t handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op; (void)args; (void)k;
    return 99; /* declines to resume */
}
int main(void) {
    static const char *ops[] = {"fail"};
    static const int64_t nparams[] = {1};
    printf("%lld\n", (long long)mx_handle(
        body, NULL, handler, NULL, "Fail", ops, nparams, 1));
    return 0;
}
""")
    assert code == 0
    assert out == "99\n"


@needs_clang
def test_post_resume_handler_code_transforms_body_value(tmp_path):
    # Deep semantics: rest = resume(1) is the body's completion value
    # (1 + 2 = 3); the handler's return 3 + 100 = 103 is the handle value.
    code, out, _ = _compile_and_run(tmp_path, _PRELUDE + r"""
static int64_t body(void *env) {
    (void)env;
    return mx_perform("Ask", "ask", NULL, 0) + 2;
}
static int64_t handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op; (void)args;
    int64_t rest = mx_resume(k, 1);
    return rest + 100;
}
int main(void) {
    static const char *ops[] = {"ask"};
    static const int64_t nparams[] = {1};
    printf("%lld\n", (long long)mx_handle(
        body, NULL, handler, NULL, "Ask", ops, nparams, 1));
    return 0;
}
""")
    assert code == 0
    assert out == "103\n"


@needs_clang
def test_resume_returns_whole_body_value_across_call(tmp_path):
    # The 300 case: helper() returns the resumed 1, body computes 1 + 2 = 3,
    # rest = 3, handler yields 3 * 100 = 300.
    code, out, _ = _compile_and_run(tmp_path, _PRELUDE + r"""
static int64_t helper(void) { return mx_perform("Ask", "ask", NULL, 0); }
static int64_t body(void *env) { (void)env; return helper() + 2; }
static int64_t handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op; (void)args;
    int64_t rest = mx_resume(k, 1);
    return rest * 100;
}
int main(void) {
    static const char *ops[] = {"ask"};
    static const int64_t nparams[] = {1};
    printf("%lld\n", (long long)mx_handle(
        body, NULL, handler, NULL, "Ask", ops, nparams, 1));
    return 0;
}
""")
    assert code == 0
    assert out == "300\n"


_NESTED_224 = _PRELUDE + r"""
/* The 224 case: two nested scopes, post-resume code in both handlers,
 * performs in a called function.
 *   fn_body(): v = get() -> 10; log("hi"); return v + 1 = 11
 *   inner (Logger) body: fn_body() + 100 = 111; log handler: r=111 -> 112
 *   outer (State) handler: rest = 112 -> 224. */
static int64_t fn_body(void) {
    int64_t v = mx_perform("State", "get", NULL, 0);
    mx_perform("Logger", "log", NULL, 0);
    return v + 1;
}
static int64_t inner_body(void *env) { (void)env; return fn_body() + 100; }
static int64_t logger_handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op; (void)args;
    int64_t r = mx_resume(k, 0);
    return r + 1;
}
static int64_t outer_body(void *env) {
    (void)env;
    static const char *ops[] = {"log"};
    static const int64_t nparams[] = {1};
    return mx_handle(inner_body, NULL, logger_handler, NULL, "Logger",
                     ops, nparams, 1);
}
static int64_t state_handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op; (void)args;
    int64_t rest = mx_resume(k, 10);
    return rest * 2;
}
int main(void) {
    static const char *ops[] = {"get"};
    static const int64_t nparams[] = {1};
    printf("%lld\n", (long long)mx_handle(
        outer_body, NULL, state_handler, NULL, "State", ops, nparams, 1));
    return 0;
}
"""


@needs_clang
def test_nested_two_effect_scopes(tmp_path):
    code, out, _ = _compile_and_run(tmp_path, _NESTED_224)
    assert code == 0
    assert out == "224\n"


@needs_clang
def test_handler_self_perform_routes_outward(tmp_path):
    # A handler case performing its own op runs with its scope busy, so the
    # perform routes to the OUTER handler: outer resumes 100, inner resumes
    # 101, both handles complete with 101.
    code, out, _ = _compile_and_run(tmp_path, _PRELUDE + r"""
static int64_t inner_body(void *env) {
    (void)env;
    return mx_perform("Ask", "ask", NULL, 0);
}
static int64_t inner_handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op; (void)args;
    int64_t outer = mx_perform("Ask", "ask", NULL, 0); /* routes OUTWARD */
    return mx_resume(k, outer + 1);
}
static int64_t outer_body(void *env) {
    (void)env;
    static const char *ops[] = {"ask"};
    static const int64_t nparams[] = {1};
    return mx_handle(inner_body, NULL, inner_handler, NULL, "Ask",
                     ops, nparams, 1);
}
static int64_t outer_handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op; (void)args;
    return mx_resume(k, 100);
}
int main(void) {
    static const char *ops[] = {"ask"};
    static const int64_t nparams[] = {1};
    printf("%lld\n", (long long)mx_handle(
        outer_body, NULL, outer_handler, NULL, "Ask", ops, nparams, 1));
    return 0;
}
""")
    assert code == 0
    assert out == "101\n"


@needs_clang
def test_rearming_after_resume_reaches_inner_handler(tmp_path):
    # After resume(), later performs must still hit the INNER handler
    # (both performs resumed with 1 => 2, never the outer 1000).
    code, out, _ = _compile_and_run(tmp_path, _PRELUDE + r"""
static int64_t inner_body(void *env) {
    (void)env;
    return mx_perform("Ask", "ask", NULL, 0) + mx_perform("Ask", "ask", NULL, 0);
}
static int64_t inner_handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op; (void)args;
    return mx_resume(k, 1);
}
static int64_t outer_body(void *env) {
    (void)env;
    static const char *ops[] = {"ask"};
    static const int64_t nparams[] = {1};
    return mx_handle(inner_body, NULL, inner_handler, NULL, "Ask",
                     ops, nparams, 1);
}
static int64_t outer_handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op; (void)args;
    return mx_resume(k, 1000);
}
int main(void) {
    static const char *ops[] = {"ask"};
    static const int64_t nparams[] = {1};
    printf("%lld\n", (long long)mx_handle(
        outer_body, NULL, outer_handler, NULL, "Ask", ops, nparams, 1));
    return 0;
}
""")
    assert code == 0
    assert out == "2\n"


@needs_clang
def test_op_arguments_and_unit_padding(tmp_path):
    # Two-arg op binds both parameters (40 + 2 = 42); a zero-arg perform
    # against a one-param case pads with UNIT (0).
    code, out, _ = _compile_and_run(tmp_path, _PRELUDE + r"""
static int64_t body(void *env) {
    (void)env;
    int64_t args[2] = {40, 2};
    int64_t sum = mx_perform("Math", "add", args, 2);
    int64_t pad = mx_perform("Math", "zero", NULL, 0); /* padded to (0) */
    return sum + pad;
}
static int64_t handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env;
    if (op == 0) return mx_resume(k, args[0] + args[1]);
    return mx_resume(k, args[0] + 7); /* sees the UNIT pad = 0 */
}
int main(void) {
    static const char *ops[] = {"add", "zero"};
    static const int64_t nparams[] = {2, 1};
    printf("%lld\n", (long long)mx_handle(
        body, NULL, handler, NULL, "Math", ops, nparams, 2));
    return 0;
}
""")
    assert code == 0
    assert out == "49\n"  # 42 + (0 + 7)


@needs_clang
def test_double_resume_aborts_with_single_shot_message(tmp_path):
    code, out, err = _compile_and_run(tmp_path, _PRELUDE + r"""
static int64_t body(void *env) {
    (void)env;
    return mx_perform("Ask", "ask", NULL, 0);
}
static int64_t handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op; (void)args;
    return mx_resume(k, 1) + mx_resume(k, 2);
}
int main(void) {
    static const char *ops[] = {"ask"};
    static const int64_t nparams[] = {1};
    printf("%lld\n", (long long)mx_handle(
        body, NULL, handler, NULL, "Ask", ops, nparams, 1));
    return 0;
}
""")
    assert code != 0
    assert "single-shot violation" in err


@needs_clang
def test_unhandled_perform_aborts_with_message(tmp_path):
    code, _, err = _compile_and_run(tmp_path, _PRELUDE + r"""
int main(void) {
    mx_perform("Ask", "ask", NULL, 0);
    return 0;
}
""")
    assert code != 0
    assert "Unhandled effect operation: 'ask'" in err


@needs_clang
def test_too_many_args_aborts_with_arity_message(tmp_path):
    code, _, err = _compile_and_run(tmp_path, _PRELUDE + r"""
static int64_t body(void *env) {
    (void)env;
    int64_t args[2] = {1, 2};
    return mx_perform("Ask", "ask", args, 2); /* case declares 1 param */
}
static int64_t handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op; (void)args;
    return mx_resume(k, 0);
}
int main(void) {
    static const char *ops[] = {"ask"};
    static const int64_t nparams[] = {1};
    return (int)mx_handle(body, NULL, handler, NULL, "Ask", ops, nparams, 1);
}
""")
    assert code != 0
    assert "performed with 2 argument(s)" in err


@needs_clang
def test_effect_name_disambiguates_same_op_name(tmp_path):
    # Two scopes both handle an op named "get"; the perform names its
    # effect, so it must skip the inner (Logger) scope and reach State.
    code, out, _ = _compile_and_run(tmp_path, _PRELUDE + r"""
static int64_t inner_body(void *env) {
    (void)env;
    return mx_perform("State", "get", NULL, 0);
}
static int64_t logger_handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op; (void)args;
    return mx_resume(k, 111111); /* wrong handler */
}
static int64_t outer_body(void *env) {
    (void)env;
    static const char *ops[] = {"get"};
    static const int64_t nparams[] = {1};
    return mx_handle(inner_body, NULL, logger_handler, NULL, "Logger",
                     ops, nparams, 1);
}
static int64_t state_handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op; (void)args;
    return mx_resume(k, 5);
}
int main(void) {
    static const char *ops[] = {"get"};
    static const int64_t nparams[] = {1};
    printf("%lld\n", (long long)mx_handle(
        outer_body, NULL, state_handler, NULL, "State", ops, nparams, 1));
    return 0;
}
""")
    assert code == 0
    assert out == "5\n"


# ---------------------------------------------------------------------------
# ASan: the effect machinery itself is leak-clean (stacks/scopes/ks freed
# on completion AND on abort); fiber annotations keep ASan accurate across
# swapcontext.
# ---------------------------------------------------------------------------

def _asan_available() -> bool:
    if shutil.which("clang") is None:
        return False
    import tempfile, os
    with tempfile.TemporaryDirectory(prefix="metaxu_asan_probe_fx_") as d:
        c = os.path.join(d, "t.c")
        with open(c, "w") as fh:
            fh.write("int main(void){return 0;}\n")
        proc = subprocess.run(
            ["clang", "-fsanitize=address", c, "-o", os.path.join(d, "t")],
            capture_output=True, text=True)
        return proc.returncode == 0


needs_asan = pytest.mark.skipif(
    not _asan_available(), reason="clang ASan runtime not available")


@needs_asan
def test_asan_nested_scopes_fully_leak_checked(tmp_path):
    # Full leak checking ON: every coroutine stack, scope record and
    # continuation record must be freed at normal completion.
    code, out, err = _compile_and_run(tmp_path, _NESTED_224, asan=True)
    assert code == 0, f"ASan flagged the nested-scope run:\n{err}"
    assert out == "224\n"


@needs_asan
def test_asan_abort_teardown_fully_leak_checked(tmp_path):
    # Abort tears down the parked body chain: the aborted scope's stack and
    # a NESTED scope's stack inside it are freed (cascade), leak-checked.
    code, out, err = _compile_and_run(tmp_path, _PRELUDE + r"""
static int64_t inner_body(void *env) {
    (void)env;
    int64_t x = mx_perform("Fail", "fail", NULL, 0); /* aborts outer */
    printf("unreachable\n");
    return x;
}
static int64_t log_handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op; (void)args;
    return mx_resume(k, 0);
}
static int64_t outer_body(void *env) {
    (void)env;
    static const char *ops[] = {"log"};
    static const int64_t nparams[] = {1};
    /* nested scope suspended inside the aborted body chain */
    return mx_handle(inner_body, NULL, log_handler, NULL, "Logger",
                     ops, nparams, 1);
}
static int64_t fail_handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op; (void)args; (void)k;
    return 7; /* declines to resume: abort */
}
int main(void) {
    static const char *ops[] = {"fail"};
    static const int64_t nparams[] = {1};
    printf("%lld\n", (long long)mx_handle(
        outer_body, NULL, fail_handler, NULL, "Fail", ops, nparams, 1));
    return 0;
}
""", asan=True)
    assert code == 0, f"ASan flagged the abort-teardown run:\n{err}"
    assert out == "7\n"


@needs_asan
def test_asan_deep_recursion_and_multiple_performs(tmp_path):
    # Heavier traffic: 200 performs through a recursive body, leak-checked.
    code, out, err = _compile_and_run(tmp_path, _PRELUDE + r"""
static int64_t count(int64_t n) {
    if (n == 0) return 0;
    return mx_perform("Ask", "one", NULL, 0) + count(n - 1);
}
static int64_t body(void *env) { (void)env; return count(200); }
static int64_t handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op; (void)args;
    return mx_resume(k, 1);
}
int main(void) {
    static const char *ops[] = {"one"};
    static const int64_t nparams[] = {1};
    printf("%lld\n", (long long)mx_handle(
        body, NULL, handler, NULL, "Ask", ops, nparams, 1));
    return 0;
}
""", asan=True)
    assert code == 0, f"ASan flagged the deep-recursion run:\n{err}"
    assert out == "200\n"


# ---------------------------------------------------------------------------
# mx_perform_or_default (increment 15): the DYNAMIC default rung
#
# Precedence, mirroring mir_interp's `perform`:
#   innermost non-busy handler frame  >  declared `= expr` default  >  error
# (the interpreter's `with SYMBOL` runtime-mapping rung between the first
# two has no native implementation; the compiler demotes ops declaring one).
#
# A default RUNS ON THE PERFORMING STACK: no coroutine, no scope record, no
# continuation.  These drivers prove each claim at the C ABI level.
# ---------------------------------------------------------------------------

@needs_clang
def test_default_taken_when_no_scope_handles_the_op(tmp_path):
    # No mx_handle at all: where mx_perform would abort, the default runs
    # and its value becomes the perform's value (1 + 100 = 101).
    code, out, _ = _compile_and_run(tmp_path, _PRELUDE + r"""
static int64_t dflt(void *env, const int64_t *args) {
    (void)env;
    return args[0] + 100;
}
int main(void) {
    int64_t args[1] = {1};
    printf("%lld\n", (long long)mx_perform_or_default(
        "Ask", "ask", args, 1, dflt, NULL));
    return 0;
}
""")
    assert code == 0
    assert out == "101\n"


@needs_clang
def test_installed_handler_beats_the_default(tmp_path):
    # Same op, same call: with a scope installed the handler answers and the
    # default is never entered (it would print if it were).
    code, out, _ = _compile_and_run(tmp_path, _PRELUDE + r"""
static int64_t dflt(void *env, const int64_t *args) {
    (void)env; (void)args;
    printf("DEFAULT RAN\n");
    return -1;
}
static int64_t body(void *env) {
    (void)env;
    int64_t args[1] = {5};
    return mx_perform_or_default("Ask", "ask", args, 1, dflt, NULL);
}
static int64_t handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op;
    return mx_resume(k, args[0] * 2);
}
int main(void) {
    static const char *ops[] = {"ask"};
    static const int64_t nparams[] = {1};
    printf("%lld\n", (long long)mx_handle(
        body, NULL, handler, NULL, "Ask", ops, nparams, 1));
    return 0;
}
""")
    assert code == 0
    assert out == "10\n"


@needs_clang
def test_same_site_resolves_both_ways_in_one_program(tmp_path):
    # THE dynamic case: ONE perform site, called once inside the handle and
    # once outside it.  Inside -> 10 (handler), outside -> 105 (default).
    code, out, _ = _compile_and_run(tmp_path, _PRELUDE + r"""
static int64_t dflt(void *env, const int64_t *args) {
    (void)env;
    return args[0] + 100;
}
static int64_t site(int64_t x) {
    int64_t args[1] = {x};
    return mx_perform_or_default("Ask", "ask", args, 1, dflt, NULL);
}
static int64_t body(void *env) { (void)env; return site(5); }
static int64_t handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op;
    return mx_resume(k, args[0] * 2);
}
int main(void) {
    static const char *ops[] = {"ask"};
    static const int64_t nparams[] = {1};
    printf("%lld\n", (long long)mx_handle(
        body, NULL, handler, NULL, "Ask", ops, nparams, 1));
    printf("%lld\n", (long long)site(5));
    return 0;
}
""")
    assert code == 0
    assert out == "10\n105\n"


@needs_clang
def test_default_skips_a_busy_scope_like_the_interpreter(tmp_path):
    # A handler case's own perform routes OUTWARD past its busy scope
    # (mx__find_scope's busy skip == mir_interp._find_mir_frame's).  There
    # is no other scope, so the default answers: (2+1)*10 = 30, resume(31).
    code, out, _ = _compile_and_run(tmp_path, _PRELUDE + r"""
static int64_t dflt(void *env, const int64_t *args) {
    (void)env;
    return args[0] * 10;
}
static int64_t site(int64_t x) {
    int64_t args[1] = {x};
    return mx_perform_or_default("Ask", "ask", args, 1, dflt, NULL);
}
static int64_t body(void *env) { (void)env; return site(2); }
static int64_t handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op;
    int64_t outer = site(args[0] + 1);  /* our own scope is busy */
    return mx_resume(k, outer + 1);
}
int main(void) {
    static const char *ops[] = {"ask"};
    static const int64_t nparams[] = {1};
    printf("%lld\n", (long long)mx_handle(
        body, NULL, handler, NULL, "Ask", ops, nparams, 1));
    return 0;
}
""")
    assert code == 0
    assert out == "31\n"


@needs_clang
def test_default_runs_on_the_performing_stack_and_may_perform(tmp_path):
    # The default is NOT a suspension: it runs right here.  Proof: it
    # performs an op that IS handled, so the park/resume happens on the
    # performing fiber and the scope's deep completion value still flows
    # out.  `base` is handled (v * 100); `ask` is defaulted (+5) in both.
    code, out, _ = _compile_and_run(tmp_path, _PRELUDE + r"""
static int64_t base_dflt(void *env, const int64_t *args) {
    (void)env; return args[0];
}
static int64_t ask_dflt(void *env, const int64_t *args) {
    (void)env;
    int64_t a[1] = {args[0]};
    return mx_perform_or_default("Cap", "base", a, 1, base_dflt, NULL) + 5;
}
static int64_t inner(int64_t x) {
    int64_t a[1] = {x};
    return mx_perform_or_default("Cap", "ask", a, 1, ask_dflt, NULL) * 2;
}
static int64_t body(void *env) { (void)env; return inner(3); }
static int64_t handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op;
    return mx_resume(k, args[0] * 100);
}
int main(void) {
    static const char *ops[] = {"base"};
    static const int64_t nparams[] = {1};
    printf("%lld\n", (long long)inner(3));          /* both defaulted: 16 */
    printf("%lld\n", (long long)mx_handle(          /* base handled: 610 */
        body, NULL, handler, NULL, "Cap", ops, nparams, 1));
    return 0;
}
""")
    assert code == 0
    assert out == "16\n610\n"


@needs_clang
def test_default_env_pointer_is_passed_through(tmp_path):
    # The env word reaches the thunk untouched (the ABI slot a capturing
    # default would use; the compiler passes NULL today).
    code, out, _ = _compile_and_run(tmp_path, _PRELUDE + r"""
static int64_t dflt(void *env, const int64_t *args) {
    (void)args;
    return *(int64_t *)env;
}
int main(void) {
    int64_t captured = 77;
    printf("%lld\n", (long long)mx_perform_or_default(
        "Ask", "ask", NULL, 0, dflt, &captured));
    return 0;
}
""")
    assert code == 0
    assert out == "77\n"


@needs_clang
def test_null_default_keeps_the_unhandled_abort(tmp_path):
    # mx_perform's abort path is untouched: an op with no default still
    # dies with the interpreter's message, through either entry point.
    code, _, err = _compile_and_run(tmp_path, _PRELUDE + r"""
int main(void) {
    mx_perform_or_default("Ask", "ask", NULL, 0, NULL, NULL);
    return 0;
}
""")
    assert code != 0
    assert "Unhandled effect operation: 'ask'" in err


@needs_clang
def test_scope_route_keeps_padding_and_arity_checks(tmp_path):
    # Routing through a scope is byte-for-byte mx_perform: missing trailing
    # args still pad with UNIT/0, and MORE args than the case declares is
    # still the loud arity error.  The default rung softens neither.
    code, out, _ = _compile_and_run(tmp_path, _PRELUDE + r"""
static int64_t dflt(void *env, const int64_t *args) {
    (void)env; (void)args; return -1;
}
static int64_t body(void *env) {
    (void)env;
    return mx_perform_or_default("Ask", "ask", NULL, 0, dflt, NULL);
}
static int64_t handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op;
    return mx_resume(k, args[0] + args[1] + 9);  /* both padded to 0 */
}
int main(void) {
    static const char *ops[] = {"ask"};
    static const int64_t nparams[] = {2};
    printf("%lld\n", (long long)mx_handle(
        body, NULL, handler, NULL, "Ask", ops, nparams, 1));
    return 0;
}
""")
    assert code == 0
    assert out == "9\n"

    code, _, err = _compile_and_run(tmp_path, _PRELUDE + r"""
static int64_t dflt(void *env, const int64_t *args) {
    (void)env; (void)args; return -1;
}
static int64_t body(void *env) {
    (void)env;
    int64_t args[2] = {1, 2};
    return mx_perform_or_default("Ask", "ask", args, 2, dflt, NULL);
}
static int64_t handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op; (void)args;
    return mx_resume(k, 0);
}
int main(void) {
    static const char *ops[] = {"ask"};
    static const int64_t nparams[] = {1};
    mx_handle(body, NULL, handler, NULL, "Ask", ops, nparams, 1);
    return 0;
}
""")
    assert code != 0
    assert "declares only 1 parameter" in err


@needs_asan
def test_asan_default_route_allocates_nothing(tmp_path):
    # Leak-checked: the default rung installs no scope, allocates no
    # coroutine stack and no continuation record — 1000 defaulted performs
    # must be as clean as none.
    code, out, err = _compile_and_run(tmp_path, _PRELUDE + r"""
static int64_t dflt(void *env, const int64_t *args) {
    (void)env; return args[0];
}
int main(void) {
    int64_t total = 0;
    for (int i = 0; i < 1000; i++) {
        int64_t a[1] = {1};
        total += mx_perform_or_default("Ask", "ask", a, 1, dflt, NULL);
    }
    printf("%lld\n", (long long)total);
    return 0;
}
""", asan=True)
    assert code == 0, f"ASan flagged the default-route run:\n{err}"
    assert out == "1000\n"


@needs_asan
def test_asan_mixed_routes_stay_leak_clean(tmp_path):
    # Alternating handler / default answers (the handler's own performs skip
    # its busy scope and default), with the scope torn down at the end.
    code, out, err = _compile_and_run(tmp_path, _PRELUDE + r"""
static int64_t dflt(void *env, const int64_t *args) {
    (void)env; return args[0] * 10;
}
static int64_t site(int64_t x) {
    int64_t a[1] = {x};
    return mx_perform_or_default("Ask", "ask", a, 1, dflt, NULL);
}
static int64_t count(int64_t n) {
    if (n == 0) return 0;
    return site(1) + count(n - 1);
}
static int64_t body(void *env) { (void)env; return count(50); }
static int64_t handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op;
    return mx_resume(k, args[0] + site(2)); /* our scope is busy -> default */
}
int main(void) {
    static const char *ops[] = {"ask"};
    static const int64_t nparams[] = {1};
    printf("%lld\n", (long long)mx_handle(
        body, NULL, handler, NULL, "Ask", ops, nparams, 1));
    printf("%lld\n", (long long)site(3));
    return 0;
}
""", asan=True)
    assert code == 0, f"ASan flagged the mixed-route run:\n{err}"
    assert out == "1050\n30\n"
