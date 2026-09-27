"""Direct scopes: handle sites whose every case tail-resumes or never
resumes run on the current stack natively (compiler/effect_tail.py
``direct_case``, runtime/native/metaxu_effects.c ``mx_handle_direct``).

A perform against such a scope is a plain call of its case: no coroutine,
no continuation record, no context switch.  A case that returns without
resuming aborts the scope by unwinding to its handle, hopping fiber by
fiber when the perform ran inside a nested coroutine scope.  The
interpreter is untouched (nothing observable changes), so every native
program here is checked against it, including the shapes where the
unwinding crosses coroutine boundaries and where a case fails.

All tests go through parsed source, per the project convention.
"""
from __future__ import annotations

import subprocess

import pytest

from metaxu.compiler.effect_tail import (
    direct_case, handler_case_fns, program_direct_cases)
from metaxu.compiler.llvm_run import compile_and_run
from metaxu.compiler.tests.test_effect_tail_resume import (
    assert_native_matches_interp, interp_run, llvm_from_source,
    mir_from_source, needs_asan, needs_clang)

# ---------------------------------------------------------------------------
# Programs
# ---------------------------------------------------------------------------

_COUNTER = """
effect Counter { tick(n: int) -> int }
fn loop_it(n: int) -> int {
    let @mut acc = 0;
    let @mut i = 0;
    while i < n { acc = perform Counter.tick(acc); i = i + 1 };
    acc
}
fn main() -> int {
    let r = handle Counter with {
        tick(n) -> resume(n + 1)
    } in { loop_it({N}) };
    print(r);
    0
}
"""

_ABORT_ONLY = """
effect Stop { stop(n: int) -> int }
fn main() -> int {
    let r = handle Stop with {
        stop(n) -> n * 2
    } in {
        let a = perform Stop.stop(21);
        print(a);
        999
    };
    print(r);
    0
}
"""

_MIXED_TAKE = """
effect Counter { tick(n: int) -> int }
fn loop_it() -> int {
    let @mut acc = 0;
    while true { acc = perform Counter.tick(acc) };
    acc
}
fn main() -> int {
    let r = handle Counter with {
        tick(n) -> if n < 10 { resume(n + 1) } else { n * 1000 }
    } in { loop_it() };
    print(r);
    0
}
"""

_NON_TAIL = """
effect Counter { tick(n: int) -> int }
fn main() -> int {
    let r = handle Counter with {
        tick(n) -> { let v = resume(n + 1); v + 0 }
    } in { perform Counter.tick(1) + perform Counter.tick(10) };
    print(r);
    0
}
"""

_TRY_CAPTURES_K = """
effect Counter { tick(n: int) -> int }
fn main() -> int {
    let r = handle Counter with {
        tick(n) -> { let v = try { resume(n) } catch e { 0 }; v + 1 }
    } in { perform Counter.tick(5) };
    print(r);
    0
}
"""

_NESTED_DIRECT_ABORT = """
effect A { a(n: int) -> int }
effect B { b(n: int) -> int }
fn main() -> int {
    let r = handle A with {
        a(n) -> if n < 3 { resume(n + 1) } else { 100 + n }
    } in {
        let s = handle B with { b(n) -> resume(n * 2) } in {
            let @mut x = 0;
            let @mut i = 0;
            while i < 10 { x = perform A.a(x); x = perform B.b(x); i = i + 1 };
            x
        };
        print(s);
        s + 1
    };
    print(r);
    0
}
"""

# A coroutine scope (C's arm works after its resume) INSIDE a direct scope's
# body; C's body performs against the direct scope, so A's case runs on
# C's fiber.  Tail variant and abort variant (the abort must hop from the
# fiber to A's frame on the root stack).
_CORO_IN_DIRECT = """
effect A { a(n: int) -> int }
effect C { c(n: int) -> int }
fn body() -> int {
    let @mut x = 0;
    let @mut i = 0;
    while i < 6 {
        x = perform C.c(x);
        x = perform A.a(x);
        i = i + 1
    };
    x
}
fn main() -> int {
    let r = handle A with {
        a(n) -> if n < {LIMIT} { resume(n + 1) } else { 1000 + n }
    } in {
        let s = handle C with {
            c(n) -> { let v = resume(n + 10); v + 0 }
        } in { body() };
        print(s);
        s + 1
    };
    print(r);
    0
}
"""

# Two coroutine scopes deep, then the abort of the direct root scope: the
# signal hops C2's fiber -> C1's fiber -> root.
_DEEP_HOP = """
effect A { a(n: int) -> int }
effect C1 { c1(n: int) -> int }
effect C2 { c2(n: int) -> int }
fn inner() -> int {
    let @mut x = 0;
    let @mut i = 0;
    while i < 5 {
        x = perform C2.c2(x);
        x = perform C1.c1(x);
        x = perform A.a(x);
        i = i + 1
    };
    x
}
fn main() -> int {
    let r = handle A with {
        a(n) -> if n < 30 { resume(n + 1) } else { 5000 + n }
    } in {
        let s1 = handle C1 with {
            c1(n) -> { let v = resume(n + 2); v + 0 }
        } in {
            let s2 = handle C2 with {
                c2(n) -> { let w = resume(n + 3); w + 0 }
            } in { inner() };
            print(s2);
            s2 + 1
        };
        print(s1);
        s1 + 1
    };
    print(r);
    0
}
"""

_DIRECT_IN_CORO = """
effect Outer { o(n: int) -> int }
effect Inner { i(n: int) -> int }
fn main() -> int {
    let r = handle Outer with {
        o(n) -> { let v = resume(n * 3); v + 1 }
    } in {
        handle Inner with {
            i(n) -> resume(n + 1)
        } in {
            let a = perform Inner.i(1);
            let b = perform Outer.o(a);
            let c = perform Inner.i(b);
            a + b + c
        }
    };
    print(r);
    0
}
"""

_FAIL_IN_BODY = """
effect A { a(n: int) -> int }
fn main() -> int {
    let r = try {
        handle A with { a(n) -> resume(n + 1) } in {
            let v = perform A.a(1);
            print(v);
            raise("boom");
            v
        }
    } catch e { print(e); -1 };
    print(r);
    0
}
"""

# A failure raised by a case surfaces at the handle expression: the try
# INSIDE the body must not catch it, the try outside does (docs/io_runtime.md
# states this rule; the coroutine pump gives it because cases run on the
# owner stack, and the direct path must agree).
_FAIL_IN_ARM = """
effect A { a(n: int) -> int }
fn main() -> int {
    let r = try {
        handle A with { a(n) -> { raise("arm failed"); 0 } } in {
            try { perform A.a(1) } catch e { print("inner caught " + e); 5 }
        }
    } catch e { print("outer caught " + e); 7 };
    print(r);
    0
}
"""

# Same rule when the case runs on a nested coroutine's fiber: neither the
# try around the perform (on the fiber) nor the try around the coroutine
# handle (in A's body) sees it -- only the try around A's handle.  (C's
# arm works after its resume, so C stays a coroutine scope and A's case
# runs on C's fiber.)
_FAIL_IN_ARM_CROSS_FIBER = """
effect A { a(n: int) -> int }
effect C { c(n: int) -> int }
fn main() -> int {
    let r = try {
        handle A with { a(n) -> { raise("arm failed"); 0 } } in {
            try {
                handle C with {
                    c(n) -> { let v = resume(n); v + 0 }
                } in {
                    try { perform C.c(1) + perform A.a(1) } catch e { print("fiber caught " + e); -3 }
                }
            } catch e { print("body caught " + e); -4 }
        }
    } catch e { print("outer caught " + e); 7 };
    print(r);
    0
}
"""

_ARM_PERFORMS_OUTWARD = """
effect Log { log(n: int) -> int }
effect A { a(n: int) -> int }
fn main() -> int {
    let r = handle Log with {
        log(n) -> { print(n); resume(0) }
    } in {
        handle A with {
            a(n) -> { let z = perform Log.log(n); resume(n + 1) }
        } in {
            let @mut x = 0;
            let @mut i = 0;
            while i < 4 { x = perform A.a(x); i = i + 1 };
            x
        }
    };
    print(r);
    0
}
"""

# A case performing its OWN op routes outward (busy scope skipped) to an
# enclosing handler of the same effect.
_ARM_PERFORMS_OWN_EFFECT = """
effect A { a(n: int) -> int }
fn main() -> int {
    let r = handle A with {
        a(n) -> resume(n * 100)
    } in {
        handle A with {
            a(n) -> { let m = perform A.a(n); resume(m + 1) }
        } in { perform A.a(2) }
    };
    print(r);
    0
}
"""

_MANY_SCOPES = """
effect A { a(n: int) -> int }
fn one_scope(i: int) -> int {
    handle A with { a(n) -> if n < 0 { 0 } else { resume(n + 1) } } in { perform A.a(i) }
}
fn main() -> int {
    let @mut total = 0;
    let @mut i = 0;
    while i < 100000 { total = total + one_scope(i); i = i + 1 };
    print(total);
    0
}
"""

_STREAM = """
from std.stream import iota, sum, map, filter, fold;
fn main() -> int {
    print(sum(map(filter(iota(3000), fn(x: int) -> x % 3 == 0), fn(x: int) -> x * x)));
    print(fold(iota(50), 5, fn(x: int, acc: int) -> x + acc));
    0
}
"""


def _cases_by_name(src: str):
    funcs = mir_from_source(src)
    by_name = {f.name: f for f in funcs}
    return funcs, by_name, handler_case_fns(funcs)


def _only_case(src: str):
    funcs, by_name, cases = _cases_by_name(src)
    assert len(cases) == 1, cases
    return by_name[next(iter(cases))]


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------

def test_tail_only_case_is_direct():
    assert direct_case(_only_case(_COUNTER.replace("{N}", "10")))


def test_abort_only_case_is_direct():
    assert direct_case(_only_case(_ABORT_ONLY))


def test_tail_or_abort_case_is_direct():
    assert direct_case(_only_case(_MIXED_TAKE))


def test_post_resume_work_is_not_direct():
    assert not direct_case(_only_case(_NON_TAIL))


def test_continuation_captured_by_a_try_is_not_direct():
    assert not direct_case(_only_case(_TRY_CAPTURES_K))


def test_first_resume_general_second_tail_is_not_direct():
    src = """
effect Ask { ask() -> int }
fn main() -> int {
    handle Ask with { ask() -> { let first = resume(1); resume(2) } } in { perform Ask.ask() }
}
"""
    assert not direct_case(_only_case(src))


def test_stream_arms_classify_as_expected():
    funcs, by_name, cases = _cases_by_name(_STREAM)
    direct = program_direct_cases(funcs)
    for stem in ("iter", "map", "filter"):
        (name,) = [n for n in cases if f"std.stream.{stem}$" in n]
        assert name in direct, name
    (fold,) = [n for n in cases if "std.stream.fold$" in n]
    assert fold not in direct


def test_emitter_picks_the_entry_point_per_site():
    ir = llvm_from_source(_STREAM)
    assert ir.count("call i64 @mx_handle_direct(") == 3   # iter, map, filter
    assert ir.count("call i64 @mx_handle(") == 1          # fold
    ir = llvm_from_source(_TRY_CAPTURES_K)
    assert "@mx_handle_direct(" not in ir
    ir = llvm_from_source(_MIXED_TAKE)
    assert "@mx_handle_direct(" in ir and "@mx_handle(" not in ir


# ---------------------------------------------------------------------------
# Native differentials against the interpreter
# ---------------------------------------------------------------------------

@needs_clang
@pytest.mark.parametrize("src", [
    _COUNTER.replace("{N}", "10"), _ABORT_ONLY, _MIXED_TAKE, _NON_TAIL,
    _NESTED_DIRECT_ABORT,
    _CORO_IN_DIRECT.replace("{LIMIT}", "1000000"),
    _CORO_IN_DIRECT.replace("{LIMIT}", "30"),
    _DEEP_HOP, _DIRECT_IN_CORO, _FAIL_IN_BODY, _FAIL_IN_ARM,
    _FAIL_IN_ARM_CROSS_FIBER, _ARM_PERFORMS_OUTWARD, _ARM_PERFORMS_OWN_EFFECT,
    _MANY_SCOPES, _STREAM,
], ids=["counter", "abort_only", "mixed_take", "non_tail",
        "nested_direct_abort", "coro_in_direct_tail", "coro_in_direct_abort",
        "deep_hop", "direct_in_coro", "fail_in_body", "fail_in_arm",
        "fail_in_arm_cross_fiber", "arm_performs_outward",
        "arm_performs_own_effect", "many_scopes", "stream"])
def test_native_matches_interpreter(src, tmp_path):
    # (_TRY_CAPTURES_K is analysis-only: a resume inside a try body is a
    # resume outside its case function, which the backend demotes today —
    # a pre-existing limit, see the ALGEBRAIC EFFECTS invariants.)
    assert_native_matches_interp(src, tmp_path)


def test_expected_outputs_on_the_interpreter():
    # Pin the semantics the differentials compare against, so a change
    # that moved BOTH engines would still be noticed.
    assert interp_run(_ABORT_ONLY)[1] == "42\n"
    assert interp_run(_NESTED_DIRECT_ABORT)[1] == "106\n"
    assert interp_run(_FAIL_IN_ARM)[1] == "outer caught arm failed\n7\n"
    assert interp_run(_FAIL_IN_ARM_CROSS_FIBER)[1] == "outer caught arm failed\n7\n"
    assert interp_run(_ARM_PERFORMS_OWN_EFFECT)[1] == "201\n"
    out = interp_run(_CORO_IN_DIRECT.replace("{LIMIT}", "30"))[1]
    assert out.endswith("\n") and "1" in out


@needs_clang
def test_native_one_million_performs_flat_stack(tmp_path):
    # A perform against a direct scope is a call that returns: a million
    # of them in one body cost no stack and no heap per event.
    ir = llvm_from_source(_COUNTER.replace("{N}", "1000000"))
    assert "@mx_handle_direct(" in ir
    exit_code, stdout = compile_and_run(ir, "main", workdir=str(tmp_path))
    assert exit_code == 0
    assert stdout == "1000000\n"


@needs_asan
@pytest.mark.parametrize("src", [
    _CORO_IN_DIRECT.replace("{LIMIT}", "30"), _DEEP_HOP,
    _FAIL_IN_ARM_CROSS_FIBER, _MANY_SCOPES,
], ids=["coro_in_direct_abort", "deep_hop", "fail_in_arm_cross_fiber", "many_scopes"])
def test_native_asan_clean(src, tmp_path):
    # detect_leaks=0 per the leak-by-design contract for closure envs and
    # produced strings; exit 0 under ASan shows the abort hops and the
    # fiber teardown they trigger touch no freed memory.
    result, expected_out = interp_run(src)
    ir = llvm_from_source(src)
    exit_code, stdout = compile_and_run(
        ir, "main", workdir=str(tmp_path),
        clang_args=("-fsanitize=address",),
        run_env={"ASAN_OPTIONS": "detect_leaks=0"})
    assert stdout == expected_out
    assert exit_code == 0


@needs_clang
def test_general_resume_of_a_direct_continuation_is_fatal(tmp_path):
    # Defense in depth: the compiler never emits mx_resume for a direct
    # scope's case; if it did, the runtime must refuse loudly rather than
    # switch to a stack that was never parked.  Drive it from C.
    from metaxu.runtime.native.build import runtime_objects
    c = tmp_path / "drive.c"
    c.write_text(r'''
#include <stdint.h>
#include <stdio.h>
#include "metaxu_effects.h"
static const char *ops[] = {"tick"};
static const int64_t np[] = {1};
static int64_t handler(void *env, int64_t op, const int64_t *args, mx_k *k) {
    (void)env; (void)op; (void)args;
    return mx_resume(k, 1);   /* general resume: invariant violation */
}
static int64_t body(void *env) { (void)env; int64_t a = 5; return mx_perform("T", "tick", &a, 1); }
int main(void) {
    int64_t v = mx_handle_direct(body, NULL, handler, NULL, "T", ops, np, 1);
    printf("%lld\n", (long long)v);
    return 0;
}
''')
    inc = str((c.parent / "..").resolve())
    from pathlib import Path
    native_dir = Path(__file__).resolve().parents[2] / "runtime" / "native"
    objs = [str(o) for o in runtime_objects()]
    exe = tmp_path / "drive"
    subprocess.run(["clang", "-std=c11", "-I", str(native_dir), str(c), *objs,
                    "-pthread", "-o", str(exe)], check=True, capture_output=True, text=True)
    proc = subprocess.run([str(exe)], capture_output=True, text=True)
    assert proc.returncode != 0
    assert "general resume of a direct scope" in proc.stderr
