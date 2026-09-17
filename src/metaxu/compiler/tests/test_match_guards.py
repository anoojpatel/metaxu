"""Match guards: `pattern if cond => body`, desugared at parse time.

A failing guard falls through to the next arm; the scrutinee is
evaluated once; guarded arms do not count toward exhaustiveness (the
remaining arms must still cover); handler arms take no guards.
"""
from __future__ import annotations

import pytest

from metaxu.errors import CompileError
from metaxu.compiler.mir_interp import InterpError
from metaxu.compiler.tests.test_codegen_llvm import (
    assert_native_matches_interp, interp_run, needs_clang)

CLASSIFY = """
fn classify(n: int) -> string {
    match n {
        v if v > 100 => "big"
        v if v > 0 => "small"
        v if v <= 0 => "non-positive"
        _ => "nothing"
    }
}

fn main() -> int {
    print(classify(500));
    print(classify(7));
    print(classify(0));
    print(classify(0 - 3));
    0
}
"""


def test_guards_fall_through_in_order():
    _res, out = interp_run(CLASSIFY)
    assert out.splitlines() == ["big", "small", "non-positive", "non-positive"]


@needs_clang
def test_guards_native_matches_interp(tmp_path):
    assert_native_matches_interp(CLASSIFY, tmp_path)


def test_effectful_scrutinee_is_evaluated_once():
    _res, out = interp_run("""
fn tick() -> int {
    print("evaluated");
    5
}

fn main() -> int {
    let r = match tick() {
        v if v > 10 => "high"
        v if v > 1 => "mid"
        _ => "low"
    };
    print(r);
    0
}
""")
    assert out.splitlines() == ["evaluated", "mid"]


def test_binder_is_in_scope_in_guard_and_body():
    _res, out = interp_run("""
enum Shape {
    Dot,
    Circle(int)
}

fn describe(s: Shape) -> string {
    match s {
        Shape::Circle(r) if r > 10 => "large circle " + r.to_string()
        Shape::Circle(r) => "circle " + r.to_string()
        Shape::Dot => "dot"
    }
}

fn main() -> int {
    print(describe(Shape::Circle(12)));
    print(describe(Shape::Circle(3)));
    print(describe(Shape::Dot));
    0
}
""")
    assert out.splitlines() == ["large circle 12", "circle 3", "dot"]


def test_uncovered_fallthrough_is_a_loud_runtime_error():
    # The checker does not look inside arm bodies yet, so a match whose
    # only arms are guarded is accepted; a value that fails every guard
    # reaches an empty rest-match and must fail loudly, never silently.
    with pytest.raises(InterpError) as ei:
        interp_run("""
fn main() -> int {
    let n = 3;
    let r = match n {
        v if v > 10 => "high"
    };
    print(r);
    0
}
""")
    assert "match failure" in str(ei.value)


def test_uncovered_fallthrough_is_catchable():
    _res, out = interp_run("""
fn pick(n: int) -> string {
    match n {
        v if v > 10 => "high"
    }
}

fn main() -> int {
    let r = try {
        pick(3)
    } catch e {
        "caught"
    };
    print(r);
    0
}
""")
    assert out.splitlines() == ["caught"]


def test_handler_arms_do_not_take_guards():
    with pytest.raises(CompileError) as ei:
        interp_run("""
effect Ask {
    ask() -> int
}

fn main() -> int {
    let n = handle Ask with {
        ask() if true -> resume(1)
    } in {
        perform Ask.ask()
    };
    print(n);
    0
}
""")
    assert "handler arms do not take guards" in str(ei.value)
