"""End-to-end tests for algebraic effects with real single-shot continuations.

Source -> parse -> desugar -> freeze -> infer -> HIR -> MIR -> interpreter.

These assert the semantics documented in docs (effects_and_handlers example):
- `perform Effect.op(args)` suspends the performing frame at the perform site
- the matching handler case runs with the op argument and the continuation
- `resume(v)` runs the rest of the suspended frame with v as the perform's
  value, exactly once (second resume raises)
- a handler that returns without resuming aborts the handle scope; the
  handler's value becomes the handle expression's value and the suspended
  code after the perform never runs
- handlers are deep: every perform in the body (including in called
  functions) is handled by the same installed handler
- handle scopes nest, innermost handler wins for its ops
"""
from __future__ import annotations

import pytest

from metaxu.compiler.pipeline import build_context_from_source
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import MirInterpreter, UNIT


def run_main(source: str):
    """Compile source down to MIR and execute main(); returns (result, prints)."""
    ctx = build_context_from_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    mir = lower_hir_to_mir(hir)
    interp = MirInterpreter()
    interp.load(mir)
    prints: list[str] = []
    interp.register_builtin("print", lambda *a: (prints.append(" ".join(str(x) for x in a)), UNIT)[1])
    return interp.call("main", []), prints


def test_resume_supplies_perform_value():
    """The value passed to resume() is the value of the perform expression."""
    result, _ = run_main("""
effect Ask {
    ask() -> int
}

fn main() -> int {
    handle Ask with {
        ask() -> resume(7)
    } in {
        perform Ask.ask() + 1
    }
}
""")
    assert result == 8


def test_deep_handler_across_function_calls():
    """A perform inside a called function is caught by the caller's handler,
    and resume threads the value back through the call."""
    result, _ = run_main("""
effect Ask {
    ask() -> int
}

fn helper() performs Ask -> int {
    let x = perform Ask.ask();
    x * 10
}

fn main() -> int {
    handle Ask with {
        ask() -> resume(4)
    } in {
        helper() + 2
    }
}
""")
    assert result == 42


def test_multiple_performs_same_handler():
    """Deep handlers: the handler stays installed for every perform in the body."""
    result, _ = run_main("""
effect Ask {
    ask() -> int
}

fn main() -> int {
    handle Ask with {
        ask() -> resume(5)
    } in {
        let a = perform Ask.ask();
        let b = perform Ask.ask();
        a + b
    }
}
""")
    assert result == 10


def test_abort_when_handler_does_not_resume():
    """A handler that returns without resuming aborts the handle scope: its
    value becomes the handle result and the code after the perform is skipped."""
    result, prints = run_main("""
effect Fail {
    fail() -> int
}

fn main() -> int {
    handle Fail with {
        fail() -> 99
    } in {
        let x = perform Fail.fail();
        print("unreachable");
        x + 1
    }
}
""")
    assert result == 99
    assert prints == []  # nothing after the perform ran


def test_handler_can_transform_resumed_result():
    """Code in the handler after resume() sees the body's final value: the
    handler's own return value is the handle result (deep-handler semantics)."""
    result, _ = run_main("""
effect Ask {
    ask() -> int
}

fn main() -> int {
    handle Ask with {
        ask() -> {
            let rest = resume(1);
            rest + 100
        }
    } in {
        perform Ask.ask() + 2
    }
}
""")
    # body result = 1 + 2 = 3; handler transforms it to 103
    assert result == 103


def test_nested_handles_two_effects():
    """Nested handle scopes: each effect's ops route to its own handler."""
    result, prints = run_main("""
effect State {
    get() -> int
}

effect Logger {
    log(message: string) -> Unit
}

fn body() performs State, Logger -> int {
    let v = perform State.get();
    perform Logger.log("got it");
    v + 1
}

fn main() -> int {
    handle State with {
        get() -> resume(41)
    } in {
        handle Logger with {
            log(message) -> {
                print(message);
                resume(())
            }
        } in {
            body()
        }
    }
}
""")
    assert result == 42
    assert prints == ["got it"]


def test_handler_receives_op_argument():
    """The op's argument arrives as the handler case's parameter."""
    result, prints = run_main("""
effect Logger {
    log(message: string) -> Unit
}

fn main() -> int {
    handle Logger with {
        log(message) -> {
            print(message);
            resume(())
        }
    } in {
        perform Logger.log("hello");
        perform Logger.log("world");
        0
    }
}
""")
    assert result == 0
    assert prints == ["hello", "world"]


def test_resume_returns_whole_body_value_across_function_call():
    """resume() returns the value of the WHOLE handle body, not just the rest
    of the innermost performing frame: helper() must return 1 (the resumed
    value), the body computes 1 + 2 = 3, so rest = 3 and the handler yields
    300 as the handle result. (Regression: this used to evaluate to 102 —
    resume returned only helper's remainder and the body's `+ 2` ran after
    the handler completed.)"""
    result, _ = run_main("""
effect Ask {
    ask() -> int
}

fn helper() performs Ask -> int {
    perform Ask.ask()
}

fn main() -> int {
    handle Ask with {
        ask() -> {
            let rest = resume(1);
            rest * 100
        }
    } in {
        helper() + 2
    }
}
""")
    assert result == 300


def test_nested_handles_post_resume_with_called_function():
    """Nested handle scopes with post-resume handler code and the performs in
    a called function: each resume() sees the completion value of its OWN
    delimited body, with the inner handle's transformed result flowing into
    the outer one."""
    result, _ = run_main("""
effect State {
    get() -> int
}

effect Logger {
    log(message: string) -> Unit
}

fn body() performs State, Logger -> int {
    let v = perform State.get();
    perform Logger.log("hi");
    v + 1
}

fn main() -> int {
    handle State with {
        get() -> {
            let rest = resume(10);
            rest * 2
        }
    } in {
        handle Logger with {
            log(message) -> {
                let r = resume(());
                r + 1
            }
        } in {
            body() + 100
        }
    }
}
""")
    # body(): v = 10 -> returns 11; inner handle body = 11 + 100 = 111;
    # Logger handler: r = 111 -> inner handle result 112;
    # State handler: rest = 112 -> outer handle result 224.
    assert result == 224


def test_abort_inside_called_function_skips_caller_remainder():
    """A non-resuming handler aborts the whole delimited body: neither the
    rest of the called function nor the rest of the handle body runs."""
    result, prints = run_main("""
effect Fail {
    fail() -> int
}

fn helper() performs Fail -> int {
    let x = perform Fail.fail();
    print("unreachable-helper");
    x
}

fn main() -> int {
    handle Fail with {
        fail() -> 7
    } in {
        let y = helper();
        print("unreachable-main");
        y
    }
}
""")
    assert result == 7
    assert prints == []


def test_single_shot_double_resume_raises():
    """Resuming the same continuation twice violates single-shot semantics."""
    with pytest.raises(Exception, match="[Ss]ingle-shot|already consumed"):
        run_main("""
effect Ask {
    ask() -> int
}

fn main() -> int {
    handle Ask with {
        ask() -> resume(1) + resume(2)
    } in {
        perform Ask.ask()
    }
}
""")


def test_handler_perform_routes_to_outer_handler():
    """A handler case performing its own effect evaluates OUTSIDE its own
    delimitation: the perform routes to the next enclosing handler of that
    effect instead of deadlocking on the handler's own frame."""
    result, _ = run_main("""
effect Ask {
    ask() -> int
}

fn main() -> int {
    handle Ask with {
        ask() -> resume(100)
    } in {
        handle Ask with {
            ask() -> {
                let outer = perform Ask.ask();
                resume(outer + 1)
            }
        } in {
            perform Ask.ask()
        }
    }
}
""")
    # inner body performs -> inner handler performs -> OUTER handler resumes
    # 100 -> inner handler resumes 101 -> inner body value 101 -> both handles
    # complete with 101.
    assert result == 101


def test_multiple_performs_still_reach_inner_handler_after_resume():
    """Deep-handler re-arming: after resume(), later performs in the body must
    still reach the SAME (inner) handler, not leak to an outer one."""
    result, _ = run_main("""
effect Ask {
    ask() -> int
}

fn main() -> int {
    handle Ask with {
        ask() -> resume(1000)
    } in {
        handle Ask with {
            ask() -> resume(1)
        } in {
            perform Ask.ask() + perform Ask.ask()
        }
    }
}
""")
    assert result == 2  # both performs hit the inner handler
