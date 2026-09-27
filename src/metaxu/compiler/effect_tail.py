"""Tail-resume detection over MIR handler-case functions.

A handler case whose ``resume(v)`` is in TAIL position — the resume's
value IS the case's return value, with nothing observable after it — does
not need the recursive handler pump: the case can hand (k, v) back to its
scope's event loop, which performs the switch into the body from its own
constant frame.  That turns the per-element (case frame + resume frame)
stack growth of stream-shaped handlers (std.stream's iter/map/filter/skip,
std.state's get/set, std.log's arms) into O(1), on both engines:

  * the LLVM backend emits ``mx_resume_tail`` instead of ``mx_resume`` for
    these sites (metaxu_effects.c trampolines them in its pump loop);
  * the MIR interpreter raises ``_TailResume`` back to ``_pump_scope``
    instead of recursing (mir_interp.py).

BOTH engines consult THIS analysis, so a resume is trampolined natively
iff it is trampolined in the interpreter — the engines cannot disagree
about which shape a case has.

THE RULE (strict — when in doubt, the general recursive path is kept,
which is always correct):

  a ``('let', dst, ('resume',), (k, v))`` op is a tail resume iff
    1. it lives in a HANDLER-CASE function (a function named as a case
       target by some ``handle_scope`` op — the only functions whose
       resume the native backend accepts anyway),
    2. ``k`` is the case's own trailing ``__k`` parameter (never a
       continuation that arrived any other way),
    3. from the op to the function's ``ret``, the value flows UNTOUCHED:
       every remaining op on the path is a ``('copy',)`` of the current
       value name into a new name, every terminator is an unconditional
       ``br`` (no branching may remain — a branch would decide something
       after the resume), and the final ``ret`` returns the current name.

Anything else — ``f(x, resume(()))`` fold shapes, ``resume(()) + 1``,
a resume under a ``try_scope`` (its body is a separate subfunction, so
rule 1 excludes it), branches after the resume — stays on the general
recursive path, whose depth is bounded by handler-code nesting, not
element count (except genuine foldr shapes, where the O(n) pending work
is the semantics).

Results are keyed by ``id(op)`` of the ``let`` op tuple: MIR ops are
positional tuples built fresh per op by the lowering, alive as long as
the loaded MIR, so their identities are stable and unambiguous within a
process.
"""
from __future__ import annotations

from typing import Iterable, Set

from .mir import MirFunc

__all__ = ["handler_case_fns", "tail_resume_ids", "program_tail_resume_ids",
           "direct_case", "program_direct_cases"]


def handler_case_fns(funcs: Iterable[MirFunc]) -> Set[str]:
    """Names of every function that is a handler-case target of some
    ``handle_scope`` op anywhere in the program."""
    cases: Set[str] = set()
    for f in funcs:
        for b in f.blocks:
            for op in b.ops:
                if op[0] != "let":
                    continue
                rhs = op[2]
                if not (isinstance(rhs, tuple) and rhs
                        and rhs[0] == "handle_scope"):
                    continue
                # ("handle_scope", site, effect, cases) with cases =
                # ((op_name, params, case_fn_name), ...)
                for case in rhs[3]:
                    cases.add(case[2])
    return cases


def tail_resume_ids(func: MirFunc) -> frozenset:
    """``id(op)`` of every tail-position resume op in ``func``.

    ``func`` must already be known to be a handler-case function (rule 1);
    this checks rules 2 and 3 only.
    """
    params = func.param_names()
    if not params:
        return frozenset()
    kname = params[-1]  # the case's trailing __k continuation parameter
    ids = set()
    for bi, b in enumerate(func.blocks):
        for oi, op in enumerate(b.ops):
            if op[0] != "let":
                continue
            rhs = op[2]
            if not (isinstance(rhs, tuple) and rhs and rhs[0] == "resume"):
                continue
            args = op[3]
            if len(args) != 2 or args[0] != kname:
                continue
            if _returned_untouched(func, bi, oi, op[1]):
                ids.add(id(op))
    return frozenset(ids)


def program_tail_resume_ids(funcs: Iterable[MirFunc]) -> frozenset:
    """Union of ``tail_resume_ids`` over every handler-case function."""
    funcs = list(funcs)
    cases = handler_case_fns(funcs)
    ids: Set[int] = set()
    for f in funcs:
        if f.name in cases:
            ids |= tail_resume_ids(f)
    return frozenset(ids)


def direct_case(func: MirFunc) -> bool:
    """True iff every path through this handler case either TAIL-resumes
    its continuation exactly once or never resumes it, and the
    continuation is used for nothing else.

    Such a case never needs a real continuation: the native runtime can
    run the scope's body on the CURRENT stack and turn each perform into a
    plain call of the case (metaxu_effects.c, mx_handle_direct) -- a tail
    resume returns the value to the perform site, and a case that returns
    without resuming aborts the scope by unwinding to its handle.  The
    interpreter needs no counterpart: nothing observable changes, only
    where the frames live.

    THE RULE (strict, like tail_resume_ids): the case's trailing ``__k``
    parameter may appear ONLY as the continuation of a ``resume`` op that
    ``tail_resume_ids`` marks.  Any other occurrence anywhere in the
    function -- a non-tail resume (so a resume-then-continue arm, or
    ``let a = resume(1); resume(2)``), a capture into a nested try/handle
    body or a closure, a copy, a call argument, a terminator -- keeps the
    case on the coroutine path.  A case with no resume at all (a pure
    abort arm) is direct.
    """
    params = func.param_names()
    if not params:
        return False
    kname = params[-1]
    tail = tail_resume_ids(func)
    for b in func.blocks:
        for op in b.ops:
            if op[0] == "params":
                continue
            if (op[0] == "let" and isinstance(op[2], tuple) and op[2]
                    and op[2][0] == "resume"):
                args = op[3]
                if len(args) == 2 and args[0] == kname:
                    if id(op) not in tail:
                        return False
                    if _mentions(args[1], kname):
                        return False
                    continue
            if _mentions(op, kname):
                return False
        if _mentions(b.term, kname):
            return False
    return True


def program_direct_cases(funcs: Iterable[MirFunc]) -> frozenset:
    """Names of the handler-case functions ``direct_case`` accepts.  A
    handle site whose EVERY case is in this set runs on the current stack
    natively (the emitter checks the site, this reports the cases)."""
    funcs = list(funcs)
    cases = handler_case_fns(funcs)
    return frozenset(f.name for f in funcs
                     if f.name in cases and direct_case(f))


def _mentions(x, name: str) -> bool:
    """True iff the string ``name`` occurs anywhere inside the (nested)
    tuple/list ``x`` -- the conservative "any other use" test."""
    if isinstance(x, str):
        return x == name
    if isinstance(x, (tuple, list)):
        return any(_mentions(y, name) for y in x)
    return False


def _returned_untouched(func: MirFunc, bi: int, oi: int, name: str) -> bool:
    """True iff, starting just after op ``oi`` of block ``bi``, the value
    bound to ``name`` reaches the function's ``ret`` through nothing but
    ``copy`` ops and unconditional branches."""
    blocks = func.blocks
    visited = {bi}
    cur = name
    ops = blocks[bi].ops[oi + 1:]
    term = blocks[bi].term
    while True:
        for op in ops:
            if (op[0] == "let" and isinstance(op[2], tuple)
                    and op[2] == ("copy",) and len(op[3]) == 1
                    and op[3][0] == cur):
                cur = op[1]
            else:
                return False  # anything else after the resume: not tail
        if term[0] == "ret":
            return len(term) == 2 and term[1] == cur
        if term[0] == "br":
            nb = term[1]
            if nb in visited or nb >= len(blocks):
                return False
            visited.add(nb)
            ops = blocks[nb].ops
            term = blocks[nb].term
            continue
        return False  # br_if / unreachable / anything conditional: not tail
