"""Frame layouts for selective CPS lowering of suspending functions.

Roadmap item 6, "Defunctionalize: generate Frame struct layouts, enum State".

A *suspending* function is one whose ``MirFunc.suspending`` flag is set, or
one that contains an effect op (``perform``, or a ``resume`` /
``handle_scope`` rhs).  Non-suspending functions stay direct and get no
frame.

Frame model (every slot is one 8-byte i64 word):

  ``[0]``   ``state``  -- the State enum discriminant: 0 = start at function
            entry, k = resume after the k-th perform (1-based, in block/op
            order).  ``%run_<fn>`` dispatches on it with ``br_table``.
  ``[8]``   ``result`` -- the value delivered by ``resume``; the
            ``%resume_<fn>_<k>`` shims store here before re-entering
            ``%run_<fn>``, which copies it into the perform's destination.
  ``[16+]`` one slot per function parameter (whoever spawns the frame stores
            the arguments here before running state 0), then
  one slot per live-across-suspension variable that is not already a param.

A variable is live across a suspension point when it has a definition that
can reach the point and a use at-or-after the point's continuation (the
resume block of a perform; the next op after a call to another suspending
function).  Both kinds of point contribute frame slots; only performs get a
state number, because at the CLIF level the park sites are the perform ops.

Liveness here is reachability-based (defined-before / used-after over the
CFG) with no kill analysis: it may keep a dead variable in the frame, never
the reverse, so it is safe for layout purposes.

Honest scope note: general effect dispatch -- finding the matching handler,
single-shot continuation bookkeeping, handler aborts -- stays in the MIR
interpreter (``mir_interp.py``).  These layouts back only the mechanical
park/wake shape that ``codegen_clif.py`` emits (state machine + enqueue /
sched_read + resume shims), per the roadmap's CLIF-level CPS scope.

Public API:
  ``compute_frame_layouts(mir_funcs) -> dict[str, dict]`` (suspending only)
  ``is_suspending(mir_func) -> bool``
  ``var_offset(layout, name) -> int``
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set, Tuple

from .mir import MirFunc

WORD = 8
STATE_OFFSET = 0
RESULT_OFFSET = 8
FIRST_VAR_OFFSET = 16

_Site = Tuple[int, int]  # (block index, op index); op index -1 = block start


def is_suspending(f: MirFunc) -> bool:
    """Suspending = flagged by lowering, or contains an effect op."""
    if f.suspending:
        return True
    for b in f.blocks:
        for op in b.ops:
            if op[0] == "perform":
                return True
            if op[0] == "let" and len(op) == 4 and op[2][0] in ("resume", "handle_scope"):
                return True
    return False


def var_offset(layout: Dict[str, Any], name: str) -> int:
    """Frame offset of a param or live-across variable."""
    if name in layout["params"]:
        return layout["params"][name]
    return layout["vars"][name]


# ---------------------------------------------------------------------------
# CFG helpers
# ---------------------------------------------------------------------------

def _successors(term: tuple, n_blocks: int) -> List[int]:
    if term[0] == "br":
        tgts = [term[1]]
    elif term[0] == "br_if":
        tgts = [term[2], term[3]]
    else:  # ret / unreachable
        tgts = []
    return [t for t in tgts if isinstance(t, int) and 0 <= t < n_blocks]


def _reach(f: MirFunc) -> List[Set[int]]:
    """reach[b] = blocks reachable from b by following terminators (b itself
    only if it sits on a cycle)."""
    n = len(f.blocks)
    succ = [_successors(b.term, n) for b in f.blocks]
    reach: List[Set[int]] = []
    for s in range(n):
        seen: Set[int] = set()
        stack = list(succ[s])
        while stack:
            b = stack.pop()
            if b in seen:
                continue
            seen.add(b)
            stack.extend(succ[b])
        reach.append(seen)
    return reach


# ---------------------------------------------------------------------------
# Def/use collection
# ---------------------------------------------------------------------------

def _flat_names(args: Sequence[Any]) -> List[str]:
    """Flatten op args that may contain (name, name) pairs (captures,
    struct fields) into the referenced variable names."""
    out: List[str] = []
    for a in args or ():
        if isinstance(a, str):
            out.append(a)
        elif isinstance(a, (tuple, list)):
            out.extend(x for x in a if isinstance(x, str))
    return out


def _defs_uses(f: MirFunc) -> Tuple[Tuple[str, ...], Dict[str, List[_Site]], Dict[str, List[_Site]]]:
    defs: Dict[str, List[_Site]] = {}
    uses: Dict[str, List[_Site]] = {}
    params: Tuple[str, ...] = ()

    def add_def(name: str, site: _Site) -> None:
        defs.setdefault(name, []).append(site)

    def add_use(name: str, site: _Site) -> None:
        uses.setdefault(name, []).append(site)

    n = len(f.blocks)
    for bi, b in enumerate(f.blocks):
        for oi, op in enumerate(b.ops):
            kind = op[0]
            if kind == "params":
                if bi == 0:
                    params = tuple(op[1])
                continue
            if kind == "perform":
                # ("perform", dst, effect, op, args, resume_bb, dst)
                _, dst, _eff, _opn, pargs, resume_bb, _dst2 = op
                for a in _flat_names(pargs):
                    add_use(a, (bi, oi))
                # The dst value only exists once the frame is resumed: treat
                # its definition site as the start of the resume block so it
                # is not "live across" its own perform.
                rb = resume_bb if isinstance(resume_bb, int) and 0 <= resume_bb < n else bi
                add_def(dst, (rb, -1))
                continue
            if kind in ("drop", "match_fail"):
                continue  # no-ops for liveness (drop is a comment in CLIF)
            if kind == "let" and len(op) == 4:
                _, dst, rhs, args = op
                for a in _flat_names(args):
                    add_use(a, (bi, oi))
                add_def(dst, (bi, oi))
        last = len(b.ops)
        t = b.term
        if t[0] in ("br_if", "ret"):
            if isinstance(t[1], str):
                add_use(t[1], (bi, last))
    for p in reversed(params):
        defs.setdefault(p, []).insert(0, (0, -1))
    return params, defs, uses


# ---------------------------------------------------------------------------
# Liveness across a suspension point
# ---------------------------------------------------------------------------

def _def_reaches(site: _Site, pb: int, pi: int, reach: List[Set[int]]) -> bool:
    db, di = site
    if db == pb and di < pi:
        return True
    if pb in reach[db]:  # includes db == pb via a cycle
        return True
    return False


def _use_after(site: _Site, cont: _Site, reach: List[Set[int]]) -> bool:
    ub, ui = site
    cb, ci = cont
    if ub == cb and ui >= ci:
        return True
    if ub in reach[cb]:  # includes ub == cb via a cycle
        return True
    return False


def _live_across(pb: int, pi: int, cont: _Site,
                 defs: Dict[str, List[_Site]], uses: Dict[str, List[_Site]],
                 reach: List[Set[int]]) -> Set[str]:
    live: Set[str] = set()
    for var, dsites in defs.items():
        if not any(_def_reaches(d, pb, pi, reach) for d in dsites):
            continue
        if any(_use_after(u, cont, reach) for u in uses.get(var, ())):
            live.add(var)
    return live


# ---------------------------------------------------------------------------
# Layout computation
# ---------------------------------------------------------------------------

def _layout_one(f: MirFunc, suspending_names: Set[str]) -> Dict[str, Any]:
    params, defs, uses = _defs_uses(f)
    reach = _reach(f)
    n = len(f.blocks)

    points: List[Dict[str, Any]] = []
    sus_calls: List[Dict[str, Any]] = []
    for bi, b in enumerate(f.blocks):
        for oi, op in enumerate(b.ops):
            if op[0] == "perform":
                _, dst, eff, opn, pargs, resume_bb, _dst2 = op
                rb = resume_bb if isinstance(resume_bb, int) and 0 <= resume_bb < n else bi
                live = _live_across(bi, oi, (rb, 0), defs, uses, reach)
                points.append({
                    "state": len(points) + 1,
                    "block": bi,
                    "op_index": oi,
                    "effect": eff,
                    "op": opn,
                    "args": list(pargs or ()),
                    "dst": dst,
                    "resume_block": rb,
                    "live": sorted(live),
                })
            elif (op[0] == "let" and len(op) == 4 and op[2][0] == "call"
                  and op[2][1] in suspending_names):
                live = _live_across(bi, oi, (bi, oi + 1), defs, uses, reach)
                sus_calls.append({
                    "block": bi,
                    "op_index": oi,
                    "callee": op[2][1],
                    "live": sorted(live),
                })

    live_union: Set[str] = set()
    for p in points:
        live_union.update(p["live"])
    for c in sus_calls:
        live_union.update(c["live"])

    param_offs: Dict[str, int] = {}
    off = FIRST_VAR_OFFSET
    for p in params:
        param_offs[p] = off
        off += WORD
    var_offs: Dict[str, int] = {}
    for v in sorted(live_union):
        if v in param_offs:
            continue
        var_offs[v] = off
        off += WORD

    return {
        "name": f.name,
        "size": off,
        "state_offset": STATE_OFFSET,
        "result_offset": RESULT_OFFSET,
        "params": param_offs,
        "vars": var_offs,
        "live_across": sorted(live_union),
        "suspend_points": points,
        "suspending_calls": sus_calls,
    }


def compute_frame_layouts(mir_funcs: Sequence[MirFunc]) -> Dict[str, Dict[str, Any]]:
    """Frame layouts for every suspending function in the module.

    Returns ``{func_name: layout}`` where each layout is the dict documented
    in :func:`_layout_one` / the module docstring: word offsets for the
    ``state`` discriminant, the ``result`` slot, every param, and every
    live-across-suspension variable, plus the ordered suspend points (state
    numbers) and the calls to other suspending functions.
    """
    funcs = list(mir_funcs)
    suspending_names = {f.name for f in funcs if is_suspending(f)}
    return {f.name: _layout_one(f, suspending_names)
            for f in funcs if f.name in suspending_names}
