"""MIR inlining: flatten a function's calls to other MIR functions into it.

Device kernels are compiled from ONE MIR function by the MSL emitter
(`emit_msl.py`), and a kernel written over the library (`std/tile.mx`'s
`exp`, `row_max`, `sub_rows`, ...) is a chain of ordinary calls.  Rather
than teach the emitter calls, frames and a device-side ABI, the kernel is
flattened first: every call to a user function is replaced by the callee's
blocks, renamed and renumbered, with the arguments bound to the renamed
parameters by copies and every `ret` turned into a copy into the call's
destination plus a jump to the continuation.  The result is one ordinary
MIR function the interpreter runs exactly like the original (that is the
test: inlined == not inlined, on the reference engine) and the emitter
translates as before.

What is inlined: calls whose callee is a MIR function of the module.
Builtins (`Tile.*`, `print`, `exp`, ...), closure calls, performs and
effect scopes are not calls to MIR functions and pass through untouched;
a lambda reached through `make_closure` is left as the closure it is (the
higher-order tile ops call it per element, and the emitter inlines THAT
body at the op).  Recursion and callees with write-back (`@mut`)
parameters are refused loudly: the first has no finite expansion and the
second would need the caller's slots rebound, which value-passing copies
do not express.  Rebinding a plain parameter is callee-local on both
engines, so it inlines as a rebinding of the parameter copy.

The LLVM backend uses the module form, `inline_into_module`, to inline
shape-generic tile helpers into their callers before kind inference, with
every inlined `make_closure` getting a private copy of its lambda
(`L$iN`), and `inline_calls` to flatten a narrow lambda's own module calls
(codegen_llvm.py, "narrow lambdas").  Inlining suffixes are unique across
passes: each pass starts above the largest `$iN` already in the module.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from .mir import MirBlock, MirFunc

__all__ = ["InlineError", "inline_calls", "inline_into_module",
           "rename_locals"]

_CAPTURE_RHS = ("make_closure", "handle_scope", "try_scope")


class InlineError(Exception):
    """The call graph cannot be flattened; the message says why."""


def inline_calls(root: MirFunc, funcs: Dict[str, MirFunc],
                 should_inline: Optional[Callable[[str], bool]] = None,
                 max_depth: int = 32) -> MirFunc:
    """A copy of ``root`` with every call to a module function inlined
    (transitively).  ``should_inline(callee)`` can narrow the set; by
    default every callee present in ``funcs`` is inlined."""
    ctx = _Inliner(funcs, should_inline, max_depth)
    ctx.seq = _seed_seq([root, *funcs.values()])
    blocks = ctx.expand(root, stack=(root.name,))
    return _with_blocks(root, blocks)


def inline_into_module(funcs: List[MirFunc], inline_names: Set[str],
                       max_depth: int = 32) -> List[MirFunc]:
    """Inline every call to a function in ``inline_names`` into every
    other function of the module, CLONING the lambdas an inlined body
    creates: each inlined copy of a `make_closure` gets its own lambda
    function (``L$iN``), so a shape- or width-generic helper such as
    `std.tile.exp` leaves one private lambda per call site behind.  The
    inlined functions themselves are kept only while something still
    mentions them.  Raises InlineError when a call cannot be inlined (the
    caller decides which functions are safe to list)."""
    by_name = {f.name: f for f in funcs}
    ctx = _Inliner(by_name, lambda n: n in inline_names, max_depth,
                   clone_lambdas=True)
    ctx.seq = _seed_seq(funcs)
    out: List[MirFunc] = []
    for f in funcs:
        if f.name in inline_names:
            out.append(f)
            continue
        blocks = ctx.expand(f, stack=(f.name,))
        out.append(_with_blocks(f, blocks))
    out.extend(ctx.new_funcs)
    # Drop the inlined functions (and their original lambdas) nothing
    # mentions any more; a remaining mention (a closure reference, a call
    # from a function that was itself not expanded) keeps them.
    droppable = set(inline_names) | {
        ln for n in inline_names for ln in _owned_lambda_names(by_name.get(n))}
    while True:
        keep = []
        for f in out:
            if f.name in droppable and not any(
                    _mentions(op, f.name)
                    for g in out if g is not f
                    for b in g.blocks for op in [*b.ops, b.term]):
                continue
            keep.append(f)
        if len(keep) == len(out):
            return keep
        out = keep


def _with_blocks(f: MirFunc, blocks: List[MirBlock]) -> MirFunc:
    return MirFunc(name=f.name, ty_sig=f.ty_sig, blocks=blocks,
                   suspending=f.suspending, globals_decl=f.globals_decl,
                   mut_params=f.mut_params, origin_name=f.origin_name,
                   location=f.location)


def _owned_lambda_names(f: Optional[MirFunc]) -> List[str]:
    if f is None:
        return []
    return [op[2][1] for b in f.blocks for op in b.ops
            if op[0] == "let" and len(op) == 4 and op[2][0] == "make_closure"]


def _mentions(obj, name: str) -> bool:
    if isinstance(obj, str):
        return obj == name
    if isinstance(obj, (tuple, list)):
        return any(_mentions(x, name) for x in obj)
    return False


class _Inliner:
    def __init__(self, funcs: Dict[str, MirFunc],
                 should_inline: Optional[Callable[[str], bool]],
                 max_depth: int, clone_lambdas: bool = False) -> None:
        self.funcs = funcs
        self.should_inline = should_inline
        self.max_depth = max_depth
        self.seq = 0
        self.clone_lambdas = clone_lambdas
        self.new_funcs: List[MirFunc] = []

    # -- the call test ---------------------------------------------------------

    def _target(self, op: tuple) -> Optional[MirFunc]:
        if op[0] != "let" or len(op) != 4:
            return None
        rhs = op[2]
        if not (isinstance(rhs, tuple) and rhs and rhs[0] == "call"
                and len(rhs) == 2):
            return None
        callee = rhs[1]
        if not isinstance(callee, str) or callee not in self.funcs:
            return None
        if self.should_inline is not None and not self.should_inline(callee):
            return None
        return self.funcs[callee]

    # -- expansion -------------------------------------------------------------

    def expand(self, f: MirFunc, stack: Tuple[str, ...]) -> List[MirBlock]:
        """``f``'s blocks with every inlinable call expanded; block indices
        are those of the returned list (callers must offset them)."""
        if len(stack) > self.max_depth:
            raise InlineError(
                f"inlining deeper than {self.max_depth} calls "
                f"({' -> '.join(stack)})")
        out: List[MirBlock] = []
        # Caller terminators are written with ('caller', j) targets and fixed
        # up once every caller block's new index is known.
        remap: Dict[int, int] = {}
        for j, b in enumerate(f.blocks):
            remap[j] = len(out)
            cur_ops: List[tuple] = []
            for op in b.ops:
                callee = self._target(op)
                if callee is None:
                    cur_ops.append(op)
                    continue
                _, dst, _rhs, args = op
                if callee.name in stack:
                    raise InlineError(
                        f"recursive call to {callee.name!r} cannot be "
                        f"inlined ({' -> '.join(stack + (callee.name,))})")
                if callee.mut_params:
                    raise InlineError(
                        f"call to {callee.name!r} cannot be inlined: it has "
                        f"write-back parameters {tuple(callee.mut_params)}")
                params = callee.param_names()
                if len(params) != len(args):
                    raise InlineError(
                        f"call to {callee.name!r} passes {len(args)} "
                        f"arguments for {len(params)} parameters")
                self.seq += 1
                sfx = f"$i{self.seq}"
                # The callee, itself expanded, renamed and renumbered.
                inner = self.expand(callee, stack + (callee.name,))
                defs = _defs(inner) | set(params)
                mapper = lambda n, _s=sfx: n + _s
                offset = len(out) + 1            # after the head block
                cont = offset + len(inner)       # the continuation block
                head_ops = cur_ops + [
                    ("let", f"{p}{sfx}", ("copy",), (a,))
                    for p, a in zip(params, args)]
                out.append(MirBlock(ops=head_ops, term=("br", offset)))
                for ib in inner:
                    ops = [_rename_op(o, defs, mapper) for o in ib.ops
                           if o[0] != "params"]
                    if self.clone_lambdas:
                        ops = [self._clone_lambda(o, sfx) for o in ops]
                    term = ib.term
                    if term[0] == "ret":
                        if len(term) >= 2:
                            ops.append(("let", dst, ("copy",),
                                        (_rn(term[1], defs, mapper),)))
                        else:
                            ops.append(("let", dst, ("const_ty", "Unit"), ()))
                        term = ("br", cont)
                    elif term[0] == "br":
                        term = ("br", term[1] + offset)
                    elif term[0] == "br_if":
                        term = ("br_if", _rn(term[1], defs, mapper),
                                term[2] + offset, term[3] + offset)
                    out.append(MirBlock(ops=ops, term=term))
                cur_ops = []
            term = b.term
            if term[0] == "br":
                term = ("br", ("caller", term[1]))
            elif term[0] == "br_if":
                term = ("br_if", term[1], ("caller", term[2]),
                        ("caller", term[3]))
            out.append(MirBlock(ops=cur_ops, term=term))
        # Fix up the caller's own jumps.
        for blk in out:
            t = blk.term
            if t[0] == "br" and isinstance(t[1], tuple):
                blk.term = ("br", remap[t[1][1]])
            elif t[0] == "br_if" and isinstance(t[2], tuple):
                blk.term = ("br_if", t[1], remap[t[2][1]], remap[t[3][1]])
        return out

    def _clone_lambda(self, op: tuple, sfx: str) -> tuple:
        """An inlined `make_closure` gets a private copy of its lambda,
        named with the inlining's suffix (nested lambdas inside the copy
        stay shared: they are the copy's own make_closure targets)."""
        if op[0] != "let" or len(op) != 4 or op[2][0] != "make_closure":
            return op
        lname = op[2][1]
        lam = self.funcs.get(lname)
        if lam is None:
            return op
        new_name = f"{lname}{sfx}"
        copy = _with_blocks(
            lam, [MirBlock(ops=list(b.ops), term=b.term) for b in lam.blocks])
        copy.name = new_name
        copy.origin_name = lam.origin_name or lam.name
        self.new_funcs.append(copy)
        return ("let", op[1], ("make_closure", new_name, *op[2][2:]), op[3])


_INLINE_SFX = re.compile(r"\$i(\d+)")


def _seed_seq(funcs: Iterable[MirFunc]) -> int:
    """The largest `$iN` suffix any name in `funcs` already carries, so a
    later inlining pass (a second module round, or the narrow-lambda
    flattening after it) never reuses a suffix an earlier one minted."""
    best = 0

    def walk(x) -> None:
        nonlocal best
        if isinstance(x, str):
            for m in _INLINE_SFX.finditer(x):
                best = max(best, int(m.group(1)))
        elif isinstance(x, (tuple, list)):
            for y in x:
                walk(y)

    for f in funcs:
        walk(f.name)
        for b in f.blocks:
            walk(b.ops)
            walk(b.term)
    return best


def rename_locals(f: MirFunc, mapper: Callable[[str], str]) -> MirFunc:
    """A copy of ``f`` with every local name (params, let and perform
    destinations, and their uses) passed through ``mapper``.  Callee names
    and the lambda-side names of captures are untouched, like inlining's
    own renaming.  The MSL emitter uses it to turn inlined names such as
    ``n$i1`` into C identifiers."""
    defs = _defs(f.blocks)
    blocks = []
    for b in f.blocks:
        ops = []
        for op in b.ops:
            if op[0] == "params":
                ops.append(("params", tuple(mapper(p) for p in op[1])))
            else:
                ops.append(_rename_op(op, defs, mapper))
        blocks.append(MirBlock(ops=ops, term=_rn(b.term, defs, mapper)))
    return MirFunc(name=f.name, ty_sig=f.ty_sig, blocks=blocks,
                   suspending=f.suspending, globals_decl=f.globals_decl,
                   mut_params=f.mut_params, origin_name=f.origin_name,
                   location=f.location)


# -- renaming ------------------------------------------------------------------

def _defs(blocks: List[MirBlock]) -> Set[str]:
    """Every name a function binds: params, let destinations, perform
    results."""
    out: Set[str] = set()
    for b in blocks:
        for op in b.ops:
            if op[0] == "params":
                out.update(op[1])
            elif op[0] in ("let", "perform") and len(op) > 1 \
                    and isinstance(op[1], str):
                out.add(op[1])
    return out


def _rn(x, defs: Set[str], mapper: Callable[[str], str]):
    if isinstance(x, str):
        return mapper(x) if x in defs else x
    if isinstance(x, tuple):
        return tuple(_rn(y, defs, mapper) for y in x)
    if isinstance(x, list):
        return [_rn(y, defs, mapper) for y in x]
    return x


def _rename_op(op: tuple, defs: Set[str], mapper: Callable[[str], str]) -> tuple:
    """Rename a callee op's local names.  The callee NAME of a call and
    the lambda-side names of captures stay: `('make_closure', lname,
    params)` with captures `((cname, cval), ...)` renames only `cval`."""
    if op[0] == "let" and len(op) == 4:
        _, dst, rhs, args = op
        rk = rhs[0] if isinstance(rhs, tuple) and rhs else None
        if rk == "call":
            # ('call', callee): the callee is a function name, never local
            new_rhs = rhs
            new_args = tuple(_rn(a, defs, mapper) for a in args)
        elif rk in _CAPTURE_RHS:
            new_rhs = rhs            # lambda / body / case names
            new_args = tuple(
                (pair[0], _rn(pair[1], defs, mapper))
                if isinstance(pair, tuple) and len(pair) == 2 else
                _rn(pair, defs, mapper)
                for pair in args)
        else:
            new_rhs = _rn(rhs, defs, mapper)
            new_args = tuple(_rn(a, defs, mapper) for a in args)
        return ("let", _rn(dst, defs, mapper), new_rhs, new_args)
    if op[0] == "perform":
        # ('perform', dst, effect, op, args, site, dst)
        return tuple(
            _rn(x, defs, mapper) if i not in (2, 3) else x
            for i, x in enumerate(op))
    return _rn(op, defs, mapper)
