"""Monomorphization pass: clone generic functions per concrete instantiation.

Design
------
This is an HIR->HIR pass (deliberately, rather than MIR->MIR): at HIR level
calls still carry their callee names, structured argument expressions, and —
via the ``HExpr.type_args`` field — explicit instantiation type arguments,
so instantiations can be resolved without re-running inference. Cloned
functions then flow through the ordinary HIR->MIR lowering, which makes the
specialized names (``identity$Int``) appear in MIR for free — the lever the
native codegen lane can pick up without any MIR-side surgery.

For every call whose callee is a generic function (declared with type
parameters) the pass resolves the instantiation:

  1. explicit type args on the call (``identity<Int>(x)``), else
  2. call-site-local inference from arguments of statically known type:
     literals, struct instantiations, enum variant constructions, calls to
     non-generic functions with declared primitive returns, resolved nested
     generic calls whose return type is a bare type parameter, parameters
     of the enclosing function whose declared type is a primitive, ``let``
     locals bound to an expression of statically known type, and — inside
     a specialized clone — parameters whose declared type is a substituted
     type parameter.

``let`` locals flow forward through a statement sequence (HIR blocks are
flat: a ``Let`` node's continuation is its following sibling), each block /
arm / loop body getting its own scope, so ``let a = 1; identity(a)``
resolves exactly like ``identity(1)``.  Names that are ever REBOUND — an
assignment target, a lambda parameter, a match/while-let pattern binder, a
handler-arm parameter — are excluded from the environment wholesale rather
than tracked per scope: the pass would otherwise have to prove the shadow
carries the same type, and a wrong answer here silently picks the wrong
clone.

A fully resolved call site is rewritten to a specialized clone named
``fn$Arg1$Arg2`` (created on demand, memoized, recursion-safe). Clone bodies
are processed under their substitution so nested generic calls specialize
transitively. Generic originals are erased from the output only when every
call site was rewritten and the function is not otherwise referenced by
name (e.g. used as a value); partially resolvable programs keep their
generic originals, so running the pass never changes observable behavior —
the interpreter dispatches dynamically either way. Correctness contract:
interpreting monomorphized MIR must produce results identical to
interpreting the unmonomorphized MIR (pinned by tests).

Module-qualified generic calls (``mod.f<Int>(x)``) resolve like plain ones:
the module system renames them to dotted callee names, which are collected
in the signature map, and the HIR call preserves their explicit type args.
Bracket-form instantiations (``Full[Int](x)``) are rewritten to the angle
form by desugaring before HIR exists, so they also specialize for free.

Who runs it
-----------
``pipeline.emit_llvm_from_source`` runs it by default: the LLVM backend's
value-kind cells are per-function and monomorphic, so one generic reached
at two types joins to ``conflict`` and demotes.  The interpreter front door
(``run_pipeline_from_source`` / ``run_pipeline_ctx``) keeps the flag OFF —
it is the semantics reference, and every native differential compares
specialized native code against unspecialized interpreted semantics.

Out of scope (groundwork, documented): substitution inside type
applications (``Vec[T]``), method/trait-dispatch callees (``__trait$m``),
and higher-order flow of generic functions as values — such call sites
simply stay generic.

Measured limit (example corpus, 17 compiling files).  Every one of the 18
"irreconcilable value kinds" placeholder reasons in that corpus comes from
ONE call site: the second instantiation of the trait method ``map`` in
``examples/06_vector_operations.mx`` (``v1.map(x -> x * x)`` at Float and
``v1.map(x -> x.to_string())`` at String).  This pass cannot reach it for
two independent reasons, and neither is a small gap:

  * the callee is ``__trait$map``, dispatched on the receiver's runtime
    type.  Rewriting such a site to a specific impl needs the receiver's
    TYPE, which HIR does not carry (``HExpr.ty`` is an unresolved variable
    at this point); guessing would silently pick a different target than
    the interpreter's dispatch;
  * even for a plain (non-trait) function of that shape, ``U`` in
    ``map<U>(self, f: fn(T) -> U) -> vector[U, N]`` occurs in NO bare
    parameter position — binding it means inferring the lambda argument's
    return type, i.e. re-running inference inside this pass.

Both are recorded here rather than attempted: a wrong clone choice is a
miscompile, which is strictly worse than the placeholder it would replace.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Sequence

import metaxu.metaxu_ast as fast

from .hir import HFun, HExpr
from .mutaxu_ast import _safe_type_display, _type_param_name

# Separator for specialized names. Matches the impl-mangling separator so
# every synthesized name family shares one reserved character.
MONO_SEP = "$"

_PRIMITIVE_NAMES = {
    "int": "Int", "Int": "Int",
    "str": "String", "string": "String", "String": "String",
    "bool": "Bool", "Bool": "Bool",
    "float": "Float", "Float": "Float",
}
_PRIMITIVE_SET = frozenset({"Int", "String", "Bool", "Float"})


def _canon(display: Any) -> str | None:
    if not isinstance(display, str) or not display:
        return None
    base = display.split("[", 1)[0].split("<", 1)[0].strip()
    return _PRIMITIVE_NAMES.get(base, base)


@dataclass(frozen=True, slots=True)
class FnSig:
    """Declared signature info for one function (from the parsed AST)."""
    name: str
    type_params: tuple[str, ...]
    param_names: tuple[str, ...]
    param_types: tuple[str | None, ...]
    return_type: str | None

    @property
    def is_generic(self) -> bool:
        return bool(self.type_params)


def collect_signatures(id_map: Dict[int, Any]) -> Dict[str, FnSig]:
    """Collect declared function signatures from the pipeline's id_map
    (frozen node_id -> original AST node)."""
    sigs: Dict[str, FnSig] = {}
    for orig in id_map.values():
        if not isinstance(orig, fast.FunctionDeclaration):
            continue
        name = str(getattr(orig, "name", "") or "")
        if not name or name in sigs:
            continue
        tparams = tuple(
            p for p in (_type_param_name(tp) for tp in getattr(orig, "type_params", None) or [])
            if isinstance(p, str)
        )
        pnames: List[str] = []
        ptypes: List[str | None] = []
        for p in getattr(orig, "params", None) or []:
            pnames.append(str(getattr(p, "name", "") or ""))
            ptypes.append(_safe_type_display(getattr(p, "type_annotation", None)))
        sigs[name] = FnSig(
            name=name,
            type_params=tparams,
            param_names=tuple(pnames),
            param_types=tuple(ptypes),
            return_type=_safe_type_display(getattr(orig, "return_type", None)),
        )
    return sigs


# ---------------------------------------------------------------------------
# HExpr tree utilities
# ---------------------------------------------------------------------------

def _clone_expr(e: HExpr) -> HExpr:
    """Deep-copy an HExpr tree (HPattern/Ty/span are shared: they are not
    mutated by this pass)."""
    def ce(x: HExpr | None) -> HExpr | None:
        return _clone_expr(x) if x is not None else None

    def ct(seq: Sequence[HExpr] | None) -> tuple[HExpr, ...] | None:
        return tuple(_clone_expr(s) for s in seq) if seq is not None else None

    return replace(
        e,
        operands=ct(e.operands),
        bindings=tuple((n, _clone_expr(s)) for (n, s) in e.bindings) if e.bindings is not None else None,
        left=ce(e.left),
        right=ce(e.right),
        cond=ce(e.cond),
        then_ops=ct(e.then_ops),
        else_ops=ct(e.else_ops),
        scrutinee=ce(e.scrutinee),
        cases=ct(e.cases),
        match_arms=tuple((p, _clone_expr(b)) for (p, b) in e.match_arms) if e.match_arms is not None else None,
        loop_body=ct(e.loop_body),
        assign_value=ce(e.assign_value),
        fields=tuple((n, _clone_expr(s)) for (n, s) in e.fields) if e.fields is not None else None,
        base=ce(e.base),
        field_val=ce(e.field_val),
        lambda_body=ce(e.lambda_body),
        perform_args=ct(e.perform_args),
        handle_cases=tuple((op, ps, _clone_expr(b)) for (op, ps, b) in e.handle_cases) if e.handle_cases is not None else None,
        handle_body=ce(e.handle_body),
    )


def _pattern_binders(p: Any, out: set) -> None:
    """Collect every name an HPattern binds (recursively)."""
    if p is None:
        return
    if getattr(p, "kind", None) == "var" and getattr(p, "name", None):
        out.add(str(p.name))
    for sub in getattr(p, "subpatterns", None) or ():
        _pattern_binders(sub, out)


def _rebound_names(e: HExpr) -> set:
    """Names in this function body that are bound somewhere OTHER than a
    plain ``let`` initializer, or reassigned after binding.

    A ``let``-bound local's type is recorded and reused at call sites below
    it; that is only sound while the name means one thing. Assignment
    targets, lambda parameters, pattern binders (match arms, ``while let``)
    and handler-arm parameters can all give the same spelling a different
    type, so they are excluded from the environment entirely instead of
    being tracked per scope. Conservative by construction: an excluded name
    simply leaves its call sites generic.
    """
    out: set = set()

    def walk(x: HExpr | None) -> None:
        if x is None:
            return
        if x.op == "Assign" and x.var_name:
            out.add(str(x.var_name))
        for n in x.lambda_params or ():
            out.add(str(n))
        for (_op, ps, _b) in x.handle_cases or ():
            # Arm parameters are a name or a tuple of names depending on
            # which handler form built the case triple.
            for n in ((ps,) if isinstance(ps, str) else (ps or ())):
                if n:
                    out.add(str(n))
        for (pat, _body) in x.match_arms or ():
            _pattern_binders(pat, out)
        _pattern_binders(x.loop_pattern, out)
        for child in _child_exprs(x):
            walk(child)

    walk(e)
    return out


def _child_exprs(e: HExpr) -> List[HExpr]:
    out: List[HExpr] = []

    def add(x: Any) -> None:
        if isinstance(x, HExpr):
            out.append(x)

    for seq in (e.operands, e.then_ops, e.else_ops, e.cases, e.loop_body, e.perform_args):
        for s in seq or ():
            add(s)
    for pair_seq in (e.bindings, e.fields):
        for (_n, s) in pair_seq or ():
            add(s)
    for (_p, b) in e.match_arms or ():
        add(b)
    for (_op, _ps, b) in e.handle_cases or ():
        add(b)
    for x in (e.left, e.right, e.cond, e.scrutinee, e.assign_value, e.base,
              e.field_val, e.lambda_body, e.handle_body):
        add(x)
    return out


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------

class _Mono:
    def __init__(self, funcs: Sequence[HFun], sigs: Dict[str, FnSig]) -> None:
        self.sigs = sigs
        self.fn_by_name: Dict[str, HFun] = {str(f.sym): f for f in funcs}
        self.generic_names = {
            name for name, sig in sigs.items()
            if sig.is_generic and name in self.fn_by_name
        }
        # (generic name, concrete type args) -> specialized clone
        self.specialized: Dict[tuple[str, tuple[str, ...]], HFun] = {}
        self.clones_in_order: List[HFun] = []
        # Generic functions with at least one call site the pass could not
        # resolve, or referenced by name as a value: must keep the original.
        self.keep_generic: set[str] = set()
        # Memo: resolved concrete return type per call expression.
        self._call_result_type: Dict[int, str] = {}

    # -- known-type analysis ------------------------------------------------

    def _known_type(self, e: HExpr, param_env: Dict[str, str]) -> str | None:
        if e.op == "Literal":
            v = e.literal
            if isinstance(v, bool):
                return "Bool"
            if isinstance(v, int):
                return "Int"
            if isinstance(v, float):
                return "Float"
            if isinstance(v, str):
                return "String"
            return None
        if e.op == "Struct" and e.struct_name:
            return e.struct_name
        if e.op == "MakeVariant" and e.enum_name:
            return e.enum_name
        if e.op == "Var" and e.var_name:
            return param_env.get(e.var_name)
        if e.op == "Call":
            memo = self._call_result_type.get(id(e))
            if memo is not None:
                return memo
            sig = self.sigs.get(e.callee or "")
            if sig is not None and not sig.is_generic:
                ret = _canon(sig.return_type)
                if ret in _PRIMITIVE_SET:
                    return ret
        return None

    # -- instantiation resolution ------------------------------------------

    def _resolve_instantiation(self, e: HExpr, sig: FnSig,
                               param_env: Dict[str, str]) -> tuple[str, ...] | None:
        if e.type_args:
            if len(e.type_args) != len(sig.type_params):
                return None  # arity mismatch: the type checker already flagged it
            canon = [_canon(t) for t in e.type_args]
            if any(c is None for c in canon):
                return None
            return tuple(c for c in canon if c is not None)
        subst: Dict[str, str] = {}
        for decl, arg in zip(sig.param_types, e.operands or ()):
            if decl in sig.type_params and decl not in subst:
                k = self._known_type(arg, param_env)
                if k is not None:
                    subst[str(decl)] = k
        if all(tp in subst for tp in sig.type_params):
            return tuple(subst[tp] for tp in sig.type_params)
        return None

    # -- specialization -----------------------------------------------------

    def _specialize(self, name: str, targs: tuple[str, ...]) -> str:
        key = (name, targs)
        cached = self.specialized.get(key)
        if cached is not None:
            return str(cached.sym)
        sig = self.sigs[name]
        original = self.fn_by_name[name]
        mangled = MONO_SEP.join((name, *targs))
        subst = dict(zip(sig.type_params, targs))
        clone = HFun(
            sym=mangled,
            params=list(original.params),
            dict_params=list(original.dict_params),
            ret_ty=original.ret_ty,
            where_cls=list(original.where_cls),
            body=_clone_expr(original.body),
            param_modes=dict(original.param_modes) if original.param_modes else None,
        )
        # Memoize BEFORE processing the body so recursive instantiations
        # (f<Int> calling f with the same T) resolve to this clone.
        self.specialized[key] = clone
        self.clones_in_order.append(clone)
        # Inside the clone, a parameter declared with a bare type parameter
        # now has a concrete type: seed the body's known-type environment.
        param_env: Dict[str, str] = {}
        for pname, ptype in zip(sig.param_names, sig.param_types):
            if ptype in subst:
                param_env[pname] = subst[str(ptype)]
        self._process_body(clone.body, param_env)
        return mangled

    # -- body processing ----------------------------------------------------

    def _process_body(self, e: HExpr, param_env: Dict[str, str]) -> None:
        """Process one function body. `param_env` maps names already known to
        have a concrete type on entry (declared primitive parameters of a
        non-generic root; substituted type parameters inside a clone)."""
        rebound = _rebound_names(e)
        self._walk(e, {k: v for k, v in param_env.items() if k not in rebound},
                   rebound)

    def _walk(self, e: HExpr, env: Dict[str, str], rebound: set) -> None:
        # Statement sequences share ONE scope: an HIR block is flat, so a
        # `Let` node's continuation is its following sibling and its binding
        # must be visible there. Every other child gets a copy, so nothing a
        # nested block binds escapes it.
        for seq in (e.operands, e.then_ops, e.else_ops, e.cases,
                    e.loop_body, e.perform_args):
            if not seq:
                continue
            scope = dict(env)
            for s in seq:
                self._walk(s, scope, rebound)
        for (_n, s) in e.fields or ():
            self._walk(s, dict(env), rebound)
        for (_p, b) in e.match_arms or ():
            self._walk(b, dict(env), rebound)
        for (_op, _ps, b) in e.handle_cases or ():
            self._walk(b, dict(env), rebound)
        for x in (e.left, e.right, e.cond, e.scrutinee, e.assign_value,
                  e.base, e.field_val, e.lambda_body, e.handle_body):
            if x is not None:
                self._walk(x, dict(env), rebound)
        if e.op == "Let":
            # Post-order within the binding, then publish the local's type
            # into the ENCLOSING sequence scope for the following siblings.
            for (n, src) in e.bindings or ():
                self._walk(src, env, rebound)
                known = self._known_type(src, env)
                if known is not None and str(n) not in rebound:
                    env[str(n)] = known
                else:
                    env.pop(str(n), None)
            return
        for (_n, s) in e.bindings or ():
            self._walk(s, dict(env), rebound)
        param_env = env
        if e.op == "Var" and e.var_name in self.generic_names:
            # The generic function escapes as a value: keep the original.
            self.keep_generic.add(str(e.var_name))
        if e.op != "Call" or not e.callee:
            return
        name = e.callee
        if name not in self.generic_names:
            return
        sig = self.sigs[name]
        targs = self._resolve_instantiation(e, sig, param_env)
        if targs is None:
            self.keep_generic.add(name)
            return
        e.callee = self._specialize(name, targs)
        if sig.return_type in sig.type_params:
            subst = dict(zip(sig.type_params, targs))
            ret = subst.get(str(sig.return_type))
            if ret is not None:
                self._call_result_type[id(e)] = ret

    def _declared_param_env(self, name: str) -> Dict[str, str]:
        """Parameters of a NON-GENERIC function whose declared type is a
        primitive are known concretely inside its body, so `fn wrap(n: int)
        { identity(n) }` resolves like `identity(1)`. Restricted to
        primitives on purpose: a declared aggregate ('LinkedList[T]') canonicalizes
        to its head constructor, which is not the same information as the
        struct/variant names `_known_type` reports, and guessing there would
        pick a clone by a name that means something else."""
        sig = self.sigs.get(name)
        if sig is None or sig.is_generic:
            return {}
        env: Dict[str, str] = {}
        for pname, ptype in zip(sig.param_names, sig.param_types):
            canon = _canon(ptype)
            if canon in _PRIMITIVE_SET and ptype == (ptype or "").split("[", 1)[0]:
                env[str(pname)] = str(canon)
        return env

    def run(self, funcs: Sequence[HFun]) -> List[HFun]:
        roots = [f for f in funcs if str(f.sym) not in self.generic_names]
        for f in roots:
            self._process_body(f.body, self._declared_param_env(str(f.sym)))

        def survives(name: str) -> bool:
            """A generic original stays loaded unless every visible call
            site was rewritten to a specialized clone AND nothing else
            references it. Synthesized names (__impl$Trait$Type$m,
            __effect_default$..., __static$...) are reached DYNAMICALLY by
            name pattern at runtime (trait dispatch, effect defaults) —
            call-site analysis cannot see those references, so they always
            survive; so do generics that were never specialized at all."""
            return (
                name.startswith("__")
                or name in self.keep_generic
                or all(spec_name != name for (spec_name, _t) in self.specialized)
            )

        # Surviving generic originals still execute at runtime: analyze
        # their bodies too, so functions THEY call stay loaded (or
        # specialize where locally resolvable). Fixpoint: analysis can mark
        # more generics as surviving.
        analyzed: set[str] = set()
        changed = True
        while changed:
            changed = False
            for name in sorted(self.generic_names):
                if name in analyzed or not survives(name):
                    continue
                analyzed.add(name)
                self._process_body(self.fn_by_name[name].body, {})
                changed = True

        out: List[HFun] = []
        for f in funcs:
            name = str(f.sym)
            if name in self.generic_names and not survives(name):
                continue
            out.append(f)
        out.extend(self.clones_in_order)
        return out


def monomorphize_hir(funcs: Sequence[HFun], sigs: Dict[str, FnSig]) -> List[HFun]:
    """Monomorphize the given HIR functions in place (call sites are
    rewritten on the given HFun bodies) and return the new function list:
    non-generic functions, still-needed generic originals, and specialized
    clones. Behavior-preserving by construction: unresolvable call sites
    keep their generic callee and the generic original stays loaded."""
    return _Mono(funcs, sigs).run(funcs)
