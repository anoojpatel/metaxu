from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

try:
    from metaxu.type_defs import CompactType
except Exception:
    CompactType = None  # type: ignore

from .mutaxu_ast import AstNode
from .simplesub_adapter import SimpleSubFacade
from .frozen_borrow_checker import FrozenBorrowChecker, BorrowError

# Emits type/effect constraints over the frozen AST (mutaxu_ast) and drives
# the frozen borrow checker (modes, locality, regions, linearity, exclave).


_UNIQUENESS_MODES = {"shared", "unique", "exclusive"}
# Surface syntax spellings (@owned/@mut/@const) map onto spec uniqueness modes.
_UNIQUENESS_ALIASES = {"owned": "unique", "mut": "exclusive", "const": "shared"}
_LOCALITY_MODES = {"local", "global"}
_LINEARITY_MODES = {"once", "separate", "many"}


def _split_mode(value: dict) -> tuple[str, str, str | None]:
    """Interpret frozen mode payloads into (uniqueness, locality, linearity).

    Handles the shapes the frozen AST can carry today or in the near future:
    - a bare string mode (e.g. "unique", "local", "once", "@mut" spellings)
    - a list/tuple of such strings
    - a dict with explicit "uniqueness"/"locality"/"linearity" entries
    - separate top-level "uniqueness"/"locality"/"linearity" payload keys
    Anything absent defaults to the permissive ("shared", "global", None) so
    unannotated code keeps compiling exactly as before.
    """
    uniqueness: str | None = None
    locality: str | None = None
    linearity: str | None = None

    def absorb(token: Any) -> None:
        nonlocal uniqueness, locality, linearity
        if not isinstance(token, str):
            return
        tok = token.lower().lstrip("@")
        tok = _UNIQUENESS_ALIASES.get(tok, tok)
        if tok in _UNIQUENESS_MODES and uniqueness is None:
            uniqueness = tok
        elif tok in _LOCALITY_MODES and locality is None:
            locality = tok
        elif tok in _LINEARITY_MODES and linearity is None:
            linearity = tok

    raw = value.get("mode")
    if isinstance(raw, dict):
        absorb(raw.get("uniqueness"))
        absorb(raw.get("locality"))
        absorb(raw.get("linearity"))
    elif isinstance(raw, (list, tuple)):
        for token in raw:
            absorb(token)
    else:
        absorb(raw)
    absorb(value.get("uniqueness"))
    absorb(value.get("locality"))
    absorb(value.get("linearity"))

    return uniqueness or "shared", locality or "global", linearity


def _explicit_locality(raw: Any) -> str | None:
    """Extract an *explicitly annotated* locality mode from a frozen mode payload.

    Unlike _split_mode this does NOT default: it returns "local"/"global" only
    when the annotation actually spells it out, and None otherwise. Deep
    ownership validation must distinguish "declared @global" from the
    permissive unannotated default (structs default to local allocation)."""
    def scan(token: Any) -> str | None:
        if isinstance(token, str):
            tok = token.lower().lstrip("@")
            if tok in _LOCALITY_MODES:
                return tok
        return None

    if isinstance(raw, dict):
        return scan(raw.get("locality"))
    if isinstance(raw, (list, tuple)):
        for token in raw:
            found = scan(token)
            if found is not None:
                return found
        return None
    return scan(raw)


_BORROW_NODE_MODES = {
    "BorrowShared": "shared",
    "BorrowUnique": "unique",
    "BorrowExclusive": "exclusive",
}


def emit_constraints(frozen_root: Any, types: Dict[int, Any], simplesub: Any) -> Tuple[None, List[BorrowError]]:
    """Walk the frozen AST and emit constraints through the provided facade.

    Arguments:
      frozen_root: root node from metaxu.compiler.mutaxu_ast
      types: mapping of node_id -> CompactType (allocated by caller)
      simplesub: an adapter/facade with add_* APIs and solve()

    Returns:
      Tuple of (None, borrow_errors) where borrow_errors is a list of
      structured BorrowError objects (kind, variable, node_id, message).
    """
    scopes: list[dict[str, Any]] = [{}]
    return_types: list[Any] = []
    borrow_checker = FrozenBorrowChecker()
    effect_classes: dict[str, str] = {}  # effect_name -> effect_class (stack/suspend)
    function_effects: dict[str, list[str]] = {}  # function_name -> list of effects it performs
    declared_linearity: dict[str, str] = {}  # callable_name -> linearity ("once"/"separate"/"many")
    function_region_stack: list[int] = []  # region id at each enclosing function's entry
    handler_contexts: list[dict[str, str]] = []  # Stack of handler contexts: effect_name -> effect_class
    enclosing_performs: list[frozenset[str]] = []  # declared performs of enclosing functions
    handler_locals: list[set[str]] = []  # Track variables assigned in handler contexts
    handler_operations: list[list[str]] = []  # Track operations in handler (for stack effect checking)
    struct_defs: dict[str, dict[str, Any]] = {}  # struct name -> frozen payload (fields/type_params)
    enum_defs: dict[str, dict[str, Any]] = {}  # enum name -> frozen payload (variants)
    fn_defs: dict[str, dict[str, Any]] = {}  # function name -> frozen payload (signature)
    variant_to_enum: dict[str, str] = {}  # variant name -> enum name
    impl_pairs: set[tuple[str, str]] = set()  # (trait name, type base name)
    # (trait, type) -> (impl where constraints, impl type-param names, node):
    # collected from implement blocks (pre-desugar Implementation payloads or
    # post-desugar __impl$ function payloads) for coherence-load checking.
    impl_wheres: dict[tuple[str, str], tuple[list[dict[str, Any]], frozenset[str], Any]] = {}
    # Variables bound by an explicit `let @global ... = StructName { ... }`:
    # var name -> struct type name. Used for deep ownership checks on later
    # field assignments (global containers cannot store locals).
    global_struct_bindings: dict[str, str] = {}

    def bind(name: Any, ty: Any) -> None:
        if isinstance(name, str):
            scopes[-1][name] = ty

    def lookup(name: Any) -> Any | None:
        if not isinstance(name, str):
            return None
        for scope in reversed(scopes):
            if name in scope:
                return scope[name]
        return None

    def payload_name(node: Any) -> Any | None:
        value = getattr(node, "value", None)
        if isinstance(value, dict):
            return value.get("name")
        return None

    def payload_operator(node: Any) -> Any | None:
        value = getattr(node, "value", None)
        if isinstance(value, dict):
            return value.get("operator")
        return None

    def payload_dict(node: Any) -> dict[str, Any]:
        value = getattr(node, "value", None)
        return value if isinstance(value, dict) else {}

    _PRIMITIVE_NAMES = {
        "int": "Int", "Int": "Int",
        "str": "String", "string": "String", "String": "String",
        "bool": "Bool", "Bool": "Bool",
        "float": "Float", "Float": "Float",
    }

    def _literal_type_name(value: Any) -> str | None:
        if isinstance(value, bool):
            return "Bool"
        if isinstance(value, int):
            return "Int"
        if isinstance(value, float):
            return "Float"
        if isinstance(value, str):
            return "String"
        return None

    _PRIMITIVE_SET = frozenset({"Int", "String", "Bool", "Float"})

    def _base_type_name(display: str) -> str:
        """Base constructor of a type display: "Vec[Int]" -> "Vec"."""
        return display.split("[", 1)[0].split("<", 1)[0].strip()

    def _canon_type(display: Any) -> str | None:
        """Canonical name for a declared/argument type display.

        Primitives normalize to Int/String/Bool/Float; other names reduce
        to their base constructor. None stays None."""
        if not isinstance(display, str) or not display:
            return None
        base = _base_type_name(display)
        return _PRIMITIVE_NAMES.get(base, base)

    def _expr_known_type(node: Any) -> str | None:
        """Concrete type name of an expression when statically KNOWN.

        Covers literals, struct instantiations, enum variant constructions,
        and calls to non-generic functions with a declared primitive return
        type. Everything else (variables, field accesses, generic calls...)
        returns None and is left to the constraint machinery."""
        kind = getattr(node, "kind", None)
        if kind == "Literal":
            return _literal_type_name(getattr(node, "value", None))
        if kind == "StructInstantiation":
            name = payload_dict(node).get("name")
            return name if isinstance(name, str) else None
        if kind in ("FunctionCall", "QualifiedFunctionCall"):
            name = payload_dict(node).get("name")
            if not isinstance(name, str):
                return None
            if name in variant_to_enum:
                return variant_to_enum[name]
            sig = fn_defs.get(name)
            if sig and not (sig.get("type_params") or []):
                ret = _canon_type(sig.get("return_type"))
                if ret in _PRIMITIVE_SET:
                    return ret
        return None

    # ------------------------------------------------------------------
    # Type-display algebra: substitution inside type applications
    # ------------------------------------------------------------------
    # A parsed display is a pair (base, args): args is None for a bare
    # name ("Int" -> ("Int", None)) and a list of parsed displays for a
    # type application ("Map[K, Vec[V]]" -> ("Map", [("K", None),
    # ("Vec", [("V", None)])])). Bases are canonicalized primitives.

    _BUILTIN_TYPE_CTORS = frozenset({"Vec", "vector", "Option", "Result"})

    def _split_top_level(s: str) -> list[str]:
        """Split "K, Vec[V]" on commas at bracket depth 0."""
        parts: list[str] = []
        depth = 0
        cur = ""
        for ch in s:
            if ch in "[<":
                depth += 1
            elif ch in "]>":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append(cur)
                cur = ""
            else:
                cur += ch
        parts.append(cur)
        return parts

    def _parse_display(display: Any) -> tuple[str, list | None] | None:
        """Parse a type display into a (base, args) tree, or None."""
        if not isinstance(display, str) or not display.strip():
            return None
        s = display.strip()
        for open_c, close_c in (("[", "]"), ("<", ">")):
            i = s.find(open_c)
            if i > 0 and s.endswith(close_c):
                base = _PRIMITIVE_NAMES.get(s[:i].strip(), s[:i].strip())
                args = [_parse_display(a) for a in _split_top_level(s[i + 1:-1])]
                if any(a is None for a in args):
                    return (base, None)  # opaque args: keep the base only
                return (base, args)
        if "[" in s or "<" in s:
            canon = _canon_type(s)
            return (canon, None) if canon else None
        return (_PRIMITIVE_NAMES.get(s, s), None)

    def _fmt_parsed(t: tuple[str, list | None]) -> str:
        base, args = t
        if not args:
            return base
        return f"{base}[{', '.join(_fmt_parsed(a) for a in args)}]"

    def _subst_parsed(t: tuple[str, list | None],
                      deep_subst: dict[str, tuple[str, list | None]],
                      tparams: set[str] | frozenset[str]) -> tuple[str, list | None]:
        """Substitute resolved type parameters inside a parsed display.
        Unresolved parameters stay as bare names (skipped by the checks)."""
        base, args = t
        if args is None:
            if base in tparams:
                return deep_subst.get(base, t)
            return t
        return (base, [_subst_parsed(a, deep_subst, tparams) for a in args])

    def _known_type_ctor(name: str) -> bool:
        """Is `name` a type constructor the checker actually knows?
        Unknown names (aliases, foreign types, unresolved parameters) make
        the surrounding check permissive — no false positives."""
        return (name in _PRIMITIVE_SET or name in struct_defs
                or name in enum_defs or name in _BUILTIN_TYPE_CTORS)

    def _expr_known_type_deep(node: Any) -> tuple[str, list | None] | None:
        """Parsed known type of an expression, including type arguments
        where the expression spells them out (`Pair<String> {...}`,
        `Full<Int>(x)`). Args are None when unknown/omitted."""
        base = _expr_known_type(node)
        if base is None:
            return None
        kind = getattr(node, "kind", None)
        if kind in ("StructInstantiation", "FunctionCall", "QualifiedFunctionCall"):
            explicit = [a for a in payload_dict(node).get("type_args", []) or []
                        if isinstance(a, str)]
            if explicit:
                args = [_parse_display(a) for a in explicit]
                if all(a is not None for a in args):
                    return (base, args)
            return (base, None)
        return (_PRIMITIVE_NAMES.get(base, base), None)

    def _parsed_conflict(expected: tuple[str, list | None],
                         actual: tuple[str, list | None],
                         tparams: set[str] | frozenset[str]) -> bool:
        """Structural conflict between a (substituted) declared type and a
        known value type. Positions whose constructor is unknown, or where
        a type parameter stayed unresolved, are permissive."""
        ebase, eargs = expected
        abase, aargs = actual
        if ebase in tparams:
            return False  # unresolved parameter position: left to inference
        if not _known_type_ctor(ebase) or not _known_type_ctor(abase):
            return False  # unknown constructor: stay permissive
        if ebase != abase:
            return True
        if eargs and aargs and len(eargs) == len(aargs):
            return any(_parsed_conflict(e, a, tparams)
                       for e, a in zip(eargs, aargs))
        return False

    def _check_declared_type(expected_display: Any,
                             tparams: list[str],
                             deep_subst: dict[str, tuple[str, list | None]],
                             value: Any,
                             mk_msg: Callable[[str, str], str],
                             err_node: Any,
                             variable: str | None = None) -> None:
        """Check one value against one declared type display, substituting
        resolved type parameters — including INSIDE type applications
        (`Vec[T]` with T=Int checks against Vec[Int]).

        Bare primitive expectations are enforced directly for known-typed
        values and via class constraints for var-typed values; bare
        struct/enum expectations by base name for known-typed values. Type
        applications check the value's base constructor, and recurse into
        argument positions where the value's own type arguments are known
        (explicit instantiations). Everything unknown stays permissive."""
        parsed = _parse_display(expected_display)
        if parsed is None:
            return
        tparam_set = set(tparams)
        resolved = _subst_parsed(parsed, deep_subst, tparam_set)
        base, args = resolved
        if args is None:
            expected = _canon_type(base)
            if expected is None or expected in tparam_set:
                return  # unresolved type parameter: left to inference
            actual = _expr_known_type(value)
            if expected in _PRIMITIVE_SET:
                if actual is not None and actual != expected:
                    _type_error(mk_msg(expected, actual), err_node,
                                variable=variable)
                value_ty = types.get(value.node_id)
                if value_ty is not None:
                    simplesub.add_class_constraint(expected, [value_ty],
                                                   value.node_id)
            elif expected in struct_defs or expected in enum_defs:
                if actual is not None and _base_type_name(actual) != expected:
                    _type_error(mk_msg(expected, actual), err_node,
                                variable=variable)
            return
        # Type application (`Vec[Int]`, `Pair[Int]`, `Map[K, Vec[V]]`).
        if not _known_type_ctor(base):
            return  # unknown constructor: permissive
        actual_deep = _expr_known_type_deep(value)
        if actual_deep is None:
            return  # value type unknown: left to inference
        if _parsed_conflict(resolved, actual_deep, tparam_set):
            _type_error(mk_msg(_fmt_parsed(resolved), _fmt_parsed(actual_deep)),
                        err_node, variable=variable)

    def _type_error(message: str, node: Any, kind: str = "type-mismatch",
                    variable: str | None = None) -> None:
        borrow_checker.errors.append(BorrowError(
            message=message, node_id=node.node_id, kind=kind, variable=variable,
        ))

    def _collect_definitions(root: Any) -> None:
        """Prepass: record struct/enum/function definitions and the trait-impl
        registry before constraint emission, so instantiation checking works
        regardless of declaration order (use-before-def is legal)."""
        def scan(n: Any) -> None:
            kind = getattr(n, "kind", None)
            value = getattr(n, "value", None)
            payload = value if isinstance(value, dict) else {}
            name = payload.get("name")
            if kind == "StructDefinition" and isinstance(name, str):
                struct_defs.setdefault(name, payload)
            elif kind == "EnumDefinition" and isinstance(name, str):
                enum_defs.setdefault(name, payload)
                for v in payload.get("variants", []) or []:
                    vname = v.get("name") if isinstance(v, dict) else None
                    if isinstance(vname, str):
                        variant_to_enum.setdefault(vname, name)
            elif kind == "FunctionDeclaration" and isinstance(name, str):
                fn_defs.setdefault(name, payload)
                # Post-desugar trait impls are functions named
                # __impl$Trait$Type$method; parse the registry out of them.
                if name.startswith("__impl$"):
                    parts = name.split("$")
                    if len(parts) >= 4:
                        impl_pairs.add((parts[1], parts[2]))
                        where = payload.get("impl_where")
                        if isinstance(where, list) and where:
                            impl_wheres.setdefault(
                                (parts[1], parts[2]),
                                (where,
                                 frozenset(p for p in payload.get("impl_params")
                                           or [] if isinstance(p, str)),
                                 n))
            elif kind == "Implementation":
                # Pre-desugar impl blocks carry {"trait", "type"} payloads.
                trait = payload.get("trait")
                type_name = payload.get("type")
                if isinstance(trait, str) and isinstance(type_name, str):
                    key = (_base_type_name(trait), _base_type_name(type_name))
                    impl_pairs.add(key)
                    where = payload.get("where")
                    if isinstance(where, list) and where:
                        impl_wheres.setdefault(
                            key,
                            (where,
                             frozenset(p for p in payload.get("type_params")
                                       or [] if isinstance(p, str)),
                             n))
            for c in getattr(n, "children", ()):
                scan(c)
        scan(root)
        _check_impl_where_clauses()

    def _check_impl_where_clauses() -> None:
        """Coherence-load enforcement of impl-block where clauses.

        A constraint over a CONCRETE type (`implement Show for P where
        P: Eq`) is decidable against the impl registry as soon as all
        impls are collected: the bound type either has the required impl
        or it never will. Constraints over the impl's own type parameters
        (`implement Show for Pair[T] where T: Show` — conditional impls)
        depend on each instantiation's type arguments, which the base-name
        registry cannot see; they stay permissive (runtime dispatch still
        enforces), which is documented in docs/v1_gap_analysis.md."""
        for (trait, type_name), (constraints, params, node) in impl_wheres.items():
            for c in constraints:
                if not isinstance(c, dict):
                    continue
                param = c.get("param")
                bound_trait = c.get("trait")
                if not isinstance(param, str) or not isinstance(bound_trait, str):
                    continue
                conc = _canon_type(param)
                if conc is None or param in params or conc in params:
                    continue  # impl type parameter: instantiation-dependent
                if not _known_type_ctor(conc):
                    continue  # unknown type name: stay permissive
                if (_base_type_name(bound_trait), conc) in impl_pairs:
                    continue
                _type_error(
                    f"no implementation of trait {bound_trait} for {conc}: "
                    f"`implement {trait} for {type_name}` requires "
                    f"{param}: {bound_trait} in its where clause "
                    f"(missing `implement {bound_trait} for {conc}`)",
                    node, kind="type-missing-impl",
                )

    def _check_where_clauses(sig: dict[str, Any], fn_name: str,
                             subst: dict[str, str], node: Any) -> None:
        """Enforce `where T: Trait` bounds for a resolved instantiation.

        Only fires when the type parameter was instantiated with a KNOWN
        concrete type; unresolved parameters stay unchecked (advisory-free:
        dynamic dispatch still enforces at runtime)."""
        for c in sig.get("where") or []:
            if not isinstance(c, dict):
                continue
            param = c.get("param")
            trait = c.get("trait")
            if not isinstance(param, str) or not isinstance(trait, str):
                continue
            conc = subst.get(param)
            if conc is None:
                continue
            if not _known_type_ctor(conc):
                # Not a type the checker knows concretely (e.g. a caller's
                # own type parameter flowing through `describe<U>(y)`):
                # stay permissive, runtime dispatch enforces.
                continue
            if (trait, conc) in impl_pairs:
                continue
            _type_error(
                f"no implementation of trait {trait} for {conc}: "
                f"call of {fn_name} instantiates {param}={conc}, but "
                f"{param}: {trait} is required by its where clause "
                f"(missing `implement {trait} for {conc}`)",
                node, kind="type-missing-impl",
            )

    def _infer_call_site_subst(
        type_params: list[str],
        declared: list[Any],
        args: list[Any],
        node: Any,
        fn_name: str,
    ) -> dict[str, str]:
        """Infer a CALL-SITE-LOCAL instantiation for omitted type args.

        Each call site gets its own substitution (let-polymorphism lite):
        nothing here touches the callee's declared parameter type vars, so
        `identity(1)` and `identity("s")` coexist. Arguments whose declared
        type is the same bare type parameter are unified with each other
        (within this call only) so var-typed args participate in the
        constraint-graph conflict detection."""
        subst: dict[str, str] = {}
        groups: dict[str, list[Any]] = {}
        for decl, arg in zip(declared, args):
            if isinstance(decl, str) and decl in type_params:
                groups.setdefault(decl, []).append(arg)
        for param, group in groups.items():
            knowns: dict[str, Any] = {}
            for arg in group:
                k = _expr_known_type(arg)
                if k is not None and k not in knowns:
                    knowns[k] = arg
            if len(knowns) > 1:
                pretty = " and ".join(sorted(knowns))
                _type_error(
                    f"conflicting instantiations of type parameter {param} "
                    f"in call of {fn_name}: arguments require {param} to be "
                    f"both {pretty}",
                    node,
                )
            elif knowns:
                subst[param] = next(iter(knowns))
            # Call-site-local unification: args sharing one type parameter
            # must agree in type. Fresh per call site — only THIS call's
            # argument vars are linked, never the callee's param vars.
            group_tys = [types.get(a.node_id) for a in group]
            group_tys = [t for t in group_tys if t is not None]
            for a, b in zip(group_tys, group_tys[1:]):
                simplesub.add_unify(a, b)
        return subst

    def _check_call_types(node: Any) -> None:
        """Parametric instantiation checking for a plain function call.

        Explicit type args (`identity<Int>(x)`) are substituted into the
        declared parameter types; omitted ones are inferred call-site-
        locally. Substituted primitive parameter types are enforced both
        directly (known-typed args) and via class constraints (var-typed
        args feed the conflict-detection machinery); struct/enum parameter
        types are enforced for known-typed args. Where clauses are checked
        against the resolved instantiation."""
        payload = payload_dict(node)
        name = payload.get("name")
        if not isinstance(name, str):
            return
        if name in variant_to_enum:
            _check_variant_construction(node, name)
            return
        sig = fn_defs.get(name)
        if not sig:
            return
        type_params = [p for p in sig.get("type_params") or [] if isinstance(p, str)]
        declared = list(sig.get("param_types") or [])
        args = list(getattr(node, "children", ()))
        explicit = [a for a in payload.get("type_args", []) or [] if isinstance(a, str)]
        subst: dict[str, str] = {}
        deep_subst: dict[str, tuple[str, list | None]] = {}
        if explicit:
            if not type_params:
                _type_error(
                    f"type arguments given to non-generic function {name}",
                    node, kind="type-arg-arity",
                )
                return
            if len(explicit) != len(type_params):
                _type_error(
                    f"wrong number of type arguments for {name}: expected "
                    f"{len(type_params)} ({', '.join(type_params)}), got "
                    f"{len(explicit)}",
                    node, kind="type-arg-arity",
                )
                return
            subst = {
                p: c for p, c in zip(type_params, (_canon_type(a) for a in explicit))
                if c is not None
            }
            deep_subst = {
                p: t for p, t in zip(type_params, (_parse_display(a) for a in explicit))
                if t is not None
            }
        elif type_params:
            subst = _infer_call_site_subst(type_params, declared, args, node, name)
            deep_subst = {p: (c, None) for p, c in subst.items()}
        # Per-argument checks against the (substituted) declared types,
        # including substitution inside type applications (`Vec[T]`).
        for decl, arg in zip(declared, args):
            if not isinstance(decl, str):
                continue
            _check_declared_type(
                decl, type_params, deep_subst, arg,
                lambda exp, act: (
                    f"type mismatch in call of {name}: argument has type "
                    f"{act}, expected {exp}"),
                arg,
            )
        # Result type: a generic function whose declared return type is a
        # resolved type parameter constrains the call's result var.
        ret = sig.get("return_type")
        if isinstance(ret, str) and ret in type_params and ret in subst:
            ret_canon = subst[ret]
            node_ty = types.get(node.node_id)
            if ret_canon in _PRIMITIVE_SET and node_ty is not None:
                simplesub.add_class_constraint(ret_canon, [node_ty], node.node_id)
        _check_where_clauses(sig, name, subst, node)

    def _check_variant_construction(node: Any, variant_name: str) -> None:
        """Check an enum variant construction (`Full(3)`, `Full<Int>(3)`)
        against the enum's declared payload types, substituting explicit or
        call-site-inferred type args for the enum's type parameters."""
        enum_name = variant_to_enum[variant_name]
        edef = enum_defs.get(enum_name) or {}
        eparams = [p for p in edef.get("type_params") or [] if isinstance(p, str)]
        variant = next(
            (v for v in edef.get("variants", []) or []
             if isinstance(v, dict) and v.get("name") == variant_name),
            None,
        )
        if variant is None:
            return
        declared = [
            f.get("type") for f in variant.get("fields", []) or []
            if isinstance(f, dict)
        ]
        args = list(getattr(node, "children", ()))
        payload = payload_dict(node)
        explicit = [a for a in payload.get("type_args", []) or [] if isinstance(a, str)]
        subst: dict[str, str] = {}
        deep_subst: dict[str, tuple[str, list | None]] = {}
        if explicit:
            if len(explicit) != len(eparams):
                _type_error(
                    f"wrong number of type arguments for {enum_name}."
                    f"{variant_name}: expected {len(eparams)}, got {len(explicit)}",
                    node, kind="type-arg-arity",
                )
                return
            subst = {
                p: c for p, c in zip(eparams, (_canon_type(a) for a in explicit))
                if c is not None
            }
            deep_subst = {
                p: t for p, t in zip(eparams, (_parse_display(a) for a in explicit))
                if t is not None
            }
        elif eparams:
            subst = _infer_call_site_subst(eparams, declared, args, node,
                                           f"{enum_name}.{variant_name}")
            deep_subst = {p: (c, None) for p, c in subst.items()}
        for decl, arg in zip(declared, args):
            if not isinstance(decl, str):
                continue
            _check_declared_type(
                decl, eparams, deep_subst, arg,
                lambda exp, act: (
                    f"type mismatch in {enum_name}.{variant_name}: payload "
                    f"has type {act}, expected {exp}"),
                arg,
            )

    def _check_struct_field_types(node: Any, struct_name: str) -> None:
        """Check field values against the struct's declared field types.

        The instantiation's explicit type arguments are substituted for the
        struct's type parameters. A substituted primitive field type is
        enforced directly against values of KNOWN type (literals, struct
        instantiations, variant constructions, calls with declared primitive
        returns) and via class constraints for var-typed values (which feeds
        the constraint-graph conflict detection). A substituted struct/enum
        field type is enforced against known-typed values by base name.
        Values of unknown type are left to inference.
        """
        definition = struct_defs.get(struct_name)
        if not definition:
            return
        declared = {
            f.get("name"): f.get("type")
            for f in definition.get("fields", []) or []
            if isinstance(f, dict)
        }
        params = [p for p in definition.get("type_params", []) or [] if isinstance(p, str)]
        args = [a for a in payload_dict(node).get("type_args", []) or [] if isinstance(a, str)]
        if args and len(args) != len(params):
            _type_error(
                f"wrong number of type arguments for {struct_name}: expected "
                f"{len(params)}, got {len(args)}",
                node, kind="type-arg-arity",
            )
            return
        deep_subst = {
            p: t for p, t in zip(params, (_parse_display(a) for a in args))
            if t is not None
        }
        for field in getattr(node, "children", ()):
            if getattr(field, "kind", None) != "StructField":
                continue
            field_name = payload_dict(field).get("name")
            expected = declared.get(field_name)
            if not isinstance(expected, str):
                continue
            value_children = list(getattr(field, "children", ()))
            value = value_children[0] if value_children else None
            if value is None:
                continue
            fname = field_name if isinstance(field_name, str) else None
            _check_declared_type(
                expected, params, deep_subst, value,
                lambda exp, act: (
                    f"type mismatch for field '{field_name}' of {struct_name}: "
                    f"expected {exp}, got {act}"),
                field, variable=fname,
            )

    def _struct_field_modes(struct_name: str) -> dict[str, tuple[str | None, str | None]]:
        """Registry view of a struct definition: field -> (type_name, locality).

        Locality is the field's *declared* locality mode (None when the field
        carries no @local/@global annotation)."""
        definition = struct_defs.get(struct_name)
        if not definition:
            return {}
        registry: dict[str, tuple[str | None, str | None]] = {}
        for f in definition.get("fields", []) or []:
            if not isinstance(f, dict):
                continue
            fname = f.get("name")
            if isinstance(fname, str):
                ftype = f.get("type")
                registry[fname] = (
                    ftype if isinstance(ftype, str) else None,
                    _explicit_locality(f.get("mode")),
                )
        return registry

    def _find_local_field_path(type_name: str, visited: frozenset[str] = frozenset()) -> list[str] | None:
        """Find a path to a transitively @local-declared field of `type_name`.

        Recurses through nested struct types (and enum variant payload types)
        recorded in the frozen definitions. Returns a list of path segments
        like ["Outer.inner", "Inner.temp"], or None when no @local field is
        reachable."""
        if type_name in visited:
            return None
        visited = visited | {type_name}
        if type_name in struct_defs:
            for fname, (ftype, flocality) in _struct_field_modes(type_name).items():
                if flocality == "local":
                    return [f"{type_name}.{fname}"]
                if isinstance(ftype, str):
                    sub = _find_local_field_path(ftype, visited)
                    if sub is not None:
                        return [f"{type_name}.{fname}"] + sub
            return None
        enum_definition = enum_defs.get(type_name)
        if enum_definition:
            for variant in enum_definition.get("variants", []) or []:
                if not isinstance(variant, dict):
                    continue
                for f in variant.get("fields", []) or []:
                    ftype = f.get("type") if isinstance(f, dict) else None
                    if isinstance(ftype, str):
                        sub = _find_local_field_path(ftype, visited)
                        if sub is not None:
                            return [f"{type_name}.{variant.get('name')}"] + sub
        return None

    def _is_local_variable(name: Any) -> bool:
        if not isinstance(name, str):
            return False
        info = borrow_checker.variables.get(name)
        return info is not None and info.locality == "local"

    def _check_global_struct_binding(var_name: str, inst_node: Any) -> None:
        """Deep ownership rules for `let @global v = S { ... }`.

        - S must not (transitively) declare any @local field.
        - The initializer must not store @local-bound values in fields."""
        struct_name = payload_dict(inst_node).get("name")
        if not isinstance(struct_name, str):
            return
        global_struct_bindings[var_name] = struct_name
        path = _find_local_field_path(struct_name)
        if path is not None:
            borrow_checker.errors.append(BorrowError(
                message=(
                    f"@global binding '{var_name}' of struct {struct_name} "
                    f"contains @local field {' -> '.join(path)}; a @global "
                    f"value must not contain (transitively) any @local field"
                ),
                node_id=inst_node.node_id,
                kind="deep-locality",
                variable=var_name,
            ))
        for field in getattr(inst_node, "children", ()):
            if getattr(field, "kind", None) != "StructField":
                continue
            field_name = payload_dict(field).get("name")
            for child in getattr(field, "children", ()):
                if getattr(child, "kind", None) != "Variable":
                    continue
                value_name = payload_dict(child).get("name")
                if _is_local_variable(value_name):
                    borrow_checker.errors.append(BorrowError(
                        message=(
                            f"cannot store @local value '{value_name}' in field "
                            f"'{field_name}' of @global {struct_name} binding "
                            f"'{var_name}'; global containers cannot store locals"
                        ),
                        node_id=child.node_id,
                        kind="deep-locality",
                        variable=value_name,
                    ))

    def param_nodes(children: Any) -> list[Any]:
        return [child for child in children if getattr(child, "kind", None) == "Parameter"]

    def non_param_nodes(children: Any) -> list[Any]:
        return [child for child in children if getattr(child, "kind", None) != "Parameter"]

    def literal_class(value: Any) -> str | None:
        if isinstance(value, bool):
            return "Bool"
        if isinstance(value, int):
            return "Int"
        if isinstance(value, float):
            return "Float"
        if isinstance(value, str):
            return "String"
        if value is None:
            return "Unit"
        return None

    # ------------------------------------------------------------------
    # Match exhaustiveness (compile time)
    # ------------------------------------------------------------------
    # Frozen MatchExpression payloads carry {"arms": [descriptor, ...]} (see
    # mutaxu_ast._pattern_descriptor for the shapes). The check is
    # pattern-driven: ctor arms determine the enum being matched (which is
    # exactly the case where the scrutinee's enum type is statically known —
    # a declared enum or the builtin Option/Result constructors), literal
    # arms determine a literal class. Where neither determines a type (or
    # any pattern is opaque/unknown), no check is performed: unknown-typed
    # scrutinees stay permissive with no false positives.
    #
    # Coverage rule (kept deliberately shallow): a variant is covered iff
    # some arm names its ctor with all-irrefutable subpatterns
    # (wildcards/bindings), OR a wildcard/binding arm exists. Deeper
    # refinement is NOT attempted: `Some(1) | Some(n)` treats Some as
    # covered by the binding arm `Some(n)`, while `Some(1)` alone leaves
    # Some incompletely covered (literal completeness over an infinite
    # domain is not analyzed).

    _BUILTIN_ENUM_VARIANTS = {"Option": ("Some", "None"), "Result": ("Ok", "Err")}
    _BUILTIN_VARIANT_ENUM = {"Some": "Option", "None": "Option",
                             "Ok": "Result", "Err": "Result"}

    def _resolve_pattern(desc: Any) -> dict[str, Any]:
        """Resolve a frozen pattern descriptor against known enum variants.

        A bare identifier ({"kind": "name"}) is a zero-arg constructor
        pattern when the name is a known variant (`Point => ...`), and an
        irrefutable binding otherwise (mirrors HIR pattern conversion)."""
        if not isinstance(desc, dict):
            return {"kind": "unknown"}
        if desc.get("kind") == "name":
            nm = desc.get("name")
            enum_name = variant_to_enum.get(nm) or _BUILTIN_VARIANT_ENUM.get(nm)
            if isinstance(nm, str) and enum_name is not None:
                return {"kind": "ctor", "name": nm, "enum": enum_name,
                        "subpatterns": []}
            return {"kind": "binding", "name": nm}
        return desc

    def _has_unknown_pattern(desc: Any) -> bool:
        if not isinstance(desc, dict):
            return True
        if desc.get("kind") == "unknown":
            return True
        return any(_has_unknown_pattern(s)
                   for s in desc.get("subpatterns") or [])

    def _check_match_exhaustiveness(node: Any) -> None:
        arms = payload_dict(node).get("arms")
        if not isinstance(arms, list) or not arms:
            return
        resolved = [_resolve_pattern(d) for d in arms]

        # Redundancy advisory (warning channel, not an error): every arm
        # after the first wildcard/binding arm is unreachable —
        # top-to-bottom, first match wins.
        advisories = getattr(simplesub, "advisories", None)
        if advisories is not None:
            for i, d in enumerate(resolved):
                if d.get("kind") in ("wildcard", "binding"):
                    if i + 1 < len(resolved):
                        advisories.append(
                            f"Unreachable match arm: arm {i + 2} follows an "
                            f"irrefutable arm (arm {i + 1}) at node {node.node_id}")
                    break

        if any(d.get("kind") in ("wildcard", "binding") for d in resolved):
            return  # a catch-all arm makes any match exhaustive
        if any(_has_unknown_pattern(d) for d in resolved):
            return  # opaque pattern somewhere: stay permissive

        ctor_arms = [d for d in resolved if d.get("kind") == "ctor"]
        if ctor_arms:
            # All ctor arms must resolve to one known enum; otherwise the
            # scrutinee's enum type is not reliably known here (or the
            # program has a type error reported elsewhere) — skip.
            enums: set[str] = set()
            for d in ctor_arms:
                enum_name = (d.get("enum") or variant_to_enum.get(d.get("name"))
                             or _BUILTIN_VARIANT_ENUM.get(d.get("name")))
                if not isinstance(enum_name, str):
                    return
                enums.add(enum_name)
            if len(enums) != 1:
                return
            enum_name = next(iter(enums))
            edef = enum_defs.get(enum_name)
            if edef is not None:
                all_variants = [v.get("name") for v in edef.get("variants") or []
                                if isinstance(v, dict) and isinstance(v.get("name"), str)]
            elif enum_name in _BUILTIN_ENUM_VARIANTS:
                all_variants = list(_BUILTIN_ENUM_VARIANTS[enum_name])
            else:
                return
            covered: set[str] = set()
            for d in ctor_arms:
                subs = [_resolve_pattern(s) for s in d.get("subpatterns") or []]
                if all(s.get("kind") in ("wildcard", "binding") for s in subs):
                    covered.add(d.get("name"))
            missing = [v for v in all_variants if v not in covered]
            if missing:
                _type_error(
                    "non-exhaustive match: missing variants "
                    + ", ".join(missing), node, kind="type-nonexhaustive-match")
            return

        literal_arms = [d for d in resolved if d.get("kind") == "literal"]
        if not literal_arms or len(literal_arms) != len(resolved):
            return
        classes = {literal_class(d.get("value")) for d in literal_arms}
        if len(classes) != 1:
            return  # mixed/unclassifiable literal arms: conflict reported elsewhere
        cls = next(iter(classes))
        if cls == "Bool":
            seen = {d.get("value") for d in literal_arms}
            missing_bools = [spelling for spelling, v in
                             (("true", True), ("false", False)) if v not in seen]
            if missing_bools:
                _type_error(
                    "non-exhaustive match: missing cases "
                    + ", ".join(missing_bools), node,
                    kind="type-nonexhaustive-match")
        elif cls in ("Int", "String", "Float"):
            _type_error(
                f"non-exhaustive match: {cls} literal patterns can never be "
                "exhaustive; add a wildcard or binding arm", node,
                kind="type-nonexhaustive-match")
        return

    # Parallel to push_scope/pop_scope: per-scope saves of
    # global_struct_bindings entries that declarations in the scope popped
    # or overwrote, restored at scope exit (a shadow in an inner block must
    # not disable gating of the outer @global container afterwards).
    gsb_saves: list[dict[str, str | None]] = []

    def push_scope() -> None:
        scopes.append({})
        gsb_saves.append({})
        borrow_checker.enter_scope()

    def pop_scope() -> None:
        scopes.pop()
        borrow_checker.exit_scope()
        for name, prev in (gsb_saves.pop() if gsb_saves else {}).items():
            if prev is None:
                global_struct_bindings.pop(name, None)
            else:
                global_struct_bindings[name] = prev

    def walk(node: Any) -> None:
        node_ty = types.get(node.node_id)
        children = getattr(node, "children", ())
        kind = getattr(node, "kind", None)
        # Record effect class from EffectDeclaration
        if kind == "EffectDeclaration":
            value = payload_dict(node)
            effect_name = value.get("name")
            effect_class = value.get("effect_class")
            if effect_name and effect_class:
                effect_classes[effect_name] = effect_class
        if kind == "HandleEffect":
            value = payload_dict(node)
            effect_name = value.get("effect_name")
            effect_class = effect_classes.get(effect_name)
            if effect_name:
                # Effects without a declared class default to "suspend" (the
                # general case); the handler context must exist either way so
                # lexically-enclosed performs are recognized as handled.
                handler_contexts.append({effect_name: effect_class or "suspend"})
                handler_locals.append(set())
                handler_operations.append([])
            # Walk handler children
            for child in children:
                walk(child)
            # Pop handler context after processing
            if handler_contexts:
                handler_contexts.pop()
            if handler_locals:
                handler_locals.pop()
            if handler_operations:
                handler_operations.pop()
            return None
        # A perform must be either lexically inside a handle for its effect
        # or covered by the enclosing function's `performs` clause. Advisory
        # diagnostic (checker channel), not a hard error: dynamically-scoped
        # handlers installed by callers are legitimate and undecidable here.
        if kind == "PerformEffect":
            effect_ref = payload_dict(node).get("effect_name")
            if isinstance(effect_ref, str) and effect_ref:
                effect = effect_ref.split(".")[0]
                handled = any(effect in ctx_frame for ctx_frame in handler_contexts)
                declared = any(effect in ps for ps in enclosing_performs)
                if not handled and not declared:
                    simplesub.add_unresolved(
                        "effect (performed without enclosing handler or "
                        "performs declaration)", effect, node.node_id)
        # Track Resume calls to check effect class restrictions
        if kind == "Resume":
            # Check if we're in a handler context
            if handler_contexts:
                current_handler = handler_contexts[-1]
                for effect_name, effect_class in current_handler.items():
                    # Resume is always allowed in handlers (it's how you return)
                    # But for stack effects, it should be immediate, not stored
                    if effect_class == "stack":
                        # For stack effects, resume should be called immediately
                        # If we're tracking continuation assignments, check if this resume
                        # is using a stored continuation vs the implicit one
                        pass
        # Track assignments to detect continuation storage
        if kind == "Assignment":
            # Check if we're in a handler context
            if handler_contexts:
                # Track this assignment in the current handler
                if handler_locals:
                    value = payload_dict(node)
                    var_name = value.get("name")
                    if isinstance(var_name, str):
                        handler_locals[-1].add(var_name)
        # Track function calls to detect continuation escape
        if kind == "FunctionCall" and node_ty is not None:
            # Check if we're in a handler context
            if handler_contexts:
                # Track this operation in the current handler
                if handler_operations:
                    handler_operations[-1].append("FunctionCall")
                current_handler = handler_contexts[-1]
                for effect_name, effect_class in current_handler.items():
                    if effect_class == "stack":
                        # Check if any argument is a variable that was assigned in this handler
                        for child in children:
                            if child.kind == "Variable":
                                var_name = payload_dict(child).get("name")
                                if isinstance(var_name, str) and handler_locals and var_name in handler_locals[-1]:
                                    borrow_checker.errors.append(
                                        BorrowError(
                                            message=f"Stack effect handler '{effect_name}' cannot pass handler-local variable '{var_name}' to function (potential continuation escape)",
                                            node_id=child.node_id,
                                            kind="stack-continuation-escape",
                                            variable=var_name,
                                        )
                                    )
        if kind == "Literal" and node_ty is not None:
            cls = literal_class(getattr(node, "value", None))
            if cls is not None:
                simplesub.add_class_constraint(cls, [node_ty], node.node_id)
        if kind == "MatchExpression":
            _check_match_exhaustiveness(node)
        if kind == "Block":
            push_scope()
            borrow_checker.enter_region()
            last_child_ty = None
            for child in children:
                child_ty = types.get(child.node_id)
                if node_ty is not None and child_ty is not None:
                    simplesub.add_subtype(child_ty, node_ty)
                walk(child)
                last_child_ty = child_ty
            if kind == "Block" and node_ty is not None and last_child_ty is not None:
                simplesub.add_unify(node_ty, last_child_ty)
            borrow_checker.exit_region()
            pop_scope()
            return None
        if kind == "FunctionDeclaration" and node_ty is not None:
            # Record effects performed by this function
            value = payload_dict(node)
            func_name = value.get("name")
            performs = value.get("performs", []) or []
            if func_name and isinstance(func_name, str):
                function_effects[func_name] = [str(e) for e in performs]
            params = value.get("params", [])
            param_children = param_nodes(children)
            param_tys: list[Any] = [
                types[child.node_id] for child in param_children if types.get(child.node_id) is not None
            ]

            # Construct CompactType function type first so we can bind the
            # function name to the function type in the *enclosing* scope.
            if CompactType is not None:
                from metaxu.type_defs import next_id
                fn_compact = CompactType(
                    id=next_id(),
                    kind='function',
                    param_types=param_tys,
                    return_type=node_ty,
                    linearity="many"
                )
                types[node.node_id] = fn_compact
                simplesub.function_types[node.node_id] = fn_compact
                simplesub.add_function_type(fn_compact, param_tys, node_ty, "many", node.node_id)
                for effect_name in value.get("performs", []) or []:
                    simplesub.add_effect(fn_compact, str(effect_name), node.node_id)
                    simplesub.add_class_constraint("Effectful", [fn_compact], node.node_id)
                # Bind function name to function type (not return type)
                bind(payload_name(node), fn_compact)
            else:
                simplesub.add_function_type(node_ty, param_tys, node_ty, "many", node.node_id)
                for effect_name in value.get("performs", []) or []:
                    simplesub.add_effect(node_ty, str(effect_name), node.node_id)
                    simplesub.add_class_constraint("Effectful", [node_ty], node.node_id)
                bind(payload_name(node), node_ty)

            # Function body is a fresh scope *and* a fresh region: locals
            # (and @local params) belong to the function's region and may not
            # escape to the caller's region except through exclave. Borrow
            # state is per-function: moves in one function must not poison
            # same-named bindings in another.
            push_scope()
            fn_state = borrow_checker.enter_function_state()
            # @global-container gating is per-function: bindings recorded in
            # one function must not poison same-named locals in another.
            saved_global_bindings = dict(global_struct_bindings)
            enclosing_performs.append(frozenset(str(e) for e in performs))
            borrow_checker.enter_region()
            function_region_stack.append(borrow_checker.current_region())

            for name, child in zip(params, param_children):
                child_ty = types.get(child.node_id)
                if child_ty is not None:
                    bind(name, child_ty)
                    simplesub.add_class_constraint("Param", [child_ty], child.node_id)
                    param_payload = payload_dict(child)
                    mode = param_payload.get("mode")
                    if isinstance(mode, str):
                        simplesub.add_class_constraint(f"Mode:{mode}", [child_ty], child.node_id)
                    # Declare parameter with its real uniqueness/locality modes
                    # (defaults: shared/global for unannotated parameters).
                    uniqueness, locality, _linearity = _split_mode(param_payload)
                    if isinstance(name, str):
                        borrow_checker.declare_variable(name, uniqueness, locality, child.node_id)

            # Push the return type (not the function type) for return statements
            # Must do this BEFORE walking the body
            return_types.append(node_ty)

            for child in non_param_nodes(children):
                child_ty = types.get(child.node_id)
                # Subtype from child to function's return type
                if CompactType is not None and types.get(node.node_id) is not None:
                    fn_ty = types[node.node_id]
                    if hasattr(fn_ty, 'return_type') and fn_ty.return_type is not None and child_ty is not None:
                        simplesub.add_subtype(child_ty, fn_ty.return_type)
                elif node_ty is not None and child_ty is not None:
                    simplesub.add_subtype(child_ty, node_ty)
                walk(child)
            return_types.pop()
            function_region_stack.pop()
            borrow_checker.exit_region()
            # Pop the function's scope BEFORE restoring the enclosing borrow
            # state: exit_scope releases borrows for names declared in the
            # function, and doing that after the restore would erase live
            # outer borrows of same-named variables (e.g. a param named like
            # an outer let that holds a reference).
            pop_scope()
            borrow_checker.exit_function_state(fn_state)
            global_struct_bindings.clear()
            global_struct_bindings.update(saved_global_bindings)
            enclosing_performs.pop()
            return None
        if kind == "LambdaExpression" and node_ty is not None:
            outer_bindings = {name: lookup(name) for name in payload_dict(node).get("captures", {})}
            push_scope()
            borrow_checker.enter_region()
            function_region_stack.append(borrow_checker.current_region())
            value = payload_dict(node)
            params = value.get("params", [])
            param_children = param_nodes(children)
            param_tys: list[Any] = []
            for name, child in zip(params, param_children):
                child_ty = types.get(child.node_id)
                if child_ty is not None:
                    bind(name, child_ty)
                    param_tys.append(child_ty)
                    simplesub.add_class_constraint("Param", [child_ty], child.node_id)
                    param_payload = payload_dict(child)
                    mode = param_payload.get("mode")
                    if isinstance(mode, str):
                        simplesub.add_class_constraint(f"Mode:{mode}", [child_ty], child.node_id)
                    # Declare parameter with its real uniqueness/locality modes
                    uniqueness, locality, _linearity = _split_mode(param_payload)
                    if isinstance(name, str):
                        borrow_checker.declare_variable(name, uniqueness, locality, child.node_id)
            linearity = value.get("linearity") or "many"
            captures = value.get("captures", {}) or {}
            for captured_name, mode in captures.items():
                captured_ty = outer_bindings.get(captured_name)
                if captured_ty is not None:
                    simplesub.add_capture(node_ty, str(captured_name), captured_ty, str(mode), node.node_id)
            if any(mode == "borrow_mut" for mode in captures.values()) and linearity == "many":
                linearity = "separate"
            # Construct CompactType function type for lambda
            # Store in types dict for proper representation
            if CompactType is not None:
                from metaxu.type_defs import next_id
                fn_compact = CompactType(
                    id=next_id(),
                    kind='function',
                    param_types=param_tys,
                    return_type=node_ty,
                    linearity=str(linearity)
                )
                types[node.node_id] = fn_compact
                simplesub.function_types[node.node_id] = fn_compact
                simplesub.add_function_type(fn_compact, param_tys, node_ty, str(linearity), node.node_id)
                simplesub.add_linearity(fn_compact, str(linearity), node.node_id)
            else:
                simplesub.add_function_type(node_ty, param_tys, node_ty, str(linearity), node.node_id)
                simplesub.add_linearity(node_ty, str(linearity), node.node_id)
            
            # Push the return type (not the function type) for return statements
            # Must do this BEFORE walking the body
            return_types.append(node_ty)
            
            for child in non_param_nodes(children):
                child_ty = types.get(child.node_id)
                # Subtype from child to function's return type
                if CompactType is not None and types.get(node.node_id) is not None:
                    fn_ty = types[node.node_id]
                    if hasattr(fn_ty, 'return_type') and fn_ty.return_type is not None and child_ty is not None:
                        simplesub.add_subtype(child_ty, fn_ty.return_type)
                elif node_ty is not None and child_ty is not None:
                    simplesub.add_subtype(child_ty, node_ty)
                walk(child)
            return_types.pop()
            function_region_stack.pop()
            borrow_checker.exit_region()
            pop_scope()
            if CompactType is not None:
                simplesub.add_class_constraint("Callable", [fn_compact], node.node_id)
            else:
                simplesub.add_class_constraint("Callable", [node_ty], node.node_id)
            return None
        if kind == "LetBinding" and node_ty is not None:
            for child in children:
                child_ty = types.get(child.node_id)
                if child_ty is not None:
                    simplesub.add_unify(node_ty, child_ty)
                    walk(child)
            # Declare variable in borrow checker with its real modes
            # (defaults to shared/global for unannotated bindings).
            var_name = payload_name(node)
            let_payload = payload_dict(node)
            uniqueness, locality, linearity = _split_mode(let_payload)
            if isinstance(var_name, str):
                borrow_checker.declare_variable(var_name, uniqueness, locality, node.node_id)
                # A rebinding of the name is a fresh binding: it must not
                # inherit @global-container gating from an earlier binding.
                # Save the outer entry so scope exit restores it.
                if gsb_saves:
                    gsb_saves[-1].setdefault(
                        var_name, global_struct_bindings.get(var_name))
                global_struct_bindings.pop(var_name, None)
                # Deep ownership: an *explicitly* @global struct binding must
                # not (transitively) contain @local fields nor store @local
                # values in its initializer. Structs default to local
                # allocation, so only spelled-out @global bindings are gated.
                if _explicit_locality(let_payload.get("mode")) == "global":
                    for child in children:
                        if child.kind == "StructInstantiation":
                            _check_global_struct_binding(var_name, child)
                # Register callable linearity when binding a lambda so calls can
                # enforce once/separate semantics by name.
                for child in children:
                    if child.kind == "LambdaExpression":
                        lam_linearity = payload_dict(child).get("linearity")
                        if isinstance(lam_linearity, str):
                            declared_linearity[var_name] = lam_linearity
                        elif isinstance(linearity, str):
                            declared_linearity[var_name] = linearity
                if isinstance(linearity, str) and var_name not in declared_linearity:
                    declared_linearity[var_name] = linearity
                # Track reference relationships created by borrow initializers,
                # e.g. `let r = &x` makes r hold a reference to x.
                for child in children:
                    ref_mode = _BORROW_NODE_MODES.get(child.kind)
                    if ref_mode is not None:
                        target = payload_dict(child).get("variable")
                        if isinstance(target, str):
                            borrow_checker.track_reference(var_name, target, ref_mode)
                            borrow_checker.check_reference_conflicts(target, child.node_id)
            bind(payload_name(node), node_ty)
            return None
        if kind == "Variable" and node_ty is not None:
            name = payload_name(node)
            binding_ty = lookup(name)
            if binding_ty is not None:
                simplesub.add_unify(node_ty, binding_ty)
            elif isinstance(name, str):
                simplesub.add_unresolved("variable", name, node.node_id)
            # Check variable use in borrow checker
            if isinstance(name, str):
                borrow_checker.check_variable_use(name, node.node_id)
        if kind in ("FunctionCall", "QualifiedFunctionCall"):
            # Parametric instantiation checking: explicit type args are
            # substituted, omitted ones inferred call-site-locally.
            # QualifiedFunctionCall covers module-qualified generic calls
            # (`mod.f<Int>(x)`): the module system renames `mod.f` to a
            # dotted function name, which fn_defs knows post-rename.
            # Method-call shapes (`recv.m(...)`) have no matching fn_defs
            # entry and fall through unchecked, exactly as before.
            _check_call_types(node)
        if kind == "FunctionCall" and node_ty is not None:
            name = payload_name(node)
            callee_ty = lookup(name)
            arg_tys = [types.get(child.node_id) for child in children if types.get(child.node_id) is not None]
            if callee_ty is not None:
                simplesub.add_class_constraint("Callable", [callee_ty, node_ty], node.node_id)
                simplesub.add_call(callee_ty, arg_tys, node_ty, node.node_id)
                simplesub.add_class_constraint("CallLinearity", [callee_ty, node_ty], node.node_id)
            elif isinstance(name, str):
                simplesub.add_unresolved("callee", name, node.node_id)
            # Enforce linearity for callables with a known linearity mode
            # (e.g. once-lambdas bound to a name may only be invoked once).
            if isinstance(name, str) and name in declared_linearity:
                borrow_checker.check_linearity(name, declared_linearity[name], node.node_id)
            for child in children:
                child_ty = types.get(child.node_id)
                if child_ty is not None:
                    simplesub.add_class_constraint("Arg", [node_ty, child_ty], child.node_id)
                    # Check locality of argument if it's a variable
                    if child.kind == "Variable":
                        var_name = payload_dict(child).get("name")
                        if isinstance(var_name, str):
                            borrow_checker.check_locality(var_name, None, child.node_id)
                            # @local values may not cross a suspension point:
                            # the frame they live in can unwind before the
                            # continuation resumes. Only suspend-class effects
                            # are affected; stack-class effects behave like
                            # ordinary calls. Effects with no declared class
                            # are conservatively treated as suspend.
                            var_info = borrow_checker.variables.get(var_name)
                            if (
                                var_info is not None
                                and var_info.locality == "local"
                                and isinstance(name, str)
                                and name in function_effects
                            ):
                                for effect in function_effects[name]:
                                    effect_class = effect_classes.get(effect, "suspend")
                                    if effect_class == "suspend":
                                        borrow_checker.errors.append(
                                            BorrowError(
                                                message=f"Local variable '{var_name}' passed to suspend effect '{effect}'",
                                                node_id=child.node_id,
                                                kind="suspend-local",
                                                variable=var_name,
                                            )
                                        )
        if kind in {"BinaryOperation", "ComparisonExpression"} and node_ty is not None and len(children) >= 2:
            left_ty = types.get(children[0].node_id)
            right_ty = types.get(children[1].node_id)
            if left_ty is not None and right_ty is not None:
                simplesub.add_unify(left_ty, right_ty)
                if kind == "BinaryOperation":
                    simplesub.add_unify(node_ty, left_ty)
                else:
                    simplesub.add_class_constraint("Bool", [node_ty], node.node_id)
                    if payload_operator(node) in {"<", "<=", ">", ">="}:
                        simplesub.add_class_constraint("Ord", [left_ty], node.node_id)
                    if payload_operator(node) in {"==", "!="}:
                        simplesub.add_class_constraint("Eq", [left_ty], node.node_id)
                if kind == "BinaryOperation" and payload_operator(node) in {"+", "-", "*", "/", "%"}:
                    simplesub.add_class_constraint("Number", [node_ty], node.node_id)
        if kind in {"IfStatement", "IfExpression"} and node_ty is not None and children:
            cond_ty = types.get(children[0].node_id)
            if cond_ty is not None:
                simplesub.add_class_constraint("Bool", [cond_ty], children[0].node_id)
            branch_types = [types.get(child.node_id) for child in children[1:] if types.get(child.node_id) is not None]
            for branch_ty in branch_types:
                simplesub.add_unify(node_ty, branch_ty)
            if len(branch_types) >= 2:
                simplesub.add_unify(branch_types[0], branch_types[1])
        if kind == "WhileStatement" and node_ty is not None and children:
            cond_ty = types.get(children[0].node_id)
            if cond_ty is not None:
                simplesub.add_class_constraint("Bool", [cond_ty], children[0].node_id)
            simplesub.add_class_constraint("Unit", [node_ty], node.node_id)
        if kind == "ReturnStatement" and node_ty is not None:
            if children:
                expr_ty = types.get(children[0].node_id)
                if expr_ty is not None:
                    # return_types now contains the actual return type
                    target_ty = return_types[-1] if return_types else node_ty
                    simplesub.add_unify(target_ty, expr_ty)
                    if return_types:
                        simplesub.add_unify(return_types[-1], expr_ty)
                    # Returning a value sends it to the *caller's* region. A
                    # @local value declared inside this function (params or
                    # body) therefore escapes -> error. Exclave is the legal
                    # way to promote a local value and is handled at the
                    # ExclaveExpression node instead.
                    if children[0].kind == "Variable":
                        var_name = payload_dict(children[0]).get("name")
                        if isinstance(var_name, str):
                            caller_region = (
                                function_region_stack[-1] - 1
                                if function_region_stack
                                else borrow_checker.current_region()
                            )
                            borrow_checker.check_locality(var_name, caller_region, children[0].node_id)
            elif return_types:
                simplesub.add_class_constraint("Unit", [return_types[-1]], node.node_id)
        if kind == "Assignment" and node_ty is not None:
            target_name = payload_name(node)
            # Deep ownership: assigning a @local-bound value into a field of a
            # @global-bound struct is an error (global containers cannot store
            # locals). Dotted targets arrive stringified, e.g. "g.field".
            if isinstance(target_name, str) and "." in target_name:
                base_name, _dot, field_path = target_name.partition(".")
                container_struct = global_struct_bindings.get(base_name)
                if container_struct is not None:
                    for child in children:
                        if getattr(child, "kind", None) != "Variable":
                            continue
                        value_name = payload_dict(child).get("name")
                        if _is_local_variable(value_name):
                            borrow_checker.errors.append(BorrowError(
                                message=(
                                    f"cannot store @local value '{value_name}' in "
                                    f"field '{field_path}' of @global {container_struct} "
                                    f"binding '{base_name}'; global containers cannot "
                                    f"store locals"
                                ),
                                node_id=child.node_id,
                                kind="deep-locality",
                                variable=value_name,
                            ))
            binding_ty = lookup(target_name)
            if binding_ty is not None:
                simplesub.add_unify(node_ty, binding_ty)
            for child in children:
                child_ty = types.get(child.node_id)
                if child_ty is not None:
                    if binding_ty is not None:
                        simplesub.add_unify(binding_ty, child_ty)
                    simplesub.add_class_constraint("Unit", [node_ty], node.node_id)
                    # Check locality if assigning a variable
                    if child.kind == "Variable":
                        var_name = payload_dict(child).get("name")
                        if isinstance(var_name, str):
                            borrow_checker.check_locality(var_name, None, child.node_id)
                # Track references created by borrow assignments, e.g.
                # `r = &x` makes r hold a reference to x.
                ref_mode = _BORROW_NODE_MODES.get(child.kind)
                if ref_mode is not None and isinstance(target_name, str):
                    ref_target = payload_dict(child).get("variable")
                    if isinstance(ref_target, str):
                        borrow_checker.track_reference(target_name, ref_target, ref_mode)
                        borrow_checker.check_reference_conflicts(ref_target, child.node_id)
        if kind == "BorrowShared" and node_ty is not None:
            value = payload_dict(node)
            var_name = value.get("variable")
            if isinstance(var_name, str):
                borrow_checker.check_borrow_shared(var_name, node.node_id)
                simplesub.add_class_constraint("BorrowShared", [node_ty], node.node_id)
        if kind == "BorrowUnique" and node_ty is not None:
            value = payload_dict(node)
            var_name = value.get("variable")
            if isinstance(var_name, str):
                # Surface `&mut x` parses to BorrowUnique, but per
                # docs/ownership_and_borrowing.md it is an exclusive
                # *reference*: aliasing-XOR-mutation while live, and the
                # source stays valid once the borrow ends. Ownership
                # transfer only happens via moves, not `&mut`.
                borrow_checker.check_borrow_exclusive(var_name, node.node_id)
                simplesub.add_class_constraint("BorrowUnique", [node_ty], node.node_id)
        if kind == "BorrowExclusive" and node_ty is not None:
            value = payload_dict(node)
            var_name = value.get("variable")
            if isinstance(var_name, str):
                borrow_checker.check_borrow_exclusive(var_name, node.node_id)
                simplesub.add_class_constraint("BorrowExclusive", [node_ty], node.node_id)
        if kind == "Move" and node_ty is not None:
            value = payload_dict(node)
            var_name = value.get("variable")
            if isinstance(var_name, str):
                borrow_checker.check_move(var_name, node.node_id)
                simplesub.add_class_constraint("Move", [node_ty], node.node_id)
        if kind == "ExclaveExpression" and node_ty is not None:
            # Exclave uses copy semantics - the value is copied to the caller's
            # frame, so it is the *legal* way for a @local value to escape.
            # The only invalid case is exclaving an already-moved value.
            inner_var = payload_dict(node).get("expression")
            if not isinstance(inner_var, str):
                inner = next((c for c in children if c.kind == "Variable"), None)
                inner_var = payload_dict(inner).get("name") if inner is not None else None
            if isinstance(inner_var, str):
                borrow_checker.check_exclave(inner_var, node.node_id)
            simplesub.add_class_constraint("Exclave", [node_ty], node.node_id)
        if kind == "StructDefinition":
            name = payload_name(node)
            if isinstance(name, str):
                struct_defs[name] = payload_dict(node)
        if kind == "EnumDefinition":
            name = payload_name(node)
            if isinstance(name, str):
                enum_defs[name] = payload_dict(node)
        if kind == "StructInstantiation" and node_ty is not None:
            simplesub.add_class_constraint("Struct", [node_ty], node.node_id)
            name = payload_name(node)
            if isinstance(name, str):
                simplesub.add_class_constraint(f"Struct:{name}", [node_ty], node.node_id)
                _check_struct_field_types(node, name)
        if kind == "StructField" and node_ty is not None:
            for child in children:
                child_ty = types.get(child.node_id)
                if child_ty is not None:
                    simplesub.add_unify(node_ty, child_ty)
            name = payload_name(node)
            if isinstance(name, str):
                simplesub.add_class_constraint(f"Field:{name}", [node_ty], node.node_id)
        if kind == "FieldAccess" and node_ty is not None and children:
            base_ty = types.get(children[0].node_id)
            if base_ty is not None:
                simplesub.add_class_constraint("HasField", [base_ty, node_ty], node.node_id)
        for child in children:
            child_ty = types.get(child.node_id)
            if node_ty is not None and child_ty is not None:
                simplesub.add_subtype(child_ty, node_ty)
            walk(child)

        if kind == "FunctionCall":
            # Borrows taken directly in argument position (`f(&mut x)`) live
            # only for the duration of the call; release them so `x` is
            # borrowable again afterwards. release_call_borrow keeps the state
            # when a live named reference (`let r = &mut x`) still holds a
            # borrow of the same variable.
            for child in children:
                mode = _BORROW_NODE_MODES.get(child.kind)
                if mode is not None:
                    arg_var = payload_dict(child).get("variable")
                    if isinstance(arg_var, str):
                        borrow_checker.release_call_borrow(arg_var, mode)

    _collect_definitions(frozen_root)
    walk(frozen_root)
    # Publish declared effect classes so the constraint checker can
    # distinguish stack-class effects (which do not suspend) from
    # suspend-class effects. Effects with no declared class are treated as
    # suspend-class (conservative) by the checker.
    if hasattr(simplesub, "add_effect_class"):
        for effect_name, effect_class in effect_classes.items():
            simplesub.add_effect_class(effect_name, effect_class)
    return None, borrow_checker.get_errors()
