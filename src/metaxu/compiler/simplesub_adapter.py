from __future__ import annotations

from typing import Any, Dict

try:
    # Optional import: we will adapt to available APIs.
    from metaxu.simplesub import TypeInferencer as _SSInferencer, Polarity as _Polarity  # type: ignore
except Exception:  # pragma: no cover
    _SSInferencer = None  # type: ignore
    _Polarity = None  # type: ignore


class SimpleSubFacade:
    """A thin façade for emitting constraints against the existing SimpleSub.

    This is a scaffold interface so we can plug frozen-AST constraints into
    the original inferencer without mutating the original AST. For now, if
    SimpleSub internals are not available, this class becomes a no-op holder
    to keep the pipeline functional.

    Biunification (principal types): `principal_type_of(node_id)` /
    `principal_types()` replay the buffered constraint stream through the
    real biunification engine (metaxu.simplesub.Biunifier) and coalesce the
    node's type into a principal type string (`'a -> 'a`, `Int ∨ String`,
    `μt. ...`). The replay is LAZY and side-effect free: it runs only when
    asked, keeps every bound in the engine's side tables, and never mutates
    the shared CompactTypes, so the compile path (solve() -> HIR) is
    byte-identical whether or not anyone queries principal types. Residual
    risk of turning eager write-back on has NOT been taken: applying
    coalesced types to `self.types` stays behind the pre-existing
    `_apply_solution` mechanism and is off by default, because the
    downstream HIR builder reads CompactTypes (not PTypes) and the
    engine's unions/intersections have no CompactType encoding yet.
    Hard `type-conflict` diagnostics continue to come exclusively from
    `_detect_class_conflicts` below; the engine's `BiunifyError` records
    are exposed as `biunify_errors()` for tooling and tests only.
    """

    def __init__(self, types: Dict[int, Any]) -> None:
        self.types = types
        # If the real inferencer is available, instantiate it lazily as needed.
        self._ss = _SSInferencer() if _SSInferencer is not None else None
        # Placeholder constraint buffer if we want to capture and replay later
        self._constraints: list[tuple] = []
        self.errors: list[str] = []
        # Warning-level advisories (plain strings) surfaced through the same
        # -1 diagnostics channel as "Unresolved ..." messages. They carry no
        # `kind` attribute, so the pipeline never promotes them to hard
        # TypeCheckErrors (only kind == "type-conflict" items are promoted).
        self.advisories: list[str] = []
        self.effect_info: Any = None
        self.function_types: Dict[int, Any] = {}  # Store CompactType function types by node_id
        self._apply_solution = False  # Flag to control whether to apply SimpleSub's solution
        self._biunify_engine: Any = None  # Lazily-built Biunifier (see biunify())

    # --- Constraint emission API (scaffold) ---
    def add_unify(self, a: Any, b: Any, variance: str = "invariant") -> None:
        if self._ss is not None:
            # Ideally the inferencer exposes an add_unify; otherwise buffer.
            self._constraints.append(("unify", a, b, variance))
        else:
            self._constraints.append(("unify", a, b, variance))

    def add_class_constraint(self, cls: str, args: list[Any], node_id: int | None = None) -> None:
        self._constraints.append(("class", cls, tuple(args), node_id))

    def add_subtype(self, a: Any, b: Any) -> None:
        self._constraints.append(("subtype", a, b))

    def add_function_type(
        self,
        fn_ty: Any,
        param_tys: list[Any],
        ret_ty: Any,
        linearity: str = "many",
        node_id: int | None = None,
    ) -> None:
        self._constraints.append(("function", fn_ty, tuple(param_tys), ret_ty, linearity, node_id))

    def add_effect(self, fn_ty: Any, effect_name: str, node_id: int | None = None) -> None:
        self._constraints.append(("effect", fn_ty, effect_name, node_id))

    def add_effect_class(self, effect_name: str, effect_class: str, node_id: int | None = None) -> None:
        """Record the declared class ("stack" or "suspend") of an effect."""
        self._constraints.append(("effect_class", effect_name, effect_class, node_id))

    def add_capture(
        self,
        fn_ty: Any,
        captured_name: str,
        captured_ty: Any,
        mode: str,
        node_id: int | None = None,
    ) -> None:
        self._constraints.append(("capture", fn_ty, captured_name, captured_ty, mode, node_id))

    def add_call(self, callee_ty: Any, arg_tys: list[Any], result_ty: Any, node_id: int | None = None) -> None:
        self._constraints.append(("call", callee_ty, tuple(arg_tys), result_ty, node_id))

    def add_linearity(self, fn_ty: Any, linearity: str, node_id: int | None = None) -> None:
        self._constraints.append(("linearity", fn_ty, linearity, node_id))

    def add_unresolved(self, kind: str, name: str, node_id: int | None = None) -> None:
        self._constraints.append(("unresolved", kind, name, node_id))

    @property
    def constraints(self) -> tuple[tuple, ...]:
        return tuple(self._constraints)

    def enable_solution_application(self) -> None:
        """Enable applying SimpleSub's solution back to the types dict after solving."""
        self._apply_solution = True

    def get_solved_bounds(self, node_id: int) -> Any | None:
        """Get the solved type bounds for a given node_id after solving.
        
        Arguments:
            node_id: Node ID to get bounds for
            
        Returns:
            The TypeBounds object for the node's type, or None if not found
        """
        ty = self.types.get(node_id)
        if ty is not None and hasattr(ty, 'bounds'):
            return ty.bounds
        return None

    # Literal classes that are mutually exclusive: one value cannot be, say,
    # both an Int and a String. Used for constraint-graph conflict detection.
    _LITERAL_CLASSES = frozenset({"Int", "String", "Bool", "Float"})

    class TypeConflict:
        """Structured hard type error from constraint-graph conflict
        detection. Carries kind="type-conflict" so the pipeline can promote
        it to TypeCheckError without string matching, and `node_id` so
        frozen_borrow_checker.locate_errors can give it a source position."""
        kind = "type-conflict"

        def __init__(self, message: str, node_id: int | None = None) -> None:
            self.message = message
            self.node_id = node_id
            self.location = None

        def __str__(self) -> str:
            if self.location is not None:
                from metaxu.errors import format_location
                return f"{format_location(self.location)}: {self.message}"
            where = f" at node {self.node_id}" if self.node_id is not None else ""
            return f"{self.message}{where}"

        def excerpt(self):
            if self.location is None:
                return None
            from metaxu.errors import source_excerpt
            return source_excerpt(self.location)

    def _detect_class_conflicts(self) -> list[str]:
        """Union type vars along *unify* edges and flag components that carry
        contradictory literal classes (e.g. `1 + \"a\"` unifies an Int-classed
        var with a String-classed var).

        Subtype edges are deliberately NOT merged: they are directional
        (every statement's type flows into its block's type), and merging
        them would conflate unrelated values.
        """
        parent: dict[int, int] = {}

        def find(x: int) -> int:
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        def key(ty: Any) -> int | None:
            tid = getattr(ty, "id", None)
            return tid if isinstance(tid, int) else None

        classes: dict[int, set[str]] = {}
        nodes: dict[int, int] = {}
        for c in self._constraints:
            if c[0] == "class" and c[1] in self._LITERAL_CLASSES:
                _, cls, args, node_id = c
                for ty in args:
                    k = key(ty)
                    if k is not None:
                        classes.setdefault(k, set()).add(cls)
                        if isinstance(node_id, int):
                            nodes.setdefault(k, node_id)
            elif c[0] == "unify":
                ka, kb = key(c[1]), key(c[2])
                if ka is not None and kb is not None:
                    union(ka, kb)

        merged: dict[int, set[str]] = {}
        rep_node: dict[int, int] = {}
        for k, cls_set in classes.items():
            rep = find(k)
            merged.setdefault(rep, set()).update(cls_set)
            if k in nodes:
                # Report the LAST of the conflicting values. Frozen node ids
                # are assigned in source order, so the highest id is the value
                # that made the conflict apparent (`a + "s"` points at the
                # string, not at the `let a = 1` that typed `a` as Int).
                rep_node[rep] = max(rep_node.get(rep, -1), nodes[k])
        errors = []
        for rep, cls_set in merged.items():
            if len(cls_set) > 1:
                errors.append(self.TypeConflict(
                    "type mismatch: one value is required to be "
                    + " and ".join(sorted(cls_set)),
                    node_id=rep_node.get(rep),
                ))
        return errors

    # --- Biunification: principal types (lazy, advisory) ---
    def biunify(self) -> Any:
        """Build (once) and return a Biunifier fed with this façade's
        constraint stream, translated into subtype constraints:

        - `unify a b`           -> a <: b and b <: a
        - `subtype a b`         -> a <: b
        - literal `class C [t]` -> C <: t and t <: C for C in
          {Int, String, Bool, Float} (the same flat, mutually-exclusive
          set `_detect_class_conflicts` uses -- so the engine sees exactly
          the concrete types the conflict detector reasons about, and the
          class-only constraints like `Unit`/`Callable` stay out, keeping
          the two reporters consistent)
        - `call callee args r`  -> callee <: (args) -> r

        Runs lazily on first use and touches only the engine's side
        tables, never the shared CompactTypes, so compiling is unaffected.

        Precision caveat: the emitter's stream contains coarse
        statement-flow edges (every statement's type flows into its
        block's / function-return's type; conditions flow into their
        statement nodes) that exist for effect propagation, not typing.
        The engine renders them faithfully, so principal types are exact
        on expression chains (literals, operators, calls, let/return)
        but over-approximate — extra union/intersection members — on
        statement-heavy functions. Sharpening those edges is emitter
        work, deliberately not done here because the same tuples drive
        effect propagation in frozen_constraint_checker.
        """
        if self._biunify_engine is not None:
            return self._biunify_engine
        from metaxu.simplesub import Biunifier
        from metaxu.type_defs import CompactType, next_id

        def is_ct(x: Any) -> bool:
            return isinstance(x, CompactType)

        eng = Biunifier()
        for c in self._constraints:
            tag = c[0] if c else None
            if tag == "unify":
                _, a, b, _variance = c
                if is_ct(a) and is_ct(b):
                    eng.constrain(a, b)
                    eng.constrain(b, a)
            elif tag == "subtype":
                _, a, b = c
                if is_ct(a) and is_ct(b):
                    eng.constrain(a, b)
            elif tag == "class" and c[1] in self._LITERAL_CLASSES:
                _, cls, args, node_id = c
                for ty in args:
                    if is_ct(ty):
                        prim = CompactType(id=next_id(), kind='primitive', name=cls)
                        eng.constrain(prim, ty, node_id)
                        eng.constrain(ty, prim, node_id)
            elif tag == "call":
                _, callee, arg_tys, result, node_id = c
                if is_ct(callee) and is_ct(result) and all(is_ct(a) for a in arg_tys):
                    wanted = CompactType(id=next_id(), kind='function',
                                         param_types=list(arg_tys),
                                         return_type=result)
                    eng.constrain(callee, wanted, node_id)
        self._biunify_engine = eng
        return eng

    def principal_type_of(self, node_id: int, positive: bool = True) -> str | None:
        """Principal type (rendered) for a frozen node id, or None when the
        node has no recorded type. Purely advisory: querying this never
        changes what compiles."""
        ty = self.types.get(node_id)
        if ty is None:
            ty = self.function_types.get(node_id)
        if ty is None:
            return None
        return self.biunify().principal_type(ty, positive)

    def principal_types(self) -> Dict[int, str]:
        """Principal types for every node id in the types table."""
        return {nid: self.biunify().principal_type(ty)
                for nid, ty in self.types.items()}

    def biunify_errors(self) -> list[Any]:
        """The biunifier's advisory BiunifyError records (kind
        'type-conflict'-shaped, but NOT promoted to compile errors -- the
        adapter's `_detect_class_conflicts` remains the sole source of hard
        type-conflict diagnostics)."""
        return list(self.biunify().errors)

    # --- Solving ---
    def solve(self) -> None:
        # Run frozen constraint checker for custom validations
        from .frozen_constraint_checker import check_constraints

        self.errors, self.effect_info = check_constraints(self._constraints, self.function_types)
        self.errors = (list(self.errors) + self._detect_class_conflicts()
                       + list(self.advisories))

        # If real TypeInferencer is available, translate buffered constraints
        if self._ss is not None and _Polarity is not None:
            for constraint in self._constraints:
                tag = constraint[0] if constraint else None
                if tag == "unify":
                    _, a, b, variance = constraint
                    polarity = _Polarity.NEUTRAL
                    if variance == "covariant":
                        polarity = _Polarity.POSITIVE
                    elif variance == "contravariant":
                        polarity = _Polarity.NEGATIVE
                    self._ss.add_constraint(a, b, polarity)
                elif tag == "subtype":
                    _, a, b = constraint
                    self._ss.add_constraint(a, b, _Polarity.POSITIVE)

            # Run the real solver
            # SimpleSub modifies CompactTypes in place (setting upper_bound/lower_bound)
            # This applies the solution to our types dict since CompactTypes are shared
            self._ss.solve_constraints()

        return None
