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
