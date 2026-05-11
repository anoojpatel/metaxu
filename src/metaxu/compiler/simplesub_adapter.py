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

    # --- Solving ---
    def solve(self) -> None:
        # Run frozen constraint checker for custom validations
        from .frozen_constraint_checker import check_constraints

        self.errors, self.effect_info = check_constraints(self._constraints, self.function_types)

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
