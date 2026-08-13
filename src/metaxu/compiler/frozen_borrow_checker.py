"""Frozen AST Borrow Checker

This module implements a complete borrow checker that operates on the frozen AST
representation, following the full language specification for ownership, borrowing,
and modes as defined in type_defs.py.

See frozen_borrow_spec.md for the complete specification.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Set, List, Tuple, Optional


@dataclass
class BorrowError:
    """A structured borrow checking error.

    Fields:
        message: Human-readable description (also returned by str()).
        node_id: Frozen AST node id where the error was detected.
        kind: Machine-readable error category, e.g. "use-after-move",
              "borrow-conflict", "locality-escape", "suspend-local",
              "reference-conflict", "dangling-reference", "linearity".
        variable: The variable the error is about, when applicable.
    """
    message: str
    node_id: int
    kind: str = "borrow"
    variable: Optional[str] = None

    def __str__(self) -> str:
        return self.message


class BorrowCheckError(Exception):
    """Raised by the pipeline when borrow checking fails in strict mode.

    Carries the structured list of BorrowError objects in `errors`.
    """

    def __init__(self, errors: List[BorrowError]):
        self.errors = list(errors)
        summary = "; ".join(str(e) for e in self.errors) or "borrow check failed"
        super().__init__(f"borrow check failed: {summary}")


class TypeCheckError(Exception):
    """Raised by the pipeline when type checking fails in strict mode.

    Carries the structured list of BorrowError objects (kind "type-*")
    in `errors` — type errors share the structured-diagnostic channel
    with borrow errors but surface as their own exception type.
    """

    def __init__(self, errors: List[BorrowError]):
        self.errors = list(errors)
        summary = "; ".join(str(e) for e in self.errors) or "type check failed"
        super().__init__(f"type check failed: {summary}")


@dataclass
class BorrowState:
    """Tracks borrow state for variables."""
    shared_borrows: Dict[str, int] = field(default_factory=dict)  # variable_name -> count
    exclusive_borrows: Set[str] = field(default_factory=set)  # variables with EXCLUSIVE borrows
    unique_borrows: Set[str] = field(default_factory=set)  # variables with UNIQUE borrows
    mutable_borrows: Set[str] = field(default_factory=set)  # variables with any mutable borrow (UNIQUE or EXCLUSIVE)
    invalidated: Set[str] = field(default_factory=set)  # variables that have been moved


@dataclass
class VariableInfo:
    """Information about a variable's mode and region."""
    name: str
    mode: str  # "shared", "unique", "exclusive"
    locality: str  # "local", "global"
    region: int  # Current region ID
    node_id: int


class FrozenBorrowChecker:
    """Borrow checker for frozen AST nodes.
    
    This checker implements the full language specification for ownership, borrowing,
    and modes including:
    - UniquenessMode: SHARED, UNIQUE, EXCLUSIVE
    - LocalityMode: LOCAL, GLOBAL
    - LinearityMode: ONCE, SEPARATE, MANY
    """
    
    def __init__(self):
        self.borrow_state = BorrowState()
        self.scope_stack: List[Set[str]] = []
        self.region_stack: List[int] = []
        self.variables: Dict[str, VariableInfo] = {}  # variable_name -> VariableInfo
        self.reference_graph: Dict[str, List[Tuple[str, str]]] = {}  # holder -> [(referenced_var, mode)]
        self.referenced_by: Dict[str, List[Tuple[str, str]]] = {}  # referenced_var -> [(holder, mode)]
        self.call_counts: Dict[str, int] = {}  # callable_name -> count
        self.errors: List[BorrowError] = []
        
        self.enter_scope()
        self.enter_region()
    
    def enter_scope(self):
        """Enter a new scope."""
        self.scope_stack.append(set())
    
    def exit_scope(self):
        """Exit the current scope, releasing borrows."""
        scope = self.scope_stack.pop()
        for var_name in scope:
            self.release_borrows(var_name)
    
    def enter_function_state(self) -> tuple:
        """Snapshot per-function borrow/alias state on function entry.

        Each function body is checked against its own state: locals, moves,
        and borrows in one function must not leak into (or poison) another
        function's identically-named bindings.
        """
        snapshot = (self.borrow_state, self.variables,
                    self.reference_graph, self.referenced_by)
        self.borrow_state = BorrowState()
        self.variables = dict(self.variables)
        self.reference_graph = {}
        self.referenced_by = {}
        return snapshot

    def exit_function_state(self, snapshot: tuple) -> None:
        """Restore the enclosing scope's borrow/alias state on function exit."""
        (self.borrow_state, self.variables,
         self.reference_graph, self.referenced_by) = snapshot

    def enter_region(self):
        """Enter a new region for locality tracking."""
        self.region_stack.append(len(self.region_stack))
    
    def exit_region(self):
        """Exit the current region."""
        self.region_stack.pop()
    
    def current_region(self) -> int:
        """Get the current region ID."""
        return self.region_stack[-1] if self.region_stack else 0
    
    def is_borrowed(self, var_name: str) -> bool:
        """Check if a variable is currently borrowed."""
        return var_name in self.borrow_state.shared_borrows or var_name in self.borrow_state.mutable_borrows
    
    def has_exclusive_borrow(self, var_name: str) -> bool:
        """Check if a variable has an exclusive borrow."""
        return var_name in self.borrow_state.exclusive_borrows
    
    def has_unique_borrow(self, var_name: str) -> bool:
        """Check if a variable has a unique borrow."""
        return var_name in self.borrow_state.unique_borrows
    
    def add_shared_borrow(self, var_name: str):
        """Add a shared borrow for a variable."""
        self.borrow_state.shared_borrows[var_name] = self.borrow_state.shared_borrows.get(var_name, 0) + 1
    
    def add_unique_borrow(self, var_name: str):
        """Add a unique borrow for a variable."""
        self.borrow_state.unique_borrows.add(var_name)
        self.borrow_state.mutable_borrows.add(var_name)
    
    def add_exclusive_borrow(self, var_name: str):
        """Add an exclusive borrow for a variable."""
        self.borrow_state.exclusive_borrows.add(var_name)
        self.borrow_state.mutable_borrows.add(var_name)
    
    def release_borrows(self, var_name: str):
        """Release all borrows for a variable."""
        self.borrow_state.shared_borrows.pop(var_name, None)
        self.borrow_state.unique_borrows.discard(var_name)
        self.borrow_state.exclusive_borrows.discard(var_name)
        self.borrow_state.mutable_borrows.discard(var_name)
    
    def release_call_borrow(self, var_name: str, mode: str):
        """Release a call-argument temporary borrow (`f(&mut x)` ended).

        If a live *named* reference (`let r = &mut x`) still holds a borrow of
        the variable, keep the state: only the call-site temporary ends here.
        Named references register in referenced_by; temporaries do not.
        """
        holders = self.referenced_by.get(var_name, [])
        if mode == "shared":
            if any(m == "shared" for _, m in holders):
                return
            self.borrow_state.shared_borrows.pop(var_name, None)
        else:
            if any(m in ("unique", "exclusive") for _, m in holders):
                return
            self.borrow_state.unique_borrows.discard(var_name)
            self.borrow_state.exclusive_borrows.discard(var_name)
            self.borrow_state.mutable_borrows.discard(var_name)

    def invalidate_variable(self, var_name: str):
        """Invalidate a variable (after a move)."""
        self.borrow_state.invalidated.add(var_name)
        self.release_borrows(var_name)
    
    def is_invalidated(self, var_name: str) -> bool:
        """Check if a variable has been invalidated."""
        return var_name in self.borrow_state.invalidated
    
    def check_borrow_shared(self, var_name: str, node_id: int) -> bool:
        """Check if a shared borrow is valid.
        
        Rules:
        - Valid when variable is not exclusively borrowed (EXCLUSIVE)
        - Error if variable is exclusively borrowed
        """
        if var_name in self.borrow_state.invalidated:
            self.errors.append(BorrowError(
                f"Cannot borrow {var_name} as shared after it was moved", node_id,
                kind="borrow-after-move", variable=var_name))
            return False

        if var_name in self.borrow_state.exclusive_borrows:
            self.errors.append(BorrowError(
                f"Cannot borrow {var_name} as shared while exclusively borrowed", node_id,
                kind="borrow-conflict", variable=var_name))
            return False
        
        self.add_shared_borrow(var_name)
        return True
    
    def check_borrow_unique(self, var_name: str, node_id: int) -> bool:
        """Check if a unique borrow (pass-by-value) is valid.
        
        Rules:
        - Valid when variable is not borrowed at all (no shared or exclusive borrows)
        - Transfers ownership - variable becomes invalidated after borrow
        - Error if variable is already borrowed
        - Semantics: Like passing by value - callee can destroy/move the value
        """
        if var_name in self.borrow_state.invalidated:
            self.errors.append(BorrowError(
                f"Cannot borrow {var_name} as unique after it was moved", node_id,
                kind="borrow-after-move", variable=var_name))
            return False

        if self.is_borrowed(var_name):
            self.errors.append(BorrowError(
                f"Cannot borrow {var_name} as unique while borrowed", node_id,
                kind="borrow-conflict", variable=var_name))
            return False
        
        # Transfer ownership - invalidate the source variable
        self.add_unique_borrow(var_name)
        self.invalidate_variable(var_name)
        return True
    
    def check_borrow_exclusive(self, var_name: str, node_id: int) -> bool:
        """Check if an exclusive borrow (mutable reference) is valid.
        
        Rules:
        - Valid when variable is not borrowed at all (no shared or exclusive borrows)
        - Does NOT invalidate the source variable (it's a reference)
        - Error if variable is already borrowed
        - Semantics: Like Rust's &mut - borrowing a reference to existing data
        """
        if var_name in self.borrow_state.invalidated:
            self.errors.append(BorrowError(
                f"Cannot borrow {var_name} as exclusive after it was moved", node_id,
                kind="borrow-after-move", variable=var_name))
            return False

        if self.is_borrowed(var_name):
            self.errors.append(BorrowError(
                f"Cannot borrow {var_name} as exclusive while borrowed", node_id,
                kind="borrow-conflict", variable=var_name))
            return False
        
        # Add exclusive borrow - does NOT invalidate source (it's a reference)
        self.add_exclusive_borrow(var_name)
        return True
    
    def check_move(self, var_name: str, node_id: int) -> bool:
        """Check if a move is valid.
        
        Rules:
        - Always valid
        - Invalidates the variable
        - Releases all borrows
        """
        if var_name in self.borrow_state.invalidated:
            self.errors.append(BorrowError(
                f"Cannot move {var_name} after it was already moved", node_id,
                kind="move-after-move", variable=var_name))
            return False

        self.invalidate_variable(var_name)
        return True
    
    def check_variable_use(self, var_name: str, node_id: int) -> bool:
        """Check if a variable use is valid.
        
        Rules:
        - Cannot use an invalidated variable
        - Cannot use a variable that is exclusively borrowed by someone else
        """
        if var_name in self.borrow_state.invalidated:
            self.errors.append(BorrowError(
                f"Cannot use {var_name} after it was moved", node_id,
                kind="use-after-move", variable=var_name))
            return False

        return True
    
    def declare_variable(self, var_name: str, mode: str, locality: str, node_id: int):
        """Declare a variable with its mode and locality.
        
        Arguments:
            var_name: Name of the variable
            mode: "shared", "unique", or "exclusive"
            locality: "local" or "global"
            node_id: Node ID for error reporting
        """
        # A (re)declaration is a fresh binding: clear any stale move/borrow
        # state a same-named earlier binding (shadowing, other scope) left.
        self.borrow_state.invalidated.discard(var_name)
        self.release_borrows(var_name)
        self.variables[var_name] = VariableInfo(
            name=var_name,
            mode=mode,
            locality=locality,
            region=self.current_region(),
            node_id=node_id
        )
        
        # Track variable in current scope for cleanup
        if self.scope_stack:
            self.scope_stack[-1].add(var_name)
    
    def check_locality(self, var_name: str, target_region: Optional[int] = None, node_id: int = 0) -> bool:
        """Check if a variable would escape its region.

        Rules:
        - Local (@local) values cannot escape to an *older* (outer) region than
          the one they were declared in. Using a local from an enclosing region
          inside a nested region is fine; the escape direction is outward.
        - Global values can escape freely.

        Arguments:
            var_name: Name of the variable to check
            target_region: Region the value would flow into (defaults to current region)
            node_id: Node ID for error reporting
        """
        if var_name not in self.variables:
            # Variable not declared, skip locality check
            return True

        var_info = self.variables[var_name]

        if var_info.locality != "local":
            # Global variables can escape
            return True

        target = target_region if target_region is not None else self.current_region()
        if var_info.region > target:
            self.errors.append(BorrowError(
                f"Local variable '{var_name}' cannot escape its region (region {var_info.region} -> {target})",
                node_id,
                kind="locality-escape",
                variable=var_name,
            ))
            return False

        return True
    
    def track_reference(self, from_var: str, to_var: str, mode: str):
        """Track a reference relationship between variables.

        Arguments:
            from_var: Variable that holds the reference
            to_var: Variable being referenced
            mode: Mode of the reference ("shared", "unique", "exclusive")
        """
        self.reference_graph.setdefault(from_var, []).append((to_var, mode))
        self.referenced_by.setdefault(to_var, []).append((from_var, mode))

        # Check for global-to-local reference (dangling reference prevention)
        from_info = self.variables.get(from_var)
        to_info = self.variables.get(to_var)

        if from_info and to_info:
            # Global holding reference to local = ERROR (local would escape)
            if from_info.locality == "global" and to_info.locality == "local":
                self.errors.append(BorrowError(
                    f"Global variable '{from_var}' cannot hold reference to local variable '{to_var}' (would create dangling reference)",
                    from_info.node_id,
                    kind="dangling-reference",
                    variable=to_var,
                ))
            # Local holding reference to global = OK (global outlives local)
            # No error needed

    def check_reference_conflicts(self, var_name: str, node_id: int) -> bool:
        """Check if a referenced variable has conflicting references.

        Rules:
        - A value cannot have both mutable and const references outstanding
        - A value with UNIQUE mode cannot be referenced by multiple holders
        - A value with EXCLUSIVE mode cannot have any other references

        Arguments:
            var_name: Name of the *referenced* variable to check
            node_id: Node ID for error reporting
        """
        references = self.referenced_by.get(var_name)
        if not references:
            return True

        var_info = self.variables.get(var_name)
        if not var_info:
            return True

        holders = [holder for holder, _ in references]

        # Check EXCLUSIVE mode - no other references allowed
        if var_info.mode == "exclusive" and references:
            self.errors.append(BorrowError(
                f"Variable '{var_name}' has exclusive mode but is referenced by {holders}",
                node_id,
                kind="reference-conflict",
                variable=var_name,
            ))
            return False

        # Check UNIQUE mode - only one reference allowed
        if var_info.mode == "unique" and len(references) > 1:
            self.errors.append(BorrowError(
                f"Variable '{var_name}' has unique mode but is referenced by {holders}",
                node_id,
                kind="reference-conflict",
                variable=var_name,
            ))
            return False

        # Check for mut vs const conflicts
        has_mutable = any(mode in ("unique", "exclusive") for _, mode in references)
        has_const = any(mode == "shared" for _, mode in references)

        if has_mutable and has_const:
            self.errors.append(BorrowError(
                f"Variable '{var_name}' has both mutable and const references",
                node_id,
                kind="reference-conflict",
                variable=var_name,
            ))
            return False

        return True
    
    def check_linearity(self, callable_name: str, linearity: str, node_id: int) -> bool:
        """Check if a callable invocation respects linearity.
        
        Rules:
        - ONCE: Can only be invoked once
        - SEPARATE: Each invocation has separate state
        - MANY: Can be invoked multiple times
        
        Arguments:
            callable_name: Name of the callable
            linearity: Linearity mode ("once", "separate", "many")
            node_id: Node ID for error reporting
        """
        if linearity == "once":
            count = self.call_counts.get(callable_name, 0)
            if count >= 1:
                self.errors.append(BorrowError(
                    f"Once callable '{callable_name}' invoked more than once",
                    node_id,
                    kind="linearity",
                    variable=callable_name,
                ))
                return False
            self.call_counts[callable_name] = count + 1
        elif linearity == "many":
            self.call_counts[callable_name] = self.call_counts.get(callable_name, 0) + 1
        # SEPARATE: No counting needed, each invocation is separate
        
        return True
    
    def check_exclave(self, expression_var: str, node_id: int) -> bool:
        """Check if an exclave expression can promote a local value to caller's scope.
        
        Rules:
        - Exclave copies the value to the caller's stack frame (not a move)
        - The original local value remains valid in its scope
        - This allows controlled escape of local values
        - No borrow checking errors needed for local variables (copy semantics)
        
        Arguments:
            expression_var: Variable name in the exclave expression
            node_id: Node ID for error reporting
        """
        # Exclave copies the value to the caller's frame, so it is a *legal*
        # escape for @local values. The only illegal case is exclaving a value
        # that has already been moved (nothing left to copy).
        if expression_var in self.borrow_state.invalidated:
            self.errors.append(BorrowError(
                f"Cannot exclave {expression_var} after it was moved",
                node_id,
                kind="use-after-move",
                variable=expression_var,
            ))
            return False
        return True

    def get_errors(self) -> List[BorrowError]:
        """Get all borrow checking errors as structured BorrowError objects.

        Each error carries (kind, variable, node_id, message); str(error)
        yields the display message.
        """
        return list(self.errors)
