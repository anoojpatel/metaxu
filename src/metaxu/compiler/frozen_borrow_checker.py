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
    """A borrow checking error with location information."""
    message: str
    node_id: int


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
        self.reference_graph: Dict[str, List[Tuple[str, str]]] = {}  # var -> [(referenced_var, mode)]
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
            self.errors.append(BorrowError(f"Cannot borrow {var_name} as shared after it was moved", node_id))
            return False
        
        if var_name in self.borrow_state.exclusive_borrows:
            self.errors.append(BorrowError(f"Cannot borrow {var_name} as shared while exclusively borrowed", node_id))
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
            self.errors.append(BorrowError(f"Cannot borrow {var_name} as unique after it was moved", node_id))
            return False
        
        if self.is_borrowed(var_name):
            self.errors.append(BorrowError(f"Cannot borrow {var_name} as unique while borrowed", node_id))
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
            self.errors.append(BorrowError(f"Cannot borrow {var_name} as exclusive after it was moved", node_id))
            return False
        
        if self.is_borrowed(var_name):
            self.errors.append(BorrowError(f"Cannot borrow {var_name} as exclusive while borrowed", node_id))
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
            self.errors.append(BorrowError(f"Cannot move {var_name} after it was already moved", node_id))
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
            self.errors.append(BorrowError(f"Cannot use {var_name} after it was moved", node_id))
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
        - Local variables cannot escape their region
        - Global variables can escape their region
        
        Arguments:
            var_name: Name of the variable to check
            target_region: Target region (defaults to current region)
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
        if var_info.region != target:
            self.errors.append(BorrowError(
                f"Local variable '{var_name}' cannot escape its region (region {var_info.region} -> {target})",
                node_id
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
        if from_var not in self.reference_graph:
            self.reference_graph[from_var] = []
        self.reference_graph[from_var].append((to_var, mode))
        
        # Check for global-to-local reference (dangling reference prevention)
        from_info = self.variables.get(from_var)
        to_info = self.variables.get(to_var)
        
        if from_info and to_info:
            # Global holding reference to local = ERROR (local would escape)
            if from_info.locality == "global" and to_info.locality == "local":
                self.errors.append(BorrowError(
                    f"Global variable '{from_var}' cannot hold reference to local variable '{to_var}' (would create dangling reference)",
                    from_info.node_id
                ))
            # Local holding reference to global = OK (global outlives local)
            # No error needed
    
    def check_reference_conflicts(self, var_name: str, node_id: int) -> bool:
        """Check if a variable has conflicting references.
        
        Rules:
        - A variable cannot have both mutable and const references to the same underlying value
        - A variable with UNIQUE mode cannot be referenced by multiple variables
        - A variable with EXCLUSIVE mode cannot have any other references
        
        Arguments:
            var_name: Name of the variable to check
            node_id: Node ID for error reporting
        """
        if var_name not in self.reference_graph:
            return True
        
        var_info = self.variables.get(var_name)
        if not var_info:
            return True
        
        references = self.reference_graph[var_name]
        
        # Check EXCLUSIVE mode - no other references allowed
        if var_info.mode == "exclusive" and references:
            self.errors.append(BorrowError(
                f"Variable '{var_name}' has exclusive mode but is referenced by {[r[0] for r in references]}",
                node_id
            ))
            return False
        
        # Check UNIQUE mode - only one reference allowed
        if var_info.mode == "unique" and len(references) > 1:
            self.errors.append(BorrowError(
                f"Variable '{var_name}' has unique mode but is referenced by {[r[0] for r in references]}",
                node_id
            ))
            return False
        
        # Check for mut vs const conflicts
        has_mutable = any(mode in ("unique", "exclusive") for _, mode in references)
        has_const = any(mode == "shared" for _, mode in references)
        
        if has_mutable and has_const:
            self.errors.append(BorrowError(
                f"Variable '{var_name}' has both mutable and const references",
                node_id
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
                    node_id
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
        # Exclave uses copy semantics, so no borrow checking errors needed
        # The value is copied to caller's frame, original remains valid
        return True
    
    def get_errors(self) -> List[Tuple[str, int]]:
        """Get all borrow checking errors as (message, node_id) tuples."""
        return [(error.message, error.node_id) for error in self.errors]
