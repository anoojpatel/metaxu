# Frozen AST Borrow Checker Specification

## Overview
This document defines the specification for a complete borrow checker that operates on the frozen AST representation. This borrow checker implements the full language specification for ownership, borrowing, and modes as defined in `type_defs.py`.

## Language Mode Specification

### UniquenessMode (from type_defs.py)
- **SHARED**: Default mode, allows multiple concurrent references (like Rust's shared references)
- **UNIQUE**: Pass-by-value semantics, callee can destroy/move the value (like Rust's owned values)
- **EXCLUSIVE**: Rust-like references, borrowing a reference to existing data (like Rust's &mut)

### LocalityMode (from type_defs.py)
- **GLOBAL**: Variable can escape its current region
- **LOCAL**: Variable is tied to its current region, cannot escape

### LinearityMode (from type_defs.py)
- **MANY**: Callable can be invoked multiple times
- **ONCE**: Callable can only be invoked once
- **SEPARATE**: Each invocation has separate state

## Borrow Checker Requirements

### Core State Tracking
The borrow checker must track:
1. **Shared borrows**: `Dict[str, int]` - variable_name -> borrow count
2. **Exclusive borrows**: `Set[str]` - variables with EXCLUSIVE borrows
3. **Unique borrows**: `Set[str]` - variables with UNIQUE borrows
4. **Mutable borrows**: `Set[str]` - variables with any mutable borrow (UNIQUE or EXCLUSIVE)
5. **Scope stack**: `List[Set[str]]` - variables in each scope
6. **Region stack**: `List[int]` - region IDs for locality tracking
7. **Reference graph**: `Dict[str, List[Tuple[str, Mode]]]` - variable -> [(referenced_var, mode)]

### Borrow Rules

#### 1. BorrowShared (read-only borrow)
- **Valid when**: Variable is not exclusively borrowed (EXCLUSIVE)
- **Effect**: Increments shared borrow count
- **Error**: "Cannot borrow {var} as shared while exclusively borrowed"

#### 2. BorrowUnique (pass-by-value/owned)
- **Valid when**: Variable is not borrowed at all (no shared or exclusive borrows)
- **Effect**: Transfers ownership, variable becomes invalidated after borrow
- **Error**: "Cannot borrow {var} as unique while borrowed"
- **Semantics**: Like passing by value - callee can destroy/move the value

#### 3. BorrowExclusive (mutable reference)
- **Valid when**: Variable is not borrowed at all (no shared or exclusive borrows)
- **Effect**: Adds to exclusive borrows set, allows mutation
- **Error**: "Cannot borrow {var} as exclusive while borrowed"
- **Semantics**: Like Rust's &mut - borrowing a reference to existing data

#### 4. Move
- **Valid when**: Always valid
- **Effect**: Invalidates the variable, releases all borrows
- **Note**: After a move, the variable cannot be used again

### Locality Rules

#### 1. Local variables
- Variables with `mode: "local"` are tied to their current region
- Cannot be assigned to variables in outer regions
- Cannot be returned from functions
- Cannot be passed as arguments to functions that escape the region

#### 2. Global variables
- Variables with `mode: "global"` can escape their region
- No locality restrictions

#### 3. Region tracking
- Enter region on function entry
- Exit region on function exit
- Check locality on:
  - Variable assignment (if value is a variable)
  - Function call arguments
  - Return statements
  - Struct field assignments

### Linearity Integration

#### 1. ONCE linearity
- Callable can only be invoked once
- After first call, the callable is invalidated
- Error: "Once callable invoked more than once"

#### 2. SEPARATE linearity
- Each invocation has separate borrow state
- Borrows from one invocation don't affect others
- Used for functions with mutable captures

#### 3. MANY linearity (default)
- Callable can be invoked multiple times
- Borrows persist across invocations
- Default for most functions

### Reference Conflict Checking

The reference graph tracks which variables reference others. Must check:
1. **Mut vs Const conflicts**: A variable cannot have both mutable and const references to the same underlying value
2. **Unique conflicts**: A variable with UNIQUE mode cannot be referenced by multiple variables
3. **Exclusive conflicts**: A variable with EXCLUSIVE mode cannot have any other references
4. **Global-to-Local references**: A global variable cannot hold a reference to a local variable (would create dangling reference)
   - Local variable would escape its region when global outlives it
   - Local holding reference to global is OK (global outlives local)

### Exclave Handling

Exclave expressions promote local values to the caller's scope by allocating them on the caller's stack frame. This allows controlled escape of local values.

Rules:
1. **Local promotion**: Exclave takes a local value and allocates it on the caller's stack frame
2. **Region transfer**: Value moves from current region to caller's region
3. **Copy semantics**: The value is copied to the caller's frame (not moved)
4. **Safety**: The original local value remains valid in its scope
5. **Use case**: Allows returning local values from functions without explicit return statements

### Scope Management

#### 1. Enter scope
- Push new empty set onto scope stack
- Call on: Block, FunctionDeclaration, LambdaExpression

#### 2. Exit scope
- Pop scope from stack
- Release all borrows for variables in the exited scope
- Call on: End of Block, FunctionDeclaration, LambdaExpression

### Frozen AST Integration

The borrow checker must integrate with the frozen constraint emitter:

#### 1. Node kinds to handle
- **BorrowShared**: Extract variable from `value["variable"]`, validate borrow rules
- **BorrowUnique**: Extract variable from `value["variable"]`, validate borrow rules
- **Move**: Extract variable from `value["variable"]`, invalidate variable
- **Parameter**: Track mode from `value["mode"]` (unique, exclusive, shared)
- **LetBinding**: Track variable in current scope
- **Assignment**: Check locality if value is a variable
- **FunctionCall**: Check locality of arguments
- **ReturnStatement**: Check locality of returned value
- **FunctionDeclaration**: Enter new scope and region
- **LambdaExpression**: Enter new scope, track captures and linearity

#### 2. Constraint emission
Emit borrow-related constraints:
- `("borrow_shared", var_name, node_id)`
- `("borrow_unique", var_name, node_id)`
- `("borrow_exclusive", var_name, node_id)`
- `("move", var_name, node_id)`
- `("locality_check", var_name, target_region, node_id)`

#### 3. Error reporting
Collect errors with node IDs for precise error reporting:
- List of `(error_message, node_id)` tuples

### Implementation Phases

1. **Phase 1**: Core borrow state tracking (shared, unique, exclusive borrows)
2. **Phase 2**: Scope and region management
3. **Phase 3**: Borrow operation validation (BorrowShared, BorrowUnique, Move)
4. **Phase 4**: Locality checking (LOCAL vs GLOBAL)
5. **Phase 5**: Linearity integration (ONCE, SEPARATE, MANY)
6. **Phase 6**: Reference graph and conflict checking
7. **Phase 7**: Integration with frozen constraint emitter
8. **Phase 8**: Testing and validation

### Deprecation Plan

Deprecate the following to avoid two-world state:
1. `build_tables_with_promoted_borrow_checks()` in infer_tables.py
2. Original BorrowChecker in type_checker.py (mark as legacy)
3. Remove borrow operation visitors from original type_checker

The frozen AST borrow checker will be the single source of truth for borrow checking.
