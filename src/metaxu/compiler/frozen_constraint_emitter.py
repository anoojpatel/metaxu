from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

try:
    from metaxu.type_defs import CompactType
except Exception:
    CompactType = None  # type: ignore

from .mutaxu_ast import AstNode
from .simplesub_adapter import SimpleSubFacade
from .frozen_borrow_checker import FrozenBorrowChecker, BorrowError

# Scaffold for emitting constraints over the frozen AST (mutaxu_ast).
# This module intentionally does nothing substantive yet; it provides a stable
# interface for future work to replay constraints against SimpleSub using
# CompactType variables keyed by frozen node_id.


def emit_constraints(frozen_root: Any, types: Dict[int, Any], simplesub: Any) -> Tuple[None, List[Tuple[str, int]]]:
    """Walk the frozen AST and emit constraints through the provided facade.

    Arguments:
      frozen_root: root node from metaxu.compiler.mutaxu_ast
      types: mapping of node_id -> CompactType (allocated by caller)
      simplesub: an adapter/facade with add_* APIs and solve()

    Returns:
      Tuple of (None, borrow_errors) where borrow_errors contains (message, node_id) tuples
    """
    scopes: list[dict[str, Any]] = [{}]
    return_types: list[Any] = []
    borrow_checker = FrozenBorrowChecker()
    effect_classes: dict[str, str] = {}  # effect_name -> effect_class (stack/suspend)
    function_effects: dict[str, list[str]] = {}  # function_name -> list of effects it performs
    handler_contexts: list[dict[str, str]] = []  # Stack of handler contexts: effect_name -> effect_class
    handler_locals: list[set[str]] = []  # Track variables assigned in handler contexts
    handler_operations: list[list[str]] = []  # Track operations in handler (for stack effect checking)

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

    def push_scope() -> None:
        scopes.append({})
        borrow_checker.enter_scope()

    def pop_scope() -> None:
        scopes.pop()
        borrow_checker.exit_scope()

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
            if effect_name and effect_class:
                handler_contexts.append({effect_name: effect_class})
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
                                            node_id=child.node_id
                                        )
                                    )
        if kind == "Literal" and node_ty is not None:
            cls = literal_class(getattr(node, "value", None))
            if cls is not None:
                simplesub.add_class_constraint(cls, [node_ty], node.node_id)
        if kind == "Block":
            push_scope()
            last_child_ty = None
            for child in children:
                child_ty = types.get(child.node_id)
                if node_ty is not None and child_ty is not None:
                    simplesub.add_subtype(child_ty, node_ty)
                walk(child)
                last_child_ty = child_ty
            if kind == "Block" and node_ty is not None and last_child_ty is not None:
                simplesub.add_unify(node_ty, last_child_ty)
            pop_scope()
            return None
        if kind == "FunctionDeclaration" and node_ty is not None:
            # Record effects performed by this function
            value = payload_dict(node)
            func_name = value.get("name")
            performs = value.get("performs", []) or []
            if func_name and isinstance(func_name, str):
                function_effects[func_name] = performs
            # Construct CompactType function type first
            # so we can bind the function name to the function type
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
                    mode = payload_dict(child).get("mode")
                    if isinstance(mode, str):
                        simplesub.add_class_constraint(f"Mode:{mode}", [child_ty], child.node_id)
                        # Declare parameter in borrow checker
                        locality = "global"  # Parameters are global by default
                        borrow_checker.declare_variable(name, mode or "shared", locality, child.node_id)
            
            # Construct CompactType function type
            # Store in types dict for proper representation
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
            
            push_scope()
            
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
            pop_scope()
            return None
        if kind == "LambdaExpression" and node_ty is not None:
            outer_bindings = {name: lookup(name) for name in payload_dict(node).get("captures", {})}
            push_scope()
            return_types.append(node_ty)
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
                    mode = payload_dict(child).get("mode")
                    if isinstance(mode, str):
                        simplesub.add_class_constraint(f"Mode:{mode}", [child_ty], child.node_id)
                        # Declare parameter in borrow checker
                        locality = "global"  # Parameters are global by default
                        borrow_checker.declare_variable(name, mode or "shared", locality, child.node_id)
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
            # Declare variable in borrow checker
            var_name = payload_name(node)
            if isinstance(var_name, str):
                borrow_checker.declare_variable(var_name, "shared", "global", node.node_id)
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
            for child in children:
                child_ty = types.get(child.node_id)
                if child_ty is not None:
                    simplesub.add_class_constraint("Arg", [node_ty, child_ty], child.node_id)
                    # Check locality of argument if it's a variable
                    if child.kind == "Variable":
                        var_name = payload_dict(child).get("name")
                        if isinstance(var_name, str):
                            borrow_checker.check_locality(var_name, None, child.node_id)
                            # Check if callee performs suspend effects
                            if isinstance(name, str) and name in function_effects:
                                for effect in function_effects[name]:
                                    effect_class = effect_classes.get(effect)
                                    if effect_class == "suspend":
                                        # Suspend effects cannot take local variables
                                        borrow_checker.errors.append(
                                            BorrowError(message=f"Local variable '{var_name}' passed to suspend effect '{effect}'", node_id=child.node_id)
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
                if kind == "BinaryOperation" and payload_operator(node) in {"+", "-", "*", "/"}:
                    simplesub.add_class_constraint("Number", [node_ty], node.node_id)
        if kind == "IfStatement" and node_ty is not None and children:
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
                    # Check locality if returning a variable
                    if children[0].kind == "Variable":
                        var_name = payload_dict(children[0]).get("name")
                        if isinstance(var_name, str):
                            borrow_checker.check_locality(var_name, None, children[0].node_id)
            elif return_types:
                simplesub.add_class_constraint("Unit", [return_types[-1]], node.node_id)
        if kind == "Assignment" and node_ty is not None:
            binding_ty = lookup(payload_name(node))
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
                borrow_checker.check_borrow_unique(var_name, node.node_id)
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
            value = payload_dict(node)
            # Exclave uses copy semantics - value is copied to caller's frame
            # No borrow checking errors needed for local variables
            simplesub.add_class_constraint("Exclave", [node_ty], node.node_id)
        if kind == "StructInstantiation" and node_ty is not None:
            simplesub.add_class_constraint("Struct", [node_ty], node.node_id)
            name = payload_name(node)
            if isinstance(name, str):
                simplesub.add_class_constraint(f"Struct:{name}", [node_ty], node.node_id)
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

    walk(frozen_root)
    return None, borrow_checker.get_errors()
