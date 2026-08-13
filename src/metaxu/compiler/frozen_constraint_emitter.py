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
    handler_locals: list[set[str]] = []  # Track variables assigned in handler contexts
    handler_operations: list[list[str]] = []  # Track operations in handler (for stack effect checking)
    struct_defs: dict[str, dict[str, Any]] = {}  # struct name -> frozen payload (fields/type_params)
    enum_defs: dict[str, dict[str, Any]] = {}  # enum name -> frozen payload (variants)
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

    def _check_struct_field_types(node: Any, struct_name: str) -> None:
        """Check literal field values against the struct's declared field types.

        Conservative: only flags a mismatch when the declared type (after
        substituting the instantiation's type arguments for the struct's
        type parameters) and the literal's type are both known primitives.
        Non-literal fields are left to inference.
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
        substitution = dict(zip(params, args))
        for field in getattr(node, "children", ()):
            if getattr(field, "kind", None) != "StructField":
                continue
            field_name = payload_dict(field).get("name")
            expected = declared.get(field_name)
            if isinstance(expected, str):
                expected = substitution.get(expected, expected)
            expected = _PRIMITIVE_NAMES.get(expected) if isinstance(expected, str) else None
            if expected is None:
                continue
            literal = next(
                (c for c in getattr(field, "children", ()) if getattr(c, "kind", None) == "Literal"),
                None,
            )
            if literal is None:
                continue
            actual = _literal_type_name(getattr(literal, "value", None))
            if actual is not None and actual != expected:
                borrow_checker.errors.append(BorrowError(
                    message=(
                        f"type mismatch for field '{field_name}' of {struct_name}: "
                        f"expected {expected}, got {actual}"
                    ),
                    node_id=field.node_id,
                    kind="type-mismatch",
                    variable=field_name if isinstance(field_name, str) else None,
                ))

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
                                            node_id=child.node_id,
                                            kind="stack-continuation-escape",
                                            variable=var_name,
                                        )
                                    )
        if kind == "Literal" and node_ty is not None:
            cls = literal_class(getattr(node, "value", None))
            if cls is not None:
                simplesub.add_class_constraint(cls, [node_ty], node.node_id)
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
                if kind == "BinaryOperation" and payload_operator(node) in {"+", "-", "*", "/"}:
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

    walk(frozen_root)
    # Publish declared effect classes so the constraint checker can
    # distinguish stack-class effects (which do not suspend) from
    # suspend-class effects. Effects with no declared class are treated as
    # suspend-class (conservative) by the checker.
    if hasattr(simplesub, "add_effect_class"):
        for effect_name, effect_class in effect_classes.items():
            simplesub.add_effect_class(effect_name, effect_class)
    return None, borrow_checker.get_errors()
