from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import metaxu_ast as ast
from symbol_table import SymbolTable, Symbol

class TypeChecker:
    def __init__(self):
        self.symbol_table = SymbolTable()
        self.comptime_context = ast.ComptimeContext()
        self.errors = []
        self.borrow_checker = BorrowChecker(self.symbol_table)
        self.setup_builtin_types()
        self.current_scope = {}
        self.reference_graph = {}  # Track references between values
        
    def add_reference(self, from_var: str, to_var: str, mode: 'ast.Mode'):
        """Track reference relationship between variables"""
        if from_var not in self.reference_graph:
            self.reference_graph[from_var] = []
        self.reference_graph[from_var].append((to_var, mode))
        
    def check_reference_conflict(self, var: str) -> bool:
        """Check if a variable has conflicting references (mut and const)"""
        if var not in self.reference_graph:
            return False
            
        has_mut = False
        has_const = False
        
        # Check direct references
        for _, mode in self.reference_graph[var]:
            if mode.is_mut:
                has_mut = True
            if mode.is_const:
                has_const = True
                
        # Check indirect references through the graph
        for ref_var, mode in self.reference_graph.get(var, []):
            if self.check_reference_conflict(ref_var):
                return True
                
        return has_mut and has_const
        
    def check_call_expression(self, call: 'ast.CallExpression'):
        """Check function call for aliasing violations"""
        if isinstance(call.function, ast.VariableExpression):
            func_name = call.function.name
            func_type = self.current_scope.get(func_name)
            
            if func_type and isinstance(func_type, ast.FunctionType):
                # Track which values are used as mut/const refs
                mut_refs = set()
                const_refs = set()
                
                for arg, param_type in zip(call.arguments, func_type.param_types):
                    if isinstance(arg, ast.VariableExpression):
                        arg_name = arg.name
                        if param_type.mode:
                            if param_type.mode.is_mut:
                                mut_refs.add(arg_name)
                            if param_type.mode.is_const:
                                const_refs.add(arg_name)
                                
                # Check for same value used as both mut and const
                for var in mut_refs & const_refs:
                    self.errors.append(
                        f"Cannot pass '{var}' as both mutable and const reference"
                    )
                    
    def check_let_binding(self, binding: 'ast.LetBinding'):
        """Check let binding for aliasing violations"""
        if binding.mode:
            if isinstance(binding.value, ast.VariableExpression):
                source_var = binding.value.name
                source_mode = self.current_scope.get(source_var + "_mode")
                
                if source_mode:
                    # Cannot alias mut as const or vice versa
                    if source_mode.is_mut and binding.mode.is_const:
                        self.errors.append(
                            f"Cannot alias mutable reference '{source_var}' as const"
                        )
                    if source_mode.is_const and binding.mode.is_mut:
                        self.errors.append(
                            f"Cannot alias const reference '{source_var}' as mutable"
                        )
                        
                # Track the reference
                self.add_reference(binding.name, source_var, binding.mode)
                
        # Store the mode for future reference
        if binding.mode:
            self.current_scope[binding.name + "_mode"] = binding.mode
        
    def setup_builtin_types(self):
        # Add built-in types
        int_type = ast.TypeInfo("int", [], is_copy=True)
        float_type = ast.TypeInfo("float", [], is_copy=True)
        bool_type = ast.TypeInfo("bool", [], is_copy=True)
        string_type = ast.TypeInfo("String", [], is_copy=False)  # String needs explicit clone
        
        self.comptime_context.add_type(int_type)
        self.comptime_context.add_type(float_type)
        self.comptime_context.add_type(bool_type)
        self.comptime_context.add_type(string_type)

    def check(self, node):
        # First pass: evaluate all compile-time constructs
        if isinstance(node, ast.ComptimeBlock):
            return self.check_comptime_block(node)
        elif isinstance(node, ast.ComptimeFunction):
            return self.check_comptime_function(node)
        
        # Second pass: normal type checking for runtime code
        method = getattr(self, f'visit_{node.__class__.__name__}', self.visit_generic)
        return method(node)

    def visit_generic(self, node):
        if hasattr(node, '__dict__'):
            for value in node.__dict__.values():
                if isinstance(value, ast.Node):
                    self.check(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.Node):
                            self.check(item)
        else:
            pass

    def visit_Variable(self, node):
        """Type check variable references"""
        symbol = self.symbol_table.lookup(node.name)
        if not symbol:
            self.errors.append(f"Name '{node.name}' not found")
            return None
        return symbol.type

    def visit_Import(self, node):
        """Type check module imports"""
        # Check if module exists
        module_path = '.'.join(node.module_path)
        module = self.symbol_table.lookup_module(module_path)
        if not module:
            self.errors.append(f"Module '{module_path}' not found")
            return None

        # Add module to current scope
        alias = node.alias or node.module_path[-1]
        self.symbol_table.define(alias, Symbol(alias, module))
        return None

    def visit_FromImport(self, node):
        """Type check from-import statements"""
        # Check if module exists
        module_path = '.'.join(node.module_path)
        module = self.symbol_table.lookup_module(module_path)
        if not module:
            self.errors.append(f"Module '{module_path}' not found")
            return None

        # Import specific names
        for name, alias in node.names:
            if name not in module.symbols:
                self.errors.append(f"Name '{name}' not found in module '{module_path}'")
                continue
            target_name = alias or name
            self.symbol_table.define(target_name, module.symbols[name])
        return None

    def visit_Assignment(self, node):
        expr_type = self.check(node.expression)
        symbol = Symbol(node.name, expr_type)
        self.symbol_table.define(node.name, symbol)
        return expr_type

    def visit_BinaryOperation(self, node):
        left_type = self.check(node.left)
        right_type = self.check(node.right)
        if left_type != right_type:
            self.errors.append(f"Type mismatch: {left_type} and {right_type}")
            return None
        return left_type

    def visit_VariableDeclaration(self, node):
        """Handle variable declarations with mode annotations"""
        var_type = self.check(node.type_annotation)
        value_type = self.check(node.value)

        # Check mode compatibility
        mode = getattr(node, 'mode', 'global')  # Default to global mode if not specified
        if mode == "local":
            # Set up region tracking for local variables
            self.borrow_checker.enter_region()
            
        symbol = Symbol(node.name, var_type, mode=mode)
        self.symbol_table.define(node.name, symbol)
        
        if mode == "local":
            symbol.set_region(self.borrow_checker.current_region())

        return var_type

    def visit_Assignment(self, node):
        """Handle assignments with mode checking"""
        target_symbol = self.symbol_table.lookup(node.target.name)
        value_type = self.check(node.value)

        if target_symbol.mode == "local":
            # Check if value would escape its region
            if isinstance(node.value, ast.Variable):
                value_symbol = self.symbol_table.lookup(node.value.name)
                if value_symbol:
                    self.borrow_checker.check_locality(node.value.name)

        return value_type

    def visit_LocalDeclaration(self, node):
        """Handle local variable declarations"""
        var_type = self.check(node.type_annotation) if node.type_annotation else None
        
        if hasattr(node, 'expression'):
            value_type = self.check(node.expression)
            if var_type and value_type != var_type:
                self.errors.append(f"Type mismatch in local declaration: expected {var_type}, got {value_type}")
            var_type = value_type
        
        # Create symbol with local mode
        symbol = Symbol(node.variable.name, var_type, mode="local")
        self.symbol_table.define(node.variable.name, symbol)
        
        # Set up region tracking
        symbol.set_region(self.borrow_checker.current_region())
        
        return var_type

    def visit_ExclaveExpression(self, node):
        """Handle exclave expressions that move values to outer region"""
        value_type = self.check(node.expression)
        
        # Exit current region before evaluating the expression
        current_region = self.borrow_checker.current_region()
        self.borrow_checker.exit_region()
        
        # Create a new local value in the outer region
        if isinstance(node.expression, ast.Variable):
            var_name = node.expression.name
            symbol = self.symbol_table.lookup(var_name)
            if symbol:
                # Check if the value would escape its region
                if symbol.mode == "local":
                    outer_region = self.borrow_checker.current_region()
                    if symbol.escapes_region(outer_region):
                        self.errors.append(f"Value '{var_name}' would escape its region")
        
        # Re-enter the region we exited
        self.borrow_checker.enter_region()
        self.borrow_checker.set_current_region(current_region)
        
        return value_type

    def visit_LocalParameter(self, node):
        """Handle local parameters in function declarations"""
        param_type = self.check(node.type_annotation) if node.type_annotation else None
        
        # Create symbol with local mode
        symbol = Symbol(node.name, param_type, mode="local")
        self.symbol_table.define(node.name, symbol)
        
        # Set up region tracking
        symbol.set_region(self.borrow_checker.current_region())
        
        return param_type

    def visit_FunctionDeclaration(self, node):
        """Handle function declarations with mode annotations on parameters"""
        param_types = []
        self.borrow_checker.enter_scope()
        self.borrow_checker.enter_region()  # New region for function body

        # Process parameters
        for param in node.params:
            param_type = self.check(param.type_annotation)
            mode = getattr(param, 'mode', 'global')  # Default to global mode
            symbol = Symbol(param.name, param_type, mode=mode)
            
            if mode == "local":
                symbol.set_region(self.borrow_checker.current_region())
                
            self.symbol_table.define(param.name, symbol)
            param_types.append(param_type)

        return_type = self.check(node.return_type)
        
        # Check function body
        self.check(node.body)
        
        # Exit function scope and region
        self.borrow_checker.exit_scope()
        self.borrow_checker.exit_region()

        return ast.FunctionType(param_types, return_type)

    def visit_Block(self, node):
        # Enter a new scope for the block
        self.borrow_checker.enter_scope()

        for statement in node.statements:
            self.check(statement)

        # Exit the block scope
        self.borrow_checker.exit_scope()
        
    def visit_VariableAssignment(self, node):
    # For assignments like p.x = value
        if isinstance(node.target, ast.FieldAccess):
            struct_name = node.target.expression.name
            field_name = node.target.field_name
            full_name = f"{struct_name}.{field_name}"
            if not self.borrow_checker.borrow(full_name, mutable=True):
                self.errors.append(f"Cannot assign to '{full_name}' because it is borrowed")
            
    def visit_Reference(self, node):
    # For expressions like &p.x
        if isinstance(node.expression, ast.FieldAccess):
            struct_name = node.expression.expression.name
            field_name = node.expression.field_name
            full_name = f"{struct_name}.{field_name}"
            if not self.borrow_checker.borrow(full_name, mutable=node.mutable):
                self.errors.append(f"Cannot borrow '{full_name}' as {'mutable' if node.mutable else 'immutable'} because it is already borrowed")
            
    def visit_Move(self, node):
        symbol = self.symbol_table.lookup(node.variable)
        if symbol:
            symbol.invalidate()
            return symbol.type
        else:
            self.errors.append(f"Variable '{node.variable}' not defined")
            return None

    def visit_EffectDeclaration(self, node):
        """Handle effect declaration with type checking"""
        # Check for duplicate effect declaration
        if self.symbol_table.lookup(node.name):
            self.errors.append(f"Effect {node.name} already declared")
            return None

        # Create effect type with operations
        operations = []
        for op in node.operations:
            # Check operation parameters
            params = []
            for param in op.params:
                param_type = self.check(param.type_annotation)
                params.append(ast.Parameter(param.name, param_type))
            
            # Check return type
            return_type = self.check(op.return_type) if op.return_type else None
            
            operations.append(ast.EffectOperation(
                op.name, params, return_type, op.type_params
            ))

        effect_type = ast.EffectType(node.name, operations, node.type_params)
        self.symbol_table.define(node.name, Symbol(node.name, effect_type))
        return effect_type

    def visit_HandleExpression(self, node):
        """Type check handle expressions with proper resource tracking"""
        # Check effect type
        effect_type = self.check(node.effect)
        if not isinstance(effect_type, ast.EffectType):
            self.errors.append(f"Cannot handle non-effect type: {effect_type}")
            return None

        # Enter handler scope for borrow checking
        self.borrow_checker.enter_handler_scope()

        # Check handler implementations
        for handler in node.handlers:
            op = next((op for op in effect_type.operations if op.name == handler.name), None)
            if not op:
                self.errors.append(f"No such operation {handler.name} in effect {effect_type.name}")
                continue

            # Check handler parameters match operation
            if len(handler.params) != len(op.params) + 1:  # +1 for continuation
                self.errors.append(
                    f"Handler {handler.name} has wrong number of parameters. "
                    f"Expected {len(op.params) + 1}, got {len(handler.params)}"
                )
                continue

            # Check continuation parameter type
            cont_param = handler.params[-1]
            cont_type = ast.ContinuationType(op.return_type, effect_type)
            self.symbol_table.define(cont_param.name, Symbol(cont_param.name, cont_type))

            # Check handler body
            handler_type = self.check(handler.body)
            if handler_type != op.return_type:
                self.errors.append(
                    f"Handler {handler.name} returns {handler_type}, "
                    f"but should return {op.return_type}"
                )

        # Check body with handlers in scope
        body_type = self.check(node.body)

        # Exit handler scope
        self.borrow_checker.exit_handler_scope()

        return body_type

    def visit_PerformExpression(self, node):
        """Type check perform expressions"""
        # Check effect operation exists
        effect_type = self.check(node.effect)
        if not isinstance(effect_type, ast.EffectType):
            self.errors.append(f"Cannot perform non-effect type: {effect_type}")
            return None

        op = next((op for op in effect_type.operations if op.name == node.operation), None)
        if not op:
            self.errors.append(f"No such operation {node.operation} in effect {effect_type.name}")
            return None

        # Check arguments
        if len(node.arguments) != len(op.params):
            self.errors.append(
                f"Wrong number of arguments for {node.operation}. "
                f"Expected {len(op.params)}, got {len(node.arguments)}"
            )
            return None

        for arg, param in zip(node.arguments, op.params):
            arg_type = self.check(arg)
            if arg_type != param.type:
                self.errors.append(
                    f"Wrong argument type for {node.operation}. "
                    f"Expected {param.type}, got {arg_type}"
                )

        return op.return_type

    def visit_Module(self, node):
        """Type check a module"""
        # Enter module scope
        self.symbol_table.enter_module(node.name, node.path)
        
        # Type check all statements
        for stmt in node.statements:
            self.check(stmt)
        
        # Exit module scope
        self.symbol_table.exit_module()

    def visit_Name(self, node):
        """Type check a name node, handling qualified names"""
        if isinstance(node.id, list):  # Qualified name (e.g. std.io.println)
            # Look up the first part (module)
            symbol = self.symbol_table.lookup(node.id[0])
            if not symbol:
                self.errors.append(f"Module '{node.id[0]}' not found")
                return ast.ErrorType()
            
            # Look up subsequent parts in the module's symbols
            current = symbol
            for part in node.id[1:]:
                if not hasattr(current, 'symbols') or part not in current.symbols:
                    self.errors.append(f"Cannot find '{part}' in module '{current.name}'")
                    return ast.ErrorType()
                current = current.symbols[part]
            
            return current.type
        else:
            # Regular name lookup
            symbol = self.symbol_table.lookup(node.id)
            if not symbol:
                self.errors.append(f"Name '{node.id}' not found")
                return ast.ErrorType()
            return symbol.type

    def check_mode_compatibility(self, expected_mode, actual_mode):
        # Check uniqueness compatibility
        if expected_mode.uniqueness.mode == ast.UniquenessMode.UNIQUE:
            if actual_mode.uniqueness.mode != ast.UniquenessMode.UNIQUE:
                return False
        elif expected_mode.uniqueness.mode == ast.UniquenessMode.EXCLUSIVE:
            if actual_mode.uniqueness.mode not in [ast.UniquenessMode.UNIQUE, ast.UniquenessMode.EXCLUSIVE]:
                return False

        # Check locality compatibility
        if expected_mode.locality.mode == ast.LocalityMode.LOCAL:
            if actual_mode.locality.mode != ast.LocalityMode.LOCAL:
                return False

        # Check linearity compatibility
        if expected_mode.linearity.mode == ast.LinearityMode.ONCE:
            if actual_mode.linearity.mode != ast.LinearityMode.ONCE:
                return False
        elif expected_mode.linearity.mode == ast.LinearityMode.SEPARATE:
            if actual_mode.linearity.mode not in [ast.LinearityMode.ONCE, ast.LinearityMode.SEPARATE]:
                return False

        return True

    def visit_FunctionCall(self, node):
        func_type = self.check(node.function)
        if not isinstance(func_type, ast.FunctionType):
            self.errors.append(f"Cannot call non-function type {func_type}")
            return None

        # Check linearity constraints
        if func_type.linearity == ast.LinearityMode.ONCE:
            if node.function.is_consumed:
                self.errors.append("Cannot call a once function multiple times")
            node.function.is_consumed = True
        elif func_type.linearity == ast.LinearityMode.SEPARATE:
            if node.function.is_active:
                self.errors.append("Cannot make reentrant call to separate function")
            node.function.is_active = True
            
        # Check arguments
        if len(node.arguments) != len(func_type.param_types):
            self.errors.append(f"Wrong number of arguments")
            return None

        for arg, param_type in zip(node.arguments, func_type.param_types):
            arg_type = self.check(arg)
            if isinstance(param_type, ast.ModeType):
                if not self.check_mode_compatibility(param_type, arg_type):
                    self.errors.append(f"Mode mismatch: expected {param_type}, got {arg_type}")
                    return None

        return func_type.return_type

    def visit_FieldAccess(self, node):
        base_type = self.check(node.base)
        if not isinstance(base_type, ast.StructType):
            self.errors.append(f"Cannot access field of non-struct type {base_type}")
            return None

        field = base_type.fields.get(node.field_name)
        if not field:
            self.errors.append(f"Field {node.field_name} not found in struct {base_type.name}")
            return None

        if field.is_exclusively_mutable:
            # Check if we have exclusive access to the base
            if not self.borrow_checker.check_exclusivity(node.base.name):
                self.errors.append(f"Cannot access exclusively mutable field without exclusive access")
                return None

        return field.field_type

    def check_struct(self, node):
        # Create TypeInfo for struct
        fields = [
            ast.FieldInfo(
                field.name,
                self.get_type_info(field.type),
                field.modifiers
            )
            for field in node.fields
        ]
        
        type_info = ast.TypeInfo(
            node.name,
            fields,
            node.implements,
            self.can_be_copy(fields)
        )
        
        self.comptime_context.add_type(type_info)
        return type_info
    
    def can_be_copy(self, fields):
        return all(field.is_copy() for field in fields)
    
    def check_comptime_block(self, node):
        """Handle compile-time block evaluation with enhanced safety"""
        try:
            # Track current block for node insertion
            self.current_comptime_block = node
            
            old_context = self.comptime_context
            self.comptime_context = ast.ComptimeContext()
            self.comptime_context.types = old_context.types.copy()
            
            # Verify all used types exist
            self.verify_types_exist(node)
            
            # Check for unsafe operations
            self.check_comptime_safety(node)
            
            # Evaluate the block
            try:
                result = node.eval(self.comptime_context)
            except ast.ComptimeError as e:
                self.handle_comptime_error(e)
                return None
            
            # Process generated code
            if self.comptime_context.generated_code:
                return self.process_generated_code(self.comptime_context.generated_code)
            
            # Restore context
            self.comptime_context = old_context
            return result
            
        finally:
            # Clean up
            self.current_comptime_block = None
    
    def verify_types_exist(self, node):
        """Verify all types used in compile-time code exist"""
        class TypeVerifier(ast.NodeVisitor):
            def __init__(self, checker):
                self.checker = checker
            
            def visit_TypeRef(self, node):
                type_info = self.checker.comptime_context.get_type(node.name)
                if not type_info:
                    self.checker.comptime_context.emit_error(
                        f"Unknown type: {node.name}",
                        node,
                        ["Types must be defined before they can be used in compile-time code"]
                    )
        
        TypeVerifier(self).visit(node)
    
    def check_comptime_safety(self, node):
        """Check for unsafe operations in compile-time code"""
        class SafetyChecker(ast.NodeVisitor):
            def __init__(self, checker):
                self.checker = checker
                self.in_loop = False
                self.unsafe_ops = {
                    'file_open', 'socket_open', 'spawn_thread',
                    'system', 'exec', 'eval'
                }
            
            def visit_FunctionCall(self, node):
                if node.name in self.unsafe_ops:
                    self.checker.comptime_context.emit_error(
                        f"Unsafe operation '{node.name}' in compile-time code",
                        node,
                        ["Compile-time code cannot perform I/O or other unsafe operations"]
                    )
            
            def visit_WhileLoop(self, node):
                if self.in_loop:
                    self.checker.comptime_context.emit_warning(
                        "Nested loops in compile-time code may be expensive",
                        node
                    )
                self.in_loop = True
                self.generic_visit(node)
                self.in_loop = False
        
        SafetyChecker(self).visit(node)
    
    def process_generated_code(self, generated_code):
        """Process and validate generated code"""
        generated_nodes = []
        for code in generated_code:
            try:
                # Parse generated code
                parser = Parser()
                new_nodes = parser.parse(code)
                
                # Validate generated code
                self.validate_generated_code(new_nodes)
                
                # Type check generated code
                for new_node in new_nodes:
                    self.check(new_node)
                
                generated_nodes.extend(new_nodes)
                
            except Exception as e:
                self.comptime_context.emit_error(
                    f"Invalid generated code: {str(e)}",
                    notes=[f"Generated code: {code}"]
                )
        
        return generated_nodes
    
    def validate_generated_code(self, nodes):
        """Validate generated code for common issues"""
        class CodeValidator(ast.NodeVisitor):
            def __init__(self, checker):
                self.checker = checker
                self.defined_names = set()
            
            def visit_FunctionDeclaration(self, node):
                if node.name in self.defined_names:
                    self.checker.comptime_context.emit_error(
                        f"Duplicate definition of '{node.name}' in generated code",
                        node
                    )
                self.defined_names.add(node.name)
            
            def visit_StructDeclaration(self, node):
                if node.name in self.defined_names:
                    self.checker.comptime_context.emit_error(
                        f"Duplicate definition of '{node.name}' in generated code",
                        node
                    )
                self.defined_names.add(node.name)
                
                # Check for common struct issues
                if not node.fields:
                    self.checker.comptime_context.emit_warning(
                        f"Empty struct '{node.name}' generated",
                        node
                    )
        
        CodeValidator(self).visit(nodes)
    
    def handle_comptime_error(self, error):
        """Handle compile-time evaluation errors"""
        if error.node and hasattr(error.node, 'location'):
            loc = error.node.location
            self.errors.append(
                f"Compile-time error at {loc.file}:{loc.line}:{loc.column}: {str(error)}"
            )
        else:
            self.errors.append(f"Compile-time error: {str(error)}")
        
        if error.note:
            self.errors.append(f"note: {error.note}")
    
    def check_comptime_function(self, node):
        """Register compile-time function"""
        # Verify function is valid for compile-time execution
        if not self.is_valid_comptime_function(node):
            self.errors.append(f"Invalid compile-time function: {node.name}")
            return None
        
        # Register in compile-time context
        self.comptime_context.add_variable(node.name, node)
        return node
    
    def is_valid_comptime_function(self, node):
        """Check if a function can be executed at compile time"""
        # Function must not have side effects
        # Only allow certain operations (arithmetic, string manipulation, etc)
        allowed_ops = {'+', '-', '*', '/', '==', '!=', '<', '>', '<=', '>='}
        
        def check_expr(expr):
            if isinstance(expr, ast.BinaryOp):
                return expr.op in allowed_ops and check_expr(expr.left) and check_expr(expr.right)
            elif isinstance(expr, (ast.Literal, ast.Variable)):
                return True
            elif isinstance(expr, ast.FunctionCall):
                # Only allow calls to other compile-time functions
                fn = self.comptime_context.get_variable(expr.name)
                return fn and isinstance(fn, ast.ComptimeFunction)
            return False
        
        # Check all expressions in function body
        def check_node(node):
            if isinstance(node, ast.Expression):
                return check_expr(node)
            elif isinstance(node, ast.Block):
                return all(check_node(stmt) for stmt in node.statements)
            return True
        
        return check_node(node.body)
    
    def get_type_info(self, type_node):
        """Get TypeInfo for a type node"""
        if isinstance(type_node, str):
            return self.comptime_context.get_type(type_node)
        # Handle other type nodes (generics, etc)
        return None
    
    def insert_generated_nodes(self, nodes):
        """Insert generated AST nodes into the appropriate location"""
        if not hasattr(self, 'current_comptime_block'):
            self.errors.append("No current compile-time block to insert nodes into")
            return False
        
        try:
            # Get the current compile-time block
            block = self.current_comptime_block
            
            # Set up proper scoping
            parent_scope = block.get_scope()
            for node in nodes:
                if isinstance(node, ast.Block):
                    node.scope = ast.Scope(parent=parent_scope)
                elif hasattr(node, 'scope'):
                    node.scope = parent_scope
            
            # Insert nodes after the compile-time block
            success = block.insert_after(nodes)
            
            if not success:
                # Fallback: try to add to parent block/program
                parent = block.parent
                while parent:
                    if isinstance(parent, (ast.Block, ast.Program)):
                        parent.add_statements(nodes, position=-1)
                        return True
                    parent = parent.parent
                
                self.errors.append("Could not find suitable location to insert generated nodes")
                return False
            
            return True
            
        except Exception as e:
            self.errors.append(f"Error inserting generated nodes: {str(e)}")
            return False

    def check_type_definition(self, node, env):
        """Type check a type definition"""
        # Create new environment for type parameters
        type_env = env.copy()
        
        # Add type parameters to environment
        if node.type_params:
            for param in node.type_params:
                if param.bounds:
                    # Check bounds are valid types
                    for bound in param.bounds:
                        self.check_type(bound, type_env)
                # Add parameter to environment
                type_env[param.name] = param

        # Check the type body is well-formed
        self.check_type(node.body, type_env)
        
        # For recursive types, check positivity condition
        if self.is_recursive(node):
            if not self.check_positivity(node, node.body):
                raise TypeError(f"Recursive type {node.name} violates positivity condition")

    def is_recursive(self, type_def):
        """Check if a type definition is recursive"""
        def contains_self_reference(type_expr):
            if isinstance(type_expr, ast.BasicType):
                return type_expr.name == type_def.name
            elif isinstance(type_expr, ast.TypeApplication):
                return (isinstance(type_expr.base_type, ast.BasicType) and 
                       type_expr.base_type.name == type_def.name)
            elif isinstance(type_expr, ast.TypeParameter):
                return False
            else:
                return any(contains_self_reference(t) for t in type_expr.get_contained_types())
        return contains_self_reference(type_def.body)

    def check_positivity(self, type_def, type_expr, positive=True):
        """Check that recursive references only occur in positive positions"""
        if isinstance(type_expr, ast.BasicType):
            # Self-reference in negative position is not allowed
            if type_expr.name == type_def.name and not positive:
                return False
            return True
        elif isinstance(type_expr, ast.TypeApplication):
            if isinstance(type_expr.base_type, ast.BasicType):
                if type_expr.base_type.name == type_def.name and not positive:
                    return False
            # Check type arguments in same position
            return all(self.check_positivity(type_def, arg, positive) 
                      for arg in type_expr.type_args)
        elif isinstance(type_expr, ast.TypeParameter):
            return True
        elif isinstance(type_expr, ast.FunctionType):
            # Parameter types are in negative position
            return (all(self.check_positivity(type_def, param_type, not positive) 
                       for param_type in type_expr.param_types) and
                    self.check_positivity(type_def, type_expr.return_type, positive))
        else:
            return all(self.check_positivity(type_def, t, positive) 
                      for t in type_expr.get_contained_types())

    def check_type_application(self, node, env):
        """Type check a type application (e.g., List<int>)"""
        # Get the type being applied
        base_type = self.resolve_type(node.base_type, env)
        if not hasattr(base_type, 'type_params'):
            raise TypeError(f"Type {base_type} is not generic")
        
        # Check number of type arguments matches
        if len(node.type_args) != len(base_type.type_params):
            raise TypeError(f"Wrong number of type arguments for {base_type}")
        
        # Check each type argument
        for arg, param in zip(node.type_args, base_type.type_params):
            arg_type = self.check_type(arg, env)
            # Check bounds
            if param.bounds:
                for bound in param.bounds:
                    if not self.is_subtype(arg_type, bound):
                        raise TypeError(f"Type argument {arg_type} does not satisfy bound {bound}")

    def resolve_type(self, type_expr, env):
        """Resolve a type expression to its definition"""
        if isinstance(type_expr, ast.BasicType):
            if type_expr.name in env:
                return env[type_expr.name]
            raise TypeError(f"Undefined type {type_expr.name}")
        elif isinstance(type_expr, ast.TypeParameter):
            if type_expr.name in env:
                return env[type_expr.name]
            raise TypeError(f"Undefined type parameter {type_expr.name}")
        elif isinstance(type_expr, ast.TypeApplication):
            base = self.resolve_type(type_expr.base_type, env)
            return self.substitute_type_params(base, type_expr.type_args, env)
        return type_expr

    def substitute_type_params(self, type_def, type_args, env):
        """Substitute type parameters with concrete types"""
        subst = dict(zip([p.name for p in type_def.type_params], type_args))
        return self.apply_substitution(type_def.body, subst, env)

    def apply_substitution(self, type_expr, subst, env):
        """Apply a type parameter substitution to a type expression"""
        if isinstance(type_expr, ast.BasicType):
            return type_expr
        elif isinstance(type_expr, ast.TypeParameter):
            if type_expr.name in subst:
                return subst[type_expr.name]
            return type_expr
        elif isinstance(type_expr, ast.TypeApplication):
            base = self.apply_substitution(type_expr.base_type, subst, env)
            args = [self.apply_substitution(arg, subst, env) 
                   for arg in type_expr.type_args]
            return ast.TypeApplication(base, args)
        else:
            return type_expr.map_types(lambda t: self.apply_substitution(t, subst, env))

    def is_subtype(self, subtype, supertype):
        # Implement subtype checking logic here
        pass

    def check_reference_compatibility(self, source_type, target_type, context):
        """Check if one reference type can be assigned to another"""
        if not isinstance(source_type, ast.ReferenceType) or not isinstance(target_type, ast.ReferenceType):
            return False
        
        # Check reference modes
        if source_type.mode == target_type.mode:
            # Same mode is always compatible
            pass
        elif source_type.mode == ast.ReferenceMode.UNIQUE:
            # Unique can be converted to shared or exclusive
            pass
        elif source_type.mode == ast.ReferenceMode.EXCLUSIVE:
            # Exclusive can be temporarily converted to shared
            if target_type.mode != ast.ReferenceMode.SHARED:
                context.emit_error(
                    f"Cannot convert {source_type.mode.value} reference to {target_type.mode.value}",
                    notes=["Only UNIQUE references can be converted to any mode",
                          "EXCLUSIVE references can only be converted to SHARED temporarily"]
                )
                return False
        else:  # SHARED
            # Shared references cannot be converted to other modes
            if target_type.mode != ast.ReferenceMode.SHARED:
                context.emit_error(
                    f"Cannot convert shared reference to {target_type.mode.value}",
                    notes=["SHARED references cannot be converted to other modes"]
                )
                return False
        
        # Check lifetimes
        if source_type.lifetime and target_type.lifetime:
            if not self.check_lifetime_outlives(source_type.lifetime, target_type.lifetime, context):
                context.emit_error(
                    f"Lifetime '{source_type.lifetime.name}' does not outlive '{target_type.lifetime.name}'",
                    notes=["The borrowed value must be valid for the entire lifetime of the reference"]
                )
                return False
        
        # Check target type compatibility
        return self.check_type_compatibility(source_type.target_type, target_type.target_type, context)

    def check_lifetime_outlives(self, source_lifetime, target_lifetime, context):
        """Check if one lifetime outlives another"""
        if source_lifetime == target_lifetime:
            return True
        
        # Check explicit constraints
        if target_lifetime in source_lifetime.constraints:
            return True
        
        # Check lexical scope
        if isinstance(source_lifetime, ast.Lifetime) and isinstance(target_lifetime, ast.Lifetime):
            return source_lifetime.scope.contains(target_lifetime.scope)
        
        return False

    def check_reference_access(self, ref_type, is_mutation, context):
        """Check if a reference access is valid"""
        if not isinstance(ref_type, ast.ReferenceType):
            return True
        
        if is_mutation:
            if not ref_type.can_mutate():
                context.emit_error(
                    f"Cannot mutate through {ref_type.mode.value} reference",
                    notes=["Only UNIQUE and EXCLUSIVE references allow mutation"]
                )
                return False
        
        return True

    def check_move_from_reference(self, ref_type, context):
        """Check if we can move a value out of a reference"""
        if not isinstance(ref_type, ast.ReferenceType):
            return True
        
        if not ref_type.can_move_from():
            context.emit_error(
                f"Cannot move value out of {ref_type.mode.value} reference",
                notes=["Only UNIQUE references allow moving out values"]
            )
            return False
        
        return True

    def visit_LambdaExpression(self, node):
        """Type check a lambda expression and verify its closure captures"""
        scope = self.symbol_table.current_scope()
        return self.check_lambda_expression(node, scope)

    def check_lambda_expression(self, node, scope):
        """Type check a lambda expression and verify its closure captures"""
        # Create new scope for lambda parameters
        lambda_scope = ast.Scope(parent=scope)
        
        # Add parameters to scope
        for param in node.params:
            param_type = self.check(param.type_annotation, scope)
            lambda_scope.add_variable(param.name, param_type, is_mutable=param.is_mutable)
        
        # Verify captured variables and determine linearity
        has_mut_captures = False
        has_once_captures = False
        
        for var_name in node.captured_vars:
            var = scope.lookup(var_name)
            if not var:
                self.errors.append(f"Undefined variable '{var_name}' captured in lambda")
                continue
                
            capture_mode = node.capture_modes[var_name]
            
            # Check capture mode compatibility
            if capture_mode == "borrow_mut":
                has_mut_captures = True
                if not var.is_mutable:
                    self.errors.append(f"Cannot mutably borrow immutable variable '{var_name}' in lambda")
                if var.is_borrowed_mut:
                    self.errors.append(f"Cannot mutably borrow '{var_name}' as it is already mutably borrowed")
                var.is_borrowed_mut = True
            else:  # borrow
                if var.is_borrowed_mut:
                    self.errors.append(f"Cannot borrow '{var_name}' as it is already mutably borrowed")
                var.is_borrowed = True
                
            # Check if we're capturing any once values
            if var.linearity == ast.LinearityMode.ONCE:
                has_once_captures = True
        
        # Determine lambda's linearity mode
        if has_once_captures:
            node.linearity = ast.LinearityMode.ONCE
        elif has_mut_captures:
            node.linearity = ast.LinearityMode.SEPARATE
        else:
            node.linearity = ast.LinearityMode.MANY
        
        # Check lambda body
        return_type = self.check(node.body, lambda_scope)
        
        # Record inferred return type if not explicitly specified
        if not node.return_type:
            node.return_type = return_type
        else:
            # Verify return type matches if explicitly specified
            explicit_type = self.check(node.return_type, scope)
            if not self.types_match(return_type, explicit_type):
                self.errors.append(f"Lambda return type mismatch: expected {explicit_type}, got {return_type}")
        
        # Create function type for lambda with appropriate linearity
        param_types = [self.check(p.type_annotation, scope) for p in node.params]
        return ast.FunctionType(param_types, node.return_type, linearity=node.linearity)

    def check_function_call(self, node, scope):
        """Type check a function call with linearity constraints"""
        # Get function type
        func_type = self.check(node.function, scope)
        if not isinstance(func_type, ast.FunctionType):
            self.errors.append(f"Cannot call non-function type {func_type}")
            return ast.ErrorType()
            
        # Check linearity constraints
        if func_type.linearity == ast.LinearityMode.ONCE:
            if node.function.is_consumed:
                self.errors.append("Cannot call a once function multiple times")
            node.function.is_consumed = True
        elif func_type.linearity == ast.LinearityMode.SEPARATE:
            if node.function.is_active:
                self.errors.append("Cannot make reentrant call to separate function")
            node.function.is_active = True
            
        # Check arguments
        if len(node.arguments) != len(func_type.param_types):
            self.errors.append(f"Wrong number of arguments: expected {len(func_type.param_types)}, got {len(node.arguments)}")
            return ast.ErrorType()
            
        for arg, param_type in zip(node.arguments, func_type.param_types):
            arg_type = self.check(arg, scope)
            if not self.types_match(arg_type, param_type):
                self.errors.append(f"Argument type mismatch: expected {param_type}, got {arg_type}")
                
        # Reset active state after call completes
        if func_type.linearity == ast.LinearityMode.SEPARATE:
            node.function.is_active = False
            
        return func_type.return_type

class BorrowChecker:
    def __init__(self, symbol_table):
        """Used for a single symbol_table and children symbol_tables"""
        self.symbol_table = symbol_table
        self.shared_borrows = {}  # variable_name -> count
        self.mutable_borrows = set()  # variable_names with exclusive borrows
        self.unique_vars = set()  # variables with unique mode
        self.scope_stack = []  # Stack of scopes
        self.region_stack = []  # Stack of regions
        self.enter_scope()     # Initialize the global scope
        self.enter_region()    # Initialize the global region

    def enter_scope(self):
        self.scope_stack.append(set())

    def exit_scope(self):
        scope = self.scope_stack.pop()
        for var_name in scope:
            self.release_borrows(var_name)

    def enter_region(self):
        """Enter a new region for locality tracking"""
        self.region_stack.append(len(self.region_stack))

    def exit_region(self):
        """Exit the current region"""
        self.region_stack.pop()

    def current_region(self):
        """Get the current region ID"""
        return self.region_stack[-1] if self.region_stack else None

    def set_current_region(self, region_id):
        """Set the current region ID"""
        self.region_stack[-1] = region_id

    def check_locality(self, var_name, target_region=None):
        """Check if a variable would escape its region"""
        symbol = self.symbol_table.lookup(var_name)
        if not symbol:
            raise Exception(f"Variable '{var_name}' not defined.")
            
        target = target_region if target_region is not None else self.current_region()
        if symbol.escapes_region(target):
            raise Exception(f"Variable '{var_name}' would escape its region")

    def check_uniqueness(self, var_name):
        # Check if a variable can be used uniquely
        if var_name in self.unique_vars:
            if var_name in self.shared_borrows or var_name in self.mutable_borrows:
                return False
            return True
        return False

    def check_exclusivity(self, var_name):
        # Check if a variable can be used exclusively
        if var_name in self.mutable_borrows:
            return False
        if var_name in self.shared_borrows and self.shared_borrows[var_name] > 0:
            return False
        return True

    def visit_BorrowShared(self, node):
        var_name = node.variable.name
        if var_name in self.mutable_borrows:
            self.errors.append(f"Cannot borrow {var_name} as shared while exclusively borrowed")
            return False
        self.add_shared_borrow(var_name)
        return True

    def visit_BorrowUnique(self, node):
        var_name = node.variable.name
        if self.is_borrowed(var_name):
            self.errors.append(f"Cannot borrow {var_name} as unique while borrowed")
            return False
        self.add_mutable_borrow(var_name)
        return True

    def visit_Move(self, node):
        # Invalidate the variable.
        symbol = self.symbol_table.lookup(node.variable)
        if symbol:
            symbol.invalidate()
        else:
            raise Exception(f"Variable '{node.variable}' not defined.")

    def is_borrowed(self, var_name):
        return var_name in self.shared_borrows or var_name in self.mutable_borrows

    def has_mutable_borrow(self, var_name):
        return var_name in self.mutable_borrows

    def add_shared_borrow(self, var_name):
        self.shared_borrows[var_name] = self.shared_borrows.get(var_name, 0) + 1

    def add_mutable_borrow(self, var_name):
        self.mutable_borrows.add(var_name)

    def release_borrows(self, var_name):
        # Called when the variable's scope ends or when borrows are no longer valid.
        self.shared_borrows.pop(var_name, None)
        self.mutable_borrows.discard(var_name)

    def enter_handler_scope(self):
        self.scope_stack.append(set())

    def exit_handler_scope(self):
        scope = self.scope_stack.pop()
        for var_name in scope:
            self.release_borrows(var_name)

    def visit_VariableDeclaration(self, node):
        """Handle variable declarations with locality"""
        symbol = self.symbol_table.lookup(node.name)
        if symbol and symbol.is_local:
            symbol.set_region(self.current_region())

    def visit_Assignment(self, node):
        """Handle assignments checking locality"""
        if isinstance(node.value, ast.Variable):
            self.check_locality(node.value.name)
        self.visit_generic(node)

    def visit_FunctionCall(self, node):
        """Handle function calls checking locality of arguments"""
        for arg in node.args:
            if isinstance(arg, ast.Variable):
                self.check_locality(arg.name)
        self.visit_generic(node)

    def visit_generic(self, node):
        if hasattr(node, '__dict__'):
            for value in node.__dict__.values():
                if isinstance(value, ast.Node):
                    self.visit(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.Node):
                            self.visit(item)
        else:
            pass

    def visit_BorrowShared(self, node):
        var_name = node.variable
        if var_name in self.mutable_borrows:
            self.errors.append(f"Cannot borrow {var_name} as shared while exclusively borrowed")
            return False
        self.add_shared_borrow(var_name)
        return True

    def visit_BorrowUnique(self, node):
        var_name = node.variable
        if self.is_borrowed(var_name):
            self.errors.append(f"Cannot borrow {var_name} as unique while borrowed")
            return False
        self.add_mutable_borrow(var_name)
        return True

    def visit_Move(self, node):
        # Invalidate the variable.
        symbol = self.symbol_table.lookup(node.variable)
        if symbol:
            symbol.invalidate()
        else:
            raise Exception(f"Variable '{node.variable}' not defined.")

    def is_borrowed(self, var_name):
        return var_name in self.shared_borrows or var_name in self.mutable_borrows

    def has_mutable_borrow(self, var_name):
        return var_name in self.mutable_borrows

    def add_shared_borrow(self, var_name):
        self.shared_borrows[var_name] = self.shared_borrows.get(var_name, 0) + 1

    def add_mutable_borrow(self, var_name):
        self.mutable_borrows.add(var_name)

    def release_borrows(self, var_name):
        # Called when the variable's scope ends or when borrows are no longer valid.
        self.shared_borrows.pop(var_name, None)
        self.mutable_borrows.discard(var_name)
