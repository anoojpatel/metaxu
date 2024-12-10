"""
Code Generator for Metaxu.
Handles translation of AST to VM IR, including structs, types, effects, and memory layout.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
from metaxu.tape_vm import Opcode, Instruction
from metaxu.metaxu_ast import * 
from metaxu.errors import CompileError, SourceLocation
from metaxu.symbol_table import SymbolTable, Symbol
from metaxu.type_defs import *
from metaxu.type_checker import TypeChecker

@dataclass
class TypeConstraint:
    """Represents a constraint on a type parameter"""
    kind: str  # 'implements', 'extends', 'equals'
    bound_type: Any  # The type that bounds this parameter

@dataclass
class GenericTypeInfo:
    """Information about a generic type"""
    name: str
    type_params: List['TypeParameter']
    constraints: Dict[str, List[TypeConstraint]]
    base_layout: Optional['MemoryLayout'] = None

class MemoryLayout:
    """Manages memory layout for structs and types"""
    def __init__(self):
        self.offsets: Dict[str, int] = {}  # Field name to offset mapping
        self.total_size: int = 0
        self.alignment: int = 8  # Default alignment
        self.field_types: Dict[str, Any] = {}  # Track field types for generic instantiation

    def add_field(self, name: str, size: int, align: int = 8, field_type: Any = None) -> int:
        """Add a field and return its offset"""
        self.total_size = (self.total_size + align - 1) & ~(align - 1)
        offset = self.total_size
        self.offsets[name] = offset
        self.field_types[name] = field_type
        self.total_size += size
        return offset

    def get_offset(self, field: str) -> int:
        """Get offset for a field"""
        return self.offsets[field]

    def substitute_types(self, type_args: Dict[str, Any]) -> 'MemoryLayout':
        """Create a new layout with substituted types"""
        new_layout = MemoryLayout()
        for field_name, field_type in self.field_types.items():
            if isinstance(field_type, TypeParameter):
                concrete_type = type_args.get(field_type.name)
                if not concrete_type:
                    raise TypeError(f"Missing type argument for {field_type.name}")
                size = concrete_type.size
                align = concrete_type.align
            else:
                size = field_type.size
                align = field_type.align
            new_layout.add_field(field_name, size, align, field_type)
        return new_layout

@dataclass
class TypeInfo:
    """Type information for code generation"""
    size: int
    align: int
    layout: Optional[MemoryLayout] = None
    is_generic: bool = False
    generic_info: Optional[GenericTypeInfo] = None
    needs_drop: bool = False

class DropFlag:
    """Tracks whether a value needs to be dropped"""
    def __init__(self, var_name: str, var_type: Type):
        self.var_name = var_name
        self.var_type = var_type
        self.needs_drop = True  # Set to False when moved

class CodeGenerator:
    def __init__(self):
        self.instructions = []
        self.symbol_table = SymbolTable()
        self.type_checker = TypeChecker()
        self.scope_stack = []
        self.function_counter = 0
        self.label_counter = 0
        self.functions = {}
        self.current_module = None
        self.generated_modules = set()
        self.output = []
        self.type_info = {
            'int': TypeInfo(size=4, align=4),
            'float': TypeInfo(size=8, align=8),
            'bool': TypeInfo(size=1, align=1),
            'char': TypeInfo(size=1, align=1),
        }
        self.struct_layouts = {}
        self.generic_types = {}
        self.current_stack_offset = 0
        self.effect_handlers = {}
        self.continuations = {}
        self.drop_flags = {}  # Dict[str, DropFlag]
        self.debug = True

    def track_drop(self, var_name: str, var_type: Type):
        """Start tracking a value that needs to be dropped"""
        if var_type.needs_drop():
            self.drop_flags[var_name] = DropFlag(var_name, var_type)

    def move_value(self, var_name: str):
        """Mark a value as moved, so it won't be dropped"""
        if var_name in self.drop_flags:
            self.drop_flags[var_name].needs_drop = False

    def generate_drops(self, scope_vars: set):
        """Generate drop code for variables going out of scope"""
        drop_code = []
        for var_name in scope_vars:
            if var_name in self.drop_flags and self.drop_flags[var_name].needs_drop:
                flag = self.drop_flags[var_name]
                drop_code.extend(self.generate_destructor(flag.var_name, flag.var_type))
                del self.drop_flags[var_name]
        return drop_code

    def generate_destructor(self, var_name: str, var_type: Type) -> List[Instruction]:
        """Generate code to clean up a value"""
        instructions = []

        if isinstance(var_type, DomainType):
            # Free domain and its contents
            instructions.extend([
                Instruction(Opcode.LOAD_VAR, var_name),
                Instruction(Opcode.CALL_FUNC, "free_domain", 1)
            ])
        elif isinstance(var_type, ModeType) and var_type.locality.mode == LocalityMode.GLOBAL:
            # Free heap-allocated global value
            instructions.extend([
                Instruction(Opcode.LOAD_VAR, var_name),
                Instruction(Opcode.CALL_FUNC, "free", 1)
            ])

            # Recursively free fields that need cleanup
            if var_type.destructor:
                for field_name, field_type in var_type.destructor.cleanup_fields.items():
                    field_var = f"{var_name}->{field_name}"
                    instructions.extend(self.generate_destructor(field_var, field_type))

        return instructions

    def new_label(self, prefix="label"):
        label = f"{prefix}_{self.label_counter}"
        self.label_counter += 1
        return label

    def enter_scope(self, scope_name):
        self.scope_stack.append(scope_name)

    def exit_scope(self):
        return self.scope_stack.pop()

    def emit(self, opcode: Opcode, *operands):
        """Emit a VM instruction"""
        self.instructions.append(Instruction(opcode, *operands))
        self.debug = True  # Enable debug logging

    def log(self, msg):
        if self.debug:
            print(f"[CodeGen Debug] {msg}")

    def generate(self, node) -> List[Instruction]:
        """Generate code for a node"""
        self.log(f"generate() called with node type: {type(node)}")
        
        if isinstance(node, Module):
            self.log(f"Handling Module node with name: {node.name}")
            # Generate code for module body
            self.generate(node.body)
            return self.instructions
            
        elif isinstance(node, ModuleBody):
            self.log(f"Handling ModuleBody node with {len(node.statements)} statements")
            # Generate code for each statement in module
            for stmt in node.statements:
                self.generate_statement(stmt)
            return self.instructions
            
        elif isinstance(node, Statement):
            self.log("Handling Statement node")
            self.generate_statement(node)
            return self.instructions
            
        elif isinstance(node, Expression):
            self.log("Handling Expression node")
            self.generate_expression(node)
            return self.instructions
            
        else:
            self.log(f"Attempting to handle unknown node type: {type(node)}")
            # Try to handle as statement if not one of the above
            try:
                self.generate_statement(node)
                return self.instructions
            except CompileError as e:
                self.log(f"Failed to handle node: {e}")
                raise CompileError(
                    message=f"Unsupported node type: {type(node)}",
                    error_type="CodeGenError",
                    notes=[f"The code generator doesn't know how to handle {type(node).__name__} nodes"]
                )

    def generate_statement(self, stmt) -> None:
        """Generate code for a statement"""
        self.log(f"generate_statement() called with statement type: {type(stmt)}")
        
        # High-level module and type-related statements
        if isinstance(stmt, Module):
            self.log("Handling Module in generate_statement")
            # Instead of recursively calling generate(), handle the module body directly
            if hasattr(stmt, 'body') and isinstance(stmt.body, ModuleBody):
                for s in stmt.body.statements:
                    self.generate_statement(s)
            
        elif isinstance(stmt, ModuleBody):
            self.log("Handling ModuleBody in generate_statement")
            for s in stmt.statements:
                self.generate_statement(s)

        elif isinstance(stmt, FunctionDeclaration):
            self.log("Handling FunctionDeclaration")
            self.gen_function_def(stmt)
 

        elif isinstance(stmt, StructDefinition):
            self.log("Handling StructDefinition")
            self.gen_struct_def(stmt)

        elif isinstance(stmt, EffectDeclaration):
            self.log("Handling EffectDeclaration")
            self.gen_EffectDeclaration(stmt)

        elif isinstance(stmt, HandleEffect):
            self.log("Handling HandleEffect")
            self.gen_HandleEffect(stmt)

        elif isinstance(stmt, PerformEffect):
            self.log("Handling PerformEffect")
            self.gen_PerformEffect(stmt)

        # Basic language constructs
        elif isinstance(stmt, LetStatement):
            self.generate_expression(stmt.expression)
            self.emit(Opcode.STORE_VAR, stmt.name)

        elif isinstance(stmt, Assignment):
            self.generate_expression(stmt.expression)
            self.emit(Opcode.STORE_VAR, stmt.name)

        elif isinstance(stmt, IfStatement):
            else_label = self.new_label("else")
            end_label = self.new_label("endif")

            # Generate condition
            self.generate_expression(stmt.condition)
            self.emit(Opcode.JUMP_IF_FALSE, else_label)

            # Generate then branch
            self.generate_statement(stmt.then_body)
            self.emit(Opcode.JUMP, end_label)

            # Generate else branch
            self.emit(Opcode.LABEL, else_label)
            if stmt.else_body:
                self.generate_statement(stmt.else_body)

            self.emit(Opcode.LABEL, end_label)

        elif isinstance(stmt, WhileStatement):
            start_label = self.new_label("while")
            end_label = self.new_label("endwhile")

            # Generate loop header
            self.emit(Opcode.LABEL, start_label)
            self.generate_expression(stmt.condition)
            self.emit(Opcode.JUMP_IF_FALSE, end_label)

            # Generate loop body
            self.generate_statement(stmt.body)
            self.emit(Opcode.JUMP, start_label)

            self.emit(Opcode.LABEL, end_label)

        elif isinstance(stmt, PrintStatement):
            if stmt.arguments:
                for arg in stmt.arguments:
                    self.generate_expression(arg)
                    self.emit(Opcode.PRINT)
            else:
                # Handle empty print statement
                self.emit(Opcode.PRINT_NEWLINE)

        else:
            self.log(f"Unknown statement type: {type(stmt)}")
            raise CompileError(
                message=f"Unsupported statement type: {type(stmt)}",
                error_type="CodeGenError",
                notes=[f"The code generator doesn't know how to handle {type(stmt).__name__} statements"]
    )

    def generate_expression(self, expr):
        """Generate code for an expression"""
        self.log(f"generate_expression() called with expression type: {type(expr)}")
        
        if isinstance(expr, Literal):
            self.log("Handling Literal")
            self.emit(Opcode.LOAD_CONST, expr.value)

        elif isinstance(expr, Variable):
            self.log("Handling Variable")
            self.emit(Opcode.LOAD_VAR, expr.name)

        elif isinstance(expr, BinaryOperation):
            self.log("Handling BinaryOperation")
            self.generate_expression(expr.left)
            self.generate_expression(expr.right)

            # Map operators to opcodes
            op_map = {
                '+': Opcode.ADD,
                '-': Opcode.SUB,
                '*': Opcode.MUL,
                '/': Opcode.DIV,
                '==': Opcode.COMPARE_EQ,
                '!=': Opcode.COMPARE_NE,
                '<': Opcode.COMPARE_LT,
                '<=': Opcode.COMPARE_LE,
                '>': Opcode.COMPARE_GT,
                '>=': Opcode.COMPARE_GE,
            }
            self.emit(op_map[expr.operator])

        elif isinstance(expr, StructInstantiation):
            self.log("Handling StructInstantiation")
            self.gen_struct_instantiation(expr)

        elif isinstance(expr, FieldAccess):
            self.log("Handling FieldAccess")
            self.gen_field_access(expr)

        elif isinstance(expr, HandleEffect):
            self.log("Handling HandleEffect")
            self.gen_HandleEffect(expr)

        elif isinstance(expr, PerformEffect):
            self.log("Handling PerformEffect")
            self.gen_PerformEffect(expr)

        elif isinstance(expr, ThreadEffect):
            self.log("Handling ThreadEffect")
            self.gen_ThreadEffect(expr)

        elif isinstance(expr, DomainEffect):
            self.log("Handling DomainEffect")
            self.gen_DomainEffect(expr)

        elif isinstance(expr, ThreadPoolEffect):
            self.log("Handling ThreadPoolEffect")
            self.gen_ThreadPoolEffect(expr)

        elif isinstance(expr, ExclaveExpression):
            self.log("Handling ExclaveExpression")
            self.gen_ExclaveExpression(expr)

        else:
            self.log(f"Unknown expression type: {type(expr)}")
            raise CompileError(
                message=f"Unsupported expression type: {type(expr)}",
                error_type="CodeGenError",
                notes=[f"The code generator doesn't know how to handle {type(expr).__name__} expressions"]
            )

    def gen_Program(self, node):
        for stmt in node.statements:
            self.generate(stmt)

    def gen_Assignment(self, node):
        # If target is being moved, mark it as moved
        if isinstance(node.expression, MoveNode):
            self.move_value(node.name)
        
        # Generate the expression code
        self.generate(node.expression)
        self.emit(Opcode.STORE_VAR, node.name)
        
        # Track for destruction if needed
        var_type = self.type_checker.check(node.expression)
        symbol = Symbol(node.name, var_type)
        self.symbol_table.define(node.name, symbol)
        if var_type.needs_drop():
            self.track_drop(node.name, var_type)

    def gen_LetStatement(self, node):
        # Similar to Assignment
        if isinstance(node.expression, MoveNode):
            self.move_value(node.name)
            
        self.generate(node.expression)
        self.emit(Opcode.STORE_VAR, node.name)
        
        var_type = self.type_checker.check(node.expression)
        if var_type.needs_drop():
            self.track_drop(node.name, var_type)

    def gen_VariableDeclaration(self, node):
        if isinstance(node.initializer, MoveNode):
            self.move_value(node.name)
            
        self.generate(node.initializer)
        self.emit(Opcode.STORE_VAR, node.name)
        
        # Track for destruction
        if node.type.needs_drop():
            self.track_drop(node.name, node.type)

    def gen_Return(self, node):
        """Generate code for return statement with proper cleanup"""
        # Get current scope's variables that need dropping
        scope_vars = set()
        for scope in reversed(self.scope_stack):
            scope_vars.update(scope.variables)
            
        # Generate drops for all variables in scope
        drop_instrs = self.generate_drops(scope_vars)
        
        # Generate the return value
        self.generate(node.expression)
        
        # Add the drop instructions before the return
        self.instructions.extend(drop_instrs)
        self.emit(Opcode.RETURN)

    def gen_Literal(self, node):
        self.emit(Opcode.LOAD_CONST, node.value)

    def gen_Variable(self, node):
        self.emit(Opcode.LOAD_VAR, node.name)

    def gen_BinaryOperation(self, node):
        left_type = self.type_checker.check(node.left)
        right_type = self.type_checker.check(node.right)
        self.generate(node.left)
        self.generate(node.right)

        if isinstance(left_type, VectorType) and isinstance(right_type, VectorType):
            op_map = {
                '+': Opcode.VEC_ADD,
                '-': Opcode.VEC_SUB,
                '*': Opcode.VEC_MUL,
                '/': Opcode.VEC_DIV,
            }
        else:
            op_map = {
                '+': Opcode.ADD,
                '-': Opcode.SUB,
                '*': Opcode.MUL,
                '/': Opcode.DIV,
                '==': Opcode.COMPARE_EQ,
                '!=': Opcode.COMPARE_NE,
                '<': Opcode.COMPARE_LT,
                '<=': Opcode.COMPARE_LE,
                '>': Opcode.COMPARE_GT,
                '>=': Opcode.COMPARE_GE,
            }
        self.emit(op_map[node.operator])

    def gen_Block(self, node):
        self.enter_scope('block')
        scope_vars = set()
        for stmt in node.statements:
            if isinstance(stmt, VarDeclNode):
                scope_vars.add(stmt.name)
                self.track_drop(stmt.name, stmt.type)
            self.generate(stmt)
        drop_instrs = self.generate_drops(scope_vars)
        self.instructions.extend(drop_instrs)
        self.exit_scope()

    def gen_EffectDeclaration(self, node):
        """Generate code for effect declaration"""
        effect_name = node.name

        # Create effect type info
        effect_layout = MemoryLayout()
        self.type_info[effect_name] = TypeInfo(
            size=effect_layout.total_size,
            align=effect_layout.alignment,
            layout=effect_layout,
            is_generic=bool(node.type_params)
        )

        # Generate labels for each operation
        for op in node.operations:
            op_label = self.new_label(f"{effect_name}_{op.name}")
            self.emit(Opcode.LABEL, op_label)
            self.generate(op)

    def gen_HandleEffect(self, node):
        """Generate code for handle expression"""
        effect_name = node.effect.name

        # Create labels for handlers
        handler_labels = {}
        for handler in node.handlers:
            handler_label = self.new_label(f"handler_{handler.name}")
            handler_labels[handler.name] = handler_label

        # Set up handlers
        for handler in node.handlers:
            self.emit(Opcode.SET_HANDLER, handler.name, handler_labels[handler.name])

        # Generate handler implementations
        for handler in node.handlers:
            self.emit(Opcode.LABEL, handler_labels[handler.name])
            self.enter_scope(f"handler_{handler.name}")

            # Set up parameters including continuation
            for param in handler.params[:-1]:  # Skip continuation param
                self.emit(Opcode.STORE_VAR, param.name)

            # Store continuation
            cont_param = handler.params[-1]
            self.emit(Opcode.CREATE_CONTINUATION, cont_param.name)

            # Generate handler body
            self.generate(handler.body)

            self.exit_scope()

        # Generate body with handlers in scope
        body_end_label = self.new_label("handle_end")
        self.generate(node.body)
        self.emit(Opcode.LABEL, body_end_label)

        # Clean up handlers
        for handler in node.handlers:
            self.emit(Opcode.UNSET_HANDLER, handler.name)


    def gen_StructDefinition(self, node):
        """Generate code for struct definition"""
        struct_name = node.name
        layout = MemoryLayout()

        # Add fields to layout
        for field in node.fields:
            field_type = self.type_checker.check(field.type)
            type_info = self.type_info[field_type.name]
            layout.add_field(field.name, type_info.size, type_info.align, field_type)

        # Store struct info
        self.type_info[struct_name] = TypeInfo(
            size=layout.total_size,
            align=layout.alignment,
            layout=layout,
            is_generic=bool(node.type_params)
        )
        self.struct_layouts[struct_name] = layout

    def gen_StructInstantiation(self, node):
        """Generate code for struct instantiation"""
        struct_type = self.type_checker.check(node)
        layout = self.struct_layouts[struct_type.name]

        # Generate field values
        for field in node.fields:
            self.generate(field.value)

        self.emit(Opcode.CREATE_STRUCT, struct_type.name, len(node.fields))

    def gen_FieldAccess(self, node):
        """Generate code for field access"""
        self.generate(node.struct)
        struct_type = self.type_checker.check(node.struct)
        layout = self.struct_layouts[struct_type.name]
        offset = layout.get_offset(node.field)

        self.emit(Opcode.ACCESS_FIELD, offset)

    def gen_TypeDefinition(self, node):
        """Handle type definitions including generic types"""
        if node.type_params:
            # This is a generic type definition
            self.ir_generator.register_generic_type(node)

            # Add to symbol table
            type_info = self.type_checker.check_type_definition(node)
            self.symbol_table.define_type(node.name, type_info)
        else:
            # Regular type definition
            self.generate(node.body)

    def gen_TypeApplication(self, node):
        """Handle generic type instantiation"""
        # Get base type info
        base_type = self.symbol_table.lookup_type(node.base_type)
        if not base_type:
            raise Exception(f"Undefined type: {node.base_type}")

        # Check and resolve type arguments
        type_args = []
        for arg in node.type_args:
            arg_type = self.type_checker.check_type(arg)
            type_args.append(arg_type)

        # Generate IR for instantiated type
        type_info = self.ir_generator.instantiate_generic_type(node.base_type, type_args)

        # Add instantiated type to instructions
        self.instructions.extend(self.ir_generator.instructions)
        self.ir_generator.instructions.clear()

        return type_info

    def gen_TypeParameter(self, node):
        """Handle type parameter declarations"""
        # Check bounds
        bound_types = []
        for bound in node.bounds:
            bound_type = self.type_checker.check_type(bound)
            bound_types.append(bound_type)

        # Create type parameter info
        type_param = self.type_checker.create_type_parameter(node.name, bound_types)

        # Add to symbol table
        self.symbol_table.define_type_parameter(node.name, type_param)

        return type_param

    def gen_RecursiveType(self, node):
        """Handle recursive type definitions"""
        # First pass: register type parameters
        for param in node.type_params:
            self.gen_TypeParameter(param)

        # Second pass: process the type body with parameters in scope
        body_type = self.type_checker.check_type(node.body)

        # Create recursive type
        type_info = self.type_checker.create_recursive_type(
            node.name,
            node.type_params,
            body_type
        )

        # Add to symbol table
        self.symbol_table.define_type(node.name, type_info)

        return type_info

    def gen_VectorLiteral(self, node):
        for element in node.elements:
            self.generate(element)
        self.emit(Opcode.CREATE_VECTOR, node.base_type, node.size)

    def gen_SpawnExpression(self, node):
        self.generate(node.function_expression)
        self.emit(Opcode.SPAWN_THREAD)

    def gen_EffectHandler(self, effect_name, handler_node):
        handler_label = f"{effect_name}_handler"
        self.enter_scope(handler_label)
        self.emit(Opcode.LABEL, handler_label)

        # Generate code for the handler body
        self.gen_Block(handler_node.body)

        # Exit scope and generate END
        self.exit_scope()
        self.emit(Opcode.END)

    def gen_PerformEffect(self, node):
        # Generate code for arguments
        for arg in node.args:
            self.generate(arg)
            self.emit(Opcode.PUSH)

        # Generate unique labels for continuation and effect handler
        continuation_label = self.new_label("cont")
        effect_handler_label = f"{node.effect_name}_handler"

        # Jump to effect handler
        self.emit(Opcode.JUMP, effect_handler_label)

        # Place continuation label
        self.emit(Opcode.LABEL, continuation_label)
        # Generate code for effect arguments
        for arg in node.args:
            self.generate(arg)
        self.emit(Opcode.PERFORM_EFFECT, node.effect_name, len(node.args))

    def gen_Move(self, node):
        # Handle move semantics
        self.emit(Opcode.MOVE_VAR, node.variable)
        symbol = self.symbol_table.lookup(node.variable)
        if symbol:
            symbol.invalidate()

    def gen_ToDevice(self, node):
        self.emit(Opcode.TO_DEVICE, node.variable)

    def gen_FromDevice(self, node):
        self.emit(Opcode.FROM_DEVICE, node.variable)

    def gen_ExclaveExpression(self, node):
        """Generate code to move a value to the caller's region and return"""
        # Generate value expression
        self.generate_expression(node.value)
        
        # Move value to caller's region and return
        self.emit(Opcode.EXCLAVE)

    def gen_FunctionDeclaration(self, node):
        """Generate code for function definition with proper environment setup"""
        function_label = node.name

        # Generate function object creation
        self.emit(Opcode.CREATE_FUNC, function_label)
        self.emit(Opcode.STORE_VAR, node.name)

        # Generate the function code
        self.enter_scope(function_label)
        self.emit(Opcode.LABEL, function_label)

        # Function parameters and body
        self.gen_FunctionBody(node)

        self.exit_scope()
        self.emit(Opcode.END)

    def gen_FunctionBody(self, node):
        """Generate VM instructions for function body with parameter handling"""
        # Handle function parameters
        param_vars = set()
        for param in node.parameters:
            self.emit(Opcode.POP)
            self.emit(Opcode.STORE_VAR, param.name)
            param_vars.add(param.name)
            # Track parameters that need dropping
            if param.type.needs_drop():
                self.track_drop(param.name, param.type)

        # Generate instructions for the function's statements
        self.gen_Block(node.body)
        
        # Generate drops for parameters before return
        drop_instrs = self.generate_drops(param_vars)
        self.instructions.extend(drop_instrs)

    def gen_FunctionCall(self, node):
        """Generate code for function calls with proper argument handling"""
        # Generate code for the function expression
        self.generate(node.function_expr)

        # Push arguments onto the stack
        for arg in node.arguments:
            if isinstance(arg, MoveNode):
                self.move_value(arg.variable)
            self.generate(arg)
            self.emit(Opcode.PUSH)

        # Call the function
        self.emit(Opcode.CALL_FUNC, len(node.arguments))

    def gen_Closure(self, node):
        """Generate code for closures with proper variable capture"""
        # Identify captured variables
        captured_vars = self.get_captured_variables(node)

        # Generate code to capture variables
        for var in captured_vars:
            self.emit(Opcode.LOAD_VAR, var)
            self.emit(Opcode.PUSH)

        # Create the function object with captured environment
        function_label = self.new_label('lambda')
        self.emit(Opcode.CREATE_CLOSURE, function_label, len(captured_vars))

        # Store the closure
        self.emit(Opcode.STORE_VAR, node.name)

        # Generate the function code
        self.enter_scope(function_label)
        self.emit(Opcode.LABEL, function_label)

        # Function parameters and body
        self.gen_FunctionBody(node)

        self.exit_scope()
        self.emit(Opcode.END)

    def get_captured_variables(self, node):
        """Helper method to identify captured variables in closures"""
        captured = set()
        def visit(n):
            if isinstance(n, Variable):
                if n.name not in self.symbol_table.current_scope():
                    captured.add(n.name)
            for child in n.children():
                visit(child)
        visit(node)
        return list(captured)

    def generate_gpu_code(self, node):
        # Placeholder for GPU code generation
        code = f"// GPU kernel code for function {node.name}\n"
        # Actual code generation logic goes here
        return code

    def register_generic_type(self, type_def) -> None:
        """Register a generic type definition"""
        constraints = {}
        for param in type_def.type_params:
            param_constraints = []
            for bound in param.bounds:
                if isinstance(bound, TypeApplication):
                    constraint = TypeConstraint('implements', bound)
                else:
                    constraint = TypeConstraint('extends', bound)
                param_constraints.append(constraint)
            constraints[param.name] = param_constraints

        generic_info = GenericTypeInfo(
            name=type_def.name,
            type_params=type_def.type_params,
            constraints=constraints
        )
        self.generic_types[type_def.name] = generic_info

        # Create base layout if it's a struct
        if hasattr(type_def.body, 'fields'):
            layout = MemoryLayout()
            for field in type_def.body.fields:
                field_type = self.type_info[field.type_annotation.name]
                layout.add_field(field.name, field_type.size, field_type.align)
            generic_info.base_layout = layout

    def check_type_constraints(self, type_param: str, concrete_type: Any, constraints: List[TypeConstraint]) -> bool:
        """Check if a concrete type satisfies the constraints of a type parameter"""
        for constraint in constraints:
            if constraint.kind == 'implements':
                if not self.implements_interface(concrete_type, constraint.bound_type):
                    return False
            elif constraint.kind == 'extends':
                if not self.extends_type(concrete_type, constraint.bound_type):
                    return False
            elif constraint.kind == 'equals':
                if not self.types_equal(concrete_type, constraint.bound_type):
                    return False
        return True

    def instantiate_generic_type(self, base_type: str, type_args: List[Any]) -> TypeInfo:
        """Instantiate a generic type with concrete type arguments"""
        generic_info = self.generic_types[base_type]

        if len(type_args) != len(generic_info.type_params):
            raise TypeError(f"Wrong number of type arguments for {base_type}")

        type_arg_map = {}
        for param, arg in zip(generic_info.type_params, type_args):
            if not self.check_type_constraints(param.name, arg, generic_info.constraints[param.name]):
                raise TypeError(f"Type argument {arg} does not satisfy constraints for {param.name}")
            type_arg_map[param.name] = arg

        layout = None
        if generic_info.base_layout:
            layout = generic_info.base_layout.substitute_types(type_arg_map)

        total_size = sum(arg.size for arg in type_args)
        max_align = max(arg.align for arg in type_args)

        return TypeInfo(
            size=total_size,
            align=max_align,
            layout=layout,
            is_generic=False
        )

    def implements_interface(self, type_: Any, interface: Any) -> bool:
        """Check if a type implements an interface"""
        if hasattr(type_, 'implements'):
            return interface in type_.implements
        return False

    def extends_type(self, type_: Any, base: Any) -> bool:
        """Check if a type extends another type"""
        if hasattr(type_, 'extends'):
            return base == type_.extends
        return False

    def types_equal(self, type1: Any, type2: Any) -> bool:
        """Check if two types are equal"""
        return type1 == type2

    def gen_struct_def(self, struct_def) -> None:
        """Generate code for struct definition"""
        if hasattr(struct_def, 'type_params'):
            self.register_generic_type(struct_def)
            return

        layout = MemoryLayout()

        # Calculate offsets for each field and track which fields need dropping
        needs_drop = False
        for field in struct_def.fields:
            field_type = self.type_info[field.type_annotation.name]
            layout.add_field(field.name, field_type.size, field_type.align, field_type)
            if field_type.needs_drop:
                needs_drop = True

        # Store layout for later use
        self.struct_layouts[struct_def.name] = layout
        self.type_info[struct_def.name] = TypeInfo(
            size=layout.total_size,
            align=layout.alignment,
            layout=layout,
            needs_drop=needs_drop
        )

        # Generate destructor for struct if needed
        if needs_drop:
            destructor_label = f"drop_{struct_def.name}"
            self.emit(Opcode.LABEL, destructor_label)
            
            # Drop each field that needs dropping
            for field in struct_def.fields:
                field_type = self.type_info[field.type_annotation.name]
                if field_type.needs_drop:
                    offset = layout.get_offset(field.name)
                    self.emit(Opcode.LOAD_ARG, 0)  # Load struct pointer
                    self.emit(Opcode.LOAD_CONST, offset)
                    self.emit(Opcode.ADD)
                    self.emit(Opcode.CALL_FUNC, f"drop_{field_type.name}")
            
            self.emit(Opcode.RETURN)

        # Emit struct creation instruction
        self.emit(Opcode.CREATE_STRUCT, struct_def.name, [f.name for f in struct_def.fields])

    def gen_struct_instantiation(self, struct_inst) -> None:
        """Generate code for struct instantiation"""
        layout = self.struct_layouts[struct_inst.struct_name]
        type_info = self.type_info[struct_inst.struct_name]
        
        # Track struct for dropping if needed
        if type_info.needs_drop:
            self.track_drop(struct_inst.target, type_info)

        # Allocate memory for struct
        self.emit(Opcode.LOAD_CONST, type_info.size)
        self.emit(Opcode.CALL_FUNC, "malloc", 1)
        self.emit(Opcode.STORE_VAR, struct_inst.target)

        # Initialize fields
        for field_name, field_value in struct_inst.field_values.items():
            if isinstance(field_value, MoveNode):
                self.move_value(field_value.variable)
                
            self.generate_expression(field_value)
            offset = layout.get_offset(field_name)
            self.emit(Opcode.LOAD_VAR, struct_inst.target)
            self.emit(Opcode.LOAD_CONST, offset)
            self.emit(Opcode.ADD)
            self.emit(Opcode.STORE_INDIRECT)

    def gen_field_access(self, field_access) -> None:
        """Generate code for field access with ownership tracking"""
        struct_type = self.get_expression_type(field_access.struct_expression)
        
        # Get struct layout info
        layout = self.struct_layouts[struct_type.name]
        field_offset = layout.get_offset(field_access.field_name)
        field_type = struct_type.fields[field_access.field_name].field_type
        
        # Generate struct pointer
        self.generate_expression(field_access.struct_expression)
        
        if field_access.is_move:
            # Moving the field - mark as moved in parent struct
            struct_name = struct_type.name
            field_path = f"{struct_name}.{field_access.field_name}"
            self.move_value(field_path)
            
            # Track field for dropping if needed
            if field_type.needs_drop:
                self.emit(Opcode.TRACK_DROP, field_access.field_name, True)
            
            # Load field value
            self.emit(Opcode.LOAD_CONST, field_offset)
            self.emit(Opcode.ADD)
            self.emit(Opcode.LOAD_INDIRECT)
            
            # Move ownership
            self.emit(Opcode.MOVE, field_path, field_access.field_name)
            
        else:
            # Borrowing the field - just load address
            self.emit(Opcode.LOAD_CONST, field_offset)
            self.emit(Opcode.ADD)
            # For borrowed fields, we return the address
            if field_access.is_mutable:
                # Mutable borrow - can write through pointer
                self.emit(Opcode.LOAD_INDIRECT)
            else:
                # Immutable borrow - can only read
                self.emit(Opcode.LOAD_INDIRECT)

    def get_expression_type(self, expr) -> str:
        """Get the type name of an expression"""
        if isinstance(expr, StructInstantiation):
            return expr.struct_name
        elif isinstance(expr, Variable):
            # Look up in symbol table
            symbol = self.symbol_table.lookup(expr.name)
            return symbol.type.name if symbol else "unknown"
        return "unknown"

    def get_output(self):
        """Get the final generated code"""
        return '\n'.join(self.output)

    def gen_DomainEffect(self, node):
        """Generate code for domain effect operations build into the langauge"""
        effect_name = node.effect.name
        operation = node.operation

        # Generate arguments
        for arg in node.arguments:
            self.generate(arg)

        if operation == "create":
            # Create domain from value on stack
            self.emit(Opcode.CREATE_DOMAIN)

        elif operation == "acquire":
            # Acquire domain and get value
            self.emit(Opcode.ACQUIRE_DOMAIN)

        elif operation == "release":
            # Release domain
            self.emit(Opcode.RELEASE_DOMAIN)

        elif operation == "transfer":
            # Transfer domain to another thread
            self.emit(Opcode.TRANSFER_DOMAIN)

        # Create continuation for effect
        cont_label = self.new_label(f"domain_{operation}_cont")
        self.emit(Opcode.CREATE_CONTINUATION, cont_label)

        # Call effect handler
        self.emit(Opcode.CALL_EFFECT_HANDLER,
                 effect_name,
                 len(node.arguments),
                 cont_label)

        self.emit(Opcode.LABEL, cont_label)

    def gen_ThreadEffect(self, node):
        """Generate code for thread effect operations builtin to the language"""
        effect_name = node.effect.name
        operation = node.operation

        # Generate arguments
        for arg in node.arguments:
            self.generate(arg)

        if operation == "spawn":
            # Create closure for thread function
            closure = node.arguments[0]
            self.gen_Closure(closure)

            # Create thread
            self.emit(Opcode.CREATE_THREAD)

        elif operation == "join":
            # Join thread
            self.emit(Opcode.JOIN_THREAD)

        elif operation == "current":
            # Get current thread
            self.emit(Opcode.CURRENT_THREAD)

        elif operation == "yield":
            # Yield current thread
            self.emit(Opcode.YIELD)

        elif operation == "detach":
            # Detach thread
            self.emit(Opcode.DETACH_THREAD)

        # Create continuation for effect
        cont_label = self.new_label(f"thread_{operation}_cont")
        self.emit(Opcode.CREATE_CONTINUATION, cont_label)

        # Call effect handler
        self.emit(Opcode.CALL_EFFECT_HANDLER,
                 effect_name,
                 len(node.arguments),
                 cont_label)

        self.emit(Opcode.LABEL, cont_label)

    def gen_ThreadPoolEffect(self, node):
        """Generate code for thread pool effect operations build into the language"""
        effect_name = node.effect.name
        operation = node.operation

        # Generate arguments
        for arg in node.arguments:
            self.generate(arg)

        if operation == "submit":
            # Create closure for task
            closure = node.arguments[0]
            self.gen_Closure(closure)

            # Submit to thread pool
            self.emit(Opcode.SUBMIT_TASK)

        elif operation == "await":
            # Await future
            self.emit(Opcode.AWAIT_FUTURE)

        elif operation == "set_pool_size":
            # Set thread pool size
            self.emit(Opcode.SET_POOL_SIZE)

        # Create continuation for effect
        cont_label = self.new_label(f"pool_{operation}_cont")
        self.emit(Opcode.CREATE_CONTINUATION, cont_label)

        # Call effect handler
        self.emit(Opcode.CALL_EFFECT_HANDLER,
                 effect_name,
                 len(node.arguments),
                 cont_label)

        self.emit(Opcode.LABEL, cont_label)

    def generate_effect_operation(self, effect_op: EffectOperation) -> str:
        """Generate code for an effect operation with C runtime mapping"""
        if effect_op.c_effect:
            # Generate C runtime function declaration
            return_type = self.get_c_type(effect_op.return_type)
            params = [f"{self.get_c_type(p.type)} {p.name}" for p in effect_op.params]
            
            return f"""
{return_type} runtime_{effect_op.c_effect.lower()}({', '.join(params)}) {{
    // Call C runtime implementation
    return {effect_op.c_effect.lower()}({', '.join(p.name for p in effect_op.params)});
}}
"""
        else:
            # Normal effect operation code generation
            return self.generate_effect_op_default(effect_op)

    def generate_effect_mapping_header(self, effect_decl: EffectDeclaration) -> str:
        """Generate header declarations for mapped C runtime effects"""
        header = []
        for op in effect_decl.operations:
            if op.c_effect:
                return_type = self.get_c_type(op.return_type)
                params = [f"{self.get_c_type(p.type)} {p.name}" for p in op.params]
                header.append(f"{return_type} runtime_{op.c_effect.lower()}({', '.join(params)});")
        return '\n'.join(header)
