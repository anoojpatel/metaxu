"""
Code Generator for Metaxu.
Handles translation of AST to VM IR, including structs, types, effects, and memory layout.
"""

from typing import Dict, List, Set, Optional, Union, Tuple, Any
from dataclasses import dataclass
from tape_vm import Opcode, Instruction
from symbol_table import SymbolTable, Symbol
from types import *
from metaxu_ast import *
from type_checker import TypeChecker

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

    def generate(self, node):
        """Generate code for an AST node"""
        method = f'generate_{node.__class__.__name__}'
        if hasattr(self, method):
            return getattr(self, method)(node)
        else:
            raise NotImplementedError(f"Code generation not implemented for {node.__class__.__name__}")

    def generate_Module(self, node):
        """Generate code for a module"""
        # Skip if already generated
        if node.name in self.generated_modules:
            return []

        self.generated_modules.add(node.name)
        old_module = self.current_module
        self.current_module = node.name

        # Generate module code
        output = [f"# Module: {node.name}"]
        if node.docstring:
            output.append(f'"""{node.docstring}"""')
        
        for stmt in node.statements:
            output.extend(self.generate(stmt))

        self.current_module = old_module
        return output

    def generate_Import(self, node):
        """Generate code for an import statement"""
        # Imports are handled during module generation
        return []

    def generate_FromImport(self, node):
        """Generate code for a from-import statement"""
        # Imports are handled during module generation
        return []

    def generate_Name(self, node):
        """Generate code for a name, handling qualified names"""
        if isinstance(node.id, list):
            # Qualified name (e.g. std.io.println)
            return '.'.join(node.id)
        return node.id

    def generate_Call(self, node):
        """Generate code for a function call, handling qualified names"""
        func = self.generate(node.func)
        args = [self.generate(arg) for arg in node.args]
        return f"{func}({', '.join(args)})"

    def get_output(self):
        """Get the final generated code"""
        return '\n'.join(self.output)

    def gen_Program(self, node):
        for stmt in node.statements:
            self.generate(stmt)

    def gen_Assignment(self, node):
        self.generate(node.expression)
        self.emit(Opcode.STORE_VAR, node.name)
        symbol = Symbol(node.name, self.type_checker.check(node.expression))
        self.symbol_table.define(node.name, symbol)

    def gen_LetStatement(self, node):
        self.generate(node.expression)
        self.emit(Opcode.STORE_VAR, node.name)

    def gen_VariableDeclaration(self, node):
        self.generate(node.initializer)
        self.emit(Opcode.STORE_VAR, node.name)

    def gen_Return(self, node):
        self.generate(node.expression)
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
            }
        self.emit(op_map[node.operator])

    def gen_Block(self, node):
        self.enter_scope('block')
        for stmt in node.statements:
            self.generate(stmt)
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

    def gen_HandleExpression(self, node):
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

    def gen_PerformExpression(self, node):
        """Generate code for perform expression"""
        # Generate arguments
        for arg in node.arguments:
            self.generate(arg)
        
        # Create continuation for resume
        cont_label = self.new_label("perform_cont")
        self.emit(Opcode.CREATE_CONTINUATION, cont_label)
        
        # Call effect handler
        self.emit(Opcode.CALL_EFFECT_HANDLER, 
                 node.effect.name,
                 len(node.arguments),
                 cont_label)
        
        # Add label for continuation
        self.emit(Opcode.LABEL, cont_label)

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
            self.emit(Instruction('PUSH'))

        # Generate unique labels for continuation and effect handler
        continuation_label = self.new_label("cont")
        effect_handler_label = f"{node.effect_name}_handler"

        # Jump to effect handler
        self.emit(Instruction('JUMP', effect_handler_label))

        # Place continuation label
        self.emit(Instruction('LABEL', continuation_label))
        # Generate code for effect arguments
        # for arg in node.arguments:
        #    self.generate(arg)
        # self.emit(Instruction(Opcode.PERFORM_EFFECT, node.effect_name, len(node.arguments)))

    def gen_HandleEffect(self, node):
        handler_label = f"handler_{node.effect_name}_{self.function_counter}"
        self.function_counter += 1
        # Save current instructions and symbol table
        current_instructions = self.instructions
        current_symbol_table = self.symbol_table
        self.instructions = []
        self.symbol_table = SymbolTable(parent=current_symbol_table)
        # Generate handler code
        self.generate(node.handler)
        handler_instructions = self.instructions
        # Restore instructions and symbol table
        self.instructions = current_instructions
        self.symbol_table = current_symbol_table
        # Set up effect handler
        self.emit(Opcode.SET_HANDLER, node.effect_name, handler_instructions)
        # Generate code for the expression
        self.generate(node.expression)
        # Unset effect handler
        self.emit(Opcode.UNSET_HANDLER, node.effect_name)

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
        # Generate code for the inner expression
        self.generate(node.expression)
        
        # Store the result in a temporary variable that will be accessible
        # in both the inner and outer scopes
        temp_var = f"_exclave_result_{self.label_counter}"
        self.emit(Opcode.STORE_VAR, temp_var)
        
        # Load the variable back to make it available for the next operation
        self.emit(Opcode.LOAD_VAR, temp_var)

    def gen_FunctionDeclaration(self, node):
        """Generate code for function declaration with proper scope handling"""
        func_label = f"func_{node.name}"
        # Save current instructions and symbol table
        current_instructions = self.instructions
        current_symbol_table = self.symbol_table
        
        # Create new scope for function
        self.instructions = []
        self.symbol_table = SymbolTable(parent=current_symbol_table)
        
        # Define function parameters in symbol table
        for param_name, param_type in node.params:
            self.symbol_table.define(param_name, Symbol(param_name, param_type))
        
        # Generate function body
        self.enter_scope(func_label)
        for stmt in node.body:
            self.generate(stmt)
        self.exit_scope()
        
        # Add return if not present
        if not self.instructions or not isinstance(self.instructions[-1].opcode, Opcode.RETURN):
            self.emit(Opcode.RETURN)
        
        # Save function instructions
        function_instructions = self.instructions
        self.functions[node.name] = function_instructions
        
        # Restore instructions and symbol table
        self.instructions = current_instructions
        self.symbol_table = current_symbol_table
        
        # If kernel, generate GPU code
        if node.is_kernel:
            gpu_code = self.generate_gpu_code(node)
            self.gpu_kernels[node.name] = gpu_code

    def gen_FunctionDefinition(self, node):
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
        for param in node.parameters:
            self.emit(Opcode.POP)
            self.emit(Opcode.STORE_VAR, param.name)

        # Generate instructions for the function's statements
        self.gen_Block(node.body)

    def gen_FunctionCall(self, node):
        """Generate code for function calls with proper argument handling"""
        # Generate code for the function expression
        self.generate(node.function_expr)
        
        # Push arguments onto the stack
        for arg in node.arguments:
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
        
        # Calculate offsets for each field
        for field in struct_def.fields:
            field_type = self.type_info[field.type_annotation.name]
            layout.add_field(field.name, field_type.size, field_type.align)
        
        # Store layout for later use
        self.struct_layouts[struct_def.name] = layout
        self.type_info[struct_def.name] = TypeInfo(
            size=layout.total_size,
            align=layout.alignment,
            layout=layout
        )
        
        # Emit struct creation instruction
        self.emit(Opcode.CREATE_STRUCT, struct_def.name, [f.name for f in struct_def.fields])

    def gen_struct_instantiation(self, struct_inst) -> None:
        """Generate code for struct instantiation"""
        layout = self.struct_layouts[struct_inst.struct_name]
        type_info = self.type_info[struct_inst.struct_name]
        
        # Allocate memory for struct
        self.emit(Opcode.LOAD_CONST, type_info.size)
        self.emit(Opcode.CALL_FUNC, "malloc", 1)
        
        # Initialize fields
        for field_name, field_value in struct_inst.field_values.items():
            self.generate_expression(field_value)
            offset = layout.get_offset(field_name)
            self.emit(Opcode.LOAD_CONST, offset)
            self.emit(Opcode.ADD)
            self.emit(Opcode.STORE_VAR, f"_{field_name}")

    def gen_field_access(self, field_access) -> None:
        """Generate code for field access"""
        self.generate_expression(field_access.struct_expression)
        
        struct_type = self.get_expression_type(field_access.struct_expression)
        layout = self.struct_layouts[struct_type]
        offset = layout.get_offset(field_access.field_name)
        
        self.emit(Opcode.LOAD_CONST, offset)
        self.emit(Opcode.ADD)
        self.emit(Opcode.ACCESS_FIELD, field_access.field_name)

    def get_expression_type(self, expr) -> str:
        """Get the type name of an expression"""
        if isinstance(expr, StructInstantiation):
            return expr.struct_name
        elif isinstance(expr, Variable):
            # Look up in symbol table
            symbol = self.symbol_table.lookup(expr.name)
            return symbol.type.name if symbol else "unknown"
        return "unknown"

    def generate_statement(self, stmt) -> None:
        """Generate code for statements"""
        if isinstance(stmt, LetStatement):
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
            
            self.emit(Opcode.LABEL, start_label)
            
            # Generate condition
            self.generate_expression(stmt.condition)
            self.emit(Opcode.JUMP_IF_FALSE, end_label)
            
            # Generate body
            self.generate_statement(stmt.body)
            self.emit(Opcode.JUMP, start_label)
            
            self.emit(Opcode.LABEL, end_label)
        
        elif isinstance(stmt, ReturnStatement):
            self.generate_expression(stmt.expression)
            self.emit(Opcode.RETURN)
        
        elif isinstance(stmt, StructDefinition):
            self.gen_struct_def(stmt)
        
        elif isinstance(stmt, EffectDeclaration):
            self.gen_EffectDeclaration(stmt)
        
        elif isinstance(stmt, HandleExpression):
            self.gen_HandleExpression(stmt)
        
        elif isinstance(stmt, PerformExpression):
            self.gen_PerformExpression(stmt)
        
        else:
            raise TypeError(f"Unsupported statement type: {type(stmt)}")

    def generate_expression(self, expr) -> None:
        """Generate code for expressions"""
        if isinstance(expr, Literal):
            self.emit(Opcode.LOAD_CONST, expr.value)
        
        elif isinstance(expr, Variable):
            self.emit(Opcode.LOAD_VAR, expr.name)
        
        elif isinstance(expr, BinaryOperation):
            self.generate_expression(expr.left)
            self.generate_expression(expr.right)
            
            # Map operators to opcodes
            op_map = {
                '+': Opcode.ADD,
                '-': Opcode.SUB,
                '*': Opcode.MUL,
                '/': Opcode.DIV,
            }
            self.emit(op_map[expr.operator])
        
        elif isinstance(expr, StructInstantiation):
            self.gen_struct_instantiation(expr)
        
        elif isinstance(expr, FieldAccess):
            self.gen_field_access(expr)
        
        elif isinstance(expr, HandleExpression):
            self.gen_HandleExpression(expr)
        
        elif isinstance(expr, PerformExpression):
            self.gen_PerformExpression(expr)
        
        else:
            raise TypeError(f"Unsupported expression type: {type(expr)}")

    def generate(self, node) -> List[Instruction]:
        """Generate code for any AST node"""
        if isinstance(node, Program):
            for stmt in node.statements:
                self.generate_statement(stmt)
        else:
            self.generate_statement(node)
        return self.instructions
