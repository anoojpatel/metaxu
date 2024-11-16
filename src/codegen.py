
from tape_vm import Instruction, Opcode
from symbol_table import SymbolTable, Symbol
from types import *
from ast import *
from type_checker import TypeChecker

class CodeGenerator:
    def __init__(self):
        self.instructions = []
        self.symbol_table = SymbolTable()
        self.type_checker = TypeChecker()
        self.scope_stack = []
        self.function_counter = 0
        self.label_counter = 0
        self.functions = {}
        self.gpu_kernels = {}

    def new_label(self, prefix="label"):
        label = f"{prefix}_{self.label_counter}"
        self.label_counter += 1
        return label

    def enter_scope(self, scope_name):
        self.scope_stack.append(scope_name)

    def exit_scope(self):
        return self.scope_stack.pop()

    def generate(self, node):
        method_name = f'gen_{type(node).__name__}'
        method = getattr(self, method_name, self.gen_generic)
        return method(node)

    def gen_generic(self, node):
        raise Exception(f"No gen_{type(node).__name__} method")

    def gen_Program(self, node):
        for stmt in node.statements:
            self.generate(stmt)

    def gen_Assignment(self, node):
        self.generate(node.expression)
        self.instructions.append(Instruction(Opcode.STORE_VAR, node.name))
        symbol = Symbol(node.name, self.type_checker.check(node.expression))
        self.symbol_table.define(node.name, symbol)

     def gen_LetStatement(self, node):
        self.generate(node.expression)
        self.instructions.append(Instruction(Opcode.STORE_VAR, node.name))

    def gen_VariableDeclaration(self, node):
        # Generate code for the initializer expression
        self.generate(node.initializer)
        # Store the value in a local variable
        self.instructions.append(Instruction(OpCode.STORE_VAR, node.name))

    def gen_Return(self, node):
        # Generate code for the return expression
        self.generate(node.expression)
        # The return value should be on the stack
        # Generate 'END' to exit the function
        self.instructions.append(Instruction(Opcode.END))

    def gen_Literal(self, node):
        self.instructions.append(Instruction(Opcode.LOAD_CONST, node.value))

    def gen_Variable(self, node):
        self.instructions.append(Instruction(Opcode.LOAD_VAR, node.name))

    def gen_BinaryOperation(self, node):
        left_type = self.type_checker.check(node.left)
        right_type = self.type_checker.check(node.right)
        self.generate(node.left)
        self.generate(node.right)
        if isinstance(left_type, VectorType) and isinstance(right_type, VectorType):
            if node.operator == '+':
                self.instructions.append(Instruction(Opcode.VEC_ADD))
            elif node.operator == '-':
                self.instructions.append(Instruction(Opcode.VEC_SUB))
            elif node.operator == '*':
                self.instructions.append(Instruction(Opcode.VEC_MUL))
            elif node.operator == '/':
                self.instructions.append(Instruction(Opcode.VEC_DIV))
        else:
            if node.operator == '+':
                self.instructions.append(Instruction(Opcode.ADD))
            elif node.operator == '-':
                self.instructions.append(Instruction(Opcode.SUB))
            elif node.operator == '*':
                self.instructions.append(Instruction(Opcode.MUL))
            elif node.operator == '/':
                self.instructions.append(Instruction(Opcode.DIV))
    
    def gen_Block(self, node):
        """
        Generate VM instructions for a block of code.
        """
        # Optionally enter a new scope if your language supports block scopes
        self.enter_scope('block')

        for statement in node.statements:
            self.generate(statement)

        # Optionally exit the scope
        self.exit_scope()
        
    def gen_FunctionDeclaration(self, node):
        func_label = f"func_{node.name}"
        # Save current instructions and symbol table
        current_instructions = self.instructions
        current_symbol_table = self.symbol_table
        self.instructions = []
        self.symbol_table = SymbolTable(parent=current_symbol_table)
        # Define function parameters in symbol table
        for param_name, param_type in node.params:
            self.symbol_table.define(param_name, Symbol(param_name, param_type))
        # Generate function body
        for stmt in node.body:
            self.generate(stmt)
        self.instructions.append(Instruction(Opcode.RETURN))
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
        function_label = node.name
        # Generate function object creation
        self.instructions.append(Instruction(Opcode.CREATE_FUNC, function_label))
        # Store the function object in the environment or variable
        self.instructions.append(Instruction(OpCode.STORE_VAR, node.name))

        # Generate the function code
        self.enter_scope(function_label)
        self.instructions.append(Instruction(OpCode.LABEL, function_label))
        # Function parameters are handled here
        self.gen_FunctionBody(node)
        self.exit_scope()
        self.instructions.append(Instruction(OpCode.END))

    def gen_FunctionBody(self, node):
        """
        Generate VM instructions for the body of a function.
        """
        # Handle function parameters
        for param in node.parameters:
            # Store each parameter as a local variable
            self.instructions.append(Instruction(OpCode.POP))
            self.instructions.append(Instruction(OpCode.STORE_VAR, param.name))

        # Generate instructions for the function's statements
        self.gen_Block(node.body)

    def gen_FunctionCall(self, node):
        # Generate code for the function expression
        self.generate(node.function_expr)
        # Push arguments onto the stack
        for arg in node.arguments:
            self.generate(arg)
            self.instructions.append(Instruction(Opcode.PUSH))

        # Call the function
        self.instructions.append(Instruction(Opcode.CALL_FUNC, len(node.arguments)))

    def gen_Closure(self, node):
        # Identify captured variables
        captured_vars = self.get_captured_variables(node)
        # Generate code to capture variables
        for var in captured_vars:
            self.instructions.append(Instruction(OpCode.LOAD_VAR, var))
            self.instructions.append(Instruction(OpCode.PUSH))
        # Create the function object with captured environment
        function_label = self.new_label('lambda')
        self.instructions.append(Instruction(OpCode.CREATE_CLOSURE, function_label, len(captured_vars)))
        # Store the closure
        self.instructions.append(Instruction(OpCode.STORE_VAR, node.name))

        # Generate the function code
        self.enter_scope(function_label)
        self.instructions.append(Instruction(OpCode.LABEL, function_label))
        # Function parameters and body
        self.gen_FunctionBody(node)
        self.exit_scope()
        self.instructions.append(Instruction(OpCode.END))
    
    def gen_StructInstantiation(self, node):
        for field_name, expr in node.field_initializers.items():
            self.generate(expr)
        field_names = list(node.field_initializers.keys())
        self.instructions.append(Instruction(Opcode.CREATE_STRUCT, node.struct_name, field_names))

    def gen_FieldAccess(self, node):
        self.generate(node.expression)
        self.instructions.append(Instruction(Opcode.LOAD_FIELD, node.field_name))

    def gen_VectorLiteral(self, node):
        for element in node.elements:
            self.generate(element)
        self.instructions.append(Instruction(Opcode.CREATE_VECTOR, node.base_type, node.size))

    def gen_SpawnExpression(self, node):
        self.generate(node.function_expression)
        self.instructions.append(Instruction(Opcode.SPAWN_THREAD))

    def gen_EffectHandler(self, effect_name, handler_node):
        handler_label = f"{effect_name}_handler"
        self.enter_scope(handler_label)
        self.instructions.append(Instruction(Opcode.LABEL, handler_label))

        # Generate code for the handler body
        self.gen_Block(handler_node.body)

        # Exit scope and generate END
        self.exit_scope()
        self.instructions.append(Instruction(OpCode.END))

    def gen_PerformEffect(self, node):
        # Generate code for arguments
        for arg in node.args:
            self.generate(arg)
            self.instructions.append(Instruction('PUSH'))

        # Generate unique labels for continuation and effect handler
        continuation_label = self.new_label("cont")
        effect_handler_label = f"{node.effect_name}_handler"

        # Jump to effect handler
        self.instructions.append(Instruction('JUMP', effect_handler_label))

        # Place continuation label
        self.instructions.append(Instruction('LABEL', continuation_label))
        # Generate code for effect arguments
        # for arg in node.arguments:
        #    self.generate(arg)
        # self.instructions.append(Instruction(Opcode.PERFORM_EFFECT, node.effect_name, len(node.arguments)))

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
        self.instructions.append(Instruction(Opcode.SET_HANDLER, node.effect_name, handler_instructions))
        # Generate code for the expression
        self.generate(node.expression)
        # Unset effect handler
        self.instructions.append(Instruction(Opcode.UNSET_HANDLER, node.effect_name))

    def gen_Move(self, node):
        # Handle move semantics
        self.instructions.append(Instruction(Opcode.MOVE_VAR, node.variable))
        symbol = self.symbol_table.lookup(node.variable)
        if symbol:
            symbol.invalidate()

    def gen_ToDevice(self, node):
        self.instructions.append(Instruction(Opcode.TO_DEVICE, node.variable))

    def gen_FromDevice(self, node):
        self.instructions.append(Instruction(Opcode.FROM_DEVICE, node.variable))

    # Implement other gen_* methods...

    def generate_gpu_code(self, node):
        # Placeholder for GPU code generation
        code = f"// GPU kernel code for function {node.name}\n"
        # Actual code generation logic goes here
        return code

