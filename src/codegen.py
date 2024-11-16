
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
        self.function_counter = 0
        self.functions = {}
        self.gpu_kernels = {}

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

    def gen_FunctionCall(self, node):
        for arg in node.arguments:
            self.generate(arg)
        self.instructions.append(Instruction(Opcode.CALL_FUNC, node.name))

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

    def gen_PerformEffect(self, node):
        # Generate code for effect arguments
        for arg in node.arguments:
            self.generate(arg)
        self.instructions.append(Instruction(Opcode.PERFORM_EFFECT, node.effect_name, len(node.arguments)))

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

