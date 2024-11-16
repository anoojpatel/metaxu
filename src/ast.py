class Node:
    pass

class Program(Node):
    def __init__(self, statements):
        self.statements = statements

class Statement(Node):
    pass

class Expression(Node):
    pass

class LetStatement(Node):
    def __init__(self, name, expression, mutable=False):
        self.name = name
        self.expression = expression
        self.mutable = mutable

class ReturnStatement(Node):
    def __init__(self, expression):
        self.expression = expression

class IfStatement(Node):
    def __init__(self, condition, then_body, else_body):
        self.condition = condition
        self.then_body = then_body
        self.else_body = else_body

class WhileStatement(Node):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

# --- Basic Expressions and Statements ---

class Literal(Expression):
    def __init__(self, value):
        self.value = value

class Variable(Expression):
    def __init__(self, name):
        self.name = name

class Assignment(Statement):
    def __init__(self, name, expression):
        self.name = name
        self.expression = expression

class BinaryOperation(Expression):
    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right
        

class WhileStatement(Node):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

class FunctionDeclaration(Statement):
    def __init__(self, name, params, body, is_kernel=False):
        self.name = name
        self.params = params
        self.body = body
        self.is_kernel = is_kernel

class FunctionCall(Expression):
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

# --- Advanced Features ---

# Pattern Matching
class MatchExpression(Expression):
    def __init__(self, expression, cases):
        self.expression = expression
        self.cases = cases  # List of (Pattern, Expression)

class Pattern(Node):
    pass

class LiteralPattern(Pattern):
    def __init__(self, value):
        self.value = value

class VariablePattern(Pattern):
    def __init__(self, name):
        self.name = name

class WildcardPattern(Pattern):
    def __init__(self):
        pass

class VariantPattern(Pattern):
    def __init__(self, enum_name, variant_name, patterns):
        self.enum_name = enum_name
        self.variant_name = variant_name
        self.patterns = patterns  # List of patterns for variant fields

# Higher-Order Functions
class LambdaExpression(Expression):
    def __init__(self, params, body):
        self.params = params
        self.body = body

# Algebraic Effects
class PerformEffect(Expression):
    def __init__(self, effect_name, arguments):
        self.effect_name = effect_name
        self.arguments = arguments

class HandleEffect(Expression):
    def __init__(self, effect_name, handler, expression):
        self.effect_name = effect_name
        self.handler = handler  # FunctionDeclaration or LambdaExpression
        self.expression = expression

class Resume(Expression):
    def __init__(self, value=None):
        self.value = value

# Ownership and Borrowing
class Move(Expression):
    def __init__(self, variable):
        self.variable = variable

class BorrowShared(Expression):
    def __init__(self, variable):
        self.variable = variable

class BorrowUnique(Expression):
    def __init__(self, variable):
        self.variable = variable

# Structs and Enums
class StructDefinition(Statement):
    def __init__(self, name, fields):
        self.name = name
        self.fields = fields  # List of (field_name, field_type)

class StructInstantiation(Expression):
    def __init__(self, struct_name, field_values):
        self.struct_name = struct_name
        self.field_values = field_values  # Dict of field_name: expression

class FieldAccess(Expression):
    def __init__(self, struct_expression, field_name):
        self.struct_expression = struct_expression
        self.field_name = field_name

class EnumDefinition(Statement):
    def __init__(self, name, variants):
        self.name = name
        self.variants = variants  # List of VariantDefinition

class VariantDefinition(Node):
    def __init__(self, name, fields):
        self.name = name
        self.fields = fields  # List of (field_name, field_type)

class VariantInstance(Expression):
    def __init__(self, enum_name, variant_name, field_values):
        self.enum_name = enum_name
        self.variant_name = variant_name
        self.field_values = field_values  # Dict of field_name: expression

# Multithreading
class SpawnExpression(Expression):
    def __init__(self, function_expression):
        self.function_expression = function_expression

# SIMD and GPU
class VectorLiteral(Expression):
    def __init__(self, base_type, elements):
        self.base_type = base_type
        self.elements = elements  # List of expressions

class KernelAnnotation(Node):
    def __init__(self, function_declaration):
        self.function_declaration = function_declaration

class ToDevice(Expression):
    def __init__(self, variable):
        self.variable = variable

class FromDevice(Expression):
    def __init__(self, variable):
        self.variable = variable

