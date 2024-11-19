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
    def __init__(self, identifier, initializer, mode=None):
        self.identifier = identifier
        self.initializer = initializer
        self.mode = mode

class ReturnStatement(Node):
    def __init__(self, expression):
        self.expression = expression

    def __repr__(self):
        return f"ReturnStatement({self.expression})"

class IfStatement(Node):
    def __init__(self, condition, then_body, else_body):
        self.condition = condition
        self.then_body = then_body
        self.else_body = else_body

class WhileStatement(Node):
    """While loop control structure"""
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

class ForStatement(Statement):
    def __init__(self, iterator, iterable, body):
        self.iterator = iterator
        self.iterable = iterable
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
        

class FunctionDeclaration(Statement):
    def __init__(self, name, params, body, return_type=None, is_kernel=False):
        self.name = name
        self.params = params
        self.body = body
        self.return_type = return_type
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
    def __init__(self, params, body, return_type=None):
        self.params = params
        self.body = body
        self.return_type = return_type

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
class StructDefinition(Node):
    def __init__(self, name, fields, implements=None, methods=None):
        self.name = name
        self.fields = fields
        self.implements = implements
        self.methods = methods if methods is not None else []

class StructField(Node):
    def __init__(self, name, type_expr=None, value=None, visibility=None):
        self.name = name
        self.type_expr = type_expr
        self.value = value
        self.visibility = visibility

class StructInstantiation(Node):
    def __init__(self, struct_name, field_assignments):
        super().__init__()
        self.struct_name = struct_name  # QualifiedName
        self.field_assignments = field_assignments  # List of (field_name, value) tuples

    def __str__(self):
        fields_str = ", ".join(f"{field}={value}" for field, value in self.field_assignments)
        return f"{self.struct_name}{{{fields_str}}}"

class FieldAccess(Node):
    def __init__(self, base, fields):
        super().__init__()
        self.base = base  # Expression
        self.fields = fields  # List of field names

    def __str__(self):
        return f"{self.base}.{'.'.join(self.fields)}"

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
    def __init__(self, base_type, size, elements):
        self.base_type = base_type
        self.size = size
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

# Locality and Regions
class LocalDeclaration(Node):
    def __init__(self, variable, type_annotation=None):
        self.variable = variable
        self.type_annotation = type_annotation

class ExclaveExpression(Expression):
    def __init__(self, expression):
        self.expression = expression

class LocalParameter(Node):
    def __init__(self, name, type_annotation=None):
        self.name = name
        self.type_annotation = type_annotation

# Function Parameters
class Parameter(Node):
    def __init__(self, name, type=None, mode=None):
        self.name = name
        self.type = type
        self.mode = mode

# Mode System
class ModeAnnotation(Node):
    def __init__(self, mode_type, base_expression):
        self.mode_type = mode_type  # UniquenessMode, LocalityMode, or LinearityMode
        self.base_expression = base_expression

class UniquenessMode(Node):
    UNIQUE = "unique"
    EXCLUSIVE = "exclusive"
    SHARED = "shared"

    def __init__(self, mode):
        if mode not in [self.UNIQUE, self.EXCLUSIVE, self.SHARED]:
            raise ValueError(f"Invalid uniqueness mode: {mode}")
        self.mode = mode

class LocalityMode(Node):
    LOCAL = "local"
    GLOBAL = "global"

    def __init__(self, mode):
        if mode not in [self.LOCAL, self.GLOBAL]:
            raise ValueError(f"Invalid locality mode: {mode}")
        self.mode = mode

class LinearityMode(Node):
    ONCE = "once"
    SEPARATE = "separate"
    MANY = "many"

    def __init__(self, mode):
        if mode not in [self.ONCE, self.SEPARATE, self.MANY]:
            raise ValueError(f"Invalid linearity mode: {mode}")
        self.mode = mode

class BorrowExpression(Expression):
    def __init__(self, variable, mode):
        self.variable = variable
        self.mode = mode  # UniquenessMode

class ModeTypeAnnotation(Node):
    def __init__(self, base_type, uniqueness=None, locality=None, linearity=None):
        self.base_type = base_type
        self.uniqueness = uniqueness
        self.locality = locality
        self.linearity = linearity

class StructFieldDefinition(Node):
    def __init__(self, name, type_annotation, is_exclusively_mutable=False):
        self.name = name
        self.type_annotation = type_annotation
        self.is_exclusively_mutable = is_exclusively_mutable

# Type System
class TypeExpression(Node):
    """Base class for type expressions in the AST"""
    pass

class BasicType(TypeExpression):
    """Built-in types like int, float, etc."""
    def __init__(self, name):
        self.name = name

class TypeParameter(TypeExpression):
    """A type parameter used in generic types (e.g., T in List<T>)"""
    def __init__(self, name, bounds=None):
        self.name = name
        self.bounds = bounds or []  # List of types that bound this parameter

class RecursiveType(TypeExpression):
    """A recursive type definition (e.g., type Tree<T> = Node<T, List<Tree<T>>>)"""
    def __init__(self, name, type_params, body):
        self.name = name  # Name of the type
        self.type_params = type_params or []  # List of TypeParameters
        self.body = body  # The actual type expression

class TypeApplication(TypeExpression):
    """Application of type arguments to a generic type (e.g., List<int>)"""
    def __init__(self, base_type, type_args):
        self.base_type = base_type  # The generic type being instantiated
        self.type_args = type_args  # List of type arguments

class InterfaceType(TypeExpression):
    """Represents a reference to an interface type"""
    def __init__(self, name, type_args=None):
        self.name = name
        self.type_args = type_args or []

class TypeDefinition(Statement):
    """Top-level type definition"""
    def __init__(self, name, type_params, body, modes=None):
        self.name = name
        self.type_params = type_params or []
        self.body = body
        self.modes = modes or []

class FunctionType(TypeExpression):
    def __init__(self, param_types, return_type):
        self.param_types = param_types
        self.return_type = return_type

# Interface and Implementation
class InterfaceDefinition(Node):
    """Interface definition with methods and associated types"""
    def __init__(self, name, type_params, methods, extends=None):
        self.name = name
        self.type_params = type_params or []  # List of TypeParameter
        self.methods = methods  # List of MethodDefinition
        self.extends = extends or []  # List of interfaces this one extends

class MethodDefinition(Node):
    """Method definition in an interface"""
    def __init__(self, name, params, return_type, type_params=None):
        self.name = name
        self.params = params  # List of Parameter
        self.return_type = return_type
        self.type_params = type_params or []  # List of TypeParameter

class Implementation(Node):
    """Implementation of an interface for a type"""
    def __init__(self, interface_name, type_name, type_params, methods, where_clause=None):
        self.interface_name = interface_name
        self.type_name = type_name
        self.type_params = type_params or []  # List of TypeParameter
        self.methods = methods  # List of MethodImplementation
        self.where_clause = where_clause  # Optional type constraints

class MethodImplementation(Node):
    """Concrete implementation of an interface method"""
    def __init__(self, name, params, body, type_params=None, return_type=None):
        self.name = name
        self.params = params  # List of Parameter
        self.body = body  # Block of statements
        self.type_params = type_params or []  # List of TypeParameter
        self.return_type = return_type  # Optional return type

class WhereClause(Node):
    """Type constraints in implementations and generic functions"""
    def __init__(self, constraints):
        self.constraints = constraints  # List of TypeConstraint

class TypeConstraint(Node):
    """A constraint on a type parameter"""
    def __init__(self, type_param, bound_type, kind='extends'):
        self.type_param = type_param  # TypeParameter
        self.bound_type = bound_type  # Type that bounds this parameter
        self.kind = kind  # 'extends', 'implements', or 'equals'

class TypeAlias(Node):
    """Type alias definition"""
    def __init__(self, name, type_expr, type_params=None):
        self.name = name
        self.type_expr = type_expr
        self.type_params = type_params or []  # List of TypeParameter

# Module imports
class Import(Node):
    """Import entire module"""
    def __init__(self, module_path, alias=None, is_public=False):
        """
        Args:
            module_path (list[str]): List of module path components, e.g. ["std", "io"] for 'import std.io'
            alias (str, optional): Optional alias for the imported module, e.g. 'io' in 'import std.io as io'
            is_public (bool): Whether this import should be re-exported from the module
        """
        self.module_path = module_path
        self.alias = alias
        self.is_public = is_public

class FromImport(Node):
    """Import specific items from a module"""
    def __init__(self, module_path, names, relative_level=0, is_public=False):
        """
        Args:
            module_path (list[str]): List of module path components, e.g. ["std", "io"] for 'from std.io import ...'
            names (list[tuple[str, str]]): List of (name, alias) tuples, e.g. [("x", "y"), ("z", None)] for 'import x as y, z'
            relative_level (int): Number of dots for relative imports, e.g. 2 for '..module'
            is_public (bool): Whether these imports should be re-exported from the module
        """
        self.module_path = module_path
        self.names = names
        self.relative_level = relative_level
        self.is_public = is_public

class VisibilityRules(Node):
    def __init__(self, rules):
        self.rules = rules  # Dictionary mapping identifiers to visibility levels

    def __str__(self):
        rules_str = ", ".join(f"{id}: {vis}" for id, vis in self.rules.items())
        return f"VisibilityRules({rules_str})"

class ModuleBody(Node):
    """Container for module contents including docstring, exports, and visibility rules"""
    def __init__(self, statements, docstring=None, exports=None, visibility_rules=None):
        """
        Args:
            statements (list[Node]): List of statements in the module
            docstring (str, optional): Optional module documentation
            exports (list[tuple[str, str]], optional): List of (name, alias) tuples for exported symbols
            visibility_rules (dict[str, str], optional): Visibility rules for module items
        """
        self.statements = statements
        self.docstring = docstring
        self.exports = exports if exports is not None else []
        self.visibility_rules = visibility_rules

    def __str__(self):
        return f"ModuleBody(statements={self.statements}, docstring={self.docstring}, exports={self.exports}, visibility_rules={self.visibility_rules})"

class Module(Node):
    """A module containing statements and declarations"""
    def __init__(self, name, body, parent=None):
        """
        Args:
            name (str): Module name (not fully qualified)
            body (ModuleBody): Module contents
            parent (Module, optional): Parent module if this is a nested module
        """
        self.name = name
        self.body = body
        self.parent = parent
        self._children = {}  # Map of child module names to Module objects

    def add_child(self, child_module):
        """Add a child module to this module"""
        self._children[child_module.name] = child_module
        child_module.parent = self

    def get_child(self, name):
        """Get a child module by name"""
        return self._children.get(name)

    def get_full_name(self):
        """Get the fully qualified module name"""
        if self.parent:
            return f"{self.parent.get_full_name()}.{self.name}"
        return self.name

    def __str__(self):
        return f"Module(name={self.get_full_name()}, body={self.body})"

class RelativePath(Node):
    """Represents a relative module path like '.module' or '...module'"""
    def __init__(self, level, path):
        """
        Args:
            level (int): Number of dots in the relative path (1 for '.', 2 for '..', etc.)
            path (str): The module path after the dots
        """
        self.level = level
        self.path = path

class QualifiedName(Node):
    def __init__(self, parts):
        super().__init__()
        self.parts = parts  # List of name parts

    def __str__(self):
        return '.'.join(self.parts)

class QualifiedFunctionCall(Node):
    def __init__(self, parts, arguments):
        super().__init__()
        self.parts = parts  # List of name parts
        self.arguments = arguments  # List of expressions

    def __str__(self):
        return f"{'.'.join(self.parts)}({', '.join(str(arg) for arg in self.arguments)})"

class Block(Node):
    def __init__(self, statements):
        super().__init__()
        self.statements = statements if statements is not None else []

    def __repr__(self):
        return f"Block({self.statements})"
