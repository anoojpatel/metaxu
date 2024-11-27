from enum import Enum
from typing import List, Optional, Dict, Set, Union


class ComptimeError(Exception):
    """Error during compile-time evaluation"""
    def __init__(self, message, node=None, note=None):
        super().__init__(message)
        self.node = node
        self.note = note

class ComptimeDiagnostic:
    """Diagnostic information for compile-time errors"""
    def __init__(self, message, level="error", node=None, notes=None):
        self.message = message
        self.level = level  # "error", "warning", or "note"
        self.node = node
        self.notes = notes or []
    
    def __str__(self):
        result = f"{self.level}: {self.message}"
        if self.node and hasattr(self.node, 'location'):
            result += f"\n  --> {self.node.location}"
        for note in self.notes:
            result += f"\n  note: {note}"
        return result

class ComptimeContext:
    """Enhanced context for compile-time execution"""
    def __init__(self):
        self.variables = {}
        self.types = {}
        self.generated_code = []
        self.diagnostics = []
        self.type_cache = {}  # Cache for type reflection
        self.const_eval_depth = 0  # Track recursion depth
        self.MAX_EVAL_DEPTH = 100  # Prevent infinite recursion
    
    def emit_error(self, message, node=None, notes=None):
        diag = ComptimeDiagnostic(message, "error", node, notes)
        self.diagnostics.append(diag)
        raise ComptimeError(message, node)
    
    def emit_warning(self, message, node=None, notes=None):
        self.diagnostics.append(
            ComptimeDiagnostic(message, "warning", node, notes)
        )
    
    def enter_const_eval(self):
        """Track recursion depth for const evaluation"""
        self.const_eval_depth += 1
        if self.const_eval_depth > self.MAX_EVAL_DEPTH:
            self.emit_error("Maximum recursion depth exceeded in compile-time evaluation")
    
    def exit_const_eval(self):
        self.const_eval_depth -= 1
    
    def get_or_create_type_info(self, type_node):
        """Get or create TypeInfo with caching"""
        key = str(type_node)
        if key not in self.type_cache:
            if isinstance(type_node, str):
                self.type_cache[key] = self.types.get(type_node)
            else:
                # Handle complex type nodes (generics, etc)
                self.type_cache[key] = self.create_type_info(type_node)
        return self.type_cache[key]
    
    def create_type_info(self, type_node):
        """Create TypeInfo for complex types"""
        if isinstance(type_node, GenericType):
            base = self.get_type(type_node.base_type)
            if not base:
                self.emit_error(f"Unknown base type: {type_node.base_type}")
            
            # Instantiate generic type
            type_args = [self.get_or_create_type_info(arg) for arg in type_node.type_args]
            return self.instantiate_generic(base, type_args)
        
        return None
    
    def instantiate_generic(self, base_type, type_args):
        """Create concrete type from generic type"""
        if len(base_type.type_params) != len(type_args):
            self.emit_error(
                f"Wrong number of type arguments for {base_type.name}. "
                f"Expected {len(base_type.type_params)}, got {len(type_args)}"
            )
        
        # Create new type with substituted type parameters
        name = f"{base_type.name}[{','.join(str(arg) for arg in type_args)}]"
        fields = []
        
        # Substitute type parameters in fields
        type_param_map = dict(zip(base_type.type_params, type_args))
        for field in base_type.fields:
            new_type = self.substitute_type_params(field.type_info, type_param_map)
            fields.append(FieldInfo(field.name, new_type, field.modifiers))
        
        return TypeInfo(name, fields, base_type.interfaces)
    
    def substitute_type_params(self, type_info, type_param_map):
        """Substitute type parameters in a type"""
        if isinstance(type_info, TypeParameter):
            return type_param_map.get(type_info.name, type_info)
        elif isinstance(type_info, GenericType):
            args = [self.substitute_type_params(arg, type_param_map) 
                   for arg in type_info.type_args]
            return GenericType(type_info.base_type, args)
        return type_info

class Node:
    def __init__(self):
        self.parent = None
        self.scope = None  # For tracking lexical scope
        self.location = None  # For source locations
    
    def add_child(self, child):
        """Add a child node and set its parent"""
        if hasattr(child, 'parent'):
            child.parent = self
        return child
        
    def replace_with(self, new_node):
        """Replace this node with another in the AST"""
        if not self.parent:
            return False
        
        # Find this node in parent's children
        for attr in vars(self.parent):
            value = getattr(self.parent, attr)
            if value is self:
                # Direct reference
                setattr(self.parent, attr, new_node)
                new_node.parent = self.parent
                return True
            elif isinstance(value, list):
                # List of nodes
                try:
                    idx = value.index(self)
                    value[idx] = new_node
                    new_node.parent = self.parent
                    return True
                except ValueError:
                    continue
        return False
    
    def insert_after(self, new_nodes):
        """Insert nodes after this one in the parent's list"""
        if not self.parent:
            return False
        
        # Find this node in parent's lists
        for attr in vars(self.parent):
            value = getattr(self.parent, attr)
            if isinstance(value, list):
                try:
                    idx = value.index(self)
                    # Insert all new nodes after this one
                    for i, node in enumerate(new_nodes, 1):
                        value.insert(idx + i, node)
                        node.parent = self.parent
                    return True
                except ValueError:
                    continue
        return False
    
    def get_scope(self):
        """Get the nearest scope containing this node"""
        node = self
        while node:
            if node.scope:
                return node.scope
            node = node.parent
        return None

class Program(Node):
    def __init__(self, statements):
        super().__init__()
        self.statements = [self.add_child(stmt) for stmt in (statements or [])]
        self.scope = Scope(parent=None)  # Global scope
    
    def add_statements(self, new_stmts, position=-1):
        """Add statements at specific position (-1 for end)"""
        new_stmts = [self.add_child(stmt) for stmt in new_stmts]
        if position == -1:
            self.statements.extend(new_stmts)
        else:
            self.statements[position:position] = new_stmts

class Type(Node):
    def __init__(self):
        super().__init__()

class Statement(Node):
    def __init__(self):
        super().__init__()

class Expression(Node):
    def __init__(self):
        super().__init__()

class LetBinding(Node):
    def __init__(self, identifier, initializer, type_annotation=None, mode=None):
        super().__init__()
        self.identifier = identifier
        self.type_annotation = type_annotation
        self.initializer = self.add_child(initializer) if initializer else None
        self.mode = mode

class LetStatement(Statement):
    def __init__(self, bindings):
        super().__init__()
        self.bindings = [self.add_child(binding) for binding in bindings]

class ReturnStatement(Statement):
    def __init__(self, expression):
        super().__init__()
        self.expression = self.add_child(expression) if expression else None

    def __repr__(self):
        return f"Return({self.expression})"

class IfStatement(Statement):
    def __init__(self, condition, then_body, else_body):
        super().__init__()
        self.condition = self.add_child(condition)
        self.then_body = self.add_child(then_body)
        self.else_body = self.add_child(else_body) if else_body else None

class WhileStatement(Statement):
    """While loop control structure"""
    def __init__(self, condition, body):
        super().__init__()
        self.condition = self.add_child(condition)
        self.body = self.add_child(body)

class ForStatement(Statement):
    def __init__(self, iterator, iterable, body):
        super().__init__()
        self.iterator = iterator  # Just a string identifier
        self.iterable = self.add_child(iterable)
        self.body = self.add_child(body)

class Block(Node):
    def __init__(self, statements=None):
        super().__init__()
        self.statements = [self.add_child(stmt) for stmt in (statements or [])]
        self.scope = Scope(parent=None)  # Will be set when added to AST

    # Add statements at specific position (-1 for end)
    def add_statements(self, new_stmts, position=-1):
        if position < 0:
            position = len(self.statements)
        for stmt in new_stmts:
            self.statements.insert(position, self.add_child(stmt))
            position += 1

class Scope:
    """Represents a lexical scope in the program"""
    def __init__(self, parent=None):
        self.parent = parent
        self.symbols = {}
        self.children = []
        if parent:
            parent.children.append(self)
    
    def add_symbol(self, name, symbol):
        self.symbols[name] = symbol
    
    def lookup(self, name):
        """Look up a symbol in this scope or parent scopes"""
        current = self
        while current:
            if name in current.symbols:
                return current.symbols[name]
            current = current.parent
        return None

# --- Basic Expressions and Statements ---

class Literal(Expression):
    def __init__(self, value):
        self.value = value

    def eval(self, context):
        type_name = "int" if isinstance(self.value, int) else \
                   "float" if isinstance(self.value, float) else \
                   "bool" if isinstance(self.value, bool) else \
                   "String"
        return ComptimeValue(self.value, context.get_type(type_name))

class Variable(Expression):
    def __init__(self, name):
        self.name = name

    def eval(self, context):
        return context.get_variable(self.name)

class Assignment(Statement):
    def __init__(self, name, expression):
        self.name = name
        self.expression = expression

class BinaryOperation(Expression):
    def __init__(self, left, operator, right):
        super().__init__()
        self.left = self.add_child(left)
        self.operator = operator
        self.right = self.add_child(right)

    def eval(self, context):
        left = self.left.eval(context)
        right = self.right.eval(context)
        
        # Implement basic operations
        if self.operator == '+':
            value = left.value + right.value
        elif self.operator == '-':
            value = left.value - right.value
        elif self.operator == '*':
            value = left.value * right.value
        elif self.operator == '/':
            value = left.value / right.value
        
        return ComptimeValue(value, left.type_info)

class FunctionDeclaration(Statement):
    def __init__(self, name, params, body, return_type=None, performs=None, type_params=None):
        self.name = name
        self.params = params
        self.body = body
        self.return_type = return_type
        self.performs = performs or []  # List of effects this function performs
        self.type_params = type_params or []  # List of TypeParameter objects

class FunctionCall(Expression):
    def __init__(self, name, arguments):
        super().__init__()
        self.name = name  # String or QualifiedName
        self.arguments = [self.add_child(arg) for arg in (arguments or [])]

    def eval(self, context):
        func = context.lookup(self.name)
        if not func:
            raise Exception(f"Function {self.name} not found")
        args = [arg.eval(context) for arg in self.arguments]
        return func(*args)

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
        super().__init__()
        self.params = params
        self.body = body
        self.return_type = return_type
        self.captured_vars = set()  # Set of variable names captured from outer scope
        self.capture_modes = {}     # Map of variable name to capture mode (borrow/borrow_mut)
        self.scope = None          # Reference to the scope where lambda is defined
        self.linearity = LinearityMode.MANY  # Default to most permissive mode

    def add_capture(self, var_name, mode='borrow'):
        """Record a captured variable and its capture mode"""
        if mode not in ('borrow', 'borrow_mut'):
            raise ValueError(f"Invalid capture mode: {mode}")
        self.captured_vars.add(var_name)
        self.capture_modes[var_name] = mode
        # If we capture any mutable references, we must be separate
        if mode == 'borrow_mut':
            self.linearity = LinearityMode.SEPARATE

    def __str__(self):
        params_str = ", ".join(str(p) for p in self.params)
        captures_str = ", ".join(f"{v}[{self.capture_modes[v]}]" for v in sorted(self.captured_vars))
        lin = f" @ {self.linearity}" if self.linearity != LinearityMode.MANY else ""
        return f"lambda[captures: {captures_str}]({params_str}) -> {self.return_type or 'auto'}{lin}"

# Algebraic Effects
class EffectDeclaration(Node):
    """Definition of an effect type (e.g., effect Reader<T>)"""
    def __init__(self, name, type_params, operations):
        self.name = name                # The effect name (e.g., "Reader")
        self.type_params = type_params  # List of TypeParameter (e.g., [T])
        self.operations = operations    # List of effect operations

class EffectOperation(Node):
    """An operation in an effect (e.g., read() -> T)"""
    def __init__(self, name, params, return_type):
        self.name = name
        self.params = params
        self.return_type = return_type

class EffectExpression(Node):
    """Base class for effect expressions in performs clauses"""
    pass
class EffectApplication(EffectExpression):
    """Application of concrete types to an effect (e.g., Reader[int] or Reader[G])"""
    def __init__(self, effect_name, type_args):
        self.effect_name = effect_name  # Name of the effect (e.g., "Reader")
        self.type_args = type_args      # List of concrete types or type parameters
        
    def substitute(self, type_params, concrete_types):
        """Substitute type parameters with concrete types"""
        # e.g., Reader[G] becomes Reader[int] when G is substituted with int
        new_args = [arg.substitute(type_params, concrete_types) 
                   for arg in self.type_args]
        return EffectApplication(self.effect_name, new_args)

class PerformEffect(Expression):
    def __init__(self, effect_name, arguments):
        self.effect_name = effect_name
        self.arguments = arguments

class HandleEffect(Expression):
    def __init__(self, effect_name, handler, continuation):
        self.effect_name = effect_name
        self.handler = handler  # List of handle cases
        self.continuation = continuation  # Expression after IN

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
    def __init__(self, name, fields, type_params=None, implements=None, methods=None):
        self.name = name
        self.fields = fields
        self.type_params = type_params
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

    def eval(self, context):
        type_info = context.get_type(self.struct_name.parts[-1])
        if not type_info:
            raise Exception(f"Type {self.struct_name.parts[-1]} not found")
        
        fields = {}
        for field, value in self.field_assignments:
            fields[field] = value.eval(context)
        
        return ComptimeValue(fields, type_info)

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
    """Promotes expression up to caller scope. Must check if expression isn't
    only local to callee scope. Otherwise we will escape and leak after stack
    unwinding."""
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
    MANY = "many"
    SEPARATE = "separate"
    ONCE = "once"

    def __init__(self, mode):
        if mode not in [self.MANY, self.SEPARATE, self.ONCE]:
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
class TypeExpression(Type):
    """Base class for type expressions in the AST"""
    pass

class BasicType(TypeExpression):
    """Built-in types like int, float, etc."""
    def __init__(self, name):
        self.name = name

class GenericType:
    def __init__(self, base_type: str, type_args: List['TypeInfo']):
        self.base_type = base_type  # e.g., "List"
        self.type_args = type_args  # e.g., [IntType, StringType] for List<int, string>

class TypeParameter(TypeExpression):
    """A type parameter used in generic types (e.g., T in List<T>)"""
    def __init__(self, name, bound=None, effect=False):
        self.name = name
        self.bound = bound  # Optional bound (e.g., 'T: Display')
        self.effect = effect  # Whether this is an effect parameter

class RecursiveType(TypeExpression):
    """A recursive type definition (e.g., type Tree<T> = Node<T, List<Tree<T>>>)"""
    def __init__(self, name, type_params, body):
        self.name = name  # Name of the type
        self.type_params = type_params or []  # List of TypeParameters
        self.body = body  # The actual type expression


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
    def __init__(self, param_types, return_type, linearity=LinearityMode.MANY):
        super().__init__()
        self.param_types = param_types
        self.return_type = return_type
        self.linearity = linearity  # Track function's linearity mode

    def __str__(self):
        params = ", ".join(str(t) for t in self.param_types)
        lin = f" @ {self.linearity}" if self.linearity != LinearityMode.MANY else "" 
        if self.linearity:
            return f"(({params}) -> {self.return_type}){lin}"  
        else:
            return f"(({params}) -> {self.return_type})"

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
        self.kind = kind  # 'extends', 'implements', or 'equals', default 'subtype e.g. T: Int'

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

class ComptimeBlock(Node):
    def __init__(self, statements):
        self.statements = statements
    
    def eval(self, context):
        """Evaluate this block at compile time"""
        result = None
        for stmt in self.statements:
            if hasattr(stmt, 'eval'):
                result = stmt.eval(context)
        return result

class ComptimeFunction(FunctionDeclaration):
    def __init__(self, name, type_params, params, return_type, body, is_comptime=True):
        super().__init__(name, params, body, return_type)
        self.type_params = type_params
        self.is_comptime = is_comptime
    
    def eval(self, context, args):
        """Execute this function at compile time"""
        # Create new scope
        new_context = ComptimeContext()
        new_context.variables = context.variables.copy()
        new_context.types = context.types
        
        # Bind arguments
        for param, arg in zip(self.params, args):
            new_context.add_variable(param.name, arg)
        
        # Execute body
        return self.body.eval(new_context)

class TypeInfo(Node):
    def __init__(self, name, fields=None, interfaces=None, variants=None,is_copy=False):
        super().__init__()
        self.name = name
        self.fields = fields or []
        self.interfaces = interfaces or []
        self.variants = variants or []  # For enums
        self.is_copy = is_copy # can we blindly copy
    
    def get_field(self, name):
        return next((f for f in self.fields if f.name == name), None)
    
    def implements(self, interface):
        return interface in self.interfaces
    
    def is_enum(self):
        return bool(self.variants)

class EnumVariant:
    def __init__(self, name, fields=None):
        self.name = name
        self.fields = fields or []  # List of FieldInfo

class FieldInfo(Node):
    def __init__(self, name, type_info, modifiers=None):
        self.name = name
        self.type_info = type_info
        self.modifiers = modifiers or []
    
    def is_reference(self):
        return any(mod in ["unique", "shared", "exclusive"] for mod in self.modifiers)
    
    def is_copy(self):
        return not self.is_reference() and self.type_info.is_copy

class ComptimeValue:
    """Represents a value known at compile time"""
    def __init__(self, value, type_info):
        self.value = value
        self.type_info = type_info
    
    def __str__(self):
        return str(self.value)

    def matches_pattern(self, pattern, context):
        """Check if this value matches a pattern"""
        if isinstance(pattern, TypePattern):
            if not isinstance(self.value, TypeInfo):
                return False
            return pattern.matches(self.value, context)
        elif isinstance(pattern, LiteralPattern):
            return self.value == pattern.value
        return False

class GetType(Node):
    """Special compile-time function to get type information"""
    def __init__(self, type_name):
        self.type_name = type_name
    
    def eval(self, context):
        type_info = context.get_type(self.type_name)
        if not type_info:
            raise Exception(f"Type {self.type_name} not found")
        return ComptimeValue(type_info, context.get_type("Type"))

class TypePattern(Node):
    """Base class for type patterns"""
    def matches(self, type_info, context):
        raise NotImplementedError

class TypeNamePattern(TypePattern):
    """Matches a type by name"""
    def __init__(self, name):
        super().__init__()
        self.name = name
    
    def matches(self, type_info, context):
        return type_info.name == self.name

class GenericTypePattern(TypePattern):
    """Matches a generic type with specific type arguments"""
    def __init__(self, base_type, type_args):
        super().__init__()
        self.base_type = base_type
        self.type_args = type_args
    
    def matches(self, type_info, context):
        if not isinstance(type_info, GenericType):
            return False
        if type_info.base_type != self.base_type:
            return False
        return all(
            pattern.matches(arg, context)
            for pattern, arg in zip(self.type_args, type_info.type_args)
        )

class TypeVarPattern(TypePattern):
    """Pattern variable that captures a type"""
    def __init__(self, name):
        super().__init__()
        self.name = name
    
    def matches(self, type_info, context):
        # Always matches and captures the type
        context.add_variable(self.name, ComptimeValue(type_info, context.get_type("Type")))
        return True

class StructPattern(TypePattern):
    """Matches a struct with specific fields"""
    def __init__(self, fields=None):
        super().__init__()
        self.fields = fields or {}  # Map of field name to TypePattern
    
    def matches(self, type_info, context):
        if not hasattr(type_info, 'fields'):
            return False
        
        # Match specified fields
        for name, pattern in self.fields.items():
            field = type_info.get_field(name)
            if not field or not pattern.matches(field.type_info, context):
                return False
        return True

class EnumPattern(TypePattern):
    """Matches an enum variant"""
    def __init__(self, variant_name, fields=None):
        super().__init__()
        self.variant_name = variant_name
        self.fields = fields or []  # List of TypePattern for tuple variants
    
    def matches(self, type_info, context):
        if not hasattr(type_info, 'variants'):
            return False
        
        variant = type_info.get_variant(self.variant_name)
        if not variant:
            return False
        
        if len(self.fields) != len(variant.fields):
            return False
        
        return all(
            pattern.matches(field.type_info, context)
            for pattern, field in zip(self.fields, variant.fields)
        )
    
    def get_reachable_variants(self):
        return {self.variant_name}

class TraitPattern(TypePattern):
    """Matches any type that implements a trait"""
    def __init__(self, trait_name):
        super().__init__()
        self.trait_name = trait_name
    
    def matches(self, type_info, context):
        return any(
            iface.name == self.trait_name
            for iface in type_info.interfaces
        )
    
    def is_exhaustive(self):
        return False  # Trait patterns are never exhaustive alone

class UnionPattern(TypePattern):
    """Matches if any sub-pattern matches"""
    def __init__(self, patterns):
        super().__init__()
        self.patterns = patterns
    
    def matches(self, type_info, context):
        for pattern in self.patterns:
            case_context = ComptimeContext()
            case_context.types = context.types
            case_context.variables = context.variables.copy()
            
            if pattern.matches(type_info, case_context):
                # Update captured variables
                context.variables.update(case_context.variables)
                return True
        return False
    
    def get_reachable_variants(self):
        variants = set()
        for pattern in self.patterns:
            if hasattr(pattern, 'get_reachable_variants'):
                variants.update(pattern.get_reachable_variants())
        return variants

class WildcardPattern(TypePattern):
    """Matches any type"""
    def matches(self, type_info, context):
        return True
    
    def is_exhaustive(self):
        return True

class TypeMatchExpression(Node):
    """Compile-time pattern matching on types"""
    def __init__(self, value, cases):
        super().__init__()
        self.value = value  # Expression that evaluates to a type
        self.cases = cases  # List of (pattern, body) tuples
    
    def check_exhaustiveness(self, context):
        """Check if patterns cover all possible cases"""
        type_value = self.value.eval(context)
        if not type_value or not isinstance(type_value.value, TypeInfo):
            context.emit_error("Expected a type value")
            return False
        
        type_info = type_value.value
        
        # For enums, check if all variants are covered
        if hasattr(type_info, 'variants'):
            covered_variants = set()
            has_wildcard = False
            
            for pattern, _ in self.cases:
                if isinstance(pattern, WildcardPattern):
                    has_wildcard = True
                    break
                if hasattr(pattern, 'get_reachable_variants'):
                    covered_variants.update(pattern.get_reachable_variants())
            
            if not has_wildcard:
                all_variants = {v.name for v in type_info.variants}
                missing = all_variants - covered_variants
                if missing:
                    context.emit_error(
                        f"Non-exhaustive patterns. Missing variants: {', '.join(missing)}",
                        self,
                        ["Consider adding patterns for these variants",
                         "Or add a wildcard pattern '_' to match any remaining cases"]
                    )
                    return False
        
        return True
    
    def eval(self, context):
        # Check exhaustiveness before evaluation
        if not self.check_exhaustiveness(context):
            return None
        
        type_value = self.value.eval(context)
        type_info = type_value.value
        
        for pattern, body in self.cases:
            # Create new context for pattern bindings
            case_context = ComptimeContext()
            case_context.types = context.types
            case_context.variables = context.variables.copy()
            
            if pattern.matches(type_info, case_context):
                # Pattern matched, evaluate body with captured variables
                context.variables.update(case_context.variables)
                return body.eval(context)
        
        context.emit_error("No pattern matched the type")

def eval_type_match(self, context):
    """Evaluate a type match expression"""
    type_value = self.value.eval(context)
    if not isinstance(type_value.value, TypeInfo):
        context.emit_error("Expected a type value to match against")
    
    type_info = type_value.value
    for pattern, body in self.cases:
        case_context = ComptimeContext()
        case_context.types = context.types
        case_context.variables = context.variables.copy()
        
        if pattern.matches(type_info, case_context):
            # Update captured variables
            context.variables.update(case_context.variables)
            return body.eval(context)
    
    context.emit_error("No pattern matched the type")



class EffectApplication(EffectExpression):
    """Application of type arguments to an effect type (e.g., Reader[T])"""
    def __init__(self, effect_name, type_args):
        self.effect_name = effect_name  # Name of the effect (e.g., "Reader")
        self.type_args = type_args      # List of type arguments (e.g., [T])

class EffectReference(EffectExpression):
    """Reference to an effect parameter (e.g., E in performs E)"""
    def __init__(self, name):
        self.name = name  # Name of the effect parameter

class TypeReference(Node):
    """Reference to a type (e.g., Int)"""
    def __init__(self, name):
        self.name = name

class TypeApplication(Node):
    """Application of type parameters to a type constructor (e.g., Stack[Int])"""
    def __init__(self, type_constructor, type_args):
        self.type_constructor = type_constructor  # Name of type (e.g., "Stack")
        self.type_args = type_args               # List of types/type params
        
    def substitute(self, type_params, concrete_types):
        """Substitute type parameters with concrete types"""
        # e.g., Stack[E] becomes Stack[Int] when E is substituted with Int
        new_args = [arg.substitute(type_params, concrete_types) 
                   for arg in self.type_args]
        return TypeApplication(self.type_constructor, new_args)

class PrintStatement(Statement):
    """AST node for print statements"""
    def __init__(self, arguments=None):
        super().__init__()
        self.arguments = [self.add_child(arg) for arg in (arguments or [])]
