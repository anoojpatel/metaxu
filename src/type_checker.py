
from type_defs import *
from ast import *
from symbol_table import SymbolTable, Symbol

class TypeChecker:
    def __init__(self):
        self.symbol_table = SymbolTable()
        self.errors = []
        self.borrow_checker = BorrowChecker(self.symbol_table)

    def check(self, node):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.visit_generic)
        return method(node)

    def visit_generic(self, node):
        if hasattr(node, '__dict__'):
            for value in node.__dict__.values():
                if isinstance(value, Node):
                    self.check(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, Node):
                            self.check(item)
        else:
            pass

    # Implement visit methods for different AST nodes, checking types and ownership

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

    def visit_Variable(self, node):
        symbol = self.symbol_table.lookup(node.name)
        if symbol:
            if not symbol.is_valid():
                self.errors.append(f"Variable '{node.name}' is invalid (moved or invalidated)")
            return symbol.type
        else:
            self.errors.append(f"Variable '{node.name}' not defined")
            return None

    def visit_FunctionDeclaration(self, node):
        # Handle function parameters and body
        # Register the function in the global symbol table
        if node.name in self.symbol_table:
            self.errors.append(f"Function '{node.name}' is already defined")
            return
        self.symbol_table[node.name] = node
        # Enter a new scope for the function
        self.borrow_checker.enter_scope()
        # Declare function parameters
        for param in node.parameters:
            self.borrow_checker.declare_variable(param.name)
        # Type check the function body
        self.check(node.body)
        # Exit the function scope
        self.borrow_checker.exit_scope()

    def visit_Block(self, node):
        # Enter a new scope for the block
        self.borrow_tracker.enter_scope()

        for statement in node.statements:
            self.visit(statement)

        # Exit the block scope
        self.borrow_tracker.exit_scope()
        
    def visit_VariableAssignment(self, node):
    # For assignments like p.x = value
    if isinstance(node.target, FieldAccess):
        struct_name = node.target.expression.name
        field_name = node.target.field_name
        full_name = f"{struct_name}.{field_name}"
        if not self.borrow_checker.borrow(full_name, mutable=True):
            self.errors.append(f"Cannot assign to '{full_name}' because it is borrowed")
            
    def visit_Reference(self, node):
    # For expressions like &p.x
    if isinstance(node.expression, FieldAccess):
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

    # TODO: Implement other visit methods...

class BorrowChecker:
    def __init__(self, symbol_table):
        """Used for a single symbol_table and children symbol_tables"""
        self.symbol_table = symbol_table
        self.shared_borrows = {}  # variable_name -> count
        self.mutable_borrows = set()  # variable_names
        self.scope_stack = []  # Stack of scopes
        self.enter_scope()     # Initialize the global scope

    def enter_scope(self):
        # Each scope contains variables and borrows
        scope = {
            'variables': {},  # Map variable names to their statuses
            'borrows': {},    # Map variable names to borrow statuses
        }
        self.scope_stack.append(scope)

    def exit_scope(self):
        # Remove the current scope
        self.scope_stack.pop()

    def current_scope(self):
        # Get the current scope (top of the stack)
        return self.scope_stack[-1]
    
    def visit(self, node):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.visit_generic)
        method(node)

    def visit_generic(self, node):
        if hasattr(node, '__dict__'):
            for value in node.__dict__.values():
                if isinstance(value, Node):
                    self.visit(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, Node):
                            self.visit(item)
        else:
            pass

    def visit_BorrowShared(self, node):
        var_name = node.variable
        self.shared_borrows[var_name] = self.shared_borrows.get(var_name, 0) + 1

    def visit_BorrowUnique(self, node):
        var_name = node.variable
        if var_name in self.shared_borrows or var_name in self.mutable_borrows:
            raise Exception(f"Variable '{var_name}' is already borrowed")
        self.mutable_borrows.add(var_name)

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

