from typing import Set, Dict, List, Optional
from dataclasses import dataclass
import metaxu_ast as ast

@dataclass
class Scope:
    """Represents a stack frame scope with its variables and their modes"""
    variables: Dict[str, 'ast.ModeTypeAnnotation']
    parent: Optional['Scope'] = None
    
    def find_variable(self, name: str) -> Optional['ast.ModeTypeAnnotation']:
        """Find a variable's mode in this scope or parent scopes"""
        if name in self.variables:
            return self.variables[name]
        if self.parent:
            return self.parent.find_variable(name)
        return None

class ContinuationSafetyChecker:
    """Checks safety of continuations and effect handlers with respect to locality and mutability"""
    
    def __init__(self):
        self.current_scope = None
        self.moved_values: Set[str] = set()  # Track moved unique values
        
    def enter_scope(self):
        """Enter a new stack frame scope"""
        self.current_scope = Scope(variables={}, parent=self.current_scope)
        
    def exit_scope(self):
        """Exit current scope"""
        if self.current_scope and self.current_scope.parent:
            self.current_scope = self.current_scope.parent
            
    def add_variable(self, name: str, mode: Optional['ast.ModeTypeAnnotation']):
        """Add a variable to current scope"""
        if self.current_scope:
            self.current_scope.variables[name] = mode
            
    def mark_moved(self, name: str):
        """Mark a value as moved"""
        self.moved_values.add(name)
        
    def is_moved(self, name: str) -> bool:
        """Check if a value has been moved"""
        return name in self.moved_values

    def check_continuation_capture(self, expr: ast.Expression) -> List[str]:
        """Check if it's safe to capture an expression in a continuation"""
        errors = []
        
        if isinstance(expr, ast.VariableExpression):
            mode = self.current_scope.find_variable(expr.name)
            if mode:
                if mode.locality and mode.locality.mode == ast.LocalityMode.LOCAL:
                    errors.append(f"Cannot capture local variable '{expr.name}' in continuation")
                if mode.uniqueness and mode.uniqueness.mode == ast.UniquenessMode.EXCLUSIVE:
                    errors.append(f"Cannot capture mutable reference '{expr.name}' in continuation")
                if mode.uniqueness and mode.uniqueness.mode == ast.UniquenessMode.UNIQUE and not self.is_moved(expr.name):
                    errors.append(f"Must move unique value '{expr.name}' into continuation")
                    
        elif isinstance(expr, ast.ExclaveExpression):
            # Check that exclave expression doesn't contain local values
            sub_errors = self.check_continuation_capture(expr.expression)
            errors.extend(sub_errors)
            
        elif isinstance(expr, ast.StructExpression):
            # Check each field's value
            for field in expr.fields:
                field_errors = self.check_continuation_capture(field.value)
                errors.extend(field_errors)
                
        return errors

    def check_effect_handler(self, handler: ast.HandleEffect) -> List[str]:
        """Check if an effect handler is safe"""
        errors = []
        
        # Enter handler scope
        self.enter_scope()
        
        try:
            # Check handler body
            if isinstance(handler.handler, ast.Block):
                for stmt in handler.handler.statements:
                    stmt_errors = self.check_handler_statement(stmt)
                    errors.extend(stmt_errors)
            else:
                stmt_errors = self.check_handler_statement(handler.handler)
                errors.extend(stmt_errors)
                
            # Check handled expression
            expr_errors = self.check_continuation_capture(handler.expression)
            errors.extend(expr_errors)
                
        finally:
            # Exit handler scope
            self.exit_scope()
            
        return errors
        
    def check_handler_statement(self, stmt: ast.Statement) -> List[str]:
        """Check if a statement in an effect handler is safe"""
        errors = []
        
        if isinstance(stmt, ast.LetStatement):
            # Check let bindings
            if stmt.mode:
                self.add_variable(stmt.identifier, stmt.mode)
            value_errors = self.check_continuation_capture(stmt.initializer)
            errors.extend(value_errors)
                
        elif isinstance(stmt, ast.ExpressionStatement):
            # Check expressions
            if isinstance(stmt.expression, ast.Resume):
                # Check resume value
                if stmt.expression.value:
                    value_errors = self.check_continuation_capture(stmt.expression.value)
                    errors.extend(value_errors)
            else:
                # Check other expressions
                expr_errors = self.check_continuation_capture(stmt.expression)
                errors.extend(expr_errors)
            
        return errors

    def check_perform_expression(self, perform: ast.PerformEffect) -> List[str]:
        """Check if a perform expression is safe"""
        errors = []
        
        # Check effect arguments
        for arg in perform.arguments:
            arg_errors = self.check_continuation_capture(arg)
            errors.extend(arg_errors)
            
        # Check handler if present
        if perform.handler:
            handler_errors = self.check_effect_handler(perform.handler)
            errors.extend(handler_errors)
            
        return errors
