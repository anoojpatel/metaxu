from typing import Set, Dict, List, Optional
from dataclasses import dataclass
import metaxu_ast as ast

@dataclass
class Scope:
    """Represents a stack frame scope with its variables and their modes"""
    variables: Dict[str, 'ast.Mode']
    parent: Optional['Scope'] = None
    
    def find_variable(self, name: str) -> Optional['ast.Mode']:
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
            
    def add_variable(self, name: str, mode: Optional['ast.Mode']):
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
                if mode.is_local:
                    errors.append(f"Cannot capture local variable '{expr.name}' in continuation")
                if mode.is_mut:
                    errors.append(f"Cannot capture mutable reference '{expr.name}' in continuation")
                if not mode.is_const and not self.is_moved(expr.name):
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

    def check_effect_handler(self, handler: ast.EffectHandler) -> List[str]:
        """Check if an effect handler is safe"""
        errors = []
        
        # Enter handler scope
        self.enter_scope()
        
        try:
            # Check continuation parameter
            k_name = handler.continuation_param
            self.add_variable(k_name, None)  # Continuation has no mode
            
            # Check handler body
            if isinstance(handler.body, ast.Block):
                for stmt in handler.body.statements:
                    stmt_errors = self.check_handler_statement(stmt)
                    errors.extend(stmt_errors)
            else:
                stmt_errors = self.check_handler_statement(handler.body)
                errors.extend(stmt_errors)
                
        finally:
            # Exit handler scope
            self.exit_scope()
            
        return errors
        
    def check_handler_statement(self, stmt: ast.Statement) -> List[str]:
        """Check if a statement in an effect handler is safe"""
        errors = []
        
        if isinstance(stmt, ast.LetStatement):
            # Check let bindings
            for binding in stmt.bindings:
                self.add_variable(binding.name, binding.mode)
                value_errors = self.check_continuation_capture(binding.value)
                errors.extend(value_errors)
                
        elif isinstance(stmt, ast.ExpressionStatement):
            # Check expressions
            if isinstance(stmt.expression, ast.CallExpression):
                # Special handling for continuation calls
                if isinstance(stmt.expression.function, ast.VariableExpression):
                    if stmt.expression.function.name == self.current_scope.variables.get('k'):
                        # This is a continuation call - check arguments
                        for arg in stmt.expression.arguments:
                            arg_errors = self.check_continuation_capture(arg)
                            errors.extend(arg_errors)
                            
        return errors

    def check_perform_expression(self, perform: ast.PerformExpression) -> List[str]:
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
