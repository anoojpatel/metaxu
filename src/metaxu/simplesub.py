"""
SimpleSub-style type inference implementation for Metaxu with CompactType support.
Based on the paper "Simple and Practical Type Inference for Higher-Rank Polymorphism"
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union
from enum import Enum
from metaxu.type_defs import (
    Type, FunctionType, TypeVar, TypeScheme, TypeConstructor,
    CompactType, RecursiveType,TypeBounds, unfold_once, unify, compose_variance,
    substitute_compact, next_id
)

class Polarity(Enum):
    POSITIVE = 1   # Covariant position
    NEGATIVE = -1  # Contravariant position
    NEUTRAL = 0    # Invariant position

    def flip(self) -> 'Polarity':
        """Flip polarity (used when going under contravariant positions)"""
        if self == Polarity.POSITIVE:
            return Polarity.NEGATIVE
        elif self == Polarity.NEGATIVE:
            return Polarity.POSITIVE
        return Polarity.NEUTRAL

@dataclass
class Constraint:
    """A type constraint between two types"""
    left: CompactType
    right: CompactType
    polarity: Polarity

class VarianceInferencer:
    """Infers variance of type parameters based on their usage"""
    def __init__(self):
        self.type_var_positions: Dict[int, Set[Polarity]] = {}  # Use CompactType ID

    def record_usage(self, type_var: CompactType, polarity: Polarity):
        """Record a usage of a type variable in a certain polarity"""
        if type_var.kind != 'var':
            return
        if type_var.id not in self.type_var_positions:
            self.type_var_positions[type_var.id] = set()
        self.type_var_positions[type_var.id].add(polarity)

    def infer_variance(self, type_var: CompactType) -> str:
        """Infer variance for a type variable based on recorded positions"""
        if type_var.kind != 'var':
            return 'invariant'
            
        positions = self.type_var_positions.get(type_var.id, set())
        
        # If never used, assume invariant
        if not positions:
            return 'invariant'
            
        # If used in both positive and negative positions -> invariant
        if Polarity.POSITIVE in positions and Polarity.NEGATIVE in positions:
            return 'invariant'
            
        # If only used in positive positions -> covariant
        if Polarity.POSITIVE in positions:
            return 'covariant'
            
        # If only used in negative positions -> contravariant
        if Polarity.NEGATIVE in positions:
            return 'contravariant'
            
        return 'invariant'

class TypeInferencer:
    """Type inference using CompactTypes"""
    def __init__(self):
        self.constraints: List[Constraint] = []
        self.type_vars: Dict[str, CompactType] = {}
        self.next_var_id = 0
        self.variance_inferencer = VarianceInferencer()
        
    def to_compact_type(self, ty: Type) -> CompactType:
        """Convert Type to CompactType"""
        if isinstance(ty, TypeVar):
            if ty.name not in self.type_vars:
                self.type_vars[ty.name] = CompactType(
                    id=next_id(),
                    kind='var',
                    bounds=TypeBounds()
                )
            return self.type_vars[ty.name]
            
        elif isinstance(ty, TypeConstructor):
            return CompactType(
                id=next_id(),
                kind='constructor',
                constructor=ty,
                type_args=[]
            )
            
        elif isinstance(ty, RecursiveType):
            compact = CompactType(
                id=next_id(),
                kind='recursive',
                recursive_ref=ty.type_name,
                type_args=[self.to_compact_type(param) 
                          for param in ty.type_parameters]
            )
            if ty.get_resolved_type():
                resolved = self.to_compact_type(ty.get_resolved_type())
                compact.bounds = TypeBounds(upper_bound=resolved)
            return compact
            
        return CompactType(
            id=next_id(),
            kind='var',
            bounds=TypeBounds()
        )

    def analyze_type_definition(self, type_def: Type, polarity: Polarity = Polarity.NEUTRAL):
        """Analyze a type definition to infer variance of its type parameters"""
        compact = self.to_compact_type(type_def)
        self._analyze_type(compact, polarity)

    def check_function_type(self, func_type: CompactType, polarity: Polarity) -> CompactType:
        """Check a function type, handling contravariance in argument positions"""
        if not isinstance(func_type, FunctionType):
            return func_type
            
        # Arguments are contravariant
        param_types = [self._analyze_type(param, polarity.flip()) 
                      for param in func_type.param_types]
                      
        # Return type is covariant
        return_type = self._analyze_type(func_type.return_type, polarity)
        
        return FunctionType(param_types, return_type, func_type.linearity)

    def _analyze_type(self, ty: CompactType, polarity: Polarity):
        """Analyze a type to record variable usage positions"""
        ty = ty.find()
        
        if ty.kind == 'var':
            self.variance_inferencer.record_usage(ty, polarity)
            
        elif ty.kind == 'constructor':
            for arg, param_variance in zip(ty.type_args or [],
                                        get_constructor_variances(ty.constructor)):
                composed = compose_variance(polarity, param_variance)
                self._analyze_type(arg, composed)
                
        elif ty.kind == 'function':
            # Check function body with proper variance
            body_type = self._analyze_type(ty.body, polarity) if hasattr(ty, 'body') else None
            
            # Check function type itself
            return self.check_function_type(ty, polarity)

        elif ty.kind == 'recursive':
            # Analyze parameters
            for param in (ty.type_args or []):
                self._analyze_type(param, polarity)
            # Analyze resolved type if available
            if ty.bounds and ty.bounds.upper_bound:
                self._analyze_type(ty.bounds.upper_bound, polarity)

    def fresh_type_var(self, name_hint: str = "T") -> CompactType:
        """Create a fresh type variable"""
        var = CompactType(
            id=next_id(),
            kind='var',
            bounds=TypeBounds()
        )
        self.type_vars[name_hint] = var
        return var

    def add_constraint(self, left: CompactType, right: CompactType, 
                      polarity: Polarity):
        """Add a new constraint to the system"""
        self.constraints.append(Constraint(left, right, polarity))

    def solve_constraints(self) -> Dict[CompactType, CompactType]:
        """Solve the collected constraints using unification"""
        solution = {}
        
        for constraint in self.constraints:
            left = constraint.left.find()
            right = constraint.right.find()
            
            # Handle recursive types
            if left.kind == 'recursive' or right.kind == 'recursive':
                # Unfold recursive types once
                if left.kind == 'recursive':
                    left = unfold_once(left)
                if right.kind == 'recursive':
                    right = unfold_once(right)
            
            # Unify with appropriate variance
            variance = 'covariant' if constraint.polarity == Polarity.POSITIVE else \
                      'contravariant' if constraint.polarity == Polarity.NEGATIVE else \
                      'invariant'
            
            if not unify(left, right, variance):
                raise TypeError(f"Cannot unify {left} with {right}")
                
            # Record solution
            if left.kind == 'var':
                solution[left] = right
            elif right.kind == 'var':
                solution[right] = left
                
        return solution

    def finalize_type_definition(self, type_def: TypeConstructor):
        """Finalize a type definition by inferring variance for all its type parameters"""
        for param in type_def.type_params:
            compact_param = self.type_vars.get(param.name)
            if compact_param:
                variance = self.variance_inferencer.infer_variance(compact_param)
                param.inferred_variance = variance

    def infer_expression(self, expr: 'ast.Expression', polarity: Polarity) -> CompactType:
        """Infer type of an expression, generating constraints with polarity"""
        if isinstance(expr, ast.Lambda):
            param_type = self.fresh_type_var("param")
            # Parameters are contravariant
            self._analyze_type(param_type, polarity.flip())
            
            body_type = self.infer_expression(expr.body, polarity)
            # Return type is covariant
            self._analyze_type(body_type, polarity)
            
            return CompactType(
                id=next_id(),
                kind='function',
                param_types=[param_type],
                return_type=body_type
            )
            
        elif isinstance(expr, ast.Application):
            func_type = self.infer_expression(expr.func, polarity)
            # Arguments are in contravariant position
            arg_type = self.infer_expression(expr.arg, polarity.flip())
            result_type = self.fresh_type_var("result")
            
            self.add_constraint(
                func_type,
                CompactType(
                    id=next_id(),
                    kind='function',
                    param_types=[arg_type],
                    return_type=result_type
                ),
                polarity
            )
            return result_type
            
        elif isinstance(expr, ast.TypeApplication):
            # Analyze how type arguments are used
            constructor = self.infer_expression(expr.constructor, polarity)
            for arg in expr.type_args:
                self._analyze_type(arg, polarity)
                
            return CompactType(
                id=next_id(),
                kind='constructor',
                constructor=constructor,
                type_args=[self.to_compact_type(arg) for arg in expr.type_args]
            )
            
        return self.fresh_type_var("unknown")

    def generalize(self, ty: CompactType, env: Dict[str, Type]) -> TypeScheme:
        """Generalize a type into a type scheme by quantifying free variables"""
        free_vars = self._free_vars(ty) - self._free_vars_env(env)
        return TypeScheme(list(free_vars), ty)

    def _free_vars(self, ty: CompactType) -> Set[CompactType]:
        """Collect free type variables in a type"""
        if ty.kind == 'var':
            return {ty}
        elif ty.kind == 'function':
            vars_params = set().union(*(self._free_vars(p) for p in ty.param_types))
            return vars_params | self._free_vars(ty.return_type)
        return set()

    def _free_vars_env(self, env: Dict[str, Type]) -> Set[CompactType]:
        """Collect free type variables in an environment"""
        return set().union(*(self._free_vars(self.to_compact_type(ty)) for ty in env.values()))
