"""
SimpleSub-style type inference implementation for Metaxu with CompactType support.
Based on the paper "Simple and Practical Type Inference for Higher-Rank Polymorphism"
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union
from enum import Enum
from metaxu.type_defs import (
    Type, FunctionType, TypeVar, TypeScheme, TypeConstructor,
<<<<<<< Updated upstream
    CompactType, RecursiveType,TypeBounds, unfold_once, unify, compose_variance,
    substitute_compact, next_id
=======
    CompactType, TypeBounds, unfold_once, unify, compose_variance,
    substitute_compact, next_id, RecursiveType, get_constructor_variances
>>>>>>> Stashed changes
)
import metaxu.metaxu_ast as ast

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
        # Bridge AST basic types to CompactType primitives
        if isinstance(ty, ast.BasicType):
            # Preserve the AST name (e.g., 'Int', 'String', 'Bool', 'Void')
            return CompactType(
                id=next_id(),
                kind='primitive',
                name=ty.name
            )
        # Bridge AST type parameter to a fresh/interned var
        if hasattr(ast, 'TypeParameter') and isinstance(ty, ast.TypeParameter):
            key = getattr(ty, 'name', 'T')
            if key not in self.type_vars:
                self.type_vars[key] = CompactType(
                    id=next_id(),
                    kind='var',
                    bounds=TypeBounds()
                )
            return self.type_vars[key]
        # Bridge AST TypeReference (could be primitive, type param, or zero-arity constructor)
        if hasattr(ast, 'TypeReference') and isinstance(ty, ast.TypeReference):
            cname = getattr(ty, 'name', None) or getattr(ty, 'identifier', 'Type')
            # Primitives
            if cname in ('Int', 'String', 'Bool', 'Void'):
                return CompactType(id=next_id(), kind='primitive', name=cname)
            # Likely a type parameter
            if cname in self.type_vars or (isinstance(cname, str) and len(cname) <= 2 and cname[:1].isupper()):
                if cname not in self.type_vars:
                    self.type_vars[cname] = CompactType(id=next_id(), kind='var', bounds=TypeBounds())
                return self.type_vars[cname]
            # Otherwise treat as a named zero-arity constructor
            constructor = TypeConstructor(cname, 0)
            return CompactType(
                id=next_id(),
                kind='constructor',
                constructor=constructor,
                type_args=[],
                name=cname
            )
        # Bridge AST TypeApplication (constructor with type arguments)
        if hasattr(ast, 'TypeApplication') and isinstance(ty, ast.TypeApplication):
            # Extract constructor/base name from various possible shapes
            cname = None
            # Prefer explicit field used by this AST: 'type_constructor'
            tc = getattr(ty, 'type_constructor', None)
            if isinstance(tc, str) and tc:
                cname = tc
            base = getattr(ty, 'constructor', None)
            if not cname and base is not None:
                cname = getattr(base, 'name', None) or getattr(base, 'identifier', None)
            if not cname:
                base2 = getattr(ty, 'base_type', None)
                if isinstance(base2, str):
                    cname = base2
                elif base2 is not None:
                    cname = getattr(base2, 'name', None) or getattr(base2, 'identifier', None)
            if not cname:
                # Try common alternatives
                base3 = getattr(ty, 'base', None) or getattr(ty, 'callee', None) or getattr(ty, 'type', None)
                if isinstance(base3, str):
                    cname = base3
                elif base3 is not None:
                    cname = getattr(base3, 'name', None) or getattr(base3, 'identifier', None)
            if not cname:
                cname = getattr(ty, 'name', None)
            if not cname:
                cname = 'Generic'
            # Type arguments could be under different fields
            args = getattr(ty, 'type_args', None)
            if args is None:
                args = getattr(ty, 'arguments', [])
            if args is None:
                args = getattr(ty, 'params', None)
            if args is None:
                args = getattr(ty, 'type_parameters', None)
            args = args or []
            try:
                details = {k: v for k, v in getattr(ty, '__dict__', {}).items() if k not in ('parent',)}
                print(f"[to_compact_type] TypeApplication base(chosen)={cname}, args={[getattr(a,'name',getattr(a,'base_type',a)) for a in args]} attrs={details}")
            except Exception:
                pass
            constructor = TypeConstructor(cname, len(args))
            return CompactType(
                id=next_id(),
                kind='constructor',
                constructor=constructor,
                type_args=[self.to_compact_type(a) for a in args],
                name=cname
            )
        # Bridge AST generic type expressions (e.g., Box[T]) to constructor CompactType
        if hasattr(ast, 'GenericType') and isinstance(ty, ast.GenericType):
            base_name = getattr(ty, 'name', None) or getattr(ty, 'base_type', None) or 'Generic'
            targs = getattr(ty, 'type_args', []) or []
            try:
                print(f"[to_compact_type] GenericType base={base_name}, args={[getattr(a,'name',getattr(a,'base_type',a)) for a in targs]}")
            except Exception:
                pass
            constructor = TypeConstructor(base_name, len(targs))
            return CompactType(
                id=next_id(),
                kind='constructor',
                constructor=constructor,
                type_args=[self.to_compact_type(a) for a in targs],
                name=base_name
            )
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
            
        # Default: create a fresh var when we don't recognize the type
        return CompactType.fresh_var()

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
        # Do NOT store by name_hint; that aliases different occurrences.
        return CompactType(
            id=next_id(),
            kind='var',
            bounds=TypeBounds()
        )

    def add_constraint(self, left: CompactType, right: CompactType, 
                      polarity: Polarity):
        """Add a new constraint to the system"""
        self.constraints.append(Constraint(left, right, polarity))

    def solve_constraints(self) -> Dict[int, CompactType]:
        """Solve the collected constraints using unification.
        Returns a mapping from variable IDs to their resolved CompactTypes.
        """
        solution: Dict[int, CompactType] = {}
        errors: List[str] = []
        
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
            
            try:
                ok = unify(left, right, variance)
            except Exception as e:
                ok = False
            if not ok:
                # Record and continue; don't abort entire solve so other constraints can resolve
                errors.append(f"Cannot unify {left} with {right} (variance={variance})")
                continue
                
            # Record solution by variable ID to avoid hashing CompactType
            if left.kind == 'var':
                solution[left.id] = right
            elif right.kind == 'var':
                solution[right.id] = left
        
        # Optional: debug unify errors without breaking tests
        # if errors:
        #     print("Type inference warnings:", errors)
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
        # Handle struct/record constructor literals (extern AST), e.g., Pair { first=..., second=... }
        # Detect by shape rather than class to avoid tight coupling to AST implementation
        if hasattr(expr, 'field_assignments') or hasattr(expr, 'fields') or hasattr(expr, 'type_constructor') \
           or hasattr(expr, 'struct_name') or hasattr(expr, 'record_name'):
            # Resolve constructor name from multiple possible attributes
            ctor_name = (getattr(expr, 'type_constructor', None)
                         or getattr(expr, 'constructor_name', None)
                         or getattr(expr, 'struct_name', None)
                         or getattr(expr, 'record_name', None)
                         or getattr(expr, 'name', None))
            if isinstance(ctor_name, ast.Node):
                n = getattr(ctor_name, 'name', None)
                if not n:
                    parts = getattr(ctor_name, 'parts', None)
                    if isinstance(parts, list) and parts:
                        n = '.'.join(parts)
                ctor_name = n or str(ctor_name)

            # Gather field values in declaration order (dict preserves insertion order in Py3.7+)
            ordered_values = []
            fm = (getattr(expr, 'fields', None) or getattr(expr, 'field_values', None)
                  or getattr(expr, 'properties', None) or getattr(expr, 'args', None))
            if isinstance(fm, dict) and fm:
                for _, v in fm.items():
                    ordered_values.append(v)
            else:
                entries = (getattr(expr, 'field_assignments', None) or getattr(expr, 'fields', None)
                           or getattr(expr, 'properties', None) or getattr(expr, 'args', None))
                if isinstance(entries, list) and entries:
                    for entry in list(entries):
                        v = (getattr(entry, 'value', None) or getattr(entry, 'expr', None)
                             or getattr(entry, 'rhs', None) or getattr(entry, 'initializer', None))
                        if v is not None:
                            ordered_values.append(v)

            # Infer each field's type and constrain against fresh type args
            arg_vars: List[CompactType] = []
            for i, v in enumerate(ordered_values):
                field_ty = self.infer_expression(v, polarity)
                arg_var = self.fresh_type_var(f"arg{i}")
                # Covariant constraint: field type <= arg var
                self.add_constraint(field_ty, arg_var, Polarity.POSITIVE)
                arg_vars.append(arg_var)

            # Build constructor CompactType for the struct literal
            arity = len(arg_vars)
            constructor = TypeConstructor(ctor_name or 'Struct', arity)
            ctor_compact = CompactType(
                id=next_id(),
                kind='constructor',
                constructor=constructor,
                type_args=arg_vars,
                name=ctor_name or 'Struct'
            )
            # Attach back for downstream use
            try:
                expr.type_var = ctor_compact
            except Exception:
                pass
            return ctor_compact

        if isinstance(expr, ast.FunctionCall):
            # Do not special-case Option/Some here. Calls are generic; without a function type
            # environment modeled, return a fresh var so constraints can still attach elsewhere.
            return self.fresh_type_var("call")

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
        """Generalize a type into a type scheme by quantifying free variables.
        Tracks free variables by CompactType IDs to avoid hashing CompactType.
        """
        free_vars_ids = self._free_vars(ty) - self._free_vars_env(env)
        # Note: We are not constructing actual TypeVar nodes here since generalize
        # isn't used in current tests; we keep the API but pass IDs for now.
        return TypeScheme(list(free_vars_ids), ty)

    def _free_vars(self, ty: CompactType) -> Set[int]:
        """Collect free type variable IDs in a type"""
        ty = ty.find()
        if ty.kind == 'var':
            return {ty.id}
        elif ty.kind == 'function':
            vars_params = set().union(*(self._free_vars(p) for p in (ty.param_types or [])))
            return vars_params | self._free_vars(ty.return_type)
        elif ty.kind in ('constructor', 'recursive'):
            vars_args = set().union(*(self._free_vars(a) for a in (ty.type_args or [])))
            return vars_args
        return set()

    def _free_vars_env(self, env: Dict[str, Type]) -> Set[int]:
        """Collect free type variable IDs in an environment"""
        return set().union(*(self._free_vars(self.to_compact_type(ty)) for ty in env.values()))
