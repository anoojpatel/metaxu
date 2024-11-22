from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple

class Type:
    pass

class NamedType(Type):
    module_path: tuple[str, ...]  # E.g., ('my_module', 'sub_module')
    name: str
    is_public: bool
    is_protected: bool

    def __init__(self, module_path, name, is_public=False, is_protected=False):
        self.module_path = module_path
        self.name = name
        self.is_public = is_public
        self.is_protected = is_protected

    def __str__(self):
        return '.'.join(self.module_path + (self.name,))

class IntegerType(Type):
    def __str__(self):
        return "Int"

class FloatType(Type):
    def __str__(self):
        return "Float"

class BooleanType(Type):
    def __str__(self):
        return "Bool"

class StringType(Type):
    def __str__(self):
        return "String"

class NoneType(Type):
    def __str__(self):
        return "None"

# Ownership Types
class UniqueType(Type):
    def __init__(self, base_type):
        self.base_type = base_type

    def __str__(self):
        return f"Unique[{self.base_type}]"

class SharedType(Type):
    def __init__(self, base_type):
        self.base_type = base_type

    def __str__(self):
        return f"Shared[{self.base_type}]"

# Function Types
class FunctionType(Type):
    def __init__(self, param_types, return_type):
        self.param_types = param_types
        self.return_type = return_type

    def __str__(self):
        params = ', '.join(map(str, self.param_types))
        return f"({params}) -> {self.return_type}"

# Vector Types for SIMD
class VectorType(Type):
    def __init__(self, base_type, size):
        self.base_type = base_type
        self.size = size

    def __str__(self):
        return f"vector[{self.base_type}, {self.size}]"

# Box Types for Heap Allocation
class BoxType(Type):
    def __init__(self, inner_type):
        self.inner_type = inner_type

    def __str__(self):
        return f"Box<{self.inner_type}>"

class ReferenceType(Type):
    def __init__(self, base_type, is_mutable=False):
        self.base_type = base_type
        self.is_mutable = is_mutable

    def __str__(self):
        mut = "mut " if self.is_mutable else ""
        return f"&{mut}{self.base_type}"

# Recursive Types
class RecursiveType(Type):
    def __init__(self, type_name, type_parameters=None):
        self.type_name = type_name
        self.type_parameters = type_parameters or []
        self._resolved_type = None

    def set_resolved_type(self, resolved_type):
        self._resolved_type = resolved_type

    def get_resolved_type(self):
        return self._resolved_type

    def __str__(self):
        if self.type_parameters:
            params = ", ".join(map(str, self.type_parameters))
            return f"{self.type_name}<{params}>"
        return self.type_name

# Type Parameters for Generics
class TypeVar(Type):
    """Type variable for polymorphic types"""
    def __init__(self, name: str, module_path: Optional[Tuple[str, ...]] = None):
        self.name = name
        self.module_path = module_path or tuple()
        self.constraints = []  # Upper/lower bounds for constrained type variables

    def qualified_name(self):
        """Get fully qualified name including module path"""
        return '.'.join(self.module_path + (self.name,))

    def __str__(self):
        return self.qualified_name()

class TypeConstructor(Type):
    """Type constructor for generic types"""
    def __init__(self, name: str, arity: int,
                 module_path: Optional[Tuple[str, ...]] = None):
        self.name = name
        self.arity = arity
        self.module_path = module_path or tuple()

    def qualified_name(self):
        return '.'.join(self.module_path + (self.name,))

    def __str__(self):
        return self.qualified_name()

# Enum and Variant Types
class RowVar(Type):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"RowVar<{self.name}>"
class EmptyRow(Type):
    def __str__(self):
        return "EmptyRow"

class EnumType(Type):
    def __init__(self, name: str, variants: Dict[str, VariantType], 
                 type_params: List[TypeVar] = None,
                 module_path: Optional[Tuple[str, ...]] = None,
                 is_open: bool = False):
        self.name = name
        self.variants = variants
        self.type_params = type_params or []
        self.module_path = module_path or tuple()
        self.is_open = is_open  # True for open variant types
        
    def qualified_name(self):
        if self.module_path:
            return '.'.join(self.module_path + (self.name,))
        return self.name
        
    def __str__(self):
        if not self.type_params:
            return self.qualified_name()
        params = ", ".join(str(p) for p in self.type_params)
        return f"{self.qualified_name()}<{params}>"

class VariantType(Type):
    def __init__(self, enum_name, name, fields=None):
        self.enum_name = enum_name
        self.name = name
        self.fields = fields or {}  # Dict of field_name: field_type

    def __str__(self):
        if self.fields:
            fields = ", ".join(f"{name}: {type_}" for name, type_ in self.fields.items())
            return f"{self.enum_name}::{self.name}({fields})"
        return f"{self.enum_name}::{self.name}"

# Mode Types
class Mode:
    pass

class UniquenessMode(Mode):
    UNIQUE = "unique"
    EXCLUSIVE = "exclusive"
    SHARED = "shared"

    def __init__(self, mode):
        if mode not in [self.UNIQUE, self.EXCLUSIVE, self.SHARED]:
            raise ValueError(f"Invalid uniqueness mode: {mode}")
        self.mode = mode

    def __str__(self):
        return f"@{self.mode}"

class LocalityMode(Mode):
    LOCAL = "local"
    GLOBAL = "global"

    def __init__(self, mode):
        if mode not in [self.LOCAL, self.GLOBAL]:
            raise ValueError(f"Invalid locality mode: {mode}")
        self.mode = mode

    def __str__(self):
        return f"@{self.mode}"

class LinearityMode(Mode):
    ONCE = "once"
    SEPARATE = "separate"
    MANY = "many"

    def __init__(self, mode):
        if mode not in [self.ONCE, self.SEPARATE, self.MANY]:
            raise ValueError(f"Invalid linearity mode: {mode}")
        self.mode = mode

    def __str__(self):
        return f"@{self.mode}"

class ModeType(Type):
    def __init__(self, base_type, uniqueness=None, locality=None, linearity=None):
        self.base_type = base_type
        self.uniqueness = uniqueness or UniquenessMode(UniquenessMode.SHARED)
        self.locality = locality or LocalityMode(LocalityMode.GLOBAL)
        self.linearity = linearity or LinearityMode(LinearityMode.MANY)

    def __str__(self):
        modes = []
        if self.uniqueness.mode != UniquenessMode.SHARED:
            modes.append(str(self.uniqueness))
        if self.locality.mode != LocalityMode.GLOBAL:
            modes.append(str(self.locality))
        if self.linearity.mode != LinearityMode.MANY:
            modes.append(str(self.linearity))
        mode_str = " ".join(modes)
        return f"{self.base_type} {mode_str}" if modes else str(self.base_type)

# Structs and Enums
class StructField:
    def __init__(self, name, field_type, is_exclusively_mutable=False):
        self.name = name
        self.field_type = field_type
        self.is_exclusively_mutable = is_exclusively_mutable

class StructType(Type):
    def __init__(self, name, fields):
        self.name = name
        self.fields = {name: StructField(name, ftype) for name, ftype in fields.items()}

    def make_field_exclusively_mutable(self, field_name):
        if field_name in self.fields:
            self.fields[field_name].is_exclusively_mutable = True

    def __str__(self):
        fields_str = []
        for field in self.fields.values():
            mut_str = "exclusively mutable " if field.is_exclusively_mutable else ""
            fields_str.append(f"{mut_str}{field.name}: {field.field_type}")
        return f"struct {self.name} {{ {', '.join(fields_str)} }}"

# Effect Types
@dataclass
class EffectType(Type):
    """Type for algebraic effects"""
    name: str
    operations: List['EffectOperation']
    type_params: Optional[List['TypeParameter']] = None

@dataclass
class EffectOperation:
    """Operation in an algebraic effect"""
    name: str
    params: List['Parameter']
    return_type: Type
    type_params: Optional[List['TypeParameter']] = None

@dataclass
class ContinuationType(Type):
    """Type for continuations"""
    resume_type: Type
    effect_type: EffectType

@dataclass
class HandlerType(Type):
    """Type for effect handlers"""
    effect: EffectType
    handled_type: Type
    resume_type: Type

# SimpleSub Type System Components
class TypeScheme:
    """Universal type scheme with quantified variables"""
    def __init__(self, type_vars: List[TypeVar], body_type: Type,
                 module_path: Optional[Tuple[str, ...]] = None):
        self.type_vars = type_vars
        self.body_type = body_type
        self.module_path = module_path or tuple()
        
    def instantiate(self, type_args: List[Type]) -> Type:
        """Instantiate scheme with concrete types"""
        if len(type_args) != len(self.type_vars):
            raise TypeError(f"Wrong number of type arguments, expected {len(self.type_vars)}")
        subst = {tv: ty for tv, ty in zip(self.type_vars, type_args)}
        return substitute(self.body_type, subst)
        
    def __str__(self):
        vars_str = ", ".join(str(tv) for tv in self.type_vars)
        return f"∀{vars_str}. {self.body_type}"

class ConstructedType(Type):
    """Type application of a type constructor to arguments"""
    def __init__(self, constructor: TypeConstructor, type_args: List[Type]):
        if len(type_args) != constructor.arity:
            raise TypeError(f"Wrong number of type arguments for {constructor}")
        self.constructor = constructor
        self.type_args = type_args
        
    def __str__(self):
        if not self.type_args:
            return str(self.constructor)
        args_str = ", ".join(str(arg) for arg in self.type_args)
        return f"{self.constructor}<{args_str}>"

class IntersectionType(Type):
    """Intersection of types (greatest lower bound)"""
    def __init__(self, types: List[Type]):
        self.types = types
        
    def __str__(self):
        return " & ".join(str(t) for t in self.types)

class UnionType(Type):
    """Union of types (least upper bound)"""
    def __init__(self, types: List[Type]):
        self.types = types
        
    def __str__(self):
        return " | ".join(str(t) for t in self.types)

@dataclass
class TypeDefinition:
    """Definition of a type with its parameters and bounds"""
    name: str
    type_params: List[TypeVar]
    body: Optional[Type]
    
    def __str__(self):
        params = f"[{', '.join(str(p) for p in self.type_params)}]" if self.type_params else ""
        return f"{self.name}{params}"

# Helper functions for type manipulation
def substitute(ty: Type, subst: Dict[TypeVar, Type], visited: Optional[Set[str]] = None) -> Type:
    """Substitute type variables in a type with cycle detection"""
    if visited is None:
        visited = set()
        
    # Handle recursive types
    if isinstance(ty, RecursiveType):
        # Check for cycles
        type_key = f"{ty.type_name}_{id(ty)}"
        if type_key in visited:
            return ty  # Return the recursive reference as is
        visited.add(type_key)
        
        # Substitute in type parameters
        new_params = [substitute(param, subst, visited) 
                     for param in ty.type_parameters]
                     
        # Get resolved type if available
        resolved = ty.get_resolved_type()
        if resolved:
            # Substitute in resolved type with updated parameters
            new_resolved = substitute(resolved, subst, visited)
            result = RecursiveType(ty.type_name, new_params)
            result.set_resolved_type(new_resolved)
            return result
        return RecursiveType(ty.type_name, new_params)
        
    elif isinstance(ty, TypeVar):
        return subst.get(ty, ty)
        
    elif isinstance(ty, ConstructedType):
        return ConstructedType(
            ty.constructor,
            [substitute(arg, subst, visited) for arg in ty.type_args]
        )
        
    elif isinstance(ty, UnionType):
        return UnionType([substitute(t, subst, visited) for t in ty.types])
        
    elif isinstance(ty, IntersectionType):
        return IntersectionType([substitute(t, subst, visited) for t in ty.types])
        
    elif isinstance(ty, FunctionType):
        return FunctionType(
            [substitute(pt, subst, visited) for pt in ty.param_types],
            substitute(ty.return_type, subst, visited)
        )
        
    elif isinstance(ty, EnumType):
        # Only substitute in variant fields
        new_variants = {
            name: VariantType(
                ty.name, 
                name,
                {fname: substitute(ftype, subst, visited) 
                 for fname, ftype in variant.fields.items()}
            )
            for name, variant in ty.variants.items()
        }
        return EnumType(ty.name, new_variants, ty.type_params,
                       ty.module_path, ty.is_open)
                       
    elif isinstance(ty, BoxType):
        return BoxType(substitute(ty.inner_type, subst, visited))
        
    elif isinstance(ty, ReferenceType):
        return ReferenceType(
            substitute(ty.base_type, subst, visited),
            ty.is_mutable
        )
        
    return ty

@dataclass
class CompactType:
    """Compact representation of types for efficient unification and variance tracking"""
    id: int  # Unique identifier
    kind: str  # 'var', 'constructor', 'recursive'
    bounds: Optional['TypeBounds'] = None
    constructor: Optional[TypeConstructor] = None
    type_args: Optional[List['CompactType']] = None
    recursive_ref: Optional[str] = None  # Name for recursive types
    variance: Optional[str] = None  # 'covariant', 'contravariant', 'invariant'
    
    @staticmethod
    def fresh_var() -> 'CompactType':
        """Create fresh type variable"""
        return CompactType(
            id=next_id(),
            kind='var',
            bounds=TypeBounds()
        )
    
    def find(self) -> 'CompactType':
        """Find representative with path compression"""
        if self.bounds and self.bounds.upper_bound:
            self.bounds.upper_bound = self.bounds.upper_bound.find()
            return self.bounds.upper_bound
        return self

@dataclass
class TypeBounds:
    """Bounds for type variables during unification"""
    upper_bound: Optional[CompactType] = None
    lower_bound: Optional[CompactType] = None
    
def unify(t1: CompactType, t2: CompactType, variance: str = 'invariant') -> bool:
    """Unify two types with variance"""
    t1, t2 = t1.find(), t2.find()
    if t1 == t2:
        return True
        
    # Handle variables
    if t1.kind == 'var':
        if occurs_check(t1, t2):
            return False
        t1.bounds.upper_bound = t2
        return True
    if t2.kind == 'var':
        if occurs_check(t2, t1):
            return False
        t2.bounds.upper_bound = t1
        return True
        
    # Handle recursive types
    if t1.kind == 'recursive' and t2.kind == 'recursive':
        if t1.recursive_ref == t2.recursive_ref:
            # Same recursive type - unify parameters
            return all(unify(a1, a2, variance) 
                      for a1, a2 in zip(t1.type_args or [], t2.type_args or []))
        # Different recursive types - unfold once and try again
        return unify(unfold_once(t1), unfold_once(t2), variance)
        
    # Handle constructors
    if t1.kind == 'constructor' and t2.kind == 'constructor':
        if t1.constructor != t2.constructor:
            return False
        # Unify arguments with composed variance
        return all(unify(a1, a2, compose_variance(variance, v))
                  for a1, a2, v in zip(t1.type_args or [], 
                                     t2.type_args or [],
                                     get_constructor_variances(t1.constructor)))
                                     
    return False

def occurs_check(var: CompactType, ty: CompactType) -> bool:
    """Check if variable occurs in type"""
    if ty == var:
        return True
    return any(occurs_check(var, arg) 
              for arg in (ty.type_args or []))

def unfold_once(ty: CompactType) -> CompactType:
    """Unfold one level of recursive type"""
    if ty.kind != 'recursive' or not ty.bounds or not ty.bounds.upper_bound:
        return ty
    # Create fresh variables for parameters
    subst = {param.id: CompactType.fresh_var() 
             for param in (ty.type_args or [])}
    # Apply substitution to unfolded type
    return substitute_compact(ty.bounds.upper_bound, subst)

def substitute_compact(ty: CompactType, 
                      subst: Dict[int, CompactType]) -> CompactType:
    """Substitute in CompactType"""
    ty = ty.find()
    if ty.kind == 'var':
        return subst.get(ty.id, ty)
    if ty.kind == 'constructor':
        return CompactType(
            id=next_id(),
            kind='constructor',
            constructor=ty.constructor,
            type_args=[substitute_compact(arg, subst) 
                      for arg in (ty.type_args or [])]
        )
    if ty.kind == 'recursive':
        return CompactType(
            id=next_id(),
            kind='recursive',
            recursive_ref=ty.recursive_ref,
            type_args=[substitute_compact(arg, subst) 
                      for arg in (ty.type_args or [])]
        )
    return ty

def compose_variance(v1: str, v2: str) -> str:
    """Compose two variance annotations"""
    if v1 == 'invariant' or v2 == 'invariant':
        return 'invariant'
    if v1 == v2:
        return 'covariant'
    return 'contravariant'

def get_constructor_variances(tc: TypeConstructor) -> List[str]:
    """Get variance annotations for constructor parameters"""
    return getattr(tc, 'variances', ['invariant'] * tc.arity)

_next_id = 0
def next_id() -> int:
    """Generate unique IDs for CompactTypes"""
    global _next_id
    _next_id += 1
    return _next_id

# Update the lexer tokens
reserved = {
    # ... existing tokens ...
}
