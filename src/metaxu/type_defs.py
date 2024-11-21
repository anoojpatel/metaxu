from dataclasses import dataclass
from typing import List, Dict, Optional, Any

class Type:
    pass

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
class TypeParameter(Type):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

# Enum and Variant Types
class EnumType(Type):
    def __init__(self, name, variants):
        self.name = name
        self.variants = variants  # Dict of variant_name: VariantType

    def __str__(self):
        variants = ", ".join(f"{name}: {variant}" for name, variant in self.variants.items())
        return f"enum {self.name} {{ {variants} }}"

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

# Update the lexer tokens
reserved = {
    # ... existing tokens ...
}
