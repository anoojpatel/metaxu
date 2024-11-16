
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

# Structs and Enums
class StructType(Type):
    def __init__(self, name, fields):
        self.name = name
        self.fields = fields  # Dict of field_name: field_type

    def __str__(self):
        return f"Struct[{self.name}]"

class EnumType(Type):
    def __init__(self, name, variants):
        self.name = name
        self.variants = variants  # Dict of variant_name: VariantType

    def __str__(self):
        return f"Enum[{self.name}]"

class VariantType(Type):
    def __init__(self, enum_name, name, fields):
        self.enum_name = enum_name
        self.name = name
        self.fields = fields  # Dict of field_name: field_type

    def __str__(self):
        return f"Variant[{self.enum_name}::{self.name}]"

