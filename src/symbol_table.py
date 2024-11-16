
class SymbolTable:
    def __init__(self, parent=None, struct_registry=None, enum_registry=None):
        self.symbols = {}
        self.parent = parent
        self.struct_registry = struct_registry or StructRegistry()
        self.enum_registry = enum_registry or EnumRegistry()

    def define(self, name, symbol):
        self.symbols[name] = symbol

    def lookup(self, name):
        value = self.symbols.get(name, None)
        if value is None and self.parent is not None:
            return self.parent.lookup(name)
        return value

class Symbol:
    def __init__(self, name, symbol_type):
        self.name = name
        self.type = symbol_type
        self.valid = True  # For ownership checking

    def invalidate(self):
        self.valid = False

    def is_valid(self):
        return self.valid

class StructRegistry:
    def __init__(self):
        self.structs = {}

    def define_struct(self, name, struct_type):
        self.structs[name] = struct_type

    def lookup_struct(self, name):
        return self.structs.get(name, None)

class EnumRegistry:
    def __init__(self):
        self.enums = {}

    def define_enum(self, name, enum_type):
        self.enums[name] = enum_type

    def lookup_enum(self, name):
        return self.enums.get(name, None)

