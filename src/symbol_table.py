from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path

@dataclass
class ModuleInfo:
    """Information about a loaded module"""
    path: Path  # Full path to module file
    symbols: Dict[str, 'Symbol']  # Exported symbols
    imports: Set[Path]  # Other modules this module imports
    is_loaded: bool = False

class SymbolTable:
    def __init__(self):
        self.scopes: List[Dict[str, 'Symbol']] = [{}]  # Stack of scopes
        self.modules: Dict[str, ModuleInfo] = {}  # Map of module name to info
        self.current_module: Optional[str] = None
        self.module_search_paths: List[Path] = []

    def enter_module(self, module_name: str, module_path: Path):
        """Enter a new module scope"""
        if module_name not in self.modules:
            self.modules[module_name] = ModuleInfo(
                path=module_path,
                symbols={},
                imports=set(),
                is_loaded=False
            )
        self.current_module = module_name
        self.enter_scope()  # Create new scope for module

    def exit_module(self):
        """Exit current module scope"""
        self.exit_scope()
        self.current_module = None

    def add_module_search_path(self, path: Path):
        """Add a directory to search for modules"""
        self.module_search_paths.append(path)

    def resolve_module_path(self, module_path: List[str], relative_level: int = 0) -> Optional[Path]:
        """Resolve a module path to an actual file path"""
        if relative_level > 0:
            # Handle relative imports
            if not self.current_module:
                raise ImportError("Relative import outside of module")
            current_path = self.modules[self.current_module].path
            start_dir = current_path.parent
            for _ in range(relative_level - 1):
                start_dir = start_dir.parent
            search_paths = [start_dir]
        else:
            search_paths = self.module_search_paths

        # Convert module path to file path
        file_path = Path(*module_path).with_suffix('.mx')
        
        # Search in all possible locations
        for search_path in search_paths:
            full_path = search_path / file_path
            if full_path.exists():
                return full_path
        return None

    def import_module(self, module_path: List[str], alias: Optional[str] = None, relative_level: int = 0) -> Optional[str]:
        """Import a module and return its qualified name"""
        # Resolve module file path
        file_path = self.resolve_module_path(module_path, relative_level)
        if not file_path:
            raise ImportError(f"Module not found: {'.'.join(module_path)}")

        # Generate qualified name
        qualified_name = '.'.join(module_path)
        if alias:
            module_name = alias
        else:
            module_name = module_path[-1]

        # Add to current module's imports
        if self.current_module:
            self.modules[self.current_module].imports.add(file_path)

        # Add module to symbol table if not already present
        if qualified_name not in self.modules:
            self.modules[qualified_name] = ModuleInfo(
                path=file_path,
                symbols={},
                imports=set(),
                is_loaded=False
            )

        # Add module symbol to current scope
        self.add_symbol(module_name, Symbol(
            name=module_name,
            symbol_type=None,  # Modules don't have a type
            mode="global",
            qualified_name=qualified_name
        ))

        return qualified_name

    def import_names(self, module_path: List[str], names: List[Tuple[str, Optional[str]]], relative_level: int = 0):
        """Import specific names from a module"""
        qualified_name = self.import_module(module_path, None, relative_level)
        if not qualified_name:
            return

        module_info = self.modules[qualified_name]
        for name, alias in names:
            if name not in module_info.symbols:
                raise ImportError(f"Cannot import name '{name}' from '{qualified_name}'")
            
            symbol = module_info.symbols[name]
            local_name = alias or name
            self.add_symbol(local_name, Symbol(
                name=local_name,
                symbol_type=symbol.symbol_type,
                mode=symbol.mode,
                qualified_name=f"{qualified_name}.{name}"
            ))

    def enter_scope(self):
        """Enter a new scope"""
        self.scopes.append({})

    def exit_scope(self):
        """Exit current scope"""
        self.scopes.pop()

    def add_symbol(self, name: str, symbol: 'Symbol'):
        """Add a symbol to the current scope"""
        self.scopes[-1][name] = symbol

    def lookup(self, name: str) -> Optional['Symbol']:
        """Lookup a symbol in the current scope and its parents"""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

class Symbol:
    def __init__(self, name: str, symbol_type, mode="global"):
        self.name = name
        self.type = symbol_type
        self.valid = True  # For ownership checking
        self.mode = mode   # "global" or "local"
        self.region = None  # For locality tracking
        self.qualified_name = None

    def invalidate(self):
        self.valid = False

    def is_valid(self):
        return self.valid

    def set_region(self, region):
        """Set the region this symbol belongs to"""
        self.region = region

    def get_region(self):
        """Get the region this symbol belongs to"""
        return self.region

    def escapes_region(self, target_region):
        """Check if this symbol would escape its region if moved to target_region"""
        if self.mode == "global":
            return False  # Global symbols can escape any region
        return target_region != self.region

    def check_deep_locality(self, target_region, symbol_table):
        """Check if this symbol and all its contained values would escape target_region"""
        if self.escapes_region(target_region):
            return False

        # For composite types, check all contained values
        if isinstance(self.type, StructType):
            # All fields of a local struct must be local and in the same region
            for field_name, field_type in self.type.fields.items():
                field_symbol = symbol_table.lookup(f"{self.name}.{field_name}")
                if field_symbol and not field_symbol.check_deep_locality(target_region, symbol_table):
                    return False
                    
        elif isinstance(self.type, VectorType):
            # For vectors/arrays, the base type must be local
            base_symbol = Symbol(f"{self.name}_base", self.type.base_type, mode=self.mode)
            base_symbol.set_region(self.region)
            if not base_symbol.check_deep_locality(target_region, symbol_table):
                return False
                
        return True

    def __str__(self):
        mode_str = f" @ {self.mode}" if self.mode == "local" else ""
        return f"{self.type}{mode_str}"

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
