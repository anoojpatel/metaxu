from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from .type_defs import TypeDefinition, Type
import ast

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
        self._loading_stack: List[str] = []  # Track module loading order
        self.type_registry = TypeRegistry()  # Registry for type definitions

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

        # Construct possible file paths
        module_file = Path(*module_path).with_suffix('.mx')
        for search_path in search_paths:
            full_path = search_path / module_file
            if full_path.exists():
                return full_path
        return None

    def lookup_module(self, module_path: str) -> Optional[ModuleInfo]:
        """Look up a module by its dot-separated path"""
        return self.modules.get(module_path)

    def import_module(self, module_path: List[str], alias: Optional[str] = None) -> None:
        """Import a module and add it to current scope"""
        # Convert path to string for module map
        module_name = '.'.join(module_path)
        
        # Check if already imported
        if module_name in self.modules and self.modules[module_name].is_loaded:
            return

        # Resolve module file path
        file_path = self.resolve_module_path(module_path)
        if not file_path:
            raise ImportError(f"Module '{module_name}' not found")

        # Create module info if doesn't exist
        if module_name not in self.modules:
            self.modules[module_name] = ModuleInfo(
                path=file_path,
                symbols={},
                imports=set(),
                is_loaded=False
            )

        # Load module if not loaded
        if not self.modules[module_name].is_loaded:
            self._load_module(module_name, file_path)

        # Add to current scope under alias or last component
        scope_name = alias or module_path[-1]
        self.define(scope_name, Symbol(scope_name, self.modules[module_name]))

    def import_names(self, module_path: List[str], names: List[Tuple[str, Optional[str]]],
                    relative_level: int = 0) -> None:
        """Import specific names from a module"""
        # Convert path to string for module map
        module_name = '.'.join(module_path)
        
        # Import the module first
        self.import_module(module_path)
        module_info = self.modules[module_name]

        # Import each name
        for name, alias in names:
            if name not in module_info.symbols:
                raise ImportError(f"Cannot import name '{name}' from '{module_name}'")
            
            # Add to current scope under alias or original name
            scope_name = alias or name
            self.define(scope_name, module_info.symbols[name])

    def _load_module(self, module_name: str, file_path: Path) -> None:
        """Load a module and all its dependencies.
        
        Args:
            module_name: Fully qualified module name (e.g. 'std.collections')
            file_path: Path to the module file
        """
        # Check for circular imports
        if module_name in self._loading_stack:
            raise ImportError(f"Circular import detected: {' -> '.join(self._loading_stack)} -> {module_name}")
        
        # Skip if already loaded
        if self.modules[module_name].is_loaded:
            return

        # Create parser and type checker
        from parser import Parser
        from type_checker import TypeChecker
        
        parser = Parser()
        type_checker = TypeChecker()
        type_checker.symbol_table = self  # Share symbol table
        
        # Parse module file to find imports
        with open(file_path) as f:
            source = f.read()
        module_ast = parser.parse(source)
        
        # Track module loading
        self._loading_stack.append(module_name)
        try:
            # First pass: collect and load all imports
            imports = []
            if isinstance(module_ast, ast.Module):
                for stmt in module_ast.body.statements:
                    if isinstance(stmt, ast.Import):
                        import_name = '.'.join(stmt.module_path)
                        import_path = self.resolve_module_path(stmt.module_path)
                        if import_path:
                            imports.append((import_name, import_path))
                    elif isinstance(stmt, ast.FromImport):
                        import_name = '.'.join(stmt.module_path)
                        import_path = self.resolve_module_path(stmt.module_path, stmt.relative_level)
                        if import_path:
                            imports.append((import_name, import_path))
            
            # Load all dependencies first
            for import_name, import_path in imports:
                if import_name not in self.modules:
                    self.modules[import_name] = ModuleInfo(
                        path=import_path,
                        symbols={},
                        imports=set(),
                        is_loaded=False
                    )
                self._load_module(import_name, import_path)
                self.modules[module_name].imports.add(import_path)
            
            # Enter module scope for type checking
            self.enter_module(module_name, file_path)
            
            try:
                # Type check module and populate symbol table
                type_checker.check(module_ast)
                
                # Get module body for visibility rules and exports
                module_body = module_ast.body if isinstance(module_ast, ast.Module) else None
                visibility_rules = {}
                exports = []
                
                if module_body:
                    # Get visibility rules
                    if module_body.visibility_rules:
                        visibility_rules = module_body.visibility_rules.rules
                    
                    # Get exports
                    if module_body.exports:
                        exports = module_body.exports
                
                # Process symbols according to visibility and exports
                module_info = self.modules[module_name]
                for name, symbol in self.scopes[-1].items():
                    # Skip internal symbols (starting with _)
                    if name.startswith('_'):
                        continue
                        
                    # Get visibility level (default to private)
                    visibility = visibility_rules.get(name, 'private')
                    
                    # Check if symbol is exported
                    is_exported = any(export_name == name for export_name, _ in exports)
                    
                    # Only add public/protected symbols or exported symbols
                    if visibility in ('public', 'protected') or is_exported:
                        qualified_name = f"{module_name}.{name}"
                        symbol.qualified_name = qualified_name
                        symbol.visibility = visibility
                        module_info.symbols[name] = symbol
                        
                        # Re-export public imports
                        if isinstance(symbol, ast.Import) and symbol.is_public:
                            module_info.symbols.update(self.modules[symbol.module_path[-1]].symbols)
                
                # Mark module as loaded
                module_info.is_loaded = True
                
            finally:
                # Always exit module scope
                self.exit_module()
                
        finally:
            # Always remove from loading stack
            self._loading_stack.pop()

    def enter_scope(self):
        """Enter a new scope"""
        self.scopes.append({})

    def exit_scope(self):
        """Exit current scope"""
        if len(self.scopes) > 1:
            self.scopes.pop()

    def define(self, name: str, symbol: 'Symbol'):
        """Define a symbol in current scope"""
        self.scopes[-1][name] = symbol

    def lookup(self, name: str) -> Optional['Symbol']:
        """Look up a symbol in all scopes"""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def lookup_type(self, name: str) -> Optional['TypeDefinition']:
        """Look up a type definition by name.
        
        This method:
        1. Checks current module scope
        2. Checks imported modules
        3. Checks builtin types
        
        Args:
            name: The type name to look up (can be qualified)
            
        Returns:
            TypeDefinition if found, None otherwise
        """
        # Handle qualified names (e.g. std.collections.List)
        if '.' in name:
            module_path, type_name = name.rsplit('.', 1)
            module = self.resolve_module_path(module_path.split('.'))
            if module and module_path in self.modules:
                module_info = self.modules[module_path]
                if type_name in module_info.symbols:
                    symbol = module_info.symbols[type_name]
                    if isinstance(symbol.type, TypeDefinition):
                        return symbol.type
            return None
            
        # Check current module scope
        if self.current_module:
            module_info = self.modules[self.current_module]
            if name in module_info.symbols:
                symbol = module_info.symbols[name]
                if isinstance(symbol.type, TypeDefinition):
                    return symbol.type
                    
        # Check imported modules
        if self.current_module:
            module_info = self.modules[self.current_module]
            for import_path in module_info.imports:
                imported_module = str(import_path)
                if imported_module in self.modules:
                    imported_info = self.modules[imported_module]
                    if name in imported_info.symbols:
                        symbol = imported_info.symbols[name]
                        if isinstance(symbol.type, TypeDefinition):
                            return symbol.type
                            
        # Check builtin types
        if name in self.type_registry.builtin_types:
            return self.type_registry.builtin_types[name]
            
        return None

class TypeRegistry:
    """Registry for type definitions"""
    def __init__(self):
        self.types = {}  # Map of type name to TypeDefinition
        self.builtin_types = {
            'int': TypeDefinition('int', [], None),
            'float': TypeDefinition('float', [], None),
            'bool': TypeDefinition('bool', [], None),
            'str': TypeDefinition('str', [], None),
            'void': TypeDefinition('void', [], None),
        }
        
    def define_type(self, name: str, type_def: 'TypeDefinition'):
        """Register a type definition.
        
        Args:
            name: The type name
            type_def: The type definition
        """
        if name in self.types:
            raise ValueError(f"Type {name} already defined")
        self.types[name] = type_def
        
    def lookup_type(self, name: str) -> Optional['TypeDefinition']:
        """Look up a type definition by name.
        
        Args:
            name: The type name to look up
            
        Returns:
            TypeDefinition if found, None otherwise
        """
        if name in self.types:
            return self.types[name]
        if name in self.builtin_types:
            return self.builtin_types[name]
        return None
        
    def register_builtin_type(self, name: str, type_def: 'TypeDefinition'):
        """Register a builtin type.
        
        Args:
            name: The type name
            type_def: The type definition
        """
        if name in self.builtin_types:
            raise ValueError(f"Builtin type {name} already registered")
        self.builtin_types[name] = type_def

class Symbol:
    def __init__(self, name: str, symbol_type, mode="global"):
        self.name = name
        self.type = symbol_type
        self.valid = True  # For ownership checking
        self.mode = mode   # "global" or "local"
        self.region = None  # For locality tracking
        self.qualified_name = None
        self.visibility = None

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
