from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from pathlib import Path
import metaxu.metaxu_ast as ast
from metaxu.symbol_table import SymbolTable, Symbol, ModuleInfo

@dataclass
class LinkageType:
    STATIC = "static"
    DYNAMIC = "dynamic"

@dataclass
class ModuleDependency:
    module_name: str
    linkage_type: str
    required_symbols: Set[str] = field(default_factory=set)
    version_constraints: Optional[str] = None

@dataclass
class LinkedModule:
    name: str
    symbols: Dict[str, Symbol]
    dependencies: List[ModuleDependency]
    is_dynamic: bool = False

class Linker:
    def __init__(self, symbol_table: SymbolTable):
        self.symbol_table = symbol_table
        self.linked_modules: Dict[str, LinkedModule] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        
    def link_module(self, module_name: str, linkage_type: str = LinkageType.STATIC) -> LinkedModule:
        """Link a module and all its dependencies"""
        if module_name in self.linked_modules:
            return self.linked_modules[module_name]
            
        # Get module info
        module_info = self.symbol_table.lookup_module(module_name)
        if not module_info:
            raise LinkError(f"Module '{module_name}' not found")
            
        # Check for circular dependencies
        if self._would_create_cycle(module_name, set()):
            raise LinkError(f"Circular dependency detected involving {module_name}")
            
        # Create linked module
        linked_module = LinkedModule(
            name=module_name,
            symbols={},
            dependencies=[],
            is_dynamic=linkage_type == LinkageType.DYNAMIC
        )
        
        # Link dependencies first
        for dep in module_info.imports:
            dep_module = self.link_module(dep, linkage_type)
            linked_module.dependencies.append(ModuleDependency(
                module_name=dep,
                linkage_type=linkage_type
            ))
            
        # Add to dependency graph
        self.dependency_graph[module_name] = {dep.module_name for dep in linked_module.dependencies}
            
        # Process symbols
        for name, symbol in module_info.symbols.items():
            if self._is_symbol_accessible(symbol, module_info):
                linked_module.symbols[name] = symbol
                
        self.linked_modules[module_name] = linked_module
        return linked_module
    
    def _would_create_cycle(self, module_name: str, visited: Set[str]) -> bool:
        """Check if adding module_name would create a cycle in the dependency graph"""
        if module_name in visited:
            return True
            
        visited.add(module_name)
        module_info = self.symbol_table.lookup_module(module_name)
        if not module_info:
            return False
            
        for dep in module_info.imports:
            if self._would_create_cycle(dep, visited.copy()):
                return True
                
        return False
    
    def _is_symbol_accessible(self, symbol: Symbol, module_info: ModuleInfo) -> bool:
        """Check if a symbol should be included in the linked module"""
        # Check visibility rules
        if hasattr(module_info, 'visibility_rules') and symbol.name in module_info.visibility_rules:
            visibility = module_info.visibility_rules[symbol.name]
            if visibility == 'private':
                return False
        return True

class DynamicLinker(Linker):
    def __init__(self, symbol_table: SymbolTable):
        super().__init__(symbol_table)
        self.loaded_modules: Dict[str, LinkedModule] = {}
        
    def load_module(self, module_name: str) -> LinkedModule:
        """Dynamically load a module at runtime"""
        if module_name in self.loaded_modules:
            return self.loaded_modules[module_name]
            
        linked_module = self.link_module(module_name, LinkageType.DYNAMIC)
        self.loaded_modules[module_name] = linked_module
        return linked_module
        
    def unload_module(self, module_name: str):
        """Unload a dynamically loaded module"""
        if module_name in self.loaded_modules:
            # Unload dependencies first
            module = self.loaded_modules[module_name]
            for dep in module.dependencies:
                if dep.linkage_type == LinkageType.DYNAMIC:
                    self.unload_module(dep.module_name)
            
            del self.loaded_modules[module_name]
            
    def reload_module(self, module_name: str) -> LinkedModule:
        """Reload a dynamically loaded module"""
        self.unload_module(module_name)
        return self.load_module(module_name)

class LinkError(Exception):
    pass
