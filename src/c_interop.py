from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union, Callable
from enum import Enum
import metaxu_ast as ast
from symbol_table import Symbol

class CFunctionCallMode(Enum):
    """Mode for C function calls"""
    BLOCKING = "blocking"       # Function blocks until completion
    NONBLOCKING = "nonblocking" # Function runs async, references locked until completion
    UNSAFE = "unsafe"          # No safety guarantees (requires unsafe block)

class CTypeKind(Enum):
    """C type categories"""
    VALUE = "value"           # Plain value types (int, float, etc)
    POINTER = "pointer"       # Pointer types (*int, *char, etc)
    ARRAY = "array"          # Array types
    STRUCT = "struct"        # Struct types
    FUNCTION = "function"    # Function pointer types

@dataclass
class CTypeInfo:
    """Information about a C type"""
    kind: CTypeKind
    name: str
    is_const: bool = False
    is_volatile: bool = False
    element_type: Optional['CTypeInfo'] = None  # For pointers/arrays
    array_size: Optional[int] = None           # For arrays
    struct_fields: Dict[str, 'CTypeInfo'] = field(default_factory=dict)

@dataclass
class CFunctionInfo:
    """Information about a C function"""
    name: str
    return_type: CTypeInfo
    param_types: List[CTypeInfo]
    is_variadic: bool = False
    call_mode: CFunctionCallMode = CFunctionCallMode.BLOCKING
    borrows_refs: bool = False  # True if function borrows references
    consumes_refs: bool = False # True if function takes ownership
    produces_refs: bool = False # True if function returns new references

class CInteropManager:
    def __init__(self):
        self.function_info: Dict[str, CFunctionInfo] = {}
        self.type_info: Dict[str, CTypeInfo] = {}
        self._register_basic_types()
        
    def _register_basic_types(self):
        """Register basic C types"""
        basic_types = [
            ("void", CTypeKind.VALUE),
            ("char", CTypeKind.VALUE),
            ("int", CTypeKind.VALUE),
            ("float", CTypeKind.VALUE),
            ("double", CTypeKind.VALUE),
            ("size_t", CTypeKind.VALUE),
        ]
        for name, kind in basic_types:
            self.type_info[name] = CTypeInfo(kind=kind, name=name)

    def register_function(self, func_info: CFunctionInfo):
        """Register a C function"""
        self.function_info[func_info.name] = func_info

    def wrap_function_call(self, func_name: str, args: List[ast.Expression]) -> ast.Expression:
        """Wrap a C function call with appropriate safety checks"""
        func_info = self.function_info.get(func_name)
        if not func_info:
            raise ValueError(f"Unknown C function: {func_name}")

        # Create safety wrapper based on call mode
        if func_info.call_mode == CFunctionCallMode.BLOCKING:
            return self._create_blocking_wrapper(func_info, args)
        elif func_info.call_mode == CFunctionCallMode.NONBLOCKING:
            return self._create_nonblocking_wrapper(func_info, args)
        else:  # UNSAFE
            return self._create_unsafe_wrapper(func_info, args)

    def _create_blocking_wrapper(self, func_info: CFunctionInfo, args: List[ast.Expression]) -> ast.Expression:
        """Create a blocking wrapper that ensures reference safety"""
        # Create a scope that will block until the function completes
        stmts = []
        
        # Add reference tracking
        if func_info.borrows_refs:
            stmts.append(ast.BorrowCheck(args))
        if func_info.consumes_refs:
            stmts.append(ast.MoveCheck(args))
            
        # Add the actual function call
        call = ast.CFunctionCall(
            name=func_info.name,
            args=args,
            return_type=self._convert_c_type(func_info.return_type)
        )
        
        # Add reference cleanup
        if func_info.produces_refs:
            stmts.append(ast.ReferenceInit(call))
        else:
            stmts.append(call)
            
        return ast.Block(stmts)

    def _create_nonblocking_wrapper(self, func_info: CFunctionInfo, args: List[ast.Expression]) -> ast.Expression:
        """Create a non-blocking wrapper that manages reference lifetimes"""
        # Similar to blocking wrapper but runs in background
        stmts = []
        
        # Lock references
        if func_info.borrows_refs or func_info.consumes_refs:
            stmts.append(ast.ReferenceLock(args))
            
        # Create async call
        call = ast.AsyncCFunctionCall(
            name=func_info.name,
            args=args,
            return_type=self._convert_c_type(func_info.return_type)
        )
        
        # Add completion handler for reference management
        completion_handler = ast.CompletionHandler([
            ast.ReferenceUnlock(args),
            ast.ReferenceCleanup(args) if func_info.consumes_refs else ast.NoOp()
        ])
        
        return ast.AsyncBlock([*stmts, call, completion_handler])

    def _create_unsafe_wrapper(self, func_info: CFunctionInfo, args: List[ast.Expression]) -> ast.Expression:
        """Create an unsafe wrapper with minimal checks"""
        return ast.UnsafeBlock([
            ast.CFunctionCall(
                name=func_info.name,
                args=args,
                return_type=self._convert_c_type(func_info.return_type)
            )
        ])

    def _convert_c_type(self, c_type: CTypeInfo) -> ast.Type:
        """Convert C type to Metaxu type"""
        if c_type.kind == CTypeKind.VALUE:
            return ast.BasicType(c_type.name)
        elif c_type.kind == CTypeKind.POINTER:
            if c_type.is_const:
                return ast.ReferenceType(
                    self._convert_c_type(c_type.element_type),
                    mode=ast.UniquenessMode.SHARED
                )
            else:
                return ast.ReferenceType(
                    self._convert_c_type(c_type.element_type),
                    mode=ast.UniquenessMode.EXCLUSIVE
                )
        elif c_type.kind == CTypeKind.ARRAY:
            return ast.ArrayType(
                self._convert_c_type(c_type.element_type),
                c_type.array_size
            )
        else:
            raise ValueError(f"Unsupported C type: {c_type.kind}")

def create_function_info(
    name: str,
    return_type: CTypeInfo,
    param_types: List[CTypeInfo],
    call_mode: CFunctionCallMode = CFunctionCallMode.BLOCKING,
    **kwargs
) -> CFunctionInfo:
    """Helper function to create C function information"""
    return CFunctionInfo(
        name=name,
        return_type=return_type,
        param_types=param_types,
        call_mode=call_mode,
        **kwargs
    )
