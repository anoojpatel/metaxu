from dataclasses import dataclass
from typing import List, Optional
from metaxu_ast import Node, Expression, Type
from decorator_ast import CFunctionDecorator

@dataclass
class ExternFunctionDeclaration(Node):
    """Declaration of an external function"""
    name: str
    params: List[Expression]
    return_type: Optional[Type]
    c_function: Optional[CFunctionDecorator] = None  # C-specific info if this is a C function

@dataclass
class ExternBlock(Node):
    """A block of extern declarations"""
    header_path: str  # Path to the header file
    declarations: List[Node]  # List of declarations (functions, types, etc.)

@dataclass
class ExternTypeDeclaration(Node):
    """Declaration of an external type"""
    name: str
    is_opaque: bool = False  # True if type internals are not exposed
    struct_name: Optional[str] = None  # Name of the struct type if referencing existing struct
