from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from metaxu_ast import Node, Expression

@dataclass
class Decorator(Node):
    """Base class for decorators"""
    name: str
    args: Dict[str, Expression] = field(default_factory=dict)

@dataclass
class CFunctionDecorator(Decorator):
    """Decorator for C functions"""
    borrows_refs: bool = False
    consumes_refs: bool = False
    produces_refs: bool = False
    call_mode: str = "blocking"
    inline: bool = True  # Default to inlining when possible
    
    def __post_init__(self):
        self.name = "c_function"  # Always set name to c_function

@dataclass
class DecoratorList(Node):
    """List of decorators"""
    decorators: List[Decorator]
