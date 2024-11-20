from dataclasses import dataclass
from typing import List, Optional
from metaxu_ast import Expression, Type, Node

@dataclass
class CFunctionCall(Expression):
    """A call to a C function"""
    name: str
    args: List[Expression]
    return_type: Type

@dataclass
class AsyncCFunctionCall(Expression):
    """An asynchronous call to a C function"""
    name: str
    args: List[Expression]
    return_type: Type

@dataclass
class UnsafeBlock(Expression):
    """A block of unsafe code"""
    body: List[Expression]

@dataclass
class AsyncBlock(Expression):
    """A block of async code"""
    body: List[Expression]

@dataclass
class CompletionHandler(Node):
    """Handler for async completion"""
    body: List[Expression]

@dataclass
class BorrowCheck(Node):
    """Check if references can be borrowed"""
    refs: List[Expression]

@dataclass
class MoveCheck(Node):
    """Check if references can be moved"""
    refs: List[Expression]

@dataclass
class ReferenceInit(Node):
    """Initialize a reference from C"""
    expr: Expression

@dataclass
class ReferenceLock(Node):
    """Lock references for async operations"""
    refs: List[Expression]

@dataclass
class ReferenceUnlock(Node):
    """Unlock references after async operations"""
    refs: List[Expression]

@dataclass
class ReferenceCleanup(Node):
    """Clean up references"""
    refs: List[Expression]

@dataclass
class NoOp(Node):
    """No operation"""
    pass
