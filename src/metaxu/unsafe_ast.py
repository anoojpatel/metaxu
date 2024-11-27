from dataclasses import dataclass
from typing import List, Optional
from metaxu.metaxu_ast import Node, Expression, Type

@dataclass
class UnsafeBlock(Node):
    """A block of unsafe code where pointer operations are allowed"""
    body: List[Node]
    
    def contains_node(self, node):
        """Check if this unsafe block contains a given node"""
        for item in self.body:
            if item == node:
                return True
            if hasattr(item, 'contains_node'):
                if item.contains_node(node):
                    return True
        return False

@dataclass
class PointerType(Type):
    """A pointer type (only valid in unsafe contexts)"""
    base_type: Type
    is_mut: bool = False  # True for *mut T, False for *const T

@dataclass
class TypeCast(Expression):
    """A type cast expression (pointer casts only valid in unsafe)"""
    expr: Expression
    target_type: Type

@dataclass
class PointerDereference(Expression):
    """Dereference a pointer (only valid in unsafe blocks)"""
    ptr: Expression

@dataclass
class AddressOf(Expression):
    """Take the address of a value (only valid in unsafe blocks)"""
    expr: Expression
    is_mut: bool = False  # True for &mut x, False for &x
