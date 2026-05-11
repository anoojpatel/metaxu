"""Desugaring pass infrastructure for Metaxu compiler.

This module provides infrastructure for AST desugaring - transforming high-level
syntactic sugar into lower-level constructs that the rest of the compiler pipeline
can handle more easily.

Desugaring passes are applied after parsing but before type checking, allowing
syntactic features to be transformed into core language constructs.

Examples of desugaring:
- if/else statements -> pattern matching on Bool type
- trait method calls -> dictionary lookups
- for loops -> while loops or recursion
- etc.
"""

from __future__ import annotations
from typing import Callable, Any
from dataclasses import dataclass

import metaxu.metaxu_ast as fast


@dataclass(slots=True)
class DesugarContext:
    """Context passed through desugaring passes."""
    source: str | None = None
    file_path: str | None = None
    traits: dict[int, Any] | None = None  # Trait definitions by node_id
    trait_impls: dict[str, Any] | None = None  # Trait implementations by type name
    tables: Any = None  # InferSideTables for type information
    # Additional context can be added as needed


class DesugarPass:
    """Base class for desugaring passes.
    
    Each desugaring pass should inherit from this class and implement
    the `apply` method to transform the AST.
    """
    
    def apply(self, node: fast.Node, ctx: DesugarContext) -> fast.Node:
        """Apply desugaring transformation to a node.
        
        Arguments:
            node: The AST node to transform
            ctx: Desugaring context
            
        Returns:
            The transformed node (may be the same node if no transformation needed)
        """
        return node
    
    def apply_recursive(self, node: fast.Node, ctx: DesugarContext) -> fast.Node:
        """Apply desugaring recursively to all child nodes."""
        # Transform this node
        node = self.apply(node, ctx)
        
        # Recursively transform children
        # Note: frozen AST nodes are immutable, so we create new nodes
        if hasattr(node, 'children') and node.children:
            new_children = tuple(self.apply_recursive(child, ctx) for child in node.children)
            # Create a new node with transformed children
            # For frozen AST, we need to reconstruct the node
            if hasattr(node, 'node_id') and hasattr(node, 'kind'):
                # This is a frozen AST node - can't modify in place
                # Just return the node as-is for now
                # Full implementation would reconstruct the node
                pass
        
        # Handle specific child attributes
        if isinstance(node, fast.MatchExpression):
            node.expression = self.apply_recursive(node.expression, ctx)
            node.cases = [
                (self.apply_recursive(pattern, ctx), self.apply_recursive(expr, ctx))
                for pattern, expr in node.cases
            ]
        elif isinstance(node, fast.Block):
            node.statements = [self.apply_recursive(stmt, ctx) for stmt in node.statements]
        elif isinstance(node, fast.FunctionDeclaration):
            node.params = [self.apply_recursive(param, ctx) for param in node.params]
            node.body = self.apply_recursive(node.body, ctx)
        
        return node


class IfDesugarPass(DesugarPass):
    """Desugar if/else statements to pattern matching on Bool type.
    
    Transforms:
        if condition { then_expr } else { else_expr }
    
    Into:
        match condition {
            true => then_expr,
            false => else_expr
        }
    """
    
    def apply(self, node: fast.Node, ctx: DesugarContext) -> fast.Node:
        """Transform if/else to pattern matching."""
        if isinstance(node, fast.IfExpression):
            condition = node.condition
            then_branch = node.then_branch
            else_branch = node.else_branch
            
            # Create pattern matching cases
            # Case 1: true => then_branch
            true_pattern = fast.LiteralPattern(True)
            true_case = (true_pattern, then_branch)
            
            # Case 2: false => else_branch (or unit if no else)
            false_pattern = fast.LiteralPattern(False)
            else_expr = else_branch if else_branch else fast.Literal(())  # unit
            false_case = (false_pattern, else_expr)
            
            # Create match expression
            match_expr = fast.MatchExpression(condition, [true_case, false_case])
            
            return match_expr
        
        return node


class TraitDictionaryDesugarPass(DesugarPass):
    """Desugar trait method calls to dictionary lookups.
    
    Trait method calls like:
        stack.push(42)
    
    Are desugared to dictionary lookups:
        stack.trait_dict["Container::push"](stack, 42)
    
    This requires:
    1. Trait dictionary to be constructed during type checking
    2. Trait implementations to be collected
    3. Method calls to be transformed to dictionary lookups
    """
    
    def apply(self, node: fast.Node, ctx: DesugarContext) -> fast.Node:
        """Transform function calls to trait dictionary lookups."""
        if isinstance(node, fast.FunctionCall):
            func_name = node.name if isinstance(node.name, str) else str(node.name)
            
            # Only desugar if this is a trait method call
            # Check trait_impls to determine if this is a trait method
            is_trait_method = False
            trait_name = None
            
            if ctx.tables and ctx.tables.trait_impls:
                # For now, check if any trait implementation has this method
                for impl_key, impl_node in ctx.tables.trait_impls.items():
                    # impl_key is "struct_name:trait_name"
                    if hasattr(impl_node, 'methods'):
                        if func_name in impl_node.methods:
                            is_trait_method = True
                            trait_name = impl_key.split(':')[1]  # Extract trait name
                            break
            
            if not is_trait_method:
                # Not a trait method, return unchanged
                return node
            
            if len(node.arguments) > 0:
                receiver = node.arguments[0]
                args = node.arguments[1:]
                
                # Create field access: receiver.trait_dict
                trait_dict_access = fast.FieldAccess(receiver, "trait_dict")
                
                # Create literal for method name
                trait_name_str = trait_name if trait_name else "Trait"
                method_name_literal = fast.Literal(f"{trait_name_str}::{func_name}")
                
                # Create dictionary lookup using BinaryOperation for indexing
                # trait_dict_access[method_name_literal]
                dict_lookup = fast.BinaryOperation(trait_dict_access, method_name_literal, "Index")
                
                # Create function call with receiver + args
                new_call = fast.FunctionCall(dict_lookup, [receiver] + args)
                
                return new_call
        
        return node


def run_desugaring_passes(
    ast_root: fast.Node, 
    passes: list[DesugarPass],
    ctx: DesugarContext | None = None
) -> fast.Node:
    """Run multiple desugaring passes on an AST.
    
    Arguments:
        ast_root: The root AST node to desugar
        passes: List of desugaring passes to apply in order
        ctx: Optional desugaring context
        
    Returns:
        The desugared AST
    """
    if ctx is None:
        ctx = DesugarContext()
    
    result = ast_root
    for pass_obj in passes:
        result = pass_obj.apply_recursive(result, ctx)
    
    return result


def run_default_desugaring(ast_root: fast.Node, ctx: DesugarContext | None = None) -> fast.Node:
    """Run the default set of desugaring passes.
    
    Arguments:
        ast_root: The root AST node to desugar
        ctx: Optional desugaring context
        
    Returns:
        The desugared AST
    """
    passes = [
        TraitDictionaryDesugarPass(),  # Desugar trait method calls to dictionary lookups
        IfDesugarPass(),  # Desugar if statements to pattern matching
    ]
    
    return run_desugaring_passes(ast_root, passes, ctx)
