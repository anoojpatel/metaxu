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
from metaxu.compiler.frozen_borrow_checker import BorrowError, TypeCheckError


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
    """Base class for desugaring passes over the *mutable* parser AST.

    Each desugaring pass should inherit from this class and implement
    the `apply` method to transform a single node. `apply_recursive`
    walks the whole tree generically: every attribute of every node that
    holds a Node (directly, or inside a list/tuple/dict) is rewritten,
    and shared references (e.g. a statement present both in
    `Block.statements` and `Block.children`) are replaced consistently.

    Desugaring runs before freezing; frozen AST nodes are never mutated.
    """

    # Bookkeeping attributes that must not be traversed: `parent` points
    # upward (would re-walk ancestors), `scope`/`location` are metadata.
    _SKIP_FIELDS = frozenset({"parent", "scope", "location"})

    def apply(self, node: fast.Node, ctx: DesugarContext) -> fast.Node:
        """Apply desugaring transformation to a node.

        Arguments:
            node: The AST node to transform
            ctx: Desugaring context

        Returns:
            The transformed node (may be the same node if no transformation needed)
        """
        return node

    def apply_recursive(
        self,
        node: fast.Node,
        ctx: DesugarContext,
        _memo: dict[int, Any] | None = None,
    ) -> fast.Node:
        """Apply desugaring recursively to a node and its entire subtree."""
        if not isinstance(node, fast.Node):
            return node
        if _memo is None:
            _memo = {}
        key = id(node)
        if key in _memo:
            return _memo[key]

        result = self.apply(node, ctx)
        # Memoize both the original and the replacement so every reference to
        # this node (attribute fields and `children` lists alike) resolves to
        # the same transformed object, and so cycles (via any stray backrefs)
        # terminate.
        _memo[key] = result
        _memo[id(result)] = result

        self._recurse_fields(result, ctx, _memo)
        return result

    def _recurse_fields(self, node: fast.Node, ctx: DesugarContext, memo: dict[int, Any]) -> None:
        """Generically rewrite all child-bearing fields of a node in place."""
        for attr, value in list(vars(node).items()):
            if attr in self._SKIP_FIELDS:
                continue
            new_value = self._rewrite_value(value, ctx, memo)
            if new_value is not value:
                # Write back via object.__setattr__-compatible plain setattr;
                # name-mangled/underscored storage attrs (e.g. _body behind the
                # FunctionDeclaration.body property) are updated directly.
                setattr(node, attr, new_value)

    def _rewrite_value(self, value: Any, ctx: DesugarContext, memo: dict[int, Any]) -> Any:
        """Rewrite a field value, recursing through containers to find Nodes."""
        if isinstance(value, fast.Node):
            return self.apply_recursive(value, ctx, memo)
        if isinstance(value, list):
            new_items = [self._rewrite_value(item, ctx, memo) for item in value]
            if any(new is not old for new, old in zip(new_items, value)):
                return new_items
            return value
        if isinstance(value, tuple):
            new_items = tuple(self._rewrite_value(item, ctx, memo) for item in value)
            if any(new is not old for new, old in zip(new_items, value)):
                return new_items
            return value
        if isinstance(value, dict):
            new_map = {k: self._rewrite_value(v, ctx, memo) for k, v in value.items()}
            if any(new_map[k] is not value[k] for k in value):
                return new_map
            return value
        return value


class IfDesugarPass(DesugarPass):
    """Desugar if/else expressions to pattern matching on Bool type.

    Transforms:
        if condition { then_expr } else { else_expr }

    Into:
        match condition {
            true => then_expr,
            false => else_expr
        }

    NOTE: This pass is intentionally NOT part of run_default_desugaring.
    If/IfExpression have a direct HIR ("If") and MIR (br_if + phi-merge)
    lowering path which produces better control flow than routing every
    conditional through match lowering. The pass is kept for opt-in use by
    pipelines that want a match-only core language.
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


# Separator used in mangled impl-method names. "$" cannot appear in Metaxu
# identifiers, so splitting on it is unambiguous even when trait/type/method
# names themselves contain underscores.
IMPL_SEP = "$"

# The compiler's own name space.  EVERY symbol the compiler synthesizes or
# emits calls to lives behind a leading double underscore:
#   __impl$T$Ty$m, __trait$m, __static$Ty$m          (trait/static dispatch)
#   __effect_default$E$op, __effect_runtime$E$op,
#   __mx_effect_runtime$SYMBOL                       (effect lowering)
#   __module_init                                    (module constants)
#   __builtin$m                                      (method-position builtins)
#   __index_get/__index_set/__index_store/__slice_get/__range/__zip/__cast/
#   __vec_lit/__vec_dim/__vec_zeros/__vec_filled/__vec_comprehension/
#   __list_lit/__list_concat                         (compiler intrinsics)
# Reserving the whole prefix (rather than an enumerated list) is what makes
# "a user function wins over a same-named builtin" safe: the names the
# compiler generates or emits calls to can never be shadowed, and a user
# declaration that tries is a loud error instead of a silent override.
# See docs/name_precedence.md.
RESERVED_NAME_PREFIX = "__"


def is_reserved_name(name: str) -> bool:
    """True for names in the compiler's reserved namespace (see above)."""
    return isinstance(name, str) and name.startswith(RESERVED_NAME_PREFIX)


class CoherenceError(TypeCheckError):
    """Two distinct implement blocks define the same (trait, type, method).

    A typed diagnostic (kind "type-coherence") so it travels the same
    structured channel as every other compile rejection; `location` is
    the source position of the offending method (None when the node
    carries none); when known, the message carries the standard
    `file:line:column` prefix and an excerpt with a caret.
    """

    def __init__(self, message: str, location: Any = None):
        self.location = location
        from metaxu.errors import format_location, source_excerpt
        if location is not None:
            message = f"{format_location(location)}: {message}"
            excerpt = source_excerpt(location)
            if excerpt:
                message = f"{message}\n{excerpt}"
        self.errors = [BorrowError(message=message, node_id=-1,
                                   kind="type-coherence", variable="",
                                   location=location)]
        Exception.__init__(self, message)


def node_location(node: Any) -> Any:
    """The parser-attached SourceLocation of `node`, when it has one."""
    from metaxu.errors import SourceLocation
    loc = getattr(node, "location", None)
    return loc if isinstance(loc, SourceLocation) and loc.line else None
IMPL_PREFIX = f"__impl{IMPL_SEP}"


def type_base_name(t: Any) -> str:
    """Best-effort base name of a type expression node.

    `Dog` -> "Dog"; `Stack<E>` -> "Stack" (type arguments are erased — v1
    trait dispatch is on the head type constructor only).
    """
    if t is None:
        return ""
    if isinstance(t, str):
        return t
    if isinstance(t, fast.TypeReference):
        return type_base_name(t.name)
    if isinstance(t, fast.TypeApplication):
        return type_base_name(t.type_constructor)
    if isinstance(t, fast.GenericInstance):
        return type_base_name(t.base)
    name = getattr(t, "name", None)
    if name is not None:
        return type_base_name(name)
    return str(t)


def mangle_impl_method(trait_name: str, type_name: str, method_name: str) -> str:
    """Mangled top-level function name for one impl-block method."""
    return f"{IMPL_PREFIX}{trait_name}{IMPL_SEP}{type_name}{IMPL_SEP}{method_name}"


def parse_impl_method_name(fn_name: str) -> tuple[str, str, str] | None:
    """Inverse of mangle_impl_method; None when fn_name is not a mangled impl."""
    if not fn_name.startswith(IMPL_PREFIX):
        return None
    parts = fn_name.split(IMPL_SEP)
    # ["__impl", trait, type, method]
    if len(parts) != 4:
        return None
    return parts[1], parts[2], parts[3]


class TraitImplDesugarPass(DesugarPass):
    """Rewrite `implement Trait for Type { fn m(self, ...) {...} }` blocks into
    plain top-level FunctionDeclarations with mangled names.

    Each impl method becomes `__impl$Trait$Type$method` with `self` as its
    first parameter (prepended when the method body uses `self` without
    declaring it, as in the `implements Type: Trait` legacy syntax). When
    `self` carries no type annotation, the impl's target type is attached so
    inference sees the receiver type.

    Dispatch itself happens at RUNTIME in the MIR interpreter: a method call
    `recv.m(args)` on a trait method lowers to a `__trait$m` call whose first
    operand is the receiver, and the interpreter picks the mangled function
    matching the receiver's runtime type name (MxStruct.name /
    MxVariant.enum_name). This static-name-erased, runtime-dispatched scheme
    is the documented v1 choice: it needs no reliable static receiver types
    and generic impls dispatch on the head type constructor.
    """

    def __init__(self) -> None:
        # Replacements memoized per Implementation object so that the same
        # impl referenced from multiple lists (`statements` and `children`)
        # expands to the same FunctionDeclaration objects.
        self._expanded: dict[int, list[fast.Node]] = {}
        # Coherence: (trait, type, method) -> id of the defining impl. A
        # second DISTINCT impl for the same key silently overwriting the
        # first would make dispatch order-dependent, so it is an error.
        self._seen_methods: dict[tuple[str, str, str], int] = {}

    def apply(self, node: fast.Node, ctx: DesugarContext) -> fast.Node:
        # Splice Implementation items out of any list-valued field
        # (ModuleBody.statements, Block.statements, children lists, ...).
        for attr, value in list(vars(node).items()):
            if attr in self._SKIP_FIELDS or not isinstance(value, list):
                continue
            if not any(isinstance(item, fast.Implementation) for item in value):
                continue
            new_items: list[Any] = []
            for item in value:
                if isinstance(item, fast.Implementation):
                    new_items.extend(self._expand_impl(item))
                else:
                    new_items.append(item)
            setattr(node, attr, new_items)
        return node

    def _expand_impl(self, impl: fast.Implementation) -> list[fast.Node]:
        cached = self._expanded.get(id(impl))
        if cached is not None:
            return cached
        trait_name = type_base_name(impl.interface_name)
        type_name = type_base_name(impl.type_name)
        const_dims = _const_generic_dims(impl)
        # Impl-level where clause + type parameters: attached to each mangled
        # function (underscored attrs, frozen into the FunctionDeclaration
        # payload as impl_where/impl_params) so the constraint emitter can
        # enforce decidable impl where clauses at coherence-load time.
        #
        # The type parameters are attached UNCONDITIONALLY: they are a
        # property of the impl block, not of its where clause. An impl's
        # parameters are also in scope as VALUE names inside its methods —
        # a const generic is bound to a receiver dimension at method entry
        # (see `_const_dims` below) and a plain parameter is the argument of
        # `type_of` — so gating them on `where_clause is not None` made
        # `implement<T, const N: int> ... { .. type_of(T) .. }` look like a
        # read of an undefined name (`compiler/name_resolution.py`).
        impl_where = getattr(impl, "where_clause", None)
        impl_tparams = _impl_type_param_names(impl)
        out: list[fast.Node] = []
        for m in impl.methods or []:
            if not isinstance(m, (fast.FunctionDeclaration, fast.MethodImplementation)):
                continue
            fn = self._method_to_function(m, trait_name, type_name)
            if fn is not None:
                fn._impl_type_params = impl_tparams
            if fn is not None and impl_where is not None:
                fn._impl_where_clause = impl_where
            if fn is not None and const_dims:
                # Record which const-generic size parameters of the impl's
                # receiver type map to which runtime dimension of `self`
                # (e.g. `vector[T,N]` -> N is dim 0; `vector[vector[T,N],M]`
                # -> M is dim 0, N is dim 1). The HIR builder turns these
                # into entry bindings so the method body's uses of N/M read
                # the receiver's actual runtime shape. Underscored attr:
                # invisible to the generic desugar walk and to freezing.
                fn._const_dims = const_dims
            if fn is not None:
                key = (trait_name, type_name, str(fn.name))
                method = key[2].split(IMPL_SEP)[-1]
                # Duplicate within THIS block (silent last-wins otherwise).
                if any(str(prev.name) == str(fn.name) for prev in out):
                    raise CoherenceError(
                        f"Conflicting implementations: method {method!r} of "
                        f"trait '{trait_name}' for type '{type_name}' is "
                        f"defined twice in the same implement block",
                        location=node_location(m) or node_location(impl))
                # Duplicate across distinct blocks.
                owner = self._seen_methods.setdefault(key, id(impl))
                if owner != id(impl):
                    raise CoherenceError(
                        f"Conflicting implementations: method {method!r} of "
                        f"trait '{trait_name}' for type '{type_name}' is "
                        f"defined by more than one implement block",
                        location=node_location(m) or node_location(impl))
                out.append(fn)
        self._expanded[id(impl)] = out
        return out

    def _method_to_function(
        self, m: fast.Node, trait_name: str, type_name: str
    ) -> fast.Node | None:
        method_name = str(getattr(m, "name", "") or "")
        if not method_name:
            return None
        params = list(getattr(m, "params", None) or [])
        self_param = next(
            (p for p in params if str(getattr(p, "name", "")) == "self"), None)
        if self_param is None and _mentions_self(getattr(m, "body", None)):
            # `implements Type: Trait` methods use `self` without declaring it.
            self_param = fast.Parameter("self")
            params.insert(0, self_param)
        # A method that neither declares nor uses `self` stays parameter-less:
        # it is a static method (e.g. `fn new(...)` in an inherent impl),
        # callable as `Type.method(args)`.
        if self_param is not None and getattr(self_param, "type_annotation", None) is None:
            self_param.type_annotation = fast.TypeReference(type_name)
        mangled = mangle_impl_method(trait_name, type_name, method_name)
        if isinstance(m, fast.FunctionDeclaration):
            # Reuse the node (body/children bookkeeping stays intact); only
            # the name and parameter list change.
            m.name = mangled
            m.params = params
            return m
        # MethodImplementation -> fresh FunctionDeclaration
        return fast.FunctionDeclaration(
            mangled, params, list(getattr(m, "body", None) or []),
            return_type=getattr(m, "return_type", None))


def _type_arg_name(t: Any) -> str | None:
    """Name of a type argument when it is a bare reference (else None)."""
    if isinstance(t, fast.TypeReference):
        return str(getattr(t, "name", "") or "") or None
    if isinstance(t, fast.TypeParameter):
        return str(getattr(t, "name", "") or "") or None
    return None


def _impl_type_param_names(impl: fast.Implementation) -> list[str]:
    """Type-parameter names an implement block binds: its declared
    `implement<T, ...>` parameters plus bare type arguments of its target
    type application (`implement Show for Pair[T]` binds T). Used to
    distinguish decidable (concrete-type) impl where constraints from
    instantiation-dependent ones."""
    names: list[str] = []
    for tp in getattr(impl, "type_params", None) or []:
        n = _type_arg_name(tp) if not isinstance(tp, str) else tp
        if n:
            names.append(str(n))
    target = getattr(impl, "type_name", None)
    if isinstance(target, fast.TypeApplication):
        for a in getattr(target, "type_args", None) or []:
            n = _type_arg_name(a)
            if n:
                names.append(n)
    return names


def _const_generic_dims(impl: fast.Implementation) -> tuple[tuple[str, int], ...]:
    """Map the impl receiver's const-generic size names to runtime dims.

    For `implement<..., const N: int> ... for vector[T, N]` the size name N
    is dimension 0 of the receiver (its length); for a matrix receiver
    `vector[vector[T, N], M]`, M is dimension 0 (rows) and N dimension 1
    (columns). Only names that are declared type parameters of the impl are
    mapped — literal sizes (vector[float, 4]) produce no binding.
    """
    recv = getattr(impl, "type_name", None)
    args = None
    if isinstance(recv, fast.TypeApplication) and str(recv.type_constructor) == "vector":
        args = list(recv.type_args or [])
    elif isinstance(recv, fast.VectorTypeExpression):
        args = [getattr(recv, "base_type", None), getattr(recv, "size", None)]
    if not args or len(args) < 2:
        return ()
    param_names = {
        str(getattr(tp, "name", "") or "")
        for tp in (getattr(impl, "type_params", None) or [])
    }
    out: list[tuple[str, int]] = []
    size0 = _type_arg_name(args[1])
    if size0 and not size0.lstrip("-").isdigit() and size0 in param_names:
        out.append((size0, 0))
    elem = args[0]
    if isinstance(elem, fast.TypeApplication) and str(elem.type_constructor) == "vector":
        eargs = list(elem.type_args or [])
        if len(eargs) >= 2:
            size1 = _type_arg_name(eargs[1])
            if size1 and not size1.lstrip("-").isdigit() and size1 in param_names:
                out.append((size1, 1))
    return tuple(out)


def _mentions_self(node: Any, _seen: set[int] | None = None) -> bool:
    """True when any node in the subtree references the identifier `self`.

    Detects `self` stored as a bare string in any AST field (Variable.name,
    FieldAccess.base, Assignment targets, ...). String literal VALUES are
    excluded so a `"self"` string constant does not count as a use.
    """
    if _seen is None:
        _seen = set()
    # `self` inside list-valued attributes: QualifiedFunctionCall/QualifiedName
    # store name parts as plain strings (e.g. parts=["self", "incr"]).
    if node == "self":
        return True
    if isinstance(node, (list, tuple)):
        return any(_mentions_self(item, _seen) for item in node)
    if not isinstance(node, fast.Node):
        return False
    if id(node) in _seen:
        return False
    _seen.add(id(node))
    for attr, value in vars(node).items():
        if attr in DesugarPass._SKIP_FIELDS:
            continue
        if isinstance(node, fast.Literal) and attr == "value":
            continue
        if value == "self":
            return True
        if isinstance(value, (fast.Node, list, tuple)):
            if _mentions_self(value, _seen):
                return True
        elif isinstance(value, dict):
            if any(_mentions_self(v, _seen) for v in value.values()):
                return True
    return False


_BRACKET_PRIMITIVES = frozenset({
    "Int", "int", "String", "str", "string", "Bool", "bool", "Float", "float",
})


class BracketCtorCallDesugarPass(DesugarPass):
    """Rewrite bracket-form explicit instantiations in call position.

    `Full[Int](x)` parses as CallExpression(IndexExpression(Full, Int), [x])
    because `[...]` in expression position is indexing — so bracket-form
    constructor/function instantiations used to bypass instantiation
    checking entirely (and HIR had no lowering for them). This pass
    recognizes the shape where it is decidable:

      - the indexed base is a bare name that is a KNOWN generic enum
        variant or a KNOWN generic function (declared with type params), and
      - every index entry is a type display: a primitive name, a declared
        struct/enum name, or a declared type-parameter name

    and rewrites it to the equivalent FunctionCall with explicit type_args,
    i.e. exactly the node `Full<Int>(x)` produces. Downstream phases
    (constraint emitter, HIR lowering, monomorphization) then treat both
    spellings identically. Any other indexed call (`arr[i](x)`, unknown
    names, value indexes) is left untouched.
    """

    def __init__(self) -> None:
        self._generic_callables: set[str] = set()
        self._type_names: set[str] = set(_BRACKET_PRIMITIVES)

    # -- definition scan ---------------------------------------------------

    def _scan(self, node: Any, _seen: set[int] | None = None) -> None:
        if _seen is None:
            _seen = set()
        if isinstance(node, (list, tuple)):
            for item in node:
                self._scan(item, _seen)
            return
        if isinstance(node, dict):
            for item in node.values():
                self._scan(item, _seen)
            return
        if not isinstance(node, fast.Node) or id(node) in _seen:
            return
        _seen.add(id(node))
        if isinstance(node, fast.EnumDefinition):
            name = str(getattr(node, "name", "") or "")
            if name:
                self._type_names.add(name)
            tparams = getattr(node, "type_params", None) or []
            for tp in tparams:
                n = tp if isinstance(tp, str) else _type_arg_name(tp)
                if n:
                    self._type_names.add(str(n))
            if tparams:
                for v in getattr(node, "variants", None) or []:
                    vname = str(getattr(v, "name", "") or "")
                    if vname:
                        self._generic_callables.add(vname)
        elif isinstance(node, fast.StructDefinition):
            name = str(getattr(node, "name", "") or "")
            if name:
                self._type_names.add(name)
            for tp in getattr(node, "type_params", None) or []:
                n = tp if isinstance(tp, str) else _type_arg_name(tp)
                if n:
                    self._type_names.add(str(n))
        elif isinstance(node, fast.FunctionDeclaration):
            tparams = getattr(node, "type_params", None) or []
            for tp in tparams:
                n = tp if isinstance(tp, str) else _type_arg_name(tp)
                if n:
                    self._type_names.add(str(n))
            if tparams:
                name = str(getattr(node, "name", "") or "")
                if name:
                    self._generic_callables.add(name)
        for attr, value in list(vars(node).items()):
            if attr in self._SKIP_FIELDS:
                continue
            if isinstance(value, (fast.Node, list, tuple, dict)):
                self._scan(value, _seen)

    # -- rewrite -----------------------------------------------------------

    @staticmethod
    def _display_name(node: Any) -> str | None:
        if isinstance(node, fast.Variable):
            n = getattr(node, "name", None)
            return str(n) if isinstance(n, str) else None
        if isinstance(node, (fast.TypeReference, fast.TypeParameter)):
            n = getattr(node, "name", None)
            return str(n) if isinstance(n, str) else None
        return None

    def apply_recursive(self, node, ctx, _memo=None):
        if _memo is None:
            # First call is the program root: collect definitions before
            # rewriting so recognition is declaration-order independent.
            self._scan(node)
        return super().apply_recursive(node, ctx, _memo)

    def apply(self, node: fast.Node, ctx: DesugarContext) -> fast.Node:
        if not isinstance(node, fast.CallExpression):
            return node
        callee = getattr(node, "callee", None)
        if not isinstance(callee, fast.IndexExpression):
            return node
        base_name = self._display_name(getattr(callee, "base", None))
        if base_name is None or base_name not in self._generic_callables:
            return node
        idx = getattr(callee, "index", None)
        idx_list = idx if isinstance(idx, list) else [idx]
        targ_names: list[str] = []
        for entry in idx_list:
            n = self._display_name(entry)
            if n is None or n not in self._type_names:
                return node          # not a type display: a real index
            targ_names.append(n)
        if not targ_names:
            return node
        call = fast.FunctionCall(base_name, list(getattr(node, "arguments", None) or []))
        call.type_args = [fast.TypeReference(n) for n in targ_names]
        call.location = getattr(node, "location", None)
        return call


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
        # Bracket-form explicit instantiations (`Full[Int](x)`) become plain
        # FunctionCalls with type_args, identical to `Full<Int>(x)`.
        BracketCtorCallDesugarPass(),
        # Rewrite implement-blocks into mangled top-level functions; method
        # calls dispatch on the receiver's runtime type in the MIR interpreter.
        TraitImplDesugarPass(),
        # TraitDictionaryDesugarPass (dictionary-passing dispatch) is not part
        # of the default pipeline: v1 uses TraitImplDesugarPass + runtime
        # dispatch instead. The class is kept for opt-in/experimental use.
        # IfDesugarPass is deliberately omitted: if/else keeps its native
        # HIR/MIR lowering (see IfDesugarPass docstring).
    ]

    return run_desugaring_passes(ast_root, passes, ctx)
