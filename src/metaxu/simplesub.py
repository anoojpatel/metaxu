"""SimpleSub-style type inference for Metaxu over CompactTypes.

Two engines live in this module:

1. ``TypeInferencer`` — the historical, unification-based solver.
   ``solve_constraints`` unifies each buffered constraint via
   ``type_defs.unify`` (which writes ``bounds.upper_bound`` as a union-find
   pointer that ``CompactType.find()`` follows). This is what the compiler
   pipeline drives today through ``compiler/simplesub_adapter.py``: the
   frozen-AST emitter produces unify/subtype/class constraints, hard type
   errors come from the adapter's constraint-graph conflict detection plus
   ``frozen_constraint_checker``, and the unified CompactTypes feed HIR.

2. ``Biunifier`` — real biunification per Lionel Parreaux's *The Simple
   Essence of Algebraic Subtyping* (ICFP 2020). ``constrain(lhs, rhs)``
   decomposes subtype constraints structurally with a processed-pair cache
   (termination on recursive constraint graphs), keeps MULTIPLE lower and
   upper bounds per variable in side tables (never touching the union-find
   pointer in ``CompactType.bounds``), and propagates transitively through
   those bounds. ``coalesce``/``principal_type`` turn the solved graph into
   principal types with unions, intersections and mu-recursive binders,
   followed by the paper's simplification passes (polar-variable removal,
   co-occurrence analysis, flattening). Levels with extrusion implement
   let-polymorphism: ``enter_level``/``exit_level``, ``generalize`` and
   ``instantiate``.

Scope, honestly stated: the pipeline's constraint stream never generalizes
— every function is bound to a single monomorphic CompactType shared by
all of its call sites — so levels/extrusion/instantiation are exercised at
the engine level only (see ``compiler/tests/test_biunification.py``). The
pipeline exposes the Biunifier lazily through
``SimpleSubFacade.principal_type_of`` for principal-type queries;
compilation-blocking diagnostics continue to come exclusively from the
adapter's conflict detection and the frozen constraint/borrow checkers,
and ``Biunifier.errors`` are advisory. The primitive lattice is flat (no
``Int <: Float``): the biunifier reports exactly the same primitive
clashes the flat unifier would, so no program changes compilability.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union
from enum import Enum
from metaxu.type_defs import (
    Type, FunctionType, TypeVar, TypeScheme, TypeConstructor,
    CompactType, TypeBounds, unfold_once, unify, compose_variance,
    substitute_compact, next_id, RecursiveType, get_constructor_variances
)
import metaxu.metaxu_ast as ast

class Polarity(Enum):
    POSITIVE = 1   # Covariant position
    NEGATIVE = -1  # Contravariant position
    NEUTRAL = 0    # Invariant position

    def flip(self) -> 'Polarity':
        """Flip polarity (used when going under contravariant positions)"""
        if self == Polarity.POSITIVE:
            return Polarity.NEGATIVE
        elif self == Polarity.NEGATIVE:
            return Polarity.POSITIVE
        return Polarity.NEUTRAL

@dataclass
class Constraint:
    """A type constraint between two types"""
    left: CompactType
    right: CompactType
    polarity: Polarity

class VarianceInferencer:
    """Infers variance of type parameters based on their usage"""
    def __init__(self):
        self.type_var_positions: Dict[int, Set[Polarity]] = {}  # Use CompactType ID

    def record_usage(self, type_var: CompactType, polarity: Polarity):
        """Record a usage of a type variable in a certain polarity"""
        if type_var.kind != 'var':
            return
        if type_var.id not in self.type_var_positions:
            self.type_var_positions[type_var.id] = set()
        self.type_var_positions[type_var.id].add(polarity)

    def infer_variance(self, type_var: CompactType) -> str:
        """Infer variance for a type variable based on recorded positions"""
        if type_var.kind != 'var':
            return 'invariant'
            
        positions = self.type_var_positions.get(type_var.id, set())
        
        # If never used, assume invariant
        if not positions:
            return 'invariant'
            
        # If used in both positive and negative positions -> invariant
        if Polarity.POSITIVE in positions and Polarity.NEGATIVE in positions:
            return 'invariant'
            
        # If only used in positive positions -> covariant
        if Polarity.POSITIVE in positions:
            return 'covariant'
            
        # If only used in negative positions -> contravariant
        if Polarity.NEGATIVE in positions:
            return 'contravariant'
            
        return 'invariant'

class TypeInferencer:
    """Type inference using CompactTypes"""
    def __init__(self):
        self.constraints: List[Constraint] = []
        self.type_vars: Dict[str, CompactType] = {}
        self.next_var_id = 0
        self.variance_inferencer = VarianceInferencer()
        self._biunifier: Optional['Biunifier'] = None
        # unification failures from the last solve_constraints(); the
        # facade surfaces them on request (a failed constraint never
        # aborts the solve, so callers read this list afterwards)
        self.errors: List[str] = []
        
    def to_compact_type(self, ty: Type) -> CompactType:
        """Convert Type to CompactType"""
        # Bridge AST basic types to CompactType primitives
        if isinstance(ty, ast.BasicType):
            # Preserve the AST name (e.g., 'Int', 'String', 'Bool', 'Void')
            return CompactType(
                id=next_id(),
                kind='primitive',
                name=ty.name
            )
        # Bridge AST type parameter to a fresh/interned var
        if hasattr(ast, 'TypeParameter') and isinstance(ty, ast.TypeParameter):
            key = getattr(ty, 'name', 'T')
            if key not in self.type_vars:
                self.type_vars[key] = CompactType(
                    id=next_id(),
                    kind='var',
                    bounds=TypeBounds()
                )
            return self.type_vars[key]
        # Bridge AST TypeReference (could be primitive, type param, or zero-arity constructor)
        if hasattr(ast, 'TypeReference') and isinstance(ty, ast.TypeReference):
            cname = getattr(ty, 'name', None) or getattr(ty, 'identifier', 'Type')
            # Primitives
            if cname in ('Int', 'String', 'Bool', 'Void'):
                return CompactType(id=next_id(), kind='primitive', name=cname)
            # Likely a type parameter
            if cname in self.type_vars or (isinstance(cname, str) and len(cname) <= 2 and cname[:1].isupper()):
                if cname not in self.type_vars:
                    self.type_vars[cname] = CompactType(id=next_id(), kind='var', bounds=TypeBounds())
                return self.type_vars[cname]
            # Otherwise treat as a named zero-arity constructor
            constructor = TypeConstructor(cname, 0)
            return CompactType(
                id=next_id(),
                kind='constructor',
                constructor=constructor,
                type_args=[],
                name=cname
            )
        # Bridge AST TypeApplication (constructor with type arguments)
        if hasattr(ast, 'TypeApplication') and isinstance(ty, ast.TypeApplication):
            # Extract constructor/base name from various possible shapes
            cname = None
            # Prefer explicit field used by this AST: 'type_constructor'
            tc = getattr(ty, 'type_constructor', None)
            if isinstance(tc, str) and tc:
                cname = tc
            base = getattr(ty, 'constructor', None)
            if not cname and base is not None:
                cname = getattr(base, 'name', None) or getattr(base, 'identifier', None)
            if not cname:
                base2 = getattr(ty, 'base_type', None)
                if isinstance(base2, str):
                    cname = base2
                elif base2 is not None:
                    cname = getattr(base2, 'name', None) or getattr(base2, 'identifier', None)
            if not cname:
                # Try common alternatives
                base3 = getattr(ty, 'base', None) or getattr(ty, 'callee', None) or getattr(ty, 'type', None)
                if isinstance(base3, str):
                    cname = base3
                elif base3 is not None:
                    cname = getattr(base3, 'name', None) or getattr(base3, 'identifier', None)
            if not cname:
                cname = getattr(ty, 'name', None)
            if not cname:
                cname = 'Generic'
            # Type arguments could be under different fields
            args = getattr(ty, 'type_args', None)
            if args is None:
                args = getattr(ty, 'arguments', [])
            if args is None:
                args = getattr(ty, 'params', None)
            if args is None:
                args = getattr(ty, 'type_parameters', None)
            args = args or []
            constructor = TypeConstructor(cname, len(args))
            return CompactType(
                id=next_id(),
                kind='constructor',
                constructor=constructor,
                type_args=[self.to_compact_type(a) for a in args],
                name=cname
            )
        # Bridge AST generic type expressions (e.g., Box[T]) to constructor CompactType
        if hasattr(ast, 'GenericType') and isinstance(ty, ast.GenericType):
            base_name = getattr(ty, 'name', None) or getattr(ty, 'base_type', None) or 'Generic'
            targs = getattr(ty, 'type_args', []) or []
            constructor = TypeConstructor(base_name, len(targs))
            return CompactType(
                id=next_id(),
                kind='constructor',
                constructor=constructor,
                type_args=[self.to_compact_type(a) for a in targs],
                name=base_name
            )
        if isinstance(ty, TypeVar):
            if ty.name not in self.type_vars:
                self.type_vars[ty.name] = CompactType(
                    id=next_id(),
                    kind='var',
                    bounds=TypeBounds()
                )
            return self.type_vars[ty.name]
            
        elif isinstance(ty, TypeConstructor):
            return CompactType(
                id=next_id(),
                kind='constructor',
                constructor=ty,
                type_args=[]
            )
            
        elif isinstance(ty, RecursiveType):
            compact = CompactType(
                id=next_id(),
                kind='recursive',
                recursive_ref=ty.type_name,
                type_args=[self.to_compact_type(param) 
                          for param in ty.type_parameters]
            )
            if ty.get_resolved_type():
                resolved = self.to_compact_type(ty.get_resolved_type())
                compact.bounds = TypeBounds(upper_bound=resolved)
            return compact
            
        # Handle ast.FunctionType from metaxu_ast
        if hasattr(ast, 'FunctionType') and isinstance(ty, ast.FunctionType):
            return CompactType(
                id=next_id(),
                kind='function',
                param_types=[self.to_compact_type(p) for p in ty.param_types],
                return_type=self.to_compact_type(ty.return_type),
                linearity=str(ty.linearity) if hasattr(ty, 'linearity') else 'many'
            )

        # Default: create a fresh var when we don't recognize the type
        return CompactType.fresh_var()

    def analyze_type_definition(self, type_def: Type, polarity: Polarity = Polarity.NEUTRAL):
        """Analyze a type definition to infer variance of its type parameters"""
        compact = self.to_compact_type(type_def)
        self._analyze_type(compact, polarity)

    def check_function_type(self, func_type: CompactType, polarity: Polarity) -> CompactType:
        """Check a function type, handling contravariance in argument positions"""
        if not isinstance(func_type, FunctionType):
            return func_type
            
        # Arguments are contravariant
        param_types = [self._analyze_type(param, polarity.flip()) 
                      for param in func_type.param_types]
                      
        # Return type is covariant
        return_type = self._analyze_type(func_type.return_type, polarity)
        
        return FunctionType(param_types, return_type, func_type.linearity)

    def _analyze_type(self, ty: CompactType, polarity: Polarity):
        """Analyze a type to record variable usage positions"""
        ty = ty.find()
        
        if ty.kind == 'var':
            self.variance_inferencer.record_usage(ty, polarity)
            
        elif ty.kind == 'constructor':
            for arg, param_variance in zip(ty.type_args or [],
                                        get_constructor_variances(ty.constructor)):
                composed = compose_variance(polarity, param_variance)
                self._analyze_type(arg, composed)
                
        elif ty.kind == 'function':
            # Check function body with proper variance
            body_type = self._analyze_type(ty.body, polarity) if hasattr(ty, 'body') else None
            
            # Check function type itself
            return self.check_function_type(ty, polarity)

        elif ty.kind == 'recursive':
            # Analyze parameters
            for param in (ty.type_args or []):
                self._analyze_type(param, polarity)
            # Analyze resolved type if available
            if ty.bounds and ty.bounds.upper_bound:
                self._analyze_type(ty.bounds.upper_bound, polarity)

    def fresh_type_var(self, name_hint: str = "T") -> CompactType:
        """Create a fresh type variable"""
        # Do NOT store by name_hint; that aliases different occurrences.
        return CompactType(
            id=next_id(),
            kind='var',
            bounds=TypeBounds()
        )

    def add_constraint(self, left: CompactType, right: CompactType, 
                      polarity: Polarity):
        """Add a new constraint to the system"""
        self.constraints.append(Constraint(left, right, polarity))

    def solve_constraints(self) -> Dict[int, CompactType]:
        """Solve the collected constraints using unification.
        Returns a mapping from variable IDs to their resolved CompactTypes.
        """
        solution: Dict[int, CompactType] = {}
        errors: List[str] = []
        
        for constraint in self.constraints:
            left = constraint.left.find()
            right = constraint.right.find()
            
            # Handle recursive types
            if left.kind == 'recursive' or right.kind == 'recursive':
                # Unfold recursive types once
                if left.kind == 'recursive':
                    left = unfold_once(left)
                if right.kind == 'recursive':
                    right = unfold_once(right)
            
            # Unify with appropriate variance
            variance = 'covariant' if constraint.polarity == Polarity.POSITIVE else \
                      'contravariant' if constraint.polarity == Polarity.NEGATIVE else \
                      'invariant'
            
            try:
                ok = unify(left, right, variance)
            except Exception as e:
                ok = False
            if not ok:
                # Record and continue; don't abort entire solve so other constraints can resolve
                errors.append(f"Cannot unify {left} with {right} (variance={variance})")
                continue
                
            # Record solution by variable ID to avoid hashing CompactType
            if left.kind == 'var':
                solution[left.id] = right
            elif right.kind == 'var':
                solution[right.id] = left
        
        # Optional: debug unify errors without breaking tests
        # if errors:
        #     print("Type inference warnings:", errors)
        self.errors = errors
        return solution

    def finalize_type_definition(self, type_def: TypeConstructor):
        """Finalize a type definition by inferring variance for all its type parameters"""
        for param in type_def.type_params:
            compact_param = self.type_vars.get(param.name)
            if compact_param:
                variance = self.variance_inferencer.infer_variance(compact_param)
                param.inferred_variance = variance

    def infer_expression(self, expr: 'ast.Expression', polarity: Polarity) -> CompactType:
        """Infer type of an expression, generating constraints with polarity"""
        # Handle struct/record constructor literals (extern AST), e.g., Pair { first=..., second=... }
        # Detect by shape rather than class to avoid tight coupling to AST implementation
        if hasattr(expr, 'field_assignments') or hasattr(expr, 'fields') or hasattr(expr, 'type_constructor') \
           or hasattr(expr, 'struct_name') or hasattr(expr, 'record_name'):
            # Resolve constructor name from multiple possible attributes
            ctor_name = (getattr(expr, 'type_constructor', None)
                         or getattr(expr, 'constructor_name', None)
                         or getattr(expr, 'struct_name', None)
                         or getattr(expr, 'record_name', None)
                         or getattr(expr, 'name', None))
            if isinstance(ctor_name, ast.Node):
                n = getattr(ctor_name, 'name', None)
                if not n:
                    parts = getattr(ctor_name, 'parts', None)
                    if isinstance(parts, list) and parts:
                        n = '.'.join(parts)
                ctor_name = n or str(ctor_name)

            # Gather field values in declaration order (dict preserves insertion order in Py3.7+)
            ordered_values = []
            fm = (getattr(expr, 'fields', None) or getattr(expr, 'field_values', None)
                  or getattr(expr, 'properties', None) or getattr(expr, 'args', None))
            if isinstance(fm, dict) and fm:
                for _, v in fm.items():
                    ordered_values.append(v)
            else:
                entries = (getattr(expr, 'field_assignments', None) or getattr(expr, 'fields', None)
                           or getattr(expr, 'properties', None) or getattr(expr, 'args', None))
                if isinstance(entries, list) and entries:
                    for entry in list(entries):
                        v = (getattr(entry, 'value', None) or getattr(entry, 'expr', None)
                             or getattr(entry, 'rhs', None) or getattr(entry, 'initializer', None))
                        if v is not None:
                            ordered_values.append(v)

            # Infer each field's type and constrain against fresh type args
            arg_vars: List[CompactType] = []
            for i, v in enumerate(ordered_values):
                field_ty = self.infer_expression(v, polarity)
                arg_var = self.fresh_type_var(f"arg{i}")
                # Covariant constraint: field type <= arg var
                self.add_constraint(field_ty, arg_var, Polarity.POSITIVE)
                arg_vars.append(arg_var)

            # Build constructor CompactType for the struct literal
            arity = len(arg_vars)
            constructor = TypeConstructor(ctor_name or 'Struct', arity)
            ctor_compact = CompactType(
                id=next_id(),
                kind='constructor',
                constructor=constructor,
                type_args=arg_vars,
                name=ctor_name or 'Struct'
            )
            # Attach back for downstream use
            try:
                expr.type_var = ctor_compact
            except Exception:
                pass
            return ctor_compact

        if isinstance(expr, ast.FunctionCall):
            # Do not special-case Option/Some here. Calls are generic; without a function type
            # environment modeled, return a fresh var so constraints can still attach elsewhere.
            return self.fresh_type_var("call")

        if isinstance(expr, ast.Lambda):
            param_type = self.fresh_type_var("param")
            # Parameters are contravariant
            self._analyze_type(param_type, polarity.flip())
            
            body_type = self.infer_expression(expr.body, polarity)
            # Return type is covariant
            self._analyze_type(body_type, polarity)
            
            return CompactType(
                id=next_id(),
                kind='function',
                param_types=[param_type],
                return_type=body_type
            )
            
        elif isinstance(expr, ast.Application):
            func_type = self.infer_expression(expr.func, polarity)
            # Arguments are in contravariant position
            arg_type = self.infer_expression(expr.arg, polarity.flip())
            result_type = self.fresh_type_var("result")
            
            self.add_constraint(
                func_type,
                CompactType(
                    id=next_id(),
                    kind='function',
                    param_types=[arg_type],
                    return_type=result_type
                ),
                polarity
            )
            return result_type
            
        elif isinstance(expr, ast.TypeApplication):
            # Analyze how type arguments are used
            constructor = self.infer_expression(expr.constructor, polarity)
            for arg in expr.type_args:
                self._analyze_type(arg, polarity)
                
            return CompactType(
                id=next_id(),
                kind='constructor',
                constructor=constructor,
                type_args=[self.to_compact_type(arg) for arg in expr.type_args]
            )
            
        return self.fresh_type_var("unknown")

    def generalize(self, ty: CompactType, env: Dict[str, Type]) -> TypeScheme:
        """Generalize a type into a type scheme by quantifying free variables.
        Tracks free variables by CompactType IDs to avoid hashing CompactType.
        """
        free_vars_ids = self._free_vars(ty) - self._free_vars_env(env)
        # Note: We are not constructing actual TypeVar nodes here since generalize
        # isn't used in current tests; we keep the API but pass IDs for now.
        return TypeScheme(list(free_vars_ids), ty)

    def _free_vars(self, ty: CompactType) -> Set[int]:
        """Collect free type variable IDs in a type"""
        ty = ty.find()
        if ty.kind == 'var':
            return {ty.id}
        elif ty.kind == 'function':
            vars_params = set().union(*(self._free_vars(p) for p in (ty.param_types or [])))
            return vars_params | self._free_vars(ty.return_type)
        elif ty.kind in ('constructor', 'recursive'):
            vars_args = set().union(*(self._free_vars(a) for a in (ty.type_args or [])))
            return vars_args
        return set()

    def _free_vars_env(self, env: Dict[str, Type]) -> Set[int]:
        """Collect free type variable IDs in an environment"""
        return set().union(*(self._free_vars(self.to_compact_type(ty)) for ty in env.values()))

    # --- Biunification (algebraic subtyping) -------------------------------
    #
    # These delegate to a lazily-created Biunifier (defined below). They are
    # additive: nothing in solve_constraints() or the pipeline calls them
    # implicitly, so existing behavior is untouched.

    def biunifier(self) -> 'Biunifier':
        """The lazily-created biunification engine for this inferencer."""
        if self._biunifier is None:
            self._biunifier = Biunifier()
        return self._biunifier

    def constrain(self, lhs: CompactType, rhs: CompactType,
                  node_id: Optional[int] = None) -> None:
        """Record the subtype constraint lhs <: rhs (biunification)."""
        self.biunifier().constrain(lhs, rhs, node_id)

    def solve_constraints_biunify(self) -> 'Biunifier':
        """Feed every buffered Constraint through the biunifier and return
        it. POSITIVE polarity means left <: right, NEGATIVE the reverse,
        NEUTRAL constrains both directions (equality). Unlike
        solve_constraints(), this never mutates the CompactTypes: bounds
        accumulate in the Biunifier's side tables only."""
        b = self.biunifier()
        for c in self.constraints:
            if c.polarity == Polarity.NEGATIVE:
                b.constrain(c.right, c.left)
            elif c.polarity == Polarity.POSITIVE:
                b.constrain(c.left, c.right)
            else:
                b.constrain(c.left, c.right)
                b.constrain(c.right, c.left)
        return b

    def principal_type(self, compact: CompactType, positive: bool = True) -> str:
        """Render the principal type of `compact` from the biunifier's
        recorded bounds, e.g. ``'a -> 'a`` or ``Int ∨ String``."""
        return self.biunifier().principal_type(compact, positive)

    def principal_ptype(self, compact: CompactType, positive: bool = True) -> 'PType':
        """Structured (PType) form of principal_type."""
        return self.biunifier().principal_ptype(compact, positive)
# ---------------------------------------------------------------------------
# Biunification (algebraic subtyping) engine
# ---------------------------------------------------------------------------
#
# This section implements the SimpleSub algorithm (Parreaux, "The Simple
# Essence of Algebraic Subtyping", ICFP 2020) over the existing CompactType
# representation:
#
#   * `Biunifier.constrain(lhs, rhs)` decomposes a subtype constraint
#     lhs <: rhs structurally, records variable bounds in SIDE TABLES
#     (never mutating CompactType.bounds, which the pipeline uses as a
#     union-find pointer), and propagates transitively: when a variable
#     gains an upper bound, every recorded lower bound is constrained
#     against it, and symmetrically. A processed-pair cache guarantees
#     termination on cyclic/recursive constraint graphs.
#   * Let-polymorphism uses levels: fresh variables carry the level current
#     at their creation, `generalize` wraps a type into a scheme at the
#     level below, and `instantiate` copies variables above the scheme
#     level. Constraints that cross levels extrude the offending type: the
#     structure is copied at the lower level with fresh variables that are
#     linked back to the originals as bounds.
#   * `coalesce` turns a constrained variable graph into a principal type:
#     positive occurrences union with their lower bounds, negative
#     occurrences intersect with their upper bounds, and in-process
#     re-entry produces a mu-binder (recursive type).
#   * `simplify` cleans the coalesced form per the paper: flattening and
#     deduplicating unions/intersections, removing variables that occur in
#     only one polarity, and merging variables that always co-occur.
#
# Known, deliberate approximations (documented, not silent):
#   * Invariant constructor parameters are decomposed into constraints in
#     BOTH directions (sound), but extrusion and coalescence treat them at
#     the ambient polarity. The pipeline's constraint stream never crosses
#     levels (it does not generalize), so extrusion of invariant arguments
#     is exercised only by engine-level code.
#   * `recursive`-kind CompactTypes with unresolved bodies constrain only
#     against the same recursive reference (invariantly in their
#     arguments).


@dataclass(frozen=True)
class PType:
    """Base class for coalesced (output) types. Immutable and hashable."""

    def render(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class PVar(PType):
    name: str

    def render(self) -> str:
        return f"'{self.name}"


@dataclass(frozen=True)
class PPrim(PType):
    name: str

    def render(self) -> str:
        return self.name


@dataclass(frozen=True)
class PTop(PType):
    def render(self) -> str:
        return "⊤"


@dataclass(frozen=True)
class PBot(PType):
    def render(self) -> str:
        return "⊥"


def _atom_render(t: PType) -> str:
    """Render, parenthesizing anything non-atomic."""
    s = t.render()
    if isinstance(t, (PFun, PUnion, PInter)):
        return f"({s})"
    return s


@dataclass(frozen=True)
class PFun(PType):
    params: Tuple[PType, ...]
    ret: PType

    def render(self) -> str:
        if len(self.params) == 1:
            left = _atom_render(self.params[0])
        else:
            left = "(" + ", ".join(p.render() for p in self.params) + ")"
        right = self.ret.render() if not isinstance(self.ret, (PUnion, PInter)) \
            else _atom_render(self.ret)
        return f"{left} -> {right}"


@dataclass(frozen=True)
class PCtor(PType):
    name: str
    args: Tuple[PType, ...]

    def render(self) -> str:
        if not self.args:
            return self.name
        return f"{self.name}[" + ", ".join(a.render() for a in self.args) + "]"


@dataclass(frozen=True)
class PUnion(PType):
    parts: Tuple[PType, ...]

    def render(self) -> str:
        return " ∨ ".join(_atom_render(p) for p in self.parts)


@dataclass(frozen=True)
class PInter(PType):
    parts: Tuple[PType, ...]

    def render(self) -> str:
        return " ∧ ".join(_atom_render(p) for p in self.parts)


@dataclass(frozen=True)
class PRec(PType):
    """mu-type binder: `name` is bound inside `body` (rendered as a PVar)."""
    name: str
    body: PType

    def render(self) -> str:
        return f"μ{self.name}. {self.body.render()}"


def _merge_parts(parts: List[PType], positive: bool) -> PType:
    """Build a union (positive) / intersection (negative), flattening one
    level and deduplicating while preserving order."""
    flat: List[PType] = []
    seen: Set[PType] = set()
    for p in parts:
        subparts = p.parts if (positive and isinstance(p, PUnion)) or \
            (not positive and isinstance(p, PInter)) else (p,)
        for sp in subparts:
            if positive and isinstance(sp, PBot):
                continue  # identity element of union
            if not positive and isinstance(sp, PTop):
                continue  # identity element of intersection
            if positive and isinstance(sp, PTop):
                return PTop()  # absorbing element of union
            if not positive and isinstance(sp, PBot):
                return PBot()  # absorbing element of intersection
            if sp not in seen:
                seen.add(sp)
                flat.append(sp)
    if not flat:
        return PBot() if positive else PTop()
    if len(flat) == 1:
        return flat[0]
    return PUnion(tuple(flat)) if positive else PInter(tuple(flat))


class BiunifyError:
    """Structured type error produced by `Biunifier.constrain`.

    Shape-compatible with the adapter's TypeConflict diagnostics
    (kind == "type-conflict", `message`, `node_id`, optional `location`) so
    that a consumer choosing to surface these can reuse the exact
    promotion/formatting machinery. The pipeline does NOT currently promote
    them: hard conflicts keep coming from the adapter's constraint-graph
    detection, and these records are advisory analysis output.
    """
    kind = "type-conflict"

    def __init__(self, message: str, node_id: Optional[int] = None) -> None:
        self.message = message
        self.node_id = node_id
        self.location = None

    def __str__(self) -> str:
        where = f" at node {self.node_id}" if self.node_id is not None else ""
        return f"{self.message}{where}"

    def __repr__(self) -> str:
        return f"BiunifyError({self.message!r}, node_id={self.node_id!r})"


@dataclass
class VarInfo:
    """Side-table state for one type variable (bounds + level)."""
    lower: List[CompactType]
    upper: List[CompactType]
    level: int


@dataclass(frozen=True)
class PolyScheme:
    """A type generalized at `level`: variables with a level strictly above
    it are quantified and copied fresh by `Biunifier.instantiate`."""
    level: int
    body: CompactType


def _short(ty: CompactType) -> str:
    """A short human-readable description of a CompactType's head."""
    ty = ty.find()
    if ty.kind == 'primitive':
        return ty.name or 'primitive'
    if ty.kind == 'function':
        return f"a function of {len(ty.param_types or [])} argument(s)"
    if ty.kind == 'constructor':
        name = ty.name or (ty.constructor.qualified_name() if ty.constructor else 'constructor')
        return name
    if ty.kind == 'recursive':
        return ty.recursive_ref or 'recursive type'
    if ty.kind == 'effect':
        return f"effect {ty.name}"
    return f"'{ty.id}"


class Biunifier:
    """Subtype-constraint solver over CompactTypes (SimpleSub biunification).

    Bounds live in side tables keyed by CompactType id; CompactType.bounds
    is never written (the pipeline's unifier uses `bounds.upper_bound` as a
    union-find pointer, and `find()` follows it — writing bounds here would
    corrupt that structure). Multiple upper/lower bounds per variable are
    supported, which TypeBounds cannot represent.
    """

    def __init__(self) -> None:
        self._info: Dict[int, VarInfo] = {}
        self._cache: Set[Tuple[int, int]] = set()
        self.errors: List[BiunifyError] = []
        self.current_level: int = 0

    # -- variables and levels -------------------------------------------------

    def fresh_var(self, level: Optional[int] = None) -> CompactType:
        v = CompactType.fresh_var()
        self._info[v.id] = VarInfo([], [], self.current_level if level is None else level)
        return v

    def info(self, var: CompactType) -> VarInfo:
        vi = self._info.get(var.id)
        if vi is None:
            # Variables created outside the engine (e.g. the pipeline's
            # per-node vars) are adopted at level 0 so they always live in
            # the outermost scope.
            vi = VarInfo([], [], 0)
            self._info[var.id] = vi
        return vi

    def level_of(self, ty: CompactType, _seen: Optional[Set[int]] = None) -> int:
        ty = ty.find()
        if _seen is None:
            _seen = set()
        if ty.id in _seen:
            return 0
        _seen.add(ty.id)
        if ty.kind == 'var':
            return self.info(ty).level
        parts: List[CompactType] = []
        parts.extend(ty.param_types or [])
        if ty.return_type is not None:
            parts.append(ty.return_type)
        parts.extend(ty.type_args or [])
        return max((self.level_of(p, _seen) for p in parts), default=0)

    def enter_level(self) -> None:
        self.current_level += 1

    def exit_level(self) -> None:
        self.current_level -= 1

    # -- constraining ---------------------------------------------------------

    def _error(self, lhs: CompactType, rhs: CompactType, node_id: Optional[int]) -> None:
        a, b = sorted((_short(lhs), _short(rhs)))
        message = f"type mismatch: one value is required to be {a} and {b}"
        if any(e.message == message and e.node_id == node_id for e in self.errors):
            return  # constraining both directions reports the clash once
        self.errors.append(BiunifyError(message, node_id))

    def constrain(self, lhs: CompactType, rhs: CompactType,
                  node_id: Optional[int] = None) -> None:
        """Record lhs <: rhs, decomposing structurally and propagating
        through variable bounds. Terminates on cyclic graphs via the
        processed-pair cache."""
        lhs = lhs.find()
        rhs = rhs.find()
        if lhs.id == rhs.id:
            return
        key = (lhs.id, rhs.id)
        if key in self._cache:
            return
        self._cache.add(key)

        lk, rk = lhs.kind, rhs.kind
        if lk == 'function' and rk == 'function':
            lp, rp = lhs.param_types or [], rhs.param_types or []
            if len(lp) != len(rp):
                self._error(lhs, rhs, node_id)
                return
            for a, b in zip(rp, lp):          # parameters: contravariant
                self.constrain(a, b, node_id)
            if lhs.return_type is not None and rhs.return_type is not None:
                self.constrain(lhs.return_type, rhs.return_type, node_id)
            return
        if lk == 'constructor' and rk == 'constructor':
            lname = lhs.constructor.qualified_name() if lhs.constructor else lhs.name
            rname = rhs.constructor.qualified_name() if rhs.constructor else rhs.name
            largs, rargs = lhs.type_args or [], rhs.type_args or []
            if lname != rname or len(largs) != len(rargs):
                self._error(lhs, rhs, node_id)
                return
            variances = get_constructor_variances(lhs.constructor) if lhs.constructor \
                else ['invariant'] * len(largs)
            for a, b, v in zip(largs, rargs, variances):
                if v == 'covariant':
                    self.constrain(a, b, node_id)
                elif v == 'contravariant':
                    self.constrain(b, a, node_id)
                else:  # invariant: both directions
                    self.constrain(a, b, node_id)
                    self.constrain(b, a, node_id)
            return
        if lk == 'primitive' and rk == 'primitive':
            # The primitive lattice is deliberately FLAT (no Int <: Float):
            # widening it would change which programs compile.
            if lhs.name != rhs.name:
                self._error(lhs, rhs, node_id)
            return
        if lk == 'effect' and rk == 'effect':
            if lhs.name != rhs.name:
                self._error(lhs, rhs, node_id)
            return
        if lk == 'var':
            vi = self.info(lhs)
            bound = rhs
            if self.level_of(rhs) > vi.level:
                bound = self.extrude(rhs, False, vi.level, {})
            vi.upper.append(bound)
            for lb in list(vi.lower):
                self.constrain(lb, bound, node_id)
            return
        if rk == 'var':
            vi = self.info(rhs)
            bound = lhs
            if self.level_of(lhs) > vi.level:
                bound = self.extrude(lhs, True, vi.level, {})
            vi.lower.append(bound)
            for ub in list(vi.upper):
                self.constrain(bound, ub, node_id)
            return
        if lk == 'recursive' or rk == 'recursive':
            if lk == 'recursive' and rk == 'recursive' \
                    and lhs.recursive_ref == rhs.recursive_ref \
                    and len(lhs.type_args or []) == len(rhs.type_args or []):
                for a, b in zip(lhs.type_args or [], rhs.type_args or []):
                    self.constrain(a, b, node_id)
                    self.constrain(b, a, node_id)
                return
            l2 = unfold_once(lhs) if lk == 'recursive' else lhs
            r2 = unfold_once(rhs) if rk == 'recursive' else rhs
            if l2 is not lhs or r2 is not rhs:
                self.constrain(l2, r2, node_id)
                return
            self._error(lhs, rhs, node_id)
            return
        self._error(lhs, rhs, node_id)

    # -- extrusion ------------------------------------------------------------

    def extrude(self, ty: CompactType, pol: bool, lvl: int,
                cache: Dict[Tuple[int, bool], CompactType]) -> CompactType:
        """Copy `ty` down to level `lvl` (SimpleSub's extrusion).

        Variables above the level are replaced by fresh variables AT the
        level, registered as bounds of the originals so information keeps
        flowing; structure is copied with parameter positions flipping
        polarity. Invariant constructor arguments are extruded at the
        ambient polarity (see the section comment above)."""
        ty = ty.find()
        if self.level_of(ty) <= lvl:
            return ty
        if ty.kind == 'var':
            key = (ty.id, pol)
            hit = cache.get(key)
            if hit is not None:
                return hit
            nv = self.fresh_var(level=lvl)
            cache[key] = nv
            vi = self.info(ty)
            nvi = self._info[nv.id]
            if pol:
                vi.upper.append(nv)
                nvi.lower = [self.extrude(lb, pol, lvl, cache) for lb in vi.lower]
            else:
                vi.lower.append(nv)
                nvi.upper = [self.extrude(ub, pol, lvl, cache) for ub in vi.upper]
            return nv
        if ty.kind == 'function':
            return CompactType(
                id=next_id(), kind='function',
                param_types=[self.extrude(p, not pol, lvl, cache)
                             for p in (ty.param_types or [])],
                return_type=(self.extrude(ty.return_type, pol, lvl, cache)
                             if ty.return_type is not None else None),
                linearity=ty.linearity)
        if ty.kind == 'constructor':
            variances = get_constructor_variances(ty.constructor) if ty.constructor \
                else ['invariant'] * len(ty.type_args or [])
            new_args = []
            for a, v in zip(ty.type_args or [], variances):
                p2 = (not pol) if v == 'contravariant' else pol
                new_args.append(self.extrude(a, p2, lvl, cache))
            return CompactType(id=next_id(), kind='constructor',
                               constructor=ty.constructor, type_args=new_args,
                               name=ty.name)
        return ty  # primitive / effect / recursive: level-closed

    # -- let-polymorphism -----------------------------------------------------

    def generalize(self, ty: CompactType) -> PolyScheme:
        """Generalize over every variable ABOVE the current level. Call
        after typing a let-bound definition one level up (enter_level /
        exit_level around the definition)."""
        return PolyScheme(self.current_level, ty)

    def instantiate(self, scheme: PolyScheme) -> CompactType:
        """Copy variables above the scheme level as fresh variables at the
        current level, duplicating their recorded bounds."""
        cache: Dict[int, CompactType] = {}

        def freshen(ty: CompactType) -> CompactType:
            ty = ty.find()
            if self.level_of(ty) <= scheme.level:
                return ty
            if ty.kind == 'var':
                hit = cache.get(ty.id)
                if hit is not None:
                    return hit
                nv = self.fresh_var()
                cache[ty.id] = nv
                vi = self.info(ty)
                nvi = self._info[nv.id]
                nvi.lower = [freshen(lb) for lb in vi.lower]
                nvi.upper = [freshen(ub) for ub in vi.upper]
                return nv
            if ty.kind == 'function':
                return CompactType(
                    id=next_id(), kind='function',
                    param_types=[freshen(p) for p in (ty.param_types or [])],
                    return_type=(freshen(ty.return_type)
                                 if ty.return_type is not None else None),
                    linearity=ty.linearity)
            if ty.kind == 'constructor':
                return CompactType(id=next_id(), kind='constructor',
                                   constructor=ty.constructor,
                                   type_args=[freshen(a) for a in (ty.type_args or [])],
                                   name=ty.name)
            return ty

        return freshen(scheme.body)

    # -- coalescence: bounds -> principal types -------------------------------

    def coalesce(self, ty: CompactType, positive: bool = True) -> PType:
        """Turn the constrained graph reachable from `ty` into a coalesced
        type: positive variables union with their lower bounds, negative
        variables intersect with their upper bounds; re-entering a variable
        at the same polarity produces a mu-binder."""
        rec_names: Dict[Tuple[int, bool], str] = {}
        rec_counter = [0]

        def go(t: CompactType, pol: bool, in_process: frozenset) -> PType:
            t = t.find()
            if t.kind == 'var':
                key = (t.id, pol)
                if key in in_process:
                    name = rec_names.get(key)
                    if name is None:
                        rec_counter[0] += 1
                        name = f"rec{rec_counter[0]}"
                        rec_names[key] = name
                    return PMuVar(name)
                vi = self._info.get(t.id)
                bounds = (vi.lower if pol else vi.upper) if vi is not None else []
                parts: List[PType] = [PVar(f"v{t.id}")]
                nested = in_process | {key}
                for b in bounds:
                    parts.append(go(b, pol, nested))
                res = _merge_parts(parts, pol)
                if key in rec_names:
                    res = PRec(rec_names[key], res)
                return res
            if t.kind == 'function':
                return PFun(
                    tuple(go(p, not pol, in_process) for p in (t.param_types or [])),
                    go(t.return_type, pol, in_process)
                    if t.return_type is not None else PPrim("Unit"))
            if t.kind == 'constructor':
                variances = get_constructor_variances(t.constructor) if t.constructor \
                    else ['invariant'] * len(t.type_args or [])
                name = t.name or (t.constructor.qualified_name() if t.constructor else 'Ctor')
                args = []
                for a, v in zip(t.type_args or [], variances):
                    p2 = (not pol) if v == 'contravariant' else pol
                    args.append(go(a, p2, in_process))
                return PCtor(name, tuple(args))
            if t.kind == 'primitive':
                return PPrim(t.name or 'Unknown')
            if t.kind == 'effect':
                return PPrim(t.name or 'Effect')
            if t.kind == 'recursive':
                return PCtor(t.recursive_ref or 'Rec',
                             tuple(go(a, pol, in_process) for a in (t.type_args or [])))
            return PPrim('Unknown')

        return go(ty, positive, frozenset())

    # -- principal types ------------------------------------------------------

    def principal_ptype(self, ty: CompactType, positive: bool = True) -> PType:
        return simplify_ptype(self.coalesce(ty, positive), positive)

    def principal_type(self, ty: CompactType, positive: bool = True) -> str:
        """Coalesce, simplify and render the principal type of `ty`,
        e.g. `'a -> 'a`, `Int ∨ String`, `μt. Cons[Int, 't] ∨ Nil`."""
        return self.principal_ptype(ty, positive).render()
# -- simplification of coalesced types ---------------------------------------

@dataclass(frozen=True)
class PMuVar(PType):
    """Occurrence of a mu-bound recursion variable (rendered bare: `t`)."""
    name: str

    def render(self) -> str:
        return self.name


def _mu_occurs(pt: PType, name: str) -> bool:
    if isinstance(pt, PMuVar):
        return pt.name == name
    if isinstance(pt, (PUnion, PInter)):
        return any(_mu_occurs(p, name) for p in pt.parts)
    if isinstance(pt, PFun):
        return any(_mu_occurs(p, name) for p in pt.params) or _mu_occurs(pt.ret, name)
    if isinstance(pt, PCtor):
        return any(_mu_occurs(a, name) for a in pt.args)
    if isinstance(pt, PRec):
        return pt.name != name and _mu_occurs(pt.body, name)
    return False


def _normalize(pt: PType) -> PType:
    """Flatten nested unions/intersections, drop identity elements,
    deduplicate, collapse singletons, and remove unused mu-binders."""
    if isinstance(pt, PUnion):
        return _merge_parts([_normalize(p) for p in pt.parts], True)
    if isinstance(pt, PInter):
        return _merge_parts([_normalize(p) for p in pt.parts], False)
    if isinstance(pt, PFun):
        return PFun(tuple(_normalize(p) for p in pt.params), _normalize(pt.ret))
    if isinstance(pt, PCtor):
        return PCtor(pt.name, tuple(_normalize(a) for a in pt.args))
    if isinstance(pt, PRec):
        body = _normalize(pt.body)
        return PRec(pt.name, body) if _mu_occurs(body, pt.name) else body
    return pt


def _occurrences(pt: PType, positive: bool = True):
    """Collect, for every inference variable (PVar): the polarities it
    occurs at, the co-occurrence contexts (sibling atoms in its immediately
    enclosing union/intersection), and whether it ever occurs standalone
    (not directly under a union/intersection). `positive` is the polarity
    of the root (False when simplifying a negative-position query)."""
    pols: Dict[str, Set[bool]] = {}
    contexts: Dict[str, List[Tuple[bool, frozenset]]] = {}
    standalone: Set[str] = set()

    def record(name: str, pol: bool) -> None:
        pols.setdefault(name, set()).add(pol)

    def go(t: PType, pol: bool) -> None:
        if isinstance(t, PVar):
            record(t.name, pol)
            standalone.add(t.name)
            return
        if isinstance(t, (PUnion, PInter)):
            parts = t.parts
            for i, p in enumerate(parts):
                if isinstance(p, PVar):
                    record(p.name, pol)
                    sibs = frozenset(q for j, q in enumerate(parts) if j != i)
                    contexts.setdefault(p.name, []).append((pol, sibs))
                else:
                    go(p, pol)
            return
        if isinstance(t, PFun):
            for p in t.params:
                go(p, not pol)
            go(t.ret, pol)
            return
        if isinstance(t, PCtor):
            for a in t.args:
                go(a, pol)
            return
        if isinstance(t, PRec):
            go(t.body, pol)
            return

    go(pt, positive)
    return pols, contexts, standalone


def _subst_var(pt: PType, name: str, pos_repl: PType, neg_repl: PType,
               pol: bool = True) -> PType:
    """Polarity-aware substitution of the inference variable `name`."""
    if isinstance(pt, PVar):
        if pt.name == name:
            return pos_repl if pol else neg_repl
        return pt
    if isinstance(pt, PUnion):
        return PUnion(tuple(_subst_var(p, name, pos_repl, neg_repl, pol)
                            for p in pt.parts))
    if isinstance(pt, PInter):
        return PInter(tuple(_subst_var(p, name, pos_repl, neg_repl, pol)
                            for p in pt.parts))
    if isinstance(pt, PFun):
        return PFun(tuple(_subst_var(p, name, pos_repl, neg_repl, not pol)
                          for p in pt.params),
                    _subst_var(pt.ret, name, pos_repl, neg_repl, pol))
    if isinstance(pt, PCtor):
        return PCtor(pt.name, tuple(_subst_var(a, name, pos_repl, neg_repl, pol)
                                    for a in pt.args))
    if isinstance(pt, PRec):
        return PRec(pt.name, _subst_var(pt.body, name, pos_repl, neg_repl, pol))
    return pt


def simplify_ptype(pt: PType, positive: bool = True) -> PType:
    """Simplify a coalesced type per the SimpleSub paper:

    1. flatten/dedupe unions and intersections (in `_normalize`);
    2. polar variable removal: a variable occurring at only one polarity,
       always inside a union/intersection with other members, adds no
       information and is dropped;
    3. co-occurrence analysis: a variable that always occurs together with
       the same other variable, at both polarities and symmetrically, is
       unified with it; a variable that always co-occurs with the same
       concrete atom at both polarities is redundant and removed.

    Finally, variables are renamed to 'a, 'b, ... and mu-binders to t, u, ...
    """
    pt = _normalize(pt)
    for _ in range(32):
        pols, contexts, standalone = _occurrences(pt, positive)
        changed = False

        # (2) polar variable removal
        for name, ps in sorted(pols.items()):
            if len(ps) == 1 and name not in standalone and contexts.get(name):
                pol = next(iter(ps))
                repl = PBot() if pol else PTop()
                pt = _normalize(_subst_var(pt, name, repl, repl))
                changed = True
                break
        if changed:
            continue

        # (3) co-occurrence analysis
        for name, ctxs in sorted(contexts.items()):
            if name in standalone or pols.get(name) != {True, False} or not ctxs:
                continue
            common = set(ctxs[0][1])
            for _, sibs in ctxs[1:]:
                common &= set(sibs)
            var_partners = sorted((a.name for a in common if isinstance(a, PVar)))
            merged = False
            for w in var_partners:
                if w == name:
                    continue
                # symmetric requirement: `name` must appear in every context
                # of `w` as well (and w must never be standalone)
                w_ctxs = contexts.get(w, [])
                if w in standalone or not w_ctxs:
                    continue
                if all(any(isinstance(a, PVar) and a.name == name for a in sibs)
                       for _, sibs in w_ctxs):
                    pt = _normalize(_subst_var(pt, name, PVar(w), PVar(w)))
                    changed = merged = True
                    break
            if merged:
                break
            if any(not isinstance(a, (PVar, PMuVar)) for a in common):
                # always accompanied by the same concrete atom on both sides
                pt = _normalize(_subst_var(pt, name, PBot(), PTop(), positive))
                changed = True
                break
        if not changed:
            break
    return _rename_ptype_vars(pt)


def _rename_ptype_vars(pt: PType) -> PType:
    """Rename inference variables to 'a, 'b, ... (first-occurrence order)
    and mu-binders to t, u, ..."""
    var_names: Dict[str, str] = {}
    mu_names: Dict[str, str] = {}
    var_alphabet = "abcdefghijklmnopqrs"
    mu_alphabet = "tuvwxyz"

    def var_name(old: str) -> str:
        if old not in var_names:
            i = len(var_names)
            var_names[old] = var_alphabet[i] if i < len(var_alphabet) else f"a{i}"
        return var_names[old]

    def mu_name(old: str) -> str:
        if old not in mu_names:
            i = len(mu_names)
            mu_names[old] = mu_alphabet[i] if i < len(mu_alphabet) else f"t{i}"
        return mu_names[old]

    def go(t: PType) -> PType:
        if isinstance(t, PVar):
            return PVar(var_name(t.name))
        if isinstance(t, PMuVar):
            return PMuVar(mu_name(t.name))
        if isinstance(t, PUnion):
            return PUnion(tuple(go(p) for p in t.parts))
        if isinstance(t, PInter):
            return PInter(tuple(go(p) for p in t.parts))
        if isinstance(t, PFun):
            return PFun(tuple(go(p) for p in t.params), go(t.ret))
        if isinstance(t, PCtor):
            return PCtor(t.name, tuple(go(a) for a in t.args))
        if isinstance(t, PRec):
            new = mu_name(t.name)
            return PRec(new, go(t.body))
        return t

    return go(pt)
