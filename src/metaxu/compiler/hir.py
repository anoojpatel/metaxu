from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .types import Ty, EffectSet
from .infer_tables import InferSideTables
from .constraints import ClassConstraint
from . import mutaxu_ast as mast
import metaxu.metaxu_ast as fast

from .desugar import IMPL_SEP, parse_impl_method_name, type_base_name

# Methods that are dispatched as interpreter builtins with the receiver as
# first argument (`x.to_string()` -> to_string(x)).
# Note: when a method of the same name is provided by a user trait/impl block,
# the call lowers to a __trait$ dispatch instead (see the QualifiedFunctionCall
# and MethodCall paths below), so user impls win over these builtins; the
# interpreter's trait dispatch falls back to the builtin for receiver types
# without an impl.
_BUILTIN_METHODS = frozenset({
    "to_string", "len",
    # Vec methods (runtime library)
    "push", "pop",
    # math methods on numbers (runtime library)
    "sqrt", "sin", "cos",
})

# Callee-name prefix marking a runtime-dispatched trait method call:
# `recv.m(args)` lowers to Call(callee="__trait$m", operands=(recv, *args))
# and the MIR interpreter picks the impl function matching the receiver's
# runtime type name (falling back to builtins, then plain functions).
TRAIT_CALL_PREFIX = f"__trait{IMPL_SEP}"

# Callee-name prefix for a static impl-method call `Type.m(args)` (no
# receiver): "__static$Type$m". The interpreter resolves it against the
# loaded __impl$Trait$Type$m functions for that type name.
STATIC_CALL_PREFIX = f"__static{IMPL_SEP}"


@dataclass(slots=True)
class ModeInfo:
    uniqueness: str | None = None   # 'unique'|'exclusive'|'shared'|'owned'|'mutable'|'const'
    locality: str | None = None     # 'local'|'global'
    linearity: str | None = None    # 'once'|'separate'|'many'

@dataclass(slots=True, frozen=True)
class HPattern:
    """A match pattern in HIR.

    kind:
      'wildcard'  matches anything, binds nothing
      'var'       matches anything, binds `name`
      'literal'   matches when scrutinee == value
      'ctor'      matches enum variant `name` (of enum `enum_name` when known),
                  recursively matching `subpatterns` against the payload fields
    """
    kind: str
    name: str | None = None          # binding name (var) or variant name (ctor)
    value: Any | None = None         # literal value (literal)
    enum_name: str | None = None     # enum type name (ctor), when known
    subpatterns: tuple['HPattern', ...] = ()


@dataclass(slots=True)
class HExpr:
    node_id: int
    kind: str
    args: tuple[Any, ...]
    ty: Ty
    effects: EffectSet
    suspends: bool
    sym: Any | None
    span: mast.Span
    # Optional op-specific fields for lowering
    op: str | None = None            # e.g., 'Literal', 'Var', 'Call', 'Let', 'Block', 'Match'
    literal: Any | None = None       # for Literal
    var_name: str | None = None      # for Var
    callee: str | None = None        # for Call (simple callee name)
    operands: tuple['HExpr', ...] | None = None  # for Call/Block
    bindings: tuple[tuple[str, 'HExpr'], ...] | None = None  # for Let: ((name, expr), ...)
    bind_modes: dict[str, ModeInfo] | None = None  # modes for Let-bound locals
    # BinOp
    binop: str | None = None
    left: 'HExpr' | None = None
    right: 'HExpr' | None = None
    # If
    cond: 'HExpr' | None = None
    then_ops: tuple['HExpr', ...] | None = None
    else_ops: tuple['HExpr', ...] | None = None
    # Match
    scrutinee: 'HExpr' | None = None
    cases: tuple['HExpr', ...] | None = None  # legacy: arm bodies only (no patterns)
    match_arms: tuple[tuple[HPattern, 'HExpr'], ...] | None = None  # (pattern, body) pairs
    # While loop: op="While" (cond field reused for the loop condition)
    loop_body: tuple['HExpr', ...] | None = None
    # Assignment: op="Assign" (var_name reused for the target)
    assign_value: 'HExpr | None' = None
    # Enum variant construction: op="MakeVariant" (operands reused for payload exprs)
    enum_name: str | None = None
    variant_name: str | None = None
    # Struct: op="Struct"
    struct_name: str | None = None                          # for Struct
    fields: tuple[tuple[str, 'HExpr'], ...] | None = None  # for Struct: ((field_name, expr), ...)
    locality: str | None = None                             # 'local'|'global' allocation site
    # FieldGet/FieldSet: op="FieldGet" | "FieldSet"
    base: 'HExpr | None' = None      # receiver object
    field_name: str | None = None    # field being accessed/set
    field_val: 'HExpr | None' = None # for FieldSet: new value
    # Lambda/Closure: op="Lambda"
    lambda_params: tuple[str, ...] | None = None           # parameter names
    lambda_body: 'HExpr | None' = None                     # body expression
    captures: tuple[tuple[str, str], ...] | None = None    # ((name, mode), ...) captured vars
    # Perform: op="Perform"
    effect_op: str | None = None        # effect operation name, e.g. 'emit'
    perform_args: tuple['HExpr', ...] | None = None  # arguments to the operation
    # Handle: op="Handle"
    handle_effect: str | None = None    # effect type name being handled
    handle_cases: tuple[tuple[str, str, 'HExpr'], ...] | None = None  # ((op, param, body), ...)
    handle_body: 'HExpr | None' = None  # the continuation expression


@dataclass(slots=True)
class HFun:
    sym: Any
    params: list[tuple[Any, Ty]]
    dict_params: list[tuple[str, Any]]  # (TraitName, DictTy placeholder)
    ret_ty: Ty
    where_cls: list[ClassConstraint]
    body: HExpr
    param_modes: dict[str, ModeInfo] | None = None


class HIRBuilder:
    """Build a typed, frozen HIR from the parsed AST and inference side-tables."""

    def __init__(self, tables: InferSideTables, id_map: dict[int, Any] | None = None) -> None:
        self.t = tables
        self.id_map = id_map or {}
        # Reverse map: id(orig_obj) -> frozen AstNode, built lazily in build()
        self._orig_to_frozen: dict[int, mast.AstNode] = {}
        # Effect op names collected from EffectDeclaration nodes
        self._effect_op_names: set[str] = set()
        # Enum variant constructors: variant_name -> enum_name
        self._variant_to_enum: dict[str, str] = {}
        # Trait method names: declared in traits (InterfaceDefinition) or
        # provided by an implement block (mangled __impl$... function names).
        self._trait_method_names: set[str] = set()
        # Type names with impl methods (targets of implement blocks), used to
        # recognize static calls `Type.method(args)`.
        self._impl_type_names: set[str] = set()

    def build(self, root: mast.AstNode) -> list[HFun]:
        funcs: list[HFun] = []

        # Build reverse map: id(orig_obj) -> frozen AstNode
        def index_nodes(n: mast.AstNode) -> None:
            orig = self.id_map.get(n.node_id)
            if orig is not None:
                self._orig_to_frozen[id(orig)] = n
            for c in n.children:
                index_nodes(c)
        index_nodes(root)

        # Collect effect operation names so bare FunctionCall(name=op) can be identified as Perform
        self._effect_op_names: set[str] = set()
        self._trait_method_names = set()
        for orig in self.id_map.values():
            if isinstance(orig, fast.EffectDeclaration):
                for op in (getattr(orig, 'operations', []) or []):
                    if hasattr(op, 'name'):
                        self._effect_op_names.add(str(op.name))
            if isinstance(orig, fast.EnumDefinition):
                ename = str(getattr(orig, 'name', '') or '')
                for v in (getattr(orig, 'variants', []) or []):
                    vname = getattr(v, 'name', None)
                    if vname is not None:
                        self._variant_to_enum[str(vname)] = ename
            # Trait method names: from trait declarations...
            if isinstance(orig, fast.InterfaceDefinition):
                for m in (getattr(orig, 'methods', []) or []):
                    mname = getattr(m, 'name', None)
                    if mname is not None:
                        self._trait_method_names.add(str(mname))
            # ...and from desugared implement-block functions (__impl$T$Ty$m),
            # so impl-only methods dispatch even without a trait declaration.
            if isinstance(orig, fast.FunctionDeclaration):
                parsed = parse_impl_method_name(str(getattr(orig, 'name', '') or ''))
                if parsed is not None:
                    self._trait_method_names.add(parsed[2])
                    self._impl_type_names.add(parsed[1])

        def visit(n: mast.AstNode) -> None:
            orig = self.id_map.get(n.node_id)
            if isinstance(orig, fast.FunctionDeclaration):
                # Determine return type from side tables for this node or fallback
                ret = self.t.apply_tyenv(self.t.types.get(n.node_id, "Unit"))  # type: ignore[index]
                body_hexpr = self._from_orig_expr(orig.body if hasattr(orig, 'body') else [], n)
                if body_hexpr is None:
                    body_hexpr = HExpr(
                        node_id=n.node_id,
                        kind=n.kind,
                        args=tuple(),
                        ty=ret,
                        effects=EffectSet(frozenset()),
                        suspends=self.t.suspends_node(n.node_id) if hasattr(self.t, "suspends_node") else False,
                        sym=getattr(orig, 'name', None),
                        span=n.span,
                        op="Block",
                        operands=tuple(),
                    )
                params: list[tuple[Any, Ty]] = []
                param_modes: dict[str, ModeInfo] = {}
                for p in getattr(orig, 'params', []) or []:
                    pname = getattr(p, 'name', None)
                    pty = getattr(p, 'type_annotation', None)
                    if pty is None:
                        pty = self.t.types.get(getattr(p, 'node_id', -1), "Unknown") if hasattr(p, 'node_id') else "Unknown"  # type: ignore[index]
                    params.append((pname, pty))
                    # Extract modes if available
                    pmode = self._extract_modeinfo(getattr(p, 'mode', None))
                    if pname is not None:
                        param_modes[str(pname)] = pmode
                hfun = HFun(
                    sym=getattr(orig, 'name', 'fun'),
                    params=params,
                    dict_params=[],
                    ret_ty=ret,
                    where_cls=[],
                    body=body_hexpr,
                    param_modes=param_modes or None,
                )
                funcs.append(hfun)
            # Recurse
            for c in n.children:
                visit(c)

        visit(root)
        # Fallback: if no functions found, produce a default wrapper
        if not funcs:
            ty = self.t.apply_tyenv(self.t.types.get(root.node_id, "Unit"))  # type: ignore[union-attr]
            hexpr = HExpr(
                node_id=root.node_id,
                kind=root.kind,
                args=tuple(),
                ty=ty,
                effects=EffectSet(frozenset()),
                suspends=self.t.suspends_node(root.node_id) if hasattr(self.t, "suspends_node") else False,
                sym=self.t.sym_of(root.node_id) if hasattr(self.t, "sym_of") else None,
                span=root.span,
                op="Block",
                operands=tuple(),
            )
            funcs.append(HFun(sym="main", params=[], dict_params=[], ret_ty=ty, where_cls=[], body=hexpr))
        return funcs

    def _mk_hexpr(self, node_id: int, kind: str, ty: Ty, span: mast.Span, **kw: Any) -> HExpr:
        return HExpr(
            node_id=node_id,
            kind=kind,
            args=tuple(),
            ty=ty,
            effects=EffectSet(frozenset()),
            suspends=self.t.suspends_node(node_id) if hasattr(self.t, "suspends_node") else False,
            sym=None,
            span=span,
            **kw,
        )

    def _frozen_for(self, orig: Any, fallback: mast.AstNode) -> mast.AstNode:
        """Return the frozen AstNode corresponding to orig, or fallback."""
        return self._orig_to_frozen.get(id(orig), fallback)

    def _from_orig_expr(self, orig: Any, frozen_ctx: mast.AstNode) -> HExpr | None:
        """Build an HExpr from an original AST node or list of nodes.

        frozen_ctx: a frozen node to provide node_id/span context when the original
        node lacks a corresponding frozen node in id_map traversal.
        """
        if orig is None:
            return None

        # Resolve the best frozen node for this orig object
        if not isinstance(orig, list):
            frozen_ctx = self._frozen_for(orig, frozen_ctx)

        def ctx_for(child: Any) -> mast.AstNode:
            return self._frozen_for(child, frozen_ctx) if child is not None else frozen_ctx

        # Lists become Block
        if isinstance(orig, list):
            ops: list[HExpr] = []
            for item in orig:
                he = self._from_orig_expr(item, self._frozen_for(item, frozen_ctx) if item is not None else frozen_ctx)
                if he is not None:
                    ops.append(he)
            return self._mk_hexpr(frozen_ctx.node_id, "Block", self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unit")), frozen_ctx.span, op="Block", operands=tuple(ops))

        # Literals
        if isinstance(orig, fast.Literal):
            ty = self.t.apply_tyenv(getattr(orig, 'type_var', None) or self.t.types.get(frozen_ctx.node_id, "Unknown"))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span, op="Literal", literal=getattr(orig, 'value', None))

        # Variables
        if isinstance(orig, fast.Variable):
            ty = self.t.apply_tyenv(getattr(orig, 'type_var', None) or self.t.types.get(frozen_ctx.node_id, "Unknown"))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span, op="Var", var_name=getattr(orig, 'name', None))

        # QualifiedName: `a` → Var; `a.b.c` → chained FieldGet
        if isinstance(orig, fast.QualifiedName):
            parts = list(getattr(orig, 'parts', []) or [])
            if not parts:
                return None
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unknown"))
            if len(parts) == 1:
                return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span, op="Var", var_name=str(parts[0]))
            # Build chained FieldGet: start from the first part as a Var
            current: HExpr = self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span, op="Var", var_name=str(parts[0]))
            for field in parts[1:]:
                current = self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span, op="FieldGet", base=current, field_name=str(field))
            return current

        # Borrow/move/exclave expressions: at runtime these evaluate to the
        # referenced value (aliasing and ownership rules are enforced earlier
        # by the frozen borrow checker, not at HIR/MIR level).
        if isinstance(orig, (fast.BorrowShared, fast.BorrowUnique, fast.Move)):
            var = getattr(orig, 'variable', None)
            if isinstance(var, str):
                ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
                return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                      op="Var", var_name=var)
            return self._from_orig_expr(var, ctx_for(var)) if var is not None else None
        if isinstance(orig, fast.ExclaveExpression):
            inner = getattr(orig, 'expression', None)
            if isinstance(inner, str):
                ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
                return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                      op="Var", var_name=inner)
            return self._from_orig_expr(inner, ctx_for(inner)) if inner is not None else None

        # Option constructors in expression position: Some(x) / None.
        # (In pattern position these are handled by _convert_pattern.)
        if isinstance(orig, fast.SomeExpression):
            inner = getattr(orig, 'value', None)
            inner_he = self._from_orig_expr(inner, ctx_for(inner)) if inner is not None else None
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                  op="MakeVariant",
                                  enum_name=self._variant_to_enum.get("Some", "Option"),
                                  variant_name="Some",
                                  operands=(inner_he,) if inner_he is not None else ())
        if isinstance(orig, fast.NoneExpression):
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                  op="MakeVariant",
                                  enum_name=self._variant_to_enum.get("None", "Option"),
                                  variant_name="None", operands=())

        # PrintStatement: `print(args)` — lower to a builtin call
        if isinstance(orig, fast.PrintStatement):
            arg_exprs = []
            for a in getattr(orig, 'arguments', []) or []:
                he = self._from_orig_expr(a, ctx_for(a))
                if he is not None:
                    arg_exprs.append(he)
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unit'))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                  op="Call", callee="print", operands=tuple(arg_exprs))

        # Dedicated Resume node: `resume(v)` inside a handle case
        if isinstance(orig, fast.Resume):
            val_node = getattr(orig, 'value', None)
            val_exprs: tuple = ()
            if val_node is not None and hasattr(val_node, '__class__') and isinstance(val_node, fast.Node):
                he = self._from_orig_expr(val_node, ctx_for(val_node))
                if he is not None:
                    val_exprs = (he,)
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                  op="Resume", perform_args=val_exprs)

        # Function calls (including perform-as-bare-call and resume)
        if isinstance(orig, fast.FunctionCall):
            callee = str(getattr(orig, 'name', None) or '')
            args_exprs = []
            for a in getattr(orig, 'arguments', []) or []:
                he = self._from_orig_expr(a, ctx_for(a))
                if he is not None:
                    args_exprs.append(he)
            ty = self.t.apply_tyenv(getattr(orig, 'type_var', None) or self.t.types.get(frozen_ctx.node_id, "Unknown"))
            # `resume(v)` in a handle case → special Resume op (returns value to handler caller)
            if callee == 'resume':
                return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                      op="Resume", perform_args=tuple(args_exprs))
            # Bare `perform emit(x)` parsed as FunctionCall when callee is a known effect op
            if callee in self._effect_op_names:
                return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                      op="Perform", effect_op=callee, perform_args=tuple(args_exprs))
            # Enum variant constructor call, e.g. `Some(5)` / `Cons(h, t)`
            if callee in self._variant_to_enum:
                return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                      op="MakeVariant", enum_name=self._variant_to_enum[callee],
                                      variant_name=callee, operands=tuple(args_exprs))
            # Builtin Option constructors when no enum declares them: the docs
            # treat Option as a language-provided type.
            if callee in ("Some", "None"):
                return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                      op="MakeVariant", enum_name="Option",
                                      variant_name=callee, operands=tuple(args_exprs))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span, op="Call", callee=callee, operands=tuple(args_exprs))

        # BinaryOperation / ComparisonExpression (same structure, both use left/operator/right)
        if isinstance(orig, (fast.BinaryOperation, fast.ComparisonExpression)):
            lnode = getattr(orig, 'left', None)
            rnode = getattr(orig, 'right', None)
            l = self._from_orig_expr(lnode, ctx_for(lnode))
            r = self._from_orig_expr(rnode, ctx_for(rnode))
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unknown"))
            op_sym = getattr(orig, 'operator', None)
            # ComparisonOperator may be an enum instance; get its value string
            if hasattr(op_sym, 'value'):
                op_sym = op_sym.value
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span, op="BinOp", binop=op_sym, left=l, right=r)

        # IfStatement / IfExpression
        if isinstance(orig, (fast.IfStatement, fast.IfExpression)):
            cond_node = getattr(orig, 'condition', None)
            c = self._from_orig_expr(cond_node, ctx_for(cond_node))
            # IfStatement uses then_body/else_body; IfExpression uses then_branch/else_branch
            then_node = getattr(orig, 'then_body', None) or getattr(orig, 'then_branch', None)
            else_node = getattr(orig, 'else_body', None) or getattr(orig, 'else_branch', None)
            tb = self._from_orig_expr(then_node, ctx_for(then_node))
            eb = self._from_orig_expr(else_node, ctx_for(else_node)) if else_node is not None else None
            # Flatten block bodies into operand lists
            def as_ops(h: HExpr | None) -> tuple[HExpr, ...]:
                if h is None:
                    return tuple()
                if h.op == "Block" and h.operands is not None:
                    return h.operands
                return (h,)
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unit"))
            return self._mk_hexpr(frozen_ctx.node_id, "Stmt", ty, frozen_ctx.span, op="If", cond=c, then_ops=as_ops(tb), else_ops=as_ops(eb) if eb else tuple())

        # MatchExpression: carry (pattern, body) pairs into HIR.
        # TODO(pattern-typing): pattern variable types are not yet threaded through
        # the constraint emitter; typing of bindings currently falls back to the
        # arm-body node types (frozen_constraint_emitter is owned by another agent).
        if isinstance(orig, fast.MatchExpression):
            expr = self._from_orig_expr(getattr(orig, 'expression', None), frozen_ctx)
            cases = getattr(orig, 'cases', []) or []
            case_exprs: list[HExpr] = []
            arms: list[tuple[HPattern, HExpr]] = []
            for case in cases:
                # Parser Option sugar: ('some', var_name, body) / ('none', None, body)
                if len(case) == 3 and case[0] in ('some', 'none'):
                    tag, var, case_body = case
                    if tag == 'some':
                        pat = HPattern(kind="ctor", name="Some", enum_name="Option",
                                       subpatterns=(HPattern(kind="var", name=str(var)),))
                    else:
                        pat = HPattern(kind="ctor", name="None", enum_name="Option")
                else:
                    pattern, case_body = case
                    pat = self._convert_pattern(pattern)
                case_hexpr = self._from_orig_expr(case_body, frozen_ctx)
                if case_hexpr is not None:
                    case_exprs.append(case_hexpr)
                    arms.append((pat, case_hexpr))
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unit"))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span, op="Match",
                                  scrutinee=expr, cases=tuple(case_exprs), match_arms=tuple(arms))

        # VariantInstance: EnumName::Variant(field=expr, ...)
        if isinstance(orig, fast.VariantInstance):
            enum_name = str(getattr(orig, 'enum_name', '') or '')
            variant_name = str(getattr(orig, 'variant_name', '') or '')
            fvals = getattr(orig, 'field_values', None) or []
            # field_values may be a dict {name: expr} or a list of (name, expr)
            if isinstance(fvals, dict):
                items = list(fvals.items())
            else:
                items = [(fn, fe) for (fn, fe) in fvals]
            payload: list[HExpr] = []
            for (_fname, fexpr) in items:
                he = self._from_orig_expr(fexpr, ctx_for(fexpr))
                if he is not None:
                    payload.append(he)
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unknown"))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                  op="MakeVariant", enum_name=enum_name,
                                  variant_name=variant_name, operands=tuple(payload))

        # WhileStatement: while cond { body }
        if isinstance(orig, fast.WhileStatement):
            cond_node = getattr(orig, 'condition', None)
            c = self._from_orig_expr(cond_node, ctx_for(cond_node))
            body_node = getattr(orig, 'body', None)
            body_he = self._from_orig_expr(body_node, ctx_for(body_node))
            if body_he is not None and body_he.op == "Block" and body_he.operands is not None:
                body_ops = body_he.operands
            elif body_he is not None:
                body_ops = (body_he,)
            else:
                body_ops = tuple()
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unit"))
            return self._mk_hexpr(frozen_ctx.node_id, "Stmt", ty, frozen_ctx.span,
                                  op="While", cond=c, loop_body=body_ops)

        # Assignment: x = expr (rebinds an existing local/param)
        if isinstance(orig, fast.Assignment):
            target = getattr(orig, 'name', None)
            value_node = getattr(orig, 'expression', None)
            val_he = self._from_orig_expr(value_node, ctx_for(value_node))
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unit"))
            return self._mk_hexpr(frozen_ctx.node_id, "Stmt", ty, frozen_ctx.span,
                                  op="Assign", var_name=str(target) if target is not None else None,
                                  assign_value=val_he)

        # ReturnStatement: lower its expression if present
        if isinstance(orig, fast.ReturnStatement):
            expr = getattr(orig, 'expression', None)
            return self._from_orig_expr(expr, frozen_ctx)

        # LetStatement
        if isinstance(orig, fast.LetStatement):
            binds: list[tuple[str, HExpr]] = []
            bind_modes: dict[str, ModeInfo] = {}
            for b in getattr(orig, 'bindings', []) or []:
                name = getattr(b, 'identifier', None)
                init = getattr(b, 'initializer', None)
                he = self._from_orig_expr(init, frozen_ctx)
                if name and he is not None:
                    binds.append((name, he))
                    bind_modes[str(name)] = self._extract_modeinfo(getattr(b, 'mode', None))
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unit"))
            return self._mk_hexpr(frozen_ctx.node_id, "Stmt", ty, frozen_ctx.span, op="Let", bindings=tuple(binds), bind_modes=bind_modes)

        # Block
        if isinstance(orig, fast.Block):
            stmts = getattr(orig, 'statements', []) or []
            ops: list[HExpr] = []
            for s in stmts:
                he = self._from_orig_expr(s, frozen_ctx)
                if he is not None:
                    ops.append(he)
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unit"))
            return self._mk_hexpr(frozen_ctx.node_id, "Block", ty, frozen_ctx.span, op="Block", operands=tuple(ops))

        # StructInstantiation
        if isinstance(orig, fast.StructInstantiation):
            sname_node = getattr(orig, 'struct_name', None)
            sname = str(sname_node) if sname_node is not None else "Unknown"
            field_assigns = getattr(orig, 'field_assignments', []) or []
            field_exprs: list[tuple[str, HExpr]] = []
            for sf in field_assigns:
                fname = getattr(sf, 'name', None)
                fval_node = getattr(sf, 'value', None)
                fval = self._from_orig_expr(fval_node, frozen_ctx)
                if fname and fval is not None:
                    field_exprs.append((str(fname), fval))
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unknown"))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                  op="Struct", struct_name=sname,
                                  fields=tuple(field_exprs),
                                  locality="local")  # default local; borrow checker promotes to global

        # QualifiedFunctionCall: effect_name.op(args) or module.fn(args)
        if isinstance(orig, fast.QualifiedFunctionCall):
            parts = list(getattr(orig, 'parts', []) or [])
            arguments = list(getattr(orig, 'arguments', []) or [])
            arg_exprs = []
            for a in arguments:
                he = self._from_orig_expr(a, ctx_for(a))
                if he is not None:
                    arg_exprs.append(he)
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
            # Enum variant constructor: Option.Some(x) / Option::Some(x)
            if len(parts) == 2 and str(parts[1]) in self._variant_to_enum:
                return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                      op='MakeVariant', enum_name=str(parts[0]),
                                      variant_name=str(parts[1]), operands=tuple(arg_exprs))
            # Trait method call on a named receiver: `d.speak()` (dispatch on
            # the receiver's runtime type; checked BEFORE builtin methods so a
            # user impl of e.g. to_string wins for its receiver type, with the
            # builtin as runtime fallback for everything else).
            # Builtin method call on a value: `x.to_string()` — lower to a
            # call with the receiver (Var or chained FieldGet) as first arg.
            last = str(parts[-1])
            # Static impl-method call on the type itself: `Buffer.new(1024)`.
            if (len(parts) == 2 and str(parts[0]) in self._impl_type_names
                    and last in self._trait_method_names):
                return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                      op='Call',
                                      callee=f"{STATIC_CALL_PREFIX}{parts[0]}{IMPL_SEP}{last}",
                                      operands=tuple(arg_exprs))
            if len(parts) >= 2 and (last in self._trait_method_names
                                    or last in _BUILTIN_METHODS):
                recv: HExpr = self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty,
                                             frozen_ctx.span, op='Var',
                                             var_name=str(parts[0]))
                for fname in parts[1:-1]:
                    recv = self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty,
                                          frozen_ctx.span, op='FieldGet',
                                          base=recv, field_name=str(fname))
                callee = (TRAIT_CALL_PREFIX + last
                          if last in self._trait_method_names else last)
                return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                      op='Call', callee=callee,
                                      operands=(recv, *arg_exprs))
            # Treat as a plain call with dotted callee name
            callee = '.'.join(str(p) for p in parts)
            return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                  op='Call', callee=callee, operands=tuple(arg_exprs))

        # MethodCall on a computed receiver: `expr.method(args)`
        if isinstance(orig, fast.MethodCall):
            recv_node = getattr(orig, 'receiver', None)
            method = str(getattr(orig, 'method', '') or '')
            # Static method on the vector TYPE itself:
            # `vector[float,4].filled(1.0)` — no runtime receiver.
            if isinstance(recv_node, fast.VectorTypeExpression) and method == "filled":
                n = self._const_int_of(getattr(recv_node, 'size', None))
                ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
                n_he = self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                      op='Literal', literal=n)
                arg_exprs = []
                for a in getattr(orig, 'arguments', []) or []:
                    he = self._from_orig_expr(a, ctx_for(a))
                    if he is not None:
                        arg_exprs.append(he)
                return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                      op='Call', callee='__vec_filled',
                                      operands=(n_he, *arg_exprs))
            recv_he = self._from_orig_expr(recv_node, ctx_for(recv_node)) if recv_node is not None else None
            arg_exprs = []
            for a in getattr(orig, 'arguments', []) or []:
                he = self._from_orig_expr(a, ctx_for(a))
                if he is not None:
                    arg_exprs.append(he)
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
            callee = (TRAIT_CALL_PREFIX + method
                      if method in self._trait_method_names else method)
            if recv_he is not None:
                return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                      op='Call', callee=callee,
                                      operands=(recv_he, *arg_exprs))
            return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                  op='Call', callee=callee, operands=tuple(arg_exprs))

        # PerformEffect: perform effect_name(args)
        if isinstance(orig, fast.PerformEffect):
            eff_name = str(getattr(orig, 'effect_name', '') or '')
            arguments = list(getattr(orig, 'arguments', []) or [])
            arg_exprs = []
            for a in arguments:
                he = self._from_orig_expr(a, ctx_for(a))
                if he is not None:
                    arg_exprs.append(he)
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
            return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                  op='Perform', effect_op=eff_name, perform_args=tuple(arg_exprs))

        # HandleEffect: handle EffectType with { cases } in body
        if isinstance(orig, fast.HandleEffect):
            eff_node = getattr(orig, 'effect_name', None)
            # effect_name may be a TypeReference, QualifiedName, string, or AST node
            if hasattr(eff_node, 'name'):
                eff_name = str(eff_node.name)
            elif hasattr(eff_node, 'parts'):
                eff_name = '.'.join(str(p) for p in eff_node.parts)
            else:
                eff_name = str(eff_node or '')
            cases_raw = list(getattr(orig, 'handler', []) or [])
            cont = getattr(orig, 'continuation', None)
            case_triples: list[tuple[str, tuple, HExpr]] = []
            for c in cases_raw:
                if isinstance(c, fast.HandleCase):
                    op_name = str(c.op_name)
                    raw_params = getattr(c, 'param_names', None)
                    if raw_params is None:
                        raw_params = [c.param_name] if c.param_name is not None else []
                    params = tuple(str(p) for p in raw_params) or ("_",)
                    case_body = self._from_orig_expr(c.body, ctx_for(c.body) if c.body is not None else frozen_ctx)
                    if case_body is not None:
                        case_triples.append((op_name, params, case_body))
            body_he = self._from_orig_expr(cont, ctx_for(cont) if cont is not None else frozen_ctx)
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
            return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                  op='Handle',
                                  handle_effect=eff_name,
                                  handle_cases=tuple(case_triples),
                                  handle_body=body_he)

        # FieldAccess
        if isinstance(orig, fast.FieldAccess):
            base_node = getattr(orig, 'base', None) or getattr(orig, 'expression', None)
            if isinstance(base_node, str):
                # The parser stores the base of `c.name` as a raw name string.
                base_ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unknown"))
                base_he = self._mk_hexpr(frozen_ctx.node_id, "Expr", base_ty,
                                         frozen_ctx.span, op="Var", var_name=base_node)
            else:
                base_he = self._from_orig_expr(base_node, frozen_ctx)
            field_names = getattr(orig, 'fields', ()) or ()
            # Chain: for a.b.c, build nested FieldGet(FieldGet(a, b), c)
            current = base_he
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unknown"))
            for fname in field_names:
                if current is not None:
                    current = self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                             op="FieldGet", base=current, field_name=str(fname))
            return current

        # IndexExpression: `base[i]` (plain index) or `base[a:b:c]` (slice).
        # Lowered to strict runtime-library builtin calls: __index_get /
        # __slice_get (absent slice parts become literal None).
        if isinstance(orig, fast.IndexExpression):
            base_node = getattr(orig, 'base', None)
            base_he = self._from_orig_expr(base_node, ctx_for(base_node))
            if base_he is None:
                return None
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
            idx = getattr(orig, 'index', None)
            idx_list = idx if isinstance(idx, list) else [idx]
            current = base_he
            for i in idx_list:
                if i is None:
                    continue
                if isinstance(i, fast.SliceExpression):
                    parts = []
                    for part in (getattr(i, 'start', None), getattr(i, 'stop', None),
                                 getattr(i, 'step', None)):
                        if part is None:
                            parts.append(self._mk_hexpr(
                                frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                op='Literal', literal=None))
                        else:
                            he = self._from_orig_expr(part, ctx_for(part))
                            if he is None:
                                return None
                            parts.append(he)
                    current = self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty,
                                             frozen_ctx.span, op='Call',
                                             callee='__slice_get',
                                             operands=(current, *parts))
                else:
                    ih = self._from_orig_expr(i, ctx_for(i))
                    if ih is None:
                        return None
                    current = self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty,
                                             frozen_ctx.span, op='Call',
                                             callee='__index_get',
                                             operands=(current, ih))
            return current

        # UnaryOperation: `-x` / `!x` — lowered to the neg/not builtins.
        if isinstance(orig, fast.UnaryOperation):
            operand = getattr(orig, 'operand', None)
            operand_he = self._from_orig_expr(operand, ctx_for(operand))
            if operand_he is None:
                return None
            op_sym = str(getattr(orig, 'operator', '') or '')
            callee = {'-': 'neg', '!': 'not', 'not': 'not'}.get(op_sym)
            if callee is None:
                return None
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
            return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                  op='Call', callee=callee, operands=(operand_he,))

        # RangeExpression: `start..end` — lowered to the __range builtin.
        if isinstance(orig, fast.RangeExpression):
            s_node = getattr(orig, 'start', None)
            e_node = getattr(orig, 'end', None)
            s_he = self._from_orig_expr(s_node, ctx_for(s_node))
            e_he = self._from_orig_expr(e_node, ctx_for(e_node))
            if s_he is None or e_he is None:
                return None
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
            return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                  op='Call', callee='__range',
                                  operands=(s_he, e_he))

        # VectorLiteral: `vector[T, N](elems...)`, `vector[T, N]()` (zeros),
        # or `vector[T, N](expr for x in iterable)` (comprehension).
        if isinstance(orig, fast.VectorLiteral):
            return self._convert_vector_literal(orig, frozen_ctx, ctx_for)

        # LambdaExpression
        if isinstance(orig, fast.LambdaExpression):
            params = getattr(orig, 'params', []) or []
            param_names = tuple(str(getattr(p, 'name', p)) for p in params)
            body_nodes = getattr(orig, 'body', None)
            body_he = self._from_orig_expr(body_nodes, frozen_ctx)
            captured_vars = getattr(orig, 'captured_vars', set()) or set()
            capture_modes = getattr(orig, 'capture_modes', {}) or {}
            captures = tuple((str(v), str(capture_modes.get(v, 'borrow'))) for v in sorted(captured_vars))
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unknown"))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                  op="Lambda",
                                  lambda_params=param_names,
                                  lambda_body=body_he,
                                  captures=captures)

        # Fallback: None
        return None

    # ------------------------------------------------------------------
    # Vector literal / comprehension helpers
    # ------------------------------------------------------------------

    def _const_int_of(self, node: Any) -> int | None:
        """Best-effort compile-time integer of a size expression node.

        `vector[float, 4]` carries its size as a TypeReference("4") (or a
        Literal). Returns None when the size is not a literal integer (e.g. a
        const generic `N`), in which case runtime checks that need it fail
        with a clear error instead of guessing.
        """
        if node is None:
            return None
        if isinstance(node, int) and not isinstance(node, bool):
            return node
        if isinstance(node, str):
            return int(node) if node.lstrip("-").isdigit() else None
        if isinstance(node, fast.Literal):
            v = getattr(node, 'value', None)
            return v if isinstance(v, int) and not isinstance(v, bool) else None
        if isinstance(node, fast.TypeReference):
            return self._const_int_of(getattr(node, 'name', None))
        return None

    def _convert_vector_literal(self, orig: Any, frozen_ctx: mast.AstNode,
                                ctx_for: Any) -> HExpr | None:
        n = self._const_int_of(getattr(orig, 'size', None))
        ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
        n_he = self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                              op='Literal', literal=n)
        elements = list(getattr(orig, 'elements', []) or [])
        # Comprehension form: vector[T, N](f(x) for x in iterable)
        if len(elements) == 1 and isinstance(elements[0], fast.Comprehension):
            comp = elements[0]
            lam = self._comprehension_lambda(comp, frozen_ctx, ctx_for)
            iter_node = getattr(comp, 'iterable', None)
            iter_he = self._from_orig_expr(iter_node, ctx_for(iter_node))
            if lam is None or iter_he is None:
                return None
            return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                  op='Call', callee='__vec_comprehension',
                                  operands=(n_he, lam, iter_he))
        # Empty form: vector[T, N]() — zero-initialized
        if not elements:
            base_name = type_base_name(getattr(orig, 'base_type', None))
            base_he = self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                     op='Literal', literal=base_name)
            return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                  op='Call', callee='__vec_zeros',
                                  operands=(n_he, base_he))
        # Explicit elements
        elem_hes: list[HExpr] = []
        for el in elements:
            he = self._from_orig_expr(el, ctx_for(el))
            if he is None:
                return None
            elem_hes.append(he)
        return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                              op='Call', callee='__vec_lit',
                              operands=(n_he, *elem_hes))

    def _comprehension_lambda(self, comp: Any, frozen_ctx: mast.AstNode,
                              ctx_for: Any) -> HExpr | None:
        """Compile a comprehension body into a Lambda HExpr.

        The comprehension targets become the lambda parameters. Free names in
        the body are captured with mode 'auto': the MIR lowering only actually
        captures the ones bound in the enclosing scope (globals/builtins
        resolve by name at call time).
        """
        targets = tuple(str(t) for t in (getattr(comp, 'targets', []) or []))
        if not targets:
            return None
        body_node = getattr(comp, 'expression', None)
        body_he = self._from_orig_expr(body_node, ctx_for(body_node))
        if body_he is None:
            return None
        free = self._free_names(body_node) - set(targets)
        captures = tuple((name, 'auto') for name in sorted(free))
        ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
        return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                              op='Lambda', lambda_params=targets,
                              lambda_body=body_he, captures=captures)

    def _free_names(self, node: Any, _seen: set[int] | None = None) -> set[str]:
        """Names referenced by an expression subtree (variables, call heads)."""
        if _seen is None:
            _seen = set()
        out: set[str] = set()
        if isinstance(node, (list, tuple)):
            for item in node:
                out |= self._free_names(item, _seen)
            return out
        if isinstance(node, dict):
            for v in node.values():
                out |= self._free_names(v, _seen)
            return out
        if not isinstance(node, fast.Node) or id(node) in _seen:
            return out
        _seen.add(id(node))
        if isinstance(node, fast.Variable):
            name = getattr(node, 'name', None)
            if isinstance(name, str):
                out.add(name)
        elif isinstance(node, fast.QualifiedName):
            parts = list(getattr(node, 'parts', []) or [])
            if parts:
                out.add(str(parts[0]))
        elif isinstance(node, fast.QualifiedFunctionCall):
            parts = list(getattr(node, 'parts', []) or [])
            if parts:
                out.add(str(parts[0]))
        elif isinstance(node, fast.FunctionCall):
            name = getattr(node, 'name', None)
            if isinstance(name, str):
                out.add(name)
        elif isinstance(node, fast.FieldAccess):
            base = getattr(node, 'base', None) or getattr(node, 'expression', None)
            if isinstance(base, str):
                out.add(base)
        for attr, value in vars(node).items():
            if attr in ('parent', 'scope', 'location', 'children'):
                continue
            out |= self._free_names(value, _seen)
        return out

    def _convert_pattern(self, p: Any) -> HPattern:
        """Convert a frozen-AST pattern node into an HPattern.

        Note: metaxu_ast defines two WildcardPattern classes (a value-level
        Pattern and a TypePattern); the later definition shadows the former in
        the module namespace, so we match by class name where needed.
        """
        if p is None:
            return HPattern(kind="wildcard")
        cls_name = type(p).__name__
        if cls_name == "WildcardPattern":
            return HPattern(kind="wildcard")
        if isinstance(p, fast.VariablePattern):
            return HPattern(kind="var", name=str(getattr(p, 'name', '_')))
        if isinstance(p, fast.LiteralPattern):
            v = getattr(p, 'value', None)
            if isinstance(v, fast.Literal):
                v = getattr(v, 'value', None)
            return HPattern(kind="literal", value=v)
        if isinstance(p, fast.VariantPattern):
            subs = tuple(self._convert_pattern(sp) for sp in (getattr(p, 'patterns', []) or []))
            enum_name = getattr(p, 'enum_name', None)
            return HPattern(kind="ctor",
                            name=str(getattr(p, 'variant_name', '')),
                            enum_name=str(enum_name) if enum_name is not None else None,
                            subpatterns=subs)
        # Raw python literal used as a pattern (e.g. IfDesugarPass emits
        # LiteralPattern(True); tolerate bare values defensively too)
        if isinstance(p, (bool, int, float, str)):
            return HPattern(kind="literal", value=p)
        # The parser's arm grammar is `expression => body`, so parsed match
        # arms carry *expression* nodes as patterns. Convert the pattern-like
        # expression forms.
        if isinstance(p, fast.Literal):
            return HPattern(kind="literal", value=getattr(p, 'value', None))
        if isinstance(p, fast.Variable):
            name = str(getattr(p, 'name', '_') or '_')
            if name == "_":
                return HPattern(kind="wildcard")
            return HPattern(kind="var", name=name)
        if isinstance(p, fast.NoneExpression):
            return HPattern(kind="ctor", name="None",
                            enum_name=self._variant_to_enum.get("None"), subpatterns=())
        if isinstance(p, fast.SomeExpression):
            inner = getattr(p, 'value', None)
            subs = (self._convert_pattern(inner),) if inner is not None else ()
            return HPattern(kind="ctor", name="Some",
                            enum_name=self._variant_to_enum.get("Some"), subpatterns=subs)
        if isinstance(p, fast.FunctionCall):
            callee = str(getattr(p, 'name', '') or '')
            # Builtin Option constructors match even without a user enum
            # declaring them (mirrors the expression-position fallback).
            if callee in self._variant_to_enum or callee in ("Some", "None"):
                subs = tuple(self._convert_pattern(a)
                             for a in getattr(p, 'arguments', []) or [])
                return HPattern(kind="ctor", name=callee,
                                enum_name=self._variant_to_enum.get(callee, "Option"),
                                subpatterns=subs)
        if isinstance(p, fast.QualifiedFunctionCall):
            parts = list(getattr(p, 'parts', []) or [])
            if len(parts) >= 2:
                subs = tuple(self._convert_pattern(a)
                             for a in getattr(p, 'arguments', []) or [])
                return HPattern(kind="ctor", name=str(parts[-1]),
                                enum_name=str(parts[-2]), subpatterns=subs)
        # Unknown pattern node: treat as wildcard so lowering stays total.
        return HPattern(kind="wildcard")

    def _extract_modeinfo(self, mode: Any) -> ModeInfo:
        mi = ModeInfo()
        if mode is None:
            return mi
        # UniquenessMode
        if isinstance(mode, fast.ModeAnnotation):
            # ModeAnnotation wraps a mode_type, which can be Uniqueness/Locality/Linearity
            mt = getattr(mode, 'mode_type', None)
            if isinstance(mt, fast.UniquenessMode):
                mi.uniqueness = getattr(mt, 'mode', None)
            if isinstance(mt, fast.LocalityMode):
                mi.locality = getattr(mt, 'mode', None)
            if isinstance(mt, fast.LinearityMode):
                mi.linearity = getattr(mt, 'mode', None)
        elif isinstance(mode, fast.UniquenessMode):
            mi.uniqueness = getattr(mode, 'mode', None)
        elif isinstance(mode, fast.LocalityMode):
            mi.locality = getattr(mode, 'mode', None)
        elif isinstance(mode, fast.LinearityMode):
            mi.linearity = getattr(mode, 'mode', None)
        # Chained/combined modes: ModeAnnotationList pattern
        # Some parser variants may combine; attempt to read common fields if iterable
        try:
            for m in getattr(mode, '__dict__', {}).values():
                if isinstance(m, fast.UniquenessMode):
                    mi.uniqueness = getattr(m, 'mode', mi.uniqueness)
                if isinstance(m, fast.LocalityMode):
                    mi.locality = getattr(m, 'mode', mi.locality)
                if isinstance(m, fast.LinearityMode):
                    mi.linearity = getattr(m, 'mode', mi.linearity)
        except Exception:
            pass
        return mi


def dump_hir(funcs: Sequence[HFun]) -> str:
    out: list[str] = []
    for f in funcs:
        out.append(f"fun {f.sym} : {f.ret_ty}")
        out.append(f"  params: {[(str(s), str(t)) for (s, t) in f.params]}")
        out.append(f"  dict_params: {f.dict_params}")
        out.append(f"  where: {f.where_cls}")
        op_str = f"/{f.body.op}" if getattr(f.body, 'op', None) else ""
        out.append(f"  body: {f.body.kind}{op_str}@{f.body.node_id}")
    return "\n".join(out)
