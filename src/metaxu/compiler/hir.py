from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .types import Ty, EffectSet
from .infer_tables import InferSideTables
from .constraints import ClassConstraint
from . import mutaxu_ast as mast
import metaxu.metaxu_ast as fast


@dataclass(slots=True)
class ModeInfo:
    uniqueness: str | None = None   # 'unique'|'exclusive'|'shared'|'owned'|'mutable'|'const'
    locality: str | None = None     # 'local'|'global'
    linearity: str | None = None    # 'once'|'separate'|'many'

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
    cases: tuple['HExpr', ...] | None = None
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
        for orig in self.id_map.values():
            if isinstance(orig, fast.EffectDeclaration):
                for op in (getattr(orig, 'operations', []) or []):
                    if hasattr(op, 'name'):
                        self._effect_op_names.add(str(op.name))

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

        # MatchExpression (desugared from if statements)
        if isinstance(orig, fast.MatchExpression):
            expr = self._from_orig_expr(getattr(orig, 'expression', None), frozen_ctx)
            cases = getattr(orig, 'cases', []) or []
            case_exprs = []
            for pattern, case_body in cases:
                # For now, just lower the case body
                case_hexpr = self._from_orig_expr(case_body, frozen_ctx)
                if case_hexpr is not None:
                    case_exprs.append(case_hexpr)
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unit"))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span, op="Match", scrutinee=expr, cases=tuple(case_exprs))

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
            # Treat as a plain call with dotted callee name
            callee = '.'.join(str(p) for p in parts)
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
            case_triples: list[tuple[str, str, HExpr]] = []
            for c in cases_raw:
                if isinstance(c, fast.HandleCase):
                    op_name = str(c.op_name)
                    param_name = str(c.param_name)
                    case_body = self._from_orig_expr(c.body, ctx_for(c.body) if c.body is not None else frozen_ctx)
                    if case_body is not None:
                        case_triples.append((op_name, param_name, case_body))
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
