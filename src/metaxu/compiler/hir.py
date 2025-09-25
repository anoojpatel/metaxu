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
    op: str | None = None            # e.g., 'Literal', 'Var', 'Call', 'Let', 'Block'
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

    def build(self, root: mast.AstNode) -> list[HFun]:
        funcs: list[HFun] = []

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

    def _from_orig_expr(self, orig: Any, frozen_ctx: mast.AstNode) -> HExpr | None:
        """Build an HExpr from an original AST node or list of nodes.

        frozen_ctx: a frozen node to provide node_id/span context when the original
        node lacks a corresponding frozen node in id_map traversal.
        """
        # Lists become Block
        if isinstance(orig, list):
            ops: list[HExpr] = []
            for item in orig:
                he = self._from_orig_expr(item, frozen_ctx)
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

        # Function calls
        if isinstance(orig, fast.FunctionCall):
            callee = getattr(orig, 'name', None)
            args_exprs = []
            for a in getattr(orig, 'arguments', []) or []:
                he = self._from_orig_expr(a, frozen_ctx)
                if he is not None:
                    args_exprs.append(he)
            ty = self.t.apply_tyenv(getattr(orig, 'type_var', None) or self.t.types.get(frozen_ctx.node_id, "Unknown"))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span, op="Call", callee=str(callee), operands=tuple(args_exprs))

        # BinaryOperation
        if isinstance(orig, fast.BinaryOperation):
            l = self._from_orig_expr(getattr(orig, 'left', None), frozen_ctx)
            r = self._from_orig_expr(getattr(orig, 'right', None), frozen_ctx)
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unknown"))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span, op="BinOp", binop=getattr(orig, 'operator', None), left=l, right=r)

        # IfStatement
        if isinstance(orig, fast.IfStatement):
            c = self._from_orig_expr(getattr(orig, 'condition', None), frozen_ctx)
            tb = self._from_orig_expr(getattr(orig, 'then_body', None), frozen_ctx)
            eb = self._from_orig_expr(getattr(orig, 'else_body', None), frozen_ctx) if getattr(orig, 'else_body', None) is not None else None
            # Flatten block bodies into operand lists
            def as_ops(h: HExpr | None) -> tuple[HExpr, ...]:
                if h is None:
                    return tuple()
                if h.op == "Block" and h.operands is not None:
                    return h.operands
                return (h,)
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unit"))
            return self._mk_hexpr(frozen_ctx.node_id, "Stmt", ty, frozen_ctx.span, op="If", cond=c, then_ops=as_ops(tb), else_ops=as_ops(eb) if eb else tuple())

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
