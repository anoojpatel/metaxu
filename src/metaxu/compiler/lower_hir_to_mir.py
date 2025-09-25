from __future__ import annotations

from typing import Sequence, List, Dict

from .hir import HFun, HExpr
from .mir import MirFunc, MirBlock
from .borrow_analysis import plan_drops


class _ANFState:
    def __init__(self) -> None:
        self.counter = 0
        self.env: Dict[str, str] = {}

    def fresh(self, hint: str = "t") -> str:
        self.counter += 1
        return f"{hint}{self.counter}"


class _FuncLowerer:
    def __init__(self, f: HFun) -> None:
        self.f = f
        self.blocks: List[MirBlock] = []
        self.ops: List[tuple] = []
        self.state = _ANFState()

    def new_block(self) -> int:
        idx = len(self.blocks)
        self.blocks.append(MirBlock(ops=[], term=("br", idx)))  # placeholder
        return idx

    def seal_current_as_block(self, term: tuple) -> int:
        idx = len(self.blocks)
        self.blocks.append(MirBlock(ops=self.ops, term=term))
        self.ops = []
        return idx

    def emit(self, op: tuple) -> None:
        self.ops.append(op)

    def lower_expr(self, e: HExpr) -> str:
        # Handle expression forms that return a value
        if e.op == "Literal":
            dst = self.state.fresh("c")
            self.emit(("let", dst, ("const", e.literal), ()))
            return dst
        if e.op == "Var" and e.var_name:
            return self.state.env.get(e.var_name, e.var_name)
        if e.op == "Call" and e.callee is not None and e.operands is not None:
            arg_names: List[str] = [self.lower_expr(a) for a in e.operands]
            dst = self.state.fresh("v")
            self.emit(("let", dst, ("call", e.callee), tuple(arg_names)))
            return dst
        if e.op == "BinOp" and e.left is not None and e.right is not None:
            l = self.lower_expr(e.left)
            r = self.lower_expr(e.right)
            dst = self.state.fresh("b")
            self.emit(("let", dst, ("binop", e.binop), (l, r)))
            return dst
        if e.op == "Let" and e.bindings is not None:
            last_name = None
            for (name, sube) in e.bindings:
                val = self.lower_expr(sube)
                self.state.env[name] = val
                last_name = name
            return self.state.env.get(last_name, self.state.fresh("unit")) if last_name else self.state.fresh("unit")
        if e.op == "Block" and e.operands is not None:
            last = self.state.fresh("unit")
            for sube in e.operands:
                last = self.lower_expr(sube)
            return last
        # If as an expression: lower control flow and return a fresh value of type
        if e.op == "If" and e.cond is not None:
            cond_val = self.lower_expr(e.cond)
            # end current ops as entry and branch
            then_label = len(self.blocks) + 1  # optimistic labels: entry -> then -> else -> join
            else_label = then_label + 1
            join_label = else_label + 1
            self.seal_current_as_block(("br_if", cond_val, then_label, else_label))
            # Then block
            then_ops_result = self.state.fresh("unit")
            if e.then_ops:
                for sub in e.then_ops:
                    then_ops_result = self.lower_expr(sub)
            self.seal_current_as_block(("br", join_label))
            # Else block
            else_ops_result = self.state.fresh("unit")
            if e.else_ops:
                for sub in e.else_ops:
                    else_ops_result = self.lower_expr(sub)
            self.seal_current_as_block(("br", join_label))
            # Join block
            # For now, produce a const ty as the If-result; future: select based on phi
            dst = self.state.fresh("if")
            self.emit(("let", dst, ("const_ty", str(e.ty)), ()))
            # Do not seal join yet; caller will decide terminator
            return dst
        # Fallback: const of type
        dst = self.state.fresh("ret")
        self.emit(("let", dst, ("const_ty", str(e.ty)), ()))
        return dst


def _lower_hexpr(e: HExpr, ops: List[tuple], st: _ANFState) -> str:
    """Lower an HExpr to ANF ops and return an SSA name for its value."""
    # Pattern by op tag first, then fallback by kind/ty into a const-typed value.
    if e.op == "Literal":
        dst = st.fresh("c")
        ops.append(("let", dst, ("const", e.literal), ()))
        return dst
    if e.op == "Var" and e.var_name:
        # In SSA-lite, a variable refers to a previously bound name or parameter.
        return st.env.get(e.var_name, e.var_name)
    if e.op == "Call" and e.callee is not None and e.operands is not None:
        arg_names: List[str] = []
        for arg in e.operands:
            arg_names.append(_lower_hexpr(arg, ops, st))
        dst = st.fresh("v")
        ops.append(("let", dst, ("call", e.callee), tuple(arg_names)))
        return dst
    if e.op == "Let" and e.bindings is not None:
        last_name = None
        for (name, sube) in e.bindings:
            val = _lower_hexpr(sube, ops, st)
            st.env[name] = val
            last_name = name
        # Value of let-expression: return last bound name if any, else unit-typed const
        if last_name is not None:
            return st.env[last_name]
    if e.op == "Block" and e.operands is not None:
        last = ""
        for sube in e.operands:
            last = _lower_hexpr(sube, ops, st)
        return last
    # Fallback: produce a const of the type for stability
    dst = st.fresh("ret")
    ops.append(("let", dst, ("const_ty", str(e.ty)), ()))
    return dst


def lower_hir_to_mir(funcs: Sequence[HFun]) -> list[MirFunc]:
    """Lower HIR to MIR (ANF direct vs CPS later).

    Now supports BinOp and If (multi-block) in addition to Literal/Var/Call/Let/Block.
    """
    out: list[MirFunc] = []
    drops = plan_drops(list(funcs))
    for f in funcs:
        fl = _FuncLowerer(f)
        # Parameters
        param_names = [str(pname) for (pname, _pty) in f.params]
        fl.emit(("params", tuple(param_names)))
        for pn in param_names:
            fl.state.env[pn] = pn
        res = fl.lower_expr(f.body)
        # Insert drops before ret
        plan = drops.get(str(f.sym))
        if plan:
            for name in plan.drop_at_end:
                fl.emit(("drop", name))
        # Seal the final block as return
        fl.seal_current_as_block(("ret", res))
        blocks = fl.blocks
        if not blocks:
            blocks = [MirBlock(ops=[], term=("ret", res))]
        out.append(MirFunc(name=str(f.sym), ty_sig=f.ret_ty, blocks=blocks, suspending=bool(f.body.suspends)))
    return out
