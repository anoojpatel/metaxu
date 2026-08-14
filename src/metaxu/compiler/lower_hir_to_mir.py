from __future__ import annotations

from typing import Any, Sequence, List, Dict

from .desugar import IMPL_PREFIX
from .hir import HFun, HExpr, HPattern, tuple_field_name
from .mir import MirFunc, MirBlock
from .borrow_analysis import plan_drops


def _span_location(span: Any) -> Any:
    """errors.SourceLocation for a HIR span (None when position-less).

    Only whole functions carry a location into MIR; see MirFunc.location.
    """
    try:
        return span.location()
    except AttributeError:
        return None


class _ANFState:
    def __init__(self) -> None:
        self.counter = 0
        self.env: Dict[str, str] = {}

    def fresh(self, hint: str = "t") -> str:
        self.counter += 1
        return f"{hint}{self.counter}"


class _FuncLowerer:
    """Lowers one HIR function body into MIR blocks.

    Block model: all blocks are allocated up-front as objects in ``self.blocks``
    (so their indices/labels are stable no matter how many blocks a nested arm
    creates), and ``self.cur`` points at the block currently being filled.
    A freshly allocated block carries an ("unreachable",) placeholder terminator
    which is overwritten via ``terminate`` once its real successor is known.
    """

    def __init__(self, f: HFun) -> None:
        self.f = f
        self.blocks: List[MirBlock] = [MirBlock(ops=[], term=("unreachable",))]
        self.cur: int = 0
        self.state = _ANFState()
        self._pending_lambdas: List[MirFunc] = []
        # Early-return plumbing: allocated lazily by the first "Return" op.
        # All returns copy into ret_var and branch to ret_bb; the function's
        # epilogue (drops + the actual ret) lives in ret_bb.
        self.ret_bb: int | None = None
        self.ret_var: str | None = None

    # ------------------------------------------------------------------
    # Block plumbing
    # ------------------------------------------------------------------

    def new_block(self) -> int:
        idx = len(self.blocks)
        self.blocks.append(MirBlock(ops=[], term=("unreachable",)))
        return idx

    def switch_to(self, idx: int) -> None:
        self.cur = idx

    def terminate(self, term: tuple) -> None:
        self.blocks[self.cur].term = term

    def emit(self, op: tuple) -> None:
        self.blocks[self.cur].ops.append(op)

    def unit_value(self) -> str:
        """Emit a unit constant and return its name (always bound at runtime)."""
        dst = self.state.fresh("unit")
        self.emit(("let", dst, ("const_ty", "Unit"), ()))
        return dst

    def finish_body(self, res: str, drop_names: Sequence[str] = ()) -> None:
        """Terminate the function body whose result value is ``res``.

        Without early returns this is the classic ``drops; ret res`` in the
        current block. When any "Return" op fired, the epilogue instead lives
        in the shared ret_bb: the fall-off-the-end path copies its result
        into ret_var and joins the early returns there.
        """
        if self.ret_bb is None:
            for name in drop_names:
                self.emit(("drop", name))
            self.terminate(("ret", res))
            return
        assert self.ret_var is not None
        self.emit(("let", self.ret_var, ("copy",), (res,)))
        self.terminate(("br", self.ret_bb))
        self.switch_to(self.ret_bb)
        for name in drop_names:
            self.emit(("drop", name))
        self.terminate(("ret", self.ret_var))

    # ------------------------------------------------------------------
    # Mut-capture analysis (write-back through shared cells)
    # ------------------------------------------------------------------

    @staticmethod
    def _assigned_names(e: "HExpr | None", acc: set | None = None) -> set:
        """Source names assigned (plain `x = ...`, not `x.f = ...`) anywhere
        inside ``e``, including nested lambdas and handler arms.

        Used to decide which enclosing bindings a sub-function mutates: those
        slots are boxed into shared MxCells (``cell_wrap``) before capture so
        the writes are visible outside the sub-function instead of being
        silently lost in a by-value environment copy.
        """
        acc = set() if acc is None else acc
        if e is None:
            return acc
        if e.op == "Assign" and e.var_name and "." not in e.var_name:
            acc.add(e.var_name)
        walk = _FuncLowerer._assigned_names
        for child in (e.assign_value, e.left, e.right, e.cond, e.scrutinee,
                      e.base, e.field_val, e.lambda_body, e.handle_body):
            walk(child, acc)
        for seq in (e.operands, e.then_ops, e.else_ops, e.loop_body,
                    e.cases, e.perform_args):
            for sub in (seq or ()):
                walk(sub, acc)
        for (_n, sub) in (e.bindings or ()):
            walk(sub, acc)
        for (_n, sub) in (e.fields or ()):
            walk(sub, acc)
        for (_p, body) in (e.match_arms or ()):
            walk(body, acc)
        for (_op, _params, body) in (e.handle_cases or ()):
            walk(body, acc)
        return acc

    def _wrap_mut_captures(self, names) -> None:
        """Emit cell_wrap for each enclosing binding in ``names`` so the
        sub-function about to capture the env aliases (not copies) them."""
        for name in sorted(names):
            slot = self.state.env.get(name)
            if slot is not None:
                self.emit(("cell_wrap", slot))

    # ------------------------------------------------------------------
    # Sub-function compilation (lambdas, effect handler cases)
    # ------------------------------------------------------------------

    def _lower_subfunc(self, name: str, params: Sequence[str], body: HExpr,
                       ty_sig: Any, suspending: bool,
                       mut_params: Sequence[str] = ()) -> None:
        """Compile ``body`` into a standalone MirFunc appended to pending lambdas.

        Swaps out the block context so nested control flow inside the
        sub-function cannot pollute the enclosing function's blocks.
        """
        saved_blocks, saved_cur = self.blocks, self.cur
        saved_env = dict(self.state.env)
        saved_ret_bb, saved_ret_var = self.ret_bb, self.ret_var
        self.blocks = [MirBlock(ops=[("params", tuple(params))], term=("unreachable",))]
        self.cur = 0
        self.ret_bb, self.ret_var = None, None
        for pn in params:
            self.state.env[pn] = pn
        result = self.lower_expr(body)
        self.finish_body(result)
        sub_blocks = self.blocks
        self.blocks, self.cur = saved_blocks, saved_cur
        self.state.env = saved_env
        self.ret_bb, self.ret_var = saved_ret_bb, saved_ret_var
        self._pending_lambdas.append(
            MirFunc(name=name, ty_sig=ty_sig, blocks=sub_blocks,
                    suspending=suspending, mut_params=tuple(mut_params))
        )

    # ------------------------------------------------------------------
    # Pattern compilation
    # ------------------------------------------------------------------

    def compile_pattern(self, pat: HPattern, val_name: str, fail_bb: int) -> None:
        """Emit test-and-branch ops for ``pat`` against ``val_name``.

        On fall-through (staying in the current block chain) the pattern has
        matched and all its variables are bound; on failure control jumps to
        ``fail_bb``.
        """
        if pat.kind == "wildcard":
            return
        if pat.kind == "var":
            dst = self.state.fresh("p")
            self.emit(("let", dst, ("copy",), (val_name,)))
            self.state.env[str(pat.name)] = dst
            return
        if pat.kind == "literal":
            c = self.state.fresh("c")
            self.emit(("let", c, ("const", pat.value), ()))
            cond = self.state.fresh("t")
            self.emit(("let", cond, ("binop", "=="), (val_name, c)))
            ok_bb = self.new_block()
            self.terminate(("br_if", cond, ok_bb, fail_bb))
            self.switch_to(ok_bb)
            return
        if pat.kind == "tuple":
            # A tuple is an anonymous struct (hir.TUPLE_STRUCT_PREFIX), so
            # destructuring is a positional field_get per element and NO tag
            # test.  The arity is not branched on because it is not a
            # runtime choice: the field names carry it (`_0of2` exists only
            # on a `__tuple2`), so a pattern whose arity disagrees with the
            # value is a loud missing-field error, not a quietly-failing arm
            # and not a quietly-succeeding prefix bind.  Refutable
            # subpatterns still branch to fail_bb.
            arity = len(pat.subpatterns)
            for i, sub in enumerate(pat.subpatterns):
                fv = self.state.fresh("pt")
                self.emit(("let", fv, ("field_get", tuple_field_name(i, arity)),
                           (val_name,)))
                self.compile_pattern(sub, fv, fail_bb)
            return
        if pat.kind == "ctor":
            tag = self.state.fresh("tag")
            self.emit(("let", tag, ("variant_tag",), (val_name,)))
            c = self.state.fresh("c")
            self.emit(("let", c, ("const", str(pat.name)), ()))
            cond = self.state.fresh("t")
            self.emit(("let", cond, ("binop", "=="), (tag, c)))
            ok_bb = self.new_block()
            self.terminate(("br_if", cond, ok_bb, fail_bb))
            self.switch_to(ok_bb)
            for i, sub in enumerate(pat.subpatterns):
                fv = self.state.fresh("pf")
                # The op carries the pattern's ctor name as a third element so
                # backends know WHICH variant's slot is being read (the read
                # sits under this ctor's tag test).  Consumers that don't need
                # it (interpreter, CLIF) read only rhs[1]; legacy two-element
                # shapes remain valid MIR.
                self.emit(("let", fv,
                           ("variant_field", i, str(pat.name)), (val_name,)))
                self.compile_pattern(sub, fv, fail_bb)
            return
        raise ValueError(f"Unknown pattern kind: {pat.kind!r}")

    # ------------------------------------------------------------------
    # Expression lowering
    # ------------------------------------------------------------------

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
            # Resolve the callee through the local env: a let-bound closure
            # (`let g = fn(y) ...; g(2)`) lives in a renamed slot (e.g. g_6),
            # so the call must reference that slot, not the source name.
            # Names with no local binding (top-level funcs, builtins) pass
            # through unchanged.
            callee = self.state.env.get(e.callee, e.callee)
            self.emit(("let", dst, ("call", callee), tuple(arg_names)))
            return dst
        if e.op == "BinOp" and e.left is not None and e.right is not None:
            l = self.lower_expr(e.left)
            r = self.lower_expr(e.right)
            dst = self.state.fresh("b")
            self.emit(("let", dst, ("binop", e.binop), (l, r)))
            return dst
        if e.op == "Let" and e.bindings is not None:
            last_val: str | None = None
            for (name, sube) in e.bindings:
                val = self.lower_expr(sube)
                # Bind into a dedicated slot (not an alias of the initializer's
                # temp) so later assignments to this variable cannot clobber
                # the initializer's own slot (e.g. `let j = b; j = j - 1`).
                slot = self.state.fresh(f"{name}_")
                self.emit(("let", slot, ("copy",), (val,)))
                self.state.env[name] = slot
                last_val = slot
            return last_val if last_val is not None else self.unit_value()
        if e.op == "Block" and e.operands is not None:
            last: str | None = None
            for sube in e.operands:
                last = self.lower_expr(sube)
                # A bare name read emits NO instruction: `lower_expr("Var")`
                # just answers the slot name. In value position the consumer
                # names that slot, so the read is real; in STATEMENT position
                # the name is discarded and the read disappears from MIR
                # entirely — which is exactly how `undefined_thing; 42`
                # compiled to `ret 42` with the undefined name nowhere in
                # sight (docs/name_resolution.md). Name resolution now
                # rejects that at compile time; force the read anyway so a
                # slot that is somehow unbound at run time is loud
                # ("Unbound variable ...") instead of vanishing. Every other
                # expression form already emits its own instruction, so its
                # evaluation survives statement position unaided.
                #
                # The TAIL is not exempt. It used to be (`i != n - 1`), on the
                # theory that the block's value is consumed by the caller —
                # but a function body's tail is consumed by `ret <slot>`, and
                # `mir_interp`'s ret keeps a legacy fallback for slots that
                # are absent from the environment. So `fn inner() -> int {
                # secret }` returned the PREVIOUS op's value (`()`) instead of
                # raising: a silent wrong answer, the very shape this fix
                # exists to remove. The copy is emitted for every position and
                # `last` still names the source slot, so the MIR is unchanged
                # apart from the forced read.
                if sube.op == "Var" and sube.var_name and last:
                    dst = self.state.fresh("r")
                    self.emit(("let", dst, ("copy",), (last,)))
            return last if last is not None else self.unit_value()
        # Assignment: write through to the variable's runtime slot so loops see
        # the updated value on the next iteration.
        if e.op == "Assign" and e.var_name is not None:
            val = self.unit_value() if e.assign_value is None else self.lower_expr(e.assign_value)
            # Field assignment: `x.f = v` (target arrives as a dotted string).
            # Structs have value semantics in MIR, so read the intermediate
            # structs, set the innermost field, and write the updated structs
            # back out to the base slot.
            if "." in e.var_name:
                base, *fields = e.var_name.split(".")
                base_slot = self.state.env.get(base, base)
                chain = [base_slot]
                for fname in fields[:-1]:
                    nxt = self.state.fresh("fg")
                    self.emit(("let", nxt, ("field_get", fname), (chain[-1],)))
                    chain.append(nxt)
                updated = val
                for fname, holder in zip(reversed(fields), reversed(chain)):
                    nxt = self.state.fresh("fs")
                    self.emit(("let", nxt, ("field_set", fname), (holder, updated)))
                    updated = nxt
                self.emit(("let", base_slot, ("copy",), (updated,)))
                return base_slot
            slot = self.state.env.get(e.var_name)
            if slot is None:
                # First assignment introduces the slot (named after the variable)
                slot = e.var_name
                self.state.env[e.var_name] = slot
            self.emit(("let", slot, ("copy",), (val,)))
            return slot
        # Match: decision-tree lowering, first-match-wins top-to-bottom.
        if e.op == "Match" and e.scrutinee is not None:
            return self._lower_match(e)
        # If as an expression: real control flow with a join block.
        # Layout: cur(br_if) -> then_bb ... -> join_bb
        #                    -> else_bb ... -> join_bb
        # Each arm copies its result into a shared result variable before
        # branching to the join, so the join sees exactly one binding.
        if e.op == "If" and e.cond is not None:
            cond_val = self.lower_expr(e.cond)
            res_var = self.state.fresh("if")
            # else_ops=None marks an else-less if: it always evaluates to
            # unit (statement rule), so the then-arm runs for effects only
            # and its value is discarded instead of merging with unit.
            has_else = e.else_ops is not None
            then_bb = self.new_block()
            else_bb = self.new_block()
            join_bb = self.new_block()
            self.terminate(("br_if", cond_val, then_bb, else_bb))
            # Then arm
            self.switch_to(then_bb)
            saved_env = dict(self.state.env)
            then_result: str | None = None
            for sub in (e.then_ops or ()):
                then_result = self.lower_expr(sub)
            if then_result is None or not has_else:
                then_result = self.unit_value()
            self.emit(("let", res_var, ("copy",), (then_result,)))
            self.terminate(("br", join_bb))
            self.state.env = dict(saved_env)
            # Else arm
            self.switch_to(else_bb)
            else_result: str | None = None
            for sub in (e.else_ops or ()):
                else_result = self.lower_expr(sub)
            if else_result is None:
                else_result = self.unit_value()
            self.emit(("let", res_var, ("copy",), (else_result,)))
            self.terminate(("br", join_bb))
            self.state.env = saved_env
            # Join
            self.switch_to(join_bb)
            return res_var
        # While loop: header/body/exit blocks with a back-edge to the header.
        if e.op == "While" and e.cond is not None:
            header_bb = self.new_block()
            body_bb = self.new_block()
            exit_bb = self.new_block()
            self.terminate(("br", header_bb))
            # Header: (re-)evaluate the condition each iteration
            self.switch_to(header_bb)
            cond_val = self.lower_expr(e.cond)
            self.terminate(("br_if", cond_val, body_bb, exit_bb))
            # Body
            self.switch_to(body_bb)
            saved_env = dict(self.state.env)
            for sub in (e.loop_body or ()):
                self.lower_expr(sub)
            self.terminate(("br", header_bb))
            self.state.env = saved_env
            # Exit: a while loop evaluates to unit
            self.switch_to(exit_bb)
            return self.unit_value()
        # While-let loop: `while let PAT = expr { body }`. The header
        # re-evaluates expr each iteration and pattern-matches it; a match
        # binds the pattern variables and runs the body, a mismatch exits.
        if e.op == "WhileLet" and e.scrutinee is not None and e.loop_pattern is not None:
            header_bb = self.new_block()
            body_bb = self.new_block()
            exit_bb = self.new_block()
            self.terminate(("br", header_bb))
            self.switch_to(header_bb)
            saved_env = dict(self.state.env)
            scrut_val = self.lower_expr(e.scrutinee)
            # On mismatch compile_pattern jumps to exit_bb; on fall-through
            # all pattern variables are bound for the body.
            self.compile_pattern(e.loop_pattern, scrut_val, exit_bb)
            self.terminate(("br", body_bb))
            self.switch_to(body_bb)
            for sub in (e.loop_body or ()):
                self.lower_expr(sub)
            self.terminate(("br", header_bb))
            self.state.env = saved_env
            # Exit: a while-let loop evaluates to unit
            self.switch_to(exit_bb)
            return self.unit_value()
        # Early return: copy the value into the shared return slot and jump
        # to the (lazily created) epilogue block. Subsequent code in the
        # current arm lowers into a fresh unreachable block and is dead.
        if e.op == "Return":
            val = self.lower_expr(e.operands[0]) if e.operands else self.unit_value()
            if self.ret_bb is None:
                self.ret_bb = self.new_block()
                self.ret_var = self.state.fresh("retv")
            assert self.ret_var is not None
            self.emit(("let", self.ret_var, ("copy",), (val,)))
            self.terminate(("br", self.ret_bb))
            self.switch_to(self.new_block())
            return val
        # Enum variant construction
        if e.op == "MakeVariant" and e.variant_name is not None:
            payload = [self.lower_expr(a) for a in (e.operands or ())]
            dst = self.state.fresh("vt")
            self.emit(("let", dst, ("make_variant", e.enum_name or "", e.variant_name), tuple(payload)))
            return dst
        # Struct instantiation
        if e.op == "Struct" and e.struct_name is not None:
            field_vals: List[tuple] = []
            for (fname, fexpr) in (e.fields or ()):
                fval = self.lower_expr(fexpr)
                field_vals.append((fname, fval))
            dst = self.state.fresh("s")
            locality = e.locality or "local"
            self.emit(("let", dst, ("alloc_struct", e.struct_name, locality), tuple(field_vals)))
            return dst
        # Field access
        if e.op == "FieldGet" and e.base is not None and e.field_name is not None:
            base_val = self.lower_expr(e.base)
            dst = self.state.fresh("f")
            self.emit(("let", dst, ("field_get", e.field_name), (base_val,)))
            return dst
        if e.op == "FieldSet" and e.base is not None and e.field_name is not None and e.field_val is not None:
            base_val = self.lower_expr(e.base)
            new_val = self.lower_expr(e.field_val)
            dst = self.state.fresh("fs")
            self.emit(("let", dst, ("field_set", e.field_name), (base_val, new_val)))
            return dst
        # Lambda / closure
        if e.op == "Lambda" and e.lambda_params is not None:
            # Qualify with the enclosing function's name: the fresh counter
            # is per-function, so bare "lambdaN" names collided ACROSS
            # functions (e.g. sum's (a,b)->a+b and prod's (a,b)->a*b both
            # lowered to "lambda1"; whichever loaded last won and sum
            # silently multiplied). MirFuncs live in one flat namespace.
            lname = f"{self.f.sym}${self.state.fresh('lambda')}"
            # Capture current env values. The lambda body is compiled against
            # the enclosing env's SLOT names (source `x` may live in slot
            # `x_2`), so the runtime closure env must be keyed by the slot
            # name the body actually references — capture (slot, slot) pairs,
            # same convention as handle_scope below.
            cap_names: List[tuple] = []
            # Captured bindings the lambda body ASSIGNS to are boxed into
            # shared cells first, so the mutation writes back to the
            # enclosing frame instead of vanishing in a by-value copy.
            # The HIR body scan is authoritative here — the parser's
            # 'borrow_mut' capture mode over-approximates (it flags any
            # variable merely PASSED to a function), and wrapping read-only
            # captures would demote them in the native backends for nothing.
            assigned = self._assigned_names(e.lambda_body)
            capture_names = {cname for (cname, _m) in (e.captures or ())}
            self._wrap_mut_captures(assigned & capture_names
                                    & set(self.state.env))
            for (cname, cmode) in (e.captures or ()):
                # 'auto' captures come from comprehension free-name analysis
                # (hir._comprehension_lambda): only names actually bound in
                # the enclosing scope are captured — the rest are globals or
                # builtins that resolve by name at call time.
                if cmode == 'auto' and cname not in self.state.env:
                    continue
                slot = self.state.env.get(cname, cname)
                cap_names.append((slot, slot))
            dst = self.state.fresh("cl")
            self.emit(("let", dst, ("make_closure", lname, e.lambda_params), tuple(cap_names)))
            # Compile the lambda body as a deferred sub-function
            if e.lambda_body is not None:
                self._lower_subfunc(lname, e.lambda_params, e.lambda_body,
                                    ty_sig=e.ty, suspending=bool(e.suspends),
                                    mut_params=e.lambda_mut_params or ())
            return dst
        # Perform: perform Effect.op(args) — a real suspension point. The op
        # ends its block; the continuation is "this function from resume_bb on"
        # plus (implicitly) the Python-level call stack below this frame.
        if e.op == "Perform" and e.effect_op is not None:
            arg_names: List[str] = [self.lower_expr(a) for a in (e.perform_args or ())]
            dst = self.state.fresh("pv")
            raw = str(e.effect_op)
            effect_name, _, op_name = raw.rpartition(".")
            resume_bb = self.new_block()
            self.emit(("perform", dst, effect_name, op_name, tuple(arg_names),
                       resume_bb, dst))
            self.terminate(("br", resume_bb))
            self.switch_to(resume_bb)
            return dst
        # Resume: resume(value) inside a handle case — consume the single-shot
        # continuation bound to the handler's implicit __k parameter.
        if e.op == "Resume":
            args_list = list(e.perform_args or ())
            val = self.lower_expr(args_list[0]) if args_list else self.unit_value()
            dst = self.state.fresh("rv")
            self.emit(("let", dst, ("resume",), ("__k", val)))
            return dst
        # Handle: handle Effect with { cases } in body. The body and each case
        # compile to sub-functions; at runtime handle_scope pushes a delimited
        # handler frame, runs the body under it, and catches handler aborts
        # (a case returning without calling resume).
        if e.op == "Try" and e.handle_body is not None and e.handle_cases:
            # Qualify with the enclosing function's name: the fresh counter is
            # per-function, so bare "__catch_tcN" names collided ACROSS
            # functions (same defect the lambda naming fix documents above).
            scope_tag = f"{self.f.sym}${self.state.fresh('tc')}"
            (_, catch_params, catch_he) = e.handle_cases[0]
            if isinstance(catch_params, str):
                catch_params = (catch_params,)
            catch_fn = f"__catch_{scope_tag}"
            self._lower_subfunc(catch_fn, tuple(catch_params), catch_he,
                                ty_sig=e.ty, suspending=False)
            body_fn = f"__try_body_{scope_tag}"
            self._lower_subfunc(body_fn, (), e.handle_body, ty_sig=e.ty,
                                suspending=False)
            assigned = self._assigned_names(e.handle_body)
            self._assigned_names(catch_he, assigned)
            self._wrap_mut_captures(assigned & set(self.state.env))
            captured = tuple(sorted({v for v in self.state.env.values() if isinstance(v, str)}))
            captures = tuple((name, name) for name in captured)
            dst = self.state.fresh("tres")
            self.emit(("let", dst, ("try_scope", body_fn, catch_fn), captures))
            return dst
        if e.op == "Handle" and e.handle_body is not None:
            effect = str(e.handle_effect or "")
            if "<" in effect:  # strip generic args: State<int> -> State
                effect = effect.split("<", 1)[0]
            # Qualified like lambda names: two functions handling the same
            # effect+op each lower handler cases to sub-functions, and the
            # per-function fresh counter alone made those names collide
            # across functions (whichever loaded last silently won).
            scope_tag = f"{self.f.sym}${self.state.fresh('hs')}"
            cases_encoded: List[tuple] = []
            for (op_name, case_params, body_he) in (e.handle_cases or ()):
                # case_params is a tuple of parameter names (multi-arg ops);
                # tolerate a legacy bare string.
                if isinstance(case_params, str):
                    case_params = (case_params,)
                hfn = f"__handler_{effect}_{op_name}_{scope_tag}"
                self._lower_subfunc(hfn, (*case_params, "__k"), body_he,
                                    ty_sig=e.ty, suspending=False)
                cases_encoded.append((op_name, tuple(case_params), hfn))
            body_fn = f"__handle_body_{effect}_{scope_tag}"
            self._lower_subfunc(body_fn, (), e.handle_body, ty_sig=e.ty,
                                suspending=False)
            # Bindings the delimited body or a handler arm assigns to are
            # boxed into shared cells so the writes survive the per-frame
            # env copies (by-value capture silently swallowed them before).
            assigned = self._assigned_names(e.handle_body)
            for (_op, _params, body_he) in (e.handle_cases or ()):
                self._assigned_names(body_he, assigned)
            self._wrap_mut_captures(assigned & set(self.state.env))
            # Capture every live MIR value so body/case sub-functions can see
            # enclosing locals (read-only value semantics, like closures).
            captured = tuple(sorted({v for v in self.state.env.values() if isinstance(v, str)}))
            captures = tuple((name, name) for name in captured)
            dst = self.state.fresh("hres")
            self.emit(("let", dst,
                       ("handle_scope", body_fn, effect, tuple(cases_encoded)),
                       captures))
            return dst
        # Fallback: const of type
        dst = self.state.fresh("ret")
        self.emit(("let", dst, ("const_ty", str(e.ty)), ()))
        return dst

    # ------------------------------------------------------------------
    # Match lowering
    # ------------------------------------------------------------------

    def _lower_match(self, e: HExpr) -> str:
        scrut = self.lower_expr(e.scrutinee)  # type: ignore[arg-type]
        res_var = self.state.fresh("m")
        join_bb = self.new_block()
        arms: tuple[tuple[HPattern, HExpr], ...]
        if e.match_arms is not None:
            arms = e.match_arms
        elif e.cases:
            # Legacy Match without patterns: first-match-wins means the first
            # arm (irrefutable) always matches.
            arms = tuple((HPattern(kind="wildcard"), body) for body in e.cases)
        else:
            arms = tuple()
        if not arms:
            self.emit(("match_fail", "empty match"))
            self.terminate(("br", join_bb))
            self.switch_to(join_bb)
            return self.unit_value()
        for (pat, body) in arms:
            fail_bb = self.new_block()
            saved_env = dict(self.state.env)
            self.compile_pattern(pat, scrut, fail_bb)
            body_val = self.lower_expr(body)
            self.emit(("let", res_var, ("copy",), (body_val,)))
            self.terminate(("br", join_bb))
            self.state.env = saved_env
            self.switch_to(fail_bb)
        # Fell off the last arm: no pattern matched.
        self.emit(("match_fail", "no pattern matched"))
        self.terminate(("br", join_bb))
        self.switch_to(join_bb)
        return res_var


def _is_matrix_annotation(pty: Any) -> bool:
    """True when a parameter's type annotation is a vector of vectors."""

    def vector_elem(t: Any) -> Any:
        # Returns the element type when t is a vector type, else a sentinel.
        ctor = getattr(t, "type_constructor", None)
        if ctor == "vector":
            args = list(getattr(t, "type_args", None) or [])
            return args[0] if args else None
        if type(t).__name__ == "VectorTypeExpression":
            return getattr(t, "base_type", None)
        return _NOT_VECTOR

    elem = vector_elem(pty)
    if elem is _NOT_VECTOR or elem is None:
        return False
    return vector_elem(elem) is not _NOT_VECTOR


_NOT_VECTOR = object()

# Uniqueness spellings that grant pass-by-reference (write-back) semantics:
# `@mut` arrives as 'mutable' (hir._absorb_mode_token), its spec spelling is
# 'exclusive'.
_MUT_UNIQUENESS = ("mutable", "exclusive")


def _mut_param_names(f: HFun) -> tuple:
    """Parameter names of ``f`` with write-back (by-reference) semantics.

    Params explicitly declared @mut, plus the `self` receiver of impl
    methods (`__impl$Trait$Type$m`): method bodies mutate the receiver via
    `self.field = ...` without an explicit mode, and both engines have
    always written the receiver back to the caller. A receiver explicitly
    declared @const/shared opts out. Everything else keeps value semantics.
    """
    modes = f.param_modes or {}
    names = []
    for (pname, _pty) in f.params:
        p = str(pname)
        mi = modes.get(p)
        uniq = getattr(mi, "uniqueness", None)
        if uniq in _MUT_UNIQUENESS:
            names.append(p)
        elif (p == "self" and str(f.sym).startswith(IMPL_PREFIX)
                and uniq not in ("shared", "const")):
            names.append(p)
    return tuple(names)


def lower_hir_to_mir(funcs: Sequence[HFun], borrow_errors: List[Any] | None = None) -> list[MirFunc]:
    """Lower HIR to MIR (ANF direct vs CPS later).

    Supports Literal/Var/Call/Let/Block/BinOp, real multi-block control flow
    (If, While, Match with pattern decision trees), enum variant construction,
    structs, lambdas, and effect perform/handle.

    Arguments:
        funcs: HIR functions to lower
        borrow_errors: Optional borrow errors from frozen borrow checker (tables.constraints.get(-2, []))
    """
    out: list[MirFunc] = []
    drops = plan_drops(list(funcs), borrow_errors)
    for f in funcs:
        fl = _FuncLowerer(f)
        # Parameters
        param_names = [str(pname) for (pname, _pty) in f.params]
        fl.emit(("params", tuple(param_names)))
        # Parameters declared as matrices (vector[vector[T,N],M]) accept a
        # flat vector as an Mx1 column: the interpreter promotes such an
        # argument to a real nested vector at entry, so generic matrix code
        # (transpose's self[j][i]) runs strictly instead of indexing
        # scalars. This is the boundary where `mat.matmul(vec)` — the
        # example's "matrix-vector multiplication" — becomes well-shaped.
        # DOCUMENTED SEMANTICS (kept per adversarial review round 5):
        # examples/06 explicitly promises matrix-vector multiplication for
        # a flat vector argument, so the column embedding is intentional,
        # not a lenient fallback. Both engines implement it identically
        # (mir_interp promote_matrix / codegen_llvm mx_fvec_promote;
        # parity pinned in test_round5_regressions).
        matrix_params = tuple(
            str(pname) for (pname, pty) in f.params
            if _is_matrix_annotation(pty)
        )
        if matrix_params:
            fl.emit(("promote_matrix", matrix_params))
        for pn in param_names:
            fl.state.env[pn] = pn
        res = fl.lower_expr(f.body)
        # Epilogue: drops + ret (joined with any early returns via ret_bb)
        plan = drops.get(str(f.sym))
        fl.finish_body(res, plan.drop_at_end if plan else ())
        out.append(MirFunc(name=str(f.sym), ty_sig=f.ret_ty, blocks=fl.blocks,
                           suspending=bool(f.body.suspends),
                           globals_decl=tuple(getattr(f, "globals_decl", ()) or ()),
                           mut_params=_mut_param_names(f),
                           location=_span_location(f.body.span)))
        # Emit any lambdas that were compiled during lowering
        out.extend(fl._pending_lambdas)
    return out
