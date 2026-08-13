"""MIR interpreter for Metaxu.

Executes MirFunc/MirBlock ops directly, providing:
- Arithmetic and comparison binary ops
- Control flow: br, br_if, ret
- Function calls (user-defined + builtins)
- Effect perform/handle with real delimited, single-shot continuations:
  MIR handle_scope bodies run on their own (parked) thread so a perform deep
  inside called functions suspends the whole delimited context; resume(v)
  returns the final value of the whole handle body (deep handlers). Host
  handlers registered via register_effect_handler keep the legacy frame-level
  MxContinuation semantics (stack and suspend classes).
- Enum variants (make_variant / variant_tag / variant_field)
- Drop (removes the binding; any later use raises InterpError)
- Strict name resolution: referencing an unbound variable raises InterpError
  instead of silently evaluating to the name string

Usage:
    interp = MirInterpreter()
    interp.load(mir_funcs)
    result = interp.call("main", [])
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from queue import SimpleQueue
from typing import Any, Callable, Dict, List, Optional, Sequence

from .mir import MirBlock, MirFunc


# ---------------------------------------------------------------------------
# Runtime values
# ---------------------------------------------------------------------------

class MxUnit:
    """Singleton unit value."""
    _instance: "MxUnit | None" = None
    def __new__(cls) -> "MxUnit":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __repr__(self) -> str:
        return "()"

UNIT = MxUnit()


@dataclass
class MxContinuation:
    """Single-shot continuation: a suspended frame waiting to be resumed once."""
    func: "MirFunc"
    block_idx: int
    env: Dict[str, Any]
    result_slot: str
    used: bool = False

    def resume(self, value: Any, interp: "MirInterpreter") -> Any:
        if self.used:
            raise RuntimeError("Continuation already consumed (single-shot violation)")
        self.used = True
        env = dict(self.env)
        env[self.result_slot] = value
        return interp._run_blocks(self.func, self.block_idx, env)


@dataclass
class MxStruct:
    """A runtime struct value.

    locality='local'  → stack/fiber allocation (non-escaping by default)
    locality='global' → heap allocation (promoted, freely shareable)
    """
    name: str
    fields: Dict[str, Any]
    locality: str = "local"

    def get(self, field_name: str) -> Any:
        if field_name not in self.fields:
            raise KeyError(f"Struct '{self.name}' has no field '{field_name}'")
        return self.fields[field_name]

    def set(self, field_name: str, value: Any) -> "MxStruct":
        """Return a new struct with the field updated (value semantics)."""
        new_fields = dict(self.fields)
        new_fields[field_name] = value
        return MxStruct(name=self.name, fields=new_fields, locality=self.locality)

    def __repr__(self) -> str:
        fields_str = ", ".join(f"{k}={v!r}" for k, v in self.fields.items())
        return f"{self.name} {{ {fields_str} }}"


@dataclass
class MxVariant:
    """A runtime enum variant value: tag + positional payload fields."""
    enum_name: str
    tag: str
    fields: tuple = ()

    def __repr__(self) -> str:
        if not self.fields:
            return f"{self.enum_name}::{self.tag}" if self.enum_name else self.tag
        payload = ", ".join(repr(f) for f in self.fields)
        prefix = f"{self.enum_name}::" if self.enum_name else ""
        return f"{prefix}{self.tag}({payload})"


@dataclass
class MxClosure:
    """A closure: a named MirFunc plus a captured environment dict."""
    func_name: str
    captured: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Effect handler registry
# ---------------------------------------------------------------------------

class EffectHandler:
    """Registered handler for one effect.

    For stack effects: handler_fn(op_name, args, k) where k must be called exactly
    once and returns immediately (no storage).
    For suspend effects: handler_fn(op_name, args, k) where k is a MxContinuation
    that may be stored and resumed later.
    """
    def __init__(self, effect_name: str, effect_class: str,
                 fn: Callable[[str, list[Any], MxContinuation], Any]) -> None:
        self.effect_name = effect_name
        self.effect_class = effect_class  # "stack" or "suspend"
        self.fn = fn


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------

class InterpError(Exception):
    pass


class _ScopeAbort(BaseException):
    """Raised inside a suspended handle-body thread to tear it down.

    Carries the scope being aborted so nested body threads can cascade the
    unwind across thread boundaries: a body thread that catches an abort for
    an OUTER scope forwards it to its own handler side (which unwinds too)
    and then exits.

    BaseException on purpose: interpreter-level `except Exception` handlers
    must never swallow a teardown in progress.
    """
    def __init__(self, scope: "_EffectScope") -> None:
        super().__init__(f"handle scope {scope.frame_id} aborted")
        self.scope = scope


class _EffectScope:
    """Runtime state for one MIR handle_scope: the delimited boundary.

    The handle body runs on its own thread so that a perform ANYWHERE in the
    delimited context — including deep inside called functions — suspends the
    whole body up to this boundary (the Python call stack between the handle
    body and the perform site simply stays parked on the blocked body thread).

    Messages flow body -> handler over `to_handler`:
      ("perform", op_name, arg_vals, k)  a suspension point was reached
      ("done", value)                    the body finished normally
      ("error", exc)                     the body raised; re-raised handler-side
      ("cascade", scope_abort)           an outer scope's teardown crossed this
                                         boundary; keep unwinding handler-side

    Each perform carries a private reply queue (`k.reply_q`) the body blocks
    on until the handler resumes it (("resume", value)) or aborts it
    (("abort", _ScopeAbort)). Exactly one side runs at a time, so the
    interpreter's shared state never sees true concurrency.
    """
    def __init__(self, frame_id: int) -> None:
        self.frame_id = frame_id
        self.to_handler: SimpleQueue = SimpleQueue()
        self.thread: Optional[threading.Thread] = None
        self.frame: Optional[Dict[str, Any]] = None
        self.pending_k: Optional["_ScopeContinuation"] = None


@dataclass
class _ScopeContinuation:
    """Single-shot continuation for a perform caught by a MIR handle_scope.

    The suspended state is the blocked body thread itself; resume(v) sends v
    to `reply_q`, unblocking the body at the perform site, then waits for the
    scope's next event. Deep-handler semantics: resume() returns the final
    value of the WHOLE delimited body (subsequent performs included).
    """
    scope: _EffectScope
    reply_q: SimpleQueue = field(default_factory=SimpleQueue)
    used: bool = False


class _EffectAbort(Exception):
    """Control exception: a handler case returned WITHOUT calling resume.

    Unwinds to the handle_scope that installed the handler frame; the handle
    expression's value becomes `value` (abort semantics).
    """
    def __init__(self, frame_id: int, value: Any) -> None:
        super().__init__(f"effect abort -> frame {frame_id}")
        self.frame_id = frame_id
        self.value = value


class MirInterpreter:
    def __init__(self) -> None:
        self._funcs: Dict[str, MirFunc] = {}
        self._effect_handlers: Dict[str, EffectHandler] = {}
        self._builtins: Dict[str, Callable[..., Any]] = {}
        # Dynamic handler stack: list of {op_name -> (param_name, handler_func_name)}
        self._handler_stack: List[Dict[str, tuple]] = []
        # Delimited MIR-level handler frames installed by handle_scope ops:
        # each is {"id", "effect", "cases": {op: (param, fn_name)}, "captured"}
        self._mir_handler_frames: List[Dict[str, Any]] = []
        self._next_frame_id: int = 1
        self._register_builtins()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, funcs: Sequence[MirFunc]) -> None:
        for f in funcs:
            self._funcs[f.name] = f

    def register_effect_handler(self, effect_name: str, effect_class: str,
                                  fn: Callable[[str, list[Any], MxContinuation], Any]) -> None:
        self._effect_handlers[effect_name] = EffectHandler(effect_name, effect_class, fn)

    def register_builtin(self, name: str, fn: Callable[..., Any]) -> None:
        self._builtins[name] = fn

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def call(self, func_name: str, args: List[Any]) -> Any:
        f = self._funcs.get(func_name)
        if f is None:
            raise InterpError(f"Unknown function: {func_name!r}")
        env: Dict[str, Any] = {}
        return self._call_func(f, args, env)

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    def _call_func(self, f: MirFunc, args: List[Any], outer_env: Dict[str, Any]) -> Any:
        env: Dict[str, Any] = dict(outer_env)
        # Bind parameters from first block's params op
        param_names: List[str] = []
        if f.blocks:
            for op in f.blocks[0].ops:
                if op[0] == "params":
                    param_names = list(op[1])
                    break
        for name, val in zip(param_names, args):
            env[name] = val
        return self._run_blocks(f, 0, env)

    def _run_blocks(self, f: MirFunc, start: int, env: Dict[str, Any]) -> Any:
        bi = start
        while True:
            if bi >= len(f.blocks):
                raise InterpError(f"Block index {bi} out of range in {f.name!r}")
            block = f.blocks[bi]
            result = self._run_ops(block.ops, env, f)
            # Process terminator
            term = block.term
            if term[0] == "ret":
                # NOTE: ret keeps a legacy fallback to the last op's value for
                # effect-continuation frames: a resumed MxContinuation re-enters
                # at a block whose ret var may only be bound on the
                # non-suspended path (see stack-effect tests). All other
                # operand lookups are strict.
                if isinstance(term[1], str) and term[1] not in env:
                    return result
                return self._lookup(term[1], env, f)
            elif term[0] == "br":
                bi = term[1]
            elif term[0] == "br_if":
                cond_val = self._lookup(term[1], env, f)
                bi = term[2] if _is_truthy(cond_val) else term[3]
            elif term[0] == "unreachable":
                raise InterpError(f"Reached unreachable block bb{bi} in {f.name!r}")
            else:
                raise InterpError(f"Unknown terminator: {term!r}")

    def _find_mir_frame(self, effect_name: str, op_name: str) -> Optional[Dict[str, Any]]:
        """Innermost MIR handler frame handling op_name (and effect, if named).

        Frames whose handler case is currently executing are skipped: a
        handler body evaluates OUTSIDE its own delimitation, so a perform
        inside it routes to an outer handler of the effect (previously this
        posted to the frame's own queue, which nobody was pumping: deadlock).
        """
        for frame in reversed(self._mir_handler_frames):
            if frame.get("busy"):
                continue
            if op_name not in frame["cases"]:
                continue
            if effect_name and frame["effect"] and frame["effect"] != effect_name:
                continue
            return frame
        return None

    def _pump_scope(self, scope: _EffectScope) -> Any:
        """Handler side of a handle_scope: wait for the delimited body's next
        event and produce the scope's final value.

        Called from handle_scope (initial wait) and from resume() (waiting for
        the body to finish or perform again). Returns the handle result;
        raises _EffectAbort(frame_id) when a handler case declines to resume.
        """
        frame = scope.frame
        msg = scope.to_handler.get()
        kind = msg[0]
        if kind == "done":
            return msg[1]
        if kind == "error":
            raise msg[1]
        if kind == "cascade":
            raise msg[1]  # _ScopeAbort for an outer scope: keep unwinding
        # ("perform", op_name, arg_vals, k)
        _, op_name, arg_vals, sk = msg
        case_params, handler_fn_name = frame["cases"][op_name]
        if isinstance(case_params, str):
            case_params = (case_params,)
        target = self._funcs.get(handler_fn_name)
        if target is None:
            raise InterpError(f"Missing handler function {handler_fn_name!r}")
        handler_env = dict(frame["captured"])
        # Bind the op's arguments positionally to the case parameters,
        # padding with UNIT when the perform supplied fewer.
        handler_args = list(arg_vals[:len(case_params)])
        handler_args += [UNIT] * (len(case_params) - len(handler_args))
        frame["busy"] = True
        try:
            handler_result = self._call_func(target, [*handler_args, sk], handler_env)
        finally:
            frame["busy"] = False
        if sk.used:
            # The handler resumed: resume() pumped the body to completion, so
            # handler_result already reflects the whole delimited body's value.
            return handler_result
        # The handler returned without resuming: abort the handle scope with
        # the handler's value.
        raise _EffectAbort(frame["id"], handler_result)

    def _abort_scope(self, scope: _EffectScope) -> None:
        """Tear down a scope's body thread if it is still suspended.

        No-op when the body already finished. Otherwise the pending perform's
        reply queue gets an abort message; the body thread raises _ScopeAbort
        at the perform site, unwinds (cascading through any nested scopes),
        and exits without ever running the suspended post-perform code.
        """
        t = scope.thread
        if t is None or not t.is_alive():
            return
        sk = scope.pending_k
        scope.pending_k = None
        if sk is not None and not sk.used:
            sk.used = True
            sk.reply_q.put(("abort", _ScopeAbort(scope)))
        # Wait for the full unwind so handler frames are cleaned up before the
        # code after the handle expression continues. The timeout is a safety
        # valve (the thread is a daemon) — it should never trip in practice.
        t.join(timeout=10.0)

    def _lookup(self, a: Any, env: Dict[str, Any], f: MirFunc) -> Any:
        """Resolve an operand: strings are variable names (must be bound);
        anything else is an immediate value."""
        if not isinstance(a, str):
            return a
        if a in env:
            return env[a]
        raise InterpError(f"Unbound variable {a!r} in {f.name!r} (bad lowering or use-after-drop)")

    def _run_ops(self, ops: List[tuple], env: Dict[str, Any], f: MirFunc) -> Any:
        last: Any = UNIT
        for op in ops:
            tag = op[0]
            if tag == "params":
                continue
            elif tag == "let":
                dst, rhs, args = op[1], op[2], op[3]
                val = self._eval_rhs(rhs, args, env, f)
                env[dst] = val
                last = val
            elif tag == "drop":
                # Remove the binding; any subsequent use raises via _lookup.
                name = op[1]
                env.pop(name, None)
            elif tag == "match_fail":
                detail = op[1] if len(op) > 1 else "no pattern matched"
                raise InterpError(f"match failure in {f.name!r}: {detail}")
            elif tag == "perform":
                # ("perform", result_dst, effect_name, op_name, arg_names, resume_block, resume_slot)
                dst = op[1]
                effect_name = op[2]
                op_name = op[3]
                arg_names: tuple = op[4]
                arg_vals = [self._lookup(a, env, f) for a in arg_names]
                resume_block: int = op[5]
                resume_slot: str = op[6]
                # MIR-level handler frames first (innermost handle wins).
                frame = self._find_mir_frame(effect_name, op_name)
                if frame is not None:
                    # We are running on the scope's (possibly indirect) body
                    # thread. Park this whole call stack at the perform site:
                    # ship (op, args, k) to the scope's handler side and block
                    # until it resumes or aborts us. The Python frames between
                    # the handle body and this perform stay suspended right
                    # here, so resume(v) continues the FULL delimited context.
                    scope: _EffectScope = frame["scope"]
                    sk = _ScopeContinuation(scope=scope)
                    scope.pending_k = sk
                    scope.to_handler.put(("perform", op_name, arg_vals, sk))
                    kind, payload = sk.reply_q.get()
                    if kind == "abort":
                        raise payload  # _ScopeAbort: tear down the delimited body
                    # Resumed: the perform expression's value is payload; keep
                    # executing this block (it branches to the resume block,
                    # whose slot is this op's dst).
                    env[dst] = payload
                    last = payload
                    continue
                # The continuation captures the CURRENT env so that after the handler
                # stores it and later calls k.resume(v), the env is correctly seeded.
                k = MxContinuation(func=f, block_idx=resume_block, env=dict(env),
                                    result_slot=resume_slot)
                # Host-registered handlers (tests/embedding): legacy semantics —
                # bind dst to the handler's return and keep running this block.
                handler = self._effect_handlers.get(effect_name)
                if handler is None:
                    raise InterpError(f"No handler for effect {effect_name!r}")
                handler_result = handler.fn(op_name, arg_vals, k)
                # For stack effects the handler called k.resume() inline and returned
                # the final value.  For suspend effects the handler returns a placeholder
                # (e.g. UNIT) and the real value comes back when the scheduler resumes k.
                # Either way, bind dst to whatever the handler returned so that code
                # after the perform op (before the br) can use it.
                env[dst] = handler_result
                last = handler_result
            else:
                raise InterpError(f"Unknown op tag: {tag!r} in {op!r}")
        return last

    def _eval_rhs(self, rhs: tuple, args: tuple, env: Dict[str, Any], f: MirFunc) -> Any:
        kind = rhs[0]
        if kind == "const":
            return rhs[1]
        elif kind == "const_ty":
            return UNIT
        elif kind == "copy":
            # ("copy",), (src,) — bind dst to the value of src (phi/assignment)
            return self._lookup(args[0], env, f)
        elif kind == "call":
            callee_name: str = rhs[1]
            arg_vals = [self._lookup(a, env, f) for a in args]
            # A local bound to a closure value (`let g = fn(y) ...; g(2)`, or a
            # closure received as a parameter) shadows funcs/builtins: call the
            # closure's MirFunc with its captured env seeding the frame.
            local_val = env.get(callee_name)
            if isinstance(local_val, MxClosure):
                target = self._funcs.get(local_val.func_name)
                if target is None:
                    raise InterpError(
                        f"call: no func {local_val.func_name!r} for closure {callee_name!r}")
                return self._call_func(target, arg_vals, local_val.captured)
            # Builtins first
            if callee_name in self._builtins:
                return self._builtins[callee_name](*arg_vals)
            # User functions
            target = self._funcs.get(callee_name)
            if target is None:
                raise InterpError(f"Unknown callee: {callee_name!r}")
            return self._call_func(target, arg_vals, {})
        elif kind == "binop":
            op_name = rhs[1]
            lv = self._lookup(args[0], env, f)
            rv = self._lookup(args[1], env, f)
            return _eval_binop(op_name, lv, rv)
        elif kind == "select":
            # ("select",), (cond, then_val, else_val) — phi merge for if/else
            cond = self._lookup(args[0], env, f)
            return self._lookup(args[1], env, f) if _is_truthy(cond) else self._lookup(args[2], env, f)
        elif kind == "make_variant":
            # ("make_variant", enum_name, variant_name), (field_val_names...)
            enum_name: str = rhs[1]
            variant_name: str = rhs[2]
            payload = tuple(self._lookup(a, env, f) for a in args)
            return MxVariant(enum_name=enum_name, tag=variant_name, fields=payload)
        elif kind == "variant_tag":
            # ("variant_tag",), (variant_name_ref,)
            v = self._lookup(args[0], env, f)
            if not isinstance(v, MxVariant):
                raise InterpError(f"variant_tag: expected MxVariant, got {type(v).__name__!r}")
            return v.tag
        elif kind == "variant_field":
            # ("variant_field", index), (variant_name_ref,)
            idx: int = rhs[1]
            v = self._lookup(args[0], env, f)
            if not isinstance(v, MxVariant):
                raise InterpError(f"variant_field: expected MxVariant, got {type(v).__name__!r}")
            if idx >= len(v.fields):
                raise InterpError(f"variant_field: index {idx} out of range for {v!r}")
            return v.fields[idx]
        elif kind == "handle_scope":
            # ("handle_scope", body_fn, effect_name, ((op, param, handler_fn), ...)),
            # args = ((name, val_name), ...) captured environment
            body_fn_name: str = rhs[1]
            scope_effect: str = rhs[2]
            case_encodings = rhs[3]
            captured: Dict[str, Any] = {}
            for (cname, cval) in args:
                # Non-strict: the lowering conservatively captures every name
                # in its symbol table; some may not be live at runtime.
                if isinstance(cval, str):
                    if cval in env:
                        captured[cname] = env[cval]
                else:
                    captured[cname] = cval
            frame_id = self._next_frame_id
            self._next_frame_id += 1
            scope = _EffectScope(frame_id)
            frame = {
                "id": frame_id,
                "effect": scope_effect,
                "cases": {op_name: (param, hfn) for (op_name, param, hfn) in case_encodings},
                "captured": captured,
                "scope": scope,
            }
            scope.frame = frame
            body_func = self._funcs.get(body_fn_name)
            if body_func is None:
                raise InterpError(f"Missing handle body function {body_fn_name!r}")

            def _body_main(scope: _EffectScope = scope, body_func: MirFunc = body_func,
                           captured: Dict[str, Any] = captured) -> None:
                try:
                    val = self._call_func(body_func, [], dict(captured))
                except _ScopeAbort as sa:
                    if sa.scope is not scope:
                        # An outer scope is tearing down THROUGH this boundary:
                        # forward the abort so our handler side unwinds too.
                        scope.to_handler.put(("cascade", sa))
                    return  # aborted: the suspended body code never completes
                except BaseException as exc:  # noqa: BLE001 — re-raised handler-side
                    scope.to_handler.put(("error", exc))
                    return
                scope.to_handler.put(("done", val))

            scope.thread = threading.Thread(
                target=_body_main, daemon=True,
                name=f"mx-handle-{scope_effect or 'any'}-{frame_id}")
            self._mir_handler_frames.append(frame)
            try:
                scope.thread.start()
                try:
                    return self._pump_scope(scope)
                except _EffectAbort as abort:
                    if abort.frame_id != frame_id:
                        raise
                    # A handler case returned without resuming: its value is
                    # the handle expression's value (abort semantics).
                    return abort.value
            finally:
                self._abort_scope(scope)
                try:
                    self._mir_handler_frames.remove(frame)
                except ValueError:
                    pass
        elif kind == "resume":
            # ("resume",), (k_name, value_name) — consume the single-shot
            # continuation: run the suspended frame from its resume block.
            k = self._lookup(args[0], env, f)
            value = self._lookup(args[1], env, f)
            if isinstance(k, _ScopeContinuation):
                if k.used:
                    raise RuntimeError(
                        "Continuation already consumed (single-shot violation)")
                k.used = True
                scope = k.scope
                scope.pending_k = None
                # Unblock the body thread at its perform site, then wait for
                # the scope's next event. Deep handlers: resume() returns the
                # final value of the whole delimited body, so a subsequent
                # perform is handled (recursively) inside this pump. While the
                # body runs, the frame's delimitation is re-armed (busy off);
                # it re-engages for the post-resume handler code, which
                # evaluates outside its own delimitation.
                frame = scope.frame
                was_busy = bool(frame and frame.get("busy"))
                if frame is not None:
                    frame["busy"] = False
                k.reply_q.put(("resume", value))
                try:
                    return self._pump_scope(scope)
                finally:
                    if frame is not None:
                        frame["busy"] = was_busy
            if not isinstance(k, MxContinuation):
                raise InterpError(f"resume: expected continuation, got {type(k).__name__!r}")
            return k.resume(value, self)
        elif kind == "push_handler":
            # ("push_handler", effect_name), ((op, param_name), ...)
            # Register handler cases on the dynamic stack
            frame: Dict[str, tuple] = {}
            for (op_name, param_name) in args:
                handler_fn_name = f"__handler_{op_name}"
                frame[op_name] = (param_name, handler_fn_name)
            self._handler_stack.append(frame)
            return UNIT
        elif kind == "pop_handler":
            if self._handler_stack:
                self._handler_stack.pop()
            return UNIT
        elif kind == "perform":
            # ("perform", op_name), (arg1, arg2, ...)
            op_name: str = rhs[1]
            perform_args = [self._lookup(a, env, f) for a in args]
            # Search handler stack top-to-bottom
            for frame in reversed(self._handler_stack):
                if op_name in frame:
                    _param_name, handler_fn_name = frame[op_name]
                    target = self._funcs.get(handler_fn_name)
                    if target is not None:
                        return self._call_func(target, perform_args, {})
            # No handler found: raise
            raise InterpError(f"Unhandled effect operation: {op_name!r}")
        elif kind == "alloc_struct":
            # ("alloc_struct", struct_name, locality), ((field, val_name), ...)
            struct_name: str = rhs[1]
            locality: str = rhs[2] if len(rhs) > 2 else "local"
            fields: Dict[str, Any] = {}
            for (fname, fval_name) in args:
                fields[fname] = self._lookup(fval_name, env, f)
            return MxStruct(name=struct_name, fields=fields, locality=locality)
        elif kind == "field_get":
            # ("field_get", field_name), (base_name,)
            field_name: str = rhs[1]
            base = self._lookup(args[0], env, f)
            if not isinstance(base, MxStruct):
                raise InterpError(f"field_get: expected MxStruct, got {type(base).__name__!r}")
            return base.get(field_name)
        elif kind == "field_set":
            # ("field_set", field_name), (base_name, new_val_name)
            field_name = rhs[1]
            base = self._lookup(args[0], env, f)
            new_val = self._lookup(args[1], env, f)
            if not isinstance(base, MxStruct):
                raise InterpError(f"field_set: expected MxStruct, got {type(base).__name__!r}")
            return base.set(field_name, new_val)
        elif kind == "make_closure":
            # ("make_closure", func_name, param_names), ((cap_name, val_name), ...)
            func_name: str = rhs[1]
            captured: Dict[str, Any] = {}
            for (cname, cval_name) in args:
                captured[cname] = self._lookup(cval_name, env, f)
            return MxClosure(func_name=func_name, captured=captured)
        elif kind == "call_closure":
            # ("call_closure",), (closure_name, arg1, arg2, ...)
            closure = self._lookup(args[0], env, f)
            if not isinstance(closure, MxClosure):
                raise InterpError(f"call_closure: expected MxClosure, got {type(closure).__name__!r}")
            target = self._funcs.get(closure.func_name)
            if target is None:
                raise InterpError(f"call_closure: no func {closure.func_name!r}")
            arg_vals = [self._lookup(a, env, f) for a in args[1:]]
            # Inject captured env on top of params
            return self._call_func(target, arg_vals, closure.captured)
        else:
            raise InterpError(f"Unknown rhs kind: {kind!r}")

    # ------------------------------------------------------------------
    # Builtins
    # ------------------------------------------------------------------

    def _register_builtins(self) -> None:
        self._builtins["print"] = lambda *args: (print(*args), UNIT)[1]
        self._builtins["println"] = lambda *args: (print(*args), UNIT)[1]
        self._builtins["assert_eq"] = _builtin_assert_eq
        self._builtins["int_to_str"] = lambda x: str(x)
        self._builtins["neg"] = lambda x: -x
        self._builtins["not"] = lambda x: not x
        # Builtin methods (receiver passed as first argument by HIR)
        self._builtins["to_string"] = lambda x: "()" if x is UNIT else str(x)
        self._builtins["len"] = lambda x: len(x)
        self._builtins["assert"] = _builtin_assert


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_truthy(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val != 0
    if isinstance(val, MxUnit):
        return False
    return bool(val)


_BINOPS: Dict[str, Callable[[Any, Any], Any]] = {
    "+":  lambda a, b: a + b,
    "-":  lambda a, b: a - b,
    "*":  lambda a, b: a * b,
    "/":  lambda a, b: a // b if isinstance(a, int) and isinstance(b, int) else a / b,
    "%":  lambda a, b: a % b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<":  lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">":  lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "&&": lambda a, b: bool(a) and bool(b),
    "||": lambda a, b: bool(a) or bool(b),
    "and": lambda a, b: bool(a) and bool(b),
    "or":  lambda a, b: bool(a) or bool(b),
}


def _eval_binop(op: str, lv: Any, rv: Any) -> Any:
    fn = _BINOPS.get(op)
    if fn is None:
        raise InterpError(f"Unknown binary operator: {op!r}")
    return fn(lv, rv)


def _builtin_assert_eq(a: Any, b: Any) -> Any:
    if a != b:
        raise AssertionError(f"assert_eq failed: {a!r} != {b!r}")
    return UNIT


def _builtin_assert(cond: Any, *msg: Any) -> Any:
    if not cond:
        raise AssertionError(f"assert failed{': ' + ' '.join(str(m) for m in msg) if msg else ''}")
    return UNIT
