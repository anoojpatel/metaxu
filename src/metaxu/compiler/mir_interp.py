"""MIR interpreter for Metaxu.

Executes MirFunc/MirBlock ops directly, providing:
- Arithmetic and comparison binary ops
- Control flow: br, br_if, ret
- Function calls (user-defined + builtins)
- Effect perform/handle using single-shot continuations (stack and suspend classes)
- Drop (no-op for now; ownership semantics validated at check time)

Usage:
    interp = MirInterpreter()
    interp.load(mir_funcs)
    result = interp.call("main", [])
"""
from __future__ import annotations

from dataclasses import dataclass, field
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


class MirInterpreter:
    def __init__(self) -> None:
        self._funcs: Dict[str, MirFunc] = {}
        self._effect_handlers: Dict[str, EffectHandler] = {}
        self._builtins: Dict[str, Callable[..., Any]] = {}
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
                return env.get(term[1], result)
            elif term[0] == "br":
                bi = term[1]
            elif term[0] == "br_if":
                cond_val = env.get(term[1], False)
                bi = term[2] if _is_truthy(cond_val) else term[3]
            else:
                raise InterpError(f"Unknown terminator: {term!r}")

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
                name = op[1]
                env.pop(name, None)
            elif tag == "perform":
                # ("perform", result_dst, effect_name, op_name, arg_names, resume_block, resume_slot)
                dst = op[1]
                effect_name = op[2]
                op_name = op[3]
                arg_names: tuple = op[4]
                arg_vals = [env.get(a, a) for a in arg_names]
                handler = self._effect_handlers.get(effect_name)
                if handler is None:
                    raise InterpError(f"No handler for effect {effect_name!r}")
                # Build single-shot continuation.
                # When resumed with value v, execution continues from resume_block with
                # resume_slot bound to v in the env.  The result returned from resume()
                # becomes the overall call result for suspend effects.
                resume_block: int = op[5]
                resume_slot: str = op[6]
                # The continuation captures the CURRENT env so that after the handler
                # stores it and later calls k.resume(v), the env is correctly seeded.
                k = MxContinuation(func=f, block_idx=resume_block, env=dict(env),
                                    result_slot=resume_slot)
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
        elif kind == "call":
            callee_name: str = rhs[1]
            arg_vals = [env.get(a, a) for a in args]
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
            lv = env.get(args[0], args[0])
            rv = env.get(args[1], args[1])
            return _eval_binop(op_name, lv, rv)
        elif kind == "alloc_struct":
            # ("alloc_struct", struct_name, locality), ((field, val_name), ...)
            struct_name: str = rhs[1]
            locality: str = rhs[2] if len(rhs) > 2 else "local"
            fields: Dict[str, Any] = {}
            for (fname, fval_name) in args:
                fields[fname] = env.get(fval_name, fval_name)
            return MxStruct(name=struct_name, fields=fields, locality=locality)
        elif kind == "field_get":
            # ("field_get", field_name), (base_name,)
            field_name: str = rhs[1]
            base = env.get(args[0], args[0])
            if not isinstance(base, MxStruct):
                raise InterpError(f"field_get: expected MxStruct, got {type(base).__name__!r}")
            return base.get(field_name)
        elif kind == "field_set":
            # ("field_set", field_name), (base_name, new_val_name)
            field_name = rhs[1]
            base = env.get(args[0], args[0])
            new_val = env.get(args[1], args[1])
            if not isinstance(base, MxStruct):
                raise InterpError(f"field_set: expected MxStruct, got {type(base).__name__!r}")
            return base.set(field_name, new_val)
        elif kind == "make_closure":
            # ("make_closure", func_name, param_names), ((cap_name, val_name), ...)
            func_name: str = rhs[1]
            captured: Dict[str, Any] = {}
            for (cname, cval_name) in args:
                captured[cname] = env.get(cval_name, cval_name)
            return MxClosure(func_name=func_name, captured=captured)
        elif kind == "call_closure":
            # ("call_closure",), (closure_name, arg1, arg2, ...)
            closure = env.get(args[0], args[0])
            if not isinstance(closure, MxClosure):
                raise InterpError(f"call_closure: expected MxClosure, got {type(closure).__name__!r}")
            target = self._funcs.get(closure.func_name)
            if target is None:
                raise InterpError(f"call_closure: no func {closure.func_name!r}")
            arg_vals = [env.get(a, a) for a in args[1:]]
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
