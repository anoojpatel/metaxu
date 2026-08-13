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

import math
import threading
from dataclasses import dataclass, field
from queue import SimpleQueue
from typing import Any, Callable, Dict, List, Optional, Sequence

from .mir import MirBlock, MirFunc
from .desugar import IMPL_SEP, parse_impl_method_name
from .hir import EFFECT_RUNTIME_CALL_PREFIX, STATIC_CALL_PREFIX, TRAIT_CALL_PREFIX


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


class MxCell:
    """A shared mutable slot backing a mut-captured variable.

    Scalars (int/bool/float/str) have value semantics, so a closure or
    handler frame that captured one by value could assign to it and the
    write was silently lost (the stdlib worked around this with
    one-element Vec "cells"). When MIR lowering sees a sub-function assign
    to an enclosing binding it emits a ``cell_wrap`` op: the enclosing slot
    is boxed into one MxCell whose identity is shared by every environment
    that captures it. Reads auto-deref (see _lookup); writes go through the
    cell (see the "let" op), so mutation is visible in every frame sharing
    the cell — real write-back, not a copy.
    """
    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"<cell {self.value!r}>"


class MxVec:
    """A growable vector (`Vec<T>`): a MUTABLE runtime object.

    Design choice (documented): unlike structs — which have value semantics in
    this interpreter (MxStruct.set returns a new struct) — a Vec deliberately
    has Python-list IDENTITY semantics. `self.elements.push(x)` inside a
    `&mut self` method mutates the one shared list, so the caller observes the
    push even though the enclosing struct was passed by value. This matches
    what programs like examples/10_traits_and_structs.mx expect from Vec.
    """
    __slots__ = ("items",)

    def __init__(self, items: Optional[List[Any]] = None) -> None:
        self.items = items if items is not None else []

    def __len__(self) -> int:
        return len(self.items)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, MxVec) and self.items == other.items

    def __repr__(self) -> str:
        return f"Vec[{', '.join(repr(i) for i in self.items)}]"


@dataclass(frozen=True)
class MxVector:
    """A fixed-size vector value `vector[T, N]`.

    Design choice (documented): represented as an immutable tuple of elements
    with VALUE semantics (like structs) — element-wise arithmetic and slicing
    produce new vectors; there is no in-place mutation. Slices are honest
    copies, not aliasing views. Nested MxVector elements model
    `vector[vector[T,N],M]` matrices.
    """
    elements: tuple

    def __len__(self) -> int:
        return len(self.elements)

    def __repr__(self) -> str:
        return f"vector[{', '.join(repr(e) for e in self.elements)}]"


@dataclass(frozen=True)
class MxPtr:
    """A raw pointer into the interpreter's simulated C heap.

    FFI runtime model (strict): `malloc` returns a fresh MxPtr handle backed
    by a Python bytearray owned by the interpreter; every read/write is
    bounds-checked and freed allocations are poisoned, so a buffer overrun,
    use-after-free or double free is a hard InterpError instead of UB.
    The null pointer is represented as Python None (the `null` literal), so
    `ptr == null` comparisons work structurally.

    readonly=True marks pointers produced by `as_ptr` on immutable data
    (string/vector byte snapshots): writing through them is an InterpError
    rather than a silent write into a snapshot nobody can observe.
    """
    alloc_id: int
    offset: int = 0
    readonly: bool = False

    def __repr__(self) -> str:
        ro = " const" if self.readonly else ""
        off = f"+{self.offset}" if self.offset else ""
        return f"<*heap#{self.alloc_id}{off}{ro}>"


class MxFile:
    """An opaque FILE* handle returned by the fopen shim (real OS file)."""
    __slots__ = ("fp", "path", "closed")

    def __init__(self, fp: Any, path: str) -> None:
        self.fp = fp
        self.path = path
        self.closed = False

    def __repr__(self) -> str:
        state = "closed" if self.closed else "open"
        return f"<*FILE {self.path!r} {state}>"


class MxMutex:
    """Runtime mutex behind the EFFECT_MUTEX_* primitives (effect_mapping.mx).

    The interpreter executes on a single logical thread (spawned "threads"
    run to completion at spawn — see _rt_thread_spawn), so a non-recursive
    mutex has exact semantics: locking a mutex that is already locked can
    never succeed later — it IS a deadlock — and unlocking an unlocked mutex
    is a program error. Both fail loudly rather than no-op.
    """
    __slots__ = ("mutex_id", "locked")

    def __init__(self, mutex_id: int) -> None:
        self.mutex_id = mutex_id
        self.locked = False

    def __repr__(self) -> str:
        state = "locked" if self.locked else "unlocked"
        return f"<Mutex#{self.mutex_id} {state}>"


class MxThread:
    """Runtime thread handle behind EFFECT_SPAWN / EFFECT_JOIN.

    The single-threaded interpreter realizes one legal schedule of real
    thread semantics: the spawned function runs to completion at spawn time
    (as if the child ran immediately and finished before the parent resumed),
    and join returns its stored result. Joining twice is an error (the handle
    is consumed), matching pthread_join.
    """
    __slots__ = ("thread_id", "result", "joined")

    def __init__(self, thread_id: int, result: Any) -> None:
        self.thread_id = thread_id
        self.result = result
        self.joined = False

    def __repr__(self) -> str:
        state = "joined" if self.joined else "done"
        return f"<Thread#{self.thread_id} {state}>"


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
        # Trait impl index built from mangled function names at load():
        # method -> type_name -> {trait_name: func_name}. Dispatch is on the
        # receiver's RUNTIME type name (MxStruct.name / MxVariant.enum_name /
        # scalar type names) — the documented v1 choice: no static receiver
        # types needed, generics dispatch on the head type constructor.
        self._impl_index: Dict[str, Dict[str, Dict[str, str]]] = {}
        # Simulated C heap for the FFI shims: alloc_id -> backing bytes.
        # Freed ids are remembered so use-after-free / double free give a
        # precise diagnostic instead of a generic "wild pointer".
        self._c_heap: Dict[int, bytearray] = {}
        self._c_freed: set[int] = set()
        self._next_alloc_id: int = 1
        # Runtime shims for effect ops mapped via `with SYMBOL` clauses
        # (effect_mapping.mx). Keyed by the declared runtime symbol; a mapped
        # op whose symbol has no shim here fails loudly at perform time.
        self._effect_runtime_shims: Dict[str, Callable[[List[Any]], Any]] = {
            "EFFECT_MUTEX_CREATE": self._rt_mutex_create,
            "EFFECT_MUTEX_LOCK": self._rt_mutex_lock,
            "EFFECT_MUTEX_UNLOCK": self._rt_mutex_unlock,
            "EFFECT_SPAWN": self._rt_thread_spawn,
            "EFFECT_JOIN": self._rt_thread_join,
        }
        self._next_mutex_id: int = 1
        self._next_thread_id: int = 1
        # Module-level constants: initialized by running __module_init (if
        # loaded) before the first entry-point call; read by _lookup as the
        # fallback after frame-local bindings.
        self._globals: Dict[str, Any] = {}
        self._globals_ready: bool = False
        self._register_builtins()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, funcs: Sequence[MirFunc]) -> None:
        for f in funcs:
            self._funcs[f.name] = f
            parsed = parse_impl_method_name(f.name)
            if parsed is not None:
                trait_name, type_name, method = parsed
                by_type = self._impl_index.setdefault(method, {})
                by_type.setdefault(type_name, {})[trait_name] = f.name

    def register_effect_handler(self, effect_name: str, effect_class: str,
                                  fn: Callable[[str, list[Any], MxContinuation], Any]) -> None:
        self._effect_handlers[effect_name] = EffectHandler(effect_name, effect_class, fn)

    def register_builtin(self, name: str, fn: Callable[..., Any]) -> None:
        self._builtins[name] = fn

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def call(self, func_name: str, args: List[Any]) -> Any:
        self._ensure_globals()
        f = self._funcs.get(func_name)
        if f is None:
            raise InterpError(f"Unknown function: {func_name!r}")
        env: Dict[str, Any] = {}
        return self._call_func(f, args, env)

    def _ensure_globals(self) -> None:
        """Run the synthesized __module_init once (module-level `let`
        bindings), publishing its declared names as globals."""
        if self._globals_ready:
            return
        self._globals_ready = True
        init = self._funcs.get("__module_init")
        if init is None:
            return
        out: Dict[str, Any] = {}
        self._call_func(init, [], {}, out_env=out)
        for name in init.globals_decl:
            if name not in out:
                raise InterpError(
                    f"module constant {name!r} was declared but its "
                    "initializer bound nothing (bad lowering)")
            self._globals[name] = out[name]

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    def _call_func(self, f: MirFunc, args: List[Any], outer_env: Dict[str, Any],
                   out_env: Optional[Dict[str, Any]] = None) -> Any:
        env: Dict[str, Any] = dict(outer_env)
        for name, val in zip(f.param_names(), args):
            env[name] = val
        result = self._run_blocks(f, 0, env)
        if out_env is not None:
            out_env.update(env)
            out_env["__params__"] = tuple(f.param_names())
        return result

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
                if (isinstance(term[1], str) and term[1] not in env
                        and term[1] not in self._globals):
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
        # Bind the op's arguments positionally to the case parameters.
        # Zero-arg ops carry a synthesized case parameter, so FEWER args than
        # params is legitimate (pad with UNIT) — but MORE args than params is
        # a program bug that must error, not silently truncate.
        if len(arg_vals) > len(case_params):
            raise InterpError(
                f"Effect op {op_name!r} performed with {len(arg_vals)} "
                f"argument(s) but its handler case declares only "
                f"{len(case_params)} parameter(s)")
        handler_args = list(arg_vals)
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
            v = env[a]
            # Mut-captured slots are boxed (see MxCell): reads auto-deref so
            # the cell is invisible everywhere except capture/write plumbing.
            if isinstance(v, MxCell):
                return v.value
            return v
        if a in self._globals:
            return self._globals[a]
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
                cur = env.get(dst)
                if isinstance(cur, MxCell) and not isinstance(val, MxCell):
                    # Write THROUGH the shared cell so every frame that
                    # captured this binding observes the new value.
                    cur.value = val
                else:
                    env[dst] = val
                last = val
            elif tag == "cell_wrap":
                # ("cell_wrap", slot): box the slot into a shared MxCell
                # (idempotent) so sub-functions capturing it can write back.
                name = op[1]
                if name not in env:
                    raise InterpError(
                        f"cell_wrap: unbound variable {name!r} in {f.name!r}")
                if not isinstance(env[name], MxCell):
                    env[name] = MxCell(env[name])
            elif tag == "drop":
                # Remove the binding; any subsequent use raises via _lookup.
                name = op[1]
                env.pop(name, None)
            elif tag == "promote_matrix":
                # ("promote_matrix", (param_names...)): the named parameters
                # are declared as matrices (vector[vector[T,N],M]). A flat
                # vector of scalars passed there is an Mx1 column; promote it
                # to a real nested vector so downstream matrix code indexes
                # strictly (e.g. mat.matmul(vec): transpose sees [[1],[2],[3]]).
                # Already-nested vectors pass through untouched.
                for pname in op[1]:
                    val = env.get(pname)
                    if isinstance(val, MxVector) and val.elements and all(
                            _is_number(x) for x in val.elements):
                        env[pname] = MxVector(elements=tuple(
                            MxVector(elements=(x,)) for x in val.elements))
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
                    # Runtime-mapped op: `op(...) -> T with SYMBOL` in the
                    # effect declaration compiled to a thunk
                    # __effect_runtime$E$op that invokes the interpreter's
                    # shim for SYMBOL (Mutex/Thread primitives). Handlers in
                    # scope always win (checked above) — effects stay
                    # virtualizable; the mapping is the op's ground
                    # implementation when nothing intercepts it, and it takes
                    # precedence over a declared `= expr` default.
                    runtime_fn = self._funcs.get(
                        f"__effect_runtime{IMPL_SEP}{effect_name}{IMPL_SEP}{op_name}")
                    if runtime_fn is not None:
                        n_params = len(runtime_fn.param_names())
                        if len(arg_vals) > n_params:
                            raise InterpError(
                                f"Effect op {op_name!r} performed with "
                                f"{len(arg_vals)} argument(s) but its runtime "
                                f"mapping declares only {n_params} parameter(s)")
                        rt_val = self._call_func(runtime_fn, arg_vals, {})
                        env[dst] = rt_val
                        last = rt_val
                        continue
                    # Declared default handler: an effect op with a
                    # `= expr` default compiles to __effect_default$E$op.
                    # With no handler in scope the perform evaluates it and
                    # continues with its value (capability-style effects:
                    # e.g. SimdOp answers None -> callers take their scalar
                    # branch). Installed handlers always take precedence
                    # (checked above); effects with no default still error.
                    default_fn = self._funcs.get(
                        f"__effect_default${effect_name}${op_name}")
                    if default_fn is not None:
                        n_params = len(default_fn.param_names())
                        if len(arg_vals) > n_params:
                            raise InterpError(
                                f"Effect op {op_name!r} performed with "
                                f"{len(arg_vals)} argument(s) but its default "
                                f"handler declares only {n_params} parameter(s)")
                        default_val = self._call_func(default_fn, arg_vals, {})
                        env[dst] = default_val
                        last = default_val
                        continue
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
            # Struct arguments are passed by reference for mutation purposes:
            # after the callee completes, any struct param it rebound (via
            # `self.field = ...` / `list.field = ...`, which the borrow
            # checker only admits through @mut-capable bindings) is written
            # back to the caller's slot. Without this, methods like
            # Stack.push updated a private copy and the mutation silently
            # vanished (Vec already has identity semantics; see MxVec).
            final_env: Dict[str, Any] = {}
            # Trait method call: dispatch on the receiver's runtime type.
            if callee_name.startswith(TRAIT_CALL_PREFIX):
                result = self._dispatch_trait_call(
                    callee_name[len(TRAIT_CALL_PREFIX):], arg_vals,
                    out_env=final_env)
                self._write_back_struct_args(args, arg_vals, final_env, env)
                return result
            # Static impl-method call: `Type.method(args)` — resolved by the
            # (type, method) pair, no receiver involved.
            if callee_name.startswith(STATIC_CALL_PREFIX):
                type_name, _, method = callee_name[len(STATIC_CALL_PREFIX):].partition(IMPL_SEP)
                result = self._dispatch_static_call(type_name, method, arg_vals,
                                                    out_env=final_env)
                self._write_back_struct_args(args, arg_vals, final_env, env)
                return result
            # A local bound to a closure value (`let g = fn(y) ...; g(2)`, or a
            # closure received as a parameter) shadows funcs/builtins: call the
            # closure's MirFunc with its captured env seeding the frame.
            local_val = env.get(callee_name)
            if local_val is None:
                # A module constant bound to a closure is callable too.
                local_val = self._globals.get(callee_name)
            if isinstance(local_val, MxCell):
                local_val = local_val.value
            if isinstance(local_val, MxClosure):
                target = self._funcs.get(local_val.func_name)
                if target is None:
                    raise InterpError(
                        f"call: no func {local_val.func_name!r} for closure {callee_name!r}")
                return self._call_func(target, arg_vals, local_val.captured)
            # Runtime primitive behind a `with SYMBOL` effect mapping: the
            # __effect_runtime$E$op thunk's body calls
            # __mx_effect_runtime$SYMBOL. Dispatch to the shim table; an
            # unmapped symbol is a loud error, never a silent no-op.
            if callee_name.startswith(EFFECT_RUNTIME_CALL_PREFIX):
                symbol = callee_name[len(EFFECT_RUNTIME_CALL_PREFIX):]
                shim = self._effect_runtime_shims.get(symbol)
                if shim is None:
                    raise InterpError(
                        f"effect op is mapped to runtime primitive {symbol!r}, "
                        f"but this interpreter provides no shim for it "
                        f"(available: {', '.join(sorted(self._effect_runtime_shims))})")
                return shim(arg_vals)
            # Builtins first
            if callee_name in self._builtins:
                return self._builtins[callee_name](*arg_vals)
            # User functions
            target = self._funcs.get(callee_name)
            if target is None:
                raise InterpError(f"Unknown callee: {callee_name!r}")
            result = self._call_func(target, arg_vals, {}, out_env=final_env)
            self._write_back_struct_args(args, arg_vals, final_env, env)
            return result
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
            # ("variant_field", index[, ctor_name]), (variant_name_ref,)
            # The optional third element (the pattern's ctor name, added for
            # variant-aware backends) is deliberately ignored here: the
            # interpreter reads the field positionally, and legacy
            # two-element ops stay valid.
            idx: int = rhs[1]
            v = self._lookup(args[0], env, f)
            if not isinstance(v, MxVariant):
                raise InterpError(f"variant_field: expected MxVariant, got {type(v).__name__!r}")
            if idx >= len(v.fields):
                raise InterpError(f"variant_field: index {idx} out of range for {v!r}")
            return v.fields[idx]
        elif kind == "try_scope":
            # ("try_scope", body_fn, catch_fn) — delimited dynamic error
            # recovery (docs/try_catch.md): run body; on InterpError anywhere
            # in its extent, the catch subfunction receives the failure
            # message and its value becomes the try expression's value.
            # Scope-teardown control exceptions (_ScopeAbort) pass through.
            body_fn = self._funcs.get(rhs[1])
            catch_fn = self._funcs.get(rhs[2])
            if body_fn is None or catch_fn is None:
                raise InterpError(f"Missing try/catch function {rhs[1]!r}/{rhs[2]!r}")
            captured: Dict[str, Any] = {}
            for (cname, cval) in args:
                if isinstance(cval, str):
                    if cval in env:
                        captured[cname] = env[cval]
                else:
                    captured[cname] = cval
            try:
                return self._call_func(body_fn, [], captured)
            except InterpError as exc:
                return self._call_func(catch_fn, [str(exc)], captured)
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
                # Raw (non-deref) capture: a cell-wrapped slot must be
                # captured as the CELL so the closure aliases the binding.
                if isinstance(cval_name, str):
                    if cval_name not in env:
                        raise InterpError(
                            f"Unbound variable {cval_name!r} in {f.name!r} "
                            "(bad lowering or use-after-drop)")
                    captured[cname] = env[cval_name]
                else:
                    captured[cname] = cval_name
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

    def _write_back_struct_args(self, arg_slots: tuple, arg_vals: List[Any],
                                final_env: Dict[str, Any],
                                caller_env: Dict[str, Any]) -> None:
        """Propagate struct mutations from a completed callee to the caller.

        For each argument that was an MxStruct, if the callee's final binding
        of the corresponding parameter is a *different* struct value (the
        callee rebound it, i.e. assigned through it), the caller's argument
        slot is updated. Non-struct args and untouched params are left alone,
        preserving value semantics everywhere else.
        """
        pnames = final_env.get("__params__")
        if not pnames:
            return
        for slot, pname, passed in zip(arg_slots, pnames, arg_vals):
            if not isinstance(passed, MxStruct):
                continue
            newv = final_env.get(pname, passed)
            if isinstance(newv, MxCell):
                newv = newv.value
            if newv is not passed and isinstance(newv, MxStruct):
                cur = caller_env.get(slot)
                if isinstance(cur, MxCell):
                    cur.value = newv
                else:
                    caller_env[slot] = newv

    # ------------------------------------------------------------------
    # Trait method dispatch (runtime, on the receiver's type name)
    # ------------------------------------------------------------------

    def _dispatch_trait_call(self, method: str, arg_vals: List[Any],
                             out_env: Optional[Dict[str, Any]] = None) -> Any:
        """Resolve `recv.method(args)` against loaded __impl$Trait$Type$method
        functions using the receiver's runtime type name.

        Resolution order:
          1. impl function for the receiver's exact type name (then a
             case-insensitive match, so `implement Show for string` finds
             String receivers);
          2. builtin of the same name (to_string/len keep working for types
             without a user impl);
          3. plain user function of the same name;
          4. clear InterpError (unimplemented / ambiguous).
        """
        by_type = self._impl_index.get(method)
        recv_ty: Optional[str] = None
        if arg_vals:
            recv_ty = _runtime_type_name(arg_vals[0])
            if by_type is not None:
                traits = by_type.get(recv_ty)
                if traits is None:
                    low = recv_ty.lower()
                    traits = next(
                        (t for k, t in by_type.items() if k.lower() == low), None)
                if traits:
                    if len(traits) > 1:
                        opts = ", ".join(
                            f"{tr} ({fn})" for tr, fn in sorted(traits.items()))
                        raise InterpError(
                            f"Ambiguous trait method call: {method!r} on type "
                            f"{recv_ty!r} is implemented by multiple traits: {opts}")
                    fname = next(iter(traits.values()))
                    return self._call_func(self._funcs[fname], arg_vals, {},
                                           out_env=out_env)
        # Fallbacks: builtin method, then a plain function of the same name.
        if method in self._builtins:
            return self._builtins[method](*arg_vals)
        target = self._funcs.get(method)
        if target is not None:
            return self._call_func(target, arg_vals, {}, out_env=out_env)
        if by_type:
            impl_types = ", ".join(sorted(by_type))
            raise InterpError(
                f"Trait method {method!r} is not implemented for type "
                f"{recv_ty!r} (implementations exist for: {impl_types})")
        raise InterpError(
            f"Trait method {method!r} has no implementation for any type "
            f"(receiver type: {recv_ty!r})")

    def _dispatch_static_call(self, type_name: str, method: str,
                              arg_vals: List[Any],
                              out_env: Optional[Dict[str, Any]] = None) -> Any:
        """Resolve `Type.method(args)` against __impl$Trait$Type$method funcs."""
        traits = self._impl_index.get(method, {}).get(type_name)
        if traits:
            if len(traits) > 1:
                opts = ", ".join(f"{tr} ({fn})" for tr, fn in sorted(traits.items()))
                raise InterpError(
                    f"Ambiguous static method call: {method!r} on type "
                    f"{type_name!r} is implemented by multiple traits: {opts}")
            fname = next(iter(traits.values()))
            return self._call_func(self._funcs[fname], arg_vals, {},
                                   out_env=out_env)
        # No impl provides it: fall back to a plain dotted function or
        # builtin (`Vec.new`) before giving up, so a type with impls keeps
        # access to same-named non-impl entry points.
        dotted = f"{type_name}.{method}"
        if dotted in self._funcs:
            return self._call_func(self._funcs[dotted], arg_vals, {},
                                   out_env=out_env)
        if dotted in self._builtins:
            return self._builtins[dotted](*arg_vals)
        raise InterpError(
            f"No implementation of method {method!r} for type {type_name!r}")

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
        self._builtins["len"] = _builtin_len
        self._builtins["assert"] = _builtin_assert
        # --- Runtime library: Vec (growable, mutable; see MxVec) ------------
        self._builtins["Vec.new"] = lambda: MxVec()
        self._builtins["push"] = _builtin_push
        self._builtins["pop"] = _builtin_pop
        # --- Runtime library: math methods on numbers -----------------------
        self._builtins["sqrt"] = _make_math_method("sqrt", math.sqrt)
        self._builtins["sin"] = _make_math_method("sin", math.sin)
        self._builtins["cos"] = _make_math_method("cos", math.cos)
        # --- Runtime library: indexing / slicing / fixed-size vectors -------
        self._builtins["__index_get"] = _builtin_index_get
        self._builtins["__index_set"] = _builtin_index_set
        self._builtins["__index_store"] = _builtin_index_store
        self._builtins["__zip"] = _builtin_zip
        self._builtins["__slice_get"] = _builtin_slice_get
        self._builtins["__range"] = _builtin_range
        self._builtins["__vec_dim"] = _builtin_vec_dim
        self._builtins["__cast"] = _builtin_cast
        self._builtins["__vec_lit"] = _builtin_vec_lit
        self._builtins["__vec_zeros"] = _builtin_vec_zeros
        self._builtins["__vec_filled"] = _builtin_vec_filled
        # Comprehension needs to call back into the interpreter for closures.
        self._builtins["__vec_comprehension"] = self._builtin_vec_comprehension
        # --- FFI shims over a simulated, bounds-checked C heap --------------
        # Extern fns lower to plain calls; these builtins are their runtime.
        # (Builtins are checked before user functions, but the only functions
        # with these bare names come from `extern` declarations — impl methods
        # are mangled to __impl$... and dispatch via __trait$/__static$.)
        self._builtins["malloc"] = self._ffi_malloc
        self._builtins["free"] = self._ffi_free
        self._builtins["memcpy"] = self._ffi_memcpy
        self._builtins["realloc"] = self._ffi_realloc
        self._builtins["ptr_read"] = self._ffi_ptr_read
        self._builtins["ptr_write"] = self._ffi_ptr_write
        self._builtins["as_ptr"] = self._ffi_as_ptr
        self._builtins["fopen"] = self._ffi_fopen
        self._builtins["fclose"] = self._ffi_fclose

    # ------------------------------------------------------------------
    # FFI shims (simulated C heap; strict bounds/lifetime checking)
    # ------------------------------------------------------------------

    def _heap_buf(self, ptr: Any, what: str, *, write: bool = False) -> bytearray:
        """Resolve an MxPtr to its live backing buffer, or raise InterpError."""
        if ptr is None:
            raise InterpError(f"{what}: null pointer dereference")
        if not isinstance(ptr, MxPtr):
            raise InterpError(
                f"{what}: expected a pointer, got {_runtime_type_name(ptr)!r}")
        if ptr.alloc_id in self._c_freed:
            raise InterpError(f"{what}: use after free ({ptr!r})")
        buf = self._c_heap.get(ptr.alloc_id)
        if buf is None:
            raise InterpError(f"{what}: wild pointer ({ptr!r})")
        if write and ptr.readonly:
            raise InterpError(f"{what}: write through read-only pointer ({ptr!r})")
        return buf

    def _heap_range(self, ptr: MxPtr, n: int, what: str, *, write: bool = False) -> tuple[bytearray, int]:
        """Bounds-check [ptr.offset, ptr.offset + n) and return (buf, start)."""
        buf = self._heap_buf(ptr, what, write=write)
        start = ptr.offset
        if start < 0 or n < 0 or start + n > len(buf):
            raise InterpError(
                f"{what}: out of bounds — [{start}, {start + n}) outside "
                f"allocation of {len(buf)} bytes ({ptr!r})")
        return buf, start

    def _c_alloc(self, data: bytearray, readonly: bool = False) -> MxPtr:
        aid = self._next_alloc_id
        self._next_alloc_id += 1
        self._c_heap[aid] = data
        return MxPtr(alloc_id=aid, offset=0, readonly=readonly)

    def _ffi_malloc(self, size: Any) -> MxPtr:
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise InterpError(
                f"malloc: size must be a non-negative integer, got {size!r}")
        return self._c_alloc(bytearray(size))

    def _ffi_free(self, ptr: Any) -> Any:
        if ptr is None:
            return UNIT  # free(NULL) is a documented no-op in C
        if not isinstance(ptr, MxPtr):
            raise InterpError(
                f"free: expected a pointer, got {_runtime_type_name(ptr)!r}")
        if ptr.alloc_id in self._c_freed:
            raise InterpError(f"free: double free ({ptr!r})")
        if ptr.alloc_id not in self._c_heap:
            raise InterpError(f"free: wild pointer ({ptr!r})")
        if ptr.offset != 0:
            raise InterpError(
                f"free: pointer does not point to the start of an allocation ({ptr!r})")
        del self._c_heap[ptr.alloc_id]
        self._c_freed.add(ptr.alloc_id)
        return UNIT

    def _src_bytes(self, src: Any, n: int, what: str) -> bytes:
        """Read n bytes from a memcpy-style source (pointer or string)."""
        if isinstance(src, str):
            data = src.encode("utf-8")
            if n > len(data):
                raise InterpError(
                    f"{what}: out of bounds — reading {n} bytes from a "
                    f"{len(data)}-byte string source")
            return bytes(data[:n])
        buf, start = self._heap_range(src, n, what)
        return bytes(buf[start:start + n])

    def _ffi_memcpy(self, dest: Any, src: Any, n: Any) -> Any:
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise InterpError(
                f"memcpy: byte count must be a non-negative integer, got {n!r}")
        data = self._src_bytes(src, n, "memcpy")
        dbuf, dstart = self._heap_range(dest, n, "memcpy", write=True)
        dbuf[dstart:dstart + n] = data
        return dest

    def _ffi_realloc(self, ptr: Any, size: Any) -> MxPtr:
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise InterpError(
                f"realloc: size must be a non-negative integer, got {size!r}")
        if ptr is None:
            return self._ffi_malloc(size)  # realloc(NULL, n) == malloc(n)
        buf = self._heap_buf(ptr, "realloc", write=True)
        if ptr.offset != 0:
            raise InterpError(
                f"realloc: pointer does not point to the start of an allocation ({ptr!r})")
        new = bytearray(size)
        keep = min(size, len(buf))
        new[:keep] = buf[:keep]
        # The old allocation is invalidated (strict: stale pointers poison).
        del self._c_heap[ptr.alloc_id]
        self._c_freed.add(ptr.alloc_id)
        return self._c_alloc(new)

    def _ffi_ptr_read(self, ptr: Any, offset: Any) -> int:
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise InterpError(
                f"ptr_read: offset must be a non-negative integer, got {offset!r}")
        buf, start = self._heap_range(
            MxPtr(alloc_id=ptr.alloc_id, offset=ptr.offset + offset,
                  readonly=ptr.readonly) if isinstance(ptr, MxPtr) else ptr,
            1, "ptr_read")
        return buf[start]

    def _ffi_ptr_write(self, ptr: Any, offset: Any, value: Any) -> Any:
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise InterpError(
                f"ptr_write: offset must be a non-negative integer, got {offset!r}")
        if not isinstance(value, int) or isinstance(value, bool) or not (0 <= value <= 255):
            raise InterpError(
                f"ptr_write: value must be a byte (0..255), got {value!r}")
        buf, start = self._heap_range(
            MxPtr(alloc_id=ptr.alloc_id, offset=ptr.offset + offset,
                  readonly=ptr.readonly) if isinstance(ptr, MxPtr) else ptr,
            1, "ptr_write", write=True)
        buf[start] = value
        return UNIT

    def _ffi_as_ptr(self, recv: Any) -> MxPtr:
        """`x.as_ptr()` — a read-only byte snapshot of a string or vector.

        Strings become NUL-terminated UTF-8 (C-string convention, so the
        result feeds fopen/strlen-style consumers); vectors of bytes become
        their raw bytes. Elements outside 0..255 are an error: the source
        surface types these as &[u8].
        """
        if isinstance(recv, str):
            return self._c_alloc(bytearray(recv.encode("utf-8") + b"\x00"),
                                 readonly=True)
        if isinstance(recv, (MxVector, MxVec)):
            items = recv.elements if isinstance(recv, MxVector) else recv.items
            out = bytearray()
            for i, e in enumerate(items):
                if not isinstance(e, int) or isinstance(e, bool) or not (0 <= e <= 255):
                    raise InterpError(
                        f"as_ptr: element {i} is not a byte (0..255): {e!r}")
                out.append(e)
            return self._c_alloc(out, readonly=True)
        raise InterpError(
            f"as_ptr: unsupported receiver type {_runtime_type_name(recv)!r}")

    def _c_string_at(self, ptr: Any, what: str) -> str:
        """Decode a NUL-terminated C string from a pointer (or pass a str)."""
        if isinstance(ptr, str):
            return ptr
        buf = self._heap_buf(ptr, what)
        end = buf.find(b"\x00", ptr.offset)
        if end < 0:
            raise InterpError(
                f"{what}: unterminated C string — no NUL byte before the end "
                f"of the allocation ({ptr!r})")
        return bytes(buf[ptr.offset:end]).decode("utf-8")

    def _ffi_fopen(self, path: Any, mode: Any) -> Any:
        """fopen shim over the real filesystem: returns null on failure."""
        path_s = self._c_string_at(path, "fopen")
        mode_s = self._c_string_at(mode, "fopen")
        if not mode_s or any(ch not in "rwa+bx" for ch in mode_s):
            raise InterpError(f"fopen: unsupported mode {mode_s!r}")
        try:
            fp = open(path_s, mode_s if "b" in mode_s else mode_s + "b")
        except OSError:
            return None  # C fopen returns NULL on failure
        return MxFile(fp, path_s)

    def _ffi_fclose(self, handle: Any) -> int:
        if handle is None:
            raise InterpError("fclose: null FILE handle")
        if not isinstance(handle, MxFile):
            raise InterpError(
                f"fclose: expected a FILE handle, got {_runtime_type_name(handle)!r}")
        if handle.closed:
            raise InterpError(f"fclose: FILE already closed ({handle!r})")
        handle.fp.close()
        handle.closed = True
        return 0

    # ------------------------------------------------------------------
    # Runtime shims for `with SYMBOL`-mapped effect ops (effect_mapping.mx)
    # ------------------------------------------------------------------
    # Single-threaded execution model, stated once: spawn runs the child
    # function to completion immediately (a legal schedule of real thread
    # semantics — child finishes before the parent resumes), so mutex
    # lock/unlock are exact, not simulated: a lock that cannot be acquired
    # NOW can never be acquired (deadlock -> loud error).

    def _rt_mutex_create(self, args: List[Any]) -> Any:
        if args:
            raise InterpError(
                f"EFFECT_MUTEX_CREATE takes no arguments, got {len(args)}")
        m = MxMutex(self._next_mutex_id)
        self._next_mutex_id += 1
        return m

    def _rt_mutex_lock(self, args: List[Any]) -> Any:
        if len(args) != 1 or not isinstance(args[0], MxMutex):
            raise InterpError(
                f"EFFECT_MUTEX_LOCK expects one Mutex argument, got "
                f"{[_runtime_type_name(a) for a in args]!r}")
        m = args[0]
        if m.locked:
            raise InterpError(
                f"deadlock: EFFECT_MUTEX_LOCK on {m!r}, which is already "
                f"locked — in the single-threaded interpreter no other "
                f"thread can ever release it")
        m.locked = True
        return UNIT

    def _rt_mutex_unlock(self, args: List[Any]) -> Any:
        if len(args) != 1 or not isinstance(args[0], MxMutex):
            raise InterpError(
                f"EFFECT_MUTEX_UNLOCK expects one Mutex argument, got "
                f"{[_runtime_type_name(a) for a in args]!r}")
        m = args[0]
        if not m.locked:
            raise InterpError(
                f"EFFECT_MUTEX_UNLOCK on {m!r}, which is not locked")
        m.locked = False
        return UNIT

    def _rt_thread_spawn(self, args: List[Any]) -> Any:
        if len(args) != 1:
            raise InterpError(
                f"EFFECT_SPAWN expects one function argument, got {len(args)}")
        fn = args[0]
        if not isinstance(fn, MxClosure):
            raise InterpError(
                f"EFFECT_SPAWN expects a closure, got "
                f"{_runtime_type_name(fn)!r}")
        target = self._funcs.get(fn.func_name)
        if target is None:
            raise InterpError(f"EFFECT_SPAWN: no func {fn.func_name!r} for closure")
        if target.param_names():
            raise InterpError(
                f"EFFECT_SPAWN: spawned function must take no arguments, "
                f"but {fn.func_name!r} declares {len(target.param_names())}")
        # Run the child to completion now (see execution model note above).
        result = self._call_func(target, [], dict(fn.captured))
        t = MxThread(self._next_thread_id, result)
        self._next_thread_id += 1
        return t

    def _rt_thread_join(self, args: List[Any]) -> Any:
        if len(args) != 1 or not isinstance(args[0], MxThread):
            raise InterpError(
                f"EFFECT_JOIN expects one Thread argument, got "
                f"{[_runtime_type_name(a) for a in args]!r}")
        t = args[0]
        if t.joined:
            raise InterpError(f"EFFECT_JOIN on {t!r}: thread already joined")
        t.joined = True
        return t.result

    def _builtin_vec_comprehension(self, n: Any, fn: Any, iterable: Any) -> Any:
        """Evaluate `vector[T, N](expr for targets in iterable)` at runtime."""
        if isinstance(iterable, MxVector):
            items: Sequence[Any] = iterable.elements
        elif isinstance(iterable, MxVec):
            items = list(iterable.items)
        elif isinstance(iterable, (list, tuple)):
            items = iterable
        else:
            raise InterpError(
                f"vector comprehension: cannot iterate a "
                f"{_runtime_type_name(iterable)!r} value")
        if not isinstance(fn, MxClosure):
            raise InterpError(
                f"vector comprehension: expected a closure body, got "
                f"{_runtime_type_name(fn)!r}")
        target = self._funcs.get(fn.func_name)
        if target is None:
            raise InterpError(
                f"vector comprehension: no func {fn.func_name!r} for closure")
        n_params = len(target.param_names()) or 1
        out: List[Any] = []
        for item in items:
            if n_params > 1:
                if not isinstance(item, (tuple, list)) or len(item) != n_params:
                    raise InterpError(
                        f"vector comprehension: cannot unpack {item!r} into "
                        f"{n_params} targets")
                args = list(item)
            else:
                args = [item]
            out.append(self._call_func(target, args, dict(fn.captured)))
        if n is not None and len(out) != n:
            raise InterpError(
                f"vector comprehension produced {len(out)} elements for a "
                f"vector of size {n}")
        return MxVector(elements=tuple(out))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _runtime_type_name(v: Any) -> str:
    """Type name of a runtime value, as used for trait impl dispatch."""
    if isinstance(v, MxStruct):
        return v.name
    if isinstance(v, MxVariant):
        return v.enum_name
    if isinstance(v, bool):
        return "Bool"
    if isinstance(v, int):
        return "Int"
    if isinstance(v, float):
        return "Float"
    if isinstance(v, str):
        return "String"
    if isinstance(v, MxUnit):
        return "Unit"
    if isinstance(v, MxVec):
        return "Vec"
    if isinstance(v, MxVector):
        # Matches the head type constructor name that `implement ... for
        # vector[T, N]` desugars to, so user impls on vectors dispatch.
        return "vector"
    if isinstance(v, MxPtr):
        return "Ptr"
    if isinstance(v, MxFile):
        return "FILE"
    if isinstance(v, MxMutex):
        return "Mutex"
    if isinstance(v, MxThread):
        return "Thread"
    if isinstance(v, MxClosure):
        return "Closure"
    if v is None:
        return "Null"
    return type(v).__name__


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


_VEC_ELEMENTWISE_OPS = frozenset({"+", "-", "*", "/", "%"})


def _eval_binop(op: str, lv: Any, rv: Any) -> Any:
    # Fixed-size vectors: element-wise arithmetic with scalar broadcasting.
    if op in _VEC_ELEMENTWISE_OPS and (isinstance(lv, MxVector) or isinstance(rv, MxVector)):
        return _vec_elementwise(op, lv, rv)
    fn = _BINOPS.get(op)
    if fn is None:
        raise InterpError(f"Unknown binary operator: {op!r}")
    return fn(lv, rv)


def _vec_elementwise(op: str, lv: Any, rv: Any) -> "MxVector":
    """Element-wise vector arithmetic; a scalar operand broadcasts.

    Recurses through _eval_binop per element, so nested vectors (matrices)
    combine element-wise too and int/int division keeps its `//` semantics.
    """
    if isinstance(lv, MxVector) and isinstance(rv, MxVector):
        if len(lv) != len(rv):
            raise InterpError(
                f"vector size mismatch for {op!r}: {len(lv)} vs {len(rv)}")
        pairs = zip(lv.elements, rv.elements)
    elif isinstance(lv, MxVector):
        pairs = ((e, rv) for e in lv.elements)
    else:
        pairs = ((lv, e) for e in rv.elements)
    return MxVector(elements=tuple(_eval_binop(op, a, b) for (a, b) in pairs))


def _builtin_assert_eq(a: Any, b: Any) -> Any:
    if a != b:
        raise AssertionError(f"assert_eq failed: {a!r} != {b!r}")
    return UNIT


def _builtin_assert(cond: Any, *msg: Any) -> Any:
    if not cond:
        raise AssertionError(f"assert failed{': ' + ' '.join(str(m) for m in msg) if msg else ''}")
    return UNIT


# ---------------------------------------------------------------------------
# Runtime library builtins (Vec, math methods, indexing, fixed-size vectors)
# ---------------------------------------------------------------------------

def _builtin_len(x: Any) -> int:
    if isinstance(x, (MxVec, MxVector, str, list, tuple)):
        return len(x)
    raise InterpError(f"len: unsupported receiver type {_runtime_type_name(x)!r}")


def _builtin_push(recv: Any, *vals: Any) -> Any:
    if not isinstance(recv, MxVec):
        raise InterpError(
            f"push: expected a Vec receiver, got {_runtime_type_name(recv)!r}")
    if len(vals) != 1:
        raise InterpError(f"push: expected exactly 1 value, got {len(vals)}")
    recv.items.append(vals[0])
    return UNIT


def _builtin_pop(recv: Any) -> Any:
    if not isinstance(recv, MxVec):
        raise InterpError(
            f"pop: expected a Vec receiver, got {_runtime_type_name(recv)!r}")
    if not recv.items:
        raise InterpError("pop: Vec is empty")
    return recv.items.pop()


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _make_math_method(name: str, fn: Callable[[float], float]) -> Callable[[Any], float]:
    def method(x: Any) -> float:
        if not _is_number(x):
            raise InterpError(
                f"{name}: expected a number, got {_runtime_type_name(x)!r}")
        try:
            return fn(x)
        except ValueError as exc:
            raise InterpError(f"{name}: domain error for {x!r} ({exc})") from None
    return method


def _index_target(base: Any, what: str) -> Sequence[Any]:
    if isinstance(base, MxVec):
        return base.items
    if isinstance(base, MxVector):
        return base.elements
    if isinstance(base, (list, tuple, str)):
        return base
    raise InterpError(
        f"{what}: cannot index a {_runtime_type_name(base)!r} value")


def _builtin_index_get(base: Any, idx: Any) -> Any:
    seq = _index_target(base, "index")
    if not isinstance(idx, int) or isinstance(idx, bool):
        raise InterpError(f"index: expected an integer index, got {idx!r}")
    if idx < 0 or idx >= len(seq):
        raise InterpError(
            f"index out of bounds: {idx} (length {len(seq)})")
    return seq[idx]


def _builtin_zip(*seqs: Any) -> Any:
    """Lockstep iteration source for zip comprehensions:
    `f(a, b) for (a, b) in (xs, ys)`. Strict: same-length sequences only."""
    if not seqs:
        raise InterpError("zip: expected at least one sequence")
    mats = [_index_target(s, "zip") for s in seqs]
    lengths = sorted({len(m) for m in mats})
    if len(lengths) != 1:
        raise InterpError(
            f"zip: sequences have different lengths {lengths}")
    return [tuple(vals) for vals in zip(*mats)]


def _builtin_index_store(base: Any, idx: Any, value: Any) -> Any:
    """Store-back form of `v[i] = x` for assignable places: returns the
    updated receiver, which the lowering rebinds to the place.

    - MxVec: mutates the one shared vector in place (the returned object is
      the same object, so the rebind is a no-op) — identity semantics.
    - MxVector (fixed `vector[T, N]`): value semantics — a functional update
      producing a new vector, written back to the variable/field, exactly
      like struct field assignment.
    Everything else errors loudly.
    """
    if isinstance(base, MxVec):
        _builtin_index_set(base, idx, value)
        return base
    if isinstance(base, MxVector):
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise InterpError(
                f"index assignment: expected an integer index, got {idx!r}")
        if idx < 0 or idx >= len(base.elements):
            raise InterpError(
                f"index assignment out of bounds: {idx} "
                f"(length {len(base.elements)})")
        elems = list(base.elements)
        elems[idx] = value
        return MxVector(elements=tuple(elems))
    if isinstance(base, str):
        raise InterpError("cannot assign into a string: strings are immutable")
    raise InterpError(
        f"index assignment: cannot assign into a "
        f"{_runtime_type_name(base)!r} value")


def _builtin_index_set(base: Any, idx: Any, value: Any) -> Any:
    """`v[i] = x`: in-place element store on the shared MxVec.

    Only Vec supports it — Vec is the one runtime type with documented
    identity semantics (see MxVec). Everything else is immutable here, and a
    silent no-op store is exactly the bug this builtin replaces, so immutable
    receivers are a hard error.
    """
    if isinstance(base, MxVector):
        raise InterpError(
            "cannot assign into an immutable vector: vector[T, N] values have "
            "value semantics (build a new vector, or use Vec for mutable data)")
    if isinstance(base, str):
        raise InterpError("cannot assign into a string: strings are immutable")
    if not isinstance(base, MxVec):
        raise InterpError(
            f"index assignment: cannot assign into a "
            f"{_runtime_type_name(base)!r} value")
    if not isinstance(idx, int) or isinstance(idx, bool):
        raise InterpError(f"index assignment: expected an integer index, got {idx!r}")
    if idx < 0 or idx >= len(base.items):
        raise InterpError(
            f"index assignment out of bounds: {idx} (length {len(base.items)})")
    base.items[idx] = value
    return UNIT


def _builtin_slice_get(base: Any, start: Any, stop: Any, step: Any) -> Any:
    seq = _index_target(base, "slice")
    for part, label in ((start, "start"), (stop, "stop"), (step, "step")):
        if part is not None and (not isinstance(part, int) or isinstance(part, bool)):
            raise InterpError(f"slice: {label} must be an integer, got {part!r}")
    if step == 0:
        raise InterpError("slice: step must be non-zero")
    out = list(seq[slice(start, stop, step)])
    if isinstance(base, MxVec):
        return MxVec(out)  # honest copy, not an aliasing view
    if isinstance(base, MxVector):
        return MxVector(elements=tuple(out))
    if isinstance(base, str):
        return "".join(out)
    return out


def _builtin_vec_dim(v: Any, dim: Any) -> int:
    """Runtime dimension of a vector receiver for const-generic binding.

    dim 0 is the vector's length. dim 1 is the length of its elements when
    they are vectors (a matrix's column count); a flat vector of scalars is
    a column (Mx1 matrix) under the const-generic matrix embedding, so its
    dim 1 is 1. Anything else errors.
    """
    if not isinstance(v, (MxVector, MxVec)):
        raise InterpError(
            f"__vec_dim: expected a vector receiver, got {_runtime_type_name(v)!r}")
    items = v.elements if isinstance(v, MxVector) else v.items
    if dim == 0:
        return len(items)
    if dim == 1:
        if not items:
            return 0
        first = items[0]
        if isinstance(first, (MxVector, MxVec)):
            return len(first.elements if isinstance(first, MxVector) else first.items)
        if _is_number(first):
            return 1  # flat vector == column matrix
        raise InterpError(
            f"__vec_dim: elements of type {_runtime_type_name(first)!r} "
            "have no second dimension")
    raise InterpError(f"__vec_dim: unsupported dimension {dim!r}")


def _builtin_cast(v: Any, target: Any) -> Any:
    """Runtime semantics of `e as T`.

    Numeric targets convert the representation (int <-> float); every other
    target is a static-level reinterpretation with no runtime effect, so the
    value passes through unchanged (e.g. `f as fn(T,T) -> T`).
    """
    t = str(target)
    if t in ("float", "f32", "f64"):
        if _is_number(v):
            return float(v)
        raise InterpError(
            f"cast: cannot convert {_runtime_type_name(v)!r} to {t}")
    if t in ("int", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"):
        if _is_number(v):
            return int(v)
        raise InterpError(
            f"cast: cannot convert {_runtime_type_name(v)!r} to {t}")
    return v


def _builtin_range(start: Any, end: Any) -> list:
    for part, label in ((start, "start"), (end, "end")):
        if not isinstance(part, int) or isinstance(part, bool):
            raise InterpError(f"range: {label} must be an integer, got {part!r}")
    return list(range(start, end))


def _builtin_vec_lit(n: Any, *elems: Any) -> MxVector:
    if n is not None and len(elems) != n:
        raise InterpError(
            f"vector literal has {len(elems)} elements for a vector of size {n}")
    return MxVector(elements=tuple(elems))


_VEC_ZERO_VALUES = {"float": 0.0, "int": 0}


def _builtin_vec_zeros(n: Any, base_name: Any) -> MxVector:
    if not isinstance(n, int) or isinstance(n, bool):
        raise InterpError(
            "vector literal without elements needs a constant integer size")
    zero = _VEC_ZERO_VALUES.get(str(base_name).lower())
    if zero is None:
        raise InterpError(
            f"cannot zero-initialize a vector of element type {base_name!r}")
    return MxVector(elements=(zero,) * n)


def _builtin_vec_filled(n: Any, value: Any) -> MxVector:
    if not isinstance(n, int) or isinstance(n, bool):
        raise InterpError("vector.filled needs a constant integer size")
    return MxVector(elements=(value,) * n)
