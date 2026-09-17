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
import struct as _structmod
import threading
from dataclasses import dataclass, field
from queue import SimpleQueue
from typing import Any, Callable, Dict, List, Optional, Sequence

from .mir import MirBlock, MirFunc
from .desugar import IMPL_SEP, parse_impl_method_name
from .recursion import current_ceiling, grant_slack, recursion_budget
from .hir import (BUILTIN_CALL_PREFIX, EFFECT_RUNTIME_CALL_PREFIX,
                  STATIC_CALL_PREFIX, TRAIT_CALL_PREFIX,
                  TUPLE_STRUCT_PREFIX as _TUPLE_STRUCT_PREFIX,
                  tuple_field_name as _tuple_field)


# ---------------------------------------------------------------------------
# Recursion budget
# ---------------------------------------------------------------------------
#
# A Metaxu call frame is NOT a Python call frame: the interpreter spends
# `_call_func` -> `_run_blocks` -> `_run_ops` -> `_eval_rhs` -> the next
# `_call_func` per Metaxu call, roughly 4 Python frames for a bare
# recursion and ~24 for a realistic structural one. `recursion.py` carries
# the measurements, the chosen ceiling and the argument for why raising it
# does not turn a clean exception into a segfault.

# Stack size for the threads a `handle` body runs on.  Python frames live on
# the heap, so this is not what bounds Metaxu recursion depth (a 128 KiB
# thread was measured carrying a 20_000-frame Metaxu recursion); it is
# headroom for the C-stack side — the C-recursion guard, the runtime's own
# C frames — so that a handle body is not more fragile than the main thread.
# 16 MiB is 2x the usual 8 MiB `RLIMIT_STACK` main-thread default; thread
# stacks are lazily committed, so nested handles cost address space, not RAM.
HANDLE_THREAD_STACK_BYTES = 16 * 1024 * 1024

# `threading.stack_size()` is a process-global read at thread START, and
# there is no per-Thread stack-size argument.  Rather than set it once at
# import (which would silently resize every unrelated thread the host
# process ever starts), set it, start our thread and put it straight back,
# with a lock so two handle scopes cannot interleave the swap.
_STACK_SIZE_LOCK = threading.Lock()


def _start_with_stack_size(thread: threading.Thread,
                           size: int = HANDLE_THREAD_STACK_BYTES) -> None:
    """Start `thread` with a `size`-byte stack, leaving the global default
    exactly as it was found.

    Falls back to starting the thread unchanged if the platform rejects the
    size (`threading.stack_size` raises ValueError below the platform
    minimum / for a bad multiple, RuntimeError where it is unsupported) —
    a smaller stack is a smaller ceiling, never a wrong answer.
    """
    with _STACK_SIZE_LOCK:
        try:
            previous = threading.stack_size()
            threading.stack_size(size)
        except (ValueError, RuntimeError):
            thread.start()
            return
        try:
            thread.start()
        finally:
            try:
                threading.stack_size(previous)
            except (ValueError, RuntimeError):  # pragma: no cover - defensive
                pass


def _tuple_field_arity(field_name: str) -> "int | None":
    """The arity a tuple field name encodes (`_0of2` -> 2), else None."""
    head, sep, tail = field_name.partition("of")
    if not sep or not tail.isdigit() or not head.startswith("_"):
        return None
    return int(tail) if head[1:].isdigit() else None


# ---------------------------------------------------------------------------
# Runtime values
# ---------------------------------------------------------------------------

def mx_display(v):
    """A value as the user-visible formatter receives it: booleans format
    as their word (1/0) on BOTH engines. Native erases bools to i64, so
    the interpreter formats the same way (print parity, codegen_llvm
    module notes)."""
    if v is True:
        return 1
    if v is False:
        return 0
    return v


def mx_repr(v) -> str:
    """repr for language values inside container reprs (Vec/struct/enum/
    vector): same bool-as-word rule as mx_display."""
    return repr(mx_display(v))


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
        return interp._run_blocks(self.func, self.block_idx, env, resumed=True)


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
            # Tuples are anonymous structs whose field names carry the arity
            # (hir.TUPLE_STRUCT_PREFIX), so an arity-mismatched destructuring
            # lands HERE rather than silently binding a prefix.  Say what
            # actually went wrong instead of naming a synthetic field.
            want = _tuple_field_arity(field_name)
            if self.is_tuple and want is not None:
                raise InterpError(
                    f"tuple arity mismatch: a {want}-element tuple pattern "
                    f"cannot destructure a {self.tuple_arity}-element tuple")
            raise KeyError(f"Struct '{self.name}' has no field '{field_name}'")
        return self.fields[field_name]

    def set(self, field_name: str, value: Any) -> "MxStruct":
        """Return a new struct with the field updated (value semantics)."""
        new_fields = dict(self.fields)
        new_fields[field_name] = value
        return MxStruct(name=self.name, fields=new_fields, locality=self.locality)

    def __repr__(self) -> str:
        n = self.tuple_arity
        if n is not None:
            # A tuple is an anonymous struct (hir.TUPLE_STRUCT_PREFIX), but
            # it must READ like a tuple: `(1, 2)`, not
            # `__tuple2 { _0of2=1, _1of2=2 }`.
            return "(" + ", ".join(
                mx_repr(self.fields[_tuple_field(i, n)]) for i in range(n)) + ")"
        fields_str = ", ".join(f"{k}={mx_repr(v)}" for k, v in self.fields.items())
        return f"{self.name} {{ {fields_str} }}"

    @property
    def is_tuple(self) -> bool:
        """True for the anonymous struct a tuple literal lowers to."""
        return self.tuple_arity is not None

    @property
    def tuple_arity(self) -> "int | None":
        """This value's tuple arity, or None when it is an ordinary struct."""
        if not self.name.startswith(_TUPLE_STRUCT_PREFIX):
            return None
        digits = self.name[len(_TUPLE_STRUCT_PREFIX):]
        if not digits.isdigit():
            return None
        n = int(digits)
        if len(self.fields) != n or any(
                _tuple_field(i, n) not in self.fields for i in range(n)):
            return None
        return n


@dataclass
class MxVariant:
    """A runtime enum variant value: tag + positional payload fields."""
    enum_name: str
    tag: str
    fields: tuple = ()

    def __repr__(self) -> str:
        if not self.fields:
            return f"{self.enum_name}::{self.tag}" if self.enum_name else self.tag
        payload = ", ".join(mx_repr(f) for f in self.fields)
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

    ``contended`` (docs/contention_as_permission.md): set once when this
    vector crosses a REAL spawn boundary (_rt_thread_spawn's marking walk
    over the closure's captures). A mutation (push/pop/index-set) of a
    contended vec on a thread whose logical-thread write permit is 0 (no
    mutex held through the runtime — _ThreadCtx.write_permit) raises
    catchably with the spec's exact wording; reads stay free.
    """
    __slots__ = ("items", "contended")

    def __init__(self, items: Optional[List[Any]] = None) -> None:
        self.items = items if items is not None else []
        self.contended = False

    def __len__(self) -> int:
        return len(self.items)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, MxVec) and self.items == other.items

    def __repr__(self) -> str:
        return f"Vec[{', '.join(mx_repr(i) for i in self.items)}]"


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
        return f"vector[{', '.join(mx_repr(e) for e in self.elements)}]"


@dataclass(frozen=True)
class MxTile:
    """A 2D tile value `Tile[T, R, C]` (docs/gpu_tiles.md, Stage 0).

    The portable-core tile: an immutable row-major block with the shape
    carried on the VALUE here (the interpreter is shape-dynamic and
    raises loud errors; the static story — `tile:<elem>:<R>x<C>` kinds
    and the compile-time shape checker — lives in the compiler).  Every
    op produces a fresh tile: functional semantics until layouts exist
    to make lane-local mutation provably sound (deliberate, documented).
    Elements are uniformly one kind (`ekind`: "int", "f64", "f32" or
    "f16"); mixing is a loud error, never a coercion.  f32 (docs/
    gpu_tiles.md Stage 1d) and f16 (Stage 1f) are TILE element kinds
    only — language scalars stay f64 — stored as the REPRESENTABLE
    double (widened bits), with every arithmetic op rounding its result
    to the element width once.  That representation is what makes them
    bit-exact across all three engines: adding/multiplying two
    representable values in f64 and rounding once IS the correctly
    rounded narrow op (24+24 < 53 for f32, 11+11 < 53 for f16
    significand bits).
    """
    rows: int
    cols: int
    elements: tuple
    ekind: str  # "int" | "f64" | "f32" | "f16"

    def __repr__(self) -> str:
        # Pinned print format, shared byte-for-byte with the native
        # mx_tile_to_str: rows joined by "; ", elements by ", ".
        body = "; ".join(
            ", ".join(repr(self.elements[r * self.cols + c])
                      for c in range(self.cols))
            for r in range(self.rows))
        return f"tile[{self.rows}x{self.cols}]({body})"


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
    """Runtime mutex behind the EFFECT_MUTEX_* primitives
    (docs/threads_runtime.md).

    An owner-tracking wrapper over ``threading.Lock`` mirroring
    PTHREAD_MUTEX_ERRORCHECK under REAL threads (spawned threads run
    concurrently — see _rt_thread_spawn): locking a mutex held by ANOTHER
    Metaxu thread BLOCKS until it is released; locking a mutex this thread
    already holds is a loud deadlock error (EDEADLK); unlocking a mutex
    this thread does not hold — unlocked, or held by someone else (EPERM
    covers both) — is a loud error. Never a silent no-op.

    ``owner`` is the holding LOGICAL Metaxu thread's ``_ThreadCtx`` (not a
    Python thread id): a handle body runs on its own parked Python thread
    but belongs to the same logical thread as the frame that installed the
    handler, so lock/unlock pair up across that seam exactly as they do
    natively, where handle bodies are fibers on the same OS thread.  Only
    the owning thread ever writes ``owner`` while holding ``_lock``, so
    the self-relock check (owner is my ctx) is race-free.
    """
    __slots__ = ("mutex_id", "_lock", "owner")

    def __init__(self, mutex_id: int) -> None:
        self.mutex_id = mutex_id
        self._lock = threading.Lock()
        self.owner: "object | None" = None

    def __repr__(self) -> str:
        state = "locked" if self.owner is not None else "unlocked"
        return f"<Mutex#{self.mutex_id} {state}>"


class MxThread:
    """Runtime thread handle behind EFFECT_SPAWN / EFFECT_JOIN
    (docs/threads_runtime.md).

    A REAL OS thread: the spawned closure starts on its own
    ``threading.Thread`` at spawn time and runs concurrently with the
    spawner; join blocks until it completes and returns its result exactly
    once. Joining twice is a loud error (the handle is consumed), matching
    pthread_join. A child that died is re-raised at join with the child's
    own error. Unjoined handles are detached (daemon threads): program
    termination does not wait for them. Handles are immortal — double-join
    is a flag check, never a use-after-free.
    """
    __slots__ = ("thread_id", "thread", "result", "error", "joined",
                 "_join_lock")

    def __init__(self, thread_id: int) -> None:
        self.thread_id = thread_id
        self.thread: "threading.Thread | None" = None
        self.result: Any = None
        self.error: "BaseException | None" = None
        self.joined = False
        self._join_lock = threading.Lock()

    def consume_join(self) -> bool:
        """Atomically claim the single join; True if this caller got it."""
        with self._join_lock:
            if self.joined:
                return False
            self.joined = True
            return True

    def __repr__(self) -> str:
        state = "joined" if self.joined else "running"
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
    """A run-time error from the MIR interpreter.

    TWO MESSAGES, ON PURPOSE — they have different audiences:

    * ``.message`` is the LANGUAGE-VISIBLE failure text: exactly what the
      raiser wrote, with no compiler context and no host filesystem paths.
      This is the value a user's ``catch e`` binds (docs/try_catch.md §
      "the error value in v1 is the failure's message string"), so it is
      part of the program's observable semantics and must stay stable
      across compilations, machines and backends.  The native backend has
      no try/catch yet (it demotes ``try_scope``); when it grows one it
      must hand the catch binding this same plain text.
    * ``str(exc)`` is the DEVELOPER-FACING diagnostic: ``.message`` plus
      ``.note``, the " [in function 'f' declared at f.mx:3:1]" context
      that ``locate`` attaches.  Tracebacks, the example gate and every
      compiler-side report go through ``str()``, so they keep naming the
      function the failure happened in.

    Granularity note: `location` is the DECLARATION site of the function the
    error happened in, not the individual operation — MIR ops are positional
    tuples with no span field (see mir.MirFunc.location).  The note is worded
    so the location can never be misread as the failing op's line.

    ``args`` is never rewritten: mutating ``args[0]`` is what leaked the
    compiler context (and absolute host paths) into the caught value.
    """
    location = None
    note = ""
    _located = False

    def __init__(self, message: str = "", *rest: Any) -> None:
        super().__init__(message, *rest)
        self.message = message

    def __str__(self) -> str:
        return f"{self.message}{self.note}"

    def locate(self, func_name: str, location) -> None:
        """Attach the innermost enclosing function once (outer frames are
        annotated first-wins, so the deepest frame is the one reported).

        Diagnostic context only: `.message` — the value `catch` binds — is
        deliberately left alone."""
        if self._located:
            return
        self._located = True
        self.location = location
        from metaxu.errors import format_location
        where = (f" declared at {format_location(location)}"
                 if location is not None else "")
        self.note = f" [in function {func_name!r}{where}]"


class RecursionLimitExceeded(InterpError):
    """The interpreter ran out of recursion budget (see recursion.py).

    An `InterpError` so it renders as a Metaxu diagnostic — with the innermost
    function's name and declaration site attached by the usual `locate` path —
    instead of leaking a host `RecursionError` traceback across the language
    boundary.

    NOT catchable by Metaxu `try`/`catch`, on purpose (see docs/try_catch.md):
    this is interpreter resource exhaustion, not a failure the program
    produced.  A catch arm would run with the stack still exhausted, so it
    would either overflow again immediately or "recover" onto a stack that
    can no longer do useful work.  The native backend uses the real machine
    stack and has no recoverable equivalent either, so leaving it uncatchable
    keeps the two backends from diverging on the recovery path.
    """


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
    (("abort", _ScopeAbort)). Exactly one side of a SCOPE runs at a time,
    so a scope's own state never sees concurrency; true concurrency exists
    only between LOGICAL threads (EFFECT_SPAWN), whose handler stacks are
    disjoint per _ThreadCtx and whose shared runtime state is lock-guarded
    (docs/threads_runtime.md).
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


class _ThreadCtx:
    """Per-LOGICAL-Metaxu-thread interpreter state (docs/threads_runtime.md
    § effect-scope isolation).

    Each spawned Metaxu thread owns its OWN handler stacks: a child's
    perform never routes to a handler installed on the spawning thread (a
    handler scope is a delimited continuation rooted in the installing
    thread's stack; crossing threads would suspend a foreign stack).

    A handle-scope BODY thread is NOT a new logical thread: it ADOPTS its
    creator's ctx (see _body_main), so nested handlers, busy flags and
    mutex ownership all behave as one thread of control — mirroring the
    native runtime, where handle bodies are fibers on the same OS thread.
    Identity of this object doubles as the mutex-ownership token
    (MxMutex.owner).
    """
    __slots__ = ("mir_handler_frames", "handler_stack", "write_permit")

    def __init__(self) -> None:
        # Delimited MIR-level handler frames installed by handle_scope ops
        # on THIS logical thread; each is {"id", "effect", "cases", ...}.
        self.mir_handler_frames: List[Dict[str, Any]] = []
        # Dynamic (push_handler/pop_handler) stack, same locality.
        self.handler_stack: List[Dict[str, tuple]] = []
        # Contention permission (docs/contention_as_permission.md): count
        # of mutexes this LOGICAL thread holds through the runtime
        # (_rt_mutex_lock +1 / _rt_mutex_unlock -1). Handle-body threads
        # adopt their creator's ctx, so the permission follows the logical
        # thread exactly like mutex ownership; spawned threads get a fresh
        # ctx and start at 0. Read by the contended-Vec write checks.
        self.write_permit: int = 0


class _EffectAbort(Exception):
    """Control exception: a handler case returned WITHOUT calling resume.

    Unwinds to the handle_scope that installed the handler frame; the handle
    expression's value becomes `value` (abort semantics).
    """
    def __init__(self, frame_id: int, value: Any) -> None:
        super().__init__(f"effect abort -> frame {frame_id}")
        self.frame_id = frame_id
        self.value = value


class _TailResume(Exception):
    """Control exception: a handler case reached a TAIL-position resume.

    compiler/effect_tail.py proved (strictly, and identically for the
    native backend) that the resume's value IS the case's return value
    with nothing after it, so instead of recursing (_pump_scope -> case ->
    resume -> _pump_scope ..., one Python frame chain per element for
    stream-shaped handlers), the resume site marks the continuation
    consumed and unwinds the case's frames back to the _pump_scope that
    dispatched it; the pump sends the resume value and LOOPS for the
    scope's next event at constant depth.  Equivalent to the recursion
    because a tail case returns resume's value unchanged: the body's
    eventual DONE value (deep semantics) reaches the pump identically
    whether propagated back through the deleted frames or read directly
    at the loop — see THE PUMP MODEL in runtime/native/metaxu_effects.c
    for the full argument (this class is the interpreter half of that
    same design).  Not an InterpError: `try` must never catch it (it is
    control flow, not a failure), exactly like _EffectAbort.
    """
    def __init__(self, k: "_ScopeContinuation", value: Any) -> None:
        super().__init__("tail resume")
        self.k = k
        self.value = value


class MirInterpreter:
    def __init__(self) -> None:
        self._funcs: Dict[str, MirFunc] = {}
        # ids of tail-position resume ops (filled by load(); see there).
        self._tail_resume_ids: frozenset = frozenset()
        self._effect_handlers: Dict[str, EffectHandler] = {}
        self._builtins: Dict[str, Callable[..., Any]] = {}
        # Handler stacks are PER LOGICAL METAXU THREAD (_ThreadCtx, reached
        # through the `_mir_handler_frames` / `_handler_stack` properties):
        # spawned threads get a fresh ctx (scope isolation), handle-body
        # threads adopt their creator's. `self._tls` maps each Python
        # thread to its current ctx.
        self._tls = threading.local()
        self._next_frame_id: int = 1
        # Guards the shared id counters (frame/mutex/thread/alloc ids) now
        # that spawned threads run concurrently. The GIL makes single dict
        # and list operations atomic, but `x += 1` is not.
        self._id_lock = threading.Lock()
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
            "EFFECT_METAL_LAUNCH": self._rt_metal_launch,
        }
        self._next_mutex_id: int = 1
        self._next_thread_id: int = 1
        # Module-level constants: initialized by running __module_init (if
        # loaded) before the first entry-point call; read by _lookup as the
        # fallback after frame-local bindings.
        self._globals: Dict[str, Any] = {}
        self._globals_ready: bool = False
        # A failed __module_init is cached here and re-raised on every
        # subsequent entry-point call (never silently skipped).
        self._globals_init_error: Exception | None = None
        self._register_builtins()

    # ------------------------------------------------------------------
    # Per-logical-thread state (docs/threads_runtime.md)
    # ------------------------------------------------------------------

    def _ctx(self) -> _ThreadCtx:
        """The calling Python thread's logical-thread context.

        Lazily created for threads that never had one installed (the main
        thread, embedder threads). Spawned Metaxu threads install a FRESH
        ctx at start (_rt_thread_spawn); handle-body threads install their
        creator's (handle_scope's _body_main)."""
        ctx = getattr(self._tls, "ctx", None)
        if ctx is None:
            ctx = _ThreadCtx()
            self._tls.ctx = ctx
        return ctx

    @property
    def _mir_handler_frames(self) -> List[Dict[str, Any]]:
        """MIR handler frames of the CURRENT logical thread (innermost
        last). A property so the ~10 touch points stay written as before
        while spawned threads each see their own stack."""
        return self._ctx().mir_handler_frames

    @property
    def _handler_stack(self) -> List[Dict[str, tuple]]:
        """Dynamic push/pop handler stack of the current logical thread."""
        return self._ctx().handler_stack

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
        # Tail-position resumes in handler-case functions (shared analysis
        # with the native backend — compiler/effect_tail.py — so both
        # engines trampoline the SAME sites): identities of the `let` op
        # tuples whose resume short-circuits back to _pump_scope's loop
        # via _TailResume instead of recursing.  Recomputed over ALL
        # loaded functions on each load() (a later load can add the
        # handle_scope that makes an earlier function a case).
        from .effect_tail import program_tail_resume_ids
        self._tail_resume_ids = program_tail_resume_ids(self._funcs.values())

    def register_effect_handler(self, effect_name: str, effect_class: str,
                                  fn: Callable[[str, list[Any], MxContinuation], Any]) -> None:
        self._effect_handlers[effect_name] = EffectHandler(effect_name, effect_class, fn)

    def register_builtin(self, name: str, fn: Callable[..., Any]) -> None:
        self._builtins[name] = fn

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def call(self, func_name: str, args: List[Any]) -> Any:
        # The budget is process-global, not per-thread, so handle-body
        # threads AND spawned Metaxu threads started inside this extent
        # inherit the same ceiling (each Python thread counts its own
        # depth against it).
        with recursion_budget():
            self._ensure_globals()
            f = self._funcs.get(func_name)
            if f is None:
                raise InterpError(f"Unknown function: {func_name!r}")
            env: Dict[str, Any] = {}
            try:
                return self._call_func(f, args, env)
            except RecursionError:
                # Belt and braces: `_call_func` converts at the innermost
                # Metaxu frame, but a RecursionError raised OUTSIDE any
                # `_run_blocks` (scope teardown, a builtin's own recursion)
                # must not escape as a host exception either.
                raise self._recursion_exhausted(None) from None

    # ------------------------------------------------------------------
    # Recursion budget
    # ------------------------------------------------------------------

    def _recursion_exhausted(self, f: "MirFunc | None") -> "RecursionLimitExceeded":
        """Turn a host `RecursionError` into a Metaxu diagnostic.

        Grants the slack FIRST: this runs with the stack at the ceiling, and
        building the message, walking `locate`, tearing down handle scopes
        and formatting the report all need frames of their own — without the
        slack the conversion would raise the very error it is converting.
        `recursion.grant_slack` is idempotent and `recursion_budget` discards
        the grant when `call` returns.
        """
        grant_slack()
        where = f" while calling {f.name!r}" if f is not None else ""
        return RecursionLimitExceeded(
            "recursion limit exceeded" + where + ": the interpreter allows "
            f"{current_ceiling()} nested Python frames (a few per Metaxu "
            "call frame). Either the program recursed without a base case, "
            "or it is deeper than the interpreter supports — the native "
            "backend uses the machine stack and has no such ceiling.")

    def _ensure_globals(self) -> None:
        """Run the synthesized __module_init once (module-level `let`
        bindings), publishing its declared names as globals.

        _globals_ready is only set after SUCCESS: a failed initializer is
        cached and re-raised on every subsequent call — running the program
        without its module constants would just misbehave later and blame
        the wrong code."""
        if self._globals_ready:
            return
        if self._globals_init_error is not None:
            raise self._globals_init_error
        init = self._funcs.get("__module_init")
        if init is None:
            self._globals_ready = True
            return
        out: Dict[str, Any] = {}
        try:
            self._call_func(init, [], {}, out_env=out)
            for name in init.globals_decl:
                if name not in out:
                    raise InterpError(
                        f"module constant {name!r} was declared but its "
                        "initializer bound nothing (bad lowering)")
                self._globals[name] = out[name]
        except Exception as exc:
            self._globals_init_error = exc
            raise
        self._globals_ready = True

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    def _call_func(self, f: MirFunc, args: List[Any], outer_env: Dict[str, Any],
                   out_env: Optional[Dict[str, Any]] = None) -> Any:
        env: Dict[str, Any] = dict(outer_env)
        for name, val in zip(f.param_names(), args):
            env[name] = val
        try:
            result = self._run_blocks(f, 0, env)
        except RecursionError:
            # The recursion ceiling, converted at the INNERMOST Metaxu frame
            # that hit it: `RecursionLimitExceeded` is an `InterpError`, so
            # every enclosing frame takes the branch below instead and
            # `locate`'s first-wins rule keeps this function's name. `from
            # None` drops the host chain: a 100_000-frame Python traceback is
            # not a Metaxu diagnostic (and printing one is its own hazard).
            err = self._recursion_exhausted(f)
            err.locate(f.name, getattr(f, "location", None))
            raise err from None
        except InterpError as exc:
            # Name the function the error happened in (innermost wins).
            exc.locate(f.name, getattr(f, "location", None))
            raise
        if out_env is not None:
            out_env.update(env)
            out_env["__params__"] = tuple(f.param_names())
            out_env["__mut_params__"] = tuple(getattr(f, "mut_params", ()) or ())
        return result

    def _run_blocks(self, f: MirFunc, start: int, env: Dict[str, Any],
                    resumed: bool = False) -> Any:
        bi = start
        while True:
            if bi >= len(f.blocks):
                raise InterpError(f"Block index {bi} out of range in {f.name!r}")
            block = f.blocks[bi]
            result = self._run_ops(block.ops, env, f)
            # Process terminator
            term = block.term
            if term[0] == "ret":
                # NOTE: ret keeps a fallback to the last op's value for
                # effect-continuation frames ONLY: a resumed MxContinuation
                # re-enters at a block whose ret var may only be bound on the
                # non-suspended path (see stack-effect tests). Everywhere else
                # the lookup is strict, because returning the previous op's
                # value for an absent slot is a SILENT WRONG ANSWER: it is how
                # `fn inner() -> int { secret }` answered `()` instead of
                # raising `Unbound variable 'secret'`. Ordinary calls enter
                # through `_call_func` with resumed=False, so they are loud.
                if (resumed and isinstance(term[1], str) and term[1] not in env
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

        THE LOOP: a case that ends in a TAIL-position resume (_TailResume,
        see effect_tail.py) hands (k, value) back here instead of recursing;
        this pump sends the value and waits for the scope's next event at
        CONSTANT depth — the interpreter half of the native runtime's
        tail-resume trampoline (metaxu_effects.c, THE PUMP MODEL, where the
        equivalence argument lives).  General (non-tail) resumes keep the
        recursion: the case's resume() re-enters this method in a fresh
        frame, so depth tracks handler-code nesting (fold's pending
        f-applications), not element count.
        """
        frame = scope.frame
        while True:
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
            # Zero-arg ops carry a synthesized case parameter, so FEWER args
            # than params is legitimate (pad with UNIT) — but MORE args than
            # params is a program bug that must error, not silently truncate.
            if len(arg_vals) > len(case_params):
                raise InterpError(
                    f"Effect op {op_name!r} performed with {len(arg_vals)} "
                    f"argument(s) but its handler case declares only "
                    f"{len(case_params)} parameter(s)")
            handler_args = list(arg_vals)
            handler_args += [UNIT] * (len(case_params) - len(handler_args))
            frame["busy"] = True
            tail: Optional[_TailResume] = None
            try:
                handler_result = self._call_func(
                    target, [*handler_args, sk], handler_env)
            except _TailResume as tr:
                tail = tr
            finally:
                frame["busy"] = False
            if tail is not None:
                # The tail resume must consume THIS dispatch's continuation
                # (effect_tail.py only marks resumes of the case's own __k).
                if tail.k is not sk:
                    raise InterpError(
                        "tail resume of a foreign continuation "
                        "(pump invariant violated)")
                # Unblock the body at its perform site and loop for the
                # scope's next event.  The case's value is resume's value
                # (tail position), which is the body's eventual DONE value
                # (deep semantics) — exactly what the next loop iteration
                # returns when the body completes.
                tail.k.reply_q.put(("resume", tail.value))
                continue
            if sk.used:
                # The handler resumed (general form): resume() pumped the
                # body to completion, so handler_result already reflects the
                # whole delimited body's value.
                return handler_result
            # The handler returned without resuming: abort the handle scope
            # with the handler's value.
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
                if rhs and rhs[0] == "resume" and id(op) in self._tail_resume_ids:
                    # TAIL-position resume of the case's own continuation
                    # (effect_tail.py): consume it and unwind to the
                    # dispatching _pump_scope, which sends the value and
                    # loops — no per-element recursion.  Only scope
                    # continuations trampoline; a stack-effect
                    # MxContinuation keeps the general path below.
                    k = self._lookup(args[0], env, f)
                    if isinstance(k, _ScopeContinuation):
                        if k.used:
                            raise RuntimeError(
                                "Continuation already consumed "
                                "(single-shot violation)")
                        k.used = True
                        k.scope.pending_k = None
                        raise _TailResume(k, self._lookup(args[1], env, f))
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
                # origin_name (pre-monomorphization) so interpreting
                # monomorphized MIR raises the SAME message the
                # unspecialized reference run raises — and the native
                # backend can bind the identical string (codegen_llvm
                # emits mx_raise with this exact text).
                fname = f.origin_name or f.name
                raise InterpError(f"match failure in {fname!r}: {detail}")
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
                # Closure calls write back @mut struct params exactly like
                # direct/trait/static calls (mut_params-gated, so plain
                # lambda params still never copy out).
                result = self._call_func(target, arg_vals, local_val.captured,
                                         out_env=final_env)
                self._write_back_struct_args(args, arg_vals, final_env, env)
                return result
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
            # Method-position builtin call (`x.len()` -> __builtin$len):
            # always the builtin, never a same-named plain function — see
            # hir._method_callee / docs/name_precedence.md.
            if callee_name.startswith(BUILTIN_CALL_PREFIX):
                bname = callee_name[len(BUILTIN_CALL_PREFIX):]
                builtin = self._builtins.get(bname)
                if builtin is None:
                    raise InterpError(
                        f"no builtin method {bname!r} (method-position call)")
                return builtin(*arg_vals)
            # NAME PRECEDENCE for plain calls: a user-defined module
            # function WINS over a same-named builtin; the builtin is the
            # fallback when no user function of that name exists.  Names
            # the compiler owns (the __-prefixed intrinsics and the
            # __trait$/__static$/... dispatch prefixes) are rejected as
            # reserved at the front end, so they can never be shadowed.
            target = self._funcs.get(callee_name)
            if target is None:
                builtin = self._builtins.get(callee_name)
                if builtin is not None:
                    return builtin(*arg_vals)
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
            except RecursionLimitExceeded:
                # Interpreter resource exhaustion, not a program failure:
                # a catch arm here would run with the stack still at the
                # ceiling. Propagates past every `try` to the top level.
                raise
            except InterpError as exc:
                # `.message`, NOT `str(exc)`: the catch binding is a
                # LANGUAGE-VISIBLE value, so it gets the plain failure text
                # without the compiler's " [in function 'f' declared at
                # /abs/host/path:2:1]" diagnostic note (see InterpError).
                return self._call_func(catch_fn, [exc.message], captured)
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
            with self._id_lock:
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

            creator_ctx = self._ctx()

            def _body_main(scope: _EffectScope = scope, body_func: MirFunc = body_func,
                           captured: Dict[str, Any] = captured) -> None:
                # The body thread is the SAME logical Metaxu thread as its
                # creator (a parked continuation, not a spawned thread):
                # adopt the creator's handler stacks and mutex-ownership
                # identity (docs/threads_runtime.md).
                self._tls.ctx = creator_ctx
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
                _start_with_stack_size(scope.thread)
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
            # Inject captured env on top of params. @mut struct params write
            # back to the caller exactly like every other call path.
            final_env: Dict[str, Any] = {}
            result = self._call_func(target, arg_vals, closure.captured,
                                     out_env=final_env)
            self._write_back_struct_args(args[1:], arg_vals, final_env, env)
            return result
        else:
            raise InterpError(f"Unknown rhs kind: {kind!r}")

    def _write_back_struct_args(self, arg_slots: tuple, arg_vals: List[Any],
                                final_env: Dict[str, Any],
                                caller_env: Dict[str, Any]) -> None:
        """Propagate struct mutations from a completed callee to the caller.

        Write-back applies ONLY to parameters with by-reference semantics
        (MirFunc.mut_params: declared @mut, or a method's `self` receiver).
        For such a parameter that was passed an MxStruct, if the callee's
        final binding is a *different* struct value (the callee rebound it,
        i.e. assigned through it), the caller's argument slot is updated.
        Plain parameters keep value semantics — a callee rebinding its own
        (non-@mut) param stays local to the callee.
        """
        pnames = final_env.get("__params__")
        if not pnames:
            return
        mut_params = final_env.get("__mut_params__") or ()
        for slot, pname, passed in zip(arg_slots, pnames, arg_vals):
            if pname not in mut_params:
                continue
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
        self._builtins["print"] = lambda *args: (print(*(mx_display(a) for a in args)), UNIT)[1]
        self._builtins["println"] = lambda *args: (print(*(mx_display(a) for a in args)), UNIT)[1]
        self._builtins["assert_eq"] = _builtin_assert_eq
        self._builtins["int_to_str"] = lambda x: str(mx_display(x))
        self._builtins["neg"] = lambda x: -x
        self._builtins["not"] = lambda x: not x
        # `~x`: bitwise complement on i64.  Python's `~` is already
        # two's-complement on unbounded ints, so for an in-range operand it
        # equals the native `xor i64 %x, -1`.
        self._builtins["bnot"] = lambda x: _wrap_i64(~_bit_operand("~", x, "left"))
        # Builtin methods (receiver passed as first argument by HIR)
        self._builtins["to_string"] = lambda x: "()" if x is UNIT else str(mx_display(x))
        self._builtins["len"] = _builtin_len
        self._builtins["assert"] = _builtin_assert
        # --- Runtime library: Tile (docs/gpu_tiles.md Stage 0) --------------
        # Dotted statics only in v1 (the Vec.new resolution path): no
        # method-position names, no collisions with std/user `dot`/`sum`.
        self._builtins["Tile.zeros"] = _tile_zeros
        self._builtins["Tile.filled"] = _tile_filled
        self._builtins["Tile.arange"] = _tile_arange
        self._builtins["Tile.from_vec"] = _tile_from_vec
        self._builtins["Tile.to_vec"] = _tile_to_vec
        self._builtins["Tile.add"] = _tile_add
        self._builtins["Tile.mul"] = _tile_mul
        self._builtins["Tile.scale"] = _tile_scale
        self._builtins["Tile.dot"] = _tile_dot
        self._builtins["Tile.sum"] = _tile_sum
        self._builtins["Tile.transpose"] = _tile_transpose
        self._builtins["Tile.get"] = _tile_get
        self._builtins["Tile.rows"] = _tile_rows
        self._builtins["Tile.cols"] = _tile_cols
        self._builtins["Tile.load"] = _tile_load
        self._builtins["Tile.load_or"] = _tile_load_or
        self._builtins["Tile.to_f32"] = _tile_to_f32
        self._builtins["Tile.to_f16"] = _tile_to_f16
        self._builtins["Tile.to_f64"] = _tile_to_f64
        self._builtins["Tile.load_rows"] = _tile_load_rows
        # --- Runtime library: Vec (growable, mutable; see MxVec) ------------
        self._builtins["Vec.new"] = lambda: MxVec()
        # `[a, b, c]` / `[]` (HIR lowers ListLiteral to this): a fresh Vec,
        # identical to Vec.new() followed by pushes.
        self._builtins["__list_lit"] = lambda *xs: MxVec(list(xs))
        self._builtins["__list_concat"] = _builtin_list_concat

        # Contended-write permission check (docs/contention_as_permission.md):
        # EVERY Vec-mutating builtin (push, pop, index-set, index-store)
        # goes through this guard; reads (len/get/iteration/slicing) stay
        # free. Bound wrappers, because the permission lives on the calling
        # LOGICAL thread's ctx (handle bodies adopt their creator's — the
        # grant follows the logical thread exactly like mutex ownership).
        def _contended_write_check(recv: Any) -> None:
            if (isinstance(recv, MxVec) and recv.contended
                    and self._ctx().write_permit == 0):
                raise InterpError(_CONTENDED_WRITE_MSG)

        def _checked_push(recv: Any, *vals: Any) -> Any:
            _contended_write_check(recv)
            return _builtin_push(recv, *vals)

        def _checked_pop(recv: Any) -> Any:
            _contended_write_check(recv)
            return _builtin_pop(recv)

        def _checked_index_set(base: Any, idx: Any, value: Any) -> Any:
            _contended_write_check(base)
            return _builtin_index_set(base, idx, value)

        def _checked_index_store(base: Any, idx: Any, value: Any) -> Any:
            _contended_write_check(base)
            return _builtin_index_store(base, idx, value)

        self._builtins["push"] = _checked_push
        self._builtins["pop"] = _checked_pop

        # Tile stores are Vec WRITES: same contended-write permission as
        # every other mutating builtin (docs/contention_as_permission.md).
        def _checked_tile_store(v: Any, off: Any, t: Any) -> Any:
            _contended_write_check(v)
            return _tile_store_unchecked(v, off, t)

        def _checked_tile_store_clipped(v: Any, off: Any, t: Any) -> Any:
            _contended_write_check(v)
            return _tile_store_clipped_unchecked(v, off, t)

        def _checked_tile_store_rows(v, off, stride, t):
            _contended_write_check(v)
            return _tile_store_rows_unchecked(v, off, stride, t)

        self._builtins["Tile.store"] = _checked_tile_store
        self._builtins["Tile.store_clipped"] = _checked_tile_store_clipped
        self._builtins["Tile.store_rows"] = _checked_tile_store_rows
        # --- Runtime library: math methods on numbers -----------------------
        self._builtins["sqrt"] = _make_math_method("sqrt", math.sqrt)
        self._builtins["sin"] = _make_math_method("sin", math.sin)
        self._builtins["cos"] = _make_math_method("cos", math.cos)
        # --- Runtime library: indexing / slicing / fixed-size vectors -------
        self._builtins["__index_get"] = _builtin_index_get
        self._builtins["__index_set"] = _checked_index_set
        self._builtins["__index_store"] = _checked_index_store
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
        # (`extern` declarations produce no MIR function, so these bare names
        # normally resolve here; a program that also defines `fn malloc`
        # WINS over the shim, like every other plain call — see
        # docs/name_precedence.md.)
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
        with self._id_lock:
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
        try:
            return bytes(buf[ptr.offset:end]).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InterpError(
                f"{what}: invalid UTF-8 in C string at {ptr!r}: {exc}") from exc

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
    # REAL-THREADS execution model (docs/threads_runtime.md), stated once:
    # spawn starts the closure on its own OS thread immediately and runs
    # it CONCURRENTLY with the spawner; join blocks for completion and
    # returns the result exactly once. Mutexes are ERRORCHECK: locking a
    # mutex held by ANOTHER thread blocks; self-relock is a loud deadlock
    # error; unlocking a mutex this thread does not hold is a loud error.
    # Error message wording is shared byte-for-byte with the native
    # runtime (metaxu_threads.c) — the caught value is language-visible.

    def _rt_mutex_create(self, args: List[Any]) -> Any:
        if args:
            raise InterpError(
                f"EFFECT_MUTEX_CREATE takes no arguments, got {len(args)}")
        with self._id_lock:
            mutex_id = self._next_mutex_id
            self._next_mutex_id += 1
        return MxMutex(mutex_id)

    def _rt_mutex_lock(self, args: List[Any]) -> Any:
        if len(args) != 1 or not isinstance(args[0], MxMutex):
            raise InterpError(
                f"EFFECT_MUTEX_LOCK expects one Mutex argument, got "
                f"{[_runtime_type_name(a) for a in args]!r}")
        m = args[0]
        me = self._ctx()
        # Only this logical thread can have set owner to `me`, so the
        # self-relock check cannot race (PTHREAD_MUTEX_ERRORCHECK EDEADLK).
        if m.owner is me:
            raise InterpError(
                f"deadlock: EFFECT_MUTEX_LOCK on <Mutex#{m.mutex_id}>: "
                f"this thread already holds it")
        m._lock.acquire()  # held by another thread: BLOCK until released
        m.owner = me
        # Lock HELD from here: grant write permission on this LOGICAL
        # thread (docs/contention_as_permission.md). Strictly after the
        # error paths — a failed ERRORCHECK lock must not bump.
        me.write_permit += 1
        return UNIT

    def _rt_mutex_unlock(self, args: List[Any]) -> Any:
        if len(args) != 1 or not isinstance(args[0], MxMutex):
            raise InterpError(
                f"EFFECT_MUTEX_UNLOCK expects one Mutex argument, got "
                f"{[_runtime_type_name(a) for a in args]!r}")
        m = args[0]
        # One unified message for "unlocked" and "held by another thread":
        # ERRORCHECK's EPERM covers both, and the momentary state of a
        # mutex someone else holds is racy to print (docs/threads_runtime.md).
        me = self._ctx()
        if m.owner is not me:
            raise InterpError(
                f"EFFECT_MUTEX_UNLOCK on <Mutex#{m.mutex_id}>: "
                f"this thread does not hold it")
        # Revoke write permission before releasing: strictly after the
        # ownership check, so a failed unlock changes nothing (mirrors
        # metaxu_threads.c mx_mutex_unlock).
        me.write_permit -= 1
        m.owner = None
        m._lock.release()
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
        with self._id_lock:
            thread_id = self._next_thread_id
            self._next_thread_id += 1
        t = MxThread(thread_id)
        captured = dict(fn.captured)
        # Contention marking (docs/contention_as_permission.md § marking
        # rule): a REAL spawn marks every captured Vec contended, recursing
        # through struct fields, STOPPING at Vec elements. This runs only
        # here — a handler-virtualized spawn never reaches this shim — and
        # BEFORE the thread starts, so the marks happen-before the child.
        # The native engine emits the identical walk inside the
        # EFFECT_SPAWN runtime thunk (codegen_llvm); asymmetry is a bug.
        for cap_value in captured.values():
            _mark_contended(cap_value)

        def _child_main() -> None:
            # A spawned thread is a NEW logical thread: fresh handler
            # stacks, so the child never sees the spawner's in-scope
            # handlers (docs/threads_runtime.md § effect-scope isolation).
            self._tls.ctx = _ThreadCtx()
            try:
                t.result = self._call_func(target, [], captured)
            except RecursionError:
                # Same conversion the entry point performs: never let a
                # host RecursionError cross the language boundary, even
                # from a child thread (see MirInterpreter.call).
                t.error = self._recursion_exhausted(target)
            except BaseException as exc:  # noqa: BLE001 — re-raised at join
                # Captured, NOT propagated: a child failure must never
                # unwind into the parent thread; join re-raises it.
                t.error = exc

        thread = threading.Thread(
            target=_child_main, daemon=True,
            name=f"mx-spawn-{thread_id}")
        t.thread = thread
        # Same stack-size discipline as handle-body threads (the child
        # runs arbitrary Metaxu code under the same recursion budget; the
        # budget itself is process-global, so the child inherits the
        # ceiling installed around the entry point).
        _start_with_stack_size(thread)
        return t

    def _rt_thread_join(self, args: List[Any]) -> Any:
        if len(args) != 1 or not isinstance(args[0], MxThread):
            raise InterpError(
                f"EFFECT_JOIN expects one Thread argument, got "
                f"{[_runtime_type_name(a) for a in args]!r}")
        t = args[0]
        if not t.consume_join():
            raise InterpError(
                f"EFFECT_JOIN on <Thread#{t.thread_id} joined>: "
                f"thread already joined")
        if t.thread is not None:
            t.thread.join()
        if t.error is not None:
            # The child's own failure, surfaced on the joining thread —
            # catchable when it was catchable in the child (InterpError),
            # uncatchable otherwise, exactly as if raised here.
            raise t.error
        return t.result

    # ------------------------------------------------------------------
    # The Metal launch shim (docs/gpu_tiles.md — handlers as backends):
    # `std.gpu.run_metal(n, f)` performs Metal.launch, whose runtime
    # symbol lands here.  The closure `f` must be the canonical launch
    # idiom — `fn(pid) -> kernel(pid, buf, buf, ...)` — because the
    # kernel must exist AS A FUNCTION for the MSL emitter; anything else
    # is a loud error, never a silent CPU fallback.
    # ------------------------------------------------------------------

    _METAL_IDIOM = ("the Metal handler launches `fn(pid) -> "
                    "kernel(pid, buffers...)` where `kernel` is a named "
                    "function and every buffer is a captured Vec")

    def _introspect_launch_closure(self, clo: "MxClosure"):
        """(kernel MirFunc, [buffer MxVecs] in kernel-parameter order)."""
        lam = self._funcs.get(clo.func_name)
        if lam is None:
            raise InterpError(
                f"Metal.launch: no func {clo.func_name!r} for closure")
        params = lam.param_names()
        if len(params) != 1:
            raise InterpError(
                f"Metal.launch: the launched closure takes {len(params)} "
                f"parameters, expected exactly the instance id — "
                + self._METAL_IDIOM)
        pid = params[0]
        if len(lam.blocks) != 1:
            raise InterpError(
                "Metal.launch: the launched closure must be a single "
                "kernel call — " + self._METAL_IDIOM)
        copies: Dict[str, str] = {}
        call = None
        for op in lam.blocks[0].ops:
            if op[0] == "params":
                continue
            if op[0] != "let":
                raise InterpError(
                    f"Metal.launch: op {op[0]!r} in the launched closure "
                    "— " + self._METAL_IDIOM)
            _, dst, rhs, aa = op
            if rhs[0] == "copy":
                copies[dst] = aa[0]
            elif rhs[0] in ("const", "const_ty"):
                continue
            elif rhs[0] == "call":
                if call is not None:
                    raise InterpError(
                        "Metal.launch: the launched closure makes more "
                        "than one call — " + self._METAL_IDIOM)
                call = (str(rhs[1]), list(aa))
            else:
                raise InterpError(
                    f"Metal.launch: {rhs[0]!r} in the launched closure "
                    "— " + self._METAL_IDIOM)
        if call is None:
            raise InterpError(
                "Metal.launch: the launched closure calls no kernel — "
                + self._METAL_IDIOM)
        kname, kargs = call

        def root(n: str) -> str:
            seen = set()
            while n in copies and n not in seen:
                seen.add(n)
                n = copies[n]
            return n

        if not kargs or root(kargs[0]) != pid:
            raise InterpError(
                f"Metal.launch: the kernel's first argument must be the "
                f"instance id {pid!r} unchanged — " + self._METAL_IDIOM)
        bufs: List[Any] = []
        for a in kargs[1:]:
            r = root(a)
            if r not in clo.captured:
                raise InterpError(
                    f"Metal.launch: kernel argument {a!r} is not a "
                    "captured value — " + self._METAL_IDIOM)
            v = clo.captured[r]
            if isinstance(v, MxCell):
                v = v.value
            if not isinstance(v, MxVec):
                raise InterpError(
                    f"Metal.launch: kernel buffer argument {a!r} is a "
                    f"{_runtime_type_name(v)!r}, expected a Vec — "
                    + self._METAL_IDIOM)
            bufs.append(v)
        kfunc = self._funcs.get(kname)
        if kfunc is None:
            raise InterpError(f"Metal.launch: kernel {kname!r} is not a "
                              "module function")
        return kfunc, bufs

    def _rt_metal_launch(self, args: List[Any]) -> Any:
        from .emit_msl import MslError, _emit
        from .metal_launch import MetalLaunchError
        from . import metal_launch as _engine

        if len(args) != 2:
            raise InterpError(
                f"EFFECT_METAL_LAUNCH expects (n, f), got {len(args)} "
                "arguments")
        n, f = args
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise InterpError(
                f"Metal.launch: grid size must be a non-negative int, "
                f"got {n!r}")
        if not isinstance(f, MxClosure):
            raise InterpError(
                f"Metal.launch: expected a closure, got "
                f"{_runtime_type_name(f)!r}")
        kfunc, vecs = self._introspect_launch_closure(f)
        # Lowering choice (docs/simdgroup_plan.md): by default a kernel
        # with an 8x8 f32/f16 dot takes the per-simdgroup lowering (the
        # matrix units); METAXU_METAL_LOWERING=thread forces the
        # per-thread one (bit-exact device results for such kernels),
        # =simdgroup forces the other.
        import os
        lowering = os.environ.get("METAXU_METAL_LOWERING", "auto")
        choice = {"auto": None, "thread": False, "simdgroup": True}
        if lowering not in choice:
            raise InterpError(
                f"Metal.launch: METAXU_METAL_LOWERING={lowering!r} is not "
                "one of auto, thread, simdgroup")
        try:
            kern = _emit(kfunc, simdgroup=choice[lowering])
        except MslError as e:
            raise InterpError(
                f"Metal.launch: kernel {kfunc.name!r} is outside the MSL "
                f"subset: {e}") from e
        if len(kern.in_bufs) != len(vecs):
            raise InterpError(
                f"Metal.launch: kernel {kfunc.name!r} declares "
                f"{len(kern.in_bufs)} buffers, the closure passes "
                f"{len(vecs)}")
        # The launch WRITES its output Vecs: same contended-write
        # permission as every mutating builtin, checked before dispatch.
        by_name = dict(zip(kern.in_bufs, vecs))
        for b in kern.out_bufs:
            v = by_name[b]
            if v.contended and self._ctx().write_permit == 0:
                raise InterpError(_CONTENDED_WRITE_MSG)
        buffers = {b: list(v.items) for b, v in by_name.items()}
        try:
            merged = _engine.run(kern, n, buffers)
        except MetalLaunchError as e:
            raise InterpError(f"Metal.launch: {e}") from e
        for b, vals in merged.items():
            by_name[b].items[:] = vals
        return None  # unit

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
    if isinstance(v, MxTile):
        return "Tile"
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


def _int_div(a: int, b: int) -> int:
    """Integer division TRUNCATING TOWARD ZERO (C / LLVM ``sdiv``).

    Python's ``//`` floors, so ``-7 // 2 == -4`` while every backend
    (``codegen_llvm``'s ``sdiv``, ``codegen_clif``'s ``sdiv``) answers
    ``-3``.  The interpreter is the semantics reference, so it must not
    be the odd one out: a program whose result depended on the sign of an
    operand used to compile to two different answers with no diagnostic.
    """
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def _int_mod(a: int, b: int) -> int:
    """Remainder with the sign of the DIVIDEND (C / LLVM ``srem``).

    Paired with :func:`_int_div` so ``(a / b) * b + a % b == a`` holds
    with truncating division, exactly as it does natively.
    """
    return a - _int_div(a, b) * b


def _div(a: Any, b: Any) -> Any:
    if isinstance(a, int) and isinstance(b, int):
        return _int_div(a, b)
    return a / b


def _mod(a: Any, b: Any) -> Any:
    """``%``: srem for ints, ``frem`` (C ``fmod``) for floats.

    Python's float ``%`` also floors (``-7.0 % 5.0 == 3.0``); LLVM's
    ``frem`` truncates like ``fmod`` (``-2.0``).  Same reasoning as
    :func:`_int_mod`.
    """
    if isinstance(a, int) and isinstance(b, int):
        return _int_mod(a, b)
    return math.fmod(a, b)


#: Metaxu's `int` is a signed 64-bit machine integer; Python's is unbounded.
#: Every bitwise result is normalised back into that range so the
#: interpreter and the backends' i64 `and`/`or`/`xor`/`shl`/`ashr` agree bit
#: for bit — without this, `x ^ (x << 13)` (the xorshift `std.random` now
#: uses) grows without bound here and wraps natively, i.e. one program with
#: two answers and no diagnostic.
_I64_MIN = -(2 ** 63)
_I64_MOD = 2 ** 64


def _wrap_i64(v: int) -> int:
    return ((v - _I64_MIN) % _I64_MOD) + _I64_MIN


def _bit_operand(op: str, v: Any, side: str) -> int:
    """An `int` operand for a bitwise operator, or a loud error.

    `bool` is rejected on purpose: `&`/`|`/`^`/`~`/`<<`/`>>` are Int-only
    (the constraint emitter classes them `Int`, so a Bool/Float/String
    operand is a compile error), and accepting `true & 1` here would make
    the interpreter more permissive than the checker.
    """
    if isinstance(v, bool) or not isinstance(v, int):
        raise InterpError(
            f"bitwise {op!r}: {side} operand must be an Int, got "
            f"{_runtime_type_name(v)!r}")
    return v


def _shift_amount(op: str, v: Any) -> int:
    """A shift count in 0..63, or a loud error.

    LLVM's `shl`/`ashr` are POISON for counts outside the bit width and
    Python's `<<` happily builds a 10000-bit integer, so an unchecked shift
    is exactly the interpreter/native divergence this compiler refuses.
    Both sides raise instead (natively: `mx_shift_check` aborts).
    """
    n = _bit_operand(op, v, "right")
    if n < 0 or n >= 64:
        raise InterpError(
            f"shift amount {n} out of range for {op!r} on a 64-bit int "
            "(must be 0..63)")
    return n


def _band(a: Any, b: Any) -> int:
    return _bit_operand("&", a, "left") & _bit_operand("&", b, "right")


def _bor(a: Any, b: Any) -> int:
    return _bit_operand("|", a, "left") | _bit_operand("|", b, "right")


def _bxor(a: Any, b: Any) -> int:
    return _bit_operand("^", a, "left") ^ _bit_operand("^", b, "right")


def _shl(a: Any, b: Any) -> int:
    return _wrap_i64(_bit_operand("<<", a, "left") << _shift_amount("<<", b))


def _shr(a: Any, b: Any) -> int:
    # ARITHMETIC shift right (LLVM `ashr`): Python's `>>` on a negative int
    # already sign-extends, so `-8 >> 1 == -4` on both sides.
    return _bit_operand(">>", a, "left") >> _shift_amount(">>", b)


_BINOPS: Dict[str, Callable[[Any, Any], Any]] = {
    "+":  lambda a, b: a + b,
    "-":  lambda a, b: a - b,
    "*":  lambda a, b: a * b,
    "/":  _div,
    "%":  _mod,
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
    # Bitwise, i64, two's complement (see _wrap_i64 / _shift_amount).
    "&":  _band,
    "|":  _bor,
    "^":  _bxor,
    "<<": _shl,
    ">>": _shr,
}


_VEC_ELEMENTWISE_OPS = frozenset({"+", "-", "*", "/", "%"})


def _eval_binop(op: str, lv: Any, rv: Any) -> Any:
    # Fixed-size vectors: element-wise arithmetic with scalar broadcasting.
    if op in _VEC_ELEMENTWISE_OPS and (isinstance(lv, MxVector) or isinstance(rv, MxVector)):
        return _vec_elementwise(op, lv, rv)
    fn = _BINOPS.get(op)
    if fn is None:
        raise InterpError(f"Unknown binary operator: {op!r}")
    try:
        return fn(lv, rv)
    except TypeError:
        # A checker gap (a value reached an operator at a type the
        # checker never saw) must be a LOUD, catchable Metaxu error, not a
        # host Python exception leaking out of the interpreter.
        raise InterpError(
            f"binary operator {op!r} cannot be applied to "
            f"{_runtime_type_name(lv)} and {_runtime_type_name(rv)}") from None


def _vec_elementwise(op: str, lv: Any, rv: Any) -> "MxVector":
    """Element-wise vector arithmetic; a scalar operand broadcasts.

    Recurses through _eval_binop per element, so nested vectors (matrices)
    combine element-wise too and int/int division keeps its truncating
    (C `sdiv`) semantics.
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


# ---------------------------------------------------------------------------
# Tiles (docs/gpu_tiles.md, Stage 0): the portable-core reference semantics.
#
# All ops are dotted statics (`Tile.dot(a, b)`) — one resolution mechanism
# (the Vec.new path), no method-name collisions with std/user code; method
# sugar can arrive later through an ordinary std trait impl.  Every check
# here is LOUD (InterpError, catchable) and its wording is the contract the
# native runtime must reproduce byte-for-byte where the check is dynamic
# there too (from_vec length, get bounds); checks that are STATIC natively
# (shape/kind agreement — carried in `tile:` kinds) are also enforced at
# compile time by the tile shape checker, so these dynamic forms are the
# interpreter's strict backstop.  Accumulation order is pinned: row-major
# for sum, k = 0..K-1 for dot — the native loops must match so float
# results are bit-identical.
# ---------------------------------------------------------------------------

def _tile_shape(op: str, r: Any, c: Any) -> tuple:
    for name, v in (("rows", r), ("cols", c)):
        if isinstance(v, bool) or not isinstance(v, int):
            raise InterpError(
                f"{op}: {name} must be an integer, got "
                f"{_runtime_type_name(v)!r}")
    if r <= 0 or c <= 0:
        raise InterpError(f"{op}: tile shape must be positive, got {r}x{c}")
    return r, c


def _tile_arg(op: str, t: Any) -> MxTile:
    if not isinstance(t, MxTile):
        raise InterpError(
            f"{op}: expected a Tile, got {_runtime_type_name(t)!r}")
    return t


def _tile_elem_ekind(op: str, x: Any) -> str:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise InterpError(
            f"{op}: tile elements must be int or float, got "
            f"{_runtime_type_name(x)!r}")
    return "f64" if isinstance(x, float) else "int"


def _kname(ekind: str) -> str:
    """Element-kind name in diagnostics ("float" is the language's name
    for f64; "f32"/"f16" name the device-width tile kinds)."""
    return {"int": "int", "f64": "float", "f32": "f32", "f16": "f16"}[ekind]


def _f32(x: float) -> float:
    """Round a double to the nearest f32, returned as the widened double
    (the f32-representable value).  Every f32 tile op rounds through
    this; _F32_PACK round-trips through IEEE binary32 exactly."""
    return _structmod.unpack("f", _structmod.pack("f", x))[0]


def _f16(x: float) -> float:
    """Round a double to the nearest f16 (IEEE binary16, round-to-
    nearest-even — Python's 'e' struct format), returned as the widened
    double.  Every f16 tile op rounds through this; it matches the C
    runtime's (double)(_Float16)x bit for bit."""
    return _structmod.unpack("e", _structmod.pack("e", x))[0]


def _tile_zero(ekind: str) -> Any:
    return 0 if ekind == "int" else 0.0


def _tile_same(op: str, a: MxTile, b: MxTile) -> None:
    if (a.rows, a.cols) != (b.rows, b.cols):
        raise InterpError(f"{op}: shape mismatch: {a.rows}x{a.cols} vs "
                          f"{b.rows}x{b.cols}")
    if a.ekind != b.ekind:
        raise InterpError(f"{op}: element kinds differ "
                          f"({_kname(a.ekind)} vs {_kname(b.ekind)})")


def _tile_zeros(r: Any, c: Any) -> MxTile:
    r, c = _tile_shape("Tile.zeros", r, c)
    return MxTile(r, c, (0.0,) * (r * c), "f64")


def _tile_filled(r: Any, c: Any, x: Any) -> MxTile:
    r, c = _tile_shape("Tile.filled", r, c)
    ek = _tile_elem_ekind("Tile.filled", x)
    return MxTile(r, c, (x,) * (r * c), ek)


def _tile_arange(r: Any, c: Any) -> MxTile:
    r, c = _tile_shape("Tile.arange", r, c)
    return MxTile(r, c, tuple(range(r * c)), "int")


def _tile_from_vec(v: Any, r: Any, c: Any) -> MxTile:
    r, c = _tile_shape("Tile.from_vec", r, c)
    if not isinstance(v, MxVec):
        raise InterpError(f"Tile.from_vec: expected a Vec, got "
                          f"{_runtime_type_name(v)!r}")
    if len(v.items) != r * c:
        raise InterpError(
            f"Tile.from_vec: Vec length {len(v.items)} does not fill "
            f"{r}x{c} (= {r * c} elements)")
    fks = {_tile_elem_ekind("Tile.from_vec", x) for x in v.items}
    if len(fks) > 1:
        raise InterpError(
            "Tile.from_vec: mixed int and float elements in the Vec")
    return MxTile(r, c, tuple(v.items), fks.pop())


def _tile_to_vec(t: Any) -> MxVec:
    t = _tile_arg("Tile.to_vec", t)
    return MxVec(list(t.elements))  # fresh, mutable, row-major


def _tile_add(a: Any, b: Any) -> MxTile:
    a = _tile_arg("Tile.add", a)
    b = _tile_arg("Tile.add", b)
    _tile_same("Tile.add", a, b)
    if a.ekind == "f32":  # per-op rounding (see MxTile docstring)
        elems = tuple(_f32(x + y)
                      for x, y in zip(a.elements, b.elements))
    elif a.ekind == "f16":
        elems = tuple(_f16(x + y)
                      for x, y in zip(a.elements, b.elements))
    else:
        elems = tuple(x + y for x, y in zip(a.elements, b.elements))
    return MxTile(a.rows, a.cols, elems, a.ekind)


def _tile_mul(a: Any, b: Any) -> MxTile:
    a = _tile_arg("Tile.mul", a)
    b = _tile_arg("Tile.mul", b)
    _tile_same("Tile.mul", a, b)
    if a.ekind == "f32":
        elems = tuple(_f32(x * y)
                      for x, y in zip(a.elements, b.elements))
    elif a.ekind == "f16":
        elems = tuple(_f16(x * y)
                      for x, y in zip(a.elements, b.elements))
    else:
        elems = tuple(x * y for x, y in zip(a.elements, b.elements))
    return MxTile(a.rows, a.cols, elems, a.ekind)


def _tile_scale(t: Any, s: Any) -> MxTile:
    t = _tile_arg("Tile.scale", t)
    sk = _tile_elem_ekind("Tile.scale", s)
    if t.ekind == "f32":
        # Language scalars are f64: the factor rounds to f32 first, the
        # multiply rounds per element — matching (float)s in C exactly.
        if sk != "f64":
            raise InterpError(
                f"Tile.scale: scalar kind must match tile elements "
                f"(f32 tile, {_kname(sk)} scalar; f32 tiles scale by "
                "float scalars)")
        sf = _f32(s)
        return MxTile(t.rows, t.cols,
                      tuple(_f32(x * sf) for x in t.elements), "f32")
    if t.ekind == "f16":
        # Same rule as f32: the f64 factor rounds to f16 first, the
        # multiply rounds per element — matching (double)(_Float16)s in C.
        if sk != "f64":
            raise InterpError(
                f"Tile.scale: scalar kind must match tile elements "
                f"(f16 tile, {_kname(sk)} scalar; f16 tiles scale by "
                "float scalars)")
        sf = _f16(s)
        return MxTile(t.rows, t.cols,
                      tuple(_f16(x * sf) for x in t.elements), "f16")
    if sk != t.ekind:
        raise InterpError(
            f"Tile.scale: scalar kind must match tile elements "
            f"({_kname(t.ekind)} tile, {_kname(sk)} scalar)")
    return MxTile(t.rows, t.cols, tuple(x * s for x in t.elements), t.ekind)


def _tile_dot(a: Any, b: Any) -> MxTile:
    a = _tile_arg("Tile.dot", a)
    b = _tile_arg("Tile.dot", b)
    if a.cols != b.rows:
        raise InterpError(
            f"Tile.dot: shape mismatch: {a.rows}x{a.cols} · "
            f"{b.rows}x{b.cols} (inner dims {a.cols} and {b.rows})")
    if a.ekind != b.ekind:
        raise InterpError(f"Tile.dot: element kinds differ "
                          f"({_kname(a.ekind)} vs {_kname(b.ekind)})")
    R, K, C = a.rows, a.cols, b.cols
    out = []
    f32 = a.ekind == "f32"
    f16 = a.ekind == "f16"
    for i in range(R):
        for j in range(C):
            acc = _tile_zero(a.ekind)
            for k in range(K):  # pinned order: k ascending
                x = a.elements[i * K + k] * b.elements[k * C + j]
                if f32:  # round the product, then the accumulation
                    acc = _f32(acc + _f32(x))
                elif f16:
                    acc = _f16(acc + _f16(x))
                else:
                    acc = acc + x
            out.append(acc)
    return MxTile(R, C, tuple(out), a.ekind)


def _tile_sum(t: Any) -> Any:
    t = _tile_arg("Tile.sum", t)
    acc = _tile_zero(t.ekind)
    for x in t.elements:  # pinned order: row-major
        if t.ekind == "f32":
            acc = _f32(acc + x)
        elif t.ekind == "f16":
            acc = _f16(acc + x)
        else:
            acc = acc + x
    return acc  # f32/f16 results reach the language as the widened double


def _tile_transpose(t: Any) -> MxTile:
    t = _tile_arg("Tile.transpose", t)
    out = tuple(t.elements[r * t.cols + c]
                for c in range(t.cols) for r in range(t.rows))
    return MxTile(t.cols, t.rows, out, t.ekind)


def _tile_get(t: Any, i: Any, j: Any) -> Any:
    t = _tile_arg("Tile.get", t)
    for name, v in (("row", i), ("col", j)):
        if isinstance(v, bool) or not isinstance(v, int):
            raise InterpError(
                f"Tile.get: {name} index must be an integer, got "
                f"{_runtime_type_name(v)!r}")
    if not (0 <= i < t.rows and 0 <= j < t.cols):
        raise InterpError(f"Tile.get: index out of bounds: ({i}, {j}) "
                          f"(shape {t.rows}x{t.cols})")
    return t.elements[i * t.cols + j]


# -- f32/f16 conversions (docs/gpu_tiles.md Stage 1d/1f) --------------------
#
# f32 and f16 exist as TILE element kinds only; language scalars stay
# f64.  The conversions are the whole new surface: everything else is the
# existing ops extended to the narrow ekinds.  Elements are stored as the
# REPRESENTABLE double, so widening back (to_f64 / to_f32 of an f16 tile /
# sum / get / store_rows into a Vec) is exact and free (f16 values are
# exactly f32-representable).

def _tile_to_f32(t: Any) -> MxTile:
    t = _tile_arg("Tile.to_f32", t)
    if t.ekind == "f32":
        return MxTile(t.rows, t.cols, t.elements, "f32")
    # f16 elements are f32-representable, so _f32 is an exact widening.
    return MxTile(t.rows, t.cols,
                  tuple(_f32(float(x)) for x in t.elements), "f32")


def _tile_to_f16(t: Any) -> MxTile:
    t = _tile_arg("Tile.to_f16", t)
    if t.ekind == "f16":
        return MxTile(t.rows, t.cols, t.elements, "f16")
    return MxTile(t.rows, t.cols,
                  tuple(_f16(float(x)) for x in t.elements), "f16")


def _tile_to_f64(t: Any) -> MxTile:
    t = _tile_arg("Tile.to_f64", t)
    return MxTile(t.rows, t.cols,
                  tuple(float(x) for x in t.elements), "f64")


def _tile_rows(t: Any) -> int:
    return _tile_arg("Tile.rows", t).rows


def _tile_cols(t: Any) -> int:
    return _tile_arg("Tile.cols", t).cols


# -- Buffer <-> tile boundary (docs/gpu_tiles.md Stage 1) -------------------
#
# Kernels move tiles in and out of Vec buffers at an element offset: the
# strict forms raise on any out-of-range element (host-side discipline);
# the MASKED forms are the kernel-side idiom — kernels cannot raise, so a
# ragged edge reads `other` and writes nothing, semantics pinned HERE
# first per the design doc.  Stores are Vec WRITES and take the
# contended-write guard exactly like push/pop/index-set (the wrapper is
# bound in _register_builtins where the thread ctx lives).

def _tile_off(op: str, off: Any) -> int:
    if isinstance(off, bool) or not isinstance(off, int):
        raise InterpError(f"{op}: offset must be an integer, got "
                          f"{_runtime_type_name(off)!r}")
    return off


def _tile_vec(op: str, v: Any) -> MxVec:
    if not isinstance(v, MxVec):
        raise InterpError(f"{op}: expected a Vec, got "
                          f"{_runtime_type_name(v)!r}")
    return v


def _tile_load(v: Any, off: Any, r: Any, c: Any) -> MxTile:
    r, c = _tile_shape("Tile.load", r, c)
    v = _tile_vec("Tile.load", v)
    off = _tile_off("Tile.load", off)
    n = r * c
    if off < 0 or off + n > len(v.items):
        raise InterpError(
            f"Tile.load: range [{off}, {off + n}) outside Vec length "
            f"{len(v.items)}")
    elems = v.items[off:off + n]
    fks = {_tile_elem_ekind("Tile.load", x) for x in elems}
    if len(fks) > 1:
        raise InterpError("Tile.load: mixed int and float elements "
                          "in the Vec range")
    return MxTile(r, c, tuple(elems), fks.pop())


def _tile_load_or(v: Any, off: Any, r: Any, c: Any, other: Any) -> MxTile:
    r, c = _tile_shape("Tile.load_or", r, c)
    v = _tile_vec("Tile.load_or", v)
    off = _tile_off("Tile.load_or", off)
    fk = _tile_elem_ekind("Tile.load_or", other)
    out = []
    for i in range(r * c):
        j = off + i
        if 0 <= j < len(v.items):
            x = v.items[j]
            if _tile_elem_ekind("Tile.load_or", x) != fk:
                raise InterpError(
                    "Tile.load_or: Vec element kind differs from `other` "
                    f"({_kname(_tile_elem_ekind('Tile.load_or', x))} vs "
                    f"{_kname(fk)})")
            out.append(x)
        else:
            out.append(other)  # masked-out element reads `other`
    return MxTile(r, c, tuple(out), fk)


def _tile_load_rows(v: Any, off: Any, stride: Any, r: Any, c: Any,
                    other: Any) -> MxTile:
    """Masked STRIDED load: element (i, j) reads v[off + i*stride + j],
    out-of-range elements read `other`.  This is the 2D form — a tile of
    a row-major matrix has row stride = the matrix width; the flat
    `Tile.load_or` is the stride == cols special case (a fact the
    benchmark that motivated this op got wrong first: flat loads on a
    matrix silently read the wrong elements, identically on both
    engines, so only a ground-truth check caught it)."""
    r, c = _tile_shape("Tile.load_rows", r, c)
    v = _tile_vec("Tile.load_rows", v)
    off = _tile_off("Tile.load_rows", off)
    stride = _tile_off("Tile.load_rows", stride)
    fk = _tile_elem_ekind("Tile.load_rows", other)
    out = []
    for i in range(r):
        for j in range(c):
            k = off + i * stride + j
            if 0 <= k < len(v.items):
                x = v.items[k]
                if _tile_elem_ekind("Tile.load_rows", x) != fk:
                    raise InterpError(
                        "Tile.load_rows: Vec element kind differs from "
                        f"`other` ({_kname(_tile_elem_ekind('Tile.load_rows', x))} "
                        f"vs {_kname(fk)})")
                out.append(x)
            else:
                out.append(other)
    return MxTile(r, c, tuple(out), fk)


def _tile_store_rows_unchecked(v: Any, off: Any, stride: Any,
                               t: Any) -> Any:
    """Masked STRIDED store: element (i, j) writes v[off + i*stride + j];
    out-of-range elements write nothing (store_clipped's 2D form)."""
    t = _tile_arg("Tile.store_rows", t)
    v = _tile_vec("Tile.store_rows", v)
    off = _tile_off("Tile.store_rows", off)
    stride = _tile_off("Tile.store_rows", stride)
    for i in range(t.rows):
        for j in range(t.cols):
            k = off + i * stride + j
            if 0 <= k < len(v.items):
                v.items[k] = t.elements[i * t.cols + j]
    return UNIT


def _tile_store_unchecked(v: Any, off: Any, t: Any) -> Any:
    t = _tile_arg("Tile.store", t)
    v = _tile_vec("Tile.store", v)
    off = _tile_off("Tile.store", off)
    n = t.rows * t.cols
    if off < 0 or off + n > len(v.items):
        raise InterpError(
            f"Tile.store: range [{off}, {off + n}) outside Vec length "
            f"{len(v.items)}")
    for i, x in enumerate(t.elements):
        v.items[off + i] = x
    return UNIT


def _tile_store_clipped_unchecked(v: Any, off: Any, t: Any) -> Any:
    t = _tile_arg("Tile.store_clipped", t)
    v = _tile_vec("Tile.store_clipped", v)
    off = _tile_off("Tile.store_clipped", off)
    for i, x in enumerate(t.elements):
        j = off + i
        if 0 <= j < len(v.items):  # masked-out element writes nothing
            v.items[j] = x
    return UNIT


# Contention as permission (docs/contention_as_permission.md). Wording is
# language-visible (a `try` binds it) and shared byte-for-byte with the
# native runtime (metaxu_rt.c mx__vec_write_check).
_CONTENDED_WRITE_MSG = (
    "write to contended Vec without a held lock: this value crossed "
    "a thread boundary at spawn; mutate it under a mutex "
    "(std.sync.with_lock) or keep it thread-local")


def _mark_contended(value: Any) -> None:
    """The spec's marking rule, applied to one spawned-closure capture:
    Vec identity -> mark contended (do NOT descend into elements — the
    vec-of-vecs hole is documented and pinned by test, because native code
    cannot enumerate runtime elements and the engines must agree); struct
    values -> recurse fields by layout; mutable-capture cells -> the cell
    aliases the binding, so walk the boxed value (native envs capture the
    cell POINTER and the emitted walk loads through it). Everything else
    (scalars, strings, fixed vectors, enum variants, closures, handles)
    stops, identically on both engines."""
    if isinstance(value, MxVec):
        value.contended = True
    elif isinstance(value, MxStruct):
        for field_value in value.fields.values():
            _mark_contended(field_value)
    elif isinstance(value, MxCell):
        _mark_contended(value.value)


def _builtin_push(recv: Any, *vals: Any) -> Any:
    if not isinstance(recv, MxVec):
        raise InterpError(
            f"push: expected a Vec receiver, got {_runtime_type_name(recv)!r}")
    if len(vals) != 1:
        raise InterpError(f"push: expected exactly 1 value, got {len(vals)}")
    recv.items.append(vals[0])
    return UNIT


def _builtin_list_concat(*parts: Any) -> Any:
    """`[a, ...xs, b]` — join the literal's segments into ONE fresh Vec.

    Spreading anything that is not a list is an error, not a skipped element:
    the whole point of lowering list literals is that nothing vanishes."""
    items: List[Any] = []
    for p in parts:
        if isinstance(p, MxVec):
            items.extend(p.items)
        elif isinstance(p, MxVector):
            items.extend(p.elements)
        else:
            raise InterpError(
                "list literal: cannot spread a "
                f"{_runtime_type_name(p)!r} (expected a list)")
    return MxVec(items)


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
