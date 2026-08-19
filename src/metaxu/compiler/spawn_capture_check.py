"""Thread-safety mode enforcement at spawn boundaries.

`perform Thread.spawn(f)` (docs/threads_runtime.md) runs the closure `f` on
a REAL OS thread.  The spawning frame keeps executing — and may return —
while the child is still running, so anything the closure captures crosses
a thread boundary and outlives the spawn site's dynamic extent.  The
declared signature (`fn spawn[T](f: fn() -> @global T)`) promises that, but
until this pass nothing enforced it (the spec recorded the gap explicitly:
"declared, not enforced").

This pass enforces the soundly-checkable subset, at compile time:

1. **No `@local` captures** (kind ``locality-spawn-capture``): a closure
   passed to a spawn-mapped operation must not capture a variable whose
   DECLARED locality is `@local`.  A `@local` value lives in the spawning
   frame's stack region, and that frame may return while the spawned thread
   still runs, leaving the capture dangling.
2. **No `@mut`-borrow captures** (kind ``borrow-spawn-capture``): the
   closure must not capture a binding that holds an active `@mut`/`&mut`
   borrow (`let r = &mut x`), nor capture a variable while such a borrow of
   it is live in scope.  An exclusive borrow shared with another thread is
   aliasing-XOR-mutation broken by construction.

What identifies a spawn is the RUNTIME SYMBOL, not the effect or operation
spelling: any operation declared `with EFFECT_SPAWN` is a spawn, whatever
the user named the effect.  The check is deliberately syntactic on the
spawn-mapped op — it applies even when a `handle` in scope overrides the
runtime mapping (a virtualized spawn that never actually threads still
promises thread-compatibility by its signature; see docs/threads_runtime.md
§ Modes).

Explicitly allowed, and pinned by tests: capturing Mutex/Thread handles
(opaque runtime words), Vec and other shared-identity values (the
mutex-counter pattern), and plain copied scalars/strings/structs — all of
which are `@global` (or default-global) bindings.

Why this pass runs on the MUTABLE, post-desugar AST
---------------------------------------------------------------------------
The frozen AST drops a `PerformEffect`'s ARGUMENTS entirely (its frozen
node has no children), so the spawned closure — and therefore its captures
— is invisible to the frozen constraint emitter at the perform site.  This
is the same lossiness that forced `name_resolution.py` off the frozen tree,
and this pass follows that precedent exactly: it walks the mutable
post-desugar AST from `pipeline.build_context_from_source` and files
structured diagnostics on the same channel (-2) the frozen checkers use.
The kinds carry no ``type-`` prefix, so `run_pipeline` promotes them to
`BorrowCheckError` like every other borrow/locality diagnostic.

Deliberately NOT covered (documented in docs/threads_runtime.md § Modes):
locality is the binding's DECLARED mode (unannotated bindings default to
global, exactly as `frozen_borrow_checker.declare_variable` treats them),
so a `@local` value aliased through an unannotated rebinding is not
tracked; the unqualified-call spelling of an effect op (`spawn(f)` without
`perform`) is not matched; and a closure reaching a spawn through a data
structure or a function return is not traced.  A sound subset, enforced
loudly, beats an unsound guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import metaxu.metaxu_ast as fast
import metaxu.unsafe_ast as uast

from .frozen_borrow_checker import BorrowError
from .mutaxu_ast import _mode_value
from .name_resolution import (_TYPE_ATTRS, _case_parts, _as_list, _is_node,
                              _module_statements, _sub_nodes, _walk_all)

#: The runtime symbol that makes an effect operation a spawn.  Matching is
#: on this symbol — never on the effect/op names, which users may rename.
SPAWN_RUNTIME_SYMBOL = "EFFECT_SPAWN"

#: Diagnostic kind for rule 1 (@local capture crossing a thread boundary).
LOCAL_CAPTURE_KIND = "locality-spawn-capture"

#: Diagnostic kind for rule 2 (active @mut borrow crossing a thread boundary).
MUT_BORROW_CAPTURE_KIND = "borrow-spawn-capture"

#: Diagnostic kind for a rejected explicit escape: `let @global g = v` where
#: `v` is @local and the checker cannot verify the type crosses modes.
LOCALITY_ESCAPE_KIND = "locality-escape"

#: Diagnostic kind for rule 3 (shared mutable identity crossing a thread
#: boundary un-protected) — docs/separate_send_sync.md.
SEPARATE_CAPTURE_KIND = "separate-spawn-capture"

#: Runtime symbols whose perform results are opaque runtime handles,
#: separate by construction (the runtime serializes all access).
_HANDLE_RUNTIME_SYMBOLS = frozenset({
    "EFFECT_SPAWN", "EFFECT_MUTEX_CREATE",
})

#: std.sync constructors whose results are separate by construction.
_PROTECT_CALLEES = frozenset({"protect", "sync.protect", "std.sync.protect"})

_LOCALITY_NAMES = frozenset({"local", "global"})
_MUT_BORROW_MODES = frozenset({"mut", "unique", "exclusive"})

#: Declared type names whose values provably contain no reference into any
#: frame region, so a @local value of such a type may CROSS to @global
#: (OCaml's "mode crossing" on immediates). Strings are included: string
#: values are pointers to immutable heap/constant data on both engines,
#: never into a frame.
_CROSSING_TYPE_NAMES = frozenset({"int", "bool", "float", "string", "str",
                                  "Int", "Bool", "Float", "String"})


@dataclass
class _Binding:
    """What the checker knows about one in-scope name."""
    locality: str = "global"          # declared OR INFERRED locality
    lambda_node: Any | None = None    # LambdaExpression bound by `let f = fn ...`
    mut_borrow_of: str | None = None  # `let r = &mut x` -> "x"
    #: Rule B provenance: how this name came to be @local, innermost first
    #: — a chain of (alias_name, source_name) steps ending at the name that
    #: was DECLARED @local. Empty for a declared-@local binding itself.
    provenance: tuple[tuple[str, str], ...] = ()
    #: Syntactic mode-crossing evidence: True when the value is known (by
    #: literal shape, scalar arithmetic, or a scalar type annotation) to
    #: contain no frame references, so it may be explicitly re-bound
    #: @global even if @local.
    crosses: bool = False
    #: Separateness (docs/separate_send_sync.md): "separate" (safe to share
    #: across threads), "shared" (known shared mutable identity — Vec),
    #: or "unknown" (unclassifiable; deliberately NOT rejected).
    separateness: str = "unknown"
    #: Rule-B-style chain for HOW the name came to be shared.
    shared_provenance: tuple[tuple[str, str], ...] = ()


def _declared_locality(mode: Any) -> str | None:
    """Explicit locality from a surface mode annotation; None when the
    binding carries no locality annotation (the tri-state matters: an
    EXPLICIT @global on a @local initializer is a checked escape, while an
    unannotated binding INHERITS the initializer's locality — Rule B)."""
    tokens = _mode_value(mode)
    if tokens is None:
        return None
    if not isinstance(tokens, (list, tuple)):
        tokens = [tokens]
    for token in tokens:
        if isinstance(token, str) and token.lower().lstrip("@") in _LOCALITY_NAMES:
            return token.lower().lstrip("@")
    return None


def _annotation_crosses(type_annotation: Any) -> bool:
    """A declared scalar type is crossing evidence."""
    if type_annotation is None:
        return False
    name = getattr(type_annotation, "name", None)
    if not isinstance(name, str):
        name = str(type_annotation) if isinstance(type_annotation, str) else None
    return name in _CROSSING_TYPE_NAMES


def _borrow_var_name(value: Any) -> str | None:
    """The variable name a borrow/move operand refers to, when it is one."""
    if isinstance(value, str):
        return value
    if isinstance(value, fast.Variable):
        name = getattr(value, "name", None)
        return name if isinstance(name, str) else None
    return None


def _mut_borrow_target(expr: Any) -> str | None:
    """`&mut x` initializer -> "x"; anything else -> None.

    Surface `&mut x` parses to `BorrowUnique` (see the frozen emitter's
    BorrowUnique note); `BorrowExpression` carries an explicit mode.
    """
    if isinstance(expr, fast.BorrowUnique):
        return _borrow_var_name(getattr(expr, "variable", None))
    if isinstance(expr, fast.BorrowExpression):
        mode = getattr(expr, "mode", None)
        mode_name = mode if isinstance(mode, str) else getattr(mode, "mode", None)
        if isinstance(mode_name, str) and \
                mode_name.lower().lstrip("@") in _MUT_BORROW_MODES:
            return _borrow_var_name(getattr(expr, "variable", None))
    return None


def _op_key(effect_ref: Any) -> tuple[str, str] | None:
    """(effect, op) key of a perform's dotted reference.

    "Thread.spawn" -> ("Thread", "spawn"); module-qualified spellings keep
    only the last two segments, matching how declarations are keyed.
    """
    if not isinstance(effect_ref, str):
        effect_ref = str(effect_ref) if effect_ref is not None else ""
    parts = [p for p in effect_ref.split(".") if p]
    if len(parts) < 2:
        return None
    return (parts[-2], parts[-1])


def _collect_spawn_ops(root: Any) -> tuple[frozenset[tuple[str, str]],
                                            frozenset[tuple[str, str]]]:
    """(spawn ops, handle-producing ops): every (effect, op) declared
    `with EFFECT_SPAWN`, and every one whose runtime symbol produces an
    opaque separate-by-construction handle (spawn, mutex-create)."""
    spawn: set[tuple[str, str]] = set()
    handles: set[tuple[str, str]] = set()
    for node in _walk_all(root):
        if not isinstance(node, fast.EffectDeclaration):
            continue
        effect_name = getattr(node, "name", None)
        if not isinstance(effect_name, str) or not effect_name:
            continue
        effect_key = effect_name.split(".")[-1]
        for op in getattr(node, "operations", None) or []:
            sym = getattr(op, "c_effect", None)
            op_name = getattr(op, "name", None)
            if not (isinstance(op_name, str) and op_name):
                continue
            if sym == SPAWN_RUNTIME_SYMBOL:
                spawn.add((effect_key, op_name))
            if sym in _HANDLE_RUNTIME_SYMBOLS:
                handles.add((effect_key, op_name))
    return frozenset(spawn), frozenset(handles)


class _SpawnCaptureChecker:
    """Scoped walk over the mutable AST (the `name_resolution._Resolver`
    shape, carrying binding INFO instead of bare names).

    Two operating modes share the walk:
      * normal (``collect_free is None``): maintain bindings, and at every
        spawn-mapped `PerformEffect` check the closure argument's captures;
      * free-variable collection (``collect_free`` is a set): the walk
        starts from a spawned LambdaExpression with EMPTY scopes, records
        every name read that no scope inside the lambda binds — exactly its
        captures — and files no diagnostics (the enclosing normal walk owns
        those; nested spawn sites are re-visited there with real scopes).
    """

    def __init__(self, spawn_ops: frozenset[tuple[str, str]],
                 file_path: str | None,
                 collect_free: set[str] | None = None,
                 handle_ops: frozenset[tuple[str, str]] = frozenset()) -> None:
        self.spawn_ops = spawn_ops
        self.handle_ops = handle_ops
        self.file_path = file_path
        self.scopes: list[dict[str, _Binding]] = [{}]
        self.errors: list[BorrowError] = []
        self.free = collect_free
        self._reported: set[tuple[int, str, str]] = set()
        #: > 0 inside `unsafe { }`: the separateness rule is suspended
        #: (docs/separate_send_sync.md § 4 — locality/@mut rules are NOT).
        self._unsafe_depth = 0
        #: (effect, op) pairs handled by a lexically enclosing `handle`:
        #: a spawn-mapped perform under one is VIRTUALIZED, so no real
        #: thread is crossed and separateness is not demanded.
        self._handled_ops: list[set[tuple[str, str]]] = []

    # -- scope helpers ---------------------------------------------------
    def push(self) -> None:
        self.scopes.append({})

    def pop(self) -> None:
        self.scopes.pop()

    def bind(self, name: Any, binding: _Binding | None = None) -> None:
        if isinstance(name, str) and name:
            self.scopes[-1][name] = binding or _Binding()

    def lookup(self, name: Any) -> _Binding | None:
        if not isinstance(name, str):
            return None
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def _mut_borrow_holder(self, name: str) -> str | None:
        """The in-scope binding holding an active @mut borrow of `name`."""
        for scope in reversed(self.scopes):
            for holder, binding in scope.items():
                if binding.mut_borrow_of == name:
                    return holder
        return None

    # -- reads (free-variable collection mode) ---------------------------
    def note_read(self, name: Any) -> None:
        if self.free is None or not isinstance(name, str) or not name:
            return
        if name == "_" or name.startswith("__") or "." in name:
            return
        if self.lookup(name) is None:
            self.free.add(name)

    # -- the spawn check -------------------------------------------------
    @staticmethod
    def _moved_names(lam: Any) -> set[str]:
        """Names the closure takes by `move(..)` anywhere in its body:
        ownership transfers into the (single) spawned thread — the Send
        pattern — so sharedness is not a race for them."""
        out: set[str] = set()
        for node in _walk_all(lam):
            if isinstance(node, fast.Move):
                name = _borrow_var_name(getattr(node, "variable", None))
                if name:
                    out.add(name)
        return out

    def _shared_note(self, name: str, binding: _Binding) -> str:
        if not binding.shared_provenance:
            return ""
        steps = "; ".join(f"'{a}' was bound from '{b}'"
                          for (a, b) in binding.shared_provenance)
        return f" (note: {steps})"

    def _resolve_closure(self, arg: Any) -> Any | None:
        """The LambdaExpression an argument denotes, if statically known.

        Inline lambdas and `move(f)` around them resolve directly; a bare
        variable resolves through the binding recorded by `let f = fn ...`.
        Anything else (a call result, a field read) is not traced —
        documented as remaining in docs/threads_runtime.md § Modes.
        """
        if isinstance(arg, fast.LambdaExpression):
            return arg
        if isinstance(arg, fast.Move):
            inner = getattr(arg, "variable", None)
            if isinstance(inner, str):
                inner = None if inner == "" else inner
                binding = self.lookup(inner)
                return binding.lambda_node if binding is not None else None
            return self._resolve_closure(inner)
        if isinstance(arg, fast.Variable):
            binding = self.lookup(getattr(arg, "name", None))
            return binding.lambda_node if binding is not None else None
        return None

    def _report(self, site: Any, lam: Any, kind: str, name: str,
                message: str) -> None:
        key = (id(site), name, kind)
        if key in self._reported:
            return
        self._reported.add(key)
        location = getattr(site, "location", None) or getattr(lam, "location", None)
        self.errors.append(BorrowError(
            message=message,
            node_id=-1,
            kind=kind,
            variable=name,
            location=location,
        ))

    def _check_spawn_perform(self, node: Any) -> None:
        key = _op_key(getattr(node, "effect_name", None))
        if key is None or key not in self.spawn_ops:
            return
        op_display = f"{key[0]}.{key[1]}"
        # Separateness is demanded only when a REAL thread is crossed:
        # a lexically enclosing handler for this op virtualizes it, and
        # `unsafe { }` is the scoped escape (docs/separate_send_sync.md).
        # Locality/@mut rules below apply regardless (signature promise).
        check_separate = (self._unsafe_depth == 0
                          and not any(key in hs for hs in self._handled_ops))
        for arg in getattr(node, "arguments", None) or []:
            lam = self._resolve_closure(arg)
            if lam is None:
                continue
            captures: set[str] = set()
            collector = _SpawnCaptureChecker(
                self.spawn_ops, self.file_path, collect_free=captures,
                handle_ops=self.handle_ops)
            collector.visit(lam)
            moved = self._moved_names(lam) if check_separate else set()
            site = arg if _is_node(arg) else node
            for name in sorted(captures):
                binding = self.lookup(name)
                if binding is None:
                    continue  # module fn / builtin / not a value binding
                if binding.locality == "local":
                    self._report(
                        site, lam, LOCAL_CAPTURE_KIND, name,
                        f"closure passed to spawn-mapped operation "
                        f"'{op_display}' (with {SPAWN_RUNTIME_SYMBOL}) "
                        f"captures @local variable '{name}': a @local value "
                        f"lives in the spawning frame's stack region, and "
                        f"that frame may return while the spawned thread is "
                        f"still running, leaving the capture dangling; "
                        f"spawn requires @global captures for exactly this "
                        f"reason{self._provenance_note(name, binding)}")
                elif binding.mut_borrow_of is not None:
                    self._report(
                        site, lam, MUT_BORROW_CAPTURE_KIND, name,
                        f"closure passed to spawn-mapped operation "
                        f"'{op_display}' (with {SPAWN_RUNTIME_SYMBOL}) "
                        f"captures '{name}', which holds an active @mut "
                        f"borrow of '{binding.mut_borrow_of}': an exclusive "
                        f"borrow must not be shared across threads")
                else:
                    holder = self._mut_borrow_holder(name)
                    if holder is not None:
                        self._report(
                            site, lam, MUT_BORROW_CAPTURE_KIND, name,
                            f"closure passed to spawn-mapped operation "
                            f"'{op_display}' (with {SPAWN_RUNTIME_SYMBOL}) "
                            f"captures '{name}' while '{holder}' holds an "
                            f"active @mut borrow of it: an exclusive borrow "
                            f"must not be shared across threads")
                    elif (check_separate and name not in moved
                          and binding.separateness == "shared"):
                        self._report(
                            site, lam, SEPARATE_CAPTURE_KIND, name,
                            f"closure passed to spawn-mapped operation "
                            f"'{op_display}' (with {SPAWN_RUNTIME_SYMBOL}) "
                            f"captures '{name}', which has shared mutable "
                            f"identity (Vec): two threads mutating through "
                            f"one handle race; protect it "
                            f"(std.sync.protect), move it into exactly one "
                            f"thread (move({name})), or take responsibility "
                            f"with `unsafe {{ .. }}`"
                            f"{self._shared_note(name, binding)}")

    # -- Rule B: initializer-driven locality -------------------------------
    def _expr_crosses(self, expr: Any) -> bool:
        """Syntactic mode-crossing evidence for an initializer expression:
        scalar literals, arithmetic/comparison/unary over crossing operands,
        and names whose bindings carry crossing evidence. Conservative —
        False means "cannot verify", not "does not cross"."""
        if isinstance(expr, fast.Literal):
            return isinstance(getattr(expr, "value", None),
                              (int, float, bool, str))
        if isinstance(expr, fast.BinaryOperation):
            return (self._expr_crosses(getattr(expr, "left", None))
                    and self._expr_crosses(getattr(expr, "right", None)))
        if isinstance(expr, fast.ComparisonExpression):
            return True   # comparisons yield bool
        if isinstance(expr, fast.UnaryOperation):
            return self._expr_crosses(getattr(expr, "operand", None))
        if isinstance(expr, fast.Variable):
            b = self.lookup(getattr(expr, "name", None))
            return b is not None and b.crosses
        return False

    def _init_local_source(self, expr: Any) -> tuple[str, _Binding] | None:
        """The @local-tracked name an initializer READS AS A WHOLE, if any:
        a bare variable, or a borrow/move of one. Field reads, calls and
        literals containing local names copy VALUES out and are not traced
        (documented limitation)."""
        name = None
        if isinstance(expr, fast.Variable):
            name = getattr(expr, "name", None)
        elif isinstance(expr, (fast.BorrowShared, fast.BorrowUnique,
                               fast.Move, fast.BorrowExpression)):
            name = _borrow_var_name(getattr(expr, "variable", None))
        if not isinstance(name, str):
            return None
        b = self.lookup(name)
        if b is not None and b.locality == "local":
            return (name, b)
        return None

    # -- separateness (docs/separate_send_sync.md) -------------------------
    def _expr_separateness(self, expr: Any) -> tuple[str, tuple]:
        """(classification, provenance) for an initializer: "separate",
        "shared" (known shared mutable identity), or "unknown". Structural
        over struct literals; name-propagating like Rule B locality."""
        if isinstance(expr, fast.Literal):
            return "separate", ()
        if isinstance(expr, (fast.BinaryOperation, fast.ComparisonExpression,
                             fast.UnaryOperation)):
            return "separate", ()   # scalar arithmetic results
        if isinstance(expr, fast.ListLiteral):
            return "shared", ()     # list literals build Vecs
        if isinstance(expr, (fast.FunctionCall, fast.QualifiedFunctionCall)):
            name = getattr(expr, "name", None)
            if not isinstance(name, str):
                parts = [str(x) for x in (getattr(expr, "parts", None) or [])]
                name = ".".join(parts)
            if name in _PROTECT_CALLEES:
                return "separate", ()   # Protected: separate by construction
            base = name.split(".")[0] if isinstance(name, str) else ""
            if base == "Vec" or name == "Vec.new":
                return "shared", ()     # Vec constructors: shared identity
            return "unknown", ()
        if isinstance(expr, fast.PerformEffect):
            key = _op_key(getattr(expr, "effect_name", None))
            if key is not None and key in self.handle_ops:
                return "separate", ()   # runtime handle (mutex/thread)
            return "unknown", ()
        if isinstance(expr, fast.StructInstantiation):
            # Structural composition: separate iff every field is; shared
            # if any field is known-shared.
            worst = "separate"
            for field in getattr(expr, "field_assignments", None) or []:
                v = getattr(field, "value", None) or getattr(field, "expression", None)
                cls, _prov = self._expr_separateness(v)
                if cls == "shared":
                    return "shared", ()
                if cls == "unknown":
                    worst = "unknown"
            return worst, ()
        if isinstance(expr, fast.Variable):
            b = self.lookup(getattr(expr, "name", None))
            if b is None:
                return "unknown", ()
            return b.separateness, b.shared_provenance
        return "unknown", ()

    # -- binder helpers ---------------------------------------------------
    def _binding_from_let(self, let_binding: Any) -> _Binding:
        init = getattr(let_binding, "initializer", None)
        declared = _declared_locality(getattr(let_binding, "mode", None))
        crosses = (_annotation_crosses(getattr(let_binding, "type_annotation",
                                               None))
                   or self._expr_crosses(init))
        source = self._init_local_source(init)
        ident = getattr(let_binding, "identifier", None)
        ident = ident if isinstance(ident, str) else ""

        if declared == "local":
            # Declared root: provenance chain starts here.
            locality, provenance = "local", ()
        elif source is not None:
            src_name, src_binding = source
            src_crosses = src_binding.crosses or crosses
            if declared == "global":
                # EXPLICIT escape: allowed only with crossing evidence —
                # the visible, checked laundering point (Rule B).
                if src_crosses:
                    locality, provenance = "global", ()
                    crosses = True
                else:
                    self._report_escape(let_binding, ident, src_name,
                                        src_binding)
                    # Keep it local so downstream diagnostics stay coherent.
                    locality = "local"
                    provenance = ((ident, src_name),) + src_binding.provenance
            else:
                # Unannotated: locality FOLLOWS THE DATA.
                locality = "local"
                provenance = ((ident, src_name),) + src_binding.provenance
                crosses = crosses or src_binding.crosses
        else:
            locality, provenance = (declared or "global"), ()

        sep, sep_prov = self._expr_separateness(init)
        if isinstance(init, fast.Variable) and sep == "shared" and ident:
            src_name = getattr(init, "name", "")
            sep_prov = ((ident, src_name),) + sep_prov
        return _Binding(
            locality=locality,
            lambda_node=init if isinstance(init, fast.LambdaExpression) else None,
            mut_borrow_of=_mut_borrow_target(init),
            provenance=provenance,
            crosses=crosses,
            separateness=sep,
            shared_provenance=sep_prov,
        )

    def _report_escape(self, let_binding: Any, ident: str, src_name: str,
                       src_binding: _Binding) -> None:
        chain = self._provenance_note(src_name, src_binding)
        location = getattr(let_binding, "location", None)
        self.errors.append(BorrowError(
            message=(
                f"cannot bind @global '{ident}' from @local '{src_name}': "
                f"the checker cannot verify the value contains no reference "
                f"into the current frame (only scalar-typed values — "
                f"int/bool/float/string — cross to @global today); keep the "
                f"binding unannotated to stay @local, or give the source a "
                f"scalar type annotation{chain}"),
            node_id=-1,
            kind=LOCALITY_ESCAPE_KIND,
            variable=ident,
            location=location,
        ))

    def _provenance_note(self, name: str, binding: _Binding) -> str:
        """Render the Rule B chain: how `name` came to be @local."""
        if not binding.provenance:
            return f" (note: '{name}' was declared @local)"
        steps = "; ".join(
            f"'{alias}' was bound from '{src}'"
            for (alias, src) in binding.provenance)
        root = binding.provenance[-1][1]
        return (f" (note: {steps}; '{root}' was declared @local — "
                f"locality follows the data)")

    def bind_param(self, param: Any) -> None:
        if isinstance(param, str):
            self.bind(param)
            return
        self.bind(getattr(param, "name", None),
                  _Binding(
                      locality=_declared_locality(getattr(param, "mode",
                                                          None)) or "global",
                      crosses=_annotation_crosses(
                          getattr(param, "type_annotation", None))))

    def bind_pattern(self, p: Any) -> None:
        """Bind every name a pattern introduces (as default-global).

        Over-binding (a bare pattern name that is really a variant) only
        SHADOWS: it can suppress a capture diagnostic for a name that was
        never a capture, never fabricate one.
        """
        if p is None:
            return
        if isinstance(p, str):
            self.bind(p)
            return
        if isinstance(p, fast.VariablePattern):
            self.bind(getattr(p, "name", None))
            return
        if isinstance(p, fast.VariantPattern):
            for sub in getattr(p, "patterns", None) or []:
                self.bind_pattern(sub)
            return
        if isinstance(p, (fast.Variable, fast.TypeReference)):
            self.bind(getattr(p, "name", None))
            return
        if isinstance(p, fast.QualifiedName):
            parts = [str(x) for x in (getattr(p, "parts", None) or [])]
            if len(parts) == 1:
                self.bind(parts[0])
            return
        if isinstance(p, (fast.BorrowShared, fast.BorrowUnique, fast.Move)):
            self.bind_pattern(getattr(p, "variable", None))
            return
        if isinstance(p, fast.ModeExpression):
            self.bind_pattern(getattr(p, "expression", None))
            return
        if isinstance(p, fast.SomeExpression):
            self.bind_pattern(getattr(p, "value", None))
            return
        if isinstance(p, (fast.FunctionCall, fast.QualifiedFunctionCall)):
            for a in getattr(p, "arguments", None) or []:
                self.bind_pattern(a)
            return
        if isinstance(p, (fast.ListLiteral, fast.TupleLiteral)):
            for e in getattr(p, "elements", None) or []:
                self.bind_pattern(e)
            return
        if isinstance(p, fast.SpreadElement):
            self.bind_pattern(getattr(p, "expression", None))
            return
        # Literals, wildcards and field accesses bind nothing.

    def bind_handler_arm(self, pattern: Any) -> None:
        if pattern is None:
            return
        if isinstance(pattern, (fast.PerformEffect, fast.FunctionCall,
                                fast.QualifiedFunctionCall)):
            for a in getattr(pattern, "arguments", None) or []:
                self.bind_pattern(a)
            return
        self.bind_pattern(pattern)

    # -- handled-op extraction (handler subtraction) ----------------------
    def _handle_effect_ops(self, node: Any) -> set[tuple[str, str]]:
        """(effect, op) pairs a `handle E with { op(..) -> .. }` covers."""
        effect = getattr(node, "effect_name", None)
        effect = str(effect).split(".")[-1] if effect is not None else ""
        out: set[tuple[str, str]] = set()
        for c in getattr(node, "handler", None) or []:
            op = getattr(c, "op_name", None) or getattr(c, "name", None)
            if isinstance(op, str) and op and effect:
                out.add((effect, op))
        return out

    def _handle_block_ops(self, node: Any) -> set[tuple[str, str]]:
        effect = getattr(node, "effect_name", None) or getattr(node, "effect", None)
        effect = str(effect).split(".")[-1] if effect is not None else ""
        out: set[tuple[str, str]] = set()
        for arm in getattr(node, "arms", None) or []:
            pattern, _body = _case_parts(arm)
            op = None
            if isinstance(pattern, (fast.FunctionCall, fast.QualifiedFunctionCall)):
                nm = getattr(pattern, "name", None)
                if not isinstance(nm, str):
                    parts = [str(x) for x in (getattr(pattern, "parts", None) or [])]
                    nm = parts[-1] if parts else None
                op = nm
            elif isinstance(pattern, fast.Variable):
                op = getattr(pattern, "name", None)
            if isinstance(op, str) and op and effect:
                out.add((effect, op.split(".")[-1]))
        return out

    # -- the walk ---------------------------------------------------------
    def visit_all(self, nodes: Iterable[Any]) -> None:
        for n in nodes:
            self.visit(n)

    def visit_body(self, body: Any) -> None:
        if body is None:
            return
        if isinstance(body, (list, tuple)):
            self.push()
            self.visit_all(body)
            self.pop()
            return
        self.visit(body)

    def visit(self, node: Any) -> None:  # noqa: C901 - one branch per form
        if node is None or not _is_node(node):
            return

        # ---------- references (free-collection mode only) ----------
        if isinstance(node, fast.Variable):
            self.note_read(getattr(node, "name", None))
            return
        if isinstance(node, fast.QualifiedName):
            parts = [str(x) for x in (getattr(node, "parts", None) or [])]
            if parts:
                self.note_read(parts[0])
            return
        if isinstance(node, (fast.BorrowShared, fast.BorrowUnique, fast.Move,
                             fast.BorrowExpression)):
            var = getattr(node, "variable", None)
            if isinstance(var, str):
                self.note_read(var)
            else:
                self.visit(var)
            return
        if isinstance(node, fast.ExclaveExpression):
            inner = getattr(node, "expression", None)
            if isinstance(inner, str):
                self.note_read(inner)
            else:
                self.visit(inner)
            return
        if isinstance(node, fast.FieldAccess):
            base = getattr(node, "base", None)
            if isinstance(base, str):
                self.note_read(base)
            else:
                self.visit(base)
            return
        if isinstance(node, fast.FunctionCall):
            name = getattr(node, "name", None)
            if isinstance(name, str) and "." not in name:
                self.note_read(name)
            self.visit_all(getattr(node, "arguments", None) or [])
            return
        if isinstance(node, fast.QualifiedFunctionCall):
            parts = [str(p) for p in (getattr(node, "parts", None) or [])]
            if parts:
                self.note_read(parts[0])
            self.visit_all(getattr(node, "arguments", None) or [])
            return
        if isinstance(node, fast.PerformEffect):
            if self.free is None:
                self._check_spawn_perform(node)
            self.visit_all(getattr(node, "arguments", None) or [])
            return
        if isinstance(node, fast.Assignment):
            self.visit(getattr(node, "expression", None))
            target = getattr(node, "name", None)
            if isinstance(target, str):
                root = target.split(".")[0].split("[")[0].strip()
                self.note_read(root)
                # `r = &mut x`: the binding now holds a mut borrow.
                mut_of = _mut_borrow_target(getattr(node, "expression", None))
                if mut_of is not None:
                    binding = self.lookup(root)
                    if binding is not None:
                        binding.mut_borrow_of = mut_of
                # Rule B through assignment: `a = local_thing` makes `a`
                # local from here on (sticky — once a name has held local
                # data in this scope, it stays tracked; an over-
                # approximation that errs toward safety).
                if self.free is None:
                    src = self._init_local_source(
                        getattr(node, "expression", None))
                    if src is not None:
                        binding = self.lookup(root)
                        if binding is not None and binding.locality != "local":
                            src_name, src_binding = src
                            binding.locality = "local"
                            binding.provenance = (
                                ((root, src_name),) + src_binding.provenance)
            else:
                self.visit(target)
            return

        # ---------- binders ----------
        if isinstance(node, fast.FunctionDeclaration):
            if self.free is not None:
                # A nested `fn` does not close over the enclosing function
                # (HIR hoists it into the flat namespace) — nothing it reads
                # is a capture of the spawned closure.
                return
            saved = self.scopes
            self.scopes = [saved[0], {}]
            for p in getattr(node, "params", None) or []:
                self.bind_param(p)
            self.bind("self")
            self.visit_body(getattr(node, "body", None))
            self.scopes = saved
            return
        if isinstance(node, fast.LambdaExpression):
            self.push()
            for p in getattr(node, "params", None) or []:
                self.bind_param(p)
            self.visit_body(getattr(node, "body", None))
            self.pop()
            return
        if isinstance(node, fast.Implementation):
            for m in getattr(node, "methods", None) or []:
                self.visit(m)
            return
        if isinstance(node, fast.LetStatement):
            for b in getattr(node, "bindings", None) or []:
                self.visit(getattr(b, "initializer", None))
                self.bind(getattr(b, "identifier", None),
                          self._binding_from_let(b))
            return
        if isinstance(node, fast.LetBinding):
            self.visit(getattr(node, "initializer", None))
            self.bind(getattr(node, "identifier", None),
                      self._binding_from_let(node))
            return
        if isinstance(node, fast.Block):
            self.push()
            self.visit_all(getattr(node, "statements", None) or [])
            self.pop()
            return
        if isinstance(node, fast.ForStatement):
            self.visit(getattr(node, "iterable", None))
            self.push()
            it = getattr(node, "iterator", None)
            if isinstance(it, str):
                self.bind(it)
            else:
                self.bind_pattern(it)
            self.visit_body(getattr(node, "body", None))
            self.pop()
            return
        if isinstance(node, fast.IfLetExpression):
            self.visit(getattr(node, "value", None))
            self.push()
            self.bind_pattern(getattr(node, "pattern", None))
            self.visit_body(getattr(node, "then_branch", None))
            self.pop()
            self.visit_body(getattr(node, "else_branch", None))
            return
        if isinstance(node, fast.WhileLetStatement):
            self.visit(getattr(node, "value", None))
            self.push()
            self.bind_pattern(getattr(node, "pattern", None))
            self.visit_body(getattr(node, "body", None))
            self.pop()
            return
        if isinstance(node, fast.MatchExpression):
            self.visit(getattr(node, "expression", None))
            for case in getattr(node, "cases", None) or []:
                pattern, body = _case_parts(case)
                self.push()
                self.bind_pattern(pattern)
                self.visit_body(body)
                self.pop()
            return
        if isinstance(node, fast.TryCatch):
            self.visit_body(getattr(node, "body", None))
            self.push()
            self.bind(getattr(node, "catch_name", None))
            self.visit_body(getattr(node, "catch_body", None))
            self.pop()
            return
        if isinstance(node, uast.UnsafeBlock):
            self._unsafe_depth += 1
            try:
                self.visit_all(getattr(node, "body", None) or [])
            finally:
                self._unsafe_depth -= 1
            return
        if isinstance(node, fast.HandleEffect):
            handled = self._handle_effect_ops(node)
            self._handled_ops.append(handled)
            try:
                for c in getattr(node, "handler", None) or []:
                    self.visit(c)
                self.visit_body(getattr(node, "continuation", None))
            finally:
                self._handled_ops.pop()
            return
        if isinstance(node, fast.HandleCase):
            self.push()
            params = getattr(node, "param_names", None)
            if params is None:
                params = [getattr(node, "param_name", None)]
            for p in params or []:
                self.bind(p if isinstance(p, str) else getattr(p, "name", None))
            self.visit_body(getattr(node, "body", None))
            self.pop()
            return
        if isinstance(node, fast.HandleBlock):
            handled = self._handle_block_ops(node)
            self._handled_ops.append(handled)
            try:
                self.visit_body(getattr(node, "subject", None))
            finally:
                pass  # arms below re-push nothing; popped after the block
            try:
                for arm in getattr(node, "arms", None) or []:
                    pattern, body = _case_parts(arm)
                    self.push()
                    self.bind_handler_arm(pattern)
                    self.visit_body(body)
                    self.pop()
            finally:
                self._handled_ops.pop()
            return
        if isinstance(node, fast.Comprehension):
            self.visit(getattr(node, "iterable", None))
            self.push()
            for t in _as_list(getattr(node, "targets", None)):
                self.bind_pattern(t)
            self.visit(getattr(node, "expression", None))
            self.pop()
            return

        # ---------- everything else: structural recursion ----------
        self.visit_all(_sub_nodes(node, skip=_TYPE_ATTRS))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def check_spawn_captures(root: Any,
                         file_path: str | None = None) -> list[BorrowError]:
    """Structured spawn-capture diagnostics for `root`, if any.

    `root` is the MUTABLE, post-desugar AST (`pipeline.PhaseContext.program`),
    the same tree `name_resolution.check_names` walks.  Returns an empty
    list for programs that declare no `with EFFECT_SPAWN` operation, so the
    pass costs nothing where threads are not in play.
    """
    spawn_ops, handle_ops = _collect_spawn_ops(root)
    if not spawn_ops:
        return []
    checker = _SpawnCaptureChecker(spawn_ops, file_path, handle_ops=handle_ops)
    statements = _module_statements(root)
    # Module-level `let` bindings are program-wide constants; bind them all
    # before walking so a spawn anywhere can resolve them.
    for stmt in statements:
        if isinstance(stmt, fast.LetStatement):
            for b in getattr(stmt, "bindings", None) or []:
                checker.bind(getattr(b, "identifier", None),
                             checker._binding_from_let(b))
    checker.visit_all(statements)
    return checker.errors
