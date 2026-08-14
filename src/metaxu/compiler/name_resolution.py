"""Compile-time name resolution: an undefined name is an error.

Until this pass existed, a name that resolved to nothing was simply
*dropped*.  `fn main() -> int { undefined_thing; 42 }` compiled, ran and
answered 42; `fn main() -> int { helpr(); 7 }` (a typo of `helper`)
compiled and only died at run time with `Unknown callee`.  That is the same
silent-degradation family as the lexer audit's `let x = 1e10; x`, which
answered `1` *because* the stray `e10` became a discarded identifier
(docs/token_reachability.md).

One rule, in three positions:

  * every **variable reference** must name something in scope — a read, the
    root of a field chain, a borrow/move operand, an assignment target;
  * every **plain call callee** (`f(a)`) must name something callable;
  * every **dotted call callee** (`a.b(...)`) must resolve the way
    `hir._from_orig_expr` resolves it — as a variant constructor, a static
    impl call, a receiver call (then the RECEIVER is the name checked), or
    a plain call of the dotted name the module system produced.

Each violation produces a structured `BorrowError` with kind
``type-unresolved-name``, so `pipeline.run_pipeline` promotes it to a
`TypeCheckError` exactly like every other `type-*` diagnostic, with a source
location and — when a plausible candidate exists within edit distance 2 — a
"did you mean" note.

Why this pass runs on the MUTABLE, post-desugar AST rather than the frozen
one
---------------------------------------------------------------------------
The frozen AST is lossy in precisely the places names are bound and used:
`MatchExpression` freezes its arms into a payload *descriptor* and drops the
arm bodies entirely, `ForStatement` freezes its body to an empty node of
kind ``list``, and `IfLetExpression` flattens its pattern into the same
child list as its value and branches, so a pattern's binder occurrence is
indistinguishable from a variable read.  A resolver over the frozen tree
would therefore both miss every name inside a match arm or a for body and
report every pattern binder as undefined.  HIR lowering itself reads the
original nodes (`HIRBuilder._from_orig_expr(orig, ...)`), so resolving over
the original AST checks exactly the tree that gets compiled.

See `docs/name_resolution.md` for the enumerated in-scope categories and
for what is deliberately out of scope.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator

import metaxu.metaxu_ast as fast
from metaxu.extern_ast import (ExternBlock, ExternFunctionDeclaration,
                               ExternTypeDeclaration)

from .frozen_borrow_checker import BorrowError
from .hir import (_BUILTIN_METHODS, BUILTIN_FUNCTION_NAMES,
                  parse_impl_method_name)


#: kind carried by every diagnostic this module raises.  The `type-` prefix
#: is what makes `pipeline.run_pipeline` raise `TypeCheckError`.
UNRESOLVED_NAME_KIND = "type-unresolved-name"


# ---------------------------------------------------------------------------
# Names the language itself provides
# ---------------------------------------------------------------------------

#: Value-position names with no declaration anywhere in user code.
#:
#: `null` is the null-pointer literal (`hir` lowers a `Variable("null")` to a
#: literal, never to a name read).  `self` is the receiver bound by every
#: impl method (bound per-function too, but a trait method declared without
#: an explicit receiver still reads it).  `resume` is the continuation a
#: handler arm calls; `hir` lowers it to a dedicated Resume op.
LANGUAGE_VALUE_NAMES = frozenset({"null", "self", "resume"})

#: Runtime builtins that exist under a dotted or type-qualified spelling,
#: plus the two reflection/runtime names the native backends know
#: (`codegen_llvm._RUNTIME_NAMES`, `codegen_clif._RUNTIME_NAMES`) but the
#: interpreter has no shim for.  Calling `type_of` still fails loudly at run
#: time — that is defence in depth, not a reason to call the *name* unknown.
EXTRA_BUILTIN_NAMES = frozenset({"type_of", "Vec", "vector"})

#: Runtime builtins registered under a DOTTED name
#: (`mir_interp._register_builtins`), reachable as `Type.method(...)` with
#: no user declaration behind them.
DOTTED_BUILTIN_NAMES = frozenset({"Vec.new"})

#: Option/Result are language-provided enums (`hir._from_orig_expr` builds
#: their variants even when no user enum declares them).
BUILTIN_VARIANTS = frozenset({"Some", "None", "Ok", "Err"})

#: Attributes that hold TYPE-level information.  The scoped walk does not
#: descend into them: a type expression names types, not values, and
#: `TypeReference("T")` is not a variable read.
_TYPE_ATTRS = frozenset({
    "type_annotation", "return_type", "type_params", "type_args", "type_info",
    "type_expr", "bounds", "where_clause", "constraints", "interface_name",
    "type_name", "base_type", "implements", "modes", "mode", "linearity",
    "uniqueness", "locality", "type_constructor", "param_types", "extends",
    "effect_class", "c_function", "c_effect", "type_var", "declared_type",
})

#: Structural attributes that are never part of the value tree.
_SKIP_ATTRS = frozenset({"parent", "scope", "location", "symbol_table"})


def _is_node(value: Any) -> bool:
    # extern_ast / unsafe_ast / decorator_ast node classes all subclass
    # fast.Node, so one check covers every AST family.
    return isinstance(value, fast.Node)


#: The two skip sets `_sub_nodes` is ever called with, precomputed: it runs
#: once per AST node per pass, so rebuilding the union there showed up.
_SKIP_ONLY = frozenset(_SKIP_ATTRS)
_SKIP_AND_TYPES = frozenset(_SKIP_ATTRS | _TYPE_ATTRS)


def _sub_nodes(node: Any, skip: Iterable[str] = ()) -> Iterator[Any]:
    """Every AST node reachable from `node`'s attributes, in a stable order.

    `children` is visited last and de-duplicated by identity against the
    named attributes, because most node classes put the same objects in
    both (`Program.statements` are also `Program.children`).
    """
    skip = (_SKIP_AND_TYPES if skip is _TYPE_ATTRS
            else (_SKIP_ONLY if not skip else frozenset(skip) | _SKIP_ATTRS))
    seen: set[int] = set()

    def emit(value: Any) -> Iterator[Any]:
        if _is_node(value):
            if id(value) not in seen:
                seen.add(id(value))
                yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from emit(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from emit(item)

    items = list(vars(node).items()) if hasattr(node, "__dict__") else []
    for attr, value in items:
        if attr in skip or attr == "children":
            continue
        yield from emit(value)
    if "children" not in skip:
        yield from emit(getattr(node, "children", None))


def _walk_all(root: Any) -> Iterator[Any]:
    """Every AST node in the tree, each yielded once."""
    seen: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        stack.extend(_sub_nodes(node))


def _edit_distance_within(a: str, b: str, limit: int) -> int | None:
    """Levenshtein distance between `a` and `b`, or None if it exceeds
    `limit`.  Bounded rows keep this cheap for the candidate scan."""
    if abs(len(a) - len(b)) > limit:
        return None
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        if min(cur) > limit:
            return None
        prev = cur
    return prev[-1] if prev[-1] <= limit else None


def _suggest(name: str, candidates: Iterable[str]) -> str | None:
    """The closest in-scope name within edit distance 2, else None.

    Ties break on the shorter name, then alphabetically, so the suggestion
    is deterministic.
    """
    limit = 1 if len(name) <= 4 else 2
    best: tuple[int, int, str] | None = None
    for cand in candidates:
        if cand == name or not cand or cand.startswith("__"):
            continue
        d = _edit_distance_within(name, cand, limit)
        if d is None or d == 0:
            continue
        key = (d, len(cand), cand)
        if best is None or key < best:
            best = key
    return best[2] if best is not None else None


# ---------------------------------------------------------------------------
# The global (module-level) name set
# ---------------------------------------------------------------------------

class _GlobalNames:
    """Names visible everywhere in a compilation unit.

    Metaxu has no forward-declaration rule: a function may call a function
    declared later in the file, and module-level `let` constants are
    published by `__module_init` before the entry point runs
    (`hir.HIRBuilder.build`).  So all of these are collected in one prepass
    and are in scope for the whole program.
    """

    def __init__(self) -> None:
        self.functions: set[str] = set()       # fn / extern fn / impl methods
        self.methods: set[str] = set()         # trait + impl method names
        self.variants: set[str] = set(BUILTIN_VARIANTS)
        self.types: set[str] = set()           # struct / enum / trait / alias
        self.impl_types: set[str] = set()      # types with an `implement` block
        self.effect_ops: set[str] = set()
        self.imported: set[str] = set()        # import / from-import locals
        # Materialized by freeze(), below.
        self.callable_names: set[str] = set()
        self.value_names: set[str] = set()

    def freeze(self) -> None:
        """Materialize the two lookup sets once.

        They are consulted at every single name reference, so recomputing
        the unions per reference made resolution quadratic in program size.
        Call once, after collection, before the scoped walk.
        """
        self.callable_names = (
            self.functions | self.methods | self.variants | self.types
            | self.effect_ops | self.imported
            | set(BUILTIN_FUNCTION_NAMES) | set(EXTRA_BUILTIN_NAMES))
        self.value_names = (
            self.functions | self.variants | self.types | self.imported
            | set(LANGUAGE_VALUE_NAMES)
            | set(BUILTIN_FUNCTION_NAMES) | set(EXTRA_BUILTIN_NAMES))


def collect_global_names(root: Any) -> _GlobalNames:
    """Every module-level name a program provides.

    Each branch corresponds to one documented category in
    `docs/name_resolution.md`; nothing here is a special case bolted on to
    make a file compile.
    """
    g = _GlobalNames()
    for node in _walk_all(root):
        # --- module functions, including NESTED ones -----------------------
        # `HIRBuilder.build`'s hoisting walk lifts every FunctionDeclaration
        # in the tree into a flat MIR function under its bare name, so a
        # `fn` declared inside another function's body occupies the global
        # namespace exactly like a top-level one (docs/name_precedence.md).
        if isinstance(node, fast.FunctionDeclaration):
            name = getattr(node, "name", None)
            if isinstance(name, str) and name:
                g.functions.add(name)
                # Post-desugar impl methods are __impl$Trait$Type$method;
                # the bare method name is reachable in method position and
                # as a UFCS plain call.
                parsed = parse_impl_method_name(name)
                if parsed is not None:
                    g.methods.add(parsed[2])
                    g.types.add(parsed[1])
                    g.impl_types.add(parsed[1])
        # --- FFI: `extern "C" { fn malloc(..); type FILE; }` ---------------
        elif isinstance(node, ExternFunctionDeclaration):
            name = getattr(node, "name", None)
            if isinstance(name, str) and name:
                g.functions.add(name)
        elif isinstance(node, ExternTypeDeclaration):
            name = getattr(node, "name", None)
            if isinstance(name, str) and name:
                g.types.add(name)
        # --- types (usable in value position: `Vec.new()`, `Buffer.new()`) -
        elif isinstance(node, fast.StructDefinition):
            _add(g.types, getattr(node, "name", None))
        elif isinstance(node, fast.EnumDefinition):
            _add(g.types, getattr(node, "name", None))
            for v in getattr(node, "variants", None) or []:
                _add(g.variants, getattr(v, "name", None))
        elif isinstance(node, fast.TypeDefinition):
            _add(g.types, getattr(node, "name", None))
        elif isinstance(node, fast.TypeAlias):
            _add(g.types, getattr(node, "name", None))
        # --- traits: the type name AND its method names (UFCS fallback) ----
        elif isinstance(node, fast.InterfaceDefinition):
            _add(g.types, getattr(node, "name", None))
            for m in getattr(node, "methods", None) or []:
                _add(g.methods, getattr(m, "name", None))
        # --- pre-desugar impl blocks (the prelim analysis round sees these) -
        elif isinstance(node, fast.Implementation):
            _add(g.impl_types, _base_name(getattr(node, "type_name", None)))
            for m in getattr(node, "methods", None) or []:
                _add(g.methods, getattr(m, "name", None))
        # --- effects: the effect type name and its operation names ---------
        # An operation is callable unqualified (`emit(x)` == `perform
        # Emit.emit(x)`), which `hir._from_orig_expr` recognizes by name.
        elif isinstance(node, fast.EffectDeclaration):
            _add(g.types, getattr(node, "name", None))
            for op in getattr(node, "operations", None) or []:
                _add(g.effect_ops, getattr(op, "name", None))
        # --- imports: every local name an import statement introduces ------
        # For a RESOLVED module the loader has already rewritten references
        # to dotted paths, so these bindings are usually unused; for the
        # `std.*` external placeholder (std.simd, std.effects, ... — modules
        # with no file under the stdlib root) nothing is rewritten and the
        # bare imported name is all the program has.  Both are the same
        # category: a name an import statement brought into scope.
        elif isinstance(node, fast.Import):
            alias = getattr(node, "alias", None)
            path = getattr(node, "module_path", None) or []
            local = alias or (str(path[-1]) if path else None)
            _add(g.imported, local)
        elif isinstance(node, fast.FromImport):
            # `names` is a list of (name, alias) pairs; the alias, when
            # present, is the local spelling.
            for entry in getattr(node, "names", None) or []:
                if isinstance(entry, (tuple, list)):
                    if not entry:
                        continue
                    alias = entry[1] if len(entry) > 1 else None
                    _add(g.imported, alias if alias else entry[0])
                elif entry is not None:
                    _add(g.imported, entry)
    # Module-level `let` constants are bound by the scoped walk, which is
    # the only place that can tell a module-level binding from a local.
    g.freeze()
    return g


def _base_name(type_node: Any) -> str | None:
    """The bare name of a (possibly generic) type reference."""
    if type_node is None:
        return None
    for attr in ("name", "type_constructor", "base_type"):
        inner = getattr(type_node, attr, None)
        if isinstance(inner, str) and inner:
            return inner
        if inner is not None and not isinstance(inner, str):
            nested = _base_name(inner)
            if nested:
                return nested
    if isinstance(type_node, str):
        return type_node.split("[")[0].split("<")[0] or None
    text = str(type_node).split("[")[0].split("<")[0]
    return text or None


def _add(target: set[str], name: Any) -> None:
    if isinstance(name, str) and name:
        target.add(name)
    elif name is not None:
        text = str(name)
        if text:
            target.add(text)


# ---------------------------------------------------------------------------
# The scoped walk
# ---------------------------------------------------------------------------

class _Resolver:
    def __init__(self, globals_: _GlobalNames, file_path: str | None) -> None:
        self.g = globals_
        self.file_path = file_path
        self.errors: list[BorrowError] = []
        # Lexical scopes of local value names (params, lets, pattern
        # bindings, loop variables, handler-arm parameters).
        self.scopes: list[set[str]] = [set()]
        # Type-parameter names of the enclosing generic declarations.  They
        # are in scope as VALUES: a const generic (`implement<const N: int>`)
        # is bound to the receiver's runtime dimension at method entry
        # (hir.build's `_const_dims`), and a plain type parameter is the
        # argument of the `type_of` reflection intrinsic.
        self.type_params: list[set[str]] = [set()]
        self._reported: set[int] = set()

    # -- scope helpers --------------------------------------------------
    def push(self) -> None:
        self.scopes.append(set())

    def pop(self) -> None:
        self.scopes.pop()

    def bind(self, name: Any) -> None:
        if isinstance(name, str) and name:
            self.scopes[-1].add(name)

    def in_scope(self, name: str) -> bool:
        if any(name in s for s in self.scopes):
            return True
        if any(name in tp for tp in self.type_params):
            return True
        return False

    def local_names(self) -> set[str]:
        out: set[str] = set()
        for s in self.scopes:
            out |= s
        for tp in self.type_params:
            out |= tp
        return out

    # -- diagnostics ----------------------------------------------------
    def report(self, node: Any, what: str, name: str, candidates: set[str]) -> None:
        if id(node) in self._reported:
            return
        self._reported.add(id(node))
        message = f"undefined {what} '{name}'"
        hint = _suggest(name, candidates)
        if hint is not None:
            message += f"; did you mean '{hint}'?"
        loc = getattr(node, "location", None)
        self.errors.append(BorrowError(
            message=message,
            node_id=-1,
            kind=UNRESOLVED_NAME_KIND,
            variable=name,
            location=loc,
        ))

    def check_value(self, node: Any, name: Any) -> None:
        if not isinstance(name, str) or not name:
            return
        if name == "_" or name.startswith("__"):
            return  # `_` is a wildcard; `__` is the compiler's namespace
        if self.in_scope(name) or name in self.g.value_names:
            return
        self.report(node, "variable", name,
                    self.local_names() | self.g.value_names)

    def check_callee(self, node: Any, name: Any) -> None:
        if not isinstance(name, str) or not name:
            return
        if name.startswith("__") or "." in name:
            return  # the compiler's namespace; dotted forms go through
                    # check_qualified_call, which knows their dispatch rules
        if self.in_scope(name) or name in self.g.callable_names:
            return
        self.report(node, "function", name,
                    self.local_names() | self.g.callable_names)

    def check_qualified_call(self, node: Any, parts: list[str]) -> None:
        """`a.b(...)` — mirrors `hir._from_orig_expr`'s dispatch for
        `QualifiedFunctionCall`, which is the only place that decides what
        the dotted form means.

        Four outcomes, in HIR's own order:
          1. `Enum.Variant(x)`     — a variant constructor;
          2. `Type.method(x)`      — a static impl call (`__static$Type$m`);
          3. `recv.method(x)`      — a receiver call, where `recv` is an
             ordinary VALUE name, so that name is what gets checked;
          4. anything else         — a plain call of the dotted name, which
             is what the module system produces for `mod.f` after renaming.
        """
        if not parts:
            return
        root, last = parts[0], parts[-1]
        if root.startswith("__"):
            return
        # 1. Enum.Variant(...)
        if len(parts) == 2 and last in self.g.variants:
            return
        # 2. Static impl-method call on the type itself (`Buffer.new(1024)`).
        if (len(parts) == 2 and root in self.g.impl_types
                and last in self.g.methods):
            return
        # 3. A method name (trait, impl or runtime builtin) as the last
        # segment means the leading segments are a RECEIVER path, so the
        # root is an ordinary value name. The exception is a known type with
        # no impls (`Vec.new()`): that is a static call on the type, which
        # case 4 resolves as a dotted name.
        base_is_foreign_type = root in self.g.types and root not in self.g.impl_types
        if not base_is_foreign_type and (last in self.g.methods
                                         or last in _BUILTIN_METHODS):
            self.check_value(node, root)
            return
        # 4. Plain dotted callee: the whole dotted spelling IS the function
        # name, because the module resolver renames an imported module's
        # functions to dotted paths (`std.vec.map`). Accepted when
        #   * some declaration in the unit carries that dotted name, or
        #   * it is one of the runtime's own dotted builtins (`Vec.new`), or
        #   * the root is a local name an import introduced — an unresolved
        #     `std.*` module resolves to the external placeholder, whose
        #     names are deliberately left untouched for the runtime to
        #     answer (module_loader's STD_ROOT note), or
        #   * the last segment is an effect operation (`Eff.op(x)` is the
        #     bare-call spelling of a perform).
        dotted = ".".join(parts)
        if (dotted in self.g.functions or dotted in self.g.imported
                or dotted in DOTTED_BUILTIN_NAMES
                or root in self.g.imported or last in self.g.effect_ops):
            return
        self.report(node, "function", dotted,
                    self.g.functions | self.g.imported | DOTTED_BUILTIN_NAMES)

    def check_assign_target(self, node: Any, target: Any) -> None:
        """`x = e` where the target is a bare name.

        The name must already be bound: MIR's `Assign` writes into
        `env[name]` unconditionally, so assigning to an undeclared name
        created a fresh slot nobody could ever read.  Compound targets
        (`x.f = e`, `v[i] = e`) arrive as nodes and are checked through
        their own branches; a defensive split is kept here because desugar
        passes may synthesize a dotted string target."""
        if not isinstance(target, str) or not target:
            return
        root = target.split(".")[0].split("[")[0].strip()
        if not root or root.startswith("__") or root == "_":
            return
        if self.in_scope(root) or root in self.g.value_names:
            return
        self.report(node, "variable", root,
                    self.local_names() | self.g.value_names)

    # -- pattern binders -------------------------------------------------
    def bind_pattern(self, p: Any) -> None:
        """Bind every name a pattern introduces.

        Mirrors `hir.HIRBuilder._convert_pattern`: the parser's arm grammar
        is `expression => body`, so patterns arrive as expression nodes and
        a bare name is a binding unless it names a variant.
        """
        if p is None:
            return
        if isinstance(p, str):
            if p not in self.g.variants:
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
            name = getattr(p, "name", None)
            if isinstance(name, str) and name not in self.g.variants:
                self.bind(name)
            return
        if isinstance(p, fast.QualifiedName):
            parts = [str(x) for x in (getattr(p, "parts", None) or [])]
            if len(parts) == 1 and parts[0] not in self.g.variants:
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
        # A tuple pattern (`(x, y) => ..`) binds each element, exactly like a
        # list pattern.  Without this every tuple-pattern binder read as an
        # undefined variable in the arm body.
        if isinstance(p, (fast.ListLiteral, fast.TupleLiteral)):
            for e in getattr(p, "elements", None) or []:
                self.bind_pattern(e)
            return
        if isinstance(p, fast.SpreadElement):
            self.bind_pattern(getattr(p, "expression", None))
            return
        # Literals, wildcards, field accesses (`Color.Red`) and anything the
        # HIR pattern converter rejects outright bind nothing.

    # -- declarations that introduce type parameters ---------------------
    @staticmethod
    def _type_param_names(node: Any) -> set[str]:
        names: set[str] = set()
        for source in ("type_params", "_impl_type_params"):
            for tp in getattr(node, source, None) or []:
                name = tp if isinstance(tp, str) else getattr(tp, "name", None)
                if isinstance(name, str) and name:
                    names.add(name)
        # Const-generic receiver dimensions recorded by the impl desugar
        # (`implement<const N: int> ... vector[T, N]`): HIR binds N at method
        # entry to the receiver's runtime length, so the body may read it.
        for (dim_name, _idx) in getattr(node, "_const_dims", None) or ():
            if isinstance(dim_name, str) and dim_name:
                names.add(dim_name)
        return names

    # -- the walk --------------------------------------------------------
    def visit_all(self, nodes: Iterable[Any]) -> None:
        for n in nodes:
            self.visit(n)

    def visit(self, node: Any) -> None:  # noqa: C901 - one branch per form
        if node is None or not _is_node(node):
            return

        # ---------- references ----------
        if isinstance(node, fast.Variable):
            self.check_value(node, getattr(node, "name", None))
            return
        if isinstance(node, fast.QualifiedName):
            parts = [str(x) for x in (getattr(node, "parts", None) or [])]
            if len(parts) == 1:
                self.check_value(node, parts[0])
            elif len(parts) >= 2:
                # `a.b.c` is a field chain rooted at `a` UNLESS it spells
                # `Enum.Variant`; either way only the root is a value name.
                if not (len(parts) == 2 and parts[1] in self.g.variants):
                    self.check_value(node, parts[0])
            return
        if isinstance(node, (fast.BorrowShared, fast.BorrowUnique, fast.Move,
                             fast.BorrowExpression)):
            var = getattr(node, "variable", None)
            if isinstance(var, str):
                self.check_value(node, var)
            else:
                self.visit(var)
            return
        if isinstance(node, fast.ExclaveExpression):
            inner = getattr(node, "expression", None)
            if isinstance(inner, str):
                self.check_value(node, inner)
            else:
                self.visit(inner)
            return
        if isinstance(node, fast.FunctionCall):
            self.check_callee(node, getattr(node, "name", None))
            self.visit_all(getattr(node, "arguments", None) or [])
            return
        if isinstance(node, fast.QualifiedFunctionCall):
            self.check_qualified_call(
                node, [str(p) for p in (getattr(node, "parts", None) or [])])
            self.visit_all(getattr(node, "arguments", None) or [])
            return
        if isinstance(node, fast.Assignment):
            self.visit(getattr(node, "expression", None))
            target = getattr(node, "name", None)
            if isinstance(target, str):
                self.check_assign_target(node, target)
            else:
                # `x.f = e` / `v[i] = e`: the target is a FieldAccess or
                # IndexExpression node, whose own branch checks its root.
                self.visit(target)
            return
        if isinstance(node, fast.FieldAccess):
            # The parser stores the base of `c.name` as a RAW STRING, so it
            # is not reachable through structural recursion; `hir` turns it
            # into a `Var` read of exactly that name.  `Color.Red` is the
            # exception: it spells a nullary enum variant, not a field of a
            # variable called `Color`.
            base = getattr(node, "base", None)
            fields = [str(f) for f in (getattr(node, "fields", None) or ())]
            if isinstance(base, str):
                if not (len(fields) == 1 and fields[0] in self.g.variants):
                    self.check_value(node, base)
            else:
                self.visit(base)
            return

        # ---------- binders ----------
        if isinstance(node, fast.FunctionDeclaration):
            self.visit_function(node)
            return
        if isinstance(node, fast.LambdaExpression):
            self.visit_lambda(node)
            return
        if isinstance(node, fast.Implementation):
            self.visit_impl(node)
            return
        if isinstance(node, fast.LetStatement):
            for b in getattr(node, "bindings", None) or []:
                self.visit(getattr(b, "initializer", None))
                self.bind(getattr(b, "identifier", None))
            return
        if isinstance(node, fast.LetBinding):
            self.visit(getattr(node, "initializer", None))
            self.bind(getattr(node, "identifier", None))
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
        if isinstance(node, fast.HandleEffect):
            for c in getattr(node, "handler", None) or []:
                self.visit(c)
            self.visit_body(getattr(node, "continuation", None))
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
            self.visit_body(getattr(node, "subject", None))
            for arm in getattr(node, "arms", None) or []:
                pattern, body = _case_parts(arm)
                self.push()
                self.bind_handler_arm(pattern)
                self.visit_body(body)
                self.pop()
            return
        if isinstance(node, fast.Comprehension):
            self.visit(getattr(node, "iterable", None))
            self.push()
            for t in _as_list(getattr(node, "targets", None)):
                self.bind_pattern(t)
            self.visit(getattr(node, "expression", None))
            self.pop()
            return

        # ---------- declarations that bind nothing in value scope ----------
        if isinstance(node, (fast.StructDefinition, fast.EnumDefinition,
                             fast.InterfaceDefinition, fast.EffectDeclaration,
                             fast.TypeDefinition, fast.TypeAlias,
                             fast.Import, fast.FromImport,
                             fast.ExportDeclaration, fast.VisibilityRules,
                             ExternBlock, ExternFunctionDeclaration,
                             ExternTypeDeclaration)):
            # Their interiors are type-level or already consumed by an
            # earlier pass; an effect operation's DEFAULT expression is the
            # one value-level body among them.
            if isinstance(node, fast.EffectDeclaration):
                for op in getattr(node, "operations", None) or []:
                    default = getattr(op, "_default_expr", None)
                    if default is not None:
                        self.push()
                        for p in getattr(op, "params", None) or []:
                            self.bind(getattr(p, "name", None))
                        self.visit(default)
                        self.pop()
            return

        # ---------- everything else: structural recursion ----------
        self.visit_all(_sub_nodes(node, skip=_TYPE_ATTRS))

    # -- helpers ---------------------------------------------------------
    def visit_body(self, body: Any) -> None:
        """Visit a body that may be a node, a list of statements, or None."""
        if body is None:
            return
        if isinstance(body, (list, tuple)):
            self.push()
            self.visit_all(body)
            self.pop()
            return
        self.visit(body)

    def bind_handler_arm(self, pattern: Any) -> None:
        """Bind the parameters of `perform Eff.op(p, q) => body`."""
        if pattern is None:
            return
        if isinstance(pattern, fast.PerformEffect):
            for a in getattr(pattern, "arguments", None) or []:
                self.bind_pattern(a)
            return
        if isinstance(pattern, (fast.FunctionCall, fast.QualifiedFunctionCall)):
            for a in getattr(pattern, "arguments", None) or []:
                self.bind_pattern(a)
            return
        self.bind_pattern(pattern)

    def visit_function(self, node: Any) -> None:
        self.push()
        self.type_params.append(self._type_param_names(node))
        for p in getattr(node, "params", None) or []:
            self.bind(p if isinstance(p, str) else getattr(p, "name", None))
        # A method body may read `self` even when the receiver is implicit.
        self.bind("self")
        self.visit_body(getattr(node, "body", None))
        self.type_params.pop()
        self.pop()

    def visit_lambda(self, node: Any) -> None:
        self.push()
        for p in getattr(node, "params", None) or []:
            self.bind(p if isinstance(p, str) else getattr(p, "name", None))
        self.visit_body(getattr(node, "body", None))
        self.pop()

    def visit_impl(self, node: Any) -> None:
        """Pre-desugar `implement` block (only the preliminary analysis
        round sees these; the final round sees mangled functions)."""
        self.type_params.append(self._type_param_names(node))
        for m in getattr(node, "methods", None) or []:
            self.visit(m)
        self.type_params.pop()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _case_parts(case: Any) -> tuple[Any, Any]:
    """(pattern, body) of a match case / handler arm in any shape the
    parser and the desugar passes produce."""
    if isinstance(case, (tuple, list)) and len(case) >= 2:
        return case[0], case[1]
    return (getattr(case, "pattern", None), getattr(case, "body", None))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def check_names(root: Any, file_path: str | None = None) -> list[BorrowError]:
    """Structured `type-unresolved-name` diagnostics for `root`, if any.

    `root` is the MUTABLE, post-desugar AST (`pipeline.PhaseContext.program`).
    """
    globals_ = collect_global_names(root)
    resolver = _Resolver(globals_, file_path)
    # Module-level `let` bindings are program-wide constants (they become
    # globals through __module_init), so bind them all before walking.
    _bind_module_constants(root, resolver)
    resolver.visit_all(_module_statements(root))
    return resolver.errors


def _module_statements(root: Any) -> list[Any]:
    """Top-level statements of every module in the program."""
    out: list[Any] = []
    for node in _walk_all(root):
        if isinstance(node, fast.ModuleBody):
            out.extend(getattr(node, "statements", None) or [])
    if out:
        return out
    if isinstance(root, fast.Program):
        return list(getattr(root, "statements", None) or [])
    return [root]


def _bind_module_constants(root: Any, resolver: _Resolver) -> None:
    """Bind the names of module-level `let` statements.

    Only DIRECT statements of a module body are module constants — the same
    rule `hir.HIRBuilder.build` uses when it hoists them into
    `__module_init`.
    """
    for stmt in _module_statements(root):
        if isinstance(stmt, fast.LetStatement):
            for b in getattr(stmt, "bindings", None) or []:
                resolver.bind(getattr(b, "identifier", None))
