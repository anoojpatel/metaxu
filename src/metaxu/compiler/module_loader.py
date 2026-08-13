"""Multi-file module resolution for the Metaxu pipeline.

This pass runs immediately after parsing (before desugaring/freezing) and
turns the surface module constructs — `module a.b { ... }` blocks,
`import a.b [as x]` and `from a.b import x [as y]` — into a single merged
Program the rest of the pipeline already understands:

- every module's top-level *functions* are renamed to their fully qualified
  dotted name (`math.vector.dot`), except the entry module (the root file's
  own top-level scope, conventionally named "main"), whose functions keep
  their plain names so `main` stays the entry point;
- every reference (unqualified `FunctionCall`, qualified
  `QualifiedFunctionCall`) inside a module is resolved against that
  module's scope — own declarations first, then its imports — and
  rewritten to the final symbol name;
- types, traits and effects are merged into a single global namespace
  (they are NOT renamed); declaring the same type/trait/effect name in two
  different modules is a loud CompileError;
- visibility (`export { ... }` lists, `visibility { name: private }`
  blocks) is enforced at import sites and at qualified references.

See docs/modules_implementation.md for the full semantics contract.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import metaxu.metaxu_ast as fast
from metaxu.errors import CompileError


# The reserved namespace for the standard library. Imports under `std.*`
# first resolve to real files under the stdlib root (see _stdlib_dir):
# `import std.fail` loads `<stdlib>/fail.mx` exactly like any other module
# file. Names with no file under the stdlib root (std.simd, std.matrix,
# std.geometry, ...) fall back to the historical external-placeholder
# behavior: the import succeeds, no names are rewritten, and calls into
# the placeholder fall through to interpreter builtins or fail loudly at
# run time — documented examples importing those keep compiling.
STD_ROOT = "std"


def _stdlib_dir() -> str | None:
    """Directory holding the standard library sources (`std/*.mx`).

    Resolution order:
    1. the METAXU_STD_PATH environment variable, when set and a directory;
    2. the repo-layout default: the `std/` directory at the repository
       root, located relative to this file (src/metaxu/compiler/ -> ../../../std).
    Returns None when neither exists (std.* imports then all resolve to
    the external placeholder, the pre-stdlib behavior).
    """
    env = os.environ.get("METAXU_STD_PATH")
    if env:
        return env if os.path.isdir(env) else None
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.normpath(os.path.join(here, "..", "..", "..", "std"))
    return cand if os.path.isdir(cand) else None


def _module_error(message: str, notes: list[str] | None = None) -> CompileError:
    return CompileError(message=message, error_type="ModuleError",
                        notes=notes or [])


@dataclass
class ModuleInfo:
    path: str                       # fully qualified dotted path
    nodes: list = field(default_factory=list)   # fast.Module nodes merged into this path
    external: bool = False          # unresolved std.* placeholder
    functions: dict = field(default_factory=dict)   # name -> FunctionDeclaration
    # module-level `let` bindings (module constants). Like types/traits/
    # effects they live in ONE global namespace (they are never renamed);
    # declaring the same constant in two modules is a loud CompileError.
    constants: set = field(default_factory=set)
    types: set = field(default_factory=set)
    traits: set = field(default_factory=set)
    effects: set = field(default_factory=set)
    exports: list = field(default_factory=list)      # [(name, alias)] from export {...}
    has_export_list: bool = False
    visibility_rules: dict = field(default_factory=dict)  # name -> "public"|"private"|"protected"
    # import bindings visible inside this module:
    #   local name -> ("module", target_path) | ("symbol", target_path, symbol_name)
    bindings: dict = field(default_factory=dict)
    # names re-exported via `public import` / `public from ... import`
    reexports: dict = field(default_factory=dict)   # name -> same binding shape
    imports: list = field(default_factory=list)      # raw Import/FromImport nodes

    def declares(self, name: str) -> bool:
        return (name in self.functions or name in self.types
                or name in self.traits or name in self.effects
                or name in self.constants)

    def is_public(self, name: str) -> bool:
        """Effective visibility of a declared symbol.

        Rules (documented in docs/modules_implementation.md):
        1. an explicit `visibility { name: ... }` entry wins
           (protected is treated as private for cross-module access);
        2. else, if the module has an `export { ... }` list, only listed
           names are public;
        3. else everything top-level is public.
        """
        rule = self.visibility_rules.get(name)
        if rule is not None:
            return rule == "public"
        if self.has_export_list:
            return any(exp_name == name for (exp_name, _alias) in self.exports)
        return True


class ModuleResolver:
    def __init__(self, program: fast.Program, file_path: str):
        self.program = program
        self.file_path = file_path
        self.root_dir = None
        if file_path and not file_path.startswith("<"):
            d = os.path.dirname(os.path.abspath(file_path))
            if os.path.isdir(d):
                self.root_dir = d
        self.registry: dict[str, ModuleInfo] = {}
        self.entry_path: str | None = None
        self.import_edges: list[tuple[str, str]] = []
        self.loaded_files: dict[str, str] = {}   # abs file path -> module path
        # from-import checks deferred until every module (and its own
        # imports, which populate re-export tables) has been processed:
        # [(target_path, symbol_name, importer_path)]
        self.pending_import_checks: list[tuple[str, str, str]] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _info(self, path: str) -> ModuleInfo:
        info = self.registry.get(path)
        if info is None:
            info = ModuleInfo(path=path)
            self.registry[path] = info
        return info

    def _register_module_node(self, node: fast.Module, path: str) -> ModuleInfo:
        """Register a module AST node (and its nested module blocks) under
        `path`. Nested `module x.y { ... }` blocks always register at their
        *declared* absolute dotted path (module paths are absolute)."""
        info = self._info(path)
        if info.external:
            info.external = False
        info.nodes.append(node)
        body = getattr(node, "body", None)
        statements = list(getattr(body, "statements", []) or []) if body else []

        # exports / visibility attached by the parser
        exports = list(getattr(body, "exports", []) or []) if body else []
        if exports:
            info.exports.extend(exports)
            info.has_export_list = True
        vis = getattr(body, "visibility_rules", None) if body else None
        if vis is not None:
            info.visibility_rules.update(getattr(vis, "rules", {}) or {})

        for stmt in statements:
            if isinstance(stmt, fast.Module):
                self._register_module_node(stmt, str(stmt.name))
            elif isinstance(stmt, fast.VisibilityRules):
                # visibility blocks can appear as plain statements in the
                # synthesized file wrapper module
                info.visibility_rules.update(getattr(stmt, "rules", {}) or {})
            elif isinstance(stmt, fast.ExportDeclaration):
                # file-level `export { ... }` list (files are modules too)
                info.exports.extend(stmt.names or [])
                info.has_export_list = True
            elif isinstance(stmt, (fast.Import, fast.FromImport)):
                info.imports.append(stmt)
            elif isinstance(stmt, fast.FunctionDeclaration):
                fname = str(getattr(stmt, "name", "") or "")
                if fname in info.functions and info.functions[fname] is not stmt:
                    raise _module_error(
                        f"duplicate function '{fname}' in module '{path}'")
                info.functions[fname] = stmt
            elif isinstance(stmt, fast.LetStatement):
                # module-level constants (initialized before the entry point
                # by the synthesized __module_init; see compiler/hir.py)
                for b in (getattr(stmt, "bindings", None) or []):
                    cname = getattr(b, "identifier", None)
                    if cname:
                        info.constants.add(str(cname))
            elif isinstance(stmt, (fast.StructDefinition, fast.EnumDefinition)):
                info.types.add(str(getattr(stmt, "name", "") or ""))
            elif isinstance(stmt, fast.InterfaceDefinition):
                info.traits.add(str(getattr(stmt, "name", "") or ""))
            elif isinstance(stmt, fast.EffectDeclaration):
                info.effects.add(str(getattr(stmt, "name", "") or ""))
        return info

    # ------------------------------------------------------------------
    # Import resolution / file loading
    # ------------------------------------------------------------------

    def _resolve_import_path(self, importer: ModuleInfo,
                             raw_path: list[str], relative_level: int) -> str:
        parts = [str(p) for p in raw_path]
        if relative_level:
            # `.x`  (level 1) = child of the current module,
            # `..x` (level 2) = sibling (child of the parent module), etc.
            # This matches examples/03: `from ..vector import ...` inside
            # `module math.transform` resolves to `math.vector`.
            base = importer.path.split(".")
            drop = relative_level - 1
            if drop > len(base):
                raise _module_error(
                    f"relative import in module '{importer.path}' escapes the "
                    f"module root ({'.' * relative_level}{'.'.join(parts)})")
            base = base[: len(base) - drop] if drop else base
            parts = base + parts
        return ".".join(parts)

    def _load_module(self, path: str, importer: ModuleInfo) -> ModuleInfo:
        """Ensure `path` is present in the registry, loading it from a file
        if necessary."""
        info = self.registry.get(path)
        if info is not None and (info.external or info.nodes):
            return info
        root = path.split(".", 1)[0]
        if root == STD_ROOT:
            # Real stdlib file first (std.fail -> <stdlib>/fail.mx), then
            # the external placeholder for unresolved std.* names.
            std_dir = _stdlib_dir()
            rel_parts = path.split(".")[1:]
            if std_dir is not None and rel_parts:
                candidate = os.path.join(std_dir, *rel_parts) + ".mx"
                if os.path.isfile(candidate):
                    return self._load_module_file(os.path.abspath(candidate), path)
            info = self._info(path)
            info.external = True
            return info
        if self.root_dir is None:
            raise _module_error(
                f"cannot resolve import of module '{path}' from module "
                f"'{importer.path}': the source has no on-disk location "
                f"(compiled from memory) and '{path}' is not declared in-file")
        rel = os.path.join(*path.split(".")) + ".mx"
        candidate = os.path.join(self.root_dir, rel)
        if not os.path.isfile(candidate):
            raise _module_error(
                f"module '{path}' not found (imported from module "
                f"'{importer.path}')",
                notes=[f"looked for {candidate}",
                       f"module paths resolve relative to the root file's "
                       f"directory: {self.root_dir}"])
        return self._load_module_file(os.path.abspath(candidate), path)

    def _load_module_file(self, candidate: str, path: str) -> ModuleInfo:
        """Parse the module file at `candidate` and register it under `path`."""
        if candidate in self.loaded_files:
            # Same file already loaded under another module path: alias it.
            return self._info(self.loaded_files[candidate])
        from metaxu.parser import Parser
        source = open(candidate).read()
        parsed = Parser().parse(source, file_path=candidate)
        if not isinstance(parsed, fast.Module):
            raise _module_error(
                f"module file {candidate} did not parse to a module")
        # The parser wraps a file's statements in a synthesized module named
        # "main"; the file *is* the module named by its path.
        parsed.name = path
        self.loaded_files[candidate] = path
        info = self._register_module_node(parsed, path)
        # merge the loaded file's top-level module into the program so the
        # rest of the pipeline sees its declarations
        self.program.add_statements([parsed])
        return info

    def _process_imports(self) -> None:
        worklist = list(self.registry.values())
        seen: set[int] = set()
        while worklist:
            info = worklist.pop(0)
            if id(info) in seen or info.external:
                continue
            seen.add(id(info))
            for imp in info.imports:
                before = set(self.registry)
                if isinstance(imp, fast.Import):
                    target_path = ".".join(str(p) for p in imp.module_path)
                    target = self._load_module(target_path, info)
                    local = imp.alias or str(imp.module_path[-1])
                    binding = ("module", target.path)
                    info.bindings[local] = binding
                    if getattr(imp, "is_public", False):
                        info.reexports[local] = binding
                    self.import_edges.append((info.path, target.path))
                elif isinstance(imp, fast.FromImport):
                    target_path = self._resolve_import_path(
                        info, imp.module_path, getattr(imp, "relative_level", 0))
                    target = self._load_module(target_path, info)
                    self.import_edges.append((info.path, target.path))
                    for (name, alias) in imp.names:
                        name = str(name)
                        local = str(alias) if alias else name
                        if not target.external:
                            # Deferred: the target's own imports may not have
                            # been processed yet, so its re-export table
                            # (public import / public from-import) can still
                            # be empty here. Checking after the worklist
                            # drains sees the complete picture.
                            self.pending_import_checks.append(
                                (target.path, name, info.path))
                        binding = ("symbol", target.path, name)
                        info.bindings[local] = binding
                        if getattr(imp, "is_public", False):
                            info.reexports[local] = binding
                # newly loaded modules need their own imports processed
                for new_path in set(self.registry) - before:
                    worklist.append(self.registry[new_path])
        for (target_path, name, importer_path) in self.pending_import_checks:
            self._check_importable(self.registry[target_path], name,
                                   self.registry[importer_path])

    def _chase_symbol(self, target: ModuleInfo, name: str,
                      _seen: set[tuple[str, str]] | None = None):
        """Follow `name` through `target`'s re-export chain to the module
        that actually declares it.

        Returns (final_info, final_name) — final_info.external is True when
        the chain ends in a std.* placeholder — or None when the name is
        neither declared nor re-exported anywhere along the chain."""
        seen = _seen or set()
        if (target.path, name) in seen:
            return None
        seen.add((target.path, name))
        if target.external or target.declares(name):
            return target, name
        b = target.reexports.get(name)
        if b is None or b[0] != "symbol":
            return None
        nxt = self.registry.get(b[1])
        if nxt is None:
            return None
        return self._chase_symbol(nxt, b[2], seen)

    def _check_importable(self, target: ModuleInfo, name: str,
                          importer: ModuleInfo) -> None:
        if target.declares(name):
            if not target.is_public(name):
                raise _module_error(
                    f"cannot import private symbol '{name}' from module "
                    f"'{target.path}' (imported by module '{importer.path}')",
                    notes=[f"'{name}' is not exported by '{target.path}'"])
            return
        # re-exported names (public import / public from-import), possibly
        # through a chain of re-exporting modules (e.g. std.prelude)
        if name in target.reexports and (
                target.reexports[name][0] != "symbol"
                or self._chase_symbol(target, name) is not None):
            return
        raise _module_error(
            f"module '{target.path}' has no symbol '{name}' "
            f"(imported by module '{importer.path}')")

    # ------------------------------------------------------------------
    # Cycle detection
    # ------------------------------------------------------------------

    def _check_cycles(self) -> None:
        graph: dict[str, set[str]] = {}
        for (a, b) in self.import_edges:
            graph.setdefault(a, set()).add(b)
        WHITE, GREY, BLACK = 0, 1, 2
        color = {p: WHITE for p in self.registry}
        stack: list[str] = []

        def visit(p: str) -> None:
            color[p] = GREY
            stack.append(p)
            for q in sorted(graph.get(p, ())):
                if color.get(q, WHITE) == GREY:
                    cycle = stack[stack.index(q):] + [q]
                    raise _module_error(
                        "import cycle detected: " + " -> ".join(cycle),
                        notes=["break the cycle by moving shared declarations "
                               "into a module both sides can import"])
                if color.get(q, WHITE) == WHITE:
                    visit(q)
            stack.pop()
            color[p] = BLACK

        for p in list(color):
            if color[p] == WHITE:
                visit(p)

    # ------------------------------------------------------------------
    # Global type/trait/effect namespace (merged, collision-checked)
    # ------------------------------------------------------------------

    def _check_global_collisions(self) -> None:
        owner: dict[tuple[str, str], str] = {}
        for info in self.registry.values():
            for (kind, names) in (("type", info.types), ("trait", info.traits),
                                  ("effect", info.effects),
                                  ("constant", info.constants)):
                for name in names:
                    if not name:
                        continue
                    prev = owner.get((kind, name))
                    if prev is not None and prev != info.path:
                        raise _module_error(
                            f"{kind} '{name}' is declared in both module "
                            f"'{prev}' and module '{info.path}': types, "
                            "traits and effects share one global namespace "
                            "and may only be declared once",
                            notes=["rename one of the declarations"])
                    owner[(kind, name)] = info.path

    # ------------------------------------------------------------------
    # Renaming + reference rewriting
    # ------------------------------------------------------------------

    def _final_name(self, mod_path: str, name: str) -> str:
        if mod_path == self.entry_path:
            return name
        return f"{mod_path}.{name}"

    def _rename_functions(self) -> None:
        for info in self.registry.values():
            if info.external or info.path == self.entry_path:
                continue
            for (name, fn) in info.functions.items():
                fn.name = f"{info.path}.{name}"

    def _resolve_qualified(self, info: ModuleInfo, parts: list[str]):
        """Resolve a dotted reference against `info`'s scope.

        Returns (target_module_info, remaining_parts) when the leading parts
        name a module (via an import binding or the global registry), or
        None when they do not (e.g. `receiver.method(...)`)."""
        parts = [str(p) for p in parts]
        # (a) leading part is an import binding
        b = info.bindings.get(parts[0])
        if b is not None:
            if b[0] == "module":
                expanded = b[1].split(".") + parts[1:]
            else:  # symbol binding used as a qualifier (module alias to symbol)
                expanded = b[1].split(".") + [b[2]] + parts[1:]
            parts = expanded
        # (b) longest registry prefix match
        for j in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:j])
            target = self.registry.get(prefix)
            if target is not None:
                return target, parts[j:]
        return None

    def _rewrite_call(self, node, info: ModuleInfo) -> None:
        if isinstance(node, fast.FunctionCall):
            name = getattr(node, "name", None)
            if not isinstance(name, str):
                return
            if name in info.functions:
                node.name = self._final_name(info.path, name)
                return
            b = info.bindings.get(name)
            if b is not None and b[0] == "symbol":
                target = self.registry.get(b[1])
                if target is None:
                    return
                resolved = self._chase_symbol(target, b[2])
                if resolved is None:
                    return
                final, fname = resolved
                if final.external:
                    return          # std.* placeholder: leave for builtins
                if fname in final.functions:
                    node.name = self._final_name(final.path, fname)
            return

        if isinstance(node, fast.QualifiedFunctionCall):
            parts = [str(p) for p in (getattr(node, "parts", None) or [])]
            if len(parts) < 2:
                return
            resolved = self._resolve_qualified(info, parts)
            if resolved is None:
                return
            target, rest = resolved
            if target.external:
                return              # std.* placeholder: leave untouched
            if not rest:
                return
            head = rest[0]
            if not target.declares(head):
                # A re-exported symbol referenced through the re-exporting
                # module (`prelude.try_opt(...)`): chase to the declarer.
                chased = (self._chase_symbol(target, head)
                          if head in target.reexports else None)
                if chased is not None:
                    final, fname = chased
                    if final.external:
                        return      # re-export of a std.* placeholder name
                    target, head = final, fname
                    rest = [head] + rest[1:]
                else:
                    raise _module_error(
                        f"module '{target.path}' has no symbol '{head}' "
                        f"(referenced from module '{info.path}' as "
                        f"'{'.'.join(parts)}')")
            if target.path != info.path and not target.is_public(head):
                raise _module_error(
                    f"symbol '{head}' of module '{target.path}' is private "
                    f"(referenced from module '{info.path}' as "
                    f"'{'.'.join(parts)}')")
            if head in target.functions and len(rest) == 1:
                # `mod.fn(args)` -> plain call of the final symbol name
                node.parts = [self._final_name(target.path, head)]
            else:
                # `mod.Type.method(args)` etc: strip the module qualifier,
                # types/traits/effects live in the global namespace
                node.parts = rest

    _SKIP_FIELDS = frozenset({"parent", "scope", "location"})

    def _walk_and_rewrite(self, value, info: ModuleInfo, memo: set[int]) -> None:
        if isinstance(value, fast.Module):
            return                  # nested modules are rewritten in their own scope
        if isinstance(value, fast.Node):
            if id(value) in memo:
                return
            memo.add(id(value))
            self._rewrite_call(value, info)
            for attr, v in list(vars(value).items()):
                if attr in self._SKIP_FIELDS:
                    continue
                self._walk_and_rewrite(v, info, memo)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                self._walk_and_rewrite(item, info, memo)
            return
        if isinstance(value, dict):
            for item in value.values():
                self._walk_and_rewrite(item, info, memo)

    def _rewrite_references(self) -> None:
        for info in self.registry.values():
            if info.external:
                continue
            memo: set[int] = set()
            for node in info.nodes:
                body = getattr(node, "body", None)
                for stmt in list(getattr(body, "statements", []) or []) if body else []:
                    self._walk_and_rewrite(stmt, info, memo)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def resolve(self) -> fast.Program:
        top_modules = [s for s in (self.program.statements or [])
                       if isinstance(s, fast.Module)]
        if not top_modules:
            return self.program
        # The first top-level module is the root file's wrapper: its own
        # top-level scope is the entry module.
        self.entry_path = str(top_modules[0].name)
        for m in top_modules:
            self._register_module_node(m, str(m.name))
        self._process_imports()
        self._check_cycles()
        self._check_global_collisions()
        self._rename_functions()
        self._rewrite_references()
        return self.program


def _has_module_constructs(program: fast.Program) -> bool:
    """Cheap scan: does this program use modules or imports at all?

    Single-file programs without imports skip resolution entirely, so they
    compile byte-for-byte exactly as before this pass existed."""
    for s in (program.statements or []):
        if not isinstance(s, fast.Module):
            continue
        body = getattr(s, "body", None)
        for stmt in (getattr(body, "statements", []) or []) if body else []:
            if isinstance(stmt, (fast.Module, fast.Import, fast.FromImport)):
                return True
    return False


def resolve_modules(program: fast.Program, file_path: str = "<mem>") -> fast.Program:
    """Resolve modules/imports in `program` in place (loading imported files
    relative to `file_path`'s directory) and return it. Programs that use no
    module constructs are returned untouched."""
    if not _has_module_constructs(program):
        return program
    return ModuleResolver(program, file_path).resolve()
