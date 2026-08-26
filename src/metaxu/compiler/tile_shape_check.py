"""Compile-time tile shape checking (docs/gpu_tiles.md, Stage 0).

Tiles carry their shape statically (`Tile[T, R, C]`), so shape misuse the
compiler can SEE must be a compile-time error, per the house rule that a
clear error beats a runtime surprise.  This pass runs from
``build_context_from_source`` over the MUTABLE post-desugar AST — the same
placement, and for the same reasons, as ``name_resolution`` — and files
kind ``type-tile-shape`` diagnostics on the structured channel (-2), which
``run_pipeline`` promotes to ``TypeCheckError`` like every other ``type-*``
kind.

Scope, stated honestly (zero-false-positive contract):

  * A bounded FORWARD shape analysis per function/lambda body: tile shapes
    propagate through `let` bindings and straight-line `Tile.*` dataflow.
    Anything the analysis does not understand (a call of a user function,
    a parameter, a capture, a rebind inside a branch) makes the value's
    shape UNKNOWN, and unknown always suppresses reporting — the
    interpreter's dynamic checks (mir_interp `_tile_*`) remain the strict
    backstop, and the native backend demotes tile ops whose shapes never
    become static (never wrong code).
  * Reported statically: wrong arities of `Tile.*` builtins, non-positive
    literal ctor shapes, elementwise shape/element-kind mismatches, dot
    inner-dimension disagreement, literal `Tile.get` indices out of a
    known shape's bounds, and scalar-kind mismatch in `Tile.scale` with a
    literal scalar.
  * Branch soundness: names assigned anywhere inside a branching
    construct (if/match/loops/try/handle) are invalidated BEFORE its
    subtrees are checked, so no branch-dependent shape is ever trusted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import metaxu.metaxu_ast as fast

from .frozen_borrow_checker import BorrowError
from .name_resolution import _sub_nodes, _walk_all

TILE_SHAPE_KIND = "type-tile-shape"

#: Tile builtin -> arity (dotted statics; docs/gpu_tiles.md).
TILE_ARITY = {
    "zeros": 2, "filled": 3, "arange": 2, "from_vec": 3, "to_vec": 1,
    "add": 2, "mul": 2, "scale": 2, "dot": 2, "sum": 1, "transpose": 1,
    "get": 3, "rows": 1, "cols": 1,
    # Buffer <-> tile boundary (Stage 1): strict load/store raise on any
    # out-of-range element; the masked forms (load_or reads `other`,
    # store_clipped writes nothing) are the kernel-side ragged-edge idiom.
    "load": 4, "load_or": 5, "store": 3, "store_clipped": 3,
    # 2D (row-strided) masked forms: element (i, j) maps to
    # off + i*stride + j — the tile-of-a-matrix idiom.
    "load_rows": 6, "store_rows": 4,
}


@dataclass(frozen=True)
class _TileInfo:
    """Static knowledge about one tile value.

    ``fkind`` is True (float elements), False (int elements) or None
    (statically unknown — e.g. `Tile.from_vec`, whose element kind is the
    Vec's runtime content).
    """
    rows: int
    cols: int
    fkind: Optional[bool]


def _lit_int(node: Any) -> Optional[int]:
    if isinstance(node, fast.Literal):
        v = node.value
        if isinstance(v, int) and not isinstance(v, bool):
            return v
    return None


def _lit_fkind(node: Any) -> Optional[bool]:
    """Element kind of a literal scalar argument, when decidable."""
    if isinstance(node, fast.Literal):
        v = node.value
        if isinstance(v, bool):
            return None
        if isinstance(v, float):
            return True
        if isinstance(v, int):
            return False
    return None


def _kname(fk: bool) -> str:
    return "float" if fk else "int"


class _TileShapeChecker:
    def __init__(self, file_path: str | None) -> None:
        self.file_path = file_path
        self.errors: list[BorrowError] = []

    # -- reporting ---------------------------------------------------------

    def report(self, node: Any, message: str) -> None:
        self.errors.append(BorrowError(
            message=message,
            node_id=-1,
            kind=TILE_SHAPE_KIND,
            variable="",
            location=getattr(node, "location", None),
        ))

    # -- entry -------------------------------------------------------------

    def check_program(self, root: Any) -> None:
        for fn in _walk_all(root):
            if isinstance(fn, fast.FunctionDeclaration):
                self._check_body(getattr(fn, "body", None), {})

    # -- statements (env threads through straight-line code) ----------------

    def _check_body(self, body: Any, env: dict) -> None:
        """``body`` is a statement list (FunctionDeclaration/lambda bodies)
        or a Block-like node carrying ``statements`` — or any expression."""
        if isinstance(body, (list, tuple)):
            for stmt in body:
                self._check_stmt(stmt, env)
            return
        stmts = getattr(body, "statements", None)
        if stmts is None:
            self._eval(body, env)
            return
        for stmt in stmts:
            self._check_stmt(stmt, env)

    def _assigned_names(self, node: Any) -> set[str]:
        out: set[str] = set()
        for n in _walk_all(node):
            if isinstance(n, fast.Assignment):
                name = getattr(n, "name", None)
                if isinstance(name, str) and "." not in name:
                    out.add(name)
        return out

    def _check_stmt(self, stmt: Any, env: dict) -> None:
        if isinstance(stmt, fast.LetStatement):
            for b in getattr(stmt, "bindings", None) or []:
                info = self._eval(getattr(b, "initializer", None), env)
                ident = getattr(b, "identifier", None)
                if isinstance(ident, str):
                    env[ident] = info
            return
        if isinstance(stmt, fast.Assignment):
            info = self._eval(getattr(stmt, "expression", None), env)
            name = getattr(stmt, "name", None)
            if isinstance(name, str) and "." not in name:
                env[name] = info
            return
        # Branching / looping / anything else: never trust a shape that a
        # subtree may rebind — invalidate first, THEN check the subtrees
        # (reads inside still use what remains known).
        for n in self._assigned_names(stmt):
            env[n] = None
        self._eval(stmt, env)

    # -- expressions --------------------------------------------------------

    def _eval(self, node: Any, env: dict) -> Optional[_TileInfo]:
        if node is None or not isinstance(node, fast.Node):
            return None
        if isinstance(node, fast.Variable):
            return env.get(getattr(node, "name", None))
        if isinstance(node, fast.LambdaExpression):
            # Fresh body, unknown captures: still checks literal misuse
            # inside, trusts nothing from the enclosing frame.
            body = getattr(node, "body", None)
            if body is not None:
                self._check_body(body, {})
            return None
        if isinstance(node, fast.FunctionDeclaration):
            # Nested declarations get their own fresh env from
            # check_program's walk; never thread the outer frame in.
            return None
        if isinstance(node, fast.QualifiedFunctionCall):
            parts = [str(p) for p in (getattr(node, "parts", None) or ())]
            args = list(getattr(node, "arguments", None) or [])
            if len(parts) == 2 and parts[0] == "Tile" \
                    and parts[1] in TILE_ARITY:
                return self._eval_tile_call(node, parts[1], args, env)
            for a in args:
                self._eval(a, env)
            return None
        if isinstance(node, fast.Block):
            # A block expression scopes its lets; evaluate against a COPY
            # so inner shadowing cannot leak out.
            inner = dict(env)
            stmts = getattr(node, "statements", None) or []
            for s in stmts[:-1]:
                self._check_stmt(s, inner)
            if stmts:
                return self._eval_stmt_value(stmts[-1], inner)
            return None
        # Generic: recurse into sub-nodes; the node's own value is unknown.
        for ch in _sub_nodes(node):
            self._eval(ch, env)
        return None

    def _eval_stmt_value(self, stmt: Any, env: dict) -> Optional[_TileInfo]:
        """Last statement of a block: its value is the block's value."""
        if isinstance(stmt, (fast.LetStatement, fast.Assignment)):
            self._check_stmt(stmt, env)
            return None
        return self._eval(stmt, env)

    # -- Tile.* ------------------------------------------------------------

    def _ctor_shape(self, node: Any, op: str, r_node: Any,
                    c_node: Any) -> Optional[tuple]:
        r, c = _lit_int(r_node), _lit_int(c_node)
        if r is None or c is None:
            return None  # dynamic shape: runtime backstop / native demotion
        if r <= 0 or c <= 0:
            self.report(node, f"Tile.{op}: tile shape must be positive, "
                              f"got {r}x{c}")
            return None
        return r, c

    def _eval_tile_call(self, node: Any, op: str, args: list,
                        env: dict) -> Optional[_TileInfo]:
        want = TILE_ARITY[op]
        if len(args) != want:
            self.report(node, f"Tile.{op}: expects {want} argument"
                              f"{'s' if want != 1 else ''}, got {len(args)}")
            for a in args:
                self._eval(a, env)
            return None
        infos = [self._eval(a, env) for a in args]

        if op in ("zeros", "arange"):
            shape = self._ctor_shape(node, op, args[0], args[1])
            if shape is None:
                return None
            # zeros builds float tiles, arange int tiles (mir_interp).
            return _TileInfo(shape[0], shape[1], op == "zeros")
        if op == "filled":
            shape = self._ctor_shape(node, op, args[0], args[1])
            if shape is None:
                return None
            return _TileInfo(shape[0], shape[1], _lit_fkind(args[2]))
        if op == "from_vec":
            shape = self._ctor_shape(node, op, args[1], args[2])
            if shape is None:
                return None
            return _TileInfo(shape[0], shape[1], None)
        if op == "load":
            shape = self._ctor_shape(node, op, args[2], args[3])
            if shape is None:
                return None
            return _TileInfo(shape[0], shape[1], None)
        if op == "load_or":
            shape = self._ctor_shape(node, op, args[2], args[3])
            if shape is None:
                return None
            return _TileInfo(shape[0], shape[1], _lit_fkind(args[4]))
        if op == "load_rows":
            shape = self._ctor_shape(node, op, args[3], args[4])
            if shape is None:
                return None
            return _TileInfo(shape[0], shape[1], _lit_fkind(args[5]))
        if op in ("add", "mul"):
            a, b = infos[0], infos[1]
            if a is not None and b is not None:
                if (a.rows, a.cols) != (b.rows, b.cols):
                    self.report(node, f"Tile.{op}: shape mismatch: "
                                      f"{a.rows}x{a.cols} vs {b.rows}x{b.cols}")
                    return None
                if a.fkind is not None and b.fkind is not None \
                        and a.fkind != b.fkind:
                    self.report(node, f"Tile.{op}: element kinds differ "
                                      f"({_kname(a.fkind)} vs "
                                      f"{_kname(b.fkind)})")
                    return None
                fk = a.fkind if a.fkind is not None else b.fkind
                return _TileInfo(a.rows, a.cols, fk)
            return None
        if op == "scale":
            t = infos[0]
            sk = _lit_fkind(args[1])
            if t is not None and t.fkind is not None and sk is not None \
                    and sk != t.fkind:
                self.report(node, f"Tile.scale: scalar kind must match tile "
                                  f"elements ({_kname(t.fkind)} tile, "
                                  f"{_kname(sk)} scalar)")
                return None
            return t
        if op == "dot":
            a, b = infos[0], infos[1]
            if a is not None and b is not None:
                if a.cols != b.rows:
                    self.report(node, f"Tile.dot: shape mismatch: "
                                      f"{a.rows}x{a.cols} · "
                                      f"{b.rows}x{b.cols} (inner dims "
                                      f"{a.cols} and {b.rows})")
                    return None
                if a.fkind is not None and b.fkind is not None \
                        and a.fkind != b.fkind:
                    self.report(node, f"Tile.dot: element kinds differ "
                                      f"({_kname(a.fkind)} vs "
                                      f"{_kname(b.fkind)})")
                    return None
                fk = a.fkind if a.fkind is not None else b.fkind
                return _TileInfo(a.rows, b.cols, fk)
            return None
        if op == "transpose":
            t = infos[0]
            if t is not None:
                return _TileInfo(t.cols, t.rows, t.fkind)
            return None
        if op == "get":
            t = infos[0]
            i, j = _lit_int(args[1]), _lit_int(args[2])
            if t is not None and i is not None and j is not None \
                    and not (0 <= i < t.rows and 0 <= j < t.cols):
                self.report(node, f"Tile.get: index out of bounds: "
                                  f"({i}, {j}) (shape {t.rows}x{t.cols})")
            return None
        # to_vec / sum / rows / cols: non-tile results.
        return None


def check_tile_shapes(root: Any,
                      file_path: str | None = None) -> list[BorrowError]:
    """Structured ``type-tile-shape`` diagnostics for ``root``, if any.

    ``root`` is the MUTABLE post-desugar AST (`pipeline.PhaseContext.program`).
    """
    checker = _TileShapeChecker(file_path)
    checker.check_program(root)
    return checker.errors
