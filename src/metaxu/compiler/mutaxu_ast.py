from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Dict
import json
import metaxu.metaxu_ast as fast


@dataclass(frozen=True, slots=True)
class Span:
    """Source range of a frozen AST node.

    Position semantics (unambiguous, and the same everywhere in the
    compiler):

      start / end   0-based CHARACTER OFFSETS into the source text of
                    `file`, half-open: `source[start:end]` is the node's
                    text.  0/0 means "position unknown".
      line / column 1-based line and column of `start` — the human-readable
                    form printed in diagnostics as `file:line:column`.
                    0 means "unknown".
      end_line /    1-based line and column of `end`, i.e. just past the
      end_column    node's last character (0 when unknown).

    Before source locations were attached at parse time, `start` held a
    column number; it is an offset now, and `line`/`column` are the fields
    to print.  Use `Span.text()` for the `file:line:column` rendering.
    """
    file: str
    start: int
    end: int
    line: int = 0
    column: int = 0
    end_line: int = 0
    end_column: int = 0

    def text(self, fallback_file: str | None = None) -> str:
        """`file:line:column`, degrading to just the file when the line is
        unknown and to `fallback_file` when even the file is unknown."""
        file = self.file
        if not file or file == "<unknown>":
            file = fallback_file or file or "<unknown>"
        return f"{file}:{self.line}:{self.column}" if self.line else str(file)

    def location(self, fallback_file: str | None = None):
        """This span as an errors.SourceLocation (None when position-less)."""
        from metaxu.errors import SourceLocation
        if not self.line:
            return None
        file = self.file
        if not file or file == "<unknown>":
            file = fallback_file or file
        return SourceLocation(
            file=file,
            line=self.line,
            column=self.column,
            end_line=self.end_line or None,
            end_column=self.end_column or None,
            offset=self.start,
            end_offset=self.end,
        )


@dataclass(frozen=True, slots=True)
class AstNode:
    """Frozen, minimal AST mirror for the HIR builder stage.

    This mirrors the existing parsed AST (already available in src/metaxu),
    but provides an immutable view with the essential fields needed for
    subsequent passes. The actual construction will wrap/copy nodes from the
    existing parser output.
    """

    node_id: int
    kind: str
    children: tuple[AstNode, ...]
    span: Span
    # Optional data for leaf nodes or annotations
    value: Any | None = None

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "span": {
                "file": self.span.file,
                "start": self.span.start,
                "end": self.span.end,
                "line": self.span.line,
                "column": self.span.column,
                "end_line": self.span.end_line,
                "end_column": self.span.end_column,
            },
            "value": self.value,
            "children": [c.to_json_obj() for c in self.children],
        }


def dump_ast_json(root: AstNode) -> str:
    """Pretty JSON for golden tests."""
    return json.dumps(root.to_json_obj(), indent=2, sort_keys=True)


def _span_of(node: Any) -> Span:
    """Freeze the parser's SourceLocation for `node` into a Span.

    The parser attaches a SourceLocation (file, 1-based line/column, and the
    0-based character offsets of the node's text) to every node it builds;
    see parser.Parser._attach_location.  Nodes synthesized after parsing
    (desugaring, module resolution) may have none — those keep a file-only
    span with zeroed positions, which renders as just the file name.
    """
    loc = getattr(node, 'location', None)
    if loc is None:
        # Fallbacks from parser where file and positions may be missing
        return Span(file=getattr(node, 'source_file', '<unknown>'), start=0, end=0)
    file = getattr(loc, 'file', None) or getattr(node, 'source_file', '<unknown>')
    line = getattr(loc, 'line', 0) or 0
    column = getattr(loc, 'column', 0) or 0
    start = getattr(loc, 'offset', None)
    end = getattr(loc, 'end_offset', None)
    return Span(
        file=file,
        start=start if isinstance(start, int) else 0,
        end=end if isinstance(end, int) else (start if isinstance(start, int) else 0),
        line=line,
        column=column,
        end_line=getattr(loc, 'end_line', 0) or 0,
        end_column=getattr(loc, 'end_column', 0) or 0,
    )


def _pattern_descriptor(p: Any) -> dict[str, Any]:
    """JSON-serializable summary of a match-arm pattern.

    The frozen AST does not traverse MatchExpression.cases (they are plain
    (pattern, body) tuples, and the patterns are parsed as *expressions* by
    the arm grammar), so the constraint emitter cannot see arm patterns
    through children. This mirrors hir.HIRBuilder._convert_pattern just far
    enough for compile-time exhaustiveness checking. Descriptor kinds:

      {"kind": "wildcard"}                      `_` / explicit wildcard
      {"kind": "binding", "name": n}            always-a-binding (VariablePattern,
                                                borrow/move-annotated bindings)
      {"kind": "name", "name": n}               bare identifier: a zero-arg
                                                variant ctor iff the checker
                                                knows n as a variant, else a
                                                binding (resolved by the
                                                emitter, which has
                                                variant_to_enum)
      {"kind": "literal", "value": v}           literal pattern
      {"kind": "ctor", "name": v, "enum": e|None, "subpatterns": [...]}
      {"kind": "unknown"}                       anything unclassified (list
                                                patterns, lambdas, ...) — the
                                                checker skips matches
                                                containing these entirely
    """
    if p is None:
        # _convert_pattern lowers a missing pattern to a wildcard; mirror it.
        return {"kind": "wildcard"}
    cls_name = type(p).__name__
    if cls_name == "WildcardPattern":
        return {"kind": "wildcard"}
    if isinstance(p, fast.VariablePattern):
        return {"kind": "binding", "name": str(getattr(p, "name", "_"))}
    if isinstance(p, fast.LiteralPattern):
        v = getattr(p, "value", None)
        if isinstance(v, fast.Literal):
            v = getattr(v, "value", None)
        return {"kind": "literal", "value": v}
    if isinstance(p, fast.VariantPattern):
        enum_name = getattr(p, "enum_name", None)
        return {
            "kind": "ctor",
            "name": str(getattr(p, "variant_name", "")),
            "enum": str(enum_name) if enum_name is not None else None,
            "subpatterns": [_pattern_descriptor(sp)
                            for sp in (getattr(p, "patterns", None) or [])],
        }
    if isinstance(p, (bool, int, float, str)):
        return {"kind": "literal", "value": p}
    # The arm grammar is `expression => body`: patterns arrive as
    # expression nodes. Classify the pattern-like expression forms.
    if isinstance(p, fast.Literal):
        return {"kind": "literal", "value": getattr(p, "value", None)}
    if isinstance(p, fast.Variable):
        name = str(getattr(p, "name", "_") or "_")
        if name == "_":
            return {"kind": "wildcard"}
        return {"kind": "name", "name": name}
    if isinstance(p, fast.UnaryOperation) and getattr(p, "operator", None) == "-":
        operand = getattr(p, "operand", None)
        if isinstance(operand, fast.Literal):
            v = getattr(operand, "value", None)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return {"kind": "literal", "value": -v}
        return {"kind": "unknown"}
    if isinstance(p, fast.TupleLiteral):
        # `(_, _)` is a CATCH-ALL: a tuple pattern's shape carries no runtime
        # test (an arity mismatch is a loud missing-field error, not a failed
        # match — see hir.TUPLE_STRUCT_PREFIX), so an all-wildcard tuple
        # pattern is irrefutable exactly like `_`.
        #
        # Every other tuple pattern is reported OPAQUE, deliberately.  A bare
        # element name freezes as {"kind": "name"}, and only the emitter's
        # variant table can say whether it is an irrefutable binding (`x`) or
        # a nullary constructor (`None`) — guessing "binding" here would make
        # a refutable arm claim to be a catch-all and could silence a real
        # non-exhaustive-match error.  Opaque keeps the check permissive,
        # which is the safe direction.
        subs = [_pattern_descriptor(e)
                for e in (getattr(p, "elements", None) or [])]
        if len(subs) >= 2 and all(s.get("kind") == "wildcard" for s in subs):
            return {"kind": "wildcard"}
        return {"kind": "unknown"}
    if isinstance(p, fast.NoneExpression):
        return {"kind": "ctor", "name": "None", "enum": "Option", "subpatterns": []}
    if isinstance(p, fast.SomeExpression):
        inner = getattr(p, "value", None)
        subs = [_pattern_descriptor(inner)] if inner is not None else []
        return {"kind": "ctor", "name": "Some", "enum": "Option", "subpatterns": subs}
    if isinstance(p, (fast.BorrowShared, fast.BorrowUnique, fast.Move)):
        var = getattr(p, "variable", None)
        if isinstance(var, str):
            return {"kind": "binding", "name": var}
        return _pattern_descriptor(var)
    if isinstance(p, fast.ModeExpression):
        return _pattern_descriptor(getattr(p, "expression", None))
    if isinstance(p, fast.FunctionCall):
        callee = getattr(p, "name", None)
        if isinstance(callee, str) and callee:
            return {
                "kind": "ctor",
                "name": callee,
                "enum": None,
                "subpatterns": [_pattern_descriptor(a)
                                for a in (getattr(p, "arguments", None) or [])],
            }
        return {"kind": "unknown"}
    if isinstance(p, fast.QualifiedFunctionCall):
        parts = list(getattr(p, "parts", None) or [])
        if len(parts) >= 2:
            return {
                "kind": "ctor",
                "name": str(parts[-1]),
                "enum": str(parts[-2]),
                "subpatterns": [_pattern_descriptor(a)
                                for a in (getattr(p, "arguments", None) or [])],
            }
        return {"kind": "unknown"}
    return {"kind": "unknown"}


def _arm_descriptor(case: Any) -> dict[str, Any]:
    """Descriptor for one match case: (pattern, body) tuples plus the legacy
    Option sugar shapes ('some', var, body) / ('none', None, body)."""
    if isinstance(case, (list, tuple)):
        if len(case) == 3 and case[0] in ("some", "none"):
            if case[0] == "some":
                return {"kind": "ctor", "name": "Some", "enum": "Option",
                        "subpatterns": [{"kind": "binding", "name": str(case[1])}]}
            return {"kind": "ctor", "name": "None", "enum": "Option",
                    "subpatterns": []}
        if len(case) == 2:
            return _pattern_descriptor(case[0])
    return {"kind": "unknown"}


def _value_of(node: Any) -> Any | None:
    if isinstance(node, fast.Literal):
        return getattr(node, "value", None)
    if isinstance(node, fast.Variable):
        return {"name": getattr(node, "name", None)}
    if isinstance(node, fast.BinaryOperation):
        return {"operator": getattr(node, "operator", None)}
    if isinstance(node, fast.ComparisonExpression):
        return {"operator": getattr(node, "operator", None)}
    if isinstance(node, fast.UnaryOperation):
        # The operator was NOT carried before, so every unary node froze as
        # an anonymous "some unary op" and no checker could tell `-e` from
        # `~e`. `~` is Int-only, which the constraint emitter can only
        # enforce if it can see which operator this is.
        return {"operator": getattr(node, "operator", None)}
    if isinstance(node, fast.LetBinding):
        return {
            "name": getattr(node, "identifier", None),
            "mode": _mode_value(getattr(node, "mode", None)),
        }
    if isinstance(node, fast.Parameter):
        return {
            "name": getattr(node, "name", None),
            "mode": _mode_value(getattr(node, "mode", None)),
        }
    if isinstance(node, fast.FunctionDeclaration):
        payload = {
            "name": getattr(node, "name", None),
            "params": [getattr(p, "name", None) for p in getattr(node, "params", [])],
            "performs": [_effect_name(e) for e in getattr(node, "performs", []) or []],
            # Generic signature info (for parametric instantiation checking):
            # declared type parameters, per-parameter declared type displays
            # (None for unannotated params), the declared return type display,
            # and where-clause / inline-bound constraints [{param, trait, kind}].
            "type_params": [
                _type_param_name(tp) for tp in getattr(node, "type_params", None) or []
            ],
            "param_types": [
                _type_display(getattr(p, "type_annotation", None))
                for p in getattr(node, "params", [])
            ],
            "return_type": _safe_type_display(getattr(node, "return_type", None)),
            "where": _where_constraints(node),
        }
        # Mangled impl methods (__impl$Trait$Type$m) carry their implement
        # block's where clause and type parameters, attached by the trait
        # impl desugar pass, so the constraint emitter can enforce impl
        # where clauses at coherence-load time.
        impl_where = getattr(node, "_impl_where_clause", None)
        if impl_where is not None:
            payload["impl_where"] = _where_clause_constraints(impl_where)
        impl_params = getattr(node, "_impl_type_params", None)
        if impl_params is not None:
            payload["impl_params"] = [str(p) for p in impl_params]
        return payload
    if isinstance(node, fast.EffectDeclaration):
        return {
            "name": getattr(node, "name", None),
            "effect_class": getattr(node, "effect_class", None),
        }
    if isinstance(node, fast.HandleEffect):
        return {
            "effect_name": getattr(node, "effect_name", None),
        }
    if isinstance(node, fast.PerformEffect):
        name = getattr(node, "effect_name", None)
        return {"effect_name": name if isinstance(name, str) or name is None else str(name)}
    if isinstance(node, fast.Resume):
        return {
            "value": getattr(node, "value", None),
        }
    if isinstance(node, fast.FunctionCall):
        name = getattr(node, "name", None)
        payload: dict[str, Any] = {
            "name": name if isinstance(name, str) or name is None else str(name)
        }
        # Explicit instantiation type args (`identity<Int>(x)`, `Full<Int>(3)`).
        type_args = getattr(node, "type_args", None) or []
        if type_args:
            payload["type_args"] = [_type_display(a) for a in type_args]
        return payload
    if isinstance(node, fast.Assignment):
        # The assignment target may be a complex expression (field access,
        # indexing); stringify it so the frozen AST stays JSON serializable.
        name = getattr(node, "name", None)
        return {"name": name if isinstance(name, str) or name is None else str(name)}
    if isinstance(node, fast.LambdaExpression):
        return {
            "params": [getattr(p, "name", None) for p in getattr(node, "params", [])],
            "captures": dict(getattr(node, "capture_modes", {}) or {}),
            "linearity": _mode_value(getattr(node, "linearity", None)),
        }
    if isinstance(node, fast.StructDefinition):
        return {
            "name": getattr(node, "name", None),
            "type_params": [
                _type_display(tp) for tp in getattr(node, "type_params", None) or []
            ],
            "fields": [
                {
                    "name": getattr(f, "name", None),
                    "type": _type_display(getattr(f, "type_info", None)),
                    # Per-field mode annotations (@mut/@const/@local ...) are
                    # attached by the parser as `field.modes`; carry them so
                    # deep ownership validation can see declared field modes.
                    "mode": _mode_value(getattr(f, "modes", None)),
                }
                for f in getattr(node, "fields", None) or []
            ],
        }
    if isinstance(node, fast.EnumDefinition):
        return {
            "name": getattr(node, "name", None),
            "type_params": [
                _type_param_name(tp) for tp in getattr(node, "type_params", None) or []
            ],
            "variants": [
                {
                    "name": getattr(v, "name", None),
                    "fields": [
                        {"name": fname, "type": _type_display(ftype)}
                        for (fname, ftype) in (getattr(v, "fields", None) or [])
                    ],
                }
                for v in getattr(node, "variants", None) or []
            ],
        }
    if isinstance(node, fast.StructInstantiation):
        struct_name = getattr(node, "struct_name", None)
        return {
            "name": str(struct_name) if struct_name is not None else None,
            "type_args": [
                _type_display(a) for a in getattr(node, "type_args", None) or []
            ],
        }
    if isinstance(node, fast.StructField):
        return {"name": getattr(node, "name", None)}
    if isinstance(node, fast.Implementation):
        # Pre-desugar impl registry info: `implement Trait for Type`.
        # (Post-desugar the same info lives in the mangled __impl$ names,
        # which also carry impl_where/impl_params — see FunctionDeclaration.)
        return {
            "trait": _safe_type_display(getattr(node, "interface_name", None)),
            "type": _safe_type_display(getattr(node, "type_name", None)),
            "where": _where_clause_constraints(getattr(node, "where_clause", None)),
            "type_params": [
                _type_param_name(tp) for tp in getattr(node, "type_params", None) or []
            ],
        }
    if isinstance(node, fast.MatchExpression):
        # Arm-pattern summaries for compile-time exhaustiveness checking:
        # cases are not frozen as children (they are plain tuples), so the
        # payload is the only window the constraint emitter has onto them.
        return {"arms": [_arm_descriptor(c) for c in getattr(node, "cases", None) or []]}
    if isinstance(node, fast.FieldAccess):
        return {"fields": tuple(getattr(node, "fields", ()) or ())}
    if isinstance(node, fast.QualifiedFunctionCall):
        payload = {"name": ".".join(getattr(node, "parts", ()) or ())}
        # Module-qualified generic calls (`mod.f<Int>(x)`) carry their
        # explicit type args exactly like plain FunctionCalls, so
        # instantiation checking survives the module system's rename of
        # `mod.f` to a dotted callee name.
        type_args = getattr(node, "type_args", None) or []
        if type_args:
            payload["type_args"] = [_type_display(a) for a in type_args]
        return payload
    # Borrow/move payloads name the borrowed variable. The parser always
    # supplies a name string here (`&x`, `move(x)`); _operand_name also
    # accepts a Variable node so a directly-constructed node freezes to the
    # same payload instead of smuggling an AST node into the JSON.
    if isinstance(node, fast.BorrowShared):
        return {"variable": _operand_name(getattr(node, "variable", None))}
    if isinstance(node, fast.BorrowUnique):
        return {"variable": _operand_name(getattr(node, "variable", None))}
    if isinstance(node, fast.Move):
        return {"variable": _operand_name(getattr(node, "variable", None))}
    if isinstance(node, fast.ExclaveExpression):
        # `exclave e`: the payload names the exclaved VARIABLE when there is
        # one (the borrow checker's check_exclave takes a name), and never
        # the operand node itself — the operand is already a frozen child,
        # and putting an AST node in the payload made `dump_ast_json` die
        # with "Object of type Literal is not JSON serializable".
        return {"expression": _operand_name(getattr(node, "expression", None))}
    if isinstance(node, fast.Resume):
        return {"value": _operand_name(getattr(node, "value", None))}
    return None


def _operand_name(operand: Any) -> str | None:
    """Variable name of a sub-expression used as a payload key, else None.

    Payloads must stay JSON-serializable (the frozen AST is dumped for the
    golden tests and for tooling), so a sub-expression is summarized by its
    variable name when it is a plain name and dropped otherwise; the operand
    itself is always reachable as a frozen child.
    """
    if operand is None:
        return None
    if isinstance(operand, str):
        return operand
    if isinstance(operand, fast.Variable):
        name = getattr(operand, "name", None)
        return name if isinstance(name, str) else None
    return None


#: Types allowed inside a frozen-AST payload (everything json.dumps handles).
_JSON_SCALARS = (str, int, float, bool, type(None))


def _first_unserializable(value: Any, depth: int = 0) -> Any | None:
    """First value inside `value` that JSON cannot represent, else None."""
    if depth > 12:
        return value
    if isinstance(value, _JSON_SCALARS):
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            bad = _first_unserializable(item, depth + 1)
            if bad is not None:
                return bad
        return None
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                return k
            bad = _first_unserializable(v, depth + 1)
            if bad is not None:
                return bad
        return None
    return value


def _checked_value_of(node: Any) -> Any | None:
    """`_value_of` plus the invariant that payloads are JSON-serializable.

    A payload holding an AST node is a compiler bug in `_value_of`, and it
    used to surface far away as a bare `TypeError: Object of type Literal is
    not JSON serializable` from `json.dumps`.  Catch it here, where the node
    (and therefore the source location) is still in hand, and report it as a
    located compile error instead of a crash.
    """
    payload = _value_of(node)
    bad = _first_unserializable(payload)
    if bad is None:
        return payload
    from metaxu.errors import CompileError
    span = _span_of(node)
    raise CompileError(
        message=(f"frozen AST payload for {node.__class__.__name__} holds a "
                 f"{type(bad).__name__} value that is not serializable "
                 f"({bad!r})"),
        error_type="FrozenAstError",
        location=span.location(),
        notes=["this is a compiler bug in mutaxu_ast._value_of: payloads must "
               "be plain JSON values; sub-expressions belong in the node's "
               "children, not in its payload"],
    )


def _type_display(t: Any) -> str | None:
    """Best-effort display name for a surface type expression.

    Handles TypeReference/TypeParameter/Variable (all carry .name) and
    TypeApplication (constructor + args). Used so the frozen AST keeps
    enough declared-type information for field-level type checking.
    """
    if t is None:
        return None
    if isinstance(t, str):
        return t
    # Mode-annotated types (`unique vector[Int,3]`, `@local Foo`) wrap the
    # type they annotate; the modes travel in their own payload keys, so the
    # display is the display of the annotated type.  Falling through to the
    # str() fallback produced "ModeTypeAnnotationat 0x7f..." — a display no
    # field-level check could read, which is how an ill-formed field type
    # slipped past the checker.
    if isinstance(t, fast.ModeTypeAnnotation):
        return _type_display(getattr(t, "base_type", None))
    # `vector[T, N]` written directly as a type: base type plus size.
    if isinstance(t, fast.VectorTypeExpression):
        base = _type_display(getattr(t, "base_type", None)) or "?"
        size = _type_display(getattr(t, "size", None)) or "?"
        return f"vector[{base}, {size}]"
    name = getattr(t, "name", None)
    if isinstance(name, str):
        return name
    ctor = getattr(t, "type_constructor", None)
    if ctor is not None:
        base = ctor if isinstance(ctor, str) else _type_display(ctor)
        args = [_type_display(a) or "?" for a in getattr(t, "type_args", None) or []]
        return f"{base}[{', '.join(args)}]" if args else base
    return str(t)


def _safe_type_display(t: Any) -> str | None:
    """_type_display that also tolerates bare Python classes (e.g. the parser
    stores the NoneType *class* as the default return type)."""
    if t is None:
        return None
    if isinstance(t, type):
        return getattr(t, "__name__", None)
    return _type_display(t)


def _type_param_name(tp: Any) -> str | None:
    """Name of a declared type parameter (TypeParameter or bare string)."""
    if isinstance(tp, str):
        return tp
    name = getattr(tp, "name", None)
    return name if isinstance(name, str) else None


def _flatten_bounds(bound: Any) -> list[str]:
    """Flatten a type bound expression into trait-name displays.

    Handles a single bound (`T: Display`), compound bounds
    (`T: Display + Ord`), and legacy list shapes.
    """
    if bound is None:
        return []
    if isinstance(bound, (list, tuple)):
        out: list[str] = []
        for b in bound:
            out.extend(_flatten_bounds(b))
        return out
    left = getattr(bound, "left", None)
    right = getattr(bound, "right", None)
    if bound.__class__.__name__ == "CompoundTypeBound" and (left is not None or right is not None):
        return _flatten_bounds(left) + _flatten_bounds(right)
    disp = _safe_type_display(bound)
    return [disp] if disp else []


def _where_clause_constraints(where: Any) -> list[dict[str, str]]:
    """Flatten one WhereClause node into constraint entries
    [{"param": <name>, "trait": <trait name>, "kind": <kind>}]."""
    out: list[dict[str, str]] = []
    for c in getattr(where, "constraints", None) or []:
        pname = _safe_type_display(getattr(c, "type_param", None))
        for trait in _flatten_bounds(getattr(c, "bound_type", None)):
            if pname:
                out.append({
                    "param": pname,
                    "trait": trait,
                    "kind": str(getattr(c, "kind", "subtype") or "subtype"),
                })
    return out


def _where_constraints(node: Any) -> list[dict[str, str]]:
    """Collect trait-bound constraints for a generic declaration.

    Merges inline bounds on type parameters (`fn f<T: Trait>`) with the
    where clause (`fn f<T>(..) -> R where T: Trait`). Each entry is
    {"param": <type param name>, "trait": <trait name>, "kind": <kind>}.
    """
    out: list[dict[str, str]] = []
    for tp in getattr(node, "type_params", None) or []:
        pname = _type_param_name(tp)
        if pname is None:
            continue
        for trait in _flatten_bounds(getattr(tp, "bounds", None)):
            out.append({"param": pname, "trait": trait, "kind": "bound"})
    out.extend(_where_clause_constraints(getattr(node, "where_clause", None)))
    return out


def _mode_value(mode: Any) -> Any | None:
    """Normalize surface mode annotations to plain strings for the payload.

    The parser produces ModeAnnotation nodes (wrapping a Uniqueness/Locality/
    LinearityMode carrying .mode) and sometimes lists of them; downstream
    (_split_mode in the constraint emitter) understands strings and lists of
    strings.
    """
    if mode is None:
        return None
    if isinstance(mode, str):
        return mode
    if isinstance(mode, (list, tuple)):
        flat: list[str] = []
        for m in mode:
            v = _mode_value(m)
            if isinstance(v, list):
                flat.extend(v)
            elif v is not None:
                flat.append(v)
        return flat or None
    inner = getattr(mode, "mode_type", None)
    if inner is not None:
        return _mode_value(inner)
    value = getattr(mode, "mode", None)
    return value if isinstance(value, str) else None


def _effect_name(effect: Any) -> str:
    name = getattr(effect, "effect_name", None)
    if name is None:
        name = getattr(effect, "name", None)
    if name is None:
        name = effect
    return str(name)


def build_frozen_ast_with_map(parsed_root: Any) -> Tuple[AstNode, Dict[int, Any]]:
    """Convert the existing AST into a frozen representation and return a mapping
    from frozen node_id to original node object.

    We rely on metaxu.metaxu_ast.Node.children for traversal.
    """
    next_id = 1
    id_map: Dict[int, Any] = {}

    def is_ast_node(value: Any) -> bool:
        return hasattr(value, "children") or isinstance(value, fast.Node)

    def go(n: Any) -> AstNode:
        nonlocal next_id
        nid = next_id
        next_id += 1
        id_map[nid] = n
        kind = n.__class__.__name__
        # Children: prefer explicit children list if present
        kids = []
        for attr in ("params",):
            for c in getattr(n, attr, []) or []:
                if c is not None:
                    kids.append(go(c))
        if hasattr(n, 'children') and isinstance(n.children, list):
            for c in n.children:
                if c is not None:
                    kids.append(go(c))
        for attr in ("expression", "body", "value", "base"):
            c = getattr(n, attr, None)
            if c is not None and c not in getattr(n, 'children', []):
                if isinstance(c, list):
                    for item in c:
                        if item is not None and item not in getattr(n, 'children', []) and is_ast_node(item):
                            kids.append(go(item))
                else:
                    if is_ast_node(c):
                        kids.append(go(c))
        span = _span_of(n)
        return AstNode(node_id=nid, kind=kind, children=tuple(kids), span=span,
                       value=_checked_value_of(n))

    root = go(parsed_root)
    return root, id_map


def build_frozen_ast(parsed_root: Any) -> AstNode:
    root, _ = build_frozen_ast_with_map(parsed_root)
    return root
