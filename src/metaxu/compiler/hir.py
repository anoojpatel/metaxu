from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .types import Ty, EffectSet
from .infer_tables import InferSideTables
from .constraints import ClassConstraint
from . import mutaxu_ast as mast
import metaxu.metaxu_ast as fast
from metaxu.unsafe_ast import AddressOf, PointerDereference, TypeCast, UnsafeBlock
from metaxu.extern_ast import (ExternBlock, ExternFunctionDeclaration,
                               ExternTypeDeclaration)

from .desugar import IMPL_SEP, parse_impl_method_name, type_base_name
from metaxu.errors import SourceLocation, format_location, source_excerpt
from .recursion import compiler_phase

# Methods that are dispatched as interpreter builtins with the receiver as
# first argument (`x.to_string()` -> __builtin$to_string(x); see
# _method_callee and BUILTIN_CALL_PREFIX below).
# Note: when a method of the same name is provided by a user trait/impl block,
# the call lowers to a __trait$ dispatch instead (see the QualifiedFunctionCall
# and MethodCall paths below), so user impls win over these builtins; the
# interpreter's trait dispatch falls back to the builtin for receiver types
# without an impl.  A plain top-level function of the same name NEVER wins in
# method position (docs/name_precedence.md).
_BUILTIN_METHODS = frozenset({
    "to_string", "len",
    # Vec methods (runtime library)
    "push", "pop",
    # math methods on numbers (runtime library)
    "sqrt", "sin", "cos",
    # FFI: `x.as_ptr()` — raw pointer view of a string/vector's bytes
    # (interpreter shim over the simulated C heap; see mir_interp).
    "as_ptr",
})

# Callee-name prefix marking a runtime-dispatched trait method call:
# `recv.m(args)` lowers to Call(callee="__trait$m", operands=(recv, *args))
# and the MIR interpreter picks the impl function matching the receiver's
# runtime type name (falling back to builtins, then plain functions).
TRAIT_CALL_PREFIX = f"__trait{IMPL_SEP}"

# Callee-name prefix for a static impl-method call `Type.m(args)` (no
# receiver): "__static$Type$m". The interpreter resolves it against the
# loaded __impl$Trait$Type$m functions for that type name.
STATIC_CALL_PREFIX = f"__static{IMPL_SEP}"

# Callee-name prefix for an effect op mapped onto a named runtime primitive
# via a `with SYMBOL` clause in its effect declaration (e.g.
# `fn lock(m: Mutex) -> () with EFFECT_MUTEX_LOCK`). Each mapped op compiles
# to a thunk `__effect_runtime$Effect$op` whose body is
# Call(callee="__mx_effect_runtime$SYMBOL", operands=(op params...)); the
# interpreter dispatches that callee to its runtime shim table (loudly
# erroring on symbols it has no shim for).
EFFECT_RUNTIME_CALL_PREFIX = f"__mx_effect_runtime{IMPL_SEP}"

# Callee-name prefix marking a METHOD-POSITION call to a runtime builtin
# method: `x.len()` lowers to Call(callee="__builtin$len", operands=(x,)).
#
# NAME PRECEDENCE (see docs/name_precedence.md).  Plain calls resolve to a
# user module function BEFORE the builtin of the same name, so a program
# that declares `fn push(list, item)` can call it.  Method position must
# not follow that rule: `x.len()` means "the receiver's length", and a
# top-level `fn len(...)` is not a method of anything, so it must not
# hijack every receiver in the program.  Marking the call site keeps the
# two positions distinguishable in MIR (they were both bare names before,
# which is exactly why the precedence could not be flipped).  Method
# position therefore resolves impl -> builtin -> plain function, the same
# order trait dispatch has always used: a method name declared by any
# trait/impl goes through TRAIT_CALL_PREFIX instead of this prefix, so a
# user impl still wins.
BUILTIN_CALL_PREFIX = f"__builtin{IMPL_SEP}"

# ---------------------------------------------------------------------------
# Tuples ARE anonymous structs.
#
# `(a, b)` lowers to `alloc_struct "__tuple2" { _0of2: a, _1of2: b }` and a
# tuple pattern reads its elements back with `field_get "_0of2"` /
# `"_1of2"`.  Nothing else in the pipeline learns a new concept: MIR gains
# no op, the interpreter gains no value class (an MxStruct IS the tuple),
# and native codegen inherits the whole struct path — layout, GEPs, byval
# parameter copies, sret returns, field kinds, the module-wide struct table
# built from `alloc_struct` sites rather than from declarations — which is
# why a tuple lowers natively with no backend change at all.
#
# WHY THE FIELD NAME REPEATS THE ARITY.  Metaxu's inference has no tuple
# type, so nothing upstream can reject `let (a, b) = triple`.  With plain
# `_0`/`_1` field names that program would SILENTLY bind the first two
# elements of a 3-tuple — a positional pattern quietly meaning something
# other than what it says, which is the exact bug family
# `docs/token_reachability.md` exists to prevent.  Because the names carry
# the arity, `_0of2` simply does not exist on a `__tuple3`, so the mismatch
# is a loud missing-field error in the interpreter (see
# `MxStruct.get`) and a demotion — never wrong code — natively, with no new
# MIR op, builtin or runtime check.  The names are an implementation
# detail — the surface way to reach an element is a pattern, not a field
# access; there is no `p.0`.
#
# The struct name carries the ARITY only, so every 2-tuple in a module is
# one layout.  The visible cost: a module that builds both `(1, 2)` and
# `(1, "s")` joins `_1of2`'s native kind across them, the join conflicts,
# and the affected functions demote to the interpreter with a reason.  The
# interpreter is unaffected (its fields are dynamically typed), and the
# alternative — a distinct struct name per element-kind tuple — cannot be
# computed in HIR, which runs before native kinds are inferred.
TUPLE_STRUCT_PREFIX = "__tuple"

# The smallest tuple.  `(e)` is parenthesized grouping and `(e,)` is a
# syntax error, so the language has NO 1-tuples: a tuple is 2+ elements,
# and `()` is the unit value (not a 0-tuple).
TUPLE_MIN_ARITY = 2


def tuple_struct_name(arity: int) -> str:
    """Struct name for an `arity`-element tuple (`(a, b)` -> `__tuple2`)."""
    return f"{TUPLE_STRUCT_PREFIX}{arity}"


def tuple_field_name(index: int, arity: int) -> str:
    """Field name for element `index` (0-based) of an `arity`-tuple.

    The arity is part of the NAME, not just the struct name, so reading a
    2-tuple's element out of a 3-tuple cannot silently succeed.
    """
    return f"_{index}of{arity}"


def is_tuple_struct(name: str) -> bool:
    """True for a struct name a tuple literal generated (`__tuple2`)."""
    return (isinstance(name, str) and name.startswith(TUPLE_STRUCT_PREFIX)
            and name[len(TUPLE_STRUCT_PREFIX):].isdigit())

# Every BARE (non-dotted, non-reserved) builtin function name the runtime
# provides — the shadowable surface of the precedence rule.  Pinned equal to
# mir_interp._register_builtins by test_name_precedence; the module resolver
# uses it to bind a module's unqualified builtin calls LEXICALLY (a bare
# `len(v)` inside std/vec.mx means the builtin even when the entry program
# defines its own `fn len`, because MIR's function namespace is flat).
BUILTIN_FUNCTION_NAMES = frozenset({
    "print", "println", "assert", "assert_eq",
    "to_string", "int_to_str", "len", "push", "pop",
    "sqrt", "sin", "cos", "neg", "not", "bnot",
    # FFI shims over the interpreter's simulated C heap
    "malloc", "free", "memcpy", "realloc",
    "ptr_read", "ptr_write", "as_ptr", "fopen", "fclose",
})


def _method_callee(method: str, trait_method_names: set[str]) -> str:
    """The MIR callee name for a method-position call `recv.method(...)`.

    Three cases, in order:
      * declared by some trait or impl -> ``__trait$method`` (runtime
        dispatch on the receiver type; a user impl wins, builtin second,
        plain function third);
      * a runtime builtin method       -> ``__builtin$method`` (always the
        builtin — a plain function never hijacks method position);
      * anything else                  -> the bare name, i.e. a UFCS-style
        call of a plain function with the receiver as first argument.
    """
    if method in trait_method_names:
        return TRAIT_CALL_PREFIX + method
    if method in _BUILTIN_METHODS:
        return BUILTIN_CALL_PREFIX + method
    return method


# ======================================================================
# AST-node triage
# ======================================================================
#
# HIR lowering used to end in `return None` for every AST node class it did
# not recognize, and every caller skipped a None result. The construct then
# VANISHED: `if let`, `while let`, `unsafe { }`, `@mut e`, `[a, b]`, `for`,
# `e as T`, early `return` and struct-field initializers were each found this
# way, one accident at a time, each one a program that compiled and quietly
# did less than it said.
#
# The fallback is now loud (see `_unlowerable`), and every AST node class is
# triaged into EXACTLY ONE bucket below, so a construct can never again be
# forgotten silently — `test_hir_coverage.py` fails if a newly added AST node
# class is missing from this table.
#
# The bucket answers one question: *what does it mean for this node class to
# reach HIR expression lowering?*
#
#   LOWERED           it is handled by `_from_orig_expr` today. This includes
#                     declaration nodes that deliberately lower to unit (their
#                     meaning is realized by an earlier pass or by
#                     HIRBuilder.build's own hoisting walk) and the two nodes
#                     handled by raising a targeted *user* diagnostic
#                     (a bare `vector[T,N]`, a non-unit tuple literal).
#   NOT_AN_EXPRESSION type-level, structural (a child consumed by its parent's
#                     branch), a pattern-position-only node, an abstract base,
#                     or a form already consumed by an earlier pass. Reaching
#                     expression lowering is a COMPILER BUG -> raise.
#   UNSUPPORTED       a real surface construct a user can write that has no
#                     HIR semantics yet. Reaching lowering is a USER error ->
#                     raise a "not supported" diagnostic naming the construct.
#                     Never silence.
#
# Pattern position has its own table (`PATTERN_TRIAGE`) because the set of
# node classes that may appear there is different: the parser's match-arm
# grammar is `expression => body`, so patterns arrive as expression nodes.

LOWERED = "lowered"
NOT_AN_EXPRESSION = "not-an-expression"
UNSUPPORTED = "unsupported"

#: class name -> (bucket, one-line reason). Covers every Node subclass of
#: metaxu_ast, unsafe_ast, extern_ast and decorator_ast.
AST_NODE_TRIAGE: dict[str, tuple[str, str]] = {
    # ---- (a) lowered: ordinary expressions and statements ----
    "AddressOf": (LOWERED, "`&e` on a non-variable operand: same borrow rule as `&x` — evaluates to the referenced value"),
    "Assignment": (LOWERED, "`x = e`, `x.f = e`, `v[i] = e`"),
    "BinaryOperation": (LOWERED, "arithmetic/logical binop"),
    "Block": (LOWERED, "`{ stmts }`"),
    "BorrowExpression": (LOWERED, "`borrow x` / `borrow x as T` — evaluates to the borrowed value"),
    "BorrowShared": (LOWERED, "`&x` — evaluates to the referenced value"),
    "BorrowUnique": (LOWERED, "`&mut x` / `@mut x` — evaluates to the referenced value"),
    "CallExpression": (LOWERED, "call with a computed callee (`(f)(x)`, `v[0](x)`) via a temp binding"),
    "ComparisonExpression": (LOWERED, "comparison binop"),
    "ExclaveExpression": (LOWERED, "`exclave e` — evaluates to the inner value"),
    "FieldAccess": (LOWERED, "`a.b.c` -> chained FieldGet"),
    "ForStatement": (LOWERED, "`for x in it { }` -> desugared to the While machinery"),
    "FunctionCall": (LOWERED, "`f(args)`, perform-as-bare-call, variant constructors"),
    "HandleBlock": (LOWERED, "`handle e { perform Op(p) => body }` (inline handler form)"),
    "HandleEffect": (LOWERED, "`handle e with { } in body`"),
    "IfExpression": (LOWERED, "`if c { } else { }`"),
    "IfLetExpression": (LOWERED, "`if let PAT = e { }` -> two-arm Match"),
    "IfStatement": (LOWERED, "statement form of `if` (built by desugar passes)"),
    "IndexExpression": (LOWERED, "`v[i]` / `v[a:b]` -> __index_get / __slice_get"),
    "LambdaExpression": (LOWERED, "`fn(x) -> e` / `x -> e`"),
    "LetStatement": (LOWERED, "`let x = e`"),
    "ListLiteral": (LOWERED, "`[]`, `[a, b]`, `[a, ...rest]` -> Vec"),
    "Literal": (LOWERED, "int/float/bool/string literal"),
    "MatchExpression": (LOWERED, "`match e { pat => body }`"),
    "MethodCall": (LOWERED, "`e.m(args)` on a computed receiver"),
    "ModeExpression": (LOWERED, "`@const e` / `@mut e` — evaluates to the inner value"),
    "Move": (LOWERED, "`move(x)` — evaluates to the moved value"),
    "NoneExpression": (LOWERED, "`None`"),
    "PerformEffect": (LOWERED, "`perform Effect.op(args)`"),
    "PrintStatement": (LOWERED, "`print(args)`"),
    "QualifiedFunctionCall": (LOWERED, "`a.b(args)`: trait/static/builtin dispatch or dotted callee"),
    "QualifiedName": (LOWERED, "`a`, `a.b.c`, `Enum.Variant`"),
    "RangeExpression": (LOWERED, "`a..b` -> __range"),
    "Resume": (LOWERED, "`resume(v)` in a handler arm"),
    "ReturnStatement": (LOWERED, "`return e` -> explicit Return op"),
    "SomeExpression": (LOWERED, "`Some(e)`"),
    "StructInstantiation": (LOWERED, "`S { f: e }`"),
    "TryCatch": (LOWERED, "`try { } catch e { }`"),
    "TupleLiteral": (LOWERED, "`()` is unit; `(a, b)` is the anonymous struct `__tuple2 { _0of2, _1of2 }`"),
    "TypeCast": (LOWERED, "`e as T` -> __cast"),
    "UnaryOperation": (LOWERED, "`-e` / `!e` / `~e` -> neg / not / bnot"),
    "UnsafeBlock": (LOWERED, "`unsafe { }` — an ordinary block (unsafe is a static permission)"),
    "Variable": (LOWERED, "name read, `null`, or a bare nullary variant"),
    "VariantInstance": (LOWERED, "`Enum::Variant(f: e)`"),
    "VectorLiteral": (LOWERED, "`vector[T,N](...)` incl. comprehension and zip forms"),
    "VectorTypeExpression": (LOWERED, "raises a targeted diagnostic: `vector[T,N]` is a TYPE, not a value"),
    "WhileLetStatement": (LOWERED, "`while let PAT = e { }`"),
    "WhileStatement": (LOWERED, "`while c { }`"),

    # ---- (a) lowered: declarations, which evaluate to unit ----
    # The parser allows all of these in statement position. Their meaning is
    # realized elsewhere (module loader, desugar, or HIRBuilder.build's own
    # hoisting walk, which lifts every FunctionDeclaration anywhere in the
    # tree), so as a *statement* each one contributes nothing at run time.
    "EffectDeclaration": (LOWERED, "declaration -> unit (ops/defaults compiled by build())"),
    "EnumDefinition": (LOWERED, "declaration -> unit (variants collected by build())"),
    "ExportDeclaration": (LOWERED, "declaration -> unit (consumed by the module loader)"),
    "ExternBlock": (LOWERED, "declaration -> unit (FFI declarations, consumed earlier)"),
    "ExternFunctionDeclaration": (LOWERED, "declaration -> unit (FFI declaration)"),
    "ExternTypeDeclaration": (LOWERED, "declaration -> unit (FFI declaration)"),
    "FromImport": (LOWERED, "declaration -> unit (consumed by the module loader)"),
    "FunctionDeclaration": (LOWERED, "declaration -> unit (hoisted to an HFun by build())"),
    "Implementation": (LOWERED, "declaration -> unit (desugared to mangled __impl$ functions)"),
    "Import": (LOWERED, "declaration -> unit (consumed by the module loader)"),
    "InterfaceDefinition": (LOWERED, "declaration -> unit (trait method names collected by build())"),
    "Module": (LOWERED, "declaration -> unit (consumed by the module loader)"),
    "StructDefinition": (LOWERED, "declaration -> unit (type names collected by build())"),
    "TypeDefinition": (LOWERED, "declaration -> unit (consumed by inference)"),
    "VisibilityRules": (LOWERED, "declaration -> unit (consumed by the module loader)"),

    # ---- (b) not an expression: abstract bases ----
    "Node": (NOT_AN_EXPRESSION, "abstract base class"),
    "Expression": (NOT_AN_EXPRESSION, "abstract base class"),
    "Statement": (NOT_AN_EXPRESSION, "abstract base class"),
    "Pattern": (NOT_AN_EXPRESSION, "abstract base class"),
    "Type": (NOT_AN_EXPRESSION, "abstract base class"),
    "TypeExpression": (NOT_AN_EXPRESSION, "abstract base class"),
    "TypePattern": (NOT_AN_EXPRESSION, "abstract base class"),
    "EffectExpression": (NOT_AN_EXPRESSION, "abstract base class"),

    # ---- (b) not an expression: type level ----
    "BasicType": (NOT_AN_EXPRESSION, "type-level: a primitive type"),
    "CompoundTypeBound": (NOT_AN_EXPRESSION, "type-level: `A + B` bound"),
    "EffectApplication": (NOT_AN_EXPRESSION, "type-level: `Reader[T]` effect application"),
    "EffectReference": (NOT_AN_EXPRESSION, "type-level: an effect parameter"),
    "FunctionType": (NOT_AN_EXPRESSION, "type-level: `fn(T) -> U`"),
    "InterfaceType": (NOT_AN_EXPRESSION, "type-level: a trait used as a type"),
    "ModeTypeAnnotation": (NOT_AN_EXPRESSION, "type-level: `@mut T`"),
    "PointerType": (NOT_AN_EXPRESSION, "type-level: `*mut T` / `*const T`"),
    "RecursiveType": (NOT_AN_EXPRESSION, "type-level: a recursive type"),
    "TypeAlias": (NOT_AN_EXPRESSION, "type-level: an alias declaration"),
    "TypeApplication": (NOT_AN_EXPRESSION, "type-level: `Stack[Int]`"),
    "TypeConstraint": (NOT_AN_EXPRESSION, "type-level: a where-clause constraint"),
    "TypeInfo": (NOT_AN_EXPRESSION, "type-level: reflected type information"),
    "TypeParameter": (NOT_AN_EXPRESSION, "type-level: a generic parameter"),
    "TypeReference": (NOT_AN_EXPRESSION, "type-level: a named type"),
    "WhereClause": (NOT_AN_EXPRESSION, "type-level: a where clause"),

    # ---- (b) not an expression: type-level patterns (comptime type matching) ----
    "EnumPattern": (NOT_AN_EXPRESSION, "type-level pattern (comptime type match)"),
    "GenericTypePattern": (NOT_AN_EXPRESSION, "type-level pattern (comptime type match)"),
    "StructPattern": (NOT_AN_EXPRESSION, "type-level pattern (comptime type match)"),
    "TraitPattern": (NOT_AN_EXPRESSION, "type-level pattern (comptime type match)"),
    "TypeNamePattern": (NOT_AN_EXPRESSION, "type-level pattern (comptime type match)"),
    "TypeVarPattern": (NOT_AN_EXPRESSION, "type-level pattern (comptime type match)"),
    "UnionPattern": (NOT_AN_EXPRESSION, "type-level pattern (comptime type match)"),
    "WildcardPattern": (NOT_AN_EXPRESSION, "pattern position only (see PATTERN_TRIAGE)"),

    # ---- (b) not an expression: value-level pattern nodes ----
    "LiteralPattern": (NOT_AN_EXPRESSION, "pattern position only (see PATTERN_TRIAGE)"),
    "VariablePattern": (NOT_AN_EXPRESSION, "pattern position only (see PATTERN_TRIAGE)"),
    "VariantPattern": (NOT_AN_EXPRESSION, "pattern position only (see PATTERN_TRIAGE)"),

    # ---- (b) not an expression: structural children consumed by a parent ----
    "Decorator": (NOT_AN_EXPRESSION, "structural: an annotation on a declaration"),
    "CFunctionDecorator": (NOT_AN_EXPRESSION, "structural: an annotation on an extern declaration"),
    "DecoratorList": (NOT_AN_EXPRESSION, "structural: a list of annotations"),
    "EffectOperation": (NOT_AN_EXPRESSION, "structural: child of EffectDeclaration"),
    "EnumVariant": (NOT_AN_EXPRESSION, "structural: child of EnumDefinition"),
    "FieldInfo": (NOT_AN_EXPRESSION, "structural: reflected field metadata"),
    "HandleCase": (NOT_AN_EXPRESSION, "structural: child of HandleEffect/HandleBlock"),
    "LetBinding": (NOT_AN_EXPRESSION, "structural: child of LetStatement"),
    "LinearityMode": (NOT_AN_EXPRESSION, "structural: a mode annotation"),
    "LocalityMode": (NOT_AN_EXPRESSION, "structural: a mode annotation"),
    "LocalDeclaration": (NOT_AN_EXPRESSION, "structural: a locality declaration (no parser production)"),
    "LocalParameter": (NOT_AN_EXPRESSION, "structural: a local parameter (no parser production)"),
    "MethodDefinition": (NOT_AN_EXPRESSION, "structural: child of InterfaceDefinition"),
    "MethodImplementation": (NOT_AN_EXPRESSION, "structural: child of Implementation"),
    "ModeAnnotation": (NOT_AN_EXPRESSION, "structural: a mode annotation"),
    "ModuleBody": (NOT_AN_EXPRESSION, "structural: child of Module"),
    "Parameter": (NOT_AN_EXPRESSION, "structural: a function parameter"),
    "Program": (NOT_AN_EXPRESSION, "structural: the compilation-unit root"),
    "RelativePath": (NOT_AN_EXPRESSION, "structural: an import path"),
    "SliceExpression": (NOT_AN_EXPRESSION, "structural: only valid inside an index (handled by IndexExpression)"),
    "SpreadElement": (NOT_AN_EXPRESSION, "structural: only valid inside a list literal (handled by ListLiteral)"),
    "StructField": (NOT_AN_EXPRESSION, "structural: child of StructDefinition"),
    "StructFieldDefinition": (NOT_AN_EXPRESSION, "structural: child of StructDefinition"),
    "UniquenessMode": (NOT_AN_EXPRESSION, "structural: a mode annotation"),
    "VariantDefinition": (NOT_AN_EXPRESSION, "structural: child of EnumDefinition"),
    "WithClause": (NOT_AN_EXPRESSION, "structural: the `with SYMBOL` effect-mapping clause"),

    # ---- (c) unsupported: real surface constructs with no semantics yet ----
    "Comprehension": (UNSUPPORTED, "a bare comprehension has no value representation; only `vector[T,N](e for x in it)` is supported"),
    "ComptimeBlock": (UNSUPPORTED, "compile-time evaluation is not implemented"),
    "ComptimeFunction": (UNSUPPORTED, "compile-time evaluation is not implemented"),
    "ComptimeValue": (UNSUPPORTED, "compile-time evaluation is not implemented"),
    "FromDevice": (UNSUPPORTED, "GPU device transfer has no runtime"),
    "GenericInstance": (UNSUPPORTED, "an uncalled generic instantiation (`f<T>`) has no value representation; call it directly (`f<T>(x)`)"),
    "GetType": (UNSUPPORTED, "compile-time type reflection is not implemented"),
    "KernelAnnotation": (UNSUPPORTED, "GPU kernels have no runtime"),
    "PointerDereference": (UNSUPPORTED, "raw pointer dereference has no HIR/MIR representation"),
    "ToDevice": (UNSUPPORTED, "GPU device transfer has no runtime"),
    "TypeMatchExpression": (UNSUPPORTED, "compile-time matching on types is not implemented"),
}

#: Declaration node classes that lower to unit in statement position (the
#: LOWERED entries above whose reason begins "declaration -> unit").
_DECLARATION_NODES: tuple[type, ...] = (
    fast.EffectDeclaration, fast.EnumDefinition, fast.ExportDeclaration,
    fast.FromImport, fast.FunctionDeclaration, fast.Implementation,
    fast.Import, fast.InterfaceDefinition, fast.Module,
    fast.StructDefinition, fast.TypeDefinition, fast.VisibilityRules,
    ExternBlock, ExternFunctionDeclaration, ExternTypeDeclaration,
)

#: Pattern-position triage. The parser's arm grammar is `expression => body`,
#: so most patterns arrive as expression nodes; the *Pattern classes are here
#: too because desugar passes emit them. Anything not listed makes
#: `_convert_pattern` raise instead of silently degrading to a wildcard —
#: which would make the arm match EVERYTHING (the seam that made
#: `match list { [] -> ..., [x, ...xs] -> ... }` always take its first arm).
# Top-level statement kinds that are NOT the body of a script file: every
# "declaration -> unit" kind above, plus module constants (compiled into
# __module_init by build()), comptime functions (rejected by build()) and
# type aliases. Anything else at the top level of the entry file is code
# that must run; see HIRBuilder._script_statements.
SCRIPT_EXCLUDED_KINDS: frozenset[str] = frozenset(
    {k for k, (_status, desc) in AST_NODE_TRIAGE.items()
     if desc.startswith("declaration -> unit")}
    | {"LetStatement", "ComptimeFunction", "TypeAlias"})

PATTERN_TRIAGE: dict[str, tuple[str, str]] = {
    "WildcardPattern": (LOWERED, "`_`"),
    "VariablePattern": (LOWERED, "a binding"),
    "LiteralPattern": (LOWERED, "a literal"),
    "VariantPattern": (LOWERED, "`Enum::Variant(sub, ...)`"),
    "Literal": (LOWERED, "literal in arm position"),
    "Variable": (LOWERED, "`_`, a nullary variant, or a binding"),
    "UnaryOperation": (LOWERED, "negative numeric literal (`-1`); any other unary form raises"),
    "NoneExpression": (LOWERED, "`None`"),
    "SomeExpression": (LOWERED, "`Some(sub)`"),
    "BorrowShared": (LOWERED, "`&x` in a pattern binds the name"),
    "BorrowUnique": (LOWERED, "`@mut x` in a pattern binds the name"),
    "Move": (LOWERED, "`move(x)` in a pattern binds the name"),
    "ModeExpression": (LOWERED, "a mode-annotated sub-pattern"),
    "FunctionCall": (LOWERED, "`Variant(sub, ...)`; a non-variant callee raises"),
    "QualifiedFunctionCall": (LOWERED, "`Enum.Variant(sub, ...)`"),
    "QualifiedName": (LOWERED, "`Enum.Variant` (nullary); a non-variant dotted name raises"),
    "FieldAccess": (LOWERED, "`Enum.Variant` (nullary, the shape the parser actually builds); any other dotted form raises"),
    "ListLiteral": (UNSUPPORTED, "list patterns (`[]`, `[x, ...xs]`) need a pattern kind MIR cannot test yet"),
    "TupleLiteral": (LOWERED, "`(x, y)` destructures the anonymous tuple struct positionally (arity-exact; `()` in pattern position raises)"),
    "StructInstantiation": (UNSUPPORTED, "struct patterns (`S { f: p }`) are not implemented"),
    "RangeExpression": (UNSUPPORTED, "range patterns (`1..5`) are not implemented"),
    "BinaryOperation": (UNSUPPORTED, "an arbitrary expression is not a pattern"),
    "ComparisonExpression": (UNSUPPORTED, "pattern guards are not implemented"),
    "LambdaExpression": (UNSUPPORTED, "matching on the structure of a function is not implemented"),
    "IndexExpression": (UNSUPPORTED, "matching against an indexed value is not implemented"),
}


class HIRLoweringError(NotImplementedError):
    """A construct reached HIR lowering that the compiler cannot lower.

    Subclasses NotImplementedError so the raises that already existed in this
    module keep their exception class. Every raise is LOUD by design: HIR
    lowering must never quietly drop a construct.

    `location` is the source position of the offending construct (None when
    the node carries none); when it is known, the message gained the
    standard excerpt-with-caret rendering below, so a caller printing the
    exception shows the offending line.
    """

    def __init__(self, message: str, location: SourceLocation | None = None):
        self.location = location
        excerpt = source_excerpt(location) if location is not None else None
        super().__init__(f"{message}\n{excerpt}" if excerpt else message)


class UnsupportedConstruct(HIRLoweringError):
    """Bucket (c): a real surface construct with no HIR semantics yet."""


class HIRCompilerBug(HIRLoweringError):
    """Bucket (b): a node that must never reach expression lowering did."""


@dataclass(slots=True)
class ModeInfo:
    uniqueness: str | None = None   # 'unique'|'exclusive'|'shared'|'owned'|'mutable'|'const'
    locality: str | None = None     # 'local'|'global'
    linearity: str | None = None    # 'once'|'separate'|'many'

@dataclass(slots=True, frozen=True)
class HPattern:
    """A match pattern in HIR.

    kind:
      'wildcard'  matches anything, binds nothing
      'var'       matches anything, binds `name`
      'literal'   matches when scrutinee == value
      'ctor'      matches enum variant `name` (of enum `enum_name` when known),
                  recursively matching `subpatterns` against the payload fields
      'tuple'     destructures an anonymous tuple struct positionally,
                  matching `subpatterns` against `_0ofN`, `_1ofN`, ...
                  The shape itself carries no tag test: the ARITY-BEARING
                  field names make a mismatched arity a loud missing-field
                  error rather than a branch, so only refutable
                  subpatterns can make the arm fail.
    """
    kind: str
    name: str | None = None          # binding name (var) or variant name (ctor)
    value: Any | None = None         # literal value (literal)
    enum_name: str | None = None     # enum type name (ctor), when known
    subpatterns: tuple['HPattern', ...] = ()


@dataclass(slots=True)
class HExpr:
    node_id: int
    kind: str
    args: tuple[Any, ...]
    ty: Ty
    effects: EffectSet
    suspends: bool
    sym: Any | None
    span: mast.Span
    # Optional op-specific fields for lowering
    op: str | None = None            # e.g., 'Literal', 'Var', 'Call', 'Let', 'Block', 'Match'
    literal: Any | None = None       # for Literal
    var_name: str | None = None      # for Var
    callee: str | None = None        # for Call (simple callee name)
    operands: tuple['HExpr', ...] | None = None  # for Call/Block
    bindings: tuple[tuple[str, 'HExpr'], ...] | None = None  # for Let: ((name, expr), ...)
    bind_modes: dict[str, ModeInfo] | None = None  # modes for Let-bound locals
    # BinOp
    binop: str | None = None
    left: 'HExpr' | None = None
    right: 'HExpr' | None = None
    # If
    cond: 'HExpr' | None = None
    then_ops: tuple['HExpr', ...] | None = None
    else_ops: tuple['HExpr', ...] | None = None
    # Match
    scrutinee: 'HExpr' | None = None
    cases: tuple['HExpr', ...] | None = None  # legacy: arm bodies only (no patterns)
    match_arms: tuple[tuple[HPattern, 'HExpr'], ...] | None = None  # (pattern, body) pairs
    # While loop: op="While" (cond field reused for the loop condition)
    loop_body: tuple['HExpr', ...] | None = None
    # While-let loop: op="WhileLet" (scrutinee reused for the matched value;
    # the loop runs while loop_pattern matches, binding its variables in the body)
    loop_pattern: HPattern | None = None
    # Assignment: op="Assign" (var_name reused for the target)
    assign_value: 'HExpr | None' = None
    # Enum variant construction: op="MakeVariant" (operands reused for payload exprs)
    enum_name: str | None = None
    variant_name: str | None = None
    # Struct: op="Struct"
    struct_name: str | None = None                          # for Struct
    fields: tuple[tuple[str, 'HExpr'], ...] | None = None  # for Struct: ((field_name, expr), ...)
    locality: str | None = None                             # 'local'|'global' allocation site
    # FieldGet/FieldSet: op="FieldGet" | "FieldSet"
    base: 'HExpr | None' = None      # receiver object
    field_name: str | None = None    # field being accessed/set
    field_val: 'HExpr | None' = None # for FieldSet: new value
    # Lambda/Closure: op="Lambda"
    lambda_params: tuple[str, ...] | None = None           # parameter names
    # Lambda params declared @mut (write-back semantics, like @mut params of
    # named functions); None/() for plain value-semantics params.
    lambda_mut_params: tuple[str, ...] | None = None
    lambda_body: 'HExpr | None' = None                     # body expression
    captures: tuple[tuple[str, str], ...] | None = None    # ((name, mode), ...) captured vars
    # Call: explicit instantiation type args (`identity<Int>(x)`), as type
    # display strings; None when the call spelled no type args.
    type_args: tuple[str, ...] | None = None
    # Perform: op="Perform"
    effect_op: str | None = None        # effect operation name, e.g. 'emit'
    perform_args: tuple['HExpr', ...] | None = None  # arguments to the operation
    # Handle: op="Handle"
    handle_effect: str | None = None    # effect type name being handled
    handle_cases: tuple[tuple[str, str, 'HExpr'], ...] | None = None  # ((op, param, body), ...)
    handle_body: 'HExpr | None' = None  # the continuation expression


@dataclass(slots=True)
class HFun:
    sym: Any
    params: list[tuple[Any, Ty]]
    dict_params: list[tuple[str, Any]]  # (TraitName, DictTy placeholder)
    ret_ty: Ty
    where_cls: list[ClassConstraint]
    body: HExpr
    param_modes: dict[str, ModeInfo] | None = None
    # Pre-monomorphization name, set by monomorphize.py on its clones
    # (identity$Int, catch_$ho1). Runtime failure messages that embed a
    # function name (match_fail) use this so the monomorphized native lane
    # binds the SAME string the unspecialized interpreter binds. None for
    # every non-clone.
    origin_sym: Any = None
    # Module-level constants this function initializes (only set on the
    # synthesized __module_init function): the interpreter runs it before
    # the entry point and publishes these names as globals.
    globals_decl: tuple[str, ...] = ()


class HIRBuilder:
    """Build a typed, frozen HIR from the parsed AST and inference side-tables."""

    def __init__(self, tables: InferSideTables, id_map: dict[int, Any] | None = None) -> None:
        self.t = tables
        self.id_map = id_map or {}
        # Reverse map: id(orig_obj) -> frozen AstNode, built lazily in build()
        self._orig_to_frozen: dict[int, mast.AstNode] = {}
        # Effect op names collected from EffectDeclaration nodes
        self._effect_op_names: set[str] = set()
        # Enum variant constructors: variant_name -> enum_name
        self._variant_to_enum: dict[str, str] = {}
        # Trait method names: declared in traits (InterfaceDefinition) or
        # provided by an implement block (mangled __impl$... function names).
        self._trait_method_names: set[str] = set()
        # Type names with impl methods (targets of implement blocks), used to
        # recognize static calls `Type.method(args)`.
        self._impl_type_names: set[str] = set()
        # All declared type names (structs, enums) plus runtime type
        # constructors — used to tell `Type.method()` from `variable.method()`.
        self._type_names: set[str] = {"Vec", "vector"}
        # Compilation-unit file name, used as the last-resort location in
        # lowering diagnostics (most parsed nodes carry no SourceLocation).
        self._root_file: str = "<unknown file>"

    @compiler_phase
    def build(self, root: mast.AstNode) -> list[HFun]:
        funcs: list[HFun] = []
        # The compilation-unit file name for diagnostics: the frozen root
        # itself carries "<unknown>", its Module child carries the real path.
        for cand in (root, *root.children):
            fname = getattr(getattr(cand, 'span', None), 'file', None)
            if fname and fname != "<unknown>":
                self._root_file = fname
                break

        # Build reverse map: id(orig_obj) -> frozen AstNode
        def index_nodes(n: mast.AstNode) -> None:
            orig = self.id_map.get(n.node_id)
            if orig is not None:
                self._orig_to_frozen[id(orig)] = n
            for c in n.children:
                index_nodes(c)
        index_nodes(root)

        # Collect effect operation names so bare FunctionCall(name=op) can be identified as Perform
        self._effect_op_names: set[str] = set()
        self._trait_method_names = set()
        for orig in self.id_map.values():
            if isinstance(orig, fast.EffectDeclaration):
                for op in (getattr(orig, 'operations', []) or []):
                    if hasattr(op, 'name'):
                        self._effect_op_names.add(str(op.name))
            if isinstance(orig, fast.StructDefinition):
                sname = getattr(orig, 'name', None)
                if sname is not None:
                    self._type_names.add(str(sname))
            if isinstance(orig, fast.EnumDefinition):
                self._type_names.add(str(getattr(orig, 'name', '') or ''))
                ename = str(getattr(orig, 'name', '') or '')
                for v in (getattr(orig, 'variants', []) or []):
                    vname = getattr(v, 'name', None)
                    if vname is not None:
                        self._variant_to_enum[str(vname)] = ename
            # Trait method names: from trait declarations...
            if isinstance(orig, fast.InterfaceDefinition):
                for m in (getattr(orig, 'methods', []) or []):
                    mname = getattr(m, 'name', None)
                    if mname is not None:
                        self._trait_method_names.add(str(mname))
            # ...and from desugared implement-block functions (__impl$T$Ty$m),
            # so impl-only methods dispatch even without a trait declaration.
            if isinstance(orig, fast.FunctionDeclaration):
                parsed = parse_impl_method_name(str(getattr(orig, 'name', '') or ''))
                if parsed is not None:
                    self._trait_method_names.add(parsed[2])
                    self._impl_type_names.add(parsed[1])

        # Module-level `let` bindings become module constants: collected (in
        # program order) into a synthesized __module_init function that the
        # interpreter runs before the entry point, publishing the bound names
        # as globals. Before this existed a top-level `let PI = 3;` compiled
        # and every read of PI misbehaved at run time.
        module_lets: list[tuple[mast.AstNode, Any]] = []

        def visit(n: mast.AstNode, in_fn: bool = False) -> None:
            orig = self.id_map.get(n.node_id)
            if not in_fn and isinstance(orig, fast.LetStatement):
                module_lets.append((n, orig))
            if isinstance(orig, fast.ComptimeFunction):
                # A ComptimeFunction IS a FunctionDeclaration; hoisting it here
                # would compile it as an ordinary run-time function and lose
                # the "evaluate at compile time" meaning entirely.
                raise UnsupportedConstruct(
                    f"comptime fn {getattr(orig, 'name', '?')!r} at "
                    f"{self._span_text(n.span, orig)} is not supported: "
                    "compile-time evaluation is not implemented",
                    location=self._loc(n.span, orig))
            if isinstance(orig, fast.FunctionDeclaration):
                # Determine return type from side tables for this node or fallback
                ret = self.t.apply_tyenv(self.t.types.get(n.node_id, "Unit"))  # type: ignore[index]
                body_hexpr = self._from_orig_expr(orig.body if hasattr(orig, 'body') else [], n)
                if body_hexpr is None:
                    body_hexpr = HExpr(
                        node_id=n.node_id,
                        kind=n.kind,
                        args=tuple(),
                        ty=ret,
                        effects=EffectSet(frozenset()),
                        suspends=self.t.suspends_node(n.node_id) if hasattr(self.t, "suspends_node") else False,
                        sym=getattr(orig, 'name', None),
                        span=n.span,
                        op="Block",
                        operands=tuple(),
                    )
                params: list[tuple[Any, Ty]] = []
                param_modes: dict[str, ModeInfo] = {}
                for p in getattr(orig, 'params', []) or []:
                    pname = getattr(p, 'name', None)
                    pty = getattr(p, 'type_annotation', None)
                    if pty is None:
                        pty = self.t.types.get(getattr(p, 'node_id', -1), "Unknown") if hasattr(p, 'node_id') else "Unknown"  # type: ignore[index]
                    params.append((pname, pty))
                    # Extract modes if available (explicit mode field or a
                    # @mut/@const-style ModeTypeAnnotation on the type)
                    pmode = self._param_modeinfo(p)
                    if pname is not None:
                        param_modes[str(pname)] = pmode
                # Const-generic receiver dimensions (recorded by the impl
                # desugar): bind e.g. N to the receiver's runtime length at
                # method entry so scalar-fallback code like `for i in 1..N`
                # reads the actual vector size. `vector[T,N]` binds N =
                # __vec_dim(self, 0); `vector[vector[T,N],M]` binds M/N to
                # dims 0/1.
                const_dims = getattr(orig, '_const_dims', None) or ()
                param_names = {str(p0) for (p0, _t) in params}
                if const_dims and 'self' in param_names:
                    dim_lets: list[HExpr] = []
                    for (dim_name, dim_idx) in const_dims:
                        if dim_name in param_names:
                            continue  # an explicit param shadows the binding
                        dim_call = self._mk_hexpr(
                            n.node_id, "Expr", "int", n.span, op="Call",
                            callee="__vec_dim",
                            operands=(self._mk_hexpr(n.node_id, "Expr", "Unknown",
                                                     n.span, op="Var", var_name="self"),
                                      self._mk_hexpr(n.node_id, "Expr", "int",
                                                     n.span, op="Literal",
                                                     literal=int(dim_idx))))
                        dim_lets.append(self._mk_hexpr(
                            n.node_id, "Stmt", "Unit", n.span, op="Let",
                            bindings=((str(dim_name), dim_call),)))
                    if dim_lets:
                        old_ops = (body_hexpr.operands
                                   if body_hexpr.op == "Block" and body_hexpr.operands is not None
                                   else (body_hexpr,))
                        body_hexpr = self._mk_hexpr(
                            n.node_id, "Block", body_hexpr.ty, n.span,
                            op="Block", operands=(*dim_lets, *old_ops))
                hfun = HFun(
                    sym=getattr(orig, 'name', 'fun'),
                    params=params,
                    dict_params=[],
                    ret_ty=ret,
                    where_cls=[],
                    body=body_hexpr,
                    param_modes=param_modes or None,
                )
                funcs.append(hfun)
            # Recurse. Only DIRECT module-level LetStatements are module
            # constants: entering any function-like body (named function or
            # lambda) — or the subexpressions of a module-level let's own
            # initializer — sets in_fn so nested lets stay local to their
            # scope instead of being hoisted into __module_init (where their
            # initializers referenced unbound locals and aborted).
            inside = in_fn or isinstance(
                orig, (fast.FunctionDeclaration, fast.LambdaExpression,
                       fast.LetStatement))
            for c in n.children:
                visit(c, inside)

        visit(root)

        if module_lets:
            init_ops: list[HExpr] = []
            global_names: list[str] = []
            for (n, orig) in module_lets:
                for b in getattr(orig, 'bindings', []) or []:
                    name = getattr(b, 'identifier', None)
                    init_node = getattr(b, 'initializer', None)
                    he = self._from_orig_expr(init_node, n)
                    if name is None or he is None:
                        raise NotImplementedError(
                            f"module-level let {name!r}: could not lower its "
                            "initializer — refusing to drop the binding")
                    # Assign (not Let) so the MIR slot keeps the source name:
                    # the interpreter publishes exactly these slots as globals.
                    init_ops.append(self._mk_hexpr(
                        n.node_id, "Stmt", "Unit", n.span, op="Assign",
                        var_name=str(name), assign_value=he))
                    global_names.append(str(name))
            body = self._mk_hexpr(root.node_id, "Block", "Unit", root.span,
                                  op="Block", operands=tuple(init_ops))
            funcs.append(HFun(sym="__module_init", params=[], dict_params=[],
                              ret_ty="Unit", where_cls=[], body=body,
                              globals_decl=tuple(global_names)))

        # Default effect handlers: an effect operation declared as
        # `op(params) -> T = expr;` compiles its default expression to a
        # plain function `__effect_default$Effect$op`. The interpreter calls
        # it when a perform finds NO handler in scope (capability-style
        # effects: absence answers the default; effects without defaults
        # still fail loudly). Handlers installed by `handle` always win.
        seen_effects: set[int] = set()
        for orig in self.id_map.values():
            if not isinstance(orig, fast.EffectDeclaration) or id(orig) in seen_effects:
                continue
            seen_effects.add(id(orig))
            eff_name = str(getattr(orig, 'name', '') or '')
            frozen = self._orig_to_frozen.get(id(orig), root)
            for op in (getattr(orig, 'operations', []) or []):
                dexpr = getattr(op, '_default_expr', None)
                dparams: list[tuple[Any, Ty]] = []
                for p in (getattr(op, 'params', []) or []):
                    pname = getattr(p, 'name', None)
                    pty = getattr(p, 'type_annotation', None) or "Unknown"
                    dparams.append((pname, pty))
                # Runtime mapping: `op(params) -> T with SYMBOL` binds the op
                # to a named runtime primitive (effect_mapping.mx: Mutex and
                # Thread ops map onto EFFECT_MUTEX_* / EFFECT_SPAWN / ...).
                # Compile a thunk __effect_runtime$Effect$op whose body calls
                # __mx_effect_runtime$SYMBOL(params); the interpreter resolves
                # that callee against its runtime shim table and fails loudly
                # for symbols it cannot honor.
                csym = getattr(op, 'c_effect', None)
                if csym:
                    operands = tuple(
                        self._mk_hexpr(frozen.node_id, "Expr", "Unknown",
                                       frozen.span, op="Var", var_name=pn)
                        for (pn, _pty) in dparams)
                    rt_body = self._mk_hexpr(
                        frozen.node_id, "Expr", "Unknown", frozen.span,
                        op="Call",
                        callee=f"{EFFECT_RUNTIME_CALL_PREFIX}{csym}",
                        operands=operands)
                    funcs.append(HFun(
                        sym=f"__effect_runtime{IMPL_SEP}{eff_name}{IMPL_SEP}{op.name}",
                        params=list(dparams), dict_params=[], ret_ty="Unknown",
                        where_cls=[], body=rt_body))
                if dexpr is None:
                    continue
                body_he = self._from_orig_expr(dexpr, frozen)
                if body_he is None:
                    raise NotImplementedError(
                        f"effect {eff_name}.{op.name}: could not lower the "
                        "declared default expression — refusing to drop it")
                funcs.append(HFun(
                    sym=f"__effect_default{IMPL_SEP}{eff_name}{IMPL_SEP}{op.name}",
                    params=dparams, dict_params=[], ret_ty="Unknown",
                    where_cls=[], body=body_he))

        # Script files. A file whose top level holds executable statements
        # and declares no `main` runs those statements as main's body
        # (chapter 1: examples/hello.mx is one line of `print`). Before this
        # existed the fallback below synthesized an EMPTY main for such a
        # file, so `print("hi")` compiled, ran and printed nothing. Top-level
        # statements next to a declared `main` are refused rather than
        # dropped: there is no order in which both could run.
        script_stmts = self._script_statements(root)
        if script_stmts:
            first = script_stmts[0]
            first_frozen = self._orig_to_frozen.get(id(first), root)
            if any(f.sym == "main" for f in funcs):
                raise UnsupportedConstruct(
                    f"top-level statement at "
                    f"{self._span_text(first_frozen.span, first)} in a file "
                    "that also declares `fn main`: a file runs EITHER its "
                    "top-level statements as a script OR `main`, never both; "
                    "move the statement into main",
                    location=self._loc(first_frozen.span, first))
            body_hexpr = self._from_orig_expr(script_stmts, first_frozen)
            assert body_hexpr is not None  # a non-empty list never lowers to None
            funcs.append(HFun(sym="main", params=[], dict_params=[],
                              ret_ty="Unit", where_cls=[], body=body_hexpr))

        # Fallback: if no functions found, produce a default wrapper
        if not funcs:
            ty = self.t.apply_tyenv(self.t.types.get(root.node_id, "Unit"))  # type: ignore[union-attr]
            hexpr = HExpr(
                node_id=root.node_id,
                kind=root.kind,
                args=tuple(),
                ty=ty,
                effects=EffectSet(frozenset()),
                suspends=self.t.suspends_node(root.node_id) if hasattr(self.t, "suspends_node") else False,
                sym=self.t.sym_of(root.node_id) if hasattr(self.t, "sym_of") else None,
                span=root.span,
                op="Block",
                operands=tuple(),
            )
            funcs.append(HFun(sym="main", params=[], dict_params=[], ret_ty=ty, where_cls=[], body=hexpr))
        return funcs

    def _script_statements(self, root: mast.AstNode) -> list[Any]:
        """The entry file's top-level statements that DO something at run
        time, in program order.

        The parser wraps a file in a Module named "main"; the module loader
        renames every imported file's module to its path, so "main" is the
        entry file. Declarations (functions, types, imports, module-level
        `let`, nested `module` blocks) are excluded: each is realized
        elsewhere. Imported modules are not scanned, because a library
        file's stray statement must not run behind the importer's back;
        the loader rejects nothing there today, which is a known gap.
        """
        prog = self.id_map.get(root.node_id)
        stmts: list[Any] = []

        def executable(s: Any) -> bool:
            return type(s).__name__ not in SCRIPT_EXCLUDED_KINDS

        for s in getattr(prog, "statements", []) or []:
            if isinstance(s, fast.Module):
                if str(getattr(s, "name", "")) != "main":
                    continue
                body = getattr(s, "body", None)
                stmts.extend(t for t in (getattr(body, "statements", []) or [])
                             if executable(t))
            elif executable(s):
                stmts.append(s)  # a Program built without the module wrapper
        return stmts

    def _mk_hexpr(self, node_id: int, kind: str, ty: Ty, span: mast.Span, **kw: Any) -> HExpr:
        return HExpr(
            node_id=node_id,
            kind=kind,
            args=tuple(),
            ty=ty,
            effects=EffectSet(frozenset()),
            suspends=self.t.suspends_node(node_id) if hasattr(self.t, "suspends_node") else False,
            sym=None,
            span=span,
            **kw,
        )

    def _frozen_for(self, orig: Any, fallback: mast.AstNode) -> mast.AstNode:
        """Return the frozen AstNode corresponding to orig, or fallback."""
        return self._orig_to_frozen.get(id(orig), fallback)

    def _from_orig_expr(self, orig: Any, frozen_ctx: mast.AstNode) -> HExpr | None:
        """Build an HExpr from an original AST node or list of nodes.

        frozen_ctx: a frozen node to provide node_id/span context when the original
        node lacks a corresponding frozen node in id_map traversal.

        None contract (the ONLY one): this returns None if and only if `orig`
        is None — i.e. the source genuinely had nothing there (an absent else
        branch, an extern function's missing body). Every other outcome is
        either a real HExpr or a raised HIRLoweringError. Callers may test the
        `orig is None` case; they must NEVER treat a None as "skip this
        construct", which is how if-let, while-let, `unsafe { }`, `@mut e`,
        list literals, `for`, `as` casts and early `return` each silently
        vanished from compiled programs.
        """
        if orig is None:
            return None

        # Resolve the best frozen node for this orig object
        if not isinstance(orig, list):
            frozen_ctx = self._frozen_for(orig, frozen_ctx)

        def ctx_for(child: Any) -> mast.AstNode:
            return self._frozen_for(child, frozen_ctx) if child is not None else frozen_ctx

        # Lists become Block
        if isinstance(orig, list):
            ops: list[HExpr] = []
            for item in orig:
                he = self._from_orig_expr(item, self._frozen_for(item, frozen_ctx) if item is not None else frozen_ctx)
                if he is not None:
                    ops.append(he)
            return self._mk_hexpr(frozen_ctx.node_id, "Block", self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unit")), frozen_ctx.span, op="Block", operands=tuple(ops))

        # Literals
        if isinstance(orig, fast.Literal):
            ty = self.t.apply_tyenv(getattr(orig, 'type_var', None) or self.t.types.get(frozen_ctx.node_id, "Unknown"))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span, op="Literal", literal=getattr(orig, 'value', None))

        # TupleLiteral: `()` is the unit value (an empty Block lowers to the
        # unit constant); `(a, b)` is an anonymous struct — see
        # TUPLE_STRUCT_PREFIX for why that representation and not a new
        # runtime value. An element that cannot lower is a LOUD error, never
        # a short tuple (the same rule StructInstantiation enforces).
        if isinstance(orig, fast.TupleLiteral):
            elements = getattr(orig, 'elements', []) or []
            if not elements:
                ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unit"))
                return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                      op="Block", operands=())
            arity = len(elements)
            if arity < TUPLE_MIN_ARITY:
                # The grammar cannot build this (`(e)` is grouping and `(e,)`
                # is a syntax error), so reaching it means a pass synthesized
                # a 1-tuple; say so instead of inventing a `__tuple1` layout
                # nothing else in the compiler knows about.
                raise UnsupportedConstruct(
                    f"tuple literal: {arity}-element tuples do not exist — "
                    "`(e)` is parenthesized grouping and a tuple has two or "
                    "more elements",
                    location=self._loc(frozen_ctx.span, orig))
            field_exprs: list[tuple[str, HExpr]] = []
            for i, el in enumerate(elements):
                he = self._from_orig_expr(el, ctx_for(el))
                if he is None:
                    raise UnsupportedConstruct(
                        f"tuple literal: could not lower element {i} "
                        f"({type(el).__name__}) — refusing to build a short "
                        "tuple", location=self._loc(frozen_ctx.span, orig))
                field_exprs.append((tuple_field_name(i, arity), he))
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unknown"))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                  op="Struct",
                                  struct_name=tuple_struct_name(arity),
                                  fields=tuple(field_exprs),
                                  locality="local")

        # Variables
        if isinstance(orig, fast.Variable):
            ty = self.t.apply_tyenv(getattr(orig, 'type_var', None) or self.t.types.get(frozen_ctx.node_id, "Unknown"))
            vname = getattr(orig, 'name', None)
            # `null` is the null-pointer literal (the parser produces a plain
            # Variable for it). Lowered as a literal so it never hits strict
            # name resolution; the interpreter represents null as Python None.
            if vname == "null":
                return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                      op="Literal", literal=None)
            # A bare zero-arg enum variant name in expression position
            # (`Point`) constructs the variant, it is not a variable read.
            if isinstance(vname, str) and vname in self._variant_to_enum:
                return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                      op="MakeVariant",
                                      enum_name=self._variant_to_enum[vname],
                                      variant_name=vname, operands=())
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span, op="Var", var_name=vname)

        # QualifiedName: `a` → Var; `a.b.c` → chained FieldGet
        if isinstance(orig, fast.QualifiedName):
            parts = list(getattr(orig, 'parts', []) or [])
            if not parts:
                raise HIRCompilerBug(
                    f"QualifiedName with no parts at "
                    f"{self._span_text(frozen_ctx.span)}",
                    location=self._loc(frozen_ctx.span, orig))
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unknown"))
            # `Enum.Variant` names a nullary variant, it is not a field read of
            # a variable called `Enum` (which is what the FieldGet chain below
            # produced — "Unbound variable 'Color'" at run time).
            if (len(parts) == 2
                    and self._variant_to_enum.get(str(parts[1])) == str(parts[0])):
                return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                      op="MakeVariant", enum_name=str(parts[0]),
                                      variant_name=str(parts[1]), operands=())
            if len(parts) == 1:
                return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span, op="Var", var_name=str(parts[0]))
            # Build chained FieldGet: start from the first part as a Var
            current: HExpr = self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span, op="Var", var_name=str(parts[0]))
            for field in parts[1:]:
                current = self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span, op="FieldGet", base=current, field_name=str(field))
            return current

        # Borrow/move/exclave expressions: at runtime these evaluate to the
        # referenced value (aliasing and ownership rules are enforced earlier
        # by the frozen borrow checker, not at HIR/MIR level).
        # `borrow x` / `borrow x as T` carries the borrowed name as a bare
        # string, like BorrowShared/BorrowUnique. It had no lowering: the whole
        # expression vanished, so `let y = borrow x;` bound nothing.
        if isinstance(orig, (fast.BorrowShared, fast.BorrowUnique, fast.Move,
                             fast.BorrowExpression)):
            var = getattr(orig, 'variable', None)
            if isinstance(var, str):
                ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
                return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                      op="Var", var_name=var)
            return self._require(
                self._from_orig_expr(var, ctx_for(var)) if var is not None else None,
                orig, frozen_ctx, "the borrowed/moved operand")
        # `&e` / `&mut e` where `e` is NOT a plain name (`&x.f`, `&v[0]`): the
        # parser only builds BorrowShared/BorrowUnique for the bare-name form
        # and an AddressOf for everything else. Same rule as the bare-name form
        # — at runtime it evaluates to the referenced value; aliasing and
        # ownership are enforced earlier by the frozen borrow checker. (It had
        # no lowering, so `&x.f` silently became nothing.)
        if isinstance(orig, AddressOf):
            inner = getattr(orig, 'expr', None)
            return self._require(
                self._from_orig_expr(inner, ctx_for(inner)) if inner is not None else None,
                orig, frozen_ctx, "the `&`-operand")
        if isinstance(orig, fast.ExclaveExpression):
            inner = getattr(orig, 'expression', None)
            if isinstance(inner, str):
                ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
                return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                      op="Var", var_name=inner)
            return self._from_orig_expr(inner, ctx_for(inner)) if inner is not None else None
        # Mode-annotated expression (`@const node.data`, `@mut x.f`): like the
        # borrow forms above, at runtime it evaluates to the underlying value.
        # (Previously unhandled: it lowered to None and e.g. the payload of
        # `Some(@const node.data)` was silently dropped, leaving a nullary
        # Some that blew up at pattern-match time.)
        if isinstance(orig, fast.ModeExpression):
            inner = getattr(orig, 'expression', None)
            return self._from_orig_expr(inner, ctx_for(inner)) if inner is not None else None

        # Option constructors in expression position: Some(x) / None.
        # (In pattern position these are handled by _convert_pattern.)
        if isinstance(orig, fast.SomeExpression):
            inner = getattr(orig, 'value', None)
            inner_he = self._from_orig_expr(inner, ctx_for(inner)) if inner is not None else None
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                  op="MakeVariant",
                                  enum_name=self._variant_to_enum.get("Some", "Option"),
                                  variant_name="Some",
                                  operands=(inner_he,) if inner_he is not None else ())
        if isinstance(orig, fast.NoneExpression):
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                  op="MakeVariant",
                                  enum_name=self._variant_to_enum.get("None", "Option"),
                                  variant_name="None", operands=())

        # PrintStatement: `print(args)` — lower to a builtin call
        if isinstance(orig, fast.PrintStatement):
            arg_exprs = []
            for a in getattr(orig, 'arguments', []) or []:
                he = self._from_orig_expr(a, ctx_for(a))
                if he is not None:
                    arg_exprs.append(he)
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unit'))
            # `print(...)` is a GRAMMAR PRODUCTION (print is a lexer
            # keyword), not an ordinary call, so it always means the
            # builtin — marked so a module function named `print`
            # anywhere in the program cannot capture it.
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                  op="Call", callee=BUILTIN_CALL_PREFIX + "print",
                                  operands=tuple(arg_exprs))

        # Dedicated Resume node: `resume(v)` inside a handle case
        if isinstance(orig, fast.Resume):
            val_node = getattr(orig, 'value', None)
            val_exprs: tuple = ()
            if val_node is not None and hasattr(val_node, '__class__') and isinstance(val_node, fast.Node):
                he = self._from_orig_expr(val_node, ctx_for(val_node))
                if he is not None:
                    val_exprs = (he,)
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                  op="Resume", perform_args=val_exprs)

        # Function calls (including perform-as-bare-call and resume)
        if isinstance(orig, fast.FunctionCall):
            callee = str(getattr(orig, 'name', None) or '')
            args_exprs = []
            for a in getattr(orig, 'arguments', []) or []:
                he = self._from_orig_expr(a, ctx_for(a))
                if he is not None:
                    args_exprs.append(he)
            ty = self.t.apply_tyenv(getattr(orig, 'type_var', None) or self.t.types.get(frozen_ctx.node_id, "Unknown"))
            # `resume(v)` in a handle case → special Resume op (returns value to handler caller)
            if callee == 'resume':
                return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                      op="Resume", perform_args=tuple(args_exprs))
            # Bare `perform emit(x)` parsed as FunctionCall when callee is a known effect op
            if callee in self._effect_op_names:
                return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                      op="Perform", effect_op=callee, perform_args=tuple(args_exprs))
            # Enum variant constructor call, e.g. `Some(5)` / `Cons(h, t)`
            if callee in self._variant_to_enum:
                return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                      op="MakeVariant", enum_name=self._variant_to_enum[callee],
                                      variant_name=callee, operands=tuple(args_exprs))
            # Builtin Option constructors when no enum declares them: the docs
            # treat Option as a language-provided type.
            if callee in ("Some", "None"):
                return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                      op="MakeVariant", enum_name="Option",
                                      variant_name=callee, operands=tuple(args_exprs))
            # Builtin Result constructors when no enum declares them: like
            # Option, the docs treat Result<T, E> as language-provided (a user
            # enum declaring Ok/Err takes precedence via _variant_to_enum).
            if callee in ("Ok", "Err"):
                return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                      op="MakeVariant", enum_name="Result",
                                      variant_name=callee, operands=tuple(args_exprs))
            # Preserve explicit instantiation type args (`identity<Int>(x)`)
            # for the (optional) monomorphization pass.
            raw_targs = getattr(orig, 'type_args', None) or ()
            targs = tuple(t for t in (mast._type_display(a) for a in raw_targs)
                          if isinstance(t, str))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span, op="Call", callee=callee, operands=tuple(args_exprs),
                                  type_args=targs or None)

        # CallExpression: a call whose callee is an arbitrary EXPRESSION rather
        # than a name — `(fn(x) -> x)(1)`, `v[0](41)`, `mk()(2)`. HIR's Call op
        # carries a callee NAME, and the interpreter/native backends already
        # resolve a name bound to a closure value by calling that closure, so
        # bind the computed callee to a fresh local first and call it by that
        # name. (This node had no lowering: the whole call vanished and the
        # expression evaluated to unit.)
        if isinstance(orig, fast.CallExpression):
            callee_node = getattr(orig, 'callee', None)
            callee_he = self._require(
                self._from_orig_expr(callee_node, ctx_for(callee_node))
                if callee_node is not None else None,
                orig, frozen_ctx, "the callee expression of an indirect call")
            arg_exprs = []
            for a in getattr(orig, 'arguments', []) or []:
                arg_exprs.append(self._require(
                    self._from_orig_expr(a, ctx_for(a)), a, frozen_ctx,
                    "an argument of an indirect call"))
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unknown"))
            tmp = f"__callee{frozen_ctx.node_id}"
            bind = self._mk_hexpr(frozen_ctx.node_id, "Stmt", "Unit", frozen_ctx.span,
                                  op="Let", bindings=((tmp, callee_he),))
            call = self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                  op="Call", callee=tmp, operands=tuple(arg_exprs))
            return self._mk_hexpr(frozen_ctx.node_id, "Block", ty, frozen_ctx.span,
                                  op="Block", operands=(bind, call))

        # BinaryOperation / ComparisonExpression (same structure, both use left/operator/right)
        if isinstance(orig, (fast.BinaryOperation, fast.ComparisonExpression)):
            lnode = getattr(orig, 'left', None)
            rnode = getattr(orig, 'right', None)
            l = self._from_orig_expr(lnode, ctx_for(lnode))
            r = self._from_orig_expr(rnode, ctx_for(rnode))
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unknown"))
            op_sym = getattr(orig, 'operator', None)
            # ComparisonOperator may be an enum instance; get its value string
            if hasattr(op_sym, 'value'):
                op_sym = op_sym.value
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span, op="BinOp", binop=op_sym, left=l, right=r)

        # IfStatement / IfExpression
        if isinstance(orig, (fast.IfStatement, fast.IfExpression)):
            cond_node = getattr(orig, 'condition', None)
            c = self._from_orig_expr(cond_node, ctx_for(cond_node))
            # IfStatement uses then_body/else_body; IfExpression uses then_branch/else_branch
            then_node = getattr(orig, 'then_body', None) or getattr(orig, 'then_branch', None)
            else_node = getattr(orig, 'else_body', None) or getattr(orig, 'else_branch', None)
            tb = self._from_orig_expr(then_node, ctx_for(then_node))
            eb = self._from_orig_expr(else_node, ctx_for(else_node)) if else_node is not None else None
            # Flatten block bodies into operand lists
            def as_ops(h: HExpr | None) -> tuple[HExpr, ...]:
                if h is None:
                    return tuple()
                if h.op == "Block" and h.operands is not None:
                    return h.operands
                return (h,)
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unit"))
            # An else-less `if` always evaluates to unit (standard statement
            # rule): the then-arm runs for its effects only and its value is
            # discarded. else_ops=None marks the else-less form for lowering;
            # an if/else keeps a (possibly empty) tuple and merges both arms.
            if eb is None:
                ty = "Unit"
            return self._mk_hexpr(frozen_ctx.node_id, "Stmt", ty, frozen_ctx.span, op="If", cond=c, then_ops=as_ops(tb), else_ops=as_ops(eb) if eb is not None else None)

        # IfLetExpression: `if let PAT = expr { then } else { else }`.
        # Desugared to a two-arm Match (PAT => then, _ => else) so the
        # pattern binding and both branches survive lowering. This node
        # used to fall through to the None fallback and be silently
        # dropped — whole function bodies degraded to unit.
        if isinstance(orig, fast.IfLetExpression):
            val_node = getattr(orig, 'value', None)
            scrut = self._from_orig_expr(val_node, ctx_for(val_node))
            if scrut is None:
                raise NotImplementedError(
                    "if let: could not lower the matched expression "
                    f"({type(val_node).__name__})")
            pat_node = getattr(orig, 'pattern', None)
            pat = self._checked_refutable_pattern(pat_node, construct="if let")
            then_node = getattr(orig, 'then_branch', None)
            then_he = self._from_orig_expr(then_node, ctx_for(then_node))
            if then_he is None:
                raise NotImplementedError(
                    "if let: could not lower the then-branch "
                    f"({type(then_node).__name__})")
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unit"))
            else_node = getattr(orig, 'else_branch', None)
            if else_node is not None:
                else_he = self._from_orig_expr(else_node, ctx_for(else_node))
                if else_he is None:
                    raise NotImplementedError(
                        "if let: could not lower the else-branch "
                        f"({type(else_node).__name__})")
            else:
                # No else: an else-less `if let` is a statement — BOTH arms
                # evaluate to unit. The then-body runs for its effects only
                # (early `return` inside it still works via the epilogue),
                # so wrap it in a block whose tail is unit rather than
                # letting its value merge with the unit of the miss arm.
                ty = "Unit"
                unit_he = self._mk_hexpr(frozen_ctx.node_id, "Block", ty,
                                         frozen_ctx.span, op="Block", operands=())
                then_he = self._mk_hexpr(frozen_ctx.node_id, "Block", ty,
                                         frozen_ctx.span, op="Block",
                                         operands=(then_he, unit_he))
                else_he = self._mk_hexpr(frozen_ctx.node_id, "Block", ty,
                                         frozen_ctx.span, op="Block", operands=())
            arms = ((pat, then_he), (HPattern(kind="wildcard"), else_he))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                  op="Match", scrutinee=scrut,
                                  cases=(then_he, else_he), match_arms=arms)

        # WhileLetStatement: `while let PAT = expr { body }` — loop while
        # PAT matches the (re-evaluated) expr, binding pattern variables in
        # the body. Previously silently dropped like IfLetExpression.
        if isinstance(orig, fast.WhileLetStatement):
            val_node = getattr(orig, 'value', None)
            scrut = self._from_orig_expr(val_node, ctx_for(val_node))
            if scrut is None:
                raise NotImplementedError(
                    "while let: could not lower the matched expression "
                    f"({type(val_node).__name__})")
            pat = self._checked_refutable_pattern(getattr(orig, 'pattern', None),
                                                  construct="while let")
            body_node = getattr(orig, 'body', None)
            body_he = self._from_orig_expr(body_node, ctx_for(body_node))
            if body_he is not None and body_he.op == "Block" and body_he.operands is not None:
                body_ops = body_he.operands
            elif body_he is not None:
                body_ops = (body_he,)
            else:
                body_ops = tuple()
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unit"))
            return self._mk_hexpr(frozen_ctx.node_id, "Stmt", ty, frozen_ctx.span,
                                  op="WhileLet", scrutinee=scrut,
                                  loop_pattern=pat, loop_body=body_ops)

        # MatchExpression: carry (pattern, body) pairs into HIR.
        # TODO(pattern-typing): pattern variable types are not yet threaded through
        # the constraint emitter; typing of bindings currently falls back to the
        # arm-body node types (frozen_constraint_emitter is owned by another agent).
        if isinstance(orig, fast.MatchExpression):
            expr = self._require(
                self._from_orig_expr(getattr(orig, 'expression', None), frozen_ctx),
                orig, frozen_ctx, "the scrutinee of a `match`")
            cases = getattr(orig, 'cases', []) or []
            case_exprs: list[HExpr] = []
            arms: list[tuple[HPattern, HExpr]] = []
            for case in cases:
                # Parser Option sugar: ('some', var_name, body) / ('none', None, body)
                if len(case) == 3 and case[0] in ('some', 'none'):
                    tag, var, case_body = case
                    if tag == 'some':
                        pat = HPattern(kind="ctor", name="Some", enum_name="Option",
                                       subpatterns=(HPattern(kind="var", name=str(var)),))
                    else:
                        pat = HPattern(kind="ctor", name="None", enum_name="Option")
                else:
                    pattern, case_body = case
                    pat = self._convert_pattern(pattern)
                # A dropped arm silently changes which arm wins for a value
                # (the next arm takes over), so an un-lowerable arm body is a
                # hard error rather than a skip.
                case_hexpr = self._require(
                    self._from_orig_expr(case_body, frozen_ctx), orig,
                    frozen_ctx, "the body of a `match` arm")
                case_exprs.append(case_hexpr)
                arms.append((pat, case_hexpr))
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unit"))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span, op="Match",
                                  scrutinee=expr, cases=tuple(case_exprs), match_arms=tuple(arms))

        # VariantInstance: EnumName::Variant(field=expr, ...)
        if isinstance(orig, fast.VariantInstance):
            enum_name = str(getattr(orig, 'enum_name', '') or '')
            variant_name = str(getattr(orig, 'variant_name', '') or '')
            fvals = getattr(orig, 'field_values', None) or []
            # field_values may be a dict {name: expr} or a list of (name, expr)
            if isinstance(fvals, dict):
                items = list(fvals.items())
            else:
                items = [(fn, fe) for (fn, fe) in fvals]
            payload: list[HExpr] = []
            for (_fname, fexpr) in items:
                he = self._from_orig_expr(fexpr, ctx_for(fexpr))
                if he is not None:
                    payload.append(he)
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unknown"))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                  op="MakeVariant", enum_name=enum_name,
                                  variant_name=variant_name, operands=tuple(payload))

        # ForStatement: `for x in iterable { body }` — desugared here to the
        # existing While machinery (no dedicated loop op below HIR):
        #     let __for_it = iterable; let __for_i = 0; let __for_n = len(it);
        #     while __for_i < __for_n {
        #         let x = __index_get(__for_it, __for_i);
        #         ...body...
        #         __for_i = __for_i + 1;
        #     }
        # Works for every runtime sequence (`0..N` ranges, vectors, Vec).
        # Unsupported shapes fail loudly instead of dropping the loop — the
        # historical seam here was for-loops vanishing so scalar fallbacks
        # returned partial results.
        if isinstance(orig, fast.ForStatement):
            loop_var = str(getattr(orig, 'iterator', '') or '')
            it_node = getattr(orig, 'iterable', None)
            iter_he = self._from_orig_expr(it_node, ctx_for(it_node))
            if not loop_var or iter_he is None:
                raise NotImplementedError(
                    "for loop: unsupported shape (iterator "
                    f"{loop_var!r}, iterable {type(it_node).__name__}) — "
                    "refusing to drop the loop")
            body_node = getattr(orig, 'body', None)
            body_he = self._from_orig_expr(body_node, ctx_for(body_node))
            if body_he is not None and body_he.op == "Block" and body_he.operands is not None:
                body_ops = body_he.operands
            elif body_he is not None:
                body_ops = (body_he,)
            else:
                body_ops = tuple()
            nid = frozen_ctx.node_id
            span = frozen_ctx.span
            it_name, i_name, n_name = f"__for_it{nid}", f"__for_i{nid}", f"__for_n{nid}"

            def mk(**kw: Any) -> HExpr:
                return self._mk_hexpr(nid, "Expr", "Unknown", span, **kw)

            let_it = self._mk_hexpr(nid, "Stmt", "Unit", span, op="Let",
                                    bindings=((it_name, iter_he),))
            let_i = self._mk_hexpr(nid, "Stmt", "Unit", span, op="Let",
                                   bindings=((i_name, mk(op="Literal", literal=0)),))
            let_n = self._mk_hexpr(
                nid, "Stmt", "Unit", span, op="Let",
                bindings=((n_name, mk(op="Call",
                                      callee=BUILTIN_CALL_PREFIX + "len",
                                      operands=(mk(op="Var", var_name=it_name),))),))
            cond = mk(op="BinOp", binop="<",
                      left=mk(op="Var", var_name=i_name),
                      right=mk(op="Var", var_name=n_name))
            bind_elem = self._mk_hexpr(
                nid, "Stmt", "Unit", span, op="Let",
                bindings=((loop_var, mk(op="Call", callee="__index_get",
                                        operands=(mk(op="Var", var_name=it_name),
                                                  mk(op="Var", var_name=i_name)))),))
            incr = self._mk_hexpr(
                nid, "Stmt", "Unit", span, op="Assign", var_name=i_name,
                assign_value=mk(op="BinOp", binop="+",
                                left=mk(op="Var", var_name=i_name),
                                right=mk(op="Literal", literal=1)))
            while_he = self._mk_hexpr(nid, "Stmt", "Unit", span, op="While",
                                      cond=cond,
                                      loop_body=(bind_elem, *body_ops, incr))
            return self._mk_hexpr(nid, "Stmt", "Unit", span, op="Block",
                                  operands=(let_it, let_i, let_n, while_he))

        # TypeCast: `e as T`. Casts are static reinterpretations; at runtime
        # only numeric targets perform an actual representation conversion
        # (int <-> float), every other target passes the value through
        # unchanged (e.g. `f as fn(T,T) -> T`). Lowered to the __cast
        # builtin, which owns that rule. Previously this node was unhandled:
        # whole statements containing a cast were silently dropped
        # (`self.sum() / N as float` degraded to returning unit).
        if isinstance(orig, TypeCast):
            inner_node = getattr(orig, 'expr', None)
            inner_he = self._from_orig_expr(inner_node, ctx_for(inner_node))
            if inner_he is None:
                raise NotImplementedError(
                    "cast: could not lower the operand expression — "
                    "refusing to drop it")
            target = type_base_name(getattr(orig, 'target_type', None))
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unknown"))
            target_he = self._mk_hexpr(frozen_ctx.node_id, "Expr", "Unknown",
                                       frozen_ctx.span, op="Literal", literal=str(target))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                  op="Call", callee="__cast",
                                  operands=(inner_he, target_he))

        # WhileStatement: while cond { body }
        if isinstance(orig, fast.WhileStatement):
            cond_node = getattr(orig, 'condition', None)
            c = self._from_orig_expr(cond_node, ctx_for(cond_node))
            body_node = getattr(orig, 'body', None)
            body_he = self._from_orig_expr(body_node, ctx_for(body_node))
            if body_he is not None and body_he.op == "Block" and body_he.operands is not None:
                body_ops = body_he.operands
            elif body_he is not None:
                body_ops = (body_he,)
            else:
                body_ops = tuple()
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unit"))
            return self._mk_hexpr(frozen_ctx.node_id, "Stmt", ty, frozen_ctx.span,
                                  op="While", cond=c, loop_body=body_ops)

        # Assignment: x = expr (rebinds an existing local/param),
        # x.f = expr (field write-back), or v[i] = expr (Vec element store).
        if isinstance(orig, fast.Assignment):
            target = getattr(orig, 'name', None)
            value_node = getattr(orig, 'expression', None)
            val_he = self._from_orig_expr(value_node, ctx_for(value_node))
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unit"))

            # `v[i] = x` — a real element store. Two lowerings (both strict;
            # before this branch existed the target stringified to a garbage
            # slot name and the store was a silent no-op):
            # - assignable place base (`v[i] = x`, `buf.data[i] = x`):
            #   `place = __index_store(place, i, x)` — Vec mutates in place
            #   (and the rebind is the same object); fixed vector[T,N] gets a
            #   value-semantics functional update written back to the place,
            #   exactly like struct field assignment.
            # - anything else (`m[i][j] = x`, call results): __index_set on
            #   the indexed object — in-place only, so immutable receivers
            #   error loudly instead of mutating a temporary.
            if isinstance(target, fast.IndexExpression):
                base_node = getattr(target, 'base', None)
                base_he = self._from_orig_expr(base_node, ctx_for(base_node))
                idx = getattr(target, 'index', None)
                idx_list = idx if isinstance(idx, list) else [idx]
                if (not idx_list or any(i is None for i in idx_list)
                        or any(isinstance(i, fast.SliceExpression) for i in idx_list)):
                    raise NotImplementedError(
                        "cannot assign into a slice (`v[a:b] = x`); assign one "
                        "element at a time")
                if base_he is None or val_he is None:
                    raise NotImplementedError(
                        "could not lower index assignment target/value — "
                        "refusing to drop the store")
                idx_hes = []
                for i in idx_list:
                    ih = self._from_orig_expr(i, ctx_for(i))
                    if ih is None:
                        raise NotImplementedError(
                            "could not lower index expression in assignment — "
                            "refusing to drop the store")
                    idx_hes.append(ih)

                # Assignable place with a single index -> store-back form.
                place: str | None = None
                if len(idx_hes) == 1:
                    if isinstance(base_node, fast.Variable):
                        place = str(getattr(base_node, 'name', ''))
                    elif isinstance(base_node, fast.FieldAccess):
                        fbase = getattr(base_node, 'base', None)
                        if isinstance(fbase, fast.Variable):
                            fbase = getattr(fbase, 'name', None)
                        if isinstance(fbase, str):
                            place = ".".join(
                                [fbase, *[str(ff) for ff in
                                          getattr(base_node, 'fields', []) or []]])
                if place:
                    store = self._mk_hexpr(frozen_ctx.node_id, "Expr", 'Unknown',
                                           frozen_ctx.span,
                                           op="Call", callee="__index_store",
                                           operands=(base_he, idx_hes[0], val_he))
                    return self._mk_hexpr(frozen_ctx.node_id, "Stmt", ty,
                                          frozen_ctx.span, op="Assign",
                                          var_name=place, assign_value=store)

                # m[i][j] = x parses as Index(Index(m,i), j): all but the last
                # index are reads, the last one is the in-place store.
                current = base_he
                for ih in idx_hes[:-1]:
                    current = self._mk_hexpr(frozen_ctx.node_id, 'Expr', 'Unknown',
                                             frozen_ctx.span, op='Call',
                                             callee='__index_get',
                                             operands=(current, ih))
                return self._mk_hexpr(frozen_ctx.node_id, "Stmt", ty, frozen_ctx.span,
                                      op="Call", callee="__index_set",
                                      operands=(current, idx_hes[-1], val_he))

            # `x.f = v` / `x.f.g = v`: carried as a dotted string; MIR reads
            # the intermediate structs, sets the innermost field, and writes
            # back. Only a plain named base supports write-back — anything
            # else (call results, indexed elements) would mutate a temporary,
            # so it must error, not silently stringify to garbage.
            if isinstance(target, fast.FieldAccess):
                base = getattr(target, 'base', None)
                if isinstance(base, fast.Variable):
                    base = getattr(base, 'name', None)
                if not isinstance(base, str):
                    raise NotImplementedError(
                        "unsupported field-assignment target: the base of "
                        f"`{'.'.join([str(f) for f in getattr(target, 'fields', [])])}` "
                        f"is a {type(getattr(target, 'base', None)).__name__}, "
                        "not a named variable — refusing to drop the store")
                dotted = ".".join([base, *[str(f) for f in getattr(target, 'fields', []) or []]])
                return self._mk_hexpr(frozen_ctx.node_id, "Stmt", ty, frozen_ctx.span,
                                      op="Assign", var_name=dotted,
                                      assign_value=val_he)

            if target is not None and not isinstance(target, str):
                raise NotImplementedError(
                    f"unsupported assignment target {type(target).__name__} — "
                    "refusing to drop the store")
            return self._mk_hexpr(frozen_ctx.node_id, "Stmt", ty, frozen_ctx.span,
                                  op="Assign", var_name=str(target) if target is not None else None,
                                  assign_value=val_he)

        # ReturnStatement: an explicit Return op so early returns (inside
        # loops, match arms, if branches) actually exit the function. The
        # old lowering reduced `return e` to just `e`, which is only correct
        # in tail position — everywhere else the value was silently discarded
        # and execution continued.
        if isinstance(orig, fast.ReturnStatement):
            expr = getattr(orig, 'expression', None)
            val_he = self._from_orig_expr(expr, frozen_ctx) if expr is not None else None
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unit"))
            return self._mk_hexpr(frozen_ctx.node_id, "Stmt", ty, frozen_ctx.span,
                                  op="Return",
                                  operands=(val_he,) if val_he is not None else ())

        # LetStatement
        if isinstance(orig, fast.LetStatement):
            binds: list[tuple[str, HExpr]] = []
            bind_modes: dict[str, ModeInfo] = {}
            for b in getattr(orig, 'bindings', []) or []:
                name = getattr(b, 'identifier', None)
                init = getattr(b, 'initializer', None)
                he = self._from_orig_expr(init, frozen_ctx)
                if name and he is not None:
                    binds.append((name, he))
                    mi = self._extract_modeinfo(getattr(b, 'mode', None))
                    bind_modes[str(name)] = mi
                    # `let @global x = S { ... }` allocates on the heap: the
                    # binding's locality annotation is the allocation site's
                    # locality, so it must reach MIR's alloc_struct (dropping
                    # it here is exactly the "silently degraded mode" seam the
                    # conventions warn about).
                    if mi.locality == "global" and he.op == "Struct":
                        he.locality = "global"
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unit"))
            return self._mk_hexpr(frozen_ctx.node_id, "Stmt", ty, frozen_ctx.span, op="Let", bindings=tuple(binds), bind_modes=bind_modes)

        # UnsafeBlock: `unsafe { stmts }` — at HIR level this is an ordinary
        # block (unsafe-ness is a static permission, not runtime behavior);
        # its value is the last statement's value, so constructor-return
        # through `unsafe { let p = malloc(n); Buffer { ... } }` survives.
        # This node used to fall through to the None fallback: every function
        # whose body was an unsafe block silently degraded to `ret Unit`
        # (which is how Buffer.new returned Unit and copy_from ended up
        # dispatching on a Unit receiver). Un-lowerable statements inside the
        # block fail loudly instead of being dropped.
        if isinstance(orig, UnsafeBlock):
            stmts = getattr(orig, 'body', []) or []
            ops = []
            for s in stmts:
                he = self._from_orig_expr(s, ctx_for(s))
                if he is None:
                    raise NotImplementedError(
                        "unsafe block: could not lower statement "
                        f"{type(s).__name__} — refusing to drop it")
                ops.append(he)
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unit"))
            return self._mk_hexpr(frozen_ctx.node_id, "Block", ty, frozen_ctx.span,
                                  op="Block", operands=tuple(ops))

        # Block
        if isinstance(orig, fast.Block):
            stmts = getattr(orig, 'statements', []) or []
            ops: list[HExpr] = []
            for s in stmts:
                he = self._from_orig_expr(s, frozen_ctx)
                if he is not None:
                    ops.append(he)
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unit"))
            return self._mk_hexpr(frozen_ctx.node_id, "Block", ty, frozen_ctx.span, op="Block", operands=tuple(ops))

        # StructInstantiation
        if isinstance(orig, fast.StructInstantiation):
            sname_node = getattr(orig, 'struct_name', None)
            sname = str(sname_node) if sname_node is not None else "Unknown"
            field_assigns = getattr(orig, 'field_assignments', []) or []
            field_exprs: list[tuple[str, HExpr]] = []
            for sf in field_assigns:
                fname = getattr(sf, 'name', None)
                fval_node = getattr(sf, 'value', None)
                fval = self._from_orig_expr(fval_node, frozen_ctx)
                if not fname:
                    raise NotImplementedError(
                        f"struct instantiation of {sname}: field assignment "
                        f"without a field name — refusing to drop it")
                if fval is None:
                    # Dropping the field here made the struct silently LOSE it:
                    # MIR's alloc_struct only lists the fields that survive, so
                    # both engines then built a short struct and native codegen
                    # blamed the *declaration* ("struct 'List' has no field
                    # 'data'") for an un-lowerable initializer.
                    raise NotImplementedError(
                        f"struct instantiation of {sname}: could not lower the "
                        f"value of field {str(fname)!r} "
                        f"({type(fval_node).__name__}) — refusing to drop it")
                field_exprs.append((str(fname), fval))
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unknown"))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                  op="Struct", struct_name=sname,
                                  fields=tuple(field_exprs),
                                  locality="local")  # default local; borrow checker promotes to global

        # QualifiedFunctionCall: effect_name.op(args) or module.fn(args)
        if isinstance(orig, fast.QualifiedFunctionCall):
            parts = list(getattr(orig, 'parts', []) or [])
            arguments = list(getattr(orig, 'arguments', []) or [])
            arg_exprs = []
            for a in arguments:
                he = self._from_orig_expr(a, ctx_for(a))
                if he is not None:
                    arg_exprs.append(he)
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
            # Enum variant constructor: Option.Some(x) / Option::Some(x)
            if len(parts) == 2 and str(parts[1]) in self._variant_to_enum:
                return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                      op='MakeVariant', enum_name=str(parts[0]),
                                      variant_name=str(parts[1]), operands=tuple(arg_exprs))
            # Trait method call on a named receiver: `d.speak()` (dispatch on
            # the receiver's runtime type; checked BEFORE builtin methods so a
            # user impl of e.g. to_string wins for its receiver type, with the
            # builtin as runtime fallback for everything else).
            # Builtin method call on a value: `x.to_string()` — lower to a
            # call with the receiver (Var or chained FieldGet) as first arg.
            last = str(parts[-1])
            # Static impl-method call on the type itself: `Buffer.new(1024)`.
            if (len(parts) == 2 and str(parts[0]) in self._impl_type_names
                    and last in self._trait_method_names):
                return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                      op='Call',
                                      callee=f"{STATIC_CALL_PREFIX}{parts[0]}{IMPL_SEP}{last}",
                                      operands=tuple(arg_exprs))
            # A KNOWN type name without impls as the base (`Vec.new()`) is a
            # static call on the type, not a runtime receiver: fall through
            # to the plain dotted-callee path so runtime builtins keep
            # working even when some unrelated impl defines a same-named
            # method. Names not known as types (however capitalized) stay
            # ordinary receivers.
            base_is_foreign_type = (str(parts[0]) in self._type_names
                                    and str(parts[0]) not in self._impl_type_names)
            if (len(parts) >= 2 and not base_is_foreign_type
                    and (last in self._trait_method_names
                         or last in _BUILTIN_METHODS)):
                recv: HExpr = self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty,
                                             frozen_ctx.span, op='Var',
                                             var_name=str(parts[0]))
                for fname in parts[1:-1]:
                    recv = self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty,
                                          frozen_ctx.span, op='FieldGet',
                                          base=recv, field_name=str(fname))
                callee = _method_callee(last, self._trait_method_names)
                return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                      op='Call', callee=callee,
                                      operands=(recv, *arg_exprs))
            # Treat as a plain call with dotted callee name. Explicit
            # instantiation type args (`mod.f<Int>(x)`) are preserved so the
            # monomorphization pass can specialize qualified generic calls.
            callee = '.'.join(str(p) for p in parts)
            raw_targs = getattr(orig, 'type_args', None) or ()
            targs = tuple(t for t in (mast._type_display(a) for a in raw_targs)
                          if isinstance(t, str))
            return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                  op='Call', callee=callee, operands=tuple(arg_exprs),
                                  type_args=targs or None)

        # MethodCall on a computed receiver: `expr.method(args)`
        if isinstance(orig, fast.MethodCall):
            recv_node = getattr(orig, 'receiver', None)
            method = str(getattr(orig, 'method', '') or '')
            # Static method on the vector TYPE itself:
            # `vector[float,4].filled(1.0)` — no runtime receiver.
            if isinstance(recv_node, fast.VectorTypeExpression) and method == "filled":
                n = self._const_int_of(getattr(recv_node, 'size', None))
                ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
                n_he = self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                      op='Literal', literal=n)
                arg_exprs = []
                for a in getattr(orig, 'arguments', []) or []:
                    he = self._from_orig_expr(a, ctx_for(a))
                    if he is not None:
                        arg_exprs.append(he)
                return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                      op='Call', callee='__vec_filled',
                                      operands=(n_he, *arg_exprs))
            recv_he = self._from_orig_expr(recv_node, ctx_for(recv_node)) if recv_node is not None else None
            arg_exprs = []
            for a in getattr(orig, 'arguments', []) or []:
                he = self._from_orig_expr(a, ctx_for(a))
                if he is not None:
                    arg_exprs.append(he)
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
            callee = _method_callee(method, self._trait_method_names)
            if recv_he is not None:
                return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                      op='Call', callee=callee,
                                      operands=(recv_he, *arg_exprs))
            return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                  op='Call', callee=callee, operands=tuple(arg_exprs))

        # PerformEffect: perform effect_name(args)
        if isinstance(orig, fast.PerformEffect):
            eff_name = str(getattr(orig, 'effect_name', '') or '')
            arguments = list(getattr(orig, 'arguments', []) or [])
            arg_exprs = []
            for a in arguments:
                he = self._from_orig_expr(a, ctx_for(a))
                if he is not None:
                    arg_exprs.append(he)
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
            return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                  op='Perform', effect_op=eff_name, perform_args=tuple(arg_exprs))

        # HandleEffect: handle EffectType with { cases } in body
        if isinstance(orig, fast.HandleEffect):
            eff_node = getattr(orig, 'effect_name', None)
            # effect_name may be a TypeReference, QualifiedName, string, or AST node
            if hasattr(eff_node, 'name'):
                eff_name = str(eff_node.name)
            elif hasattr(eff_node, 'parts'):
                eff_name = '.'.join(str(p) for p in eff_node.parts)
            else:
                eff_name = str(eff_node or '')
            cases_raw = list(getattr(orig, 'handler', []) or [])
            cont = getattr(orig, 'continuation', None)
            case_triples: list[tuple[str, tuple, HExpr]] = []
            for c in cases_raw:
                if isinstance(c, fast.HandleCase):
                    op_name = str(c.op_name)
                    raw_params = getattr(c, 'param_names', None)
                    if raw_params is None:
                        raw_params = [c.param_name] if c.param_name is not None else []
                    params = tuple(str(p) for p in raw_params) or ("_",)
                    case_body = self._from_orig_expr(c.body, ctx_for(c.body) if c.body is not None else frozen_ctx)
                    if case_body is None:
                        # A dropped arm silently changes runtime behavior (the
                        # op becomes unhandled): refuse instead of skipping.
                        raise NotImplementedError(
                            f"handle {eff_name}: could not lower the body of "
                            f"handler arm {op_name!r} "
                            f"({type(c.body).__name__}) — refusing to drop the arm")
                    case_triples.append((op_name, params, case_body))
            # An absent/undroppable `in` target would make the whole handled
            # computation disappear (MIR falls back to a typed constant), so
            # require it.
            body_he = self._require(
                self._from_orig_expr(cont, ctx_for(cont)) if cont is not None else None,
                orig, frozen_ctx, f"the `in` body of `handle {eff_name}`")
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
            return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                  op='Handle',
                                  handle_effect=eff_name,
                                  handle_cases=tuple(case_triples),
                                  handle_body=body_he)

        # HandleBlock: the inline handler form
        #     handle SUBJECT { perform Eff.op(p, q) => body, ... }
        # (the `handle e with { } in body` form is HandleEffect above). Same
        # delimited semantics: SUBJECT is the delimited body, each arm handles
        # one operation. This node had NO lowering — `with_simd(...)` in
        # examples/06_vector_operations.mx compiled to a function that ran
        # nothing and returned unit.
        if isinstance(orig, fast.HandleBlock):
            subject = getattr(orig, 'subject', None)
            body_he = self._require(
                self._from_orig_expr(subject, ctx_for(subject))
                if subject is not None else None,
                orig, frozen_ctx, "the subject of a `handle` block")
            eff_names: set[str] = set()
            case_triples: list[tuple[str, tuple, HExpr]] = []
            for arm in (getattr(orig, 'arms', []) or []):
                pat_node, arm_body = arm
                eff, op_name, params = self._handle_arm_signature(pat_node, frozen_ctx)
                if eff:
                    eff_names.add(eff)
                arm_he = self._require(
                    self._from_orig_expr(arm_body, ctx_for(arm_body))
                    if arm_body is not None else None,
                    orig, frozen_ctx,
                    f"the body of handler arm {op_name!r}")
                case_triples.append((op_name, params, arm_he))
            if len(eff_names) > 1:
                raise UnsupportedConstruct(
                    f"handle block at {self._span_text(frozen_ctx.span)} handles "
                    f"operations of more than one effect ({', '.join(sorted(eff_names))}); "
                    "a handler frame names a single effect — use one `handle` per effect",
                    location=self._loc(frozen_ctx.span, orig))
            # No qualified prefix on any arm: leave the effect unnamed, which
            # the runtime treats as "match by operation name alone".
            eff_name = next(iter(eff_names)) if eff_names else ""
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
            return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                  op='Handle', handle_effect=eff_name,
                                  handle_cases=tuple(case_triples),
                                  handle_body=body_he)

        # TryCatch: try { body } catch e { handler } — delimited dynamic
        # error recovery (docs/try_catch.md). Reuses the Handle field slots:
        # handle_body is the try body, handle_cases holds the single catch
        # arm as ("catch", (param,), body).
        if isinstance(orig, fast.TryCatch):
            body_node = getattr(orig, 'body', None)
            catch_node = getattr(orig, 'catch_body', None)
            body_he = self._from_orig_expr(body_node, ctx_for(body_node) if body_node is not None else frozen_ctx)
            catch_he = self._from_orig_expr(catch_node, ctx_for(catch_node) if catch_node is not None else frozen_ctx)
            param = str(getattr(orig, 'catch_name', None) or '_')
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
            if body_he is None:
                raise ValueError("try expression with no body")
            if catch_he is None:
                raise ValueError("catch block with no body")
            return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                  op='Try',
                                  handle_cases=(("catch", (param,), catch_he),),
                                  handle_body=body_he)

        # FieldAccess
        if isinstance(orig, fast.FieldAccess):
            base_node = getattr(orig, 'base', None) or getattr(orig, 'expression', None)
            # `Color.Red` parses as a FieldAccess, but it names a nullary enum
            # variant — not a field of a variable called `Color` (which is what
            # the FieldGet chain below built: "Unbound variable 'Color'").
            enum_ctor = self._enum_variant_of(base_node,
                                              getattr(orig, 'fields', ()) or ())
            if enum_ctor is not None:
                ety = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unknown"))
                return self._mk_hexpr(frozen_ctx.node_id, "Expr", ety, frozen_ctx.span,
                                      op="MakeVariant", enum_name=enum_ctor[0],
                                      variant_name=enum_ctor[1], operands=())
            if isinstance(base_node, str):
                # The parser stores the base of `c.name` as a raw name string.
                base_ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unknown"))
                base_he = self._mk_hexpr(frozen_ctx.node_id, "Expr", base_ty,
                                         frozen_ctx.span, op="Var", var_name=base_node)
            else:
                base_he = self._require(
                    self._from_orig_expr(base_node, frozen_ctx), orig,
                    frozen_ctx, "the base of a field access")
            field_names = getattr(orig, 'fields', ()) or ()
            # Chain: for a.b.c, build nested FieldGet(FieldGet(a, b), c)
            current = base_he
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unknown"))
            for fname in field_names:
                current = self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                         op="FieldGet", base=current, field_name=str(fname))
            return current

        # IndexExpression: `base[i]` (plain index) or `base[a:b:c]` (slice).
        # Lowered to strict runtime-library builtin calls: __index_get /
        # __slice_get (absent slice parts become literal None).
        if isinstance(orig, fast.IndexExpression):
            base_node = getattr(orig, 'base', None)
            base_he = self._require(
                self._from_orig_expr(base_node, ctx_for(base_node)), orig,
                frozen_ctx, "the base of an index expression")
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
            idx = getattr(orig, 'index', None)
            idx_list = idx if isinstance(idx, list) else [idx]
            current = base_he
            for i in idx_list:
                if i is None:
                    continue
                if isinstance(i, fast.SliceExpression):
                    parts = []
                    for part in (getattr(i, 'start', None), getattr(i, 'stop', None),
                                 getattr(i, 'step', None)):
                        if part is None:
                            parts.append(self._mk_hexpr(
                                frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                op='Literal', literal=None))
                        else:
                            parts.append(self._require(
                                self._from_orig_expr(part, ctx_for(part)), part,
                                frozen_ctx, "a slice bound"))
                    current = self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty,
                                             frozen_ctx.span, op='Call',
                                             callee='__slice_get',
                                             operands=(current, *parts))
                else:
                    ih = self._require(
                        self._from_orig_expr(i, ctx_for(i)), i, frozen_ctx,
                        "an index expression")
                    current = self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty,
                                             frozen_ctx.span, op='Call',
                                             callee='__index_get',
                                             operands=(current, ih))
            return current

        # UnaryOperation: `-x` / `!x` — lowered to the neg/not builtins.
        if isinstance(orig, fast.UnaryOperation):
            operand = getattr(orig, 'operand', None)
            operand_he = self._require(
                self._from_orig_expr(operand, ctx_for(operand)), orig,
                frozen_ctx, "the operand of a unary operator")
            op_sym = str(getattr(orig, 'operator', '') or '')
            callee = {'-': 'neg', '!': 'not', 'not': 'not',
                      '~': 'bnot'}.get(op_sym)
            if callee is None:
                # Dropping the whole expression here turned `~x` into nothing.
                raise UnsupportedConstruct(
                    f"unary operator {op_sym!r} at "
                    f"{self._span_text(frozen_ctx.span)} is not supported "
                    "(only `-`, `!`/`not` and `~` lower)",
                    location=self._loc(frozen_ctx.span, orig))
            # Unary operators are compiler-synthesized builtin calls:
            # marked so a module function named `neg`/`not` cannot capture
            # `-x` / `!x`.
            callee = BUILTIN_CALL_PREFIX + callee
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
            return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                  op='Call', callee=callee, operands=(operand_he,))

        # RangeExpression: `start..end` — lowered to the __range builtin.
        if isinstance(orig, fast.RangeExpression):
            s_node = getattr(orig, 'start', None)
            e_node = getattr(orig, 'end', None)
            s_he = self._require(self._from_orig_expr(s_node, ctx_for(s_node)),
                                 orig, frozen_ctx, "the start of a range")
            e_he = self._require(self._from_orig_expr(e_node, ctx_for(e_node)),
                                 orig, frozen_ctx, "the end of a range")
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
            return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                  op='Call', callee='__range',
                                  operands=(s_he, e_he))

        # VectorLiteral: `vector[T, N](elems...)`, `vector[T, N]()` (zeros),
        # or `vector[T, N](expr for x in iterable)` (comprehension).
        if isinstance(orig, fast.VectorLiteral):
            return self._convert_vector_literal(orig, frozen_ctx, ctx_for)

        # ListLiteral: `[]`, `[a, b, c]` — a growable Vec, the same runtime
        # object `Vec.new()` returns (fixed-size `vector[T, N]` is the literal
        # above).  This node had NO lowering: every list literal silently
        # vanished, which is how `Queue { items: [], capacity: 10 }` in
        # examples/04_advanced_types.mx built a Queue with no `items` field.
        if isinstance(orig, fast.ListLiteral):
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))

            def mk_call(callee: str, ops: tuple) -> HExpr:
                return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty,
                                      frozen_ctx.span, op='Call', callee=callee,
                                      operands=ops)

            # `[a, ...rest, b]` splits into segments: runs of ordinary
            # elements become __list_lit calls, each spread contributes its
            # own list, and the segments are joined by __list_concat (strict:
            # a spread of a non-list is a runtime error, never a silent skip).
            segments: list[HExpr] = []
            run: list[HExpr] = []
            saw_spread = False
            for el in getattr(orig, 'elements', []) or []:
                if isinstance(el, fast.SpreadElement):
                    saw_spread = True
                    inner = getattr(el, 'expression', None)
                    he = self._from_orig_expr(inner, ctx_for(inner)) if inner is not None else None
                    if he is None:
                        raise NotImplementedError(
                            "list literal: could not lower the spread element "
                            f"...{type(inner).__name__} — refusing to drop it")
                    if run:
                        segments.append(mk_call('__list_lit', tuple(run)))
                        run = []
                    segments.append(he)
                    continue
                he = self._from_orig_expr(el, ctx_for(el))
                if he is None:
                    raise NotImplementedError(
                        f"list literal: could not lower element "
                        f"{type(el).__name__} — refusing to drop it")
                run.append(he)
            if run or not segments:
                segments.append(mk_call('__list_lit', tuple(run)))
            if not saw_spread:
                return segments[0]
            # A literal containing a spread always goes through __list_concat,
            # even when it is the only segment: `[...xs]` is a fresh list, and
            # Vec has identity semantics, so returning `xs` itself would alias.
            return mk_call('__list_concat', tuple(segments))

        # A bare `vector[T, N]` in VALUE position is a TYPE, not a value: the
        # value forms all spell the call (`vector[T,N]()` zeros,
        # `vector[T,N](a, b)` elements, `vector[T,N].filled(x)`, which are
        # VectorLiteral / MethodCall nodes handled above).  This used to fall
        # into the None fallback and vanish wherever it appeared.
        if isinstance(orig, fast.VectorTypeExpression):
            base = mast._type_display(getattr(orig, 'base_type', None)) or '?'
            size = mast._type_display(getattr(orig, 'size', None)) or '?'
            raise NotImplementedError(
                f"vector[{base}, {size}] is a TYPE, not a value: write "
                f"vector[{base}, {size}]() for a zero-initialized vector, "
                f"vector[{base}, {size}](e1, ...) for one with elements, or "
                f"vector[{base}, {size}].filled(e)")

        # LambdaExpression
        if isinstance(orig, fast.LambdaExpression):
            params = getattr(orig, 'params', []) or []
            param_names = tuple(str(getattr(p, 'name', p)) for p in params)
            body_nodes = getattr(orig, 'body', None)
            body_he = self._require(
                self._from_orig_expr(body_nodes, frozen_ctx), orig, frozen_ctx,
                "the body of a lambda")
            captured_vars = {str(v) for v in (getattr(orig, 'captured_vars', set()) or set())}
            capture_modes = getattr(orig, 'capture_modes', {}) or {}
            # The parser's scope-based capture analysis only sees scopes it
            # links and populates (function/lambda scopes); variables bound
            # in enclosing loop or block scopes are invisible to it, which
            # used to yield silently-empty capture lists ("Unbound variable"
            # at run time). Supplement it with a free-variable analysis of
            # the body: free names that are not parameters are captured with
            # mode 'auto' — the MIR lowering captures only the ones actually
            # bound in the enclosing scope (the rest are globals/builtins
            # that resolve by name at call time).
            free = self._free_names(body_nodes) - set(param_names)
            captures = tuple(
                (name, str(capture_modes.get(name, 'auto')))
                for name in sorted(captured_vars | free)
            )
            ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, "Unknown"))
            # Lambda params declared @mut get the same write-back semantics
            # as @mut params of named functions (threaded to MirFunc.mut_params).
            mut_names = tuple(
                str(getattr(p, 'name', p)) for p in params
                if self._param_modeinfo(p).uniqueness in ("mutable", "exclusive"))
            return self._mk_hexpr(frozen_ctx.node_id, "Expr", ty, frozen_ctx.span,
                                  op="Lambda",
                                  lambda_params=param_names,
                                  lambda_mut_params=mut_names or None,
                                  lambda_body=body_he,
                                  captures=captures)

        # ComptimeFunction subclasses FunctionDeclaration, so it must be
        # rejected BEFORE the declaration branch below (and before build()'s
        # hoisting walk treats it as an ordinary function): compiling a
        # `comptime fn` as a run-time function is exactly the "quietly does
        # something else" failure this triage exists to stop.
        if isinstance(orig, fast.ComptimeFunction):
            return self._unlowerable(orig, frozen_ctx)

        # Declarations in statement position. The parser allows `fn`, `struct`,
        # `enum`, `trait`, `implement`, `import`, `module`, `effect`, `extern`
        # and friends inside any block; their meaning is realized by the module
        # loader, the desugar passes, or build()'s own hoisting walk (which
        # lifts every FunctionDeclaration anywhere in the tree into an HFun).
        # As a *statement* each one therefore contributes nothing at run time —
        # an EXPLICIT unit, not a silently skipped None.
        if isinstance(orig, _DECLARATION_NODES):
            return self._mk_hexpr(frozen_ctx.node_id, "Block", "Unit",
                                  frozen_ctx.span, op="Block", operands=())

        # Loud fallback. There is no silent "unknown node" path any more:
        # every AST node class is triaged in AST_NODE_TRIAGE and an unhandled
        # one raises, naming the class and the source span.
        return self._unlowerable(orig, frozen_ctx)

    # ------------------------------------------------------------------
    # The loud fallback
    # ------------------------------------------------------------------

    def _loc(self, span: Any = None, orig: Any = None) -> SourceLocation | None:
        """SourceLocation for a diagnostic about `orig` / frozen `span`.

        Prefers the ORIGINAL parsed node's location (the parser attaches one
        to every node it builds) and falls back to the frozen Span, which
        carries the same information for nodes that survived freezing.
        Nodes synthesized after parsing have neither; those report the
        compilation unit's file with no line.
        """
        loc = getattr(orig, 'location', None)
        if isinstance(loc, SourceLocation) and loc.line:
            if not loc.file or loc.file == "<unknown>":
                return SourceLocation(file=self._root_file, line=loc.line,
                                      column=loc.column, end_line=loc.end_line,
                                      end_column=loc.end_column,
                                      offset=loc.offset, end_offset=loc.end_offset)
            return loc
        if span is not None and hasattr(span, 'location'):
            resolved = span.location(self._root_file)
            if resolved is not None:
                return resolved
        return None

    def _span_text(self, span: Any, orig: Any = None) -> str:
        """Human-readable `file:line:column` for a diagnostic.

        Degrades to the compilation unit's file name for nodes that have no
        location (everything synthesized after parsing).
        """
        loc = self._loc(span, orig)
        if loc is not None:
            return format_location(loc, fallback=self._root_file)
        fpath = getattr(span, 'file', None)
        if not fpath or fpath == "<unknown>":
            fpath = self._root_file
        return str(fpath)

    def _pat_where(self, p: Any) -> str:
        """`file:line:column` of a pattern node for a diagnostic.

        Patterns are converted without a frozen context (they are plain
        expression nodes hanging off a match arm), so the location comes
        straight off the parsed node; the compilation-unit file is the
        fallback for synthesized patterns.
        """
        loc = self._loc(None, p)
        return format_location(loc, fallback=self._root_file) if loc is not None \
            else self._root_file

    def _unlowerable(self, orig: Any, frozen_ctx: mast.AstNode) -> HExpr:
        """Raise for an AST node expression lowering does not handle.

        Consults AST_NODE_TRIAGE so the diagnostic says WHICH kind of problem
        this is: a construct with no semantics yet (user error), a node that
        should never have reached here (compiler bug), or a node class nobody
        has triaged (also a compiler bug — the table is the checklist).
        """
        cls = type(orig).__name__
        span = getattr(frozen_ctx, 'span', None)
        where = self._span_text(span, orig)
        loc = self._loc(span, orig)
        entry = AST_NODE_TRIAGE.get(cls)
        if entry is None:
            raise HIRCompilerBug(
                f"{cls} at {where} is not in AST_NODE_TRIAGE: HIR lowering "
                "does not know whether it is an expression, and refuses to "
                "guess. Add it to the table in hir.py (see test_hir_coverage).",
                location=loc)
        bucket, reason = entry
        if bucket == UNSUPPORTED:
            raise UnsupportedConstruct(
                f"{cls} at {where} is not supported: {reason}", location=loc)
        if bucket == NOT_AN_EXPRESSION:
            raise HIRCompilerBug(
                f"{cls} at {where} reached HIR expression lowering, but it is "
                f"not an expression ({reason}) — an earlier pass should have "
                "consumed it", location=loc)
        raise HIRCompilerBug(
            f"{cls} at {where} is registered as lowered ({reason}) but fell "
            "through to the fallback in _from_orig_expr", location=loc)

    def _enum_variant_of(self, base: Any, fields: Any) -> tuple[str, str] | None:
        """(enum, variant) if `base.fields` spells a nullary variant, else None.

        `Color.Red` reaches HIR as FieldAccess(base=Variable('Color'),
        fields=['Red']) in expression position and in pattern position alike.
        The match is deliberately tight — the base must be exactly the enum
        that declares the variant — so a real field read of a same-named
        variable is untouched.
        """
        names = [str(f) for f in (fields or [])]
        if len(names) != 1:
            return None
        if isinstance(base, fast.Variable):
            base = getattr(base, 'name', None)
        elif isinstance(base, fast.QualifiedName):
            parts = list(getattr(base, 'parts', []) or [])
            base = str(parts[0]) if len(parts) == 1 else None
        if not isinstance(base, str):
            return None
        if self._variant_to_enum.get(names[0]) != base:
            return None
        return base, names[0]

    def _handle_arm_signature(self, pat_node: Any,
                              frozen_ctx: mast.AstNode) -> tuple[str, str, tuple[str, ...]]:
        """(effect, op, param names) of an inline `handle { ... }` arm pattern.

        Arms are written `perform Eff.op(p, q) => body` (a PerformEffect) or
        `op(p) => body` / `op => body` (a bare call/name). Anything else is a
        loud error: guessing an operation name would silently install a
        handler for the wrong op — which is indistinguishable from installing
        no handler at all.
        """
        raw_name: str | None = None
        args: list = []
        if isinstance(pat_node, fast.PerformEffect):
            raw_name = str(getattr(pat_node, 'effect_name', '') or '')
            args = list(getattr(pat_node, 'arguments', []) or [])
        elif isinstance(pat_node, fast.QualifiedFunctionCall):
            raw_name = '.'.join(str(p) for p in (getattr(pat_node, 'parts', []) or []))
            args = list(getattr(pat_node, 'arguments', []) or [])
        elif isinstance(pat_node, fast.FunctionCall):
            raw_name = str(getattr(pat_node, 'name', '') or '')
            args = list(getattr(pat_node, 'arguments', []) or [])
        elif isinstance(pat_node, fast.QualifiedName):
            raw_name = '.'.join(str(p) for p in (getattr(pat_node, 'parts', []) or []))
        elif isinstance(pat_node, fast.Variable):
            raw_name = str(getattr(pat_node, 'name', '') or '')
        if not raw_name:
            raise UnsupportedConstruct(
                f"handle block at {self._span_text(frozen_ctx.span)}: arm "
                f"{type(pat_node).__name__} is not an operation pattern; write "
                "`perform Effect.op(params) => body`",
                location=self._loc(frozen_ctx.span, pat_node))
        effect, _, op_name = raw_name.rpartition(".")
        params: list[str] = []
        for a in args:
            if isinstance(a, fast.Variable):
                params.append(str(getattr(a, 'name', '_')))
            elif isinstance(a, str):
                params.append(a)
            else:
                raise UnsupportedConstruct(
                    f"handle block at {self._span_text(frozen_ctx.span)}: arm "
                    f"{op_name!r} binds a {type(a).__name__} where a parameter "
                    "name is required — handler arms bind plain names",
                    location=self._loc(frozen_ctx.span, a))
        # Zero-arg ops still need a slot: the MIR handler sub-function takes
        # (params..., __k), and the interpreter tolerates fewer args than
        # params. Mirrors the parser's HandleCase default.
        return effect, op_name, tuple(params) or ("_",)

    def _require(self, he: HExpr | None, orig: Any, frozen_ctx: mast.AstNode,
                 what: str) -> HExpr:
        """Return `he`, or raise naming `what` — used where a sub-expression is
        mandatory (dropping it would silently change the program)."""
        if he is not None:
            return he
        cls = type(orig).__name__
        span = getattr(frozen_ctx, 'span', None)
        where = self._span_text(span, orig)
        raise HIRCompilerBug(
            f"{what} at {where} is missing ({cls}) — refusing to drop it",
            location=self._loc(span, orig))

    # ------------------------------------------------------------------
    # Vector literal / comprehension helpers
    # ------------------------------------------------------------------

    def _const_int_of(self, node: Any) -> int | None:
        """Best-effort compile-time integer of a size expression node.

        `vector[float, 4]` carries its size as a TypeReference("4") (or a
        Literal). Returns None when the size is not a literal integer (e.g. a
        const generic `N`), in which case runtime checks that need it fail
        with a clear error instead of guessing.
        """
        if node is None:
            return None
        if isinstance(node, int) and not isinstance(node, bool):
            return node
        if isinstance(node, str):
            return int(node) if node.lstrip("-").isdigit() else None
        if isinstance(node, fast.Literal):
            v = getattr(node, 'value', None)
            return v if isinstance(v, int) and not isinstance(v, bool) else None
        if isinstance(node, fast.TypeReference):
            return self._const_int_of(getattr(node, 'name', None))
        return None

    def _convert_vector_literal(self, orig: Any, frozen_ctx: mast.AstNode,
                                ctx_for: Any) -> HExpr:
        n = self._const_int_of(getattr(orig, 'size', None))
        ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
        n_he = self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                              op='Literal', literal=n)
        elements = list(getattr(orig, 'elements', []) or [])
        # Comprehension form: vector[T, N](f(x) for x in iterable)
        if len(elements) == 1 and isinstance(elements[0], fast.Comprehension):
            comp = elements[0]
            lam = self._comprehension_lambda(comp, frozen_ctx, ctx_for)
            iter_node = getattr(comp, 'iterable', None)
            # Zip form: `f(a, b) for (a, b) in (xs, ys)` — a tuple of
            # sequences iterates them in lockstep. Lowered to the strict
            # __zip builtin (length mismatch errors at runtime). This used
            # to fall into the generic None fallback and silently drop the
            # whole vector literal.
            if isinstance(iter_node, fast.TupleLiteral):
                part_nodes = list(getattr(iter_node, 'elements', []) or [])
                parts = [self._from_orig_expr(el, ctx_for(el)) for el in part_nodes]
                if not parts or any(p is None for p in parts):
                    raise NotImplementedError(
                        "could not lower the tuple iterable of a zip "
                        "comprehension — refusing to drop it")
                iter_he = self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty,
                                         frozen_ctx.span, op='Call',
                                         callee='__zip', operands=tuple(parts))
            else:
                iter_he = self._require(
                    self._from_orig_expr(iter_node, ctx_for(iter_node)), comp,
                    frozen_ctx, "the iterable of a vector comprehension")
            return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                  op='Call', callee='__vec_comprehension',
                                  operands=(n_he, lam, iter_he))
        # Empty form: vector[T, N]() — zero-initialized
        if not elements:
            base_name = type_base_name(getattr(orig, 'base_type', None))
            base_he = self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                     op='Literal', literal=base_name)
            return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                                  op='Call', callee='__vec_zeros',
                                  operands=(n_he, base_he))
        # Explicit elements
        elem_hes: list[HExpr] = []
        for el in elements:
            elem_hes.append(self._require(
                self._from_orig_expr(el, ctx_for(el)), el, frozen_ctx,
                "an element of a vector literal"))
        return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                              op='Call', callee='__vec_lit',
                              operands=(n_he, *elem_hes))

    def _comprehension_lambda(self, comp: Any, frozen_ctx: mast.AstNode,
                              ctx_for: Any) -> HExpr:
        """Compile a comprehension body into a Lambda HExpr.

        The comprehension targets become the lambda parameters. Free names in
        the body are captured with mode 'auto': the MIR lowering only actually
        captures the ones bound in the enclosing scope (globals/builtins
        resolve by name at call time).
        """
        targets = tuple(str(t) for t in (getattr(comp, 'targets', []) or []))
        if not targets:
            raise HIRCompilerBug(
                f"comprehension at {self._span_text(frozen_ctx.span, comp)} has no "
                "loop target — refusing to drop it",
                location=self._loc(frozen_ctx.span, comp))
        body_node = getattr(comp, 'expression', None)
        body_he = self._require(
            self._from_orig_expr(body_node, ctx_for(body_node)), comp,
            frozen_ctx, "the body of a comprehension")
        free = self._free_names(body_node) - set(targets)
        captures = tuple((name, 'auto') for name in sorted(free))
        ty = self.t.apply_tyenv(self.t.types.get(frozen_ctx.node_id, 'Unknown'))
        return self._mk_hexpr(frozen_ctx.node_id, 'Expr', ty, frozen_ctx.span,
                              op='Lambda', lambda_params=targets,
                              lambda_body=body_he, captures=captures)

    def _free_names(self, node: Any, _seen: set[int] | None = None) -> set[str]:
        """Names referenced by an expression subtree (variables, call heads)."""
        if _seen is None:
            _seen = set()
        out: set[str] = set()
        if isinstance(node, (list, tuple)):
            for item in node:
                out |= self._free_names(item, _seen)
            return out
        if isinstance(node, dict):
            for v in node.values():
                out |= self._free_names(v, _seen)
            return out
        if not isinstance(node, fast.Node) or id(node) in _seen:
            return out
        _seen.add(id(node))
        if isinstance(node, fast.Variable):
            name = getattr(node, 'name', None)
            if isinstance(name, str):
                out.add(name)
        elif isinstance(node, fast.QualifiedName):
            parts = list(getattr(node, 'parts', []) or [])
            if parts:
                out.add(str(parts[0]))
        elif isinstance(node, fast.QualifiedFunctionCall):
            parts = list(getattr(node, 'parts', []) or [])
            if parts:
                out.add(str(parts[0]))
        elif isinstance(node, fast.FunctionCall):
            name = getattr(node, 'name', None)
            if isinstance(name, str):
                out.add(name)
        elif isinstance(node, fast.FieldAccess):
            base = getattr(node, 'base', None) or getattr(node, 'expression', None)
            if isinstance(base, str):
                out.add(base)
        elif isinstance(node, fast.Assignment):
            # An assignment TARGET references the name too. Without this a
            # lambda whose body only assigns an enclosing variable (never
            # reads it) failed to capture it, so the write landed in a
            # frame-local slot and was silently lost.
            tname = getattr(node, 'name', None)
            if isinstance(tname, str):
                out.add(tname)
            elif isinstance(tname, fast.FieldAccess):
                tbase = getattr(tname, 'base', None)
                if isinstance(tbase, fast.Variable):
                    tbase = getattr(tbase, 'name', None)
                if isinstance(tbase, str):
                    out.add(tbase)
        for attr, value in vars(node).items():
            if attr in ('parent', 'scope', 'location', 'children'):
                continue
            out |= self._free_names(value, _seen)
        return out

    def _checked_refutable_pattern(self, pat_node: Any, construct: str) -> HPattern:
        """Convert an `if let` / `while let` pattern, failing loudly when the
        pattern shape is not understood.

        _convert_pattern is itself loud now, but it still legitimately answers
        a wildcard for `_`; for `if let`/`while let` a wildcard reached any
        other way would make the branch unconditionally taken, so this second
        check stays as a belt-and-braces guard.
        """
        try:
            pat = self._convert_pattern(pat_node)
        except HIRLoweringError as exc:
            # Name the construct: `if let`/`while let` read very differently
            # from a match arm even for the same rejected pattern shape.
            raise type(exc)(f"{construct}: {exc}") from exc
        is_source_wildcard = (
            pat_node is None
            or type(pat_node).__name__ == "WildcardPattern"
            or (isinstance(pat_node, fast.Variable)
                and getattr(pat_node, 'name', None) == "_")
        )
        if pat.kind == "wildcard" and not is_source_wildcard:
            raise NotImplementedError(
                f"{construct}: unsupported pattern shape "
                f"{type(pat_node).__name__} — refusing to degrade it to a "
                "match-anything wildcard")
        return pat

    def _convert_pattern(self, p: Any) -> HPattern:
        """Convert a frozen-AST pattern node into an HPattern.

        Note: metaxu_ast defines two WildcardPattern classes (a value-level
        Pattern and a TypePattern); the later definition shadows the former in
        the module namespace, so we match by class name where needed.
        """
        if p is None:
            return HPattern(kind="wildcard")
        cls_name = type(p).__name__
        if cls_name == "WildcardPattern":
            return HPattern(kind="wildcard")
        if isinstance(p, fast.VariablePattern):
            return HPattern(kind="var", name=str(getattr(p, 'name', '_')))
        if isinstance(p, fast.LiteralPattern):
            v = getattr(p, 'value', None)
            if isinstance(v, fast.Literal):
                v = getattr(v, 'value', None)
            return HPattern(kind="literal", value=v)
        if isinstance(p, fast.VariantPattern):
            subs = tuple(self._convert_pattern(sp) for sp in (getattr(p, 'patterns', []) or []))
            enum_name = getattr(p, 'enum_name', None)
            return HPattern(kind="ctor",
                            name=str(getattr(p, 'variant_name', '')),
                            enum_name=str(enum_name) if enum_name is not None else None,
                            subpatterns=subs)
        # Raw python literal used as a pattern (e.g. IfDesugarPass emits
        # LiteralPattern(True); tolerate bare values defensively too)
        if isinstance(p, (bool, int, float, str)):
            return HPattern(kind="literal", value=p)
        # The parser's arm grammar is `expression => body`, so parsed match
        # arms carry *expression* nodes as patterns. Convert the pattern-like
        # expression forms.
        if isinstance(p, fast.Literal):
            return HPattern(kind="literal", value=getattr(p, 'value', None))
        if isinstance(p, fast.Variable):
            name = str(getattr(p, 'name', '_') or '_')
            if name == "_":
                return HPattern(kind="wildcard")
            # A bare zero-arg variant name (`Point => ...`) is a constructor
            # pattern, not a catch-all binding.
            if name in self._variant_to_enum:
                return HPattern(kind="ctor", name=name,
                                enum_name=self._variant_to_enum[name], subpatterns=())
            return HPattern(kind="var", name=name)
        # Negative number literal pattern: `-1 => ...` parses as UnaryOperation.
        if isinstance(p, fast.UnaryOperation):
            operand = getattr(p, 'operand', None)
            if (getattr(p, 'operator', None) == '-'
                    and isinstance(operand, fast.Literal)):
                v = getattr(operand, 'value', None)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return HPattern(kind="literal", value=-v)
            raise UnsupportedConstruct(
                f"[{self._pat_where(p)}] unsupported pattern: `{getattr(p, 'operator', '?')}"
                f"{type(operand).__name__}` — only a negative numeric literal "
                "(`-1`) is a valid unary pattern", location=self._loc(None, p))
        # `(x, y) => ...` — positional destructuring of the anonymous tuple
        # struct.  ARITY-EXACT: the field names carry the arity
        # (TUPLE_STRUCT_PREFIX), so a 2-element pattern cannot read a
        # 3-tuple.  `()` is REJECTED in pattern position: it has no elements
        # to read, so it would match every value — a silent catch-all, which
        # is precisely the degradation PATTERN_TRIAGE exists to stop.
        if isinstance(p, fast.TupleLiteral):
            elements = list(getattr(p, 'elements', []) or [])
            if len(elements) < TUPLE_MIN_ARITY:
                raise UnsupportedConstruct(
                    f"[{self._pat_where(p)}] unsupported pattern: "
                    f"`({', '.join('_' for _ in elements)})` has "
                    f"{len(elements)} elements — a tuple pattern needs two or "
                    "more (write `_` for a catch-all)",
                    location=self._loc(None, p))
            subs = tuple(self._convert_pattern(el) for el in elements)
            return HPattern(kind="tuple", subpatterns=subs)
        if isinstance(p, fast.NoneExpression):
            return HPattern(kind="ctor", name="None",
                            enum_name=self._variant_to_enum.get("None"), subpatterns=())
        if isinstance(p, fast.SomeExpression):
            inner = getattr(p, 'value', None)
            subs = (self._convert_pattern(inner),) if inner is not None else ()
            return HPattern(kind="ctor", name="Some",
                            enum_name=self._variant_to_enum.get("Some"), subpatterns=subs)
        # A mode/borrow-annotated binding inside a pattern (`Ok(@mut file)`
        # parses its payload as BorrowUnique('file')): at pattern level the
        # annotation is a static property — the pattern just binds the name.
        # Previously these fell through to the wildcard fallback, so the
        # binding silently vanished and the arm body saw an unbound variable.
        if isinstance(p, (fast.BorrowShared, fast.BorrowUnique, fast.Move)):
            var = getattr(p, 'variable', None)
            if isinstance(var, str):
                return HPattern(kind="var", name=var)
            return self._convert_pattern(var)
        if isinstance(p, fast.ModeExpression):
            return self._convert_pattern(getattr(p, 'expression', None))
        if isinstance(p, fast.FunctionCall):
            callee = str(getattr(p, 'name', '') or '')
            # Builtin Option/Result constructors match even without a user
            # enum declaring them (mirrors the expression-position fallback).
            if callee in self._variant_to_enum or callee in ("Some", "None", "Ok", "Err"):
                default_enum = "Option" if callee in ("Some", "None") else "Result"
                subs = tuple(self._convert_pattern(a)
                             for a in getattr(p, 'arguments', []) or [])
                return HPattern(kind="ctor", name=callee,
                                enum_name=self._variant_to_enum.get(callee, default_enum),
                                subpatterns=subs)
            raise UnsupportedConstruct(
                f"[{self._pat_where(p)}] unsupported pattern: `{callee}(...)` is not a known enum "
                "variant, and a call is not a pattern", location=self._loc(None, p))
        # `Enum.Variant(sub, ...)` and the nullary `Enum.Variant`.
        if isinstance(p, fast.QualifiedFunctionCall):
            parts = list(getattr(p, 'parts', []) or [])
            if len(parts) >= 2:
                subs = tuple(self._convert_pattern(a)
                             for a in getattr(p, 'arguments', []) or [])
                return HPattern(kind="ctor", name=str(parts[-1]),
                                enum_name=str(parts[-2]), subpatterns=subs)
            raise UnsupportedConstruct(
                f"[{self._pat_where(p)}] unsupported pattern: a qualified call "
                "pattern needs at least `Enum.Variant`, got "
                f"{'.'.join(str(x) for x in parts)!r}", location=self._loc(None, p))
        # `Color.Red => ...` parses as a FieldAccess. It used to fall into the
        # wildcard fallback, so the arm matched EVERYTHING and every later arm
        # became dead code.
        if isinstance(p, fast.FieldAccess):
            ctor = self._enum_variant_of(getattr(p, 'base', None),
                                         getattr(p, 'fields', ()) or ())
            if ctor is not None:
                return HPattern(kind="ctor", name=ctor[1], enum_name=ctor[0])
            raise UnsupportedConstruct(
                f"[{self._pat_where(p)}] unsupported pattern: `{p}` is not a known enum variant, and "
                "matching against a field's value is not implemented",
                location=self._loc(None, p))
        if isinstance(p, fast.QualifiedName):
            parts = [str(x) for x in (getattr(p, 'parts', []) or [])]
            # `Color.Red => ...` is a nullary variant pattern; it used to fall
            # into the wildcard fallback, so the arm matched EVERYTHING and
            # every later arm became dead code.
            if len(parts) == 2 and self._variant_to_enum.get(parts[1]) == parts[0]:
                return HPattern(kind="ctor", name=parts[1],
                                enum_name=parts[0], subpatterns=())
            if len(parts) == 1:
                name = parts[0]
                if name == "_":
                    return HPattern(kind="wildcard")
                if name in self._variant_to_enum:
                    return HPattern(kind="ctor", name=name,
                                    enum_name=self._variant_to_enum[name])
                return HPattern(kind="var", name=name)
            raise UnsupportedConstruct(
                f"[{self._pat_where(p)}] unsupported pattern: `{'.'.join(parts)}` is not a known enum "
                "variant, and a dotted name is not a pattern", location=self._loc(None, p))
        # Unknown pattern node. Degrading to a wildcard here made the arm match
        # EVERYTHING — the seam that turned examples/02's
        # `match list { [] -> ..., [x, ...xs] -> ... }` into "always return the
        # first arm". Consult the pattern triage table so the diagnostic says
        # what is unsupported and why.
        entry = PATTERN_TRIAGE.get(cls_name)
        if entry is not None and entry[0] == UNSUPPORTED:
            raise UnsupportedConstruct(
                f"[{self._pat_where(p)}] unsupported pattern {cls_name}: {entry[1]}",
                location=self._loc(None, p))
        raise HIRCompilerBug(
            f"[{self._pat_where(p)}] {cls_name} reached pattern conversion and is "
            "not in PATTERN_TRIAGE: refusing to degrade it to a match-anything "
            "wildcard. Add it to the table in hir.py (see test_hir_coverage).",
            location=self._loc(None, p))

    # Plain-string mode tokens as produced by the parser's binding_prefix
    # (`let @global x = ...` builds ModeAnnotation('global'): mode_type is the
    # bare token, not a Uniqueness/Locality/LinearityMode node).
    _UNIQUENESS_TOKENS = {"unique", "exclusive", "shared", "owned", "const"}
    _LOCALITY_TOKENS = {"local", "global"}
    _LINEARITY_TOKENS = {"once", "separate", "many"}

    def _absorb_mode_token(self, mi: ModeInfo, tok: Any) -> None:
        if not isinstance(tok, str):
            return
        if tok == "mut":
            mi.uniqueness = mi.uniqueness or "mutable"
        elif tok in self._UNIQUENESS_TOKENS:
            mi.uniqueness = mi.uniqueness or tok
        elif tok in self._LOCALITY_TOKENS:
            mi.locality = mi.locality or tok
        elif tok in self._LINEARITY_TOKENS:
            mi.linearity = mi.linearity or tok

    def _param_modeinfo(self, p: Any) -> ModeInfo:
        """ModeInfo of a function/lambda parameter: merges the explicit
        `mode` field with modes carried by a ModeTypeAnnotation type
        (`c: @mut Counter` stores 'mut' on the annotation, not on mode)."""
        ann = getattr(p, 'type_annotation', None)
        return self._extract_modeinfo([
            getattr(p, 'mode', None),
            ann if isinstance(ann, fast.ModeTypeAnnotation) else None,
        ])

    def _extract_modeinfo(self, mode: Any) -> ModeInfo:
        mi = ModeInfo()
        if mode is None:
            return mi
        # A list of annotations (parser binding_prefix): merge every entry.
        if isinstance(mode, (list, tuple)):
            for m in mode:
                sub = self._extract_modeinfo(m)
                mi.uniqueness = mi.uniqueness or sub.uniqueness
                mi.locality = mi.locality or sub.locality
                mi.linearity = mi.linearity or sub.linearity
            return mi
        # Mode-carrying type annotation (`c: @mut Counter` parses the modes
        # onto a ModeTypeAnnotation wrapping the base type, with plain-string
        # tokens like 'mut'/'const'/'local').
        if isinstance(mode, fast.ModeTypeAnnotation):
            self._absorb_mode_token(mi, getattr(mode, 'uniqueness', None))
            self._absorb_mode_token(mi, getattr(mode, 'locality', None))
            self._absorb_mode_token(mi, getattr(mode, 'linearity', None))
            return mi
        # UniquenessMode
        if isinstance(mode, fast.ModeAnnotation):
            # ModeAnnotation wraps a mode_type, which can be Uniqueness/Locality/Linearity
            mt = getattr(mode, 'mode_type', None)
            if isinstance(mt, fast.UniquenessMode):
                mi.uniqueness = getattr(mt, 'mode', None)
            if isinstance(mt, fast.LocalityMode):
                mi.locality = getattr(mt, 'mode', None)
            if isinstance(mt, fast.LinearityMode):
                mi.linearity = getattr(mt, 'mode', None)
            self._absorb_mode_token(mi, mt)
        elif isinstance(mode, fast.UniquenessMode):
            mi.uniqueness = getattr(mode, 'mode', None)
        elif isinstance(mode, fast.LocalityMode):
            mi.locality = getattr(mode, 'mode', None)
        elif isinstance(mode, fast.LinearityMode):
            mi.linearity = getattr(mode, 'mode', None)
        # Chained/combined modes: ModeAnnotationList pattern
        # Some parser variants may combine; attempt to read common fields if iterable
        try:
            for m in getattr(mode, '__dict__', {}).values():
                if isinstance(m, fast.UniquenessMode):
                    mi.uniqueness = getattr(m, 'mode', mi.uniqueness)
                if isinstance(m, fast.LocalityMode):
                    mi.locality = getattr(m, 'mode', mi.locality)
                if isinstance(m, fast.LinearityMode):
                    mi.linearity = getattr(m, 'mode', mi.linearity)
        except Exception:
            pass
        return mi


def dump_hir(funcs: Sequence[HFun]) -> str:
    out: list[str] = []
    for f in funcs:
        out.append(f"fun {f.sym} : {f.ret_ty}")
        out.append(f"  params: {[(str(s), str(t)) for (s, t) in f.params]}")
        out.append(f"  dict_params: {f.dict_params}")
        out.append(f"  where: {f.where_cls}")
        op_str = f"/{f.body.op}" if getattr(f.body, 'op', None) else ""
        out.append(f"  body: {f.body.kind}{op_str}@{f.body.node_id}")
    return "\n".join(out)
