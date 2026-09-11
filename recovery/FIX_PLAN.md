# Six compiler fixes to re-apply after restoring the 258af2d base

Apply IN THIS ORDER (later patches' context strings assume earlier ones
landed). Each was verified on the pre-reset tree; the suite after all
six: **2148 passed, 1 skipped**, gates 20/20 + 20/20. All `rep(old, new)`
patch scripts assert occurrence counts, so a failed assert means the
base is wrong, not that you should force it.

## 1. Bool print parity (interp formats bools as their word, 1/0)

Native erases bools to i64 and prints 1/0; the interpreter printed
Python True/False. Fix in `src/metaxu/compiler/mir_interp.py`:

- Add before `class MxUnit`:

```python
def mx_display(v):
    """A value as the user-visible formatter receives it: booleans format
    as their word (1/0) on BOTH engines — native erases bools to i64, so
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
```

- Route `print`/`println` builtins through
  `print(*(mx_display(a) for a in args))`; `int_to_str` through
  `str(mx_display(x))`; `to_string` through
  `"()" if x is UNIT else str(mx_display(x))`.
- In container reprs replace `repr(...)` on user payloads with
  `mx_repr(...)`: MxStruct tuple repr and named-field repr
  (`f"{k}={mx_repr(v)}"`), MxVariant payload join, MxVec items join,
  MxVector elements join.
- `test_codegen_llvm.py::interp_run`'s `_print` capture must apply
  `str(mx_display(a))` too (it overrides the builtin), and the file's
  mir_interp import gains `mx_display`.
- Regression test appended to test_codegen_llvm.py:

```python
@needs_clang
def test_bool_print_parity_native_matches_interp(tmp_path):
    # Booleans format as their word (1/0) on BOTH engines: native erases
    # bools to i64, and the interpreter's mx_display matches it, closing
    # the divergence the book surfaced (print(true) was "True" vs "1").
    src = """
fn main() -> int {
    print(true);
    print(false);
    print(1 < 2);
    print((3 > 2).to_string() + "!");
    0
}
"""
    _res, out = interp_run(src)
    assert out.splitlines() == ["1", "0", "1", "1!"]
    assert_native_matches_interp(src, tmp_path)
    # container reprs format bools the same way (interp-only: native
    # print of a whole Vec demotes honestly rather than diverging)
    _res, out = interp_run("""
fn main() -> int {
    let @mut v = Vec.new();
    v.push(true);
    v.push(false);
    print(v);
    0
}
""")
    assert out.splitlines() == ["Vec[1, 0]"]
```

## 2. Bare qualified payload-less variants (`Shape::Dot`)

`src/metaxu/parser.py`, p_postfix_expression: add the production line
`| postfix_expression DOUBLECOLON IDENTIFIER` directly under the
6-token DOUBLECOLON call form, and change the `'::'` action to:

```python
        elif p[2] == '::':
            base = p[1]
            # Bare `Enum::Variant` (no parens) is the zero-argument form:
            # payload-less variants construct and PATTERN-MATCH under
            # their qualified name, same as `Enum::Variant()`.
            args = (p[5] if p[5] else []) if len(p) == 7 else []
            if isinstance(base, ast.GenericInstance):
                base = base.base
            parts = self._name_parts(base) or [str(base)]
            p[0] = ast.QualifiedFunctionCall(parts + [p[3]], args)
```

This adds exactly ONE shift/reduce conflict (resolved as shift on
LPAREN). Bump `_EXPECTED_SHIFT_REDUCE_CONFLICTS` in
test_token_coverage.py from 162 to 163 with this comment:

```
#: 162 -> 163: bare `Enum::Variant` (the zero-argument qualified variant
#: form, usable in construction AND pattern position) shares a prefix
#: with the call form `Enum::Variant(args)`; on LPAREN the parser must
#: SHIFT into the call reading, which is exactly the maximal-munch
#: resolution this test requires.
```

Regression test appended to test_codegen_llvm.py
(`test_bare_qualified_variant_construct_and_match`): match on
`Shape::Dot` and `Shape::Circle(7)` in both positions, interp output
["dot", "7"], plus `assert_native_matches_interp`.

## 3. CoherenceError becomes a typed TypeCheckError

`src/metaxu/compiler/desugar.py`: import BorrowError + TypeCheckError
from frozen_borrow_checker (no cycle) and rebase the class:

```python
class CoherenceError(TypeCheckError):
    """... (docstring: typed diagnostic, kind "type-coherence") ..."""

    def __init__(self, message: str, location: Any = None):
        self.location = location
        from metaxu.errors import format_location, source_excerpt
        if location is not None:
            message = f"{format_location(location)}: {message}"
            excerpt = source_excerpt(location)
            if excerpt:
                message = f"{message}\n{excerpt}"
        self.errors = [BorrowError(message=message, node_id=-1,
                                   kind="type-coherence", variable="",
                                   location=location)]
        Exception.__init__(self, message)
```

Existing test_trait_dispatch tests still pass (subclass + message).

## 4. @const field-write enforcement

`src/metaxu/compiler/frozen_constraint_emitter.py`. Five pieces, all
mirroring the existing global_struct_bindings scoping machinery:

- New map next to `global_struct_bindings: dict[str, str] = {}`:
  `struct_type_bindings: dict[str, str] = {}` (any-locality name ->
  struct name; a name not present is simply unchecked — zero false
  positives).
- New `stb_saves` list parallel to `gsb_saves`; push/pop_scope mirror
  the save/restore; the FunctionDeclaration handler saves/restores a
  `saved_struct_typing = dict(struct_type_bindings)` beside
  `saved_global_bindings`.
- LetBinding handler: after the gsb shadow-save/pop block, do the same
  for struct_type_bindings, then record
  `struct_type_bindings[var_name] = payload name` for any
  StructInstantiation child (regardless of locality).
- Parameter loop (BOTH occurrences — function decls and lambdas):
  change `for name, child in zip(params, param_children):` to
  enumerate with `_declared_ptypes = list(payload_dict(node).get("param_types") or [])`
  read before the loop, and inside, after declare_variable:

```python
                        _pt = (_declared_ptypes[_pidx]
                               if _pidx < len(_declared_ptypes) else None)
                        if isinstance(_pt, str) and _pt in struct_defs:
                            if stb_saves:
                                stb_saves[-1].setdefault(
                                    name, struct_type_bindings.get(name))
                            struct_type_bindings[name] = _pt
```

  (Parameter frozen payloads carry NO type; the function payload's
  aligned "param_types" list is the source.)
- The check, at the top of the dotted-target Assignment branch (before
  `container_struct = global_struct_bindings.get(base_name)`): walk
  `field_path.split(".")` through struct_defs; a segment whose field
  dict has `"const" in (f.get("mode") or [])` appends a BorrowError
  kind "const-field-write", message
  `cannot assign to @const field '{seg}' of {cur} (binding '{base}')`;
  otherwise descend into the field's type when it's a struct name.

Field mode payloads look like `{"name": "name", "type": "string",
"mode": ["const"]}`; plain fields have mode None.

Regression tests appended to test_deep_field_modes.py: let/param/nested
rejections, mut+plain fields writable, and the shadowing
zero-false-positive case (P has @const name; a Q-typed `p` writes
`name` fine).

## 5. Rebinding discipline (plain `let`, params, module consts immutable)

Same file, in the Assignment handler BEFORE `binding_ty = lookup(...)`:

```python
            if (isinstance(target_name, str) and "." not in target_name
                    and not target_name.startswith("__")):
                # Rebinding discipline: assignment needs a binding declared
                # mutable (`let mut x` / `let @mut x` / a mut-mode param).
                # Compiler-generated names (__ prefix) and names the
                # checker never saw (captures resolved elsewhere) are left
                # alone: never a false positive on generated lowerings.
                _info = borrow_checker.variables.get(target_name)
                if _info is not None and _info.mode == "shared":
                    borrow_checker.errors.append(BorrowError(
                        message=(
                            f"cannot assign twice to immutable binding "
                            f"'{target_name}'; declare it mutable "
                            f"(`let mut {target_name} = ...`)"),
                        node_id=node.node_id,
                        kind="immutable-rebind",
                        variable=target_name,
                    ))
```

Fallout (three sites, all fixed on the old tree):
- examples/linked_list.mx ~line 143: the if-let binder rebind becomes
  `let mut updated = value; updated = 42;` (output unchanged).
- test_round5_regressions.py: `test_native_plain_rebinding_matches_interp`
  (with its @needs_clang decorator) becomes
  `test_plain_rebinding_is_now_rejected` pinning BorrowCheckError
  "cannot assign twice to immutable binding 'p'" on PLAIN_REBIND_SRC.
- test_codegen_llvm.py: `test_local_shadow_of_module_constant_demotes`
  becomes `test_assignment_to_module_constant_is_rejected` pinning the
  same error for 'BASE' (llvm_from_source now raises).

## 6. Match guards (`pattern if cond => body`)

`src/metaxu/parser.py`:
- p_arm grammar gains `| expression IF expression arm_arrow arm_body`;
  guarded arms are 3-tuples (pattern, guard, body). NO new grammar
  conflicts (count stays 163).
- p_match_expression desugars when any arm has a guard, via two new
  methods (verbatim):

```python
    def _desugar_guarded_match(self, scrut, arms, pos):
        """Match guards, desugared at parse time.

        `p if g => b` becomes `p => if g { b } else { match s { <rest> } }`
        with the remaining arms duplicated into the else.  The scrutinee
        binds ONCE: when it isn't already a variable, the whole match is
        wrapped in an immediately-called lambda taking the scrutinee, so
        effects in it never run twice.  Guarded arms deliberately do not
        count toward exhaustiveness (their inner rest-match must still
        cover, or the checker rejects the program), which is exactly the
        semantics guards need: a failing guard falls through."""
        if isinstance(scrut, ast.Variable):
            return self._guarded_arms_match(scrut.name, arms)
        temp = f"__guard_scrut_at{pos}"
        lam = self._make_lambda(
            [ast.Parameter(temp)],
            self._guarded_arms_match(temp, arms))
        return self._make_call(lam, [scrut])

    def _guarded_arms_match(self, scrut_name, arms):
        cases = []
        for i, arm in enumerate(arms):
            if len(arm) == 3:
                pat, guard, body = arm
                rest = arms[i + 1:]
                fallback = (self._guarded_arms_match(scrut_name, rest)
                            if rest else
                            ast.MatchExpression(
                                ast.Variable(scrut_name), []))
                cases.append((pat,
                              ast.IfExpression(guard, body, fallback)))
            else:
                cases.append((arm[0], arm[1]))
        return ast.MatchExpression(ast.Variable(scrut_name), cases)
```

  IMPORTANT: build the IIFE with `self._make_lambda` (it wires the
  lambda's scope; a bare ast.LambdaExpression crashes
  _update_child_scopes).
- p_handle_expression rejects 3-tuple arms with a CompileError
  ("handler arms do not take guards", ParseError, note that
  `pattern if cond =>` is a MATCH arm form).
- New file test_match_guards.py (7 tests): fallthrough order
  (big/small/non-positive/nothing classify), native differential,
  effectful scrutinee evaluates once ("evaluated" printed once), binder
  in scope in guard and body, uncovered fallthrough is a loud runtime
  InterpError "no pattern matched" (checker doesn't look inside arm
  bodies yet — documented), catchable via try/catch, handler-arm guard
  rejection.

## Book chapter updates that accompany the fixes

- ch02: `let` is immutable; pinned error block "cannot assign twice to
  immutable binding 'x'".
- ch04: a Guards section (runnable classify example) + the caveat that
  guarded arms don't count toward exhaustiveness; bare `Command::Quit`
  qualifies in both positions.
- ch07: coherence gets a pinned ```metaxu error block (fragment "more
  than one implement block").
- ch10: the "@const not enforced" honesty note becomes a pinned error
  block "cannot assign to @const field 'name' of P".
