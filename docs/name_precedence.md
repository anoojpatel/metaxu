# Name-resolution precedence: user functions win over builtins

Metaxu's runtime provides builtins with ordinary-looking names (`len`,
`push`, `pop`, `print`, `sqrt`, `to_string`, `malloc`, …).  Before this
change, a plain call resolved builtins **first**, so a program that
declared `fn push(list: List, item: T)` could never call it — the Vec
builtin shadowed it and the call died with
`push: expected a Vec receiver, got 'List'`
(`examples/collections.mx` is the motivating case).  That contradicted the
language's own rule for trait dispatch, where user impls have always won
over builtins.

The rule below is now enforced identically by the MIR interpreter
(`compiler/mir_interp.py`, the semantics reference), the LLVM backend
(`compiler/codegen_llvm.py`) and the CLIF backend
(`compiler/codegen_clif.py`), and is pinned by
`src/metaxu/compiler/tests/test_name_precedence.py`.

## 1. Plain call position — `f(a, b)`

```
1. a local/parameter bound to a closure value named f   (unchanged)
2. a compiler dispatch form (__trait$ / __static$ / __mx_effect_runtime$)
3. a USER MODULE FUNCTION named f                       <-- new
4. the BUILTIN named f
5. loud error
```

A user function wins; the builtin is the fallback when no user function of
that name exists.  Locals still shadow both, exactly as before.

## 2. Method position — `x.m(args)`

Unchanged in behaviour, but now *marked* so it stays distinguishable from
a plain call in MIR:

```
1. a user IMPL of m for the receiver's runtime type   (__trait$m)
2. the BUILTIN m                                      (__builtin$m)
3. a plain user function m                            (UFCS fallback)
4. loud error
```

`x.len()` means "the receiver's length".  A top-level `fn len(n: int)` is
not a method of anything, so it must not hijack every receiver in the
program; and this is exactly the order trait dispatch already used
(impl → builtin → plain function), so the two positions now agree with
each other instead of only one of them agreeing with traits.

The rule matters concretely: `std/math.mx` contains
`fn sqrt(x) -> float { x.sqrt() }`.  If method position followed the
plain-call rule, that wrapper would call itself forever.  A user *impl*
still wins in method position, so nothing is taken away from user code —
only plain functions are kept out of method position.

**Implementation.** A method-position call to a runtime builtin method
(`to_string`, `len`, `push`, `pop`, `sqrt`, `sin`, `cos`, `as_ptr`) lowers
to `Call(callee="__builtin$m", operands=(recv, *args))`
(`hir._method_callee`).  A method name declared by any trait or impl keeps
lowering to `__trait$m`, so impls still win.  Any other method name keeps
lowering to a bare call with the receiver first (UFCS), which now prefers
a plain user function like every other plain call.

Compiler-synthesized builtin calls carry the same marker, so surface
syntax can never be rebound:

| syntax | lowers to |
| --- | --- |
| `print(...)` (a grammar production; `print` is a lexer keyword) | `__builtin$print` |
| `-x` / `!x` | `__builtin$neg` / `__builtin$not` |
| `for x in c` (the desugared bound) | `__builtin$len` |

## 3. Module scope — unqualified builtin calls bind lexically

MIR's function namespace is flat, and the module resolver renames an
imported module's functions to dotted paths (`std.vec.map`).  A bare
`len(v)` inside `std/vec.mx` therefore refers to no module function at
all — it is the builtin.  Since plain calls now prefer user functions, an
entry program declaring `fn len` would otherwise have captured every
`len(v)` inside `std/vec.mx`, `std/string.mx` and `std/map.mx`.

`module_loader.ModuleResolver._rewrite_call` closes that hole: an
unqualified call that **this module's scope does not provide** and whose
name is a builtin is rewritten to `__builtin$name` at resolution time.
Effect operation names (one global namespace, called unqualified) are
excluded.  The result is lexical scoping for builtins: each module's bare
`len(...)` means what it means *in that module*.

The shadowable builtin surface is `hir.BUILTIN_FUNCTION_NAMES`, pinned
equal to `mir_interp._register_builtins` by a test.

## 4. Reserved names

**Every name beginning with `__` is reserved for the compiler.**  A user
function declaration with such a name is a loud `ReservedNameError`
(`module_loader.check_reserved_names`, run on the entry file and on every
imported module before renaming) — never a silent override.

Reserving the whole prefix, rather than an enumerated list, is what makes
"user functions win" safe.  The namespace holds:

| name | produced by |
| --- | --- |
| `__impl$Trait$Type$m` | trait impl desugaring |
| `__trait$m`, `__static$Type$m` | method / static dispatch |
| `__effect_default$E$op`, `__effect_runtime$E$op`, `__mx_effect_runtime$SYMBOL` | effect lowering |
| `__module_init` | module-constant initializer |
| `__builtin$m` | resolved builtin calls (sections 2–3) |
| `__index_get`, `__index_set`, `__index_store`, `__slice_get`, `__range`, `__zip`, `__cast` | indexing / slicing / iteration intrinsics |
| `__vec_lit`, `__vec_dim`, `__vec_zeros`, `__vec_filled`, `__vec_comprehension` | fixed-vector intrinsics |
| `__list_lit`, `__list_concat` | list-literal intrinsics |
| `<owner>$lambda<N>`, handler sub-functions | lambda / handle-scope lowering (already unreachable: `$` is not an identifier character) |

Single-underscore names (`_private_helper`) are ordinary user names and
are unaffected.

## 5. What is NOT changed

* Trait dispatch order (`impl → builtin → plain function`) is untouched;
  `codegen_llvm._TRAIT_BUILTIN_FALLBACK` still keeps a trait call that
  falls through to a builtin from landing on a same-named plain function.
* Static dispatch (`Type.method(...)`) already preferred a module function
  over the dotted builtin (`Vec.new`); unchanged.
* Local closure variables still shadow everything (checked first).
* Nothing is weakened: a builtin applied to a receiver it rejects still
  raises the same loud error.

## 6. Corpus impact

Across all 19 gate files, `std/*.mx` and the test suite, exactly two
programs declare a function whose name collides with a builtin:

* `examples/collections.mx` — `fn push<T, const N: int>(list, item)`.
  Previously uncallable; now it is the target of `push(l, x)`.
* `std/math.mx` — `fn sqrt/sin/cos`, thin wrappers over the builtin
  methods.  Imported, they are `std.math.sqrt` and never collided; as the
  entry file their bare names now shadow the builtins for plain calls,
  while their `x.sqrt()` bodies still reach the builtin (section 2).

No other program in the corpus changed behaviour; gates stayed 19/19
pipeline and 19/19 run.
