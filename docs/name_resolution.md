# Name resolution: an undefined name is a compile-time error

This is the last member of the silent-degradation family that
`docs/token_reachability.md` documents. Until this pass existed, a name that
resolved to nothing was **dropped**:

| program | was | now |
| --- | --- | --- |
| `fn main() -> int { undefined_thing; 42 }` | compiled, ran, answered `42` | `undefined variable 'undefined_thing'` |
| `fn main() -> int { undefined_thing }` | compiled, ran, answered `()` | same |
| `fn main() -> int { helpr(); 7 }` (typo of `helper`) | compiled; died at RUN time with `Unknown callee` | `undefined function 'helpr'; did you mean 'helper'?` |

The lexer audit's headline bug depended on exactly this: `let x = 1e10; x`
compiled and answered `1` **because** the stray `e10` became an identifier in
statement position, where an unused undefined name vanished.

The check lives in `src/metaxu/compiler/name_resolution.py`, runs from
`pipeline.build_context_from_source` (the single front door for source
input), and files structured diagnostics of kind `type-unresolved-name` on
the same channel the frozen checkers use, so `run_pipeline` promotes them to
`TypeCheckError` with a `file:line:column`, a source excerpt with a caret,
and — when an in-scope name is within edit distance 1–2 — a
`did you mean 'x'?` note.

`src/metaxu/compiler/tests/test_name_resolution.py` pins all of it, most of
it in the anti-false-positive direction.

## Why the pass runs on the mutable, post-desugar AST

Every other checker in the front end runs over the frozen AST. This one
cannot, because the frozen AST is lossy in exactly the places names are
bound and used:

* `MatchExpression` freezes its arms into a payload *descriptor* and does
  not carry the arm **bodies** as children at all;
* `ForStatement` freezes its body to an empty node of kind `list`;
* `IfLetExpression` flattens its pattern into the same child tuple as its
  value and its branches, so a pattern's binder occurrence is structurally
  indistinguishable from a variable read.

A resolver over the frozen tree would therefore miss every name inside a
match arm or a for body *and* report every pattern binder as undefined. HIR
lowering itself reads the original nodes
(`HIRBuilder._from_orig_expr(orig, ...)`), so resolving over the mutable
post-desugar AST checks exactly the tree that gets compiled — module
resolution and desugaring have already run, so imported functions carry
their dotted names and trait impls are already mangled `__impl$…` functions.

## What is checked

1. **Variable references** — `x`, the root of a field chain (`x.f.g`), the
   operand of `&x` / `&mut x` / `move(x)` / `exclave x`, and the target of an
   assignment (`x = e`, and the root of `x.f = e` / `v[i] = e`).
2. **Plain call callees** — `f(a, b)`.
3. **Dotted call callees** — `a.b(...)`, following `hir`'s own dispatch
   order for `QualifiedFunctionCall`: an `Enum.Variant(x)` constructor; a
   static impl call `Type.method(x)`; a receiver call `recv.method(x)`
   (where the *receiver* is the value name that gets checked); or a plain
   call of the dotted name, which is what the module system produces for
   `mod.f` after renaming.

## What is in scope

Each entry is a real category of the language, not a whitelist bolted on to
make a file compile. The battery in `TestNoFalsePositives` has one test per
row.

### Lexically scoped (a stack of scopes)

| category | bound by |
| --- | --- |
| parameters | `fn f(a, b)`, `fn(x) -> …`, `x -> …` |
| local bindings | `let x = e` — bound **after** its own initializer, so `let x = x + 1` shadows an outer `x` |
| loop variables | `for x in it { }` |
| pattern bindings | `match`, `if let`, `while let`, including nested subpatterns and the mode-annotated forms (`Ok(@mut v)`, `Some(&r)`, `move(x)`) |
| catch bindings | `try { } catch e { }` |
| handler-arm parameters | `perform Eff.op(p, q) => …` in both the `handle e { … }` and `handle e with { … } in …` forms |
| comprehension targets | `vector[T,N](e for x in it)` |
| effect-operation default parameters | `op(x: int) -> int = x + 1;` |
| `self` | every method body (the receiver is implicit in trait/impl methods) |
| type parameters | the enclosing `fn f<T>` **and** the enclosing `implement<T, const N: int>` block — see below |

**Type parameters are value names.** A const generic is bound to the
receiver's runtime dimension at method entry (`hir.build`'s `_const_dims`
turns `implement<const N: int> … for vector[T,N]` into `let N =
__vec_dim(self, 0)`), so `for i in 0..N` really reads a value; and a plain
type parameter is the argument of the `type_of` reflection intrinsic
(`examples/06_vector_operations.mx` writes `type_of(U) == type_of(T)`).
`desugar` attaches the impl's parameters to each mangled method
unconditionally for this reason — it used to attach them only when the impl
had a `where` clause.

### Program-wide (Metaxu has no forward-declaration rule)

| category | source |
| --- | --- |
| module functions | every `FunctionDeclaration` in the tree — **including nested ones**, because `HIRBuilder.build`'s hoisting walk lifts them all into the flat MIR namespace (docs/name_precedence.md § 3) |
| dotted module functions | the same, after `module_loader` renames an imported module's functions to `std.vec.map`-style paths |
| trait and impl methods | `trait { fn m(…); }` and `implement … { fn m(…) }`, the latter recovered from the mangled `__impl$Trait$Type$m` names |
| extern FFI declarations | `extern "C" { fn malloc(…); type FILE; }` |
| type names | `struct`, `enum`, `trait`, `type`, `effect` and extern type names — legal in value position as static-call receivers (`Buffer.new()`, `Vec.new()`) |
| enum variant constructors | every variant of every `enum`, plus the language-provided `Some` / `None` / `Ok` / `Err` (`hir` builds `Option`/`Result` variants even when no user enum declares them) |
| effect operation names | every `op` of every `effect`, callable unqualified (`emit(x)` is `perform Emit.emit(x)`) |
| module constants | direct module-level `let` statements, which `hir.build` hoists into `__module_init` and publishes as globals before the entry point runs |
| imported names | every local name an `import` / `from … import` statement introduces, alias included |
| runtime builtins | `hir.BUILTIN_FUNCTION_NAMES` (pinned equal to `mir_interp._register_builtins`), plus the dotted `Vec.new`, plus `type_of` / `Vec` / `vector` |
| the `__` namespace | every name starting with `__` is the compiler's (`module_loader.check_reserved_names`) and is exempt |
| `null` | the null-pointer literal; the parser produces a plain `Variable`, which `hir` lowers to a literal |
| `_` | the wildcard |

### The `std.*` placeholder namespace

`std.simd`, `std.effects`, `std.matrix`, … have no file under the stdlib
root, so `module_loader` takes its documented external-placeholder path: the
import succeeds, **nothing is rewritten**, and calls fall through to
interpreter builtins or fail loudly at run time. Both halves are covered by
one rule rather than a special case: a name an import statement introduces
is in scope, and a dotted callee whose root is such a name is accepted.

## What is deliberately NOT checked, and why

* **Method names on a computed receiver** (`f().m()`, `(a + b).m()`,
  `x.to_string()`). `hir._method_callee` lowers these to `__trait$m` —
  runtime dispatch on the receiver's *runtime type* (docs/name_precedence.md
  § 2). Which impl answers is not decidable in the front end, and the
  builtin fallback means "no impl" is not the same as "no such name". The
  receiver expression itself is fully checked; only the method name is left
  to the runtime, which raises loudly when nothing answers.
* **Effect names in `perform Eff.op(…)`**. An unhandled or misspelled
  effect already fails loudly at run time (`No handler for effect 'Eff'`),
  and the constraint emitter already reports a perform with no enclosing
  handler and no `performs` clause as an advisory. Dynamically-scoped
  handlers installed by callers are legitimate and undecidable here, so this
  stays where it is rather than becoming a hard error under a different
  banner.
* **Struct field names** (`s.no_such_field`). That is a type question, not a
  name-scope one; it belongs with the field-mode checks in
  `frozen_constraint_emitter.py`.
* **Statement lists that are not `Block`s.** Scopes are pushed where the AST
  makes the scope explicit: function and lambda bodies, `Block`s, loop
  bodies, pattern arms, handler arms, `catch` bodies. Other statement lists
  (an `unsafe { }` interior, a bare `if` branch list) are walked in the
  enclosing scope. That can only make the check *more permissive* — a
  binding leaks outward and a use of it is accepted — never wrong.

## Statement position does not swallow

The second half of the original bug was in MIR lowering, and it was worth
auditing separately: does an expression whose *value* is unused still get
*evaluated*?

Surveyed form by form, everything that can be observed survives statement
position — `f()` keeps its `call`, `v[i]` keeps its `__index_get` (and so
its bounds check), `x / 0` keeps its `binop`, `s.a` keeps its `field_get`,
`x as float` keeps its `__cast`, `x = e` keeps its write. Each of those
lowers by *emitting* an instruction, so discarding the resulting slot name
discards nothing.

The one exception was a bare name read. `lower_hir_to_mir.lower_expr`
answers a `Var` with the slot name and emits **no instruction at all**; in
value position the consumer names that slot so the read is real, but in
statement position the name was thrown away and the read disappeared —
which is precisely how `undefined_thing; 42` compiled to `ret 42` with the
undefined name nowhere in the MIR. `Block` lowering now emits an explicit
`copy` for a statement-position `Var`, so the read happens.

## Defence in depth

The compile-time check does not replace the engines' own guards; a check
that ever misses must still fail loudly rather than silently:

* `mir_interp` raises `Unknown callee: 'f'` for a call it cannot resolve;
* `mir_interp._lookup` raises `Unbound variable 'x'` for a slot read that
  finds nothing — which the statement-position fix above now makes
  reachable for a discarded read too.

Both are pinned by tests that switch the front-end pass off and run the
program anyway.

## Corpus impact

Across the 19 gate files, every `std/*.mx` and the ~1,340 programs the test
suite compiles, the check reports **zero** false positives. It found two
real defects in shipped code, both of which were fixed rather than
whitelisted:

* `examples/06_vector_operations.mx` — `zip` asked
  `SimdOp.try_vectorize(self, fn(x: T) -> V { f(x, other[i]) })`, where `i`
  was never bound. It compiled because the undefined name was dropped, and
  it never blew up because the capability answers `None` before the lambda
  is ever called. `try_vectorize` takes a *unary* function, so there is no
  vectorized form to ask for in a two-operand zip; the body is now the
  scalar element-wise form it always actually ran.
* `implement<T, const N: int> …` methods could not see `T` at all unless the
  impl also had a `where` clause (`desugar` gated `_impl_type_params` on
  it). Fixed at the source rather than by exempting type parameters from the
  check.

One test moved its diagnostic earlier rather than changing what it asserts:
`handle (body()) { }` — the parenthesized-subject spelling that
`docs/token_reachability.md` records as "a loud unknown-callee error, not a
silent no-op" — is now a compile-time `undefined function 'handle'` instead
of a run-time `Unknown callee`.
