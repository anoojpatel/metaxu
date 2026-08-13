# Metaxu standard library

A standard library for Metaxu modeled on [Ante's stdlib](https://github.com/jfecher/ante/tree/master/stdlib/src)
— the closest relative language (algebraic effects + ownership). The
*design* is ported, not the code: each module's header comment names its
Ante source, and every function here is written in the subset of Metaxu
the compiler gates prove out today (effects with deep single-shot
handlers, Option/Result builtins, closures, Vec/string builtins,
structs, modules).

## Using it

```
from std.fail import Fail, try_opt, fail_if;
from std.prelude import unwrap_or, clamp;      # curated re-exports
import std.option as opt;                       # qualified use: opt.map(...)
```

`import std.foo` resolves to `std/foo.mx` under the stdlib root:

1. `METAXU_STD_PATH` (environment variable), when set to a directory;
2. otherwise the repo's `std/` directory, located relative to the
   compiler package (`src/metaxu/compiler/module_loader.py` →
   `../../../std`).

`std.*` names with **no** file under the root (`std.simd`, `std.matrix`,
`std.geometry`, …) keep the historical placeholder behavior: the import
succeeds, nothing is rewritten, and calls fall through to interpreter
builtins or fail loudly at run time. Documented examples importing those
keep compiling unchanged.

## Module inventory

| Module | Ante source | Status | Contents |
| --- | --- | --- | --- |
| `std.fail` | `Fail.an` | implemented | `effect Fail`; `try_opt`, `on_fail`, `on_fail_else`, `fail_if`, `ok_if`, `ignore_fail` |
| `std.throw` | `Throw.an` | implemented | `effect Throw`; `catch_`, `catch_or`, `catch_or_else`, `map_err`, `unwrap_err`, `throw_if` |
| `std.early_return` | `EarlyReturn.an` | implemented | `effect EarlyReturn`; `with_early_return`, `return_if` |
| `std.stream` | `Stream.an` | implemented (core) | `effect Emit`, `effect Loop`; producers `iota`/`emit_range`/`emit_vec`; consumers `iter`/`for_`/`fold`/`sum`/`product`/`count`/`collect`/`all_of`/`any_of`/`find`; transformers `map`/`filter`/`take`/`skip`/`chain` |
| `std.option` | Maybe surface (`Prelude.an`) | implemented | `map`, `and_then`, `filter`, `or_else`, `unwrap_or(_else)`, `is_some`, `is_none`, `contains`, `ok_or`, `flatten` |
| `std.result` | Result (`Throw.an`) | implemented | `map`, `map_err`, `and_then`, `unwrap_or(_else)`, `is_ok`, `is_err`, `ok`, `err` |
| `std.math` | `Math.an` | partial | `pi`/`tau`/`e` (as functions), `abs`, `min`, `max`, `clamp`, `sign`, `sqrt`, `sin`, `cos`, `powi` |
| `std.vec` | `Vec.an` | implemented | `of1..of3`, `range_vec`, `sum`, `product`, `contains`, `index_of`, `first`, `last`, `is_empty`, `map`, `filter`, `reverse`, `concat`, `max_of`, `min_of` |
| `std.string` | `String.an` | partial | `is_empty`, `eq`, `concat`, `repeat`, `join`, `char_at`, `contains_char`, `count_char`, `index_of_char`, `starts_with`, `ends_with`, `reverse` |
| `std.map` | `HashMap.an` | placeholder (by design) | assoc-list `Map` struct; `empty`, `size`, `is_empty`, `contains_key`, `get`, `get_or`, `put`, `remove`, `keys`, `values` — **every op is O(n)**; API shaped so a real hash map can replace the representation |
| `std.prelude` | `Prelude.an` | implemented | curated re-exports (`public from ... import`) of the unambiguous names |

Not ported (no Metaxu runtime surface yet): `IO.an`, `Env.an`,
`Time.an`, `Rc.an`, `Sync.an`, `C.an`, `Char.an`, `Hash.an`, `Seq.an`,
`Slice.an`.

## The effect idioms (what makes this Ante's design)

- **`Fail` / `Throw` / `EarlyReturn` are effects, not control-flow
  statements.** `try_opt(f)` installs a handler that answers `None` when
  the computation performs `fail()` — the handler *aborts* (returns
  without `resume`), so nothing after the failing perform runs; a clean
  run answers `Some(result)`. `catch_` is the same shape producing
  `Ok`/`Err`.
- **Streams are the `Emit` effect** (`Stream.an`): a stream is a thunk
  performing `emit(x)` per element. Consumers are handlers around
  calling the thunk; transformers are handlers that re-emit and are
  returned as new thunks, so `sum(filter(map(iota(10), f), p))` chains
  lazily with no intermediate collections.
- **`break_`/`continue_` are ABORT-style handlers per iteration**
  (`Stream.an`'s `for_`): `for_` delivers each element under a fresh
  `Loop` handler whose cases return without resuming, tearing down just
  that iteration's delimited body; `break_` additionally stops pulling
  the source.
- **`fold` is Ante's `foldr`**: the handler case computes
  `f(x, resume(()))`, and `resume` returns the completion value of the
  whole rest of the delimited body — accumulation with no mutable state.
- **`map_err` re-throws from inside a handler case**: a case evaluates
  outside its own delimitation, so its `perform Throw.throw(...)` routes
  to the next enclosing handler, exactly like Ante's version.

## Deviations from Ante, and why

- `try` → `try_opt`, `catch` → `catch_`: `try`/`catch` are reserved
  words in Metaxu (held for the deferred try/catch statement).
- Ante's `assert` → `ok_if`: `assert` is an interpreter builtin.
- `Throw t` (generic payload) → unannotated payload: Metaxu effect ops
  accept an unannotated parameter, which the checker treats permissively.
  One program can throw ints and strings; a generic-effects pass can
  tighten this later without changing callers.
- Constants (`pi`, `tau`, `e`) are zero-argument functions: module-level
  `let` bindings do not resolve across module boundaries yet (see gaps).
- No `Stream` trait / implicit impls: producers are explicitly thunks.
- `panic_on_fail`, `or_panic`, `retry_until_success` (Fail.an),
  `enumerate`, `zip`, `map2`, `intersperse` (Stream.an) are deferred —
  see gaps below for the specific blockers.

## Language gaps this library exposed

Real library code turned out to be an effective compiler test. Found
while building it (status as of this port):

1. **Handler sub-function name collisions across functions** (fixed in
   `lower_hir_to_mir.py` as part of this work): two functions handling
   the same effect+op lowered their handler cases to identically-named
   MIR functions (`__handler_Emit_emit_hs1`); the last one loaded won,
   silently running the wrong handler body. Names are now qualified with
   the enclosing function symbol, like lambda names already were.
   Regression test: `test_stdlib.py::test_two_functions_handling_same_effect_do_not_collide`.
2. **Re-export chains resolved too early** (fixed in `module_loader.py`):
   `public from a import x` in module `b` was invisible to
   `from b import x` when `b`'s own imports had not been processed yet,
   and calls through re-exported bindings were never rewritten to the
   declaring module's symbol. Import checks are now deferred until all
   modules load, and bindings chase re-export chains. `std.prelude`
   depends on this.
3. **A handler arm of the form `op() -> ()` is silently dropped** by the
   parser's arm list (comma-separated arms with a unit-literal body eat
   the following arm too). Workaround used throughout: block bodies
   (`op() -> { () }`) and newline-separated arms.
4. **Closure captures of scalars are by value**: mutating a captured
   `int`/`bool` inside a closure or handler case does not write back
   (silently). Vec captures alias the shared runtime vector, so the
   library uses Vec cells where cross-frame mutation is needed
   (`for_`'s `broke` flag, `take`/`skip` counters).
5. **Module-level `let` bindings do not resolve**: a top-level
   `let PI = 3.14;` compiles, but a function in the same module file
   reading it gets unit back (silent). Constants are functions for now.
6. **No modulo operator**: `%` is not a lexer token (the MIR interpreter
   supports the binop; the surface syntax cannot produce it). Parity in
   library/test code is `x - (x / 2) * 2`.
7. **`v[i] = x` (Vec index assignment) is a silent no-op**: the parse
   succeeds but no store happens. `std.map.remove` rebuilds its vectors
   through pop/push instead of writing in place.
8. **No tuple destructuring** (`let (a, b) = p` is a parse error), which
   is what defers `enumerate`/`zip`: they would emit pairs no consumer
   could take apart.
9. **Unqualified keywords**: `try`, `catch`, `some`, `none`, `option`
   are reserved and unusable as function names, even where the grammar
   would be unambiguous (contextual-keyword handling already exists for
   the `x.keyword` position).

Items 3, 4, 5 and 7 are *silent* — code compiles and runs with the
wrong meaning. They are the ones most worth fixing next; the library
deliberately avoids them rather than depending on today's behavior.

## Testing

`src/metaxu/compiler/tests/test_stdlib.py` exercises every module
through the full pipeline (parse → module resolution → strict
infer/borrow-check → HIR → MIR → interpreter), plus the loader's
placeholder fallback, the `METAXU_STD_PATH` override, and regressions
for gaps 1–2.
