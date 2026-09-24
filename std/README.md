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
| `std.math` | `Math.an` | partial | `pi`/`tau`/`e` (module constants), `abs`, `min`, `max`, `clamp`, `sign`, `sqrt`, `sin`, `cos`, `powi` |
| `std.vec` | `Vec.an` | implemented | `of1..of3`, `range_vec`, `sum`, `product`, `contains`, `index_of`, `first`, `last`, `is_empty`, `map`, `filter`, `reverse`, `concat`, `max_of`, `min_of` |
| `std.string` | `String.an` | partial | `is_empty`, `eq`, `concat`, `repeat`, `join` (over the `join` builtin), `char_at`, `contains_char`, `count_char`, `index_of_char`, `starts_with`, `ends_with`, `reverse` |
| `std.map` | `HashMap.an` | placeholder (by design) | assoc-list `Map` struct; `empty`, `size`, `is_empty`, `contains_key`, `get`, `get_or`, `put`, `remove`, `keys`, `values` — **every op is O(n)**; API shaped so a real hash map can replace the representation |
| `std.prelude` | `Prelude.an` | implemented | curated re-exports (`public from ... import`) of the unambiguous names |

### Round 2 — the effect-shaped modules Ante does not have

Round 1 ported Ante's design. These six are Metaxu's own: each one exists
because *handlers* are a better answer than the usual global (a logger
object, a seeded RNG singleton, an aborting `assert`).

| Module | Status | Contents |
| --- | --- | --- |
| `std.state` | implemented | `effect State { get, put }`; `eval_state`/`with_state` (result), `exec_state` (final state), `run_state` (both, via `StateResult`), `modify`, `gets`, `update`, `increment` |
| `std.log` | implemented | `effect Log { debug, info, warn, error }` with stdout defaults; `log_*` wrappers; `level_*` constants; handlers `with_stdout_logging`, `quietly`, `collect_logs`, `with_collected_logs`, `run_collected` (`LogRun`), `with_min_level` |
| `std.random` | implemented (seeded only) | `effect Random { next }` with **no default** (see below); `with_seed` (xorshift64), `with_sequence` (scripted draws); `next_below`, `next_range`, `next_bool`, `next_sign`, `choose`, `take_random`, `shuffle` |
| `std.parse` | implemented | `parse_int` (Option) / `parse_int_or` / `parse_int_or_fail` (Fail), `parse_bool`, `digit_value`, `is_digit`, `is_space`, `trim` and `split_on` (over the `trim` and `split` builtins), `parse_int_vec` |
| `std.test` | implemented | `effect Report { passed, failed }`; `assert_true`/`assert_false`/`assert_eq`/`assert_ne`/`check`/`check_eq`; runners `run_suite` (failure count), `run_tests` (`TestReport`), `collect_failures` |

### Round 3 — glade in Metaxu

The package manager's pieces, ported from `src/metaxu/glade/` with the
Python module as the oracle for each (`docs/glade_in_metaxu.md`). These
run on both engines: each module's differential test compares the
interpreter to the Python module and then the native binary to the
interpreter.

| Module | Oracle | Status | Contents |
| --- | --- | --- | --- |
| `std.semver` | `glade/semver.py` | implemented | `Version`, `Interval`, `Range`; `parse_version`, `version_to_string`, `compare_version`; `range_any/empty/exact/between`, `range_intersect/union/complement`, `range_contains` (prerelease rule), `range_is_subset/disjoint`, `range_to_string`; `parse_requirement` in the Cargo dialect (`^`, `~`, `*`, `=`, comparators, commas) |
| `std.solve` | `glade/pubgrub.py` | implemented | PubGrub over a `Graph` of `PackageVersion`s (`graph_new`, `graph_add`, `dep`); `solve(graph, root, root_version, prefer_names, prefer_versions)` returns an `Outcome` with the picks or an `explanation` whose sentences match the Python solver's word for word |
| `std.hex` | `bytes.hex` / `bytes.fromhex` | implemented | `to_hex` (lowercase), `from_hex` (Option), `hex_digit_value`, over bytes as a Vec of ints |
| `std.sha256` | `hashlib.sha256` | implemented | `sha256` (32 bytes), `sha256_hex`, `sha256_string` (UTF-8 of a string); pure Metaxu, so native binaries need no libcrypto |
| `std.iter` | implemented | the adapters `std.stream` defers, over real tuples: `enumerate`, `zip`, `zip_with`, `take_while`, `drop_while`, `step_by`, `windows`, `chunks` |

### Round 3

| Module | Status | Contents |
| --- | --- | --- |
| `std.sort` | implemented | stable merge sort: `sort`, `sort_desc`, `sort_by`, `sort_by_key`; `merge`, `merge_by`; `is_sorted`, `is_sorted_by`; `binary_search` (Option), `unique_sorted`, `min_by_key`, `max_by_key` |
| `std.sync` | implemented | thread-safe sharing (docs/separate_send_sync.md): `Protected`, `protect(v)` (separate by construction — the compiler's spawn checker recognizes it), `with_lock`, `read`, `write`, `update` (atomic read-modify-write; the counter idiom) |

`std.sort` fills a gap neither round covered: `std.vec`'s combinators
are all single-pass and a stream cannot sort at all (sorting needs the
whole sequence). It follows `std.vec`'s contract exactly — eager, Vec
in, fresh Vec out, argument never mutated.

Two choices in it are deliberate:

- **Merge sort, not quicksort.** Stability is the point (it is what
  makes chained `sort_by_key` calls order by several keys: sort by the
  least significant first), it needs no in-place swapping so it fits the
  fresh-Vec-out contract without copying twice, and its worst case is
  its average case. A standard library should not ship an algorithm with
  an adversarial input.
- **Comparators are a strict LESS-THAN predicate** (`less(a, b) -> bool`),
  not a three-way `cmp` returning -1/0/1. One predicate orders a
  sequence, it is cheaper to write at the call site, and it removes the
  question of what a comparator returning 2 means. Stability then falls
  out of a single rule in `merge_by`: take from the left run unless the
  right element is *strictly* less, so ties never cross.

Design notes worth knowing before using them:

- **`std.state` reads its final state out of the handler's own capture
  cell.** `exec_state`/`run_state` install the handler, run the block,
  and then read `current` — the same shared cell both arms write. That
  is the whole trick behind `execState`/`runState` without a second
  effect or a mutable out-parameter.
- **`std.log`'s `with_min_level` RE-PERFORMS.** A handler arm evaluates
  outside its own delimitation, so a `perform Log.warn(...)` inside an
  arm routes to the *next* enclosing `Log` handler (or to the stdout
  default). Filters therefore compose with collectors:
  `with_collected_logs(fn() -> with_min_level(level_warn, body))`
  collects only warnings and errors. Same idiom as `std.throw.map_err`.
- **`std.random` ships NO real-entropy default, deliberately.** The
  `with SYMBOL` runtime-mapping mechanism
  (`examples/effect_mapping.mx`) is how one would be declared, but the
  interpreter's shim table only has `EFFECT_MUTEX_*`/`EFFECT_SPAWN`/
  `EFFECT_JOIN` — there is no entropy shim, and a mapped symbol with no
  shim fails loudly. So `Random.next` has no default clause at all:
  performing it unhandled is a loud unhandled-effect error, never a
  silently constant "random" number. Adding an `EFFECT_RANDOM_SEED` shim
  later is a one-line change in `std/random.mx`.
  The generator is **Marsaglia's xorshift64** (it was an ANSI-C LCG until
  the language grew bitwise operators — gap 13 below, now fixed), with
  one adjustment: Metaxu's `>>` is arithmetic, so the `x >> 7` step is
  masked with 57 ones to make it the logical shift the algorithm's
  GF(2)-linearity argument requires. `next()` answers bits 33..62 of the
  state. A seed is folded away from the generator's one fixed point (0)
  and stirred with three discarded steps, so adjacent small seeds give
  uncorrelated streams. Not cryptographic.
- **`std.test.assert_eq` deliberately shadows the `assert_eq`
  builtin.** The builtin aborts the program on the first mismatch; the
  module's version reports through `Report`, so a suite runs to the end
  and `run_tests` answers totals plus every failure's description. Plain
  calls prefer a user function (`docs/name_precedence.md`), so importing
  the module is all it takes.
- **`std.iter` emits real tuples.** It used to carry a hand-rolled
  `struct Pair<A, B>` with `p.first` / `p.second` accessors, because
  Metaxu had no tuples (gap 8) — which is also why `std.stream` defers
  `enumerate`/`zip`. Gap 8 is fixed, so the struct and its `pair`
  constructor are gone: `enumerate` emits `(index, element)`, `zip`
  emits `(a, b)`, and consumers write `let (i, x) = p;`,
  `for (i, x) in ps { .. }` or `match p { (i, x) => .. }`.
- **`std.iter.zip`/`zip_with` realize their SECOND stream eagerly** (into
  a Vec) before pulling the first. Metaxu's continuations are single-shot
  and delimited, so two producers cannot be stepped in lockstep without
  one being materialized. The module says so rather than hiding it —
  never pass an unbounded stream as `second`.

Not ported (no Metaxu runtime surface yet): `IO.an`, `Env.an`,
`Time.an`, `Rc.an`, `Sync.an`, `C.an`, `Char.an`, `Hash.an`, `Seq.an`,
`Slice.an`.

Deferred inside the round-2 modules, with the blocker:

- `std.random`: real entropy (no runtime shim, above); a float
  `next_float()` (no int→float conversion builtin); weighted sampling
  (wants floats).
- `std.parse`: floats (no string→float builtin, and accumulating digits
  would lose precision), radix prefixes, digit separators, and overflow
  detection (no checked arithmetic — accumulation wraps natively and
  grows unbounded in the interpreter; documented in the module header).
- `std.iter`: `unzip` and `flat_map` (both want a stream of streams or a
  multi-value return); `intersperse` (expressible, but only useful with
  `join`-style consumers that `std.string` already covers).
- `std.test`: a `#[test]`-style registry (no attributes/macros); test
  names are strings passed to each assertion.

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
  `std.log`'s `with_min_level` is the same idiom used as a *filter*.
- **State is a capture cell shared by the handler's arms** (`std.state`):
  `get` resumes with it, `put` writes it, and the runner reads it back
  after the handled block ends — which is how `exec_state`/`run_state`
  answer the final state without a second effect.
- **The sink is a handler choice, not a global** (`std.log`, `std.test`,
  `std.random`): the same computation prints, is silent, is collected
  into a Vec, is filtered by level, is tallied, or draws from a scripted
  sequence, depending only on which handler is lexically in scope.

## Deviations from Ante, and why

- `try` → `try_opt`, `catch` → `catch_`: `try`/`catch` are reserved
  words in Metaxu (held for the deferred try/catch statement).
- Ante's `assert` → `ok_if`: `assert` is an interpreter builtin.
- `Throw t` (generic payload) → unannotated payload: Metaxu effect ops
  accept an unannotated parameter, which the checker treats permissively.
  One program can throw ints and strings; a generic-effects pass can
  tighten this later without changing callers.
- Constants (`pi`, `tau`, `e`) are real module-level `let` bindings,
  read as plain names (`pi`, not `pi()`). Module constants live in one
  global namespace (like types/traits/effects): a duplicate constant
  name in two modules is a loud compile error, and initializers run in
  module-load order before the entry point.
- No `Stream` trait / implicit impls: producers are explicitly thunks.
- `panic_on_fail`, `or_panic`, `retry_until_success` (Fail.an) are
  deferred — see gaps below for the specific blockers.
- `enumerate`, `zip`, `map2` (Stream.an) live in **`std.iter`**, and
  emit real tuples (they emitted a `Pair` struct until gap 8 was fixed).
  `intersperse` is still deferred (see the round-2 deferrals above).

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
3. **FIXED — `op() -> ()` handler arms are no longer dropped.** The
   unit-literal body failed to lower (it had no HIR representation) and
   the arm was silently skipped downstream of the parser. `()` now
   lowers as the unit value — `op() -> ()` is a valid ABORT-style arm
   (returns unit without resuming) — and an arm whose body cannot lower
   is a loud compile error instead of a vanished arm. (Non-empty tuple
   literals were a loud error at the time for want of a runtime
   representation; they are real values now — gap 8.)
4. **FIXED — mutations of captured scalars write back.** A binding that
   a closure, handler arm, or delimited handle body assigns to is boxed
   into one shared cell, so the write is visible in every frame that
   captured it (and writes made outside are visible inside).
   `std.stream`'s Vec-as-cell workarounds (`for_`'s `broke` flag,
   `take`/`skip`'s counters) are now plain mutable scalars.
5. **FIXED — module-level `let` bindings are real constants.** They are
   initialized before the entry point (in module-load order, via a
   synthesized `__module_init`) and are readable from any function.
   Like types/traits/effects they share ONE global namespace: declaring
   the same constant name in two modules is a loud compile error.
   `std.math`'s `pi`/`tau`/`e` are real constants now (read `pi`, not
   `pi()`). Qualified reads (`math.pi`) and import aliases
   (`import ... as`) of constants are not resolved yet — those fail
   loudly at run time, they do not read back as unit.
6. **FIXED — `%` is a real modulo operator** (lexer token, grammar rule
   at multiplicative precedence, `Number` typing constraint; the MIR
   binop and native `srem` already existed).
7. **FIXED — `v[i] = x` stores through to the Vec** (`__index_set` /
   `__index_store`, mirroring `__index_get`), including nested
   `m[i][j] = x`. A fixed `vector[T,N]` in an assignable place
   (`v[0] = 9`, `buf.data[0] = 42`) gets a value-semantics functional
   update written back to the place, like struct field assignment;
   an immutable receiver with no place to write back to (a vector
   reached through another index, a string, a slice target) is a loud
   error, never a no-op. `std.map.remove` now shifts elements in place
   instead of rebuilding through pop/push (still O(n), as documented).
8. **FIXED — tuples exist.** `(a, b)` / `(a, b, c)` are values, `(A, B)`
   is a type, and `let (a, b) = p;`, `match p { (x, y) => .. }` and
   `for (k, v) in pairs { .. }` all destructure. A tuple **is an
   anonymous struct**: `(a, b)` lowers to
   `alloc_struct "__tuple2" { _0of2: a, _1of2: b }` and a pattern reads
   its elements with `field_get`, so MIR gained no op, the interpreter
   gained no value class, and native codegen inherited the whole struct
   path (layout, GEPs, byval params, sret returns) with no backend
   change of its own.
   The field name repeats the arity on purpose: inference has no tuple
   type, so nothing upstream can reject `let (a, b) = triple`, and with
   plain `_0`/`_1` names that would silently bind a *prefix*. Because
   `_0of2` does not exist on a `__tuple3`, an arity mismatch is a loud
   error in either direction with no new runtime check.
   There are **no 1-tuples**: `(e)` is parenthesized grouping (it always
   was), `(e,)` is a syntax error, `()` is unit, and `let (a) = e` /
   `()` in pattern position are rejected by name rather than given a
   `__tuple1` layout.
   Zip *comprehensions* keep their old meaning:
   `f(a, b) for (a, b) in (xs, ys)` is still lockstep iteration over two
   sequences, not a tuple value. A shorthand lambda `(a, b) -> e` is
   still a two-parameter lambda, so lambda parameters are the one
   position that does not destructure.
   Still open: nested destructuring in a `let`/`for` binder list
   (`let ((a, b), c) = ..` — a match arm nests fine), tuple element
   access without a pattern (there is no `p.0`), and two *different*
   tuple types of the same arity in one module, or a tuple nested
   directly inside a same-arity tuple, which demote to the interpreter
   with a reason instead of miscompiling.
   Tests: `src/metaxu/compiler/tests/test_tuples.py`.
9. **Unqualified keywords**: `try`, `catch`, `some`, `none` are reserved
   and unusable as function names, even where the grammar would be
   unambiguous (contextual-keyword handling already exists for the
   `x.keyword` position). `option` is no longer among them: the token
   reachability audit (`docs/token_reachability.md`) found it reserved
   with no production, no AST node and no mention in the docs, and
   de-reserved it along with `box` and `async`.

Items 3, 4, 5, 6, 7 and 8 are fixed (regression tests:
`src/metaxu/compiler/tests/test_silent_seams.py`, and
`test_tuples.py` for 8); the remaining gap (9) is parse-time-loud, not
silent.

10. Block-bodied lambdas (`fn() -> { stmt; stmt }`) do not parse in
    expression position — only expression-bodied lambdas work. Found
    writing `std/state.mx`'s nested-scope test (worked around with named
    helper functions). Parse-time-loud, not silent.

### Found in round 2 (state / log / random / parse / test / iter)

All five of these were found by writing the modules above, and all five
are fixed; the regression tests are at the bottom of
`src/metaxu/compiler/tests/test_silent_seams.py`.

11. **FIXED — `!e` silently compiled as `e`.** `!` was not a lexer token
    at all, and `t_error` merely *logged a warning and skipped* the
    character, so `!cond` lost the `!` and computed the opposite answer
    with no diagnostic — even though `hir` has always lowered
    `UnaryOperation('!')` to `__builtin$not` and
    `docs/name_precedence.md` documents `!x`. Found by
    `std/test.mx`'s `assert_false`, which reported passing tests as
    failures. `!` is a token and a `unary_expression` rule now, and an
    unlexable character is a loud `LexError` instead of vanishing.

12. **FIXED — `&&` and `||` did not exist in the grammar.** The MIR
    interpreter and both backends have always had `&&`/`||` binops;
    nothing could produce them. They now parse at a level between
    `expression` and `comparison_expression`, and they **short-circuit**:
    `a && b` is `if a { b } else { false }` and `a || b` is
    `if a { true } else { b }`. Desugaring to `if` (rather than a MIR
    binop, which would take two already-evaluated operands) is what makes
    guards like `i < len(s) && s[i] == c` safe, and it inherits
    `IfExpression`'s typing, so `1 && 2` is a loud type error. `||` is
    still the empty-parameter lambda opener; the two uses never collide
    because one is at expression start.

13. **FIXED — bitwise operators exist** (`&`, `|`, `^`, `~`, `<<`, `>>`).
    They keep the C/Rust spelling with **Rust's precedence**: shifts bind
    tighter than `&`, `&` tighter than `^`, `^` tighter than `|`, and all
    four tighter than the comparisons, so `flags & MASK == 0` groups the
    way it reads. Four of the six spellings are strict extensions — `^`
    and `~` were illegal characters, `|` in expression position was a
    syntax error, and `<<`/`>>` are synthesized by a new lexer pass from
    *adjacent* angle brackets the generic-argument pass did not claim, so
    `Vec<Vec<int>>` is untouched and `a > > b` is still an error. Only
    binary `&` displaces an old reading (two juxtaposed statements `a`
    and `&b`), exactly as `a` `-b` has always been read as subtraction.
    The operators are **Int-only** (a Float or String operand is a
    compile error via the same class-conflict detection that rejects
    `1 + "a"`), i64 two's complement on both engines, and a shift count
    outside `0..63` is a **loud error** on both — natively via
    `mx_shift_check`, because LLVM's `shl`/`ashr` would be poison there.
    `std/random.mx` is a real xorshift64 now.
    Tests: `src/metaxu/compiler/tests/test_bitwise.py` (spelling, lexing,
    semantics, typing, and native differentials including negative
    operands and the xorshift step).
    Found while fixing it: **the grammar's "no parser conflicts" test was
    vacuous** — `ply.yacc` only reports conflict counts under `debug=True`
    and the parser builds with `debug=False`, so the assertion ran against
    a logger PLY never wrote to and passed on a grammar carrying 154
    shift/reduce conflicts. The test now forces the flag on and asserts
    what is true: zero reduce/reduce conflicts, every shift/reduce
    conflict resolved as shift, count pinned as a ratchet
    (`docs/token_reachability.md`).

14. **FIXED — string literals kept their backslashes verbatim.** `"a\nb"`
    was the four characters `a \ n b` and printed that way, and
    `"say \"hi\""` could not be written at all (the old `"[^"]*"` pattern
    stopped at the escaped quote). Found by `std/parse.mx`'s `is_space`,
    which compared a one-character string against the two-character
    `"\t"` and so never matched. String and f-string literals now decode
    `\n \t \r \0 \\ \" \'`, and an **unknown escape is a loud
    `LexError`** rather than a silently-kept backslash.

15. **FIXED — an `if`/`else` whose branches assign differently-typed
    variables was a spurious type error.** An `Assignment` node's own
    type was unified with the assigned *variable's* type, so
    `if c { flag = false } else { n = n + 1 }` demanded `Bool ~ Int`
    ("one value is required to be Bool and Int") on a perfectly
    well-typed program. An assignment is a statement: its type is `Unit`
    now. The value still has to match the binding, which is the real
    check. Found writing `std/parse.mx`'s `parse_int`; it is also why
    older modules write `if c { x = ... } else { () }` everywhere.

16. **FIXED — `/` and `%` disagreed between the interpreter and the
    backends on negative operands.** The interpreter used Python's
    flooring `//`/`%` (`-7 / 2 == -4`, `-7 % 5 == 3`) while
    `codegen_llvm` and `codegen_clif` emit `sdiv`/`srem`, which truncate
    toward zero (`-3`, `-2`). The same program had two answers and no
    diagnostic — it was even documented as a known divergence in
    `codegen_llvm`'s header comment. The interpreter is the semantics
    reference, so it must not be the odd one out: `mir_interp` now
    truncates toward zero and takes the sign of the dividend, and
    `(a / b) * b + a % b == a` holds on both sides. (The two stale
    "interpreter floors" comments inside `codegen_llvm.py` still need
    updating; that file is owned elsewhere.) Found writing
    `std/random.mx`'s seed normalization.

## Testing

`src/metaxu/compiler/tests/test_stdlib.py` exercises every round-1 module
through the full pipeline (parse → module resolution → strict
infer/borrow-check → HIR → MIR → interpreter), plus the loader's
placeholder fallback, the `METAXU_STD_PATH` override, and regressions
for gaps 1–2.

`src/metaxu/compiler/tests/test_stdlib_effects.py` does the same for the
round-2 modules — state threading through `resume`, a collector handler
answering the lines a computation logged, a filter re-performing into an
enclosing collector, `with_seed` reproducible across two runs (and
seed-dependent), `parse_int` on good and bad input, assertions tallying
instead of aborting, and adapters that stop pulling their source —
alongside `test_std_state.py`, `test_std_log.py` and `test_std_test.py`,
which pin the first cut of those three.

Every `std/*.mx` file is additionally walked by
`test_hir_coverage.py`, which parametrizes over `std/*.mx`, so a new
module is compiled by the suite the moment it lands.
