# Pin audit: invented syntax/pins in the reconstructed chapters

The reconstructed chapters (see MANIFEST.md) pinned outputs by hand.
Each agent reported what it took verbatim from the pre-reset reports
versus what it had to guess. The guesses below are where harness
failures are MOST likely on the restored tree; check them first.

## Chapters 02-05

Error fragments: none invented (all from the report or FIX_PLAN).
Syntax/semantics guesses to vet:
1. Match arms newline-separated, no commas; `=>` arrows.
2. `;` after brace-ended `if` / `if let` / `try` in statement position.
3. Range syntax `0..5`, exclusive upper bound (sum pinned as 10).
4. F-string form `f"hello from {name}"`.
5. Tuple type `(int, int)` and `let (lo, hi) = ...` destructuring.
6. `Tree[T]` bracket generics (normalized from the report's `Tree<T>`).
7. Generic struct literal with inferred T: `Pair { a: 1, b: 2 }`.
8. `Option[int]` as a type annotation.
9. `from std.sort import sort;` and sort returning a NEW Vec (if
   in-place, fix the ch05 example and prose).
10. `let a: vector[int, 3] = [1, 2, 3];` coercion, elementwise `+`,
    value-semantics pin `print(a[0])` = 1 after mutating a copy.
11. `print(v.pop())` pins popped value 30 (pop returns the element).

## Chapters 15-17

Taken verbatim from the report: `ptr_read: use after free (<*heap#1>)`,
`one value is required to be Int and String`,
`<mem>:4:5: undefined function 'helpr'; did you mean 'helper'?`,
len-precedence 999/5, iota/sum values.
Invented, vet first:
1. ch15 ModuleError wording: "cannot import private name 'shave' from
   module 'geometry'" (class + non-pinnability from report, words mine).
2. ch15 ReservedNameError wording for `__tmp`.
3. ch15 `export a, b;` list syntax, `import geometry;` sibling form,
   and the reconstructed norun main.mx import head.
4. ch16 slot-granular heap (malloc counts slots not bytes) and the
   104/105 byte reads.
5. ch16 messages: `free: double free (<*heap#1>)`, the out-of-bounds
   wording (em-dash inside per report), `ptr_write: readonly snapshot`,
   heap identity for string snapshots.
6. ch16 `is_null(f)` as the null test name.
7. ch17 caret-excerpt format, `type check failed:` prefix, 2:13
   location for the Int/String fence.
8. ch17 `run_pipeline_ctx` import path, `ctx.mir.dump()`, abridged MIR
   dump (display-only).

## Chapters 11, 13, 14

Taken verbatim: bare-Vec-capture fence output (ch10's full line
"captures AND WRITES 'shared', which has shared mutable identity"),
4x250 counter = 1000, to_f32(0.1)=0.10000000149011612,
to_f16(0.1)=0.0999755859375, `12.0`, `1e-05`, `pop: Vec is empty`,
1.02x fib / 1.09x nsieve endpoints, ~280ns crossing.
Invented, vet first:
1. ch11 contended-write runtime message wording: "contended write to a
   value with shared mutable identity; protect it with std.sync".
2. ch11 `EFFECT_MUTEX_UNLOCK` name (prose); std.sync import form and
   free-function shapes protect/read/write/update.
3. ch11 caught-error binds as printable message string; `fn bump(v:
   Vec[int])` pushes through a plain parameter.
4. ch13 Tile repr format `Tile[[0, 1, 2], [3, 4, 5]]`.
5. ch13 shape-error wording "shape mismatch: 2x3 vs 3x2".
6. ch13 buffer IO names load_rows/load_rows_or/store_rows/
   store_rows_clipped and launch(n, k) handler arity.
7. ch13 f16 sum pin 0.199951171875 (assumes accumulation in the
   element kind).
8. ch14 middle ratios mandelbrot 1.00x / orbit 1.03x / par_sum 1.06x
   (chosen inside the surviving 1.00-1.09 range; re-pin from the
   suite). The run_metal example is now the chapter's one norun fence.

## Chapters 06, 07, 09, 12

Taken verbatim: the ch06 inference diagnostics ("one value is required
to be Int and String", "Bool and Int", "conflicting instantiations of
type parameter T", "missing `implement Show for Int`"), ch07
"more than one implement block", ch09 `pop: Vec is empty` /
`No handler for effect 'Parser'`, `Option::Some(3)`.
Invented, vet first:
1. ch06 arity fragment "type argument"; "type mismatch for field
   'tag'" on a generic struct; `undefined variable 'totl'`.
2. ch06 generic-fn syntax `fn describe[T](x: T) -> string where T:
   Show` (bracket params extrapolated from struct generics).
3. ch09/ch12 std API shapes, ALL invented: option.map/unwrap_or,
   result.to_option, fail.fail (caught value prints as the message;
   "divide by zero" pin depends on it), early_return.with_return,
   iter.zip, vec.map/filter/sum, sort.sort (copy)/binary_search
   (Option), map.Map new/insert/get/len, string.split/join/to_upper,
   parse.parse_int (Option), math.abs/min/max/pow, random.Rng
   .seed/.range, state.get/put/with_state, test.run_tests.
4. ch12 log format "[info] starting up" / "[warn] low disk" (most
   provisional pins in the book).
5. ch12 random pins properties (1/1/1 bool prints), not draws:
   deliberate deviation, stated in the chapter's prose.
6. Syntax assumptions: `;` after statement-position match, list
   literals build Vecs, method call on a struct literal, Ok/Err as
   prelude constructors.
