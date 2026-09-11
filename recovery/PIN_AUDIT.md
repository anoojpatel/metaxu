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
