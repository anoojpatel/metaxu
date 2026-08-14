# Token and grammar reachability

`hir.AST_NODE_TRIAGE` guarantees that no AST node class can reach lowering
without a decision having been made about it. This document is the same
guarantee one layer up, for the layer where the worst bug of this project
actually lived: **`!` was never a lexer token**, and `t_error` merely logged
and skipped characters it did not recognise, so `!e` compiled as `e` — with
the wrong answer, no diagnostic, and nothing a test would notice — for the
entire history of the compiler.

The invariant is checked by `src/metaxu/compiler/tests/test_token_coverage.py`
against the **live** parser (PLY's own production table), not against a copy
of the grammar, so it cannot drift.

## The triage table

`lexer.TOKEN_TRIAGE` classifies every token name in `Lexer.tokens` into
exactly one bucket, with a reason:

| bucket | meaning |
| --- | --- |
| `GRAMMAR` | the token appears in at least one grammar production |
| `CONTEXTUAL` | no production names it, but the *feature* it spells is reachable another way, and the reason says how |
| `RESERVED_ONLY` | no production and no other route: the word is reserved and unusable |

Two properties make the table load-bearing rather than decorative:

* the `GRAMMAR` bucket is recomputed from the parser's production table, so
  wiring a token in (or deleting the last rule that mentions one) without
  updating the table is a test failure;
* every token outside `GRAMMAR` must have an entry in
  `lexer.RESERVED_WITHOUT_GRAMMAR`. Since no production names those tokens,
  **every** appearance of one is a syntax error, which makes
  `Parser.p_error` their complete diagnostic surface — so they answer with a
  route instead of a bare `Syntax error at 'use'`.

## Current contents

102 tokens: 97 `GRAMMAR`, 3 `CONTEXTUAL`, 2 `RESERVED_ONLY`.

### `CONTEXTUAL` — reachable through the `@` rewrite

`ONCE`, `SEPARATE`, `MANY`. No production mentions them, but the linearity
modes they name are reachable as annotations: the lexer's Pass A retags any
keyword following `@` as an `IDENTIFIER`, and `mode_annotation : AT
IDENTIFIER` accepts it. `let @once f = fn(x: int) -> int { x };` really does
bind a once-callable — `frozen_borrow_checker.check_linearity` rejects the
second call.

Writing the bare word (`let once = 1;`) is a syntax error that points at the
`@once` spelling.

### `RESERVED_ONLY` — reserved, with a route in the diagnostic

* `USE` — there is no `use` statement and never has been. The word stays
  reserved because `use std.math;` is the likely mistake; the error routes
  the reader to `import std.math;`.
* `KERNEL` — GPU kernel annotations have no syntax and no runtime
  (`docs/v1_gap_analysis.md`). Reserved so the word reads as unimplemented
  rather than as a free identifier, matching its siblings `to_device` /
  `from_device`, which parse and then raise `UnsupportedConstruct`.

### Resolved during the audit

| word | was | now | why |
| --- | --- | --- | --- |
| `impl` | `Syntax error at 'impl'` | accepted spelling of `implement` | it is the spelling `docs/ownership_and_borrowing.md` and `docs/type_system.md` use throughout, and the lexer's generic disambiguation already listed `IMPL` beside `IMPLEMENT`, so `impl<T>` was being retagged for a production that did not exist. `interface`/`trait` is the existing alias precedent. |
| `box` | reserved keyword | ordinary identifier | no `box` type in the grammar, the AST, `examples/`, `std/` or `docs/`; the documented `Box<T>` is a user-defined struct, i.e. an ordinary name. |
| `option` | reserved keyword | ordinary identifier | the surface type is `Option`; lowercase `option` only ever appeared as the module name `std.option`, which already reached the parser as an `IDENTIFIER` through the `.`-rewrite. |
| `async` | reserved keyword | ordinary identifier | no `async` syntax anywhere; concurrency is expressed with suspend effects and handlers (`docs/effects/`), not a keyword. |

## Grammar reachability (the reverse direction)

Checked on the live grammar and currently clean: every nonterminal is
reachable from the start symbol `program`, every nonterminal is referenced by
some production, no production is orphaned, and the grammar has **zero**
shift/reduce and reduce/reduce conflicts. Every terminal a production names
is one the lexer can actually produce, and every token the lexer declares is
producible (reserved word, `t_*` rule, or one of the three tokens
`_transform` synthesizes: `LGENERIC`, `RGENERIC`, `LBRACE_STRUCT`).

## Lexer silent paths closed by the audit

Each of these accepted a program and quietly meant something other than what
was written.

* **Numeric literal forms the language does not have** split into two tokens
  and the second half became an identifier in statement position — where an
  unused undefined name is dropped. `let x = 1e10; x` compiled, ran, and
  answered `1`. Likewise `0x1f` → `0 x1f`, `1_000` → `1 _000`, `123abc` →
  `123 abc`, and `1.5.2` → `FLOAT(1.5) FLOAT(0.2)`. A digit sequence glued to
  a letter, `_`, or a second dotted group is now a `LexError` naming the
  whole literal.
* **Integer literals outside i64.** `int(...)` yields an unbounded Python
  int, so a literal past the i64 range made the interpreter (exact bignum)
  and native code (`i64`, wrapping) mean different things. Literals above
  `2**63` are rejected; the bound is `2**63`, not `2**63 - 1`, because the
  most negative i64 is written as unary minus applied to that literal.
  Float literals that overflow to infinity are rejected too.
* **Unknown mode annotations.** Pass A retags *any* keyword after `@` as an
  identifier, `mode_annotation` took whatever followed, and
  `_split_mode` keeps only the names it knows and **drops the rest** — so
  `@moot` (a plausible typo for `@mut`) compiled clean and bound a shared
  value, as did the older `@mutable` spelling. `Parser.p_mode_annotation`
  now validates against `Parser.MODE_NAMES`, which a test keeps equal to the
  emitter's vocabulary.
* **The import-list keyword rewrite fired outside imports.** Its test was
  "previous token is a comma", true inside any comma-separated list, so
  `g(x, match, x)` turned the keyword `match` into a variable reference in
  exactly one argument position and nowhere else. It is now restricted to
  the token range of a real `import ... ;` statement.
* **The generic-argument scan gave up silently past 80 tokens**, falling back
  to comparison, so a long-but-legal argument list reported `Syntax error at
  '<'` — a cliff unrelated to what was wrong. The cap is gone; the scan is
  bounded by the first token that cannot appear between `<` and `>`.
* **Unterminated string literals** reported `illegal character '"'`, sending
  the reader to look at a closing quote that does not exist.
* **`LexError` diagnostics excerpted the wrong file.** `lexer.input` runs the
  whole scan, so a lex error was raised before `register_source`, and the
  caret was drawn over the previously parsed file's text.

## Known, documented, not silent

* `let @once f = fn(..) ..` used to lose its `once`: `LambdaExpression`
  initialises `.linearity` to the default `many`, and the emitter read the
  lambda before the binding. The binding's own annotation now wins; the
  lambda's linearity applies only when the binding says nothing. (Found
  while establishing that `ONCE` belongs in `CONTEXTUAL` rather than
  `RESERVED_ONLY` — if `@once` had been a no-op, the token would have had no
  route at all.)
* A function whose name is a keyword can only be reached again from a
  position where the contextual rewrite applies. That is what makes
  `fn spawn[T](..)` inside an `effect` block callable as
  `perform Thread.spawn(..)` (`examples/effect_mapping.mx`) — and what makes
  `fn if(x) { .. }` a function nothing can call. This is the same gap
  `std/README.md` item 9 records for `try`/`catch`/`some`/`none`.
* `handle` is the keyword only when an identifier follows it, because a bare
  block is a statement and `handle(x) { }` is otherwise ambiguous with a call
  followed by a block. The parenthesized-subject spelling
  (`handle (f()) { .. }`) therefore does not install a handler; it is a loud
  unknown-callee error, not a silent no-op. Write `handle f() { .. }`.
* `fn f() -> int @once { .. }` (a mode after a return type, as in
  `docs/ownership_and_borrowing.md`) has no grammar production. `@once` in a
  type position (`x: @once int`) and on a binding (`let @once f = ..`) do.
