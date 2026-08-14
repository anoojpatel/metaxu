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

106 tokens: 101 `GRAMMAR`, 3 `CONTEXTUAL`, 2 `RESERVED_ONLY`.

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

## The bitwise operators

`&`, `|`, `^`, `~`, `<<`, `>>` — the C/Rust spelling, with **Rust's
precedence**, not C's: shifts bind tighter than `&`, `&` tighter than `^`,
`^` tighter than `|`, and all four bind tighter than the comparisons, so
`flags & MASK == 0` groups as `(flags & MASK) == 0`. They are Int-only
(`frozen_constraint_emitter` classes both operands and the result `Int`, so
a Float or String operand is a compile error through the same conflict
detection that rejects `1 + "a"`), i64 two's complement on both engines,
`>>` is arithmetic, and a shift count outside `0..63` is a **loud error**
on both engines rather than LLVM poison / an unbounded Python shift
(natively: `mx_shift_check` aborts before the `shl`/`ashr`).

Four of the six spellings are *strict extensions* — nothing that compiled
before changes meaning:

* `^` and `~` were illegal characters, i.e. a `LexError`;
* `|` in expression position was `Syntax error at '|'`;
* `<<` and `>>` are **synthesized by Pass D** of `Lexer._transform` from an
  *adjacent* `LESS LESS` / `GREATER GREATER` pair that Pass B did not claim
  for a generic argument list. They are deliberately not lexer regexes: a
  `t_SHR = r'>>'` rule would have eaten the two closing brackets of
  `Vec<Vec<int>>` before Pass B ever saw them — the C++ nested-generics
  bug, and precisely the silent misparse this document exists to prevent.
  Adjacency is required, so `a > > b` stays the syntax error it was.
  Pass B additionally refuses to *start* a generic scan at an adjacent
  `<<`: a type argument list can never open with `<`, and without that
  refusal `a << b >> (c)` (balanced angles, `(` in the follow set) would
  have been retagged as generic arguments. It refuses to *close* one on
  an adjacent `>>` for the mirror-image reason: a `>` that would close the
  OUTERMOST bracket with another `>` glued to it is the shift operator.
  Without that half `a < b >> c` matched `a<b>` and left a stray `>`, so a
  plain shift failed with an unrelated "uncalled generic instantiation"
  (`g(a < b, c >> d)` broke the same way). Nesting is unaffected because it
  is counted with `depth`: `Map<String, Vec<Int>>` closes the outermost
  bracket on the *second* `>`, where the guard cannot fire. `GREATER` was
  dropped from the follow set at the same time — it was justified there as
  "nesting `>>`", which `depth` already handles, and a spaced `X<T> > y` is
  an uncalled generic instantiation with no value representation anyway.

Only binary `&` overlaps something the old grammar accepted: two juxtaposed
statements, `a` followed by `&b`. That reading is gone, exactly as `a` `-b`
has always read as subtraction rather than as a statement pair — the same
grammar shape and the same shift/reduce resolution (see below). A borrow in
statement position still works after a `;`.

`~e` lowers to the `bnot` builtin, joining `-e` (`neg`) and `!e` (`not`);
the frozen AST now carries a `UnaryOperation`'s operator, which it did not
before, so a checker can tell the three apart.

## Tuples added productions but no tokens

`(a, b)` already lexed and parsed — `primary_expression : LPAREN
expression COMMA expression_seq RPAREN` has built a `TupleLiteral` for as
long as the grammar has existed; what it lacked was a lowering, so every
non-unit tuple raised. Making tuples real therefore added **no token**:
`LPAREN`, `COMMA` and `RPAREN` were already in the `GRAMMAR` bucket, and
the triage table is unchanged.

Three productions were added, and none of them adds a conflict (the
shift/reduce count below is unmoved at 164, reduce/reduce still zero):

* `let_statement : LET binding_prefix LPAREN identifier_seq RPAREN EQUALS
  expression` and `for_statement : FOR LPAREN identifier_seq RPAREN IN
  expression LBRACE statement_list RBRACE`. Both share their whole prefix
  with the existing `IDENTIFIER` forms up to the token that distinguishes
  them (`LPAREN` vs `IDENTIFIER`), which one lookahead settles.
* `type_postfix : LPAREN type_expression COMMA type_list RPAREN` — the
  tuple TYPE. Nothing else in type position starts with `LPAREN` except
  `fn (..) -> ..`, which is preceded by `FN`.

The **1-tuple** is the grammar hazard this feature had to answer, because
`(e)` is parenthesized grouping and has been since the beginning. The rule
is that Metaxu has no 1-tuples at all: `(e)` stays grouping, `(e,)` stays
a syntax error (the alternative — accepting Rust's spelling — would mean
inventing a `__tuple1` layout no other pass understands), `()` stays the
unit value, and the two shapes that would need one are rejected by name:
`let (a) = e` reports "a 1-element tuple, which does not exist" and a
`()` *pattern* is a loud `UnsupportedConstruct` rather than a match-
anything arm. The other two spellings tuples might have claimed keep their
old meanings, and both are pinned by tests: `f(a, b) for (a, b) in (xs,
ys)` is still lockstep zip iteration over two sequences, and `(a, b) -> e`
is still a two-parameter lambda.

## Grammar reachability (the reverse direction)

Checked on the live grammar and currently clean: every nonterminal is
reachable from the start symbol `program`, every nonterminal is referenced by
some production, no production is orphaned, and the grammar has **zero
reduce/reduce** conflicts. Every terminal a production names
is one the lexer can actually produce, and every token the lexer declares is
producible (reserved word, `t_*` rule, or one of the five tokens
`_transform` synthesizes: `LGENERIC`, `RGENERIC`, `LBRACE_STRUCT`, `SHL`,
`SHR`).

### The conflict check was vacuous (found during the bitwise work)

`test_the_grammar_has_no_parser_conflicts` asserted that PLY reported no
conflicts — but `ply.yacc` only reports conflict counts inside `if debug:`,
and `Parser.__init__` builds with `debug=False`. The assertion therefore
ran against a logger PLY never wrote conflicts to, and passed on a grammar
carrying **154 shift/reduce conflicts**. A check that cannot fail is worse
than no check.

The test now forces `debug=True` and asserts what is actually true:

* **zero reduce/reduce conflicts** (one of those silently discards a rule,
  making the language depend on the order of methods in `parser.py`);
* **every shift/reduce conflict resolves as shift**, with the count pinned
  as a ratchet (`_EXPECTED_SHIFT_REDUCE_CONFLICTS`, 164 after the bitwise
  levels were added).

They all come from one shape: `statements : statements statement` juxtaposes
statements with no required separator, while a statement may itself be an
expression that *starts* with a prefix operator (`-x`, `!x`, `&x`, `@mut x`,
`~x`) or *continues* one (`a - b`). Shift is the maximal-munch reading —
`a - b` is a subtraction, never the two statements `a` and `-b` — which is
the pre-existing, documented behaviour of `-`; binary `&` now joins it.
`^`, `|`, `<<` and `>>` add none, because none of them can start an
expression.

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
  and native code (`i64`, wrapping) mean different things. The largest
  positive literal is `2**63 - 1`. The bound used to be `2**63` so that the
  most negative i64 — unary minus applied to the literal
  `9223372036854775808` — stayed writable, but that let the *bare* positive
  `9223372036854775808` through and reintroduced the very divergence
  (interpreter: the exact bignum; `emit_llvm`: "integer constant … outside
  i64 range", i.e. a demoted placeholder). Both halves are solved by making
  the sign part of the literal: Pass 0 of `Lexer._transform`
  (`_fold_most_negative_int`) accepts `2**63` **only** as the operand of a
  unary minus and folds the pair into the single constant `-2**63`, so the
  backends see an in-range i64 and every other occurrence — including
  `x - 9223372036854775808` and the doubled `--9223372036854775808` — is a
  loud `LexError`. Folding also removed the *negative* form's demotion:
  `-9223372036854775808` used to reach codegen_llvm as
  `neg(const 9223372036854775808)`. Float literals that overflow to
  infinity are rejected too.
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
  (`handle (f()) { .. }`) therefore does not install a handler; it parses as
  a call of a function named `handle`, which nothing declares, so it is now
  a compile-time `undefined function 'handle'`
  (`docs/name_resolution.md` — it used to reach the interpreter and die
  there with `Unknown callee`). Either way, never a silent no-op. Write
  `handle f() { .. }`.
* `fn f() -> int @once { .. }` (a mode after a return type, as in
  `docs/ownership_and_borrowing.md`) has no grammar production. `@once` in a
  type position (`x: @once int`) and on a binding (`let @once f = ..`) do.
