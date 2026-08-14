"""The lexer/grammar layer has no silently-unreachable corners.

`test_hir_coverage.py` pins the AST->HIR layer: no AST node class can reach
lowering without a triage bucket, and nothing may silently vanish there.
This file pins the layer ABOVE it, which is where the worst bug of this
project actually lived: `!` was never a lexer token, and `t_error` merely
logged-and-skipped characters it did not recognise, so `!e` compiled as `e`
for the entire history of the compiler with no diagnostic anywhere.

Two directions are checked, plus the lexer's own silent paths:

1. TOKEN -> GRAMMAR. Every token in `Lexer.tokens` is triaged into exactly
   one bucket of `lexer.TOKEN_TRIAGE`:
     - GRAMMAR       the token appears in a grammar production;
     - CONTEXTUAL    no production names it, but the FEATURE is reachable by
                     another documented route (`@once` reaches the grammar
                     as `AT IDENTIFIER`);
     - RESERVED_ONLY no production and no route: the word is reserved, and
                     the parser must answer with guidance, not
                     "Syntax error at 'use'".
   The GRAMMAR bucket is recomputed here from PLY's own production table on
   the live parser, so the table cannot rot in either direction.

2. GRAMMAR -> START SYMBOL. No nonterminal may be unreachable from `program`
   and none may be defined-but-never-referenced: dead grammar is where
   precedence bugs hide.

3. LEXER ROBUSTNESS. The regression tests at the bottom pin the silent paths
   found by the audit: numeric literals that split into two tokens
   (`1e10` ran as `1`), literals outside i64 that made the interpreter and
   native code disagree, mode annotations whose name was silently dropped
   (`@moot` bound a shared value), the keyword->IDENTIFIER rewrite firing
   outside import lists, and the generic-argument scan giving up silently
   past a fixed token count.

All behavioural tests go through parsed source, per the repo convention.
"""
from __future__ import annotations

import pytest

from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import MirInterpreter, UNIT
from metaxu.compiler.pipeline import build_context_from_source, run_pipeline_ctx
from metaxu.compiler.shared_parser import shared_parser
from metaxu.errors import CompileError
from metaxu.lexer import (
    CONTEXTUAL,
    GRAMMAR,
    RESERVED_ONLY,
    RESERVED_WITHOUT_GRAMMAR,
    SYNTHESIZED_TOKENS,
    TOKEN_TRIAGE,
    Lexer,
)
from metaxu.parser import Parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse(source: str, file_path: str = "<mem>"):
    return shared_parser().parse(source, file_path)


def run_main(source: str, file_path: str = "<mem>"):
    """Full strict pipeline, then execute main(); returns (result, prints)."""
    ctx = build_context_from_source(source, file_path=file_path)
    run_pipeline_ctx(ctx)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    prints: list[str] = []
    interp.register_builtin(
        "print", lambda *a: (prints.append(" ".join(str(x) for x in a)), UNIT)[1])
    return interp.call("main", []), prints


def lex_types(source: str) -> list[str]:
    lx = Lexer()
    lx.input(source)
    out: list[str] = []
    while True:
        tok = lx.token()
        if tok is None:
            return out
        out.append(tok.type)


def _grammar():
    """(nonterminal names, terminals used, productions) of the LIVE grammar.

    Read off PLY's own production table rather than off a copy of the
    grammar text, so this cannot drift from what the parser actually
    accepts.
    """
    prods = shared_parser().parser.productions
    nonterminals = {pr.name for pr in prods}
    terminals = {s for pr in prods for s in pr.prod if s not in nonterminals}
    return nonterminals, terminals, prods


# ---------------------------------------------------------------------------
# 1. Token triage: exactly one bucket per token, and the GRAMMAR bucket is
#    recomputed from the live grammar
# ---------------------------------------------------------------------------

def test_every_declared_token_is_triaged():
    """A newly added token must be classified before it can be used.

    This is the whole point of the table: `!` was missing for years because
    nothing ever compared the token set against the grammar.
    """
    missing = sorted(set(Lexer.tokens) - set(TOKEN_TRIAGE))
    assert not missing, (
        "tokens with no triage bucket — add each to TOKEN_TRIAGE in "
        f"lexer.py: {missing}")


def test_triage_table_has_no_stale_entries():
    stale = sorted(set(TOKEN_TRIAGE) - set(Lexer.tokens))
    assert not stale, f"TOKEN_TRIAGE names tokens that no longer exist: {stale}"


def test_every_triage_entry_has_a_valid_bucket_and_a_reason():
    for name, entry in TOKEN_TRIAGE.items():
        bucket, reason = entry
        assert bucket in (GRAMMAR, CONTEXTUAL, RESERVED_ONLY), (name, bucket)
        assert reason and reason.strip(), f"{name} has an empty reason"


def test_buckets_are_disjoint_and_cover_the_table():
    by_bucket: dict[str, list[str]] = {GRAMMAR: [], CONTEXTUAL: [],
                                       RESERVED_ONLY: []}
    for name, (bucket, _r) in TOKEN_TRIAGE.items():
        by_bucket[bucket].append(name)
    assert sum(len(v) for v in by_bucket.values()) == len(TOKEN_TRIAGE)
    for bucket, names in by_bucket.items():
        assert names, f"bucket {bucket} is empty"
        assert len(names) == len(set(names))


def test_grammar_bucket_equals_the_terminals_the_grammar_actually_uses():
    """The load-bearing check, in both directions.

    Wiring a token into the grammar without moving it out of CONTEXTUAL /
    RESERVED_ONLY fails here, and so does deleting the last production that
    mentions a GRAMMAR token — which is exactly how `impl`, `box`, `option`,
    `async`, `use` and `kernel` came to be reserved words the parser could
    never accept.
    """
    _nts, terminals, _prods = _grammar()
    declared = set(Lexer.tokens)
    tabled = {n for n, (b, _r) in TOKEN_TRIAGE.items() if b == GRAMMAR}
    used = terminals & declared
    assert tabled == used, (
        f"GRAMMAR bucket is wrong. In the grammar but not the bucket: "
        f"{sorted(used - tabled)}; in the bucket but unused by the grammar: "
        f"{sorted(tabled - used)}")


def test_grammar_uses_no_terminal_the_lexer_cannot_produce():
    """A production naming a token the lexer never emits is dead grammar."""
    nonterminals, terminals, _prods = _grammar()
    phantom = sorted(terminals - set(Lexer.tokens) - {"error", "$end"})
    assert not phantom, (
        f"grammar productions use terminals no lexer rule produces: {phantom}")


def test_unreachable_tokens_all_carry_user_guidance():
    """Every token outside the grammar can ONLY ever be a syntax error, so
    `p_error` is its complete diagnostic surface: it must say something
    better than "Syntax error at 'use'"."""
    unreachable = {n for n, (b, _r) in TOKEN_TRIAGE.items() if b != GRAMMAR}
    missing = sorted(unreachable - set(RESERVED_WITHOUT_GRAMMAR))
    assert not missing, (
        "tokens with no production and no guidance in "
        f"RESERVED_WITHOUT_GRAMMAR: {missing}")
    stale = sorted(set(RESERVED_WITHOUT_GRAMMAR) - unreachable)
    assert not stale, (
        f"RESERVED_WITHOUT_GRAMMAR has guidance for reachable/absent tokens: "
        f"{stale}")
    for name, notes in RESERVED_WITHOUT_GRAMMAR.items():
        assert notes and all(n.strip() for n in notes), name


def test_every_declared_token_is_producible():
    """A token nothing can emit is as dead as a token nothing accepts."""
    reserved_types = set(Lexer.reserved.values())
    orphans = []
    for name in Lexer.tokens:
        if name in reserved_types or name in SYNTHESIZED_TOKENS:
            continue
        if not hasattr(Lexer, f"t_{name}"):
            orphans.append(name)
    assert not orphans, (
        "tokens with no lexer rule, no reserved word and no synthesis: "
        f"{orphans}")


def test_synthesized_tokens_really_are_synthesized():
    """LGENERIC/RGENERIC/LBRACE_STRUCT have no regex; `_transform` makes
    them. Pin that they are produced, so the set cannot go stale."""
    for name in SYNTHESIZED_TOKENS:
        assert not hasattr(Lexer, f"t_{name}"), f"{name} has a regex rule"
    types = lex_types("fn main() -> int { let s = Pair<int>{ a: 1 }; 0 }")
    assert "LGENERIC" in types and "RGENERIC" in types
    assert "LBRACE_STRUCT" in types


# ---------------------------------------------------------------------------
# 2. Grammar reachability (the reverse direction)
# ---------------------------------------------------------------------------

def test_no_nonterminal_is_unreachable_from_the_start_symbol():
    nonterminals, _terminals, prods = _grammar()
    by_name: dict[str, list] = {}
    for pr in prods:
        by_name.setdefault(pr.name, []).append(pr)
    start = prods[0].name          # PLY's augmented S' -> program
    seen: set[str] = set()
    stack = [start]
    while stack:
        n = stack.pop()
        if n in seen or n not in by_name:
            continue
        seen.add(n)
        for pr in by_name[n]:
            stack.extend(s for s in pr.prod if s in by_name and s not in seen)
    assert not sorted(nonterminals - seen), (
        f"nonterminals unreachable from {start}: {sorted(nonterminals - seen)}")


def test_no_nonterminal_is_defined_but_never_referenced():
    nonterminals, _terminals, prods = _grammar()
    start = prods[0].name
    referenced = {s for pr in prods for s in pr.prod if s in nonterminals}
    dead = sorted(nonterminals - referenced - {start})
    assert not dead, f"nonterminals nothing refers to: {dead}"


def test_every_production_is_reachable():
    """A production for a reachable nonterminal is itself reachable; this
    pins that no rule hangs off a nonterminal nothing derives."""
    nonterminals, _terminals, prods = _grammar()
    by_name: dict[str, list] = {}
    for pr in prods:
        by_name.setdefault(pr.name, []).append(pr)
    start = prods[0].name
    seen: set[str] = set()
    stack = [start]
    while stack:
        n = stack.pop()
        if n in seen or n not in by_name:
            continue
        seen.add(n)
        for pr in by_name[n]:
            stack.extend(s for s in pr.prod if s in by_name and s not in seen)
    unreachable = [str(pr) for pr in prods if pr.name not in seen]
    assert not unreachable, f"unreachable productions: {unreachable}"


#: Shift/reduce conflicts PLY reports for the live grammar, all resolved as
#: SHIFT.  This is a RATCHET, not an aspiration: the number may only go down
#: without a stated reason, and a new conflict must be justified in the same
#: commit that raises it.
#:
#: The whole family comes from one grammar shape: `statements : statements
#: statement` juxtaposes statements with no required separator, while a
#: statement may itself BE an expression that starts with a prefix operator
#: (`-x`, `!x`, `&x`, `@mut x`) or continue one (`a - b`).  Every conflict
#: resolves as shift, i.e. the maximal-munch reading — `a - b` is a
#: subtraction, never the two statements `a` and `-b`.  Binary `&` (bitwise
#: and) joined that family; `^`, `|`, `<<` and `>>` cannot conflict at all,
#: because none of them can START an expression.
_EXPECTED_SHIFT_REDUCE_CONFLICTS = 164


def _grammar_build_warnings() -> list[str]:
    """Every warning PLY emits while building the LIVE grammar.

    `debug=True` MATTERS: ply.yacc only reports conflict counts inside
    `if debug:` (yacc.py), and `Parser.__init__` builds with `debug=False`.
    An earlier version of this test therefore asserted "zero conflicts"
    against a logger PLY never wrote conflicts to — the assertion passed on
    a grammar carrying 154 of them.  A check that cannot fail is worse than
    no check, so the flag is forced on here.
    """
    warnings: list[str] = []

    class _Log:
        def warning(self, fmt, *a):
            warnings.append(fmt % a if a else fmt)

        error = critical = warning

        def info(self, *a, **k):
            pass

        debug = info

    import ply.yacc as yacc
    real = yacc.yacc

    def capturing(*a, **kw):
        kw["errorlog"] = _Log()
        kw["debug"] = True
        kw["debuglog"] = _Log()
        return real(*a, **kw)

    yacc.yacc = capturing
    try:
        Parser()
    finally:
        yacc.yacc = real
    return warnings


def test_the_grammar_has_no_reduce_reduce_conflicts():
    """A reduce/reduce conflict silently DISCARDS one of two rules — the
    surviving one is whichever was defined first, so the language depends on
    the order of methods in parser.py.  There must be none."""
    conflicts = [w for w in _grammar_build_warnings()
                 if "reduce/reduce" in w]
    assert not conflicts, f"reduce/reduce conflicts: {conflicts}"


def test_every_shift_reduce_conflict_resolves_as_shift():
    """Shift/reduce conflicts are real and are all resolved the same way.

    See `_EXPECTED_SHIFT_REDUCE_CONFLICTS` for why they exist.  What must
    stay true is that every one resolves as SHIFT: a conflict resolved as
    reduce would cut an expression short mid-parse, which is the silent
    misreading this file exists to catch."""
    warnings = _grammar_build_warnings()
    resolutions = [w for w in warnings
                   if "shift/reduce conflict for" in w]
    as_reduce = [w for w in resolutions if "resolved as reduce" in w]
    assert not as_reduce, f"conflicts resolved as reduce: {as_reduce}"
    assert len(resolutions) == _EXPECTED_SHIFT_REDUCE_CONFLICTS, (
        f"shift/reduce conflict count moved to {len(resolutions)} (expected "
        f"{_EXPECTED_SHIFT_REDUCE_CONFLICTS}); a new conflict needs a reason "
        "in the constant's comment, a removed one needs the number lowered")


def test_unused_token_warnings_match_the_non_grammar_buckets():
    unused = sorted(w.split("'")[1] for w in _grammar_build_warnings()
                    if "defined, but not used" in w)
    tabled = sorted(n for n, (b, _r) in TOKEN_TRIAGE.items() if b != GRAMMAR)
    assert unused == tabled, (unused, tabled)


# ---------------------------------------------------------------------------
# 3. Resolved (c) tokens: `impl` was wired up
# ---------------------------------------------------------------------------

def test_impl_block_dispatches_like_implement():
    """`impl Trait for Type { .. }` was `Syntax error at 'impl'`, even though
    it is the spelling docs/ownership_and_borrowing.md uses throughout."""
    result, _ = run_main('''
        trait Speak { fn speak(self) -> string }
        struct Dog { name: string }
        impl Speak for Dog { fn speak(self) -> string { "woof" } }
        fn main() -> string {
            let d = Dog { name: "rex" };
            d.speak()
        }
    ''')
    assert result == "woof"


def test_impl_and_implement_are_the_same_construct():
    src = '''
        trait Speak {{ fn speak(self) -> string }}
        struct Dog {{ name: string }}
        {kw} Speak for Dog {{ fn speak(self) -> string {{ "woof" }} }}
        fn main() -> string {{ let d = Dog {{ name: "rex" }}; d.speak() }}
    '''
    assert (run_main(src.format(kw="impl"))[0]
            == run_main(src.format(kw="implement"))[0] == "woof")


def test_impl_inherent_block():
    result, _ = run_main('''
        struct Counter { n: int }
        impl Counter { fn get(self) -> int { self.n } }
        fn main() -> int { let c = Counter { n: 7 }; c.get() }
    ''')
    assert result == 7


def test_impl_with_type_params_lexes_as_generics():
    """The lexer has always listed IMPL in `_GENERIC_PREV`, so `impl<T>` was
    already retagged LGENERIC/RGENERIC for a production that did not exist."""
    types = lex_types("impl<T> Box<T> { fn get(self) -> T { self.v } }")
    assert types[1] == "LGENERIC" and types[3] == "RGENERIC"
    parse('''
        struct Holder<T> { v: T }
        impl<T> Holder<T> { fn get(self) -> T { self.v } }
        fn main() -> int { 0 }
    ''')


# ---------------------------------------------------------------------------
# 3b. Resolved (c) tokens: box / option / async returned to the user
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("word", ["box", "option", "async"])
def test_dereserved_words_are_ordinary_identifiers(word):
    """These three were reserved with no production, no AST node and no
    mention in docs/ — so the word was taken from the user for nothing."""
    result, _ = run_main(f"fn main() -> int {{ let {word} = 41; {word} + 1 }}")
    assert result == 42


@pytest.mark.parametrize("word", ["box", "option", "async"])
def test_dereserved_words_work_as_function_and_field_names(word):
    result, _ = run_main(f'''
        struct S {{ {word}: int }}
        fn {word}(x: int) -> int {{ x * 2 }}
        fn main() -> int {{ let s = S {{ {word}: 20 }}; {word}(s.{word}) + 2 }}
    ''')
    assert result == 42


@pytest.mark.parametrize("word", ["box", "option", "async"])
def test_dereserved_words_are_no_longer_tokens(word):
    assert word not in Lexer.reserved
    assert lex_types(word) == ["IDENTIFIER"]


def test_std_option_still_imports_after_dereserving_option():
    """`std.option` reached the parser as IDENTIFIER via the `.`-rewrite
    before; it must still resolve now that `option` is a plain name."""
    result, _ = run_main('''
        from std.option import is_some;
        fn main() -> int { if is_some(Some(1)) { 1 } else { 0 } }
    ''')
    assert result == 1


# ---------------------------------------------------------------------------
# 3c. Resolved (c) tokens: use / kernel stay reserved but explain themselves
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source,needle", [
    ("use std.math;\nfn main() -> int { 0 }", "no statement form"),
    ("fn main() -> int { let use = 1; use }", "no statement form"),
    ("kernel fn f() -> int { 1 }\nfn main() -> int { 0 }", "not implemented"),
    ("fn main() -> int { let kernel = 1; kernel }", "not implemented"),
])
def test_reserved_only_words_get_a_route_not_a_bare_syntax_error(source, needle):
    with pytest.raises(CompileError) as exc:
        parse(source)
    rendered = str(exc.value)
    assert "ParseError" in rendered
    assert needle in rendered, rendered


def test_use_points_at_import():
    with pytest.raises(CompileError) as exc:
        parse("use std.math;\nfn main() -> int { 0 }")
    assert "import std.math;" in str(exc.value)


# ---------------------------------------------------------------------------
# 3d. CONTEXTUAL tokens: the `@` route really works, and the bare word
#     explains itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("word", ["once", "separate", "many"])
def test_bare_linearity_keyword_points_at_the_mode_annotation(word):
    with pytest.raises(CompileError) as exc:
        parse(f"fn main() -> int {{ let {word} = 1; {word} }}")
    rendered = str(exc.value)
    assert "linearity mode" in rendered
    assert f"@{word}" in rendered


@pytest.mark.parametrize("word", ["once", "separate", "many"])
def test_linearity_modes_are_reachable_as_annotations(word):
    """The documented route for the CONTEXTUAL bucket: the lexer retags a
    keyword after `@` as IDENTIFIER and `mode_annotation : AT IDENTIFIER`
    takes it."""
    types = lex_types(f"@{word}")
    assert types == ["AT", "IDENTIFIER"]
    result, _ = run_main(f"fn main() -> int {{ let @{word} x = 42; x }}")
    assert result == 42


def test_once_annotation_is_enforced_not_just_parsed():
    """`@once` must MEAN something, or ONCE belongs in RESERVED_ONLY.

    It very nearly did not: `LambdaExpression.linearity` defaults to `many`
    and the emitter read the lambda before the binding, so an annotated
    `let @once f = fn(..) -> ..` recorded "many" and the once-ness of the
    binding was dropped without a word.
    """
    from metaxu.compiler.pipeline import BorrowCheckError

    with pytest.raises(BorrowCheckError, match="invoked more than once"):
        run_main('''
            fn main() -> int {
                let @once f = fn(x: int) -> int { x + 1 };
                let a = f(1);
                let b = f(2);
                a + b
            }
        ''')


def test_once_annotation_permits_a_single_call():
    assert run_main('''
        fn main() -> int {
            let @once f = fn(x: int) -> int { x + 1 };
            f(41)
        }
    ''')[0] == 42


def test_many_annotation_permits_repeated_calls():
    assert run_main('''
        fn main() -> int {
            let @many f = fn(x: int) -> int { x + 1 };
            f(1) + f(2)
        }
    ''')[0] == 5


# ---------------------------------------------------------------------------
# 4. Lexer robustness
# ---------------------------------------------------------------------------

def test_mode_vocabulary_matches_the_constraint_emitter():
    """`p_mode_annotation` rejects names `_split_mode` would drop; if the
    emitter learns a new mode this test says so."""
    from metaxu.compiler.frozen_constraint_emitter import (
        _LINEARITY_MODES, _LOCALITY_MODES, _UNIQUENESS_ALIASES,
        _UNIQUENESS_MODES,
    )
    expected = (set(_UNIQUENESS_MODES) | set(_UNIQUENESS_ALIASES)
                | set(_LOCALITY_MODES) | set(_LINEARITY_MODES))
    assert set(Parser.MODE_NAMES) == expected


@pytest.mark.parametrize("mode", list(Parser.MODE_NAMES))
def test_every_valid_mode_name_parses(mode):
    parse(f"fn main() -> int {{ let @{mode} x = 1; x }}")


@pytest.mark.parametrize("bad", ["moot", "mutable", "wibble", "if", "fn"])
def test_unknown_mode_annotation_is_loud(bad):
    """`@moot` used to compile clean: the lexer retags ANY keyword after `@`
    as an identifier, the grammar took whatever followed, and `_split_mode`
    kept only names it knew and DROPPED the rest — so a typo for `@mut`
    silently bound a shared value."""
    with pytest.raises(CompileError) as exc:
        parse(f"fn main() -> int {{ let @{bad} x = 1; x }}")
    rendered = str(exc.value)
    assert f"unknown mode '@{bad}'" in rendered
    assert "@mut" in rendered            # the valid list is offered


@pytest.mark.parametrize("literal,note", [
    ("1e10", "exponent"),
    ("0x1f", "hex"),
    ("0b101", "binary"),
    ("1_000", "digit-separator"),
    ("123abc", "decimal integer"),
    ("1.5.2", "float literals"),
])
def test_unsupported_numeric_literal_forms_are_loud(literal, note):
    """Each of these used to split into two tokens whose second half became
    an identifier in statement position — where an unused undefined name is
    dropped — so `let x = 1e10; x` compiled, ran, and answered 1."""
    with pytest.raises(CompileError) as exc:
        parse(f"fn main() -> int {{ let x = {literal}; 0 }}")
    rendered = str(exc.value)
    assert "LexError" in rendered
    assert "invalid numeric literal" in rendered
    assert literal in rendered
    assert note in rendered


def test_supported_numeric_literals_still_lex():
    assert lex_types("42") == ["NUMBER"]
    assert lex_types("3.14") == ["FLOAT"]
    assert lex_types(".5") == ["FLOAT"]
    assert lex_types("1..5") == ["NUMBER", "DOTDOT", "NUMBER"]
    assert lex_types("1.to_string()") == [
        "NUMBER", "DOT", "IDENTIFIER", "LPAREN", "RPAREN"]
    assert run_main("fn main() -> int { 40 + 2 }")[0] == 42


def test_integer_literal_beyond_i64_is_loud():
    """The interpreter answers with Python's exact bignum while every `i64`
    in codegen_llvm wraps: a literal out of range made the two backends mean
    different things, which is the divergence the differential tests exist
    to catch."""
    with pytest.raises(CompileError) as exc:
        parse("fn main() -> int { 99999999999999999999 }")
    assert "out of range for a 64-bit int" in str(exc.value)


def test_i64_boundary_literals_are_accepted():
    # 2**63 is allowed so the most negative i64 can be written at all
    # (`-9223372036854775808` is unary minus applied to that literal).
    parse("fn main() -> int { 9223372036854775807 }")
    parse("fn main() -> int { -9223372036854775808 }")
    with pytest.raises(CompileError):
        parse("fn main() -> int { 9223372036854775809 }")


def test_float_literal_beyond_f64_is_loud():
    huge = "1" * 400 + ".0"
    with pytest.raises(CompileError) as exc:
        parse(f"fn main() -> float {{ {huge} }}")
    assert "out of range for f64" in str(exc.value)


def test_unterminated_string_says_so():
    """It used to report `illegal character '"'`, which sends the reader
    looking at the closing quote of a string that has none."""
    with pytest.raises(CompileError) as exc:
        parse('fn main() -> int { let s = "abc; 0 }')
    assert "unterminated string literal" in str(exc.value)


def test_illegal_character_is_still_loud():
    with pytest.raises(CompileError) as exc:
        parse("fn main() -> int { let x = 1 $ 2; x }")
    assert "illegal character" in str(exc.value)


def test_lex_errors_excerpt_their_own_source():
    """`lexer.input` runs the whole scan, so a LexError was raised BEFORE
    `register_source`, and the caret was drawn over the previously parsed
    file's text."""
    parse("fn main() -> int { 1 }", "first.mx")
    with pytest.raises(CompileError) as exc:
        parse("fn other() -> int { 1e10 }", "second.mx")
    rendered = str(exc.value)
    assert "second.mx" in rendered
    assert "fn other()" in rendered
    assert "fn main()" not in rendered


# --- the keyword -> IDENTIFIER rewrite passes ------------------------------

def test_import_lists_may_shadow_keywords():
    types = lex_types("from std.effects import Effect, handle, perform;")
    assert "HANDLE" not in types and "PERFORM" not in types


def test_the_import_rewrite_does_not_leak_into_other_comma_lists():
    """The rewrite used to fire on "previous token is a comma", which is
    true inside ANY comma-separated list: `g(x, match, x)` quietly turned
    the keyword `match` into a variable reference in exactly one argument
    position and nowhere else."""
    types = lex_types("g(x, match, x)")
    assert "MATCH" in types
    with pytest.raises(CompileError) as exc:
        parse("fn g(a: int, b: int, c: int) -> int { a }\n"
              "fn main() -> int { let x = 1; g(x, match, x) }")
    assert "ParseError" in str(exc.value)
    # ... while the import list it exists for still works.
    assert "MATCH" not in lex_types("from m import a, match, b;")


def test_keyword_after_dot_or_at_is_still_a_name():
    assert lex_types("a.match") == ["IDENTIFIER", "DOT", "IDENTIFIER"]
    assert lex_types("@const") == ["AT", "IDENTIFIER"]


def test_effect_operation_may_be_named_with_a_keyword():
    """The documented reason the `fn <keyword>` rewrite exists: an effect
    operation named `spawn` (examples/effect_mapping.mx), reached again
    through the `.name` position that `perform E.op` uses."""
    result, _ = run_main('''
        effect Thread {
            fn spawn(n: int) -> int;
        }
        fn body() -> int performs Thread { perform Thread.spawn(20) }
        fn main() -> int {
            handle body() {
                perform Thread.spawn(n) => { resume(n + 22) }
            }
        }
    ''')
    assert result == 42


def test_handle_with_a_parenthesized_subject_is_loud_not_a_silent_no_op():
    """`handle` is a keyword only when an identifier follows, because a bare
    block is a statement and `handle(x) { }` is otherwise ambiguous with a
    call. The parenthesized-subject spelling therefore does NOT install a
    handler — and must not do so silently.

    It parses as a CALL of a function named `handle`, which no program
    declares, so name resolution rejects it at compile time
    (docs/name_resolution.md). It used to survive to run time and die there
    with `Unknown callee: 'handle'`; the interpreter still raises that if a
    call ever reaches it unresolved (defence in depth), but the front end no
    longer lets this one through."""
    from metaxu.compiler.frozen_borrow_checker import TypeCheckError

    assert "HANDLE" not in lex_types("handle (body()) { }")
    with pytest.raises(TypeCheckError, match="undefined function 'handle'"):
        run_main('''
            fn body() -> int { 1 }
            fn main() -> int { handle (body()) { } }
        ''')


# --- generic angle-bracket disambiguation ----------------------------------

def test_comparisons_are_not_mistaken_for_generics():
    assert lex_types("if a < b { 1 } else { 0 }").count("LESS") == 1
    assert "LGENERIC" not in lex_types("f(a < b, c > d)")


def test_generic_call_is_retagged():
    types = lex_types("identity<int>(3)")
    assert "LGENERIC" in types and "RGENERIC" in types


def test_long_generic_argument_lists_no_longer_fall_off_a_cliff():
    """The scan used to stop after 80 tokens and SILENTLY fall back to
    comparison, so a long-but-legal list reported `Syntax error at '<'`."""
    params = ", ".join(f"T{i}" for i in range(45))
    args = ", ".join("int" for _ in range(45))
    types = lex_types(f"fn f<{params}>(x: int) -> int {{ x }}")
    assert "LGENERIC" in types and "RGENERIC" in types
    parse(f"fn f<{params}>(x: int) -> int {{ x }}\n"
          f"fn main() -> int {{ f<{args}>(1) }}")


def test_generic_scan_still_stops_at_a_token_that_cannot_be_inside():
    """Removing the fixed cap must not make the scan swallow a whole file:
    the first token that cannot appear between `<` and `>` still ends it."""
    types = lex_types("if a < b { let c = 1; } else { let d = 2; }")
    assert "LGENERIC" not in types and "RGENERIC" not in types
