import ply.lex as lex
from metaxu.errors import CompileError, SourceLocation, get_source_context
from typing import List
import logging

logger = logging.getLogger(__name__)


class Lexer:
    # A string containing ignored characters (spaces, tabs, carriage returns)
    t_ignore = ' \t\r'

    #: Token most recently handed to the parser (class-level default so the
    #: lineno/lexpos properties are safe to read during lex.lex() setup,
    #: which walks dir(self) before __init__ has run).
    current_token = None
    line_starts = [0]

    # Keywords
    reserved = {
        'if': 'IF',
        'else': 'ELSE',
        'while': 'WHILE',
        'for': 'FOR',
        'fn': 'FN',
        'return': 'RETURN',
        'struct': 'STRUCT',
        'enum': 'ENUM',
        'match': 'MATCH',
        'effect': 'EFFECT',
        'handle': 'HANDLE',
        'perform': 'PERFORM',
        'performs': 'PERFORMS',
        'resume': 'RESUME',
        'with': 'WITH',
        'in': 'IN',
        'let': 'LET',
        'mut': 'MUT',
        'type': 'TYPE',
        'extern': 'EXTERN',
        'const': 'CONST',
        'move': 'MOVE',
        'exclave': 'EXCLAVE',
        'once': 'ONCE',
        'spawn': 'SPAWN',
        'kernel': 'KERNEL',
        'to_device': 'TO_DEVICE',
        'from_device': 'FROM_DEVICE',
        'print': 'PRINT',
        # Interface / trait / implementation keywords
        'interface': 'INTERFACE',
        'trait': 'TRAIT',
        'impl': 'IMPL',
        'implement': 'IMPLEMENT',
        'implements': 'IMPLEMENTS',
        'where': 'WHERE',
        'extends': 'EXTENDS',
        # Mode-related keywords
        'unique': 'UNIQUE',
        'exclusive': 'EXCLUSIVE',
        'separate': 'SEPARATE',
        'many': 'MANY',
        'borrow': 'BORROW',
        # Module-related keywords
        'import': 'IMPORT',
        'from': 'FROM',
        'module': 'MODULE',
        'export': 'EXPORT',
        'use': 'USE',
        'public': 'PUBLIC',
        'private': 'PRIVATE',
        'protected': 'PROTECTED',
        'visibility': 'VISIBILITY',
        'comptime': 'COMPTIME',
        'some': 'SOME',
        'none': 'NONE',
        # NOTE: `box`, `option` and `async` were reserved here for years with
        # no grammar production, no AST node and no mention in docs/, so
        # `let box = 1` was a syntax error for a keyword the language does
        # not have.  They were removed during the token reachability audit
        # (docs/token_reachability.md); the surface types are the ordinary
        # identifiers `Box`/`Option`, and concurrency is expressed with
        # effect handlers rather than an `async` keyword.
        'vector': 'VECTOR',
        'unsafe': 'UNSAFE',
        'void': 'VOID',
        'size_t': 'SIZE_T',
        'as': 'AS',
        'try': 'TRY',
        'catch': 'CATCH',
    }

    # List of token names
    tokens = [
        'IDENTIFIER', 'NUMBER', 'FLOAT', 'STRING', 'FSTRING',
        'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'MOD',
        'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'LBRACKET', 'RBRACKET',
        'EQUALS', 'SEMICOLON', 'COLON', 'COMMA', 'DOT', 'DOTDOT', 'TRIPLE_DOT',
        'DOUBLECOLON', 'ARROW', 'FATARROW', 'BACKSLASH', 'AT', 'AMPERSAND',
        'PIPE', 'OROR', 'ANDAND',
        'LESS', 'GREATER', 'LESSEQUAL', 'GREATEREQUAL', 'EQUALEQUAL', 'NOTEQUAL',
        'NOT', 'CARET', 'TILDE',
        # Synthesized by the token-stream disambiguation filter (never produced
        # directly by a regex): generic type argument brackets, the opening
        # brace of a struct literal, and the two shift operators (Pass D
        # merges an ADJACENT pair of LESS/GREATER that Pass B did not claim
        # for a generic argument list — a `>>` regex would have eaten the
        # closing brackets of `Vec<Vec<int>>`).
        'LGENERIC', 'RGENERIC', 'LBRACE_STRUCT', 'SHL', 'SHR',
    ] + list(set(reserved.values()))

    # Regular expression rules for simple tokens
    t_PLUS = r'\+'
    t_MINUS = r'-'
    t_TIMES = r'\*'
    t_DIVIDE = r'/'
    t_MOD = r'%'
    t_EQUALS = r'='
    t_EQUALEQUAL = r'=='
    t_NOTEQUAL = r'!='
    # Logical negation.  `!=` out-ranks it automatically (PLY orders string
    # token rules by decreasing regex length).  Without this rule `!` was an
    # ILLEGAL CHARACTER that t_error merely warned about and skipped, so
    # `!cond` silently compiled as `cond` — with the wrong answer and no
    # diagnostic — even though `hir` has always lowered `!e` to
    # `__builtin$not` and docs/name_precedence.md documents it.
    t_NOT = r'!'
    t_LESSEQUAL = r'<='
    t_GREATEREQUAL = r'>='
    t_LESS = r'<'
    t_GREATER = r'>'
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_LBRACE = r'\{'
    t_RBRACE = r'\}'
    t_LBRACKET = r'\['
    t_RBRACKET = r'\]'
    t_SEMICOLON = r';'
    t_COLON = r':'
    t_COMMA = r','
    t_DOT = r'\.'
    t_DOTDOT = r'\.\.'
    t_TRIPLE_DOT = r'\.\.\.'
    t_DOUBLECOLON = r'::'
    t_ARROW = r'->'
    t_FATARROW = r'=>'
    t_BACKSLASH = r'\\'  # Used in function type annotations (fn\(T) -> U)
    t_AT = r'@'
    # `&&` must out-rank `&` (PLY orders string token rules by decreasing
    # regex length, so this is automatic) — otherwise `a && b` lexes as two
    # borrows.
    t_ANDAND = r'&&'
    t_AMPERSAND = r'&'
    t_OROR = r'\|\|'
    t_PIPE = r'\|'
    # Bitwise xor and complement.  Both characters were ILLEGAL until now
    # (t_error rejects them), so adding the tokens cannot change the meaning
    # of any program that compiles today.  `<<`/`>>` are NOT regex rules —
    # see SHL/SHR in `tokens` and `_transform`'s Pass D.
    t_CARET = r'\^'
    t_TILDE = r'~'

    # Comments: both '#' and '//' styles
    def t_COMMENT(self, t):
        r'\#.*|//.*'
        pass

    #: Metaxu's `int` is a signed 64-bit machine integer everywhere below the
    #: front end (MIR, the interpreter and every `i64` in codegen_llvm), but
    #: `int(...)` here yields an unbounded Python int.  A literal past the
    #: i64 range therefore used to sail through the front end and then MEAN
    #: DIFFERENT THINGS in the two backends — the interpreter answered with
    #: the exact bignum while native code wrapped — which is precisely the
    #: interpreter/native divergence the differential tests exist to prevent.
    #: The bound is `2**63` rather than `2**63 - 1` because the most negative
    #: i64 is written `-9223372036854775808`, i.e. unary minus applied to the
    #: literal `9223372036854775808`.
    _INT_LITERAL_MAX = 2 ** 63

    def _reject_numeric_junk(self, t) -> None:
        """Reject a numeric literal glued to a letter, `_`, or a second dot.

        Metaxu has exactly two numeric literal forms: decimal integers and
        `digits.digits` floats.  There is no exponent, hex, binary or
        digit-separator syntax.  Without this check the lexer split the
        unsupported forms into two tokens and the second half QUIETLY
        BECAME SOMETHING ELSE:

            1e10     ->  NUMBER(1)  IDENTIFIER(e10)   # exponent vanished:
                                                      # `let x = 1e10;` bound 1
            0x1f     ->  NUMBER(0)  IDENTIFIER(x1f)
            1_000    ->  NUMBER(1)  IDENTIFIER(_000)
            1.5.2    ->  FLOAT(1.5) FLOAT(0.2)

        The `IDENTIFIER` half landed in statement position, where an unused
        undefined name is dropped, so `1e10` compiled and ran as `1`.  This
        is the same shape of silent seam as `!e` compiling as `e`.
        """
        data = t.lexer.lexdata
        end = t.lexer.lexpos          # PLY has already advanced past the match
        n = len(data)
        bad = ''
        if end < n and (data[end].isalpha() or data[end] == '_'):
            j = end
            while j < n and (data[j].isalnum() or data[j] == '_'):
                j += 1
            bad = data[end:j]
        elif (end + 1 < n and data[end] == '.' and data[end + 1].isdigit()
                and '.' in t.value):
            j = end + 1
            while j < n and data[j].isdigit():
                j += 1
            bad = data[end:j]
        if not bad:
            return
        self._numeric_error(
            t, f"invalid numeric literal {t.value + bad!r}",
            ["Metaxu has decimal integer literals (`42`) and `d.d` float "
             "literals (`3.14`) only",
             "There is no exponent (`1e10`), hex (`0x1f`), binary (`0b1`) or "
             "digit-separator (`1_000`) form"])

    def _numeric_error(self, t, message: str, notes):
        line_start = self.line_starts[min(t.lineno - 1, len(self.line_starts) - 1)]
        raise CompileError(
            message=message,
            error_type="LexError",
            location=SourceLocation(
                file=self.source_file, line=t.lineno,
                column=t.lexpos - line_start + 1),
            notes=list(notes),
        )

    # NOTE: function rules are matched in definition order; FLOAT must come
    # before NUMBER so that "3.14" lexes as a single float.
    def t_FLOAT(self, t):
        r'\d+\.\d+|\.\d+'
        self._reject_numeric_junk(t)
        t.endlexpos = t.lexpos + len(t.value)
        value = float(t.value)
        if value in (float('inf'), float('-inf')):
            self._numeric_error(
                t, f"float literal {t.value!r} is out of range for f64",
                ["The largest finite f64 is about 1.8e308"])
        t.value = value
        return t

    def t_NUMBER(self, t):
        r'\d+'
        self._reject_numeric_junk(t)
        t.endlexpos = t.lexpos + len(t.value)
        value = int(t.value)
        if value > self._INT_LITERAL_MAX:
            self._numeric_error(
                t, f"integer literal {t.value} is out of range for a 64-bit int",
                [f"Metaxu's `int` is a signed 64-bit integer: "
                 f"-{self._INT_LITERAL_MAX} .. {self._INT_LITERAL_MAX - 1}"])
        t.value = value
        return t

    #: Recognised backslash escapes in string and f-string literals.
    #: Anything else after a backslash is a LOUD error rather than a
    #: silently-kept backslash: `"\d"` is far more likely a typo than an
    #: intended two-character string, and a silent pass-through is exactly
    #: the kind of seam this compiler refuses elsewhere.
    _ESCAPES = {
        'n': '\n', 't': '\t', 'r': '\r', '0': '\0',
        '\\': '\\', '"': '"', "'": "'",
    }

    def _decode_escapes(self, raw: str, t) -> str:
        """Interpret backslash escapes in a string literal's inner text.

        Before this existed the lexer kept the raw characters, so `"a\\nb"`
        was the four-character string `a`, `\\`, `n`, `b` — it printed as
        `a\\nb` with no diagnostic, and `"\\""` could not be written at all
        (the old `"[^"]*"` pattern stopped at the escaped quote).
        """
        if '\\' not in raw:
            return raw
        out: List[str] = []
        i = 0
        n = len(raw)
        while i < n:
            ch = raw[i]
            if ch != '\\':
                out.append(ch)
                i += 1
                continue
            if i + 1 >= n:
                self._string_error(t, "string literal ends with a lone backslash")
            nxt = raw[i + 1]
            decoded = self._ESCAPES.get(nxt)
            if decoded is None:
                self._string_error(
                    t, f"unknown escape sequence '\\{nxt}' in string literal")
            out.append(decoded)
            i += 2
        return "".join(out)

    def _string_error(self, t, message: str):
        line_start = self.line_starts[min(t.lineno - 1, len(self.line_starts) - 1)]
        raise CompileError(
            message=message,
            error_type="LexError",
            location=SourceLocation(
                file=self.source_file, line=t.lineno,
                column=t.lexpos - line_start + 1),
            notes=["Valid escapes are \\n \\t \\r \\0 \\\\ \\\" \\'"],
        )

    def t_FSTRING(self, t):
        r'f"([^"\\]|\\.)*"'
        t.endlexpos = t.lexpos + len(t.value)
        # Tuple with (value, type)
        t.value = (self._decode_escapes(t.value[2:-1], t), 'string')
        return t

    def t_STRING(self, t):
        r'"([^"\\]|\\.)*"'
        t.endlexpos = t.lexpos + len(t.value)
        # Tuple with (value, type)
        t.value = (self._decode_escapes(t.value[1:-1], t), 'string')
        return t

    def t_IDENTIFIER(self, t):
        r'[a-zA-Z_][a-zA-Z_0-9]*'
        t.type = self.reserved.get(t.value, 'IDENTIFIER')
        return t

    # Define a rule so we can track line numbers
    def t_newline(self, t):
        r'\n+'
        # Track the position after each newline
        for i in range(len(t.value)):
            self.line_starts.append(t.lexpos + i + 1)
        t.lexer.lineno += len(t.value)

    # Error handling rule
    def t_error(self, t):
        # LOUD, not skipped.  This used to log a warning and `skip(1)`, which
        # made every unlexable character vanish from the token stream: `!cond`
        # (before `!` had a token) compiled as `cond`, with the wrong answer
        # and nothing on stderr that a test would notice.  A character the
        # lexer does not know is a compile error.
        line_start = self.line_starts[min(t.lineno - 1, len(self.line_starts) - 1)]
        column = t.lexpos - line_start + 1
        # A well-formed string always matches t_STRING/t_FSTRING, so a `"`
        # that reaches the error rule is an opening quote with no closing
        # one.  Saying "illegal character '\"'" for that sends the reader
        # looking at the wrong thing entirely.
        if t.value[0] == '"':
            message = "unterminated string literal"
            notes = ["Close it with a matching '\"' on the same line",
                     "Metaxu string literals do not span lines"]
        else:
            message = f"illegal character {t.value[0]!r}"
            notes = ["Remove it, or quote it inside a string literal"]
        raise CompileError(
            message=message,
            error_type="LexError",
            location=SourceLocation(
                file=self.source_file, line=t.lineno, column=column),
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Token-stream disambiguation
    # ------------------------------------------------------------------

    #: Keyword token types (values of ``reserved``)
    _KEYWORD_TYPES = frozenset(reserved.values())

    #: Tokens permitted inside a generic argument list ``< ... >``.
    _GENERIC_INSIDE = frozenset({
        'IDENTIFIER', 'NUMBER', 'COMMA', 'LBRACKET', 'RBRACKET',
        'LESS', 'GREATER', 'CONST', 'COLON', 'DOT', 'VECTOR', 'MUT',
    })

    #: Tokens that may directly precede a ``<`` that opens generic args.
    _GENERIC_PREV = frozenset({'IDENTIFIER', 'IMPLEMENT', 'IMPL'})

    #: Openers after which ``<`` is unambiguously a generic-parameter list
    #: (``implement<T> ...`` can never be a comparison), so the follow-set
    #: check below is skipped.
    _GENERIC_PREV_UNAMBIGUOUS = frozenset({'IMPLEMENT', 'IMPL'})

    #: Tokens that may directly follow the closing ``>`` of a generic
    #: argument list (call ``identity<Int>(..)``, struct literal
    #: ``Stack<Int>{..}``, type positions ``: Stack<Int> =``, ``-> Opt<T>``,
    #: trait clauses ``where``/``with``/``for``, nesting ``>>``, etc.).
    #: A token outside this set — in particular an identifier or a literal,
    #: as in ``f(a < b, c > d)`` — means the angle brackets were comparison
    #: operators, so they are left as LESS/GREATER.  When ambiguous we
    #: prefer comparison: expression-level generic instantiation is rare
    #: and is virtually always followed by ``(`` or ``{``.
    _GENERIC_FOLLOW = frozenset({
        'LPAREN', 'LBRACE', 'LBRACE_STRUCT', 'RPAREN', 'RBRACKET',
        'COMMA', 'SEMICOLON', 'COLON', 'DOT', 'DOUBLECOLON', 'EQUALS',
        'ARROW', 'GREATER', 'RGENERIC', 'WHERE', 'WITH', 'FOR',
    })

    # NOTE: there is deliberately no scan limit here.  An earlier
    # `_GENERIC_SCAN_LIMIT = 80` made the scan give up SILENTLY once a
    # candidate argument list ran past 80 tokens, so a long-but-legal
    # generic list (`fn f<T0, ..., T44>(..)`) fell back to comparison and
    # reported `Syntax error at '<'` — a cliff with no relation to what was
    # wrong.  The scan is already bounded by the first token that cannot
    # appear inside `< ... >` (`_GENERIC_INSIDE`), which in real source is a
    # handful of tokens away, so the cap bought nothing.

    def _import_statement_spans(self, toks):
        """Index set covering every `import ... ;` statement's token range.

        Pass A lets an import list shadow keywords
        (`from std.effects import Effect, handle, perform;`).  The old test
        for "am I in an import list?" was `previous token is COMMA`, which
        is true inside ANY comma-separated list: `g(x, match, x)` quietly
        turned the keyword `match` into a variable reference in a plain call
        argument, so the same word was an identifier in one argument
        position and a syntax error in every other.  Restricting the rewrite
        to the real statement keeps the import behaviour and drops the
        accidental one.
        """
        inside: set[int] = set()
        i = 0
        n = len(toks)
        while i < n:
            if toks[i].type == 'IMPORT':
                j = i
                while j < n and toks[j].type != 'SEMICOLON':
                    inside.add(j)
                    j += 1
                i = j
            i += 1
        return inside

    def _transform(self, toks):
        """Rewrite the raw token list to resolve context-sensitive ambiguity.

        Pass A: keywords used as plain names (after '.', '@', 'fn', and a
                contextual rule for 'handle') become IDENTIFIER tokens.
        Pass B: '<' ... '>' pairs that enclose type arguments become
                LGENERIC/RGENERIC so the grammar can distinguish generics
                from comparisons.
        Pass C: a '{' that opens a struct literal (previous token is a name
                or a closing type-argument bracket and the next tokens look
                like `field :`) becomes LBRACE_STRUCT.
        Pass D: an ADJACENT pair of LESS/LESS or GREATER/GREATER that Pass B
                did not claim for a generic argument list becomes SHL/SHR.
        """
        # --- Pass A: contextual keywords -------------------------------
        import_span = self._import_statement_spans(toks)
        for i, tok in enumerate(toks):
            prev = toks[i - 1] if i > 0 else None
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            if tok.type in self._KEYWORD_TYPES and prev is not None and \
                    prev.type in ('DOT', 'DOTDOT', 'TRIPLE_DOT', 'AT', 'FN'):
                # Member access (thread.spawn), relative paths (..vector),
                # mode names (@mut/@const), and function names
                # (fn spawn[...]) may reuse keywords.
                #
                # After FN this DEFINES a name that only the `.name` position
                # can reach again — which is exactly what an effect operation
                # `fn spawn[T](..)` called as `perform Thread.spawn(..)`
                # needs (examples/effect_mapping.mx), and what makes
                # `fn if(x) {..}` a function nothing can call.  See
                # docs/token_reachability.md; @-mode names are validated in
                # the parser (Parser.p_mode_annotation), so `@moot` is a
                # loud error rather than a silently dropped mode.
                tok.type = 'IDENTIFIER'
            elif tok.type in self._KEYWORD_TYPES and i in import_span and \
                    prev is not None and prev.type in ('COMMA', 'IMPORT') and \
                    nxt is not None and nxt.type in ('COMMA', 'SEMICOLON') and \
                    tok.type != 'HANDLE':
                # Imported names may shadow keywords:
                #   from std.effects import Effect, handle, perform, resume;
                tok.type = 'IDENTIFIER'
            elif tok.type == 'HANDLE':
                # 'handle' is only the keyword when introducing a handler
                # (handle Effect with { ... } / handle f() { ... }); in other
                # positions it is an ordinary identifier (e.g. file handles).
                if nxt is None or nxt.type not in ('IDENTIFIER', 'VECTOR'):
                    tok.type = 'IDENTIFIER'

        # --- Pass B: generic angle brackets ----------------------------
        i = 0
        n = len(toks)
        while i < n:
            tok = toks[i]
            if tok.type == 'LESS' and i > 0 and toks[i - 1].type in self._GENERIC_PREV \
                    and not self._adjacent(toks, i, 'LESS'):
                # A generic argument list can never OPEN with another `<`
                # (no type is spelled starting with an angle bracket), so an
                # adjacent `<<` is always the shift operator.  Refusing to
                # start the scan here is what keeps `a << b >> (c)` from
                # being retagged as `a<<b>>` generic arguments: the angle
                # counts are balanced and `(` is in _GENERIC_FOLLOW, so the
                # scan would otherwise have "matched".
                depth = 1
                angle_positions = [i]
                j = i + 1
                matched = -1
                while j < n:
                    tt = toks[j].type
                    if tt == 'LESS':
                        depth += 1
                        angle_positions.append(j)
                    elif tt == 'GREATER':
                        depth -= 1
                        angle_positions.append(j)
                        if depth == 0:
                            matched = j
                            break
                    elif tt not in self._GENERIC_INSIDE:
                        break
                    j += 1
                if matched >= 0:
                    # An IMPLEMENT/IMPL opener is unambiguous; after a plain
                    # identifier, only re-tag when the token following the
                    # closing '>' can legally follow a type instantiation.
                    # An identifier/literal there (``f(a < b, c > d)``)
                    # means these were comparisons — leave LESS/GREATER.
                    follow = toks[matched + 1].type if matched + 1 < n else None
                    if toks[i - 1].type in self._GENERIC_PREV_UNAMBIGUOUS \
                            or follow is None or follow in self._GENERIC_FOLLOW:
                        for pos in angle_positions:
                            toks[pos].type = 'LGENERIC' if toks[pos].type == 'LESS' else 'RGENERIC'
                        i = matched + 1
                        continue
            i += 1

        # --- Pass C: struct literal braces -----------------------------
        for i, tok in enumerate(toks):
            if tok.type != 'LBRACE':
                continue
            prev = toks[i - 1] if i > 0 else None
            n1 = toks[i + 1] if i + 1 < len(toks) else None
            n2 = toks[i + 2] if i + 2 < len(toks) else None
            if prev is not None and prev.type in ('IDENTIFIER', 'RGENERIC', 'RBRACKET') \
                    and n1 is not None and n1.type == 'IDENTIFIER' \
                    and n2 is not None and n2.type == 'COLON':
                tok.type = 'LBRACE_STRUCT'

        # --- Pass D: shift operators -----------------------------------
        # `<<` and `>>` are deliberately NOT lexer regexes.  A `>>` rule
        # would swallow the two closing brackets of `Vec<Vec<int>>` before
        # Pass B ever saw them — the classic C++ nested-generics bug, and
        # exactly the kind of silent misparse docs/token_reachability.md
        # exists to prevent.  By the time this pass runs, every angle
        # bracket Pass B recognised as a type argument is LGENERIC/RGENERIC,
        # so the only LESS/GREATER pairs left are operators.
        #
        # ADJACENCY IS REQUIRED: `a > > b` keeps two GREATER tokens and
        # stays the syntax error it is today, so the spelling of a shift is
        # exactly the two-character one.
        merged: list = []
        i = 0
        n = len(toks)
        while i < n:
            tok = toks[i]
            if tok.type in ('LESS', 'GREATER') and self._adjacent(toks, i, tok.type):
                nxt = toks[i + 1]
                tok.type = 'SHL' if tok.type == 'LESS' else 'SHR'
                tok.value = '<<' if tok.type == 'SHL' else '>>'
                tok.endlexpos = getattr(nxt, 'endlexpos', nxt.lexpos + 1)
                merged.append(tok)
                i += 2
                continue
            merged.append(tok)
            i += 1
        return merged

    @staticmethod
    def _adjacent(toks, i: int, ttype: str) -> bool:
        """True when toks[i+1] has type `ttype` and touches toks[i] in the
        source text (no whitespace or comment between them)."""
        if i + 1 >= len(toks):
            return False
        nxt = toks[i + 1]
        return nxt.type == ttype and nxt.lexpos == toks[i].lexpos + 1

    # Build the lexer
    def __init__(self):
        self.lexer = lex.lex(module=self)
        self.line_starts = [0]  # Track start of each line
        self._tokens = []
        self._index = 0
        self.source_file = "<unknown>"
        self.source = ""
        self.current_token = None

    def input(self, data):
        self.lexer.lineno = 1
        self.lexer.input(data)
        self.source = data
        self.line_starts = [0]  # Reset line starts
        self.current_token = None
        toks = []
        while True:
            tok = self.lexer.token()
            if tok is None:
                break
            line_start = self.line_starts[min(tok.lineno - 1, len(self.line_starts) - 1)]
            tok.column = tok.lexpos - line_start + 1  # 1-based column
            # End offset (exclusive) of the token's raw text.  PLY's
            # `tracking=True` reduce path copies `endlexpos` from the LAST
            # symbol of a production onto the nonterminal, so setting it here
            # is what makes production end positions point PAST the final
            # token instead of at its first character.  Function rules that
            # rewrite t.value (numbers, strings) set it themselves above.
            if not hasattr(tok, 'endlexpos'):
                tok.endlexpos = tok.lexpos + (
                    len(tok.value) if isinstance(tok.value, str) else 1)
            toks.append(tok)
        self._tokens = self._transform(toks)
        self._index = 0

    def token(self):
        if self._index >= len(self._tokens):
            return None
        tok = self._tokens[self._index]
        self._index += 1
        self.current_token = tok
        return tok

    # PLY's tracking-enabled reduce path reads `lexer.lineno` / `lexer.lexpos`
    # for EMPTY productions (there is no first symbol to copy a position
    # from).  This wrapper hands PLY a pre-tokenized stream, so the inner
    # lex object has already run to end-of-input and its own lineno/lexpos
    # are stale; report the position of the token most recently handed to
    # the parser instead, which is the empty production's insertion point.
    @property
    def lineno(self) -> int:
        tok = self.current_token
        return getattr(tok, 'lineno', 1) if tok is not None else 1

    @property
    def lexpos(self) -> int:
        tok = self.current_token
        return getattr(tok, 'lexpos', 0) if tok is not None else 0


# =====================================================================
# Token reachability triage
# =====================================================================
#
# The worst bug of this project's history lived one layer ABOVE the AST:
# `!` was never a lexer token at all, and `t_error` merely logged-and-
# skipped characters it did not know, so `!e` compiled as `e` — silently,
# for the entire life of the compiler.  `hir.AST_NODE_TRIAGE` pins the
# AST->HIR layer against the same failure; this table pins the
# source->token->grammar layer.
#
# Every token name in `Lexer.tokens` is classified into EXACTLY ONE bucket
# with a reason.  `test_token_coverage.py` recomputes the GRAMMAR bucket
# from PLY's own production table on the live parser, so the table cannot
# rot: adding a token without classifying it fails, and moving a token
# into or out of the grammar without moving it in this table fails too.

#: (a) The token appears in at least one grammar production.
GRAMMAR = "grammar"

#: (b) No grammar production names the token, but the FEATURE it spells is
#: reachable by another route, which the reason must state.
CONTEXTUAL = "contextual"

#: (c) No grammar production names the token and there is no other route:
#: the word is reserved and unusable.  Every token in this bucket must have
#: an entry in `RESERVED_WITHOUT_GRAMMAR` so the parser answers with a route
#: instead of a bare "Syntax error at 'use'".
RESERVED_ONLY = "reserved-only"

#: Tokens no regex rule produces — the `_transform` passes synthesize them.
SYNTHESIZED_TOKENS = frozenset({'LGENERIC', 'RGENERIC', 'LBRACE_STRUCT',
                                'SHL', 'SHR'})

TOKEN_TRIAGE: dict[str, tuple[str, str]] = {
    # -- literals and names --------------------------------------------
    'IDENTIFIER': (GRAMMAR, "every name: bindings, calls, fields, types"),
    'NUMBER': (GRAMMAR, "integer literal; also const generic arguments"),
    'FLOAT': (GRAMMAR, "float literal"),
    'STRING': (GRAMMAR, "string literal; also an extern block's ABI string"),
    'FSTRING': (GRAMMAR, "f-string literal (desugared at parse time)"),

    # -- operators ------------------------------------------------------
    'PLUS': (GRAMMAR, "addition; also `+` in type bounds and const generics"),
    'MINUS': (GRAMMAR, "subtraction and unary negation"),
    'TIMES': (GRAMMAR, "multiplication; also `*T` pointer types"),
    'DIVIDE': (GRAMMAR, "division"),
    'MOD': (GRAMMAR, "remainder"),
    'NOT': (GRAMMAR, "logical negation `!e` (unary_expression)"),
    'ANDAND': (GRAMMAR, "short-circuit `&&`"),
    'OROR': (GRAMMAR, "short-circuit `||`; also the empty-parameter lambda"),
    'AMPERSAND': (GRAMMAR, "address-of `&x`, reference types, and bitwise "
                           "and `a & b` (bitand_expression)"),
    'PIPE': (GRAMMAR, "effect-set union `E | F` and bitwise or `a | b` "
                      "(bitor_expression)"),
    'CARET': (GRAMMAR, "bitwise xor `a ^ b` (bitxor_expression)"),
    'TILDE': (GRAMMAR, "bitwise complement `~e` (unary_expression)"),
    'LESS': (GRAMMAR, "`<` comparison (generic `<` is retagged LGENERIC)"),
    'GREATER': (GRAMMAR, "`>` comparison (generic `>` is retagged RGENERIC)"),
    'LESSEQUAL': (GRAMMAR, "`<=` comparison"),
    'GREATEREQUAL': (GRAMMAR, "`>=` comparison"),
    'EQUALEQUAL': (GRAMMAR, "`==` comparison"),
    'NOTEQUAL': (GRAMMAR, "`!=` comparison"),
    'EQUALS': (GRAMMAR, "binding and assignment `=`"),

    # -- punctuation ----------------------------------------------------
    'LPAREN': (GRAMMAR, "grouping, calls, parameter lists"),
    'RPAREN': (GRAMMAR, "grouping, calls, parameter lists"),
    'LBRACE': (GRAMMAR, "blocks, handler arms, module bodies"),
    'RBRACE': (GRAMMAR, "blocks, handler arms, module bodies"),
    'LBRACKET': (GRAMMAR, "indexing, list literals, bracket-form type args"),
    'RBRACKET': (GRAMMAR, "indexing, list literals, bracket-form type args"),
    'SEMICOLON': (GRAMMAR, "statement and item separator"),
    'COLON': (GRAMMAR, "type ascription, struct-literal fields, type bounds"),
    'COMMA': (GRAMMAR, "list separator"),
    'DOT': (GRAMMAR, "field access, module paths"),
    'DOTDOT': (GRAMMAR, "range `a..b`; also relative import paths"),
    'TRIPLE_DOT': (GRAMMAR, "list spread `...xs`; also relative import paths"),
    'DOUBLECOLON': (GRAMMAR, "path separator in postfix and index positions"),
    'ARROW': (GRAMMAR, "return types, lambda bodies, comprehension targets"),
    'FATARROW': (GRAMMAR, "match and handler arm `=>`"),
    'BACKSLASH': (GRAMMAR, "function type `fn\\(T) -> U`"),
    'AT': (GRAMMAR, "mode annotation `@mut` (mode_annotation)"),

    # -- synthesized by _transform --------------------------------------
    'LGENERIC': (GRAMMAR, "synthesized `<` of a type argument list (Pass B)"),
    'RGENERIC': (GRAMMAR, "synthesized `>` of a type argument list (Pass B)"),
    'LBRACE_STRUCT': (GRAMMAR, "synthesized `{` of a struct literal (Pass C)"),
    'SHL': (GRAMMAR, "synthesized shift-left `<<` from an adjacent LESS pair "
                     "Pass B left unclaimed (Pass D)"),
    'SHR': (GRAMMAR, "synthesized shift-right `>>` from an adjacent GREATER "
                     "pair Pass B left unclaimed (Pass D)"),

    # -- control flow and declarations -----------------------------------
    'IF': (GRAMMAR, "`if` / `if let`"),
    'ELSE': (GRAMMAR, "`else`"),
    'WHILE': (GRAMMAR, "`while` / `while let`"),
    'FOR': (GRAMMAR, "`for` loops, comprehensions, `implement T for S`"),
    'IN': (GRAMMAR, "`for x in xs`, comprehensions, `handle .. in ..`"),
    'FN': (GRAMMAR, "function declarations, lambdas, method signatures"),
    'RETURN': (GRAMMAR, "`return`"),
    'LET': (GRAMMAR, "`let` bindings, `if let`, `while let`"),
    'MUT': (GRAMMAR, "`let mut`, `&mut`, `@mut` type positions"),
    'CONST': (GRAMMAR, "`const N: int` generic params and `const T` types"),
    'MATCH': (GRAMMAR, "`match` expressions"),
    'STRUCT': (GRAMMAR, "struct definitions"),
    'ENUM': (GRAMMAR, "enum definitions"),
    'TYPE': (GRAMMAR, "type aliases and extern type declarations"),
    'PRINT': (GRAMMAR, "`print(..)` (expands to a `__builtin$` call)"),
    'MOVE': (GRAMMAR, "`move e`"),
    'EXCLAVE': (GRAMMAR, "`exclave e`"),
    'BORROW': (GRAMMAR, "`borrow x` / `borrow x as T`"),
    'AS': (GRAMMAR, "casts, import aliases, `borrow x as T`"),
    'SOME': (GRAMMAR, "`Some(e)` builtin option constructor"),
    'NONE': (GRAMMAR, "`None` builtin option constructor"),
    'VECTOR': (GRAMMAR, "`vector[T, N]` types and vector literals"),
    'VOID': (GRAMMAR, "`void` (FFI type)"),
    'SIZE_T': (GRAMMAR, "`size_t` (FFI type)"),
    'UNSAFE': (GRAMMAR, "`unsafe { .. }` blocks"),
    'EXTERN': (GRAMMAR, "`extern` blocks and extern type statements"),
    'COMPTIME': (GRAMMAR, "`comptime` blocks and `comptime fn`"),
    'SPAWN': (GRAMMAR, "`spawn(e)`"),
    'TO_DEVICE': (GRAMMAR, "`to_device(x)`"),
    'FROM_DEVICE': (GRAMMAR, "`from_device(x)`"),
    'TRY': (GRAMMAR, "`try`/`catch` expressions"),
    'CATCH': (GRAMMAR, "`try`/`catch` expressions"),

    # -- effects ---------------------------------------------------------
    'EFFECT': (GRAMMAR, "`effect E { .. }` declarations"),
    'PERFORM': (GRAMMAR, "`perform E.op(..)`"),
    'PERFORMS': (GRAMMAR, "`performs E` effect rows on signatures"),
    'HANDLE': (GRAMMAR, "`handle e { .. }` / `handle e with { .. } in ..`"),
    'RESUME': (GRAMMAR, "`resume(v)` inside a handler arm"),
    'WITH': (GRAMMAR, "`handle .. with { .. }`, `effect .. with ..`"),

    # -- traits ----------------------------------------------------------
    'TRAIT': (GRAMMAR, "`trait T { .. }` (trait_keyword)"),
    'INTERFACE': (GRAMMAR, "`interface T { .. }`, a spelling of `trait`"),
    'IMPLEMENT': (GRAMMAR, "`implement .. { .. }` (implement_keyword)"),
    'IMPL': (GRAMMAR, "`impl .. { .. }`, a spelling of `implement`"),
    'IMPLEMENTS': (GRAMMAR, "`implements T: S { .. }` and `T implements B`"),
    'EXTENDS': (GRAMMAR, "trait supertraits and `T extends B` constraints"),
    'WHERE': (GRAMMAR, "`where` clauses"),

    # -- modes in type position -------------------------------------------
    'UNIQUE': (GRAMMAR, "`unique T` type expression"),
    'EXCLUSIVE': (GRAMMAR, "`exclusive T` type expression"),

    # -- modules -----------------------------------------------------------
    'IMPORT': (GRAMMAR, "`import ..` / `from .. import ..`"),
    'FROM': (GRAMMAR, "`from .. import ..`"),
    'MODULE': (GRAMMAR, "`module m { .. }`"),
    'EXPORT': (GRAMMAR, "`export { .. }`"),
    'PUBLIC': (GRAMMAR, "visibility modifier and `public import`"),
    'PRIVATE': (GRAMMAR, "visibility modifier"),
    'PROTECTED': (GRAMMAR, "visibility modifier"),
    'VISIBILITY': (GRAMMAR, "`visibility { .. }` blocks"),

    # -- (b) reachable only through a lexer rewrite -------------------------
    'ONCE': (CONTEXTUAL,
             "no production names ONCE, but the linearity mode is reachable "
             "as the annotation `@once`: Pass A retags any keyword after `@` "
             "as IDENTIFIER, and `mode_annotation : AT IDENTIFIER` accepts "
             "it, so `let @once f = ..` really does bind a once-callable "
             "(frozen_borrow_checker.check_linearity enforces it)"),
    'SEPARATE': (CONTEXTUAL,
                 "same route as ONCE: `@separate` reaches the grammar as "
                 "`AT IDENTIFIER` and _split_mode reads it as a linearity"),
    'MANY': (CONTEXTUAL,
             "same route as ONCE: `@many` reaches the grammar as "
             "`AT IDENTIFIER` and _split_mode reads it as a linearity"),

    # -- (c) reserved, no feature ------------------------------------------
    'USE': (RESERVED_ONLY,
            "there is no `use` statement and never has been "
            "(docs/modules_implementation.md); the word stays reserved "
            "because `use std.math;` is the likely mistake, and p_error "
            "routes it to `import`"),
    'KERNEL': (RESERVED_ONLY,
               "GPU kernel annotations have no syntax and no runtime "
               "(docs/v1_gap_analysis.md); the word stays reserved so it "
               "reads as unimplemented rather than as a free identifier, "
               "matching its siblings `to_device`/`from_device`, which "
               "parse and then raise UnsupportedConstruct"),
}

#: Guidance `Parser.p_error` attaches when one of these tokens is what the
#: parse choked on.  Since no production names them, EVERY appearance is a
#: syntax error, so this covers the token completely.  Keyed by token type.
RESERVED_WITHOUT_GRAMMAR: dict[str, list[str]] = {
    'USE': ["`use` is a reserved word with no statement form in Metaxu",
            "To bring names into scope write `import std.math;` or "
            "`from std.math import abs;`"],
    'KERNEL': ["`kernel` is reserved for GPU kernel annotations, which are "
               "not implemented (see docs/v1_gap_analysis.md)",
               "`to_device(x)` and `from_device(x)` parse but have no runtime"],
    'ONCE': ["`once` is a linearity mode, not a bare keyword",
             "Write it as a mode annotation: `let @once f = fn(x: int) -> "
             "int { x };`"],
    'SEPARATE': ["`separate` is a linearity mode, not a bare keyword",
                 "Write it as a mode annotation: `let @separate x = ..;`"],
    'MANY': ["`many` is a linearity mode, not a bare keyword",
             "Write it as a mode annotation: `let @many f = ..;`"],
}
