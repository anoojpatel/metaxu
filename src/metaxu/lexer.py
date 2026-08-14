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
        'box': 'BOX',
        'option': 'OPTION',
        'vector': 'VECTOR',
        'unsafe': 'UNSAFE',
        'async': 'ASYNC',
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
        'NOT',
        # Synthesized by the token-stream disambiguation filter (never produced
        # directly by a regex): generic type argument brackets and the opening
        # brace of a struct literal.
        'LGENERIC', 'RGENERIC', 'LBRACE_STRUCT',
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

    # Comments: both '#' and '//' styles
    def t_COMMENT(self, t):
        r'\#.*|//.*'
        pass

    # NOTE: function rules are matched in definition order; FLOAT must come
    # before NUMBER so that "3.14" lexes as a single float.
    def t_FLOAT(self, t):
        r'\d+\.\d+|\.\d+'
        t.endlexpos = t.lexpos + len(t.value)
        t.value = float(t.value)
        return t

    def t_NUMBER(self, t):
        r'\d+'
        t.endlexpos = t.lexpos + len(t.value)
        t.value = int(t.value)
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
        raise CompileError(
            message=f"illegal character {t.value[0]!r}",
            error_type="LexError",
            location=SourceLocation(
                file=self.source_file, line=t.lineno, column=column),
            notes=["Remove it, or quote it inside a string literal"],
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

    _GENERIC_SCAN_LIMIT = 80

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
        """
        # --- Pass A: contextual keywords -------------------------------
        for i, tok in enumerate(toks):
            prev = toks[i - 1] if i > 0 else None
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            if tok.type in self._KEYWORD_TYPES and prev is not None and \
                    prev.type in ('DOT', 'DOTDOT', 'TRIPLE_DOT', 'AT', 'FN'):
                # Member access (thread.spawn), relative paths (..vector),
                # mode names (@mut/@const), and function names
                # (fn spawn[...]) may reuse keywords.
                tok.type = 'IDENTIFIER'
            elif tok.type in self._KEYWORD_TYPES and prev is not None and \
                    prev.type in ('COMMA', 'IMPORT') and \
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
            if tok.type == 'LESS' and i > 0 and toks[i - 1].type in self._GENERIC_PREV:
                depth = 1
                angle_positions = [i]
                j = i + 1
                matched = -1
                limit = min(n, i + 1 + self._GENERIC_SCAN_LIMIT)
                while j < limit:
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

        return toks

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
