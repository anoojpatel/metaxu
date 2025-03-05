import ply.lex as lex
from metaxu.errors import CompileError, SourceLocation, get_source_context
from typing import List
import logging
class Lexer:
    # A string containing ignored characters (spaces and tabs)
    t_ignore = ' \t'

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
        'with': 'WITH',
        'in': 'IN',
        'let': 'LET',
        #'mut': 'MUT',
        'type': 'TYPE',
        'extern': 'EXTERN',
        'const': 'CONST',
        'move': 'MOVE',
        'local': 'LOCAL',
        'exclave': 'EXCLAVE',
        'once': 'ONCE',
        'spawn': 'SPAWN',
        'kernel': 'KERNEL',
        'to_device': 'TO_DEVICE',
        'from_device': 'FROM_DEVICE',
        'print': 'PRINT',  # Add print keyword
        # Control flow keywords
        'if': 'IF',
        'else': 'ELSE',
        'while': 'WHILE',
        'for': 'FOR',
        'let': 'LET',
        # Interface and implementation keywords
        'interface': 'INTERFACE',
        'impl': 'IMPL',
        'for': 'FOR',
        'in': 'IN',
        'where': 'WHERE',
        'extends': 'EXTENDS',
        'implements': 'IMPLEMENTS',
        'type': 'TYPE',
        'fn': 'FN',
        # Mode-related keywords
        'unique': 'UNIQUE',
        'exclusive': 'EXCLUSIVE',
        'const': 'CONST',
        'global': 'GLOBAL',
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
        'extern': 'EXTERN',
        'unsafe': 'UNSAFE',  # Add unsafe keyword
        'async': 'ASYNC',    # Add async keyword
        'void': 'VOID',      # Add void type
        'size_t': 'SIZE_T',  # Add size_t type
        'as': 'AS',          # Ensure 'as' is in the reserved keywords
    }

    # List of token names
    tokens = [
        'IDENTIFIER', 'NUMBER', 'FLOAT', 'STRING', 'BOOL',
        'PLUS', 'MINUS', 'TIMES', 'DIVIDE',
        'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'LBRACKET', 'RBRACKET',
        'EQUALS', 'SEMICOLON', 'COLON', 'COMMA', 'DOT', 'TRIPLE_DOT',
        'DOUBLECOLON', 'ARROW', 'BACKSLASH', 'AT', 'AMPERSAND',
        'LESS', 'GREATER', 'LESSEQUAL', 'GREATEREQUAL', 'EQUALEQUAL', 'NOTEQUAL',
        'MUT', 'AS'  # Add AS for type casts
    ] + list(reserved.values())

    # Ensure AS is recognized as a keyword
    t_AS = r'as'

    # Regular expression rules for simple tokens
    t_PLUS = r'\+'
    t_MINUS = r'-'
    t_TIMES = r'\*'
    t_DIVIDE = r'/'
    t_EQUALS = r'='
    t_EQUALEQUAL = r'=='
    t_NOTEQUAL = r'!='
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
    t_TRIPLE_DOT = r'\.\.\.'
    t_DOUBLECOLON = r'::'
    t_ARROW = r'->'
    t_BACKSLASH = r'\\' # Added for function type annotations
    t_AT = r'@'
    t_AMPERSAND = r'&'

    # Regular expression rules with actions
    def t_IDENTIFIER(self, t):
        r'[a-zA-Z_][a-zA-Z_0-9]*'
        # Check for reserved words
        t.type = self.reserved.get(t.value, 'IDENTIFIER')
        logging.debug(f"Token recognized: {t.type}, value: {t.value}")  # Debug statement
        return t

    def t_NUMBER(self, t):
        r'\d+'
        t.value = int(t.value)
        return t

    def t_FLOAT(self, t):
        r'\d*\.\d+'
        t.value = float(t.value)
        return t
    def t_BOOL(self, t):
        r'(true|false)'
        t.value = True if t.value == 'true' else False
        return t

    def t_STRING(self, t):
        r'"[^"]*"'
        t.value = (t.value[1:-1], 'string')  # Tuple with (value, type)
        return t


    # Comments
    def t_COMMENT(self, t):
        r'\#.*'
        pass

    # Define a rule so we can track line numbers
    def t_newline(self, t):
        r'\n+'
        # Track the position after each newline
        for i in range(len(t.value)):
            self.line_starts.append(t.lexpos + i + 1)
        t.lexer.lineno += len(t.value)

    # Error handling rule
    def t_error(self, t):
        # Calculate column based on the last line start
        line_start = self.line_starts[t.lineno - 1]
        column = t.lexpos - line_start
        print(f"\n=== Lexer Error ===")
        print(f"Illegal character '{t.value[0]}' at line {t.lineno}, column {column}")
        t.lexer.skip(1)

    # Build the lexer
    def __init__(self):
        self.lexer = lex.lex(module=self)
        self.line_starts = [0]  # Track start of each line

    def input(self, data):
        self.lexer.input(data)
        self.line_starts = [0]  # Reset line starts

    def token(self):
        tok = self.lexer.token()
        if tok:
            # Calculate column based on the last line start
            line_start = self.line_starts[min(tok.lineno - 1, len(self.line_starts) - 1)]
            tok.column = tok.lexpos - line_start + 1  # Make columns 1-based
        return tok
