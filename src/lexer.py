import ply.lex as lex

class Lexer:
    # Reserved words
    reserved = {
        'function': 'FUNCTION',
        'effect': 'EFFECT',
        'struct': 'STRUCT',
        'enum': 'ENUM',
        'match': 'MATCH',
        'with': 'WITH',
        'handle': 'HANDLE',
        'perform': 'PERFORM',
        'resume': 'RESUME',
        'move': 'MOVE',
        'mut': 'MUT',
        'local': 'LOCAL',
        'exclave': 'EXCLAVE',
        'once': 'ONCE',
        'spawn': 'SPAWN',
        'kernel': 'KERNEL',
        'to_device': 'TO_DEVICE',
        'from_device': 'FROM_DEVICE',
        # Interface and implementation keywords
        'interface': 'INTERFACE',
        'impl': 'IMPL',
        'for': 'FOR',
        'where': 'WHERE',
        'extends': 'EXTENDS',
        'implements': 'IMPLEMENTS',
        'type': 'TYPE',
        'fn': 'FN',
        # Mode-related keywords
        'unique': 'UNIQUE',
        'exclusive': 'EXCLUSIVE',
        'shared': 'SHARED',
        'global': 'GLOBAL',
        'separate': 'SEPARATE',
        'many': 'MANY',
        'borrow': 'BORROW',
        'import': 'IMPORT',
        'from': 'FROM',
        'as': 'AS',
    }

    # List of token names
    tokens = [
        'IDENTIFIER', 'NUMBER', 'FLOAT', 'STRING',
        'PLUS', 'MINUS', 'TIMES', 'DIVIDE',
        'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'LBRACKET', 'RBRACKET',
        'SEMICOLON', 'COLON', 'COMMA', 'DOT', 'EQUALS', 'ARROW', 'BACKSLASH',
        'AMPERSAND', 'PIPE', 'UNDERSCORE', 'DOUBLECOLON', 'AT',
        'PLUS_EQUALS', 'MINUS_EQUALS', 'TIMES_EQUALS', 'DIVIDE_EQUALS',
        'DOUBLE_PLUS', 'DOUBLE_MINUS', 'DOUBLE_ARROW',
    ] + list(reserved.values())

    # Regular expression rules for simple tokens
    def t_PLUS(self, t):
        r'\+'
        return t

    def t_MINUS(self, t):
        r'-'
        return t

    def t_TIMES(self, t):
        r'\*'
        return t

    def t_DIVIDE(self, t):
        r'/'
        return t

    def t_LPAREN(self, t):
        r'\('
        return t

    def t_RPAREN(self, t):
        r'\)'
        return t

    def t_LBRACE(self, t):
        r'\{'
        return t

    def t_RBRACE(self, t):
        r'\}'
        return t

    def t_LBRACKET(self, t):
        r'\['
        return t

    def t_RBRACKET(self, t):
        r'\]'
        return t

    def t_SEMICOLON(self, t):
        r';'
        return t

    def t_COLON(self, t):
        r':'
        return t

    def t_COMMA(self, t):
        r','
        return t

    def t_DOT(self, t):
        r'\.'
        return t

    def t_EQUALS(self, t):
        r'='
        return t

    def t_ARROW(self, t):
        r'->'
        return t

    def t_BACKSLASH(self, t):
        r'\\'
        return t

    def t_AMPERSAND(self, t):
        r'&'
        return t

    def t_PIPE(self, t):
        r'\|'
        return t

    def t_UNDERSCORE(self, t):
        r'_'
        return t

    def t_DOUBLECOLON(self, t):
        r'::'
        return t

    def t_AT(self, t):
        r'@'
        return t

    def t_PLUS_EQUALS(self, t):
        r'\+='
        return t

    def t_MINUS_EQUALS(self, t):
        r'-='
        return t

    def t_TIMES_EQUALS(self, t):
        r'\*='
        return t

    def t_DIVIDE_EQUALS(self, t):
        r'/='
        return t

    def t_DOUBLE_PLUS(self, t):
        r'\+\+'
        return t

    def t_DOUBLE_MINUS(self, t):
        r'--'
        return t

    def t_DOUBLE_ARROW(self, t):
        r'=>'
        return t

    # Ignored characters
    t_ignore = ' \t'

    def t_FLOAT(self, t):
        r'\d+\.\d+'
        t.value = float(t.value)
        return t

    def t_NUMBER(self, t):
        r'\d+'
        t.value = int(t.value)
        return t

    def t_STRING(self, t):
        r'"([^"\\]|\\.)*"'
        t.value = t.value[1:-1]  # Remove quotes
        return t

    def t_IDENTIFIER(self, t):
        r'[a-zA-Z_][a-zA-Z0-9_]*'
        t.type = self.reserved.get(t.value, 'IDENTIFIER')
        return t

    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)

    def t_error(self, t):
        print(f"Illegal character '{t.value[0]}'")
        t.lexer.skip(1)

    def __init__(self):
        self.lexer = lex.lex(module=self)

    def input(self, data):
        self.lexer.input(data)

    def token(self):
        return self.lexer.token()
