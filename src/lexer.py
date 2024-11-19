import ply.lex as lex

class Lexer:
    # A string containing ignored characters (spaces and tabs)
    t_ignore = ' \t'

    # Reserved words
    reserved = {
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
        # Control flow keywords
        'return': 'RETURN',
        'if': 'IF',
        'else': 'ELSE',
        'while': 'WHILE',
        'for': 'FOR',
        'let': 'LET',
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
        # Module-related keywords
        'import': 'IMPORT',
        'from': 'FROM',
        'as': 'AS',
        'module': 'MODULE',
        'export': 'EXPORT',
        'use': 'USE',
        'public': 'PUBLIC',
        'private': 'PRIVATE',
        'protected': 'PROTECTED',
        'visibility': 'VISIBILITY',
        'some': 'SOME',
        'none': 'NONE',
        'box': 'BOX',
        'option': 'OPTION',
        'vector': 'VECTOR'
    }

    # List of token names
    tokens = [
        'IDENTIFIER', 'NUMBER', 'FLOAT', 'STRING',
        'PLUS', 'MINUS', 'TIMES', 'DIVIDE',
        'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'LBRACKET', 'RBRACKET',
        'EQUALS', 'SEMICOLON', 'COLON', 'COMMA', 'DOT', 'TRIPLE_DOT',
        'DOUBLECOLON', 'ARROW', 'BACKSLASH', 'AT', 'AMPERSAND',
        'LESS', 'GREATER'
    ] + list(reserved.values())

    # Regular expression rules for simple tokens
    t_PLUS = r'\+'
    t_MINUS = r'-'
    t_TIMES = r'\*'
    t_DIVIDE = r'/'
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_LBRACE = r'\{'
    t_RBRACE = r'\}'
    t_LBRACKET = r'\['
    t_RBRACKET = r'\]'
    t_EQUALS = r'='
    t_SEMICOLON = r';'
    t_COLON = r':'
    t_COMMA = r','
    t_DOT = r'\.'
    t_TRIPLE_DOT = r'\.\.\.'
    t_DOUBLECOLON = r'::'
    t_ARROW = r'->'
    t_BACKSLASH = r'\\'
    t_AT = r'@'
    t_AMPERSAND = r'&'
    t_LESS = r'<'
    t_GREATER = r'>'

    # Regular expression rules with actions
    def t_IDENTIFIER(self, t):
        r'[a-zA-Z_][a-zA-Z_0-9]*'
        t.type = self.reserved.get(t.value, 'IDENTIFIER')
        return t

    def t_NUMBER(self, t):
        r'\d+'
        t.value = int(t.value)
        return t

    def t_FLOAT(self, t):
        r'\d*\.\d+'
        t.value = float(t.value)
        return t

    def t_STRING(self, t):
        r'"[^"]*"'
        t.value = t.value[1:-1]  # Remove quotes
        return t


    # Comments
    def t_COMMENT(self, t):
        r'//.*'
        pass

    # Define a rule so we can track line numbers
    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)
 

    # Error handling rule
    def t_error(self, t):
        print(f"Illegal character '{t.value[0]}' at line {t.lexer.lineno}")
        t.lexer.skip(1)

    # Build the lexer
    def __init__(self):
        self.lexer = lex.lex(module=self)

    def input(self, data):
        self.lexer.input(data)

    def token(self):
        return self.lexer.token()
