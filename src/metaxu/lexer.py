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
    }

    # List of token names
    tokens = [
        'IDENTIFIER', 'INTEGER', 'FLOAT', 'STRING',
        'PLUS', 'MINUS', 'TIMES', 'DIVIDE',
        'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'LBRACKET', 'RBRACKET',
        'SEMICOLON', 'COLON', 'COMMA', 'DOT',
        'EQUALS', 'ARROW', 'PIPE',
        'LT', 'GT', 'LE', 'GE', 'EQ', 'NE',
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
    t_SEMICOLON = r';'
    t_COLON = r':'
    t_COMMA = r','
    t_DOT = r'\.'
    t_EQUALS = r'='
    t_ARROW = r'->'
    t_PIPE = r'\|'
    t_LT = r'<'
    t_GT = r'>'
    t_LE = r'<='
    t_GE = r'>='
    t_EQ = r'=='
    t_NE = r'!='

    # Regular expression rules with actions
    def t_IDENTIFIER(self, t):
        r'[a-zA-Z_][a-zA-Z_0-9]*'
        # Check for reserved words
        t.type = self.reserved.get(t.value, 'IDENTIFIER') 
        return t

    def t_INTEGER(self, t):
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
