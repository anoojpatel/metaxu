
import ply.lex as lex

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
    'once': 'ONCE',
    'spawn': 'SPAWN',
    'kernel': 'KERNEL',
    'to_device': 'TO_DEVICE',
    'from_device': 'FROM_DEVICE',
}

# List of token names
tokens = [
    'IDENTIFIER', 'NUMBER', 'FLOAT', 'STRING',
    'PLUS', 'MINUS', 'TIMES', 'DIVIDE',
    'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'LBRACKET', 'RBRACKET',
    'SEMICOLON', 'COLON', 'COMMA', 'DOT', 'EQUALS', 'ARROW', 'BACKSLASH',
    'AMPERSAND', 'PIPE', 'UNDERSCORE', 'DOUBLECOLON',
] + list(reserved.values())

# Regular expression rules for simple tokens
t_PLUS         = r'\+'
t_MINUS        = r'-'
t_TIMES        = r'\*'
t_DIVIDE       = r'/'
t_LPAREN       = r'\('
t_RPAREN       = r'\)'
t_LBRACE       = r'\{'
t_RBRACE       = r'\}'
t_LBRACKET     = r'\['
t_RBRACKET     = r'\]'
t_SEMICOLON    = r';'
t_COLON        = r':'
t_COMMA        = r','
t_DOT          = r'\.'
t_EQUALS       = r'='
t_ARROW        = r'->'
t_BACKSLASH    = r'\\'
t_AMPERSAND    = r'&'
t_PIPE         = r'\|'
t_UNDERSCORE   = r'_'
t_DOUBLECOLON  = r'::'

def t_FLOAT(t):
    r'\d+\.\d+'
    t.value = float(t.value)
    return t

def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t

def t_STRING(t):
    r'"([^"\\]|\\.)*"'
    t.value = t.value[1:-1]
    return t

def t_IDENTIFIER(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    t.type = reserved.get(t.value, 'IDENTIFIER')
    return t

# Ignored characters
t_ignore = ' \t'

# Newlines
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

# Error handling
def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

lexer = lex.lex()

