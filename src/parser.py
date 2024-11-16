
import ply.yacc as yacc
from lexer import tokens
from ast import *

def p_program(p):
    '''program : statement_list'''
    p[0] = Program(p[1])

def p_statement_list(p):
    '''statement_list : statement_list statement
                      | statement'''
    if len(p) == 3:
        p[0] = p[1] + [p[2]]
    else:
        p[0] = [p[1]]

def p_statement(p):
    '''statement : expression SEMICOLON
                 | assignment SEMICOLON
                 | function_declaration
                 | struct_definition
                 | enum_definition
                 | effect_declaration
                 | handle_expression SEMICOLON
                 | perform_expression SEMICOLON'''
    p[0] = p[1]

def p_assignment(p):
    '''assignment : IDENTIFIER EQUALS expression'''
    p[0] = Assignment(p[1], p[3])

def p_expression(p):
    '''expression : term
                  | expression PLUS term
                  | expression MINUS term
                  | lambda_expression
                  | match_expression
                  | perform_expression
                  | handle_expression
                  | spawn_expression'''
    if len(p) == 2:
        p[0] = p[1]
    elif len(p) == 4:
        p[0] = BinaryOperation(p[1], p[2], p[3])
    else:
        p[0] = p[1]

def p_term(p):
    '''term : factor
            | term TIMES factor
            | term DIVIDE factor'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = BinaryOperation(p[1], p[2], p[3])

def p_factor(p):
    '''factor : NUMBER
              | FLOAT
              | STRING
              | IDENTIFIER
              | function_call
              | LPAREN expression RPAREN
              | vector_literal
              | struct_instantiation
              | field_access
              | move_expression
              | borrow_shared_expression
              | borrow_unique_expression
              | variant_instantiation'''
    if isinstance(p[1], (int, float, str)):
        p[0] = Literal(p[1])
    elif isinstance(p[1], str):
        p[0] = Variable(p[1])
    else:
        p[0] = p[1]

def p_function_call(p):
    '''function_call : IDENTIFIER LPAREN argument_list RPAREN'''
    p[0] = FunctionCall(p[1], p[3])

def p_argument_list(p):
    '''argument_list : argument_list COMMA expression
                     | expression
                     | empty'''
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    elif len(p) == 2:
        if p[1] is None:
            p[0] = []
        else:
            p[0] = [p[1]]

def p_function_declaration(p):
    '''function_declaration : FUNCTION IDENTIFIER LPAREN parameter_list RPAREN block
                            | KERNEL FUNCTION IDENTIFIER LPAREN parameter_list RPAREN block'''
    if len(p) == 7:
        p[0] = FunctionDeclaration(p[2], p[4], p[6])
    else:
        p[0] = FunctionDeclaration(p[3], p[5], p[7], is_kernel=True)

def p_parameter_list(p):
    '''parameter_list : parameter_list COMMA parameter
                      | parameter
                      | empty'''
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    elif len(p) == 2:
        if p[1] is None:
            p[0] = []
        else:
            p[0] = [p[1]]

def p_parameter(p):
    '''parameter : IDENTIFIER COLON type_specification'''
    p[0] = (p[1], p[3])

def p_type_specification(p):
    '''type_specification : IDENTIFIER
                          | VECTOR LBRACKET IDENTIFIER COMMA NUMBER RBRACKET'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = ('vector', p[3], p[5])

def p_block(p):
    '''block : LBRACE statement_list RBRACE'''
    p[0] = p[2]

def p_struct_definition(p):
    '''struct_definition : STRUCT IDENTIFIER LBRACE struct_fields RBRACE'''
    p[0] = StructDefinition(p[2], p[4])

def p_struct_fields(p):
    '''struct_fields : struct_fields struct_field
                     | struct_field'''
    if len(p) == 3:
        p[0] = p[1] + [p[2]]
    else:
        p[0] = [p[1]]

def p_struct_field(p):
    '''struct_field : IDENTIFIER COLON type_specification SEMICOLON'''
    p[0] = (p[1], p[3])

def p_struct_instantiation(p):
    '''struct_instantiation : IDENTIFIER LBRACE field_assignments RBRACE'''
    p[0] = StructInstantiation(p[1], p[3])

def p_struct_initializers(p):
    '''struct_initializers : struct_initializers COMMA struct_initializer
                           | struct_initializer'''
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    else:
        p[0] = [p[1]]

def p_struct_initializer(p):
    '''struct_initializer : IDENTIFIER COLON expression'''
    p[0] = (p[1], p[3])

def p_field_assignments(p):
    '''field_assignments : field_assignments field_assignment
                         | field_assignment'''
    if len(p) == 3:
        p[0] = {**p[1], **p[2]}
    else:
        p[0] = p[1]

def p_field_assignment(p):
    '''field_assignment : IDENTIFIER EQUALS expression SEMICOLON'''
    p[0] = {p[1]: p[3]}

def p_field_access(p):
    '''field_access : expression DOT IDENTIFIER'''
    p[0] = FieldAccess(p[1], p[3])

def p_enum_definition(p):
    '''enum_definition : ENUM IDENTIFIER LBRACE variant_definitions RBRACE'''
    p[0] = EnumDefinition(p[2], p[4])

def p_variant_definitions(p):
    '''variant_definitions : variant_definitions variant_definition
                           | variant_definition'''
    if len(p) == 3:
        p[0] = p[1] + [p[2]]
    else:
        p[0] = [p[1]]

def p_variant_definition(p):
    '''variant_definition : IDENTIFIER variant_fields SEMICOLON'''
    p[0] = VariantDefinition(p[1], p[2])

def p_variant_fields(p):
    '''variant_fields : LPAREN variant_field_list RPAREN
                      | empty'''
    if len(p) == 4:
        p[0] = p[2]
    else:
        p[0] = []

def p_variant_field_list(p):
    '''variant_field_list : variant_field_list COMMA variant_field
                          | variant_field'''
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    else:
        p[0] = [p[1]]

def p_variant_field(p):
    '''variant_field : IDENTIFIER COLON type_specification'''
    p[0] = (p[1], p[3])

def p_variant_instantiation(p):
    '''variant_instantiation : IDENTIFIER DOUBLECOLON IDENTIFIER variant_field_values'''
    p[0] = VariantInstance(p[1], p[3], p[4])

def p_variant_field_values(p):
    '''variant_field_values : LPAREN field_assignments RPAREN
                            | empty'''
    if len(p) == 4:
        p[0] = p[2]
    else:
        p[0] = {}

def p_match_expression(p):
    '''match_expression : MATCH expression LBRACE case_list RBRACE'''
    p[0] = MatchExpression(p[2], p[4])

def p_case_list(p):
    '''case_list : case_list case
                 | case'''
    if len(p) == 3:
        p[0] = p[1] + [p[2]]
    else:
        p[0] = [p[1]]

def p_case(p):
    '''case : pattern ARROW expression SEMICOLON'''
    p[0] = (p[1], p[3])

def p_pattern(p):
    '''pattern : NUMBER
               | STRING
               | IDENTIFIER
               | UNDERSCORE
               | IDENTIFIER DOUBLECOLON IDENTIFIER'''
    if len(p) == 2:
        if isinstance(p[1], int):
            p[0] = LiteralPattern(p[1])
        elif p[1] == '_':
            p[0] = WildcardPattern()
        else:
            p[0] = VariablePattern(p[1])
    else:
        p[0] = VariantPattern(p[1], p[3], [])

def p_lambda_expression(p):
    '''lambda_expression : BACKSLASH parameter_list ARROW expression'''
    p[0] = LambdaExpression(p[2], p[4])

def p_perform_expression(p):
    '''perform_expression : PERFORM IDENTIFIER LPAREN argument_list RPAREN'''
    p[0] = PerformEffect(p[2], p[4])

def p_handle_expression(p):
    '''handle_expression : HANDLE IDENTIFIER WITH handler_expression IN expression'''
    p[0] = HandleEffect(p[2], p[4], p[6])

def p_handler_expression(p):
    '''handler_expression : function_declaration
                          | lambda_expression'''
    p[0] = p[1]

def p_resume_expression(p):
    '''resume_expression : RESUME LPAREN expression RPAREN
                         | RESUME'''
    if len(p) == 5:
        p[0] = Resume(p[3])
    else:
        p[0] = Resume()

def p_move_expression(p):
    '''move_expression : MOVE LPAREN IDENTIFIER RPAREN'''
    p[0] = Move(p[3])

def p_borrow_shared_expression(p):
    '''borrow_shared_expression : AMPERSAND IDENTIFIER'''
    p[0] = BorrowShared(p[2])

def p_borrow_unique_expression(p):
    '''borrow_unique_expression : AMPERSAND MUT IDENTIFIER'''
    p[0] = BorrowUnique(p[3])

def p_spawn_expression(p):
    '''spawn_expression : SPAWN LPAREN expression RPAREN'''
    p[0] = SpawnExpression(p[3])

def p_vector_literal(p):
    '''vector_literal : VECTOR LBRACKET IDENTIFIER COMMA NUMBER RBRACKET LPAREN element_list RPAREN'''
    p[0] = VectorLiteral(p[3], p[5], p[8])

def p_element_list(p):
    '''element_list : element_list COMMA expression
                    | expression'''
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    else:
        p[0] = [p[1]]

def p_to_device(p):
    '''expression : TO_DEVICE LPAREN IDENTIFIER RPAREN'''
    p[0] = ToDevice(p[3])

def p_from_device(p):
    '''expression : FROM_DEVICE LPAREN IDENTIFIER RPAREN'''
    p[0] = FromDevice(p[3])

def p_empty(p):
    'empty :'
    pass

def p_error(p):
    if p:
        print(f"Syntax error at '{p.value}'")
    else:
        print("Syntax error at EOF")

parser = yacc.yacc()

