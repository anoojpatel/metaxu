import ply.yacc as yacc
from lexer import Lexer
import metaxu_ast as ast

class Parser:
    tokens = Lexer.tokens  # Add tokens at class level
    
    def __init__(self):
        self.lexer = Lexer()
        self.parser = yacc.yacc(module=self)

    def parse(self, data):
        self.lexer.input(data)
        return self.parser.parse(lexer=self.lexer.lexer)

def p_program(p):
    '''program : statement_list'''
    p[0] = ast.Program(p[1])

def p_statement_list(p):
    '''statement_list : statement_list statement
                      | statement'''
    if len(p) == 3:
        p[0] = p[1] + [p[2]]
    else:
        p[0] = [p[1]]

def p_statement(p):
    '''statement : expression
                 | expression SEMICOLON
                 | assignment
                 | assignment SEMICOLON
                 | function_declaration
                 | struct_definition
                 | enum_definition
                 | effect_declaration
                 | handle_expression
                 | perform_expression
                 | local_declaration
                 | local_declaration SEMICOLON
                 | type_definition
                 | type_definition SEMICOLON
                 | interface_definition
                 | implementation
                 | import_statement
                 | from_import_statement'''
    p[0] = p[1]

def p_local_declaration(p):
    '''local_declaration : LOCAL IDENTIFIER type_annotation
                        | LOCAL IDENTIFIER EQUALS expression'''
    if len(p) == 4:
        p[0] = ast.LocalDeclaration(ast.Variable(p[2]), p[3])
    else:  # len(p) == 5
        p[0] = ast.LocalDeclaration(ast.Variable(p[2]), None, p[4])

def p_type_annotation(p):
    '''type_annotation : COLON type_specification'''
    p[0] = p[2]

def p_parameter(p):
    '''parameter : IDENTIFIER COLON type_specification mode_annotation
                | IDENTIFIER COLON type_specification'''
    if len(p) == 5:
        p[0] = ast.Parameter(p[1], p[3], p[4])
    else:  # len(p) == 4
        p[0] = ast.Parameter(p[1], p[3])

def p_assignment(p):
    '''assignment : IDENTIFIER EQUALS expression'''
    p[0] = ast.Assignment(p[1], p[3])

def p_expression(p):
    '''expression : term
                  | expression PLUS term
                  | expression MINUS term
                  | lambda_expression
                  | match_expression
                  | perform_expression
                  | handle_expression
                  | spawn_expression
                  | exclave_expression
                  | borrow_expression'''
    if len(p) == 2:
        p[0] = p[1]
    elif len(p) == 4:
        p[0] = ast.BinaryOperation(p[1], p[2], p[3])
    else:
        p[0] = p[1]

def p_exclave_expression(p):
    '''exclave_expression : EXCLAVE expression'''
    p[0] = ast.ExclaveExpression(p[2])

def p_term(p):
    '''term : factor
            | term TIMES factor
            | term DIVIDE factor'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = ast.BinaryOperation(p[1], p[2], p[3])

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
        p[0] = ast.Literal(p[1])
    elif isinstance(p[1], str):
        p[0] = ast.Variable(p[1])
    else:
        p[0] = p[1]

def p_function_call(p):
    '''function_call : IDENTIFIER LPAREN argument_list RPAREN'''
    p[0] = ast.FunctionCall(p[1], p[3])

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
        p[0] = ast.FunctionDeclaration(p[2], p[4], p[6])
    else:
        p[0] = ast.FunctionDeclaration(p[3], p[5], p[7], is_kernel=True)

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

def p_mode_annotation(p):
    '''mode_annotation : AT mode_type
                      | empty'''
    if len(p) == 3:
        p[0] = p[2]
    else:
        p[0] = None

def p_mode_type(p):
    '''mode_type : uniqueness_mode
                | locality_mode
                | linearity_mode'''
    p[0] = p[1]

def p_uniqueness_mode(p):
    '''uniqueness_mode : UNIQUE
                      | EXCLUSIVE
                      | SHARED'''
    p[0] = ast.UniquenessMode(p[1])

def p_locality_mode(p):
    '''locality_mode : LOCAL
                    | GLOBAL'''
    p[0] = ast.LocalityMode(p[1])

def p_linearity_mode(p):
    '''linearity_mode : ONCE
                     | SEPARATE
                     | MANY'''
    p[0] = ast.LinearityMode(p[1])

def p_type_parameters(p):
    '''type_parameters : LESS type_parameter_list GREATER
                      | empty'''
    if len(p) == 4:
        p[0] = p[2]
    else:
        p[0] = []

def p_type_parameter_list(p):
    '''type_parameter_list : type_parameter_list COMMA IDENTIFIER
                         | IDENTIFIER'''
    if len(p) == 4:
        p[0] = p[1] + [ast.TypeParameter(p[3])]
    else:
        p[0] = [ast.TypeParameter(p[1])]

def p_type_specification(p):
    '''type_specification : IDENTIFIER mode_annotation
                        | IDENTIFIER LESS type_argument_list GREATER mode_annotation
                        | BOX LESS type_specification GREATER mode_annotation
                        | OPTION LESS type_specification GREATER mode_annotation
                        | VECTOR LBRACKET IDENTIFIER COMMA NUMBER RBRACKET mode_annotation
                        | AMPERSAND type_specification mode_annotation
                        | AMPERSAND MUT type_specification mode_annotation'''
    if len(p) == 3:  # Simple type
        base_type = p[1]
        mode = p[2]
    elif len(p) == 6 and p[1] == 'Box':  # Box type
        base_type = ast.BoxType(p[3])
        mode = p[5]
    elif len(p) == 6 and p[1] == 'Option':  # Option type
        base_type = ast.OptionType(p[3])
        mode = p[5]
    elif len(p) == 6:  # Generic type
        base_type = ast.RecursiveType(p[1], p[3])
        mode = p[5]
    elif len(p) == 8:  # Vector type
        base_type = ast.VectorType(p[3], int(p[5]))
        mode = p[7]
    elif len(p) == 4:  # Immutable reference
        base_type = ast.ReferenceType(p[2], False)
        mode = p[3]
    else:  # len(p) == 5, Mutable reference
        base_type = ast.ReferenceType(p[3], True)
        mode = p[4]
    
    if mode:
        p[0] = ast.ModeTypeAnnotation(base_type, mode)
    else:
        p[0] = base_type

def p_type_argument_list(p):
    '''type_argument_list : type_specification
                        | type_argument_list COMMA type_specification'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_borrow_expression(p):
    '''borrow_expression : BORROW mode_type IDENTIFIER
                        | BORROW IDENTIFIER'''
    if len(p) == 4:
        p[0] = ast.BorrowExpression(ast.Variable(p[3]), p[2])
    else:  # len(p) == 3
        p[0] = ast.BorrowExpression(ast.Variable(p[2]), ast.UniquenessMode(ast.UniquenessMode.SHARED))

def p_block(p):
    '''block : LBRACE statement_list RBRACE'''
    p[0] = p[2]

def p_struct_definition(p):
    '''struct_definition : STRUCT IDENTIFIER LBRACE struct_fields RBRACE'''
    p[0] = ast.StructDefinition(p[2], p[4])

def p_struct_fields(p):
    '''struct_fields : struct_fields struct_field
                     | struct_field'''
    if len(p) == 3:
        p[0] = p[1] + [p[2]]
    else:
        p[0] = [p[1]]

def p_struct_instantiation(p):
    '''struct_instantiation : IDENTIFIER LBRACE field_assignments RBRACE'''
    p[0] = ast.StructInstantiation(p[1], p[3])

def p_struct_initializers(p):
    '''struct_initializers : struct_initializers COMMA struct_initializer
                           | struct_initializer'''
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    else:
        p[0] = [p[1]]

def p_struct_initializer(p):
    '''struct_initializer : IDENTIFIER COLON expression
                        | IDENTIFIER EQUALS expression'''
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
    p[0] = ast.FieldAccess(p[1], p[3])

def p_enum_definition(p):
    '''enum_definition : ENUM IDENTIFIER LBRACE variant_definitions RBRACE'''
    p[0] = ast.EnumDefinition(p[2], p[4])

def p_variant_definitions(p):
    '''variant_definitions : variant_definitions variant_definition
                           | variant_definition'''
    if len(p) == 3:
        p[0] = p[1] + [p[2]]
    else:
        p[0] = [p[1]]

def p_variant_definition(p):
    '''variant_definition : IDENTIFIER variant_fields SEMICOLON'''
    p[0] = ast.VariantDefinition(p[1], p[2])

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
    p[0] = ast.VariantInstance(p[1], p[3], p[4])

def p_variant_field_values(p):
    '''variant_field_values : LPAREN field_assignments RPAREN
                            | empty'''
    if len(p) == 4:
        p[0] = p[2]
    else:
        p[0] = {}

def p_match_expression(p):
    '''match_expression : MATCH expression LBRACE match_cases RBRACE'''
    p[0] = ast.MatchExpression(p[2], p[4])

def p_match_cases(p):
    '''match_cases : option_match_case
                  | match_cases option_match_case'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]

def p_option_match_case(p):
    '''option_match_case : SOME LPAREN IDENTIFIER RPAREN ARROW expression SEMICOLON
                        | NONE ARROW expression SEMICOLON'''
    if len(p) == 8:  # Some case
        p[0] = ('some', p[3], p[6])
    else:  # None case
        p[0] = ('none', None, p[3])

def p_option_expression(p):
    '''option_expression : NONE
                        | SOME LPAREN expression RPAREN'''
    if len(p) == 2:
        p[0] = ast.NoneExpression()
    else:
        p[0] = ast.SomeExpression(p[3])

def p_lambda_expression(p):
    '''lambda_expression : BACKSLASH parameter_list ARROW expression'''
    p[0] = ast.LambdaExpression(p[2], p[4])

def p_perform_expression(p):
    '''perform_expression : PERFORM effect_operation
                        | PERFORM effect_operation SEMICOLON'''
    p[0] = p[2]

def p_effect_operation(p):
    '''effect_operation : IDENTIFIER DOT IDENTIFIER LPAREN argument_list RPAREN
                       | IDENTIFIER DOUBLECOLON IDENTIFIER LPAREN argument_list RPAREN'''
    p[0] = ast.EffectOperation(p[1], p[3], p[5])

def p_struct_field(p):
    '''struct_field : IDENTIFIER COLON type_specification
                   | IDENTIFIER EQUALS type_specification'''
    p[0] = ast.StructField(p[1], p[3])

def p_struct_initializer(p):
    '''struct_initializer : IDENTIFIER COLON expression
                        | IDENTIFIER EQUALS expression'''
    p[0] = ast.StructInitializer(p[1], p[3])

def p_move_expression(p):
    '''move_expression : MOVE LPAREN IDENTIFIER RPAREN'''
    p[0] = ast.Move(p[3])

def p_borrow_shared_expression(p):
    '''borrow_shared_expression : AMPERSAND IDENTIFIER'''
    p[0] = ast.BorrowShared(p[2])

def p_borrow_unique_expression(p):
    '''borrow_unique_expression : AMPERSAND MUT IDENTIFIER'''
    p[0] = ast.BorrowUnique(p[3])

def p_spawn_expression(p):
    '''spawn_expression : SPAWN LPAREN expression RPAREN'''
    p[0] = ast.SpawnExpression(p[3])

def p_vector_literal(p):
    '''vector_literal : VECTOR LBRACKET IDENTIFIER COMMA NUMBER RBRACKET LPAREN element_list RPAREN'''
    p[0] = ast.VectorLiteral(p[3], p[5], p[8])

def p_element_list(p):
    '''element_list : element_list COMMA expression
                    | expression'''
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    else:
        p[0] = [p[1]]

def p_to_device(p):
    '''expression : TO_DEVICE LPAREN IDENTIFIER RPAREN'''
    p[0] = ast.ToDevice(p[3])

def p_from_device(p):
    '''expression : FROM_DEVICE LPAREN IDENTIFIER RPAREN'''
    p[0] = ast.FromDevice(p[3])

def p_empty(p):
    'empty :'
    pass

def p_type_expression(p):
    '''type_expression : IDENT
                      | type_application
                      | recursive_type'''
    if isinstance(p[1], str):
        p[0] = ast.BasicType(p[1])
    else:
        p[0] = p[1]

def p_type_parameter(p):
    '''type_parameter : IDENT
                     | IDENT COLON type_bound_list'''
    if len(p) == 2:
        p[0] = ast.TypeParameter(p[1])
    else:
        p[0] = ast.TypeParameter(p[1], p[3])

def p_type_parameter_list(p):
    '''type_parameter_list : type_parameter
                          | type_parameter_list COMMA type_parameter'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_type_bound_list(p):
    '''type_bound_list : type_expression
                      | type_bound_list PLUS type_expression'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_type_application(p):
    '''type_application : IDENT LT type_argument_list GT'''
    p[0] = ast.TypeApplication(p[1], p[3])

def p_type_argument_list(p):
    '''type_argument_list : type_expression
                         | type_argument_list COMMA type_expression'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_type_definition(p):
    '''type_definition : TYPE IDENT type_params EQUALS type_expression mode_annotations
                      | TYPE IDENT EQUALS type_expression mode_annotations
                      | TYPE IDENT type_params EQUALS type_expression
                      | TYPE IDENT EQUALS type_expression'''
    name = p[2]
    if len(p) == 7:  # With type params and modes
        type_params = p[3]
        body = p[5]
        modes = p[6]
    elif len(p) == 6:  # With type params, no modes
        type_params = p[3]
        body = p[5]
        modes = None
    elif len(p) == 6:  # No type params, with modes
        type_params = None
        body = p[4]
        modes = p[5]
    else:  # No type params, no modes
        type_params = None
        body = p[4]
        modes = None
    p[0] = ast.TypeDefinition(name, type_params, body, modes)

def p_type_params(p):
    '''type_params : LT type_parameter_list GT'''
    p[0] = p[2]

def p_interface_definition(p):
    '''interface_definition : INTERFACE IDENTIFIER type_params_opt LBRACE method_list RBRACE
                           | INTERFACE IDENTIFIER type_params_opt EXTENDS type_list LBRACE method_list RBRACE'''
    if len(p) == 7:
        p[0] = ast.InterfaceDefinition(p[2], p[3], p[5])
    else:
        p[0] = ast.InterfaceDefinition(p[2], p[3], p[7], extends=p[5])

def p_method_list(p):
    '''method_list : method_definition
                  | method_list method_definition'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]

def p_method_definition(p):
    '''method_definition : FN IDENTIFIER type_params_opt LPAREN param_list_opt RPAREN ARROW type_expr
                       | FN IDENTIFIER type_params_opt LPAREN param_list_opt RPAREN'''
    if len(p) == 9:
        p[0] = ast.MethodDefinition(p[2], p[5], p[8], type_params=p[3])
    else:
        p[0] = ast.MethodDefinition(p[2], p[5], None, type_params=p[3])

def p_implementation(p):
    '''implementation : IMPL type_params_opt interface_type FOR type_expr where_clause_opt LBRACE method_impl_list RBRACE'''
    p[0] = ast.Implementation(p[3], p[5], p[2], p[8], where_clause=p[6])

def p_interface_type(p):
    '''interface_type : IDENTIFIER
                    | IDENTIFIER LBRACKET type_list RBRACKET'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = (p[1], p[3])

def p_method_impl_list(p):
    '''method_impl_list : method_implementation
                      | method_impl_list method_implementation'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]

def p_method_implementation(p):
    '''method_implementation : FN IDENTIFIER type_params_opt LPAREN param_list_opt RPAREN block'''
    p[0] = ast.MethodImplementation(p[2], p[5], p[7], type_params=p[3])

def p_where_clause(p):
    '''where_clause : WHERE type_constraint_list'''
    p[0] = ast.WhereClause(p[2])

def p_where_clause_opt(p):
    '''where_clause_opt : where_clause
                      | empty'''
    p[0] = p[1]

def p_type_constraint_list(p):
    '''type_constraint_list : type_constraint
                           | type_constraint_list COMMA type_constraint'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_type_constraint(p):
    '''type_constraint : type_expr COLON type_bound
                     | type_expr EXTENDS type_bound
                     | type_expr IMPLEMENTS type_bound'''
    p[0] = ast.TypeConstraint(p[1], p[3], kind=p[2].lower())

def p_type_bound(p):
    '''type_bound : type_expr
                 | type_bound PLUS type_expr'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = ('compound', p[1], p[3])

def p_type_alias(p):
    '''type_alias : TYPE IDENTIFIER type_params_opt EQUALS type_expr'''
    p[0] = ast.TypeAlias(p[2], p[5], type_params=p[3])

def p_type_params_opt(p):
    '''type_params_opt : type_params
                      | empty'''
    p[0] = p[1]

def p_param_list_opt(p):
    '''param_list_opt : param_list
                     | empty'''
    p[0] = p[1]

def p_type_list(p):
    '''type_list : type_expr
                | type_list COMMA type_expr'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_import_statement(p):
    '''import_statement : IMPORT module_path
                       | IMPORT module_path AS IDENTIFIER'''
    if len(p) == 3:
        p[0] = ast.Import(p[2], None)
    else:
        p[0] = ast.Import(p[2], p[4])

def p_from_import_statement(p):
    '''from_import_statement : FROM relative_path IMPORT import_names
                           | FROM module_path IMPORT import_names'''
    if isinstance(p[2], tuple):
        # Relative import
        level, path = p[2]
        p[0] = ast.FromImport(path, p[4], relative_level=level)
    else:
        # Absolute import
        p[0] = ast.FromImport(p[2], p[4])

def p_relative_path(p):
    '''relative_path : DOT module_path
                    | DOT DOT module_path
                    | DOT DOT DOT module_path'''
    level = len(p) - 2  # Count number of dots
    path = p[len(p)-1]  # Get the module path after dots
    p[0] = (level, path)

def p_module_path(p):
    '''module_path : IDENTIFIER
                  | module_path DOT IDENTIFIER'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_import_names(p):
    '''import_names : IDENTIFIER
                   | IDENTIFIER AS IDENTIFIER
                   | import_names COMMA IDENTIFIER
                   | import_names COMMA IDENTIFIER AS IDENTIFIER'''
    if len(p) == 2:
        p[0] = [(p[1], None)]
    elif len(p) == 4:
        if p[2] == ',':
            p[0] = p[1] + [(p[3], None)]
        else:
            p[0] = [(p[1], p[3])]
    else:
        p[0] = p[1] + [(p[3], p[5])]

def p_error(p):
    if p:
        print(f"Syntax error at line {p.lineno}, position {p.lexpos}: Unexpected token {p.type}")
    else:
        print("Syntax error at EOF")
