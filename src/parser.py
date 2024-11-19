import ply.yacc as yacc
from lexer import Lexer
import metaxu_ast as ast

class Parser:
    def __init__(self):
        self.lexer = Lexer()
        self.tokens = self.lexer.tokens
        self.module_names = set()
        self.parser = yacc.yacc(module=self)

    def parse(self, data):
        self.module_names.clear()  # Reset module names for each parse
        try:
            result = self.parser.parse(data, lexer=self.lexer)
            if result is None:
                return ast.Program([])  # Return empty program instead of None
            return result
        except Exception as e:
            print(f"Error during parsing: {e}")
            raise

    def p_program(self, p):
        '''program : statement_list'''
        p[0] = ast.Program(p[1])

    def p_statement_list(self,p):
        '''statement_list : statement_list statement
                        | statement
                        | empty'''
        if len(p) == 3:
            if p[1] is None:
                p[0] = [p[2]] if p[2] is not None else []
            else:
                p[0] = p[1] + ([p[2]] if p[2] is not None else [])
        else:
            p[0] = [p[1]] if p[1] is not None else []

    def p_statement(self, p):
        '''statement : return_statement
                    | let_statement SEMICOLON
                    | let_statement
                    | for_statement
                    | block
                    | assignment SEMICOLON
                    | assignment
                    | function_declaration
                    | struct_definition
                    | enum_definition
                    | type_definition SEMICOLON
                    | type_definition
                    | interface_definition
                    | implementation
                    | import_statement
                    | from_import_statement
                    | module_declaration
                    | visibility_block
                    | expression SEMICOLON
                    | expression'''
        if len(p) == 3 and isinstance(p[1], ast.LetStatement):
            p[0] = p[1]  # Let statement with semicolon
        elif len(p) == 3 and isinstance(p[1], ast.Assignment):
            p[0] = p[1]  # Assignment with semicolon
        else:
            p[0] = p[1]  # All other statements

    def p_lambda_statement(self, p):
        '''lambda_statement : lambda_expression SEMICOLON
                          | lambda_expression'''
        p[0] = p[1]

    def p_for_statement(self, p):
        '''for_statement : FOR IDENTIFIER IN expression LBRACE statement_list RBRACE'''
        p[0] = ast.ForStatement(p[2], p[4], p[6])

    def p_let_statement(self, p):
        '''let_statement : LET reference_mode IDENTIFIER EQUALS expression SEMICOLON
                        | LET MUT IDENTIFIER EQUALS expression SEMICOLON
                        | LET IDENTIFIER EQUALS expression SEMICOLON
                        | LET reference_mode IDENTIFIER EQUALS expression
                        | LET MUT IDENTIFIER EQUALS expression
                        | LET IDENTIFIER EQUALS expression'''
        
        # Strip off semicolon if present
        has_semicolon = p[-1] == ';'
        effective_length = len(p) - (1 if has_semicolon else 0)
        
        if effective_length == 7:  # With reference mode
            p[0] = ast.LetStatement(identifier=p[3], initializer=p[5], mode=p[2])
        else:  # Simple let statement
            p[0] = ast.LetStatement(identifier=p[2], initializer=p[4])

    def p_return_statement(self, p):
        '''return_statement : RETURN expression_or_empty SEMICOLON
                          | RETURN expression_or_empty'''
        p[0] = ast.ReturnStatement(p[2])

    def p_expression_or_empty(self, p):
        '''expression_or_empty : expression
                             | empty'''
        p[0] = p[1]

    def p_module_declaration(self, p):
        '''module_declaration : MODULE module_path LBRACE module_body RBRACE
                            | MODULE module_path LBRACE RBRACE
                            | MODULE module_path SEMICOLON'''
        module_name = p[2]
        if module_name in self.module_names:
            raise Exception(f"Duplicate module name: {module_name}")
        self.module_names.add(module_name)

        if len(p) == 6:
            p[0] = ast.Module(name=module_name, body=p[4])
        else:
            p[0] = ast.Module(name=module_name, body=ast.Block([]))

    def p_module_path(self, p):
        '''module_path : IDENTIFIER
                      | module_path DOT IDENTIFIER'''
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = f"{p[1]}.{p[3]}"

    def p_module_body(self, p):
        '''module_body : docstring_opt exports_opt statement_list'''
        statements = []
        visibility_rules = None
        
        if p[3]:
            # Extract visibility rules from statements if present
            for stmt in p[3]:
                if isinstance(stmt, ast.VisibilityRules):
                    visibility_rules = stmt.rules
                else:
                    statements.append(stmt)
        
        p[0] = ast.ModuleBody(statements=statements, docstring=p[1], exports=p[2], visibility_rules=visibility_rules)

    def p_docstring_opt(self, p):
        '''docstring_opt : STRING
                        | empty'''
        p[0] = p[1] if p[1] != None else None

    def p_exports_opt(self, p):
        '''exports_opt : EXPORT LBRACE export_list RBRACE
                    | empty'''
        p[0] = p[3] if len(p) == 5 else []

    def p_export_list(self, p):
        '''export_list : export_list COMMA export_item
                    | export_item'''
        if len(p) == 4:
            p[0] = p[1] + [p[3]]
        else:
            p[0] = [p[1]]

    def p_export_item(self, p):
        '''export_item : IDENTIFIER
                    | IDENTIFIER AS IDENTIFIER'''
        if len(p) == 4:
            p[0] = (p[1], p[3])
        else:
            p[0] = (p[1], None)

    def p_visibility_block(self, p):
        '''visibility_block : VISIBILITY LBRACE visibility_rule_list RBRACE'''
        p[0] = ast.VisibilityRules(rules=p[3])

    def p_visibility_rule_list(self, p):
        '''visibility_rule_list : visibility_rule_list COMMA visibility_rule
                            | visibility_rule'''
        if len(p) == 4:
            p[1].update(p[3])
            p[0] = p[1]
        else:
            p[0] = p[1]

    def p_visibility_rule(self, p):
        '''visibility_rule : IDENTIFIER COLON visibility_level'''
        p[0] = {p[1]: p[3]}

    def p_visibility_level(self, p):
        '''visibility_level : PUBLIC
                          | PRIVATE
                          | PROTECTED'''
        p[0] = p[1].lower()

    def p_import_statement(self, p):
        '''import_statement : PUBLIC IMPORT module_path SEMICOLON
                          | PUBLIC IMPORT module_path AS IDENTIFIER SEMICOLON
                          | IMPORT module_path SEMICOLON
                          | IMPORT module_path AS IDENTIFIER SEMICOLON'''
        if len(p) == 7:  # public import with alias
            p[0] = ast.Import(module_path=p[3].split('.'), alias=p[5], is_public=True)
        elif len(p) == 5:  # public import without alias
            p[0] = ast.Import(module_path=p[3].split('.'), alias=None, is_public=True)
        elif len(p) == 6:  # private import with alias
            p[0] = ast.Import(module_path=p[2].split('.'), alias=p[4], is_public=False)
        else:  # private import without alias
            p[0] = ast.Import(module_path=p[2].split('.'), alias=None, is_public=False)

    def p_from_import_statement(self, p):
        '''from_import_statement : PUBLIC FROM relative_path IMPORT import_names SEMICOLON
                                | PUBLIC FROM module_path IMPORT import_names SEMICOLON
                                | FROM relative_path IMPORT import_names SEMICOLON
                                | FROM module_path IMPORT import_names SEMICOLON'''
        if len(p) == 7:  # public from import
            if isinstance(p[3], ast.RelativePath):
                p[0] = ast.FromImport(module_path=p[3].path.split('.'), names=p[5], relative_level=p[3].level, is_public=True)
            else:
                p[0] = ast.FromImport(module_path=p[3].split('.'), names=p[5], is_public=True)
        else:  # private from import
            if isinstance(p[2], ast.RelativePath):
                p[0] = ast.FromImport(module_path=p[2].path.split('.'), names=p[4], relative_level=p[2].level)
            else:
                p[0] = ast.FromImport(module_path=p[2].split('.'), names=p[4])

    def p_relative_path(self, p):
        '''relative_path : DOT module_path
                        | DOT DOT module_path
                        | DOT DOT DOT module_path
                        | TRIPLE_DOT module_path'''
        if len(p) == 3 and p[1] == '.':  # .module_path
            p[0] = ast.RelativePath(1, p[2])
        elif len(p) == 4 and p[1] == '.' and p[2] == '.':  # ..module_path
            p[0] = ast.RelativePath(2, p[3])
        elif len(p) == 5 and p[1] == '.' and p[2] == '.' and p[3] == '.':  # ...module_path
            p[0] = ast.RelativePath(3, p[4])
        else:  # TRIPLE_DOT case
            p[0] = ast.RelativePath(3, p[2])

    def p_import_names(self, p):
        '''import_names : import_name
                    | import_names COMMA import_name'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_import_name(self, p):
        '''import_name : IDENTIFIER
                    | IDENTIFIER AS IDENTIFIER'''
        if len(p) == 2:
            p[0] = (p[1], None)
        else:
            p[0] = (p[1], p[3])

    def p_assignment(self, p):
        '''assignment : IDENTIFIER EQUALS expression
                    | IDENTIFIER EQUALS expression SEMICOLON'''
        if len(p) == 5:  # With semicolon
            p[0] = ast.Assignment(p[1], p[3])
        else:  # Without semicolon
            p[0] = ast.Assignment(p[1], p[3])

    def p_expression(self, p):
        '''expression : term
                    | expression PLUS term
                    | expression MINUS term
                    | expression GREATER term
                    | expression LESS term
                    | expression TIMES term
                    | expression DIVIDE term
                    | LPAREN expression RPAREN'''
        if len(p) == 2:
            p[0] = p[1]
        elif len(p) == 4:
            if p[1] == '(':
                p[0] = p[2]
            else:
                p[0] = ast.BinaryOperation(p[1], p[2], p[3])

    def p_exclave_expression(self, p):
        '''exclave_expression : EXCLAVE expression'''
        p[0] = ast.ExclaveExpression(p[2])

    def p_term(self, p):
        '''term : factor
                | function_call
                | lambda_expression
                | match_expression
                | perform_expression
                | handle_expression
                | spawn_expression
                | exclave_expression
                | borrow_expression
                | qualified_name
                | variant_instantiation
                | vector_literal
                | LPAREN expression RPAREN'''
        if len(p) == 4 and p[1] == '(':  # Parenthesized expression
            p[0] = p[2]
        else:  # All other terms
            p[0] = p[1]

    def p_factor(self, p):
        '''factor : NUMBER
                | FLOAT
                | STRING
                | BOOL
                | NONE
                | SOME LPAREN expression RPAREN
                | NONE LPAREN RPAREN
                | IDENTIFIER
                | IDENTIFIER LPAREN argument_list RPAREN
                | IDENTIFIER LPAREN RPAREN
                | LPAREN expression RPAREN'''
        if len(p) == 2:
            if isinstance(p[1], str):
                if p[1].lower() == 'true':
                    p[0] = ast.Literal(True)
                elif p[1].lower() == 'false':
                    p[0] = ast.Literal(False)
                elif p[1].lower() == 'none':
                    p[0] = ast.NoneExpression()
                else:
                    p[0] = ast.Variable(p[1])
            else:
                p[0] = ast.Literal(p[1])
        elif len(p) == 3:
            p[0] = ast.NoneExpression()
        elif len(p) == 4:
            if p[1] == '(':
                p[0] = p[2]
            else:
                p[0] = ast.FunctionCall(p[1], [])
        elif len(p) == 5:
            if p[1].lower() == 'some':
                p[0] = ast.SomeExpression(p[3])
            else:
                p[0] = ast.FunctionCall(p[1], p[3] if p[3] is not None else [])

    def p_function_call(self, p):
        '''function_call : IDENTIFIER LPAREN argument_list RPAREN
                        | IDENTIFIER LPAREN RPAREN
                        | qualified_name LPAREN argument_list RPAREN
                        | qualified_name LPAREN RPAREN'''
        if isinstance(p[1], ast.QualifiedName):
            if len(p) == 4:  # Empty args
                p[0] = ast.QualifiedFunctionCall(p[1].parts, [])
            else:  # With args
                p[0] = ast.QualifiedFunctionCall(p[1].parts, p[3] if p[3] is not None else [])
        else:
            if len(p) == 4:  # Empty args
                p[0] = ast.FunctionCall(p[1], [])
            else:  # With args
                p[0] = ast.FunctionCall(p[1], p[3] if p[3] is not None else [])

    def p_qualified_name(self, p):
        '''qualified_name : IDENTIFIER
                        | IDENTIFIER DOT IDENTIFIER
                        | qualified_name DOT IDENTIFIER'''
        if len(p) == 2:
            p[0] = ast.QualifiedName([p[1]])
        elif len(p) == 4:
            if isinstance(p[1], ast.QualifiedName):
                parts = p[1].parts + [p[3]]
            else:
                parts = [p[1], p[3]]
            p[0] = ast.QualifiedName(parts)
        else:
            p[0] = ast.QualifiedName(p[1].parts + [p[3]])

    def p_argument_list(self, p):
        '''argument_list : expression
                        | argument_list COMMA expression
                        | empty'''
        if len(p) == 2:
            if p[1] is None:  # empty
                p[0] = []
            else:  # single expression
                p[0] = [p[1]]
        else:  # argument_list COMMA expression
            p[0] = p[1] + [p[3]]

    def p_function_declaration(self, p):
        '''function_declaration : FN IDENTIFIER LPAREN parameter_list RPAREN type_annotation LBRACE statement_list RBRACE
                              | FN IDENTIFIER LPAREN parameter_list RPAREN LBRACE statement_list RBRACE'''
        if len(p) == 10:  # With return type
            p[0] = ast.FunctionDeclaration(p[2], p[4] if p[4] is not None else [], p[8], return_type=p[6])
        else:  # Without return type
            p[0] = ast.FunctionDeclaration(p[2], p[4] if p[4] is not None else [], p[7])

    def p_parameter_list(self, p):
        '''parameter_list : parameter_list COMMA parameter
                        | parameter
                        | empty'''
        if len(p) == 4:
            p[0] = p[1] + [p[3]]
        elif len(p) == 2 and p[1] is None:
            p[0] = []
        else:
            p[0] = [p[1]]

    def p_parameter(self, p):
        '''parameter : IDENTIFIER COLON mode_type_expression
                    | reference_mode IDENTIFIER COLON type_expression
                    | IDENTIFIER'''
        if len(p) == 4:
            p[0] = ast.Parameter(p[1], p[3].base_type, p[3].uniqueness)
        elif len(p) == 5:
            p[0] = ast.Parameter(p[2], p[4], p[1])
        else:  # Just identifier
            p[0] = ast.Parameter(p[1])

    def p_mode_type_expression(self, p):
        '''mode_type_expression : reference_mode type_expression
                               | type_expression'''
        if len(p) == 3:
            p[0] = ast.ModeTypeAnnotation(p[2], uniqueness=p[1])
        else:
            p[0] = ast.ModeTypeAnnotation(p[1])

    def p_reference_mode(self, p):
        '''reference_mode : UNIQUE
                        | SHARED
                        | EXCLUSIVE
                        | BORROW SHARED
                        | BORROW EXCLUSIVE'''
        if len(p) == 2:
            p[0] = ast.UniquenessMode(p[1].lower())
        else:
            mode = ast.UniquenessMode(p[2].lower())
            mode.is_borrowed = True
            p[0] = mode

    def p_type_expression(self, p):
        '''type_expression : IDENTIFIER
                         | type_application
                         | function_type
                         | struct_type
                         | enum_type
                         | array_type
                         | reference_mode type_expression
                         | LPAREN type_expression RPAREN'''
        if len(p) == 4:  # Parenthesized or reference mode
            if p[1] == '(':
                p[0] = p[2]
            else:
                p[0] = ast.TypeApplication("Reference", [p[2]], mode=p[1])
        else:
            p[0] = p[1]

    def p_array_type(self, p):
        '''array_type : LBRACKET RBRACKET type_expression'''
        p[0] = ast.TypeApplication("Array", [p[3]])

    def p_type_application(self, p):
        '''type_application : IDENTIFIER LESS type_argument_list GREATER'''
        p[0] = ast.TypeApplication(p[1], p[3])

    def p_type_argument_list(self, p):
        '''type_argument_list : type_expression
                            | type_argument_list COMMA type_expression'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_type_definition(self, p):
        '''type_definition : TYPE IDENTIFIER LESS type_parameter_list GREATER EQUALS type_expression
                         | TYPE IDENTIFIER EQUALS type_expression'''
        if len(p) == 8:
            p[0] = ast.TypeDefinition(p[2], p[4], p[7])
        else:
            p[0] = ast.TypeDefinition(p[2], None, p[4])

    def p_type_parameter(self, p):
        '''type_parameter : IDENTIFIER mode_annotations
                        | IDENTIFIER'''
        if len(p) == 3:
            p[0] = ast.TypeParameter(p[1], p[2])
        else:
            p[0] = ast.TypeParameter(p[1], None)

    def p_type_parameter_list(self, p):
        '''type_parameter_list : type_parameter
                            | type_parameter_list COMMA type_parameter'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_borrow_expression(self, p):
        '''borrow_expression : BORROW IDENTIFIER AS type_expression
                           | BORROW IDENTIFIER'''
        if len(p) == 5:
            p[0] = ast.BorrowExpression(p[2], p[4])
        else:
            p[0] = ast.BorrowExpression(p[2], None)

    def p_block(self, p):
        '''block : LBRACE statement_list RBRACE'''
        p[0] = ast.Block(p[2] if p[2] is not None else [])

    def p_struct_definition(self, p):
        '''struct_definition : STRUCT IDENTIFIER LBRACE struct_fields RBRACE
                           | STRUCT IDENTIFIER IMPLEMENTS IDENTIFIER LBRACE struct_fields method_impl_list RBRACE'''
        if len(p) == 6:
            p[0] = ast.StructDefinition(name=p[2], fields=p[4])
        else:
            p[0] = ast.StructDefinition(name=p[2], fields=p[6], implements=p[4], methods=p[7])

    def p_struct_fields(self, p):
        '''struct_fields : struct_fields struct_field
                        | struct_field
                        | empty'''
        if len(p) == 3:
            p[0] = p[1] + [p[2]]
        elif len(p) == 2 and p[1] is None:  # empty
            p[0] = []
        else:
            p[0] = [p[1]]

    def p_struct_field(self, p):
        '''struct_field : visibility_modifier IDENTIFIER COLON type_specification
                       | visibility_modifier IDENTIFIER EQUALS expression
                       | IDENTIFIER COLON type_specification
                       | IDENTIFIER EQUALS expression'''
        if len(p) == 5:
            if p[3] == ':':
                p[0] = ast.StructField(name=p[2], type_expr=p[4], visibility=p[1])
            else:  # p[3] == '='
                p[0] = ast.StructField(name=p[2], value=p[4], visibility=p[1])
        else:  # len(p) == 4
            if p[2] == ':':
                p[0] = ast.StructField(name=p[1], type_expr=p[3])
            else:  # p[2] == '='
                p[0] = ast.StructField(name=p[1], value=p[3])

    def p_enum_definition(self, p):
        '''enum_definition : ENUM IDENTIFIER LBRACE variant_list RBRACE'''
        p[0] = ast.EnumDefinition(p[2], p[4])

    def p_variant_list(self, p):
        '''variant_list : variant_definition
                       | variant_list variant_definition'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_variant_definition(self, p):
        '''variant_definition : IDENTIFIER LPAREN variant_fields RPAREN SEMICOLON
                            | IDENTIFIER SEMICOLON'''
        if len(p) == 6:
            p[0] = ast.VariantDefinition(p[1], p[3])
        else:
            p[0] = ast.VariantDefinition(p[1], [])

    def p_variant_fields(self, p):
        '''variant_fields : variant_field
                        | variant_fields COMMA variant_field
                        | empty'''
        if len(p) == 2:
            if p[1] is None:
                p[0] = []
            else:
                p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_variant_field(self, p):
        '''variant_field : IDENTIFIER COLON type_specification'''
        p[0] = (p[1], p[3])

    def p_variant_instantiation(self, p):
        '''variant_instantiation : IDENTIFIER DOUBLECOLON IDENTIFIER LPAREN field_assignments RPAREN
                                | IDENTIFIER DOUBLECOLON IDENTIFIER LPAREN RPAREN'''
        if len(p) == 7:
            p[0] = ast.VariantInstance(p[1], p[3], p[5])
        else:
            p[0] = ast.VariantInstance(p[1], p[3], [])

    def p_field_assignments(self, p):
        '''field_assignments : field_assignment
                           | field_assignments COMMA field_assignment'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_field_assignment(self, p):
        '''field_assignment : IDENTIFIER EQUALS expression'''
        p[0] = (p[1], p[3])

    def p_match_expression(self, p):
        '''match_expression : MATCH expression LBRACE match_cases RBRACE'''
        p[0] = ast.MatchExpression(p[2], p[4])

    def p_match_cases(self, p):
        '''match_cases : option_match_case
                    | match_cases option_match_case'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_option_match_case(self, p):
        '''option_match_case : SOME LPAREN IDENTIFIER RPAREN ARROW expression SEMICOLON
                            | NONE ARROW expression SEMICOLON'''
        if len(p) == 8:  # Some case
            p[0] = ('some', p[3], p[6])
        else:  # None case
            p[0] = ('none', None, p[3])

    def p_option_expression(self, p):
        '''option_expression : NONE
                            | SOME LPAREN expression RPAREN'''
        if len(p) == 2:
            p[0] = ast.NoneExpression()
        else:
            p[0] = ast.SomeExpression(p[3])

    def p_lambda_body(self, p):
        '''lambda_body : LBRACE statement_list RBRACE
                      | LBRACE statement_list RBRACE SEMICOLON
                      | expression SEMICOLON
                      | expression'''
        if len(p) == 4:  # Block without semicolon
            p[0] = ast.Block(p[2] if p[2] is not None else [])
        elif len(p) == 5:  # Block with semicolon
            p[0] = ast.Block(p[2] if p[2] is not None else [])
        else:  # Expression with or without semicolon
            p[0] = p[1]

    def p_lambda_expression(self, p):
        '''lambda_expression : FN LPAREN parameter_list RPAREN ARROW type_expression lambda_body
                           | FN LPAREN parameter_list RPAREN ARROW lambda_body
                           | FN LPAREN RPAREN ARROW type_expression lambda_body
                           | FN LPAREN RPAREN ARROW lambda_body'''
        if len(p) == 7 and p[3] != ')':  # With params, no return type
            p[0] = ast.LambdaExpression(p[3] if p[3] is not None else [], p[6])
        elif len(p) == 8:  # With params and return type
            p[0] = ast.LambdaExpression(p[3] if p[3] is not None else [], p[7], return_type=p[6])
        elif len(p) == 7:  # No params, with return type
            p[0] = ast.LambdaExpression([], p[6], return_type=p[5])
        else:  # No params, no return type
            p[0] = ast.LambdaExpression([], p[5])

    def p_perform_expression(self, p):
        '''perform_expression : PERFORM effect_operation
                            | PERFORM effect_operation SEMICOLON'''
        p[0] = p[2]

    def p_effect_operation(self, p):
        '''effect_operation : IDENTIFIER DOT IDENTIFIER LPAREN argument_list RPAREN'''
        p[0] = ast.EffectOperation(p[1], p[3], p[5])

    def p_struct_instantiation(self, p):
        '''struct_instantiation : IDENTIFIER LBRACE field_assignments RBRACE
                               | qualified_name LPAREN argument_list RPAREN
                               | qualified_name LBRACE field_assignments RBRACE'''
        if len(p) == 5:
            if p[2] == '(':  # Function-style constructor
                p[0] = ast.QualifiedFunctionCall(p[1].parts, p[3])
            else:  # Record-style constructor
                if isinstance(p[1], ast.QualifiedName):
                    p[0] = ast.StructInstantiation(p[1], p[3])
                else:
                    p[0] = ast.StructInstantiation(ast.QualifiedName([p[1]]), p[3])

    def p_field_access(self, p):
        '''field_access : expression DOT IDENTIFIER
                       | field_access DOT IDENTIFIER'''
        if isinstance(p[1], ast.QualifiedName):
            # Convert QualifiedName to field access
            p[0] = ast.FieldAccess(p[1].parts[0], p[1].parts[1:] + [p[3]])
        elif isinstance(p[1], ast.FieldAccess):
            # Append to existing field access
            p[0] = ast.FieldAccess(p[1].base, p[1].fields + [p[3]])
        else:
            # Start new field access
            p[0] = ast.FieldAccess(p[1], [p[3]])

    def p_method_impl_list(self, p):
        '''method_impl_list : method_implementation
                        | method_impl_list method_implementation
                        | empty'''
        if len(p) == 2:
            if p[1] is None:  # empty
                p[0] = []
            else:
                p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_method_implementation(self, p):
        '''method_implementation : FN IDENTIFIER type_params_opt LPAREN param_list_opt RPAREN ARROW type_expression block
                               | FN IDENTIFIER type_params_opt LPAREN param_list_opt RPAREN block'''
        if len(p) == 9:
            p[0] = ast.MethodImplementation(p[2], p[5], p[8], type_params=p[3])
        else:
            p[0] = ast.MethodImplementation(p[2], p[5], p[7], type_params=p[3], return_type=p[8])

    def p_interface_definition(self, p):
        '''interface_definition : INTERFACE IDENTIFIER type_params_opt LBRACE method_list RBRACE
                            | INTERFACE IDENTIFIER type_params_opt EXTENDS type_list LBRACE method_list RBRACE'''
        if len(p) == 7:
            p[0] = ast.InterfaceDefinition(p[2], p[3], p[5])
        else:
            p[0] = ast.InterfaceDefinition(p[2], p[3], p[7], extends=p[5])

    def p_method_list(self, p):
        '''method_list : method_definition
                    | method_list method_definition'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_method_definition(self, p):
        '''method_definition : FN IDENTIFIER type_params_opt LPAREN param_list_opt RPAREN ARROW type_expression
                           | FN IDENTIFIER type_params_opt LPAREN param_list_opt RPAREN'''
        if len(p) == 9:
            p[0] = ast.MethodDefinition(p[2], p[5], p[8], type_params=p[3])
        else:
            p[0] = ast.MethodDefinition(p[2], p[5], None, type_params=p[3])

    def p_implementation(self, p):
        '''implementation : IMPL type_params_opt interface_type FOR type_expression where_clause_opt LBRACE method_impl_list RBRACE
                        | IMPL interface_type LBRACE method_impl_list RBRACE'''
        if len(p) == 9:
            p[0] = ast.Implementation(p[3], p[5], p[2], p[8], where_clause=p[6])
        else:
            p[0] = ast.Implementation(p[2], None, None, p[4])

    def p_interface_type(self, p):
        '''interface_type : IDENTIFIER
                        | IDENTIFIER LESS type_list GREATER'''
        if len(p) == 2:
            p[0] = ast.InterfaceType(p[1], None)
        else:
            p[0] = ast.InterfaceType(p[1], p[3])

    def p_where_clause(self, p):
        '''where_clause : WHERE type_constraint_list'''
        p[0] = ast.WhereClause(p[2])

    def p_where_clause_opt(self, p):
        '''where_clause_opt : where_clause
                        | empty'''
        p[0] = p[1]

    def p_type_constraint_list(self, p):
        '''type_constraint_list : type_constraint
                            | type_constraint_list COMMA type_constraint'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_type_constraint(self, p):
        '''type_constraint : type_expression COLON type_bound
                         | type_expression EXTENDS type_bound
                         | type_expression IMPLEMENTS type_bound
                         | type_expression EQUALS type_expression'''
        if p[2] == '=':
            p[0] = ast.TypeConstraint(p[1], 'equals', p[3])
        else:
            p[0] = ast.TypeConstraint(p[1], p[2].lower(), p[3])

    def p_type_alias(self, p):
        '''type_alias : TYPE IDENTIFIER type_params_opt EQUALS type_expression'''
        p[0] = ast.TypeAlias(p[2], p[5], type_params=p[3])

    def p_type_params(self, p):
        '''type_params : LESS type_parameter_list GREATER'''
        p[0] = p[2]

    def p_param_list(self, p):
        '''param_list : parameter_list'''
        p[0] = p[1]

    def p_type_params_opt(self, p):
        '''type_params_opt : type_params
                        | empty'''
        p[0] = p[1]

    def p_param_list_opt(self, p):
        '''param_list_opt : param_list
                        | empty'''
        p[0] = p[1]

    def p_type_list(self, p):
        '''type_list : type_expression
                    | type_list COMMA type_expression'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_function_type(self, p):
        '''function_type : FN BACKSLASH LPAREN type_list RPAREN ARROW type_expression
                        | FN LPAREN type_list RPAREN ARROW type_expression
                        | FN BACKSLASH LPAREN RPAREN ARROW type_expression
                        | FN LPAREN RPAREN ARROW type_expression'''
        if len(p) == 8:  # With params and backslash
            p[0] = ast.FunctionType(p[4], p[7])
        elif len(p) == 7:  # With params, no backslash
            p[0] = ast.FunctionType(p[3], p[6])
        elif len(p) == 7:  # No params, with backslash
            p[0] = ast.FunctionType([], p[6])
        else:  # No params, no backslash
            p[0] = ast.FunctionType([], p[5])

    def p_struct_type(self, p):
        '''struct_type : STRUCT LBRACE struct_field_list RBRACE'''
        p[0] = ast.StructType(p[3])

    def p_struct_field_list(self, p):
        '''struct_field_list : struct_field
                           | struct_field_list COMMA struct_field'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_enum_type(self, p):
        '''enum_type : ENUM LBRACE enum_variant_list RBRACE'''
        p[0] = ast.EnumType(p[3])

    def p_enum_variant_list(self, p):
        '''enum_variant_list : enum_variant
                           | enum_variant_list COMMA enum_variant'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_enum_variant(self, p):
        '''enum_variant : IDENTIFIER
                      | IDENTIFIER LPAREN type_list RPAREN'''
        if len(p) == 2:
            p[0] = ast.EnumVariant(p[1], None)
        else:
            p[0] = ast.EnumVariant(p[1], p[3])

    def p_mode_annotations(self, p):
        '''mode_annotations : mode_annotation
                          | mode_annotations mode_annotation'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_mode_annotation(self, p):
        '''mode_annotation : AT IDENTIFIER'''
        p[0] = ast.ModeAnnotation(p[2])

    def p_type_annotation(self, p):
        '''type_annotation : ARROW type_expression'''
        p[0] = p[2]

    def p_handle_expression(self, p):
        '''handle_expression : HANDLE expression WITH LBRACE handle_cases RBRACE'''
        p[0] = ast.HandleExpression(p[2], p[5])

    def p_handle_cases(self, p):
        '''handle_cases : handle_case
                       | handle_cases handle_case'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_handle_case(self, p):
        '''handle_case : IDENTIFIER LPAREN IDENTIFIER RPAREN ARROW expression SEMICOLON
                      | IDENTIFIER LPAREN IDENTIFIER RPAREN ARROW block'''
        p[0] = ast.HandleCase(p[1], p[3], p[6])

    def p_type_specification(self, p):
        '''type_specification : type_expression
                            | type_specification DOT IDENTIFIER
                            | type_specification LBRACKET type_specification RBRACKET'''
        if len(p) == 2:
            p[0] = p[1]
        elif p[2] == '.':
            p[0] = ast.QualifiedType(p[1], p[3])
        else:  # p[2] == '['
            p[0] = ast.ArrayType(p[1], p[3])

    def p_type_constraints(self, p):
        '''type_constraints : type_constraint
                          | type_constraints COMMA type_constraint'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_type_bound(self, p):
        '''type_bound : type_expression
                     | type_bound PLUS type_expression'''
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = ast.CompoundTypeBound(p[1], p[3])

    def p_visibility_block(self, p):
        '''visibility_block : VISIBILITY LBRACE visibility_rule_list RBRACE'''
        p[0] = ast.VisibilityRules(rules=p[3])

    def p_visibility_modifier(self, p):
        '''visibility_modifier : PUBLIC
                             | PRIVATE
                             | PROTECTED'''
        p[0] = p[1]

    def p_move_expression(self, p):
        '''move_expression : MOVE LPAREN IDENTIFIER RPAREN'''
        p[0] = ast.Move(p[3])

    def p_borrow_shared_expression(self, p):
        '''borrow_shared_expression : AMPERSAND IDENTIFIER'''
        p[0] = ast.BorrowShared(p[2])

    def p_borrow_unique_expression(self, p):
        '''borrow_unique_expression : AMPERSAND MUT IDENTIFIER'''
        p[0] = ast.BorrowUnique(p[3])

    def p_spawn_expression(self, p):
        '''spawn_expression : SPAWN LPAREN expression RPAREN'''
        p[0] = ast.SpawnExpression(p[3])

    def p_vector_literal(self, p):
        '''vector_literal : VECTOR LBRACKET IDENTIFIER COMMA NUMBER RBRACKET LPAREN element_list RPAREN
                        | VECTOR LBRACKET IDENTIFIER COMMA NUMBER RBRACKET
                        | VECTOR LBRACKET IDENTIFIER RBRACKET'''
        if len(p) == 10:  # Full form with element list
            p[0] = ast.VectorLiteral(p[3], p[5], p[8])
        elif len(p) == 7:  # With size but no elements
            p[0] = ast.VectorLiteral(p[3], p[5], [])
        else:  # Just type
            p[0] = ast.VectorLiteral(p[3], None, [])

    def p_element_list(self, p):
        '''element_list : element_list COMMA expression
                       | expression'''
        if len(p) == 4:
            p[0] = p[1] + [p[3]]
        else:
            p[0] = [p[1]]

    def p_to_device(self, p):
        '''expression : TO_DEVICE LPAREN IDENTIFIER RPAREN'''
        p[0] = ast.ToDevice(p[3])

    def p_from_device(self, p):
        '''expression : FROM_DEVICE LPAREN IDENTIFIER RPAREN'''
        p[0] = ast.FromDevice(p[3])


    def p_empty(self, p):
        'empty :'
        pass

    def p_error(self, p):
        if p:
            print(f"Syntax error at token {p.type}, value '{p.value}', line {p.lineno}, position {p.lexpos}")
            # Print the next few tokens to help with debugging
            tok = self.parser.token()
            print("Next tokens:")
            for _ in range(5):  # Print next 5 tokens
                if tok:
                    print(f"  {tok.type}: {tok.value}")
                    tok = self.parser.token()
        else:
            print("Syntax error at EOF")
        # Don't return None here, instead raise an exception to be caught by parse()
        raise SyntaxError("Failed to parse input")
