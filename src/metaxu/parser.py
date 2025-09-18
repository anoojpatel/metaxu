import ply.yacc as yacc
from metaxu.lexer import Lexer
import metaxu.metaxu_ast as ast
from metaxu.decorator_ast import Decorator, CFunctionDecorator, DecoratorList
from metaxu.extern_ast import ExternBlock, ExternFunctionDeclaration, ExternTypeDeclaration
from metaxu.unsafe_ast import (UnsafeBlock, PointerType, TypeCast, 
                       PointerDereference, AddressOf)
from metaxu.type_defs import (SharedType, BoxType, ReferenceType,NoneType)
from metaxu.errors import CompileError, SourceLocation, get_source_context
import traceback
import logging

# Configure logging with a custom format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Print to stdout
    ]
)
logger = logging.getLogger(__name__)

scoped_nodes = (ast.FunctionDeclaration, ast.LambdaExpression, ast.Block, ast.WhileStatement, ast.ForStatement,ast.ModuleBody)

class Parser:
    start = 'program'

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.precedence = (
            ('left', 'PLUS', 'MINUS'),
            ('left', 'TIMES', 'DIVIDE'),
        )
        
        # Initialize deferred processing system
        self.deferred_processing = []
        
        # Initialize the lexer
        self.lexer = Lexer()
        self.tokens = self.lexer.tokens  # Get token list from lexer
        self.parser = yacc.yacc(module=self)
        self.module_names = set()
        self.parse_stack = []
        self.current_scope = None
        self.scope_stack = []  # Stack to track nested scopes
        self.logger = logging.getLogger(__name__)
        # Explicitly add a StreamHandler to ensure output to stdout
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.WARNING)
        self.logger.debug("Parser initialized with debug logging enabled.")
        self.current_module = None
        
    def log_rule(self, rule_name, p=None):
        try:
            if p:
                tokens = [f"{tok.type}({tok.value})" for tok in p.slice[1:]]
                print(f"\n=== Parser Rule: {rule_name} ===")
                print(f"Tokens: {' '.join(tokens)}")
                print(f"Production: {p.slice[0].type} -> {' '.join(tok.type for tok in p.slice[1:])}")
                
                # Store location info in the resulting AST node if it has a location attribute
                if hasattr(p[0], 'location') and len(p.slice) > 1:
                    first_token = p.slice[1]
                    p[0].line = first_token.lineno
                    p[0].column = first_token.column if hasattr(first_token, 'column') else 0
                    
        except Exception as e:
            print(f"Error in log_rule: {e}")
        production = f" -> {p.slice[1].type}" if p and len(p.slice) > 1 else ""
        self.logger.debug(f"Entering {rule_name}{production}")
        print(f"PARSER DEBUG: {rule_name}{production}")  # Direct print for immediate visibility

    def parse(self, source: str, file_path: str = "<unknown>") -> 'ast.Module':
        """Parse source code into an AST"""
        try:
            print("\n=== Starting Parse ===")
            # Initialize lexer with source
            self.lexer.source_file = file_path
            self.lexer.input(source)  # Make sure lexer has the source
            self._enter_scope(ast.Scope(name="global"))
            # Parse using PLY
            result = self.parser.parse(source, lexer=self.lexer, debug=False)  # Enable debug here too
            
            # If the result is a list of statements and not a module, wrap it in a module
            if isinstance(result, list) and result and not any(isinstance(stmt, ast.Module) for stmt in result):
                module_body = ast.ModuleBody(statements=result)
                result = ast.Module(name="main", body=module_body)
            
            # Set source file for all modules
            if isinstance(result, ast.Module): 
                result.source_file = file_path
            elif isinstance(result, list):
                for module in result:
                    if isinstance(module, ast.Module):
                        module.source_file = file_path
                        
            self._exit_scope()
            # Process any pending lambda expressions
            #self.process_pending_lambdas()
            # Process deferred items
            self.process_deferred()
            return result
            
        except Exception as e:
            # Get current token for error location
            token = self.lexer.current_token if hasattr(self.lexer, 'current_token') else None
            print(f"\n=== Parser Error ===")
            
            # Get parser state information if available
            if hasattr(self.parser, 'symstack'):
                stack_state = self.parser.symstack
                stack_str = ' '.join([str(sym.type) for sym in stack_state[1:]])
                print(f"Parser stack: {stack_str}")
                
                # Get expected tokens in current state
                if self.parser.state < len(self.parser.action):
                    state = self.parser.state
                    expected = [token for token in self.parser.action[state].keys() 
                              if isinstance(token, str) and token != 'error']
                    print(f"Expected one of: {', '.join(expected)}")
            
            print(f"Current token: {token}")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            print(f"Error hierarchy: {type(e).__mro__}")
            location = SourceLocation(
                file=file_path,
                line=token.lineno if token else 0,
                column=token.column if token else 0
            ) if token else None
            
            error = CompileError(
                message=str(e),
                error_type="ParseError",
                location=location,
                context=get_source_context(file_path, location.line) if location else None,
                stack_trace=traceback.format_stack(),
                notes=["Check syntax near this location"]
            )
            print(f"Created CompileError: {error}")
            raise error from e
        finally:
            print("\n=== Exiting Parse ===")
            self._exit_scope() if self.current_scope else None

    #def p_module(self, p):
    #    '''module : module_body'''
    #    p[0] = ast.Module(
    #        name=None,  # Module name will be set later
    #        body=p[1]
    #    )

    # def p_module_body(self, p):
        # '''module_body : docstring_opt exports_opt statement_list'''
        # statements = []
        # visibility_rules = None
        
        # if p[3]:
            # # Extract visibility rules from statements if present
            # for stmt in p[3]:
                # if isinstance(stmt, ast.VisibilityRules):
                    # visibility_rules = stmt.rules
                # else:
                    # statements.append(stmt)
                
        # p[0] = ast.ModuleBody(
            # statements=statements,
            # docstring=p[1],
            # exports=p[2],
            # visibility_rules=visibility_rules
        # )
    def p_program(self, p):
        '''program : statement_list'''
        p[0] = p[1]

    def p_statement_list(self, p):
        '''statement_list : statements
                        | empty'''
        p[0] = p[1] if p[1] else []

    def p_statements(self, p):
        '''statements : statement
                    | statements statement'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_statement(self, p):
        '''statement : return_statement
                    | let_statement
                    | assignment
                    | type_definition
                    | print_statement
                    | expression
                    | function_declaration
                    | struct_definition
                    | struct_instantiation
                    | enum_definition
                    | interface_definition
                    | implementation
                    | import_statement
                    | from_import_statement
                    | module_declaration
                    | visibility_block
                    | unsafe_block
                    | effect_declaration
                    | block
                    | for_statement
                    | comptime_block
                    | comptime_function
                    | extern_block'''
        p[0] = p[1]

    def p_lambda_statement(self, p):
        '''lambda_statement : lambda_expression'''
        p[0] = p[1]

    def p_for_statement(self, p):
        '''for_statement : FOR IDENTIFIER IN expression LBRACE statement_list RBRACE'''
        p[0] = ast.ForStatement(p[2], p[4], p[6])

    def p_let_binding(self, p):
        '''let_binding : LET mode_annotation_list IDENTIFIER EQUALS expression
                      | LET mode_annotation_list IDENTIFIER type_annotation EQUALS expression
                      | LET IDENTIFIER EQUALS expression
                      | LET IDENTIFIER type_annotation EQUALS expression'''
        if len(p) == 7:  # let @mode x: T = e
            p[0] = ast.LetBinding(p[3], p[6], mode=p[2], type_annotation=p[4])
        elif len(p) == 6:  # let @mode x = e or let x: T = e
            if isinstance(p[2], ast.ModeAnnotation):
                p[0] = ast.LetBinding(p[3], p[5], mode=p[2])
            else:
                p[0] = ast.LetBinding(p[2], p[5], type_annotation=p[3])
        else:  # let x = e
            p[0] = ast.LetBinding(p[2], p[4])

    def p_let_statement(self, p):
        '''let_statement : let_binding
                        | let_binding COMMA let_bindings'''
        if len(p) == 2:
            p[0] = ast.LetStatement(bindings=[p[1]])
        else:
            p[0] = ast.LetStatement(bindings=[p[1]] + p[3])

    def p_let_bindings(self, p):
        '''let_bindings : let_binding
                       | let_binding COMMA let_bindings'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_mode_annotation_list(self, p):
        '''mode_annotation_list : mode_annotation
                               | mode_annotation_list mode_annotation'''
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = p[1].combine(p[2])
 

    def p_return_statement(self, p):
        '''return_statement : RETURN expression_or_empty'''
        p[0] = ast.ReturnStatement(p[2])

    def p_expression_or_empty(self, p):
        '''expression_or_empty : expression
                             | empty'''
        p[0] = p[1]

    def p_module_declaration(self, p):
        '''module_declaration : MODULE module_path LBRACE module_body RBRACE
                            | MODULE module_path LBRACE RBRACE
                            | MODULE module_path'''
        name = p[2]
        
        # Check for duplicate module names
        if name in self.module_names:
            self.logger.error(f"Duplicate module name '{name}'")
            # Create a CompileError directly
            raise CompileError(
                message=f"Duplicate module name '{name}'",
                error_type="ParseError",
                location=SourceLocation(
                    file=self.lexer.source_file,
                    line=p.lineno(1),  # Get line number of MODULE token
                    column=p.slice[1].column if hasattr(p.slice[1], 'column') else 0  # Get column from token
                ),
                context=get_source_context(self.lexer.source_file, p.lineno(1)),
                stack_trace=traceback.format_stack(),
                notes=[f"Module '{name}' was already declared"]
            )
        
        # Add the module name to our set
        self.module_names.add(name)
        
        # Create a module body with empty statements if none provided
        body = ast.ModuleBody(statements=[]) if len(p) == 3 or len(p) == 5 else p[4]
        
        # Create the module with the body
        module = ast.Module(name=name, body=body)
        
        # Set up scope
        self._enter_scope(ast.Scope())
        
        self._exit_scope()
        p[0] = module

    def p_module_path(self, p):
        '''module_path : IDENTIFIER
                      | module_path DOT IDENTIFIER'''
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = f"{p[1]}.{p[3]}"

    def p_module_body(self, p):
        '''module_body : exports statement_list'''
        statements = []
        visibility_rules = None
        
        if p[2]:
            # Extract visibility rules from statements if present
            for stmt in p[2]:
                if isinstance(stmt, ast.VisibilityRules):
                    visibility_rules = stmt
                else:
                    statements.append(stmt)
               
        p[0] = ast.ModuleBody(statements=statements, exports=p[1], visibility_rules=visibility_rules)



    def p_exports(self, p):
        '''exports : EXPORT LBRACE export_list RBRACE
                  | empty'''
        if len(p) == 5:
            p[0] = p[3]
        else:
            p[0] = []

    def p_export_list(self, p):
        '''export_list : export_item
                      | export_list COMMA export_item'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_export_item(self, p):
        '''export_item : IDENTIFIER
                      | IDENTIFIER AS IDENTIFIER'''
        if len(p) == 2:
            p[0] = (p[1], None)
        else:
            p[0] = (p[1], p[3])

    def p_visibility_rule_list(self, p):
        '''visibility_rule_list : visibility_rule
                            | visibility_rule_list COMMA visibility_rule'''
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
        '''import_statement : PUBLIC IMPORT module_path
                          | PUBLIC IMPORT module_path AS IDENTIFIER
                          | IMPORT module_path
                          | IMPORT module_path AS IDENTIFIER'''
        if len(p) == 6:  # public import with alias
            p[0] = ast.Import(module_path=p[3].split('.'), alias=p[5], is_public=True)
        elif len(p) == 4:  # public import without alias
            p[0] = ast.Import(module_path=p[3].split('.'), alias=None, is_public=True)
        elif len(p) == 5:  # private import with alias
            p[0] = ast.Import(module_path=p[2].split('.'), alias=p[4], is_public=False)
        else:  # private import without alias
            p[0] = ast.Import(module_path=p[2].split('.'), alias=None, is_public=False)

    def p_from_import_statement(self, p):
        '''from_import_statement : PUBLIC FROM relative_path IMPORT import_names
                                | PUBLIC FROM module_path IMPORT import_names
                                | FROM relative_path IMPORT import_names
                                | FROM module_path IMPORT import_names'''
        if len(p) == 6:  # public from import
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
        '''assignment : IDENTIFIER EQUALS expression'''
        p[0] = ast.Assignment(p[1], p[3])

    def p_expression(self, p):
        '''expression : comparison_expression'''
        p[0] = p[1]

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
                | struct_instantiation
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
            if isinstance(p[1], tuple) and p[1][1] == 'string':
                p[0] = ast.Literal(p[1][0])  # Create string literal
            elif isinstance(p[1], str):
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
            p[0] = ast.FunctionCall(p[1], p[3] if p[3] is not None else [])

    def p_function_call(self, p):
        '''function_call : IDENTIFIER LPAREN argument_list RPAREN
                        | IDENTIFIER LPAREN RPAREN
                        | qualified_name LPAREN argument_list RPAREN
                        | qualified_name LPAREN RPAREN'''
        if len(p) == 5:  # With arguments
            if isinstance(p[1], str):
                # Handle built-in functions like print specially
                if p[1] == 'print':
                    p[0] = ast.PrintStatement(p[3] if p[3] is not None else [])
                else:
                    p[0] = ast.FunctionCall(p[1], p[3] if p[3] is not None else [])
            elif isinstance(p[1], ast.QualifiedName):
                p[0] = ast.QualifiedFunctionCall(p[1].parts, p[3] if p[3] is not None else [])
        else:  # No arguments (len(p) == 4)
            if isinstance(p[1], str):
                if p[1] == 'print':
                    p[0] = ast.PrintStatement([])
                else:
                    p[0] = ast.FunctionCall(p[1], [])
            elif isinstance(p[1], ast.QualifiedName):
                p[0] = ast.QualifiedFunctionCall(p[1].parts, [])

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
        '''function_declaration : FN IDENTIFIER type_param_list LPAREN parameter_list RPAREN type_annotation_opt LBRACE statement_list RBRACE
                              | FN IDENTIFIER type_param_list LPAREN parameter_list RPAREN LBRACE statement_list RBRACE
                              | FN IDENTIFIER type_param_list LPAREN RPAREN type_annotation_opt LBRACE statement_list RBRACE
                              | FN IDENTIFIER type_param_list LPAREN RPAREN LBRACE statement_list RBRACE
                              | FN IDENTIFIER LPAREN parameter_list RPAREN type_annotation_opt LBRACE statement_list RBRACE
                              | FN IDENTIFIER LPAREN parameter_list RPAREN LBRACE statement_list RBRACE
                              | FN IDENTIFIER LPAREN RPAREN type_annotation_opt LBRACE statement_list RBRACE
                              | FN IDENTIFIER LPAREN RPAREN LBRACE statement_list RBRACE'''
        
        
        # Create function node
        func = ast.FunctionDeclaration(name=p[2], params=[], body=[])
        
        # Create function scope
        func.scope = ast.Scope(name=f"function_{p[2]}")
        self._enter_scope(func.scope)
        # Parse function parameters and add them to scope
        if len(p) == 11:  # Full form with type params
            func.type_params = p[3]
            func.params = p[5] if p[5] else []
            func.return_type = p[7]
            func.body = p[9] if p[9] else []
        elif len(p) == 10:  # No return type
            func.type_params = p[3] if p[3] != "(" else None
            func.return_type = p[6] if p[6] else NoneType
            func.params = p[4] if p[4] else []
            func.body = p[8] if p[8] else []
        elif len(p) == 9:  # No params
            if p[3] == '(':  # No type params
                func.params = []
                func.return_type = p[6]
                func.body = p[7] if p[7] else []
            else:  # Has type params
                func.type_params = p[3]
                func.params = []
                func.body = p[7] if p[7] else []
        else:  # Simplest form
            func.body = p[6] if p[6] else []
            
        # Add symbols and update child scopes
        self._populate_scope_symbols(func, func.scope)
        
        # Update any lambda scopes in the body to point to this function's scope
        self._update_child_scopes(func)
            
        self._exit_scope()
        p[0] = func
        
    def p_type_param_list(self, p):
        '''type_param_list : LESS type_params GREATER
                          | empty'''
        if len(p) > 2:
            p[0] = p[2]
        else:
            p[0] = []

    def p_type_params(self, p):
        '''type_params : type_param
                      | type_params COMMA type_param'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_type_param(self, p):
        '''type_param : IDENTIFIER
                     | IDENTIFIER COLON type_constraint'''
        if len(p) == 2:
            p[0] = ast.TypeParameter(p[1], None)
        else:
            p[0] = ast.TypeParameter(p[1], p[3])

    def p_parameter(self, p):
        '''parameter : IDENTIFIER COLON type_expression
                    | IDENTIFIER mode_annotation_list COLON type_expression'''
        if len(p) == 4:  # x: T
            p[0] = ast.Parameter(p[1], p[3])
        else:  # x @mode: T
            p[0] = ast.Parameter(p[1], p[4], mode=p[2])

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

    def p_mode_type_expression(self, p):
        '''mode_type_expression : reference_mode type_expression
                               | type_expression'''
        if len(p) == 3:
            p[0] = ast.ModeTypeAnnotation(p[2], uniqueness=p[1])
        else:
            p[0] = ast.ModeTypeAnnotation(p[1])

    def p_reference_mode(self, p):
        '''reference_mode : UNIQUE
                        | CONST
                        | EXCLUSIVE
                        | BORROW CONST
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
                         | LPAREN type_expression RPAREN
                         | type_expression mode_annotation_list'''
        if len(p) == 4:  # Parenthesized, reference mode, or mode annotation
            if p[1] == '(':
                p[0] = p[2]
            elif p[2] == '@':  # type @ mode
                p[0] = ast.TypeWithMode(p[1], p[3])
            else:  # reference_mode type
                p[0] = ast.TypeApplication("Reference", [p[2]], mode=p[1])
        elif len(p) == 2 and isinstance(p[1], str):  # Simple type name (IDENTIFIER)
            p[0] = ast.TypeReference(p[1])
        else:
            p[0] = p[1]

    def p_array_type(self, p):
        '''array_type : LBRACKET RBRACKET type_expression'''
        p[0] = ast.TypeApplication("Array", [p[3]])

    def p_type_application(self, p):
        '''type_application : IDENTIFIER LBRACKET type_argument_list RBRACKET'''
        p[0] = ast.TypeApplication(p[1], p[3])

    def p_type_argument_list(self, p):
        '''type_argument_list : type_list'''
        p[0] = p[1]

    def p_type_list(self, p):
        '''type_list : type_expression
                    | type_list COMMA type_expression'''
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
        block = ast.Block(statements=[])
        self._enter_scope(ast.Scope())
        block.scope = self.current_scope
        if p[2]:
            block.statements = [block.add_child(stmt) for stmt in p[2]]
        self._exit_scope()
        p[0] = block

    def p_struct_definition(self, p):
        '''struct_definition : STRUCT IDENTIFIER type_params_opt LBRACE struct_fields RBRACE
                           | STRUCT IDENTIFIER  type_params_opt IMPLEMENTS IDENTIFIER LBRACE struct_fields method_impl_list RBRACE'''
        if len(p) == 7:
            # STRUCT IDENTIFIER type_params_opt LBRACE struct_fields RBRACE
            # indexes:   1       2           3           4       5            6
            p[0] = ast.StructDefinition(name=p[2], fields=p[5])
        else:
            # STRUCT IDENTIFIER type_params_opt IMPLEMENTS IDENTIFIER LBRACE struct_fields method_impl_list RBRACE
            # indexes:   1       2           3              4         5         6       7             8                9
            p[0] = ast.StructDefinition(name=p[2], fields=p[7], implements=p[5], methods=p[8])

    def p_struct_fields(self, p):
        '''struct_fields : struct_fields COMMA struct_field
                        | struct_field
                        | empty'''
        if len(p) == 4:
            p[0] = p[1] + [p[3]]
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
                p[0] = ast.StructField(name=p[2], type_info=p[4], visibility=p[1])
            else:  # p[3] == '='
                    p[0] = ast.StructField(name=p[2], value=p[4], visibility=p[1])
        else:  # len(p) == 4
            if p[2] == ':':
                p[0] = ast.StructField(name=p[1], type_info=p[3])
            else:  # p[2] == '='
                p[0] = ast.StructField(name=p[1], value=p[3])

    def p_enum_definition(self, p):
        '''enum_definition : ENUM IDENTIFIER LBRACE variant_list RBRACE'''
        p[0] = ast.EnumDefinition(p[2], p[4])

    def p_variant_list(self, p):
        '''variant_list : variant_definition
                       | variant_list COMMA variant_definition'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_variant_definition(self, p):
        '''variant_definition : IDENTIFIER LPAREN variant_fields RPAREN
                            | IDENTIFIER'''
        if len(p) == 6:
            p[0] = ast.VariantDefinition(p[1], p[3])
        elif len(p) == 2:
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
        '''option_match_case : SOME LPAREN IDENTIFIER RPAREN ARROW expression
                            | NONE ARROW expression'''
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
                      | expression'''
        if len(p) == 4:  # Block without semicolon
            p[0] = ast.Block(p[2] if p[2] is not None else [])
        else:  # Expression with or without semicolon
            p[0] = p[1]

    def p_lambda_expression(self, p):
        '''lambda_expression : FN LPAREN parameter_list RPAREN type_annotation lambda_body
                           | FN LPAREN parameter_list RPAREN lambda_body
                           | FN LPAREN RPAREN type_annotation lambda_body
                           | FN LPAREN RPAREN lambda_body'''
        #lambda_expr = ast.LambdaExpression(params=[], body=None)
        #self._enter_scope(ast.Scope())
        
        # Debug parent scope before parsing lambda
        self.logger.debug(f"Current scope: {self.current_scope}")
        if self.current_scope:
            self.logger.debug(f"Parent scope: {self.current_scope.parent}")
            if self.current_scope.parent and hasattr(self.current_scope.parent, 'declarations'):
                self.logger.debug(f"Parent scope declarations: {[decl.name for decl in self.current_scope.parent.declarations if hasattr(decl, 'name')]}")
        
        # Debug parent scope declarations
        if self.current_scope.parent and hasattr(self.current_scope.parent, 'declarations'):
            parent_declarations = [decl.name for decl in self.current_scope.parent.declarations if hasattr(decl, 'name')]
            self.logger.debug(f"Parent scope declarations: {parent_declarations}")
            if 'x' not in parent_declarations or 'y' not in parent_declarations:
                self.logger.warning("Variables 'x' or 'y' not found in parent scope declarations.")
        
        if len(p) == 7:  # With params and return type
            lambda_expr = ast.LambdaExpression(params = p[3] if p[3] else [], return_type=p[5], body=p[6])
        elif len(p) == 6:  # With params, no return type
            lambda_expr = ast.LambdaExpression(params = p[3] if p[3] != ")" else [], body=p[5])
        elif len(p) == 6:  # No params, with return type
            lambda_expr = ast.LambdaExpression(params= [], return_type = p[4], body=p[5])
        else:  # No params, no return type
            lambda_expr = ast.LambdaExpression(params= [], body = p[4])
        
        # Create scope and add parameters
        lambda_expr.scope = ast.Scope(name=f"lambda_{id(lambda_expr)}")
        self._enter_scope(lambda_expr.scope)
        self._populate_scope_symbols(lambda_expr, lambda_expr.scope)
        
        # Defer capture processing until we have proper parent scope
        self.defer_processing(lambda_expr, 'captures')
        # Queue for deferred processing
        self.defer_processing(lambda_expr, 'scope')  # Will link scope
        self.defer_processing(lambda_expr, 'captures')
        self._exit_scope()
        p[0] = lambda_expr

    def p_perform_expression(self, p):
        '''perform_expression : PERFORM effect_operation
                            | empty'''
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = p[2]

    def p_effect_operation(self, p):
        '''effect_operation : FN IDENTIFIER LPAREN param_list RPAREN ARROW type_expression
                          | FN IDENTIFIER LPAREN param_list RPAREN ARROW type_expression WITH IDENTIFIER
                          | empty'''
        if len(p) == 8:
            name = p[2]
            params = p[4]
            return_type = p[7]
            c_effect = p[9] if len(p) > 8 else None
            p[0] = ast.EffectOperation(name, params, return_type, c_effect)
        else:
            p[0] = None

    def p_struct_instantiation(self, p):
        '''struct_instantiation : IDENTIFIER LBRACE field_assignments RBRACE
                               | type_expression LBRACE field_assignments RBRACE
                               | qualified_name LBRACE argument_list RBRACE
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
            p[0] = ast.InterfaceType(p[1], [])
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
        '''type_constraint : type_expression COLON EFFECT
                         | type_expression COLON type_bound
                         | type_expression EXTENDS type_bound
                         | type_expression IMPLEMENTS type_bound
                         | type_expression EQUALS type_expression'''
        if p[2] == '=':
            p[0] = ast.TypeConstraint(p[1], 'equals', p[3])
        elif p[2] == ':' and p[3] == 'effect':
            p[0] = ast.EffectApplication(p[1])
        elif p[2] == ':':
            p[0] = ast.TypeConstraint(p[1], 'subtype', p[3])
        elif p[2] == 'extends':
            p[0] = ast.TypeConstraint(p[1], 'extends', p[3])
        elif p[2] == 'implements':
            p[0] = ast.TypeConstraint(p[1], 'implements', p[3])

        else:
            p[0] = ast.TypeConstraint(p[1], p[2].lower(), p[3])

    def p_type_alias(self, p):
        '''type_alias : TYPE IDENTIFIER type_params_opt EQUALS type_expression'''
        p[0] = ast.TypeAlias(p[2], p[5], type_params=p[3])

    

    def p_param_list(self, p):
        '''param_list : parameter_list'''
        p[0] = p[1]

    def p_type_params_opt(self, p):
        '''type_params_opt : type_param_list
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
        '''function_type : FN BACKSLASH LPAREN type_list RPAREN type_annotation
                        | FN BACKSLASH LPAREN RPAREN type_annotation'''
        if len(p) == 7:  # fn\(params) -> T @ linearity
            p[0] = ast.FunctionType(p[4] if p[4] is not None else [], p[4], linearity=p[6])
        elif len(p) == 6 and p[4] != ')':  # fn\(params) -> T
            p[0] = ast.FunctionType(p[4] if p[4] is not None else [], p[5])
        elif len(p) == 6:  # fn\() -> T @ linearity
            p[0] = ast.FunctionType([], p[5], linearity=p[5])
        else:  # fn\() -> T
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

    def p_type_annotation_opt(self, p):
        '''type_annotation_opt : type_annotation
                            | empty'''

        if len(p) == 1:
            p[0] = ast.TypeAnnotation(None)
        else:
            p[0] = p[1]

    def p_type_annotation(self, p):
        '''type_annotation : ARROW type_expression'''
        p[0] = p[2]
 
    def p_effect_declaration(self, p):
        '''effect_declaration : EFFECT IDENTIFIER EQUALS LBRACE effect_operation_list RBRACE
                            | EFFECT IDENTIFIER type_params EQUALS LBRACE effect_operation_list RBRACE'''
        name = p[2]
        if len(p) == 7:
            type_params = []
            operations = p[5]
        else:
            type_params = p[3]
            operations = p[6]
        p[0] = ast.EffectDeclaration(name, type_params, operations)

    def p_effect_operation_list(self, p):
        '''effect_operation_list : effect_operation
                                | effect_operation_list effect_operation
                                | empty'''
        if len(p) == 2:
            p[0] = [p[1]]
        elif len(p) == 3:
            p[0] = p[1] + [p[2]]
        else:
            p[0] = []

    def p_with_clause(self, p):
        '''with_clause : WITH IDENTIFIER'''
        p[0] = ast.WithClause(p[2])

    def p_handle_expression(self, p):
        '''handle_expression : HANDLE type_expression WITH LBRACE handle_cases RBRACE IN expression'''
        p[0] = ast.HandleEffect(p[2], p[5], p[8])

    def p_handle_cases(self, p):
        '''handle_cases : handle_case
                       | handle_cases handle_case'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_handle_case(self, p):
        '''handle_case : IDENTIFIER LPAREN IDENTIFIER RPAREN ARROW expression
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
        assert p[2] == 'mut', 'Expected mut, '
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

    def p_comptime_block(self, p):
        '''comptime_block : COMPTIME LBRACE statement_list RBRACE'''
        p[0] = ast.ComptimeBlock(statements=p[3])

    def p_comptime_function(self, p):
        '''comptime_function : COMPTIME FN IDENTIFIER type_params_opt LPAREN param_list_opt RPAREN type_annotation block'''
        p[0] = ast.ComptimeFunction(
            name=p[3],
            type_params=p[4],
            params=p[6],
            return_type=p[8],
            body=p[9],
            is_comptime=True
        )

    def p_extern_block(self, p):
        '''extern_block : EXTERN STRING LBRACE extern_declarations RBRACE
                       | EXTERN STRING LBRACE RBRACE'''
        breakpoint()
        header_path = p[2][0].strip('"')  # Remove quotes from string
        if len(p) == 6:
            declarations = p[4]
        else:
            declarations = []
        p[0] = ExternBlock(header_path=header_path, declarations=declarations)

    def p_extern_declarations(self, p):
        '''extern_declarations : extern_declaration
                             | extern_declarations extern_declaration'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_extern_declaration(self, p):
        '''extern_declaration : extern_function_declaration
                            | extern_type_declaration'''
        p[0] = p[1]

    def p_extern_type_declaration(self, p):
        '''extern_type_declaration : TYPE IDENTIFIER
                                 | TYPE IDENTIFIER EQUALS STRUCT LBRACKET RBRACKET
                                 | TYPE IDENTIFIER EQUALS STRUCT IDENTIFIER'''
        if len(p) == 3:
            # Opaque type declaration: e.g. type FILE;
            p[0] = ExternTypeDeclaration(name=p[2], is_opaque=True)
        elif len(p) == 6:
            # Named struct type: e.g. type stat = struct mystat;
            p[0] = ExternTypeDeclaration(name=p[2], is_opaque=False, struct_name=p[5])
        else:
            # Anonymous struct type: e.g. type stat = struct {};
            p[0] = ExternTypeDeclaration(name=p[2], is_opaque=False)

    def p_unsafe_block(self, p):
        '''unsafe_block : UNSAFE LBRACE statement_list RBRACE
                       | UNSAFE LBRACE RBRACE'''
        if len(p) == 5:
            p[0] = UnsafeBlock(body=p[3])
        else:
            p[0] = UnsafeBlock(body=[])

    def p_pointer_type(self, p):
        '''pointer_type : TIMES MUT TYPE
                       | TIMES CONST TYPE
                       | TIMES TYPE'''
        # Only allow pointer types in unsafe contexts or extern blocks
        if not self._in_unsafe_or_extern_context():
            raise SyntaxError("Pointer types are only allowed in unsafe blocks or extern declarations")
            
        if len(p) == 4:
            is_mut = p[2] == 'mut'
            base_type = p[3]
        else:
            is_mut = False
            base_type = p[2]
        p[0] = PointerType(base_type=base_type, is_mut=is_mut)

    def p_type_cast(self, p):
        '''type_cast : expression AS TYPE'''
        # If casting to/from a pointer type, require unsafe context
        if (isinstance(p[3], PointerType) or 
            (hasattr(p[1], 'type') and isinstance(p[1].type, PointerType))):
            if not self._in_unsafe_context():
                raise SyntaxError("Pointer type casts are only allowed in unsafe blocks")
        p[0] = TypeCast(expr=p[1], target_type=p[3])

    def p_pointer_dereference(self, p):
        '''pointer_dereference : TIMES expression'''
        if not self._in_unsafe_context():
            raise SyntaxError("Pointer dereference is only allowed in unsafe blocks")
        p[0] = PointerDereference(ptr=p[2])

    def p_address_of(self, p):
        '''address_of : AMPERSAND expression
                     | AMPERSAND MUT expression'''
        if not self._in_unsafe_context():
            raise SyntaxError("Taking address of value is only allowed in unsafe blocks")
        if len(p) == 3:
            p[0] = AddressOf(expr=p[2], is_mut=False)
        else:
            p[0] = AddressOf(expr=p[3], is_mut=True)

    def _enter_scope(self, scope):
        """Enter a new scope, setting its parent to the current scope"""
        scope.parent = self.current_scope
        self.scope_stack.append(self.current_scope)
        self.current_scope = scope
        return scope

    def _exit_scope(self):
        """Exit the current scope, restoring the parent scope"""
        self.current_scope = self.scope_stack.pop()

    def _is_variable_defined(self, var_name, scope):
        """Check if a variable is defined in the given scope or its parents"""
        current_scope = scope
        while current_scope:
            # Check function parameters
            if hasattr(current_scope, 'params'):
                for param in current_scope.params:
                    if param.name == var_name:
                        return True
                        
            # Check let bindings
            if hasattr(current_scope, 'declarations'):
                for decl in current_scope.declarations:
                    if hasattr(decl, 'name') and decl.name == var_name:
                        return True
                        
            # Check scope symbols
            if var_name in current_scope.symbols:
                return True
                
            current_scope = current_scope.parent
            
        return False
        
    def _find_variable_in_node(self, var_name, node):
        """Find a variable definition in a node or its parents"""
        if node is None:
            return None
            
        # Check if this node defines the variable
        if isinstance(node, ast.FunctionDeclaration):
            # Check parameters
            for param in node.params:
                if param.name == var_name:
                    return param
            # Check let bindings in body
                for stmt in node.body:
                    if isinstance(stmt, ast.LetStatement):
                        for binding in stmt.bindings:
                            if binding.identifier == var_name:
                                return binding
                                
        elif isinstance(node, ast.LambdaExpression):
            # Check parameters
            for param in node.params:
                if param.name == var_name:
                    return param
            # Check let bindings in body if it's a block
            if isinstance(node.body, list):
                for stmt in node.body:
                    if isinstance(stmt, ast.LetStatement):
                        for binding in stmt.bindings:
                            if binding.identifier == var_name:
                                return binding
                                
        elif isinstance(node, ast.LetStatement):
            for binding in node.bindings:
                if binding.identifier == var_name:
                    return binding
                    
        # Recursively check parent
        return self._find_variable_in_node(var_name, node.parent) if hasattr(node, 'parent') else None

    def _is_variable_mutated(self, var_name, node):
        """Check if a variable is mutated in the given AST node"""
        self.logger.debug(f"Checking mutation for {var_name} in node type: {type(node)}")
        
        if node is None:
            self.logger.warning("Warning: None node encountered in mutation check")
            return False
            
        if isinstance(node, list):
            self.logger.warning(f"Warning: List encountered in mutation check: {node}")
            return any(self._is_variable_mutated(var_name, item) for item in node)
            
        # Direct mutations
        if isinstance(node, ast.Assignment) and node.name == var_name:
            return True
            
        # Mutations through function calls
        if isinstance(node, ast.FunctionCall):
            # Check if variable is passed as argument
            for arg in node.args:
                if isinstance(arg, ast.Variable) and arg.name == var_name:
                    # Conservatively assume function calls might mutate
                    return True
                    
        # Block statements
        if isinstance(node, ast.Block):
            return any(self._is_variable_mutated(var_name, stmt) for stmt in node.statements)
            
        # Check all child nodes
        for child in node.children:
            if self._is_variable_mutated(var_name, child):
                return True
                
        return False
        
    def _get_node_children(self, node):
        """Helper to get all child nodes of an AST node"""
        children = []
        for attr_name in dir(node):
            if attr_name.startswith('_'):
                continue
            attr = getattr(node, attr_name)
            if isinstance(attr, (ast.Node, list)):
                children.append(attr)
        return children

    def parse_effect_type(self):
        """Parse an effect type (e.g., State[T])"""
        name = self.expect_identifier()
        type_args = []
        
        if self.match('['):
            type_args = self.parse_type_args()
            self.expect(']')
            
        return ast.EffectApplication(name, type_args)

    def parse_type_args(self):
        """Parse type arguments between [ and ]"""
        args = []
        while not self.check(']'):
            args.append(self.parse_type())
            if not self.check(']'):
                self.expect(',')
        return args

    def parse_type_application(self):
        """Parse a type application (e.g., List[T])"""
        base = self.parse_type_name()
        if self.match('['):
            args = self.parse_type_args()
            self.expect(']')
            return ast.TypeApplication(base, args)
        return base

    def p_comparison_expression(self, p):
        '''comparison_expression : additive_expression
                               | comparison_expression EQUALEQUAL additive_expression
                               | comparison_expression NOTEQUAL additive_expression
                               | comparison_expression LESS additive_expression
                               | comparison_expression LESSEQUAL additive_expression
                               | comparison_expression GREATER additive_expression
                               | comparison_expression GREATEREQUAL additive_expression'''
        if len(p) == 2:
            p[0] = p[1]
        else:
            operator_map = {
                '==': ast.ComparisonOperator.EQUAL,
                '!=': ast.ComparisonOperator.NOT_EQUAL,
                '<': ast.ComparisonOperator.LESS,
                '<=': ast.ComparisonOperator.LESS_EQUAL,
                '>': ast.ComparisonOperator.GREATER,
                '>=': ast.ComparisonOperator.GREATER_EQUAL
            }
            p[0] = ast.ComparisonExpression(p[1], operator_map[p[2]], p[3])

    def p_additive_expression(self, p):
        '''additive_expression : multiplicative_expression
                             | additive_expression PLUS multiplicative_expression
                             | additive_expression MINUS multiplicative_expression'''
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = ast.BinaryOperation(p[1], p[2], p[3])

    def p_multiplicative_expression(self, p):
        '''multiplicative_expression : term
                                   | multiplicative_expression TIMES term
                                   | multiplicative_expression DIVIDE term'''
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = ast.BinaryOperation(p[1], p[2], p[3])

    def _in_unsafe_context(self):
        """Check if we're currently parsing inside an unsafe block"""
        # Walk up the parse stack to find if we're in an unsafe block
        for item in reversed(self.parse_stack):
            if isinstance(item, UnsafeBlock):
                return True
        return False
    
    def _in_unsafe_or_extern_context(self):
        """Check if we're in an unsafe block or extern declaration"""
        for item in reversed(self.parse_stack):
            if isinstance(item, (UnsafeBlock, ExternBlock)):
                return True
        return False

    def p_decorator(self, p):
        '''decorator : AT IDENTIFIER
                    | AT IDENTIFIER LPAREN decorator_args RPAREN'''
        if len(p) == 3:
            p[0] = Decorator(name=p[2])
        else:
            p[0] = Decorator(name=p[2], args=p[4])

    def p_decorator_args(self, p):
        '''decorator_args : decorator_arg
                        | decorator_args COMMA decorator_arg'''
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = {**p[1], **p[3]}

    def p_decorator_arg(self, p):
        '''decorator_arg : IDENTIFIER EQUALS expression'''
        p[0] = {p[1]: p[3]}

    def p_decorator_list(self, p):
        '''decorator_list : decorator
                        | decorator_list decorator'''
        if len(p) == 2:
            p[0] = DecoratorList([p[1]])
        else:
            p[0].decorators.append(p[2])

    def p_extern_function_declaration(self, p):
        '''extern_function_declaration : decorator_list FN IDENTIFIER LPAREN param_list_opt RPAREN ARROW type_expression
                                     | decorator_list FN IDENTIFIER LPAREN param_list_opt RPAREN
                                     | FN IDENTIFIER LPAREN param_list_opt RPAREN ARROW type_expression
                                     | FN IDENTIFIER LPAREN param_list_opt RPAREN'''
        if len(p) == 9:  # With return type and decorators
            decorators = p[1]
            name = p[3]
            params = p[5]
            return_type = p[8]
        elif len(p) == 7:  # With decorators, no return type
            decorators = p[1]
            name = p[3]
            params = p[5]
            return_type = None
        elif len(p) == 8:  # No decorators, with return type
            decorators = DecoratorList([])
            name = p[2]
            params = p[4]
            return_type = p[7]
        else:  # No decorators, no return type
            decorators = DecoratorList([])
            name = p[2]
            params = p[4]
            return_type = None

        # Process C function decorators
        c_func = None
        for decorator in decorators.decorators:
            if decorator.name == "c_function":
                c_func = CFunctionDecorator(
                    borrows_refs=decorator.args.get("borrows_refs", False),
                    consumes_refs=decorator.args.get("consumes_refs", False),
                    produces_refs=decorator.args.get("produces_refs", False),
                    call_mode=decorator.args.get("call_mode", "blocking"),
                    inline=decorator.args.get("inline", True)
                )
                break

        p[0] = ast.ExternFunctionDeclaration(
            name=name,
            params=params,
            return_type=return_type,
            c_function=c_func
        )

    def p_print_statement(self, p):
        '''print_statement : PRINT LPAREN argument_list RPAREN
                         | PRINT LPAREN RPAREN'''
        if len(p) == 5:
            p[0] = ast.PrintStatement(p[3])
        else:
            p[0] = ast.PrintStatement([])

    def p_empty(self, p):
        'empty :'
        pass

    def p_error(self, p):
        if p:
            # If p is a YaccProduction, get the last token
            if hasattr(p, 'slice'):
                token = p.slice[-1]
                msg = f"Syntax error at '{token.value}'"
                lineno = token.lineno
                lexpos = token.lexpos
            else:
                # Regular token
                msg = f"Syntax error at '{p.value}'"
                lineno = p.lineno
                lexpos = p.lexpos
                
            raise CompileError(
                message=msg,
                error_type="ParseError",
                location=SourceLocation(
                    file=self.lexer.source_file,
                    line=lineno,
                    column=lexpos
                ),
                context=get_source_context(self.lexer.source_file, lineno),
                stack_trace=traceback.format_stack(),
                notes=["Check syntax near this location"]
            )
        else:
            raise CompileError(
                message="Syntax error at EOF",
                error_type="ParseError",
                location=None,
                context=None,
                stack_trace=traceback.format_stack(),
                notes=["Unexpected end of file"]
            )
 

    def defer_processing(self, node, task):
        """Queue a node for deferred processing"""
        self.deferred_processing.append((node, task))
        
    def process_deferred(self):
        """Process all deferred tasks"""
        # First pass: link all scopes
        for node, task in self.deferred_processing:
            if isinstance(node, scoped_nodes):
                parent_scope = self._find_parent_scope(node)
                if parent_scope:
                    self.logger.debug(f"Setting parent scope for {type(node)} to {parent_scope}")
                    node.scope.parent = parent_scope
        
        # Second pass: process captures now that scopes are linked
        for node, task in self.deferred_processing:
            if task == 'captures':
                self._process_captures(node)
                
    def _process_captures(self, node):
        """Process variable captures for a node"""
        if isinstance(node, ast.LambdaExpression):
            self._process_lambda_captures(node)
        elif isinstance(node, ast.HandleEffect):
            self._process_handler_captures(node)
        elif isinstance(node, ast.SpawnExpression):
            self._process_spawn_captures(node)

    def _process_lambda_captures(self, lambda_expr):
        """Process variable captures for a lambda expression."""
        lambda_expr.captured_vars = set()
        lambda_expr.capture_modes = {}
        
        # Find all variable references in the lambda body
        var_refs = self._find_variables_in_body(lambda_expr.body)
        self.logger.debug(f"Found variable references in lambda: {var_refs}")
        
        # Process each variable reference
        for var_name in var_refs:
            # Skip lambda parameters
            if any(param.name == var_name for param in lambda_expr.params):
                continue
                
            # Check if variable exists in parent scope
            if self._is_variable_defined(var_name, lambda_expr.scope.parent):
                lambda_expr.captured_vars.add(var_name)
                # Determine capture mode based on usage
                if self._is_variable_mutated(var_name, lambda_expr.body):
                    lambda_expr.capture_modes[var_name] = "borrow_mut"
                else:
                    lambda_expr.capture_modes[var_name] = "borrow"
    
    def _process_handler_captures(self, handler):
        """Process variable captures for an effect handler."""
        handler.captured_vars = set()
        handler.capture_modes = {}
        
        # Find variables referenced in handler operations
        for operation in handler.handler:
            var_refs = self._find_variables_in_body(operation.body)
            for var_name in var_refs:
                if self._is_variable_defined(var_name, handler.scope.parent):
                    handler.captured_vars.add(var_name)
                    # Effect handlers typically need mutable access
                    handler.capture_modes[var_name] = "borrow_mut"
                    
    def _process_spawn_captures(self, spawn):
        """Process variable captures for a spawn expression."""
        spawn.captured_vars = set()
        spawn.capture_modes = {}
        
        # Find variables referenced in spawned function
        var_refs = self._find_variables_in_body(spawn.function_expression)
        for var_name in var_refs:
            if self._is_variable_defined(var_name, spawn.scope.parent):
                spawn.captured_vars.add(var_name)
                # Spawned tasks need their own copies
                spawn.capture_modes[var_name] = "move"
                
    def _check_borrow_lifetime(self, borrow, context):
        """Check if a borrow expression respects lifetime rules."""
        var_name = borrow.variable
        if not self._is_variable_defined(var_name, context):
            self.emit_error(f"Cannot borrow undefined variable '{var_name}'", borrow)
            
        # Check if the borrow outlives the borrowed value
        if isinstance(borrow, ast.BorrowUnique):
            # Track that this variable has a unique borrow
            context.unique_borrows.add(var_name)
        else:
            # Track shared borrow
            context.shared_borrows.add(var_name)
            
    def _check_move_validity(self, move, context):
        """Check if a move expression is valid."""
        var_name = move.variable
        if not self._is_variable_defined(var_name, context):
            self.emit_error(f"Cannot move undefined variable '{var_name}'", move)
            
        # Check if variable was already moved
        if var_name in context.moved_vars:
            self.emit_error(f"Cannot move variable '{var_name}' more than once", move)
            
        # Mark variable as moved
        context.moved_vars.add(var_name)
        
    def _check_exclave_escape(self, exclave, context):
        """Check if an exclave expression would cause invalid escapes."""
        # Check if the expression contains any local variables
        local_vars = self._find_local_variables(exclave.expression)
        if local_vars:
            self.emit_error(
                f"Exclave expression would cause local variables to escape: {local_vars}",
                exclave
            )
            
    def _resolve_recursive_type(self, type_def, context):
        """Resolve a recursive type definition."""
        # Add the type name to the scope before resolving the body
        context.add_type(type_def.name, type_def)
        
        # Now resolve the body which might reference the type name
        self._resolve_type_expression(type_def.body, context)
        
    def _resolve_implementation(self, impl, context):
        """Resolve an interface implementation."""
        # Check that the interface exists
        if not self._is_interface_defined(impl.interface_name, context):
            self.emit_error(f"Undefined interface '{impl.interface_name}'", impl)
            
        # Check that all required methods are implemented
        interface = self._get_interface(impl.interface_name, context)
        for method in interface.methods:
            if not any(m.name == method.name for m in impl.methods):
                self.emit_error(
                    f"Missing implementation for method '{method.name}'",
                    impl
                )
                
    def _find_variables_in_body(self, node):
        """Recursively find all variable references in a node"""
        vars = set()
        self.logger.debug(f"Finding variables in node type: {type(node)}")
        
        if node is None:
            self.logger.warning("Warning: None node encountered")
            return vars
            
        if isinstance(node, list):
            for item in node:
                vars.update(self._find_variables_in_body(item))
            return vars
            
        if isinstance(node, ast.Variable):
            vars.add(node.name)
            self.logger.debug(f"Found variable reference: {node.name}")
        elif isinstance(node, ast.Block):
            self.logger.debug(f"Processing Block node with statements: {len(node.statements)} statements")
            for stmt in node.statements:
                vars.update(self._find_variables_in_body(stmt))
        elif isinstance(node, ast.BinaryOperation):
            vars.update(self._find_variables_in_body(node.left))
            vars.update(self._find_variables_in_body(node.right))
        elif isinstance(node, ast.FunctionCall):
            for arg in node.arguments:
                vars.update(self._find_variables_in_body(arg))
        elif isinstance(node, ast.LetBinding):
            if node.initializer:
                vars.update(self._find_variables_in_body(node.initializer))
        elif isinstance(node, ast.LetStatement):
            for binding in node.bindings:
                vars.update(self._find_variables_in_body(binding))
        elif isinstance(node, ast.Assignment):
            vars.update(self._find_variables_in_body(node.name))
            if node.expression:
                vars.update(self._find_variables_in_body(node.expression))
        elif isinstance(node, ast.LambdaExpression):
            for param in node.params:
                vars.add(param.name)
            if isinstance(node.body, list):
                for stmt in node.body:
                    vars.update(self._find_variables_in_body(stmt))
            elif isinstance(node.body, ast.Block):
                for stmt in node.body.statements:
                    vars.update(self._find_variables_in_body(stmt))
            else:
                vars.update(self._find_variables_in_body(node.body))
        elif isinstance(node, ast.ReturnStatement):
            if node.expression:
                vars.update(self._find_variables_in_body(node.expression))
        elif isinstance(node, ast.PrintStatement):
            for arg in node.arguments:
                vars.update(self._find_variables_in_body(arg))
        elif isinstance(node, ast.Program):
            for stmt in node.statements:
                vars.update(self._find_variables_in_body(stmt))
        
        self.logger.debug(f"Variables found in {type(node).__name__}: {vars}")
        return vars

    def _set_parent_scope(self, node, init_scope=None):
        """Set up scoping based on AST parent relationships"""
        if node is None:
            return node
            
        self.logger.debug(f"Setting parent scope for node type: {type(node)}")
        self.logger.debug(f"Node AST parent: {type(node.parent) if node.parent else None}")
        if hasattr(node, 'scope'):
            self.logger.debug(f"Node scope: {node.scope}")
            self.logger.debug(f"Node scope parent: {type(node.scope.parent) if node.scope and node.scope.parent else None}")
            
        # Find nearest parent that introduces a scope
        current = node.scope
        while current and not isinstance(current, scoped_nodes):
            self.logger.debug(f"Walking up AST, current node: {type(current)}")
            current = current.parent
            
        self.logger.debug(f"Found nearest scoping parent: {type(current) if current else None}")
        
        # Set up scope if we have one
        if hasattr(node, 'scope'):
            if current and hasattr(current, 'scope'):
                self.logger.debug(f"Setting scope parent. Current scope: {current.scope}")
                node.scope.parent = current.scope
                if node.scope not in current.scope.children:
                    current.scope.children.append(node.scope)
            else:
                self.logger.debug(f"No valid parent scope found for node type: {type(node)}")
                    
            # Process children in their own scope if needed
            if isinstance(node, (ast.FunctionDeclaration, ast.LambdaExpression)):
                old_scope = self.current_scope
                self.current_scope = node.scope
                self._process_children(node)
                self.current_scope = old_scope
            else:
                self._process_children(node)
        else:
            self._process_children(node)
            
        return node

    def _process_children(self, node):
        """Process child nodes in the current scope."""
        if hasattr(node, 'body') and node.body:
            if isinstance(node.body, list):
                for child in node.body:
                    self._set_parent_scope(child)
            else:
                self._set_parent_scope(node.body)
                
        if hasattr(node, 'statements') and node.statements:
            for stmt in node.statements:
                self._set_parent_scope(stmt)
    
    def _populate_scope_symbols(self, node, scope):
        """Populate a scope's symbol table with all declarations from a node."""
        if isinstance(node, ast.FunctionDeclaration):
            # Add parameters
            for param in node.params:
                scope.add_symbol(param.name, param)
            # Process body for let bindings
            for stmt in node.body:
                if isinstance(stmt, ast.LetStatement):
                    for binding in stmt.bindings:
                        scope.add_symbol(binding.identifier, binding)
        elif isinstance(node, ast.LambdaExpression):
            # Add parameters
            for param in node.params:
                scope.add_symbol(param.name, param)
            # Process body for let bindings
            if isinstance(node.body, list):
                for stmt in node.body:
                    if isinstance(stmt, ast.LetStatement):
                        for binding in stmt.bindings:
                            scope.add_symbol(binding.identifier, binding)
            elif isinstance(node.body, ast.Block):
                for stmt in node.body.statements:
                    if isinstance(stmt, ast.LetStatement):
                        for binding in stmt.bindings:
                            scope.add_symbol(binding.identifier, binding)
            else:
                if isinstance(node.body, ast.LetStatement):
                    for binding in node.body.bindings:
                        scope.add_symbol(binding.identifier, binding)

    def _find_local_variables(self, node):
        """Find local variables in a node"""
        local_vars = set()
        self.logger.debug(f"Finding local variables in node type: {type(node)}")
        
        if node is None:
            self.logger.warning("Warning: None node encountered")
            return local_vars
            
        if isinstance(node, list):
            for item in node:
                local_vars.update(self._find_local_variables(item))
            return local_vars
            
        if isinstance(node, ast.Variable):
            local_vars.add(node.name)
            self.logger.debug(f"Found local variable reference: {node.name}")
        elif isinstance(node, ast.Block):
            self.logger.debug(f"Processing Block node with statements: {len(node.statements)} statements")
            for stmt in node.statements:
                local_vars.update(self._find_local_variables(stmt))
        elif isinstance(node, ast.FunctionCall):
            for arg in node.arguments:
                local_vars.update(self._find_local_variables(arg))
        elif isinstance(node, ast.LetBinding):
            if node.initializer:
                local_vars.update(self._find_local_variables(node.initializer))
        elif isinstance(node, ast.LetStatement):
            for binding in node.bindings:
                local_vars.update(self._find_local_variables(binding))
        elif isinstance(node, ast.ReturnStatement):
            if node.expression:
                local_vars.update(self._find_local_variables(node.expression))
        elif isinstance(node, ast.PrintStatement):
            for arg in node.arguments:
                local_vars.update(self._find_local_variables(arg))
        elif isinstance(node, ast.Program):
            for stmt in node.statements:
                local_vars.update(self._find_local_variables(stmt))
        
        self.logger.debug(f"Local variables found in {type(node).__name__}: {local_vars}")
        return local_vars

    def _find_parent_scope(self, node):
        """Walk up the AST to find the nearest enclosing scope"""
        self.logger.debug(f"Finding parent scope for {type(node)}")
        current = node.parent
        while current:
            self.logger.debug(f"Checking node {type(current)}")
            if hasattr(current, 'scope') and isinstance(current, scoped_nodes) and current.scope:
                self.logger.debug(f"Found parent scope in {type(current)}")
                return current.scope
            current = current.parent
        self.logger.debug("No parent scope found, using global")
        return self.current_module.scope if self.current_module else None

    def _update_child_scopes(self, node):
        """Update scope parents for any child nodes that need it"""
        
        if isinstance(node, (ast.LambdaExpression, ast.FunctionDeclaration)):
            # Find parent scope by walking up AST
            parent_scope = self._find_parent_scope(node)
            if parent_scope:
                node.scope.parent = parent_scope
            
        # Process all children
        for child in node.children:
            self._update_child_scopes(child)

    def _is_variable_mutated_new(self, var_name, node):
        """Check if a variable is mutated in the given AST node"""
        self.logger.debug(f"Checking mutation for {var_name} in node type: {type(node)}")
        
        if node is None:
            return False
            
        # Check assignment statements
        if isinstance(node, ast.Assignment):
            if isinstance(node.target, ast.Variable) and node.target.name == var_name:
                return True
                
        # Check let bindings
        if isinstance(node, ast.LetStatement):
            for binding in node.bindings:
                if binding.identifier == var_name:
                    return True
                if binding.initializer:
                    if self._is_variable_mutated(var_name, binding.initializer):
                        return True
                        
        # Check function/lambda bodies
        if isinstance(node, (ast.FunctionDeclaration, ast.LambdaExpression)):
            if hasattr(node, 'body'):
                if isinstance(node.body, list):
                    for stmt in node.body:
                        if self._is_variable_mutated(var_name, stmt):
                            return True
                else:
                    return self._is_variable_mutated(var_name, node.body)
                    
        # Check block statements
        if isinstance(node, ast.Block):
            for stmt in node.statements:
                if self._is_variable_mutated(var_name, stmt):
                    return True
                    
        # Check binary operations (in case of compound assignment)
        if isinstance(node, ast.BinaryOperation):
            if self._is_variable_mutated(var_name, node.left):
                return True
            if self._is_variable_mutated(var_name, node.right):
                return True
                
        return False