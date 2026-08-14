import ply.yacc as yacc
from metaxu.lexer import Lexer
import metaxu.metaxu_ast as ast
from metaxu.decorator_ast import Decorator, CFunctionDecorator, DecoratorList
from metaxu.extern_ast import ExternBlock, ExternFunctionDeclaration, ExternTypeDeclaration
from metaxu.unsafe_ast import (UnsafeBlock, PointerType, TypeCast,
                       PointerDereference, AddressOf)
from metaxu.type_defs import (SharedType, BoxType, ReferenceType, NoneType)
from metaxu.errors import CompileError, SourceLocation, register_source
import bisect
import functools
import traceback
import logging

logger = logging.getLogger(__name__)

scoped_nodes = (ast.FunctionDeclaration, ast.LambdaExpression, ast.Block, ast.WhileStatement, ast.ForStatement, ast.ModuleBody)

# Lazily-built Parser dedicated to parsing `{expr}` segments of f-strings
# (see Parser._parse_fstring_expr).  Module-level so the ~0.5s PLY table
# build happens at most once per process.
_FSTRING_SEGMENT_PARSER = None


class _GrammarNamespace:
    """Namespace handed to ``yacc.yacc(module=...)``.

    Holds the grammar actions with the location-attaching wrapper applied
    (see Parser._locating); PLY reads rules off this object exactly as it
    would off the Parser instance.
    """


class Parser:
    start = 'program'

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize deferred processing system
        self.deferred_processing = []

        # Initialize the lexer
        self.lexer = Lexer()
        self.tokens = self.lexer.tokens  # Get token list from lexer
        self.parser = yacc.yacc(module=self._grammar_namespace(), debug=False,
                                write_tables=False, errorlog=yacc.NullLogger())
        self.module_names = set()
        self.parse_stack = []
        self.current_scope = None
        self.scope_stack = []  # Stack to track nested scopes
        self.current_module = None

    # ------------------------------------------------------------------
    # Source locations
    #
    # Every grammar action is wrapped so the node it produces gets a
    # SourceLocation covering the whole production, without touching any of
    # the ~130 productions individually.  PLY's `tracking=True` reduce path
    # stamps each nonterminal symbol with the start position of its first
    # token and the end position of its last one BEFORE calling the action,
    # so the wrapper can read them off `p` afterwards.
    # ------------------------------------------------------------------

    def _grammar_namespace(self) -> '_GrammarNamespace':
        """Build the object PLY reads the grammar from: every ``p_*`` action
        except ``p_error`` wrapped in the location-attaching decorator."""
        ns = _GrammarNamespace()
        ns.tokens = self.tokens
        ns.start = self.start
        precedence = getattr(self, 'precedence', None)
        if precedence is not None:
            ns.precedence = precedence
        for name in dir(self):
            if not name.startswith('p_'):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            setattr(ns, name, method if name == 'p_error' else self._locating(method))
        return ns

    def _locating(self, method):
        """Wrap one grammar action so its result carries a source location."""
        attach = self._attach_location

        @functools.wraps(method)
        def action(p):
            method(p)
            attach(p)

        # PLY sorts productions by their definition line and reports errors
        # against it; the wrapper must not collapse every rule onto the same
        # line (rule ORDER decides reduce/reduce conflicts, so changing it
        # would change the language).  `get_pfunctions` reads an explicit
        # `co_firstlineno` attribute in preference to the code object's.
        action.co_firstlineno = method.__func__.__code__.co_firstlineno
        return action

    #: Attributes that are back-references or bookkeeping, never child nodes.
    _NON_CHILD_ATTRS = frozenset({'parent', 'scope', 'location'})

    def _attach_location(self, p) -> None:
        """Give `p[0]` a SourceLocation spanning the reduced production.

        Only fills in nodes that do not have one yet, so a pass-through rule
        (`p[0] = p[1]`) keeps the inner node's own, tighter location.  The
        same span is then pushed down into descendants that are still
        unlocated — the helper-built nodes (HandleCase, EffectApplication,
        QualifiedName, ...) that no production of their own ever produced.
        The descent STOPS at any node that already has a location, so a
        located subtree keeps its own tighter positions and the fill costs
        one visit per node overall.
        """
        try:
            node = p[0]
        except Exception:
            return
        if not isinstance(node, ast.Node) or getattr(node, 'location', None) is not None:
            return
        try:
            start = p.lexpos(0)
            end = getattr(p.slice[0], 'endlexpos', start)
        except Exception:
            return
        loc = self.location_for_offsets(start, end)
        if loc is None:
            return
        try:
            node.location = loc
        except Exception:  # frozen/slotted node: not locatable, skip
            return
        self._fill_locations(node, loc)

    def _fill_locations(self, node, loc, depth: int = 0) -> None:
        """Give `loc` to every still-unlocated node reachable from `node`."""
        if depth > 60:
            return
        for attr, value in vars(node).items():
            if attr in self._NON_CHILD_ATTRS:
                continue
            self._fill_value(value, loc, depth)

    def _fill_value(self, value, loc, depth: int) -> None:
        if isinstance(value, ast.Node):
            if getattr(value, 'location', None) is None:
                try:
                    value.location = loc
                except Exception:
                    return
                self._fill_locations(value, loc, depth + 1)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                self._fill_value(item, loc, depth)
        elif isinstance(value, dict):
            for item in value.values():
                self._fill_value(item, loc, depth)

    def location_for_offsets(self, start: int, end: int | None = None) -> 'SourceLocation | None':
        """SourceLocation for a half-open character range of the current file."""
        if start is None or start < 0:
            return None
        line, column = self._line_col(start)
        end_line = end_column = None
        if end is not None and end >= start:
            end_line, end_column = self._line_col(end)
        return SourceLocation(
            file=self.lexer.source_file,
            line=line,
            column=column,
            end_line=end_line,
            end_column=end_column,
            offset=start,
            end_offset=end,
        )

    def _line_col(self, offset: int) -> tuple[int, int]:
        """1-based (line, column) of a 0-based character offset.

        Derived from the lexer's `line_starts` table (filled while tokenizing)
        rather than from token linenos, so a position and its line always
        agree.
        """
        starts = self.lexer.line_starts or [0]
        idx = bisect.bisect_right(starts, offset) - 1
        if idx < 0:
            idx = 0
        return idx + 1, offset - starts[idx] + 1

    def parse(self, source: str, file_path: str = "<unknown>") -> 'ast.Module':
        """Parse source code into an AST"""
        try:
            self.logger.debug("=== Starting Parse (%s) ===", file_path)
            # Reset per-parse state so one Parser instance is reusable
            # across files (PLY table construction costs ~430ms, so the
            # pipeline shares an instance; see compiler/shared_parser.py).
            self.deferred_processing = []
            self.module_names = set()
            self.parse_stack = []
            self.current_scope = None
            self.scope_stack = []
            self.current_module = None
            # Initialize lexer with source
            self.lexer.source_file = file_path
            self.lexer.input(source)
            # Diagnostics excerpt the offending line; register the text so
            # in-memory sources ("<mem>") render like on-disk ones.
            register_source(file_path, source)
            self._enter_scope(ast.Scope(name="global"))
            # Parse using PLY. tracking=True makes PLY record the token span
            # of every reduced nonterminal, which is what _attach_location
            # turns into node locations.
            result = self.parser.parse(source, lexer=self.lexer, debug=False,
                                       tracking=True)

            # If the result is a list of statements, wrap it in a module
            if isinstance(result, list):
                module_body = ast.ModuleBody(statements=result)
                module_body.location = self.location_for_offsets(0, len(source))
                result = ast.Module(name="main", body=module_body)
                result.location = module_body.location

            # Set source file for all modules
            if isinstance(result, ast.Module):
                result.source_file = file_path
            elif isinstance(result, list):
                for module in result:
                    if isinstance(module, ast.Module):
                        module.source_file = file_path

            self._exit_scope()
            # Process deferred items (lambda captures, scope linking)
            self.process_deferred()
            return result

        except CompileError:
            raise
        except Exception as e:
            # Location of the token the parser was looking at when it failed.
            token = getattr(self.lexer, 'current_token', None)
            self.logger.debug("Parser error: %s: %s", type(e).__name__, e)
            location = self.location_for_offsets(
                token.lexpos, getattr(token, 'endlexpos', token.lexpos)
            ) if token is not None else None

            error = CompileError(
                message=str(e),
                error_type="ParseError",
                location=location,
                stack_trace=traceback.format_stack(),
                notes=["Check syntax near this location"]
            )
            raise error from e
        finally:
            self.logger.debug("=== Exiting Parse ===")
            self._exit_scope() if self.current_scope else None

    # ------------------------------------------------------------------
    # AST construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _name_parts(node):
        """Return list of name parts if node is a chain of plain names, else None."""
        if isinstance(node, ast.Variable):
            return [node.name]
        if isinstance(node, ast.QualifiedName):
            return list(node.parts)
        if isinstance(node, ast.FieldAccess):
            base = node.base
            if isinstance(base, str):
                return [base] + list(node.fields)
            base_parts = Parser._name_parts(base)
            if base_parts is not None:
                return base_parts + list(node.fields)
        return None

    def _make_call(self, callee, args):
        """Build the best-fitting call node for `callee(args)`."""
        args = args or []
        if isinstance(callee, ast.Variable):
            return ast.FunctionCall(callee.name, args)
        if isinstance(callee, ast.VectorTypeExpression):
            return ast.VectorLiteral(callee.base_type, callee.size, args)
        if isinstance(callee, ast.GenericInstance):
            inner = self._make_call(callee.base, args)
            inner.type_args = callee.type_args
            return inner
        parts = self._name_parts(callee)
        if parts is not None:
            return ast.QualifiedFunctionCall(parts, args)
        return ast.CallExpression(callee, args)

    @staticmethod
    def _struct_name_of(node):
        """Extract a struct name (QualifiedName) and type args from a postfix expr."""
        type_args = []
        if isinstance(node, ast.GenericInstance):
            type_args = node.type_args
            node = node.base
        if isinstance(node, ast.IndexExpression):
            type_args = node.index if isinstance(node.index, list) else [node.index]
            node = node.base
        parts = Parser._name_parts(node)
        if parts is None:
            parts = [str(node)]
        return ast.QualifiedName(parts), type_args

    @staticmethod
    def _effect_name_of(node):
        """Extract an effect name string (and type args) from an expression/type."""
        type_args = []
        if isinstance(node, ast.GenericInstance):
            type_args = node.type_args
            node = node.base
        if isinstance(node, ast.TypeApplication):
            return node.type_constructor, node.type_args
        if isinstance(node, ast.TypeReference):
            return node.name, []
        parts = Parser._name_parts(node)
        if parts is not None:
            return '.'.join(parts), type_args
        return str(node), type_args

    @staticmethod
    def _params_from_expr(node):
        """Convert an expression (Variable or TupleLiteral of Variables) to parameters."""
        if isinstance(node, ast.Variable):
            return [ast.Parameter(node.name)]
        if isinstance(node, ast.TupleLiteral):
            params = []
            for el in node.elements:
                if isinstance(el, ast.Variable):
                    params.append(ast.Parameter(el.name))
                else:
                    params.append(ast.Parameter(str(el)))
            return params
        return [ast.Parameter(str(node))]

    @staticmethod
    def _branch_of(statements):
        """Turn a statement list into a single branch node (unwrap single expr)."""
        statements = statements or []
        if len(statements) == 1:
            return statements[0]
        return ast.Block(statements)

    def _make_lambda(self, params, body, return_type=None, performs=None):
        lambda_expr = ast.LambdaExpression(params=params or [], body=body,
                                           return_type=return_type)
        if performs:
            lambda_expr.performs = performs
        lambda_expr.scope = ast.Scope(name=f"lambda_{id(lambda_expr)}")
        self._enter_scope(lambda_expr.scope)
        self._populate_scope_symbols(lambda_expr, lambda_expr.scope)
        self.defer_processing(lambda_expr, 'scope')
        self.defer_processing(lambda_expr, 'captures')
        self._exit_scope()
        return lambda_expr

    def _effect_app_from_type(self, type_node):
        if isinstance(type_node, ast.TypeApplication):
            return ast.EffectApplication(type_node.type_constructor, type_node.type_args)
        if isinstance(type_node, ast.TypeReference):
            return ast.EffectApplication(type_node.name, [])
        return ast.EffectApplication(str(type_node), [])

    # ------------------------------------------------------------------
    # Program structure
    # ------------------------------------------------------------------

    def p_program(self, p):
        '''program : statement_list'''
        p[0] = p[1]

    def p_statement_list(self, p):
        '''statement_list : statements
                          | empty'''
        p[0] = p[1] if p[1] else []

    def p_statements(self, p):
        '''statements : statement
                      | statements statement
                      | statements SEMICOLON
                      | SEMICOLON'''
        if len(p) == 2:
            p[0] = [] if p[1] == ';' else [p[1]]
        else:
            if p[2] == ';':
                p[0] = p[1]
            else:
                p[0] = p[1] + [p[2]]

    def p_statement(self, p):
        '''statement : expression
                     | assignment
                     | let_statement
                     | return_statement
                     | function_declaration
                     | struct_definition
                     | enum_definition
                     | trait_definition
                     | implementation
                     | import_statement
                     | from_import_statement
                     | module_declaration
                     | export_statement
                     | visibility_block
                     | unsafe_block
                     | effect_declaration
                     | extern_block
                     | extern_type_statement
                     | type_definition
                     | for_statement
                     | while_statement
                     | comptime_block
                     | comptime_function
                     | block'''
        p[0] = p[1]

    def p_block(self, p):
        '''block : LBRACE statement_list RBRACE'''
        block = ast.Block(statements=[])
        self._enter_scope(ast.Scope())
        block.scope = self.current_scope
        if p[2]:
            block.statements = [block.add_child(stmt) for stmt in p[2]]
        self._exit_scope()
        p[0] = block

    # ------------------------------------------------------------------
    # Let / assignment / return
    # ------------------------------------------------------------------

    def p_let_statement(self, p):
        '''let_statement : let_binding
                         | let_statement COMMA let_binding'''
        if len(p) == 2:
            p[0] = ast.LetStatement(bindings=[p[1]])
        else:
            p[1].bindings.append(p[1].add_child(p[3]))
            p[0] = p[1]

    def p_let_binding(self, p):
        '''let_binding : LET binding_prefix IDENTIFIER EQUALS expression
                       | LET binding_prefix IDENTIFIER COLON type_expression EQUALS expression'''
        mode = p[2]
        if len(p) == 6:
            p[0] = ast.LetBinding(p[3], p[5], mode=mode)
        else:
            p[0] = ast.LetBinding(p[3], p[7], mode=mode, type_annotation=p[5])

    def p_binding_prefix(self, p):
        '''binding_prefix : empty
                          | MUT
                          | mode_annotation_list
                          | MUT mode_annotation_list'''
        if len(p) == 2:
            if p[1] == 'mut':
                p[0] = [ast.ModeAnnotation('mut')]
            else:
                p[0] = p[1]  # None (empty) or mode list
        else:
            p[0] = [ast.ModeAnnotation('mut')] + p[2]

    def p_assignment(self, p):
        '''assignment : postfix_expression EQUALS expression'''
        target = p[1]
        if isinstance(target, ast.Variable):
            p[0] = ast.Assignment(target.name, p[3])
        else:
            p[0] = ast.Assignment(target, p[3])

    def p_return_statement(self, p):
        '''return_statement : RETURN expression_or_empty'''
        p[0] = ast.ReturnStatement(p[2])

    def p_expression_or_empty(self, p):
        '''expression_or_empty : expression
                               | empty'''
        p[0] = p[1]

    # ------------------------------------------------------------------
    # Modes
    # ------------------------------------------------------------------

    def p_mode_annotation_list(self, p):
        '''mode_annotation_list : mode_annotation
                                | mode_annotation_list mode_annotation'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_mode_annotation(self, p):
        '''mode_annotation : AT IDENTIFIER'''
        p[0] = ast.ModeAnnotation(p[2])

    # ------------------------------------------------------------------
    # Expressions
    # ------------------------------------------------------------------

    def p_expression(self, p):
        '''expression : comparison_expression
                      | comparison_expression DOTDOT comparison_expression'''
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = ast.RangeExpression(p[1], p[3])

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
        '''multiplicative_expression : cast_expression
                                   | multiplicative_expression TIMES cast_expression
                                   | multiplicative_expression DIVIDE cast_expression
                                   | multiplicative_expression MOD cast_expression'''
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = ast.BinaryOperation(p[1], p[2], p[3])

    def p_cast_expression(self, p):
        '''cast_expression : unary_expression
                           | cast_expression AS type_expression'''
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = TypeCast(expr=p[1], target_type=p[3])

    def p_unary_expression(self, p):
        '''unary_expression : postfix_expression
                            | MINUS unary_expression
                            | AMPERSAND MUT unary_expression
                            | AMPERSAND unary_expression
                            | mode_annotation unary_expression'''
        if len(p) == 2:
            p[0] = p[1]
        elif p[1] == '-':
            p[0] = ast.UnaryOperation('-', p[2])
        elif p[1] == '&':
            if len(p) == 4:  # &mut x
                operand = p[3]
                if isinstance(operand, ast.Variable):
                    p[0] = ast.BorrowUnique(operand.name)
                else:
                    p[0] = AddressOf(expr=operand, is_mut=True)
            else:  # &x
                operand = p[2]
                if isinstance(operand, ast.Variable):
                    p[0] = ast.BorrowShared(operand.name)
                else:
                    p[0] = AddressOf(expr=operand, is_mut=False)
        else:  # mode-annotated expression: @mut x, @const x.y ...
            mode = p[1].mode_type
            operand = p[2]
            if isinstance(operand, ast.Variable):
                if mode == 'mut':
                    p[0] = ast.BorrowUnique(operand.name)
                elif mode in ('const', 'shared'):
                    p[0] = ast.BorrowShared(operand.name)
                else:
                    p[0] = ast.ModeExpression(mode, operand)
            else:
                p[0] = ast.ModeExpression(mode, operand)

    def p_postfix_expression(self, p):
        '''postfix_expression : primary_expression
                              | postfix_expression DOT IDENTIFIER
                              | postfix_expression DOT IDENTIFIER LPAREN argument_list_opt RPAREN
                              | postfix_expression DOUBLECOLON IDENTIFIER LPAREN argument_list_opt RPAREN
                              | postfix_expression LPAREN argument_list_opt RPAREN
                              | postfix_expression LBRACKET index_content RBRACKET
                              | postfix_expression LGENERIC type_list RGENERIC
                              | postfix_expression LBRACE_STRUCT struct_init_seq RBRACE'''
        if len(p) == 2:
            p[0] = p[1]
        elif p[2] == '.':
            if len(p) == 4:  # field access
                base = p[1]
                if isinstance(base, ast.FieldAccess):
                    p[0] = ast.FieldAccess(base.base, base.fields + [p[3]])
                elif isinstance(base, ast.Variable):
                    p[0] = ast.FieldAccess(base.name, [p[3]])
                else:
                    p[0] = ast.FieldAccess(base, [p[3]])
            else:  # method call
                base = p[1]
                parts = self._name_parts(base)
                args = p[5] if p[5] else []
                if parts is not None:
                    p[0] = ast.QualifiedFunctionCall(parts + [p[3]], args)
                else:
                    p[0] = ast.MethodCall(base, p[3], args)
        elif p[2] == '::':
            base = p[1]
            args = p[5] if p[5] else []
            if isinstance(base, ast.GenericInstance):
                base = base.base
            parts = self._name_parts(base) or [str(base)]
            p[0] = ast.QualifiedFunctionCall(parts + [p[3]], args)
        elif p[2] == '(':
            p[0] = self._make_call(p[1], p[3])
        elif p[2] == '[':
            p[0] = ast.IndexExpression(p[1], p[3])
        elif p.slice[2].type == 'LGENERIC':
            p[0] = ast.GenericInstance(p[1], p[3])
        else:  # struct literal
            name, type_args = self._struct_name_of(p[1])
            struct = ast.StructInstantiation(name, p[3])
            struct.type_args = type_args
            p[0] = struct

    def p_index_content(self, p):
        '''index_content : expression
                         | expression COLON expression
                         | expression COLON
                         | COLON expression
                         | COLON
                         | COLON COLON expression
                         | DOUBLECOLON expression'''
        if len(p) == 2:
            if p[1] == ':':
                p[0] = ast.SliceExpression(None, None, None)
            else:
                p[0] = p[1]
        elif len(p) == 3:
            if p[1] == '::':
                p[0] = ast.SliceExpression(None, None, p[2])
            elif p[1] == ':':
                p[0] = ast.SliceExpression(None, p[2], None)
            else:
                p[0] = ast.SliceExpression(p[1], None, None)
        else:
            if p[1] == ':':  # [::step]
                p[0] = ast.SliceExpression(None, None, p[3])
            else:  # [start:stop]
                p[0] = ast.SliceExpression(p[1], p[3], None)

    def p_struct_init_seq(self, p):
        '''struct_init_seq : struct_init
                           | struct_init_seq COMMA struct_init
                           | struct_init_seq COMMA'''
        if len(p) == 2:
            p[0] = [p[1]]
        elif len(p) == 3:
            p[0] = p[1]
        else:
            p[0] = p[1] + [p[3]]

    def p_struct_init(self, p):
        '''struct_init : IDENTIFIER COLON expression
                       | IDENTIFIER EQUALS expression'''
        p[0] = (p[1], p[3])

    def p_primary_expression(self, p):
        '''primary_expression : literal
                              | IDENTIFIER
                              | NONE
                              | SOME LPAREN expression RPAREN
                              | LPAREN RPAREN
                              | LPAREN expression RPAREN
                              | LPAREN expression COMMA expression_seq RPAREN
                              | list_literal
                              | vector_expression
                              | lambda_expression
                              | match_expression
                              | if_expression
                              | handle_expression
                              | perform_expression
                              | resume_expression
                              | try_expression
                              | print_expression
                              | spawn_expression
                              | exclave_expression
                              | move_expression
                              | borrow_expression
                              | to_device_expression
                              | from_device_expression'''
        if len(p) == 2:
            if p.slice[1].type == 'IDENTIFIER':
                name = p[1]
                # Exact keyword spellings only: `true`/`false` literals and
                # the `None` option constructor. Other capitalizations are
                # ordinary identifiers.
                if name == 'true':
                    p[0] = ast.Literal(True)
                elif name == 'false':
                    p[0] = ast.Literal(False)
                elif name == 'None':
                    p[0] = ast.NoneExpression()
                else:
                    p[0] = ast.Variable(name)
            elif p.slice[1].type == 'NONE':
                p[0] = ast.NoneExpression()
            else:
                p[0] = p[1]
        elif len(p) == 3:  # ( )
            p[0] = ast.TupleLiteral([])
        elif len(p) == 4:  # ( expr )
            p[0] = p[2]
        elif len(p) == 5:  # Some( expr )
            p[0] = ast.SomeExpression(p[3])
        else:  # tuple
            p[0] = ast.TupleLiteral([p[2]] + p[4])

    def p_expression_seq(self, p):
        '''expression_seq : expression
                          | expression_seq COMMA expression'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_literal(self, p):
        '''literal : NUMBER
                   | FLOAT
                   | STRING'''
        value = p[1]
        if isinstance(value, tuple) and len(value) == 2 and value[1] == 'string':
            p[0] = ast.Literal(value[0])
        else:
            p[0] = ast.Literal(value)

    def p_literal_fstring(self, p):
        '''literal : FSTRING'''
        # F-string interpolation desugars AT PARSE TIME into a chain of
        # string concatenations: literal segments become string Literals,
        # `{expr}` segments are parsed as ordinary Metaxu expressions and
        # wrapped in the `to_string` builtin (identity on strings, so the
        # concat chain is well-typed for any to_string-able value).  The
        # downstream pipeline (freeze/infer/HIR/MIR, both engines) only ever
        # sees plain `+` and `to_string` calls, which are already supported
        # natively — no new AST node, no codegen changes.
        p[0] = self._desugar_fstring(p[1][0], p.lineno(1))

    # ------------------------------------------------------------------
    # F-string desugaring (parse-time)
    # ------------------------------------------------------------------

    def _fstring_error(self, message: str, lineno: int) -> None:
        raise CompileError(
            message=message,
            error_type="ParseError",
            location=SourceLocation(
                file=getattr(self.lexer, 'source_file', None) or "<unknown>",
                line=lineno, column=0))

    def _split_fstring(self, raw: str, lineno: int):
        """Split f-string text into ('lit', text) / ('expr', text) segments.

        `{{` and `}}` escape to literal braces; `{}` (or whitespace-only
        braces) and unbalanced braces are compile errors — never a silent
        literal fallback.  An expression segment ends at its BALANCING `}`
        (nested braces are tracked), so struct literals and blocks inside
        `{...}` stay whole instead of being mis-split at the first `}`.
        (The FSTRING lexeme cannot contain a quote, so no string literal
        inside a segment can carry a brace that would fool the counter.)
        """
        segments = []
        buf = []
        i, n = 0, len(raw)
        while i < n:
            ch = raw[i]
            if ch == '{':
                if i + 1 < n and raw[i + 1] == '{':
                    buf.append('{')
                    i += 2
                    continue
                depth = 1
                end = i + 1
                while end < n:
                    if raw[end] == '{':
                        depth += 1
                    elif raw[end] == '}':
                        depth -= 1
                        if depth == 0:
                            break
                    end += 1
                if depth != 0:
                    self._fstring_error(
                        f"f-string: unterminated '{{' in f\"{raw}\" "
                        "(use '{{' for a literal brace)", lineno)
                inner = raw[i + 1:end]
                if inner.strip() == "":
                    self._fstring_error(
                        f"f-string: empty expression '{{{inner}}}' in "
                        f"f\"{raw}\"", lineno)
                if buf:
                    segments.append(('lit', ''.join(buf)))
                    buf = []
                segments.append(('expr', inner))
                i = end + 1
            elif ch == '}':
                if i + 1 < n and raw[i + 1] == '}':
                    buf.append('}')
                    i += 2
                    continue
                self._fstring_error(
                    f"f-string: single '}}' in f\"{raw}\" "
                    "(use '}}' for a literal brace)", lineno)
            else:
                buf.append(ch)
                i += 1
        if buf:
            segments.append(('lit', ''.join(buf)))
        return segments

    def _parse_fstring_expr(self, text: str, raw: str, lineno: int):
        """Parse one `{...}` segment as a Metaxu expression.

        Uses a dedicated cached Parser instance (building PLY tables is
        expensive; an FSTRING lexeme cannot contain a quote, hence cannot
        contain a nested f-string, so this parser is never re-entered).
        A segment that fails to parse — or parses to anything other than a
        single expression — is a clear CompileError naming the segment.
        """
        global _FSTRING_SEGMENT_PARSER
        if _FSTRING_SEGMENT_PARSER is None:
            _FSTRING_SEGMENT_PARSER = Parser()
        wrapper = "fn __fstring_expr__() { " + text + " }"
        try:
            module = _FSTRING_SEGMENT_PARSER.parse(
                wrapper, file_path=getattr(self.lexer, 'source_file', None)
                or "<fstring>")
        except CompileError as exc:
            self._fstring_error(
                f"f-string: cannot parse expression segment '{{{text}}}' in "
                f"f\"{raw}\": {exc.message}", lineno)
        fn = module.body.statements[0] if module.body.statements else None
        body = list(getattr(fn, 'body', None) or [])
        if (fn is None or len(body) != 1
                or isinstance(body[0], (ast.Statement, ast.LetBinding))
                or not isinstance(body[0], ast.Node)):
            self._fstring_error(
                f"f-string: segment '{{{text}}}' in f\"{raw}\" is not a "
                "single expression", lineno)
        return body[0]

    def _desugar_fstring(self, raw: str, lineno: int):
        """Desugar f-string text into `lit + to_string(expr) + ...`."""
        segments = self._split_fstring(raw, lineno)
        parts = []
        for kind, text in segments:
            if kind == 'lit':
                parts.append(ast.Literal(text))
            else:
                expr = self._parse_fstring_expr(text, raw, lineno)
                parts.append(ast.FunctionCall("to_string", [expr]))
        if not parts:
            return ast.Literal("")
        result = parts[0]
        for nxt in parts[1:]:
            result = ast.BinaryOperation(result, '+', nxt)
        return result

    def p_list_literal(self, p):
        '''list_literal : LBRACKET RBRACKET
                        | LBRACKET list_elements RBRACKET
                        | LBRACKET list_elements COMMA RBRACKET'''
        if len(p) == 3:
            p[0] = ast.ListLiteral([])
        else:
            p[0] = ast.ListLiteral(p[2])

    def p_list_elements(self, p):
        '''list_elements : list_element
                         | list_elements COMMA list_element'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_list_element(self, p):
        '''list_element : expression
                        | TRIPLE_DOT expression'''
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = ast.SpreadElement(p[2])

    def p_vector_expression(self, p):
        '''vector_expression : VECTOR LBRACKET type_list RBRACKET'''
        type_args = p[3]
        base_type = type_args[0] if type_args else None
        size = type_args[1] if len(type_args) > 1 else None
        p[0] = ast.VectorTypeExpression(base_type, size, type_args)

    # -- arguments ------------------------------------------------------

    def p_argument_list_opt(self, p):
        '''argument_list_opt : argument_list
                             | empty'''
        p[0] = p[1] if p[1] else []

    def p_argument_list(self, p):
        '''argument_list : argument
                         | argument_list COMMA argument'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_argument(self, p):
        '''argument : expression
                    | expression FOR comp_target IN expression
                    | expression ARROW expression'''
        if len(p) == 2:
            p[0] = p[1]
        elif p[2] == '->':  # shorthand lambda: x -> expr, (a, b) -> expr
            params = self._params_from_expr(p[1])
            p[0] = self._make_lambda(params, p[3])
        else:  # comprehension
            p[0] = ast.Comprehension(p[1], p[3], p[5])

    def p_comp_target(self, p):
        '''comp_target : IDENTIFIER
                       | LPAREN identifier_seq RPAREN'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[2]

    def p_identifier_seq(self, p):
        '''identifier_seq : IDENTIFIER
                          | identifier_seq COMMA IDENTIFIER'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    # -- lambdas --------------------------------------------------------

    def p_lambda_expression(self, p):
        '''lambda_expression : FN LPAREN param_list_opt RPAREN block
                             | FN LPAREN param_list_opt RPAREN ARROW block
                             | FN LPAREN param_list_opt RPAREN ARROW expression
                             | FN LPAREN param_list_opt RPAREN ARROW expression block
                             | FN LPAREN param_list_opt RPAREN ARROW expression PERFORMS effect_seq block
                             | OROR block'''
        if p[1] == '||':
            p[0] = self._make_lambda([], p[2])
        elif len(p) == 6:
            p[0] = self._make_lambda(p[3], p[5])
        elif len(p) == 7:
            # fn(params) -> expr (expression-bodied) or fn(params) -> { ... }
            # (block-bodied: the block's tail expression is the result, same
            # as function bodies -- p[6] is an ast.Block in that case). No
            # ambiguity with `-> Type { body }` or struct literals: a bare
            # `{` right after ARROW is never rewritten to LBRACE_STRUCT (the
            # lexer only rewrites `{` after an identifier/`]`/`>`), while a
            # struct-literal body like `-> Point { x: 1 }` arrives as
            # IDENTIFIER LBRACE_STRUCT and parses as an expression.
            p[0] = self._make_lambda(p[3], p[6])
        elif len(p) == 8:  # fn(params) -> Type { body }
            p[0] = self._make_lambda(p[3], p[7], return_type=p[6])
        else:  # fn(params) -> Type performs E { body }
            performs = [self._effect_app_from_type(t) if not isinstance(t, ast.EffectApplication) else t
                        for t in p[8]]
            p[0] = self._make_lambda(p[3], p[9], return_type=p[6], performs=performs)

    # -- control flow expressions --------------------------------------

    def p_if_expression(self, p):
        '''if_expression : IF expression LBRACE statement_list RBRACE
                         | IF expression LBRACE statement_list RBRACE ELSE LBRACE statement_list RBRACE
                         | IF expression LBRACE statement_list RBRACE ELSE if_expression
                         | if_let_expression'''
        if len(p) == 2:
            p[0] = p[1]
        elif len(p) == 6:
            p[0] = ast.IfExpression(p[2], self._branch_of(p[4]), None)
        elif len(p) == 8:
            p[0] = ast.IfExpression(p[2], self._branch_of(p[4]), p[7])
        else:
            p[0] = ast.IfExpression(p[2], self._branch_of(p[4]), self._branch_of(p[8]))

    def p_if_let_expression(self, p):
        '''if_let_expression : IF LET expression EQUALS expression LBRACE statement_list RBRACE
                             | IF LET expression EQUALS expression LBRACE statement_list RBRACE ELSE LBRACE statement_list RBRACE'''
        if len(p) == 9:
            p[0] = ast.IfLetExpression(p[3], p[5], self._branch_of(p[7]), None)
        else:
            p[0] = ast.IfLetExpression(p[3], p[5], self._branch_of(p[7]), self._branch_of(p[11]))

    def p_while_statement(self, p):
        '''while_statement : WHILE expression LBRACE statement_list RBRACE
                           | WHILE LET expression EQUALS expression LBRACE statement_list RBRACE'''
        if len(p) == 6:
            p[0] = ast.WhileStatement(p[2], ast.Block(p[4] or []))
        else:
            p[0] = ast.WhileLetStatement(p[3], p[5], ast.Block(p[7] or []))

    def p_for_statement(self, p):
        '''for_statement : FOR IDENTIFIER IN expression LBRACE statement_list RBRACE'''
        p[0] = ast.ForStatement(p[2], p[4], p[6] or [])

    def p_match_expression(self, p):
        '''match_expression : MATCH expression LBRACE arm_list RBRACE'''
        p[0] = ast.MatchExpression(p[2], p[4])

    def p_arm_list(self, p):
        '''arm_list : arm
                    | arm_list arm
                    | arm_list COMMA arm
                    | arm_list COMMA'''
        if len(p) == 2:
            p[0] = [p[1]]
        elif len(p) == 3:
            if p[2] == ',':
                p[0] = p[1]
            else:
                p[0] = p[1] + [p[2]]
        else:
            p[0] = p[1] + [p[3]]

    def p_arm(self, p):
        '''arm : expression arm_arrow arm_body'''
        p[0] = (p[1], p[3])

    def p_arm_arrow(self, p):
        '''arm_arrow : ARROW
                     | FATARROW'''
        p[0] = p[1]

    def p_arm_body(self, p):
        '''arm_body : expression
                    | block'''
        p[0] = p[1]

    # -- effects --------------------------------------------------------

    def p_handle_expression(self, p):
        '''handle_expression : HANDLE expression WITH LBRACE arm_list RBRACE IN in_target
                             | HANDLE expression LBRACE arm_list RBRACE'''
        if len(p) == 9:
            effect_name, type_args = self._effect_name_of(p[2])
            cases = [self._handle_case_of(pat, body) for pat, body in p[5]]
            handle = ast.HandleEffect(effect_name, cases, p[8])
            handle.type_args = type_args
            p[0] = handle
        else:
            p[0] = ast.HandleBlock(p[2], p[4])

    def _handle_case_of(self, pattern, body):
        """Convert an arm pattern (parsed as a call expression) to a HandleCase."""
        op_name = None
        param_name = None
        if isinstance(pattern, ast.FunctionCall):
            op_name = pattern.name
            args = pattern.arguments
        elif isinstance(pattern, ast.QualifiedFunctionCall):
            op_name = '.'.join(pattern.parts)
            args = pattern.arguments
        elif isinstance(pattern, ast.Variable):
            op_name = pattern.name
            args = []
        else:
            op_name = str(pattern)
            args = []
        param_names = [a.name if isinstance(a, ast.Variable) else str(a) for a in args]
        if param_names:
            param_name = param_names[0]
        case = ast.HandleCase(op_name, param_name, body)
        case.param_names = param_names  # full parameter list (multi-arg ops)
        return case

    def p_in_target(self, p):
        '''in_target : expression
                     | block'''
        p[0] = p[1]

    def p_perform_expression(self, p):
        '''perform_expression : PERFORM postfix_expression'''
        operand = p[2]
        if isinstance(operand, ast.QualifiedFunctionCall):
            p[0] = ast.PerformEffect('.'.join(operand.parts), operand.arguments)
        elif isinstance(operand, ast.FunctionCall):
            p[0] = ast.PerformEffect(operand.name, operand.arguments)
        elif isinstance(operand, ast.MethodCall):
            name, _ = self._effect_name_of(operand.receiver)
            p[0] = ast.PerformEffect(f"{name}.{operand.method}", operand.arguments)
        else:
            name, _ = self._effect_name_of(operand)
            p[0] = ast.PerformEffect(name, [])

    def p_resume_expression(self, p):
        '''resume_expression : RESUME LPAREN RPAREN
                             | RESUME LPAREN expression RPAREN'''
        if len(p) == 4:
            p[0] = ast.Resume(None)
        else:
            p[0] = ast.Resume(p[3])

    def p_try_expression(self, p):
        '''try_expression : TRY block CATCH IDENTIFIER block'''
        p[0] = ast.TryCatch(p[2], p[4], p[5])

    # -- misc primaries -------------------------------------------------

    def p_print_expression(self, p):
        '''print_expression : PRINT LPAREN argument_list_opt RPAREN'''
        p[0] = ast.PrintStatement(p[3])

    def p_spawn_expression(self, p):
        '''spawn_expression : SPAWN LPAREN expression RPAREN'''
        p[0] = ast.SpawnExpression(p[3])

    def p_exclave_expression(self, p):
        '''exclave_expression : EXCLAVE expression'''
        p[0] = ast.ExclaveExpression(p[2])

    def p_move_expression(self, p):
        '''move_expression : MOVE LPAREN IDENTIFIER RPAREN'''
        p[0] = ast.Move(p[3])

    def p_borrow_expression(self, p):
        '''borrow_expression : BORROW IDENTIFIER AS type_expression
                             | BORROW IDENTIFIER'''
        if len(p) == 5:
            p[0] = ast.BorrowExpression(p[2], p[4])
        else:
            p[0] = ast.BorrowExpression(p[2], None)

    def p_to_device_expression(self, p):
        '''to_device_expression : TO_DEVICE LPAREN IDENTIFIER RPAREN'''
        p[0] = ast.ToDevice(p[3])

    def p_from_device_expression(self, p):
        '''from_device_expression : FROM_DEVICE LPAREN IDENTIFIER RPAREN'''
        p[0] = ast.FromDevice(p[3])

    # ------------------------------------------------------------------
    # Functions
    # ------------------------------------------------------------------

    def p_function_declaration(self, p):
        '''function_declaration : FN IDENTIFIER type_params_opt LPAREN param_list_opt RPAREN fn_tail LBRACE statement_list RBRACE'''
        return_type, performs, where_clause = p[7]

        func = ast.FunctionDeclaration(name=p[2], params=[], body=[])
        func.scope = ast.Scope(name=f"function_{p[2]}")
        self._enter_scope(func.scope)
        func.type_params = p[3] or []
        func.params = p[5] or []
        func.return_type = return_type if return_type is not None else NoneType
        func.performs = performs or []
        func.where_clause = where_clause
        func.body = p[9] or []

        self._populate_scope_symbols(func, func.scope)
        self._update_child_scopes(func)
        self._exit_scope()
        p[0] = func

    def p_fn_tail(self, p):
        '''fn_tail : empty
                   | ARROW type_expression
                   | ARROW type_expression where_clause
                   | ARROW type_expression PERFORMS effect_seq
                   | ARROW type_expression PERFORMS effect_seq where_clause
                   | PERFORMS effect_seq
                   | PERFORMS effect_seq ARROW type_expression'''
        return_type = None
        performs = []
        where_clause = None
        if len(p) == 2:
            pass
        elif p[1] == '->':
            return_type = p[2]
            if len(p) == 4:
                where_clause = p[3]
            elif len(p) == 5:
                performs = p[4]
            elif len(p) == 6:
                performs = p[4]
                where_clause = p[5]
        else:  # performs first
            performs = p[2]
            if len(p) == 5:
                return_type = p[4]
        performs = [self._effect_app_from_type(t) if not isinstance(t, ast.EffectApplication) else t
                    for t in (performs or [])]
        p[0] = (return_type, performs, where_clause)

    def p_effect_seq(self, p):
        '''effect_seq : type_postfix
                      | effect_seq COMMA type_postfix'''
        if len(p) == 2:
            p[0] = [self._effect_app_from_type(p[1])]
        else:
            p[0] = p[1] + [self._effect_app_from_type(p[3])]

    def p_param_list_opt(self, p):
        '''param_list_opt : param_list
                          | empty'''
        p[0] = p[1] if p[1] else []

    def p_param_list(self, p):
        '''param_list : parameter
                      | param_list COMMA parameter'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_parameter(self, p):
        '''parameter : IDENTIFIER COLON type_expression
                     | mode_annotation_list IDENTIFIER COLON type_expression
                     | IDENTIFIER mode_annotation_list COLON type_expression
                     | IDENTIFIER
                     | mode_annotation_list IDENTIFIER'''
        if len(p) == 2:
            p[0] = ast.Parameter(p[1])
        elif len(p) == 3:
            p[0] = ast.Parameter(p[2], mode=p[1])
        elif len(p) == 4:
            p[0] = ast.Parameter(p[1], p[3])
        else:
            if isinstance(p[1], list):  # @mode name : T
                p[0] = ast.Parameter(p[2], p[4], mode=p[1])
            else:  # name @mode : T (legacy)
                p[0] = ast.Parameter(p[1], p[4], mode=p[2])

    # ------------------------------------------------------------------
    # Type parameters (declarations)
    # ------------------------------------------------------------------

    def p_type_params_opt(self, p):
        '''type_params_opt : LGENERIC type_param_seq RGENERIC
                           | empty'''
        p[0] = p[2] if len(p) == 4 else []

    def p_type_param_seq(self, p):
        '''type_param_seq : type_param
                          | type_param_seq COMMA type_param'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_type_param(self, p):
        '''type_param : IDENTIFIER
                      | IDENTIFIER COLON type_bound
                      | CONST IDENTIFIER COLON type_expression
                      | PLUS IDENTIFIER
                      | MINUS IDENTIFIER'''
        if len(p) == 2:
            p[0] = ast.TypeParameter(p[1], None)
        elif len(p) == 3:  # variance-annotated
            p[0] = ast.TypeParameter(p[2], None)
        elif p[1] == 'const':
            param = ast.TypeParameter(p[2], p[4])
            param.is_const = True
            p[0] = param
        else:
            p[0] = ast.TypeParameter(p[1], p[3])

    def p_type_bound(self, p):
        '''type_bound : type_postfix
                      | type_bound PLUS type_postfix'''
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = ast.CompoundTypeBound(p[1], p[3]) if hasattr(ast, 'CompoundTypeBound') else [p[1], p[3]]

    # ------------------------------------------------------------------
    # Type expressions
    # ------------------------------------------------------------------

    def p_type_expression(self, p):
        '''type_expression : type_postfix
                           | mode_annotation type_expression
                           | UNIQUE type_expression
                           | EXCLUSIVE type_expression
                           | TIMES type_expression
                           | TIMES MUT type_expression
                           | TIMES CONST type_expression
                           | AMPERSAND type_expression
                           | LBRACKET type_expression RBRACKET
                           | fn_type'''
        if len(p) == 2:
            p[0] = p[1]
        elif isinstance(p[1], ast.ModeAnnotation):
            p[0] = ast.ModeTypeAnnotation(p[2], uniqueness=p[1].mode_type)
        elif p[1] == 'unique' or p[1] == 'exclusive':
            p[0] = ast.ModeTypeAnnotation(p[2], uniqueness=p[1])
        elif p[1] == '*':
            if len(p) == 3:
                p[0] = PointerType(base_type=p[2], is_mut=False)
            else:
                p[0] = PointerType(base_type=p[3], is_mut=(p[2] == 'mut'))
        elif p[1] == '&':
            p[0] = ast.TypeApplication("Ref", [p[2]])
        elif p[1] == '[':
            p[0] = ast.TypeApplication("Slice", [p[2]])
        else:
            p[0] = p[1]

    def p_fn_type(self, p):
        '''fn_type : FN LPAREN type_list RPAREN ARROW type_expression fn_type_performs_opt
                   | FN LPAREN RPAREN ARROW type_expression fn_type_performs_opt
                   | FN BACKSLASH LPAREN type_list RPAREN ARROW type_expression
                   | FN BACKSLASH LPAREN RPAREN ARROW type_expression'''
        if p[2] == '\\':
            if len(p) == 8:
                fn_type = ast.FunctionType(p[4], p[7])
            else:
                fn_type = ast.FunctionType([], p[6])
        else:
            if len(p) == 8:
                fn_type = ast.FunctionType(p[3], p[6])
                fn_type.performs = p[7] or []
            else:
                fn_type = ast.FunctionType([], p[5])
                fn_type.performs = p[6] or []
        p[0] = fn_type

    def p_fn_type_performs_opt(self, p):
        '''fn_type_performs_opt : PERFORMS effect_union
                                | empty'''
        p[0] = p[2] if len(p) == 3 else []

    def p_effect_union(self, p):
        '''effect_union : type_postfix
                        | effect_union PIPE type_postfix'''
        if len(p) == 2:
            p[0] = [self._effect_app_from_type(p[1])]
        else:
            p[0] = p[1] + [self._effect_app_from_type(p[3])]

    def p_type_postfix(self, p):
        '''type_postfix : IDENTIFIER
                        | VOID
                        | SIZE_T
                        | LPAREN RPAREN
                        | VECTOR LBRACKET type_list RBRACKET
                        | VECTOR LBRACKET type_list RBRACKET LPAREN RPAREN
                        | type_postfix LBRACKET type_list RBRACKET
                        | type_postfix LGENERIC type_list RGENERIC
                        | type_postfix DOT IDENTIFIER'''
        if len(p) == 2:
            p[0] = ast.TypeReference(p[1])
        elif len(p) == 3:  # ( )
            p[0] = ast.TypeReference("Unit")
        elif p[1] == 'vector':
            p[0] = ast.TypeApplication("vector", p[3])
        elif p.slice[2].type == 'DOT':
            base = p[1]
            base_name = base.name if isinstance(base, ast.TypeReference) else str(base)
            p[0] = ast.TypeReference(f"{base_name}.{p[3]}")
        else:  # type application with [] or <>
            base = p[1]
            base_name = base.name if isinstance(base, ast.TypeReference) else base
            if isinstance(base_name, ast.TypeApplication):
                p[0] = ast.TypeApplication(base_name.type_constructor, base_name.type_args + p[3])
            else:
                p[0] = ast.TypeApplication(base_name, p[3])

    def p_type_list(self, p):
        '''type_list : type_item
                     | type_list COMMA type_item'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_type_item(self, p):
        '''type_item : type_expression
                     | NUMBER'''
        if isinstance(p[1], int):
            p[0] = ast.TypeReference(str(p[1]))
        else:
            p[0] = p[1]

    # ------------------------------------------------------------------
    # Struct / enum definitions
    # ------------------------------------------------------------------

    def p_any_lbrace(self, p):
        '''any_lbrace : LBRACE
                      | LBRACE_STRUCT'''
        p[0] = p[1]

    def p_struct_definition(self, p):
        '''struct_definition : STRUCT IDENTIFIER type_params_opt any_lbrace struct_field_seq RBRACE'''
        p[0] = ast.StructDefinition(name=p[2], fields=p[5], type_params=p[3] or None)

    def p_struct_field_seq(self, p):
        '''struct_field_seq : struct_field
                            | struct_field_seq COMMA struct_field
                            | struct_field_seq struct_field
                            | struct_field_seq COMMA
                            | empty'''
        if len(p) == 2:
            p[0] = [] if p[1] is None else [p[1]]
        elif len(p) == 3:
            if p[2] == ',':
                p[0] = p[1]
            else:
                p[0] = p[1] + [p[2]]
        else:
            p[0] = p[1] + [p[3]]

    def p_struct_field(self, p):
        '''struct_field : IDENTIFIER COLON type_expression
                        | mode_annotation_list IDENTIFIER COLON type_expression
                        | visibility_modifier IDENTIFIER COLON type_expression
                        | visibility_modifier mode_annotation_list IDENTIFIER COLON type_expression
                        | IDENTIFIER EQUALS expression'''
        if len(p) == 4:
            if p[2] == ':':
                p[0] = ast.StructField(name=p[1], type_info=p[3])
            else:
                p[0] = ast.StructField(name=p[1], value=p[3])
        elif len(p) == 5:
            if isinstance(p[1], list):  # mode-annotated field
                field = ast.StructField(name=p[2], type_info=p[4])
                field.modes = p[1]
                p[0] = field
            else:  # visibility-annotated
                field = ast.StructField(name=p[2], type_info=p[4])
                field.visibility = p[1]
                p[0] = field
        else:
            field = ast.StructField(name=p[3], type_info=p[5])
            field.visibility = p[1]
            field.modes = p[2]
            p[0] = field

    def p_enum_definition(self, p):
        '''enum_definition : ENUM IDENTIFIER type_params_opt LBRACE variant_seq RBRACE'''
        enum = ast.EnumDefinition(p[2], p[5])
        enum.type_params = p[3] or []
        p[0] = enum

    def p_variant_seq(self, p):
        '''variant_seq : variant_definition
                       | variant_seq COMMA variant_definition
                       | variant_seq COMMA'''
        if len(p) == 2:
            p[0] = [p[1]]
        elif len(p) == 3:
            p[0] = p[1]
        else:
            p[0] = p[1] + [p[3]]

    def p_variant_definition(self, p):
        '''variant_definition : IDENTIFIER
                              | IDENTIFIER LPAREN variant_field_seq RPAREN'''
        if len(p) == 2:
            p[0] = ast.VariantDefinition(p[1], [])
        else:
            p[0] = ast.VariantDefinition(p[1], p[3])

    def p_variant_field_seq(self, p):
        '''variant_field_seq : variant_field
                             | variant_field_seq COMMA variant_field'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_variant_field(self, p):
        '''variant_field : IDENTIFIER COLON type_expression
                         | type_expression'''
        if len(p) == 4:
            p[0] = (p[1], p[3])
        else:
            p[0] = (None, p[1])

    # ------------------------------------------------------------------
    # Traits (interfaces) and implementations
    # ------------------------------------------------------------------

    def p_trait_definition(self, p):
        '''trait_definition : trait_keyword IDENTIFIER type_params_opt LBRACE trait_item_seq RBRACE
                            | trait_keyword IDENTIFIER type_params_opt EXTENDS type_list LBRACE trait_item_seq RBRACE'''
        if len(p) == 7:
            p[0] = ast.InterfaceDefinition(p[2], p[3], p[5])
        else:
            p[0] = ast.InterfaceDefinition(p[2], p[3], p[7], extends=p[5])

    def p_trait_keyword(self, p):
        '''trait_keyword : TRAIT
                         | INTERFACE'''
        p[0] = p[1]

    def p_trait_item_seq(self, p):
        '''trait_item_seq : method_signature
                          | trait_item_seq method_signature
                          | trait_item_seq SEMICOLON
                          | empty'''
        if len(p) == 2:
            p[0] = [] if p[1] is None else [p[1]]
        else:
            if p[2] == ';':
                p[0] = p[1]
            else:
                p[0] = p[1] + [p[2]]

    def p_method_signature(self, p):
        '''method_signature : FN IDENTIFIER type_params_opt LPAREN param_list_opt RPAREN
                            | FN IDENTIFIER type_params_opt LPAREN param_list_opt RPAREN ARROW type_expression'''
        if len(p) == 7:
            p[0] = ast.MethodDefinition(p[2], p[5], None, type_params=p[3])
        else:
            p[0] = ast.MethodDefinition(p[2], p[5], p[8], type_params=p[3])

    def p_implementation(self, p):
        '''implementation : IMPLEMENT type_params_opt type_expression FOR type_expression where_clause_opt LBRACE impl_item_seq RBRACE
                          | IMPLEMENT type_params_opt type_expression where_clause_opt LBRACE impl_item_seq RBRACE
                          | IMPLEMENTS type_expression COLON type_expression LBRACE impl_item_seq RBRACE'''
        if p[1] == 'implements':
            impl = ast.Implementation(p[4], p[2], None, p[6])
            p[0] = impl
        elif len(p) == 10:
            p[0] = ast.Implementation(p[3], p[5], p[2], p[8], where_clause=p[6])
        else:
            impl = ast.Implementation(None, p[3], p[2], p[6], where_clause=p[4])
            p[0] = impl

    def p_impl_item_seq(self, p):
        '''impl_item_seq : function_declaration
                         | impl_item_seq function_declaration
                         | impl_item_seq SEMICOLON
                         | empty'''
        if len(p) == 2:
            p[0] = [] if p[1] is None else [p[1]]
        else:
            if p[2] == ';':
                p[0] = p[1]
            else:
                p[0] = p[1] + [p[2]]

    def p_where_clause_opt(self, p):
        '''where_clause_opt : where_clause
                            | empty'''
        p[0] = p[1]

    def p_where_clause(self, p):
        '''where_clause : WHERE type_constraint_seq'''
        p[0] = ast.WhereClause(p[2])

    def p_type_constraint_seq(self, p):
        '''type_constraint_seq : type_constraint
                               | type_constraint_seq COMMA type_constraint'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_type_constraint(self, p):
        '''type_constraint : type_postfix COLON type_bound
                           | type_postfix EXTENDS type_bound
                           | type_postfix IMPLEMENTS type_bound'''
        kind = {'extends': 'extends', 'implements': 'implements'}.get(p[2], 'subtype')
        p[0] = ast.TypeConstraint(p[1], p[3], kind=kind)

    # ------------------------------------------------------------------
    # Effect declarations
    # ------------------------------------------------------------------

    def p_effect_declaration(self, p):
        '''effect_declaration : EFFECT IDENTIFIER type_params_opt effect_class_opt effect_eq_opt LBRACE effect_op_seq RBRACE'''
        p[0] = ast.EffectDeclaration(p[2], p[3] or [], p[7], p[4])

    def p_effect_class_opt(self, p):
        '''effect_class_opt : COLON IDENTIFIER
                            | empty'''
        p[0] = p[2] if len(p) == 3 else None

    def p_effect_eq_opt(self, p):
        '''effect_eq_opt : EQUALS
                         | empty'''
        p[0] = p[1]

    def p_effect_op_seq(self, p):
        '''effect_op_seq : effect_operation
                         | effect_op_seq effect_operation
                         | effect_op_seq SEMICOLON
                         | empty'''
        if len(p) == 2:
            p[0] = [] if p[1] is None else [p[1]]
        else:
            if p[2] == ';':
                p[0] = p[1]
            else:
                p[0] = p[1] + [p[2]]

    def p_effect_operation(self, p):
        '''effect_operation : IDENTIFIER type_params_opt LPAREN param_list_opt RPAREN ARROW type_expression effect_op_default_opt effect_with_opt
                            | FN IDENTIFIER type_params_opt LPAREN param_list_opt RPAREN ARROW type_expression effect_op_default_opt effect_with_opt
                            | FN IDENTIFIER LBRACKET type_param_seq RBRACKET LPAREN param_list_opt RPAREN ARROW type_expression effect_op_default_opt effect_with_opt'''
        if len(p) == 10:
            op = ast.EffectOperation(p[1], p[4], p[7], c_effect=p[9])
            op.type_params = p[2] or []
            default = p[8]
        elif len(p) == 11:
            op = ast.EffectOperation(p[2], p[5], p[8], c_effect=p[10])
            op.type_params = p[3] or []
            default = p[9]
        else:
            op = ast.EffectOperation(p[2], p[7], p[10], c_effect=p[12])
            op.type_params = p[4] or []
            default = p[11]
        # Default handler expression: `op(...) -> T = expr;` declares the
        # value the operation yields when performed with NO handler in
        # scope (capability-style effects: absence of a handler answers
        # the default instead of aborting). Stored on an underscored attr
        # so the generic desugar/freeze walks leave it alone; the HIR
        # builder compiles it into a __effect_default$Effect$op function.
        op._default_expr = default
        p[0] = op

    def p_effect_op_default_opt(self, p):
        '''effect_op_default_opt : EQUALS expression
                                 | empty'''
        p[0] = p[2] if len(p) == 3 else None

    def p_effect_with_opt(self, p):
        '''effect_with_opt : WITH IDENTIFIER
                           | empty'''
        p[0] = p[2] if len(p) == 3 else None

    # ------------------------------------------------------------------
    # Extern / unsafe
    # ------------------------------------------------------------------

    def p_extern_block(self, p):
        '''extern_block : EXTERN STRING LBRACE extern_item_seq RBRACE'''
        header_path = p[2][0] if isinstance(p[2], tuple) else str(p[2]).strip('"')
        p[0] = ExternBlock(header_path=header_path, declarations=p[4])

    def p_extern_item_seq(self, p):
        '''extern_item_seq : extern_item
                           | extern_item_seq extern_item
                           | extern_item_seq SEMICOLON
                           | empty'''
        if len(p) == 2:
            p[0] = [] if p[1] is None else [p[1]]
        else:
            if p[2] == ';':
                p[0] = p[1]
            else:
                p[0] = p[1] + [p[2]]

    def p_extern_item(self, p):
        '''extern_item : FN IDENTIFIER LPAREN param_list_opt RPAREN ARROW type_expression
                       | FN IDENTIFIER LPAREN param_list_opt RPAREN
                       | TYPE IDENTIFIER'''
        if p[1] == 'type':
            p[0] = ExternTypeDeclaration(name=p[2], is_opaque=True)
        elif len(p) == 8:
            p[0] = ExternFunctionDeclaration(name=p[2], params=p[4], return_type=p[7])
        else:
            p[0] = ExternFunctionDeclaration(name=p[2], params=p[4], return_type=None)

    def p_extern_type_statement(self, p):
        '''extern_type_statement : EXTERN TYPE IDENTIFIER
                                 | EXTERN TYPE IDENTIFIER LBRACKET type_list RBRACKET'''
        p[0] = ExternTypeDeclaration(name=p[3], is_opaque=True)

    def p_unsafe_block(self, p):
        '''unsafe_block : UNSAFE LBRACE statement_list RBRACE'''
        p[0] = UnsafeBlock(body=p[3] or [])

    # ------------------------------------------------------------------
    # Type definitions
    # ------------------------------------------------------------------

    def p_type_definition(self, p):
        '''type_definition : TYPE IDENTIFIER type_params_opt EQUALS type_expression'''
        p[0] = ast.TypeDefinition(p[2], p[3] or [], p[5])

    # ------------------------------------------------------------------
    # Modules and imports
    # ------------------------------------------------------------------

    def p_module_declaration(self, p):
        '''module_declaration : MODULE module_path LBRACE module_body RBRACE
                              | MODULE module_path'''
        name = p[2]
        if name in self.module_names:
            raise CompileError(
                message=f"Duplicate module name '{name}'",
                error_type="ParseError",
                location=self.location_for_offsets(
                    p.lexpos(1), getattr(p.slice[1], 'endlexpos', p.lexpos(1))),
                notes=[f"Module '{name}' was already declared"]
            )
        self.module_names.add(name)
        body = p[4] if len(p) == 6 else ast.ModuleBody(statements=[])
        p[0] = ast.Module(name=name, body=body)

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
            for stmt in p[2]:
                if isinstance(stmt, ast.VisibilityRules):
                    visibility_rules = stmt
                else:
                    statements.append(stmt)
        p[0] = ast.ModuleBody(statements=statements, exports=p[1], visibility_rules=visibility_rules)

    def p_exports(self, p):
        '''exports : EXPORT LBRACE export_list RBRACE
                   | EXPORT LBRACE export_list COMMA RBRACE
                   | empty'''
        if len(p) >= 5:
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

    def p_export_statement(self, p):
        '''export_statement : EXPORT LBRACE export_list RBRACE
                            | EXPORT LBRACE export_list COMMA RBRACE'''
        # A file-level `export { ... }` list: files are modules too, so they
        # need the same explicit-export syntax module blocks get from
        # p_module_body. Carried as a statement and consumed by the module
        # resolution pass (compiler/module_loader.py).
        p[0] = ast.ExportDeclaration(names=p[3])

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
                         | DOTDOT module_path
                         | TRIPLE_DOT module_path'''
        level = {'.': 1, '..': 2, '...': 3}[p[1]]
        p[0] = ast.RelativePath(level, p[2])

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

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def p_visibility_block(self, p):
        '''visibility_block : VISIBILITY any_lbrace visibility_rule_list RBRACE'''
        p[0] = ast.VisibilityRules(rules=p[3])

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

    def p_visibility_modifier(self, p):
        '''visibility_modifier : PUBLIC
                             | PRIVATE
                             | PROTECTED'''
        p[0] = p[1]

    # ------------------------------------------------------------------
    # Comptime
    # ------------------------------------------------------------------

    def p_comptime_block(self, p):
        '''comptime_block : COMPTIME LBRACE statement_list RBRACE'''
        p[0] = ast.ComptimeBlock(statements=p[3] or [])

    def p_comptime_function(self, p):
        '''comptime_function : COMPTIME function_declaration'''
        func = p[2]
        comptime = ast.ComptimeFunction(
            name=func.name,
            type_params=func.type_params,
            params=func.params,
            return_type=func.return_type,
            body=func.body,
            is_comptime=True,
        )
        comptime.scope = func.scope
        p[0] = comptime

    # ------------------------------------------------------------------
    # Empty / errors
    # ------------------------------------------------------------------

    def p_empty(self, p):
        'empty :'
        pass

    def p_error(self, p):
        if p:
            msg = f"Syntax error at '{p.value}'"
            lexpos = getattr(p, 'lexpos', None)
            location = self.location_for_offsets(
                lexpos, getattr(p, 'endlexpos', lexpos)) if lexpos is not None else \
                SourceLocation(file=self.lexer.source_file,
                               line=getattr(p, 'lineno', 0),
                               column=getattr(p, 'column', 0))
            # The location now renders the offending line with a caret (see
            # errors.format_diagnostic), so the old multi-line `context`
            # block and the Python-level stack trace are redundant noise on
            # what is a plain user syntax error.
            raise CompileError(
                message=msg,
                error_type="ParseError",
                location=location,
                notes=["Check syntax near this location"]
            )
        else:
            raise CompileError(
                message="Syntax error at EOF",
                error_type="ParseError",
                location=self._eof_location(),
                notes=["Unexpected end of file"]
            )

    def _eof_location(self) -> 'SourceLocation | None':
        """Location of the end of the current source (for EOF errors)."""
        source = getattr(self.lexer, 'source', None)
        if not isinstance(source, str) or not source:
            return None
        return self.location_for_offsets(len(source.rstrip()) - 1)

    # ------------------------------------------------------------------
    # Scope management helpers
    # ------------------------------------------------------------------

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
            if hasattr(current_scope, 'params'):
                for param in current_scope.params:
                    if param.name == var_name:
                        return True
            if hasattr(current_scope, 'declarations'):
                for decl in current_scope.declarations:
                    if hasattr(decl, 'name') and decl.name == var_name:
                        return True
            if var_name in current_scope.symbols:
                return True
            current_scope = current_scope.parent
        return False

    def _is_variable_mutated(self, var_name, node):
        """Check if a variable is mutated in the given AST node"""
        if node is None:
            return False
        if isinstance(node, list):
            return any(self._is_variable_mutated(var_name, item) for item in node)
        if isinstance(node, ast.Assignment) and node.name == var_name:
            return True
        if isinstance(node, ast.FunctionCall):
            for arg in node.arguments:
                if isinstance(arg, ast.Variable) and arg.name == var_name:
                    # Conservatively assume function calls might mutate
                    return True
        if isinstance(node, ast.Block):
            return any(self._is_variable_mutated(var_name, stmt) for stmt in node.statements)
        for child in getattr(node, 'children', []):
            if self._is_variable_mutated(var_name, child):
                return True
        return False

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
                    node.scope.parent = parent_scope

        # Second pass: process captures now that scopes are linked
        for node, task in self.deferred_processing:
            if task == 'captures':
                self._process_captures(node)
        self.deferred_processing = []

    def _process_captures(self, node):
        """Process variable captures for a node"""
        if isinstance(node, ast.LambdaExpression):
            self._process_lambda_captures(node)
        elif isinstance(node, ast.SpawnExpression):
            self._process_spawn_captures(node)

    def _process_lambda_captures(self, lambda_expr):
        """Process variable captures for a lambda expression."""
        lambda_expr.captured_vars = set()
        lambda_expr.capture_modes = {}

        var_refs = self._find_variables_in_body(lambda_expr.body)
        for var_name in var_refs:
            if any(param.name == var_name for param in lambda_expr.params):
                continue
            if lambda_expr.scope and self._is_variable_defined(var_name, lambda_expr.scope.parent):
                lambda_expr.captured_vars.add(var_name)
                if self._is_variable_mutated(var_name, lambda_expr.body):
                    lambda_expr.capture_modes[var_name] = "borrow_mut"
                else:
                    lambda_expr.capture_modes[var_name] = "borrow"

    def _process_spawn_captures(self, spawn):
        """Process variable captures for a spawn expression."""
        spawn.captured_vars = set()
        spawn.capture_modes = {}
        var_refs = self._find_variables_in_body(spawn.function_expression)
        scope = getattr(spawn, 'scope', None)
        for var_name in var_refs:
            if scope and self._is_variable_defined(var_name, scope.parent):
                spawn.captured_vars.add(var_name)
                spawn.capture_modes[var_name] = "move"

    def _find_variables_in_body(self, node):
        """Recursively find all variable references in a node"""
        vars = set()
        if node is None:
            return vars
        if isinstance(node, list):
            for item in node:
                vars.update(self._find_variables_in_body(item))
            return vars
        if isinstance(node, ast.Variable):
            vars.add(node.name)
        elif isinstance(node, ast.Block):
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
        return vars

    def _populate_scope_symbols(self, node, scope):
        """Populate a scope's symbol table with all declarations from a node."""
        if isinstance(node, (ast.FunctionDeclaration, ast.LambdaExpression)):
            for param in node.params:
                scope.add_symbol(param.name, param)
            body = node.body
            statements = []
            if isinstance(body, list):
                statements = body
            elif isinstance(body, ast.Block):
                statements = body.statements
            elif isinstance(body, ast.LetStatement):
                statements = [body]
            for stmt in statements:
                if isinstance(stmt, ast.LetStatement):
                    for binding in stmt.bindings:
                        scope.add_symbol(binding.identifier, binding)

    def _find_parent_scope(self, node):
        """Walk up the AST to find the nearest enclosing scope"""
        current = getattr(node, 'parent', None)
        while current:
            if hasattr(current, 'scope') and isinstance(current, scoped_nodes) and current.scope:
                return current.scope
            current = getattr(current, 'parent', None)
        return self.current_module.scope if self.current_module else None

    def _update_child_scopes(self, node):
        """Update scope parents for any child nodes that need it"""
        if isinstance(node, (ast.LambdaExpression, ast.FunctionDeclaration)):
            parent_scope = self._find_parent_scope(node)
            if parent_scope:
                node.scope.parent = parent_scope
        for child in getattr(node, 'children', []):
            if hasattr(child, 'children'):
                self._update_child_scopes(child)
