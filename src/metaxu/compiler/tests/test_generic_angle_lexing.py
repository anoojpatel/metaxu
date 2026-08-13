"""Regression tests for generic-angle-bracket disambiguation (lexer Pass B).

The lexer re-tags ``<``/``>`` as LGENERIC/RGENERIC when they enclose type
arguments.  These tests pin down the boundary: comparison expressions in
call-argument lists (``f(a < b, c > d)``) must stay LESS/GREATER, while real
generic uses (``Stack<Int>{..}``, ``identity<Int>(5)``, ``LinkedList[T]``)
keep working.
"""

import pytest

from metaxu.lexer import Lexer
from metaxu.parser import Parser
import metaxu.metaxu_ast as ast


def lex_types(source):
    lx = Lexer()
    lx.input(source)
    return [t.type for t in lx._tokens]


def angle_types(source):
    return [t for t in lex_types(source)
            if t in ('LESS', 'GREATER', 'LGENERIC', 'RGENERIC')]


def parse(source):
    return Parser().parse(source)


def main_body(module):
    fn = module.body.statements[0]
    assert isinstance(fn, ast.FunctionDeclaration)
    return fn.body


def let_initializer(stmt):
    assert isinstance(stmt, ast.LetStatement)
    return stmt.bindings[0].initializer


# ---------------------------------------------------------------------------
# The bug: comparisons in argument lists must not become generics
# ---------------------------------------------------------------------------

class TestComparisonsInCallArguments:
    SRC = "fn main() { let r = f(a < b, c > d); }"

    def test_angles_lex_as_comparisons(self):
        assert angle_types(self.SRC) == ['LESS', 'GREATER']

    def test_parses_as_call_with_two_comparison_args(self):
        module = parse(self.SRC)
        call = let_initializer(main_body(module)[0])
        assert isinstance(call, ast.FunctionCall)
        assert call.name == 'f'
        assert len(call.arguments) == 2
        first, second = call.arguments
        assert isinstance(first, ast.ComparisonExpression)
        assert first.operator == '<'
        assert first.left.name == 'a'
        assert first.right.name == 'b'
        assert isinstance(second, ast.ComparisonExpression)
        assert second.operator == '>'
        assert second.left.name == 'c'
        assert second.right.name == 'd'

    def test_comparison_followed_by_number_stays_comparison(self):
        src = "fn main() { let r = f(a < b, c > 3); }"
        assert angle_types(src) == ['LESS', 'GREATER']
        call = let_initializer(main_body(parse(src))[0])
        assert isinstance(call, ast.FunctionCall)
        assert len(call.arguments) == 2
        assert all(isinstance(a, ast.ComparisonExpression)
                   for a in call.arguments)


class TestBareComparisons:
    def test_less_alone_is_comparison(self):
        src = "fn main() { let r = a < b; }"
        assert angle_types(src) == ['LESS']
        cmp_expr = let_initializer(main_body(parse(src))[0])
        assert isinstance(cmp_expr, ast.ComparisonExpression)
        assert cmp_expr.operator == '<'

    def test_greater_alone_is_comparison(self):
        src = "fn main() { let r = c > d; }"
        assert angle_types(src) == ['GREATER']
        cmp_expr = let_initializer(main_body(parse(src))[0])
        assert isinstance(cmp_expr, ast.ComparisonExpression)
        assert cmp_expr.operator == '>'

    def test_comparison_in_if_condition(self):
        src = "fn main() { if a < b { return 1; } }"
        assert angle_types(src) == ['LESS']
        parse(src)


# ---------------------------------------------------------------------------
# Real generic uses must keep re-tagging
# ---------------------------------------------------------------------------

class TestGenericsStillWork:
    def test_generic_struct_literal(self):
        src = ("struct Stack<T> { data: T }\n"
               "fn main() { let s = Stack<Int>{ data: 1 }; }")
        assert angle_types(src) == ['LGENERIC', 'RGENERIC',
                                    'LGENERIC', 'RGENERIC']
        parse(src)

    def test_generic_type_annotation(self):
        src = "fn main() { let s: Stack<Int> = make(); }"
        assert angle_types(src) == ['LGENERIC', 'RGENERIC']
        parse(src)

    def test_nested_generic_type_annotation(self):
        src = "fn main() { let m: Map<String, Stack<Int>> = make(); }"
        assert angle_types(src) == ['LGENERIC', 'LGENERIC',
                                    'RGENERIC', 'RGENERIC']
        parse(src)

    def test_generic_call(self):
        src = "fn main() { let x = identity<Int>(5); }"
        assert angle_types(src) == ['LGENERIC', 'RGENERIC']
        call = let_initializer(main_body(parse(src))[0])
        assert not isinstance(call, ast.ComparisonExpression)

    def test_bracket_generic_signature(self):
        # Node[Int]-style bracket generics do not involve angle brackets
        # and must be untouched.
        src = ("fn get<T>(list: @const LinkedList[T], index: Int) "
               "-> Option[@const T] { return none; }")
        toks = lex_types(src)
        assert 'LBRACKET' in toks and 'RBRACKET' in toks
        # The <T> parameter list on the function name is a real generic.
        assert angle_types(src) == ['LGENERIC', 'RGENERIC']
        parse(src)

    def test_implement_header_keeps_generics(self):
        # After 'implement' a '<' is unambiguously generic even when the
        # closing '>' is followed by an identifier.
        src = ("implement<T> VectorOps<T> for vector[T,4] {\n"
               "}\n")
        toks = angle_types(src)
        assert toks[0] == 'LGENERIC' and toks[1] == 'RGENERIC'
