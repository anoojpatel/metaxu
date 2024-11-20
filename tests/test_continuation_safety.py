import unittest
from lexer import Lexer
from parser import Parser
from type_checker import TypeChecker
from continuation_safety import ContinuationSafetyChecker
import metaxu_ast as ast

class TestContinuationSafety(unittest.TestCase):
    def setUp(self):
        self.lexer = Lexer()
        self.parser = Parser()
        self.type_checker = TypeChecker()
        self.safety_checker = ContinuationSafetyChecker()
        
    def test_local_value_capture(self):
        """Test that capturing local values in continuations is rejected"""
        code = """
        module test {
            effect Read<T> {
                fn read() -> T
            }
            
            fn unsafe_local_capture() {
                let @local x = 42;
                
                try {
                    perform Read::read[int]() with |k| {
                        k(x)  # Should fail - capturing local value
                    }
                }
            }
        }
        """
        ast_tree = self.parser.parse(code)
        errors = self.safety_checker.check_perform_expression(
            ast_tree.declarations[0].body.statements[0].expression
        )
        self.assertTrue(any("local variable" in err for err in errors))
        
    def test_mutable_reference_capture(self):
        """Test that capturing mutable references in continuations is rejected"""
        code = """
        module test {
            effect State<T> {
                fn get() -> T
            }
            
            fn unsafe_mut_capture() {
                let @mut x = 42;
                
                try {
                    perform State::get[int]() with |k| {
                        k(x)  # Should fail - capturing mutable reference
                    }
                }
            }
        }
        """
        ast_tree = self.parser.parse(code)
        errors = self.safety_checker.check_perform_expression(
            ast_tree.declarations[0].body.statements[0].expression
        )
        self.assertTrue(any("mutable reference" in err for err in errors))
        
    def test_safe_global_capture(self):
        """Test that capturing global values in continuations is allowed"""
        code = """
        module test {
            effect Read<T> {
                fn read() -> T
            }
            
            fn safe_global_capture() {
                let x = 42;  # Global by default
                
                try {
                    perform Read::read[int]() with |k| {
                        k(x)  # Should succeed - global value
                    }
                }
            }
        }
        """
        ast_tree = self.parser.parse(code)
        errors = self.safety_checker.check_perform_expression(
            ast_tree.declarations[0].body.statements[0].expression
        )
        self.assertEqual(len(errors), 0)
        
    def test_moved_unique_capture(self):
        """Test that moving unique values into continuations is allowed"""
        code = """
        module test {
            effect Resource<T> {
                fn take() -> T
            }
            
            fn safe_move_capture() {
                let resource = create_resource();
                
                try {
                    perform Resource::take[Resource]() with |k| {
                        k(resource)  # Should succeed - unique value is moved
                    }
                }
            }
        }
        """
        ast_tree = self.parser.parse(code)
        # Mark resource as moved
        self.safety_checker.mark_moved("resource")
        errors = self.safety_checker.check_perform_expression(
            ast_tree.declarations[0].body.statements[0].expression
        )
        self.assertEqual(len(errors), 0)
        
    def test_const_reference_capture(self):
        """Test that capturing const references in continuations is allowed"""
        code = """
        module test {
            effect Read<T> {
                fn read() -> T
            }
            
            fn safe_const_capture() {
                let @const x = 42;
                
                try {
                    perform Read::read[int]() with |k| {
                        k(x)  # Should succeed - const reference
                    }
                }
            }
        }
        """
        ast_tree = self.parser.parse(code)
        errors = self.safety_checker.check_perform_expression(
            ast_tree.declarations[0].body.statements[0].expression
        )
        self.assertEqual(len(errors), 0)
