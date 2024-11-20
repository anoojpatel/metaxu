import unittest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from lexer import Lexer
from parser import Parser
from type_checker import TypeChecker
import metaxu_ast as ast

class TestModuleSystem(unittest.TestCase):
    def setUp(self):
        self.lexer = Lexer()
        self.parser = Parser()
        self.type_checker = TypeChecker()

    def test_basic_module(self):
        code = """
        module test {
            "Module for testing basic functionality"
            
            fn add(x: int, y: int) -> int {
                return x + y;
            }
        }
        """
        ast_tree = self.parser.parse(code)
        self.assertIsInstance(ast_tree, ast.Program)
        self.assertEqual(len(ast_tree.statements), 1)
        
        module = ast_tree.statements[0]
        self.assertIsInstance(module, ast.Module)
        self.assertEqual(module.name, "test")
        self.assertEqual(len(module.body.statements), 1)
        self.assertEqual(module.body.docstring, "Module for testing basic functionality")

    def test_module_exports(self):
        code = """
        module math {
            export {
                add,
                subtract as sub,
                multiply
            }
            
            fn add(x: int, y: int) -> int {
                return x + y;
            }
            
            fn subtract(x: int, y: int) -> int {
                return x - y;
            }
            
            fn multiply(x: int, y: int) -> int {
                return x * y;
            }
            
            fn internal_helper() -> int {
                return 42;
            }
        }
        """
        ast_tree = self.parser.parse(code)
        module = ast_tree.statements[0]
        
        self.assertEqual(len(module.body.exports), 3)
        self.assertEqual(module.body.exports[0], ("add", None))
        self.assertEqual(module.body.exports[1], ("subtract", "sub"))
        self.assertEqual(module.body.exports[2], ("multiply", None))

    def test_module_visibility(self):
        code = """
        module data {
            visibility {
                process: private,
                validate: protected,
                transform: public
            }
            
            fn process(data: int) -> int {
                return data * 2;
            }
            
            fn validate(data: int) -> bool {
                return data > 0;
            }
            
            fn transform(data: int) -> int {
                return data + 1;
            }
        }
        """
        ast_tree = self.parser.parse(code)
        module = ast_tree.statements[0]
        
        self.assertEqual(module.body.visibility_rules["process"], "private")
        self.assertEqual(module.body.visibility_rules["validate"], "protected")
        self.assertEqual(module.body.visibility_rules["transform"], "public")

    def test_public_imports(self):
        code = """
        module utils {
            public import std.io;
            public import std.math as math;
            public from std.collections import Vector, HashMap as Map;
            
            fn helper() -> int {
                return io.read_int();
            }
        }
        """
        ast_tree = self.parser.parse(code)
        module = ast_tree.statements[0]
        
        imports = [stmt for stmt in module.body.statements if isinstance(stmt, (ast.Import, ast.FromImport))]
        self.assertEqual(len(imports), 3)
        
        # Check first import
        self.assertIsInstance(imports[0], ast.Import)
        self.assertEqual(imports[0].module_path, ["std", "io"])
        self.assertTrue(imports[0].is_public)
        self.assertIsNone(imports[0].alias)
        
        # Check second import
        self.assertIsInstance(imports[1], ast.Import)
        self.assertEqual(imports[1].module_path, ["std", "math"])
        self.assertTrue(imports[1].is_public)
        self.assertEqual(imports[1].alias, "math")
        
        # Check from import
        self.assertIsInstance(imports[2], ast.FromImport)
        self.assertEqual(imports[2].module_path, ["std", "collections"])
        self.assertTrue(imports[2].is_public)
        self.assertEqual(len(imports[2].names), 2)
        self.assertEqual(imports[2].names[0], ("Vector", None))
        self.assertEqual(imports[2].names[1], ("HashMap", "Map"))

    def test_relative_imports(self):
        code = """
        module utils.strings {
            from ..math import add, multiply;
            from ...core.base import Object;
            from .helpers import format;
        }
        """
        ast_tree = self.parser.parse(code)
        module = ast_tree.statements[0]
        
        imports = [stmt for stmt in module.body.statements if isinstance(stmt, ast.FromImport)]
        self.assertEqual(len(imports), 3)
        
        # Check first relative import
        self.assertEqual(imports[0].relative_level, 2)
        self.assertEqual(imports[0].module_path, ["math"])
        self.assertEqual(imports[0].names, [("add", None), ("multiply", None)])
        
        # Check second relative import
        self.assertEqual(imports[1].relative_level, 3)
        self.assertEqual(imports[1].module_path, ["core", "base"])
        self.assertEqual(imports[1].names, [("Object", None)])
        
        # Check third relative import
        self.assertEqual(imports[2].relative_level, 1)
        self.assertEqual(imports[2].module_path, ["helpers"])
        self.assertEqual(imports[2].names, [("format", None)])

    def test_nested_modules(self):
        code = """
        module outer {
            module inner {
                fn nested_func() -> int {
                    return 42;
                }
            }


            fn outer_func() -> int {
                return inner.nested_func();
            }
        }
        """
        ast_tree = self.parser.parse(code)
        outer_module = ast_tree.statements[0]
        
        self.assertEqual(outer_module.name, "outer")
        inner_module = [stmt for stmt in outer_module.body.statements if isinstance(stmt, ast.Module)][0]
        self.assertEqual(inner_module.name, "inner")
        
        # Check function calls
        outer_func = [stmt for stmt in outer_module.body.statements if isinstance(stmt, ast.FunctionDeclaration)][0]
        return_stmt = outer_func.body[0]
        self.assertIsInstance(return_stmt.expression, ast.QualifiedFunctionCall)
        self.assertEqual(return_stmt.expression.parts, ["inner", "nested_func"])

    def test_module_interface(self):
        code = """
        module math {
            interface Calculator {
                fn add(x: int, y: int) -> int
                fn subtract(x: int, y: int) -> int
            }
            
            impl Calculator {
                fn add(x: int, y: int) -> int {
                    return x + y;
                }
                
                fn subtract(x: int, y: int) -> int {
                    return x - y;
                }
            }
        }
        """
        ast_tree = self.parser.parse(code)
        module = ast_tree.statements[0]
 
        interface = [stmt for stmt in module.body.statements if isinstance(stmt, ast.InterfaceDefinition)][0]
        self.assertEqual(len(interface.methods), 2)
        self.assertEqual(interface.methods[0].name, "add")
        self.assertEqual(interface.methods[1].name, "subtract")

    def test_module_error_cases(self):
        # Test duplicate module names
        code = """
        module test {}
        module test {}
        """
        with self.assertRaises(Exception):
            self.parser.parse(code)
        
        # Test invalid module path
        code = """
        module a.b.c {
            from ....too.many.dots import something;
        }
        """
        with self.assertRaises(Exception):
            self.parser.parse(code)
        
        # Test invalid visibility modifier
        code = """
        module test {
            visibility {
                func: invalid_modifier
            }
        }
        """
        with self.assertRaises(Exception):
            self.parser.parse(code)

    def test_module_function_resolution(self):
        code = """
        module math {
            export {
                Calculator,
                BasicCalc
            }
            
            interface Calculator {
                fn calculate(x: int, y: int) -> int
            }
            
            struct BasicCalc implements Calculator {
                fn calculate(x: int, y: int) -> int {
                    return x + y;
                }
                
                fn private_helper(x: int) -> int {
                    return x * 2;
                }
            }
        }
        
        module main {
            import math;
            
            fn test() -> int {
                let calc = math.BasicCalc();
                return calc.calculate(1, 2);
            }
        }
        """
        ast_tree = self.parser.parse(code)
        
        # Check module exports
        math_module = ast_tree.statements[0]
        self.assertEqual(len(math_module.body.exports), 2)
        
        # Check function resolution
        main_module = ast_tree.statements[1]
        test_func = [stmt for stmt in main_module.body.statements if isinstance(stmt, ast.FunctionDeclaration)][0]
        
        # Verify qualified name resolution
        let_stmt = test_func.body[0]
        self.assertIsInstance(let_stmt.initializer, ast.QualifiedFunctionCall)
        self.assertEqual(let_stmt.initializer.parts, ["math", "BasicCalc"])

    def test_variant_instantiation(self):
        """Test that variant instantiation with :: is parsed correctly while module function calls with . work"""
        source = """
        module test {
            enum Color {
                RGB(r: int, g: int, b: int);
            }

            fn test() {
                let color = Color::RGB(r=255, g=0, b=0);
                return inner.nested_func();
            }
        }
        """
        ast_tree = self.parser.parse(source)
        #breakpoint()
        module = ast_tree.statements[0]
        
        # Check enum definition
        enum_def = [stmt for stmt in module.body.statements if isinstance(stmt, ast.EnumDefinition)][0]
        self.assertEqual(enum_def.name, "Color")
        self.assertEqual(len(enum_def.variants), 1)
        self.assertEqual(enum_def.variants[0].name, "RGB")
        
        # Check function body
        func = [stmt for stmt in module.body.statements if isinstance(stmt, ast.FunctionDeclaration)][0]
        
        # Check variant instantiation
        let_stmt = func.body[0]
        variant_inst = let_stmt.initializer
        self.assertIsInstance(variant_inst, ast.VariantInstance)
        self.assertEqual(variant_inst.enum_name, "Color")
        self.assertEqual(variant_inst.variant_name, "RGB")
        
        # Check module function call
        return_stmt = func.body[1]
        func_call = return_stmt.expression
        self.assertIsInstance(func_call, ast.QualifiedFunctionCall)
        self.assertEqual(func_call.parts, ["inner", "nested_func"])

    def test_closure_capture(self):
        """Test closure capture semantics in module context"""
        code = """
        module counter {
            export {
                make_counter,
                make_accumulator
            }

            # Test basic closure capture
            fn make_counter(start: int) -> fn\() -> int {
                let mut count = start;  # Variable to be captured
                
                fn() -> int {
                    count = count + 1;  # Mutates captured variable
                    count
                }
            }

            # Test multiple captures with different modes
            fn make_accumulator(initial: int) -> fn\(int) -> int {
                let mut sum = initial;     # Mutable capture
                let factor = 2;            # Immutable capture
                
                fn(x: int) -> int {
                    sum = sum + (x * factor);  # Uses both captures
                    sum
                }
            }
        }

        module usage {
            import counter;

            fn test_counters() -> bool {
                let c1 = counter.make_counter(0);
                let c2 = counter.make_counter(10);
                
                assert(c1() == 1);  # First counter: 0 -> 1
                assert(c1() == 2);  # First counter: 1 -> 2
                assert(c2() == 11); # Second counter: 10 -> 11
                
                let acc = counter.make_accumulator(0);
                assert(acc(3) == 6);   # (0 + (3 * 2)) = 6
                assert(acc(4) == 14);  # (6 + (4 * 2)) = 14
                
                true
            }
        }
        """
        ast_tree = self.parser.parse(code)
        self.assertIsInstance(ast_tree, ast.Program)
        
        # Get the counter module
        counter_module = ast_tree.statements[0]
        self.assertEqual(counter_module.name, "counter")
        
        # Check make_counter function
        make_counter = [f for f in counter_module.body.statements 
                       if isinstance(f, ast.FunctionDeclaration) and f.name == "make_counter"][0]
        
        # Verify closure in make_counter
        closure = make_counter.body.statements[-1]
        self.assertIsInstance(closure, ast.LambdaExpression)
        self.assertIn("count", closure.captured_vars)
        self.assertEqual(closure.capture_modes["count"], "borrow_mut")
        
        # Check make_accumulator function
        make_acc = [f for f in counter_module.body.statements 
                   if isinstance(f, ast.FunctionDeclaration) and f.name == "make_accumulator"][0]
        
        # Verify closure in make_accumulator
        closure = make_acc.body.statements[-1]
        self.assertIsInstance(closure, ast.LambdaExpression)
        self.assertIn("sum", closure.captured_vars)
        self.assertIn("factor", closure.captured_vars)
        self.assertEqual(closure.capture_modes["sum"], "borrow_mut")
        self.assertEqual(closure.capture_modes["factor"], "borrow")
        
        # Verify type checking passes
        self.type_checker.check(ast_tree)
        self.assertEqual(len(self.type_checker.errors), 0)

if __name__ == '__main__':
    unittest.main()
