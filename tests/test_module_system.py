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
                return std.io.read_int();
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

if __name__ == '__main__':
    unittest.main()
