import unittest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from metaxu.parser import Parser
import metaxu.metaxu_ast as ast

class TestTypeInference(unittest.TestCase):
    def setUp(self):
        self.parser = Parser()
        # Read the test file content
        self.test_file = Path(__file__).parent / 'test_type_inference.mx'
        with open(self.test_file, 'r') as f:
            self.test_code = f.read()
        print("\nParsing code:")
        print(self.test_code)
        self.ast_tree = self.parser.parse(self.test_code)
        print("\nParsed AST:")
        print(self.ast_tree)

    def test_basic_type_inference(self):
        """Test basic type inference for primitive types"""
        # Find the test_basic_inference function in the AST
        basic_inference_fn = self._find_function('test_basic_inference')
        print("\nFound function:", basic_inference_fn)
        
        # Get variable declarations
        vars = [stmt for stmt in basic_inference_fn.body.statements 
                if isinstance(stmt, ast.VariableDeclaration)]
        print("\nFound variable declarations:", vars)
        
        # Check x: Int
        self.assertEqual(vars[0].name, 'x')
        self.assertIsInstance(vars[0].inferred_type, ast.IntType,
                            f"Expected IntType for x but got {type(vars[0].inferred_type)}")
        
        # Check y: String
        self.assertEqual(vars[1].name, 'y')
        self.assertIsInstance(vars[1].inferred_type, ast.StringType,
                            f"Expected StringType for y but got {type(vars[1].inferred_type)}")
        
        # Check z: Bool
        self.assertEqual(vars[2].name, 'z')
        self.assertIsInstance(vars[2].inferred_type, ast.BoolType,
                            f"Expected BoolType for z but got {type(vars[2].inferred_type)}")

    @unittest.skip("Focusing on basic type inference first")
    def test_function_return_type_inference(self):
        """Test function return type inference"""
        add_fn = self._find_function('add')
        self.assertIsInstance(add_fn.inferred_return_type, ast.IntType)

    @unittest.skip("Focusing on basic type inference first")
    def test_generic_type_inference(self):
        """Test generic type inference with Box type"""
        create_box_fn = self._find_function('create_box')
        
        # Find the return type
        self.assertIsInstance(create_box_fn.inferred_return_type, ast.GenericType)
        self.assertEqual(create_box_fn.inferred_return_type.name, 'Box')

    @unittest.skip("Focusing on basic type inference first")
    def test_complex_type_inference(self):
        """Test complex nested type inference"""
        complex_fn = self._find_function('test_complex_inference')
        
        # Get variable declarations
        vars = [stmt for stmt in complex_fn.body.statements 
                if isinstance(stmt, ast.VariableDeclaration)]
        
        # Check nested Pair[Box[Int], Box[String]]
        nested_var = vars[0]
        self.assertEqual(nested_var.name, 'nested')
        self.assertIsInstance(nested_var.inferred_type, ast.GenericType)
        self.assertEqual(nested_var.inferred_type.name, 'Pair')
        
        # Check Option[Box[Int]]
        maybe_box = vars[1]
        self.assertEqual(maybe_box.name, 'maybe_box')
        self.assertIsInstance(maybe_box.inferred_type, ast.GenericType)
        self.assertEqual(maybe_box.inferred_type.name, 'Option')
        
        box_type = maybe_box.inferred_type.type_args[0]
        self.assertIsInstance(box_type, ast.GenericType)
        self.assertEqual(box_type.name, 'Box')
        self.assertIsInstance(box_type.type_args[0], ast.IntType)

    def _find_function(self, name):
        """Helper to find a function in the AST by name"""
        for stmt in self.ast_tree.statements:
            if isinstance(stmt, ast.FunctionDeclaration) and stmt.name == name:
                return stmt
        raise ValueError(f'Function {name} not found in AST')

if __name__ == '__main__':
    unittest.main()
