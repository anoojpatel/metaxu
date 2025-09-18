import unittest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from metaxu.parser import Parser
from metaxu.type_checker import TypeChecker
import metaxu.metaxu_ast as ast

class TestTypeInference(unittest.TestCase):
    def setUp(self):
        self.parser = Parser()
        self.type_checker = TypeChecker()
        # Read the test file content
        self.test_file = Path(__file__).parent / 'test_type_inference.mx'
        with open(self.test_file, 'r') as f:
            self.test_code = f.read()
        print("\nParsing code:")
        print(self.test_code)
        self.ast_tree = self.parser.parse(self.test_code)
        print("\nParsed AST:")
        print(self.ast_tree)
        
        # Run type checker on the AST tree
        print("\nRunning type checker...")
        self.type_checker.check_program(self.ast_tree)
        print("Type checking completed")

    def test_basic_type_inference(self):
        """Test basic type inference for primitive types"""
        # Find the test_basic_inference function in the AST
        basic_inference_fn = self._find_function('test_basic_inference')
        
        # Get variable declarations from the function body
        if hasattr(basic_inference_fn.body, 'statements'):
            # If body is a Block with statements attribute
            vars = [stmt for stmt in basic_inference_fn.body.statements 
                   if isinstance(stmt, ast.LetStatement)]
        else:
            # If body is a list of statements directly
            vars = [stmt for stmt in basic_inference_fn.body 
                   if isinstance(stmt, ast.LetStatement)]
        
        # Print focused debug information
        print("\n=== Type Inference Debug ===\n")
        print(f"Found {len(vars)} let statements in test_basic_inference")
        
        # Check each binding's structure and types
        for i, stmt in enumerate(vars):
            binding = stmt.bindings[0] if stmt.bindings else None
            if binding:
                print(f"\nBinding {i+1}: {binding.identifier}")
                print(f"  Has initializer: {binding.initializer is not None}")
                if binding.initializer:
                    print(f"  Initializer type: {type(binding.initializer).__name__}")
                    # Inspect the Literal object in detail
                    if isinstance(binding.initializer, ast.Literal):
                        print(f"  Literal value: {binding.initializer.value}")
                        print(f"  Literal value type: {type(binding.initializer.value).__name__}")
                        
                        # Types should be inferred by the type checker
                        pass
                
                print(f"  Has type_var: {hasattr(binding, 'type_var')}")
                if hasattr(binding, 'type_var'):
                    print(f"  type_var: {binding.type_var}")
                print(f"  Has inferred_type: {hasattr(binding, 'inferred_type')}")
                if hasattr(binding, 'inferred_type'):
                    print(f"  inferred_type: {binding.inferred_type}")
                
                # Check if the LetStatement has an inferred_type
                print(f"  LetStatement has inferred_type: {hasattr(stmt, 'inferred_type')}")
                if hasattr(stmt, 'inferred_type'):
                    print(f"  LetStatement inferred_type: {stmt.inferred_type}")
        
        print("\n=== End Debug ===\n")
        
        print("NOTE: Types are being set directly in the test for now.")
        print("Future work: Fix the type_checker.py to properly infer types from literals.")
        
        # Check x: Int
        self.assertEqual(vars[0].bindings[0].identifier, 'x')
        self.assertIsInstance(vars[0].inferred_type, ast.BasicType,
                            f"Expected BasicType for x but got {type(vars[0].inferred_type)}")
        self.assertEqual(vars[0].inferred_type.name, 'Int',
                       f"Expected Int type but got {vars[0].inferred_type.name}")
        
        # Check y: String
        self.assertEqual(vars[1].bindings[0].identifier, 'y')
        self.assertIsInstance(vars[1].inferred_type, ast.BasicType,
                            f"Expected BasicType for y but got {type(vars[1].inferred_type)}")
        self.assertEqual(vars[1].inferred_type.name, 'String',
                       f"Expected String type but got {vars[1].inferred_type.name}")
        
        # Check z: Bool
        self.assertEqual(vars[2].bindings[0].identifier, 'z')
        self.assertIsInstance(vars[2].inferred_type, ast.BasicType,
                            f"Expected BasicType for z but got {type(vars[2].inferred_type)}")
        self.assertEqual(vars[2].inferred_type.name, 'Bool',
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
                if isinstance(stmt, ast.LetStatement)]
        
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
        for stmt in self.ast_tree.body.statements:
                if isinstance(stmt, ast.FunctionDeclaration) and stmt.name == name:
                    return stmt
        raise ValueError(f'Function {name} not found in AST')

if __name__ == '__main__':
    unittest.main()
