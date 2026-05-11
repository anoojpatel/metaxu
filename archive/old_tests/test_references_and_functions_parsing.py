import unittest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from parser import Parser
import metaxu_ast as ast

class TestReferencesAndFunctions(unittest.TestCase):
    def setUp(self):
        self.parser = Parser()

    def test_unique_references(self):
        """Test unique references - only one reference can exist at a time"""
        code = """
        module test {
            fn transfer_ownership(unique x: int) -> unique int {
                return x;  // Ownership is transferred
            }

            fn test() {
                let unique x = 42;
                let y = transfer_ownership(x);
                // x is no longer valid here
            }
        }
        """
        ast_tree = self.parser.parse(code)
        module = ast_tree.statements[0]
        
        # Check transfer_ownership function
        transfer_fn = [stmt for stmt in module.body.statements 
                      if isinstance(stmt, ast.FunctionDeclaration) 
                      and stmt.name == "transfer_ownership"][0]
        self.assertEqual(transfer_fn.params[0].mode.mode, "unique")
        self.assertEqual(transfer_fn.return_type.mode, "unique")

    def test_shared_references(self):
        """Test shared references - multiple immutable references can exist"""
        code = """
        module test {
            fn read_shared(shared x: int) -> int {
                return x;  // Only reading, no modification
            }

            fn test() {
                let shared x = 42;
                let a = read_shared(x);
                let b = read_shared(x);  // Multiple shared references are OK
            }
        }
        """
        ast_tree = self.parser.parse(code)
        module = ast_tree.statements[0]
        
        # Check read_shared function
        read_fn = [stmt for stmt in module.body.statements 
                   if isinstance(stmt, ast.FunctionDeclaration) 
                   and stmt.name == "read_shared"][0]
        self.assertEqual(read_fn.params[0].mode.mode, "shared")

    def test_exclusive_references(self):
        """Test exclusive references - single mutable reference"""
        code = """
        module test {
            fn modify_exclusive(exclusive x: int) -> int {
                x = x + 1;  // Can modify exclusive reference
                return x;
            }

            fn test() {
                let exclusive x = 42;
                let y = modify_exclusive(x);  // x is temporarily borrowed
                let z = modify_exclusive(x);  // Can use x again after borrow ends
            }
        }
        """
        ast_tree = self.parser.parse(code)
        module = ast_tree.statements[0]
        
        # Check modify_exclusive function
        modify_fn = [stmt for stmt in module.body.statements 
                    if isinstance(stmt, ast.FunctionDeclaration) 
                    and stmt.name == "modify_exclusive"][0]
        self.assertEqual(modify_fn.params[0].mode.mode, "exclusive")

    def test_borrow_references(self):
        """Test borrowing references temporarily"""
        code = """
        module test {
            fn borrow_shared(borrow shared x: int) -> int {
                return x;  // Borrowing shared reference
            }

            fn borrow_exclusive(borrow exclusive x: int) -> int {
                x = x + 1;  // Borrowing exclusive reference
                return x;
            }

            fn test() {
                let exclusive x = 42;
                let a = borrow_shared(x);     // Can borrow shared
                let b = borrow_shared(x);     // Multiple shared borrows OK
                let c = borrow_exclusive(x);  // Exclusive borrow
            }
        }
        """
        ast_tree = self.parser.parse(code)
        module = ast_tree.statements[0]
        
        # Check borrow functions
        shared_fn = [stmt for stmt in module.body.statements 
                    if isinstance(stmt, ast.FunctionDeclaration) 
                    and stmt.name == "borrow_shared"][0]
        exclusive_fn = [stmt for stmt in module.body.statements 
                       if isinstance(stmt, ast.FunctionDeclaration) 
                       and stmt.name == "borrow_exclusive"][0]
        
        self.assertEqual(shared_fn.params[0].mode.mode, "shared")
        self.assertEqual(exclusive_fn.params[0].mode.mode, "exclusive")

    def test_closures(self):
        """Test closures capturing variables from outer scope"""
        code = """
        module test {
            fn make_counter(start: int) -> fn\() -> int {
                let mut count = start;
                return fn() -> int {
                    count = count + 1;
                    return count;
                }
            }

            fn test() {
                let counter = make_counter(0);
                let a = counter();  // Returns 1
                let b = counter();  // Returns 2
            }
        }
        """
        ast_tree = self.parser.parse(code)
        module = ast_tree.statements[0]
        
        # Check make_counter function
        counter_fn = [stmt for stmt in module.body.statements 
                     if isinstance(stmt, ast.FunctionDeclaration) 
                     and stmt.name == "make_counter"][0]
        
        # Verify return type is a function type
        self.assertIsInstance(counter_fn.return_type, ast.FunctionType)
        
        # Verify closure captures count variable
        closure = counter_fn.body[-1]
        self.assertIsInstance(closure, ast.Block)

        self.assertIn("count", closure.statements[0].name)

    def test_higher_order_functions(self):
        """Test higher-order functions that take or return functions"""
        code = """
        module test {
            fn map(arr: []int, f: fn\(int) -> int) -> []int {
                let mut result = vector [int]
                
                for x in arr {
                    result.push(f(x))
                }
                
                return result
            }

            fn compose(f: fn(int) -> int, g: fn\(int) -> int) -> fn\(int) -> int {
                return fn(x: int) -> int {
                    return f(g(x))
                }
            }

            fn test() {
                let double = fn(x: int) -> int { return x * 2 }
                let add_one = fn(x: int) -> int { return x + 1 }
                let double_then_add = compose(add_one, double)
                
                let numbers = vector[int,3](1, 2, 3)
                let doubled = map(numbers, double)  // [2, 4, 6]
                let result = map(numbers, double_then_add)  // [3, 5, 7]
            }
        }
        """
        ast_tree = self.parser.parse(code)
        module = ast_tree.statements[0]
        
        # Check map function
        map_fn = [stmt for stmt in module.body.statements 
                 if isinstance(stmt, ast.FunctionDeclaration) 
                 and stmt.name == "map"][0]
        
        # Verify map takes a function parameter
        self.assertIsInstance(map_fn.params[1].type, ast.FunctionType)
        
        # Check compose function
        compose_fn = [stmt for stmt in module.body.statements 
                     if isinstance(stmt, ast.FunctionDeclaration) 
                     and stmt.name == "compose"][0]
        
        # Verify compose returns a function
        self.assertIsInstance(compose_fn.return_type, ast.FunctionType)

if __name__ == '__main__':
    unittest.main()
