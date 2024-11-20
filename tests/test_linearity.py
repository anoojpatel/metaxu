import unittest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from lexer import Lexer
from parser import Parser
from type_checker import TypeChecker
import metaxu_ast as ast

class TestLocalityMutabilityLinearity(unittest.TestCase):
    def setUp(self):
        self.lexer = Lexer()
        self.parser = Parser()
        self.type_checker = TypeChecker()

    def test_separate_function(self):
        """Test that functions capturing mutable references are separate"""
        code = """
        module test {
            fn test_separate() -> bool {
                let @mut x = 0;
                
                # This lambda captures a mutable reference, should be separate
                let counter = fn() -> int {
                    x = x + 1;
                    x
                };
                
                # Multiple non-overlapping calls are ok
                let a = counter();  # x = 1
                let b = counter();  # x = 2
                assert(a == 1 && b == 2);
                return true
            }
            
            fn test_separate_error() -> bool {
                let @mut x = 0;
                
                let counter = fn() -> int {
                    x = x + 1;
                    x
                };
                
                # Error: Reentrant call to separate function
                let f = fn() -> int {
                    counter() + counter()  # Error: counter() is active during second call
                };
                
                f();  # Will fail type checking due to reentrant call
                return true
            }
        }
        """
        ast_tree = self.parser.parse(code)
        self.type_checker.check(ast_tree)
        self.assertEqual(len(self.type_checker.errors), 1)  # Should have one error for reentrant call

    def test_once_function(self):
        """Test that functions consuming once values are once"""
        code = """
        module test {
            fn test_once() -> bool {
                let @once x = unique_resource();  # Once binding
                
                # This lambda captures a once value, should be once
                let consumer = fn() {  
                    consume(x)  # Consumes once binding
                };
                
                consumer();  # Ok
                consumer();  # Error: once function already consumed
                
                return true
            }
            
            fn unique_resource() -> Resource {
                Resource::new()
            }
            
            # Return type none is implicit
            fn consume(r: Resource) {
                r.dispose()
            }
        }
        """
        ast_tree = self.parser.parse(code)
        self.type_checker.check(ast_tree)
        self.assertEqual(len(self.type_checker.errors), 1)  # Should have one error for double consumption

    def test_iter_separate(self):
        """Test that List.iter works with separate functions"""
        code = """
        module test {
            fn test_iter() {
                let @mut sum = 0;  # Mutable accumulator
                let list = vector[int,3](1, 2, 3);
                
                # Separate function is ok for iter since calls don't overlap
                iter(list, fn(x @const: int) {  
                    sum = sum + x;
                });
                
                assert(sum == 6);
            }
            
            fn test_iter_error() {
                let @mut x = 0;  # Mutable binding
                let list = vector[int,3](1, 2, 3);
                
                # Error: Trying to use iter with a function that makes reentrant calls
                iter(list, fn(n @const: int) {  
                    iter(vector[int,1](1), fn(_ @const: int) {
                        x = x + 1;  # Error: Reentrant modification of x
                    });
                });
            }
            
            # Parameter mode and function linearity
            fn iter<T>(list: Vector<T>, f: fn\(@const T) -> none @ separate) {  
                match list {
                    [] -> none,
                    [x, ...xs] -> {
                        f(x);
                        iter(xs, f)
                    }
                }
            }
        }
        """
        ast_tree = self.parser.parse(code)
        self.type_checker.check(ast_tree)
        self.assertEqual(len(self.type_checker.errors), 1)  # Should have one error for reentrant modification

    def test_many_function(self):
        """Test that functions with only immutable captures are many"""
        code = """
        module test {
            fn test_many() {
                let x = 42;  # Immutable value
                
                # This lambda only captures immutable references, should be many
                let reader = fn() -> int {
                    x  # Just reads x
                };
                
                # Multiple overlapping calls are ok
                let f = fn() -> int {
                    reader() + reader()  # Ok: reader is many
                };
                
                assert(f() == 84);
            }
        }
        """
        ast_tree = self.parser.parse(code)
        self.type_checker.check(ast_tree)
        self.assertEqual(len(self.type_checker.errors), 0)  # Should have no errors

    def test_modes(self):
        """Test that default global/unique values and mode annotations"""
        code = """
        module test {
            fn test_modes() -> bool {
                # By default values are global and unique
                let x = 42;  # Global unique value
                
                # Local mutable binding
                let @local @mut counter = 0;
                
                # Local const (shared) binding
                let @local @const shared = 10;
                
                # Struct with mode fields
                struct Counter {
                    value: int,           # Global unique by default
                    @mut count: int,      # Global exclusive mutable reference
                    @local name: string,  # Local unique value
                    @const max: int       # Global shared reference
                }
                
                let c = Counter {
                    value: 0,
                    count: 0,
                    name: "test",
                    max: 100
                };
                
                # Function taking references with specific modes
                let increment = fn(c: Counter, amount @const: int) {
                    c.count = c.count + amount;  # Ok to modify through mut field
                };
                
                increment(c, 5);
                assert(c.count == 5);
                return true;
            }
        }
        """
        ast_tree = self.parser.parse(code)
        self.type_checker.check(ast_tree)
        self.assertEqual(len(self.type_checker.errors), 0)  # Should have no errors

    def test_multiple_modes(self):
        """Test that functions with multiple modes work correctly"""
        code = """
        module test {
            fn test_multiple_modes() -> bool {
                # Local mutable binding
                let @local @mut counter = 0;
                
                # Local once binding
                let @local @once resource = unique_resource();
                
                # Struct with mode fields
                struct Counter {
                    @local @mut value: int,  # Local mutable counter
                    @const name: string      # Immutable name
                }
                
                let c = Counter {
                    value: 0,
                    name: "test"
                };
                
                # Function with multiple mode parameters
                let increment = fn(c: Counter, amount @local @const: int) {
                    c.value = c.value + amount;
                };
                
                increment(c, 5);
                assert(c.value == 5);
                return true
            }
            
            fn unique_resource() -> Resource {
                Resource::new()
            }
        }
        """
        ast_tree = self.parser.parse(code)
        self.type_checker.check(ast_tree)
        self.assertEqual(len(self.type_checker.errors), 0)  # Should have no errors

    def test_aliasing_safety(self):
        """Test that aliasing safety with mut and const references"""
        code = """
        module test {
            struct Vector2 {
                @mut x: int,
                @mut y: int
            }

            # Function that takes mutable target and const delta
            fn offset_inout(target @mut: Vector2, delta @const: Vector2) {
                target.x = target.x + delta.x;
                target.y = target.y + delta.y;
            }

            # Double apply the offset
            fn double_offset_inout(target @mut: Vector2, delta @const: Vector2) {
                offset_inout(target, delta);  # First call
                offset_inout(target, delta);  # Second call - delta should be unchanged
            }

            # Test with aliasing references
            let v = Vector2 { x: 3, y: 4 };
            
            # This should be caught at compile time - same value passed as both mut and const
            double_offset_inout(v, v);  # Error: Cannot pass same value as both mut and const
            
            # This is the correct way:
            let source = Vector2 { x: 3, y: 4 };
            let target = Vector2 { x: 3, y: 4 };
            double_offset_inout(target, source);  # Ok: Different values
            
            assert(target.x == 9 && target.y == 12);  # {9, 12} - correct!
            assert(source.x == 3 && source.y == 4);   # Original unchanged
            
            return true;
        }
        """
        ast_tree = self.parser.parse(code)
        self.type_checker.check(ast_tree)
        self.assertEqual(len(self.type_checker.errors), 1)  # Should have one error for aliasing

    def test_aliasing_safety_error(self):
        """Test that aliasing safety with mut and const references"""
        code = """
        module test {
            struct Vector2 {
                @mut x: int,
                @mut y: int
            }

            # This should fail type checking - cannot have mut and const refs to same value
            fn unsafe_alias(v @mut: Vector2) -> bool {
                let same @const = v;  # Error: Cannot alias mut as const
                return true;
            }
            
            let v = Vector2 { x: 1, y: 2 };
            return unsafe_alias(v);
        }
        """
        ast_tree = self.parser.parse(code)
        self.type_checker.check(ast_tree)
        self.assertEqual(len(self.type_checker.errors), 1)  # Should have one error for aliasing
