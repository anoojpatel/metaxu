import unittest
from parser import Parser
from type_checker import TypeChecker
from metaxu_ast import (
    EffectDeclaration, PerformEffect, HandleEffect,
    Resume, FunctionDeclaration, Block
)

class TestEffectfulCode(unittest.TestCase):
    def setUp(self):
        self.parser = Parser()
        self.type_checker = TypeChecker()

    def test_effectful_function_inference(self):
        """Test that functions performing effects are properly marked as effectful"""
        code = '''
        effect Console {
            print(msg: String)
        }

        # Should be inferred as effectful
        fn log_message(msg: String) {
            perform Console.print(msg)
        }

        # Should be inferred as pure
        fn pure_function(x: int) -> int {
            return x + 1
        }

        # Effect type should be inferred from called functions
        fn indirect_effect(msg: String) {
            log_message(msg)  # Makes this function effectful
        }
        '''
        result = self.parser.parse(code)
        self.type_checker.check(result)
        self.assertEqual(len(self.type_checker.errors), 0)

    def test_effect_polymorphism(self):
        """Test polymorphic effect handling with lambda functions and proper borrowing"""
        code = '''
        effect Reader<T> {
            read() -> T
        }

        effect Writer<T> {
            write(value: T) -> Unit
        }

        # Polymorphic over effect E
        fn transform<T, U, E>(f: fn\(&T) -> U performs E) performs Reader<T>, Writer<U>, E {
            let x = perform Reader.read();
            let y = f(&x);  # Borrow x when passing to f
            perform Writer.write(y)
        }

        fn main() {
            handle Reader<int> with {
                read() -> resume(42)
            } in {
                handle Writer<String> with {
                    write(value) -> resume(())
                } in {
                    # Lambda that borrows its input
                    transform(fn(&x: int) -> String { x.to_string() })
                }
            }
        }
        '''
        result = self.parser.parse(code)
        self.type_checker.check(result)
        self.assertEqual(len(self.type_checker.errors), 0)

    def test_effect_composition(self):
        """Test composing multiple effects"""
        code = '''
        effect State<T> {
            get() -> T
            put(value: T) -> Unit
        }

        effect Logger {
            log(msg: String) -> Unit
        }

        # Function using multiple effects
        fn increment_and_log() performs State<int>, Logger {
            let value = perform State.get();
            perform Logger.log("Current value: " + value.to_string());
            perform State.put(value + 1);
            perform Logger.log("Incremented to: " + (value + 1).to_string())
        }

        fn main() {
            handle State<int> with {
                get() -> resume(0)
                put(x) -> resume(())
            } in {
                handle Logger with {
                    log(msg) -> resume(())
                } in {
                    increment_and_log()
                }
            }
        }
        '''
        result = self.parser.parse(code)
        self.type_checker.check(result)
        self.assertEqual(len(self.type_checker.errors), 0)

    def test_effect_error_handling(self):
        """Test error handling with effects"""
        code = '''
        effect Error<E> {
            raise(error: E) -> never
            try<T>(block: fn() -> T) -> Result<T, E>
        }

        fn might_fail() performs Error<String> {
            if some_condition {
                perform Error.raise("Something went wrong")
            }
            42
        }

        fn safe_computation() -> Result<int, String> {
            handle Error<String> with {
                raise(error) -> resume(Err(error))
                try(block) -> {
                    let result = block();
                    resume(Ok(result))
                }
            } in {
                might_fail()
            }
        }
        '''
        result = self.parser.parse(code)
        self.type_checker.check(result)
        self.assertEqual(len(self.type_checker.errors), 0)

    def test_effect_in_async(self):
        """Test effects in async contexts"""
        code = '''
        effect Async<T> {
            await(future: Future<T>) -> T
        }

        effect IO {
            read_file(path: String) -> String
            write_file(path: String, content: String)
        }

        fn async process_file(path: String) performs Async<Unit>, IO {
            let content = perform IO.read_file(path);
            let processed = perform Async.await(process_content(content));
            perform IO.write_file(path + ".processed", processed)
        }

        fn main() {
            handle Async<Unit> with {
                await(future) -> {
                    runtime.spawn(future);
                    resume(runtime.join())
                }
            } in {
                handle IO with {
                    read_file(path) -> resume(fs.read_file(path))
                    write_file(path, content) -> resume(fs.write_file(path, content))
                } in {
                    process_file("data.txt")
                }
            }
        }
        '''
        result = self.parser.parse(code)
        self.type_checker.check(result)
        self.assertEqual(len(self.type_checker.errors), 0)

    def test_effect_inference_errors(self):
        """Test error cases in effect inference"""
        code = '''
        effect Database {
            query(sql: String) -> Vec<Row>
        }

        # Error: Unhandled effect
        fn unhandled_effect() {
            perform Database.query("SELECT * FROM users")
        }

        # Error: Missing effect annotation
        fn missing_annotation() -> Vec<Row> {
            perform Database.query("SELECT * FROM users")
        }

        # Error: Effect annotation doesn't match actual effects
        fn wrong_annotation() performs Logger {
            perform Database.query("SELECT * FROM users")
        }
        '''
        result = self.parser.parse(code)
        with self.assertRaises(TypeError):
            self.type_checker.check(result)

    def test_effect_scoping(self):
        """Test effect handler scoping rules"""
        code = '''
        effect Resource<T> {
            acquire() -> T
            release(resource: T)
        }

        fn nested_resources() {
            handle Resource<File> with {
                acquire() -> resume(File.open("outer.txt"))
                release(f) -> resume(f.close())
            } in {
                let outer = perform Resource.acquire();
                
                handle Resource<File> with {
                    acquire() -> resume(File.open("inner.txt"))
                    release(f) -> resume(f.close())
                } in {
                    let inner = perform Resource.acquire();
                    # Inner handler shadows outer
                    perform Resource.release(inner)
                };
                
                # Back to outer handler
                perform Resource.release(outer)
            }
        }
        '''
        result = self.parser.parse(code)
        self.type_checker.check(result)
        self.assertEqual(len(self.type_checker.errors), 0)

    def test_effect_tail_call(self):
        """Test tail-call optimization with effects"""
        code = '''
        effect Counter {
            increment() -> int
        }

        # Should be tail-call optimized
        fn count_to(n: int) performs Counter {
            let current = perform Counter.increment();
            if current < n {
                count_to(n)  # Tail position
            }
        }

        fn main() {
            handle Counter with {
                increment() -> {
                    count += 1;
                    resume(count)
                }
            } in {
                count_to(1000000)  # Should not overflow stack
            }
        }
        '''
        result = self.parser.parse(code)
        self.type_checker.check(result)
        self.assertEqual(len(self.type_checker.errors), 0)

    def test_tail_call_optimization(self):
        """Test that tail-recursive functions are properly optimized"""
        code = '''
        effect Counter {
            increment() -> int
        }

        # Should be tail-call optimized
        fn count_to(n: int) performs Counter {
            let current = perform Counter.increment();
            if current < n {
                count_to(n)  # Tail position - will be optimized
            } else {
                current  # Return in tail position
            }
        }

        # Non-tail recursive version for comparison
        fn count_to_non_tail(n: int) performs Counter {
            let current = perform Counter.increment();
            if current < n {
                let next = count_to_non_tail(n);  # Not in tail position
                next + 1
            } else {
                current
            }
        }

        fn main() {
            handle Counter with {
                increment() -> {
                    static mut count = 0;
                    count = count + 1;
                    resume(count)
                }
            } in {
                count_to(1000000)  # Should not overflow stack
            }
        }
        '''
        ast = self.parser.parse(code)
        self.assertIsNotNone(ast)
        self.type_checker.check(ast)

    def test_continuation_based_effects(self):
        """Test continuation-based effects with recursion"""
        code = '''
        effect Yield {
            yield(value: int) -> Unit
        }

        fn generate_numbers(n: int) performs Yield {
            if n > 0 {
                perform Yield.yield(n);
                generate_numbers(n - 1)  # Tail call with effect
            }
        }

        fn main() {
            handle Yield with {
                yield(value) -> {
                    # Store value in accumulator
                    let mut values = Vector[int, 100]();
                    let mut index = 0;
                    values[index] = value;
                    index = index + 1;
                    resume()
                }
            } in {
                generate_numbers(10)
            }
        }
        '''
        ast = self.parser.parse(code)
        self.assertIsNotNone(ast)
        self.type_checker.check(ast)

    def test_recursive_types_with_effects(self):
        """Test recursive types combined with effects"""
        code = '''
        effect Traverse {
            visit(value: int) -> Unit
        }

        enum Tree { 
            Leaf(value: int),
            Node(left: Tree, value: int, right: Tree)
        }

        fn traverse(tree: Tree) performs Traverse {
            match tree {
                Leaf(value) -> perform Traverse.visit(value),
                Node(left, value, right) -> {
                    traverse(left);
                    perform Traverse.visit(value);
                    traverse(right)  # Tail call in last position
                }
            }
        }

        fn main() {
            let tree = Node(
                Node(Leaf(1), 2, Leaf(3)),
                4,
                Node(Leaf(5), 6, Leaf(7))
            );

            handle Traverse with {
                visit(value) -> {
                    # Verify traversal order
                    let mut values = Vector[int, 100]();
                    let mut index = 0;
                    values[index] = value;
                    index = index + 1;
                    resume()
                }
            } in {
                traverse(tree)
            }
        }
        '''
        ast = self.parser.parse(code)
        self.assertIsNotNone(ast)
        self.type_checker.check(ast)

if __name__ == '__main__':
    unittest.main()
