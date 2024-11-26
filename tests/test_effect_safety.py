import unittest
from parser import Parser
from type_checker import TypeChecker
from metaxu_ast import (
    EffectDeclaration, PerformEffect, HandleEffect,
    Resume, BorrowShared, BorrowUnique, Move
)

class TestEffectSafety(unittest.TestCase):
    def setUp(self):
        self.parser = Parser()
        self.type_checker = TypeChecker()

    def test_effect_with_references(self):
        code = '''
        effect State<T> {
            get() -> &T
            put(value: &mut T) -> Unit
        }

        fn use_state() {
            handle State[i32] with {
                get() -> {
                    let value = &state;
                    resume(value)
                }
                put(new_value) -> {
                    state = *new_value;
                    resume(())
                }
            } in {
                let x = perform State.get();
                let y = &mut x;
                perform State.put(y);
            }
        }
        '''
        result = self.parser.parse(code)
        self.type_checker.check(result)
        self.assertEqual(len(self.type_checker.errors), 0)

    def test_linear_effect_handler(self):
        code = '''
        effect Channel<T> {
            send(msg: T) -> Unit
            recv() -> T
        }

        fn channel_usage() {
            handle Channel[String] with {
                send(msg) -> {
                    buffer = move msg;  # Linear use
                    resume(())
                }
                recv() -> {
                    let value = move buffer;  # Linear use
                    resume(value)
                }
            } in {
                let msg = "hello";
                perform Channel.send(move msg);  # Move into effect
                let received = perform Channel.recv();
            }
        }
        '''
        result = self.parser.parse(code)
        self.type_checker.check(result)
        self.assertEqual(len(self.type_checker.errors), 0)

    def test_nested_effect_borrows(self):
        code = '''
        effect Reader<T> {
            read() -> &T
        }

        effect Writer<T> {
            write(value: &T) -> Unit
        }

        fn nested_effects() {
            handle Reader[String] with {
                read() -> {
                    let value = &data;
                    resume(value)
                }
            } in {
                handle Writer[String] with {
                    write(value) -> {
                        let borrowed = &*value;  # Nested borrow
                        resume(())
                    }
                } in {
                    let x = perform Reader.read();
                    perform Writer.write(&x);
                }
            }
        }
        '''
        result = self.parser.parse(code)
        self.type_checker.check(result)
        self.assertEqual(len(self.type_checker.errors), 0)

    def test_effect_closure_capture(self):
        code = '''
        effect Closure<T> {
            capture(value: &T) -> Unit
            use() -> &T
        }

        fn closure_effect() {
            let data = "captured";
            handle Closure<String> with {
                capture(value) -> {
                    closed_value = &*value;  # Borrow in closure
                    resume(())
                }
                use() -> {
                    resume(&closed_value)
                }
            } in {
                perform Closure.capture(&data);
                let borrowed = perform Closure.use();
            }
        }
        '''
        result = self.parser.parse(code)
        self.type_checker.check(result)
        self.assertEqual(len(self.type_checker.errors), 0)

    def test_effect_borrow_errors(self):
        code = '''
        effect InvalidBorrow<T> {
            escape(value: &T) -> &T  # Should fail - escaping reference
        }

        fn invalid_borrow() {
            handle InvalidBorrow[i32] with {
                escape(value) -> {
                    resume(value)  # Error: returning borrowed value
                }
            } in {
                let x = 42;
                let escaped = perform InvalidBorrow.escape(&x);
            }
        }
        '''
        result = self.parser.parse(code)
        with self.assertRaises(TypeError):
            self.type_checker.check(result)

    def test_effect_move_errors(self):
        code = '''
        effect InvalidMove<T> {
            consume(value: T) -> T
        }

        fn invalid_move() {
            let x = "moved";
            handle InvalidMove[String] with {
                consume(value) -> {
                    resume(value)  # Error: value used after move
                    print(value);  # Error: use after move
                }
            } in {
                perform InvalidMove.consume(move x);
                print(x);  # Error: use after move
            }
        }
        '''
        result = self.parser.parse(code)
        with self.assertRaises(TypeError):
            self.type_checker.check(result)

    def test_nested_effect_handlers(self):
        code = '''
        effect State<T> {
            get() -> T
            put(value: T) -> Unit
        }

        effect Logger {
            log(message: String) -> Unit
        }

        fn increment_and_log() {
            handle State[int] with {
                get() -> resume(0)
                put(value) -> resume(())
            } in 
            handle Logger with {
                log(message) -> resume(())
            } in {
                let x = perform State.get();
                perform Logger.log("Got value: " + x);
                perform State.put(x + 1);
                perform Logger.log("Incremented value");
            }
        }
        '''
        result = self.parser.parse(code)
        self.type_checker.check(result)
        self.assertEqual(len(self.type_checker.errors), 0)

    def test_effect_type_application(self):
        code = '''
        effect State<T> {
            get() -> T
            put(value: T) -> Unit
        }

        fn increment_counter() {
            handle State[int] with {
                get() -> resume(0)
                put(value) -> resume(())
            } in {
                let x = perform State.get();
                perform State.put(x + 1);
                x + 1
            }
        }
        '''
        result = self.parser.parse(code)
        self.type_checker.check(result)
        self.assertEqual(len(self.type_checker.errors), 0)

    def test_domain_effect(self):
        code = '''
        effect Domain<T> {
            create(value: T) -> Domain<T>
            acquire(domain: Domain<T>) -> T
            release(domain: Domain<T>) -> Unit
            transfer(domain: Domain<T>, thread: Thread) -> Unit
        }

        fn parallel_increment(x: i32) -> i32 {
            // Create domain for shared value
            let domain = perform Domain.create(x);
            
            // Spawn worker thread
            let worker = spawn {
                // Acquire domain in worker
                let value = perform Domain.acquire(domain);
                let new_value = value + 1;
                perform Domain.release(domain);
                
                // Transfer domain back to main thread
                perform Domain.transfer(domain, current_thread());
            };
            
            // Wait for worker
            join(worker);
            
            // Get final value
            let result = perform Domain.acquire(domain);
            perform Domain.release(domain);
            result
        }
        '''
        result = self.parser.parse(code)
        self.type_checker.check(result)
        self.assertEqual(len(self.type_checker.errors), 0)

if __name__ == '__main__':
    unittest.main()
