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

    def test_thread_effects(self):
        code = '''
        effect Thread {
            spawn(f: () -> a) -> Thread
            join(t: Thread) -> Unit
            current() -> Thread
            yield_() -> Unit
            detach(t: Thread) -> Unit
        }

        effect ThreadPool {
            submit(f: () -> a) -> Future<a>
            await_(f: Future<a>) -> a
            set_pool_size(n: i32) -> Unit
        }

        // Example: Parallel map using thread pool
        fn parallel_map<T, U>(items: Vec<T>, f: (T) -> U) -> Vec<U> {
            // Set up thread pool
            perform ThreadPool.set_pool_size(4);
            
            // Submit tasks for each item
            let futures = items.map(|item| {
                perform ThreadPool.submit(|| f(item))
            });
            
            // Await all results
            futures.map(|future| perform ThreadPool.await_(future))
        }

        // Example: Producer-consumer with threads
        fn producer_consumer(items: Vec<i32>) -> Vec<i32> {
            let queue = new Queue<i32>();
            let results = new Vec<i32>();
            
            // Spawn producer thread
            let producer = perform Thread.spawn(|| {
                for item in items {
                    queue.push(item);
                    perform Thread.yield_();
                }
            });
            
            // Spawn consumer thread
            let consumer = perform Thread.spawn(|| {
                while !queue.is_empty() {
                    let item = queue.pop();
                    results.push(item * 2);
                    perform Thread.yield_();
                }
            });
            
            // Wait for both threads
            perform Thread.join(producer);
            perform Thread.join(consumer);
            
            results
        }
        '''
        result = self.parser.parse(code)
        self.type_checker.check(result)
        self.assertEqual(len(self.type_checker.errors), 0)

    def test_thread_effect_safety(self):
        """Test thread effect safety rules"""
        code = '''
        # Thread with result type
        fn compute_value() -> Int {
            42
        }

        fn thread_test() {
            # Spawn thread
            let t = perform Thread::spawn(compute_value)
            
            # Can't use thread value before join
            let x = t.result  # Should error
            
            # Must join to get result
            let result = perform Thread.join(t)
            assert(result == 42)
        }
        '''
        result = self.parser.parse(code)
        errors = self.type_checker.check(result)
        self.assertTrue(any('Cannot access thread result before join' in str(e) for e in errors))

    def test_domain_effect_safety(self):
        """Test domain effect safety rules"""
        code = '''
        fn domain_test() {
            # Create domain
            let d1 = new_domain(42)
            let d2 = new_domain(0)
            
            # Can't move after borrow
            let x = perform Domain.borrow(d1)
            perform Domain.move(d1, d2)  # Should error
            
            # Can't borrow twice
            let y = perform Domain.borrow(d1)  # Should error
            
            # Can't use after move
            perform Domain.move(d1, d2)
            let z = perform Domain.borrow(d1)  # Should error
        }
        '''
        result = self.parser.parse(code)
        errors = self.type_checker.check(result)
        self.assertTrue(any('Cannot move domain while borrowed' in str(e) for e in errors))
        self.assertTrue(any('Cannot borrow domain twice' in str(e) for e in errors))
        self.assertTrue(any('Cannot use domain after move' in str(e) for e in errors))

    def test_effect_handler_safety(self):
        """Test effect handler safety rules"""
        code = '''
        effect State<T> {
            get() -> T
            put(value: T) -> ()
        }

        fn handler_test() {
            let state = 42
            
            handle State[Int] with {
                get() -> resume(state)
                put(value) -> {
                    state = value  # Should error: can't modify captured variable
                    resume(())
                }
            } in {
                perform State.put(100)
            }
        }
        '''
        result = self.parser.parse(code)
        errors = self.type_checker.check(result)
        self.assertTrue(any('Cannot modify captured variable in handler' in str(e) for e in errors))

if __name__ == '__main__':
    unittest.main()
