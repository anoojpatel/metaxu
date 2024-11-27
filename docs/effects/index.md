# Algebraic Effects in Metaxu

Metaxu implements a powerful algebraic effects system that provides fine-grained control over side effects while maintaining type safety and performance.

## Overview

Algebraic effects in Metaxu allow you to:
- Express and handle side effects in a type-safe manner
- Compose effects cleanly using handlers
- Optimize effect handling at compile time
- Map effects directly to efficient runtime implementations

## Builtin Effect Categories

These are the built-in effect categories in Metaxu that provide multithreading, safe mutable memory management, and concurrency primitives:

### Thread Effects
Thread effects provide safe concurrency primitives:
```metaxu
effect Thread {
  spawn<T>(fn: () -> T) -> ThreadHandle<T>
  join<T>(handle: ThreadHandle<T>) -> T
  yield() -> ()
}
```

### Domain Effects
Domain effects manage memory ownership and regions:
```metaxu
effect Domain {
  alloc<T>(size: Size) -> Domain<T>
  move<T>(from: Domain<T>, to: Domain<T>) -> ()
  borrow<T>(domain: Domain<T>) -> &T
  free<T>(domain: Domain<T>) -> ()
}
```

### Atomic Effects
Atomic effects provide low-level atomic operations:
```metaxu
effect Atomic {
  alloc<T>() -> Atomic<T>
  load<T>(atomic: &Atomic<T>) -> T
  store<T>(atomic: &Atomic<T>, value: T) -> ()
  cas<T>(atomic: &Atomic<T>, expected: T, new: T) -> bool
  add<T: Numeric>(atomic: &Atomic<T>, value: T) -> T
  sub<T: Numeric>(atomic: &Atomic<T>, value: T) -> T
}
```

### RwLock Effects
RwLock effects implement reader-writer locks:
```metaxu
effect RwLock {
  create<T>() -> RwLock<T>
  read_lock<T>(lock: &RwLock<T>) -> ReadGuard<T>
  read_unlock<T>(guard: ReadGuard<T>) -> ()
  write_lock<T>(lock: &RwLock<T>) -> WriteGuard<T>
  write_unlock<T>(guard: WriteGuard<T>) -> ()
}
```

## Implementation Details

The effect system is implemented using:
- Fixed-size hash tables for O(1) effect handler lookups
- FNV-1a hashing for efficient handler registration
- Linear probing for collision resolution
- Compile-time effect handler optimization
- Type-safe value abstraction layer

For more details on the implementation, see the [runtime code](../../src/metaxu/runtimes/c/effects.h).

## Usage Examples

### Basic Effect Usage
```metaxu
// Create and use an atomic counter
let counter = perform Atomic::alloc<Int>();
perform Atomic::store(counter, 0);
perform Atomic::add(counter, 1);
let value = perform Atomic::load(counter);
```

### Effect Composition
```metaxu
// Combine thread and atomic effects
let counter = perform Atomic::alloc<Int>();
let handle = perform Thread::spawn(() => {
  perform Atomic::add(counter, 1);
});
perform Thread::join(handle);
```

## Performance Considerations

The effect system is designed for minimal runtime overhead:
- Effect handlers are registered at compile time
- Handler lookup is O(1)
- Value handling is zero-cost when possible
- Memory management is explicit and efficient

## Future Enhancements

Planned improvements to the effect system:
- Dynamic hash table resizing
- More advanced collision resolution strategies
- Additional effect categories
- Cross-platform effect implementations
