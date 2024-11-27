# Metaxu Documentation

Welcome to the Metaxu documentation! Metaxu is a self-hosted, low-level functional programming language designed for performance, safety, and expressiveness.

## Getting Started
- [Installation and Usage](../README.md#installation)
- [Command Line Interface](../README.md#use-metaxu-compiler)

## Language Features

### Type System and Safety
- [Type System](type_system.md)
- [Ownership and Borrowing](ownership_and_borrowing.md)

### Effects and Runtime
- [Builtin Algebraic Effects](effects/index.md)
  - Thread Effects
  - Domain Effects
  - Atomic Effects
  - RwLock Effects

### Future Development
- [Future Considerations](future_considerations.md)
  - SIMD and Multithreading
  - Algebraic Subtyping
  - Pattern Matching
  - Generics
  - Compilation Time Execution

## Development
- [Building and Testing](../README.md#development)
- [Contributing Guidelines](contributing.md)

## Design Philosophy
Metaxu is deeply inspired by languages such as Ante, Hylo, Sage, Oxidized OCaml, Rust, Zig, and Python. The language aims to combine:

- Strong type safety through static analysis
- Efficient memory management with ownership and borrowing
- Powerful effect system for controlled side effects
- Zero-cost abstractions
- Clear and expressive syntax

The name "Metaxu" comes from Simone Weil's philosophy, representing the interconnections between objects in our world. This reflects our goal of creating a language that elegantly bridges the gap between high-level abstractions and low-level performance.
