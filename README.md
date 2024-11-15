# Metaxu
A self-hosted Low level functional-style programming language 🌀 
## Structure
A language currently will be compiled (very slowly) in Python into a TapeVM. This lowered Tape IR will  
then emit to a bespoke C program. As the syntax, ergonomics and safety is matured, we will reimplement 
the compiler in pure Metaxu. The goal is to design a language that is fast, fun and safe. The metric  
will be how "fun" will it be to write a compiler for itself.

## Features
The planned language features will include:
- [ ] Unboxed modal references and mutable move semantics
- [ ] Borrow Checking
- [ ] Algebraic Effects
- [ ] Structs and Methods
- [ ] Algebraic Subtyping
- [ ] SIMD and Multithreading
- [ ] Portability
- [ ] Pattern Matching
- [ ] Generics
- [ ] Compilation time execution


### Inspiration
Metaxu is deeply inspired by languages such as Ante, Hylo, Sage, Oxidized Ocaml, Rust and Python.

The term "Metaux" comes from Simone Weil and neoplatonists, describing the links and seperations between the objects  
in the world around us. Classically, the wall between two prisoners who talk via tapping is the separator and  
the very essence of what connects them. Weil's philosophy (theopicy really) is that through these links we  
experience the world and bring ourselves to capital T "Truth." Given this is a compiler that I want to bootstrap,  
I found "metaxu" to be fitting. I hope what what we create can live up to the mantle. 

