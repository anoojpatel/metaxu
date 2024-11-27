# Metaxu🌀  
A self-hosted Low level functional-style programming language  
[Documentation](docs/index.md)
## Installation

### Prerequisites
- Python 3.11 or higher
- GCC compiler
- uv (Fast Python package installer)

### Setup
1. Install uv:
```bash
# On macOS
brew install uv

# On Linux/WSL
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:
```bash
git clone https://github.com/yourusername/metaxu.git
cd metaxu
```

3. Install dependencies:
```bash
# Install all dependencies including dev tools
uv sync --all

# Or install only runtime dependencies
uv sync
```

4. Use Metaxu Compiler:
```bash
$ metaxu --help

 usage: metaxu [-h] [--target {vm,c}] [--output OUTPUT] [--optimize] [--debug] [--dump-ir] [--dump-ast] [--run] [--link-mode {static,dynamic,library}]
              [--library-name LIBRARY_NAME] [--cuda]
              files [files ...]

Metaxu compiler

positional arguments:
  files                 Source files to compile

options:
  -h, --help            show this help message and exit
  --target {vm,c}       Compilation target (default: vm)
  --output, -o OUTPUT   Output file
  --optimize, -O        Enable optimizations
  --debug, -g           Include debug information
  --dump-ir             Dump VM IR
  --dump-ast            Dump AST
  --run                 Run in VM after compilation
  --link-mode {static,dynamic,library}
                        Linking mode (default: static)
  --library-name LIBRARY_NAME
                        Library name (for library builds)
  --cuda                Enable CUDA support
```

## Development

### Running Tests
The project uses `doit` as its task runner. Here are the common commands:

```bash
# List all available tasks
doit list

# Run all tests (C and Python)
doit test

# Run specific test suites
doit test_python           # Run Python tests only
doit test_atomic_effects   # Run atomic effects test
doit test_effect_runtime   # Run effect runtime test

# Clean build artifacts
doit clean
```

Test binaries are output to the `outputs/` directory.

## Structure
The language currently will be compiled (very slowly) in Python into a TapeVM. This lowered Tape IR will  
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
    - [ ] SimpleSub with Row Polymorphism  
- [ ] SIMD and Multithreading
- [ ] Portability
- [ ] Pattern Matching
- [ ] Generics
- [ ] Compilation time execution

### Inspiration
Metaxu is deeply inspired by languages such as Ante, Hylo, Sage, Oxidized Ocaml, Rust, Zig and Python.

The term "Metaux" comes from Simone Weil and neoplatonists, describing the links and seperations  
between the objects in the world around us. Classically, the wall between two prisoners who talk  
via tapping is the separator and the very essence of what connects them. Weil's philosophy  
(theopicy really) is that, through these links we experience the world and bring ourselves  
to capital T "Truth." Given this is a compiler that I want to bootstrap,  I found "metaxu" to be  
fitting. I hope what we create can live up to the mantle. 
