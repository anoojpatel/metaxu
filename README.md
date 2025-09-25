<div align="center">
<img src=https://github.com/user-attachments/assets/9a8ec76c-224a-4662-96fb-bdf0b420e01f width="12%" height="12%"></img>


<h1 style="text-align: center;"> Metaxu🌀  </h1>
<p style="text-align: center;">
A self-hosted Low level functional-style programming language <br>    
[<a href="docs/index.md">Documentation</a>]
</p>
</div>

## What is Metaxu?
Metaxu is a modern systems functional programming language that prioritizes speed, safety, and developer joy. 
Currently written in Python, it compiles through several intermediate representations into optimized Cranelift codegen. 
Our ultimate goal? To bootstrap Metaxu in itself, creating a language that's so elegant and intuitive 
that writing its own compiler becomes a delightful challenge.

## Core Features
- 🚀 Zero-cost abstractions with unboxed references and move semantics
- 🛡️ Memory safety through simple modal borrow checking
- ⚡ Powerful algebraic effects system
- 🧬 Advanced type system with algebraic subtyping and row polymorphism
- 🔄 Pattern matching and compile-time metaprogramming
- 🧵 First-class concurrency with SIMD and multithreading
- 📦 Cross-platform portability

## Philosophy
The name "Metaxu" comes from philosopher Simone Weil's concept of divine intermediaries. Just as a wall 
between prison cells becomes both a barrier and a medium for communication through taps, Metaxu serves 
as a bridge between high-level abstractions and low-level performance. It connects human intention to 
machine execution, striving to make systems programming both powerful and accessible.

Deeply inspired by [Ante](https://ante-lang.org), [Hylo](https://hylo-lang.org), [Sage](https://github.com/adam-mcdaniel/sage), [Oxidized OCaml](https://blog.janestreet.com/oxidizing-ocaml-locality/), [Rust](https://rust-lang.org), [Zig](https://ziglang.org), and [Python](https://www.python.org), we're building a 
language that embraces both pragmatism and purity. Our goal is to create a tool that helps programmers 
express their ideas clearly and efficiently, while ensuring their code remains fast and reliable. Furthermore, we want to create a language that's accessible to everyone, regardless of their background or experience level, so that anyone can learn how to build compilers.


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
#### Helping LLMs
We have a `llm-ctx.txt` file that contains some examples of how to use Metaxu with LLMs. It follows [llms.txt](https://llmstxt.org/) standards. Add it via `@docs` for IDEs or into relvant context manager protocols.

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

