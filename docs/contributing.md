# Contributing to Metaxu

Thank you for your interest in contributing to Metaxu! This document provides guidelines for contributing to the project.

## Development Setup

1. Install prerequisites:
   - Python 3.11 or higher
   - GCC compiler
   - uv package manager

2. Clone and setup:
```bash
git clone https://github.com/yourusername/metaxu.git
cd metaxu
uv sync --all  # Install all dependencies including dev tools
```

## Development Workflow

### Running Tests
```bash
# Run all tests
doit test

# Run specific test suites
doit test_python
doit test_atomic_effects
doit test_effect_runtime
```

### Code Style
- Use type hints for all Python code
- Follow PEP 8 guidelines
- Use beartype for runtime type checking
- Keep functions focused and well-documented

### Making Changes
1. Create a new branch for your changes
2. Write tests for new functionality
3. Ensure all tests pass
4. Submit a pull request

## Project Structure

```
metaxu/
├── src/metaxu/           # Main source code
│   ├── runtimes/        # Runtime implementations
│   │   ├── c/          # C runtime
│   │   │   ├── effects.c    # Effect system implementation
│   │   │   ├── effects.h    # Effect system interface
│   │   │   ├── values.c     # Value handling implementation
│   │   │   └── values.h     # Value handling interface
│   │   └── std/        # Standard library
│   │       ├── prelude.mx   # Core language features
│   │       └── effects/     # Effect implementations
│   │           ├── state.mx    # State effects
│   │           ├── thread.mx   # Threading effects
│   │           ├── domain.mx   # Domain effects
│   │           ├── sync.mx     # Synchronization effects
│   │           ├── advanced.mx # Advanced effects
│   │           └── concurrent.mx # Concurrency effects
│   ├── metaxu_ast.py   # AST definitions
│   ├── type_checker.py # Type checking
│   ├── symbol_table.py # Symbol resolution
│   ├── vm_to_c.py      # VM to C compilation
│   ├── c_linker.py     # C linking utilities
│   ├── unsafe_ast.py   # Unsafe operation AST
│   └── extern_ast.py   # External binding AST
├── tests/              # Test files
│   ├── *.py           # Python test files
│   ├── *.c            # C runtime tests
│   └── *.mx           # Metaxu test files
├── docs/               # Documentation
│   ├── index.md       # Documentation home
│   ├── effects/       # Effect system docs
│   ├── type_system.md # Type system docs
│   └── *.md           # Other documentation
├── vscode-metaxu/     # VSCode extension
│   ├── syntaxes/      # Syntax highlighting
│   └── themes/        # Color themes
├── outputs/           # Build outputs
├── dodo.py           # Build automation
├── pyproject.toml    # Project configuration
└── README.md         # Project overview
```

## Documentation

- Update relevant documentation when making changes
- Add docstrings to new functions and classes
- Update type annotations when modifying interfaces

## Testing

- Write tests for new features
- Update existing tests when changing behavior
- Use pytest for Python tests
- Test both success and failure cases

## Commit Messages

Follow conventional commits format:
- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- test: Test changes
- refactor: Code refactoring
- chore: Maintenance tasks

## Getting Help

- Check existing documentation
- Look through related issues
- Ask questions in discussions
- Join our community chat (soonTM)
