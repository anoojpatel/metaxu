from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union
import argparse
import sys

from parser import Parser
from type_checker import TypeChecker
from codegen import CodeGenerator
from tape_vm import TapeVM
from symbol_table import SymbolTable
from linker import Linker, DynamicLinker
from c_linker import CLinker, LinkageMode, create_library_config
import metaxu_ast as ast  # Rename to avoid conflict with Python's ast module

@dataclass
class CompileOptions:
    """Compilation options for Metaxu"""
    target: str = "vm"  # "vm" or "c"
    optimize: bool = False
    debug: bool = False
    output: Optional[str] = None
    dump_ir: bool = False
    dump_ast: bool = False
    run_in_vm: bool = False

class MetaxuCompiler:
    """Main compiler interface for Metaxu"""
    
    def __init__(self, options: CompileOptions = None):
        self.options = options or CompileOptions()
        self.parser = Parser()
        self.type_checker = TypeChecker()
        self.code_gen = CodeGenerator()
        self.vm = TapeVM()
        
        # Add standard library path
        self.symbol_table = SymbolTable()
        self.symbol_table.add_module_search_path(Path("examples/std"))
        
        # Initialize linkers
        self.static_linker = Linker(self.symbol_table)
        self.dynamic_linker = DynamicLinker(self.symbol_table)
        
        # Initialize C linker
        self.c_linker = CLinker(Path("build"))
        self._register_standard_c_libraries()
        
    def _register_standard_c_libraries(self):
        """Register standard C libraries"""
        # System libc
        self.c_linker.register_library(create_library_config(
            name="c",
            version="system",
            headers=["/usr/include/stdlib.h", "/usr/include/stdio.h"],
            link_mode=LinkageMode.DYNAMIC
        ))
        
        # Math library
        self.c_linker.register_library(create_library_config(
            name="m",
            version="system",
            headers=["/usr/include/math.h"],
            link_mode=LinkageMode.DYNAMIC
        ))
        
    def compile_str(self, source: str, filename: str = "<string>") -> Any:
        """Compile a string of Metaxu code"""
        try:
            # Parse source into AST
            ast_tree = self.parser.parse(source)
            if self.options.dump_ast:
                print("AST:", ast_tree)
                
            # Create module
            module = ast.Module(
                name=Path(filename).stem,
                statements=ast_tree.statements if hasattr(ast_tree, 'statements') else [ast_tree],
                path=Path(filename)
            )
            
            # Type check
            self.type_checker.check(module)
            if self.type_checker.errors:
                for error in self.type_checker.errors:
                    print(f"Type Error: {error}", file=sys.stderr)
                return None
                
            # Generate code
            code = self.code_gen.generate(module)
            if self.options.dump_ir:
                print("VM IR:", code)
                
            return code
            
        except Exception as e:
            print(f"Compilation Error: {str(e)}", file=sys.stderr)
            return None
            
    def compile_file(self, filepath: Union[str, Path]) -> Any:
        """Compile a Metaxu source file"""
        path = Path(filepath)
        with open(path) as f:
            return self.compile_str(f.read(), str(path))
            
    def run_vm(self, code: Any) -> Any:
        """Run compiled code in the TapeVM"""
        return self.vm.execute(code)

def create_example_files():
    """Create example Metaxu source files"""
    examples = {
        "hello.mx": """
# Basic hello world
print("Hello from Metaxu!")
""",

        "effects.mx": """
# Example of algebraic effects
effect Logger {
    log(message: String) -> Unit
}

effect State<T> {
    get() -> T
    put(value: T) -> Unit
}

# Counter example using State effect
fn counter() -> Int {
    perform State.get() + 1
}

# Main program using effects
fn main() {
    handle State with {
        get() -> resume(0)
        put(x) -> resume(())
    } in {
        # Use Logger effect
        handle Logger with {
            log(msg) -> {
                print(msg)
                resume(())
            }
        } in {
            perform Logger.log("Starting counter...")
            let result = counter()
            perform Logger.log(f"Counter result: {result}")
        }
    }
}
""",

        "ownership.mx": """
# Example of ownership and borrowing
struct Buffer {
    data: unique [Int]
}

fn process(buf: &mut Buffer) {
    # Mutable borrow of buffer
    buf.data[0] = 42
}

fn main() {
    let buf = Buffer { data: [1, 2, 3] }
    process(&mut buf)  # Borrow buffer mutably
    print(buf.data[0]) # Should print 42
}
""",

        "modules/std/io.mx": """
# Standard IO module
effect IO {
    print(msg: String) -> Unit
    read_line() -> String
}

fn println(msg: String) {
    perform IO.print(msg + "\n")
}
""",

        "modules/std/collections.mx": """
# Collections module
struct List<T> {
    data: [T]
    len: Int
}

fn empty<T>() -> List<T> {
    List { data: [], len: 0 }
}

fn push<T>(list: &mut List<T>, item: T) {
    list.data[list.len] = item
    list.len = list.len + 1
}
""",

        "modules/example.mx": """
# Example using modules
import std.io
import std.collections as col
from std.io import println
from ..other.utils import format_string

fn main() {
    # Create a new list
    let mut numbers = col.empty<Int>()
    
    # Add some numbers
    col.push(&mut numbers, 1)
    col.push(&mut numbers, 2)
    
    # Print using imported println
    println("Numbers: " + numbers.len)
    
    # Use relative import
    let msg = format_string("Total: {}", numbers.len)
    io.println(msg)
}
"""
    }
    
    example_dir = Path("examples")
    example_dir.mkdir(exist_ok=True)
    
    # Create std library directory
    std_dir = example_dir / "std"
    std_dir.mkdir(exist_ok=True)
    
    for filename, content in examples.items():
        file_path = example_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(content)
    
    print(f"Created example files in {example_dir}/")

def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Metaxu compiler")
    parser.add_argument("file", nargs="?", help="Source file to compile")
    parser.add_argument("--target", choices=["vm", "c"], default="vm",
                       help="Compilation target")
    parser.add_argument("--optimize", "-O", action="store_true",
                       help="Enable optimizations")
    parser.add_argument("--debug", "-g", action="store_true",
                       help="Include debug information")
    parser.add_argument("--output", "-o", help="Output file")
    parser.add_argument("--dump-ir", action="store_true",
                       help="Dump intermediate representation")
    parser.add_argument("--dump-ast", action="store_true",
                       help="Dump abstract syntax tree")
    parser.add_argument("--create-examples", action="store_true",
                       help="Create example source files")
    parser.add_argument("--run", "-r", action="store_true",
                       help="Run the compiled code in VM")
    
    args = parser.parse_args()
    
    if args.create_examples:
        create_example_files()
        return
        
    if not args.file:
        parser.print_help()
        return
        
    options = CompileOptions(
        target=args.target,
        optimize=args.optimize,
        debug=args.debug,
        output=args.output,
        dump_ir=args.dump_ir,
        dump_ast=args.dump_ast,
        run_in_vm=args.run
    )
    
    compiler = MetaxuCompiler(options)
    
    try:
        code = compiler.compile_file(args.file)
        if code is None:
            sys.exit(1)
            
        if args.run:
            result = compiler.run_vm(code)
            print("Result:", result)
            
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
