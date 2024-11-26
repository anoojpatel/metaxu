from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union, List, Dict
import argparse
import sys
import os

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
            
    def compile_files(self, filepaths: List[Union[str, Path]]) -> Dict[str, Any]:
        """Compile multiple Metaxu source files"""
        results = {}
        
        # First pass - parse all files to collect dependencies
        for filepath in filepaths:
            path = Path(filepath)
            module_name = path.stem
            self.symbol_table.enter_module(module_name, path)
            
            with open(path) as f:
                source = f.read()
                ast_tree = self.parser.parse(source)
                module = ast.Module(
                    name=module_name,
                    statements=ast_tree.statements if hasattr(ast_tree, 'statements') else [ast_tree],
                    path=path
                )
                self.type_checker.collect_imports(module)
            
            self.symbol_table.exit_module()
            
        # Second pass - compile in dependency order
        for module_name in self.symbol_table.get_load_order():
            if module_name in results:
                continue
                
            module_info = self.symbol_table.modules[module_name]
            with open(module_info.path) as f:
                code = self.compile_str(f.read(), str(module_info.path))
                results[module_name] = code
                
        return results

    def run_vm(self, code: Any) -> Any:
        """Run compiled code in the TapeVM"""
        return self.vm.execute(code)

def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Metaxu compiler")
    parser.add_argument('files', nargs='+', help='Source files to compile')
    parser.add_argument('--target', choices=['vm', 'c'], default='vm',
                       help='Compilation target (default: vm)')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('--optimize', '-O', action='store_true',
                       help='Enable optimizations')
    parser.add_argument('--debug', '-g', action='store_true',
                       help='Include debug information')
    parser.add_argument('--dump-ir', action='store_true',
                       help='Dump VM IR')
    parser.add_argument('--dump-ast', action='store_true',
                       help='Dump AST')
    parser.add_argument('--run', action='store_true',
                       help='Run in VM after compilation')
    
    args = parser.parse_args()
    
   
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
        # Compile all input files
        results = compiler.compile_files(args.files)
        
        if not results:
            print("Compilation failed", file=sys.stderr)
            sys.exit(1)
            
        # Handle output based on target
        if args.target == 'vm':
            if args.run:
                # Run the main module
                main_module = Path(args.files[0]).stem
                if main_module in results:
                    compiler.run_vm(results[main_module])
            else:
                # Save VM bytecode
                output = args.output or 'out.mxb'
                with open(output, 'wb') as f:
                    # TODO: Implement bytecode serialization
                    pass
                    
        else:  # target == 'c'
            # Generate C output
            output_dir = args.output or 'build'
            os.makedirs(output_dir, exist_ok=True)
            
            for module_name, code in results.items():
                output_file = Path(output_dir) / f"{module_name}.c"
                with open(output_file, 'w') as f:
                    f.write(code)
                    
            print(f"Generated C code in {output_dir}")
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
