from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union, List, Dict
import argparse
import sys
import os
from metaxu.errors import CompileError, SourceLocation
from metaxu.parser import Parser
from metaxu.type_checker import TypeChecker
from metaxu.codegen import CodeGenerator
from metaxu.tape_vm import TapeVM
from metaxu.symbol_table import SymbolTable
from metaxu.linker import Linker, DynamicLinker
from metaxu.c_linker import CLinker, LinkageMode, create_library_config
import metaxu.metaxu_ast as ast  # Rename to avoid conflict with Python's ast module
from metaxu.vm_to_c import VMToCCompiler
import traceback

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
    link_mode: str = "static"  # "static", "dynamic", or "library"
    library_name: Optional[str] = None  # For library builds
    cuda_enabled: bool = False  # Enable CUDA support

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
        
        # Initialize C linker with absolute build path
        build_dir = Path("build").resolve()
        self.c_linker = CLinker(build_dir)
        self._register_standard_c_libraries()
        
    def _register_standard_c_libraries(self):
        """Register standard C libraries"""
        # System libc
        self.c_linker.add_library(create_library_config(
            name="c",
            version="system",
            headers=["/usr/include/stdlib.h", "/usr/include/stdio.h"],
            link_mode=LinkageMode.DYNAMIC
        ))
        
        # Math library
        self.c_linker.add_library(create_library_config(
            name="m",
            version="system",
            headers=["/usr/include/math.h"],
            link_mode=LinkageMode.DYNAMIC
        ))
        
    def compile_str(self, source: str, source_path: str = "<string>") -> Any:
        """Compile a string of Metaxu code"""
        try:
            parser = Parser()
            ast_tree = parser.parse(source, file_path=source_path)
            if self.options.dump_ast:
                print("AST:", ast_tree)
                
            if ast_tree is None:
                raise CompileError(
                    message="Failed to parse source code",
                    location=SourceLocation(source_path, 1, 1)
                )
            
            # Create module body
            module_body = ast.ModuleBody(
                statements=ast_tree.statements if hasattr(ast_tree, 'statements') else [ast_tree],
                docstring=None,  # TODO: Extract docstring from source
                exports=[]  # TODO: Handle exports
            )
            
            # Create module
            module = ast.Module(
                name=Path(source_path).stem,
                body=module_body
            )
            
            # Initialize module in symbol table
            self.symbol_table.enter_module(module.name, Path(source_path))
            
            # Type check
            self.type_checker.check(module)
            if self.type_checker.errors:
                for error in self.type_checker.errors:
                    print(f"Type Error: {error}", file=sys.stderr)
                self.symbol_table.exit_module()
                return None
                
            # Generate code
            code = self.code_gen.generate(module)
            if self.options.dump_ir:
                print("VM IR:", code)
                
            # Exit module scope
            self.symbol_table.exit_module()
                
            return code
            
        except CompileError as e:
            print("Compilation Error:")
            print(str(e))
            sys.exit(1)
        except Exception as e:
            # Unexpected error - convert to CompileError with full traceback
            error = CompileError.from_exception(
                e,
                location=SourceLocation(source_path, 1, 1)
            )
            print("Internal Compiler Error:")
            print(str(error))
            sys.exit(1)

    def compile_file(self, filepath: Union[str, Path]) -> Any:
        """Compile a Metaxu source file"""
        try:
            path = Path(filepath)
            with open(path) as f:
                source = f.read()
                
            return self.compile_str(source, str(path))
            
        except CompileError as e:
            print("Compilation Error:")
            print(str(e))
            sys.exit(1)
        except Exception as e:
            # Unexpected error - convert to CompileError with full traceback
            error = CompileError.from_exception(
                e, 
                location=SourceLocation(str(filepath), 1, 1)
            )
            print("Internal Compiler Error:")
            print(str(error))
            sys.exit(1)
            
    def compile_files(self, files: List[str]) -> Dict[str, Any]:
        """Compile multiple source files"""
        results = {}
        
        # First pass: Parse all files and collect imports
        for file in files:
            path = Path(file).resolve()
            module_name = path.stem
            
            self.symbol_table.enter_module(module_name, path)
            
            with open(path) as f:
                source = f.read()
                ast_tree = self.parser.parse(source, file_path=str(path))
                statements = ast_tree.statements if hasattr(ast_tree, 'statements') else [ast_tree]
                module = ast.Module(
                    name=module_name,
                    body=ast.ModuleBody(
                        statements=statements,
                        docstring=None,  # TODO: Extract docstring from source
                        exports=[]  # TODO: Handle exports
                    )
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
        return self.vm.run(code)

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
    parser.add_argument('--link-mode', choices=['static', 'dynamic', 'library'], default='static',
                       help='Linking mode (default: static)')
    parser.add_argument('--library-name', help='Library name (for library builds)')
    parser.add_argument('--cuda', action='store_true', help='Enable CUDA support')
    
    args = parser.parse_args()
    
   
    if not args.files:
        parser.print_help()
        return
    
    options = CompileOptions(
        target=args.target,
        optimize=args.optimize,
        debug=args.debug,
        output=args.output,
        dump_ir=args.dump_ir,
        dump_ast=args.dump_ast,
        run_in_vm=args.run,
        link_mode=args.link_mode,
        library_name=args.library_name,
        cuda_enabled=args.cuda
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
            # Ensure output directory exists
            output_dir = args.output or "build"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            main_module = Path(args.files[0]).stem
            transpiler = VMToCCompiler(results[main_module], compiler.c_linker, options=options)
            results = {main_module: transpiler.compile()}  # TODO: Handle multiple modules
            
            # Write output files
            for module_name, code in results.items():
                if not isinstance(code, (str, bytes)):
                    raise CompileError(
                        message=f"Generated code for module '{module_name}' has invalid type {type(code)}",
                        error_type="CodegenError",
                        location=None,
                        notes=[
                            "Expected string or bytes output from code generator",
                            "This may indicate a problem with the code generator or C backend"
                        ]
                    )
                
                # Determine output filename based on link mode
                if options.link_mode == "library":
                    if options.link_mode == "dynamic":
                        output_file = Path(output_dir) / f"lib{module_name}.so"
                    else:
                        output_file = Path(output_dir) / f"lib{module_name}.a"
                else:
                    output_file = Path(output_dir) / f"{module_name}.c"
                
                try:
                    with open(output_file, 'w') as f:
                        f.write(code)
                except IOError as e:
                    raise CompileError(
                        message=f"Failed to write output file {output_file}: {str(e)}",
                        error_type="IOError",
                        location=None,
                        notes=[
                            f"Make sure you have write permissions for directory: {output_dir}",
                            "Check if the disk has enough space"
                        ]
                    )
            
            # Compile C code if necessary
            if options.link_mode != "library":
                try:
                    compiler.c_linker.compile_and_link(
                        [str(Path(output_dir) / f"{module_name}.c") for module_name in results.keys()],
                        str(Path(output_dir) / main_module),
                        options.link_mode == "dynamic"
                    )
                except Exception as e:
                    raise CompileError(
                        message=f"C compilation failed: {str(e)}",
                        error_type="LinkError",
                        location=None,
                        notes=["Check C compiler output for details"]
                    )
            
            print(f"Generated {'library' if options.link_mode == 'library' else 'executable'} in {output_dir}")
            
    except CompileError as e:
        print(str(e))
        sys.exit(1)
    except Exception as e:
        # Unexpected error - provide as much context as possible
        error = CompileError(
            message=str(e),
            error_type="CompilationError",
            location=SourceLocation("", 0, 0),
            stack_trace=traceback.format_stack(),
            notes=["This may be a compiler bug - please report it"]
        )
        print(str(error))
        sys.exit(1)

if __name__ == "__main__":
    main()
