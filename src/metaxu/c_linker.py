from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import subprocess
from typing import List, Dict, Optional, Set, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from metaxu.errors import CompileError

class LinkageMode(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    HEADER = "header"  # For header-only libraries

class OptimizationLevel(Enum):
    NONE = auto()     # -O0
    DEBUG = auto()    # -Og
    SIZE = auto()     # -Os
    SPEED = auto()    # -O2
    AGGRESSIVE = auto() # -O3

@dataclass
class CompilerConfig:
    """Configuration for compiler behavior"""
    optimization: OptimizationLevel = OptimizationLevel.NONE
    debug_symbols: bool = False
    parallel: bool = True
    max_jobs: Optional[int] = None  # None means use CPU count
    arch: Optional[str] = None  # Target architecture (e.g., 'x86_64', 'arm64')
    custom_flags: List[str] = field(default_factory=list)
    header_only: bool = False
    position_independent: bool = False

@dataclass
class LibraryConfig:
    """Configuration for a C library dependency"""
    name: str
    version: str = "system"
    headers: List[str] = field(default_factory=list)
    link_mode: LinkageMode = LinkageMode.DYNAMIC
    pkg_config: Optional[str] = None
    include_dirs: List[Path] = field(default_factory=list)
    lib_dirs: List[Path] = field(default_factory=list)

def create_library_config(name: str, **kwargs) -> LibraryConfig:
    return LibraryConfig(name, **kwargs)

class CLinker:
    def __init__(self, build_dir: Path, config: Optional[CompilerConfig] = None):
        """Initialize the C linker with a build directory"""
        self.build_dir = Path(build_dir).resolve()  # Get absolute normalized path
        self.config = config or CompilerConfig()
        self.libraries: Dict[str, LibraryConfig] = {}
        self.include_paths: List[Path] = []
        self.library_paths: List[Path] = []
        self.header_files: Set[Path] = set()

    def _resolve_path(self, path: Union[str, Path], relative_to_build: bool = True) -> Path:
        """Resolve a path, optionally making it relative to build_dir"""
        path = Path(path)
        
        # If path starts with build_dir, don't add it again
        if str(path).startswith(str(self.build_dir)):
            return path.resolve()
            
        if path.is_absolute():
            return path.resolve()
            
        if relative_to_build:
            return (self.build_dir / path).resolve()
            
        return path.resolve()

    def add_library(self, config: LibraryConfig):
        """Add a library dependency"""
        self.libraries[config.name] = config
        
    def add_include_path(self, path: Path):
        """Add an include search path"""
        self.include_paths.append(Path(path))
        
    def add_library_path(self, path: Path):
        """Add a library search path"""
        self.library_paths.append(Path(path))

    def add_header(self, path: Path):
        """Add a header file to be generated/processed"""
        self.header_files.add(Path(path))

    def _get_optimization_flags(self) -> List[str]:
        """Get optimization-related compiler flags"""
        flags = []
        
        # Optimization level
        opt_map = {
            OptimizationLevel.NONE: "-O0",
            OptimizationLevel.DEBUG: "-Og",
            OptimizationLevel.SIZE: "-Os",
            OptimizationLevel.SPEED: "-O2",
            OptimizationLevel.AGGRESSIVE: "-O3"
        }
        flags.append(opt_map[self.config.optimization])
        
        # Debug symbols
        if self.config.debug_symbols:
            flags.append("-g")
            
        # Position independent code
        if self.config.position_independent:
            flags.append("-fPIC")
            
        return flags

    def _get_architecture_flags(self) -> List[str]:
        """Get architecture-specific compiler flags"""
        flags = []
        
        if self.config.arch:
            flags.extend(["-march=" + self.config.arch])
            
            # Add additional arch-specific optimizations
            if self.config.arch in ["x86_64", "amd64"]:
                flags.extend(["-mtune=generic"])
            elif self.config.arch.startswith("arm"):
                flags.extend(["-mfpu=neon", "-mfloat-abi=hard"])
                
        return flags

    def _get_compiler_flags(self) -> List[str]:
        """Get all compiler flags"""
        flags = ["-Wall", "-Wextra"]  # Basic warning flags
        
        # Add optimization flags
        flags.extend(self._get_optimization_flags())
        
        # Add architecture flags
        flags.extend(self._get_architecture_flags())
        
        # Add include paths
        for path in self.include_paths:
            flags.extend(["-I", str(path)])
            
        # Add library paths
        for path in self.library_paths:
            flags.extend(["-L", str(path)])
            
        # Add custom flags
        flags.extend(self.config.custom_flags)
            
        # Add pkg-config flags
        for lib in self.libraries.values():
            if lib.pkg_config:
                try:
                    pkg_flags = subprocess.check_output(
                        ["pkg-config", "--cflags", "--libs", lib.pkg_config],
                        text=True
                    ).strip().split()
                    flags.extend(pkg_flags)
                except subprocess.CalledProcessError as e:
                    raise CompileError(
                        message=f"Failed to get pkg-config flags for {lib.pkg_config}",
                        error_type="LinkError",
                        notes=[f"Make sure {lib.pkg_config} is installed"]
                    )
                    
        return flags

    def _compile_source(self, source_file: str, output_file: str, compiler_flags: List[str]) -> Optional[str]:
        """Compile a single source file"""
        try:
            # Resolve source and output paths
            source_path = self._resolve_path(source_file, relative_to_build=False)  # Don't add build_dir to source
            output_path = self._resolve_path(output_file)  # Add build_dir to output
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if not source_path.exists():
                return f"Source file not found: {source_path}"

            cmd = ["gcc"] + compiler_flags + ["-c", str(source_path), "-o", str(output_path)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.build_dir)  # Convert Path to str for subprocess
            )
            
            if result.returncode != 0:
                return f"Error compiling {source_path}:\n{result.stderr}"
                
            return None
            
        except subprocess.SubprocessError as e:
            return f"Failed to compile {source_file}: {str(e)}"

    def compile_and_link(self, source_files: List[str], output_file: str, is_dynamic: bool = False):
        """Compile and link C source files into an executable or library"""
        try:
            # Ensure build directory exists
            self.build_dir.mkdir(parents=True, exist_ok=True)
            
            # Resolve source files without adding build_dir prefix
            abs_source_files = [str(self._resolve_path(src, relative_to_build=False)) for src in source_files]
            output_path = self._resolve_path(output_file)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Get compiler flags
            compiler_flags = self._get_compiler_flags()
            
            # Generate header files if needed
            for header in self.header_files:
                header_path = self._resolve_path(header.name)
                header_path.parent.mkdir(parents=True, exist_ok=True)
                if not header_path.exists():
                    # TODO: Implement header generation logic
                    pass
            
            # Compile source files in parallel if enabled
            object_files = []
            errors = []
            
            if self.config.parallel:
                max_workers = self.config.max_jobs or multiprocessing.cpu_count()
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    
                    for source in abs_source_files:
                        obj_file = str(self.build_dir / Path(source).name.replace('.c', '.o'))
                        # Ensure object file directory exists
                        Path(obj_file).parent.mkdir(parents=True, exist_ok=True)
                        object_files.append(obj_file)
                        futures.append(
                            executor.submit(
                                self._compile_source,
                                source,
                                obj_file,
                                compiler_flags
                            )
                        )
                        
                    for future in as_completed(futures):
                        if error := future.result():
                            errors.append(error)
            else:
                # Sequential compilation
                for source in abs_source_files:
                    obj_file = str(self.build_dir / Path(source).name.replace('.c', '.o'))
                    # Ensure object file directory exists
                    Path(obj_file).parent.mkdir(parents=True, exist_ok=True)
                    object_files.append(obj_file)
                    if error := self._compile_source(source, obj_file, compiler_flags):
                        errors.append(error)
                        
            if errors:
                raise CompileError(
                    message="Compilation failed",
                    error_type="CompilationError",
                    notes=errors
                )
                
            # Link object files
            linker_flags = []
            if is_dynamic:
                linker_flags.extend(["-shared"])
                
            cmd = ["gcc"] + object_files + ["-o", str(output_path)] + linker_flags
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.build_dir)  # Convert Path to str for subprocess
            )
            
            if result.returncode != 0:
                raise CompileError(
                    message="Linking failed",
                    error_type="LinkError",
                    notes=[result.stderr]
                )
                
        except subprocess.SubprocessError as e:
            raise CompileError(
                message=f"Failed to run C compiler: {str(e)}",
                error_type="CompilationError",
                notes=["Make sure gcc is installed and in your PATH"]
            )
