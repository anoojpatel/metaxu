from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union
from pathlib import Path
import subprocess
import platform
from enum import Enum
import os

class LinkageMode(Enum):
    STATIC = "static"    # .a files
    DYNAMIC = "dynamic"  # .so/.dylib files
    HEADER = "header"    # .h files only

@dataclass
class CLibraryConfig:
    name: str
    version: str
    headers: List[str]
    link_mode: LinkageMode
    lib_paths: List[str] = field(default_factory=list)
    include_paths: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    target_specific: bool = False
    compiler_flags: List[str] = field(default_factory=list)
    linker_flags: List[str] = field(default_factory=list)

class CLinker:
    def __init__(self, build_dir: Path):
        self.build_dir = build_dir
        self.libraries: Dict[str, CLibraryConfig] = {}
        self.linked_libraries: Set[str] = set()
        self._setup_platform()
        
    def _setup_platform(self):
        """Setup platform-specific configurations"""
        self.platform = platform.system().lower()
        if self.platform == "darwin":
            self.lib_extension = ".dylib"
            self.static_lib_extension = ".a"
            self.default_compiler = "clang"
            self.default_linker = "ld"
        elif self.platform == "linux":
            self.lib_extension = ".so"
            self.static_lib_extension = ".a"
            self.default_compiler = "gcc"
            self.default_linker = "ld"
        else:
            raise RuntimeError(f"Unsupported platform: {self.platform}")

    def register_library(self, config: CLibraryConfig):
        """Register a C library for linking"""
        self.libraries[config.name] = config

    def _compile_library(self, config: CLibraryConfig) -> Path:
        """Compile a C library if needed"""
        output_dir = self.build_dir / config.name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Compile source files if any
        objects = []
        for src in config.headers:
            if src.endswith('.c'):  # Only compile .c files
                obj_file = output_dir / f"{Path(src).stem}.o"
                if not obj_file.exists() or self._is_outdated(obj_file, Path(src)):
                    cmd = [
                        self.default_compiler,
                        "-c",
                        "-fPIC",
                        *config.compiler_flags,
                        *[f"-I{p}" for p in config.include_paths],
                        src,
                        "-o",
                        str(obj_file)
                    ]
                    subprocess.run(cmd, check=True)
                objects.append(obj_file)
        
        # Create library
        if config.link_mode == LinkageMode.STATIC:
            lib_file = output_dir / f"lib{config.name}{self.static_lib_extension}"
            if objects and (not lib_file.exists() or self._is_outdated(lib_file, *objects)):
                subprocess.run(["ar", "rcs", str(lib_file), *[str(o) for o in objects]], check=True)
        else:  # Dynamic
            lib_file = output_dir / f"lib{config.name}{self.lib_extension}"
            if objects and (not lib_file.exists() or self._is_outdated(lib_file, *objects)):
                cmd = [
                    self.default_compiler,
                    "-shared",
                    *config.linker_flags,
                    "-o",
                    str(lib_file),
                    *[str(o) for o in objects]
                ]
                subprocess.run(cmd, check=True)
        
        return lib_file if objects else None

    def _is_outdated(self, target: Path, *sources: Path) -> bool:
        """Check if target is older than any of the sources"""
        if not target.exists():
            return True
        target_mtime = target.stat().st_mtime
        return any(s.stat().st_mtime > target_mtime for s in sources if s.exists())

    def link_library(self, name: str, target_file: Path):
        """Link a C library into the target file"""
        if name in self.linked_libraries:
            return
            
        config = self.libraries.get(name)
        if not config:
            raise ValueError(f"Unknown library: {name}")
            
        # Link dependencies first
        for dep in config.dependencies:
            self.link_library(dep, target_file)
            
        # Compile and link the library
        lib_file = self._compile_library(config)
        if lib_file:
            if config.link_mode == LinkageMode.STATIC:
                # For static linking, we need to include the whole archive
                subprocess.run([
                    self.default_linker,
                    "-r",
                    str(target_file),
                    str(lib_file),
                    *config.linker_flags,
                    "-o",
                    str(target_file) + ".tmp"
                ], check=True)
                os.rename(str(target_file) + ".tmp", str(target_file))
            else:
                # For dynamic linking, we just need to add the library path and name
                subprocess.run([
                    self.default_linker,
                    "-r",
                    str(target_file),
                    f"-L{lib_file.parent}",
                    f"-l{config.name}",
                    *config.linker_flags,
                    "-o",
                    str(target_file) + ".tmp"
                ], check=True)
                os.rename(str(target_file) + ".tmp", str(target_file))
                
        self.linked_libraries.add(name)

    def generate_bindings(self, config: CLibraryConfig) -> str:
        """Generate Metaxu bindings for a C library"""
        bindings = []
        bindings.append(f"module {config.name} {{")
        
        # Add extern declarations for all functions
        for header in config.headers:
            if header.endswith('.h'):
                # Parse header and generate bindings
                # This is a simplified version - you'd want to use a proper C parser
                bindings.append(f'    extern "{header}" {{')
                # Add function declarations
                bindings.append('    }')
                
        bindings.append("}")
        return "\n".join(bindings)

def create_library_config(
    name: str,
    version: str,
    headers: List[str],
    link_mode: LinkageMode,
    target_specific: bool = False,
    **kwargs
) -> CLibraryConfig:
    """Helper function to create library configurations"""
    return CLibraryConfig(
        name=name,
        version=version,
        headers=headers,
        link_mode=link_mode,
        target_specific=target_specific,
        **kwargs
    )
