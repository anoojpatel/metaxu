from dataclasses import dataclass, field
from typing import List, Optional, Any
from pathlib import Path

@dataclass
class SourceLocation:
    """Location in source code"""
    file: str
    line: int
    column: int
    
    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}"

@dataclass
class CompileError(Exception):
    """Detailed compile error with source location and context"""
    message: str
    error_type: str = "CompilationError"  # e.g. "LexError", "ParseError", "TypeError"
    location: Optional[SourceLocation] = None
    node: Optional[Any] = None  # AST node if available
    context: Optional[str] = None
    stack_trace: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)  # Additional notes/hints
    traceback: Optional[str] = None  # For internal errors, full Python traceback

    def __str__(self) -> str:
        parts = []
        
        # Error type and location
        loc = str(self.location) if self.location else "unknown location"
        parts.append(f"{self.error_type} at {loc}: {self.message}")
        
        # Source context if available
        if self.context:
            parts.append("\nContext:")
            parts.append(self.context)
            if self.location and self.location.column:
                parts.append(" " * (self.location.column - 1) + "^")
        
        # Additional notes
        if self.notes:
            parts.append("\nNotes:")
            parts.extend(f"  - {note}" for note in self.notes)
        
        # Stack trace from compiler
        if self.stack_trace:
            parts.append("\nStack trace:")
            parts.extend(f"  {frame}" for frame in self.stack_trace)

        # Python traceback for internal errors
        if self.traceback:
            parts.append("\nPython traceback:")
            parts.append(self.traceback)
        
        return "\n".join(parts)

    @classmethod
    def from_exception(cls, e: Exception, location: Optional[SourceLocation] = None) -> 'CompileError':
        """Create a CompileError from a Python exception with full traceback"""
        import traceback
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        return cls(
            message=str(e),
            error_type="InternalError",
            location=location,
            traceback=tb,
            notes=["This may be a compiler bug - please report it"]
        )

def get_source_context(file_path: str, line: int, context_lines: int = 3) -> Optional[str]:
    """Get source code context around a location"""
    try:
        path = Path(file_path)
        if not path.exists():
            return None
            
        with open(path) as f:
            lines = f.readlines()
            
        start = max(0, line - context_lines - 1)
        end = min(len(lines), line + context_lines)
        
        context = []
        for i in range(start, end):
            line_num = i + 1
            prefix = '> ' if line_num == line else '  '
            context.append(f"{prefix}{line_num:4d} | {lines[i].rstrip()}")
            
        return '\n'.join(context)
    except:
        return None
