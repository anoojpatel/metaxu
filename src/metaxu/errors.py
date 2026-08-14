from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from pathlib import Path

@dataclass
class SourceLocation:
    """Location in source code.

    `line` and `column` are 1-based and describe the FIRST character of the
    construct.  `end_line`/`end_column` (when known) point just PAST its last
    character, so `column..end_column` on a single line is a half-open range
    usable for underlining.  `offset`/`end_offset` are the same range as
    0-based character offsets into the source text.
    """
    file: str
    line: int
    column: int
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    offset: Optional[int] = None
    end_offset: Optional[int] = None

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column}"


# ---------------------------------------------------------------------------
# Source registry
#
# Diagnostics are rendered with an excerpt of the offending line.  Sources
# compiled from memory (file_path "<mem>", the test/REPL front door) have no
# file to read back, so the front end registers the text here keyed by the
# path it was compiled under.  On-disk files are read from disk as before,
# so nothing needs to be registered for them.
# ---------------------------------------------------------------------------

_SOURCE_TEXTS: Dict[str, str] = {}


def register_source(file_path: Optional[str], text: Optional[str]) -> None:
    """Remember `text` as the contents of `file_path` for diagnostics."""
    if file_path and isinstance(text, str):
        _SOURCE_TEXTS[file_path] = text


def get_source_text(file_path: Optional[str]) -> Optional[str]:
    """Registered text for `file_path`, else its contents on disk, else None."""
    if not file_path:
        return None
    text = _SOURCE_TEXTS.get(file_path)
    if text is not None:
        return text
    try:
        path = Path(file_path)
        if path.is_file():
            return path.read_text()
    except OSError:
        return None
    return None


def source_excerpt(location: Optional['SourceLocation'], indent: str = "  ") -> Optional[str]:
    """A two-line excerpt for `location`: the source line and a caret line.

        3 | struct Buf { data: vector[T, N] }
          |                    ^~~~~~~~~~~~

    Returns None when the source text is unavailable or the location does not
    name a real line.  The caret column is clamped to the line's length so a
    stale column can never mis-render.
    """
    if location is None or not getattr(location, 'line', 0):
        return None
    text = get_source_text(getattr(location, 'file', None))
    if text is None:
        return None
    lines = text.splitlines()
    line_no = location.line
    if line_no < 1 or line_no > len(lines):
        return None
    src = lines[line_no - 1].replace('\t', ' ')
    gutter = str(line_no)
    col = max(1, min(location.column or 1, len(src) + 1))
    width = 1
    end_col = getattr(location, 'end_column', None)
    if end_col and (getattr(location, 'end_line', line_no) or line_no) == line_no:
        width = max(1, min(end_col, len(src) + 1) - col)
    caret = '^' + '~' * (width - 1)
    return (f"{indent}{gutter} | {src}\n"
            f"{indent}{' ' * len(gutter)} | {' ' * (col - 1)}{caret}")


def format_location(location: Optional['SourceLocation'], fallback: str = "unknown location") -> str:
    """`file:line:column` for a location, `file` alone when the line is
    unknown, and `fallback` when there is no location at all."""
    if location is None:
        return fallback
    file = getattr(location, 'file', None) or fallback
    line = getattr(location, 'line', 0) or 0
    if not line:
        return str(file)
    return f"{file}:{line}:{getattr(location, 'column', 0) or 0}"


def format_diagnostic(message: str,
                      location: Optional['SourceLocation'],
                      error_type: Optional[str] = None,
                      notes: Optional[List[str]] = None) -> str:
    """The one diagnostic format every loud compiler error uses:

        <file>:<line>:<col>: <error type>: <message>
          <line> | <source line>
                 | <caret>
          note: ...

    The location prefix is omitted entirely when nothing is known about the
    location, so a message never grows a misleading `<unknown>:0:0`.
    """
    head = format_location(location, fallback="")
    label = f"{error_type}: " if error_type else ""
    parts = [f"{head}: {label}{message}" if head else f"{label}{message}"]
    excerpt = source_excerpt(location)
    if excerpt:
        parts.append(excerpt)
    for note in notes or []:
        parts.append(f"  note: {note}")
    return "\n".join(parts)


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

    def __post_init__(self):
        # A caller that only had the offending AST node still gets a located
        # diagnostic: the parser attaches SourceLocation to every node it
        # builds (see parser.Parser._attach_location).
        if self.location is None and self.node is not None:
            loc = getattr(self.node, 'location', None)
            if isinstance(loc, SourceLocation):
                self.location = loc
        # Snapshot the excerpt now: in-memory sources all register under the
        # same "<mem>" path, so rendering it lazily could quote a LATER
        # compilation's text.
        self._excerpt = source_excerpt(self.location)

    def __str__(self) -> str:
        # `file:line:column: ErrorType: message` plus the offending source
        # line with a caret — the same shape every located diagnostic uses.
        head = format_location(self.location, fallback="")
        label = f"{self.error_type}: " if self.error_type else ""
        parts = [f"{head}: {label}{self.message}" if head else f"{label}{self.message}"]
        if getattr(self, '_excerpt', None):
            parts.append(self._excerpt)

        # Explicit context passed by the caller (legacy multi-line form)
        if self.context:
            parts.append("\nContext:")
            parts.append(self.context)

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
        text = get_source_text(file_path)
        if text is None:
            return None
        lines = text.splitlines(keepends=True)

        start = max(0, line - context_lines - 1)
        end = min(len(lines), line + context_lines)

        context = []
        for i in range(start, end):
            line_num = i + 1
            prefix = '> ' if line_num == line else '  '
            context.append(f"{prefix}{line_num:4d} | {lines[i].rstrip()}")

        return '\n'.join(context)
    except Exception:
        return None
