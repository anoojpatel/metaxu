"""Shared Parser cache for the compiler pipeline.

Parser() construction rebuilds PLY's LALR tables from scratch
(write_tables=False), costing ~430ms per instance while an actual parse
costs ~0ms. The grammar is fixed at import time, and Parser instances are
reusable across parse() calls (verified: sequential files, error recovery,
line tracking), so the pipeline shares one instance per process.
"""
from __future__ import annotations

_PARSER = None


def shared_parser():
    """Process-wide Parser instance (built once, ~430ms; reused after)."""
    global _PARSER
    if _PARSER is None:
        from metaxu.parser import Parser
        _PARSER = Parser()
    return _PARSER
