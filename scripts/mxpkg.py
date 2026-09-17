"""Command-line entry for Metaxu packages; see docs/packages.md.

    uv run python scripts/mxpkg.py sync
"""
import sys

from metaxu.packages import main

if __name__ == "__main__":
    sys.exit(main())
