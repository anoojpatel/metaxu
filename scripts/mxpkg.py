"""The old name of the package manager. `glade` replaced it (docs/glade.md);
this script forwards so existing instructions keep working.

    uv run python scripts/mxpkg.py sync      ==  uv run glade sync
"""
import sys

from metaxu.glade.cli import main

if __name__ == "__main__":
    print("mxpkg is now `glade`; forwarding", file=sys.stderr)
    sys.exit(main())
