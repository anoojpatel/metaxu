"""The old name of the package manager. `tap` replaced it (docs/tap.md);
this script forwards so existing instructions keep working.

    uv run python scripts/mxpkg.py sync      ==  uv run tap sync
"""
import sys

from metaxu.tap.cli import main

if __name__ == "__main__":
    print("mxpkg is now `tap`; forwarding", file=sys.stderr)
    sys.exit(main())
