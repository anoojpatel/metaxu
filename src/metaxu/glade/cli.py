"""`glade`: the Metaxu package manager's command line.

    glade init [DIR] [--name NAME]       start a project (mx.toml, main.mx)
    glade add NAME [REQ]                 add a registry dependency (REQ defaults to ^newest)
    glade add NAME --git URL --rev REV   add a git dependency
    glade add NAME --path DIR            add a path dependency
    glade remove NAME
    glade sync                           resolve, fetch, vendor, write mx.lock
    glade update [NAME ...]              re-resolve, letting locked versions move
    glade tree                           the dependency tree with versions
    glade check                          verify mx_modules/ against mx.lock (CI)
    glade search TEXT                    find packages in the registry
    glade paths                          name -> root as JSON (what the compiler reads)

`add` and `remove` sync afterwards, so the lock and mx_modules/ always
match the manifest you just edited.  Every command reads and writes
the project in the current directory (or --project DIR).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from metaxu.packages import PackageError, check, package_roots, read_lock

from .project import Project, init


def _project(args) -> Project:
    return Project(Path(args.project), registry_override=args.registry, log=print)


def cmd_init(args) -> int:
    path = init(Path(args.dir), args.name)
    print(f"wrote {path}")
    return 0


def cmd_add(args) -> int:
    p = _project(args)
    spec = p.add(args.name, args.requirement, git=args.git, rev=args.rev,
                 path=args.path, registry=args.from_registry)
    if spec.kind == "registry":
        print(f"added {spec.name} = {spec.requirement_text!r}")
    else:
        print(f"added {spec.name} ({spec.kind})")
    p = _project(args)
    p.sync()
    return 0


def cmd_remove(args) -> int:
    p = _project(args)
    p.remove(args.name)
    print(f"removed {args.name}")
    _project(args).sync()
    return 0


def cmd_sync(args) -> int:
    locked = _project(args).sync(refresh=not args.offline)
    print(f"locked {len(locked)} package(s)")
    return 0


def cmd_update(args) -> int:
    p = _project(args)
    names = set(args.names) if args.names else {n for n in read_lock(p.root)}
    before = {n: e.version for n, e in read_lock(p.root).items()}
    locked = p.sync(update=names)
    moved = [f"{n} {before.get(n) or '-'} -> {e.version}" for n, e in locked.items()
             if e.version and before.get(n) != e.version]
    print("\n".join(moved) if moved else "nothing to update")
    return 0


def cmd_tree(args) -> int:
    print(_project(args).tree())
    return 0


def cmd_check(args) -> int:
    problems = check(Path(args.project))
    for p in problems:
        print(p)
    return 1 if problems else 0


def cmd_search(args) -> int:
    p = _project(args)
    reg = p.default_registry
    reg.refresh()
    hits = [n for n in reg.names() if args.text.lower() in n.lower()]
    for name in hits:
        entry = reg.entry(name)
        assert entry is not None
        newest = max(entry.versions) if entry.versions else "-"
        print(f"{name} {newest}  {entry.description}".rstrip())
    if not hits:
        print(f"no packages matching {args.text!r} in {reg.source}")
    return 0


def cmd_paths(args) -> int:
    print(json.dumps({k: str(v) for k, v in package_roots(Path(args.project)).items()},
                     indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="glade", description="The Metaxu package manager.")
    ap.add_argument("--project", default=".", help="project root (default: .)")
    ap.add_argument("--registry", help="registry URL or directory (default: mx.toml [glade] registry, "
                                       "then $GLADE_REGISTRY, then the Metaxu index)")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="start a project")
    p.add_argument("dir", nargs="?", default=".")
    p.add_argument("--name")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("add", help="add a dependency and sync")
    p.add_argument("name")
    p.add_argument("requirement", nargs="?", help='e.g. "^0.2", "~1.4.2", ">=1, <3"')
    p.add_argument("--git")
    p.add_argument("--rev", help="tag, branch or commit for --git")
    p.add_argument("--path")
    p.add_argument("--from", dest="from_registry", help="a registry other than the project's")
    p.set_defaults(fn=cmd_add)

    p = sub.add_parser("remove", help="remove a dependency and sync")
    p.add_argument("name")
    p.set_defaults(fn=cmd_remove)

    p = sub.add_parser("sync", help="resolve, fetch, vendor, write mx.lock")
    p.add_argument("--offline", action="store_true", help="do not refresh the registry index")
    p.set_defaults(fn=cmd_sync)

    p = sub.add_parser("update", help="move locked versions forward within their requirements")
    p.add_argument("names", nargs="*")
    p.set_defaults(fn=cmd_update)

    sub.add_parser("tree", help="print the dependency tree").set_defaults(fn=cmd_tree)
    sub.add_parser("check", help="verify vendored trees against mx.lock").set_defaults(fn=cmd_check)

    p = sub.add_parser("search", help="find packages in the registry")
    p.add_argument("text")
    p.set_defaults(fn=cmd_search)

    sub.add_parser("paths", help="print name -> root as JSON").set_defaults(fn=cmd_paths)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except PackageError as e:
        print(f"glade: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
