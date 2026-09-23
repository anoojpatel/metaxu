"""tap: the Metaxu package manager (docs/tap.md).

Named for Simone Weil's wall: prisoners in neighbouring cells tap
messages through the wall that separates them. A package manager is the
same thing between repositories.

    semver     versions and version requirements (^, ~, >=, <, =, *)
    pubgrub    the version solver (the algorithm Cargo, uv and pub use)
    index      the registry: a git repository of package descriptions
    project    manifest, lockfile, resolution, vendoring
    cli        the `tap` command
"""
