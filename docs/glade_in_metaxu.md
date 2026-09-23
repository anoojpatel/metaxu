# Writing glade in Metaxu: what has to exist first

`glade` is about 1,500 lines of Python. Rewriting it in Metaxu is a
useful target because the things it needs are the things any real
program needs: files, processes, the environment, hashing, parsing.
None of them exist in the language yet, and the compiler will need
the same ones on its own road to self-hosting. This page is the
inventory: what glade uses, what Metaxu has today, and for each gap
whether it should be written in Metaxu or bound to C.

The Python glade is the oracle throughout. Every Metaxu piece below
gets a differential test against it on the same inputs, the way the
native backend is tested against the interpreter.

## What glade actually uses

| glade does | Python it leans on |
|---|---|
| parse and compare versions and requirements | `re`, tuple ordering (semver.py) |
| solve the dependency graph | dicts, lists, dataclasses, recursion (pubgrub.py) |
| read the index and manifests, write the lock and manifest | `tomllib`, string formatting |
| clone, fetch, tag lookup, `rev-parse` | `subprocess.run(["git", ...])` |
| vendor a tree, prune, hash it | `shutil`, `pathlib`, `hashlib.sha256` |
| find the cache, the registry, the home directory | `os.environ`, `Path.home()` |
| the command line | `argparse`, `sys.argv`, `sys.stderr`, exit codes |
| `paths` output | `json.dumps` |

No networking of its own: git does all of it. That stays true in the
rewrite, which is the design paying off.

## What Metaxu has today that this can stand on

- Strings with `len`, slicing, `+`, `==`; `std.string` (join, repeat,
  `char_at`, `starts_with`, `ends_with`, `index_of_char`); `std.parse`
  (`trim`, `split_on`, `parse_int`, `parse_bool`).
- `Vec`, `std.vec`, `std.sort`, `std.map` (an association list with
  the HashMap API, correct and O(n)), `std.option`, `std.result`.
- Effects for control and IO shape: `std.fail`, `std.throw`,
  `std.early_return`, `std.state`, `std.log`, `std.test`.
- Structs, enums, pattern matching, traits, generics, closures,
  recursion with a checked depth budget.
- 64-bit integers with defined wrapping bitwise ops (`&`, `|`, `^`,
  `~`, shifts).
- `extern "C"` declarations with pointer types, `unsafe` blocks, and a
  simulated heap in the interpreter that matches native byte for byte.
  The interpreter ships shims for exactly six symbols: `malloc`,
  `free`, `realloc`, `memcpy`, `fopen`, `fclose`. An extern with no
  shim fails loudly at call time.
- `print` to stdout and `main`'s return as the exit code.

What it does not have: reading a file, listing a directory, running a
process, reading an environment variable, arguments to `main`, writing
to stderr, a byte type, a hash function, or a TOML parser.

## The list

Three buckets. "Build" means pure Metaxu in `std/`. "FFI" means a C
implementation reached through an effect whose operations map to
runtime symbols, the pattern `Thread` and `Mutex` already use
(`with EFFECT_*`), so each one needs a native C side in
`src/metaxu/runtime/native/` and an interpreter shim in
`mir_interp.py`, differentially tested against each other.

### Build in Metaxu

| module | what | notes |
|---|---|---|
| `std.semver` | `Version`, `Range` as sorted disjoint intervals with intersect, union, complement; requirement parsing (`^`, `~`, `>=`, `<`, `=`, `*`, commas) | Pure. Portable today. The prerelease rule and partial versions are the fiddly parts; the Python tests transfer one to one. |
| `std.solve` | PubGrub: terms, incompatibilities, partial solution, propagation, conflict resolution, the widening step, the explanation writer | Pure. Needs `std.map` keyed by strings, recursion, and a `Provider` trait with `versions` and `dependencies`. The largest single piece, and the best stress test of traits and enums the language has had. |
| `std.toml` | a reader for the subset the manifests use (tables, arrays of tables, inline tables, strings, ints, bools, arrays) and a writer | Needs `std.bytes` for correct string escapes. A full TOML 1.0 reader can come later; the lock and index never use dates or multiline strings. |
| `std.path` | join, dirname, basename, normalize, relative-to, `is_absolute` | String only. `relpath` is the one glade needs and gets wrong without care. |
| `std.json` | a writer | `glade paths` prints a JSON object; no reader needed. |
| `std.args` | flag and subcommand parsing over `Vec` of strings | Replaces argparse. Depends on `std.env` for the argument vector. |
| `std.hex` and `std.sha256` | hex encoding; SHA-256 over bytes | SHA-256 is 32-bit arithmetic with wrapping adds and rotates, which the integer ops already give. Writing it in Metaxu keeps the native binary free of libcrypto. Bind C only if the interpreter's speed on large trees becomes a problem. |
| the tool itself | manifest and lock models, index parsing, resolution, sync, add, remove, tree, check, search | Ordinary code once the pieces above exist. Lives in its own package, built and vendored by glade. |

### Fill in the language and standard library first

These are not glade features. They are gaps the modules above run
into, and each needs work in the compiler as well as `std/`.

| gap | why glade hits it | what to add |
|---|---|---|
| a byte type and string ↔ bytes | hashing a tree, reading files that are not UTF-8, TOML escapes | `bytes` as a `Vec` of `u8` or a distinct runtime value; `string.to_bytes()`, `bytes.to_string()`, byte indexing. Both engines and the C runtime. |
| string builtins with linear cost | `std.string` does `starts_with` and `index_of_char` by `char_at` loops, which is O(n²) over a manifest | `split`, `find`, `replace`, `trim`, `to_int`, `join` as `__builtin$m` methods backed by `mx_str_*` in C, with the interpreter matching. |
| a hashed `Map` | the solver keeps a dozen maps keyed by package name; O(n) lookups are fine at glade's sizes but wrong in spirit | a `Hash` trait, `impl Hash for string/int`, and a bucketed `std.map` behind the same API. Not blocking. |
| `main` arguments | the command line | either `fn main(args: Vec) -> int` accepted by both engines, or `std.env.args()`. The native `@main` wrapper already exists; it needs to carry `argv` through. |
| stderr | diagnostics must not go to stdout | `eprint` builtin, or `std.io.stderr` as an effect operation. |
| a process-exit with a message | `glade: <error>` then exit 2 | falls out of `std.fail` plus `eprint` and `main`'s return. |

### FFI out, as effects with C handlers

| effect | operations glade needs | C side | interpreter shim |
|---|---|---|---|
| `std.fs` | `read(path) -> Result[bytes]`, `write(path, bytes)`, `exists`, `is_dir`, `is_file`, `list_dir`, `mkdir_all`, `remove_all`, `rename`, `walk` (or `list_dir` and recursion in Metaxu) | `fopen`/`fread`/`fwrite`/`fclose`, `stat`, `opendir`/`readdir`/`closedir`, `mkdir`, `unlink`/`rmdir`, `rename`; POSIX first, Windows later | Python `os`, `pathlib`, `shutil` |
| `std.process` | `run(argv: Vec, cwd) -> {status, stdout, stderr}` | `posix_spawn` or `fork`+`execvp`, pipes, `waitpid`; capture both streams without deadlock | `subprocess.run` |
| `std.env` | `get(name) -> Option[string]`, `home()`, `cwd()`, `args()` | `getenv`, `getcwd`, `argv` from the entry wrapper | `os.environ`, `os.getcwd`, `sys.argv` |
| `std.io` | `stderr(text)`, and `stdin` later | `fputs` on `stderr` | `sys.stderr` |

Making these effects rather than plain externs is the point of the
language: a test installs a handler that serves an in-memory
filesystem and a fake `git`, and the whole of glade runs under it with
no mocking library. The native handler and the interpreter shim must
agree on results and on error text, which is the same contract every
other builtin lives under.

What is deliberately not on the list: `std.net` (git is the network),
`std.time` (glade records no timestamps), threads for parallel fetches
(the `Thread` effect exists already if wanted), and Windows paths
(POSIX first, as a stated limit).

## Order of work

1. **`std.semver` and `std.solve`.** Pure code, possible today, with
   the Python tests ported as the oracle. This proves the language can
   carry a real algorithm and shakes out whatever the trait and enum
   paths still hide.
2. **Bytes, string builtins, `std.sha256`, `std.toml`, `std.path`.**
   Still pure Metaxu, but the first two need compiler and runtime
   work. After this step every non-IO part of glade can be written.
3. **`std.fs`, `std.process`, `std.env`, `std.io` and `main` arguments.**
   Runtime work on both engines with differential tests, the same
   discipline as `Thread`. These are the primitives the compiler
   itself needs later, so nothing here is glade-only.
4. **glade in Metaxu.** A package, built by the Python glade at first,
   then by itself. Parity test: both implementations run the fixtures
   in `test_glade.py` and must produce identical lockfiles and
   identical error text. Then the console script switches over and
   the Python version becomes the reference the way the interpreter
   is for the backends.

Steps 1 and 2 can start now. Step 3 is where the design decisions
are, chiefly what a `bytes` value is and how effect operations pass
`Vec` and `bytes` across the C boundary; those deserve their own
notes before code.
