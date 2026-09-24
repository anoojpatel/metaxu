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
| `std.toml` | a reader for the subset the manifests use (tables, arrays of tables, inline tables, strings, ints, bools, arrays) and a writer | Done. The reader accepts exactly what the manifests, locks and index entries use and rejects the rest by name (floats, dates, literal strings, hex); the writer produces glade's layout (inline tables inside sections, one `[[package]]` per lock entry). A full TOML 1.0 reader can come later. |
| `std.path` | join, dirname, basename, normalize, relative-to, `is_absolute` | Done, against `posixpath` case for case. `relpath` takes two absolute or two relative paths, since there is no working directory to resolve against yet. |
| `std.json` | a writer | Done: `to_json` and `to_json_pretty` match `json.dumps` compact and `indent=2`. `glade paths` prints a JSON object; no reader needed. |
| `std.args` | flag and subcommand parsing over `Vec` of strings | Replaces argparse. Depends on `std.env` for the argument vector. |
| `std.hex` and `std.sha256` | hex encoding; SHA-256 over bytes | Done. SHA-256 is 32-bit arithmetic with wrapping adds and rotates, which the integer ops already give; the port is FIPS 180-4 in a hundred lines with the constants in decimal (no hex literals yet), tested against hashlib on both engines. Bind C only if the interpreter's speed on large trees becomes a problem. |
| the tool itself | manifest and lock models, index parsing, resolution, sync, add, remove, tree, check, search | Ordinary code once the pieces above exist. Lives in its own package, built and vendored by glade. |

### Fill in the language and standard library first

These are not glade features. They are gaps the modules above run
into, and each needs work in the compiler as well as `std/`.

| gap | why glade hits it | what to add |
|---|---|---|
| a byte type and string ↔ bytes | hashing a tree, reading files that are not UTF-8, TOML escapes | Done, as the cheaper of the two options: bytes are a `Vec` of ints in 0..255 (what `as_ptr` already accepted), `s.to_bytes()` gives a string's UTF-8 bytes and `v.from_bytes()` decodes them, rejecting a non-byte, a NUL and invalid UTF-8 with the same three messages on both engines. A distinct byte type can replace the representation later without changing callers. |
| string builtins with linear cost | `std.string` does `starts_with` and `index_of_char` by `char_at` loops, which is O(n²) over a manifest | Done: `split`, `find`, `replace`, `trim`, `join` are `__builtin$m` methods backed by `mx_str_*` in C, with the interpreter matching; `std.parse.trim`/`split_on` and `std.string.join` are thin names over them. `to_int` stays `std.parse.parse_int`, which is linear already. |
| a hashed `Map` | the solver keeps a dozen maps keyed by package name; O(n) lookups are fine at glade's sizes but wrong in spirit | a `Hash` trait, `impl Hash for string/int`, and a bucketed `std.map` behind the same API. Not blocking. |
| `main` arguments | the command line | Done as `std.env.args()`: the native `@main` wrapper takes `argc, argv` and hands them to the IO runtime; `metaxuc run file.mx -- a b` sets them on the interpreter. `main` keeps its signature. |
| stderr | diagnostics must not go to stdout | Done: `std.io.eprint` / `eprintln` as an effect operation. |
| a process-exit with a message | `glade: <error>` then exit 2 | falls out of `try`/`catch` plus `eprintln` and `main`'s return. |

### FFI out, as effects with C handlers

| effect | operations glade needs | C side | interpreter shim |
|---|---|---|---|
| `std.fs` | Done: `read_bytes`, `write_bytes`, `exists`, `is_dir`, `is_file`, `list_dir`, `mkdir_all`, `remove_all`, `rename`, with `read_text`, `write_text`, `try_read_text` (Result) and `walk` in Metaxu | `metaxu_io.c`: `open`/`read`/`write`, `stat`, `opendir`/`readdir`, `mkdir`, `unlink`/`rmdir`, `rename`; POSIX first, Windows later | Python `os`, `shutil` |
| `std.process` | Done: `run(argv, cwd)` answers a handle read by `status`, `stdout`, `stderr`; `run`, `run_in`, `output_of` wrap it into a `Completed` record or a Result | `fork`+`execvp` with an exec-failure pipe, both streams drained with `poll`, `waitpid` | `subprocess.run` |
| `std.env` | Done: `lookup(name) -> Option`, `get_or`, `has`, `home()`, `cwd()`, `args()` | `getenv`, `getcwd`, `argv` handed to `mx_io_set_args` by the entry wrapper | `os.environ`, `os.getcwd`, `MirInterpreter.program_args` (set by `metaxuc run file.mx -- a b`) |
| `std.io` | Done: `eprint`, `eprintln` over `write_err`; `stdin` later | `fputs` on `stderr` after flushing stdout | `sys.stderr` |

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

## Progress

**Step 1, `std.semver`: done on the interpreter.** `std/semver.mx`
mirrors the Python module function for function, and
`test_std_semver.py` runs both over a few hundred generated cases
(parsing, ordering, requirements, membership with the prerelease rule,
and the range algebra) and compares every line. The port took one
syntax correction (an assignment is not a match arm; wrap it in
braces) and fixed two laxities in the Python oracle on the way: it
accepted empty identifiers in `1.0.0-a..b` and in requirements.

**Step 1, `std.solve`: done on the interpreter.** `std/solve.mx` is
PubGrub over a data `Graph` (a registry index lists every version's
dependencies, so no Provider trait is needed yet).
`test_std_solve.py` runs thirteen graphs through both solvers,
including deep backtracking and the fifty-version widening case, and
requires identical picks and identical explanation sentences. It
matched on the first run. Two Metaxu facts shaped the port: a struct
passed as `@mut` is shared, so the solver state is one struct that
helpers mutate; and `std.map` moves a key to the end on overwrite,
which would change which package the solver tries first, so the
port keeps its own insertion-ordered tables.

**Both modules compile and run natively.** Getting there took four
backend fixes, each found by these two modules and each now pinned in
`test_codegen_llvm.py`; the native differential tests in
`test_std_semver.py` and `test_std_solve.py` are required passes.

- **String indexing was not lowered.** `s[i]` and `s[a:b:c]` on a
  string were interpreter-only, which made every helper in `std.parse`
  and `std.string` interpreter-only too. They now lower to
  `mx_str_index` and `mx_str_slice` in the C runtime, fresh copies with
  the interpreter's exact bounds diagnostic. The linear string builtins
  above remain the way to make that text code fast rather than merely
  native.
- **A nested enum lost its payload kinds.** `component_or_wild` returns
  `Some(Some(n))` or `Some(None)`. The backend recorded the inner
  `Option` name-only and read its payload back through the module-wide
  `Option` cells, which conflict as soon as one file puts ints, Vecs
  and structs into `Option`; that conflict then spread through
  two-way signature unification into sixty functions. Nested enums now
  keep their own refinement three levels deep, so the inner read is
  exact and never touches the cells.
- **Kind-polymorphic helpers conflicted.** `is_none(o: Option)` is
  called with `Option` of string and `Option` of `Incompat`, and a
  signature is one join over every call site. The backend now clones
  such a function per call-site kind tuple before inference, the way
  the HIR monomorphizer clones declared generics. The same pass fixes a
  silent wrong answer: a helper reached with `Vec` of int and `Vec` of
  float used to merge to float and print `10.0` for `10`.
- **An assignment evaluated to the struct it wrote.** `record` ends in
  an `if` whose arms are a `push` and a field assignment; the second
  arm's value was the whole `Solver`, merged with unit, and the
  function demoted. Assignments now evaluate to `()` on both engines.

**Step 2, string builtins: done.** `split`, `find`, `replace`, `trim`
and `join` are builtin methods on both engines, Python-backed in the
interpreter and `mx_str_*` in C natively, with the two catchable
diagnostics (`split: empty separator`, `replace: empty pattern`)
byte-identical. `test_string_builtins.py` runs a few hundred generated
cases against Python's `str` methods on the interpreter and the same
program natively.

**Step 2, bytes, `std.hex`, `std.sha256`: done.** Bytes are a `Vec`
of ints (see the table above), `std.hex` renders and parses them, and
`std.sha256` is FIPS 180-4 in Metaxu; `test_std_hash.py` checks every
padding boundary and a few hundred bytes of patterns against hashlib,
then runs the same program natively. The interpreter hashes about
twenty kilobytes a second, enough for manifests and small trees.

**Step 2, `std.path`, `std.json`, `std.toml`: done.** Each has a
Python oracle test (`posixpath`, `json.dumps`, `tomllib`) on the
interpreter and a native differential. The TOML reader and the JSON
writer are the first recursive-enum programs (a `Toml` holds `Vec`s of
`Toml`) to compile natively, and they took three more backend fixes:
a name-only enum kind (past the refinement depth) must never become a
value's kind but read as the canonical refinement, an empty `Vec`
stored into a payload slot learns the slot's element kind from the
module-wide cells so those cells stay unmixed, and a recursive
function's self-call never opens a new specialization group. With
that, every non-IO part of glade can now be written.

**Step 3, the IO effects: done.** `std.fs`, `std.process`, `std.env`
and `std.io` are effects whose operations map to runtime primitives on
both engines, POSIX in `metaxu_io.c` natively and `os`/`shutil`/
`subprocess` on the interpreter, with the error text shared word for
word; `docs/io_runtime.md` is the contract (one word per value, bytes
as a `Vec` of ints, catchable `<op>: <path>: <reason>` failures, a
handle per finished process, `args()` for the command line).
`test_std_io.py` runs one program through both engines inside a scratch
directory and a second one entirely under handlers that serve an
in-memory tree and a fake `git`, which is the shape glade's own tests
will take. Every primitive the package manager needs now exists.

What the two programs still leave as placeholders is `std.fail`'s
higher-order handlers (`try_opt(f)` and friends call a closure passed as
a parameter), which they import but never call.

**Step 4, glade itself: done.** `glade/*.mx` at the repository root is
the package manager, about 1,600 lines over the modules above:
`manifest.mx` and `lock.mx` read and write `mx.toml` and `mx.lock` in
the Python layout byte for byte, `index.mx` is the registry with its
cache and the moved-tag check, `project.mx` resolves through
`std.solve` and does `sync`, `add`, `remove`, `update`, `tree`,
`check`, `paths` and `init`, `hash.mx` is the tree hash, `git.mx` runs
git, and `main.mx` is the command. It type-checked after three fixes
(`where` and `public` are keywords, so a parameter and a manifest field
were renamed, and `std.process` and `std.solve` both declared an
`Outcome`) and matched the Python glade on its first run: `add`,
`check`, `tree`, `paths`, `search`, an unsatisfiable requirement and an
unknown package all agreed on the lockfile, stdout and stderr.

`test_glade_metaxu.py` is the parity test the plan asked for: both
implementations run the scenarios of `test_glade.py` (the directory
registry and tagged git repositories, an older line and `update`,
conflicts between two requesters, a moved tag, path and git
dependencies, `remove`, drift under `check`, the usage errors), each in
its own project and cache, and every step must agree on the exit code,
stdout, stderr, manifest, lockfile and vendored trees. It found two
things in the reference: a malformed requirement on the command line
escaped as a traceback instead of `glade: ...`, and stale vendored
packages were removed in directory-listing order rather than sorted, so
the log line order was not deterministic. The Metaxu side adopted the
Python error texts word for word.

The same program compiles natively; two more backend fixes came out of
it, both pinned in `test_codegen_llvm.py`:

- **String ordering.** The sorts compare names with `<`, which the
  backend demoted (`only ==/!= lower to mx_str_eq`). The four ordering
  comparisons now call `mx_str_cmp`, strcmp on the UTF-8 bytes, which
  is code point order and so the interpreter's.
- **A `let` evaluated to its value.** A branch ending in
  `let a = git(...)` handed back the string, the other branch was `()`,
  and the merge demoted the function. A `let` is a statement whose
  value is `()` on both engines now, as an assignment already was.

What still demotes is unreached: `std.toml`'s writer (glade writes its
files line by line), `std.string.is_empty` and `std.fail`'s
higher-order handlers, imported by modules glade uses.
`test_native_glade_matches_and_the_launcher_caches_the_binary` runs the
native binary against the Python glade on the main scenario.

The console script switched over: `glade` is `metaxu.glade.launch`,
which compiles the program once into a cached native binary when clang
is installed (about a minute, printed as `building the native glade`)
and otherwise runs it on the interpreter at a few seconds per command;
`GLADE_IMPL=python` is the reference. The Python implementation stays as
the oracle for the parity test, the way the interpreter is for the
backends. What the port cost in language terms, beyond the keyword
clashes: no `Map` of mixed value kinds (the struct-kinds limit below), so
the project keeps parallel `Vec` tables; `assert` is uncatchable, so
every user-facing failure is a `raise`; and an effect performed in
method position (`perform Fs.read_bytes(p).from_bytes()`) must be bound
first.

Two limits worth knowing before writing more library code:

- A helper taking an empty `Vec` and one taking a `Vec` of strings get
  two clones where one would do, because an empty `Vec` and a `Vec` of
  ints have the same kind. Correct, just redundant.
- A struct's field kinds are per declaration, not per use, so a
  `std.map` holding strings in one place and `Version`s in another
  would conflict the `Map` struct module-wide. `std.solve` avoids it
  with its own parallel-`Vec` tables; glade proper will need either
  struct specialization by field kinds or a boxed value type.

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

All four steps are done. `docs/io_runtime.md` holds step 3's design
decisions (what a `bytes` value is and how effect operations pass
`Vec` and `bytes` across the C boundary); `docs/glade.md` describes
the command as it now runs, and the section above what each step found
in the compiler. What remains is the language work the port exposed
rather than glade work: struct specialization by field kinds (or a
boxed value type) so a `std.map` can hold different value kinds in one
program, closures passed as parameters natively (the `std.fail`
helpers), and the interpreter's speed, which is why the console script
prefers the native binary.
