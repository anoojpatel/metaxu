# IO runtime: files, processes, the environment and stderr as effects

This is the contract for the four IO effects the standard library
declares with `with EFFECT_*` runtime mappings, the way `Thread` and
`Mutex` are (`docs/threads_runtime.md`):

```
std.fs       effect Fs      read_bytes, write_bytes, exists, is_dir, is_file,
                            list_dir, mkdir_all, remove_all, rename
std.process  effect Process run, status, stdout, stderr
std.env      effect Env     args, get_raw, has, home, cwd
std.io       effect Io      write_err
```

Both engines implement every operation: the MIR interpreter on Python's
`os`, `shutil` and `subprocess` (`mir_interp._rt_fs_*` and friends), the
LLVM backend on POSIX calls in `src/metaxu/runtime/native/metaxu_io.c`.
The interpreter is the semantics reference; `tests/test_std_io.py` runs
the same programs through both and compares output, exit code and error
text.

## Why effects

Making these effects rather than plain externs is the point of the
language. A handler in scope on the performing thread intercepts the
operation before the runtime mapping is consulted (dispatch order is the
threads runtime's: innermost handler, then the `with SYMBOL` mapping,
then a declared default, then a loud error). A test therefore installs
a handler that serves an in-memory filesystem, or a fake `git`, and the
whole program runs under it with no mocking library. The package
manager's tests will do exactly that.

One consequence of the delimitation semantics (`docs/try_catch.md`) is
worth stating: a failure raised inside a handler ARM propagates from
the `handle` expression, not into a `try` that sits inside the handled
body, because the arm runs in the handler's frame. A virtual
filesystem therefore answers "missing" through `exists` (or resumes a
Result-shaped value) instead of raising the real runtime's message.

## What crosses the boundary

Every operation argument and result is one 8-byte word, the same
encoding the effects runtime already uses at every perform:

| Metaxu value | word | notes |
|---|---|---|
| `int`, `bool` | the integer | handles are ints |
| `string` | `char*` to NUL-terminated UTF-8 | a string never holds NUL, so nothing is lost; `from_bytes` rejects a zero byte for this reason |
| bytes | `mx_vec*` of ints in 0..255 | the representation `to_bytes`/`from_bytes` use; eight bytes per byte, fine for manifests and sources, not for large binaries |
| `Vec` of strings | `mx_vec*` whose words are `char*` | `list_dir`, `args`, `run`'s argv |

Nothing here is retained by the runtime: every returned string or Vec is
a fresh allocation the program owns (and, like every other produced
string natively, leaks by design). Files are read whole and written
whole; there are no open handles to leak or to share across threads.

## Errors

A failure is a CATCHABLE runtime error with a fixed shape, so a `try`
around the operation binds a message that is byte-identical on both
engines:

```
<op>: <path>: <strerror>          read: mx.toml: No such file or directory
run: <argv0 or cwd>: <strerror>   run: nosuch-binary: No such file or directory
run: empty argv
write: element 3 is not a byte (0..255): 300
```

`<strerror>` is the C library's text for the errno (`strerror`), which is
also what Python puts in `OSError.strerror`, so the two engines agree
without a translation table. The interpreter's argument TYPE errors
(`EFFECT_FS_READ expects a string path, got 'Int'`) have no native
counterpart: the backend rejects such programs at compile time, the way
it does for every builtin.

## Process semantics

`run(argv, cwd)` starts the program named by `argv[0]` (searched on
`PATH`, like `execvp` and `subprocess.run`), with `cwd` as its working
directory when non-empty, waits for it, and answers a handle. `status`,
`stdout` and `stderr` read the handle. The exit status is the child's
exit code, or minus the signal number when a signal ended it (Python's
`returncode` convention). Both streams are captured in full, without
deadlock (the native runtime drains them with `poll`). Output is
expected to be UTF-8 text: the native side keeps the raw bytes, the
interpreter decodes with replacement characters, so only that case can
differ. There is no stdin plumbing and no environment override: glade
needs neither.

Handles are immortal records, never freed, so a stale handle is a range
check (`status: no such process handle 7`), never a use after free.

## Arguments to the program

`std.env.args()` answers the arguments after the program name. Natively
the entry wrapper that `llvm_run` appends is `main(argc, argv)` and hands
both to `mx_io_set_args` before the module initializer runs; on the
interpreter `metaxuc run file.mx -- a b` passes `a b`, and the embedding
API sets `MirInterpreter.program_args`. `main` keeps its
`fn main() -> int` signature on both engines.

## What is deliberately not here

No stdin, no time, no networking (git is the network), no file handles
or streaming, no environment mutation, no Windows paths. Each of these
would be a new operation on one of the four effects with the same
one-word-per-value contract; none is needed to write the package
manager.
