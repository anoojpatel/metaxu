# Metaxu native runtime (C shim)

A small, dependency-free C11 library implementing the semantics of the
interpreter's runtime library (`src/metaxu/compiler/mir_interp.py`:
`MxVec`, `_builtin_push` / `_builtin_pop` / `_builtin_len` /
`__index_get`, plus string helpers). **The interpreter remains the
reference semantics**; this library must match it observably.

Planned wiring (a later increment, owned elsewhere): the LLVM backend
(`codegen_llvm.py`) declares these symbols as external functions and links
`metaxu_rt.o` (or `libmetaxu_rt.a`) into native binaries, replacing its
current inline lowering of runtime builtins.

## Files

- `metaxu_rt.h` / `metaxu_rt.c` — the runtime (no dependencies beyond libc/libm)
- `build.py` — build recipe: `compile_runtime()` returns the cached `.o`,
  `build_archive()` the `.a`; also runnable as a script
- Tests: `src/metaxu/compiler/tests/test_native_runtime.py`

## ABI

All Metaxu runtime values are 8 bytes, matching the LLVM backend's
conventions: `int`/`bool` are `int64_t`, `float` is `double`, `Vec` is an
opaque `mx_vec*`, `str` is a NUL-terminated `char*`.

| Symbol          | Signature                                | Behavior |
|-----------------|------------------------------------------|----------|
| `mx_vec_new`    | `mx_vec* (void)`                         | heap-allocate an empty growable vector |
| `mx_vec_push`   | `void (mx_vec*, int64_t)`                | amortized O(1) append (cap 8, then ×2) |
| `mx_vec_pop`    | `int64_t (mx_vec*)`                      | remove+return last; **aborts** on empty (`pop: Vec is empty`) |
| `mx_vec_len`    | `int64_t (const mx_vec*)`                | element count |
| `mx_vec_get`    | `int64_t (const mx_vec*, int64_t)`       | bounds-checked read; **aborts** on OOB/negative index |
| `mx_vec_set`    | `void (mx_vec*, int64_t, int64_t)`       | bounds-checked write; **aborts** on OOB/negative index |
| `mx_vec_free`   | `void (mx_vec*)`                         | free buffer + header; `NULL` is a no-op |
| `mx_str_concat` | `char* (const char*, const char*)`       | fresh malloc'd concatenation |
| `mx_str_len`    | `int64_t (const char*)`                  | `strlen` |
| `mx_i64_to_str` | `char* (int64_t)`                        | fresh malloc'd decimal string |
| `mx_f64_to_str` | `char* (double)`                         | fresh malloc'd string in Python `str(float)` format |
| `mx_str_eq`     | `int64_t (const char*, const char*)`     | 1 if contents equal, else 0 |

Vec elements are opaque 8-byte words — the runtime never inspects them, so
int64 payloads, doubles reinterpreted as i64 bits, and pointers all fit.
A Vec has **identity semantics** (like the interpreter's `MxVec`, unlike
value-semantics structs): every `mx_vec*` aliases the one shared vector.

## Error philosophy

Matching the interpreter's strict `InterpError` philosophy: every invalid
operation prints one clear line to stderr, prefixed `metaxu runtime
error: `, and calls `abort()`. No error codes, no silent fallbacks.
Message wording reuses the interpreter's where one exists:

- pop on empty Vec: `pop: Vec is empty`
- out-of-bounds get/set: `index out of bounds: <idx> (length <len>)`
- `NULL` receivers/arguments and allocation failures also abort.

## Memory ownership

- `mx_vec_new` allocates; the caller owns the vector and must release it
  with `mx_vec_free` (never plain `free()` — the element buffer would
  leak). Double-free is undefined, as with `free()`.
- `mx_str_concat`, `mx_i64_to_str`, `mx_f64_to_str` return fresh
  `malloc`'d buffers owned by the caller (`free()` them). Inputs are never
  modified or retained.
- `mx_str_len`, `mx_str_eq`, and the Vec accessors allocate nothing.

## Float formatting

`mx_f64_to_str` reproduces Python's `str(float)`: shortest round-tripping
decimal digits, fixed notation for decimal exponents in `[-4, 16)`,
scientific otherwise (`1e+16`, `1.5e-05`), trailing `.0` on integral
fixed-notation values (`3.0`, not `3`), `-0.0`, and `inf`/`-inf`/`nan`.

Known divergence: the implementation relies on `printf`/`strtod` and
assumes the `"C"` locale's `.` decimal point; a host program that calls
`setlocale()` with a `,`-based locale changes the output. Metaxu-generated
binaries never call `setlocale`, so this only affects foreign embedders.

## Building

```bash
python src/metaxu/runtime/native/build.py            # -> _build/metaxu_rt.o
python src/metaxu/runtime/native/build.py --archive  # -> _build/libmetaxu_rt.a
```

Artifacts are cached by mtime and rebuilt when `metaxu_rt.c`,
`metaxu_rt.h`, or `build.py` changes. A failed compile raises with clang's
stderr.
