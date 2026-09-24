/* metaxu_rt.h -- native runtime library for Metaxu (Vec + strings).
 *
 * This is the C shim implementing the semantics of the interpreter's
 * runtime library (src/metaxu/compiler/mir_interp.py).  The interpreter
 * remains the reference semantics; this library must match it observably.
 * The LLVM backend will, in a later increment, declare these symbols and
 * link the compiled object (metaxu_rt.o) into native binaries.
 *
 * ABI conventions
 * ---------------
 * All Metaxu runtime values are 8 bytes, matching the LLVM backend's
 * conventions:
 *   - int / bool           : int64_t
 *   - float                : double
 *   - Vec                  : mx_vec* (opaque pointer; identity semantics)
 *   - str                  : char*  (NUL-terminated, UTF-8/byte string)
 *
 * | symbol           | signature                                  | notes  |
 * |------------------|--------------------------------------------|--------|
 * | mx_vec_new       | mx_vec* (void)                             | heap-allocates an empty growable vector |
 * | mx_vec_push      | void (mx_vec*, int64_t)                    | amortized O(1) append |
 * | mx_vec_pop       | int64_t (mx_vec*)                          | aborts on empty ("pop: Vec is empty") |
 * | mx_vec_len       | int64_t (const mx_vec*)                    | current element count |
 * | mx_vec_get       | int64_t (const mx_vec*, int64_t idx)       | aborts on out-of-bounds (incl. negative) |
 * | mx_vec_set       | void (mx_vec*, int64_t idx, int64_t val)   | aborts on out-of-bounds (incl. negative) |
 * | mx_vec_free      | void (mx_vec*)                             | frees storage + header; NULL is a no-op |
 * | mx_str_concat    | char* (const char*, const char*)           | returns a fresh malloc'd string |
 * | mx_str_len       | int64_t (const char*)                      | strlen |
 * | mx_i64_to_str    | char* (int64_t)                            | fresh malloc'd decimal string |
 * | mx_f64_to_str    | char* (double)                             | fresh malloc'd string, Python str(float) format |
 * | mx_str_eq        | int64_t (const char*, const char*)         | 1 if contents equal, else 0 |
 * | mx_str_free      | void (char*)                               | frees a produced string; NULL is a no-op |
 * | mx_shift_check   | int64_t (int64_t count, int64_t is_left)   | returns count; aborts unless 0 <= count < 64 |
 * | mx_vec_as_bytes  | unsigned char* (const mx_vec*)             | fresh malloc'd byte SNAPSHOT of the elements |
 *
 * Vec semantics (mirrors MxVec in mir_interp.py)
 * ----------------------------------------------
 * A Vec is a MUTABLE heap object with IDENTITY semantics: passing the
 * mx_vec* around aliases one shared vector (unlike structs, which the
 * backend copies by value).  Elements are opaque 8-byte words; the runtime
 * never inspects them, so int64 payloads, doubles reinterpreted as i64
 * bits, and pointers all fit.
 *
 * Error philosophy
 * ----------------
 * Matching the interpreter's strict InterpError philosophy, every invalid
 * operation reports a clear one-line message and terminates.  There are no
 * error codes and no silent fallbacks.  Messages reuse the interpreter's
 * wording where one exists:
 *   - pop on empty:      "pop: Vec is empty"
 *   - out-of-bounds get/set: "index out of bounds: <idx> (length <len>)"
 *   - NULL receiver / NULL string argument / allocation failure also abort.
 *
 * Since native try/catch landed, that split matters: a contract violation
 * whose wording reproduces an InterpError byte for byte is CATCHABLE (it
 * goes through mx_raise in metaxu_effects.c, so `try { ... } catch e` binds
 * exactly this text, and with no `try` installed it prints and aborts as
 * before) -- pop on empty, index out of bounds, index assignment out of
 * bounds, slice step, vector size mismatch, zip length mismatch,
 * comprehension length, as_ptr byte range, shift count range.  Everything
 * else stays FATAL because the interpreter does not raise InterpError for
 * it either (integer division by zero is ZeroDivisionError there) or it has
 * no interpreter counterpart at all (allocation failure, NULL receivers,
 * capacity overflow, internal formatting invariants) and must never become
 * a program value.  See metaxu_effects.h for the full contract.
 *
 * Memory ownership
 * ----------------
 *   - mx_vec_new allocates; the caller owns the vector and must release it
 *     with mx_vec_free (never plain free(): the element buffer would leak).
 *   - mx_vec_free(NULL) is a no-op; freeing the same vector twice is
 *     undefined (same as free()).
 *   - mx_str_concat / mx_i64_to_str / mx_f64_to_str return fresh buffers
 *     allocated with malloc; the caller owns them and frees them with
 *     mx_str_free (or plain free()).  Inputs are never modified or
 *     retained — a concat result NEVER aliases an input, which is what
 *     lets the backend free an operand right after producing from it.
 *   - mx_str_free(NULL) is a no-op; only PRODUCED strings may be freed
 *     (never interned literal constants — the backend distinguishes
 *     provenance statically).
 *   - mx_str_len / mx_str_eq allocate nothing.
 *   - mx_vec_as_bytes (`vec.as_ptr()`) returns a fresh malloc'd BYTE
 *     SNAPSHOT of the vector's current elements (len bytes, one byte per
 *     element, NO NUL terminator), mirroring the interpreter's _ffi_as_ptr
 *     exactly: the snapshot is an independent allocation that never
 *     aliases the vector's own word buffer, so later pushes/growth cannot
 *     invalidate it and freeing the vector leaves it intact.  The caller
 *     owns the snapshot (plain free()); the LLVM backend currently leaks
 *     it by design.  An element outside 0..255 aborts (the interpreter's
 *     "as_ptr: element ... is not a byte" strictness), never truncates.
 *
 * Float formatting
 * ----------------
 * mx_f64_to_str reproduces Python's str(float): the shortest decimal
 * string that round-trips to the same double, fixed notation for decimal
 * exponents in [-4, 16), scientific ("1e+16", "1.5e-05") otherwise, a
 * trailing ".0" on integral fixed-notation values ("3.0", not "3"),
 * "-0.0" for negative zero, and "inf"/"-inf"/"nan" for non-finite values.
 * Known divergence: the implementation assumes the "C" locale's '.'
 * decimal point; a host program that calls setlocale() with a ','-based
 * locale changes printf/strtod behavior and thus the output.
 */
/* Fixed-size vectors (mx_fvec; mirrors MxVector in mir_interp.py)
 * ---------------------------------------------------------------
 * A fixed vector `vector[T, N]` is a length-prefixed heap block
 * { int64_t len; int64_t elems[len]; } of opaque 8-byte element words
 * (i64 as-is, doubles bit-cast, str/vector pointers) — IMMUTABLE after
 * construction.  The interpreter's MxVector has VALUE semantics; because
 * no operation mutates a filled block, sharing the pointer shallowly is
 * observationally identical to copying the value, exactly the write-once
 * payload-box argument in codegen_llvm.  Construction protocol: the
 * compiler calls mx_fvec_new (zeroed) and fills the block with
 * mx_fvec_init before the pointer is shared; nothing writes it afterwards.
 * Blocks are never freed (leak by design — shallow sharing makes
 * ownership non-unique; a leak is provably sound where a free is not).
 *
 * | symbol           | signature                                        |
 * |------------------|--------------------------------------------------|
 * | mx_fvec_new      | mx_fvec* (int64_t len)          zero-filled      |
 * | mx_fvec_len      | int64_t (const mx_fvec*)                         |
 * | mx_fvec_get      | int64_t (const mx_fvec*, int64_t) aborts on OOB  |
 * | mx_fvec_init     | void (mx_fvec*, int64_t, int64_t) fill-only store|
 * | mx_fvec_filled   | mx_fvec* (int64_t n, int64_t word)               |
 * | mx_fvec_range    | mx_fvec* (int64_t start, int64_t end)            |
 * | mx_fvec_dim      | int64_t (const mx_fvec*, int64_t dim,            |
 * |                  |          int64_t elems_are_vecs)                 |
 * | mx_fvec_slice    | mx_fvec* (v, start, stop, step, mask)            |
 * | mx_fvec_binop    | mx_fvec* (op, base, depth, mode, lhs, rhs)       |
 * | mx_fvec_promote  | mx_fvec* (const mx_fvec*)  flat -> Mx1 matrix    |
 * | mx_fvec_map      | mx_fvec* (v, fn, env, expected_n)  comprehension |
 * | mx_fvec_set_copy | mx_fvec* (v, idx, word)  functional update copy  |
 * | mx_fvec_zip_map  | mx_fvec* (a, b, fn2, env, expected_n)  zip compr.|
 * | mx_fvec_to_str   | char* (v, base, depth)  Python repr, fresh malloc|
 * | mx_fvec_as_bytes | unsigned char* (const mx_fvec*)  byte snapshot   |
 *
 * Conventions shared by the mx_fvec entry points:
 *   - `mask` (mx_fvec_slice): bit 0 = start given, bit 1 = stop given,
 *     bit 2 = step given; omitted parts take Python's slice defaults and
 *     the index arithmetic is exactly CPython's slice.indices() (negative
 *     indices wrap, everything clamps, step 0 aborts with the
 *     interpreter's "slice: step must be non-zero").  The result is a
 *     fresh copy, never an aliasing view (interpreter parity).
 *   - mx_fvec_binop: op 0..4 = + - * / %, `base` 0 = int64 / 1 = double
 *     (the LEAF element type), `depth` = how many nesting levels the
 *     elements are still vectors (0 = scalar elements), `mode` 0 =
 *     vec(+)vec / 1 = vec(+)scalar / 2 = scalar(+)vec (lhs/rhs are element
 *     words; vector operands pass the mx_fvec* as a word).  Element-wise
 *     with scalar broadcasting, recursing through nested vectors — the
 *     interpreter's _vec_elementwise.  Length mismatch aborts with the
 *     interpreter's "vector size mismatch for '+': N vs M"; integer ops
 *     use C truncating /,% (the backend's documented scalar convention;
 *     the interpreter floors) and integer division by zero aborts.
 *     Float % aborts (it would need libm's fmod and this object stays
 *     libm-free; the compiler demotes float vector % instead).
 *   - mx_fvec_dim mirrors __vec_dim: dim 0 -> len; dim 1 -> 0 when empty,
 *     the first element's length when elems_are_vecs, else 1 (a flat
 *     numeric vector is a column matrix).
 *   - mx_fvec_map applies fn(env, word) to each element word in order
 *     (the compiled comprehension-body closure) and aborts when
 *     expected_n >= 0 differs from the length (the interpreter's
 *     "vector comprehension produced N elements ..." check).
 *   - mx_fvec_set_copy is the FUNCTIONAL update behind `v[i] = x` on an
 *     immutable vector[T,N] place: blocks are shallow-shared and
 *     write-once, so the update COPIES the block, stores the element
 *     word, and returns the fresh block for the compiler to rebind —
 *     other shares never observe the write.  Aborts on out-of-bounds
 *     with the interpreter's "index assignment out of bounds" message.
 *   - mx_fvec_zip_map is the lockstep pair form of mx_fvec_map
 *     (`f(a, b) for (a, b) in (xs, ys)`): aborts when the two lengths
 *     differ (the interpreter's "zip: sequences have different lengths")
 *     or when expected_n >= 0 differs from the common length, then
 *     applies fn2(env, wa, wb) elementwise.
 *   - mx_fvec_to_str reproduces repr(MxVector): "vector[e0, e1, ...]"
 *     with elements rendered like mx_i64_to_str / mx_f64_to_str
 *     (base 0 / 1), recursing while depth > 0.
 *   - mx_fvec_as_bytes matches mx_vec_as_bytes (fresh snapshot, byte
 *     range checked).
 */
#ifndef METAXU_RT_H
#define METAXU_RT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mx_vec mx_vec;
typedef struct mx_fvec mx_fvec;
typedef struct mx_tile mx_tile;

/* Comprehension body: compiled closure thunk (env, element word) -> word. */
typedef int64_t (*mx_fvec_map_fn)(void *env, int64_t word);

/* Zip-comprehension body: (env, word from a, word from b) -> word. */
typedef int64_t (*mx_fvec_zip_fn)(void *env, int64_t wa, int64_t wb);

/* --- Vec ---------------------------------------------------------------- */
mx_vec *mx_vec_new(void);
void    mx_vec_push(mx_vec *v, int64_t value);
int64_t mx_vec_pop(mx_vec *v);
int64_t mx_vec_len(const mx_vec *v);
int64_t mx_vec_get(const mx_vec *v, int64_t idx);
void    mx_vec_set(mx_vec *v, int64_t idx, int64_t value);

/* Cold-path terminators for the inline Vec fast paths (codegen_llvm).
 * Generated code branches here only after the inlined checks already
 * failed; each re-runs its op's canonical check order so the diagnostic
 * stays byte-identical to calling the full op, then aborts if somehow
 * nothing fired.  They never return — which is what lets the emitted
 * declares carry `noreturn` plus a narrow memory contract, keeping the
 * hot loop's header loads hoistable around the never-taken branch. */
__attribute__((cold, noreturn)) void mx__vec_get_fail(const mx_vec *v,
                                                      int64_t idx);
__attribute__((cold, noreturn)) void mx__vec_set_fail(mx_vec *v,
                                                      int64_t idx);
__attribute__((cold, noreturn)) void mx__vec_pop_fail(mx_vec *v);
__attribute__((cold, noreturn)) void mx__vec_len_fail(const mx_vec *v);
void    mx_vec_free(mx_vec *v);
unsigned char *mx_vec_as_bytes(const mx_vec *v);
/* Contention (docs/contention_as_permission.md): mark a vector as having
 * crossed a real spawn boundary.  After this, mutating it (push/pop/set)
 * from a thread whose write permit is 0 (no mutex held through the
 * runtime, see metaxu_threads.h mx__write_permit) raises catchably with
 * the spec's exact wording; reads stay free.  void* so generated code can
 * pass the raw env/field word; NULL is a no-op. */
void    mx_vec_mark_contended(void *v);

/* --- Fixed-size vectors (immutable, write-once fill) --------------------- */
mx_fvec *mx_fvec_new(int64_t len);
int64_t  mx_fvec_len(const mx_fvec *v);
int64_t  mx_fvec_get(const mx_fvec *v, int64_t idx);
void     mx_fvec_init(mx_fvec *v, int64_t idx, int64_t word);
mx_fvec *mx_fvec_filled(int64_t len, int64_t word);
mx_fvec *mx_fvec_range(int64_t start, int64_t end);
int64_t  mx_fvec_dim(const mx_fvec *v, int64_t dim, int64_t elems_are_vecs);
mx_fvec *mx_fvec_slice(const mx_fvec *v, int64_t start, int64_t stop,
                       int64_t step, int64_t mask);
mx_fvec *mx_fvec_binop(int64_t op, int64_t base, int64_t depth, int64_t mode,
                       int64_t lhs, int64_t rhs);
mx_fvec *mx_fvec_promote(const mx_fvec *v);
mx_fvec *mx_fvec_map(const mx_fvec *v, mx_fvec_map_fn fn, void *env,
                     int64_t expected_n);
mx_fvec *mx_fvec_set_copy(const mx_fvec *v, int64_t idx, int64_t word);
mx_fvec *mx_fvec_zip_map(const mx_fvec *a, const mx_fvec *b,
                         mx_fvec_zip_fn fn, void *env, int64_t expected_n);
char    *mx_fvec_to_str(const mx_fvec *v, int64_t base, int64_t depth);

/* Tiles (docs/gpu_tiles.md Stage 0): immutable 2D row-major word blocks,
 * write-once, leak by design (like mx_fvec).  is_f64 selects the element
 * arithmetic; the emitter passes it from the static `tile:` kind.  Shape
 * and element-kind agreement is static natively; from_vec length and get
 * bounds raise catchably, wording byte-identical to the interpreter. */
mx_tile *mx_tile_zeros(int64_t rows, int64_t cols);
mx_tile *mx_tile_filled(int64_t rows, int64_t cols, int64_t word);
mx_tile *mx_tile_arange(int64_t rows, int64_t cols);
mx_tile *mx_tile_from_vec(const mx_vec *v, int64_t rows, int64_t cols);
mx_vec  *mx_tile_to_vec(const mx_tile *t);
/* ekind codes for tile arithmetic: 0 = int, 1 = f64, 2 = f32, 3 = f16
 * (narrow elements stored as the representable double; every op rounds
 * through float / _Float16 — docs/gpu_tiles.md Stage 1d/1f). */
mx_tile *mx_tile_add(const mx_tile *a, const mx_tile *b, int64_t ekind);
mx_tile *mx_tile_mul(const mx_tile *a, const mx_tile *b, int64_t ekind);
mx_tile *mx_tile_scale(const mx_tile *t, int64_t sword, int64_t ekind);
mx_tile *mx_tile_dot(const mx_tile *a, const mx_tile *b, int64_t ekind);
int64_t  mx_tile_sum(const mx_tile *t, int64_t ekind);
mx_tile *mx_tile_to_f32(const mx_tile *t, int64_t src_ekind);
mx_tile *mx_tile_to_f16(const mx_tile *t, int64_t src_ekind);
mx_tile *mx_tile_to_f64(const mx_tile *t, int64_t src_ekind);
mx_tile *mx_tile_transpose(const mx_tile *t);
int64_t  mx_tile_get(const mx_tile *t, int64_t i, int64_t j);
int64_t  mx_tile_rows(const mx_tile *t);
int64_t  mx_tile_cols(const mx_tile *t);
char    *mx_tile_to_str(const mx_tile *t, int64_t is_f64);
/* Buffer <-> tile boundary (Stage 1): strict load/store raise on any
 * out-of-range element; masked load_or reads `other` and store_clipped
 * writes nothing for out-of-range elements.  Stores take the
 * contended-write guard like every Vec mutator. */
mx_tile *mx_tile_load(const mx_vec *v, int64_t off, int64_t rows,
                      int64_t cols);
mx_tile *mx_tile_load_or(const mx_vec *v, int64_t off, int64_t rows,
                         int64_t cols, int64_t other);
void     mx_tile_store(mx_vec *v, int64_t off, const mx_tile *t);
void     mx_tile_store_clipped(mx_vec *v, int64_t off, const mx_tile *t);
/* 2D row-strided masked forms: element (i,j) <-> off + i*stride + j. */
mx_tile *mx_tile_load_rows(const mx_vec *v, int64_t off, int64_t stride,
                           int64_t rows, int64_t cols, int64_t other);
void     mx_tile_store_rows(mx_vec *v, int64_t off, int64_t stride,
                            const mx_tile *t);
unsigned char *mx_fvec_as_bytes(const mx_fvec *v);

/* --- Strings ------------------------------------------------------------ */
char   *mx_str_concat(const char *a, const char *b);
int64_t mx_str_len(const char *s);
/* `s[i]`: a FRESH one-character string; out of range raises the
   interpreter's exact diagnostic ("index out of bounds: i (length n)"). */
char   *mx_str_index(const char *s, int64_t idx);
/* `s[a:b:c]`: a FRESH copy with CPython slice.indices() semantics; `mask`
   bits 1/2/4 say which of start/stop/step were given (the same protocol
   as mx_fvec_slice).  A zero step raises "slice: step must be non-zero". */
char   *mx_str_slice(const char *s, int64_t start, int64_t stop,
                     int64_t step, int64_t mask);
char   *mx_i64_to_str(int64_t value);
char   *mx_f64_to_str(double value);
int64_t mx_str_eq(const char *a, const char *b);
void    mx_str_free(char *s);
int64_t mx_shift_check(int64_t count, int64_t is_left);

#ifdef __cplusplus
}
#endif

#endif /* METAXU_RT_H */
