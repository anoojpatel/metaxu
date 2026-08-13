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
 * operation prints a clear one-line message to stderr and calls abort().
 * There are no error codes and no silent fallbacks.  Messages reuse the
 * interpreter's wording where one exists:
 *   - pop on empty:      "pop: Vec is empty"
 *   - out-of-bounds get/set: "index out of bounds: <idx> (length <len>)"
 *   - NULL receiver / NULL string argument / allocation failure also abort.
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
#ifndef METAXU_RT_H
#define METAXU_RT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mx_vec mx_vec;

/* --- Vec ---------------------------------------------------------------- */
mx_vec *mx_vec_new(void);
void    mx_vec_push(mx_vec *v, int64_t value);
int64_t mx_vec_pop(mx_vec *v);
int64_t mx_vec_len(const mx_vec *v);
int64_t mx_vec_get(const mx_vec *v, int64_t idx);
void    mx_vec_set(mx_vec *v, int64_t idx, int64_t value);
void    mx_vec_free(mx_vec *v);

/* --- Strings ------------------------------------------------------------ */
char   *mx_str_concat(const char *a, const char *b);
int64_t mx_str_len(const char *s);
char   *mx_i64_to_str(int64_t value);
char   *mx_f64_to_str(double value);
int64_t mx_str_eq(const char *a, const char *b);
void    mx_str_free(char *s);

#ifdef __cplusplus
}
#endif

#endif /* METAXU_RT_H */
