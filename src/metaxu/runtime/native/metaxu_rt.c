/* metaxu_rt.c -- native runtime library for Metaxu (Vec + strings).
 *
 * Dependency-free C11.  See metaxu_rt.h for the ABI contract; the
 * interpreter (src/metaxu/compiler/mir_interp.py) is the reference for
 * every observable behavior here, including error message wording.
 */
#include "metaxu_rt.h"
#include "metaxu_effects.h"   /* mx_raisef: catchable failures (try/catch) */

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------
 * Error paths: strict and loud, in two flavors.
 *
 *   mx_rt_fail  -- FATAL.  Message on stderr, then abort().  Used for
 *                  failures the INTERPRETER does not raise InterpError for
 *                  (so a `try` must not catch them either) and for
 *                  allocation / internal-invariant failures, which have no
 *                  interpreter counterpart at all and must never become a
 *                  program value.
 *   mx_rt_raise -- CATCHABLE.  Routed through mx_raise (metaxu_effects.c):
 *                  an enclosing `try` binds this exact text to its catch
 *                  parameter, and with no try installed the behavior is
 *                  identical to mx_rt_fail (same stderr line, abort()).
 *                  Used ONLY where the wording already reproduces the
 *                  interpreter's InterpError message byte for byte -- the
 *                  caught value is language-visible (docs/try_catch.md).
 * ---------------------------------------------------------------------- */
static void mx_rt_fail(const char *fmt, ...) {
    va_list ap;
    fflush(stdout);   /* abort() does not: keep the printed prefix visible */
    fputs("metaxu runtime error: ", stderr);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    fflush(stderr);
    abort();
}

#define mx_rt_raise(...) mx_raisef(__VA_ARGS__)

static void *mx_rt_malloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (p == NULL) {
        mx_rt_fail("out of memory (requested %zu bytes)", n);
    }
    return p;
}

/* ------------------------------------------------------------------------
 * Vec: growable mutable vector of 8-byte words, identity semantics.
 * ---------------------------------------------------------------------- */
struct mx_vec {
    int64_t  len;   /* current element count */
    int64_t  cap;   /* allocated slots */
    int64_t *data;  /* element buffer (NULL iff cap == 0) */
};

#define MX_VEC_INITIAL_CAP 8

static void mx_vec_check(const mx_vec *v, const char *op) {
    if (v == NULL) {
        mx_rt_fail("%s: expected a Vec receiver, got NULL", op);
    }
}

mx_vec *mx_vec_new(void) {
    mx_vec *v = (mx_vec *)mx_rt_malloc(sizeof(mx_vec));
    v->len = 0;
    v->cap = 0;
    v->data = NULL;
    return v;
}

void mx_vec_push(mx_vec *v, int64_t value) {
    mx_vec_check(v, "push");
    if (v->len == v->cap) {
        int64_t new_cap = v->cap == 0 ? MX_VEC_INITIAL_CAP : v->cap * 2;
        if (new_cap <= v->cap ||
            (uint64_t)new_cap > SIZE_MAX / sizeof(int64_t)) {
            mx_rt_fail("push: Vec capacity overflow (len %lld)",
                       (long long)v->len);
        }
        int64_t *data =
            (int64_t *)realloc(v->data, (size_t)new_cap * sizeof(int64_t));
        if (data == NULL) {
            mx_rt_fail("out of memory (requested %zu bytes)",
                       (size_t)new_cap * sizeof(int64_t));
        }
        v->data = data;
        v->cap = new_cap;
    }
    v->data[v->len++] = value;
}

int64_t mx_vec_pop(mx_vec *v) {
    mx_vec_check(v, "pop");
    if (v->len == 0) {
        mx_rt_raise("pop: Vec is empty");   /* catchable (InterpError) */
    }
    return v->data[--v->len];
}

int64_t mx_vec_len(const mx_vec *v) {
    mx_vec_check(v, "len");
    return v->len;
}

int64_t mx_vec_get(const mx_vec *v, int64_t idx) {
    mx_vec_check(v, "index");
    if (idx < 0 || idx >= v->len) {
        mx_rt_raise("index out of bounds: %lld (length %lld)",
                    (long long)idx, (long long)v->len);
    }
    return v->data[idx];
}

void mx_vec_set(mx_vec *v, int64_t idx, int64_t value) {
    mx_vec_check(v, "index");
    if (idx < 0 || idx >= v->len) {
        mx_rt_raise("index out of bounds: %lld (length %lld)",
                    (long long)idx, (long long)v->len);
    }
    v->data[idx] = value;
}

void mx_vec_free(mx_vec *v) {
    if (v == NULL) {
        return;
    }
    free(v->data);
    free(v);
}

/* `vec.as_ptr()`: a fresh byte SNAPSHOT of the elements (interpreter parity
 * with _ffi_as_ptr — an independent allocation, never a view into the
 * vector's word buffer, so growth/free of the vector cannot invalidate it).
 * len bytes, no NUL terminator.  Elements outside 0..255 abort, exactly the
 * interpreter's "as_ptr: element i is not a byte (0..255)" strictness. */
unsigned char *mx_vec_as_bytes(const mx_vec *v) {
    mx_vec_check(v, "as_ptr");
    unsigned char *out = (unsigned char *)mx_rt_malloc((size_t)v->len);
    for (int64_t i = 0; i < v->len; i++) {
        int64_t e = v->data[i];
        if (e < 0 || e > 255) {
            mx_rt_raise("as_ptr: element %lld is not a byte (0..255): %lld",
                        (long long)i, (long long)e);
        }
        out[i] = (unsigned char)e;
    }
    return out;
}

/* ------------------------------------------------------------------------
 * Fixed-size vectors: immutable length-prefixed word blocks (MxVector).
 * See metaxu_rt.h for the full semantics contract; the interpreter
 * (mir_interp.py MxVector + _vec_elementwise + _builtin_slice_get + ...)
 * is the reference, including error message wording.
 * ---------------------------------------------------------------------- */
struct mx_fvec {
    int64_t len;
    int64_t elems[];  /* element words */
};

static void mx_fvec_check(const mx_fvec *v, const char *op) {
    if (v == NULL) {
        mx_rt_fail("%s: expected a vector receiver, got NULL", op);
    }
}

mx_fvec *mx_fvec_new(int64_t len) {
    if (len < 0) {
        mx_rt_fail("vector: negative length %lld", (long long)len);
    }
    if ((uint64_t)len > (SIZE_MAX - sizeof(mx_fvec)) / sizeof(int64_t)) {
        mx_rt_fail("vector: length overflow (%lld)", (long long)len);
    }
    size_t bytes = sizeof(mx_fvec) + (size_t)len * sizeof(int64_t);
    mx_fvec *v = (mx_fvec *)calloc(1, bytes ? bytes : 1);
    if (v == NULL) {
        mx_rt_fail("out of memory (requested %zu bytes)", bytes);
    }
    v->len = len;
    return v;
}

int64_t mx_fvec_len(const mx_fvec *v) {
    mx_fvec_check(v, "len");
    return v->len;
}

int64_t mx_fvec_get(const mx_fvec *v, int64_t idx) {
    mx_fvec_check(v, "index");
    if (idx < 0 || idx >= v->len) {
        mx_rt_raise("index out of bounds: %lld (length %lld)",
                    (long long)idx, (long long)v->len);
    }
    return v->elems[idx];
}

void mx_fvec_init(mx_fvec *v, int64_t idx, int64_t word) {
    mx_fvec_check(v, "vector init");
    if (idx < 0 || idx >= v->len) {
        mx_rt_raise("index out of bounds: %lld (length %lld)",
                    (long long)idx, (long long)v->len);
    }
    v->elems[idx] = word;
}

mx_fvec *mx_fvec_filled(int64_t len, int64_t word) {
    mx_fvec *v = mx_fvec_new(len);
    for (int64_t i = 0; i < len; i++) {
        v->elems[i] = word;
    }
    return v;
}

mx_fvec *mx_fvec_range(int64_t start, int64_t end) {
    int64_t n = end > start ? end - start : 0;
    mx_fvec *v = mx_fvec_new(n);
    for (int64_t i = 0; i < n; i++) {
        v->elems[i] = start + i;
    }
    return v;
}

/* __vec_dim: dim 0 -> length; dim 1 -> 0 when empty, the first element's
 * length when elements are vectors, else 1 (flat vector == column matrix).
 * The compiler passes elems_are_vecs from the static element kind and
 * demotes non-numeric non-vector element kinds, mirroring the
 * interpreter's per-value checks exactly for every accepted program. */
int64_t mx_fvec_dim(const mx_fvec *v, int64_t dim, int64_t elems_are_vecs) {
    mx_fvec_check(v, "__vec_dim");
    if (dim == 0) {
        return v->len;
    }
    if (dim != 1) {
        mx_rt_fail("__vec_dim: unsupported dimension %lld", (long long)dim);
    }
    if (v->len == 0) {
        return 0;
    }
    if (elems_are_vecs) {
        const mx_fvec *first = (const mx_fvec *)(intptr_t)v->elems[0];
        mx_fvec_check(first, "__vec_dim");
        return first->len;
    }
    return 1;
}

/* CPython slice.indices() over [0, len): negative wraps, clamps, and the
 * omitted-part defaults depend on the step sign.  `mask` bits: 1 start
 * given, 2 stop given, 4 step given. */
mx_fvec *mx_fvec_slice(const mx_fvec *v, int64_t start, int64_t stop,
                       int64_t step, int64_t mask) {
    mx_fvec_check(v, "slice");
    int64_t len = v->len;
    if (!(mask & 4)) {
        step = 1;
    }
    if (step == 0) {
        mx_rt_raise("slice: step must be non-zero");  /* catchable */
    }
    int64_t lo_clamp = step < 0 ? -1 : 0;
    int64_t hi_clamp = step < 0 ? len - 1 : len;
    if (mask & 1) {
        if (start < 0) {
            start += len;
            if (start < 0) {
                start = lo_clamp;
            }
        } else if (start >= len) {
            start = hi_clamp;
        }
    } else {
        start = step < 0 ? len - 1 : 0;
    }
    if (mask & 2) {
        if (stop < 0) {
            stop += len;
            if (stop < 0) {
                stop = lo_clamp;
            }
        } else if (stop >= len) {
            stop = hi_clamp;
        }
    } else {
        stop = step < 0 ? -1 : len;  /* -1 here means "past the front" */
    }
    int64_t count;
    if (step > 0) {
        count = stop > start ? (stop - start + step - 1) / step : 0;
    } else {
        count = start > stop ? (start - stop + (-step) - 1) / (-step) : 0;
    }
    mx_fvec *out = mx_fvec_new(count);
    int64_t idx = start;
    for (int64_t i = 0; i < count; i++, idx += step) {
        out->elems[i] = v->elems[idx];
    }
    return out;
}

static const char *mx_fvec_op_name(int64_t op) {
    switch (op) {
    case 0: return "+";
    case 1: return "-";
    case 2: return "*";
    case 3: return "/";
    case 4: return "%";
    default:
        mx_rt_fail("vector binop: unknown opcode %lld", (long long)op);
        return "?";  /* unreachable */
    }
}

static int64_t mx_fvec_scalar_op(int64_t op, int64_t base, int64_t a,
                                 int64_t b) {
    if (base == 1) {
        double x, y, r;
        memcpy(&x, &a, sizeof x);
        memcpy(&y, &b, sizeof y);
        switch (op) {
        case 0: r = x + y; break;
        case 1: r = x - y; break;
        case 2: r = x * y; break;
        case 3: r = x / y; break;  /* IEEE; the interpreter REJECTS /0 */
        default:
            /* Float %% would need libm's fmod; the compiler demotes float
             * vector %% instead (this runtime object stays libm-free). */
            mx_rt_fail("vector binop: float %% has no native lowering");
            r = 0.0;  /* unreachable */
            break;
        }
        int64_t out;
        memcpy(&out, &r, sizeof out);
        return out;
    }
    switch (op) {
    case 0: return a + b;
    case 1: return a - b;
    case 2: return a * b;
    default:
        if (b == 0) {
            /* FATAL, deliberately: the interpreter raises ZeroDivisionError,
             * NOT InterpError, so a `try` does not recover from it there
             * either (scalar native sdiv is plain UB; the vector runtime
             * aborts loudly instead of guessing). */
            mx_rt_fail("vector binop: integer division by zero");
        }
        /* C truncating semantics, the backend's documented sdiv/srem
         * convention (the interpreter floors; they agree for non-negative
         * operands). */
        return op == 3 ? a / b : a % b;
    }
}

/* Element-wise arithmetic with scalar broadcasting, recursing through
 * nested vectors — the interpreter's _vec_elementwise.  See the header
 * for op/base/depth/mode. */
mx_fvec *mx_fvec_binop(int64_t op, int64_t base, int64_t depth, int64_t mode,
                       int64_t lhs, int64_t rhs) {
    const mx_fvec *lv = NULL;
    const mx_fvec *rv = NULL;
    int64_t n;
    if (mode == 0) {
        lv = (const mx_fvec *)(intptr_t)lhs;
        rv = (const mx_fvec *)(intptr_t)rhs;
        mx_fvec_check(lv, "vector binop");
        mx_fvec_check(rv, "vector binop");
        if (lv->len != rv->len) {
            mx_rt_raise("vector size mismatch for '%s': %lld vs %lld",
                        mx_fvec_op_name(op), (long long)lv->len,
                        (long long)rv->len);
        }
        n = lv->len;
    } else if (mode == 1) {
        lv = (const mx_fvec *)(intptr_t)lhs;
        mx_fvec_check(lv, "vector binop");
        n = lv->len;
    } else {
        rv = (const mx_fvec *)(intptr_t)rhs;
        mx_fvec_check(rv, "vector binop");
        n = rv->len;
    }
    mx_fvec *out = mx_fvec_new(n);
    for (int64_t i = 0; i < n; i++) {
        int64_t a = lv != NULL ? lv->elems[i] : lhs;
        int64_t b = rv != NULL ? rv->elems[i] : rhs;
        if (depth > 0) {
            /* Elements are vectors: recurse; a scalar operand keeps
             * broadcasting downward (the interpreter's recursion through
             * _eval_binop). */
            out->elems[i] = (int64_t)(intptr_t)mx_fvec_binop(
                op, base, depth - 1, mode, a, b);
        } else {
            out->elems[i] = mx_fvec_scalar_op(op, base, a, b);
        }
    }
    return out;
}

/* promote_matrix: wrap a flat vector's elements as one-element rows so a
 * vector passed where a matrix is expected indexes as an Mx1 column. */
mx_fvec *mx_fvec_promote(const mx_fvec *v) {
    mx_fvec_check(v, "promote_matrix");
    mx_fvec *out = mx_fvec_new(v->len);
    for (int64_t i = 0; i < v->len; i++) {
        mx_fvec *cell = mx_fvec_new(1);
        cell->elems[0] = v->elems[i];
        out->elems[i] = (int64_t)(intptr_t)cell;
    }
    return out;
}

mx_fvec *mx_fvec_map(const mx_fvec *v, mx_fvec_map_fn fn, void *env,
                     int64_t expected_n) {
    mx_fvec_check(v, "vector comprehension");
    if (fn == NULL) {
        mx_rt_fail("vector comprehension: NULL body function");
    }
    if (expected_n >= 0 && v->len != expected_n) {
        mx_rt_raise("vector comprehension produced %lld elements for a "
                    "vector of size %lld",
                    (long long)v->len, (long long)expected_n);
    }
    mx_fvec *out = mx_fvec_new(v->len);
    for (int64_t i = 0; i < v->len; i++) {
        out->elems[i] = fn(env, v->elems[i]);
    }
    return out;
}

/* Functional update behind `v[i] = x` on an immutable vector[T,N] place:
 * blocks are shallow-shared and write-once, so the update must COPY the
 * block, store the element, and hand the fresh block back for the
 * compiler to rebind — a mutation in place would be observed by every
 * other share (exactly the silent bug the interpreter's value semantics
 * forbid). */
mx_fvec *mx_fvec_set_copy(const mx_fvec *v, int64_t idx, int64_t word) {
    mx_fvec_check(v, "index assignment");
    if (idx < 0 || idx >= v->len) {
        mx_rt_raise("index assignment out of bounds: %lld (length %lld)",
                    (long long)idx, (long long)v->len);
    }
    mx_fvec *out = mx_fvec_new(v->len);
    memcpy(out->elems, v->elems, (size_t)v->len * sizeof(int64_t));
    out->elems[idx] = word;
    return out;
}

/* Lockstep pair comprehension: `f(a, b) for (a, b) in (xs, ys)`.  The
 * interpreter's __zip is strict about lengths, so a mismatch aborts
 * loudly; expected_n mirrors mx_fvec_map's declared-size check. */
mx_fvec *mx_fvec_zip_map(const mx_fvec *a, const mx_fvec *b,
                         mx_fvec_zip_fn fn, void *env, int64_t expected_n) {
    mx_fvec_check(a, "zip");
    mx_fvec_check(b, "zip");
    if (fn == NULL) {
        mx_rt_fail("zip comprehension: NULL body function");
    }
    if (a->len != b->len) {
        int64_t lo = a->len < b->len ? a->len : b->len;
        int64_t hi = a->len < b->len ? b->len : a->len;
        mx_rt_raise("zip: sequences have different lengths [%lld, %lld]",
                    (long long)lo, (long long)hi);
    }
    if (expected_n >= 0 && a->len != expected_n) {
        mx_rt_raise("vector comprehension produced %lld elements for a "
                    "vector of size %lld",
                    (long long)a->len, (long long)expected_n);
    }
    mx_fvec *out = mx_fvec_new(a->len);
    for (int64_t i = 0; i < a->len; i++) {
        out->elems[i] = fn(env, a->elems[i], b->elems[i]);
    }
    return out;
}

/* Append helper for mx_fvec_to_str: exact-size accounting is not worth the
 * complexity; grow a buffer geometrically. */
static void mx_buf_append(char **buf, size_t *len, size_t *cap,
                          const char *piece) {
    size_t pl = strlen(piece);
    if (*len + pl + 1 > *cap) {
        size_t ncap = *cap ? *cap * 2 : 64;
        while (ncap < *len + pl + 1) {
            ncap *= 2;
        }
        char *nb = (char *)realloc(*buf, ncap);
        if (nb == NULL) {
            mx_rt_fail("out of memory (requested %zu bytes)", ncap);
        }
        *buf = nb;
        *cap = ncap;
    }
    memcpy(*buf + *len, piece, pl + 1);
    *len += pl;
}

/* repr(MxVector): "vector[e0, e1, ...]" with Python-repr elements. */
char *mx_fvec_to_str(const mx_fvec *v, int64_t base, int64_t depth) {
    mx_fvec_check(v, "to_string");
    char *buf = NULL;
    size_t len = 0, cap = 0;
    mx_buf_append(&buf, &len, &cap, "vector[");
    for (int64_t i = 0; i < v->len; i++) {
        if (i > 0) {
            mx_buf_append(&buf, &len, &cap, ", ");
        }
        char *piece;
        if (depth > 0) {
            piece = mx_fvec_to_str(
                (const mx_fvec *)(intptr_t)v->elems[i], base, depth - 1);
        } else if (base == 1) {
            double x;
            memcpy(&x, &v->elems[i], sizeof x);
            piece = mx_f64_to_str(x);
        } else {
            piece = mx_i64_to_str(v->elems[i]);
        }
        mx_buf_append(&buf, &len, &cap, piece);
        free(piece);
    }
    mx_buf_append(&buf, &len, &cap, "]");
    return buf;
}

/* `vector.as_ptr()`: fresh byte snapshot, mirroring mx_vec_as_bytes. */
unsigned char *mx_fvec_as_bytes(const mx_fvec *v) {
    mx_fvec_check(v, "as_ptr");
    unsigned char *out = (unsigned char *)mx_rt_malloc((size_t)v->len);
    for (int64_t i = 0; i < v->len; i++) {
        int64_t e = v->elems[i];
        if (e < 0 || e > 255) {
            mx_rt_raise("as_ptr: element %lld is not a byte (0..255): %lld",
                        (long long)i, (long long)e);
        }
        out[i] = (unsigned char)e;
    }
    return out;
}

/* ------------------------------------------------------------------------
 * Strings: NUL-terminated byte strings; results are fresh malloc'd buffers.
 * ---------------------------------------------------------------------- */
static void mx_str_check(const char *s, const char *op, const char *which) {
    if (s == NULL) {
        mx_rt_fail("%s: expected a str %s, got NULL", op, which);
    }
}

char *mx_str_concat(const char *a, const char *b) {
    mx_str_check(a, "concat", "operand");
    mx_str_check(b, "concat", "operand");
    size_t la = strlen(a);
    size_t lb = strlen(b);
    char *out = (char *)mx_rt_malloc(la + lb + 1);
    memcpy(out, a, la);
    memcpy(out + la, b, lb + 1); /* +1 copies b's NUL */
    return out;
}

int64_t mx_str_len(const char *s) {
    mx_str_check(s, "len", "receiver");
    return (int64_t)strlen(s);
}

int64_t mx_str_eq(const char *a, const char *b) {
    mx_str_check(a, "str_eq", "operand");
    mx_str_check(b, "str_eq", "operand");
    return strcmp(a, b) == 0 ? 1 : 0;
}

/* ------------------------------------------------------------------------
 * Bitwise shifts: validate the COUNT, then let the caller emit shl/ashr.
 *
 * LLVM makes `shl`/`ashr` POISON when the count is negative or >= the bit
 * width, while mir_interp raises a loud InterpError there -- the same
 * program with two behaviours and no diagnostic, which is exactly what the
 * differential tests exist to prevent.  Returning the count (rather than
 * the shifted value) keeps the shift itself a single native instruction.
 * ---------------------------------------------------------------------- */
int64_t mx_shift_check(int64_t count, int64_t is_left) {
    if (count < 0 || count >= 64) {
        mx_rt_raise("shift amount %lld out of range for '%s' on a 64-bit int "
                    "(must be 0..63)",
                    (long long)count, is_left ? "<<" : ">>");
    }
    return count;
}

void mx_str_free(char *s) {
    if (s != NULL) {
        free(s);
    }
}

static char *mx_strdup_fresh(const char *s) {
    size_t n = strlen(s) + 1;
    char *out = (char *)mx_rt_malloc(n);
    memcpy(out, s, n);
    return out;
}

char *mx_i64_to_str(int64_t value) {
    char buf[32]; /* INT64_MIN is 20 chars + NUL */
    snprintf(buf, sizeof buf, "%lld", (long long)value);
    return mx_strdup_fresh(buf);
}

/* Python-str(float) formatting: shortest round-tripping decimal digits,
 * fixed notation for decimal exponent in [-4, 16), scientific otherwise,
 * and a trailing ".0" on integral fixed-notation output.  See the header
 * for the documented divergence (non-"C" locales). */
char *mx_f64_to_str(double value) {
    char buf[64];
    char out[64];

    if (isnan(value)) {
        return mx_strdup_fresh("nan");
    }
    if (isinf(value)) {
        return mx_strdup_fresh(value < 0 ? "-inf" : "inf");
    }
    if (value == 0.0) {
        return mx_strdup_fresh(signbit(value) ? "-0.0" : "0.0");
    }

    /* 1. Find the minimal number of significant digits that round-trips.
     *    %e keeps the digit count independent of the notation choice. */
    int prec = 17;
    for (int p = 1; p <= 17; p++) {
        snprintf(buf, sizeof buf, "%.*e", p - 1, value);
        double back = strtod(buf, NULL);
        if (memcmp(&back, &value, sizeof value) == 0) {
            prec = p;
            break;
        }
    }
    snprintf(buf, sizeof buf, "%.*e", prec - 1, value);

    /* 2. Decimal exponent from the %e output ("d.ddde[+-]XX"). */
    const char *epos = strchr(buf, 'e');
    if (epos == NULL) {
        mx_rt_fail("to_string: internal float formatting error for %%e output"
                   " %s", buf);
    }
    long exp10 = strtol(epos + 1, NULL, 10);

    /* 3. Choose notation the way Python's float repr does. */
    if (exp10 >= -4 && exp10 < 16) {
        int decimals = prec - 1 - (int)exp10;
        if (decimals < 0) {
            decimals = 0;
        }
        snprintf(out, sizeof out, "%.*f", decimals, value);
        if (strchr(out, '.') == NULL) {
            strcat(out, ".0");
        }
    } else {
        /* %e already yields Python's scientific form: at least two exponent
         * digits, no trailing ".0" on a bare mantissa ("1e+16"). */
        snprintf(out, sizeof out, "%s", buf);
    }

    /* Internal consistency: the printed string must parse back exactly. */
    {
        double back = strtod(out, NULL);
        if (memcmp(&back, &value, sizeof value) != 0) {
            mx_rt_fail("to_string: internal float formatting error"
                       " (%s does not round-trip)", out);
        }
    }
    return mx_strdup_fresh(out);
}
