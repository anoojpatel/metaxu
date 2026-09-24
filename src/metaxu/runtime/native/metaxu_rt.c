/* metaxu_rt.c -- native runtime library for Metaxu (Vec + strings).
 *
 * Dependency-free C11.  See metaxu_rt.h for the ABI contract; the
 * interpreter (src/metaxu/compiler/mir_interp.py) is the reference for
 * every observable behavior here, including error message wording.
 */
#include "metaxu_rt.h"
#include "metaxu_effects.h"   /* mx_raisef: catchable failures (try/catch) */
#include "metaxu_threads.h"   /* mx__write_permit: contention permission */

#include <math.h>
#include <stdarg.h>
#include <stdatomic.h>
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
_Noreturn static void mx_rt_fail(const char *fmt, ...) {
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
    /* Contention flag (docs/contention_as_permission.md): set once when
     * this vector crosses a real spawn boundary; checked by every MUTATOR
     * (push/pop/set) against the per-thread write permit.  Atomic with
     * relaxed ordering so a re-mark at a second spawn cannot race a
     * concurrent flag read in another thread's mutator (the initial mark
     * happens-before the child via pthread_create); relaxed loads/stores
     * compile to plain moves on x86-64, so the uncrossed fast path pays
     * exactly one flag test on the header cache line.
     *
     * Layout note: this grows the header from 24 to 32 bytes, which moves
     * it from glibc malloc's 32-byte chunk into the 48-byte one.  Measured
     * cost: see the spec's "Measured" section (per-vector, one-time). */
    _Atomic int64_t contended;
};

#define MX_VEC_INITIAL_CAP 8

static void mx_vec_check(const mx_vec *v, const char *op) {
    if (v == NULL) {
        mx_rt_fail("%s: expected a Vec receiver, got NULL", op);
    }
}

/* The contended-write guard, shared by every Vec mutator.  Order of tests:
 * the uncontended path (the overwhelming common case) pays ONLY the flag
 * test -- the thread-local permit read (a direct TLS load: the exported
 * mx__tls_write_permit variable, NOT the accessor call, which measurably
 * bloated the mutators' fast path) happens after the unlikely branch, and
 * the raise itself is outlined cold so the mutators carry just a
 * compare-and-jump.  Wording is byte-identical to the interpreter's
 * InterpError (mir_interp._CONTENDED_WRITE_MSG); the caught value is
 * language-visible. */
__attribute__((cold, noinline))
static void mx__vec_contended_raise(void) {
    mx_rt_raise(
        "write to contended Vec without a held lock: this value crossed "
        "a thread boundary at spawn; mutate it under a mutex "
        "(std.sync.with_lock) or keep it thread-local");
}

static inline void mx__vec_write_check(mx_vec *v) {
    if (__builtin_expect(
            atomic_load_explicit(&v->contended, memory_order_relaxed)
            && mx__tls_write_permit == 0, 0)) {
        mx__vec_contended_raise();
    }
}

/* Mark one vector contended (spawn-boundary crossing).  Called from
 * generated code at the real-spawn path (codegen_llvm emits the marking
 * walk over the spawned closure's captures inside the EFFECT_SPAWN
 * runtime thunk, which executes iff the spawn is real); NULL is a no-op
 * so struct fields holding a never-initialized vec slot stay safe. */
void mx_vec_mark_contended(void *vp) {
    mx_vec *v = (mx_vec *)vp;
    if (v == NULL) {
        return;
    }
    atomic_store_explicit(&v->contended, 1, memory_order_relaxed);
}

mx_vec *mx_vec_new(void) {
    mx_vec *v = (mx_vec *)mx_rt_malloc(sizeof(mx_vec));
    v->len = 0;
    v->cap = 0;
    v->data = NULL;
    atomic_store_explicit(&v->contended, 0, memory_order_relaxed);
    return v;
}

void mx_vec_push(mx_vec *v, int64_t value) {
    mx_vec_check(v, "push");
    mx__vec_write_check(v);
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
    mx__vec_write_check(v);
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
    mx_vec_check(v, "index assignment");
    mx__vec_write_check(v);
    if (idx < 0 || idx >= v->len) {
        /* Interpreter parity (mir_interp index assignment): stores say
         * "index assignment", reads say "index" — mx_fvec_set_copy
         * already matched and this site lagged until the inline-fast-path
         * differential caught it. */
        mx_rt_raise("index assignment out of bounds: %lld (length %lld)",
                    (long long)idx, (long long)v->len);
    }
    v->data[idx] = value;
}

/* Inline-fast-path cold terminators (metaxu_rt.h): re-run the op's
 * canonical check order and raise/abort exactly as the full op would.
 * The trailing mx_rt_fail is defensive — generated code reaches these
 * only when an inlined check failed, so one of the canonical checks must
 * fire; if a racing writer changed the header in between (possible only
 * under a data race the language already forbids), fail loudly. */
void mx__vec_get_fail(const mx_vec *v, int64_t idx) {
    mx_vec_check(v, "index");
    if (idx < 0 || idx >= v->len) {
        mx_rt_raise("index out of bounds: %lld (length %lld)",
                    (long long)idx, (long long)v->len);
    }
    mx_rt_fail("metaxu internal: Vec get fast-path miss did not fail");
}

void mx__vec_set_fail(mx_vec *v, int64_t idx) {
    mx_vec_check(v, "index assignment");
    mx__vec_write_check(v);
    if (idx < 0 || idx >= v->len) {
        mx_rt_raise("index assignment out of bounds: %lld (length %lld)",
                    (long long)idx, (long long)v->len);
    }
    mx_rt_fail("metaxu internal: Vec set fast-path miss did not fail");
}

void mx__vec_pop_fail(mx_vec *v) {
    mx_vec_check(v, "pop");
    mx__vec_write_check(v);
    if (v->len == 0) {
        mx_rt_raise("pop: Vec is empty");
    }
    mx_rt_fail("metaxu internal: Vec pop fast-path miss did not fail");
}

void mx__vec_len_fail(const mx_vec *v) {
    mx_vec_check(v, "len");
    mx_rt_fail("metaxu internal: Vec len fast-path miss did not fail");
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
 * Tiles (docs/gpu_tiles.md, Stage 0): immutable 2D row-major blocks.
 *
 * Same word-block model as mx_fvec: elements are int64 words (f64 via
 * bitcast; the emitter's `tile:` kind carries the element type and passes
 * is_f64 to the ops whose arithmetic differs).  Blocks are write-once and
 * leak by design, exactly like fvec (immutability makes sharing sound and
 * ownership never unique).  Shape/kind agreement is STATIC natively (the
 * tile shape checker rejects it at compile time and the kind system
 * demotes anything unresolvable), so mismatches reaching these functions
 * are internal errors (mx_rt_fail), not language-visible raises; the two
 * checks that are dynamic even natively — from_vec length and get bounds
 * — raise catchably with wording byte-identical to mir_interp's _tile_*.
 * Accumulation order is pinned to match the interpreter bit-for-bit:
 * row-major for sum, k ascending for dot.
 * ---------------------------------------------------------------------- */

struct mx_tile {
    int64_t rows;
    int64_t cols;
    int64_t elems[];  /* rows*cols element words, row-major */
};

static mx_tile *mx_tile_new(int64_t rows, int64_t cols) {
    if (rows <= 0 || cols <= 0) {
        mx_rt_fail("metaxu internal: non-positive tile shape %lldx%lld "
                   "reached the runtime", (long long)rows, (long long)cols);
    }
    mx_tile *t = (mx_tile *)mx_rt_malloc(
        sizeof(mx_tile) + (size_t)(rows * cols) * sizeof(int64_t));
    t->rows = rows;
    t->cols = cols;
    return t;
}

static void mx_tile_check(const mx_tile *t, const char *op) {
    if (t == NULL) {
        mx_rt_fail("%s: expected a Tile receiver, got NULL", op);
    }
}

mx_tile *mx_tile_zeros(int64_t rows, int64_t cols) {
    mx_tile *t = mx_tile_new(rows, cols);
    /* +0.0's bit pattern is all-zeros, so float and int zero coincide. */
    memset(t->elems, 0, (size_t)(rows * cols) * sizeof(int64_t));
    return t;
}

mx_tile *mx_tile_filled(int64_t rows, int64_t cols, int64_t word) {
    mx_tile *t = mx_tile_new(rows, cols);
    for (int64_t i = 0; i < rows * cols; i++) t->elems[i] = word;
    return t;
}

mx_tile *mx_tile_arange(int64_t rows, int64_t cols) {
    mx_tile *t = mx_tile_new(rows, cols);
    for (int64_t i = 0; i < rows * cols; i++) t->elems[i] = i;
    return t;
}

mx_tile *mx_tile_from_vec(const mx_vec *v, int64_t rows, int64_t cols) {
    if (v == NULL) {
        mx_rt_raise("Tile.from_vec: expected a Vec, got NULL");
    }
    mx_tile *t = mx_tile_new(rows, cols);
    if (v->len != rows * cols) {
        mx_rt_raise("Tile.from_vec: Vec length %lld does not fill "
                    "%lldx%lld (= %lld elements)", (long long)v->len,
                    (long long)rows, (long long)cols,
                    (long long)(rows * cols));
    }
    memcpy(t->elems, v->data, (size_t)v->len * sizeof(int64_t));
    return t;
}

mx_vec *mx_tile_to_vec(const mx_tile *t) {
    mx_tile_check(t, "Tile.to_vec");
    mx_vec *v = mx_vec_new();
    for (int64_t i = 0; i < t->rows * t->cols; i++) {
        mx_vec_push(v, t->elems[i]);
    }
    return v;
}

static void mx_tile_same(const mx_tile *a, const mx_tile *b,
                         const char *op) {
    mx_tile_check(a, op);
    mx_tile_check(b, op);
    if (a->rows != b->rows || a->cols != b->cols) {
        mx_rt_fail("metaxu internal: %s shape mismatch reached the runtime "
                   "(%lldx%lld vs %lldx%lld)", op, (long long)a->rows,
                   (long long)a->cols, (long long)b->rows,
                   (long long)b->cols);
    }
}

/* f32 elements (ekind 2) live as the f32-REPRESENTABLE double: adding or
 * multiplying two such values in double and rounding once through float
 * IS the correctly-rounded f32 op (24+24 < 53 significand bits), which is
 * what keeps all three engines bit-identical.  mx_f32r is that rounding. */
static double mx_f32r(double x) { return (double)(float)x; }

/* f16 elements (ekind 3, docs/gpu_tiles.md Stage 1f) follow the same
 * recipe (11+11 < 53); mx_f16r matches the interpreter's struct 'e'
 * round-trip bit for bit.  _Float16 is required — an unsupported
 * toolchain must fail LOUDLY at build, never emulate approximately. */
#if !defined(__FLT16_MANT_DIG__)
#error "f16 tiles need _Float16 (clang on x86-64/arm64 provides it)"
#endif
static double mx_f16r(double x) { return (double)(_Float16)x; }

/* One rounding step for a narrow float ekind (2 f32, 3 f16); identity
 * for f64. */
static double mx_ekr(int64_t ekind, double x) {
    if (ekind == 2) return mx_f32r(x);
    if (ekind == 3) return mx_f16r(x);
    return x;
}

/* Elementwise add/mul: ekind selects the arithmetic (0 int, 1 f64,
 * 2 f32 / 3 f16 = f64 op + one rounding); words in, words out. */
#define MX_TILE_EW(name, fop, iop)                                          \
    mx_tile *name(const mx_tile *a, const mx_tile *b, int64_t ekind) {      \
        mx_tile_same(a, b, #name);                                          \
        mx_tile *out = mx_tile_new(a->rows, a->cols);                       \
        int64_t n = a->rows * a->cols;                                      \
        for (int64_t i = 0; i < n; i++) {                                   \
            if (ekind != 0) {                                               \
                double x, y, r;                                             \
                memcpy(&x, &a->elems[i], sizeof x);                         \
                memcpy(&y, &b->elems[i], sizeof y);                         \
                r = fop;                                                    \
                r = mx_ekr(ekind, r);                                       \
                memcpy(&out->elems[i], &r, sizeof r);                       \
            } else {                                                        \
                int64_t x = a->elems[i], y = b->elems[i];                   \
                out->elems[i] = (iop);                                      \
            }                                                               \
        }                                                                   \
        return out;                                                         \
    }

MX_TILE_EW(mx_tile_add, x + y, x + y)
MX_TILE_EW(mx_tile_mul, x * y, x * y)

mx_tile *mx_tile_scale(const mx_tile *t, int64_t sword, int64_t ekind) {
    mx_tile_check(t, "Tile.scale");
    mx_tile *out = mx_tile_new(t->rows, t->cols);
    int64_t n = t->rows * t->cols;
    for (int64_t i = 0; i < n; i++) {
        if (ekind != 0) {
            double x, s, r;
            memcpy(&x, &t->elems[i], sizeof x);
            memcpy(&s, &sword, sizeof s);
            /* f32/f16 tiles scale by a LANGUAGE (f64) scalar: the factor
             * rounds to the element width first, the multiply rounds per
             * element — exactly (float)s / (_Float16)s in C, exactly
             * mir_interp. */
            s = mx_ekr(ekind, s);
            r = x * s;
            r = mx_ekr(ekind, r);
            memcpy(&out->elems[i], &r, sizeof r);
        } else {
            out->elems[i] = t->elems[i] * sword;
        }
    }
    return out;
}

mx_tile *mx_tile_dot(const mx_tile *a, const mx_tile *b, int64_t ekind) {
    mx_tile_check(a, "Tile.dot");
    mx_tile_check(b, "Tile.dot");
    if (a->cols != b->rows) {
        mx_rt_fail("metaxu internal: Tile.dot shape mismatch reached the "
                   "runtime (%lldx%lld · %lldx%lld)", (long long)a->rows,
                   (long long)a->cols, (long long)b->rows,
                   (long long)b->cols);
    }
    int64_t R = a->rows, K = a->cols, C = b->cols;
    mx_tile *out = mx_tile_new(R, C);
    for (int64_t i = 0; i < R; i++) {
        for (int64_t j = 0; j < C; j++) {
            if (ekind != 0) {
                double acc = 0.0;
                for (int64_t k = 0; k < K; k++) {  /* pinned: k ascending */
                    double x, y;
                    memcpy(&x, &a->elems[i * K + k], sizeof x);
                    memcpy(&y, &b->elems[k * C + j], sizeof y);
                    if (ekind == 2 || ekind == 3) {
                        /* round the product, then the accumulation */
                        acc = mx_ekr(ekind, acc + mx_ekr(ekind, x * y));
                    } else {
                        acc = acc + x * y;
                    }
                }
                memcpy(&out->elems[i * C + j], &acc, sizeof acc);
            } else {
                int64_t acc = 0;
                for (int64_t k = 0; k < K; k++) {
                    acc = acc + a->elems[i * K + k] * b->elems[k * C + j];
                }
                out->elems[i * C + j] = acc;
            }
        }
    }
    return out;
}

int64_t mx_tile_sum(const mx_tile *t, int64_t ekind) {
    mx_tile_check(t, "Tile.sum");
    int64_t n = t->rows * t->cols;
    if (ekind != 0) {
        double acc = 0.0;
        for (int64_t i = 0; i < n; i++) {  /* pinned: row-major */
            double x;
            memcpy(&x, &t->elems[i], sizeof x);
            acc = mx_ekr(ekind, acc + x);
        }
        int64_t w;
        memcpy(&w, &acc, sizeof w);
        return w;
    }
    int64_t acc = 0;
    for (int64_t i = 0; i < n; i++) acc = acc + t->elems[i];
    return acc;
}

/* f32/f16 conversions (docs/gpu_tiles.md Stage 1d/1f): total and shape-
 * preserving.  src_ekind says how to READ the words (0 int, 1/2/3 double
 * bits); the result of to_f32 / to_f16 is every element rounded through
 * the narrow width and stored back as double bits, of to_f64 every
 * element widened to double bits (f32/f16-representables pass through
 * exactly). */
mx_tile *mx_tile_to_f32(const mx_tile *t, int64_t src_ekind) {
    mx_tile_check(t, "Tile.to_f32");
    mx_tile *out = mx_tile_new(t->rows, t->cols);
    int64_t n = t->rows * t->cols;
    for (int64_t i = 0; i < n; i++) {
        double x;
        if (src_ekind == 0) {
            x = (double)t->elems[i];
        } else {
            memcpy(&x, &t->elems[i], sizeof x);
        }
        x = mx_f32r(x);
        memcpy(&out->elems[i], &x, sizeof x);
    }
    return out;
}

mx_tile *mx_tile_to_f16(const mx_tile *t, int64_t src_ekind) {
    mx_tile_check(t, "Tile.to_f16");
    mx_tile *out = mx_tile_new(t->rows, t->cols);
    int64_t n = t->rows * t->cols;
    for (int64_t i = 0; i < n; i++) {
        double x;
        if (src_ekind == 0) {
            x = (double)t->elems[i];
        } else {
            memcpy(&x, &t->elems[i], sizeof x);
        }
        x = mx_f16r(x);
        memcpy(&out->elems[i], &x, sizeof x);
    }
    return out;
}

mx_tile *mx_tile_to_f64(const mx_tile *t, int64_t src_ekind) {
    mx_tile_check(t, "Tile.to_f64");
    mx_tile *out = mx_tile_new(t->rows, t->cols);
    int64_t n = t->rows * t->cols;
    for (int64_t i = 0; i < n; i++) {
        double x;
        if (src_ekind == 0) {
            x = (double)t->elems[i];
        } else {
            memcpy(&x, &t->elems[i], sizeof x);
        }
        memcpy(&out->elems[i], &x, sizeof x);
    }
    return out;
}

mx_tile *mx_tile_transpose(const mx_tile *t) {
    mx_tile_check(t, "Tile.transpose");
    mx_tile *out = mx_tile_new(t->cols, t->rows);
    for (int64_t r = 0; r < t->rows; r++) {
        for (int64_t c = 0; c < t->cols; c++) {
            out->elems[c * t->rows + r] = t->elems[r * t->cols + c];
        }
    }
    return out;
}

int64_t mx_tile_get(const mx_tile *t, int64_t i, int64_t j) {
    mx_tile_check(t, "Tile.get");
    if (i < 0 || i >= t->rows || j < 0 || j >= t->cols) {
        mx_rt_raise("Tile.get: index out of bounds: (%lld, %lld) "
                    "(shape %lldx%lld)", (long long)i, (long long)j,
                    (long long)t->rows, (long long)t->cols);
    }
    return t->elems[i * t->cols + j];
}

int64_t mx_tile_rows(const mx_tile *t) {
    mx_tile_check(t, "Tile.rows");
    return t->rows;
}

int64_t mx_tile_cols(const mx_tile *t) {
    mx_tile_check(t, "Tile.cols");
    return t->cols;
}

/* Buffer <-> tile boundary (Stage 1).  Strict load/store raise on any
 * out-of-range element; the masked forms are the kernel-side ragged-edge
 * idiom (load_or reads `other`, store_clipped writes nothing) — wording
 * and check ORDER byte-identical to mir_interp (stores: contended-write
 * guard BEFORE the range check, like every Vec mutator). */

mx_tile *mx_tile_load(const mx_vec *v, int64_t off, int64_t rows,
                      int64_t cols) {
    if (v == NULL) {
        mx_rt_fail("Tile.load: expected a Vec, got NULL");
    }
    mx_tile *t = mx_tile_new(rows, cols);
    int64_t n = rows * cols;
    if (off < 0 || off + n > v->len) {
        mx_rt_raise("Tile.load: range [%lld, %lld) outside Vec length %lld",
                    (long long)off, (long long)(off + n), (long long)v->len);
    }
    memcpy(t->elems, v->data + off, (size_t)n * sizeof(int64_t));
    return t;
}

mx_tile *mx_tile_load_or(const mx_vec *v, int64_t off, int64_t rows,
                         int64_t cols, int64_t other) {
    if (v == NULL) {
        mx_rt_fail("Tile.load_or: expected a Vec, got NULL");
    }
    mx_tile *t = mx_tile_new(rows, cols);
    int64_t n = rows * cols;
    for (int64_t i = 0; i < n; i++) {
        int64_t j = off + i;
        t->elems[i] = (j >= 0 && j < v->len) ? v->data[j] : other;
    }
    return t;
}

void mx_tile_store(mx_vec *v, int64_t off, const mx_tile *t) {
    mx_vec_check(v, "Tile.store");
    mx_tile_check(t, "Tile.store");
    mx__vec_write_check(v);  /* contended-write guard, canonical order */
    int64_t n = t->rows * t->cols;
    if (off < 0 || off + n > v->len) {
        mx_rt_raise("Tile.store: range [%lld, %lld) outside Vec length "
                    "%lld", (long long)off, (long long)(off + n),
                    (long long)v->len);
    }
    memcpy(v->data + off, t->elems, (size_t)n * sizeof(int64_t));
}

void mx_tile_store_clipped(mx_vec *v, int64_t off, const mx_tile *t) {
    mx_vec_check(v, "Tile.store_clipped");
    mx_tile_check(t, "Tile.store_clipped");
    mx__vec_write_check(v);  /* contended-write guard, canonical order */
    int64_t n = t->rows * t->cols;
    for (int64_t i = 0; i < n; i++) {
        int64_t j = off + i;
        if (j >= 0 && j < v->len) {  /* masked-out element writes nothing */
            v->data[j] = t->elems[i];
        }
    }
}

/* 2D (row-strided) masked forms: element (i, j) maps to
 * off + i*stride + j — the tile-of-a-matrix idiom.  Same masked
 * semantics as load_or / store_clipped, per element. */

mx_tile *mx_tile_load_rows(const mx_vec *v, int64_t off, int64_t stride,
                           int64_t rows, int64_t cols, int64_t other) {
    if (v == NULL) {
        mx_rt_fail("Tile.load_rows: expected a Vec, got NULL");
    }
    mx_tile *t = mx_tile_new(rows, cols);
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            int64_t k = off + i * stride + j;
            t->elems[i * cols + j] =
                (k >= 0 && k < v->len) ? v->data[k] : other;
        }
    }
    return t;
}

void mx_tile_store_rows(mx_vec *v, int64_t off, int64_t stride,
                        const mx_tile *t) {
    mx_vec_check(v, "Tile.store_rows");
    mx_tile_check(t, "Tile.store_rows");
    mx__vec_write_check(v);  /* contended-write guard, canonical order */
    for (int64_t i = 0; i < t->rows; i++) {
        for (int64_t j = 0; j < t->cols; j++) {
            int64_t k = off + i * stride + j;
            if (k >= 0 && k < v->len) {
                v->data[k] = t->elems[i * t->cols + j];
            }
        }
    }
}

/* repr(MxTile): "tile[RxC](e, e, ...; e, ...)" — rows joined by "; ",
 * byte-identical to the interpreter's MxTile.__repr__. */
char *mx_tile_to_str(const mx_tile *t, int64_t is_f64) {
    mx_tile_check(t, "to_string");
    char *buf = NULL;
    size_t len = 0, cap = 0;
    char head[64];
    snprintf(head, sizeof head, "tile[%lldx%lld](",
             (long long)t->rows, (long long)t->cols);
    mx_buf_append(&buf, &len, &cap, head);
    for (int64_t r = 0; r < t->rows; r++) {
        if (r > 0) {
            mx_buf_append(&buf, &len, &cap, "; ");
        }
        for (int64_t c = 0; c < t->cols; c++) {
            if (c > 0) {
                mx_buf_append(&buf, &len, &cap, ", ");
            }
            char *piece;
            if (is_f64) {
                double x;
                memcpy(&x, &t->elems[r * t->cols + c], sizeof x);
                piece = mx_f64_to_str(x);
            } else {
                piece = mx_i64_to_str(t->elems[r * t->cols + c]);
            }
            mx_buf_append(&buf, &len, &cap, piece);
            free(piece);
        }
    }
    mx_buf_append(&buf, &len, &cap, ")");
    return buf;
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

char *mx_str_index(const char *s, int64_t idx) {
    mx_str_check(s, "index", "receiver");
    int64_t len = (int64_t)strlen(s);
    if (idx < 0 || idx >= len) {
        mx_rt_raise("index out of bounds: %lld (length %lld)",
                    (long long)idx, (long long)len);
    }
    char *out = (char *)mx_rt_malloc(2);
    out[0] = s[idx];
    out[1] = '\0';
    return out;
}

char *mx_str_slice(const char *s, int64_t start, int64_t stop,
                   int64_t step, int64_t mask) {
    /* The same normalisation as mx_fvec_slice: CPython's slice.indices(). */
    mx_str_check(s, "slice", "receiver");
    int64_t len = (int64_t)strlen(s);
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
        stop = step < 0 ? -1 : len;
    }
    int64_t count;
    if (step > 0) {
        count = stop > start ? (stop - start + step - 1) / step : 0;
    } else {
        count = start > stop ? (start - stop + (-step) - 1) / (-step) : 0;
    }
    char *out = (char *)mx_rt_malloc((size_t)count + 1);
    for (int64_t i = 0; i < count; i++) {
        out[i] = s[start + i * step];
    }
    out[count] = '\0';
    return out;
}

int64_t mx_str_eq(const char *a, const char *b) {
    mx_str_check(a, "str_eq", "operand");
    mx_str_check(b, "str_eq", "operand");
    return strcmp(a, b) == 0 ? 1 : 0;
}

/* The linear string builtins (`s.split(sep)`, `s.find(sub)`,
 * `s.replace(old, new)`, `s.trim()`, `parts.join(sep)`): each mirrors the
 * interpreter's Python-backed builtin of the same name, including the two
 * catchable diagnostics ("split: empty separator", "replace: empty
 * pattern").  Every result is a FRESH allocation (a string, or a Vec whose
 * elements are fresh strings); nothing retains its arguments.  Receiver
 * TYPE errors cannot arise here: the backend's kind checks reject a
 * non-string receiver at compile time, where the interpreter raises. */

static char *mx_str_dup_n(const char *s, size_t n) {
    char *out = (char *)mx_rt_malloc(n + 1);
    memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

int64_t mx_str_find(const char *s, const char *sub) {
    mx_str_check(s, "find", "receiver");
    mx_str_check(sub, "find", "argument");
    const char *p = strstr(s, sub);  /* an empty `sub` is found at 0 */
    return p == NULL ? -1 : (int64_t)(p - s);
}

mx_vec *mx_str_split(const char *s, const char *sep) {
    mx_str_check(s, "split", "receiver");
    mx_str_check(sep, "split", "separator");
    size_t ls = strlen(sep);
    if (ls == 0) {
        mx_rt_raise("split: empty separator");  /* catchable */
    }
    mx_vec *out = mx_vec_new();
    const char *cur = s;
    for (;;) {
        const char *p = strstr(cur, sep);
        if (p == NULL) {
            break;
        }
        mx_vec_push(out, (int64_t)(intptr_t)mx_str_dup_n(cur, (size_t)(p - cur)));
        cur = p + ls;
    }
    mx_vec_push(out, (int64_t)(intptr_t)mx_str_dup_n(cur, strlen(cur)));
    return out;
}

char *mx_str_replace(const char *s, const char *old, const char *new_) {
    mx_str_check(s, "replace", "receiver");
    mx_str_check(old, "replace", "pattern");
    mx_str_check(new_, "replace", "replacement");
    size_t lo = strlen(old);
    if (lo == 0) {
        mx_rt_raise("replace: empty pattern");  /* catchable */
    }
    size_t ln = strlen(new_);
    size_t count = 0;
    for (const char *p = strstr(s, old); p != NULL; p = strstr(p + lo, old)) {
        count++;
    }
    size_t ls = strlen(s);
    char *out = (char *)mx_rt_malloc(ls - count * lo + count * ln + 1);
    char *w = out;
    const char *cur = s;
    for (;;) {
        const char *p = strstr(cur, old);
        if (p == NULL) {
            break;
        }
        memcpy(w, cur, (size_t)(p - cur));
        w += p - cur;
        memcpy(w, new_, ln);
        w += ln;
        cur = p + lo;
    }
    size_t tail = strlen(cur);
    memcpy(w, cur, tail);
    w[tail] = '\0';
    return out;
}

static int mx_str_is_trim_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

char *mx_str_trim(const char *s) {
    mx_str_check(s, "trim", "receiver");
    size_t start = 0;
    size_t stop = strlen(s);
    while (start < stop && mx_str_is_trim_space(s[start])) {
        start++;
    }
    while (stop > start && mx_str_is_trim_space(s[stop - 1])) {
        stop--;
    }
    return mx_str_dup_n(s + start, stop - start);
}

char *mx_str_join(const mx_vec *parts, const char *sep) {
    if (parts == NULL) {
        mx_rt_fail("join: expected a Vec receiver, got NULL");
    }
    mx_str_check(sep, "join", "separator");
    int64_t n = mx_vec_len(parts);
    size_t ls = strlen(sep);
    size_t total = 0;
    for (int64_t i = 0; i < n; i++) {
        const char *e = (const char *)(intptr_t)mx_vec_get(parts, i);
        mx_str_check(e, "join", "element");
        total += strlen(e);
    }
    if (n > 1) {
        total += ls * (size_t)(n - 1);
    }
    char *out = (char *)mx_rt_malloc(total + 1);
    char *w = out;
    for (int64_t i = 0; i < n; i++) {
        if (i > 0) {
            memcpy(w, sep, ls);
            w += ls;
        }
        const char *e = (const char *)(intptr_t)mx_vec_get(parts, i);
        size_t le = strlen(e);
        memcpy(w, e, le);
        w += le;
    }
    *w = '\0';
    return out;
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
