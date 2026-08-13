/* metaxu_rt.c -- native runtime library for Metaxu (Vec + strings).
 *
 * Dependency-free C11.  See metaxu_rt.h for the ABI contract; the
 * interpreter (src/metaxu/compiler/mir_interp.py) is the reference for
 * every observable behavior here, including error message wording.
 */
#include "metaxu_rt.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------
 * Error path: strict, loud, fatal.  Mirrors the interpreter's InterpError
 * philosophy -- a clear message on stderr, then abort().  No fallbacks.
 * ---------------------------------------------------------------------- */
static void mx_rt_fail(const char *fmt, ...) {
    va_list ap;
    fputs("metaxu runtime error: ", stderr);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    fflush(stderr);
    abort();
}

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
        mx_rt_fail("pop: Vec is empty");
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
        mx_rt_fail("index out of bounds: %lld (length %lld)",
                   (long long)idx, (long long)v->len);
    }
    return v->data[idx];
}

void mx_vec_set(mx_vec *v, int64_t idx, int64_t value) {
    mx_vec_check(v, "index");
    if (idx < 0 || idx >= v->len) {
        mx_rt_fail("index out of bounds: %lld (length %lld)",
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
