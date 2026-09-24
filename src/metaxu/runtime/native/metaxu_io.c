/* metaxu_io.c -- POSIX-backed IO effect primitives.  Contract and the
 * reasoning behind it: docs/io_runtime.md; API table: metaxu_io.h. */
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#include "metaxu_io.h"
#include "metaxu_effects.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pwd.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* helpers                                                             */

static void *io_malloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (p == NULL) {
        fprintf(stderr, "out of memory (requested %zu bytes)\n", n);
        abort();
    }
    return p;
}

static char *io_dup(const char *s) {
    size_t n = strlen(s);
    char *out = (char *)io_malloc(n + 1);
    memcpy(out, s, n + 1);
    return out;
}

static char *io_dup_n(const char *s, size_t n) {
    char *out = (char *)io_malloc(n + 1);
    memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

static void io_check(const char *s, const char *op, const char *which) {
    if (s == NULL) {
        fprintf(stderr, "%s: expected a str %s, got NULL\n", op, which);
        abort();
    }
}

/* The catchable failure shape every operation shares (docs/io_runtime.md):
 * "<op>: <path>: <strerror>". */
static _Noreturn void io_raise(const char *op, const char *path, int err) {
    mx_raisef("%s: %s: %s", op, path, strerror(err));
}

/* A growable byte buffer for reads and captured output. */
typedef struct { char *data; size_t len, cap; } io_buf;

static void buf_push(io_buf *b, const char *src, size_t n) {
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 4096;
        while (cap < b->len + n + 1) cap *= 2;
        char *nd = (char *)realloc(b->data, cap);
        if (nd == NULL) {
            fprintf(stderr, "out of memory (requested %zu bytes)\n", cap);
            abort();
        }
        b->data = nd;
        b->cap = cap;
    }
    memcpy(b->data + b->len, src, n);
    b->len += n;
    b->data[b->len] = '\0';
}

static char *buf_take(io_buf *b) {
    if (b->data == NULL) return io_dup("");
    return b->data;  /* NUL-terminated by buf_push */
}

/* ------------------------------------------------------------------ */
/* arguments and environment                                           */

static int g_argc = 0;
static char **g_argv = NULL;

void mx_io_set_args(int argc, char **argv) {
    g_argc = argc;
    g_argv = argv;
}

mx_vec *mx_env_args(void) {
    mx_vec *out = mx_vec_new();
    for (int i = 1; i < g_argc; i++) {
        mx_vec_push(out, (int64_t)(intptr_t)io_dup(g_argv[i]));
    }
    return out;
}

char *mx_env_get(const char *name) {
    io_check(name, "get", "name");
    const char *v = getenv(name);
    return io_dup(v ? v : "");
}

int64_t mx_env_has(const char *name) {
    io_check(name, "has", "name");
    return getenv(name) != NULL;
}

char *mx_env_home(void) {
    const char *h = getenv("HOME");
    if (h != NULL && h[0] != '\0') return io_dup(h);
    struct passwd *pw = getpwuid(getuid());
    return io_dup(pw != NULL && pw->pw_dir != NULL ? pw->pw_dir : "");
}

char *mx_env_cwd(void) {
    size_t cap = 256;
    for (;;) {
        char *buf = (char *)io_malloc(cap);
        if (getcwd(buf, cap) != NULL) return buf;
        int err = errno;
        free(buf);
        if (err != ERANGE) io_raise("cwd", ".", err);
        cap *= 2;
    }
}

int64_t mx_io_stderr(const char *text) {
    io_check(text, "write_err", "text");
    fflush(stdout);
    fputs(text, stderr);
    fflush(stderr);
    return 0;
}

/* ------------------------------------------------------------------ */
/* files                                                               */

mx_vec *mx_fs_read(const char *path) {
    io_check(path, "read", "path");
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) io_raise("read", path, errno);
    struct stat st;
    if (fstat(fd, &st) == 0 && S_ISDIR(st.st_mode)) {
        close(fd);
        io_raise("read", path, EISDIR);
    }
    mx_vec *out = mx_vec_new();
    unsigned char chunk[65536];
    for (;;) {
        ssize_t n = read(fd, chunk, sizeof chunk);
        if (n < 0) {
            if (errno == EINTR) continue;
            int err = errno;
            close(fd);
            io_raise("read", path, err);
        }
        if (n == 0) break;
        for (ssize_t i = 0; i < n; i++) mx_vec_push(out, (int64_t)chunk[i]);
    }
    close(fd);
    return out;
}

int64_t mx_fs_write(const char *path, const mx_vec *bytes) {
    io_check(path, "write", "path");
    if (bytes == NULL) {
        fprintf(stderr, "write: expected a Vec of bytes, got NULL\n");
        abort();
    }
    int64_t n = mx_vec_len(bytes);
    unsigned char *buf = (unsigned char *)io_malloc((size_t)n);
    for (int64_t i = 0; i < n; i++) {
        int64_t w = mx_vec_get(bytes, i);
        if (w < 0 || w > 255) {
            free(buf);
            mx_raisef("write: element %lld is not a byte (0..255): %lld",
                      (long long)i, (long long)w);
        }
        buf[i] = (unsigned char)w;
    }
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
    if (fd < 0) {
        int err = errno;
        free(buf);
        io_raise("write", path, err);
    }
    size_t done = 0;
    while (done < (size_t)n) {
        ssize_t k = write(fd, buf + done, (size_t)n - done);
        if (k < 0) {
            if (errno == EINTR) continue;
            int err = errno;
            close(fd);
            free(buf);
            io_raise("write", path, err);
        }
        done += (size_t)k;
    }
    free(buf);
    if (close(fd) != 0) io_raise("write", path, errno);
    return 0;
}

int64_t mx_fs_exists(const char *path) {
    io_check(path, "exists", "path");
    struct stat st;
    return stat(path, &st) == 0;
}

int64_t mx_fs_is_dir(const char *path) {
    io_check(path, "is_dir", "path");
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

int64_t mx_fs_is_file(const char *path) {
    io_check(path, "is_file", "path");
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static int cmp_names(const void *a, const void *b) {
    return strcmp(*(char *const *)a, *(char *const *)b);
}

mx_vec *mx_fs_list_dir(const char *path) {
    io_check(path, "list_dir", "path");
    DIR *d = opendir(path);
    if (d == NULL) io_raise("list_dir", path, errno);
    size_t count = 0, cap = 16;
    char **names = (char **)io_malloc(cap * sizeof *names);
    for (;;) {
        errno = 0;
        struct dirent *e = readdir(d);
        if (e == NULL) {
            if (errno != 0) {
                int err = errno;
                closedir(d);
                io_raise("list_dir", path, err);
            }
            break;
        }
        if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0) continue;
        if (count == cap) {
            cap *= 2;
            char **nn = (char **)realloc(names, cap * sizeof *names);
            if (nn == NULL) abort();
            names = nn;
        }
        names[count++] = io_dup(e->d_name);
    }
    closedir(d);
    /* Byte order, which is what Python's sorted() gives for str keys of
     * UTF-8 names (code points sort like their UTF-8 bytes). */
    qsort(names, count, sizeof *names, cmp_names);
    mx_vec *out = mx_vec_new();
    for (size_t i = 0; i < count; i++) {
        mx_vec_push(out, (int64_t)(intptr_t)names[i]);
    }
    free(names);
    return out;
}

int64_t mx_fs_mkdir_all(const char *path) {
    io_check(path, "mkdir_all", "path");
    size_t n = strlen(path);
    char *buf = io_dup(path);
    for (size_t i = 1; i <= n; i++) {
        if (buf[i] == '/' || buf[i] == '\0') {
            char saved = buf[i];
            buf[i] = '\0';
            if (buf[0] != '\0' && mkdir(buf, 0777) != 0) {
                int err = errno;
                struct stat st;
                if (!(err == EEXIST && stat(buf, &st) == 0 && S_ISDIR(st.st_mode))) {
                    free(buf);
                    /* Python's makedirs names the full path in its error. */
                    io_raise("mkdir_all", path, err == EEXIST ? EEXIST : err);
                }
            }
            buf[i] = saved;
        }
    }
    free(buf);
    return 0;
}

static void remove_tree(const char *path) {
    struct stat st;
    if (lstat(path, &st) != 0) io_raise("remove_all", path, errno);
    if (S_ISDIR(st.st_mode)) {
        DIR *d = opendir(path);
        if (d == NULL) io_raise("remove_all", path, errno);
        /* Collect first: unlinking while iterating is unspecified. */
        size_t count = 0, cap = 16;
        char **names = (char **)io_malloc(cap * sizeof *names);
        for (;;) {
            errno = 0;
            struct dirent *e = readdir(d);
            if (e == NULL) break;
            if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0) continue;
            if (count == cap) {
                cap *= 2;
                char **nn = (char **)realloc(names, cap * sizeof *names);
                if (nn == NULL) abort();
                names = nn;
            }
            names[count++] = io_dup(e->d_name);
        }
        closedir(d);
        size_t plen = strlen(path);
        for (size_t i = 0; i < count; i++) {
            size_t clen = plen + 1 + strlen(names[i]) + 1;
            char *child = (char *)io_malloc(clen);
            snprintf(child, clen, "%s/%s", path, names[i]);
            remove_tree(child);
            free(child);
            free(names[i]);
        }
        free(names);
        if (rmdir(path) != 0) io_raise("remove_all", path, errno);
    } else {
        if (unlink(path) != 0) io_raise("remove_all", path, errno);
    }
}

int64_t mx_fs_remove_all(const char *path) {
    io_check(path, "remove_all", "path");
    remove_tree(path);
    return 0;
}

int64_t mx_fs_rename(const char *from, const char *to) {
    io_check(from, "rename", "path");
    io_check(to, "rename", "path");
    if (rename(from, to) != 0) io_raise("rename", from, errno);
    return 0;
}

/* ------------------------------------------------------------------ */
/* processes                                                           */

typedef struct { int64_t status; char *out; char *err; } proc_rec;

static proc_rec *g_procs = NULL;
static size_t g_nprocs = 0, g_cap_procs = 0;

static proc_rec *proc_lookup(const char *op, int64_t handle) {
    if (handle < 1 || (size_t)handle > g_nprocs) {
        mx_raisef("%s: no such process handle %lld", op, (long long)handle);
    }
    return &g_procs[handle - 1];
}

static void drain(int fd_out, int fd_err, io_buf *out, io_buf *err) {
    struct pollfd fds[2] = {{fd_out, POLLIN, 0}, {fd_err, POLLIN, 0}};
    io_buf *bufs[2] = {out, err};
    int open_count = 2;
    char chunk[65536];
    while (open_count > 0) {
        if (poll(fds, 2, -1) < 0) {
            if (errno == EINTR) continue;
            break;
        }
        for (int i = 0; i < 2; i++) {
            if (fds[i].fd < 0) continue;
            if (fds[i].revents & (POLLIN | POLLHUP | POLLERR)) {
                ssize_t n = read(fds[i].fd, chunk, sizeof chunk);
                if (n > 0) {
                    buf_push(bufs[i], chunk, (size_t)n);
                } else if (n == 0 || (n < 0 && errno != EINTR)) {
                    fds[i].fd = -1;
                    open_count--;
                }
            }
        }
    }
}

int64_t mx_process_run(const mx_vec *argv, const char *cwd) {
    if (argv == NULL) {
        fprintf(stderr, "run: expected a Vec argv, got NULL\n");
        abort();
    }
    io_check(cwd, "run", "cwd");
    int64_t n = mx_vec_len(argv);
    if (n == 0) mx_raisef("run: empty argv");
    char **args = (char **)io_malloc(((size_t)n + 1) * sizeof *args);
    for (int64_t i = 0; i < n; i++) {
        args[i] = (char *)(intptr_t)mx_vec_get(argv, i);
        io_check(args[i], "run", "argument");
    }
    args[n] = NULL;

    int out_p[2], err_p[2], fail_p[2];
    if (pipe(out_p) != 0 || pipe(err_p) != 0 || pipe(fail_p) != 0) {
        io_raise("run", args[0], errno);
    }
    fcntl(fail_p[1], F_SETFD, FD_CLOEXEC);
    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid < 0) io_raise("run", args[0], errno);
    if (pid == 0) {
        /* child */
        close(out_p[0]);
        close(err_p[0]);
        close(fail_p[0]);
        dup2(out_p[1], 1);
        dup2(err_p[1], 2);
        close(out_p[1]);
        close(err_p[1]);
        if (cwd[0] != '\0' && chdir(cwd) != 0) {
            unsigned char tag = 1;
            int err = errno;
            (void)!write(fail_p[1], &tag, 1);
            (void)!write(fail_p[1], &err, sizeof err);
            _exit(127);
        }
        execvp(args[0], args);
        unsigned char tag = 0;
        int err = errno;
        (void)!write(fail_p[1], &tag, 1);
        (void)!write(fail_p[1], &err, sizeof err);
        _exit(127);
    }
    /* parent */
    close(out_p[1]);
    close(err_p[1]);
    close(fail_p[1]);
    unsigned char tag = 0;
    int child_err = 0;
    ssize_t got = read(fail_p[0], &tag, 1);
    if (got == 1) {
        (void)!read(fail_p[0], &child_err, sizeof child_err);
    }
    close(fail_p[0]);
    io_buf out = {0}, err = {0};
    drain(out_p[0], err_p[0], &out, &err);
    close(out_p[0]);
    close(err_p[0]);
    int wstatus = 0;
    while (waitpid(pid, &wstatus, 0) < 0) {
        if (errno != EINTR) break;
    }
    if (got == 1) {
        free(out.data);
        free(err.data);
        const char *what = tag == 1 ? cwd : args[0];
        free(args);
        io_raise("run", what, child_err);
    }
    free(args);
    int64_t status = 0;
    if (WIFEXITED(wstatus)) status = WEXITSTATUS(wstatus);
    else if (WIFSIGNALED(wstatus)) status = -(int64_t)WTERMSIG(wstatus);
    if (g_nprocs == g_cap_procs) {
        g_cap_procs = g_cap_procs ? g_cap_procs * 2 : 8;
        proc_rec *np = (proc_rec *)realloc(g_procs, g_cap_procs * sizeof *np);
        if (np == NULL) abort();
        g_procs = np;
    }
    g_procs[g_nprocs].status = status;
    g_procs[g_nprocs].out = buf_take(&out);
    g_procs[g_nprocs].err = buf_take(&err);
    g_nprocs++;
    return (int64_t)g_nprocs;
}

int64_t mx_process_status(int64_t handle) {
    return proc_lookup("status", handle)->status;
}

char *mx_process_stdout(int64_t handle) {
    return io_dup(proc_lookup("stdout", handle)->out);
}

char *mx_process_stderr(int64_t handle) {
    return io_dup(proc_lookup("stderr", handle)->err);
}
