/* metaxu_io.h -- POSIX-backed IO effect primitives (docs/io_runtime.md).
 *
 * The native implementation of the `with EFFECT_*` runtime mappings the
 * standard library's std.fs / std.process / std.env / std.io declare.
 * Reference semantics: mir_interp._rt_fs_* / _rt_process_* / _rt_env_* /
 * _rt_io_stderr, whose error-message wording is shared byte for byte
 * (the caught value is language-visible).
 *
 * Every argument and result is one word: int64_t, char* (NUL-terminated
 * UTF-8, fresh when returned) or mx_vec* (bytes as ints 0..255, or
 * strings as char* words; fresh when returned).  Failures are CATCHABLE
 * (mx_raisef) with the shape "<op>: <path>: <strerror>".
 *
 * | symbol                | signature                                       |
 * |-----------------------|-------------------------------------------------|
 * | (entry wrapper)       | void mx_io_set_args(int argc, char **argv)      |
 * | EFFECT_ENV_ARGS       | mx_vec *mx_env_args(void)                       |
 * | EFFECT_ENV_GET        | char *mx_env_get(const char *name)  ("" unset)  |
 * | EFFECT_ENV_HAS        | int64_t mx_env_has(const char *name)            |
 * | EFFECT_ENV_HOME       | char *mx_env_home(void)                         |
 * | EFFECT_ENV_CWD        | char *mx_env_cwd(void)                          |
 * | EFFECT_STDERR         | int64_t mx_io_stderr(const char *text)          |
 * | EFFECT_FS_READ        | mx_vec *mx_fs_read(const char *path)            |
 * | EFFECT_FS_WRITE       | int64_t mx_fs_write(const char *path, const mx_vec *bytes) |
 * | EFFECT_FS_EXISTS      | int64_t mx_fs_exists(const char *path)          |
 * | EFFECT_FS_IS_DIR      | int64_t mx_fs_is_dir(const char *path)          |
 * | EFFECT_FS_IS_FILE     | int64_t mx_fs_is_file(const char *path)         |
 * | EFFECT_FS_LIST_DIR    | mx_vec *mx_fs_list_dir(const char *path)        |
 * | EFFECT_FS_MKDIR_ALL   | int64_t mx_fs_mkdir_all(const char *path)       |
 * | EFFECT_FS_REMOVE_ALL  | int64_t mx_fs_remove_all(const char *path)      |
 * | EFFECT_FS_RENAME      | int64_t mx_fs_rename(const char *a, const char *b) |
 * | EFFECT_PROCESS_RUN    | int64_t mx_process_run(const mx_vec *argv, const char *cwd) |
 * | EFFECT_PROCESS_STATUS | int64_t mx_process_status(int64_t handle)       |
 * | EFFECT_PROCESS_STDOUT | char *mx_process_stdout(int64_t handle)         |
 * | EFFECT_PROCESS_STDERR | char *mx_process_stderr(int64_t handle)         |
 */
#ifndef METAXU_IO_H
#define METAXU_IO_H

#include <stdint.h>
#include "metaxu_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

void    mx_io_set_args(int argc, char **argv);
mx_vec *mx_env_args(void);
char   *mx_env_get(const char *name);
int64_t mx_env_has(const char *name);
char   *mx_env_home(void);
char   *mx_env_cwd(void);
int64_t mx_io_stderr(const char *text);
mx_vec *mx_fs_read(const char *path);
int64_t mx_fs_write(const char *path, const mx_vec *bytes);
int64_t mx_fs_exists(const char *path);
int64_t mx_fs_is_dir(const char *path);
int64_t mx_fs_is_file(const char *path);
mx_vec *mx_fs_list_dir(const char *path);
int64_t mx_fs_mkdir_all(const char *path);
int64_t mx_fs_remove_all(const char *path);
int64_t mx_fs_rename(const char *from, const char *to);
int64_t mx_process_run(const mx_vec *argv, const char *cwd);
int64_t mx_process_status(int64_t handle);
char   *mx_process_stdout(int64_t handle);
char   *mx_process_stderr(int64_t handle);

#ifdef __cplusplus
}
#endif

#endif /* METAXU_IO_H */
