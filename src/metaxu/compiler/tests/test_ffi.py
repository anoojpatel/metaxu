"""FFI end-to-end tests: extern declarations, unsafe blocks, and the
interpreter's simulated C heap.

Every test goes through the REAL front end (parse -> desugar -> freeze ->
infer -> HIR -> MIR -> interpreter), per the repo convention: the historical
bug pattern here was seams where constructs silently degraded. The specific
seams pinned by this file:

- unsafe blocks used to fall through HIR's None fallback, so any function
  whose body was `unsafe { ... }` silently degraded to `ret Unit` (which is
  how example 05's Buffer.new returned Unit and copy_from ended up
  dispatching on a Unit receiver);
- `Ok(@mut file)` match patterns degraded to wildcards (Ok/Err had no
  builtin Result enum, and borrow-annotated payload bindings vanished).

The runtime model (strict, simulated heap): malloc returns an MxPtr handle
backed by a Python bytearray; every read/write is bounds-checked; freed
allocations are poisoned. A buffer overrun, use-after-free or double free is
a hard InterpError — never UB.
"""
from __future__ import annotations

import pytest

from metaxu.compiler.pipeline import build_context_from_source, run_pipeline_from_source
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import InterpError, MirInterpreter, MxPtr, UNIT


def build(source: str) -> MirInterpreter:
    ctx = build_context_from_source(source)
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    interp = MirInterpreter()
    interp.load(lower_hir_to_mir(hir))
    return interp


def call(source: str, fn: str = "main", args: list | None = None):
    return build(source).call(fn, args or [])


EXTERN_HEADER = """
extern "C" {
    fn malloc(size: uint) -> *void;
    fn free(ptr: *void);
    fn memcpy(dest: *void, src: *void, n: uint) -> *void;
    fn realloc(ptr: *void, size: uint) -> *void;
}
"""


# ---------------------------------------------------------------------------
# Front end: extern declarations + unsafe blocks flow through the pipeline
# ---------------------------------------------------------------------------

def test_extern_and_unsafe_pass_strict_pipeline():
    # The full strict pipeline (type + borrow checks) accepts extern blocks,
    # opaque extern types, pointer types (*void, *FILE) and unsafe blocks.
    src = EXTERN_HEADER + """
extern "C" {
    type FILE;
    fn fopen(filename: *char, mode: *char) -> *FILE;
    fn fclose(file: *FILE) -> int;
}

fn main() {
    unsafe {
        let p = malloc(4);
        free(p)
    }
}
"""
    run_pipeline_from_source(src)  # must not raise


def test_unsafe_block_yields_its_tail_value():
    # THE Buffer.new seam: an unsafe block is a block expression — its value
    # is the last statement's value. (It used to lower to None and the whole
    # function body degraded to unit.)
    src = "fn make() -> int { unsafe { 5 } }"
    assert call(src, "make") == 5


def test_unsafe_constructor_return_survives():
    # Constructor-return through unsafe: exactly the Buffer.new shape.
    src = EXTERN_HEADER + """
struct Buffer { ptr: *void, size: uint }

fn make(size: uint) -> Buffer {
    unsafe {
        let ptr = malloc(size);
        Buffer { ptr: ptr, size: size }
    }
}
"""
    result = call(src, "make", [16])
    assert result is not UNIT
    assert result.name == "Buffer"
    assert isinstance(result.get("ptr"), MxPtr)
    assert result.get("size") == 16


# ---------------------------------------------------------------------------
# Simulated heap: malloc / ptr_write / ptr_read / free round-trip
# ---------------------------------------------------------------------------

def test_malloc_write_read_free_roundtrip():
    src = EXTERN_HEADER + """
fn main() -> int {
    unsafe {
        let p = malloc(2);
        ptr_write(p, 0, 65);
        ptr_write(p, 1, 200);
        let v = ptr_read(p, 0) + ptr_read(p, 1);
        free(p);
        v
    }
}
"""
    assert call(src) == 265


def test_malloc_zero_initialized():
    src = EXTERN_HEADER + """
fn main() -> int {
    unsafe {
        let p = malloc(8);
        ptr_read(p, 7)
    }
}
"""
    assert call(src) == 0


def test_memcpy_between_buffers():
    src = EXTERN_HEADER + """
fn main() -> int {
    unsafe {
        let a = malloc(3);
        ptr_write(a, 0, 1);
        ptr_write(a, 1, 2);
        ptr_write(a, 2, 3);
        let b = malloc(3);
        memcpy(b, a, 3);
        free(a);
        ptr_read(b, 0) + ptr_read(b, 1) + ptr_read(b, 2)
    }
}
"""
    assert call(src) == 6


def test_memcpy_from_string_source():
    # memcpy also reads from a string operand ("hi" -> UTF-8 bytes).
    src = EXTERN_HEADER + """
fn main() -> int {
    unsafe {
        let p = malloc(2);
        memcpy(p, "hi", 2);
        ptr_read(p, 0)
    }
}
"""
    assert call(src) == ord("h")


def test_realloc_preserves_bytes_and_poisons_old_pointer():
    src = EXTERN_HEADER + """
fn main() -> int {
    unsafe {
        let p = malloc(2);
        ptr_write(p, 0, 9);
        let q = realloc(p, 4);
        ptr_write(q, 3, 1);
        ptr_read(q, 0) + ptr_read(q, 3)
    }
}
"""
    assert call(src) == 10
    stale = EXTERN_HEADER + """
fn main() -> int {
    unsafe {
        let p = malloc(2);
        let q = realloc(p, 4);
        ptr_read(p, 0)
    }
}
"""
    with pytest.raises(InterpError, match="use after free"):
        call(stale)


# ---------------------------------------------------------------------------
# Strictness: overruns and lifetime violations are errors, not UB
# ---------------------------------------------------------------------------

def test_memcpy_overrun_is_an_error():
    src = EXTERN_HEADER + """
fn main() {
    unsafe {
        let p = malloc(4);
        memcpy(p, "hello", 5)
    }
}
"""
    with pytest.raises(InterpError, match="out of bounds"):
        call(src)


def test_read_out_of_bounds_is_an_error():
    src = EXTERN_HEADER + """
fn main() -> int {
    unsafe {
        let p = malloc(4);
        ptr_read(p, 4)
    }
}
"""
    with pytest.raises(InterpError, match="out of bounds"):
        call(src)


def test_use_after_free_is_an_error():
    src = EXTERN_HEADER + """
fn main() -> int {
    unsafe {
        let p = malloc(4);
        free(p);
        ptr_read(p, 0)
    }
}
"""
    with pytest.raises(InterpError, match="use after free"):
        call(src)


def test_double_free_is_an_error():
    src = EXTERN_HEADER + """
fn main() {
    unsafe {
        let p = malloc(4);
        free(p);
        free(p)
    }
}
"""
    with pytest.raises(InterpError, match="double free"):
        call(src)


def test_write_through_readonly_as_ptr_is_an_error():
    # as_ptr produces a read-only byte snapshot: writing through it would be
    # a silent write into data nobody can observe, so it fails loudly.
    src = EXTERN_HEADER + """
fn main() {
    unsafe {
        let s = "abc".as_ptr();
        ptr_write(s, 0, 66)
    }
}
"""
    with pytest.raises(InterpError, match="read-only"):
        call(src)


def test_ptr_write_requires_a_byte_value():
    src = EXTERN_HEADER + """
fn main() {
    unsafe {
        let p = malloc(1);
        ptr_write(p, 0, 300)
    }
}
"""
    with pytest.raises(InterpError, match="byte"):
        call(src)


# ---------------------------------------------------------------------------
# The Buffer.copy_from shape from example 05 (trait method + vector as_ptr)
# ---------------------------------------------------------------------------

BUFFER_SRC = EXTERN_HEADER + """
struct Buffer { ptr: *void, size: uint }

implement Buffer {
    fn new(size: uint) -> Buffer {
        unsafe {
            let ptr = malloc(size);
            Buffer { ptr: ptr, size: size }
        }
    }

    fn copy_from(@mut self, @const source: &[u8]) {
        if source.len() <= self.size {
            unsafe {
                memcpy(self.ptr, source.as_ptr(), source.len())
            }
        }
    }

    fn free(@mut self) {
        unsafe {
            free(self.ptr);
            self.ptr = null;
            self.size = 0
        }
    }
}
"""


def test_buffer_copy_from_dispatches_on_buffer_and_copies_bytes():
    # THE run-gate failure shape: Buffer.new must return a Buffer (not Unit)
    # so copy_from dispatches on it; the bytes must actually land.
    src = BUFFER_SRC + """
fn main() -> int {
    let @mut b = Buffer.new(8);
    let data = vector[int,3](7, 8, 9);
    b.copy_from(data);
    ptr_read(b.ptr, 0) + ptr_read(b.ptr, 2)
}
"""
    assert call(src) == 16


def test_buffer_copy_from_too_large_source_is_skipped():
    # The guard in copy_from (source.len() <= self.size) makes the oversized
    # copy a no-op — the buffer keeps its zeroed bytes.
    src = BUFFER_SRC + """
fn main() -> int {
    let @mut b = Buffer.new(2);
    let data = vector[int,3](7, 8, 9);
    b.copy_from(data);
    ptr_read(b.ptr, 0)
}
"""
    assert call(src) == 0


def test_buffer_free_nulls_pointer_and_write_back_reaches_caller():
    # free(self.ptr) then self.ptr = null: the field assignment must survive
    # (dotted-target Assign -> field_set) and the @mut self rebind must write
    # back to the caller's binding.
    src = BUFFER_SRC + """
fn main() -> int {
    let @mut b = Buffer.new(4);
    b.free();
    if b.ptr == null { 1 } else { 0 }
}
"""
    assert call(src) == 1


def test_as_ptr_rejects_non_byte_elements():
    src = EXTERN_HEADER + """
fn main() {
    unsafe {
        let data = vector[int,2](1, 999);
        data.as_ptr()
    }
}
"""
    with pytest.raises(InterpError, match="not a byte"):
        call(src)


# ---------------------------------------------------------------------------
# Builtin Result (Ok/Err) and borrow-annotated pattern bindings
# ---------------------------------------------------------------------------

def test_builtin_result_constructors_and_match():
    # No enum declares Ok/Err: like Option, Result is language-provided.
    src = """
fn classify(x: int) -> Result<int, string> {
    if x >= 0 { Ok(x) } else { Err("negative") }
}

fn main() -> int {
    match classify(41) {
        Ok(v) -> v + 1,
        Err(e) -> 0
    }
}
"""
    assert call(src) == 42


def test_result_err_arm_taken():
    src = """
fn classify(x: int) -> Result<int, string> {
    if x >= 0 { Ok(x) } else { Err("negative") }
}

fn main() -> string {
    match classify(0 - 5) {
        Ok(v) -> "ok",
        Err(e) -> e
    }
}
"""
    assert call(src) == "negative"


def test_mode_annotated_pattern_binding_binds():
    # `Ok(@mut t)` parses its payload as a borrow node; the pattern must
    # still bind t (it used to degrade to a wildcard, leaving t unbound and
    # the whole match arm order broken).
    src = """
struct Thing { n: int }

fn make() -> Result<Thing, string> {
    Ok(Thing { n: 7 })
}

fn main() -> int {
    match make() {
        Ok(@mut t) -> t.n,
        Err(e) -> 0
    }
}
"""
    assert call(src) == 7


# ---------------------------------------------------------------------------
# fopen / fclose shims (real filesystem; null on failure like C)
# ---------------------------------------------------------------------------

FOPEN_SRC = """
extern "C" {
    type FILE;
    fn fopen(filename: *char, mode: *char) -> *FILE;
    fn fclose(file: *FILE) -> int;
}

fn try_open(path: string) -> int {
    unsafe {
        let h = fopen(path.as_ptr(), "r".as_ptr());
        if h == null {
            0
        } else {
            fclose(h);
            1
        }
    }
}
"""


def test_fopen_missing_file_returns_null(tmp_path):
    missing = str(tmp_path / "does_not_exist.txt")
    assert call(FOPEN_SRC, "try_open", [missing]) == 0


def test_fopen_existing_file_opens_and_closes(tmp_path):
    p = tmp_path / "exists.txt"
    p.write_text("hello")
    assert call(FOPEN_SRC, "try_open", [str(p)]) == 1
