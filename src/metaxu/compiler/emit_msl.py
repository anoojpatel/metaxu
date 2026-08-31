"""MSL emission for tile kernels (docs/gpu_tiles.md, Stage 1c/1d).

Compiles ONE kernel function (a `fn k(pid: int, <buffers: Vec...>) -> ()`)
from its MIR into a Metal Shading Language kernel body, plus a
self-checking MLX harness for a Mac.  The container this compiler
develops in has no Metal, so the emitter is designed to be testable
anyway: the emitted body is deliberately **C++-compatible MSL** — control
flow is a block switch-machine (MSL has no goto), buffers are plain
pointers, and the only Metal-ism is `thread_position_in_grid`, which a
five-line shim provides — so the differential tests compile the emitted
body with clang++ (with -ffp-contract=off, pinning the float rounding)
and race it against the interpreter END TO END.  What remains untested
here is exactly the MLX binding and Metal address spaces, which the
generated harness self-checks on a Mac.

The kernel subset (out-of-subset input raises MslError with the reason;
never wrong code):

  * signature `fn k(pid: int, b1: Vec, b2: Vec, ...) -> ()` — parameter 0
    is the instance id, every other parameter is a buffer whose element
    type (long or float) the kernel's own usage decides;
  * int scalars for arithmetic/control flow; INT, F32 and F16 tiles —
    Metal has no f64, so f64 tiles are outside the subset.  The ONLY way
    an f64 tile may appear is as the immediate operand of `Tile.to_f32`
    wrapped around a float-filled masked load: the pair fuses into one
    direct float-buffer load (`Tile.to_f32(Tile.load_rows(v, ..., 0.0))`
    — on CPU that reads f64s and rounds; on the device the buffer IS
    float32, so the fused load is the same op);
  * f16 tiles (Stage 1f) are COMPUTE-ONLY in kernels: buffers stay
    long/float, an f16 tile arises via `Tile.to_f16` of an int/f32 tile
    inside the kernel, and it must convert back through `Tile.to_f32`
    before storing (a half store raises MslError with exactly that fix —
    half buffers wait until a workload needs the bandwidth);
  * float literals are allowed and round exactly like the CPU engines
    (a double literal cast to float once, at the point of use);
  * the MASKED buffer forms only (`Tile.load_or` / `Tile.store_clipped`
    / the row-strided pair): device kernels cannot raise, so the strict
    forms are host-side;
  * tile ops: filled / arange / load_or / store_clipped / load_rows /
    store_rows / add / mul / scale / dot / sum / transpose / rows /
    cols / to_f32 / to_f16; int binops; any control flow the language
    produces (the switch-machine handles the CFG);
  * literal tile shapes (same rule as the native backend's kinds).

f32 rounding parity (docs/gpu_tiles.md Stage 1d): f32 elements are
f32-representable values, and every op rounds once — which is exactly
what `float` arithmetic does, PROVIDED no fused-multiply-add sneaks in.
The C++ shim compiles with -ffp-contract=off so the dot product's
round-product-then-round-accumulate order is bit-identical to the
interpreter; Metal's compiler is fast-math, so the generated Mac harness
compares float buffers with a tolerance instead of bit equality (the
container-side shim differential stays exact).

f16 rounding parity (Stage 1f): the same recipe in `half`.  The C++ shim
maps `half` to clang's `_Float16`, whose arithmetic on x86-64 uses float
excess precision rounded at each assignment/cast — and rounding an exact
half+half or half*half result to float and then to half equals rounding
it to half directly (24 >= 2*11 + 2, the innocuous-double-rounding
bound), so every emitted half op still rounds correctly ONCE.  The dot
product casts the product to half explicitly ((half)(x*y)) so the pinned
round-product-then-round-accumulate order survives excess precision.
Conversions also single-round: (half) of a float equals rounding the
original f64 straight to f16 by the same bound.

Execution contract (documented, enforced by construction): buffer READS
see the launch-entry snapshot, WRITES land in fresh output buffers merged
by a written-mask — so an instance may read what it wrote only through
values it still holds in tiles, and kernels relying on cross-instance
read-after-write within one launch are outside the contract (they would
be racy on any real GPU; the sequential CPU reference would hide that).
The mask merge is what makes `store_clipped`'s partial writes correct
under MLX's const-input model.
"""
from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .mir import MirFunc

_INT_BINOPS = {"+": "+", "-": "-", "*": "*", "/": "/", "%": "%",
               "==": "==", "!=": "!=", "<": "<", "<=": "<=",
               ">": ">", ">=": ">="}

_TILE_OPS = {"filled", "arange", "load_or", "store_clipped", "load_rows",
             "store_rows", "add", "mul", "scale", "dot", "sum",
             "transpose", "rows", "cols", "to_f32", "to_f16"}


class MslError(Exception):
    """The kernel is outside the emitting subset; the message says why."""


def _flit(v: float) -> str:
    """A float value as C++/MSL source: the double literal cast to float
    ONCE — matching the CPU engines, where the literal is an f64 the op
    rounds at the point of use (a direct `0.1f` literal could double-round
    differently)."""
    if not math.isfinite(v):
        raise MslError(f"non-finite float constant {v!r} in a kernel")
    return f"(float)({v!r})"


@dataclass
class MslKernel:
    """An emitted kernel: the body is one C++-compatible MSL fragment."""
    name: str
    pid_var: str
    in_bufs: List[str]           # every buffer param, in signature order
    out_bufs: List[str]          # the subset the kernel writes
    buf_types: Dict[str, str]    # buffer -> "long" | "float"
    body: str                    # references <b>_in / <b>_out / <b>_wm / lens
    uses_half: bool = False      # body declares f16 (`half`) values

    def _bt(self, b: str) -> str:
        return self.buf_types.get(b, "long")

    def cpp_wrapper(self) -> str:
        """A C++ translation unit that runs the body sequentially over the
        grid — the container-side differential harness (no Metal here).
        Compile with -ffp-contract=off: FMA contraction would break the
        pinned f32 rounding order."""
        params = ["uint3 thread_position_in_grid"]
        params += [f"const {self._bt(b)}* {b}_in" for b in self.in_bufs]
        for b in self.out_bufs:
            params += [f"{self._bt(b)}* {b}_out", f"long* {b}_wm"]
        params += ["const long* lens"]
        lines = [
            "// generated by metaxu emit_msl (C++ shim differential harness)",
            "// compile with -ffp-contract=off (f32 rounding parity)",
            "#include <cstdint>",
            "#include <cstdio>",
            "#include <cstdlib>",
            "#include <vector>",
            *([
                "// `half` is MSL-native; the shim maps it to _Float16,",
                "// whose correct rounding is the f16 parity contract —",
                "// an unsupported toolchain must fail loudly here.",
                "#if !defined(__FLT16_MANT_DIG__)",
                "#error \"f16 kernels need _Float16 (clang on x86-64/"
                "arm64 provides it)\"",
                "#endif",
                "typedef _Float16 half;",
            ] if self.uses_half else []),
            "struct uint3 { unsigned x, y, z; };",
            "static_assert(sizeof(long) == 8, \"long must be 64-bit\");",
            f"static void kernel_body({', '.join(params)}) {{",
            self.body,
            "}",
            "",
            "// main: argv = grid_n; stdin = n_bufs, then len + elems per",
            "// buffer (whitespace-separated; float buffers as decimals).",
            "int main(int argc, char** argv) {",
            "    int grid_n = atoi(argv[1]);",
            "    int nb; if (scanf(\"%d\", &nb) != 1) return 2;",
            f"    if (nb != {len(self.in_bufs)}) return 2;",
            f"    std::vector<long> lens({len(self.in_bufs)});",
        ]
        # Per-buffer typed reads, unrolled at generation time.  Int
        # buffers parse %ld (full 64-bit fidelity); float buffers parse
        # %lf into double then cast once (exact for the f32-representable
        # values the CPU engines feed in).
        for i, b in enumerate(self.in_bufs):
            t = self._bt(b)
            lines += [
                f"    if (scanf(\"%ld\", &lens[{i}]) != 1) return 2;",
                f"    std::vector<{t}> buf_{b}(lens[{i}]);",
                f"    for (long j = 0; j < lens[{i}]; j++) {{",
            ]
            if t == "float":
                lines += [
                    "        double x;"
                    " if (scanf(\"%lf\", &x) != 1) return 2;",
                    f"        buf_{b}[j] = (float)x;",
                ]
            else:
                lines += [
                    f"        if (scanf(\"%ld\", &buf_{b}[j]) != 1)"
                    " return 2;",
                ]
            lines.append("    }")
        for b in self.out_bufs:
            i = self.in_bufs.index(b)
            t = self._bt(b)
            lines += [
                f"    std::vector<{t}> out_{b}(lens[{i}], 0);",
                f"    std::vector<long> wm_{b}(lens[{i}], 0);",
            ]
        lines += [
            "    for (int pid = 0; pid < grid_n; pid++) {",
            "        uint3 tpg{(unsigned)pid, 0, 0};",
            "        kernel_body(tpg"
            + "".join(f", buf_{b}.data()" for b in self.in_bufs)
            + "".join(f", out_{b}.data(), wm_{b}.data()"
                      for b in self.out_bufs)
            + ", lens.data());",
            "    }",
            "    // merge: written elements from out, the rest from input",
        ]
        for b in self.out_bufs:
            i = self.in_bufs.index(b)
            lines.append(
                f"    for (long j = 0; j < lens[{i}]; j++)"
                f" if (!wm_{b}[j]) out_{b}[j] = buf_{b}[j];")
        for b in self.out_bufs:
            i = self.in_bufs.index(b)
            if self._bt(b) == "float":
                # %.17g round-trips the widened double exactly, so the
                # differential can compare VALUES, not formatting.
                lines.append(
                    f"    for (long j = 0; j < lens[{i}]; j++)"
                    f" printf(\"%.17g\\n\", (double)out_{b}[j]);")
            else:
                lines.append(
                    f"    for (long j = 0; j < lens[{i}]; j++)"
                    f" printf(\"%ld\\n\", out_{b}[j]);")
        lines += [
            "    return 0;",
            "}",
        ]
        return "\n".join(lines)

    def mlx_harness(self, grid_n: int, buffers: Dict[str, list],
                    expected: Dict[str, list]) -> str:
        """A self-checking Mac harness: binds the SAME body via
        mx.fast.metal_kernel, merges written masks, compares against the
        interpreter-computed expected buffers baked in here.  Int buffers
        compare exactly; float buffers with a tolerance (Metal compiles
        fast-math, so bit equality with the CPU engines is not promised
        there — the container-side C++ shim is the bit-exact leg)."""
        input_names = [f"{b}_in" for b in self.in_bufs] + ["lens"]
        output_names: List[str] = []
        for b in self.out_bufs:
            output_names += [f"{b}_out", f"{b}_wm"]
        lens = [len(buffers[b]) for b in self.in_bufs]
        lines = [
            "#!/usr/bin/env python3",
            f'"""Self-checking Metal harness for kernel {self.name!r},',
            "generated by metaxu emit_msl (docs/gpu_tiles.md Stage 1c).",
            "Run on a Mac with mlx installed:  python3 <this file>",
            "Exit 0 = Metal output matches the interpreter reference.",
            '"""',
            "import sys",
            "try:",
            "    import mlx.core as mx",
            "except ImportError:",
            "    print('mlx not installed (pip install mlx); this harness "
            "needs Apple silicon')",
            "    sys.exit(2)",
            "",
            "SOURCE = r'''",
            self.body,
            "'''",
            "",
            f"kernel = mx.fast.metal_kernel(",
            f"    name={self.name!r},",
            f"    input_names={input_names!r},",
            f"    output_names={output_names!r},",
            "    source=SOURCE,",
            ")",
            "",
            f"GRID_N = {grid_n}",
            f"LENS = {lens!r}",
            "FLOAT_TOL = 1e-5  # Metal is fast-math; CPU shim is bit-exact",
        ]
        for b in self.in_bufs:
            lines.append(f"BUF_{b} = {buffers[b]!r}")
        for b in self.out_bufs:
            lines.append(f"EXPECTED_{b} = {expected[b]!r}")

        def dt(b: str) -> str:
            return "mx.float32" if self._bt(b) == "float" else "mx.int64"

        lines += [
            "",
            "inputs = ["
            + ", ".join(f"mx.array(BUF_{b}, dtype={dt(b)})"
                        for b in self.in_bufs)
            + ", mx.array(LENS, dtype=mx.int64)]",
            "output_shapes = []",
            "output_dtypes = []",
        ]
        for b in self.out_bufs:
            lines += [
                f"output_shapes += [(len(BUF_{b}),), (len(BUF_{b}),)]",
                f"output_dtypes += [{dt(b)}, mx.int64]",
            ]
        lines += [
            "outs = kernel(",
            "    inputs=inputs,",
            "    grid=(GRID_N, 1, 1),",
            "    threadgroup=(min(GRID_N, 32), 1, 1),",
            "    output_shapes=output_shapes,",
            "    output_dtypes=output_dtypes,",
            "    init_value=0,",
            ")",
            "",
            "ok = True",
        ]
        for i, b in enumerate(self.out_bufs):
            lines += [
                f"out = outs[{2 * i}].tolist()",
                f"wm = outs[{2 * i + 1}].tolist()",
                f"merged = [o if m else v for o, m, v in "
                f"zip(out, wm, BUF_{b})]",
            ]
            if self._bt(b) == "float":
                lines += [
                    f"close = all(abs(m - e) <= FLOAT_TOL * max(1.0, abs(e))"
                    f" for m, e in zip(merged, EXPECTED_{b}))",
                    "if not close:",
                    f"    print('FAIL {b}: metal', merged, "
                    f"'!~ expected', EXPECTED_{b})",
                    "    ok = False",
                    "else:",
                    f"    print('OK {b}:', merged)",
                ]
            else:
                lines += [
                    f"if merged != EXPECTED_{b}:",
                    f"    print('FAIL {b}: metal', merged, "
                    f"'!= expected', EXPECTED_{b})",
                    "    ok = False",
                    "else:",
                    f"    print('OK {b}:', merged)",
                ]
        lines += [
            "sys.exit(0 if ok else 1)",
            "",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

def emit_msl_kernel(source: str, kernel_name: str) -> MslKernel:
    """Compile ``kernel_name`` from metaxu ``source`` into an MslKernel.

    Runs the same strict front end as every backend (build context ->
    type/borrow gate -> HIR -> monomorphize -> MIR) and translates the
    kernel's MIR.  Raises MslError for anything outside the subset."""
    from .hir import HIRBuilder
    from .lower_hir_to_mir import lower_hir_to_mir
    from .monomorphize import collect_signatures, monomorphize_hir
    from .pipeline import build_context_from_source, run_pipeline_ctx

    ctx = build_context_from_source(source)
    run_pipeline_ctx(ctx)  # strict gate: production parity
    hir = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    hir = monomorphize_hir(hir, collect_signatures(ctx.id_map))
    funcs = lower_hir_to_mir(hir)
    matches = [f for f in funcs if f.name == kernel_name]
    if not matches:
        near = sorted(f.name for f in funcs if kernel_name in f.name)
        raise MslError(f"kernel {kernel_name!r} not found in the module"
                       + (f" (near: {', '.join(near)})" if near else ""))
    return _emit(matches[0])


def _emit(f: MirFunc) -> MslKernel:
    if not f.blocks or not f.blocks[0].ops \
            or f.blocks[0].ops[0][0] != "params":
        raise MslError(f"kernel {f.name!r} has no parameter list")
    params = list(f.blocks[0].ops[0][1])
    if not params:
        raise MslError(f"kernel {f.name!r} takes no parameters; the first "
                       "must be the instance id `pid: int`")
    pid, bufs = params[0], params[1:]
    bufset = set(bufs)

    consts: Dict[str, int] = {}
    fconsts: Dict[str, float] = {}
    shapes: Dict[str, Tuple[int, int]] = {}  # tile temps -> (rows, cols)
    written: List[str] = []                  # buffers stored to, in order
    decls: Dict[str, str] = {}               # var -> long/float[/[N]]
    buf_types: Dict[str, str] = {}           # buffer -> long/float
    # Float-filled masked loads produce f64 tiles, which do not exist in
    # kernels: each must be consumed by the IMMEDIATELY FOLLOWING
    # Tile.to_f32 in the same block, and the pair fuses into one direct
    # float-buffer load.  f64loads maps the load's dst to (top, args,
    # shape, block, op_index); fused_from maps a to_f32 dst back to it.
    f64loads: Dict[str, Tuple[str, tuple, Tuple[int, int], int, int]] = {}
    fused_from: Dict[str, str] = {}

    def cint(n: str, what: str) -> int:
        if n not in consts:
            raise MslError(f"{what} must be an integer literal in kernels "
                           f"(got the computed value {n!r})")
        return consts[n]

    def shape_of(n: str, what: str) -> Tuple[int, int]:
        if n in f64loads:
            raise MslError(
                f"{what}: {n!r} is an f64 tile (a float-filled load) — "
                "f64 tiles are outside kernels; wrap the load directly: "
                "Tile.to_f32(Tile.load_rows(...))")
        if n not in shapes:
            raise MslError(f"{what}: {n!r} is not a tile the emitter can "
                           "shape statically")
        return shapes[n]

    def ek_of(n: str) -> str:
        """Element type of a declared value: half/float scalars/tiles or
        long."""
        d = decls.get(n, "long")
        if d.startswith("half"):
            return "half"
        return "float" if d.startswith("float") else "long"

    def declare(dst: str, shape: Optional[Tuple[int, int]],
                ek: str = "long") -> None:
        if shape is None:
            want = ek
        else:
            want = f"{ek}[{shape[0] * shape[1]}]"
            if dst in shapes and shapes[dst] != shape:
                raise MslError(f"{dst!r} holds tiles of different shapes "
                               f"({shapes[dst]} vs {shape}); one shape per "
                               "variable in kernels")
            shapes[dst] = shape
        have = decls.get(dst)
        if have is None:
            decls[dst] = want
        elif have != want:
            raise MslError(f"{dst!r} holds values of different kernel types "
                           f"({have} vs {want}); one type per variable in "
                           "kernels")

    def buf_index(n: str, op: str) -> int:
        if n not in bufset:
            raise MslError(f"{op}: buffer argument {n!r} is not a kernel "
                           "buffer parameter")
        return bufs.index(n)

    def set_buf_type(b: str, ek: str, op: str) -> None:
        have = buf_types.get(b)
        if have is None:
            buf_types[b] = ek
        elif have != ek:
            raise MslError(f"{op}: buffer {b!r} is used as both {have} and "
                           f"{ek} elements; one element type per buffer")

    def tile_binop_ek(top: str, a: str, b: str) -> str:
        ea, eb = ek_of(a), ek_of(b)
        if ea != eb:
            raise MslError(f"Tile.{top}: element kinds differ in a kernel "
                           f"({ea} vs {eb} tiles)")
        return ea

    # Pass 1: collect consts, shapes, declarations, buffer element types,
    # written buffers, fuse float loads into their to_f32, and validate
    # the subset.  Pass 2 emits, so forward-referenced facts (like which
    # buffers are written) are complete.
    for bi_, b in enumerate(f.blocks):
        for oi_, op in enumerate(b.ops):
            if op[0] == "params":
                continue
            if op[0] != "let" or len(op) != 4:
                raise MslError(f"kernel op {op[0]!r} is outside the MSL "
                               "subset")
            _, dst, rhs, args = op
            rk = rhs[0]
            if rk == "const":
                v = rhs[1]
                if isinstance(v, bool):
                    raise MslError(f"non-integer constant {v!r} in a kernel")
                if isinstance(v, float):
                    fconsts[dst] = v
                    _flit(v)  # rejects non-finite now, loudly
                    declare(dst, None, "float")
                elif isinstance(v, int):
                    consts[dst] = v
                    declare(dst, None)
                else:
                    raise MslError(f"non-numeric constant {v!r} in a kernel")
            elif rk == "const_ty":
                declare(dst, None)  # unit -> 0
            elif rk == "binop":
                if rhs[1] not in _INT_BINOPS:
                    raise MslError(f"binop {rhs[1]!r} is outside the MSL "
                                   "subset")
                for a in args:
                    if a in shapes or a in f64loads:
                        raise MslError("elementwise operators on tiles use "
                                       "Tile.add/Tile.mul in kernels")
                    if ek_of(a) in ("float", "half"):
                        raise MslError(
                            "float scalar arithmetic is outside the kernel "
                            "subset (float values flow through tile ops; "
                            "scalar control flow stays int)")
                declare(dst, None)
            elif rk == "copy":
                src = args[0]
                if src in f64loads:
                    raise MslError(
                        "an f64 tile (a float-filled load) can only be "
                        "consumed by Tile.to_f32 — wrap the load directly: "
                        "Tile.to_f32(Tile.load_rows(...))")
                if src in consts:
                    consts[dst] = consts[src]
                if src in fconsts:
                    fconsts[dst] = fconsts[src]
                declare(dst, shapes.get(src), ek_of(src))
            elif rk == "call" and len(rhs) > 1 \
                    and str(rhs[1]).startswith("Tile."):
                top = str(rhs[1])[len("Tile."):]
                if top not in _TILE_OPS:
                    hint = ""
                    if top in ("load", "store", "get", "from_vec",
                               "to_vec"):
                        hint = (" (kernels use the masked forms "
                                "Tile.load_or / Tile.store_clipped — "
                                "device code cannot raise)")
                    elif top == "zeros":
                        hint = (" (Tile.zeros builds f64 tiles and Metal "
                                "has no f64 — use Tile.filled(r, c, 0), "
                                "or Tile.to_f32(Tile.filled(r, c, 0)) for "
                                "an f32 accumulator)")
                    elif top == "to_f64":
                        hint = (" (Metal has no f64; kernels stay in "
                                "int/f32/f16)")
                    raise MslError(
                        f"Tile.{top} is outside the kernel subset{hint}")
                if top == "filled":
                    if ek_of(args[2]) == "float":
                        raise MslError(
                            "Tile.filled with a float builds an f64 tile "
                            "and Metal has no f64 — use "
                            "Tile.to_f32(Tile.filled(r, c, 0)) and "
                            "Tile.scale for f32 accumulators")
                    declare(dst, (cint(args[0], "Tile.filled rows"),
                                  cint(args[1], "Tile.filled cols")))
                elif top == "arange":
                    declare(dst, (cint(args[0], "Tile.arange rows"),
                                  cint(args[1], "Tile.arange cols")))
                elif top == "load_or":
                    buf_index(args[0], "Tile.load_or")
                    shape = (cint(args[2], "Tile.load_or rows"),
                             cint(args[3], "Tile.load_or cols"))
                    if ek_of(args[4]) == "float":
                        set_buf_type(args[0], "float", "Tile.load_or")
                        f64loads[dst] = (top, tuple(args), shape, bi_, oi_)
                    else:
                        set_buf_type(args[0], "long", "Tile.load_or")
                        declare(dst, shape)
                elif top == "load_rows":
                    buf_index(args[0], "Tile.load_rows")
                    shape = (cint(args[3], "Tile.load_rows rows"),
                             cint(args[4], "Tile.load_rows cols"))
                    if ek_of(args[5]) == "float":
                        set_buf_type(args[0], "float", "Tile.load_rows")
                        f64loads[dst] = (top, tuple(args), shape, bi_, oi_)
                    else:
                        set_buf_type(args[0], "long", "Tile.load_rows")
                        declare(dst, shape)
                elif top == "to_f32":
                    src = args[0]
                    if src in f64loads:
                        # THE FUSION: the to_f32 must immediately follow
                        # its load in the same block (which is exactly
                        # what `Tile.to_f32(Tile.load_rows(...))`
                        # lowers to) — anything looser could re-order a
                        # load across redefinitions of its arguments.
                        _t, _a, shape, lb, lo = f64loads[src]
                        if lb != bi_ or oi_ != lo + 1:
                            raise MslError(
                                "Tile.to_f32 must directly wrap a "
                                "float-filled load in kernels "
                                "(Tile.to_f32(Tile.load_rows(...))); "
                                f"the load of {src!r} is separated from "
                                "its conversion")
                        fused_from[dst] = src
                        declare(dst, shape, "float")
                    else:
                        declare(dst, shape_of(src, "Tile.to_f32"), "float")
                elif top == "to_f16":
                    # f16 tiles are compute-only: they arise HERE, from an
                    # int/f32 tile already in the kernel (a float-filled
                    # load still fuses into Tile.to_f32 — shape_of raises
                    # with that fix if one reaches us).
                    declare(dst, shape_of(args[0], "Tile.to_f16"), "half")
                elif top == "store_clipped":
                    bi = buf_index(args[0], "Tile.store_clipped")
                    shape_of(args[2], "Tile.store_clipped")
                    if ek_of(args[2]) == "half":
                        raise MslError(
                            "Tile.store_clipped of an f16 tile: f16 tiles "
                            "are compute-only in kernels (buffers stay "
                            "long/float) — convert back first: "
                            "Tile.store_clipped(v, off, Tile.to_f32(t))")
                    set_buf_type(args[0], ek_of(args[2]),
                                 "Tile.store_clipped")
                    if bufs[bi] not in written:
                        written.append(bufs[bi])
                    declare(dst, None)  # unit
                elif top == "store_rows":
                    bi = buf_index(args[0], "Tile.store_rows")
                    shape_of(args[3], "Tile.store_rows")
                    if ek_of(args[3]) == "half":
                        raise MslError(
                            "Tile.store_rows of an f16 tile: f16 tiles "
                            "are compute-only in kernels (buffers stay "
                            "long/float) — convert back first: "
                            "Tile.store_rows(v, off, stride, "
                            "Tile.to_f32(t))")
                    set_buf_type(args[0], ek_of(args[3]), "Tile.store_rows")
                    if bufs[bi] not in written:
                        written.append(bufs[bi])
                    declare(dst, None)  # unit
                elif top in ("add", "mul"):
                    sa = shape_of(args[0], f"Tile.{top}")
                    sb = shape_of(args[1], f"Tile.{top}")
                    if sa != sb:
                        raise MslError(f"Tile.{top}: shape mismatch "
                                       f"{sa} vs {sb}")
                    declare(dst, sa, tile_binop_ek(top, args[0], args[1]))
                elif top == "scale":
                    sh = shape_of(args[0], "Tile.scale")
                    ek = ek_of(args[0])
                    sk = ek_of(args[1])
                    if ek == "half":
                        # Language scalars are f64 and kernel float
                        # consts are float: the factor rounds once more
                        # to half at the use ((half) of a float equals
                        # rounding the f64 straight to f16 — module
                        # docstring).
                        if sk != "float":
                            raise MslError(
                                f"Tile.scale: half tile scaled by a "
                                f"{sk} scalar in a kernel (f16 tiles "
                                "scale by float literals)")
                    elif sk != ek:
                        raise MslError(
                            f"Tile.scale: {ek} tile scaled by a "
                            f"{sk} scalar in a kernel (f32 "
                            "tiles scale by float literals, int tiles by "
                            "ints)")
                    declare(dst, sh, ek)
                elif top == "dot":
                    ra, ca = shape_of(args[0], "Tile.dot")
                    rb, cb = shape_of(args[1], "Tile.dot")
                    if ca != rb:
                        raise MslError(f"Tile.dot: inner dims {ca} and "
                                       f"{rb} disagree")
                    declare(dst, (ra, cb),
                            tile_binop_ek("dot", args[0], args[1]))
                elif top == "transpose":
                    r, c = shape_of(args[0], "Tile.transpose")
                    declare(dst, (c, r), ek_of(args[0]))
                elif top == "sum":
                    ek = ek_of(args[0])
                    shape_of(args[0], "Tile.sum")
                    declare(dst, None, ek)
                elif top in ("rows", "cols"):
                    r, c = shape_of(args[0], f"Tile.{top}")
                    consts[dst] = r if top == "rows" else c
                    declare(dst, None)
            else:
                what = rhs[1] if rk == "call" and len(rhs) > 1 else rk
                raise MslError(f"kernel op {what!r} is outside the MSL "
                               "subset (kernels are tile programs: tile "
                               "ops, int arithmetic and control flow)")
        t = b.term
        if t[0] not in ("br", "br_if", "ret", "unreachable"):
            raise MslError(f"terminator {t[0]!r} is outside the MSL subset")

    # Every float-filled load must have fused into a to_f32.
    fused_loads = set(fused_from.values())
    for lv in f64loads:
        if lv not in fused_loads:
            raise MslError(
                f"the float-filled load {lv!r} produces an f64 tile, "
                "which is outside kernels — wrap it directly: "
                "Tile.to_f32(Tile.load_rows(...))")

    # Pass 2: emit the switch-machine body.
    L: List[str] = []
    L.append(f"const long {pid} = (long)thread_position_in_grid.x;")
    for name, ty in sorted(decls.items()):
        if "[" not in ty:
            L.append(f"{ty} {name} = 0;")
        else:  # long[N] / float[N]
            base, n = ty[:-1].split("[")
            L.append(f"{base} {name}[{n}] = {{0}};")
    L.append("int __bb = 0;")
    L.append("bool __run = true;")
    L.append("while (__run) { switch (__bb) {")

    def loop(dst: str, n: int, expr: str) -> List[str]:
        return [f"for (int __i = 0; __i < {n}; __i++) "
                f"{dst}[__i] = {expr};"]

    def emit_masked_load(dst: str, top: str, largs: tuple) -> None:
        """Emit load_or/load_rows into ``dst`` (also the fused-float
        form, where dst is the to_f32 result and the buffer is float)."""
        r, c = shapes[dst]
        bi = bufs.index(largs[0])
        if top == "load_or":
            L.append(f"for (int __i = 0; __i < {r * c}; __i++) {{")
            L.append(f"  long __j = {largs[1]} + __i;")
            L.append(f"  {dst}[__i] = (__j >= 0 && __j < "
                     f"lens[{bi}]) ? {largs[0]}_in[__j] : "
                     f"{largs[4]};")
            L.append("}")
        else:  # load_rows
            L.append(f"for (int __r = 0; __r < {r}; __r++) "
                     f"for (int __c = 0; __c < {c}; __c++) {{")
            L.append(f"  long __j = {largs[1]} + __r * {largs[2]} "
                     f"+ __c;")
            L.append(f"  {dst}[__r * {c} + __c] = (__j >= 0 && "
                     f"__j < lens[{bi}]) ? {largs[0]}_in[__j] : "
                     f"{largs[5]};")
            L.append("}")

    for bi_, b in enumerate(f.blocks):
        L.append(f"case {bi_}: {{")
        for op in b.ops:
            if op[0] == "params":
                continue
            _, dst, rhs, args = op
            rk = rhs[0]
            if rk == "const":
                if dst in fconsts:
                    L.append(f"{dst} = {_flit(fconsts[dst])};")
                else:
                    L.append(f"{dst} = {rhs[1]};")
            elif rk == "const_ty":
                L.append(f"{dst} = 0;")
            elif rk == "binop":
                o = _INT_BINOPS[rhs[1]]
                if o in ("==", "!=", "<", "<=", ">", ">="):
                    L.append(f"{dst} = (long)({args[0]} {o} {args[1]});")
                else:
                    L.append(f"{dst} = {args[0]} {o} {args[1]};")
            elif rk == "copy":
                if dst in shapes:
                    r, c = shapes[dst]
                    L += loop(dst, r * c, f"{args[0]}[__i]")
                else:
                    L.append(f"{dst} = {args[0]};")
            else:  # Tile.*
                top = str(rhs[1])[len("Tile."):]
                if dst in f64loads:
                    # a fused float load: emitted at its to_f32 below
                    continue
                if top == "filled":
                    r, c = shapes[dst]
                    L += loop(dst, r * c, args[2])
                elif top == "arange":
                    r, c = shapes[dst]
                    L += loop(dst, r * c, "(long)__i")
                elif top in ("load_or", "load_rows"):
                    emit_masked_load(dst, top, tuple(args))
                elif top == "to_f32":
                    if dst in fused_from:
                        ltop, largs, _s, _b, _o = f64loads[fused_from[dst]]
                        emit_masked_load(dst, ltop, largs)
                    else:
                        # (float) of a half widens exactly; of a long it
                        # rounds once, matching the CPU engines.
                        r, c = shapes[dst]
                        src_ek = ek_of(args[0])
                        expr = (f"{args[0]}[__i]" if src_ek == "float"
                                else f"(float){args[0]}[__i]")
                        L += loop(dst, r * c, expr)
                elif top == "to_f16":
                    # (half) of a float or long rounds once — equal to
                    # rounding the CPU engines' f64 straight to f16
                    # (module docstring).
                    r, c = shapes[dst]
                    src_ek = ek_of(args[0])
                    expr = (f"{args[0]}[__i]" if src_ek == "half"
                            else f"(half){args[0]}[__i]")
                    L += loop(dst, r * c, expr)
                elif top == "store_clipped":
                    r, c = shapes[args[2]]
                    bi = bufs.index(args[0])
                    bn = args[0]
                    L.append(f"for (int __i = 0; __i < {r * c}; __i++) {{")
                    L.append(f"  long __j = {args[1]} + __i;")
                    L.append(f"  if (__j >= 0 && __j < lens[{bi}]) {{")
                    L.append(f"    {bn}_out[__j] = {args[2]}[__i];")
                    L.append(f"    {bn}_wm[__j] = 1;")
                    L.append("  }")
                    L.append("}")
                    L.append(f"{dst} = 0;")
                elif top == "store_rows":
                    r, c = shapes[args[3]]
                    bi = bufs.index(args[0])
                    bn = args[0]
                    L.append(f"for (int __r = 0; __r < {r}; __r++) "
                             f"for (int __c = 0; __c < {c}; __c++) {{")
                    L.append(f"  long __j = {args[1]} + __r * {args[2]} "
                             f"+ __c;")
                    L.append(f"  if (__j >= 0 && __j < lens[{bi}]) {{")
                    L.append(f"    {bn}_out[__j] = {args[3]}[__r * {c} "
                             f"+ __c];")
                    L.append(f"    {bn}_wm[__j] = 1;")
                    L.append("  }")
                    L.append("}")
                    L.append(f"{dst} = 0;")
                elif top in ("add", "mul"):
                    o = "+" if top == "add" else "*"
                    r, c = shapes[dst]
                    L += loop(dst, r * c,
                              f"{args[0]}[__i] {o} {args[1]}[__i]")
                elif top == "scale":
                    r, c = shapes[dst]
                    if ek_of(dst) == "half":
                        # The float factor rounds once to half at the
                        # use; the product then rounds per element at
                        # the assignment.
                        L += loop(dst, r * c,
                                  f"{args[0]}[__i] * (half){args[1]}")
                    else:
                        L += loop(dst, r * c, f"{args[0]}[__i] * {args[1]}")
                elif top == "dot":
                    ra, ca = shapes[args[0]]
                    _rb, cb = shapes[args[1]]
                    acc_ty = ek_of(dst)
                    zero = {"float": "0.0f", "half": "(half)0",
                            "long": "0"}[acc_ty]
                    L.append(f"for (int __r = 0; __r < {ra}; __r++) "
                             f"for (int __c = 0; __c < {cb}; __c++) {{")
                    L.append(f"  {acc_ty} __acc = {zero};")
                    if acc_ty == "half":
                        # Explicit rounding steps: _Float16 excess
                        # precision (float) would otherwise evaluate
                        # acc + x*y in float and round ONCE — breaking
                        # the pinned round-product-then-round-accumulate
                        # order (module docstring).
                        L.append(f"  for (int __k = 0; __k < {ca}; __k++) "
                                 f"__acc = (half)(__acc + "
                                 f"(half)({args[0]}[__r * {ca} + __k] * "
                                 f"{args[1]}[__k * {cb} + __c]));")
                    else:
                        L.append(f"  for (int __k = 0; __k < {ca}; __k++) "
                                 f"__acc += {args[0]}[__r * {ca} + __k] * "
                                 f"{args[1]}[__k * {cb} + __c];")
                    L.append(f"  {dst}[__r * {cb} + __c] = __acc;")
                    L.append("}")
                elif top == "transpose":
                    r, c = shapes[args[0]]  # source shape
                    L.append(f"for (int __r = 0; __r < {r}; __r++) "
                             f"for (int __c = 0; __c < {c}; __c++) "
                             f"{dst}[__c * {r} + __r] = "
                             f"{args[0]}[__r * {c} + __c];")
                elif top == "sum":
                    r, c = shapes[args[0]]
                    L.append(f"{dst} = 0;")
                    L.append(f"for (int __i = 0; __i < {r * c}; __i++) "
                             f"{dst} += {args[0]}[__i];")
                elif top in ("rows", "cols"):
                    L.append(f"{dst} = {consts[dst]};")
        t = b.term
        if t[0] == "br":
            L.append(f"__bb = {t[1]};")
        elif t[0] == "br_if":
            L.append(f"__bb = {t[1]} ? {t[2]} : {t[3]};")
        else:  # ret / unreachable
            L.append("__run = false;")
        L.append("} break;")
    L.append("} }")

    return MslKernel(name=f.name, pid_var=pid, in_bufs=bufs,
                     out_bufs=written,
                     buf_types={b: buf_types.get(b, "long") for b in bufs},
                     body="\n".join(L),
                     uses_half=any(d.startswith("half")
                                   for d in decls.values()))
