# GPU tiles

GPU kernels in Metaxu are written at the tile level, the way Triton
kernels are: programs over small matrices with the shape fixed at
compile time, not per-thread code. A `Tile` is an immutable value:
`Tile.arange(2, 3)` is a 2x3 row-major iota, `Tile.filled(r, c, x)`
broadcasts a scalar, and every op produces a fresh tile:

```metaxu
fn main() -> int {
    let a = Tile.arange(2, 3);
    print(a);
    let b = Tile.transpose(a);
    let c = Tile.dot(a, b);
    print(c);
    print(Tile.sum(c));
    let z = Tile.filled(2, 2, 1.5);
    print(Tile.sum(Tile.scale(z, 2.0)));
    0
}
```
```output
tile[2x3](0, 1, 2; 3, 4, 5)
tile[2x2](5, 14; 14, 50)
83
12.0
```

`scale` gave back a new tile and left `z` untouched; every op does.
Accumulation order is part of the semantics too: `dot` rounds each
product, then accumulates in a pinned order, and `sum` reduces
row-major. That's what lets three engines (interpreter, native
runtime, the Metal path below) agree bit for bit on floats.

## Shapes are checked at compile time

The shape lives in the tile's type, so mismatched elementwise ops,
disagreeing inner dimensions in `dot`, statically out-of-range `get`
indices, and non-positive constructor shapes are compile errors:

```metaxu error
fn main() -> int {
    let a = Tile.zeros(2, 3);
    let b = Tile.zeros(3, 2);
    Tile.sum(Tile.add(a, b));
    0
}
```
```output
Tile.add: shape mismatch: 2x3 vs 3x2
```

The checker promises zero false positives: rebind a tile inside a
branch and the shape becomes unknown, the program compiles, and the
same checks fire at run time as catchable errors, same wording.

## Element kinds and rounding

Tiles come in four element kinds: `int`, `f64` (any float literal),
and the GPU-oriented `f32` and `f16`, reached only through
`Tile.to_f32` and `Tile.to_f16`. Kinds never mix; adding an f32 tile
to an f64 tile is a compile error, and f16 mixes with nothing.

The narrow kinds have an honest bit-exact story. An f32 element is
stored as the f32-representable double, and every op rounds its
result to f32 once; that single rounding is exactly the
correctly-rounded f32 operation, because arithmetic on two
f32-representables is exact in f64 (24+24 significand bits fit under
53; for f16, 11+11 does). All engines implement that rule, so these
are the pinned outputs of real f32 and f16 arithmetic:

```metaxu
fn main() -> int {
    let f = Tile.to_f32(Tile.filled(2, 2, 0.1));
    print(f);
    print(Tile.sum(f));
    print(Tile.to_f16(Tile.filled(1, 1, 0.1)));
    0
}
```
```output
tile[2x2](0.10000000149011612, 0.10000000149011612; 0.10000000149011612, 0.10000000149011612)
0.4000000059604645
tile[1x1](0.0999755859375)
```

## Buffers in, tiles out

Kernels meet memory through `Vec` buffers. `Tile.load(v, off, r, c)`
reads r*c elements starting at `off`, `Tile.store(v, off, t)` writes
them back, and both raise catchably on an out-of-range window. The
masked forms are the kernel-side idiom: `Tile.load_or` reads a
default past the end, `Tile.store_clipped` drops out-of-range writes.
Row-strided forms (`Tile.load_rows`, `Tile.store_rows`) handle 2D
tiles inside a larger row-major matrix.

```metaxu
fn main() -> int {
    let @mut v = Vec.new();
    let mut i = 0;
    while i < 10 { v.push(i * i); i = i + 1 };
    let t = Tile.load(v, 2, 2, 3);
    print(t);
    let edge = Tile.load_or(v, 8, 1, 4, 0 - 1);
    print(edge);
    Tile.store_clipped(v, 8, Tile.filled(1, 4, 7));
    print(v[9]);
    0
}
```
```output
tile[2x3](4, 9, 16; 25, 36, 49)
tile[1x4](64, 81, -1, -1)
7
```

Tile stores are Vec writes, so the contended-write rule from chapter
11 applies unchanged: a spawned kernel writing a crossed buffer
without a lock raises the same error as any other Vec mutation.

## Kernels and the launch effect

A kernel is an ordinary function taking an instance id `pid` and its
buffers. Launching it is an effect, `Gpu.launch(n, f)` from `std.gpu`,
whose default handler runs instances sequentially in pid order 0..n-1
on the CPU. That default is the reference semantics, so every kernel
is a deterministic program you can run and pin before any GPU is
involved.

```metaxu
from std.gpu import Gpu, run_grid;

fn vecadd_kernel(pid: int, a: Vec, b: Vec, out: Vec) -> () {
    let ta = Tile.load(a, pid * 4, 1, 4);
    let tb = Tile.load(b, pid * 4, 1, 4);
    Tile.store(out, pid * 4, Tile.add(ta, tb));
    ()
}

fn main() -> int {
    let @mut a = Vec.new();
    let @mut b = Vec.new();
    let @mut out = Vec.new();
    let mut i = 0;
    while i < 8 { a.push(i); b.push(i * 10); out.push(0); i = i + 1 };
    perform Gpu.launch(2, fn(pid: int) -> vecadd_kernel(pid, a, b, out));
    print(out[3]);
    print(out[7]);
    0
}
```
```output
33
77
```

Because the launch is an effect, a handler can observe or replace it
without touching kernel code; that's how tests interpose recording
harnesses, and how backends arrive:

```metaxu
from std.gpu import Gpu, run_grid;

fn main() -> int {
    let @mut out = Vec.new();
    let mut i = 0;
    while i < 4 { out.push(0); i = i + 1 };
    handle Gpu with {
        launch(n, f) -> {
            print("launch of " + n.to_string() + " instances");
            run_grid(n, f);
            resume(())
        }
    } in {
        perform Gpu.launch(4, fn(pid: int) ->
            Tile.store(out, pid, Tile.filled(1, 1, pid * pid)));
        ()
    };
    print(out[3]);
    0
}
```
```output
launch of 4 instances
9
```

## Metal, as a handler

The Metal backend is exactly such a handler's tool. `run_metal(n, f)`
introspects the closure, which must be the canonical launch idiom
(`fn(pid) -> kernel(pid, buffers...)`: a named kernel, captured Vec
buffers), emits the kernel's Metal Shading Language, and dispatches
on the best engine available. On Apple silicon that's
`mx.fast.metal_kernel` through MLX; elsewhere it's a clang++ shim,
bit-exact with the CPU reference, which is what lets this book's
harness run and pin the example below in a Linux container with no
GPU:

```metaxu
from std.gpu import Gpu, run_grid, Metal, run_metal;

fn double_kernel(pid: int, v: Vec) -> () {
    let t = Tile.load_or(v, pid * 4, 1, 4, 0);
    Tile.store_clipped(v, pid * 4, Tile.scale(t, 2));
    ()
}

fn main() -> int {
    let @mut v = Vec.new();
    let mut i = 0;
    while i < 10 { v.push(i + 1); i = i + 1 };
    handle Gpu with {
        launch(n, f) -> { run_metal(n, f); resume(()) }
    } in {
        perform Gpu.launch(3, fn(pid: int) -> double_kernel(pid, v));
        ()
    };
    print(v[9]);
    0
}
```
```output
20
```

The kernel didn't change; only the handler did. Two rules of the
Metal contract matter. Kernels use the masked load/store forms,
because device kernels can't raise (use a strict form and the error
message says exactly that). And instances read buffers as they were
at launch entry, writes mask-merged afterward: cross-instance
read-after-write inside one launch is outside the contract, racy on a
real GPU and hidden by the sequential reference. Everything outside
the contract, a kernel outside the MSL subset, a non-canonical
closure, a missing engine, is a loud catchable error, never a silent
CPU fallback. For an actual Mac,
`scripts/emit_metal_harness.py` generates a self-checking harness
that bakes in the interpreter's expected outputs and exits 0 only if
the Metal results match:

```bash
uv run python scripts/emit_metal_harness.py kernels.mx mm_kernel \
    --grid 4 --buf a=0,1,2,3 --buf out=0,0,0,0 -o harness_mm.py
python3 harness_mm.py    # on a Mac with mlx installed
```

## Operations as functions

The compiler knows a handful of structural tile operations and no
more. Everything elementwise is a function you pass in: `Tile.map(t,
f)` applies `f` to every element, `Tile.zip(t, u, f)` pairs two tiles
of one shape, `Tile.reduce_rows(t, init, f)` and `Tile.reduce_cols`
fold each row or column left to right into an R x 1 or 1 x C tile, and
`Tile.broadcast_rows(t, v, f)` and `Tile.broadcast_cols` combine each
element with its row's or column's entry of a vector tile. The
function sees plain `float` or `int` values, and the result tile has
the receiver's kind.

Elements of f32 and f16 tiles reach the function in narrow mode: the
value remembers its width, and every operation and math call on it
rounds the result to that width once, so `x.exp() * 0.5` inside a
`Tile.map` over an f32 tile computes what f32 arithmetic computes,
with no new type syntax. `std.tile` is the vocabulary built this way,
one line per operation; the row sums of a softmax below show the
rounding at work, since one of them is the f32 just under 1:

```metaxu
from std.tile import softmax_rows, row_max, row_sum;

fn main() -> int {
    let @mut v = Vec.new();
    v.push(1.0); v.push(2.0); v.push(3.0);
    v.push(0.5); v.push(0.5); v.push(-4.0);
    let t = Tile.to_f32(Tile.from_vec(v, 2, 3));
    print(row_max(t));
    print(softmax_rows(t));
    print(row_sum(softmax_rows(t)));
    print(Tile.map(t, fn(x: float) -> x * 0.5 + 1.0));
    print(Tile.reduce_cols(Tile.arange(2, 3), 0, fn(a: int, b: int) -> a + b));
    0
}
```
```output
tile[2x1](3.0; 0.5)
tile[2x3](0.09003057330846786, 0.2447284758090973, 0.6652409434318542; 0.49723806977272034, 0.49723806977272034, 0.005523815751075745)
tile[2x1](1.0; 0.9999999403953552)
tile[2x3](1.5, 2.0, 2.5; 1.25, 1.25, -1.0)
tile[1x3](3, 5, 7)
```

Inside a kernel these calls cost nothing: the Metal emitter inlines
the library functions and the lambdas into one device program, so
`softmax_rows(t)` becomes three loops over a threadgroup array. The
same words run on the interpreter, in the C++ shim and on the device,
and the interpreter remains the bit-exact reference for the shim.

## Matrix units

A kernel whose `Tile.dot` multiplies 8x8 f32 or f16 tiles gets a
second lowering, chosen automatically: one launch instance becomes one
32-lane simdgroup instead of one thread, its tiles live in threadgroup
memory, and the dot is the collective `simdgroup_load`,
`simdgroup_multiply_accumulate`, `simdgroup_store` on a
`simdgroup_float8x8`, which is what Apple's matrix units execute. The
kernel and the `Gpu.launch` idiom do not change; `pid` still names the
instance, and every instance still computes its own output block. The
16x16 matmul below runs each of its four blocks through two 8x8 dots,
and the printed corner is the plain sum it should be:

```metaxu
from std.gpu import Gpu, run_grid, Metal, run_metal;

fn block_kernel(pid: int, a: Vec, b: Vec, c: Vec) -> () {
    let ti = pid / 2;
    let tj = pid % 2;
    let mut k = 0;
    let mut r = Tile.to_f32(Tile.filled(8, 8, 0));
    while k < 2 {
        let ta = Tile.to_f32(
            Tile.load_rows(a, (ti * 8) * 16 + k * 8, 16, 8, 8, 0.0));
        let tb = Tile.to_f32(
            Tile.load_rows(b, (k * 8) * 16 + tj * 8, 16, 8, 8, 0.0));
        r = Tile.add(r, Tile.dot(ta, tb));
        k = k + 1
    };
    Tile.store_rows(c, (ti * 8) * 16 + tj * 8, 16, r);
    ()
}

fn main() -> int {
    let @mut a = Vec.new();
    let @mut b = Vec.new();
    let @mut c = Vec.new();
    let mut i = 0;
    while i < 256 {
        a.push(1.0);
        b.push(if i % 17 == 0 { 2.0 } else { 0.0 });
        c.push(0.0);
        i = i + 1
    };
    handle Gpu with {
        launch(n, f) -> { run_metal(n, f); resume(()) }
    } in {
        perform Gpu.launch(4, fn(pid: int) -> block_kernel(pid, a, b, c));
        ()
    };
    print(c[0]);
    print(c[255]);
    0
}
```
```output
2.0
2.0
```

The shim leg stays bit-exact: it emulates each collective as the
whole-tile operation the interpreter defines, in the same rounding
order. The device's matrix unit fuses the multiply and the add, so on
a Mac the float tolerance the harness already applies is where that
difference lands; a kernel that needs bit-exact device results can
force the per-thread lowering with `METAXU_METAL_LOWERING=thread`.
In the simdgroup lowering the higher-order operations are strided
across the 32 lanes, each lane taking every 32nd element, behind a
barrier; the older fixed operations still run on one lane. Inferred
layouts and threadgroup-memory tiling across instances are the rest of
Stage 2 (`docs/simdgroup_plan.md`).

## Attention

With the matrix units for the two products and `std.tile` for the
online softmax between them, FlashAttention-2 is library code:
`std.attention.attention8` owns one block of 8 query rows and streams
the keys and values past it 8 at a time, keeping a running row maximum,
a running denominator and an unnormalized output, and rescaling them
when the maximum moves. The inner step is the algorithm as written in
the paper:

```metaxu norun
let s = Tile.scale(Tile.dot(qb, Tile.transpose(kb)), qk_scale8());
let m_new = maximum(m, row_max(s));
let p = exp(sub_rows(s, m_new));
let alpha = exp(sub(m, m_new));
l = Tile.add(Tile.mul(l, alpha), row_sum(p));
o = Tile.add(mul_rows(o, alpha), Tile.dot(p, vb));
m = m_new;
```

`causal_attention8` skips the key blocks past the query block and
masks the upper triangle of the diagonal one with a flag tile built
from `Tile.arange`. Both are launched with `Gpu.launch(n / 8, ...)`
over row-major [n, 8] buffers, and both are checked against
`attention_ref`, the plain f64 definition in the same module, and bit
for bit against the shim in either lowering (`test_std_attention.py`).
Head dimension 8 is literal in this version; other block shapes are
copies with other literals until const generics reach kernels.
