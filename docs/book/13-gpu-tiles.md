# GPU tiles

A `Tile` is a small matrix with a fixed shape and a fixed element
kind, the unit GPU kernels compute on. Tile operations run on the
CPU as ordinary values, kernels dispatch through a `Gpu.launch`
effect whose default handler is a plain loop, and on a Mac a
different handler sends the same kernel to Metal. Nothing here
needs a GPU except the last section.

## Tiles are values

`Tile.arange(r, c)` builds an `r` by `c` tile counting up from 0,
row-major. `Tile.filled(r, c, v)` repeats one element. Every
operation returns a new tile; nothing mutates in place.

```metaxu
fn main() -> int {
    let a = Tile.arange(2, 3);
    let b = Tile.filled(2, 3, 10);
    print(a);
    let s = a.add(b);
    print(s.get(0, 0), s.get(1, 2));
    print(a.get(0, 0));
    print(s.sum());
    let m = a.dot(a.transpose());
    print(m.get(0, 0), m.get(0, 1), m.get(1, 0), m.get(1, 1));
    print(a.mul(a).scale(2).sum());
    0
}
```
```output
Tile[[0, 1, 2], [3, 4, 5]]
10 15
0
75
5 14 14 50
110
```

`add` and `mul` are elementwise, `scale` multiplies by a scalar,
`dot` is matrix multiplication (2x3 times its 3x2 transpose gives
2x2), `get(r, c)` reads one element, `sum` folds to a scalar. The
third output line shows the functional semantics: `a` still starts
with 0 after `add` built `s` from it.

Order is part of the contract. `sum` accumulates row-major, left
to right; `dot` walks its inner dimension in index order. For ints
you can't tell. For floats you can, so the order is pinned and
every handler accumulates the same way: never a "close enough"
float.

## Shapes are compile errors

A tile's shape is in its type. Combining mismatched shapes is
rejected before anything runs:

```metaxu error
fn main() -> int {
    let a = Tile.arange(2, 3);
    let b = Tile.arange(3, 2);
    let c = a.add(b);
    print(c.sum());
    0
}
```
```output
shape mismatch: 2x3 vs 3x2
```

The same check accepts `a.dot(b)`: 2x3 against 3x2 is exactly what
`dot` wants. By launch time the indexing is already checked.

## Element kinds

Tiles carry one of four element kinds: `int`, `f64`, `f32`, `f16`;
the narrow ones are what GPUs are fastest in. `to_f32` and
`to_f16` convert a scalar, and a tile built from one is of that
kind:

```metaxu
fn main() -> int {
    print(to_f32(0.1));
    print(to_f16(0.1));
    let t = Tile.filled(1, 2, to_f16(0.1));
    print(t.sum());
    0
}
```
```output
0.10000000149011612
0.0999755859375
0.199951171875
```

Those long decimals are not noise; they are the exact values the
narrow formats hold. Both engines simulate f32 and f16 over the
host double, and the simulation is bit-exact. The argument is
short: an f32 significand carries 24 bits, a product of two needs
at most 24 + 24 < 53, and 53 bits is what a double holds. Compute
in double, round once to the narrow format, and you get the same
bit the hardware would produce; f16, with 11-bit significands, has
more headroom still. The last output line is the promise in action:
two f16 copies of 0.0999755859375 sum to exactly 0.199951171875 on
every engine, GPU included.

## Buffers in, buffers out

Kernels meet the outside world through flat buffers.
`Tile.load_rows` reads a shape's worth of elements from a `Vec` in
row-major order; `Tile.store_rows` writes a tile back the same way.
Each has a masked form for ragged edges: `load_rows_or` fills
out-of-range elements with a default, and `store_rows_clipped`
drops writes that would land past the end.

```metaxu
fn main() -> int {
    let @mut buf = Vec.new();
    let mut i = 0;
    while i < 6 { buf.push(10 * i); i = i + 1; }
    let t = Tile.load_rows(buf, 2, 3);
    print(t.get(1, 2));
    # a 3x3 load overruns the 6-element buffer; the mask fills -1
    let padded = Tile.load_rows_or(buf, 3, 3, -1);
    print(padded.get(1, 2), padded.get(2, 0));
    Tile.store_rows(buf, t.scale(2));
    print(buf[0], buf[5]);
    # padded holds 9 elements; only the first 6 fit, the rest drop
    Tile.store_rows_clipped(buf, padded);
    print(buf[5]);
    0
}
```
```output
50
50 -1
0 100
50
```

The unmasked forms raise if the buffer is too short, in the usual
catchable way. The masked forms are what a kernel's last ragged
tile uses: edge handling is two function choices, not a nest of
bounds checks.

## Kernels and Gpu.launch

A kernel is a function of a lane id. `Gpu.launch(n, k)` runs `k`
for every id from 0 to `n - 1`. As chapter 8 showed, `launch` has
a default: no handler in scope means a sequential CPU loop in pid
order, deterministic and hardware-free. Vector addition:

```metaxu
from std.gpu import Gpu;

fn main() -> int {
    let @mut a = Vec.new();
    let @mut b = Vec.new();
    let @mut c = Vec.new();
    let mut i = 0;
    while i < 4 {
        a.push(i); b.push(10 * i); c.push(0);
        i = i + 1;
    }
    perform Gpu.launch(4, fn(pid: int) -> () {
        c[pid] = a[pid] + b[pid];
        ()
    });
    print(c[0], c[1], c[2], c[3]);
    0
}
```
```output
0 11 22 33
```

Each lane owns index `pid` and touches nothing else. That
discipline, one lane one slot, makes the same closure correct
under any handler: run the lanes in order, in parallel, or on
another device, and the result cannot differ. Real kernels do the
same at tile granularity: `load_rows` at a lane-dependent offset,
compute, `store_rows_clipped` back.

## Any handler can be the launch

A handler above the perform owns the launch completely. The kernel
doesn't run at all unless the handler runs it:

```metaxu
from std.gpu import Gpu;

fn main() -> int {
    handle Gpu with {
        launch(n, k) -> {
            print("launch intercepted: " + n.to_string() + " lanes");
            resume(())
        }
    } in {
        perform Gpu.launch(8, fn(pid: int) -> print(pid))
    };
    0
}
```
```output
launch intercepted: 8 lanes
```

The tests virtualize launches this way to check plumbing without
executing lanes; a profiling handler can count launches before
resuming into the default. The kernel never learns which handler
it ran under.

## Metal

`run_metal` is the handler that means it: a `launch` under it
compiles the kernel's tile operations to MSL and dispatches them
on the machine's GPU. The program is the vecadd above, wrapped in
one call:

```metaxu norun
from std.gpu import Gpu, run_metal;

fn main() -> int {
    # a, b, c built exactly as in the vecadd above
    run_metal(fn() -> () {
        perform Gpu.launch(4, fn(pid: int) -> () {
            c[pid] = a[pid] + b[pid];
            ()
        })
    });
    print(c[0], c[1], c[2], c[3]);
    0
}
```

The output is `0 11 22 33`, the same bytes the default handler
prints. The fence is display-only because the dispatch goes
through a clang++/MLX shim this book's harness can't assume: the
emitted MSL is compiled and executed by a small C++ host built at
run time, a real GPU round trip on a Mac with the toolchain. The
pinned accumulation order from the first section is what lets both
handlers promise identical floats, not just identical ints. To
read the MSL, or to build a standalone harness for Xcode:

```bash
uv run python scripts/emit_metal_harness.py examples/vecadd.mx
# writes the .metal source and a host harness next to the example
```

Kernels stay in Metaxu either way. The MSL is an artifact you can
read and check in, never a second source of truth.
