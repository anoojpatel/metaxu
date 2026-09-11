<div align="center">
<img src=https://github.com/user-attachments/assets/9a8ec76c-224a-4662-96fb-bdf0b420e01f width="12%" height="12%"></img>


<h1 style="text-align: center;"> Metaxu🌀  </h1>
<p style="text-align: center;">
A systems language with algebraic effects, mode-based memory safety,
and tile-level GPU kernels <br>
[<a href="docs/book/README.md">The Metaxu Book</a>] &middot;
[<a href="docs/index.md">Documentation</a>]
</p>
</div>

## What is Metaxu?

Metaxu is a systems programming language that compiles to native code
through LLVM, built around three ideas:

1. **Algebraic effects as the concurrency and capability model.**
   Threads, mutexes, streams, GPU kernel launches — all effects with
   handlers. A handler can virtualize any of them, which is how the
   same program runs on a test double, a thread pool, or a GPU.
2. **Modes instead of lifetimes.** Memory safety comes from a small set
   of inferred modal axes — locality (`@local`/`@global`), mutability
   (`@mut`/`@const`), linearity (`once`/`separate`/`many`) — checked by
   a borrow checker that propagates properties instead of asking you to
   annotate them. Thread-safety (send/sync) is *inferred*, never
   declared; crossing a value into another thread marks it, and
   unsynchronized writes to crossed values are caught — statically where
   provable, dynamically otherwise, with the same error wording either
   way.
3. **An interpreter as the semantics reference.** Every backend is
   differentially tested against a strict reference interpreter,
   byte-for-byte — output, error messages, float formatting. A whole
   class of "silently wrong code" bugs is structurally hunted this way.

**Measured performance** (see `benchmarks/`): the native backend runs
1.00–1.09x C on recursion, float ALU, struct-rebuild and sieve
workloads, ~1.3–1.7x C on bounds-checked Vec loops (the checked-indexing
price, after inline fast paths), with OS threads at parity and effect
handler crossings at ~280 ns.

**GPU tiles** (in progress — `docs/gpu_tiles.md`): Triton/Gluon-class
tile kernels (`Tile.load_rows`, `Tile.dot`, masked stores) written in
Metaxu, launched through the `Gpu` effect, and compiled to Metal Shading
Language for Apple GPUs via MLX. The CPU handler is the deterministic
reference; the emitted MSL is differentially tested against it.

```
source → parser → module resolution → desugar → typecheck
       → (SimpleSub inference + borrow/mode checking)
       → HIR → MIR ─┬→ reference interpreter   (semantics)
                    ├→ LLVM → clang → native   (differentially tested)
                    └→ MSL → Metal via MLX     (tile kernels)
```

## Core Features
- 🚀 Zero-cost abstractions with unboxed references and move semantics
- 🛡️ Memory safety through simple modal borrow checking — inferred, not annotated
- ⚡ Algebraic effects with real delimited continuations (native `try`/`catch`, streams, generators)
- 🧵 Real OS threads with inferred thread-safety and contention-as-permission write guards
- 🧬 SimpleSub type inference with algebraic subtyping (biunification)
- 🎯 Tile-level GPU kernels compiled to Metal, with compile-time shape checking
- 🔄 Pattern matching with exhaustiveness checking
- 📦 An Ante-modeled standard library where effects are the idiom (`std.stream`, `std.sync`, `std.gpu`)

## Philosophy
The name "Metaxu" comes from philosopher Simone Weil's concept of divine intermediaries. Just as a wall
between prison cells becomes both a barrier and a medium for communication through taps, Metaxu serves
as a bridge between high-level abstractions and low-level performance. It connects human intention to
machine execution, striving to make systems programming both powerful and accessible.

Deeply inspired by [Ante](https://ante-lang.org), [Hylo](https://hylo-lang.org), [Sage](https://github.com/adam-mcdaniel/sage), [Oxidized OCaml](https://blog.janestreet.com/oxidizing-ocaml-locality/), [Rust](https://rust-lang.org), [Zig](https://ziglang.org), and [Python](https://www.python.org), we're building a
language that embraces both pragmatism and purity. Our goal is to create a tool that helps programmers
express their ideas clearly and efficiently, while ensuring their code remains fast and reliable. Furthermore, we want to create a language that's accessible to everyone, regardless of their background or experience level, so that anyone can learn how to build compilers.

## A taste

```rust
// Streams are effects; consumers are handlers.
from std.stream import iota, map, filter, sum;

fn main() -> int {
    let total = sum(filter(map(iota(200000), fn(x: int) -> x * 2),
                           fn(x: int) -> x > 5));
    print(total);
    0
}
```

```rust
// A GPU kernel: one instance per output tile, masked edges, no raw pointers.
from std.gpu import Gpu, run_grid;

fn mm_kernel(pid: int, a: Vec, b: Vec, c: Vec) -> () {
    let ti = pid / 16;  let tj = pid % 16;
    let mut k = 0;
    let mut r = Tile.filled(8, 8, 0);
    while k < 16 {
        let ta = Tile.load_rows(a, (ti * 8) * 128 + k * 8, 128, 8, 8, 0);
        let tb = Tile.load_rows(b, (k * 8) * 128 + tj * 8, 128, 8, 8, 0);
        r = Tile.add(r, Tile.dot(ta, tb));
        k = k + 1
    };
    Tile.store_rows(c, (ti * 8) * 128 + tj * 8, 128, r);
    ()
}
// perform Gpu.launch(256, fn(pid: int) -> mm_kernel(pid, a, b, c));
// — the default handler runs it deterministically on CPU;
//   the Metal handler runs it on your GPU. Same kernel.
```

## Status

The v1 compiler is working end to end: 19 example programs compile and
run natively with outputs pinned against the interpreter, ~2000 tests
pass, and the benchmark suites (`benchmarks/suite`,
`benchmarks/diagnostics`, `benchmarks/contention`) track measured
numbers with an alignment-controlled, rotation-interleaved methodology.
Current work: the GPU tile pipeline (Stage 1 landed: kernel seam, MSL
emitter, f32 tiles with bit-exact rounding across all three engines,
float Metal kernels, Mac self-check harness, and the Metal runtime
handler — `run_metal` installed over `Gpu.launch` dispatches kernels
through MLX on a Mac or a bit-exact shim engine elsewhere; next:
layouts and `simdgroup_matrix`). See `docs/compiler_roadmap.md` and
`docs/gpu_tiles.md`.

## Installation

### Prerequisites
- Python 3.11 or higher
- clang (for the native backend)
- uv (fast Python package installer)

### Setup
1. Install uv:
```bash
# On macOS
brew install uv

# On Linux/WSL
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:
```bash
git clone https://github.com/anoojpatel/metaxu.git
cd metaxu
```

3. Install dependencies:
```bash
uv sync --all-groups
```

4. Run the gates:
```bash
uv run python -m pytest src/metaxu/compiler/tests -q   # full suite
uv run python scripts/run_examples.py                  # pipeline gate
uv run python scripts/run_examples.py --stage run      # execution gate
```

5. Emit a Metal kernel harness for a Mac:
```bash
uv run python scripts/emit_metal_harness.py kernels.mx my_kernel \
    --grid 4 --buf a=1,2,3,4 --buf out=0,0,0,0
```

#### Helping LLMs
We have a `llm-ctx.txt` file that contains some examples of how to use Metaxu with LLMs. It follows [llms.txt](https://llmstxt.org/) standards. Add it via `@docs` for IDEs or into relvant context manager protocols.

## Development

Conventions that keep this codebase honest (see `CLAUDE.md` for the
full set):

- Tests go through parsed source, never hand-built IR fixtures.
- The interpreter is strict; no lenient fallbacks to make a test pass.
- Native lowering claims need differential tests (native output ==
  interpreter, byte for byte) and, for memory claims, ASan runs scoped
  to the documented contract.
- Benchmarks pin function alignment and rotate run order — unaligned
  microbenchmarks measure linker luck, not your change.
