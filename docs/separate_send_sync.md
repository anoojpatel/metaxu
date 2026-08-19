# Separateness: thread-compatibility without annotations

Design record for Metaxu's send/sync story, chosen 2026-08-18. The goal,
verbatim from the design conversation: *no annotation hell, no bleeding
into struct/field definitions, most of the power via standard-library
tooling, unsafe when needed — not the Rust world of complicated typing.*

## The four commitments

### 1. Separateness is inferred, never declared

Whether a value may be shared across threads is a STRUCTURAL property the
compiler computes — the good half of Rust's auto-trait mechanism with none
of the syntax. No struct, field, or function declaration ever mentions it.

Base facts (the compiler's, not the user's):

| Value | Separate? | Why |
| --- | --- | --- |
| int, bool, float, string | yes | no mutable identity — copies are free |
| Mutex / Thread handles | yes | the runtime serializes all access |
| `std.sync.protect(..)` result | yes | by construction (see § 3) |
| Vec, list literals | **no** | shared mutable identity: two threads through one handle race |
| struct / enum values | iff every field / payload is | structural composition |
| closures | iff every capture is | structural composition |
| everything else (call results…) | unknown | see enforcement policy below |

### 2. Checked at the boundary, carried by the effect row

The ONLY place separateness is demanded is a spawn-mapped operation (any
op declared `with EFFECT_SPAWN`) — never in generic signatures, never as
bounds. Interprocedural propagation, when incremental compilation makes
callees opaque, rides the effect row the language already requires:
`performs Thread` in a signature IS the "may cross threads" marker Rust
spells as `F: Send`. (Staged — see § Status.)

**Handlers subtract the requirement.** A spawn-mapped perform lexically
inside a `handle` that handles that op is virtualized: the closure never
crosses a real thread, so no separateness is demanded. (Locality and
@mut-borrow rules still apply there by the documented signature-promise
choice — separateness is about REAL sharing, those are about the
declared contract.) This is a property Rust cannot express: thread-
safety obligations that relax under test harnesses, statically, because
effects compose that way.

**Enforcement policy:** only KNOWN-shared captures that the closure
WRITES (assignments, mutator methods like push/pop — including through
struct fields) are rejected; read-only captures of shared values are
admitted, because contention weakens access rather than revoking it
(design A, docs/contention_as_permission.md). A mutation hidden behind
a helper call is statically untraced by design — the dynamic contention
layer owns it. A call result the checker cannot classify passes — a
deliberate false-negative bias, because rejecting every unclassifiable
capture would make the check unusable and push everyone straight to
`unsafe`. The trace is syntactic and name-based, like Rule B locality
(docs/ownership_and_borrowing.md § 2.2b), and shares its propagation:
`let w = v` inherits v's sharedness with provenance.

### 3. The power lives in the standard library

Users should meet idioms, not the checker:

- **`std.sync.protect(v)`** wraps a value with its own mutex into a
  `Protected` handle — separate by construction. Access only through
  `with_lock(p, fn(v) -> ..)`, which locks, runs, unlocks (also on the
  value-return path). "Vec is not separate" is answered by a library
  call, not a language negotiation.
- **Move = Send.** A capture the closure takes by `move(..)` transfers
  unique ownership into exactly one thread: allowed with no mutex, the
  fast pattern. (Post-spawn use of the moved-from name is the frozen
  borrow checker's use-after-move territory; the seam between move
  tracking and closure captures is documented as staged, not claimed.)
- Channels and atomics are future std work on the same foundation.

### 4. Escape is scoped `unsafe`, not a type-global vow

`unsafe { perform Thread.spawn(..) }` disables the separateness check
for spawns inside the block — you vouch for THIS crossing, at the call
site, greppably. There is no `unsafe impl`-style type-global assertion:
a vow about every future crossing of a type, made in one invisible
place, is exactly the Rust shape this design rejects. Locality and
@mut rules are NOT disabled by unsafe (dangling and aliased exclusivity
are not things a programmer can promise away by being careful).

## What this rejects, and what the fix looks like

```metaxu
let mut v = Vec.new();
v.push(0);
perform Thread.spawn(|| { v[0] = v[0] + 1; 0 });   // ✗ separate-spawn-capture
```

> closure passed to spawn-mapped operation 'Thread.spawn' captures 'v',
> which has shared mutable identity (Vec): two threads mutating through
> one handle race; protect it (std.sync.protect), move it into exactly
> one thread, or take responsibility with `unsafe { .. }`

The three fixes, in order of preference:

```metaxu
// 1. Protect: the blessed idiom
let p = protect(v);
perform Thread.spawn(|| { with_lock(p, fn(v) -> { v[0] = v[0] + 1; () }); 0 });

// 2. Move: unique transfer, no lock needed
perform Thread.spawn(|| { let mine = move(v); mine[0] = 1; 0 });

// 3. Unsafe: manual discipline, visibly owned
unsafe { perform Thread.spawn(|| { v[0] = v[0] + 1; 0 }) }
```

## The trade, stated honestly

Given up: early errors at generic definitions (errors fire at the
concrete spawn site, with provenance, C++-style) and module-boundary
precision until effect-row propagation lands. Gained: zero annotations
on types, zero bounds on functions, idioms in std, a scoped escape.
The design degrades gracefully — every stage is an extension of the
spawn-capture checker (compiler/spawn_capture_check.py).

## Status

- **Implemented**: base facts, structural composition through struct
  literals and closure captures, Rule-B-style propagation with
  provenance, spawn-boundary enforcement (kind
  `separate-spawn-capture`), handler subtraction, moved-capture
  allowance, `unsafe {}` scoping, `std.sync` with
  `Protected`/`protect`/`with_lock`.
- **Staged**: effect-row interprocedural propagation (with §9
  incremental compilation); deep sharedness through data structures /
  field reads / call results; the move/use-after-move seam across
  closure captures; channels and atomics in std; an optional
  `@separate` struct ASSERTION (documentation aid, never a requirement).
