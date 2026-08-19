# Contention as permission (design B)

Chosen 2026-08-18 as Metaxu's contention story — the effects-native
alternative to OxCaml's contention mode axis (see
docs/separate_send_sync.md for the separateness layer this builds on).

## The idea

OxCaml tracks "may another thread be touching this?" as a mode on every
value, threaded through inference, with mode-polymorphic functions. Metaxu
does not need the axis, because both facts contention cares about are
ALREADY reified here:

- **Where contention begins** is a thread crossing, and every crossing is
  an effect op the compiler and runtime intercept (`with EFFECT_SPAWN`).
- **What grants write permission** is a held lock, and every lock is an
  effect op with a runtime handler (`with EFFECT_MUTEX_LOCK/UNLOCK`).

So contention becomes a DYNAMIC fact attached to the value at the
crossing, and permission a per-thread fact granted by the lock runtime:

1. When a closure crosses a real spawn, every shared-identity value it
   captures is **marked contended** (a bit on the Vec header).
2. A **mutation** of a contended value requires the mutating thread to
   hold a lock (any mutex acquired through the runtime — see
   "Granularity" below). Otherwise it raises, loudly and catchably, with
   identical wording on both engines:

   > write to contended Vec without a held lock: this value crossed a
   > thread boundary at spawn; mutate it under a mutex
   > (std.sync.with_lock) or keep it thread-local

3. **Reads stay free.** Contention weakens access (writes need
   permission); it never revokes it. This is OxCaml's contended/shared
   split realized dynamically.

`std.sync` needs no changes: `with_lock`/`read`/`write`/`update` acquire
the mutex through the runtime, so the permission is held exactly while
the lock is. The `Protected` cell is itself marked contended when its
handle crosses — and every access to it happens under the lock, so it
never trips the check. No exemption machinery.

## Why this is still "mutation as an effect"

Permission is handler presence: the lock primitives ARE the Mutex
effect's runtime handlers, and holding one is what a `Write`-permission
handler being "in scope" means dynamically. v1 deliberately does NOT
surface a user-visible `Write` effect (no lowering of `v.push` to
`perform` — that dispatch on every mutation is the catastrophic-overhead
version). The staged v2, if wanted: a declared `Write` effect whose
user handlers can also grant permission, making the grant virtualizable
like every other effect. Recorded, not promised.

## Marking rule (identical on BOTH engines — divergence is a bug)

At a real (non-virtualized) spawn, mark contended: every captured value
of Vec identity, recursing through **struct fields by static layout**,
STOPPING at Vec elements (the contents of a captured Vec-of-Vecs are not
marked — enumerating runtime elements is possible for the interpreter
but not for native code, and the engines must agree; the hole is pinned
by a test, not silently divergent). Values that cross by `move(..)` and
`Protected` handles mark the same way — moves are uniquely owned so
their marks are unreachable races anyway, and Protected accesses hold
the lock.

Native marking is emitted at the spawn call site, where the closure's
member kinds are statically known. A spawn site where codegen cannot
enumerate capture kinds DEMOTES with a reason (house rule: honest
placeholder, never an engine asymmetry).

## Granularity, honestly

Permission is **any held mutex**, not "the mutex associated with this
value" — a per-thread counter, not a lockset. Two threads mutating one
vec under two DIFFERENT mutexes satisfies the check but still races;
TSan remains the net for that misuse, and the typed lock-to-data
association (capsule-style phantom keys, per OxCaml) is the recorded
upgrade path when Metaxu has the type machinery. What the check DOES
catch — deterministically, on both engines, in production builds — is
the overwhelmingly common bug: mutating shared state with no lock at
all, including through every flow the static checker cannot trace
(data structures, call results, field reads).

## Interaction with the static layer

The static separateness check (docs/separate_send_sync.md) is unchanged
and remains the early-error path. What changes is the meaning of its
escape hatch: `unsafe { }` now drops you onto this dynamic enforcement
rather than onto nothing — manual-discipline code keeps its loud
runtime net. (The TSan non-vacuity experiment from the threads work
changes accordingly: deleting the locks from the counter now aborts
deterministically with the contended-write error instead of racing.)

Follow-up enabled by this backstop (recorded, not implemented): the
static rule can be RELAXED to writes-only — read-only captures of
shared values are safe to admit statically once unprotected writes are
dynamically caught. That is option A of the design discussion, and it
becomes strictly safe to adopt after this lands.

## Runtime cost (measured, not asserted)

The implementation must ship with measured numbers, filled in below at
merge time; the design targets are:

- **Uncrossed Vec mutation** (the overwhelming common case): one flag
  test on a header word that is on the same cache line as len/cap, in
  an already-branchy runtime call — target indistinguishable from
  noise, must be measured with a tight native push/set loop.
- **Contended Vec mutation**: flag test plus one thread-local read —
  only paid by values that actually crossed threads.
- **Lock/unlock**: one thread-local increment/decrement on top of a
  pthread mutex operation — noise.
- **Spawn**: one marking walk over the captures (bounded by capture
  count and struct depth), once per spawn.
- **Interpreter**: one attribute check per mutating builtin, against an
  interpreter whose per-op cost is already orders of magnitude larger.

### Measured (2026-08-18, this container, clang -O2)

FILLED IN AT MERGE — native tight-loop mutation benchmark (old runtime
vs new, same IR, ≥5 runs, medians), interpreter benchmark, and spawn
marking overhead. Any claim in this section without a number next to it
is a bug in this document.
