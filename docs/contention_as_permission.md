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

Native marking is emitted where the closure's member kinds are
statically known. A spawn whose capture kinds codegen cannot enumerate
DEMOTES with a reason (house rule: honest placeholder, never an engine
asymmetry). See Status for where the emission actually lives.

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

### Measured (2026-08-19, this container, clang -O2)

Methodology: wall clock around the whole process; 15 runs per side,
A/B-interleaved so machine drift hits both equally; medians reported.
Native mutation loop: 10M iterations of `v[0] = v[0] + i` + `push` +
`pop` (30M runtime mutator calls, single thread), compiled ONCE to IR by
the new codegen and linked against (a) runtime objects built from a
pristine pre-change source snapshot and (b) the new runtime — same IR,
same link, only the .o files differ.

- **Uncrossed Vec mutation: +15.8% on the pure-mutator microloop**
  (71.4 ms → 82.7 ms median), i.e. **+0.38 ns per mutator call**. The
  design target ("indistinguishable from noise") was NOT met on this
  microloop and the number is not softened: one header-flag load+test
  added to runtime calls that were ~6 instructions long is a measurable
  fraction of them. It IS the floor: the loop does nothing but mutate,
  so any real per-element work dilutes the percentage proportionally
  (the same 0.38 ns against even a 100 ns loop body is 0.4%). Two
  first-cut mistakes were found and fixed by this benchmark before
  landing (reading the permit through a cross-TU accessor call and
  general-dynamic TLS under -fPIC forced register spills in the
  mutators even on the never-taken path: +37%; now the guard is an
  unlikely-hinted flag test, an initial-exec `%fs`-relative load behind
  it, and an outlined cold raise).
- **Contended Vec mutation** (same loop, vec marked via a trivial spawn,
  mutex held): +3.3% over the new-runtime uncontended loop (82.8 ms →
  85.5 ms, interleaved) — the thread-local permit read, paid only by
  values that actually crossed.
- **Lock/unlock**: 1M uncontended lock+unlock pairs, same IR, old vs
  new runtime: 17.8 ms → 18.6 ms median (**+0.83 ns per pair, +4.7%**
  on an ~18 ns ERRORCHECK pair) — the two TLS bumps.
- **Spawn marking, native**: 2000 spawns of a closure capturing two
  structs (a Vec field each) plus two bare Vecs (4 emitted mark calls):
  128.7 ms with marks vs 125.1 ms with the mark calls stripped from the
  IR (+2.9%, ≈ +1.8 µs/spawn nominal) — **within run-to-run noise**
  (spreads overlap; the walk is physically four relaxed stores plus a
  few GEP/loads against a ~60 µs pthread_create).
- **Interpreter mutation loop** (100k iterations × the same three
  mutations, `time.perf_counter` around `interp.call`, old = pre-change
  `mir_interp.py` restored from git): 1616-1663 ms → 1697-1711 ms
  medians across two repeats — **+3-5%** for the per-mutating-builtin
  wrapper (contended-flag + permit test), against a per-op cost already
  ~5.5 µs.
- **Interpreter spawn marking** (300 spawns × the same captures):
  medians 87.0-96.6 ms old vs 94.4-100.0 ms new across interleaved
  repeats — distributions overlap; bounded by ≤ ~10% on a pure
  spawn-join storm and not separable from Python thread-start noise.
- **Vec header**: 24 → 32 bytes (the flag word), which moves the header
  from glibc malloc's 32-byte chunk to the 48-byte one — +16 bytes of
  real memory per vector, once.

## Status (landed 2026-08-19)

Implemented as specced, on both engines, with the wording above shared
byte-for-byte:

- **Interpreter** (`mir_interp.py`): `MxVec.contended` flag; the marking
  walk in `_rt_thread_spawn` (direct Vec captures; struct fields
  recursively; mutable-capture cells through the cell; STOP at Vec
  elements); the permission is `_ThreadCtx.write_permit`, bumped by
  `_rt_mutex_lock`/`_rt_mutex_unlock` strictly after their error checks
  (handle-body threads adopt their creator's ctx, so the grant follows
  the logical thread exactly like mutex ownership); every mutating
  builtin (push, pop, index-set, index-store) checks
  `contended and write_permit == 0` and raises `InterpError` with the
  exact message. Reads untouched.
- **Native runtime**: `mx_vec.contended` (relaxed C11 atomic — a
  re-mark at a second spawn must not race a concurrent flag read;
  relaxed loads/stores are plain moves on x86-64);
  `mx_vec_mark_contended` exported; the guard in
  `mx_vec_push`/`pop`/`set` is an unlikely-hinted flag test, then a
  direct initial-exec TLS read of `mx__tls_write_permit`
  (metaxu_threads.c: +1 after a successful lock only, -1 before the
  unlock's pthread call with the EPERM path restoring it), then an
  outlined cold raise through `mx_raise` (catchable).
- **Native codegen** (`codegen_llvm.py`): the marking walk is emitted
  INSIDE the `__effect_runtime$E$spawn` thunk, branching on the fn
  pointer when the closure kind admits several member lambdas, walking
  each member's env layout (vec fields marked; struct fields recursed by
  static layout; `cell:` fields loaded through the cell pointer; stop at
  vec elements). Unenumerable capture kinds (conflicts, aggregate cells,
  unknown struct layouts) demote the thunk with reason "cannot emit
  contention marks for spawn captures (kinds unavailable)".

**Scoped-site resolution** (the direct-vs-virtualized question the
design left open): no per-site branching and no demotion of scoped
sites was needed, because the runtime thunk IS the branch. A perform
with no handle scope for the op calls the thunk directly; a scoped
perform routes through `mx_perform_or_default`, which runs the thunk
only AFTER the innermost-scope lookup found no handler — i.e. exactly
when the spawn is real. Marking inside the thunk therefore covers both
routes and can never mark a virtualized spawn, mirroring the
interpreter (whose marking lives in the `_rt_thread_spawn` shim, which
handler-intercepted performs never reach). The trade taken: the
demotion unit for unenumerable captures is the thunk (all spawns in the
module) rather than one call site — coarser, loud, and honest.

Tests: `tests/test_contention.py` — the raise (in-child catch AND
uncaught-surfaces-at-join), all three mutators guarded, locked writes
pass, reads free, the spawner's own post-crossing writes caught,
std.sync untouched, struct-field marking, the vec-of-vecs hole PINNED,
virtualized spawns unmarked, native differentials (message
byte-identical, zero placeholders), TSan-clean locked path, and the old
locks-deleted racy counter now failing deterministically with this
error instead of racing. Non-vacuity was verified by disabling the
marking on each engine in turn and watching the tests fail.
