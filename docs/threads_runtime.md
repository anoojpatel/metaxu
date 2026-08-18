# Threads runtime: real OS threads behind `Thread`/`Mutex`

This document is the contract for the `with EFFECT_*`-mapped thread and
mutex primitives (`examples/effect_mapping.mx` declares them):

```
effect Thread = {
    fn spawn[T](f: fn() -> @global T) -> @global Thread[T] with EFFECT_SPAWN
    fn join[T](thread: @global Thread[T]) -> @global T with EFFECT_JOIN
}
effect Mutex = {
    fn create() -> @global Mutex with EFFECT_MUTEX_CREATE
    fn lock(mutex: @global Mutex) -> () with EFFECT_MUTEX_LOCK
    fn unlock(mutex: @global Mutex) -> () with EFFECT_MUTEX_UNLOCK
}
```

Both engines implement REAL OS threads: the MIR interpreter (the semantics
reference) on Python `threading.Thread`s, the LLVM backend on pthreads
(`src/metaxu/runtime/native/metaxu_threads.c`). This replaces the previous
documented single-threaded model (spawn-runs-to-completion-at-spawn);
`docs/v1_gap_analysis.md` records the history.

## Execution model

* `perform Thread.spawn(f)` starts `f` — a zero-argument closure — on a
  NEW thread immediately and returns a thread handle without waiting.
  The child runs CONCURRENTLY with the parent from that point on.
* `perform Thread.join(t)` blocks until the child completes and returns
  its result, exactly once. A second join of the same handle is a loud
  error (matching `pthread_join` on an already-joined thread):
  `EFFECT_JOIN on <Thread#N joined>: thread already joined` — a CATCHABLE
  failure (`try` recovers it) on both engines.
* A handle that is never joined is detached: program termination does NOT
  wait for it (interpreter: daemon threads; native: `main` returning ends
  the process). Nothing about a detached child's progress is specified.

### Scheduling and determinism

Interleaving between threads is UNSPECIFIED. The only specified
observations are schedule-independent ones: a join returns the child's
completion value; mutual exclusion holds between `lock` and `unlock`;
writes made under a mutex are visible to the next holder of that mutex
(the interpreter inherits this from the GIL + `threading.Lock`; native
from pthread mutex acquire/release ordering). Tests — differential tests
especially — MUST only assert schedule-independent outcomes (a final
counter value, a join result), never output interleavings or progress
ordering between unjoined threads. Thread/mutex ids embedded in error
messages are assigned in creation order, which is only deterministic when
all creation happens on one thread.

## Mutex semantics

Mutexes are ERRORCHECK mutexes (native: `PTHREAD_MUTEX_ERRORCHECK`;
interpreter: an owner-tracking wrapper around `threading.Lock` that
mirrors it):

* `lock` on a mutex held by ANOTHER thread BLOCKS until it is released.
  This is the semantic change from the old single-threaded model, and the
  point of real threads.
* `lock` on a mutex this thread ALREADY holds is still a loud, CATCHABLE
  error (ERRORCHECK's `EDEADLK`):
  `deadlock: EFFECT_MUTEX_LOCK on <Mutex#N>: this thread already holds it`
* `unlock` of a mutex this thread does not hold — because it is unlocked,
  or because another thread holds it (ERRORCHECK's `EPERM` covers both) —
  is a loud, CATCHABLE error, deliberately worded without the mutex's
  momentary state (which is racy to read):
  `EFFECT_MUTEX_UNLOCK on <Mutex#N>: this thread does not hold it`
* Mutexes are NOT recursive.

Ownership in the interpreter is tracked per LOGICAL Metaxu thread, not
per Python thread: a `handle` body runs on its own parked Python thread
but belongs to the same logical thread as the frame that installed the
handler, so `lock` outside a handle body and `unlock` inside it (or vice
versa) pair up exactly as they do natively, where handle bodies are
ucontext fibers on the SAME OS thread.

## Effect-scope isolation

Each thread owns its OWN effect scope stack. A spawned child does NOT
inherit the handlers in scope at the spawn site: a perform in the child
routes against the child's own (initially empty) scope stack, then to the
op's `with SYMBOL` runtime mapping or declared `= expr` default, and
otherwise fails with the usual catchable `No handler for effect 'E'`.

Rationale: a handler scope is a delimited continuation rooted in the
frame that installed it, on the INSTALLING thread's stack. Letting a
child's perform park itself against a parent-stack scope would suspend a
foreign stack and resume across threads — a dangling-continuation hazard
natively (the parent's fiber pump runs on the parent's stack) and a
deadlock generator in the interpreter. `handle` inside the child works
normally; handlers simply do not cross `spawn`.

Native mechanics: ALL effects-scheduler state in `metaxu_effects.c`
(`g_top`, `g_cur`, `g_main`, `g_pad`, `g_fiber`, `g_raise_msg`) is
`_Thread_local`, so each thread gets a fresh scope stack, pad chain and
fiber bookkeeping; the per-fiber pad-chain semantics are unchanged WITHIN
each thread. Interpreter mechanics: the handler-frame stacks live on a
per-logical-thread context; handle-body threads adopt their creator's
context, spawned threads get a fresh one.

Handlers in scope ON THE PERFORMING THREAD still override the runtime
mapping (effects stay virtualizable): a `handle Thread with { spawn(f) ->
... }` in scope intercepts `perform Thread.spawn` and no OS thread is
created. This holds on both engines; dispatch order is unchanged
(innermost non-busy handler frame > `with SYMBOL` runtime mapping >
declared `= expr` default > loud error).

## Failure in a child thread

A CATCHABLE failure (`fail`, unhandled effect, runtime raise) inside a
child that no `try` inside the child recovers does NOT unwind into the
parent and does NOT kill the process at failure time: it marks the
child's handle errored, and `join` re-raises it — catchably — on the
joining thread. An enclosing `try` around the join binds the child's
exact failure message. Natively the child's entry point wraps the closure
in the standard `mx_try` landing pad; the interpreter's child wrapper
captures the exception and re-raises it at join.

FATAL failures (assert, integer division by zero, allocation failure,
single-shot violation — everything `try` does not catch, see
docs/try_catch.md) remain fatal. Divergence, documented: the native
runtime aborts the process AT THE FAILURE POINT on whatever thread it
happens; the interpreter surfaces the error at the join. Both end the
program naming the failure; only schedule-independent tests are valid
here anyway.

A failure in a DETACHED (never-joined) child is unobservable, except a
native fatal one, which still aborts the process.

## Memory model: what crosses the spawn boundary

* INTO the child: the closure `{fn, env}` only. Natively the environment
  is forced to a heap (immortal, leak-by-design) block — the same
  contract as effect-boundary boxes — because the child dereferences it
  after the parent's frame has moved on; when the spawn perform is
  intercepted-or-mapped dynamically (a handle scope for `spawn` exists in
  the module), the pair itself additionally crosses inside an immortal
  boundary box, which is safe for exactly the boundary-box reasons: the
  box is write-once, never freed, and the child holds the only reference
  the runtime hands out. The child invokes the closure through the
  word-uniform ABI (`i64 (ptr env)`), so natively-spawnable closures are
  zero-argument closures with a word-encodable result; anything else
  demotes with a reason (the interpreter accepts any closure).
* OUT of the child: the result travels as one opaque 8-byte word stored
  in the handle (aggregates are already boxed by the word-uniform return
  path — immortal, so join can read it at any later time).
* The Thread handle and the Mutex are heap records that are NEVER freed
  (immortal, leak by design). That is what makes double-join a flag
  check — never a use-after-free — and lets handles/mutexes be stored,
  copied and shared freely as opaque words. ASan runs for thread
  programs therefore use `detect_leaks=0`.
* Shared mutable state between threads is only whatever already has
  identity semantics across the boundary (e.g. a captured `Vec` aliases
  one vector on both engines). Data races on it are the program's
  problem; the mutex primitives exist to prevent them, and the native
  test suite proves the mutex path clean under `-fsanitize=thread`.

### Modes: declared vs enforced

`spawn`'s declared signature is `fn() -> @global T` and it returns
`@global Thread[T]`. What today's checkers actually enforce at these call
sites is only the ordinary borrow/mode discipline of the surrounding
code; no thread-specific rule (e.g. "captures must be @global", "no
@local borrows may cross spawn") is enforced beyond it. That is a
DOCUMENTED GAP, not a guarantee: the runtime keeps itself memory-safe
(heap envs, immortal boxes/handles) regardless, but cross-thread aliasing
of mutable state is not statically excluded. Do not claim send/sync-style
checking anywhere until it exists.

## Native lowering (codegen_llvm)

`with SYMBOL` runtime-mapped ops no longer demote. The routing mirrors
the declared-default machinery exactly, one rung higher in precedence:

* No handle scope in the module lists the op: the perform IS a direct
  call to the `__effect_runtime$E$op` thunk (ordinary call conventions;
  this is the effect_mapping.mx path).
* Some scope lists it: the perform lowers to `mx_perform_or_default`
  with the runtime thunk as the fallback — the identical
  innermost-non-busy scope lookup decides at run time, so an in-scope
  handler wins and only otherwise does the mapping run.

Inside the thunks, `__mx_effect_runtime$SYMBOL` calls lower to the C
primitives in `metaxu_threads.c`:

| symbol              | C entry point                             |
|---------------------|-------------------------------------------|
| EFFECT_SPAWN        | `int64_t mx_thread_spawn(void* fn, void* env)` |
| EFFECT_JOIN         | `int64_t mx_thread_join(int64_t handle)`  |
| EFFECT_MUTEX_CREATE | `int64_t mx_mutex_create(void)`           |
| EFFECT_MUTEX_LOCK   | `int64_t mx_mutex_lock(int64_t m)`        |
| EFFECT_MUTEX_UNLOCK | `int64_t mx_mutex_unlock(int64_t m)`      |

`Thread[T]` and `Mutex` are extern types with no native layout: their
values are the opaque i64 handle words above (kind-inferred as i64), and
lock/unlock "return" the unit word 0. A mapped symbol outside this table
demotes with a reason (the interpreter errors loudly at perform time for
the same program). Every native binary now links `-pthread`.

## Static state audit (native runtime)

Every file-scope mutable object in `src/metaxu/runtime/native/`,
dispositioned:

| file              | object                    | disposition            |
|-------------------|---------------------------|------------------------|
| metaxu_effects.c  | `g_top` scope stack       | `_Thread_local` (per-thread scope stack — scope isolation) |
| metaxu_effects.c  | `g_cur` running-fiber stack info | `_Thread_local` |
| metaxu_effects.c  | `g_main` root-stack bounds | `_Thread_local` (each thread's own root stack) |
| metaxu_effects.c  | `g_pad` landing-pad chain | `_Thread_local` (pads never cross threads) |
| metaxu_effects.c  | `g_fiber` running body scope | `_Thread_local` |
| metaxu_effects.c  | `g_raise_msg` in-flight raise | `_Thread_local` (a raise resolves on its own thread) |
| metaxu_threads.c  | `g_next_thread_id`, `g_next_mutex_id` | shared, C11 atomics (ids in error messages) |
| metaxu_rt.c       | — none —                  | audited: every `static` is a function or const; no mutable data |

`malloc`/`free`/`realloc` and stdio are thread-safe per POSIX; the Vec /
string runtime has no global state, so concurrent use of DISTINCT values
is safe and concurrent use of ONE value is the program's race (see
above).

## Verification bar

* The full pytest suite and both example gates stay green.
* Interpreter semantics tests in `tests/test_threads.py` (through parsed
  source): N threads x M mutex-protected increments == N*M; join value;
  child failure surfaces at join; double join; unlock-not-held; self-
  relock deadlock; scope isolation (child perform does NOT reach the
  parent's handler); handler-over-mapping virtualization.
* Native differentials for the schedule-independent programs (counter,
  join value), plus `-fsanitize=thread` (zero reports) and
  `-fsanitize=address` with `detect_leaks=0` (immortal handles/boxes)
  runs of the counter program.
* TSan non-vacuity was proven during development by deleting the
  lock/unlock calls from the counter program's source: TSan reports a
  data race on the vector slot (see the commit message that landed the
  native path).
