# Threads and sharing

Metaxu has real OS threads, and it gets them the same way it gets
everything else: through effects. There is no `std::thread` API baked
into the compiler. There is an effect whose operations are mapped to
runtime symbols, and the runtime behind those symbols starts pthreads
(natively) or Python threads (in the interpreter). The first example
shows the whole declaration, exactly as programs write it.

`extern type Thread[T]` says the handle has no Metaxu-visible layout;
it's an opaque word the runtime owns. The `with EFFECT_SPAWN` clause
maps the operation to a runtime symbol instead of requiring a handler.
`Mutex` is declared the same way, with `create`, `lock`, and `unlock`
mapped to `EFFECT_MUTEX_*`. You can write these declarations yourself,
as the examples here do, but `std.sync` already ships the mutex ones
and wraps them in something better (more on that shortly).

`spawn` takes a zero-argument closure, starts it on a new thread
immediately, and returns without waiting. `join` blocks until the child
finishes and returns its value, exactly once. A handle you never join
is detached: the program doesn't wait for it at exit.

```metaxu
extern type Thread[T];

effect Thread = {
    fn spawn[T](f: fn() -> @global T) -> @global Thread[T] with EFFECT_SPAWN
    fn join[T](thread: @global Thread[T]) -> @global T with EFFECT_JOIN
}

fn main() -> int {
    let x = 40;
    let t = perform Thread.spawn(|| { x + 2 });
    print(perform Thread.join(t));
    0
}
```
```output
42
```

Interleaving between threads is unspecified. The only guarantees are
schedule-independent ones: a join returns the child's completion value,
mutual exclusion holds between `lock` and `unlock`, and writes made
under a mutex are visible to the next holder. Every example in this
chapter prints only after joining, so its output is deterministic; your
programs should assert the same kinds of facts.

## What the checker rejects

The compiler infers whether a value may cross into another thread. You
never declare it. There is no `Send`, no `Sync`, no marker trait, no
bound on any signature. The compiler knows structurally which values
have shared mutable identity: ints, floats, strings, and structs of
them copy freely; a `Vec` is one heap object that every alias sees, so
two threads writing through one handle race. Capture such a value in a
spawned closure and write to it, and the checker rejects the spawn
site:

```metaxu error
extern type Thread[T];

effect Thread = {
    fn spawn[T](f: fn() -> @global T) -> @global Thread[T] with EFFECT_SPAWN
    fn join[T](thread: @global Thread[T]) -> @global T with EFFECT_JOIN
}

fn main() -> int {
    let mut v = Vec.new();
    v.push(0);
    let t = perform Thread.spawn(|| { v[0] = v[0] + 1; 0 });
    0
}
```
```output
shared mutable identity
```

The full diagnostic names the three fixes in order of preference:
protect it (`std.sync.protect`), move it into exactly one thread
(`move(v)`), or take responsibility with `unsafe { .. }`. Read-only
captures of shared values are allowed; contention weakens access, it
doesn't revoke it. And the check follows aliases the way locality does
(chapter 10): `let w = v` inherits `v`'s sharedness, and the diagnostic
shows the provenance chain.

The checker enforces two more spawn rules regardless of `unsafe`,
because they aren't things careful discipline can promise away: no
`@local` captures (the spawning frame may return while the child still
runs, leaving the capture dangling) and no captures of live `&mut`
borrows (an exclusive borrow shared with another thread breaks
exclusivity by construction).

## std.sync: the blessed surface

`std.sync.protect(v)` pairs a value with its own mutex into a
`Protected` handle that's shareable by construction: every access goes
through the lock. `read` and `write` do one access each, `update` is an
atomic read-modify-write, and `with_lock(p, f)` runs `f` on the
underlying one-slot cell with the lock held. The classic counter:

```metaxu
extern type Thread[T];

effect Thread = {
    fn spawn[T](f: fn() -> @global T) -> @global Thread[T] with EFFECT_SPAWN
    fn join[T](thread: @global Thread[T]) -> @global T with EFFECT_JOIN
}

from std.sync import protect, update, read;

fn main() -> int {
    let p = protect(0);
    let @mut handles = Vec.new();
    let @mut i = 0;
    while i < 4 {
        let t = perform Thread.spawn(|| {
            let @mut j = 0;
            while j < 250 {
                update(p, fn(n: int) -> n + 1);
                j = j + 1
            };
            0
        });
        handles.push(t);
        i = i + 1
    };
    let @mut k = 0;
    while k < 4 {
        perform Thread.join(handles[k]);
        k = k + 1
    };
    print(read(p));
    0
}
```
```output
1000
```

Four threads, 250 increments each, and the final value is 1000 on
every run because each `update` locks around its read-modify-write.
Capturing `p` needed no `unsafe` and no annotation: the checker
recognizes `protect(..)` results as safe to share.

Under the hood these are ERRORCHECK mutexes. Locking a mutex held by
another thread blocks until it's released. Locking one this thread
already holds is a loud, catchable deadlock error, and so is unlocking
a mutex this thread doesn't hold; mutexes are not recursive. A second
`join` of the same handle errors the same loud, catchable way.

## Contention as permission

The static check is deliberately shallow: it traces names, not data
flow through structures or helper calls, and `unsafe` turns it off for
a spawn you vouch for. What sits underneath is dynamic. When a closure
crosses a real spawn, every shared-identity value it captures is marked
contended at the crossing. Writing a contended value while holding no
runtime mutex raises a catchable error, with identical wording on both
engines. Reads stay free.

```metaxu
extern type Thread[T];

effect Thread = {
    fn spawn[T](f: fn() -> @global T) -> @global Thread[T] with EFFECT_SPAWN
    fn join[T](thread: @global Thread[T]) -> @global T with EFFECT_JOIN
}

fn main() -> int {
    let @mut v = Vec.new();
    v.push(10);
    let @mut handles = Vec.new();
    unsafe {
        let t = perform Thread.spawn(|| {
            try { v[0] = 11; "written" } catch e { e }
        });
        handles.push(t);
    }
    print(perform Thread.join(handles[0]));
    print(v[0]);
    0
}
```
```output
write to contended Vec without a held lock: this value crossed a thread boundary at spawn; mutate it under a mutex (std.sync.with_lock) or keep it thread-local
10
```

The `unsafe` block got this past the static checker, and the runtime
still refused the unlocked write. Note the second line: the parent
reads `v[0]` after the join and gets 10 without complaint, because the
value is contended for writes only. The permission is any held mutex
(a per-thread count, not a lockset), so two threads writing one vec
under two different mutexes still races; TSan remains the net for that
misuse. What this catches deterministically, in production builds, is
the common bug: mutating shared state with no lock at all.

## Handlers still win

`with EFFECT_SPAWN` is a runtime mapping, not a bypass of the effect
system. A `Thread` handler in scope on the performing thread
intercepts the perform, and no OS thread is created. That's how a test
double or a custom scheduler replaces threading without the spawning
code changing:

```metaxu
extern type Thread[T];

effect Thread = {
    fn spawn[T](f: fn() -> @global T) -> @global Thread[T] with EFFECT_SPAWN
    fn join[T](thread: @global Thread[T]) -> @global T with EFFECT_JOIN
}

fn main() -> int {
    let r = handle Thread with {
        spawn(f) -> {
            print("intercepted; no OS thread");
            resume(1234)
        }
    } in {
        perform Thread.spawn(|| { 1 })
    };
    print(r);
    0
}
```
```output
intercepted; no OS thread
1234
```

One caveat cuts the other way: handlers do not cross `spawn`. Each
thread owns its own effect scope stack, so a child's perform never
reaches a handler the parent installed. A handler scope is a delimited
continuation rooted in the installing thread's stack; letting another
thread suspend it would be a dangling continuation natively and a
deadlock generator in the interpreter. `handle` inside the child works
normally. A failure inside a child that the child doesn't catch marks
the handle errored, and `join` re-raises it, catchably, on the joining
thread.
