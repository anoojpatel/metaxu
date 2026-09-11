# Threads and sharing

Chapter 8 ended on effects that map straight onto the runtime instead
of onto user handlers. Threads are those effects. `Thread.spawn` and
`Thread.join` are ordinary operations: a program performs them, a
handler can interpose, and when none does, the `with EFFECT_SPAWN`
declarations bind them to real OS threads, pthreads natively and
host threads in the interpreter. The unusual part is not the API.
It's that the checker decides what may cross into a spawned closure
by reading your code, with no Send or Sync annotations anywhere.

## Spawn and join

The declaration comes first. `extern type Thread[T]` names an opaque
runtime type; the effect declaration binds each operation to its
runtime implementation:

```metaxu
extern type Thread[T];

effect Thread = {
    fn spawn[T](f: fn() -> @global T) -> @global Thread[T] with EFFECT_SPAWN
    fn join[T](thread: @global Thread[T]) -> @global T with EFFECT_JOIN
}

fn main() -> int {
    let t = perform Thread.spawn(|| {
        21 * 2
    });
    let answer = perform Thread.join(t);
    print(answer);
    0
}
```
```output
42
```

`spawn` takes a zero-argument closure, starts it on its own thread,
and returns a handle. `join` blocks until the closure finishes and
hands back its value. The `with EFFECT_SPAWN` clause is the entire
foreign interface: it tells the checker no handler is required in
scope, and tells the emitter which runtime call to make. You write
this prelude once per file; `std.sync` writes the same one.

The child may run whenever the OS likes; the example is
deterministic anyway, because nothing prints until `join` has the
result. Every runnable example here keeps that discipline, since
the harness pins stdout byte for byte.

There is a `Mutex` effect in the same style, its operations bound by
`EFFECT_MUTEX_LOCK` and `EFFECT_MUTEX_UNLOCK`. You'll rarely perform
it directly: `std.sync` wraps it, pairing each lock with the data it
guards, and that pairing is why the wrapper is the blessed surface.

## What a closure may capture

Chapter 10 stated the rule; this chapter lives by it. At every
spawn the checker looks at what the closure captures. Values
without identity cross freely. A `Vec` is different: two handles
alias one buffer, so a second thread writing through a captured
handle is a race. The checker rejects and names the capture:

```metaxu error
extern type Thread[T];

effect Thread = {
    fn spawn[T](f: fn() -> @global T) -> @global Thread[T] with EFFECT_SPAWN
    fn join[T](thread: @global Thread[T]) -> @global T with EFFECT_JOIN
}

fn main() -> int {
    let @mut shared = Vec.new();
    shared.push(1);
    let t = perform Thread.spawn(|| {
        shared.push(2);
        0
    });
    perform Thread.join(t);
    0
}
```
```output
captures AND WRITES 'shared', which has shared mutable identity
```

The diagnostic is the model in one line. Capturing is fine.
Reading is fine: crossing a value into another thread weakens
access, it does not revoke it. Writes need a permission, and no
`Send` bound or wrapper type is involved anywhere.

## std.sync

The sanctioned permission is `std.sync.protect`, which pairs a
value with its own mutex. `read(p)` takes the lock, copies the
value out, and releases. `write(p, v)` replaces the value under the
lock. `update(p, f)` applies a function under the lock, the form
you want for read-modify-write, since a separate `read` then
`write` would let another thread slip between them. Four threads,
250 locked increments each:

```metaxu
from std.sync import protect, read, update;

extern type Thread[T];

effect Thread = {
    fn spawn[T](f: fn() -> @global T) -> @global Thread[T] with EFFECT_SPAWN
    fn join[T](thread: @global Thread[T]) -> @global T with EFFECT_JOIN
}

fn main() -> int {
    let counter = protect(0);
    let @mut handles = Vec.new();
    let mut i = 0;
    while i < 4 {
        handles.push(perform Thread.spawn(|| {
            let mut j = 0;
            while j < 250 {
                update(counter, fn(n: int) -> n + 1);
                j = j + 1;
            }
            0
        }));
        i = i + 1;
    }
    for t in handles {
        perform Thread.join(t);
    }
    print(read(counter));
    0
}
```
```output
1000
```

The interleaving of those thousand increments is up to the
scheduler, and the count is 1000 regardless: each increment is
atomic with respect to the others, so the interleaving stops
mattering. A protected value captures cleanly into any number of
closures; the checker recognizes `protect` as exactly the
permission it was asking for.

## Contention as permission

Suppose you dodge the static check. The write below hides inside a
helper, and the capture checker doesn't chase writes through calls,
so the program compiles. It still can't race: spawning marks the
captured `nums` as contended, and an unpermitted write to a
contended value raises, with one wording on both engines. The child
catches it with `try` and returns the message as its value, so
`join` surfaces it deterministically:

```metaxu
extern type Thread[T];

effect Thread = {
    fn spawn[T](f: fn() -> @global T) -> @global Thread[T] with EFFECT_SPAWN
    fn join[T](thread: @global Thread[T]) -> @global T with EFFECT_JOIN
}

fn bump(v: Vec[int]) -> () {
    v.push(99);
    ()
}

fn main() -> int {
    let @mut nums = Vec.new();
    nums.push(7);
    let t = perform Thread.spawn(|| {
        let seen = nums[0];    # reads stay free: seen is 7
        try {
            bump(nums);
            "wrote " + seen.to_string()
        } catch e {
            e
        }
    });
    print(perform Thread.join(t));
    print(nums[0]);
    0
}
```
```output
contended write to a value with shared mutable identity; protect it with std.sync
7
```

The raise speaks the same shared-mutable-identity vocabulary as
the compile error, because it's the same rule enforced later. The
child's read of `nums[0]` succeeded; the write did not, so the
parent's Vec still holds one element. This is contention as
permission: caught statically where provable, dynamically
otherwise, one behavior either way. Chapter 10 called that the
honest cost of having no lifetimes; this is the dynamic half.

## Virtualizing spawn

A runtime binding is a default, not a monopoly. Install a handler
over the mapping and the perform goes to you instead; no OS thread
is created:

```metaxu
extern type Thread[T];

effect Thread = {
    fn spawn[T](f: fn() -> @global T) -> @global Thread[T] with EFFECT_SPAWN
    fn join[T](thread: @global Thread[T]) -> @global T with EFFECT_JOIN
}

fn main() -> int {
    let r = handle Thread with {
        spawn(f) -> {
            print("no thread spawned");
            resume(f())
        }
        join(t) -> resume(t)
    } in {
        let t = perform Thread.spawn(|| {
            3 + 4
        });
        perform Thread.join(t)
    };
    print(r);
    0
}
```
```output
no thread spawned
7
```

This handler runs the closure inline and resumes with its result. A
`Thread[T]` is whatever the handler says it is; here it's the
finished value, and `join` hands it straight back. Tests use this
shape to make scheduling deterministic, and a `suspend`-class
handler (chapter 8) can hold the continuations and interleave them
instead, which is a cooperative scheduler in a page of code.

## Scopes and failures

Two facts round out the model, both about isolation. A spawned
closure starts with a fresh handler stack: parent handlers do not
span into the child, so runtime-bound operations and defaults work
there but anything else raises `No handler for effect ...` in the
child. And a raise the child doesn't catch is held and rethrown at
`join`, where an ordinary `try` catches it. Nothing about a
thread's failure is asynchronous: you meet it exactly where you
asked for the result.
