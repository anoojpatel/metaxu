# Modes and memory

Metaxu has no lifetimes and no garbage collector. Memory safety comes
from modes: a small set of properties the checker attaches to bindings
and propagates through your program. You annotate a mode when you want
a guarantee; unannotated code gets permissive defaults and stays
ordinary. The design follows Jane Street's OxCaml work on modal types
more than it follows Rust.

Three axes, one idea each:

* locality: does this value stay inside its region (`@local`) or may
  it outlive it (`@global`)?
* mutability: is this access mutable (`@mut`) or read-only (`@const`)?
* linearity: how many times may this value be consumed (`once`,
  `separate`, `many`)?

The axes are independent. A value can be local and mutable, global and
const, any combination.

## Locality

A `@local` binding belongs to the region (roughly, the function frame)
that created it. The checker rejects any path that would let it
escape. Returning one is the simplest escape:

```metaxu error
fn leak() -> int {
    let @local x = 1;
    x
}

fn main() -> int {
    print(leak());
    0
}
```
```output
Local variable 'x' cannot escape its region
```

The native backend uses the same information for allocation: locals
live on the stack, `@global` values get heap storage. The rule and the
representation are one decision.

When you do want a local value to cross outward, say so. `exclave e`
evaluates `e` and hands a copy to the enclosing region. The escape is
in the source, not smuggled through an alias:

```metaxu
fn give() -> int {
    let @local x = 41;
    exclave x + 1
}

fn main() -> int {
    print(give());
    0
}
```
```output
42
```

Unannotated bindings default to `@global` behavior and return freely.
Annotating `@local` is a promise you're asking the checker to hold you
to, and stack allocation is what the promise buys.

## Mutability and borrows

`let` binds a name; `let mut` makes the name rebindable. The `@mut`
mode is about the value: an exclusive claim on mutating it. Taking two
exclusive borrows of one value is the classic aliasing bug, and the
checker stops it:

```metaxu error
fn main() -> int {
    let @mut x = 5;
    let @mut r1 = @mut x;
    let @mut r2 = @mut x;
    0
}
```
```output
Cannot borrow x as exclusive while borrowed
```

Shared reads don't conflict with each other; an exclusive borrow
conflicts with everything, including a live shared borrow. Struct
fields carry modes too (`@mut value: int`, `@const name: string`,
`@local temp: int` in a declaration), and the deep rules propagate
locality through fields: a `@global` struct can't hold a reference to
`@local` data. The checker walks the whole shape, not just the top
binding. A field declared `@const` rejects assignment, through let
bindings, parameters, and nested paths alike:

```metaxu error
struct P { @const name: string, @mut n: int }

fn main() -> int {
    let @mut p = P { name: "a", n: 1 };
    p.name = "b";
    0
}
```
```output
cannot assign to @const field 'name' of P
```

The same discipline applies to plain bindings: `let` without `mut` is
immutable, and so are parameters and module constants (chapter 2 pins
the error).

## Linearity

The third axis bounds consumption. `once` values must be consumed
exactly once (a file handle, a continuation). `many` values can be
used freely. Between them, `separate` is the interesting one: a value
with no shared mutable identity, safe to hand to another thread.

You rarely write these. Linearity mostly surfaces through inference,
and the place you'll meet it is threads.

## Thread-safety is inferred

Metaxu never asks you to declare Send or Sync. Instead the checker
looks at what a spawned closure captures. A `Vec` has identity
semantics: two handles to it alias one buffer, so letting a second
thread write through a captured handle is a race, and the checker
names it:

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

Read the message closely, because it states the model: reads are free.
Crossing a value into another thread weakens access, it does not
revoke it. Writes are what need a permission, and the sanctioned way
to get one is `std.sync.protect`, which pairs the value with its own
mutex so every write goes through a lock. Chapter 11 shows that in
full, along with what happens if you take the `unsafe` route instead:
the value is marked contended at spawn, and an unpermitted write
raises the same error, with the same wording, in the interpreter and
in native code.

## What there isn't

No lifetime parameters. No borrow-checker annotations on every
function signature. No `Rc`/`RefCell` escape hatches, because the
default mode already is shared-and-permissive; you opt into strictness
where you want the machine to hold a line. The cost of this design is
that some guarantees are dynamic where Rust's are static. The
contended-write check is the honest example: caught statically where
provable, dynamically otherwise, one error either way.
