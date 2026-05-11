# Metaxu Effect System / Continuation Design (V1)

## Core philosophy

Metaxu uses:

* algebraic effects + handlers
* CPS-based MIR
* single-shot continuations
* ownership/locality/modal typing
* fibers/async/multithreading implemented via suspendable effects

Primary design goal:

```text
Practical systems/runtime semantics with efficient lowering to C/Cranelift.
```

Avoid full general multi-shot continuations in V1 and design for continuation cloning if multi-shot is desired.

---

# 1. Continuation model

## V1 continuations are SINGLE-SHOT ONLY

A continuation may:

```text
resume exactly once
```

and is then consumed.

This supports:

* async/await
* fibers
* schedulers
* channels
* thread pools
* actors
* structured concurrency

without continuation cloning overhead.

---

# 2. Effect classes

V1 has exactly two effect classes:

```text
stack
suspend
```

## Stack effects

Semantics:

```text
Continuation may not escape current dynamic extent.
```

Handler restrictions:

* cannot store continuation
* cannot enqueue continuation
* cannot resume later
* cannot move continuation to another thread

Equivalent to:

```text
fancy function call
```

Safe for:

* local values
* local refs
* exclusive refs

Example:

```metaxu
effect Console : stack {
  op print(msg: &local Str) -> Unit
}
```

---

## Suspend effects

Semantics:

```text
Continuation may escape and resume later.
```

Handler MAY:

* store continuation
* enqueue continuation
* resume later
* migrate continuation to scheduler/thread

Equivalent to:

```text
capturing time/control
```

Examples:

* yield
* await
* recv
* async runtime
* scheduler operations

Example:

```metaxu
effect Async : suspend {
  op yield() -> Unit
  op await(task: Task[A]) -> A
}
```

---

# 3. Safety rules

## Stack effects

Locals may cross freely.

Allowed:

```metaxu
let local x = Buffer(...)
perform Console.print(&x)
```

---

## Suspend effects

No local values/refs may live across suspension unless explicitly promoted.

Rejected:

```metaxu
let local x = Buffer(...)

perform Async.yield()

use(x)
```

because continuation captures `x`.

Core rule:

```text
No stack-local values may be live across a suspend effect.
```

---

# 4. Continuation capture analysis

Compiler performs free-variable/liveness analysis on continuation environments.

A continuation captures all live variables after a `perform`.

Example:

```metaxu
let x = 1
let y = Buffer(...)

perform Async.yield()

print(x)
use(y)
```

Continuation captures:

* x
* y

Suspend effects therefore require captured vars to be escapable.

---

# 5. Local/global regions

Metaxu distinguishes:

```text
local region
global region
```

Local:

* stack/fiber-local lifetime
* non-escaping

Global:

* heap/promoted/escapable

---

# 6. Promotion

Escaping data must be explicitly promoted.

Example:

```metaxu
let local x = Buffer(...)

let gx = promote x

perform Thread.spawn(move || use(gx))
```

Promotion converts:

* local owned values → global owned values

Rejected:

* local refs
* exclusive local borrows escaping

---

# 7. Threading model

Multithreading uses suspend effects.

Example:

```metaxu
effect Thread : suspend {
  op spawn(f: global once Fn() -> Unit + Send)
}
```

Thread/task migration requires:

```text
global + Send + owned
```

Closures passed to threads cannot capture local refs.

---

# 8. MIR design

MIR is CPS-based.

Core representation:

```rust
enum ContEscape {
    Stack,
    Suspend,
}

enum ContUse {
    Once,
}

struct Continuation {
    escape: ContEscape,
    use_mode: ContUse,
    captures: Vec<ValueId>,
}
```

---

# 9. Lowering strategy

## Stack effects

Lower to:

* direct calls
* tailcalls
* inline handler dispatch

No heap continuation required.

---

## Suspend effects

Lower to:

* heap continuation/fiber frame
* scheduler queue entries
* state machine/fiber resumption

Continuation environment is closure-converted.

---

# 10. Async runtime semantics

Suspend effects are the foundation of:

* async
* fibers
* green threads
* schedulers
* channels

NOT multi-shot continuations.

All runtime resumptions are one-shot.

---

# 11. Multi-shot continuations (NOT V1)

Not part of V1 semantics.

Possible future addition.

Would require:

```text
explicit continuation cloning
```

Example future syntax:

```metaxu
resume clone(k)(x)
```

Only allowed if continuation captures are safely duplicable.

Use cases:

* nondeterminism
* backtracking
* probabilistic programming
* theorem proving
* symbolic execution

NOT required for async/threading.

---

# 12. Design rationale

Metaxu intentionally follows the practical direction of:

* one-shot continuations
* efficient runtime lowering
* ownership-compatible async semantics

rather than fully general continuation semantics.

Key conceptual split:

```text
stack effect   = ordinary control flow

suspend effect = continuation may outlive current stack/frame
```

This is the central semantic boundary of the V1 effect system.
