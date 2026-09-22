# Unsafe and FFI

Everything before this chapter is checked code: the type system,
modes, and the borrow checker stand between you and a bad memory
access. `unsafe` opens a door in that wall. Behind it are raw
pointers, manual allocation, and the C boundary. What makes Metaxu's
version unusual is that the reference interpreter runs unsafe code on
a strict simulated heap where every access is validated, so the
classic undefined behaviors of C are defined, catchable errors here.
You get the sharp tools and a machine that tells you the moment you
slip.

## `unsafe` is a statement

The first thing to learn is grammatical. `unsafe { ... }` is a
statement, not an expression. This does not parse:

```
let p = unsafe { malloc(2) };   # parse error: unsafe is a statement
```

Two idioms replace it. Use an unsafe block in statement position and
do your printing or mutation inside it, or make a whole function body
end in an unsafe block: the block's tail expression becomes the
function's return value, exactly as a function body's last expression
always does.

## The heap primitives

Four functions run the manual heap: `malloc(n)` allocates a block of
`n` slots and returns a pointer, `ptr_write(p, i, v)` stores into
slot `i`, `ptr_read(p, i)` loads from it, and `free(p)` releases the
block. Here is the second idiom from above, a function that returns
through its unsafe tail:

```metaxu
fn boxed_sum() -> int {
    unsafe {
        let p = malloc(2);
        ptr_write(p, 0, 41);
        ptr_write(p, 1, 1);
        let total = ptr_read(p, 0) + ptr_read(p, 1);
        free(p);
        total
    }
}

fn main() -> int {
    print(boxed_sum());
    0
}
```
```output
42
```

Nothing reclaims that block for you. Forget the `free` and the block
leaks, in the interpreter and in native code alike. The discipline is
yours again, which is the point of the keyword.

## Strings have an address

`s.as_ptr()` gives a pointer to a string's bytes, and `memcpy(dst,
src, n)` copies `n` slots between pointers. Copy a string's bytes
into a block you own and they're just numbers:

```metaxu
fn main() -> int {
    let s = "hi";
    unsafe {
        let src = s.as_ptr();
        let dst = malloc(2);
        memcpy(dst, src, 2);
        print(ptr_read(dst, 0));
        print(ptr_read(dst, 1));
        free(dst);
    }
    0
}
```
```output
104
105
```

104 and 105 are the ASCII codes for `h` and `i`. This is the shape of
every FFI marshalling routine: get an address, copy across the
boundary, work on the copy.

## The heap is strict

On the simulated heap, a freed block doesn't linger as garbage you
might get away with reading. It's gone, and the interpreter says so.
Use-after-free and double free are runtime errors, and like the
runtime errors of chapter 9 they're catchable with `try`:

```metaxu
fn main() -> int {
    unsafe {
        let p = malloc(1);
        ptr_write(p, 0, 7);
        print(ptr_read(p, 0));
        free(p);
        try {
            print(ptr_read(p, 0));
        } catch e {
            print(e);
        };
        try {
            free(p);
        } catch e {
            print(e);
        };
    }
    0
}
```
```output
7
ptr_read: use after free (<*heap#1>)
free: double free (<*heap#1>)
```

`<*heap#1>` is the pointer's identity: the first heap block this
program allocated. The message names the operation that went wrong
and the block it went wrong on, which is more than a segfault ever
told you.

Out-of-bounds access is caught the same way. A block of two slots has
slots 0 and 1, and slot 2 is a refusal, not a read of whatever lies
next door:

```metaxu
fn main() -> int {
    unsafe {
        let p = malloc(2);
        ptr_write(p, 0, 10);
        ptr_write(p, 1, 20);
        try {
            print(ptr_read(p, 2));
        } catch e {
            print(e);
        };
        free(p);
    }
    0
}
```
```output
ptr_read: out of bounds — [2, 3) outside allocation of 2 bytes (<*heap#1+2>)
```

The message spells out the byte range the read wanted, `[2, 3)`, the
size of the block it fell outside, and the offending address as an
offset from the block's identity, `<*heap#1+2>`.

## Read-only snapshots

The pointer `as_ptr` hands you is a snapshot, and it's read-only.
Strings are immutable values in Metaxu, and no pointer is allowed to
launder that away:

```metaxu
fn main() -> int {
    let s = "hi";
    unsafe {
        let p = s.as_ptr();
        try {
            ptr_write(p, 0, 72);
        } catch e {
            print(e);
        };
    }
    print(s);
    0
}
```
```output
ptr_write: write through read-only pointer (<*heap#1 const>)
hi
```

The final line is the proof: `s` still reads `hi`. To mutate the
bytes, copy them into your own block first, as the `memcpy` example
did. The snapshot rule is the mode system of chapter 10 reaching
through the unsafe door: immutability survives taking an address.

## The C boundary

Extern declarations bind Metaxu names to C symbols; you've already
met one, the `extern type Thread[T]` that chapter 11 spawns through.
The file API is the everyday example. `fopen` is C's `fopen`, and it
keeps C's contract: a path that doesn't exist gets you a null
pointer, not an exception.

```metaxu
fn main() -> int {
    unsafe {
        let f = fopen("definitely-not-here.txt", "r");
        if f == null {
            print("fopen returned null");
        };
    }
    0
}
```
```output
fopen returned null
```

`f == null` is the test, and `null` is a pointer value that exists
only inside `unsafe`; nothing outside the block can hold one, so the
check lives here at the boundary and doesn't leak out. The pattern
for wrapping any C function is the same as wrapping `fopen`: call it
inside `unsafe`, check its sentinel values at the edge, and hand back
an `Option` or a `Result` so callers outside the block never see a
pointer.

## The native boundary

One question hangs over a simulated heap: what happens when the same
program compiles to native code, where `malloc` is the real
allocator? The differential contract from chapter 14 answers it. The
native runtime implements the same checked discipline as the
interpreter, tracking block identity and liveness, so the pinned
messages in this chapter are the program's behavior on both engines,
byte for byte. The checks cost a bounds compare per access, and they
are the language's definition, not a debug mode.

That's the trade this chapter runs on. You give up the checker's
static guarantees inside an `unsafe` block; you do not give up having
semantics. Undefined behavior isn't a performance feature Metaxu
forgot, it's one it declined.
