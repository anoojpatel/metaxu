# Effects

An effect is an operation a function can perform without saying how it
gets done. The caller decides that, by installing a handler. Metaxu
uses this one mechanism for things other languages build separately:
exceptions, generators, mutable state, threads, GPU launches. All of
them are effects with different handlers.

## Declaring and performing

`effect` declares a named set of operations. `perform` invokes one.

```metaxu
effect Say {
    say(msg: string) -> Unit
}

fn greet() performs Say {
    perform Say.say("hello");
    perform Say.say("goodbye")
}

fn main() -> int {
    handle Say with {
        say(msg) -> {
            print(msg);
            resume(())
        }
    } in {
        greet()
    };
    0
}
```
```output
hello
goodbye
```

The `performs Say` clause on `greet` says the function needs a `Say`
handler somewhere above it. Inside the handler arm, `resume(v)`
continues the suspended computation, delivering `v` as the value of the
`perform` expression. The continuation is delimited and single-shot:
it runs from the perform site to the end of the handled block, and you
can call it once.

## Handlers that answer

Operations return values. Whatever you pass to `resume` is what the
`perform` expression evaluates to.

```metaxu
effect Ask {
    ask() -> int
}

fn add_two_answers() -> int performs Ask {
    perform Ask.ask() + perform Ask.ask()
}

fn main() -> int {
    let total = handle Ask with {
        ask() -> resume(21)
    } in {
        add_two_answers()
    };
    print(total);
    0
}
```
```output
42
```

## Handlers that abort

An arm doesn't have to resume. If it returns without calling `resume`,
the rest of the handled block is torn down and the arm's value becomes
the value of the whole `handle` expression. That's an exception, built
from nothing but a handler.

```metaxu
effect Bail {
    bail(code: int) -> Unit
}

fn work() -> int performs Bail {
    print("before");
    perform Bail.bail(3);
    print("after");    # never runs
    99
}

fn main() -> int {
    let r = handle Bail with {
        bail(code) -> code
    } in {
        work()
    };
    print(r);
    0
}
```
```output
before
3
```

`std.throw`, `std.fail`, and `std.early_return` are this pattern with
names; chapter 9 uses them.

## State without mutation

The classic demonstration: `get` and `set` as operations, interpreted
by handlers. This is `std.state`'s design in miniature.

```metaxu
effect Counter {
    get() -> int
    bump() -> Unit
}

fn count_to_three() performs Counter {
    perform Counter.bump();
    perform Counter.bump();
    perform Counter.bump();
    print(perform Counter.get())
}

fn main() -> int {
    let @mut cell = Vec.new();
    cell.push(0);
    handle Counter with {
        get() -> resume(cell[0])
        bump() -> {
            cell[0] = cell[0] + 1;
            resume(())
        }
    } in {
        count_to_three()
    };
    0
}
```
```output
3
```

The handler closes over `cell`, so state lives in the handler, not the
code performing the operations. Swap the handler and the same
`count_to_three` logs every bump, or counts down, or feeds a test.

## Default implementations

An operation can carry a default. Performing it with no handler in
scope runs the default; installing a handler overrides it. This is how
`std.gpu` ships a launch that works everywhere:

```metaxu
fn run_twice(f: fn(int) -> ()) -> () {
    f(0);
    f(1);
    ()
}

effect Job {
    submit(f: fn(int) -> ()) -> () = run_twice(f)
}

fn main() -> int {
    # no handler: the default runs the job twice, inline
    perform Job.submit(fn(i: int) -> print(i));
    # a handler can virtualize the same perform completely
    handle Job with {
        submit(f) -> {
            print("intercepted");
            resume(())
        }
    } in {
        perform Job.submit(fn(i: int) -> print(i * 100))
    };
    0
}
```
```output
0
1
intercepted
```

In `std.gpu`, `Gpu.launch(n, f)` defaults to a sequential CPU loop in
pid order. The Metal handler intercepts the same operation and sends
the kernel to a GPU. Kernel code doesn't change; chapter 13 shows it.

## Streams are effects

`std.stream` builds iteration from a single `Emit` effect, following
Ante's stdlib. A stream is a thunk that performs `emit` once per
element. Consumers are handlers around calling it. Transformers are
handlers that re-emit.

```metaxu
from std.stream import iota, map, filter, sum;

fn main() -> int {
    let total = sum(filter(map(iota(200000), fn(x: int) -> x * 2),
                           fn(x: int) -> x > 5));
    print(total);
    0
}
```
```output
39999799994
```

No intermediate collections exist here. Each element flows from `iota`
through `map`'s handler, `filter`'s handler, into `sum`'s accumulator,
one at a time, by suspending and resuming. `for_` in the same module
adds `break_`/`continue_` as abort-style operations of a `Loop` effect,
which is all a for loop is.

## Effect classes

Every effect belongs to a class that bounds what its handlers may do.
A `stack` effect's handlers finish or abort on the spot; a `suspend`
effect's handlers may hold the continuation across a suspension (this
is what threads and schedulers need). The emitter checks the class, and
the native backend uses it to decide when a function needs the
coroutine machinery. You'll mostly never write the class; the checker
infers the need.

Some effects map straight onto the runtime instead of user handlers:
`Thread.spawn` and `Mutex.lock` are declared `with EFFECT_SPAWN`,
`with EFFECT_MUTEX_LOCK`, and so on, binding them to pthreads natively
and to real OS threads in the interpreter. Chapter 11 is about those.
A handler can still interpose on them, which is how tests virtualize
spawning.
