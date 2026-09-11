# Traits

A trait names a piece of behavior. An `implement` block supplies that
behavior for one concrete type. At a method call, the implementation
is chosen by the runtime type of the receiver: the value's actual
type at the moment of the call decides which block runs.

## Declaring and implementing

```metaxu
trait Show {
    fn show(self) -> string;
}

struct Dog { name: string }
struct Cat { name: string }

implement Show for Dog {
    fn show(self) -> string { "dog " + self.name }
}

implement Show for Cat {
    fn show(self) -> string { "cat " + self.name }
}

fn main() -> int {
    let d = Dog { name: "rex" };
    let c = Cat { name: "mia" };
    print(d.show());
    print(c.show());
    0
}
```
```output
dog rex
cat mia
```

The trait declares the shape once; each block fills it in for one
type. `d.show()` runs the Dog block because `d` is a Dog at runtime.
No cast, no registration, no vtable you manage.

`self` is the first parameter; anything after it is an ordinary
argument:

```metaxu
trait Scale { fn scale(self, k: int) -> int; }

struct Price { cents: int }

implement Scale for Price {
    fn scale(self, k: int) -> int { self.cents * k }
}

fn main() -> int {
    print(Price { cents: 21 }.scale(2));
    0
}
```
```output
42
```

## One block per trait and type

For a given trait and type there is exactly one `implement` block. A
second is rejected at compile time, even with an identical body:

```metaxu error
trait Show { fn show(self) -> string; }

struct Dog { name: string }

implement Show for Dog {
    fn show(self) -> string { "dog" }
}

implement Show for Dog {
    fn show(self) -> string { "DOG" }
}

fn main() -> int {
    print(Dog { name: "rex" }.show());
    0
}
```
```output
more than one implement block
```

Coherence keeps dispatch predictable: which code runs for `d.show()`
never depends on import order or which file the checker read first.

## Inherent implementations and statics

Drop the trait and you get an inherent implementation: methods that
belong to the type itself. A function without `self` is a static,
called through the type name; that's the idiomatic constructor.

```metaxu
struct Counter { n: int }

implement Counter {
    fn make() -> Counter { Counter { n: 0 } }
    fn bump(self) -> Counter { Counter { n: self.n + 1 } }
    fn get(self) -> int { self.n }
}

fn main() -> int {
    let c = Counter.make();
    print(c.bump().bump().get());
    0
}
```
```output
2
```

`Counter.make()` reaches the static through the type; `.bump()`
dispatches on the value. No trait needed, because no second type will
share this interface.

## Conditional implementations

A generic type can implement a trait on the condition that its
parameter does. The `where` clause states the condition; the body
gets to use it.

```metaxu
trait Show { fn show(self) -> string; }

struct Pair[T] { a: T, b: T }

implement Show for int {
    fn show(self) -> string { self.to_string() }
}

implement Show for Pair[T] where T: Show {
    fn show(self) -> string {
        "(" + self.a.show() + ", " + self.b.show() + ")"
    }
}

fn main() -> int {
    let p = Pair { a: 1, b: 2 };
    print(p.show());
    0
}
```
```output
(1, 2)
```

`Pair[int]` is showable because int is; note that builtin types take
implementations like any other. A `Pair` of a type without `Show`
simply lacks the method, and a `where` bound asking for it fails as
chapter 6 pinned, naming the missing block.

## Your methods beat the builtins

Numbers and bools come with a builtin `to_string`; other values have
a default rendering. An implementation you write takes precedence for
your type:

```metaxu
struct Temp { celsius: int }

implement Temp {
    fn to_string(self) -> string { self.celsius.to_string() + " C" }
}

fn main() -> int {
    let t = Temp { celsius: 20 };
    print(t.to_string());
    0
}
```
```output
20 C
```

Inside the method, `self.celsius.to_string()` is the builtin working
on an int. Outside, `t.to_string()` is yours. Same name, two levels,
no conflict.

## Call position and method position

Method syntax also works on plain functions: `x.f(y)` with no
matching implementation looks for a free function `f` and passes `x`
as its first argument. Going the other way, a plain call prefers your
functions over builtins of the same name.

```metaxu
fn double(n: int) -> int { n * 2 }

fn len(s: string) -> int { 99 }

fn main() -> int {
    print("hello".len());
    print(21.double());
    print(len("hello"));
    0
}
```
```output
5
42
99
```

Three lookups, three rules. `"hello".len()` in method position finds
the builtin string length. `21.double()` has nothing builtin to find
and falls through to the free `double`: uniform function call syntax,
the reason library functions chain like methods. `len("hello")` in
call position resolves to the `len` you wrote, shadowing the builtin.
Chapter 15 returns to precedence across modules.

## When no implementation exists

A `where` bound turns a missing implementation into a compile error.
Without one, dispatch happens at runtime, and a receiver whose type
has no block for the method raises. `try` catches it like any raise;
chapter 9 covers those.

```metaxu
trait Show { fn show(self) -> string; }

struct Dog { name: string }
struct Rock { kg: int }

implement Show for Dog {
    fn show(self) -> string { "dog " + self.name }
}

fn main() -> int {
    print(Dog { name: "rex" }.show());
    let r = Rock { kg: 3 };
    let msg = try {
        r.show()
    } catch e {
        "no show for Rock"
    };
    print(msg);
    0
}
```
```output
dog rex
no show for Rock
```

The raised error names the trait and the type, so the fix is never a
mystery. But prefer the static version: put `where T: Show` on the
generic function and the mistake can't reach runtime at all.
