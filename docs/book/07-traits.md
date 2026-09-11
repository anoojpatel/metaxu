# Traits

A trait declares methods; an `implement` block provides them for one
type. Calling a method picks the impl by the runtime type of the
receiver, the same way the interpreter and the native backend both do
it (they are pinned to agree).

```metaxu
trait Speak {
    fn speak(self) -> string
}

struct Dog { name: string }
struct Cat { name: string }

implement Speak for Dog {
    fn speak(self) -> string { self.name + " says woof" }
}

implement Speak for Cat {
    fn speak(self) -> string { self.name + " says meow" }
}

fn announce(animal) -> string {
    animal.speak()
}

fn main() -> int {
    print(announce(Dog { name: "Rex" }));
    print(announce(Cat { name: "Tom" }));
    0
}
```
```output
Rex says woof
Tom says meow
```

`announce` doesn't say what type it takes. It doesn't have to: the
`speak` call dispatches on what actually arrives. `self` is the
receiver; methods can read its fields and take further arguments after
it:

```metaxu
trait Greet {
    fn greet(self, other: string) -> string
}

struct Dog { name: string }

implement Greet for Dog {
    fn greet(self, other: string) -> string {
        self.name + " greets " + other
    }
}

fn main() -> int {
    print(Dog { name: "Rex" }.greet("Tom"));
    0
}
```
```output
Rex greets Tom
```

## Inherent impls and static methods

`implement Type { ... }` with no trait attaches methods directly. A
method that neither declares nor uses `self` is static, called as
`Type.method(...)`. The usual constructor idiom:

```metaxu
struct Counter { count: int }

implement Counter {
    fn make(start: int) -> Counter { Counter { count: start } }
    fn value(self) -> int { self.count }
    fn bump(self) -> Counter { Counter { count: self.count + 1 } }
}

fn main() -> int {
    let c = Counter.make(40);
    print(c.bump().bump().value());
    0
}
```
```output
42
```

## Generic impls

An impl target can be a type application, with a `where` clause making
the impl conditional. `Pair[T]` is showable when `T` is:

```metaxu
struct Pair<T> { x: T, y: T }

trait Show { fn show(self) -> string }

implement Show for Pair[T] where T: Show {
    fn show(self) -> string {
        "<" + self.x.show() + ", " + self.y.show() + ">"
    }
}

struct N { v: int }

implement Show for N {
    fn show(self) -> string { self.v.to_string() }
}

fn main() -> int {
    let p = Pair<N> { x: N { v: 1 }, y: N { v: 2 } };
    print(p.show());
    0
}
```
```output
<1, 2>
```

Where clauses over concrete types are enforced when impls load;
conditional ones like the above are enforced per instantiation, at call
sites the checker can resolve (chapter 6 shows the diagnostic).

## Your impls beat the builtins

The runtime ships builtin methods with ordinary names: `to_string`,
`len`, `push`, `sqrt`. Method dispatch resolves a user impl first, then
the builtin, then a plain user function (UFCS). So implementing
`to_string` for your type changes what your type prints as, without
touching what ints do:

```metaxu
trait Show { fn to_string(self) -> string }

struct Temp { celsius: int }

implement Show for Temp {
    fn to_string(self) -> string { self.celsius.to_string() + " deg C" }
}

fn main() -> int {
    print(Temp { celsius: 21 }.to_string());
    print((7).to_string());
    0
}
```
```output
21 deg C
7
```

Plain call position has its own rule (`docs/name_precedence.md`): a
user function beats a same-named builtin, while method position keeps
meaning "the receiver's own operation". Both directions at once:

```metaxu
fn len(x: int) -> int { 99 }

fn double(x: int) -> int { x * 2 }

fn main() -> int {
    print(len(5));          # plain call: your len wins
    print("hello".len());   # method: still the string's length
    print(21.double());     # no impl or builtin named double: UFCS
    0
}
```
```output
99
5
42
```

`std.test` uses the plain-call rule on purpose: importing its
`assert_eq` upgrades every plain `assert_eq(...)` in your file from the
aborting builtin to a tallying assertion.

## When there is no impl

Method lookup that finds nothing is a runtime error, not a silent
no-op, and the message names the receiver type and the impls that do
exist. Like other builtin contract violations it can be caught
(chapter 9):

```metaxu
trait Speak { fn speak(self) -> string }

struct Dog { name: string }
struct Robot { id: int }

implement Speak for Dog {
    fn speak(self) -> string { "woof" }
}

fn main() -> int {
    let r = try { Robot { id: 3 }.speak() } catch e { "caught: " + e };
    print(r);
    0
}
```
```output
caught: Trait method 'speak' is not implemented for type 'Robot' (implementations exist for: Dog)
```

## Coherence

One type gets one impl of a trait. A second
`implement Speak for Dog`, or the same method twice in one block, is
rejected while impls are being desugared, before anything type-checks
or runs: the compiler raises a `CoherenceError` telling you the trait,
the type, and that more than one implement block was found. Two types
implementing the same trait, or one type implementing two traits that
happen to share a method name, are both fine; dispatch keys on the
(trait, type, method) triple, not on the name alone.
