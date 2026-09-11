# Vectors and collections

Two vector types, on purpose. `Vec` is the growable workhorse: heap
storage, identity semantics, push and pop. `vector[T, N]` is
fixed-size and value-semantic, sized in its type, built for the
numeric and GPU work in chapter 13. Strings round the chapter out.

## Vec

`Vec.new()` makes an empty vector. Growing it is mutation, so the
binding is declared `let @mut`, the collection idiom from chapter 2.
Indexing reads and writes with `[]`, `.len()` counts, `.pop()`
removes and returns the last element.

```metaxu
fn main() -> int {
    let @mut v = Vec.new();
    v.push(10);
    v.push(20);
    v.push(30);
    print(v.len());
    print(v[1]);
    v[1] = 25;
    print(v[1]);
    print(v.pop());
    print(v.len());
    0
}
```
```output
3
20
25
30
2
```

## Literals

A bracketed list builds a `Vec` with its elements in place. Without
`@mut` it's a perfectly good immutable sequence.

```metaxu
fn main() -> int {
    let primes = [2, 3, 5, 7];
    print(primes.len());
    print(primes[3]);
    0
}
```
```output
4
7
```

## A Vec is a handle

Binding a `Vec` to a second name does not copy the elements. Both
names are handles on one buffer, and a push through either is visible
through both. This is what chapter 10 calls shared mutable identity,
and it's the property that makes the thread checker in chapter 11
care about `Vec` specifically.

```metaxu
fn main() -> int {
    let @mut a = Vec.new();
    a.push(1);
    let b = a;
    a.push(2);
    print(b.len());
    0
}
```
```output
2
```

## Out of bounds is a runtime error you can catch

Indexing past the end and popping an empty `Vec` raise at runtime
(the messages are `index out of bounds` and `pop: Vec is empty`).
`try`/`catch` contains them; execution continues after the `catch`
block. Chapter 9 covers the machinery.

```metaxu
fn main() -> int {
    let @mut v = Vec.new();
    v.push(1);
    try {
        print(v[5]);
    } catch e {
        print("index 5 is out of bounds");
    };
    try {
        v.pop();
        v.pop();
    } catch e {
        print("popped too far");
    };
    print("still running");
    0
}
```
```output
index 5 is out of bounds
popped too far
still running
```

Compile errors are different: nothing catches those, because the
program never runs.

## for loops

`for` iterates a `Vec` front to back, binding each element in turn.

```metaxu
fn main() -> int {
    let names = ["ada", "grace", "alan"];
    for name in names {
        print(name);
    }
    0
}
```
```output
ada
grace
alan
```

## Ranges, and the index idiom

`for` also iterates ranges. `0..5` counts 0 through 4; the upper
bound is excluded, so a range's length is exactly the difference.
When you need the position as well as the element, the plain `while`
over indices does it:

```metaxu
fn main() -> int {
    let mut sum = 0;
    for i in 0..5 {
        sum = sum + i;
    }
    print(sum);

    let v = [3, 1, 4];
    let mut i = 0;
    let mut total = 0;
    while i < v.len() {
        total = total + v[i];
        i = i + 1;
    }
    print(total);
    0
}
```
```output
10
8
```

## Sorting

`std.sort` provides `sort` (and `binary_search`, which chapter 12
shows). `sort` gives you back the sorted vector; the original handle
is untouched.

```metaxu
from std.sort import sort;

fn main() -> int {
    let v = [3, 1, 2];
    let s = sort(v);
    print(s[0]);
    print(s[1]);
    print(s[2]);
    0
}
```
```output
1
2
3
```

`std.vec` collects the everyday `Vec` helpers the same way; the
standard library tour in chapter 12 goes module by module.

## Fixed-size vectors

`vector[T, N]` carries its length in its type. Arithmetic on two of
them is elementwise, and unlike `Vec`, assignment copies: a
`vector` is a value, not a handle, so writing through one binding
never shows through another. A fresh `vector[int, N]` with no
elements given starts zeroed. Mixing lengths is a runtime error, not
a compile error, so keep the `N`s honest yourself.

```metaxu
fn main() -> int {
    let a: vector[int, 3] = [1, 2, 3];
    let b: vector[int, 3] = [10, 20, 30];
    let c = a + b;
    print(c[0]);
    print(c[2]);

    let mut d = a;
    d[0] = 99;
    print(d[0]);
    print(a[0]);
    0
}
```
```output
11
33
99
1
```

That last pair of prints is the value semantics: `d` diverged, `a`
didn't. The GPU tiles in chapter 13 push this same idea much
further.

## Strings

Strings know their length, index to 1-character strings, concatenate
with `+`, and order lexicographically with `<` (bools print as `1`
and `0`, per chapter 2).

```metaxu
fn main() -> int {
    let word = "metaxu";
    print(word.len());
    print(word[0]);
    print(word + "!");
    print("apple" < "banana");
    0
}
```
```output
6
m
metaxu!
1
```

There is no separate char type; the element of a string is a string.
`std.string` has the split/join/case helpers, and chapter 12 shows
them alongside the rest of the library.
