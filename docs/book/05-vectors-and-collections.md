# Vectors and collections

Metaxu has two vector types with two different characters. `Vec` is
the growable heap collection: push, pop, index, one identity shared by
everyone holding it. `vector[T, N]` is a fixed-size value: its length
is part of its type, it copies on assignment, and arithmetic on it is
elementwise. This chapter covers both, plus strings.

## Vec

`Vec.new()` makes an empty vector; a literal `[a, b, c]` makes a full
one. `push` appends, `pop` removes and returns the last element,
`len` answers the count, and `v[i]` indexes from zero, for reading or
writing:

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
    let primes = [2, 3, 5, 7];
    print(primes[3]);
    0
}
```
```output
3
20
25
30
2
7
```

`len(v)` and `v.len()` are the same call; `std/` uses the function
form, most examples the method form.

## Vec has identity

Assigning a `Vec` to another name or passing it to a function hands
over the same vector, not a copy. That's the point: a helper that
pushes into its argument is how you build things.

```metaxu
fn extend_with_squares(target: Vec, upto: int) {
    let @mut i = 0;
    while i < upto {
        target.push(i * i);
        i = i + 1
    }
}

fn main() -> int {
    let @mut v = Vec.new();
    let same = v;
    extend_with_squares(v, 4);
    print(same.len());
    print(same[3]);
    0
}
```
```output
4
9
```

Both `same` and the callee saw `v` itself change. Contrast with
structs, which chapter 2 showed pass as copies unless the parameter
says `@mut`.

## Out of bounds is a failure you can catch

Indexing past the end doesn't read garbage and doesn't kill the
process behind your back. It's a runtime failure, and `try`/`catch`
(chapter 9 has the full story) turns it into a value:

```metaxu
fn main() -> int {
    let v = [1, 2];
    let got = try {
        v[10]
    } catch e {
        print("caught: " + e);
        -1
    };
    print(got);
    let @mut empty = Vec.new();
    let popped = try { empty.pop() } catch e { 0 };
    print(popped);
    0
}
```
```output
caught: index out of bounds: 10 (length 2)
-1
0
```

## Iterating

`for` walks a `Vec`, or an integer range `a..b` (end exclusive).
When you need the index too, use the `while i < len(v)` loop, the
idiom all of `std/vec.mx` is written with; `extend_with_squares`
above is one:

```metaxu
fn main() -> int {
    let names = ["ada", "grace", "edsger"];
    for name in names {
        print(name);
    }
    for k in 0..3 {
        print(k * 10);
    }
    0
}
```
```output
ada
grace
edsger
0
10
20
```

There is no slice syntax; `std.sort` builds its halves with an
explicit copy loop for the same reason you would.

## std.vec and std.sort

The combinators live in the library, not the language. `import
std.vec` gives map, filter, sum, reverse and friends, eager and
Vec-in, fresh-Vec-out; `std.sort` adds a stable merge sort driven by a
strict less-than predicate:

```metaxu
import std.vec
import std.sort

fn main() -> int {
    let v = std.vec.range_vec(1, 6);
    print(std.vec.sum(v));
    let evens = std.vec.filter(v, fn(x) -> x % 2 == 0);
    print(evens.len());
    let sorted = std.sort.sort_desc([3, 1, 4, 1, 5]);
    print(sorted[0]);
    print(sorted[4]);
    0
}
```
```output
15
2
5
1
```

## Fixed vectors: vector[T, N]

A `vector[T, N]` holds exactly `N` elements of `T`. Construct it with
values, with nothing (zero-initialized), or with a comprehension.
Arithmetic is elementwise, and the operands' sizes must agree:

```metaxu
fn main() -> int {
    let a = vector[int, 4](1, 2, 3, 4);
    let b = vector[int, 4](10, 20, 30, 40);
    let c = a + b;
    print(c.to_string());
    print((a * a).to_string());
    let z = vector[int, 3]();
    print(z[0]);
    let sq = vector[float, 4](x * 0.5 for x in 0..4);
    print(sq.to_string());
    0
}
```
```output
vector[11, 22, 33, 44]
vector[1, 4, 9, 16]
0
vector[0.0, 0.5, 1.0, 1.5]
```

Because `N` is in the type, the native backend compiles these to real
SIMD when it can prove the size, which is why they exist as a
separate type instead of a small `Vec`.

## Fixed vectors are values

Unlike `Vec`, a `vector[T, N]` copies on assignment. Writing through
one name never surprises another:

```metaxu
fn main() -> int {
    let @mut a = vector[int, 3](1, 2, 3);
    let @mut b = a;
    b[0] = 99;
    print(a[0]);
    print(b[0]);
    0
}
```
```output
1
99
```

## Strings

Strings concatenate with `+`, compare with `==` and order with `<`,
and index to one-character strings; there's no separate char type.
`len` counts characters:

```metaxu
fn main() -> int {
    let s = "metaxu";
    print(len(s));
    print(s[2]);
    print(s + "!");
    if "abc" < "abd" {
        print("abc sorts first");
    }
    let @mut spelled = "";
    let @mut i = len(s);
    while i > 0 {
        spelled = spelled + s[i - 1];
        i = i - 1
    }
    print(spelled);
    0
}
```
```output
6
t
metaxu!
abc sorts first
uxatem
```

`std.string` packages the loops you'd otherwise rewrite:
`starts_with`, `count_char`, `repeat`, `join` over a `Vec` of parts,
and more:

```metaxu
import std.string

fn main() -> int {
    print(std.string.repeat("ab", 3));
    if std.string.starts_with("metaxu", "meta") {
        print("has the prefix");
    }
    print(std.string.join(["a", "b", "c"], ", "));
    print(std.string.count_char("banana", "a"));
    0
}
```
```output
ababab
has the prefix
a, b, c
3
```

That's the eager story. `std.stream` and `std.iter` do the lazy
version, pipelines built from the `Emit` effect, and they belong to
chapter 8, after effects themselves. Next, though: the types you never
had to write in any of these examples, and where they came from.
