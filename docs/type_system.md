# Metaxu Type System

## Type Definitions and Applications

### Type Definitions
Type definitions in Metaxu use angle brackets `<>` and introduce new type parameters:

```metaxu
effect Reader<T> {
    read() -> T
}

type List<T> { # Declare a generic type with a type parameter
    head: T
    tail: Option[List[T]]
}
```

### Type Applications
Type applications use square brackets `[]` to provide concrete types or type parameters:

```metaxu
let numbers: List[int] = ...
handle Reader[String] with { ... }
```
```metaxu
type List<T> {
    head: T
    tail: Option[List[T]] # Type annotation with type parameter and recursive positive type
}
```

## Type Inference

Metaxu uses bidirectional type inference based on SimpleSub, which combines:

1. **Local Type Inference**: Infers types from local context
   ```metaxu
   let x = 42;  // Inferred as int
   let pairs = [(1, "one"), (2, "two")];  # Inferred as List[(int, String)]
   ```

2. **Flow-Based Inference**: Uses control flow information
   ```metaxu
   fn get_value<T>() -> Option[T] {
       if condition {
           Some(42)  # T inferred as int
       } else {
           None
       }
   }
   ```

3. **Constraint-Based Inference**: Solves type constraints
   ```metaxu
   fn map<T, U>(list: List[T], f: fn\(T) -> U) -> List[U] {
       # T and U inferred from usage
   }
   ```

## Type Parameter Inference

Type parameters can be either explicit or inferred:

1. **Explicit Type Parameters**: Declared using angle brackets
   ```metaxu
   fn get_value<T>() -> Option[T] {
       if condition {
           Some(42)  # T is explicitly declared and inferred as int
       } else {
           None
       }
   }
   ```

2. **Implicit Type Parameters**: Inferred from usage
   ```metaxu
   fn get_value() -> Option[T] {  # T is implicitly created
       if condition {
           Some(42)  # T is inferred as int
       } else {
           None
       }
   }
   ```

While both styles are supported, explicit type parameters are recommended as they:
- Make the code more self-documenting
- Allow adding constraints on type parameters
- Help catch type errors earlier
- Make type checking more predictable

## Subtyping and Variance

### Variance Annotations
Variance Annotations are used to specify the subtyping relationship between type parameters. They are added to type parameters in type definitions; **However, they are inferred for type applications** and not necessary. 

Variance annotations can be any of the following:
- **Covariant** (`+T`): Preserves subtyping relationship
  ```metaxu
  type Box<+T> { value: T }  # If Dog <: Animal then Box[Dog] <: Box[Animal]
  ```
  
- **Contravariant** (`-T`): Reverses subtyping relationship
  ```metaxu
  type Consumer<-T> { consume: fn(T) -> Unit }  # If Dog <: Animal then Consumer[Animal] <: Consumer[Dog]
  ```

- **Invariant** (default): No subtyping relationship
  ```metaxu
  type Cell<T> { mut value: T }  # No subtyping relationship between Cell[Dog] and Cell[Animal]
  ```

Our SimpleSub implementation automatically infers variance by:

1. Tracking how each type parameter is used (through type_var_positions)
2. If a type parameter appears only in positive (output) positions -> inferred as covariant
3. If it appears only in negative (input) positions -> inferred as contravariant
4. If it appears in both or neither -> inferred as invariant

### Effect Variance
Effects follow specific variance rules:

```metaxu
effect State<T> {
    get() -> T       # Covariant position
    put(x: T)        # Contravariant position
}
```

## Type Positions and Polarity

Type positions determine how types flow through the program and affect subtyping relationships. A type parameter's position (positive or negative) determines its variance capabilities.

### Positive Positions
A type parameter appears in a positive position when it's used as:
- Return type
- Field type
- Type argument to a covariant parameter
- Right side of a function arrow (->)

```metaxu
type Box<T> {
    value: T  # T in positive position
}

fn produce() -> T  # T in positive position
```

### Negative Positions
A type parameter appears in a negative position when it's used as:
- Function parameter type
- Type argument to a contravariant parameter
- Left side of a function arrow (->)

```metaxu
fn consume(x: T) { ... }  # T in negative position
type Consumer<T> {
    handler: fn(T) -> Unit  # T in negative position
}
```

### Mixed Positions
A type parameter can appear in both positions:

```metaxu
type Function<A, B> {
    apply: fn(A) -> B  # A in negative position, B in positive position
}
```

### Complex Polarity Examples

Higher-kinded type parameters also follow polarity rules:

```metaxu
fn sequence<F<A>, T>(x: F[Option[T]]) -> Option[F[T]] { ... }
```

In this example:
1. `F<A>` is a higher-kinded type parameter (type constructor that takes one type argument)
2. `T` appears in both:
   - Negative position: `F[Option[T]]` (input parameter)
   - Positive position: `Option[F[T]]` (return type)
3. `F` appears in both:
   - Negative position: `F[Option[T]]` (input)
   - Positive position: `F[T]` (inside return type)

This polarity pattern is common in monadic operations where types are "rewritten" while preserving their structure.

Another example with nested polarity:
```metaxu
type Transformer<F<A>, G<B>, A, B> {
    transform: fn(F[A]) -> G[B]  # F,A in negative position, G,B in positive
}
```

### Higher-Kinded Polymorphism
Type constructors as parameters:
```metaxu
# Using explicit type parameter
fn sequence<F<A>, T>(x: F[Option[T]]) -> Option[F[T]] { ... }

# Two-parameter type constructor
fn transform<F<A, B>, T, U>(x: F[T, U]) -> F[U, T] { ... }

# Nested type constructors though we can just use Algebraic Effects!
type Monad<M<A>> {
    bind: fn(M[A], fn(A) -> M[B]) -> M[B]
}
```

### Polarity and Recursion
Recursive type definitions must respect positivity to ensure type safety:

```metaxu
// Valid: positive recursion
type List<T> {
    head: T
    tail: Option[List[T]]  # List appears in positive position
}

// Invalid: negative recursion
type Bad<T> {
    f: fn(Bad[T]) -> T  # Bad appears in negative position
}
```

### Effect Operations and Polarity
Effect operations follow polarity rules for type safety:

```metaxu
effect State<T> {
    get() -> T       # T in positive position (output)
    put(x: T)        # T in negative position (input)
}

effect Transform<A, B> {
    map(f: fn\(A) -> B)  # A in negative, B in positive position
}
```

### Polarity in Type Inference
The type checker uses polarity information to:
1. Verify recursive type definitions
2. Determine variance capabilities
3. Guide type inference in constraint solving
4. Ensure soundness of subtyping

Example of polarity-aware type inference:
```metaxu
fn transform<T, U>(x: T, f: fn\(T) -> U) -> U {
    # T appears in both positive (x: T) and negative (f: fn(T) -> U) positions
    # U appears only in positive position (return type)
    f(x)
}
```

### Practical Impact
Understanding type positions helps in:
1. Designing safe generic types
2. Understanding subtyping relationships
3. Writing effect handlers correctly
4. Avoiding unsound recursive types
5. Working with the type inference system

## Understanding Positions, Variance, and Polarity

The relationship between these concepts is fundamental to type safety:

1. **Positions** refer to where a type parameter appears:
   - **Positive Position**: Output/Return type (produces values)
   ```metaxu
   fn get() -> T       // T in positive position (produces T)
   ```
   - **Negative Position**: Input/Parameter type (consumes values)
   ```metaxu
   fn set(x: T)        // T in negative position (consumes T)
   ```

2. **Polarity** tracks how positions compose:
   - Going under a negative position flips polarity
   ```metaxu
   fn transform(f: fn(T) -> U) {  
       // T is in positive position within fn(T) -> U
       // But fn(T) -> U is in negative position
       // So T ends up in negative position overall
   }
   ```

3. **Variance** is determined by the positions where a type appears:
   - Only positive positions -> Covariant (+T)
   - Only negative positions -> Contravariant (-T)
   - Both or neither -> Invariant (T)

Example showing all three concepts:
```metaxu
type Transformer<A, B, C> {
    // A appears only in negative positions -> Contravariant
    input: fn(A) -> B,  
    
    // B appears in both positive and negative positions -> Invariant
    process: fn(B) -> B,
    
    // C appears only in positive positions -> Covariant
    output: fn(B) -> C   
}
```

## Subtyping Rules
Notation: `<:` means "subtype of".  
Metaxu supports several forms of subtyping:

1. **Nominal Subtyping** with visibility rules:
   ```metaxu
   struct Animal { name: String }
   struct Dog: Animal { # How we make Dog a subtype of Animal
       breed: String  # Dog is a subtype of Animal
   }
   ```

2. **Variance-Based Subtyping** (inferred automatically):
   ```metaxu
   struct Producer<T> {
       get: fn() -> T   # T only appears in output -> covariant
   }
   # Producer[Dog] <: Producer[Animal] when Dog <: Animal

   struct Consumer<T> {
       set: fn(T)      # T only appears in input -> contravariant
   }
   # Consumer[Animal] <: Consumer[Dog] when Dog <: Animal
   ```

3. **Structural Subtyping** for records:
   ```metaxu
   struct Point2D { x: Int, y: Int }
   struct Point3D { x: Int, y: Int, z: Int }
   # Point3D is a structural subtype of Point2D
   ```

4. **Width Subtyping** for variants:
   ```metaxu
   enum Shape {
       Circle(radius: Float)
       Rectangle(width: Float, height: Float)
   }
   enum ExtendedShape {
       Circle(radius: Float)
       Rectangle(width: Float, height: Float)
       Triangle(base: Float, height: Float)
   }
   # Shape is a width subtype of ExtendedShape
   ```

5. **Effect Subtyping**:
   ```metaxu
   effect Reader<T> {
       read() -> T    # Covariant in T
   }
   effect Writer<T> {
       write(T)      # Contravariant in T
   }
   # Reader[Dog] <: Reader[Animal] when Dog <: Animal
   # Writer[Animal] <: Writer[Dog] when Dog <: Animal
   ```

6. **Interface Implementation**:
   ```metaxu
   interface Printable {
       print() -> String
   }
   struct Document implements Printable {
       content: String
       print() -> String { content }
   }
   ```

Note: All variance annotations are inferred automatically by analyzing how type parameters are used. You don't need to specify them explicitly.

## Polymorphism

### Parametric Polymorphism
Generic types and functions:
```metaxu
fn id<T>(x: T) -> T { x }
type Option<T> { Some(T) | None }
```

### Bounded Polymorphism
Type constraints using bounds:
```metaxu
fn print<T: Display>(x: T) { ... }
fn sort<T: Ord>(list: List[T]) -> List[T] { ... }
```

### Effect Polymorphism
Functions polymorphic over effects:
```metaxu
fn transform<T, U, E>(
    f: fn(T) -> U performs E
) performs Reader[T], Writer[U], E {
    # Polymorphic over effect E
}
```

## Type Classes and Interfaces

Metaxu supports type classes for ad-hoc polymorphism:

```metaxu
interface Show<T> {
    fn show(x: T) -> String
}

impl Show[int] {
    fn show(x: int) -> String { x.to_string() }
}

fn display<T: Show>(x: T) {
    print(x.show())
}
```

## Linear Types and Ownership

Metaxu has a sophisticated ownership system that ensures memory safety and data race freedom. For a complete guide, see [Ownership and Borrowing](ownership_and_borrowing.md).

Basic ownership annotations:
```metaxu
fn eat(x: String) { ... }  # Takes ownership by value
fn consume(x: @owned String) { ... }  # Same as above with explicit annotation
fn borrow(x: @const String) { ... }   # Borrows immutably
fn modify(x: @mut String) { ... }     # Borrows mutably
```

## Effect Safety

The type system ensures effect safety:

1. **Effect Isolation**: Effects can't escape their handlers
2. **Effect Linearity**: Linear effects are used exactly once
3. **Effect Polymorphism**: Functions can be polymorphic over effects
4. **Effect Subtyping**: Effects follow variance rules

Example:
```metaxu
effect State<T> {
    get() -> T
    put(value: T) -> Unit
}

fn increment() performs State[int] {
    let x = perform State.get();
    perform State.put(x + 1)
}

// Safe handling
handle State[int] with {
    get() -> resume(0)
    put(x) -> resume(())
} in {
    increment()
}
