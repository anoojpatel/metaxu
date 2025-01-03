# Ownership and Borrowing in Metaxu

Metaxu's type system includes a sophisticated ownership system that ensures memory safety and data race freedom without garbage collection. The system is built on three fundamental concepts: Modes, Locality, and Linearity.

## 1. Value Access Modes

Metaxu's type system has two independent dimensions that control how values can be accessed and where they can live:

### 1.1 Uniqueness Modes

These modes determine how variables can reference and access values:

#### Owned Mode
```metaxu
# Owned mode indicates a reference to a unique value
let @owned file = File.open("data.txt")  # file uniquely owns the File value
let @owned vec = Vec::new()  # vec uniquely owns the vector

# Once moved, the original variable cannot be used
let new_owner = file
# Error: file has been moved
# println(file)  
```

#### Mutable Mode
```metaxu
# Mutable mode represents an exclusively owned reference to a value
let @mutable x = &mut some_value
x.modify()  # Can modify the referenced value
let @mutable y = x  # Transfers the exclusive reference
# Error: x no longer has access
# x.modify()
```

#### Const Mode
```metaxu
# Const mode represents immutable shared references
let @const x = &some_value
let @const y = x  # Multiple const references are allowed
println(x)  # Can read the value
println(y)  # Can read through multiple references
# Error: Cannot modify through a const reference
# x.modify()
```

### 1.2 Locality Modes

These modes determine where values can live and how they're allocated:

#### Local Mode
```metaxu
# Local values are stack-allocated and bound to a specific region
fn process() {
    let @local x: String = "hello"  # x is local to this function's region
    let @local y: Vec<Int> = vec![1, 2, 3]  # Local array
    
    # Error: Cannot return local value outside its region
    # return x  
}
```

#### Global Mode
```metaxu
# Global values are heap-allocated and can be shared across regions
let @global x: String = "hello"  # Explicitly global
let @global data = Vec::new()    # Can be passed between regions

fn process(data: @global String) {
    # Can accept and return global values
    return data
}
```

Note: These two dimensions are orthogonal - any uniqueness mode (@owned, @mutable, @const) can be combined with either locality mode (@local, @global) to control both how a value is accessed and where it can live.

## 2. Locality and Regions

Metaxu uses a region system to track where values can live and how they can be moved:

### 2.1 Region Declaration
```metaxu
fn outer() {
    region r1 {
        let @local x: String = "hello"  # x lives in r1
        
        region r2 {
            let @local y: String = "world"  # y lives in r2
            # Error: Cannot move y out of r2
            # x = y
        }
    }
}
```

### 2.2 Region Escape Prevention
```metaxu
fn returns_local() -> String {
    let @local x: String = "hello"
    # Error: Cannot return local value
    # return x
    
    # Must convert to global or unique first
    return x.to_global()
}
```

### 2.3 Exclave Expressions

Exclave expressions provide a way to initialize pre-declared variables in a parent region using values from an inner region. This includes both nested block regions and function returns:

```metaxu
// Example 1: Block-level exclave
fn process_data() {
    region outer {
        let @local promoted_data: Vec<Int>;
        
        region inner {
            let @local data: Vec<Int> = vec![1, 2, 3];
            
            exclave {
                data  # Moves/copies data's value to promoted_data
            }
            
            process(data);  # Original data still accessible
        }
        
        use_data(promoted_data);  # promoted_data available here
    }
}

// Example 2: Function return exclave
fn create_buffer() -> @local Vec<Int> {
    let @local buffer = vec![1, 2, 3];
    
    // Return local buffer to caller's scope
    exclave {
        buffer  # Promotes buffer to caller's scope
    }
}

fn use_buffer() {
    let @local my_buffer = create_buffer();  # Receives the promoted buffer
    process(my_buffer);
}

// Example 3: Nested function calls with exclave
fn inner_computation() -> @local Int {
    let @local result = 42;
    exclave {
        result  # Promotes to caller's scope
    }
}

fn outer_computation() -> @local Int {
    let @local value = inner_computation();  # Receives from inner
    exclave {
        value  # Promotes further to its caller
    }
}
```

The exclave expression allows controlled initialization of variables in parent regions using values from inner regions. This works both for:
1. Block-level promotion: Initializing pre-declared variables in parent block scopes
2. Function returns: Promoting local variables to the caller's scope

The compiler ensures that receiving variables are properly declared in the target scope before they can be initialized through exclave expressions. For function returns, the compiler handles the necessary setup in the caller's scope automatically.

This pattern maps directly to C's scope rules, where each region corresponds to a C block scope, and function returns naturally promote values to the caller's stack frame.

## 3. Linearity and Separation

Metaxu enforces linear use of resources and separation of mutable state:

### 3.1 Borrowing Rules
```metaxu
fn main() {
    let @local mut x: String = "hello"
    
    # Shared borrows
    let @global r1: &String = &x
    let @global r2: &String = &x  # Multiple shared borrows OK
    
    # Error: Cannot mutably borrow while shared borrows exist
    # let @mut m1: &String = &mut x
    
    # Mutable borrow
    drop(r1)
    drop(r2)
    let @mut m1: &String = &mut x
    # Error: Cannot have multiple mutable borrows
    # let @mut m2: String = &mut x
}
```

### 3.2 Move Semantics
```metaxu
struct Buffer {
    @unique data: Vec<u8>
}

fn process_buffer(buf: Buffer) {
    # buf is moved here and consumed
}

fn main() {
    let @unique buf: Buffer = Buffer { data: vec![1, 2, 3] }
    process_buffer(buf)
    # Error: buf has been moved
    # process_buffer(buf)
}
```

### 3.3 Separation of Mutable State
```metaxu
struct Counter {
    value: Int
}

fn increment(c: @mut Counter) {
    c.value += 1
}

fn main() {
    let @local @mut c: Counter = Counter { value: 0 }
    
    # Error: Cannot have overlapping mutable borrows
    let r1 = &mut c
    let r2 = &mut c
    increment(r1)
    increment(r2)
}
```

## 4. Advanced Features

### 4.1 Mode Polymorphism
```metaxu
# Function that works with any mode
fn process<M: Mode>(x: String M) {
    # ...
}

# Can be called with any mode
process("hello" global)
process("world" local)
process("unique" unique)
```

### 4.2 Mode Constraints
```metaxu
# Require specific mode capabilities
trait Copyable: global {
    fn copy(self) -> Self;
}

# Only global types can implement Copyable
impl Copyable for Int {
    fn copy(self) -> Int { self }
}
```

### 4.3 Mode Conversion
```metaxu
fn convert_modes() {
    let @local x: String = "hello"
    
    # Convert to global (if type supports it)
    let @global y: String = x.to_global()
    
    # Convert to unique (consumes original)
    let @unique z: String = y.to_unique()
}
```

## 5. Best Practices

1. **Default to Local Mode**
   - Use local mode by default for better safety
   - Only use global mode for truly shared data
   - Reserve unique mode for resources that need cleanup

2. **Region Minimization**
   - Keep regions as small as possible
   - Use exclave expressions for temporary escapes
   - Explicitly mark region boundaries

3. **Borrowing Guidelines**
   - Prefer immutable borrows when possible
   - Keep mutable borrows short-lived
   - Use scoped blocks to limit borrow lifetimes

4. **Resource Management**
   - Use RAII patterns with unique mode
   - Implement Drop for cleanup
   - Consider using smart pointers for complex ownership

## 6. Common Patterns

### 6.1 RAII Resource Management
```metaxu
struct File {
    @unique handle: FileHandle
}

impl Drop for File {
    fn drop(@mut self) {
        # Automatically close file when dropped
        self.handle.close()
    }
}
```

### 6.2 Builder Pattern with Mode Transition
```metaxu
struct Builder {
    @local data: Vec<String>
}

impl Builder {
    fn build(self) -> Product unique {
        # Convert local data to unique product
        Product::new(self.data.to_unique())
    }
}
```

### 6.3 Safe Concurrent Access
```metaxu
struct SharedState {
    @global data: Vec<Int>
}

fn concurrent_process(state: &SharedState) {
    # Safe concurrent read-only access
}
```

## 7. Closures and Functions

### 7.1 Closure Capture Rules
```metaxu
fn example() {
    let @local x: String = "hello"
    let @global y: Int = 42
    
    # Closure capturing local value - Error
    let bad_closure = fn() {
        println(x)  # Error: Cannot capture local value
    }
    
    # Closure capturing global value - OK
    let good_closure = fn() {
        println(y)  # OK: Global values can be captured
    }
    
    # Explicit borrowing in closure
    let borrow_closure = fn() {
        let r: &String = &x  # Must explicitly borrow
        println(r)
    }
}
```

### 7.2 Once Functions
```metaxu
# Once functions can only be called once to prevent aliasing of mutable borrows
fn get_mut_ref() -> @mut Counter @ once {
    return &mut GLOBAL_COUNTER
}

fn main() {
    let r1 = get_mut_ref()  # First call OK
    # let r2 = get_mut_ref()  # Error: Cannot call once function again
    
    # Must drop first reference before calling again
    drop(r1)
    let r3 = get_mut_ref()  # OK after dropping r1
}
```

### 7.3 Non-reentrant Functions
```metaxu
# Prevent reentrancy in recursive functions
nonreentrant fn recursive(n: Int) {
    if n > 0 {
        # recursive(n)  # Error: Cannot reenter nonreentrant function
        recursive(n - 1)  # OK: Different argument
    }
}

# Also applies to recursive closures
let @mut rec: fn\(Int) = &mut fn(n: Int) {
    if n > 0 {
        # rec(n)  # Error: Would create circular reference
        rec(n - 1)  # OK: Different argument
    }
}
```

### 7.4 Effect Safety with Closures
```metaxu
effect Logger {
    log(msg: String)
}

fn unsafe_example() {
    let @local local_buf: String = "hello"
    
    # Error: Cannot capture local value in effect operation
    perform Logger.log(local_buf)
    
    # Error: Cannot capture local in closure used with effect
    let closure = fn() {
        perform Logger.log(local_buf)
    }
}

fn safe_example() {
    let @global global_buf: String = "sharable"
    
    # OK: Effects can use global data
    perform Logger.log(global_buf)
    
    # OK: Closure captures global value
    let closure = fn() {
        perform Logger.log(global_buf)
    }
}
```

### 7.5 Closure Ownership Transfer
```metaxu
fn make_counter() -> fn() -> Int {
    let @unique count: Cell<Int> = Cell::new(0)
    
    # Move ownership of count into closure
    move fn() -> Int {
        let current = count.get()
        count.set(current + 1)
        current
    }
}

fn main() {
    let counter = make_counter()
    assert_eq!(counter(), 0)
    assert_eq!(counter(), 1)
}
```

### 7.6 Safe Recursive Closures
```metaxu
# Safe recursion with explicit ownership transfer
fn make_factorial() -> fn(Int) -> Int {
    let @local factorial: @mut fn\(Int) -> Int = &mut fn(n: Int) -> Int {
        if n <= 1 {
            return1
        } else {
            return n * factorial(n - 1)  # OK: Ownership properly tracked
        }
    }
    exclave factorial  # Move ownership to caller
}
```

## 8. Effect Safety Rules

### 8.1 Effect Locality Constraints
```metaxu
effect FileSystem {
    read_file(path: String) -> String
    write_file(path: String, content: String)
}

fn unsafe_example() {
    let @local local_path: String = get_path()
    
    # Error: Cannot use local value in effect
    perform FileSystem.read_file(local_path)
}

fn safe_example() {
    let @global global_path: String = "/tmp/file.txt"
    
    # OK: Using global value in effect
    perform FileSystem.read_file(global_path)
}
```

### 8.2 Effect Handler Safety
```metaxu
effect State<T> {
    get() -> T
    put(value: T)
}

# Handler must ensure state doesn't escape
fn run_with_state<T, R>(initial: T, f: fn() -> R) -> R {
    let @unique state: Cell<T> = Cell::new(initial)
    
    with {
        State.get() -> T {
            state.get()  # Safe: state is unique
        }
        State.put(value: T) {
            state.set(value)
        }
    } in {
        f()  # Run computation with handler
    }
}  # state is dropped here
```

### 8.3 Effect Capture Rules
```metaxu
effect Console {
    print(msg: String)
}

fn example() {
    let @local local_buf: String = "hello"
    
    # Error: Cannot capture local in effect operation
    perform Console.print(local_buf)
    
    # Error: Cannot capture local in closure used with effect
    let closure = fn() {
        perform Console.print(local_buf)
    }
    
    # OK: Convert to global first
    let @global global_buf: String = local_buf.to_global()
    perform Console.print(global_buf)
}
```

## 9. Local References to Global Values

### 9.1 Local Collections of Global References
```metaxu
struct GlobalNode {
    @global data: Int,
    @global next: Option<Box<GlobalNode>>
}

fn process_nodes() {
    # Local list containing references to global nodes
    let @local nodes: Vec<&GlobalNode> = vec![]
    
    # OK: Add references to global nodes
    nodes.push(&GLOBAL_NODE1)
    nodes.push(&GLOBAL_NODE2)
    
    # OK: Local list can be dropped, globals remain
}
```

### 9.2 Local Caches of Global Data
```metaxu
struct Cache {
    # Local map containing references to global data
    @local entries: HashMap<String, &GlobalData>
}

impl Cache {
    fn new() -> Self {
        Cache { entries: HashMap::new() }
    }
    
    fn get_or_load(@mut self, key: String) -> &GlobalData {
        if let Some(entry) = self.entries.get(&key) {
            entry
        } else {
            let data = &GLOBAL_DATA_STORE.get(&key)
            self.entries.insert(key, data)
            data
        }
    }
}
```

### 9.3 Local Indices into Global Arrays
```metaxu
struct LocalIndex {
    # Local vector containing indices into global array
    @local indices: Vec<usize>,
    # Reference to global array for safety checking
    @global array: &GlobalArray
}

impl LocalIndex {
    fn new(array: &GlobalArray) -> Self {
        LocalIndex {
            indices: vec![],
            array
        }
    }
    
    fn add(@mut self, index: usize) -> Result<(), Error> {
        if index < self.array.len() {
            self.indices.push(index)
            Ok(())
        } else {
            Err(Error::OutOfBounds)
        }
    }
    
    fn get(&self, i: usize) -> Option<&GlobalData> {
        self.indices.get(i).map(|&idx| &self.array[idx])
    }
}
```

### 9.4 Local Views of Global Graphs
```metaxu
struct GlobalGraph {
    @global nodes: Vec<Node>,
    @global edges: Vec<Edge>
}

struct LocalView {
    # Local sets containing indices into global graph
    @local visible_nodes: HashSet<NodeId>,
    @local visible_edges: HashSet<EdgeId>,
    # Reference to global graph for safety
    @global graph: &GlobalGraph
}

impl LocalView {
    fn new(@const graph:  GlobalGraph) -> Self {
        LocalView {
            visible_nodes: HashSet::new(),
            visible_edges: HashSet::new(),
            graph
        }
    }
    
    fn add_node(@mut self, id: NodeId) -> bool {
        if id < self.graph.nodes.len() {
            self.visible_nodes.insert(id)
        } else {
            false
        }
    }
    
    # Get all visible nodes
    fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.visible_nodes
            .iter()
            .map(|&id| &self.graph.nodes[id])
    }
}
```

### 9.5 Local Workspace with Global Resources
```metaxu
struct GlobalResource {
    @global data: Vec[u8]
}

struct LocalWorkspace {
    # Local vector of references to global resources
    @local resources: Vec[&GlobalResource],
    # Local temporary data
    @local temp_data: Vec[u8>
}

impl LocalWorkspace {
    fn new() -> Self {
        LocalWorkspace {
            resources: vec![],
            temp_data: vec![]
        }
    }
    
    fn add_resource(@mut self, resource: &GlobalResource) {
        self.resources.push(resource)
    }
    
    fn process(@mut self) {
        for resource in &self.resources {
            # Process global data locally
            for byte in &resource.data {
                self.temp_data.push(*byte)
            }
        }
    }
}
```

## 10. Deep Ownership and Nested Modes

### 10.1 Nested Local Values
```metaxu
struct LocalTree {
    # Both the value and children are local
    @local value: String,
    @local children: Vec[Box[LocalTree]]
}

fn process_local_tree() {
    let tree = LocalTree {
        value: "root".to_local(),
        children: vec![
            Box::new(LocalTree {
                value: "child1".to_local(),
                children: vec![]
            }),
            Box::new(LocalTree {
                value: "child2".to_local(),
                children: vec![]
            })
        ]
    }
    # When tree is dropped, all nested local values are dropped
}
```

### 10.2 Global Container with References
```metaxu
struct GlobalContainer<T> {
    @global @mut elements: Vec[T]
}

impl<T> GlobalContainer[T] {
    fn new() -> Self {
        GlobalContainer { elements: vec![] }
    }
    
    # Error: Cannot store reference to local value
    fn bad_add(@mut self, @local value: T) {
        self.elements.push(value)  # Error: Reference would outlive local value
    }
    
    # OK: Can store reference to global value
    fn good_add(@mut self, @global value: T) {
        self.elements.push(value)  # OK: Global reference lives long enough
    }
}

# Example usage
fn example() {
    let @global container = GlobalContainer::new();
    
    let @local local_val: String = "local";
    container.bad_add(local_val);  # Error: Local reference would escape
    
    let @global global_val: String = "global";
    container.good_add(global_val);  # OK: Global reference is safe
}
```

### 10.3 Mixed Mode Data Structures
```metaxu
struct MixedTree<T> {
    # Global node with local metadata
    @global value: T,
    @local metadata: HashMap<String, String>,
    # References to other global nodes
    @local @mut children: Vec<MixedTree<T>>
}

impl<T> MixedTree[T] {
    fn new(@global value: T) -> Self { # Since `metadata` & `children` is local, it will exist only where `new()` is called
        MixedTree {
            value,
            metadata: HashMap::new(),
            children: vec![]
        }
    }
    
    fn add_child(@mut self, @global @const child: MixedTree<T>) {
        self.children.push(child)  # OK: Adding reference to global
    }
    
    fn add_metadata(@mut self, key: String, value: String) {
        self.metadata.insert(key, value)  # OK: Local modification
    }
}
```

### 10.4 Layered Ownership
```metaxu
struct Layer<T> {
    # Each layer has its own mode
    value: T,
    parent: Option<Box<Layer<T>>>
}

fn create_layers() {
    # Global root with local layers
    let @global root = Layer {
        value: "root".to_global(),
        parent: None
    };
    
    let @local layer1 = Layer {
        value: "layer1".to_local(),
        parent: Some(Box::new(root))
    };
    
    let @local layer2 = Layer {
        value: "layer2".to_local(),
        parent: Some(Box::new(layer1))
    };
}
```

### 10.5 Deep Copy and Mode Conversion
```metaxu
trait DeepCopy {
    fn deep_copy(@const self) -> Self;
    fn to_global(@const self) -> Self @global;
    fn to_local(@const self) -> Self @local;
}

struct DeepStruct {
    @local value: String,
    @local nested: Vec[Box[DeepStruct]]
}

impl DeepCopy for DeepStruct {
    fn deep_copy(@const self) -> Self {
        DeepStruct {
            value: self.value.clone(),
            nested: self.nested.iter().map(fn\(child) -> Box[DeepStruct] {
                Box::new(child.deep_copy())
            }).collect()
        }
    }
    
    fn to_global(@const self) -> Self global {
        # Recursively convert all nested values to global
        DeepStruct {
            value: self.value.to_global(),
            nested: self.nested.iter().map(fn\(child) -> Box[DeepStruct] {
                Box::new(child.to_global())
            }).collect()
        }
    }
    
    fn to_local(@const self) -> Self @local {
        # Recursively convert all nested values to local
        DeepStruct {
            value: self.value.to_local(),
            nested: self.nested.iter().map(fn\(child) -> Box[DeepStruct] {
                Box::new(child.to_local())
            }).collect()
        }
    }
}
```

Key points about deep ownership:
1. Local values can contain other local values (nested ownership)
2. Global containers cannot store local values (would violate lifetime)
3. Local containers can store references to globals (safe reference)
4. Mode conversion must be deep (recursive through structure)
5. Each layer in a structure can have its own mode

These patterns ensure that:
1. Ownership semantics are preserved at all depths
2. Local values cannot escape their scope through nesting
3. References to globals remain valid
4. Mode conversion is consistent throughout data structures

This document provides a comprehensive overview of Metaxu's ownership system, demonstrating how modes, locality, and linearity work together to ensure memory safety and prevent data races while maintaining expressiveness.
