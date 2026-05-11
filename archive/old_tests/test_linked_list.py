import pytest
import subprocess
import os

def run_mx_compiler(source_file):
    """Helper function to run the compiler on a source file and return the result."""
    result = subprocess.run(['python', '-m', 'metaxu.metaxu', source_file], 
                          capture_output=True, 
                          text=True)
    return result

def test_linked_list_compilation():
    """Test that the linked list example compiles successfully."""
    source_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              'examples', 'linked_list.mx')
    result = run_mx_compiler(source_file)
    assert result.returncode == 0, f"Compilation failed with error:\n{result.stderr}"

def test_type_errors():
    """Test that type errors are caught properly."""
    # Create a temporary file with intentional type errors
    test_content = """
    struct Node<T> {
        data: T,
        @mut next: Option[Node[T]]
    }

    fn main() {
        # Type error: trying to create Node with wrong type
        let node = Node[Int] {
            data: "string",  # Should be Int
            next: None
        };
    }
    """
    with open('test_type_error.mx', 'w') as f:
        f.write(test_content)
    
    result = run_mx_compiler('test_type_error.mx')
    assert result.returncode != 0, "Expected compilation to fail due to type error"
    assert "type error" in result.stderr.lower()
    
    # Clean up
    os.remove('test_type_error.mx')

def test_borrow_checking():
    """Test that borrow checking rules are enforced."""
    # Create a temporary file with borrow checking violations
    test_content = """
    struct Node<T> {
        data: T,
        @mut next: Option[Node[T]]
    }

    fn main() {
        let @mut node = Node[Int] {
            data: 42,
            next: None
        };
        
        # Borrow checking error: multiple mutable references
        let @mut ref1 = @mut node;
        let @mut ref2 = @mut node;  # Should fail
    }
    """
    with open('test_borrow_check.mx', 'w') as f:
        f.write(test_content)
    
    result = run_mx_compiler('test_borrow_check.mx')
    assert result.returncode != 0, "Expected compilation to fail due to borrow checking violation"
    assert "borrow" in result.stderr.lower()
    
    # Clean up
    os.remove('test_borrow_check.mx')

def test_linked_list_operations():
    """Test that linked list operations type check correctly."""
    test_content = """
    struct Node<T> {
        data: T,
        @mut next: Option[Node[T]]
    }

    struct LinkedList<T> {
        @mut head: Option[Node[T]>
    }

    fn new_list<T>() -> LinkedList<T> {
        LinkedList[T] { head: None }
    }

    fn push_front<T>(list: @mut LinkedList<T>, value: T) {
        let new_node = Node[T] {
            data: value,
            next: list.head
        };
        list.head = Some(new_node);
    }

    fn main() {
        let @mut list = new_list();
        push_front(@mut list, 42);
        push_front(@mut list, 43);
        
        # This should type check correctly
        if let Some(head) = list.head {
            assert(head.data == 43);
        }
    }
    """
    with open('test_operations.mx', 'w') as f:
        f.write(test_content)
    
    result = run_mx_compiler('test_operations.mx')
    assert result.returncode == 0, f"Valid operations failed to compile:\n{result.stderr}"
    
    # Clean up
    os.remove('test_operations.mx')

def test_locality_local_only():
    """Test that local variables cannot escape their scope."""
    test_content = """
    struct Node<T> {
        data: T,
        @mut next: Option[Node[T]]
    }

    fn create_local_node() -> @const Node[Int] {
        # Create a node in local scope
        let node = Node[Int] {
            data: 42,
            next: None
        };
        return @const node  # Should fail - cannot return local variable
    }

    fn main() {
        let escaped = create_local_node();  # Should fail
    }
    """
    with open('test_locality_local.mx', 'w') as f:
        f.write(test_content)
    
    result = run_mx_compiler('test_locality_local.mx')
    assert result.returncode != 0, "Expected compilation to fail due to locality violation"
    assert "locality" in result.stderr.lower() or "scope" in result.stderr.lower()
    
    # Clean up
    os.remove('test_locality_local.mx')

def test_locality_heap_allocation():
    """Test that heap allocated objects can be returned from functions."""
    test_content = """
    struct Node<T> {
        data: T,
        @mut next: Option[Node[T]]
    }

    fn create_heap_node() -> Node[Int] {
        # This should work since Node is heap allocated
        Node[Int] {
            data: 42,
            next: None
        }
    }

    fn main() {
        let node = create_heap_node();  # Should succeed
        assert(node.data == 42);
    }
    """
    with open('test_locality_heap.mx', 'w') as f:
        f.write(test_content)
    
    result = run_mx_compiler('test_locality_heap.mx')
    assert result.returncode == 0, f"Valid heap allocation failed:\n{result.stderr}"
    
    # Clean up
    os.remove('test_locality_heap.mx')

def test_locality_reference_escape():
    """Test that references to local variables cannot escape their scope."""
    test_content = """
    struct Container {
        value: Int
    }

    fn get_local_ref() -> @const Container {
        let local = Container { value: 42 };
        return @const local  # Should fail - reference to local variable
    }

    fn main() {
        let escaped_ref = get_local_ref();  # Should fail
    }
    """
    with open('test_locality_ref.mx', 'w') as f:
        f.write(test_content)
    
    result = run_mx_compiler('test_locality_ref.mx')
    assert result.returncode != 0, "Expected compilation to fail due to escaping reference"
    assert "reference" in result.stderr.lower() or "scope" in result.stderr.lower()
    
    # Clean up
    os.remove('test_locality_ref.mx')

if __name__ == '__main__':
    pytest.main([__file__])
