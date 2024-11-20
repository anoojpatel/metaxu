import unittest
from parser import Parser
from unsafe_ast import UnsafeBlock, PointerType, TypeCast, PointerDereference, AddressOf

class TestUnsafeParsing(unittest.TestCase):
    def setUp(self):
        self.parser = Parser()

    def test_unsafe_block(self):
        code = '''
        fn unsafe_copy<T>(dst: &mut T, src: &T) {
            unsafe {
                memcpy(
                    dst as *mut void,
                    src as *const void,
                    size_of::<T>()
                );
            }
        }
        '''
        result = self.parser.parse(code)
        func = result.body[0]
        unsafe_block = func.body[0]
        self.assertIsInstance(unsafe_block, UnsafeBlock)
        self.assertEqual(len(unsafe_block.body), 1)  # memcpy call

    def test_pointer_types_in_extern(self):
        code = '''
        extern "string.h" {
            fn memcpy(dst: *mut void, src: *const void, n: size_t) -> *mut void;
        }
        '''
        result = self.parser.parse(code)
        func = result.body[0].declarations[0]
        dst_type = func.params[0].type
        src_type = func.params[1].type
        self.assertIsInstance(dst_type, PointerType)
        self.assertTrue(dst_type.is_mut)
        self.assertIsInstance(src_type, PointerType)
        self.assertFalse(src_type.is_mut)

    # Happy path tests for properly used unsafe operations
    def test_safe_pointer_ops_in_unsafe(self):
        code = '''
        fn test() {
            let x = 42;
            unsafe {
                let ptr = &mut x;       # OK - address-of in unsafe
                *ptr = 43;              # OK - dereference in unsafe
                let raw = ptr as *mut i32;  # OK - pointer cast in unsafe
                let val = *raw;         # OK - raw pointer deref in unsafe
            }
        }
        '''
        result = self.parser.parse(code)
        func = result.body[0]
        unsafe_block = func.body[1]  # After let x = 42
        self.assertIsInstance(unsafe_block, UnsafeBlock)
        self.assertEqual(len(unsafe_block.body), 4)  # Four operations in unsafe block

    def test_safe_pointer_type_declaration_in_unsafe(self):
        code = '''
        fn test() {
            unsafe {
                let ptr: *mut i32;      # OK - pointer type in unsafe
                let const_ptr: *const i32;  # OK - const pointer in unsafe
                let void_ptr: *mut void;    # OK - void pointer in unsafe
            }
        }
        '''
        result = self.parser.parse(code)
        unsafe_block = result.body[0].body[0]
        self.assertIsInstance(unsafe_block, UnsafeBlock)
        self.assertEqual(len(unsafe_block.body), 3)  # Three pointer declarations

    def test_safe_pointer_arithmetic_in_unsafe(self):
        code = '''
        fn test() {
            let arr = [1, 2, 3, 4, 5];
            unsafe {
                let ptr = &arr[0] as *const i32;  # OK - array to pointer
                let ptr2 = ptr.offset(2);         # OK - pointer arithmetic
                let val = *ptr2;                  # OK - dereference
            }
        }
        '''
        result = self.parser.parse(code)
        unsafe_block = result.body[0].body[1]  # After array declaration
        self.assertIsInstance(unsafe_block, UnsafeBlock)
        self.assertEqual(len(unsafe_block.body), 3)  # Three pointer operations

    def test_safe_nested_unsafe_blocks(self):
        code = '''
        fn test() {
            let x = 42;
            unsafe {
                let ptr1 = &mut x;
                unsafe {  # OK - nested unsafe blocks
                    let ptr2 = ptr1 as *mut i32;
                    *ptr2 = 42;
                }
            }
        }
        '''
        result = self.parser.parse(code)
        outer_unsafe = result.body[0].body[1]
        self.assertIsInstance(outer_unsafe, UnsafeBlock)
        inner_unsafe = outer_unsafe.body[1]
        self.assertIsInstance(inner_unsafe, UnsafeBlock)

    # Error cases for unsafe operations outside unsafe blocks
    def test_pointer_types_require_unsafe(self):
        code = '''
        fn test() {
            let ptr: *mut i32;  # Should fail - pointer type outside unsafe
        }
        '''
        with self.assertRaises(SyntaxError) as cm:
            self.parser.parse(code)
        self.assertIn("Pointer types are only allowed in unsafe blocks", str(cm.exception))

    def test_type_cast_requires_unsafe(self):
        code = '''
        fn test() {
            let x = 42;
            let ptr = x as *const i32;  # Should fail - pointer cast outside unsafe
        }
        '''
        with self.assertRaises(SyntaxError) as cm:
            self.parser.parse(code)
        self.assertIn("Pointer type casts are only allowed in unsafe blocks", str(cm.exception))

    def test_pointer_ops_require_unsafe(self):
        code = '''
        fn test() {
            let x = 42;
            let ptr = &mut x;  # Should fail - address-of outside unsafe
            *ptr = 43;         # Should fail - dereference outside unsafe
        }
        '''
        with self.assertRaises(SyntaxError) as cm:
            self.parser.parse(code)
        self.assertIn("Taking address of value is only allowed in unsafe blocks", str(cm.exception))

    def test_safe_type_cast(self):
        code = '''
        fn test() {
            let x = 42;
            let y = x as f64;  # OK - numeric cast doesn't require unsafe
        }
        '''
        # Should parse without error
        self.parser.parse(code)

if __name__ == '__main__':
    unittest.main()
