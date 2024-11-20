import unittest
from parser import Parser
from extern_ast import ExternBlock, ExternFunctionDeclaration, ExternTypeDeclaration

class TestExternParsing(unittest.TestCase):
    def setUp(self):
        self.parser = Parser()

    def test_empty_extern_block(self):
        code = '''extern "stdio.h" {}'''
        result = self.parser.parse(code)
        self.assertIsInstance(result.body[0], ExternBlock)
        self.assertEqual(result.body[0].header_path, "stdio.h")
        self.assertEqual(len(result.body[0].declarations), 0)

    def test_extern_block_with_function(self):
        code = '''
        extern "stdio.h" {
            @c_function(borrows_refs=true)
            fn printf(fmt: *const char, ...) -> i32;
        }
        '''
        result = self.parser.parse(code)
        self.assertIsInstance(result.body[0], ExternBlock)
        self.assertEqual(result.body[0].header_path, "stdio.h")
        self.assertEqual(len(result.body[0].declarations), 1)
        self.assertIsInstance(result.body[0].declarations[0], ExternFunctionDeclaration)

    def test_extern_block_with_types(self):
        code = '''
        extern "sys/stat.h" {
            type FILE;  # Opaque type
            type stat = struct {};  # Anonymous struct type
            type timespec = struct timespec;  # Named struct type
        }
        '''
        result = self.parser.parse(code)
        self.assertIsInstance(result.body[0], ExternBlock)
        self.assertEqual(len(result.body[0].declarations), 3)
        
        # Check opaque type
        file_type = result.body[0].declarations[0]
        self.assertIsInstance(file_type, ExternTypeDeclaration)
        self.assertTrue(file_type.is_opaque)
        self.assertEqual(file_type.name, "FILE")
        self.assertIsNone(file_type.struct_name)
        
        # Check anonymous struct type
        stat_type = result.body[0].declarations[1]
        self.assertIsInstance(stat_type, ExternTypeDeclaration)
        self.assertFalse(stat_type.is_opaque)
        self.assertEqual(stat_type.name, "stat")
        self.assertIsNone(stat_type.struct_name)
        
        # Check named struct type
        timespec_type = result.body[0].declarations[2]
        self.assertIsInstance(timespec_type, ExternTypeDeclaration)
        self.assertFalse(timespec_type.is_opaque)
        self.assertEqual(timespec_type.name, "timespec")
        self.assertEqual(timespec_type.struct_name, "timespec")

    def test_complex_extern_block(self):
        code = '''
        extern "openssl/ssl.h" {
            // Opaque types for handle-based API
            type SSL_CTX;
            type SSL;
            
            // Function that uses opaque types
            @c_function(borrows_refs=true)
            fn SSL_new(ctx: *SSL_CTX) -> *SSL;
            
            // Anonymous struct for platform-specific types
            type ssl_method = struct {};
            
            // Named struct with explicit reference
            type CRYPTO_dynlock = struct CRYPTO_dynlock_value;
        }
        '''
        result = self.parser.parse(code)
        block = result.body[0]
        self.assertIsInstance(block, ExternBlock)
        self.assertEqual(block.header_path, "openssl/ssl.h")
        self.assertEqual(len(block.declarations), 5)
        
        # Check SSL_CTX opaque type
        ssl_ctx = block.declarations[0]
        self.assertIsInstance(ssl_ctx, ExternTypeDeclaration)
        self.assertTrue(ssl_ctx.is_opaque)
        self.assertEqual(ssl_ctx.name, "SSL_CTX")
        
        # Check SSL opaque type
        ssl = block.declarations[1]
        self.assertTrue(ssl.is_opaque)
        self.assertEqual(ssl.name, "SSL")
        
        # Check SSL_new function
        ssl_new = block.declarations[2]
        self.assertIsInstance(ssl_new, ExternFunctionDeclaration)
        self.assertEqual(ssl_new.name, "SSL_new")
        
        # Check ssl_method anonymous struct
        method = block.declarations[3]
        self.assertFalse(method.is_opaque)
        self.assertEqual(method.name, "ssl_method")
        self.assertIsNone(method.struct_name)
        
        # Check CRYPTO_dynlock named struct
        dynlock = block.declarations[4]
        self.assertFalse(dynlock.is_opaque)
        self.assertEqual(dynlock.name, "CRYPTO_dynlock")
        self.assertEqual(dynlock.struct_name, "CRYPTO_dynlock_value")

if __name__ == '__main__':
    unittest.main()
