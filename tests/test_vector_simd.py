import unittest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from lexer import Lexer
from parser import Parser
from type_checker import TypeChecker
import metaxu_ast as ast

class TestVectorSimd(unittest.TestCase):
    def setUp(self):
        self.lexer = Lexer()
        self.parser = Parser()
        self.type_checker = TypeChecker()

    def test_vector_literal_syntax(self):
        """Test parsing of vector literal syntax"""
        code = """
        let v1 = vector[float,4](1.0, 2.0, 3.0, 4.0);
        let v2 = vector[int,3](1, 2, 3);
        let v3 = vector[float,4](x * 2.0 for x in 0..4);
        """
        result = self.parser.parse(code)
        self.assertEqual(len(self.type_checker.errors), 0)
        
        # Check vector types
        v1_type = result.statements[0].value.type
        self.assertEqual(v1_type.base_type, "float")
        self.assertEqual(v1_type.size, 4)

    def test_vector_operations(self):
        """Test type checking of vector operations"""
        code = """
        let v1 = vector[float,4](1.0, 2.0, 3.0, 4.0);
        let v2 = vector[float,4](2.0, 3.0, 4.0, 5.0);
        
        # Element-wise operations
        let sum = v1 + v2;
        let prod = v1 * v2;
        let scaled = v1 * 2.0;
        
        # Reductions
        let total = v1.sum();
        let dot = v1.dot(v2);
        """
        result = self.parser.parse(code)
        self.type_checker.check(result)
        self.assertEqual(len(self.type_checker.errors), 0)

    def test_simd_effect_declaration(self):
        """Test parsing and type checking of SIMD effect declaration"""
        code = """
        effect SimdOp {
            try_vectorize<T,U,const N: int>(v: vector[T,N], f: fn(T) -> U) -> Option<vector[U,N]>;
            try_horizontal<T,const N: int>(v: vector[T,N], f: fn(T,T) -> T) -> Option<T>;
        }
        """
        result = self.parser.parse(code)
        self.type_checker.check(result)
        self.assertEqual(len(self.type_checker.errors), 0)
        
        # Check effect declaration
        effect_decl = result.statements[0]
        self.assertIsInstance(effect_decl, ast.EffectDeclaration)
        self.assertEqual(len(effect_decl.operations), 2)

    def test_simd_handler(self):
        """Test parsing and type checking of SIMD effect handler"""
        code = """
        fn with_simd<T>(f: fn() -> T performs SimdOp) -> T {
            handle SimdOp with {
                SimdOp.try_vectorize(v, f) => {
                    match typeof(T) {
                        float => {
                            unsafe {
                                let reg = SimdRegister.from_vector(v);
                                let result = match f {
                                    fn(x) -> x * x => SimdIntrinsic.mul_ps(reg, reg),
                                    _ => resume(None)
                                };
                                resume(Some(result.to_vector()))
                            }
                        },
                        _ => resume(None)
                    }
                }
                
                SimdOp.try_horizontal(v, f) => {
                    match typeof(T) {
                        float => {
                            unsafe {
                                let reg = SimdRegister.from_vector(v);
                                let result = match f {
                                    fn(a,b) -> a + b => SimdIntrinsic.horizontal_add_ps(reg),
                                    _ => resume(None)
                                };
                                resume(Some(result))
                            }
                        },
                        _ => resume(None)
                    }
                }
            } in {
                f()
            }
        }
        """
        result = self.parser.parse(code)
        self.type_checker.check(result)
        self.assertEqual(len(self.type_checker.errors), 0)

    def test_effectful_vector_operations(self):
        """Test type checking of vector operations with SIMD effects"""
        code = """
        let v1 = vector[float,4](1.0, 2.0, 3.0, 4.0);
        let v2 = vector[float,4](2.0, 3.0, 4.0, 5.0);
        
        with_simd(fn() -> float performs SimdOp {
            # These should use SIMD
            let sum = v1.reduce((a, b) -> a + b);
            let squares = v1.map(x -> x * x);
            let product = v1.zip(v2, (a, b) -> a * b);
            sum
        });
        
        # This should not use SIMD
        let strings = v1.map(x -> x.to_string());
        """
        result = self.parser.parse(code)
        self.type_checker.check(result)
        self.assertEqual(len(self.type_checker.errors), 0)

    def test_simd_safety(self):
        """Test safety checks for SIMD operations"""
        code = """
        fn unsafe_simd() {
            let v1 = vector[float,4](1.0, 2.0, 3.0, 4.0);
            
            with_simd(fn() -> float performs SimdOp {
                unsafe {
                    # Error: Cannot use SIMD intrinsics outside handler
                    let reg = SimdRegister.from_vector(v1);
                    SimdIntrinsic.mul_ps(reg, reg)
                }
            });
        }
        """
        result = self.parser.parse(code)
        self.type_checker.check(result)
        self.assertGreater(len(self.type_checker.errors), 0)  # Should have safety errors

if __name__ == '__main__':
    unittest.main()
