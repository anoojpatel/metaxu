"""End-to-end tests for trait method dispatch: parsed source -> desugar ->
HIR -> MIR -> interpreter.

v1 architecture (documented in desugar.TraitImplDesugarPass):
- `implement Trait for Type { ... }` blocks are desugared into top-level
  functions named `__impl$Trait$Type$method` with `self` as first parameter.
- `recv.method(args)` lowers to a `__trait$method` call whose first operand
  is the receiver; the MIR interpreter dispatches on the receiver's RUNTIME
  type name (MxStruct.name / MxVariant.enum_name / scalar names), falling
  back to builtins (to_string/len) and then plain functions.
- `Type.method(args)` on an impl target type lowers to a `__static$Type$method`
  call resolved by the (type, method) pair.
"""

import pytest

from metaxu.compiler.pipeline import build_context_from_source, run_pipeline_ctx
from metaxu.compiler.hir import HIRBuilder
from metaxu.compiler.lower_hir_to_mir import lower_hir_to_mir
from metaxu.compiler.mir_interp import InterpError, MirInterpreter


def run_main(source: str, strict: bool = True):
    """Compile parsed source through the full pipeline and run main()."""
    ctx = build_context_from_source(source)
    if strict:
        run_pipeline_ctx(ctx)  # raises TypeCheckError/BorrowCheckError
    hir_funcs = HIRBuilder(ctx.tables, id_map=ctx.id_map).build(ctx.frozen_root)
    mir_funcs = lower_hir_to_mir(hir_funcs)
    interp = MirInterpreter()
    interp.load(mir_funcs)
    return interp.call("main", [])


def test_two_types_dispatch_to_their_own_impls():
    """Dog and Cat both implement Speak; each call picks its own method."""
    result = run_main('''
        trait Speak {
            fn speak(self) -> string
        }

        struct Dog { name: string }
        struct Cat { name: string }

        implement Speak for Dog {
            fn speak(self) -> string { "woof" }
        }

        implement Speak for Cat {
            fn speak(self) -> string { "meow" }
        }

        fn main() -> string {
            let d = Dog { name: "rex" };
            let c = Cat { name: "tom" };
            d.speak() + " " + c.speak()
        }
    ''')
    assert result == "woof meow"


def test_impl_method_can_read_self_fields():
    result = run_main('''
        trait Named {
            fn get_name(self) -> string
        }
        struct Dog { name: string }
        implement Named for Dog {
            fn get_name(self) -> string { self.name }
        }
        fn main() -> string {
            let d = Dog { name: "rex" };
            d.get_name()
        }
    ''')
    assert result == "rex"


def test_trait_method_with_extra_argument():
    result = run_main('''
        trait Greet {
            fn greet(self, other: string) -> string
        }
        struct Dog { name: string }
        implement Greet for Dog {
            fn greet(self, other: string) -> string {
                self.name + " greets " + other
            }
        }
        fn main() -> string {
            let d = Dog { name: "rex" };
            d.greet("tom")
        }
    ''')
    assert result == "rex greets tom"


def test_method_call_on_struct_literal_receiver():
    """MethodCall on a computed receiver (not a named variable)."""
    result = run_main('''
        trait Speak {
            fn speak(self) -> string
        }
        struct Dog { name: string }
        implement Speak for Dog {
            fn speak(self) -> string { "woof" }
        }
        fn main() -> string {
            Dog { name: "rex" }.speak()
        }
    ''')
    assert result == "woof"


def test_user_to_string_impl_overrides_builtin_for_its_type():
    """A user impl of to_string wins for its receiver type; the builtin still
    serves receivers of other types."""
    result = run_main('''
        trait Show {
            fn to_string(self) -> string
        }
        struct Point { x: Int, y: Int }
        implement Show for Point {
            fn to_string(self) -> string { "custom!" }
        }
        fn main() -> string {
            let p = Point { x: 1, y: 2 };
            let n = 42;
            p.to_string() + " " + n.to_string()
        }
    ''')
    assert result == "custom! 42"


def test_unimplemented_trait_method_for_type_raises_clear_error():
    """speak is implemented for Cat but not Dog: calling it on a Dog names
    both the method and the receiver type in the error."""
    with pytest.raises(InterpError, match=r"'speak'.*not implemented.*'Dog'"):
        run_main('''
            trait Speak {
                fn speak(self) -> string
            }
            struct Dog { name: string }
            struct Cat { name: string }
            implement Speak for Cat {
                fn speak(self) -> string { "meow" }
            }
            fn main() -> string {
                let d = Dog { name: "rex" };
                d.speak()
            }
        ''')


def test_trait_method_with_no_impls_raises_clear_error():
    """A declared trait method with no implement block anywhere."""
    with pytest.raises(InterpError, match=r"'speak'.*no implementation"):
        run_main('''
            trait Speak {
                fn speak(self) -> string
            }
            struct Dog { name: string }
            fn main() -> string {
                let d = Dog { name: "rex" };
                d.speak()
            }
        ''')


def test_static_method_call_on_impl_type():
    """A method that neither declares nor uses self is a static method,
    callable as Type.method(args)."""
    result = run_main('''
        struct Counter { count: Int }
        implement Counter {
            fn make(start: Int) -> Counter {
                Counter { count: start }
            }
            fn value(self) -> Int { self.count }
        }
        fn main() -> Int {
            let c = Counter.make(7);
            c.value()
        }
    ''')
    assert result == 7


def test_inherent_impl_without_trait_dispatches():
    """implement Type { ... } (no trait) also dispatches on the receiver."""
    result = run_main('''
        struct Point { x: Int, y: Int }
        implement Point {
            fn total(self) -> Int { self.x + self.y }
        }
        fn main() -> Int {
            let p = Point { x: 3, y: 4 };
            p.total()
        }
    ''')
    assert result == 7


def test_builtin_len_still_works_without_user_impl():
    result = run_main('''
        fn main() -> Int {
            let s = "hello";
            s.len()
        }
    ''')
    assert result == 5
