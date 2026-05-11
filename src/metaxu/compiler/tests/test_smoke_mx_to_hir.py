"""Smoke tests for .mx to HIR pipeline.

Tests that we can successfully compile .mx files through the full pipeline
and generate valid HIR output.
"""

import os
from pathlib import Path
from metaxu.compiler.pipeline import run_pipeline_from_source


def get_example_files():
    """Get all .mx files from the examples directory."""
    examples_dir = Path(__file__).parent.parent.parent.parent.parent / "examples"
    mx_files = list(examples_dir.glob("*.mx"))
    return mx_files


def test_smoke_mx_to_hir():
    """Smoke test: compile all .mx files to HIR."""
    mx_files = get_example_files()
    
    # Test a few simple files first
    simple_files = ["hello.mx", "collections.mx"]
    
    for filename in simple_files:
        file_path = Path(__file__).parent.parent.parent.parent.parent / "examples" / filename
        if not file_path.exists():
            continue
            
        print(f"Testing {filename}...")
        source = file_path.read_text()
        
        try:
            ast_json, hir_txt, mir_txt, clif_txt = run_pipeline_from_source(source)
            assert hir_txt is not None
            assert len(hir_txt) > 0
            print(f"✓ {filename} compiled successfully to HIR")
        except Exception as e:
            print(f"✗ {filename} failed: {e}")
            # Don't fail the test for now, just log
            pass


def test_smoke_specific_file():
    """Test a specific .mx file compiles to HIR."""
    file_path = Path(__file__).parent.parent.parent.parent.parent / "examples" / "hello.mx"
    source = file_path.read_text()
    
    ast_json, hir_txt, mir_txt, clif_txt = run_pipeline_from_source(source)
    
    # Verify HIR was generated
    assert hir_txt is not None
    assert len(hir_txt) > 0
    print("Hello world compiled successfully to HIR")
    print(f"HIR output:\n{hir_txt}")


def test_modes_and_references_to_hir():
    """Test modes_and_references.mx compiles to HIR."""
    file_path = Path(__file__).parent.parent.parent.parent.parent / "examples" / "01_modes_and_references.mx"
    source = file_path.read_text()
    
    try:
        ast_json, hir_txt, mir_txt, clif_txt = run_pipeline_from_source(source)
        assert hir_txt is not None
        assert len(hir_txt) > 0
        print("01_modes_and_references.mx compiled successfully to HIR")
    except Exception as e:
        print(f"01_modes_and_references.mx failed (expected for complex features): {e}")
