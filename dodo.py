"""
doit tasks for building and testing Metaxu.
Run with: doit
"""

import os
from pathlib import Path
from doit.tools import run_once

# Common variables
C_COMPILER = 'gcc'
C_FLAGS = ['-pthread']
INCLUDE_FLAGS = ['-I.']

# Directories
OUTPUT_DIR = 'outputs'

# Source files
EFFECTS_SRC = 'src/metaxu/runtimes/c/effects.c'
VALUES_SRC = 'src/metaxu/runtimes/c/values.c'
COMMON_SRCS = [EFFECTS_SRC, VALUES_SRC]

# Python test files
PYTHON_TESTS = [
    'tests/test_references_and_functions_parsing.py',
    'tests/test_unsafe_parsing.py',
    'tests/test_effect_safety.py',
    'tests/test_vector_simd.py',
    'tests/test_continuation_safety.py',
    'tests/test_extern_parsing.py',
    'tests/test_module_system_parsing.py',
    'tests/test_linearity.py',
    'tests/test_effectful_code.py'
]

def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def task_build_atomic_effects():
    """Build atomic effects test"""
    ensure_output_dir()
    target = os.path.join(OUTPUT_DIR, 'test_atomic_effects')
    sources = ['tests/test_atomic_effects.c'] + COMMON_SRCS
    
    return {
        'actions': [
            f'{C_COMPILER} -o {target} {" ".join(sources)} {" ".join(INCLUDE_FLAGS)} {" ".join(C_FLAGS)}'
        ],
        'file_dep': sources,
        'targets': [target],
        'clean': True,
    }

def task_build_effect_runtime():
    """Build effect runtime test"""
    ensure_output_dir()
    target = os.path.join(OUTPUT_DIR, 'test_effect_runtime')
    sources = ['tests/test_effect_runtime.c'] + COMMON_SRCS
    
    return {
        'actions': [
            f'{C_COMPILER} -o {target} {" ".join(sources)} {" ".join(INCLUDE_FLAGS)} {" ".join(C_FLAGS)}'
        ],
        'file_dep': sources,
        'targets': [target],
        'clean': True,
    }

def task_test_atomic_effects():
    """Run atomic effects test"""
    return {
        'actions': [os.path.join(OUTPUT_DIR, 'test_atomic_effects')],
        'task_dep': ['build_atomic_effects'],
        'verbosity': 2,
    }

def task_test_effect_runtime():
    """Run effect runtime test"""
    return {
        'actions': [os.path.join(OUTPUT_DIR, 'test_effect_runtime')],
        'task_dep': ['build_effect_runtime'],
        'verbosity': 2,
    }

def task_test_python():
    """Run Python tests"""
    def run_python_tests():
        import pytest
        return pytest.main(['-v'] + PYTHON_TESTS)

    return {
        'actions': [run_python_tests],
        'file_dep': PYTHON_TESTS,
        'verbosity': 2,
    }

def task_test():
    """Run all tests"""
    return {
        'actions': None,
        'task_dep': ['test_atomic_effects', 'test_effect_runtime', 'test_python'],
    }
