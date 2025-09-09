"""
Deterministic execution test module that runs core application functions multiple times 
with identical inputs to verify consistent outputs.
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Callable, Optional, Tuple
import importlib.util
import inspect

import pytest


class DeterministicExecutionTest:
    """Test suite for verifying deterministic execution of core functions."""
    
    # Number of times to run each function to check for consistency
    CONSISTENCY_RUNS = 3
    
    @pytest.fixture
    def core_modules(self) -> List[Any]:
        """Discover and import core modules for testing."""
        project_root = Path(__file__).parent.parent
        core_modules = []
        
        # Core module directories to scan
        core_directories = [
            "egw_query_expansion/core",
            "retrieval_engine", 
            "canonical_flow",
            "analysis_nlp",
            "src"
        ]
        
        for core_dir in core_directories:
            dir_path = project_root / core_dir
            if dir_path.exists():
                # Import Python modules from this directory
                for py_file in dir_path.rglob("*.py"):
                    if (py_file.name != "__init__.py" and 
                        not py_file.name.startswith("test_")):
                        try:
                            module_name = py_file.stem
                            spec = importlib.util.spec_from_file_location(module_name, py_file)
                            if spec and spec.loader:
                                module = importlib.util.module_from_spec(spec)
                                sys.modules[module_name] = module
                                spec.loader.exec_module(module)
                                core_modules.append(module)
                        except Exception as e:
                            # Skip modules that can't be imported
                            print(f"Warning: Could not import {py_file}: {e}")
        
        return core_modules
    
    def _get_testable_functions(self, module: Any) -> List[Callable]:
        """Extract testable functions from a module."""
        functions = []
        
        for name in dir(module):
            obj = getattr(module, name)
            
            # Only test public functions that are callable
            if (callable(obj) and 
                not name.startswith('_') and 
                inspect.isfunction(obj)):
                
                # Skip functions that require complex setup or are clearly not deterministic
                skip_patterns = [
                    'random', 'uuid', 'time', 'now', 'today',
                    'log', 'print', 'write', 'save', 'load',
                    'fetch', 'download', 'upload', 'request',
                    'connect', 'disconnect', 'close', 'open'
                ]
                
                if not any(pattern in name.lower() for pattern in skip_patterns):
                    functions.append(obj)
        
        return functions
    
    def _generate_test_inputs(self, func: Callable) -> List[Tuple[tuple, dict]]:
        """Generate test inputs for a function based on its signature."""
        try:
            sig = inspect.signature(func)
            test_cases = []
            
            # Simple test cases with basic types
            basic_test_cases = [
                # Empty case
                ((), {}),
                
                # Simple values for different parameter types
                (("test_string",), {}),
                ((42,), {}),
                (([1, 2, 3],), {}),
                (({"key": "value"},), {}),
            ]
            
            # Try to match parameters with appropriate test values
            params = list(sig.parameters.values())
            if not params:
                test_cases.append(((), {}))
            else:
                # Generate kwargs based on parameter names and types
                kwargs_case = {}
                for param in params:
                    if param.default != inspect.Parameter.empty:
                        # Use default value
                        continue
                    elif param.annotation in [str, 'str']:
                        kwargs_case[param.name] = "test_input"
                    elif param.annotation in [int, 'int']:
                        kwargs_case[param.name] = 42
                    elif param.annotation in [float, 'float']:
                        kwargs_case[param.name] = 3.14
                    elif param.annotation in [list, 'list', List]:
                        kwargs_case[param.name] = [1, 2, 3]
                    elif param.annotation in [dict, 'dict', Dict]:
                        kwargs_case[param.name] = {"key": "value"}
                    elif param.annotation in [bool, 'bool']:
                        kwargs_case[param.name] = True
                    else:
                        # Fallback to string for unknown types
                        kwargs_case[param.name] = "test_value"
                
                if kwargs_case:
                    test_cases.append(((), kwargs_case))
            
            # Add basic test cases if function can accept them
            for args, kwargs in basic_test_cases:
                try:
                    if len(args) <= len([p for p in params if p.default == inspect.Parameter.empty]):
                        test_cases.append((args, kwargs))
                except:
                    pass
            
            return test_cases[:5]  # Limit to 5 test cases per function
            
        except Exception:
            # Fallback to simple test case
            return [((), {})]
    
    def _serialize_result(self, result: Any) -> str:
        """Serialize a result for comparison purposes."""
        try:
            # Handle common types
            if result is None:
                return "None"
            elif isinstance(result, (str, int, float, bool)):
                return str(result)
            elif isinstance(result, (list, tuple)):
                return json.dumps(sorted([self._serialize_result(item) for item in result]))
            elif isinstance(result, dict):
                return json.dumps({k: self._serialize_result(v) for k, v in sorted(result.items())})
            elif hasattr(result, '__dict__'):
                # Custom objects
                return json.dumps({k: self._serialize_result(v) for k, v in sorted(result.__dict__.items())})
            else:
                # Fallback to string representation
                return str(result)
        except Exception:
            return f"<non-serializable:{type(result).__name__}>"
    
    def _hash_result(self, result: Any) -> str:
        """Create a hash of the result for comparison."""
        serialized = self._serialize_result(result)
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()
    
    def test_core_functions_deterministic_execution(self, core_modules: List[Any]) -> None:
        """Test that core functions produce consistent outputs for identical inputs."""
        non_deterministic_functions = []
        
        for module in core_modules:
            functions = self._get_testable_functions(module)
            
            for func in functions:
                test_cases = self._generate_test_inputs(func)
                
                for args, kwargs in test_cases:
                    try:
                        # Run the function multiple times with identical inputs
                        results = []
                        result_hashes = []
                        
                        for run_num in range(self.CONSISTENCY_RUNS):
                            try:
                                result = func(*args, **kwargs)
                                results.append(result)
                                result_hashes.append(self._hash_result(result))
                            except Exception as e:
                                # Skip functions that raise exceptions
                                break
                        
                        # Check if all runs produced the same result
                        if len(result_hashes) == self.CONSISTENCY_RUNS:
                            unique_hashes = set(result_hashes)
                            if len(unique_hashes) > 1:
                                non_deterministic_functions.append({
                                    'module': module.__name__ if hasattr(module, '__name__') else 'unknown',
                                    'function': func.__name__,
                                    'args': args,
                                    'kwargs': kwargs,
                                    'results': [self._serialize_result(r) for r in results],
                                    'hashes': result_hashes
                                })
                    
                    except Exception:
                        # Skip test cases that cause errors
                        continue
        
        if non_deterministic_functions:
            error_messages = []
            for item in non_deterministic_functions:
                message = (
                    f"Non-deterministic function found:\n"
                    f"  Module: {item['module']}\n"
                    f"  Function: {item['function']}\n"
                    f"  Args: {item['args']}\n"
                    f"  Kwargs: {item['kwargs']}\n"
                    f"  Result hashes: {item['hashes'][:3]}..."  # Show first 3 hashes
                )
                if len(set(item['results'])) <= 3:  # Only show results if there aren't too many variants
                    message += f"\n  Results: {item['results']}"
                error_messages.append(message)
            
            pytest.fail(
                f"Found {len(non_deterministic_functions)} non-deterministic functions:\n\n" +
                "\n\n".join(error_messages[:10])  # Limit to first 10 for readability
            )
    
    def test_hash_function_consistency(self, core_modules: List[Any]) -> None:
        """Test that any hashing functions in the codebase are deterministic."""
        hash_functions = []
        
        # Find functions that might be hash-related
        for module in core_modules:
            for name in dir(module):
                obj = getattr(module, name)
                if (callable(obj) and 
                    ('hash' in name.lower() or 'checksum' in name.lower() or 'digest' in name.lower()) and
                    not name.startswith('_')):
                    hash_functions.append((module, obj))
        
        inconsistent_hash_functions = []
        
        for module, func in hash_functions:
            test_inputs = [
                ("test_string",),
                (b"test_bytes",),
                ({"key": "value"},),
                ([1, 2, 3],),
            ]
            
            for test_input in test_inputs:
                try:
                    # Run hash function multiple times
                    hash_results = []
                    for _ in range(self.CONSISTENCY_RUNS):
                        try:
                            result = func(*test_input)
                            hash_results.append(result)
                        except Exception:
                            break
                    
                    # Check consistency
                    if len(hash_results) == self.CONSISTENCY_RUNS:
                        if len(set(hash_results)) > 1:
                            inconsistent_hash_functions.append({
                                'module': module.__name__ if hasattr(module, '__name__') else 'unknown',
                                'function': func.__name__,
                                'input': test_input,
                                'results': hash_results
                            })
                
                except Exception:
                    continue
        
        if inconsistent_hash_functions:
            error_messages = []
            for item in inconsistent_hash_functions:
                message = (
                    f"Inconsistent hash function found:\n"
                    f"  Module: {item['module']}\n"
                    f"  Function: {item['function']}\n"
                    f"  Input: {item['input']}\n"
                    f"  Results: {item['results']}"
                )
                error_messages.append(message)
            
            pytest.fail(
                f"Found {len(inconsistent_hash_functions)} inconsistent hash functions:\n\n" +
                "\n\n".join(error_messages)
            )
    
    def test_random_seed_isolation(self) -> None:
        """Test that random operations don't affect each other between test runs."""
        try:
            import random
            import numpy as np
            
            # Test Python's random module
            original_state = random.getstate()
            
            # Generate some random numbers
            random.seed(42)
            first_run = [random.random() for _ in range(10)]
            
            random.seed(42) 
            second_run = [random.random() for _ in range(10)]
            
            # Restore original state
            random.setstate(original_state)
            
            if first_run != second_run:
                pytest.fail(
                    f"Random number generation is not deterministic with same seed:\n"
                    f"First run: {first_run[:5]}...\n"
                    f"Second run: {second_run[:5]}..."
                )
            
            # Test NumPy random if available
            try:
                np.random.seed(42)
                first_np = np.random.random(10).tolist()
                
                np.random.seed(42)
                second_np = np.random.random(10).tolist()
                
                if first_np != second_np:
                    pytest.fail(
                        f"NumPy random generation is not deterministic with same seed:\n"
                        f"First run: {first_np[:5]}...\n" 
                        f"Second run: {second_np[:5]}..."
                    )
            except ImportError:
                pass  # NumPy not available
                
        except ImportError:
            pytest.skip("Random module not available for testing")


# Global test functions for pytest discovery
def test_deterministic_execution_suite():
    """Entry point for deterministic execution tests."""
    test_instance = DeterministicExecutionTest()
    
    # Get core modules manually (simplified version)
    project_root = Path(__file__).parent.parent
    core_modules = []
    
    core_directories = [
        "egw_query_expansion/core",
        "retrieval_engine", 
        "canonical_flow",
        "analysis_nlp"
    ]
    
    for core_dir in core_directories:
        dir_path = project_root / core_dir
        if dir_path.exists():
            for py_file in dir_path.rglob("*.py"):
                if (py_file.name != "__init__.py" and 
                    not py_file.name.startswith("test_")):
                    try:
                        module_name = py_file.stem
                        spec = importlib.util.spec_from_file_location(module_name, py_file)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            sys.modules[module_name] = module
                            spec.loader.exec_module(module)
                            core_modules.append(module)
                    except Exception:
                        pass  # Skip modules that can't be imported
    
    # Run individual tests
    test_instance.test_core_functions_deterministic_execution(core_modules)
    test_instance.test_hash_function_consistency(core_modules)
    test_instance.test_random_seed_isolation()