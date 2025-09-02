#!/usr/bin/env python3
"""
Test script to validate the distributed processor modifications.
"""

import sys
import logging
from pathlib import Path

def test_module_import():
    """Test importing the module_distributed_processor module."""
    print("Testing module_distributed_processor import...")
    try:
        import module_distributed_processor
        print("✅ Module import successful")
        return True
    except ImportError as e:
        print(f"❌ Module import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        return False

def test_function_access():
    """Test accessing the process_document function."""
    print("Testing process_document function access...")
    try:
        import module_distributed_processor
        func = getattr(module_distributed_processor, 'process_document', None)
        if func and callable(func):
            print("✅ Function access successful")
            return True
        else:
            print("❌ Function not found or not callable")
            return False
    except Exception as e:
        print(f"❌ Function access failed: {e}")
        return False

def test_distributed_processor_validation():
    """Test the DistributedProcessor validation functionality."""
    print("Testing DistributedProcessor validation...")
    try:
        from distributed_processor import DistributedProcessor
        
        # Create a processor instance (this will run validation)
        processor = DistributedProcessor()
        
        # Check if validation results are available
        if hasattr(processor, 'validation_results'):
            validation = processor.validation_results
            print(f"  Module importable: {validation.get('module_importable', False)}")
            print(f"  Function accessible: {validation.get('process_document_accessible', False)}")
            print(f"  Function serializable: {validation.get('function_serializable', False)}")
            print(f"  Subprocess validation: {validation.get('subprocess_validation_passed', False)}")
            print(f"  All checks passed: {validation.get('all_checks_passed', False)}")
            
            if validation.get('all_checks_passed', False):
                print("✅ DistributedProcessor validation successful")
                return True
            else:
                print("❌ DistributedProcessor validation failed")
                return False
        else:
            print("❌ Validation results not available")
            return False
            
    except Exception as e:
        print(f"❌ DistributedProcessor validation failed: {e}")
        return False

def test_setup_validation():
    """Test the distributed processing setup validation."""
    print("Testing setup validation...")
    try:
        from distributed_processor import DistributedProcessor
        
        processor = DistributedProcessor()
        result = processor.validate_distributed_processing_setup()
        
        if result:
            print("✅ Setup validation passed")
            return True
        else:
            print("❌ Setup validation failed")
            return False
            
    except Exception as e:
        print(f"❌ Setup validation error: {e}")
        return False

def main():
    """Run all validation tests."""
    print("=== Distributed Processor Validation Tests ===\n")
    
    tests = [
        ("Module Import", test_module_import),
        ("Function Access", test_function_access),
        ("DistributedProcessor Validation", test_distributed_processor_validation),
        ("Setup Validation", test_setup_validation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"--- {test_name} ---")
        result = test_func()
        results.append(result)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=== Summary ===")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())