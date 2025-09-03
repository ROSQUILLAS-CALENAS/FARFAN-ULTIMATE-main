#!/usr/bin/env python3
"""
Validate the import_safety module functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'egw_query_expansion', 'core'))

try:
    from import_safety import (
        ImportSafety, safe_import, get_import_report, 
        safe_import_numpy, safe_import_scipy, safe_import_torch,
        safe_import_sklearn, safe_import_faiss, safe_import_transformers,
        safe_import_sentence_transformers, safe_import_pot,
        check_dependencies, log_import_summary
    )
    print("✓ Successfully imported import_safety module")
except ImportError as e:
    print(f"✗ Failed to import import_safety: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic import safety functionality"""
    print("\n=== Testing Basic Functionality ===")
    
    # Test singleton pattern
    safety1 = ImportSafety()
    safety2 = ImportSafety()
    assert safety1 is safety2, "ImportSafety should be a singleton"
    print("✓ Singleton pattern works")
    
    # Test successful import
    result = safe_import('os')
    assert result.success, "Should successfully import 'os'"
    assert result.module is not None, "Should have module object"
    print("✓ Successful import works")
    
    # Test failed import with graceful handling
    result = safe_import('nonexistent_module', required=False)
    assert not result.success, "Should fail for nonexistent module"
    assert result.error is not None, "Should have error information"
    print("✓ Failed import handled gracefully")
    
    # Test fallback functionality
    def mock_fallback():
        class MockModule:
            def test_method(self):
                return "fallback works"
        return MockModule()
    
    result = safe_import('another_nonexistent_module', fallback_factory=mock_fallback)
    assert result.success, "Should succeed with fallback"
    assert result.fallback_used == "factory", "Should indicate fallback was used"
    assert result.module.test_method() == "fallback works", "Fallback should be functional"
    print("✓ Fallback functionality works")

def test_specialized_imports():
    """Test specialized import methods"""
    print("\n=== Testing Specialized Imports ===")
    
    specialized_imports = [
        ('numpy', safe_import_numpy),
        ('scipy', safe_import_scipy),
        ('torch', safe_import_torch),
        ('sklearn', safe_import_sklearn),
        ('faiss', safe_import_faiss),
        ('transformers', safe_import_transformers),
        ('sentence_transformers', safe_import_sentence_transformers),
        ('pot', safe_import_pot),
    ]
    
    for name, import_func in specialized_imports:
        try:
            result = import_func()
            status = "✓ Available" if result.success else "✗ Missing"
            fallback_info = f" (using {result.fallback_used})" if result.fallback_used else ""
            print(f"{name}: {status}{fallback_info}")
        except Exception as e:
            print(f"{name}: ✗ Error: {e}")

def test_batch_checking():
    """Test batch dependency checking"""
    print("\n=== Testing Batch Dependency Checking ===")
    
    deps = ['os', 'sys', 'math', 'nonexistent_module1', 'nonexistent_module2']
    results = check_dependencies(deps, verbose=False)
    
    assert results['os'] == True, "os should be available"
    assert results['sys'] == True, "sys should be available"
    assert results['math'] == True, "math should be available"
    assert results['nonexistent_module1'] == False, "nonexistent module should be unavailable"
    print("✓ Batch dependency checking works")

def test_import_report():
    """Test import reporting functionality"""
    print("\n=== Testing Import Report ===")
    
    # Generate some import activity
    safe_import('os')
    safe_import('sys')
    safe_import('math')
    safe_import('nonexistent_test_module', required=False)
    
    report = get_import_report()
    assert 'summary' in report, "Report should contain summary"
    assert 'total_attempts' in report['summary'], "Summary should contain total attempts"
    assert 'successful_imports' in report['summary'], "Summary should contain successful imports"
    assert 'failed_imports' in report['summary'], "Summary should contain failed imports"
    
    print(f"✓ Report generated: {report['summary']['total_attempts']} total attempts")
    
    # Test logging summary
    try:
        log_import_summary()
        print("✓ Import summary logging works")
    except Exception as e:
        print(f"✗ Import summary logging failed: {e}")

def main():
    """Run all validation tests"""
    print("Validating import_safety module...")
    
    try:
        test_basic_functionality()
        test_specialized_imports()
        test_batch_checking()
        test_import_report()
        
        print("\n=== All Tests Passed! ===")
        print("The import_safety module is working correctly.")
        
        # Final comprehensive report
        print("\n=== Final Import Report ===")
        report = get_import_report()
        print(f"Total import attempts: {report['summary']['total_attempts']}")
        print(f"Successful imports: {report['summary']['successful_imports']}")
        print(f"Failed imports: {report['summary']['failed_imports']}")
        print(f"Fallbacks used: {report['summary']['fallbacks_used']}")
        if report['summary']['critical_failures'] > 0:
            print(f"Critical failures: {report['summary']['critical_failures']}")
            print(f"Failed modules: {report['critical_failures']}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)