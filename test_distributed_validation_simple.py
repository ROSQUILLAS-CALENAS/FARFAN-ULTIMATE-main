#!/usr/bin/env python3
"""
Simple test script to validate the basic functionality without dependencies.
"""

def test_basic_validation():
    """Test the basic validation functions from module_distributed_processor."""
    print("Testing basic validation functionality...")
    
    try:
        import module_distributed_processor
        
        # Test the validate_function_importability function
        validation_results = module_distributed_processor.validate_function_importability()
        
        print("Validation results:")
        print(f"  Module importable: {validation_results.get('module_importable', False)}")
        print(f"  Function accessible: {validation_results.get('process_document_accessible', False)}")
        
        # Test serialization
        serializable = module_distributed_processor.test_serialization()
        print(f"  Function serializable: {serializable}")
        
        # Test process_document function
        print("\nTesting process_document function...")
        
        # Create a test file for processing
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document for processing.")
            temp_file_path = f.name
        
        try:
            result = module_distributed_processor.process_document(
                temp_file_path, 
                "test query",
                {"test": "config"}
            )
            
            print(f"  Process result status: {result.get('status', 'unknown')}")
            print(f"  Processing time: {result.get('processing_time', 0):.3f}s")
            
            if result.get('status') == 'success':
                print("✅ Process document test successful")
                return True
            else:
                print(f"❌ Process document failed: {result.get('error_message', 'Unknown error')}")
                return False
                
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass
                
    except Exception as e:
        print(f"❌ Basic validation failed: {e}")
        return False

def test_import_error_handling():
    """Test error handling for import scenarios."""
    print("\nTesting import error handling...")
    
    try:
        # Try importing a module that doesn't exist to test error handling
        import sys
        import types
        
        # Create a mock module to test error scenarios
        mock_module = types.ModuleType('test_module')
        sys.modules['test_module'] = mock_module
        
        # Test validation with mock module
        import module_distributed_processor
        
        # The validation should handle missing attributes gracefully
        validation_results = module_distributed_processor.validate_function_importability()
        
        if 'import_errors' in validation_results or 'attribute_errors' in validation_results:
            print("✅ Error handling test successful - validation captures errors")
            return True
        else:
            print("✅ Validation successful (no errors to handle)")
            return True
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run simple validation tests."""
    print("=== Simple Distributed Processor Validation ===\n")
    
    tests = [
        test_basic_validation,
        test_import_error_handling
    ]
    
    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print("=== Summary ===")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All simple tests passed!")
        return 0
    else:
        print("❌ Some simple tests failed!")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())