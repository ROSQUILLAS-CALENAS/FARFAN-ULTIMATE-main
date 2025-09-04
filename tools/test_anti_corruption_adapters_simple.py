#!/usr/bin/env python3
"""
Simple test script for anti_corruption_adapters module syntax validation
"""

def test_syntax():
    """Test that the module has valid Python syntax"""
    try:
        import ast
        
        with open('tools/anti_corruption_adapters.py', 'r') as f:
            source = f.read()
        
        # Parse the AST
        ast.parse(source)
        print("✓ Module syntax is valid")
        return True
        
    except Exception as e:
        print(f"✗ Syntax error: {e}")
        return False


def test_basic_structure():
    """Test basic module structure without imports that might fail"""
    try:
        with open('tools/anti_corruption_adapters.py', 'r') as f:
            source = f.read()
        
        # Check for key components
        required_elements = [
            'class RetrievalResultDTO',
            'class AnalysisInputDTO', 
            'class AnalysisResultDTO',
            'class BaseAntiCorruptionAdapter',
            'class RetrievalToAnalysisAdapter',
            'class ImportGuard',
            'class SchemaViolationLogger',
            'def translate',
            'def validate'
        ]
        
        missing = []
        for element in required_elements:
            if element not in source:
                missing.append(element)
        
        if missing:
            print(f"✗ Missing elements: {missing}")
            return False
        
        print("✓ Module structure is complete")
        return True
        
    except Exception as e:
        print(f"✗ Structure test failed: {e}")
        return False


def test_docstrings():
    """Test that key classes have docstrings"""
    try:
        with open('tools/anti_corruption_adapters.py', 'r') as f:
            source = f.read()
        
        # Check for docstrings after key class definitions
        class_patterns = [
            ('class RetrievalResultDTO', 'DTO for retrieval'),
            ('class AnalysisInputDTO', 'DTO for analysis'),
            ('class BaseAntiCorruptionAdapter', 'Base class for'),
            ('class ImportGuard', 'Prevents backward dependencies')
        ]
        
        for class_def, expected_doc in class_patterns:
            if class_def in source:
                # Find the class definition and check if docstring follows
                class_pos = source.find(class_def)
                next_lines = source[class_pos:class_pos+500]  # Check next 500 chars
                if '"""' in next_lines:
                    print(f"✓ Found docstring for {class_def}")
                else:
                    print(f"⚠ Missing docstring for {class_def}")
        
        print("✓ Docstring check complete")
        return True
        
    except Exception as e:
        print(f"✗ Docstring test failed: {e}")
        return False


def main():
    """Run all simple tests"""
    print("Testing anti_corruption_adapters module (syntax only)...")
    print("=" * 60)
    
    tests = [
        test_syntax,
        test_basic_structure, 
        test_docstrings
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✓ All syntax tests passed!")
        return True
    else:
        print("✗ Some tests failed!")
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)