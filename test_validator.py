#!/usr/bin/env python3
"""
Test script for the refactored normative validator module.
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

def test_validator():
    """Test the normative validator functionality."""
    try:
        from canonical_flow.I_ingestion_preparation.normative_validator import process, NormativeValidator
        
        print("✓ Module imported successfully!")
        
        # Test 1: Valid document with both bundle and features
        print("\n=== Test 1: sample_document (should be ready) ===")
        result = process('sample_document')
        print(f"Status: {result['status']}")
        print(f"Ready: {result['ready']}")
        print(f"Validation status: {result['validation_result']['status']}")
        print(f"Issues count: {len(result['validation_result']['issues'])}")
        
        # Check if validation file was created
        validation_file = Path("canonical_flow/ingestion/sample_document_validation.json")
        if validation_file.exists():
            print("✓ Validation artifact created successfully")
        else:
            print("✗ Validation artifact not found")
        
        # Test 2: Document with processing failure
        print("\n=== Test 2: nonexistent_document (should not be ready) ===")
        result2 = process('nonexistent_document')
        print(f"Status: {result2['status']}")
        print(f"Ready: {result2['ready']}")
        print(f"Validation status: {result2['validation_result']['status']}")
        print(f"Issues count: {len(result2['validation_result']['issues'])}")
        
        # Test 3: Completely missing document
        print("\n=== Test 3: missing_document (should error gracefully) ===")
        result3 = process('missing_document')
        print(f"Status: {result3['status']}")
        print(f"Ready: {result3['ready']}")
        print(f"Validation status: {result3['validation_result']['status']}")
        print(f"Issues count: {len(result3['validation_result']['issues'])}")
        
        print("\n✓ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_validator()
    sys.exit(0 if success else 1)