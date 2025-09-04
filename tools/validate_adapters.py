#!/usr/bin/env python3
"""Validation script for anti_corruption_adapters module"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def validate_module():
    """Validate the anti_corruption_adapters module"""
    try:
        # Import the module
        import tools.anti_corruption_adapters as aca
        
        print("✓ Module imported successfully")
        
        # Check key classes exist
        required_classes = [
            'RetrievalResultDTO',
            'AnalysisInputDTO', 
            'AnalysisResultDTO',
            'BaseAntiCorruptionAdapter',
            'RetrievalToAnalysisAdapter',
            'ImportGuard',
            'SchemaViolationLogger',
            'AdapterFactory'
        ]
        
        for cls_name in required_classes:
            if hasattr(aca, cls_name):
                print(f"✓ Found {cls_name}")
            else:
                print(f"✗ Missing {cls_name}")
                return False
        
        # Test DTO creation
        retrieval_dto = aca.RetrievalResultDTO(
            query="test",
            documents=[{"id": "1", "content": "test"}],
            scores=[0.8],
            retrieval_method="test"
        )
        print("✓ RetrievalResultDTO created successfully")
        
        # Test DTO validation
        errors = retrieval_dto.validate()
        if not errors:
            print("✓ RetrievalResultDTO validation passed")
        else:
            print(f"✗ RetrievalResultDTO validation failed: {errors}")
            return False
        
        # Test adapter creation
        adapter = aca.RetrievalToAnalysisAdapter()
        print("✓ Adapter created successfully")
        
        # Test import guard
        guard = aca.ImportGuard()
        print("✓ ImportGuard created successfully")
        
        print("\n✓ All validation tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = validate_module()
    sys.exit(0 if success else 1)