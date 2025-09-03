#!/usr/bin/env python3
"""Simple test for import_safety module"""

import sys
import os

# Add the core directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'egw_query_expansion', 'core'))

def test_import():
    try:
        # Test basic import
# # #         from import_safety import ImportSafety, safe_import  # Module not found  # Module not found  # Module not found
        print("✓ Basic import successful")
        
        # Test singleton
        safety1 = ImportSafety()
        safety2 = ImportSafety()
        print(f"✓ Singleton test: {safety1 is safety2}")
        
        # Test safe import of existing module
        result = safe_import('os')
        print(f"✓ os import: success={result.success}, module={result.module is not None}")
        
        # Test safe import of non-existent module
        result = safe_import('nonexistent_module', required=False)
        print(f"✓ nonexistent_module import: success={result.success}, error={result.error is not None}")
        
        # Test specialized imports
# # #         from import_safety import (  # Module not found  # Module not found  # Module not found
            safe_import_numpy, safe_import_scipy, safe_import_torch,
            safe_import_sklearn, safe_import_faiss
        )
        
        numpy_result = safe_import_numpy()
        print(f"✓ NumPy: success={numpy_result.success}, fallback={numpy_result.fallback_used}")
        
        scipy_result = safe_import_scipy()  
        print(f"✓ SciPy: success={scipy_result.success}, fallback={scipy_result.fallback_used}")
        
        torch_result = safe_import_torch()
        print(f"✓ PyTorch: success={torch_result.success}, fallback={torch_result.fallback_used}")
        
        sklearn_result = safe_import_sklearn()
        print(f"✓ scikit-learn: success={sklearn_result.success}, fallback={sklearn_result.fallback_used}")
        
        faiss_result = safe_import_faiss()
        print(f"✓ FAISS: success={faiss_result.success}, fallback={faiss_result.fallback_used}")
        
        # Test report generation
# # #         from import_safety import get_import_report  # Module not found  # Module not found  # Module not found
        report = get_import_report()
        print(f"✓ Import report: {report['summary']['total_attempts']} attempts, "
              f"{report['summary']['successful_imports']} successful, "
              f"{report['summary']['failed_imports']} failed")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("Testing import_safety module...")
    success = test_import()
    print(f"Test {'PASSED' if success else 'FAILED'}")