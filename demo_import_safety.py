#!/usr/bin/env python3
"""
Demo script showing import_safety module capabilities
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'egw_query_expansion', 'core'))

def main():
    print("EGW Query Expansion - Import Safety Module Demo")
    print("=" * 50)
    
    try:
        # Import the module
        from import_safety import (
            ImportSafety, safe_import, get_import_report,
            safe_import_numpy, safe_import_scipy, safe_import_torch,
            safe_import_sklearn, safe_import_faiss, safe_import_transformers,
            check_dependencies, log_import_summary
        )
        
        print("✓ Successfully imported import_safety module")
        
        # Test core Python modules (should succeed)
        print("\n1. Testing Core Python Modules:")
        core_modules = ['os', 'sys', 'math', 'json', 'collections']
        for module in core_modules:
            result = safe_import(module)
            print(f"   {module}: {'✓' if result.success else '✗'}")
        
        # Test optional dependencies
        print("\n2. Testing Optional Dependencies:")
        
        # NumPy
        numpy_result = safe_import_numpy()
        print(f"   NumPy: {'✓' if numpy_result.success else '✗'} "
              f"{'(fallback: ' + numpy_result.fallback_used + ')' if numpy_result.fallback_used else ''}")
        
        # SciPy  
        scipy_result = safe_import_scipy()
        print(f"   SciPy: {'✓' if scipy_result.success else '✗'} "
              f"{'(fallback: ' + scipy_result.fallback_used + ')' if scipy_result.fallback_used else ''}")
        
        # PyTorch
        torch_result = safe_import_torch()
        print(f"   PyTorch: {'✓' if torch_result.success else '✗'} "
              f"{'(fallback: ' + torch_result.fallback_used + ')' if torch_result.fallback_used else ''}")
        
        # scikit-learn
        sklearn_result = safe_import_sklearn()
        print(f"   scikit-learn: {'✓' if sklearn_result.success else '✗'} "
              f"{'(fallback: ' + sklearn_result.fallback_used + ')' if sklearn_result.fallback_used else ''}")
        
        # FAISS
        faiss_result = safe_import_faiss()
        print(f"   FAISS: {'✓' if faiss_result.success else '✗'} "
              f"{'(fallback: ' + faiss_result.fallback_used + ')' if faiss_result.fallback_used else ''}")
        
        # Transformers
        transformers_result = safe_import_transformers()
        print(f"   Transformers: {'✓' if transformers_result.success else '✗'} "
              f"{'(fallback: ' + transformers_result.fallback_used + ')' if transformers_result.fallback_used else ''}")
        
        # Test fallback functionality
        print("\n3. Testing Fallback Functionality:")
        
        def mock_module_factory():
            class MockModule:
                def hello(self):
                    return "Hello from fallback!"
            return MockModule()
        
        result = safe_import('definitely_nonexistent_module', fallback_factory=mock_module_factory)
        if result.success and result.fallback_used:
            print(f"   ✓ Fallback test: {result.module.hello()}")
        else:
            print("   ✗ Fallback test failed")
        
        # Test batch dependency checking
        print("\n4. Testing Batch Dependency Checking:")
        deps_to_check = ['os', 'numpy', 'torch', 'sklearn', 'nonexistent1', 'nonexistent2']
        results = check_dependencies(deps_to_check, verbose=False)
        available = sum(1 for r in results.values() if r)
        print(f"   {available}/{len(deps_to_check)} dependencies available")
        
        # Generate final report
        print("\n5. Final Import Report:")
        report = get_import_report()
        summary = report['summary']
        print(f"   Total attempts: {summary['total_attempts']}")
        print(f"   Successful imports: {summary['successful_imports']}")
        print(f"   Failed imports: {summary['failed_imports']}")
        print(f"   Fallbacks used: {summary['fallbacks_used']}")
        print(f"   Critical failures: {summary['critical_failures']}")
        
        if summary['critical_failures'] > 0:
            print(f"   Critical failures: {report['critical_failures']}")
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        
        # Test some mock functionality if scipy fallback is active
        if scipy_result.fallback_used:
            print("\n6. Testing SciPy Fallback Functionality:")
            scipy_mock = scipy_result.module
            
            # Test distance calculations
            u = [1, 2, 3]
            v = [4, 5, 6]
            
            cosine_dist = scipy_mock.spatial.distance.cosine(u, v)
            euclidean_dist = scipy_mock.spatial.distance.euclidean(u, v)
            
            print(f"   Cosine distance: {cosine_dist:.4f}")
            print(f"   Euclidean distance: {euclidean_dist:.4f}")
            
            # Test optimization
            def test_func(x):
                return x[0]**2 + x[1]**2
            
            opt_result = scipy_mock.optimize.minimize(test_func, [1.0, 1.0])
            print(f"   Optimization result: success={opt_result.success}")
            
            # Test entropy
            pk = [0.5, 0.3, 0.2]
            entropy_val = scipy_mock.stats.entropy(pk)
            print(f"   Entropy calculation: {entropy_val:.4f}")
        
        # Test FAISS fallback functionality
        if faiss_result.fallback_used and numpy_result.success:
            print("\n7. Testing FAISS Fallback Functionality:")
            faiss_mock = faiss_result.module
            np = numpy_result.module
            
            # Create a simple index
            d = 64  # dimension
            index = faiss_mock.IndexFlatL2(d)
            
            # Add some vectors
            vectors = np.random.randn(10, d).astype(np.float32)
            index.add(vectors)
            
            print(f"   Added {index.ntotal} vectors to index")
            
            # Search
            query = np.random.randn(1, d).astype(np.float32)
            distances, indices = index.search(query, 3)
            print(f"   Search returned {len(indices[0])} results")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)