#!/usr/bin/env python3
"""
Validation script for EGW Query Expansion installation
"""

import importlib
import sys
import traceback


def test_imports():
    """Test importing core modules."""
    print("🔍 Testing module imports...")

    try:
        # Test numpy/scipy
        import numpy as np
        import scipy

        print("✅ NumPy and SciPy imported successfully")

        # Test PyTorch
        import torch

        print("✅ PyTorch imported successfully")

        # Test FAISS
        import faiss

        print("✅ FAISS imported successfully")

        # Test POT
        import ot

        print("✅ POT (Python Optimal Transport) imported successfully")

        # Test transformers and sentence-transformers
        import sentence_transformers
        import transformers

        print("✅ Transformers and Sentence-Transformers imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality without heavy computation."""
    print("\n🧪 Testing basic functionality...")

    try:
        import numpy as np

        # Test basic array operations
        a = np.random.rand(10, 5)
        b = np.random.rand(5, 8)
        c = np.dot(a, b)
        print("✅ Basic NumPy operations work")

        # Test FAISS basic operations
        import faiss

        index = faiss.IndexFlatL2(128)
        vectors = np.random.random((10, 128)).astype("float32")
        index.add(vectors)
        print("✅ Basic FAISS operations work")

        # Test POT basic operations
        import ot

        a = np.ones(10) / 10
        b = np.ones(10) / 10
        M = np.random.rand(10, 10)
        transport = ot.emd(a, b, M)
        print("✅ Basic POT operations work")

        return True

    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_egw_modules():
    """Test EGW-specific module imports."""
    print("\n🎯 Testing EGW module imports...")

    try:
        sys.path.append(".")

        # Test conformal risk control module
        from egw_query_expansion.core.conformal_risk_control import (
            ConformalRiskController,
            RiskControlConfig,
            create_conformal_system,
        )

        print("✅ Conformal Risk Control module imported successfully")

        # Test other core modules if they exist
        try:
            from egw_query_expansion.core.gw_alignment import GromovWassersteinAligner

            print("✅ GromovWassersteinAligner imported successfully")
        except ImportError:
            print("ℹ️  GromovWassersteinAligner not available (module not implemented)")

        try:
            from egw_query_expansion.core.query_generator import QueryGenerator

            print("✅ QueryGenerator imported successfully")
        except ImportError:
            print("ℹ️  QueryGenerator not available (module not implemented)")

        try:
            from egw_query_expansion.core.hybrid_retrieval import HybridRetriever

            print("✅ HybridRetriever imported successfully")
        except ImportError:
            print("ℹ️  HybridRetriever not available (module not implemented)")

        try:
            from egw_query_expansion.core.pattern_matcher import PatternMatcher

            print("✅ PatternMatcher imported successfully")
        except ImportError:
            print("ℹ️  PatternMatcher not available (module not implemented)")

        # Test main package import
        import egw_query_expansion

        print("✅ Main package imported successfully")

        return True

    except ImportError as e:
        print(f"❌ EGW module import error: {e}")
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration loading."""
    print("\n⚙️  Testing configuration...")

    try:
        import os

        import yaml

        config_path = "egw_query_expansion/configs/default_config.yaml"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            print("✅ Configuration file loaded successfully")
            print(f"   - Models configured: {list(config.get('models', {}).keys())}")
            print(
                f"   - GW epsilon: {config.get('gw_alignment', {}).get('epsilon', 'N/A')}"
            )
            return True
        else:
            print(f"❌ Configuration file not found: {config_path}")
            return False

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_conformal_functionality():
    """Test conformal risk control functionality."""
    print("\n🔬 Testing Conformal Risk Control functionality...")

    try:
        import numpy as np

        from egw_query_expansion.core.conformal_risk_control import (
            ClassificationScore,
            RegressionScore,
            create_conformal_system,
        )

        # Test score functions
        score_fn = ClassificationScore()
        y_pred = np.array([[0.1, 0.9], [0.7, 0.3]])
        y_true = np.array([1, 0])
        scores = score_fn(y_pred, y_true)
        print("✅ Classification scoring function works")

        reg_score_fn = RegressionScore()
        y_pred_reg = np.array([1.5, 2.8])
        y_true_reg = np.array([1.0, 3.0])
        reg_scores = reg_score_fn(y_pred_reg, y_true_reg)
        print("✅ Regression scoring function works")

        # Test system creation
        controller, score_fn = create_conformal_system(
            task_type="classification", alpha=0.1, seed=42
        )
        print("✅ Conformal system creation works")

        return True

    except Exception as e:
        print(f"❌ Conformal functionality test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("🚀 EGW Query Expansion with Conformal Risk Control - Installation Validation")
    print("=" * 70)

    tests = [
        ("Import Tests", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("EGW Modules", test_egw_modules),
        ("Conformal Functionality", test_conformal_functionality),
        ("Configuration", test_configuration),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n📊 Validation Summary")
    print("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All validation tests passed! EGW Query Expansion is ready to use.")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the installation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
