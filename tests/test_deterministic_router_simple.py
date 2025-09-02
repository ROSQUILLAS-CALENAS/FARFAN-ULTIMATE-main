#!/usr/bin/env python3
"""
Simple test script for the Deterministic Router without external dependencies
"""

import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.getcwd())


def test_imports():
    """Test that the deterministic router can be imported"""
    try:
        from egw_query_expansion.core.deterministic_router import (
            ActionType,
            DeterministicRouter,
            ImmutableConfig,
            RoutingContext,
            create_deterministic_router,
        )

        print("✅ Successfully imported deterministic router components")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality without NumPy"""
    try:
        # Import after checking dependencies
        from egw_query_expansion.core.deterministic_router import (
            ActionType,
            ImmutableConfig,
            SeedDerivation,
        )

        # Test config creation
        config = ImmutableConfig()
        print(f"✅ Created config with hash: {config.config_hash[:8]}...")

        # Test action enum
        actions = list(ActionType)
        print(f"✅ ActionType enum has {len(actions)} actions")

        # Test seed derivation (without numpy arrays)
        test_context = type(
            "MockContext",
            (),
            {
                "query_hash": "testhash123",
                "query_embedding": (0.1, 0.2, 0.3),
                "corpus_size": 1000,
                "retrieval_mode": "hybrid",
            },
        )()

        seed = SeedDerivation.derive_seed(test_context, 1, "test_module", "test_hash")
        print(f"✅ Generated deterministic seed: {seed}")

        # Test that same inputs produce same seed
        seed2 = SeedDerivation.derive_seed(test_context, 1, "test_module", "test_hash")
        if seed == seed2:
            print("✅ Deterministic seed generation verified")
        else:
            print("❌ Seed generation is not deterministic")
            return False

        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_without_numpy():
    """Test components that don't require NumPy"""
    try:
        from egw_query_expansion.core.deterministic_router import (
            ActionType,
            LexicographicComparator,
        )

        comp = LexicographicComparator()

        # Test action comparison
        result = comp.compare_actions(
            ActionType.ROUTE_TO_SPARSE, ActionType.ROUTE_TO_DENSE, "hash1", "hash2"
        )

        if result == -1:
            print("✅ Lexicographic comparator working correctly")
        else:
            print("❌ Lexicographic comparator failed")
            return False

        return True

    except Exception as e:
        print(f"❌ Non-NumPy test failed: {e}")
        return False


def main():
    print("🧪 Testing Deterministic Router (Simple Version)")
    print("=" * 50)

    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Non-NumPy Components", test_without_numpy),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")

    print(f"\n📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
