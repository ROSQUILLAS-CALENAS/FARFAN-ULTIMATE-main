#!/usr/bin/env python3
"""Simple test runner for the evidence system"""

from test_evidence_system import TestEvidenceSystem, test_evidence_msgspec_serialization


def main():
    print("Running Evidence System Tests")
    print("=" * 40)

    test_methods = [
        ("Idempotent insertion", "test_add_evidence_idempotent"),
        ("Evidence retrieval", "test_get_evidence_for_question"),
        ("Group by dimension", "test_group_by_dimension"),
        ("Shuffle invariance", "test_shuffle_invariance"),
        ("Coverage calculation", "test_coverage_calculation"),
        ("Coverage audit", "test_coverage_audit"),
        ("Evidence hash consistency", "test_evidence_hash_consistency"),
        ("DR-submodular selection", "test_dr_submodular_selection"),
        ("System statistics", "test_system_stats"),
        ("msgspec serialization", lambda: test_evidence_msgspec_serialization()),
    ]

    passed = 0
    failed = 0

    for test_name, test_method in test_methods:
        try:
            # Create fresh instance for each test
            if callable(test_method):
                test_method()  # For standalone functions
            else:
                test_system = TestEvidenceSystem()
                test_system.setup_method()
                getattr(test_system, test_method)()

            print(f"✓ {test_name}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name}: {str(e)}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"❌ {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
