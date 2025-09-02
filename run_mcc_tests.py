#!/usr/bin/env python3
"""
Standalone test runner for MCC (Monotone Consistency Check) tests.
Runs without pytest dependency for basic validation.
"""

import json
import sys
from pathlib import Path

# Import test components
sys.path.append(str(Path(__file__).parent))
from tests.test_mcc import (
    SatLabel, Clause, Evidence, HornEvaluator,
    test_monotone_consistency_basic,
    test_monotone_consistency_comprehensive,
    test_mandatory_failure_deterministic_downgrade,
    test_clause_triggering_trace
)


def run_test(test_func, test_name):
    """Run a single test function"""
    try:
        print(f"Running {test_name}...")
        test_func()
        print(f"✓ {test_name} PASSED")
        return True
    except Exception as e:
        print(f"✗ {test_name} FAILED: {e}")
        return False


def main():
    """Run all MCC tests"""
    print("=" * 60)
    print("MONOTONE CONSISTENCY CHECK (MCC) TEST SUITE")
    print("=" * 60)
    
    tests = [
        (test_monotone_consistency_basic, "Basic Monotonicity"),
        (test_monotone_consistency_comprehensive, "Comprehensive Monotonicity"),
        (test_mandatory_failure_deterministic_downgrade, "Deterministic Downgrade"),
        (test_clause_triggering_trace, "Clause Triggering Trace")
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func, test_name in tests:
        if run_test(test_func, test_name):
            passed += 1
        print()
    
    print("=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    # Check if certificate was generated
    cert_path = Path('mcc_certificate.json')
    if cert_path.exists():
        with open(cert_path, 'r') as f:
            cert = json.load(f)
        print(f"Certificate: {json.dumps(cert, indent=2)}")
    else:
        print("No certificate generated")
    
    print("=" * 60)
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)