#!/usr/bin/env python3
"""
Run refusal tests without pytest dependency
"""

import sys
import json
# # # from tests.test_refusal import TestRefusalScenarios  # Module not found  # Module not found  # Module not found

def main():
    """Run comprehensive refusal tests"""
    
    # Run comprehensive test
    test_suite = TestRefusalScenarios()
    test_suite.setup_method()

    print('Running comprehensive refusal tests...')
    try:
        summary = test_suite.test_generate_refusal_summary()
        
        print(f'\nRefusal Test Summary:')
        print(f'Total tests: {summary["total_tests"]}')
        print(f'Refusal tests: {summary["refusal_tests"]}')
        print(f'Acceptance tests: {summary["acceptance_tests"]}')
        print(f'Passed: {summary["passed_tests"]}')
        print(f'Failed: {summary["failed_tests"]}')
        print(f'Coverage: {summary["coverage"]}')
        
        if summary["failed_tests"] == 0:
            print('\n✓ All refusal scenarios tested successfully!')
            return True
        else:
            print(f'\n✗ {summary["failed_tests"]} tests failed')
            return False
            
    except Exception as e:
        print(f'Error during testing: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)