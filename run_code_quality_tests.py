#!/usr/bin/env python3
"""
Standalone test runner for code quality fixes validation
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

# Now run the tests
if __name__ == "__main__":
# # #     from tests.test_code_quality_fixes import run_basic_tests  # Module not found  # Module not found  # Module not found
    success = run_basic_tests()
    sys.exit(0 if success else 1)