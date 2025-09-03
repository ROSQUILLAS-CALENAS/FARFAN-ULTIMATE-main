"""
Pytest configuration for the main test suite.
"""

import pytest
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", 
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", 
        "slow: marks tests as slow running tests"
    )
    config.addinivalue_line(
        "markers", 
        "architecture: marks tests as architecture fitness function tests"
    )
    config.addinivalue_line(
        "markers", 
        "phase_enforcement: marks tests as phase layer enforcement tests"
    )