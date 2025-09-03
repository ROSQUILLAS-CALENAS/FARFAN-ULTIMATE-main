"""
Pytest configuration for architecture fitness function tests.
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
        "architecture: marks tests as architecture fitness function tests"
    )
    config.addinivalue_line(
        "markers", 
        "phase_enforcement: marks tests as phase layer enforcement tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add custom markers."""
    for item in items:
        # Mark all tests in architecture_tests as architecture tests
        if "architecture_tests" in str(item.fspath):
            item.add_marker(pytest.mark.architecture)
            
        # Mark phase enforcement tests
        if "test_phase_enforcement" in item.name or "phase_enforcement" in str(item.fspath):
            item.add_marker(pytest.mark.phase_enforcement)