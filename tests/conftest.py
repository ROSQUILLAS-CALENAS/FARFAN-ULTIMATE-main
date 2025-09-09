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
    config.addinivalue_line(
        "markers",
        "syntax_validation: marks tests that validate Python syntax"
    )
    config.addinivalue_line(
        "markers",
        "security_regression: marks tests that check for security regressions"
    )
    config.addinivalue_line(
        "markers", 
        "phase_annotation: marks tests that validate phase decorators"
    )
    config.addinivalue_line(
        "markers",
        "deterministic_execution: marks tests that verify deterministic behavior"
    )


@pytest.fixture(scope="session")
def project_root():
    """Provide the project root directory as a fixture."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session") 
def python_files(project_root):
    """Discover all Python files in the project."""
    python_files = []
    for pattern in ["**/*.py"]:
        python_files.extend(project_root.glob(pattern))
    
    # Filter out common exclusions
    excluded_patterns = [
        "venv/", ".venv/", "__pycache__/", ".pytest_cache/", 
        ".mypy_cache/", ".git/", "build/", "dist/", "test_deps/"
    ]
    
    filtered_files = []
    for file_path in python_files:
        should_exclude = any(pattern in str(file_path) for pattern in excluded_patterns)
        if not should_exclude and file_path.is_file():
            filtered_files.append(file_path)
    
    return sorted(filtered_files)


@pytest.fixture(scope="session")
def core_directories(project_root):
    """Get core directories for testing."""
    core_dirs = []
    potential_dirs = [
        "egw_query_expansion/core",
        "retrieval_engine",
        "canonical_flow", 
        "analysis_nlp",
        "src",
        "phases",
        "adapters",
        "microservices"
    ]
    
    for dir_name in potential_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            core_dirs.append(dir_path)
    
    return core_dirs


@pytest.fixture(scope="function")
def temp_test_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path


# Configure pytest collection to include our new test modules
def pytest_collection_modifyitems(config, items):
    """Add markers to test items based on their module."""
    for item in items:
        # Add markers based on test file names
        if "syntax_validation_test" in item.nodeid:
            item.add_marker(pytest.mark.syntax_validation)
        elif "security_regression_test" in item.nodeid:
            item.add_marker(pytest.mark.security_regression)  
        elif "phase_annotation_test" in item.nodeid:
            item.add_marker(pytest.mark.phase_annotation)
        elif "deterministic_execution_test" in item.nodeid:
            item.add_marker(pytest.mark.deterministic_execution)
        
        # Mark slow tests
        if any(name in item.nodeid for name in ["security_regression", "deterministic_execution"]):
            item.add_marker(pytest.mark.slow)