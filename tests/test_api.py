"""Tests for main API functionality."""

import hashlib
import tempfile
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

import orjson
import pytest

# # # from standards_alignment.api import (  # Module not found  # Module not found  # Module not found
    get_dimension_patterns,
    get_point_requirements,
    get_verification_criteria,
    load_standards,
)


class TestAPI:
    def test_load_standards_creates_sample(self):
        """Test that load_standards creates sample file if none exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            standards_path = Path(tmpdir) / "standards.json"

            # Should create sample standards
            result = load_standards(str(standards_path))

            assert standards_path.exists()
            assert "data" in result
            assert "graph" in result
            assert "checksum" in result
            assert "version" in result

    def test_round_trip_integrity(self):
        """Test round-trip integrity: checksum should be identical."""
        with tempfile.TemporaryDirectory() as tmpdir:
            standards_path = Path(tmpdir) / "standards.json"

            # First load
            result1 = load_standards(str(standards_path))
            checksum1 = result1["checksum"]

            # Second load should use cache
            result2 = load_standards(str(standards_path))
            checksum2 = result2["checksum"]

            assert checksum1 == checksum2

    def test_get_dimension_patterns(self):
        """Test dimension pattern retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            standards_path = Path(tmpdir) / "standards.json"
            load_standards(str(standards_path))  # Create sample

            patterns = get_dimension_patterns("security")
            assert "authentication" in patterns
            assert patterns["authentication"].pattern_type.value == "regex"

    def test_get_point_requirements(self):
        """Test point requirements retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            standards_path = Path(tmpdir) / "standards.json"
            load_standards(str(standards_path))  # Create sample

            requirements = get_point_requirements(1)
            assert "role_definition" in requirements
            assert (
                requirements["role_definition"].description
                == "Define user roles clearly"
            )

    def test_get_verification_criteria(self):
        """Test verification criteria retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            standards_path = Path(tmpdir) / "standards.json"
            load_standards(str(standards_path))  # Create sample

            criteria = get_verification_criteria("security", "access_control")
            assert "user_management" in criteria
            assert criteria["user_management"].name == "user_management"


if __name__ == "__main__":
    pytest.main([__file__])
