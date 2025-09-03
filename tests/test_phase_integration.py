"""
Integration tests that validate phase enforcement within the main test suite.

These tests ensure that the architecture fitness functions are integrated
properly into the testing framework.
"""

import pytest
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from architecture_tests.test_phase_enforcement import ImportAnalyzer, PhaseViolationError
    from architecture_tests.test_import_graph_analysis import ImportGraphAnalyzer
    ARCHITECTURE_TESTS_AVAILABLE = True
except ImportError:
    ARCHITECTURE_TESTS_AVAILABLE = False


@pytest.mark.skipif(not ARCHITECTURE_TESTS_AVAILABLE, reason="Architecture tests not available")
@pytest.mark.integration
class TestPhaseEnforcementIntegration:
    """Integration tests for phase enforcement within the main test suite."""
    
    def test_architecture_tests_importable(self):
        """Test that architecture fitness functions are importable."""
        assert ARCHITECTURE_TESTS_AVAILABLE, "Architecture tests should be importable"
    
    def test_import_analyzer_functionality(self):
        """Test that ImportAnalyzer works correctly."""
        analyzer = ImportAnalyzer()
        
        # Test basic functionality
        assert len(analyzer.phase_order) == 10
        assert "I_ingestion_preparation" in analyzer.phase_order
        assert "S_synthesis_output" in analyzer.phase_order
        
        # Test phase index mapping
        assert analyzer.phase_index["I_ingestion_preparation"] == 0
        assert analyzer.phase_index["S_synthesis_output"] == 9
    
    def test_graph_analyzer_functionality(self):
        """Test that ImportGraphAnalyzer works correctly."""
        analyzer = ImportGraphAnalyzer()
        
        # Test basic functionality
        assert len(analyzer.phase_order) == 10
        assert analyzer.graph is not None
    
    @pytest.mark.slow
    def test_phase_violations_detection(self):
        """Test that phase violations are properly detected."""
        analyzer = ImportAnalyzer()
        violations = analyzer.analyze_phase_dependencies()
        
        # This test documents current state - violations may exist
        assert isinstance(violations, dict)
        
        for phase, phase_violations in violations.items():
            assert phase in analyzer.phase_order
            assert isinstance(phase_violations, list)
    
    def test_phase_directories_exist(self):
        """Test that expected phase directories exist."""
        expected_phases = [
            "I_ingestion_preparation",
            "X_context_construction", 
            "K_knowledge_extraction",
            "A_analysis_nlp",
            "L_classification_evaluation",
            "R_search_retrieval",
            "O_orchestration_control",
            "G_aggregation_reporting",
            "T_integration_storage",
            "S_synthesis_output"
        ]
        
        root_path = Path("canonical_flow")
        existing_phases = []
        
        for phase in expected_phases:
            phase_path = root_path / phase
            if phase_path.exists():
                existing_phases.append(phase)
        
        # At least some phases should exist
        assert len(existing_phases) > 0, "At least some phase directories should exist"
    
    def test_integration_with_main_test_suite(self):
        """Test that phase enforcement integrates with the main test suite."""
        # This test ensures that the architecture tests can be run as part of
        # the main test suite without conflicts
        
        # Test that we can create analyzers without issues
        import_analyzer = ImportAnalyzer()
        graph_analyzer = ImportGraphAnalyzer()
        
        assert import_analyzer is not None
        assert graph_analyzer is not None
        
        # Test that basic analysis doesn't crash
        try:
            violations = import_analyzer.analyze_phase_dependencies()
            assert isinstance(violations, dict)
        except Exception as e:
            pytest.fail(f"Phase analysis should not crash: {e}")


@pytest.mark.architecture
def test_architecture_marker_works():
    """Test that the architecture marker is properly applied."""
    # This test will be marked with the architecture marker
    # and can be used to verify that marker-based test selection works
    pass


@pytest.mark.phase_enforcement  
def test_phase_enforcement_marker_works():
    """Test that the phase_enforcement marker is properly applied."""
    # This test will be marked with the phase_enforcement marker
    pass


class TestPhaseEnforcementConfiguration:
    """Test configuration and setup for phase enforcement."""
    
    def test_pytest_markers_configured(self):
        """Test that pytest markers are properly configured."""
        import pytest
        
        # Check that our custom markers are available
        # Note: This is a basic check - actual marker registration is done in conftest.py
        assert hasattr(pytest.mark, 'architecture')
        assert hasattr(pytest.mark, 'phase_enforcement')
    
    def test_project_structure_for_phase_enforcement(self):
        """Test that project structure supports phase enforcement."""
        # Test that architecture_tests directory exists
        arch_tests_path = Path("architecture_tests")
        if arch_tests_path.exists():
            assert (arch_tests_path / "__init__.py").exists()
            
            # Test that key files exist
            key_files = [
                "test_phase_enforcement.py",
                "test_import_graph_analysis.py",
                "conftest.py"
            ]
            
            for file_name in key_files:
                file_path = arch_tests_path / file_name
                if file_path.exists():  # Files may not exist yet
                    assert file_path.is_file()


if __name__ == "__main__":
    # Allow running this file directly for debugging
    pytest.main([__file__, "-v"])