#!/usr/bin/env python3
"""
Test Suite for Phase Layering Enforcement System

This module contains comprehensive tests to validate the phase layering
enforcement mechanisms.
"""

import unittest
from pathlib import Path
import tempfile
import shutil
import sys
import os

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from architecture_fitness_functions import PhaseLayeringValidator, DependencyViolation


class TestPhaseLayeringValidator(unittest.TestCase):
    """Test the PhaseLayeringValidator class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory structure for testing
        self.test_dir = Path(tempfile.mkdtemp())
        self.canonical_flow_dir = self.test_dir / "canonical_flow"
        
        # Create phase directories
        for phase in PhaseLayeringValidator.CANONICAL_PHASES:
            phase_dir = self.canonical_flow_dir / phase
            phase_dir.mkdir(parents=True)
            
            # Create __init__.py
            (phase_dir / "__init__.py").write_text("")
            
            # Create a sample module
            (phase_dir / "sample_module.py").write_text(f'"""Sample module for {phase}."""\n')
        
        self.validator = PhaseLayeringValidator(self.test_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_phase_extraction_from_path(self):
        """Test extracting phase name from file paths."""
        # Test valid phase path
        test_path = self.canonical_flow_dir / "I_ingestion_preparation" / "test.py"
        phase = self.validator.get_phase_from_path(test_path)
        self.assertEqual(phase, "I_ingestion_preparation")
        
        # Test invalid path
        invalid_path = self.test_dir / "other" / "test.py"
        phase = self.validator.get_phase_from_path(invalid_path)
        self.assertIsNone(phase)
    
    def test_backward_dependency_detection(self):
        """Test detection of backward dependencies."""
        # Test backward dependency (later phase importing earlier)
        self.assertTrue(
            self.validator.is_backward_dependency(
                "A_analysis_nlp", 
                "I_ingestion_preparation"
            )
        )
        
        # Test forward dependency (earlier phase importing later) - should be False
        self.assertFalse(
            self.validator.is_backward_dependency(
                "I_ingestion_preparation", 
                "A_analysis_nlp"
            )
        )
        
        # Test same phase - should be False
        self.assertFalse(
            self.validator.is_backward_dependency(
                "A_analysis_nlp", 
                "A_analysis_nlp"
            )
        )
    
    def test_import_extraction(self):
        """Test extraction of import statements from Python files."""
        # Create a test file with various import types
        test_file = self.canonical_flow_dir / "test_imports.py"
        test_content = '''
import os
from pathlib import Path
from canonical_flow.I_ingestion_preparation import advanced_loader
from canonical_flow.A_analysis_nlp.question_analyzer import QuestionAnalyzer
import canonical_flow.K_knowledge_extraction.embedding_generator
'''
        test_file.write_text(test_content)
        
        imports = self.validator.extract_imports_from_file(test_file)
        
        # Check that canonical_flow imports are detected
        canonical_imports = [imp for imp, _ in imports if imp.startswith("canonical_flow")]
        self.assertEqual(len(canonical_imports), 3)
        
        expected_imports = {
            "canonical_flow.I_ingestion_preparation",
            "canonical_flow.A_analysis_nlp.question_analyzer", 
            "canonical_flow.K_knowledge_extraction.embedding_generator"
        }
        
        actual_imports = {imp for imp, _ in imports if imp.startswith("canonical_flow")}
        self.assertEqual(actual_imports, expected_imports)
    
    def test_violation_detection_with_backward_imports(self):
        """Test detection of violations with backward imports."""
        # Create a file in A_analysis_nlp that imports from I_ingestion_preparation
        violating_file = (
            self.canonical_flow_dir / "A_analysis_nlp" / "violating_module.py"
        )
        violating_content = '''
from canonical_flow.I_ingestion_preparation import advanced_loader
from canonical_flow.X_context_construction import context_adapter
'''
        violating_file.write_text(violating_content)
        
        violations = self.validator.validate_file_dependencies(violating_file)
        
        # Should detect 2 backward dependencies
        self.assertEqual(len(violations), 2)
        
        # Check violation details
        violation_targets = {v.target_phase for v in violations}
        expected_targets = {"I_ingestion_preparation", "X_context_construction"}
        self.assertEqual(violation_targets, expected_targets)
    
    def test_no_violations_with_forward_imports(self):
        """Test that forward imports don't generate violations."""
        # Create a file in I_ingestion_preparation that imports from later phases
        valid_file = (
            self.canonical_flow_dir / "I_ingestion_preparation" / "valid_module.py"
        )
        valid_content = '''
from canonical_flow.A_analysis_nlp import question_analyzer
from canonical_flow.L_classification_evaluation import score_calculator
'''
        valid_file.write_text(valid_content)
        
        violations = self.validator.validate_file_dependencies(valid_file)
        
        # Should detect no violations (forward dependencies are allowed)
        self.assertEqual(len(violations), 0)
    
    def test_canonical_phase_sequence(self):
        """Test that the canonical phase sequence is correct."""
        expected_sequence = [
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
        
        self.assertEqual(
            PhaseLayeringValidator.CANONICAL_PHASES, 
            expected_sequence
        )
    
    def test_phase_indices(self):
        """Test that phase indices are correctly calculated."""
        validator = PhaseLayeringValidator()
        
        # Check that earlier phases have lower indices
        self.assertLess(
            validator.phase_indices["I_ingestion_preparation"],
            validator.phase_indices["S_synthesis_output"]
        )
        
        self.assertLess(
            validator.phase_indices["A_analysis_nlp"],
            validator.phase_indices["O_orchestration_control"]
        )
    
    def test_full_validation_with_mixed_violations(self):
        """Test full validation with a mix of valid and invalid imports."""
        # Create files with various import patterns
        
        # Valid file (forward imports only)
        valid_file = (
            self.canonical_flow_dir / "I_ingestion_preparation" / "valid.py"
        )
        valid_file.write_text('''
from canonical_flow.A_analysis_nlp import question_analyzer
import canonical_flow.L_classification_evaluation.score_calculator
''')
        
        # Violating file (backward imports)
        violating_file = (
            self.canonical_flow_dir / "A_analysis_nlp" / "violating.py"  
        )
        violating_file.write_text('''
from canonical_flow.I_ingestion_preparation import advanced_loader
from canonical_flow.X_context_construction import context_adapter
from canonical_flow.L_classification_evaluation import score_calculator  # This is valid (forward)
''')
        
        # Another violating file
        another_violating_file = (
            self.canonical_flow_dir / "O_orchestration_control" / "bad_imports.py"
        )
        another_violating_file.write_text('''
from canonical_flow.K_knowledge_extraction import embedding_generator
from canonical_flow.R_search_retrieval import hybrid_retrieval  # This is valid (forward)
''')
        
        result = self.validator.validate_architecture()
        
        # Should detect violations but not fail completely
        self.assertFalse(result.passed)  # Has violations
        self.assertGreater(result.backward_dependencies, 0)  # Has backward dependencies
        self.assertGreater(result.total_files_scanned, 0)
        
        # Check that violations are properly categorized
        source_phases = {v.source_phase for v in result.violations}
        expected_source_phases = {"A_analysis_nlp", "O_orchestration_control"}
        self.assertEqual(source_phases, expected_source_phases)


class TestArchitectureFitnessFunctions(unittest.TestCase):
    """Test the overall architecture fitness functions."""
    
    def test_dependency_violation_dataclass(self):
        """Test the DependencyViolation dataclass."""
        violation = DependencyViolation(
            source_phase="A_analysis_nlp",
            target_phase="I_ingestion_preparation", 
            source_file="test.py",
            target_file="canonical_flow.I_ingestion_preparation",
            line_number=10,
            import_statement="from canonical_flow.I_ingestion_preparation import test"
        )
        
        self.assertEqual(violation.source_phase, "A_analysis_nlp")
        self.assertEqual(violation.target_phase, "I_ingestion_preparation")
        self.assertEqual(violation.severity, "ERROR")  # Default value
    
    def test_real_project_structure(self):
        """Test validation against the actual project structure."""
        # This test uses the real canonical_flow directory if it exists
        project_root = Path(__file__).parent
        canonical_flow_path = project_root / "canonical_flow"
        
        if not canonical_flow_path.exists():
            self.skipTest("canonical_flow directory not found in project")
        
        validator = PhaseLayeringValidator(project_root)
        result = validator.validate_architecture()
        
        # Basic sanity checks
        self.assertIsInstance(result.passed, bool)
        self.assertIsInstance(result.violations, list)
        self.assertGreaterEqual(result.total_files_scanned, 0)
        self.assertGreater(result.execution_time_ms, 0)
        
        # Print results for manual inspection
        print(f"\n--- Real Project Validation Results ---")
        print(f"Files scanned: {result.total_files_scanned}")
        print(f"Violations found: {len(result.violations)}")
        print(f"Validation passed: {result.passed}")
        
        if result.violations:
            print("Sample violations:")
            for i, violation in enumerate(result.violations[:3]):
                print(f"  {i+1}. {violation.source_phase} → {violation.target_phase}")


def run_tests():
    """Run all tests and return success status."""
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestPhaseLayeringValidator))
    test_suite.addTest(unittest.makeSuite(TestArchitectureFitnessFunctions))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)