#!/usr/bin/env python3
"""
Comprehensive unit tests for AuditLogger functionality.
"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from canonical_flow.evaluation.audit_logger import (
    AuditLogger, ComponentTrace, DecalogoPointTrace,
    CoverageValidationResult, ScoringConsistencyResult
)


class TestComponentTrace(unittest.TestCase):
    """Test ComponentTrace dataclass functionality."""
    
    def test_component_trace_creation(self):
        """Test creating ComponentTrace instance."""
        trace = ComponentTrace(
            component_name="test_component",
            entry_timestamp=time.time(),
            inputs={"test": "input"}
        )
        
        self.assertEqual(trace.component_name, "test_component")
        self.assertEqual(trace.status, "running")
        self.assertEqual(trace.inputs, {"test": "input"})
        self.assertIsNone(trace.exit_timestamp)
    
    def test_component_trace_completion(self):
        """Test completing a component trace."""
        start_time = time.time()
        trace = ComponentTrace(
            component_name="test_component",
            entry_timestamp=start_time
        )
        
        time.sleep(0.01)  # Small delay
        trace.complete("completed", {"result": "success"})
        
        self.assertEqual(trace.status, "completed")
        self.assertEqual(trace.outputs, {"result": "success"})
        self.assertIsNotNone(trace.exit_timestamp)
        self.assertIsNotNone(trace.duration_ms)
        self.assertGreater(trace.duration_ms, 0)


class TestCoverageValidationResult(unittest.TestCase):
    """Test CoverageValidationResult functionality."""
    
    def test_excellent_coverage(self):
        """Test excellent coverage calculation."""
        result = CoverageValidationResult(
            dimension="DE1",
            expected_question_count=10,
            actual_question_count=10
        )
        
        self.assertEqual(result.coverage_percentage, 100.0)
        self.assertEqual(result.validation_status, "excellent")
    
    def test_poor_coverage(self):
        """Test poor coverage calculation."""
        result = CoverageValidationResult(
            dimension="DE1", 
            expected_question_count=10,
            actual_question_count=3
        )
        
        self.assertEqual(result.coverage_percentage, 30.0)
        self.assertEqual(result.validation_status, "poor")
    
    def test_zero_expected_questions(self):
        """Test handling of zero expected questions."""
        result = CoverageValidationResult(
            dimension="DE1",
            expected_question_count=0,
            actual_question_count=5
        )
        
        self.assertEqual(result.coverage_percentage, 0.0)
        self.assertEqual(result.validation_status, "poor")


class TestScoringConsistencyResult(unittest.TestCase):
    """Test ScoringConsistencyResult functionality."""
    
    def test_consistent_scoring(self):
        """Test consistent scoring calculation."""
        result = ScoringConsistencyResult(
            dimension="DE1",
            raw_scores=[0.8, 0.8, 0.8],
            final_classification="good",
            expected_classification="good"
        )
        
        self.assertEqual(result.consistency_status, "consistent")
        self.assertLess(result.score_variance, 0.01)
    
    def test_inconsistent_scoring(self):
        """Test inconsistent scoring calculation."""
        result = ScoringConsistencyResult(
            dimension="DE1",
            raw_scores=[0.2, 0.8, 0.9],
            final_classification="poor",
            expected_classification="excellent"
        )
        
        self.assertEqual(result.consistency_status, "inconsistent")
        self.assertGreater(result.score_variance, 0.01)


class TestAuditLogger(unittest.TestCase):
    """Test AuditLogger main functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.audit_file = Path(self.temp_dir) / "test_audit.json"
        self.audit_logger = AuditLogger(str(self.audit_file))
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.audit_file.exists():
            self.audit_file.unlink()
    
    def test_session_id_generation(self):
        """Test unique session ID generation."""
        logger1 = AuditLogger()
        logger2 = AuditLogger()
        
        self.assertNotEqual(logger1.session_id, logger2.session_id)
        self.assertEqual(len(logger1.session_id), 16)
    
    def test_start_point_evaluation(self):
        """Test starting point evaluation."""
        eval_id = self.audit_logger.start_point_evaluation(1, "Test Point")
        
        self.assertIn(eval_id, self.audit_logger.decalogo_traces)
        self.assertEqual(self.audit_logger.metrics["total_points_evaluated"], 1)
        
        trace = self.audit_logger.decalogo_traces[eval_id]
        self.assertEqual(trace.point_id, 1)
        self.assertEqual(trace.point_name, "Test Point")
        self.assertEqual(trace.status, "running")
    
    def test_component_logging(self):
        """Test component entry and exit logging."""
        eval_id = self.audit_logger.start_point_evaluation(1, "Test Point")
        
        # Test component entry
        trace_id = self.audit_logger.log_component_entry(
            "test_component", 
            {"input": "test"},
            eval_id
        )
        
        self.assertIn(trace_id, self.audit_logger.decalogo_traces[eval_id].components)
        
        # Test component exit
        self.audit_logger.log_component_exit(
            "test_component",
            {"output": "result"}, 
            evaluation_id=eval_id
        )
        
        component_trace = self.audit_logger.decalogo_traces[eval_id].components[trace_id]
        self.assertEqual(component_trace.status, "completed")
        self.assertEqual(component_trace.outputs, {"output": "result"})
        self.assertIsNotNone(component_trace.duration_ms)
    
    def test_question_processing_logging(self):
        """Test question processing logging."""
        eval_id = self.audit_logger.start_point_evaluation(1, "Test Point")
        
        self.audit_logger.log_question_processing(
            "Q1", 150.5, "Test question?", eval_id
        )
        
        trace = self.audit_logger.decalogo_traces[eval_id]
        self.assertIn("Q1", trace.question_processing)
        self.assertEqual(trace.question_processing["Q1"], 150.5)
        self.assertEqual(self.audit_logger.metrics["total_questions_processed"], 1)
    
    def test_evidence_validation_logging(self):
        """Test evidence validation logging."""
        eval_id = self.audit_logger.start_point_evaluation(1, "Test Point")
        
        self.audit_logger.log_evidence_validation(
            "E1", 200.0, ["p.1", "p.2"], "completed", eval_id
        )
        
        trace = self.audit_logger.decalogo_traces[eval_id]
        self.assertIn("E1", trace.evidence_validation)
        self.assertEqual(trace.evidence_validation["E1"], 200.0)
        self.assertEqual(self.audit_logger.metrics["total_evidence_validated"], 1)
    
    def test_scoring_computation_logging(self):
        """Test scoring computation logging."""
        eval_id = self.audit_logger.start_point_evaluation(1, "Test Point")
        
        self.audit_logger.log_scoring_computation(
            "DE1", [0.8, 0.9], [0.4, 0.45], [0.5, 0.5], 0.85, "good", eval_id
        )
        
        trace = self.audit_logger.decalogo_traces[eval_id]
        self.assertIn("DE1", trace.scoring_computation)
        
        scoring_data = trace.scoring_computation["DE1"]
        self.assertEqual(scoring_data["raw_scores"], [0.8, 0.9])
        self.assertEqual(scoring_data["classification"], "good")
    
    def test_coverage_validation(self):
        """Test coverage validation functionality."""
        expected_questions = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        actual_questions = ["Q1", "Q2", "Q4"]
        
        result = self.audit_logger.validate_coverage(
            "DE1", expected_questions, actual_questions
        )
        
        self.assertEqual(result.expected_question_count, 5)
        self.assertEqual(result.actual_question_count, 3)
        self.assertEqual(result.coverage_percentage, 60.0)
        self.assertEqual(result.validation_status, "fair")
        self.assertEqual(sorted(result.missing_question_ids), ["Q3", "Q5"])
    
    def test_scoring_consistency_validation(self):
        """Test scoring consistency validation."""
        # First log some scoring data
        eval_id = self.audit_logger.start_point_evaluation(1, "Test Point")
        self.audit_logger.log_scoring_computation(
            "DE1", [0.8, 0.8, 0.8], [0.4, 0.4, 0.4], [0.5, 0.5, 0.5], 0.8, "good", eval_id
        )
        
        # Then validate consistency
        result = self.audit_logger.validate_scoring_consistency("DE1", "good")
        
        self.assertEqual(result.final_classification, "good") 
        self.assertEqual(result.expected_classification, "good")
        self.assertEqual(result.consistency_status, "consistent")
        self.assertLess(result.score_variance, 0.01)
    
    def test_complete_point_evaluation(self):
        """Test completing point evaluation."""
        eval_id = self.audit_logger.start_point_evaluation(1, "Test Point")
        
        self.audit_logger.complete_point_evaluation(eval_id)
        
        trace = self.audit_logger.decalogo_traces[eval_id]
        self.assertEqual(trace.status, "completed")
        self.assertIsNotNone(trace.end_timestamp)
        self.assertIsNotNone(trace.total_duration_ms)
    
    def test_generate_audit_report(self):
        """Test audit report generation."""
        # Create some test data
        eval_id = self.audit_logger.start_point_evaluation(1, "Test Point")
        self.audit_logger.log_question_processing("Q1", 100.0)
        self.audit_logger.log_evidence_validation("E1", 150.0, ["p.1"])
        self.audit_logger.complete_point_evaluation(eval_id)
        
        # Generate report
        report = self.audit_logger.generate_audit_report()
        
        # Verify report structure
        required_keys = [
            "session_id", "audit_timestamp", "decalogo_traces",
            "coverage_validation", "scoring_consistency", "metrics", 
            "summary", "recommendations"
        ]
        
        for key in required_keys:
            self.assertIn(key, report)
        
        # Verify metrics
        self.assertEqual(report["metrics"]["total_points_evaluated"], 1)
        self.assertEqual(report["metrics"]["total_questions_processed"], 1)
        self.assertEqual(report["metrics"]["total_evidence_validated"], 1)
    
    def test_save_and_load_audit(self):
        """Test saving and loading audit reports."""
        # Create test data
        eval_id = self.audit_logger.start_point_evaluation(1, "Test Point")
        self.audit_logger.complete_point_evaluation(eval_id)
        
        # Save audit
        saved_path = self.audit_logger.save_audit()
        self.assertEqual(saved_path, str(self.audit_file))
        self.assertTrue(self.audit_file.exists())
        
        # Load audit
        loaded_data = self.audit_logger.load_audit()
        self.assertEqual(loaded_data["session_id"], self.audit_logger.session_id)
        
        # Verify JSON structure
        with open(self.audit_file, 'r') as f:
            json_data = json.load(f)
        
        self.assertIsInstance(json_data, dict)
        self.assertIn("decalogo_traces", json_data)
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test component exit without entry
        self.audit_logger.log_component_exit("nonexistent_component")
        
        # Test logging without active evaluation
        self.audit_logger.log_question_processing("Q1", 100.0)
        self.audit_logger.log_evidence_validation("E1", 150.0)
        
        # Test loading nonexistent file
        nonexistent_logger = AuditLogger("nonexistent_file.json")
        loaded_data = nonexistent_logger.load_audit()
        self.assertEqual(loaded_data, {})
    
    def test_recommendations_generation(self):
        """Test recommendations generation based on findings."""
        # Create poor coverage scenario
        self.audit_logger.validate_coverage("DE1", ["Q1", "Q2", "Q3", "Q4", "Q5"], ["Q1"])
        
        # Create inconsistent scoring scenario
        eval_id = self.audit_logger.start_point_evaluation(1, "Test Point")
        self.audit_logger.log_scoring_computation(
            "DE2", [0.2, 0.9], [0.1, 0.45], [0.5, 0.5], 0.55, "fair", eval_id
        )
        self.audit_logger.validate_scoring_consistency("DE2", "excellent")
        
        report = self.audit_logger.generate_audit_report()
        recommendations = report["recommendations"]
        
        self.assertGreater(len(recommendations), 0)
        
        # Check for coverage recommendation
        coverage_recs = [r for r in recommendations if r["type"] == "coverage_improvement"]
        self.assertGreater(len(coverage_recs), 0)
        
        # Check for consistency recommendation
        consistency_recs = [r for r in recommendations if r["type"] == "scoring_consistency"]
        self.assertGreater(len(consistency_recs), 0)


class TestGlobalAuditFunctions(unittest.TestCase):
    """Test global audit logging functions."""
    
    def test_global_audit_logger(self):
        """Test global audit logger functionality."""
        from canonical_flow.evaluation.audit_logger import (
            get_audit_logger, start_point_evaluation, 
            log_component_entry, log_component_exit, save_audit_report
        )
        
        # Test getting global logger
        logger1 = get_audit_logger()
        logger2 = get_audit_logger()
        self.assertIs(logger1, logger2)  # Should be same instance
        
        # Test global functions
        eval_id = start_point_evaluation(1, "Global Test")
        self.assertIsInstance(eval_id, str)
        
        trace_id = log_component_entry("global_component", {"test": "input"})
        self.assertIsInstance(trace_id, str)
        
        log_component_exit("global_component", {"test": "output"})
        
        # Test saving report
        audit_path = save_audit_report()
        self.assertTrue(Path(audit_path).exists())
        
        # Cleanup
        Path(audit_path).unlink()


if __name__ == "__main__":
    unittest.main(verbosity=2)