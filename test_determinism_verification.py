#!/usr/bin/env python3
"""
Test Cases for Determinism Verification System
=============================================
Comprehensive test suite that validates the determinism verification mechanisms
and ensures they correctly detect both deterministic and non-deterministic behavior.

Test Coverage:
- Verification system initialization and configuration
- Hash consistency detection across multiple runs
- Component ordering stability validation  
- Confidence scoring reproducibility testing
- Tie-breaking logic consistency verification
- Error handling and edge cases
- Report generation and serialization
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, MagicMock

# Import the verification system
from determinism_verification_system import (
    DeterminismVerifier, 
    DeterminismTestResult,
    VerificationReport
)


class TestDeterminismVerifier(unittest.TestCase):
    """Test the core DeterminismVerifier functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.verifier = DeterminismVerifier()
        self.test_data = {
            "test_id": "unit_test",
            "components": [
                {"name": "comp_a", "phase": "I", "confidence": 0.8},
                {"name": "comp_b", "phase": "A", "confidence": 0.9},
            ],
            "metadata": {"version": "1.0"}
        }
    
    def test_initialization(self):
        """Test verifier initialization and environment setup"""
        self.assertIsInstance(self.verifier, DeterminismVerifier)
        self.assertEqual(self.verifier.deterministic_seed, 42)
        self.assertEqual(os.environ.get('DETERMINISTIC_MODE'), '1')
        self.assertEqual(os.environ.get('PYTHONHASHSEED'), '42')
    
    def test_stable_hash_computation(self):
        """Test stable hash computation for various data types"""
        # Test dictionary hashing
        dict_data = {"b": 2, "a": 1, "c": 3}
        hash1 = self.verifier.compute_stable_hash(dict_data)
        hash2 = self.verifier.compute_stable_hash(dict_data)
        self.assertEqual(hash1, hash2, "Dictionary hashes should be identical")
        
        # Test that key order doesn't matter
        dict_data_reordered = {"c": 3, "a": 1, "b": 2}
        hash3 = self.verifier.compute_stable_hash(dict_data_reordered)
        self.assertEqual(hash1, hash3, "Hash should be independent of key order")
        
        # Test list hashing
        list_data = [1, 2, 3]
        hash4 = self.verifier.compute_stable_hash(list_data)
        hash5 = self.verifier.compute_stable_hash(list_data)
        self.assertEqual(hash4, hash5, "List hashes should be identical")
        
        # Test that different data produces different hashes
        self.assertNotEqual(hash1, hash4, "Different data should produce different hashes")
    
    def test_test_data_generation(self):
        """Test deterministic test data generation"""
        data1 = self.verifier.generate_test_data()
        data2 = self.verifier.generate_test_data()
        
        # Should be identical
        self.assertEqual(data1, data2, "Generated test data should be identical")
        
        # Should have required fields
        self.assertIn("test_id", data1)
        self.assertIn("timestamp", data1)
        self.assertIn("components", data1)
        self.assertIn("evidence", data1)
        self.assertIn("metadata", data1)
        
        # Timestamp should be deterministic
        self.assertEqual(data1["timestamp"], 1640995200.0)
    
    def test_hash_consistency_test(self):
        """Test the hash consistency verification"""
        result = self.verifier.test_hash_consistency(runs=3)
        
        self.assertIsInstance(result, DeterminismTestResult)
        self.assertEqual(result.test_name, "hash_consistency")
        self.assertTrue(result.success, "Hash consistency test should pass")
        self.assertTrue(result.hash_matches, "Hashes should match")
        self.assertEqual(result.consistency_score, 1.0)
        self.assertGreater(result.execution_time_ms, 0)
        
        # Check details
        self.assertIn("hashes", result.details)
        self.assertIn("unique_count", result.details)
        self.assertEqual(result.details["unique_count"], 1)
    
    def test_component_ordering_test(self):
        """Test component ordering consistency"""
        result = self.verifier.test_component_ordering(runs=3)
        
        self.assertIsInstance(result, DeterminismTestResult)
        self.assertEqual(result.test_name, "component_ordering")
        self.assertTrue(result.success, "Component ordering test should pass")
        self.assertTrue(result.ordering_stable, "Component ordering should be stable")
        self.assertEqual(result.consistency_score, 1.0)
        
        # Check that all orderings are identical
        orderings = result.details["orderings"]
        reference = result.details["reference"]
        for ordering in orderings:
            self.assertEqual(ordering, reference, "All orderings should be identical")
    
    def test_confidence_scoring_test(self):
        """Test confidence scoring stability"""
        result = self.verifier.test_confidence_scoring(runs=3)
        
        self.assertIsInstance(result, DeterminismTestResult)
        self.assertEqual(result.test_name, "confidence_scoring")
        self.assertTrue(result.success, "Confidence scoring test should pass")
        self.assertLess(result.confidence_variance, 0.001, "Confidence variance should be very low")
        self.assertEqual(result.consistency_score, 1.0)
        
        # Check that all confidence sets are identical
        confidence_sets = result.details["confidence_sets"]
        reference = confidence_sets[0]
        for conf_set in confidence_sets:
            self.assertEqual(conf_set, reference, "All confidence sets should be identical")
    
    def test_tie_breaking_consistency_test(self):
        """Test tie-breaking logic consistency"""
        result = self.verifier.test_tie_breaking_consistency(runs=3)
        
        self.assertIsInstance(result, DeterminismTestResult)
        self.assertEqual(result.test_name, "tie_breaking_consistency")
        self.assertTrue(result.success, "Tie-breaking consistency test should pass")
        self.assertTrue(result.ordering_stable, "Tie-breaking should be stable")
        self.assertEqual(result.consistency_score, 1.0)
        
        # Check that ordering is correct (delta first due to higher score, then alphabetical)
        reference = result.details["reference"]
        self.assertEqual(reference[0], "item_d", "Highest score item should be first")
        # Among tied items, should be alphabetical by name
        tied_items = reference[1:]  # Items with score 0.85
        expected_tied = ["item_a", "item_b", "item_c"]  # alpha, beta, gamma alphabetically
        self.assertEqual(tied_items, expected_tied, "Tied items should be in alphabetical order")
    
    def test_phase_priority_mapping(self):
        """Test phase priority assignment"""
        # Test known phases
        self.assertEqual(self.verifier._get_phase_priority("I"), 1)
        self.assertEqual(self.verifier._get_phase_priority("A"), 4)
        self.assertEqual(self.verifier._get_phase_priority("L"), 5)
        self.assertEqual(self.verifier._get_phase_priority("S"), 10)
        
        # Test unknown phase
        self.assertEqual(self.verifier._get_phase_priority("Z"), 99)
    
    def test_comprehensive_verification(self):
        """Test the comprehensive verification suite"""
        # Mock canonical auditor to avoid dependency issues
        with patch('determinism_verification_system.HAS_CANONICAL_AUDITOR', True):
            with patch('determinism_verification_system.coa') as mock_coa:
                mock_coa.process.return_value = {
                    "canonical_audit": {"status": "complete", "hash": "test123"}
                }
                
                report = self.verifier.run_comprehensive_verification(runs_per_test=2)
                
                self.assertIsInstance(report, VerificationReport)
                self.assertGreater(report.total_tests, 0)
                self.assertGreaterEqual(report.passed_tests, 0)
                self.assertGreaterEqual(report.failed_tests, 0)
                self.assertEqual(report.total_tests, report.passed_tests + report.failed_tests)
                self.assertIsInstance(report.test_results, list)
                self.assertIsInstance(report.system_info, dict)
                self.assertIsInstance(report.recommendations, list)
    
    def test_error_handling(self):
        """Test error handling in verification tests"""
        # Test with corrupted verifier
        corrupted_verifier = DeterminismVerifier()
        
        # Mock a method to raise an exception
        with patch.object(corrupted_verifier, 'compute_stable_hash', side_effect=Exception("Test error")):
            result = corrupted_verifier.test_hash_consistency(runs=2)
            
            self.assertFalse(result.success)
            self.assertEqual(result.consistency_score, 0.0)
            self.assertIsNotNone(result.error_message)
            self.assertIn("Test error", result.error_message)
    
    def test_report_serialization(self):
        """Test verification report serialization and deserialization"""
        # Create a mock report
        test_result = DeterminismTestResult(
            test_name="test_case",
            success=True,
            consistency_score=0.95,
            hash_matches=True,
            ordering_stable=True,
            confidence_variance=0.001,
            execution_time_ms=100.5,
            details={"test": "data"}
        )
        
        report = VerificationReport(
            timestamp=1640995200.0,
            total_tests=1,
            passed_tests=1,
            failed_tests=0,
            overall_success=True,
            test_results=[test_result],
            system_info={"python": "3.8"},
            recommendations=["All tests passed"]
        )
        
        # Test serialization
        report_dict = report.to_dict()
        self.assertIsInstance(report_dict, dict)
        self.assertIn("timestamp", report_dict)
        self.assertIn("test_results", report_dict)
        self.assertEqual(len(report_dict["test_results"]), 1)
        
        # Test JSON serialization
        json_str = json.dumps(report_dict, indent=2)
        self.assertIsInstance(json_str, str)
        
        # Test deserialization
        parsed_dict = json.loads(json_str)
        self.assertEqual(parsed_dict["total_tests"], 1)
        self.assertEqual(parsed_dict["passed_tests"], 1)
        self.assertTrue(parsed_dict["overall_success"])
    
    def test_save_verification_report(self):
        """Test saving verification report to file"""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock report
            test_result = DeterminismTestResult(
                test_name="save_test",
                success=True,
                consistency_score=1.0,
                hash_matches=True,
                ordering_stable=True,
                confidence_variance=0.0,
                execution_time_ms=50.0
            )
            
            report = VerificationReport(
                timestamp=1640995200.0,
                total_tests=1,
                passed_tests=1,
                failed_tests=0,
                overall_success=True,
                test_results=[test_result],
                system_info={"test": True},
                recommendations=["Test passed"]
            )
            
            # Save report
            output_path = temp_path / "test_report.json"
            saved_path = self.verifier.save_verification_report(report, output_path)
            
            # Verify file was created
            self.assertTrue(saved_path.exists())
            self.assertEqual(saved_path, output_path)
            
            # Verify file contents
            with open(saved_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            self.assertEqual(loaded_data["total_tests"], 1)
            self.assertEqual(loaded_data["passed_tests"], 1)
            self.assertTrue(loaded_data["overall_success"])


class TestDeterminismTestResult(unittest.TestCase):
    """Test DeterminismTestResult data class"""
    
    def test_creation_and_serialization(self):
        """Test creating and serializing test results"""
        result = DeterminismTestResult(
            test_name="test_creation",
            success=True,
            consistency_score=0.98,
            hash_matches=True,
            ordering_stable=True,
            confidence_variance=0.002,
            execution_time_ms=123.45,
            error_message=None,
            details={"extra": "data"}
        )
        
        # Test attributes
        self.assertEqual(result.test_name, "test_creation")
        self.assertTrue(result.success)
        self.assertEqual(result.consistency_score, 0.98)
        self.assertTrue(result.hash_matches)
        self.assertTrue(result.ordering_stable)
        self.assertEqual(result.confidence_variance, 0.002)
        self.assertEqual(result.execution_time_ms, 123.45)
        self.assertIsNone(result.error_message)
        self.assertEqual(result.details["extra"], "data")
        
        # Test serialization
        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict["test_name"], "test_creation")
        self.assertTrue(result_dict["success"])
        self.assertEqual(result_dict["consistency_score"], 0.98)
    
    def test_failure_result(self):
        """Test creating failure result with error message"""
        result = DeterminismTestResult(
            test_name="test_failure",
            success=False,
            consistency_score=0.0,
            hash_matches=False,
            ordering_stable=False,
            confidence_variance=float('inf'),
            execution_time_ms=0.0,
            error_message="Test failed due to inconsistency"
        )
        
        self.assertFalse(result.success)
        self.assertEqual(result.consistency_score, 0.0)
        self.assertFalse(result.hash_matches)
        self.assertFalse(result.ordering_stable)
        self.assertEqual(result.confidence_variance, float('inf'))
        self.assertEqual(result.error_message, "Test failed due to inconsistency")


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for realistic scenarios"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.verifier = DeterminismVerifier()
    
    def test_identical_repository_states(self):
        """Test that identical repository states produce identical results"""
        # Simulate identical repository state by using fixed test data
        data1 = self.verifier.generate_test_data()
        data2 = self.verifier.generate_test_data()
        
        hash1 = self.verifier.compute_stable_hash(data1)
        hash2 = self.verifier.compute_stable_hash(data2)
        
        self.assertEqual(hash1, hash2, 
                        "Identical repository states should produce identical hashes")
    
    def test_different_repository_states(self):
        """Test that different repository states produce different results"""
        data1 = self.verifier.generate_test_data()
        data2 = data1.copy()
        data2["metadata"] = data2["metadata"].copy()  # Deep copy the metadata
        data2["metadata"]["version"] = "2.0.0"  # Change version
        
        hash1 = self.verifier.compute_stable_hash(data1)
        hash2 = self.verifier.compute_stable_hash(data2)
        
        self.assertNotEqual(hash1, hash2,
                           "Different repository states should produce different hashes")
    
    def test_consecutive_runs_stability(self):
        """Test stability across consecutive runs"""
        # Run the same test multiple times
        results = []
        for i in range(5):
            # Reset environment for each run to simulate fresh execution
            self.verifier._setup_deterministic_environment()
            result = self.verifier.test_hash_consistency(runs=2)
            results.append(result.success)
        
        # All runs should succeed
        self.assertTrue(all(results), "All consecutive runs should succeed")
        
        # All runs should have same outcome
        unique_outcomes = set(results)
        self.assertEqual(len(unique_outcomes), 1, "All runs should have identical outcomes")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def setUp(self):
        """Set up edge case test environment"""
        self.verifier = DeterminismVerifier()
    
    def test_empty_data_handling(self):
        """Test handling of empty or minimal data"""
        empty_data = {}
        hash1 = self.verifier.compute_stable_hash(empty_data)
        hash2 = self.verifier.compute_stable_hash(empty_data)
        
        self.assertEqual(hash1, hash2, "Empty data should produce consistent hashes")
    
    def test_large_data_handling(self):
        """Test handling of large data structures"""
        large_data = {
            "components": [{"id": i, "data": f"component_{i}" * 100} for i in range(1000)],
            "metadata": {"size": "large", "items": list(range(1000))}
        }
        
        hash1 = self.verifier.compute_stable_hash(large_data)
        hash2 = self.verifier.compute_stable_hash(large_data)
        
        self.assertEqual(hash1, hash2, "Large data should produce consistent hashes")
    
    def test_unicode_data_handling(self):
        """Test handling of unicode and special characters"""
        unicode_data = {
            "text": "Test with unicode: é, ñ, 中文, 🔍, математика",
            "special_chars": "!@#$%^&*()[]{}|\\:;\"'<>?,./",
            "numbers": [1.23456789, -0.0, float('inf'), float('-inf')]
        }
        
        hash1 = self.verifier.compute_stable_hash(unicode_data)
        hash2 = self.verifier.compute_stable_hash(unicode_data)
        
        self.assertEqual(hash1, hash2, "Unicode data should produce consistent hashes")
    
    def test_nested_data_structures(self):
        """Test deeply nested data structures"""
        nested_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": ["deep", "nested", "data"]
                        }
                    }
                }
            }
        }
        
        hash1 = self.verifier.compute_stable_hash(nested_data)
        hash2 = self.verifier.compute_stable_hash(nested_data)
        
        self.assertEqual(hash1, hash2, "Nested data should produce consistent hashes")


if __name__ == "__main__":
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)