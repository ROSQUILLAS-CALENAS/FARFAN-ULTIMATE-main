"""
Contract validation tests for meso_aggregator.py component.

This test suite validates:
- API schema compliance including input parameter validation
- Output format verification and exception handling for malformed inputs
- Deterministic behavior testing for identical outputs across multiple runs  
- JSON field ordering and numerical precision consistency
- Comprehensive edge case coverage (empty data, missing fields, oversized inputs)
- Graceful degradation without exceptions for partial failures
"""

import copy
import json
import time
import unittest
# # # from typing import Any, Dict, List  # Module not found  # Module not found  # Module not found
# # # from unittest.mock import patch, MagicMock  # Module not found  # Module not found  # Module not found

# Mock numpy and scipy modules to avoid dependencies
class MockNumpyArray:
    def __init__(self, data):
        self.data = data if isinstance(data, list) else [data]
        
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return MockNumpyArray([x + other for x in self.data])
        elif hasattr(other, 'data'):
            return MockNumpyArray([a + b for a, b in zip(self.data, other.data)])
        else:
            return MockNumpyArray([a + b for a, b in zip(self.data, other)])
        
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return MockNumpyArray([x * other for x in self.data])
        elif hasattr(other, 'data'):
            return MockNumpyArray([a * b for a, b in zip(self.data, other.data)])
        else:
            return MockNumpyArray([a * b for a, b in zip(self.data, other)])
            
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return MockNumpyArray([x / other for x in self.data])
        elif hasattr(other, 'data'):
            return MockNumpyArray([a / b for a, b in zip(self.data, other.data)])
        else:
            return MockNumpyArray([a / b for a, b in zip(self.data, other)])
        
    def __iter__(self):
        return iter(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
        
    def __len__(self):
        return len(self.data)

class MockNumPy:
    def array(self, data):
        return MockNumpyArray(data)
    
    def sum(self, data):
        if hasattr(data, 'data'):
            return sum(data.data)
        return sum(data)
    
    def std(self, data):
        if hasattr(data, '__iter__'):
            data_list = list(data)
        else:
            data_list = [data]
            
        if len(data_list) <= 1:
            return 0.0
        mean = sum(data_list) / len(data_list)
        variance = sum((x - mean) ** 2 for x in data_list) / len(data_list)
        return variance ** 0.5
    
    def log(self, data):
        import math
        if hasattr(data, 'data'):
            return MockNumpyArray([math.log(x) for x in data.data])
        elif hasattr(data, '__iter__'):
            return [math.log(x) for x in data]
        else:
            return math.log(data)

class MockSciPy:
    class spatial:
        class distance:
            @staticmethod
            def cosine(vec1, vec2):
                if not vec1 or not vec2 or len(vec1) != len(vec2):
                    return 1.0
                
                dot_product = sum(a * b for a, b in zip(vec1, vec2))
                norm_a = sum(a * a for a in vec1) ** 0.5
                norm_b = sum(b * b for b in vec2) ** 0.5
                
                if norm_a == 0 or norm_b == 0:
                    return 1.0
                
                cos_sim = dot_product / (norm_a * norm_b)
                return 1.0 - cos_sim  # Convert similarity to distance

import sys
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Mock the modules before importing
sys.modules['numpy'] = MockNumPy()
sys.modules['scipy'] = MockSciPy() 
sys.modules['scipy.spatial'] = MockSciPy.spatial
sys.modules['scipy.spatial.distance'] = MockSciPy.spatial.distance

import meso_aggregator
# # # from meso_aggregator import (  # Module not found  # Module not found  # Module not found
    process, jensen_shannon_divergence, cosine_similarity_score,
    extract_components_from_answer, build_coverage_matrix,
    calculate_cluster_divergences, DEVELOPMENT_PLAN_COMPONENTS, REQUIRED_CLUSTERS
)


class TestMesoAggregatorContract(unittest.TestCase):
    """Contract validation tests for meso_aggregator component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_input = {
            "cluster_audit": {
                "micro": {
                    "C1": {
                        "answers": [
                            {
                                "question_id": "Q1",
                                "verdict": "YES",
                                "score": 0.85,
                                "evidence_ids": ["E1", "E2"],
                                "components": ["OBJECTIVES", "STRATEGIES"]
                            },
                            {
                                "question_id": "Q2", 
                                "verdict": "NO",
                                "score": 0.45,
                                "evidence_ids": ["E3"],
                                "components": ["BUDGET", "RISKS"]
                            }
                        ]
                    },
                    "C2": {
                        "answers": [
                            {
                                "question_id": "Q1",
                                "verdict": "YES", 
                                "score": 0.78,
                                "evidence_ids": ["E4", "E5", "E6"],
                                "components": ["OBJECTIVES", "INDICATORS"]
                            },
                            {
                                "question_id": "Q2",
                                "verdict": "YES",
                                "score": 0.62,
                                "evidence_ids": ["E7"],
                                "components": ["BUDGET"]
                            }
                        ]
                    },
                    "C3": {
                        "answers": [
                            {
                                "question_id": "Q1",
                                "verdict": "YES",
                                "score": 0.90,
                                "evidence_ids": ["E8", "E9"],
                                "components": ["OBJECTIVES", "TIMELINES"]
                            }
                        ]
                    },
                    "C4": {
                        "answers": [
                            {
                                "question_id": "Q2",
                                "verdict": "NO", 
                                "score": 0.35,
                                "evidence_ids": ["E10", "E11", "E12"],
                                "components": ["STAKEHOLDERS", "COMPLIANCE"]
                            }
                        ]
                    }
                }
            }
        }
        
        self.valid_context = {
            "environment": "test",
            "request_id": "test_req_001",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
    def test_input_parameter_validation(self):
        """Test strict input parameter validation."""
        # Valid input should work
        result = process(self.valid_input, self.valid_context)
        self.assertIsInstance(result, dict)
        self.assertIn("meso_summary", result)
        self.assertIn("coverage_matrix", result)
        
        # None input should be handled gracefully
        result = process(None, self.valid_context)
        self.assertIsInstance(result, dict)
        
        # Empty dict input should be handled
        result = process({}, self.valid_context)
        self.assertIsInstance(result, dict)
        
        # Context can be None
        result = process(self.valid_input, None)
        self.assertIsInstance(result, dict)
        
    def test_output_format_verification(self):
        """Test output format verification and schema compliance."""
        result = process(self.valid_input, self.valid_context)
        
        # Check main structure
        self.assertIsInstance(result, dict)
        self.assertIn("meso_summary", result)
        self.assertIn("coverage_matrix", result)
        
        # Validate meso_summary structure
        meso_summary = result["meso_summary"]
        required_summary_fields = [
            "items", "divergence_stats", "cluster_participation", 
            "component_coverage_summary"
        ]
        for field in required_summary_fields:
            self.assertIn(field, meso_summary, f"Missing required field: {field}")
        
        # Validate coverage_matrix structure
        coverage_matrix = result["coverage_matrix"] 
        self.assertIsInstance(coverage_matrix, dict)
        
        # Each component should have required fields
        for component in DEVELOPMENT_PLAN_COMPONENTS:
            if component in coverage_matrix:
                component_data = coverage_matrix[component]
                required_fields = [
                    "clusters_evaluating", "questions_addressing", 
                    "total_evaluations", "coverage_percentage"
                ]
                for field in required_fields:
                    self.assertIn(field, component_data)
                    
                # Validate field types
                self.assertIsInstance(component_data["clusters_evaluating"], list)
                self.assertIsInstance(component_data["questions_addressing"], list)
                self.assertIsInstance(component_data["total_evaluations"], int)
                self.assertIsInstance(component_data["coverage_percentage"], (int, float))
                
                # Coverage percentage should be 0-100
                self.assertGreaterEqual(component_data["coverage_percentage"], 0.0)
                self.assertLessEqual(component_data["coverage_percentage"], 100.0)
                
    def test_deterministic_behavior(self):
        """Test deterministic behavior with identical outputs across multiple runs."""
        # Run process function multiple times with same input
        results = []
        for _ in range(5):
            result = process(copy.deepcopy(self.valid_input), copy.deepcopy(self.valid_context))
            results.append(result)
            
        # Compare all results should be identical
        first_result = results[0]
        for i, result in enumerate(results[1:], 1):
            # Compare meso_summary structure
            self.assertEqual(
                result["meso_summary"]["cluster_participation"],
                first_result["meso_summary"]["cluster_participation"],
                f"Cluster participation differs in run {i}"
            )
            
            # Compare coverage matrix
            self.assertEqual(
                result["coverage_matrix"].keys(),
                first_result["coverage_matrix"].keys(),
                f"Coverage matrix keys differ in run {i}"
            )
            
            # Compare numerical values with precision
            for component in result["coverage_matrix"]:
                self.assertAlmostEqual(
                    result["coverage_matrix"][component]["coverage_percentage"],
                    first_result["coverage_matrix"][component]["coverage_percentage"],
                    places=10,
                    msg=f"Coverage percentage precision differs for {component} in run {i}"
                )
                
    def test_json_serialization_consistency(self):
        """Test JSON field ordering and serialization consistency."""
        result = process(self.valid_input, self.valid_context)
        
        # Serialize to JSON multiple times
        json_strings = []
        for _ in range(3):
            json_str = json.dumps(result, sort_keys=True, separators=(',', ':'))
            json_strings.append(json_str)
            
        # All JSON strings should be identical
        first_json = json_strings[0]
        for i, json_str in enumerate(json_strings[1:], 1):
            self.assertEqual(json_str, first_json, f"JSON serialization differs in attempt {i}")
            
        # Deserialize and verify structure preserved
        deserialized = json.loads(first_json)
        self.assertEqual(deserialized.keys(), result.keys())
        
    def test_numerical_precision_consistency(self):
        """Test numerical precision consistency across operations."""
        # Test with high precision numbers
        precise_input = copy.deepcopy(self.valid_input)
        precise_input["cluster_audit"]["micro"]["C1"]["answers"][0]["score"] = 0.123456789012345
        precise_input["cluster_audit"]["micro"]["C2"]["answers"][0]["score"] = 0.987654321098765
        
        result = process(precise_input, self.valid_context)
        
        # Check that precision is maintained in calculations
        items = result["meso_summary"]["items"]
        for question_id, item in items.items():
            if "score_summary" in item and item["score_summary"]["count"] > 0:
                # Verify scores maintain reasonable precision
                score_avg = item["score_summary"]["avg"]
                self.assertIsInstance(score_avg, float)
                # Should maintain at least 6 decimal places in most cases
                self.assertGreater(len(str(score_avg).split('.')[-1]), 3)
                
    def test_empty_input_data(self):
        """Test handling of empty input data."""
        empty_inputs = [
            {},
            {"cluster_audit": {}},
            {"cluster_audit": {"micro": {}}},
            {"cluster_audit": {"micro": {"C1": {}}}},
            {"cluster_audit": {"micro": {"C1": {"answers": []}}}},
        ]
        
        for empty_input in empty_inputs:
            with self.subTest(empty_input=empty_input):
                result = process(empty_input, self.valid_context)
                
                # Should return valid structure even with empty input
                self.assertIsInstance(result, dict)
                self.assertIn("meso_summary", result)
                self.assertIn("coverage_matrix", result)
                
                # Coverage matrix should still have all components
                coverage_matrix = result["coverage_matrix"]
                for component in DEVELOPMENT_PLAN_COMPONENTS:
                    self.assertIn(component, coverage_matrix)
                    self.assertEqual(coverage_matrix[component]["total_evaluations"], 0)
                    self.assertEqual(coverage_matrix[component]["coverage_percentage"], 0.0)
                    
    def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        # Test missing score field
        invalid_input = copy.deepcopy(self.valid_input)
        del invalid_input["cluster_audit"]["micro"]["C1"]["answers"][0]["score"]
        
        result = process(invalid_input, self.valid_context)
        self.assertIsInstance(result, dict)  # Should not crash
        
        # Test missing verdict field
        invalid_input2 = copy.deepcopy(self.valid_input)
        del invalid_input2["cluster_audit"]["micro"]["C1"]["answers"][0]["verdict"]
        
        result2 = process(invalid_input2, self.valid_context)
        self.assertIsInstance(result2, dict)  # Should not crash
        
        # Test missing evidence_ids field
        invalid_input3 = copy.deepcopy(self.valid_input)
        del invalid_input3["cluster_audit"]["micro"]["C1"]["answers"][0]["evidence_ids"]
        
        result3 = process(invalid_input3, self.valid_context)
        self.assertIsInstance(result3, dict)  # Should not crash
        
    def test_oversized_inputs(self):
        """Test handling of oversized inputs."""
        # Create input with many clusters
        oversized_input = copy.deepcopy(self.valid_input)
        micro = oversized_input["cluster_audit"]["micro"]
        
        # Add many clusters
        for i in range(100):
            cluster_id = f"C{i+10}"
            micro[cluster_id] = {
                "answers": [
                    {
                        "question_id": f"Q{i}",
                        "verdict": "YES" if i % 2 == 0 else "NO",
                        "score": 0.5 + (i % 50) / 100.0,
                        "evidence_ids": [f"E{i}_{j}" for j in range(10)],
                        "components": [DEVELOPMENT_PLAN_COMPONENTS[i % len(DEVELOPMENT_PLAN_COMPONENTS)]]
                    }
                ]
            }
            
        # Should handle large input without crashing
        result = process(oversized_input, self.valid_context)
        self.assertIsInstance(result, dict)
        self.assertIn("meso_summary", result)
        self.assertIn("coverage_matrix", result)
        
        # Check that all clusters were processed
        cluster_participation = result["meso_summary"]["cluster_participation"]
        processed_clusters = sum(count for count in cluster_participation.values())
        self.assertGreater(processed_clusters, 4)  # Should process more than the original clusters
        
    def test_partial_failure_conditions(self):
        """Test graceful degradation for partial failure conditions."""
        # Test with invalid answer structure
        partial_failure_input = copy.deepcopy(self.valid_input)
        micro = partial_failure_input["cluster_audit"]["micro"]
        
        # Add invalid answer entries
        micro["C1"]["answers"].append("invalid_answer_string")
        micro["C1"]["answers"].append(None)
        micro["C1"]["answers"].append({"malformed": "no required fields"})
        
        # Add cluster with invalid structure
        micro["INVALID_CLUSTER"] = "not_a_dict"
        micro["C_PARTIAL"] = {"answers": "not_a_list"}
        
        result = process(partial_failure_input, self.valid_context)
        
        # Should not crash and return valid structure
        self.assertIsInstance(result, dict)
        self.assertIn("meso_summary", result)
        self.assertIn("coverage_matrix", result)
        
        # Should process valid answers and skip invalid ones
        items = result["meso_summary"]["items"]
        self.assertGreater(len(items), 0)  # Should have processed some valid answers
        
    def test_malformed_cluster_data(self):
        """Test exception handling for malformed cluster data."""
        malformed_inputs = [
            # Invalid cluster_audit structure
            {"cluster_audit": "not_a_dict"},
            {"cluster_audit": None},
            {"cluster_audit": {"micro": "not_a_dict"}},
            {"cluster_audit": {"micro": None}},
        ]
        
        for malformed_input in malformed_inputs:
            with self.subTest(malformed_input=malformed_input):
                # Should not raise exception
                try:
                    result = process(malformed_input, self.valid_context)
                    self.assertIsInstance(result, dict)
                except Exception as e:
                    self.fail(f"process() raised exception with malformed input: {e}")
                    
    def test_invalid_score_values(self):
        """Test handling of invalid score values."""
        invalid_score_input = copy.deepcopy(self.valid_input)
        answers = invalid_score_input["cluster_audit"]["micro"]["C1"]["answers"]
        
        # Add answers with invalid scores
        answers.append({
            "question_id": "Q_invalid_score",
            "verdict": "YES",
            "score": "not_a_number",
            "evidence_ids": ["E_test"],
            "components": ["OBJECTIVES"]
        })
        
        answers.append({
            "question_id": "Q_negative_score", 
            "verdict": "YES",
            "score": -0.5,
            "evidence_ids": ["E_test2"],
            "components": ["STRATEGIES"]
        })
        
        answers.append({
            "question_id": "Q_huge_score",
            "verdict": "YES", 
            "score": 999.99,
            "evidence_ids": ["E_test3"],
            "components": ["BUDGET"]
        })
        
        result = process(invalid_score_input, self.valid_context)
        
        # Should handle gracefully without crashing
        self.assertIsInstance(result, dict)
        self.assertIn("meso_summary", result)
        
    def test_component_extraction_robustness(self):
        """Test robustness of component extraction."""
        # Test extract_components_from_answer with various inputs
        test_cases = [
            {},  # Empty dict
            {"components": []},  # Empty components
            {"components": "SINGLE_COMPONENT"},  # String component
            {"components": ["OBJECTIVES", "invalid_component", "STRATEGIES"]},  # Mixed valid/invalid
            {"verdict": "This mentions OBJECTIVES and BUDGET explicitly"},  # Text references
            {"question": "What about TIMELINES and STAKEHOLDERS?"},  # Question text
            {"components": None},  # None components
            {"components": 123},  # Invalid type
        ]
        
        for test_case in test_cases:
            with self.subTest(test_case=test_case):
                try:
                    components = extract_components_from_answer(test_case)
                    self.assertIsInstance(components, set)
                    # All components should be valid or empty
                    for comp in components:
                        self.assertIsInstance(comp, str)
                except Exception as e:
                    self.fail(f"extract_components_from_answer failed with input {test_case}: {e}")


class TestMesoAggregatorMathFunctions(unittest.TestCase):
    """Test mathematical functions used in meso aggregator."""
    
    def test_jensen_shannon_divergence(self):
        """Test Jensen-Shannon divergence calculation."""
        # Test identical distributions
        p = [0.5, 0.3, 0.2]
        q = [0.5, 0.3, 0.2]
        result = jensen_shannon_divergence(p, q)
        self.assertAlmostEqual(result, 0.0, places=10)
        
        # Test different distributions
        p = [1.0, 0.0, 0.0]
        q = [0.0, 1.0, 0.0]
        result = jensen_shannon_divergence(p, q)
        self.assertGreater(result, 0.0)
        
        # Test edge cases
        self.assertEqual(jensen_shannon_divergence([], []), 0.0)
        self.assertEqual(jensen_shannon_divergence([1.0], [1.0, 0.0]), 0.0)
        
    def test_cosine_similarity_score(self):
        """Test cosine similarity score calculation."""
        # Test identical vectors
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.0, 2.0, 3.0]
        result = cosine_similarity_score(vec1, vec2)
        self.assertAlmostEqual(result, 1.0, places=10)
        
        # Test orthogonal vectors
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        result = cosine_similarity_score(vec1, vec2)
        self.assertAlmostEqual(result, 0.0, places=10)
        
        # Test edge cases
        self.assertEqual(cosine_similarity_score([], []), 0.0)
        self.assertEqual(cosine_similarity_score([1.0], [1.0, 2.0]), 0.0)
        
    def test_mathematical_precision(self):
        """Test mathematical precision in calculations."""
        # Test with small numbers that could cause precision issues
        small_p = [1e-10, 1e-10, 1.0 - 2e-10]
        small_q = [1e-11, 1e-11, 1.0 - 2e-11]
        
        result = jensen_shannon_divergence(small_p, small_q)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        
        # Test cosine similarity with small vectors
        small_vec1 = [1e-10, 1e-10, 1e-10]
        small_vec2 = [1e-11, 1e-11, 1e-11]
        
        result = cosine_similarity_score(small_vec1, small_vec2)
        self.assertIsInstance(result, float)
        # Should be close to 1.0 for proportional small vectors
        self.assertAlmostEqual(result, 1.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)