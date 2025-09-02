#!/usr/bin/env python3
"""
Runner for L-stage test suite without pytest dependency.
Executes all three test modules: preflight, assertions, and determinism.
"""

import sys
import json
import hashlib
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from copy import deepcopy

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "canonical_flow"))

# Import L-stage components
from canonical_flow.A_analysis_nlp.question_analyzer import (
    DecalogoQuestionRegistry,
    get_decalogo_question_registry
)
from canonical_flow.L_classification_evaluation.decalogo_scoring_system import (
    ScoringSystem,
    QuestionResponse,
    DimensionScore,
    PointScore
)


class TestResult:
    """Simple test result tracking."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.passed = True
        self.error = None
    
    def fail(self, error_msg: str):
        self.passed = False
        self.error = error_msg
    
    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        if self.error:
            return f"{self.test_name}: {status} - {self.error}"
        return f"{self.test_name}: {status}"


class LStageTestRunner:
    """Test runner for L-stage validation suite."""
    
    def __init__(self):
        self.results = []
    
    def run_test(self, test_name: str, test_func):
        """Run a single test and record result."""
        result = TestResult(test_name)
        try:
            test_func()
            print(f"✓ {test_name}")
        except AssertionError as e:
            result.fail(f"Assertion failed: {e}")
            print(f"✗ {test_name}: {e}")
        except Exception as e:
            result.fail(f"Error: {e}")
            print(f"✗ {test_name}: {e}")
        
        self.results.append(result)
        return result.passed
    
    def run_preflight_tests(self):
        """Run preflight validation tests."""
        print("\n" + "=" * 60)
        print("RUNNING PREFLIGHT TESTS")
        print("=" * 60)
        
        registry = get_decalogo_question_registry()
        scoring_system = ScoringSystem(precision=4)
        
        # Test 1: Registry checksum verification
        def test_registry_checksum():
            questions = registry.get_all_questions()
            question_data = []
            for q in questions:
                question_data.append({
                    "question_id": q.question_id,
                    "point_number": q.point_number,
                    "dimension_code": q.dimension_code,
                    "question_text": q.question_text
                })
            question_data.sort(key=lambda x: x["question_id"])
            
            registry_json = json.dumps(question_data, sort_keys=True, ensure_ascii=False)
            checksum = hashlib.sha256(registry_json.encode('utf-8')).hexdigest()
            
            registry2 = get_decalogo_question_registry()
            questions2 = registry2.get_all_questions()
            question_data2 = []
            for q in questions2:
                question_data2.append({
                    "question_id": q.question_id,
                    "point_number": q.point_number,
                    "dimension_code": q.dimension_code,
                    "question_text": q.question_text
                })
            question_data2.sort(key=lambda x: x["question_id"])
            registry_json2 = json.dumps(question_data2, sort_keys=True, ensure_ascii=False)
            checksum2 = hashlib.sha256(registry_json2.encode('utf-8')).hexdigest()
            
            assert checksum == checksum2, "Registry checksum must be deterministic"
            assert len(checksum) == 64, "SHA256 checksum must be 64 characters"
        
        # Test 2: 470-question count validation
        def test_470_question_count():
            questions = registry.get_all_questions()
            assert len(questions) == 470, f"Expected exactly 470 questions, found {len(questions)}"
            
            for point_num in range(1, 11):
                point_questions = registry.get_questions_by_point(point_num)
                assert len(point_questions) == 47, f"Point {point_num} must have exactly 47 questions"
            
            question_ids = [q.question_id for q in questions]
            assert len(question_ids) == len(set(question_ids)), "All question IDs must be unique"
        
        # Test 3: Decálogo points distribution  
        def test_decalogo_distribution():
            questions = registry.get_all_questions()
            point_counts = {}
            for q in questions:
                point_counts[q.point_number] = point_counts.get(q.point_number, 0) + 1
            
            expected_points = set(range(1, 11))
            actual_points = set(point_counts.keys())
            assert actual_points == expected_points, f"Point mismatch: expected {expected_points}, got {actual_points}"
            
            for point_num, count in point_counts.items():
                assert count == 47, f"Point {point_num} has {count} questions, expected 47"
        
        # Test 4: Dimension distribution
        def test_dimension_distribution():
            expected_dimensions = {"DE1", "DE2", "DE3", "DE4", "GEN"}
            questions = registry.get_all_questions()
            
            for point_num in range(1, 11):
                point_questions = registry.get_questions_by_point(point_num)
                point_dim_counts = {}
                
                for q in point_questions:
                    dim = q.dimension_code
                    point_dim_counts[dim] = point_dim_counts.get(dim, 0) + 1
                
                for dim in ["DE1", "DE2", "DE3", "DE4"]:
                    actual_count = point_dim_counts.get(dim, 0)
                    assert actual_count == 11, f"Point {point_num} dimension {dim}: expected 11, got {actual_count}"
                
                gen_count = point_dim_counts.get("GEN", 0)
                assert gen_count == 3, f"Point {point_num} expected 3 GEN questions, got {gen_count}"
        
        # Test 5: Scoring system configuration
        def test_scoring_configuration():
            system_info = scoring_system.get_system_info()
            
            expected_base_scores = {"Sí": 1.0, "Parcial": 0.5, "No": 0.0, "NI": 0.0}
            assert system_info["base_scores"] == expected_base_scores, "Base scores mismatch"
            
            expected_weights = {"DE-1": 0.30, "DE-2": 0.25, "DE-3": 0.25, "DE-4": 0.20}
            assert system_info["decalogo_weights"] == expected_weights, "Decálogo weights mismatch"
            
            assert system_info["evidence_multiplier_range"] == [0.5, 1.2], "Evidence multiplier range invalid"
            assert system_info["deterministic"] is True, "Scoring must be deterministic"
        
        # Run preflight tests
        tests = [
            ("Registry checksum verification", test_registry_checksum),
            ("470-question count validation", test_470_question_count),
            ("Decálogo points distribution", test_decalogo_distribution),
            ("Dimension distribution validation", test_dimension_distribution),
            ("Scoring system configuration", test_scoring_configuration)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            if self.run_test(test_name, test_func):
                passed += 1
        
        print(f"\nPreflight tests: {passed}/{len(tests)} passed")
        return passed == len(tests)
    
    def run_assertion_tests(self):
        """Run post-execution assertion tests."""
        print("\n" + "=" * 60)
        print("RUNNING ASSERTION TESTS")
        print("=" * 60)
        
        scoring_system = ScoringSystem(precision=4)
        
        # Sample evaluation data
        sample_evaluation_data = {
            "DE-1": [
                {
                    "question_id": f"DQ_01_DE1_{i:08d}",
                    "response": "Sí",
                    "evidence_completeness": 0.9,
                    "page_reference_quality": 0.8
                }
                for i in range(11)
            ],
            "DE-2": [
                {
                    "question_id": f"DQ_01_DE2_{i:08d}",
                    "response": "Parcial",
                    "evidence_completeness": 0.7,
                    "page_reference_quality": 0.6
                }
                for i in range(11)
            ],
            "DE-3": [
                {
                    "question_id": f"DQ_01_DE3_{i:08d}",
                    "response": "No",
                    "evidence_completeness": 0.5,
                    "page_reference_quality": 0.4
                }
                for i in range(11)
            ],
            "DE-4": [
                {
                    "question_id": f"DQ_01_DE4_{i:08d}",
                    "response": "NI",
                    "evidence_completeness": 0.3,
                    "page_reference_quality": 0.2
                }
                for i in range(11)
            ]
        }
        
        # Test 1: Coverage completeness validation
        def test_coverage_completeness():
            point_score = scoring_system.process_point_evaluation(1, sample_evaluation_data)
            assert point_score.total_questions == 44, f"Expected 44 questions, got {point_score.total_questions}"
            
            expected_dimensions = {"DE-1", "DE-2", "DE-3", "DE-4"}
            actual_dimensions = {dim_score.dimension_id for dim_score in point_score.dimension_scores}
            assert actual_dimensions == expected_dimensions, f"Missing dimensions: {expected_dimensions - actual_dimensions}"
            
            for dim_score in point_score.dimension_scores:
                assert dim_score.total_questions == 11, f"Dimension {dim_score.dimension_id} expected 11 questions"
        
        # Test 2: Evidence quality thresholds
        def test_evidence_quality_thresholds():
            test_cases = [
                (1.0, 1.0, 1.15, 1.2),  # Perfect evidence (adjusted range)
                (0.0, 0.0, 0.5, 0.5),   # No evidence
                (0.7, 0.8, 0.95, 1.15)  # Good evidence (adjusted range)
            ]
            
            for completeness, reference_quality, min_expected, max_expected in test_cases:
                multiplier = scoring_system.calculate_evidence_multiplier(completeness, reference_quality)
                
                assert 0.5 <= float(multiplier) <= 1.2, f"Multiplier {multiplier} outside bounds [0.5, 1.2]"
                assert min_expected <= float(multiplier) <= max_expected, f"Multiplier {multiplier} outside expected range [{min_expected}, {max_expected}] for inputs ({completeness}, {reference_quality})"
        
        # Test 3: Score bounds verification
        def test_score_bounds():
            responses = ["Sí", "Parcial", "No", "NI"]
            evidence_qualities = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
            
            for response in responses:
                for completeness, reference_quality in evidence_qualities:
                    base_score, multiplier, final_score = scoring_system.calculate_final_score(
                        response, completeness, reference_quality
                    )
                    
                    assert 0.0 <= float(base_score) <= 1.0, f"Base score {base_score} outside bounds [0, 1]"
                    assert 0.5 <= float(multiplier) <= 1.2, f"Multiplier {multiplier} outside bounds [0.5, 1.2]"
                    assert 0.0 <= float(final_score) <= 1.2, f"Final score {final_score} outside bounds [0, 1.2]"
        
        # Test 4: Guard rule satisfaction
        def test_guard_rules():
            point_score = scoring_system.process_point_evaluation(1, sample_evaluation_data)
            
            # Guard rule 1: All dimension scores must be >= 0
            for dim_score in point_score.dimension_scores:
                assert dim_score.weighted_average >= 0.0, f"Dimension {dim_score.dimension_id} score negative"
            
            # Guard rule 2: Final point score within bounds
            assert 0.0 <= point_score.final_score <= 1.2, f"Point score {point_score.final_score} outside bounds"
            
            # Guard rule 3: Evidence scores within bounds
            for dim_score in point_score.dimension_scores:
                for question_response in dim_score.question_responses:
                    assert 0.0 <= question_response.evidence_completeness <= 1.0, "Evidence completeness outside bounds"
                    assert 0.0 <= question_response.page_reference_quality <= 1.0, "Page reference quality outside bounds"
            
            # Guard rule 4: Dimension weights sum to 1.0
            total_weight = sum(scoring_system.DECALOGO_WEIGHTS.values())
            assert abs(float(total_weight) - 1.0) < 1e-10, f"Weights sum to {total_weight}, expected 1.0"
        
        # Run assertion tests
        tests = [
            ("Coverage completeness validation", test_coverage_completeness),
            ("Evidence quality thresholds", test_evidence_quality_thresholds),
            ("Score bounds verification", test_score_bounds),
            ("Guard rule satisfaction", test_guard_rules)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            if self.run_test(test_name, test_func):
                passed += 1
        
        print(f"\nAssertion tests: {passed}/{len(tests)} passed")
        return passed == len(tests)
    
    def run_determinism_tests(self):
        """Run determinism verification tests."""
        print("\n" + "=" * 60)
        print("RUNNING DETERMINISM TESTS") 
        print("=" * 60)
        
        scoring_system = ScoringSystem(precision=4)
        
        # Deterministic evaluation data
        deterministic_evaluation_data = {
            "DE-1": [
                {
                    "question_id": f"DQ_01_DE1_{i:08d}",
                    "response": "Sí" if i % 3 == 0 else "Parcial" if i % 3 == 1 else "No",
                    "evidence_completeness": round(0.5 + (i * 0.1) % 0.5, 3),
                    "page_reference_quality": round(0.4 + (i * 0.2) % 0.6, 3)
                }
                for i in range(11)
            ],
            "DE-2": [
                {
                    "question_id": f"DQ_01_DE2_{i:08d}",
                    "response": "Parcial" if i % 2 == 0 else "No",
                    "evidence_completeness": round(0.6 + (i * 0.05) % 0.4, 3),
                    "page_reference_quality": round(0.3 + (i * 0.15) % 0.7, 3)
                }
                for i in range(11)
            ],
            "DE-3": [
                {
                    "question_id": f"DQ_01_DE3_{i:08d}",
                    "response": "No" if i % 4 == 0 else "NI" if i % 4 == 1 else "Sí",
                    "evidence_completeness": round(0.7 + (i * 0.03) % 0.3, 3),
                    "page_reference_quality": round(0.5 + (i * 0.25) % 0.5, 3)
                }
                for i in range(11)
            ],
            "DE-4": [
                {
                    "question_id": f"DQ_01_DE4_{i:08d}",
                    "response": "Sí" if i < 3 else "Parcial" if i < 7 else "No",
                    "evidence_completeness": round(0.8 + (i * 0.02) % 0.2, 3),
                    "page_reference_quality": round(0.6 + (i * 0.12) % 0.4, 3)
                }
                for i in range(11)
            ]
        }
        
        # Test 1: Identical document processing determinism
        def test_identical_processing():
            result1 = scoring_system.process_point_evaluation(1, deterministic_evaluation_data)
            result2 = scoring_system.process_point_evaluation(1, deepcopy(deterministic_evaluation_data))
            
            assert result1.point_id == result2.point_id, "Point IDs must be identical"
            assert result1.final_score == result2.final_score, f"Final scores differ: {result1.final_score} vs {result2.final_score}"
            assert result1.total_questions == result2.total_questions, "Question counts must be identical"
            
            assert len(result1.dimension_scores) == len(result2.dimension_scores), "Dimension counts must match"
            
            for dim1, dim2 in zip(result1.dimension_scores, result2.dimension_scores):
                assert dim1.dimension_id == dim2.dimension_id, f"Dimension IDs differ: {dim1.dimension_id} vs {dim2.dimension_id}"
                assert dim1.weighted_average == dim2.weighted_average, f"Dimension averages differ"
        
        # Test 2: JSON output consistency
        def test_json_consistency():
            outputs = []
            for run in range(3):
                result = scoring_system.process_point_evaluation(1, deepcopy(deterministic_evaluation_data))
                
                output_dict = {
                    "point_id": result.point_id,
                    "final_score": result.final_score,
                    "total_questions": result.total_questions,
                    "dimension_scores": [
                        {
                            "dimension_id": dim.dimension_id,
                            "weighted_average": dim.weighted_average,
                            "total_questions": dim.total_questions
                        }
                        for dim in sorted(result.dimension_scores, key=lambda x: x.dimension_id)
                    ]
                }
                
                json_output = json.dumps(output_dict, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
                outputs.append(json_output)
            
            for i in range(1, len(outputs)):
                assert outputs[i] == outputs[0], f"JSON output run {i} differs from run 0"
        
        # Test 3: Registry determinism across instances
        def test_registry_determinism():
            registries = [get_decalogo_question_registry() for _ in range(3)]
            
            registry_data = []
            for registry in registries:
                questions = registry.get_all_questions()
                data = [(q.question_id, q.point_number, q.dimension_code) for q in questions]
                registry_data.append(sorted(data))
            
            for i in range(1, len(registry_data)):
                assert registry_data[i] == registry_data[0], f"Registry instance {i} differs from instance 0"
            
            for i, data in enumerate(registry_data):
                assert len(data) == 470, f"Registry instance {i} has {len(data)} questions, expected 470"
        
        # Test 4: Scoring precision determinism
        def test_scoring_precision():
            test_cases = [
                ("Sí", 0.8, 0.7),
                ("Parcial", 0.5, 0.5),
                ("No", 1.0, 1.0),
            ]
            
            for response, completeness, reference_quality in test_cases:
                results = []
                for _ in range(5):
                    base_score, multiplier, final_score = scoring_system.calculate_final_score(
                        response, completeness, reference_quality
                    )
                    results.append((float(base_score), float(multiplier), float(final_score)))
                
                for i in range(1, len(results)):
                    assert results[i] == results[0], f"Precision determinism failed for {response}"
        
        # Run determinism tests
        tests = [
            ("Identical document processing determinism", test_identical_processing),
            ("JSON output consistency", test_json_consistency),
            ("Registry determinism across instances", test_registry_determinism),
            ("Scoring precision determinism", test_scoring_precision)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            if self.run_test(test_name, test_func):
                passed += 1
        
        print(f"\nDeterminism tests: {passed}/{len(tests)} passed")
        return passed == len(tests)
    
    def run_all_tests(self):
        """Run all test suites."""
        print("L-STAGE CI VALIDATION SUITE")
        print("=" * 60)
        
        preflight_passed = self.run_preflight_tests()
        assertion_passed = self.run_assertion_tests() 
        determinism_passed = self.run_determinism_tests()
        
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        
        print(f"Total tests run: {total_tests}")
        print(f"Tests passed: {passed_tests}")
        print(f"Tests failed: {total_tests - passed_tests}")
        
        if passed_tests == total_tests:
            print("\n✓ ALL TESTS PASSED")
            print("L-stage CI validation suite is READY FOR PRODUCTION")
        else:
            print("\n✗ SOME TESTS FAILED")
            print("Failed tests:")
            for result in self.results:
                if not result.passed:
                    print(f"  - {result}")
        
        return passed_tests == total_tests


def main():
    """Main entry point."""
    runner = LStageTestRunner()
    success = runner.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())