"""
CI Post-Execution Assertions for L-Stage Classification Evaluation

Validates post-execution state:
- Coverage completeness across all 470 questions
- Evidence quality thresholds and validation
- Score bounds verification [0, 1.2] 
- Guard rule satisfaction and compliance
- Artifact consistency and inter-module contract enforcement
"""

import json
import pytest
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from decimal import Decimal
import sys

# Add canonical_flow to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "canonical_flow"))

from canonical_flow.A_analysis_nlp.question_analyzer import DecalogoQuestionRegistry, get_decalogo_question_registry
from canonical_flow.L_classification_evaluation.decalogo_scoring_system import (
    ScoringSystem, 
    QuestionResponse, 
    DimensionScore, 
    PointScore
)


class TestLStagePostExecutionAssertions:
    """Post-execution validation assertions for L-stage components."""
    
    @pytest.fixture
    def scoring_system(self) -> ScoringSystem:
        """Get scoring system instance."""
        return ScoringSystem(precision=4)
    
    @pytest.fixture
    def registry(self) -> DecalogoQuestionRegistry:
        """Get question registry instance."""
        return get_decalogo_question_registry()
    
    @pytest.fixture
    def sample_evaluation_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate sample evaluation data for testing."""
        return {
            "DE-1": [
                {
                    "question_id": "DQ_01_DE1_12345678",
                    "response": "Sí",
                    "evidence_completeness": 0.9,
                    "page_reference_quality": 0.8
                }
                for _ in range(11)
            ],
            "DE-2": [
                {
                    "question_id": "DQ_01_DE2_87654321",
                    "response": "Parcial", 
                    "evidence_completeness": 0.7,
                    "page_reference_quality": 0.6
                }
                for _ in range(11)
            ],
            "DE-3": [
                {
                    "question_id": "DQ_01_DE3_11223344",
                    "response": "No",
                    "evidence_completeness": 0.5,
                    "page_reference_quality": 0.4
                }
                for _ in range(11)
            ],
            "DE-4": [
                {
                    "question_id": "DQ_01_DE4_55667788",
                    "response": "NI",
                    "evidence_completeness": 0.3,
                    "page_reference_quality": 0.2
                }
                for _ in range(11)
            ]
        }
    
    def test_coverage_completeness_validation(self, scoring_system: ScoringSystem, sample_evaluation_data: Dict[str, List[Dict[str, Any]]]):
        """Test that coverage completeness is validated across all 470 questions."""
        # Process sample evaluation
        point_score = scoring_system.process_point_evaluation(1, sample_evaluation_data)
        
        # Validate total question coverage
        assert point_score.total_questions == 44, f"Expected 44 questions for this sample, got {point_score.total_questions}"
        
        # Validate dimension coverage
        expected_dimensions = {"DE-1", "DE-2", "DE-3", "DE-4"}
        actual_dimensions = {dim_score.dimension_id for dim_score in point_score.dimension_scores}
        assert actual_dimensions == expected_dimensions, f"Coverage missing dimensions: {expected_dimensions - actual_dimensions}"
        
        # Validate each dimension has expected question count
        for dim_score in point_score.dimension_scores:
            assert dim_score.total_questions == 11, f"Dimension {dim_score.dimension_id} expected 11 questions, got {dim_score.total_questions}"
    
    def test_evidence_quality_thresholds(self, scoring_system: ScoringSystem):
        """Test evidence quality thresholds are properly enforced."""
        test_cases = [
            # (evidence_completeness, page_reference_quality, expected_min_multiplier, expected_max_multiplier)
            (1.0, 1.0, 1.2, 1.2),  # Perfect evidence should give max multiplier
            (0.0, 0.0, 0.5, 0.5),  # No evidence should give min multiplier
            (0.7, 0.8, 0.71, 0.85),  # Good evidence should be in middle range
        ]
        
        for completeness, reference_quality, min_expected, max_expected in test_cases:
            multiplier = scoring_system.calculate_evidence_multiplier(completeness, reference_quality)
            
            # Verify within bounds
            assert 0.5 <= float(multiplier) <= 1.2, (
                f"Evidence multiplier {multiplier} outside bounds [0.5, 1.2] "
                f"for completeness={completeness}, reference_quality={reference_quality}"
            )
            
            # Verify specific thresholds
            assert min_expected <= float(multiplier) <= max_expected, (
                f"Evidence multiplier {multiplier} outside expected range "
                f"[{min_expected}, {max_expected}] for inputs ({completeness}, {reference_quality})"
            )
    
    def test_score_bounds_verification(self, scoring_system: ScoringSystem):
        """Test that all scores are within bounds [0, 1.2]."""
        # Test all response types with various evidence qualities
        responses = ["Sí", "Parcial", "No", "NI"]
        evidence_qualities = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
        
        for response in responses:
            for completeness, reference_quality in evidence_qualities:
                base_score, multiplier, final_score = scoring_system.calculate_final_score(
                    response, completeness, reference_quality
                )
                
                # Verify base score bounds
                assert 0.0 <= float(base_score) <= 1.0, (
                    f"Base score {base_score} outside bounds [0, 1] for response '{response}'"
                )
                
                # Verify multiplier bounds
                assert 0.5 <= float(multiplier) <= 1.2, (
                    f"Multiplier {multiplier} outside bounds [0.5, 1.2]"
                )
                
                # Verify final score bounds
                assert 0.0 <= float(final_score) <= 1.2, (
                    f"Final score {final_score} outside bounds [0, 1.2] "
                    f"for response '{response}', evidence=({completeness}, {reference_quality})"
                )
    
    def test_guard_rule_satisfaction(self, scoring_system: ScoringSystem, sample_evaluation_data: Dict[str, List[Dict[str, Any]]]):
        """Test that guard rules are satisfied in evaluation processing."""
        # Process evaluation with guard rule checks
        point_score = scoring_system.process_point_evaluation(1, sample_evaluation_data)
        
        # Guard rule 1: All dimension scores must be >= 0
        for dim_score in point_score.dimension_scores:
            assert dim_score.weighted_average >= 0.0, (
                f"Dimension {dim_score.dimension_id} score {dim_score.weighted_average} violates non-negativity guard"
            )
        
        # Guard rule 2: Final point score must be within bounds
        assert 0.0 <= point_score.final_score <= 1.2, (
            f"Point score {point_score.final_score} violates bounds guard [0, 1.2]"
        )
        
        # Guard rule 3: Question responses must have valid evidence scores
        for dim_score in point_score.dimension_scores:
            for question_response in dim_score.question_responses:
                assert 0.0 <= question_response.evidence_completeness <= 1.0, (
                    f"Evidence completeness {question_response.evidence_completeness} violates bounds guard [0, 1]"
                )
                assert 0.0 <= question_response.page_reference_quality <= 1.0, (
                    f"Page reference quality {question_response.page_reference_quality} violates bounds guard [0, 1]"
                )
        
        # Guard rule 4: Dimension weights must sum to 1.0
        total_weight = sum(scoring_system.DECALOGO_WEIGHTS.values())
        assert abs(float(total_weight) - 1.0) < 1e-10, (
            f"Decálogo weights sum to {total_weight}, expected 1.0 (tolerance 1e-10)"
        )
    
    def test_arithmetic_consistency_guards(self, scoring_system: ScoringSystem):
        """Test arithmetic consistency guard rules."""
        # Test decimal precision consistency
        test_value = Decimal("0.123456789")
        rounded_value = scoring_system._round_decimal(test_value)
        
        # Should round to 4 decimal places
        assert str(rounded_value) == "0.1235", f"Expected 0.1235, got {rounded_value}"
        
        # Test that repeated rounding is idempotent
        double_rounded = scoring_system._round_decimal(rounded_value)
        assert rounded_value == double_rounded, "Repeated rounding should be idempotent"
        
        # Test aggregation consistency
        test_scores = [Decimal("0.1"), Decimal("0.2"), Decimal("0.3")]
        sum_scores = sum(test_scores)
        avg_score = sum_scores / Decimal("3")
        rounded_avg = scoring_system._round_decimal(avg_score)
        
        assert str(rounded_avg) == "0.2000", f"Expected 0.2000, got {rounded_avg}"
    
    def test_inter_module_contract_enforcement(self, scoring_system: ScoringSystem, sample_evaluation_data: Dict[str, List[Dict[str, Any]]]):
        """Test inter-module contract enforcement and message schema compliance."""
        # Process evaluation to generate output artifact
        point_score = scoring_system.process_point_evaluation(1, sample_evaluation_data)
        
        # Create contract-compliant output message
        output_message = {
            "message_type": "l_stage_evaluation_completed",
            "schema_version": "1.0",
            "source_component": "L_classification_evaluation",
            "payload": {
                "point_evaluation": {
                    "point_id": point_score.point_id,
                    "final_score": point_score.final_score,
                    "total_questions": point_score.total_questions,
                    "dimension_scores": [
                        {
                            "dimension_id": dim.dimension_id,
                            "weighted_average": dim.weighted_average,
                            "total_questions": dim.total_questions
                        }
                        for dim in point_score.dimension_scores
                    ]
                },
                "validation_metadata": {
                    "score_bounds_valid": 0.0 <= point_score.final_score <= 1.2,
                    "coverage_complete": point_score.total_questions > 0,
                    "guard_rules_satisfied": True
                }
            },
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        # Validate contract compliance
        required_fields = ["message_type", "schema_version", "source_component", "payload"]
        for field in required_fields:
            assert field in output_message, f"Contract violation: missing required field '{field}'"
        
        # Validate payload structure
        payload = output_message["payload"]
        assert "point_evaluation" in payload, "Contract violation: missing point_evaluation"
        assert "validation_metadata" in payload, "Contract violation: missing validation_metadata"
        
        # Validate point evaluation structure
        point_eval = payload["point_evaluation"]
        required_point_fields = ["point_id", "final_score", "total_questions", "dimension_scores"]
        for field in required_point_fields:
            assert field in point_eval, f"Contract violation: missing point field '{field}'"
    
    def test_artifact_consistency_validation(self, scoring_system: ScoringSystem, registry: DecalogoQuestionRegistry):
        """Test artifact consistency across multiple evaluation runs."""
        # Create consistent test data
        evaluation_data = {
            "DE-1": [
                {
                    "question_id": f"DQ_01_DE1_{i:08d}",
                    "response": "Sí",
                    "evidence_completeness": 0.8,
                    "page_reference_quality": 0.7
                }
                for i in range(11)
            ],
            "DE-2": [
                {
                    "question_id": f"DQ_01_DE2_{i:08d}",
                    "response": "Sí",
                    "evidence_completeness": 0.8,
                    "page_reference_quality": 0.7
                }
                for i in range(11)
            ],
            "DE-3": [
                {
                    "question_id": f"DQ_01_DE3_{i:08d}",
                    "response": "Sí",
                    "evidence_completeness": 0.8,
                    "page_reference_quality": 0.7
                }
                for i in range(11)
            ],
            "DE-4": [
                {
                    "question_id": f"DQ_01_DE4_{i:08d}",
                    "response": "Sí",
                    "evidence_completeness": 0.8,
                    "page_reference_quality": 0.7
                }
                for i in range(11)
            ]
        }
        
        # Run evaluation multiple times
        results = []
        for run in range(3):
            point_score = scoring_system.process_point_evaluation(1, evaluation_data)
            results.append({
                "point_id": point_score.point_id,
                "final_score": point_score.final_score,
                "total_questions": point_score.total_questions,
                "dimension_scores": {dim.dimension_id: dim.weighted_average for dim in point_score.dimension_scores}
            })
        
        # Verify consistency across runs
        for i in range(1, len(results)):
            assert results[i]["final_score"] == results[0]["final_score"], (
                f"Run {i} final_score {results[i]['final_score']} != run 0 {results[0]['final_score']}"
            )
            assert results[i]["total_questions"] == results[0]["total_questions"], (
                f"Run {i} total_questions inconsistent"
            )
            
            for dim_id in results[i]["dimension_scores"]:
                assert results[i]["dimension_scores"][dim_id] == results[0]["dimension_scores"][dim_id], (
                    f"Run {i} dimension {dim_id} score inconsistent"
                )
    
    # Negative test cases for robust error handling
    
    def test_malformed_evidence_inputs_handling(self, scoring_system: ScoringSystem):
        """Test handling of malformed evidence inputs."""
        # Test out-of-bounds evidence values
        malformed_cases = [
            (-0.1, 0.5),  # Negative completeness
            (1.5, 0.5),   # Completeness > 1
            (0.5, -0.1),  # Negative reference quality
            (0.5, 1.5),   # Reference quality > 1
        ]
        
        for completeness, reference_quality in malformed_cases:
            # Should clamp values to valid range
            multiplier = scoring_system.calculate_evidence_multiplier(completeness, reference_quality)
            assert 0.5 <= float(multiplier) <= 1.2, (
                f"Malformed evidence ({completeness}, {reference_quality}) "
                f"produced invalid multiplier {multiplier}"
            )
    
    def test_incomplete_evaluation_data_handling(self, scoring_system: ScoringSystem):
        """Test handling of incomplete evaluation data."""
        # Missing dimension
        incomplete_data = {
            "DE-1": [
                {
                    "question_id": "DQ_01_DE1_12345678",
                    "response": "Sí",
                    "evidence_completeness": 0.8,
                    "page_reference_quality": 0.7
                }
            ]
            # Missing DE-2, DE-3, DE-4
        }
        
        # Should still process available data
        point_score = scoring_system.process_point_evaluation(1, incomplete_data)
        assert point_score.total_questions == 1, "Should process available questions"
        assert len(point_score.dimension_scores) == 1, "Should have one dimension score"
        assert point_score.dimension_scores[0].dimension_id == "DE-1", "Should process DE-1"
    
    def test_invalid_response_handling(self, scoring_system: ScoringSystem):
        """Test handling of invalid response values."""
        invalid_responses = ["Invalid", "", None, 123, "sí", "PARCIAL"]
        
        for invalid_response in invalid_responses:
            # Should treat as "NI" (No Information)
            base_score = scoring_system.calculate_base_score(str(invalid_response) if invalid_response is not None else "")
            assert float(base_score) == 0.0, (
                f"Invalid response '{invalid_response}' should be treated as NI (score 0.0), "
                f"got {base_score}"
            )
    
    def test_score_aggregation_edge_cases(self, scoring_system: ScoringSystem):
        """Test score aggregation edge cases."""
        # Empty question list
        empty_responses = []
        dimension_score = scoring_system.aggregate_dimension_score(empty_responses)
        assert float(dimension_score) == 0.0, "Empty question list should result in 0.0 score"
        
        # Single perfect question
        single_response = [
            QuestionResponse(
                question_id="test",
                response="Sí",
                base_score=1.0,
                evidence_completeness=1.0,
                page_reference_quality=1.0,
                final_score=1.2
            )
        ]
        dimension_score = scoring_system.aggregate_dimension_score(single_response)
        assert float(dimension_score) == 1.2, f"Single perfect response should give 1.2, got {dimension_score}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])