"""
CI Determinism Verification Suite for L-Stage Classification Evaluation

Validates deterministic behavior:
- Repeatability verification by running identical documents twice
- JSON output consistency and byte-level comparison
- Deterministic ordering and processing stability
- State isolation between evaluation runs
- Random seed independence and deterministic scoring
"""

import json
import pytest
import hashlib
from typing import Dict, List, Any, Optional
from pathlib import Path
from copy import deepcopy
import sys

# Add canonical_flow to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "canonical_flow"))

from canonical_flow.A_analysis_nlp.question_analyzer import (
    DecalogoQuestionRegistry, 
    get_decalogo_question_registry,
    analyze_all_decalogo_questions
)
from canonical_flow.L_classification_evaluation.decalogo_scoring_system import (
    ScoringSystem, 
    QuestionResponse, 
    DimensionScore, 
    PointScore
)


class TestLStageDeterminismVerification:
    """Determinism verification for L-stage classification evaluation."""
    
    @pytest.fixture
    def scoring_system(self) -> ScoringSystem:
        """Get fresh scoring system instance."""
        return ScoringSystem(precision=4)
    
    @pytest.fixture 
    def sample_document_data(self) -> Dict[str, Any]:
        """Generate consistent sample document data."""
        return {
            "document_id": "test_document_001",
            "content": """
            Este documento describe el derecho a la vida, seguridad y convivencia.
            Se establecen productos medibles para garantizar estos derechos humanos.
            El diagnóstico muestra la situación actual de la población objetivo.
            Se incluyen mecanismos de seguimiento y evaluación del impacto.
            """,
            "metadata": {
                "source": "test_suite",
                "pages": 1,
                "language": "es"
            }
        }
    
    @pytest.fixture
    def deterministic_evaluation_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate deterministic evaluation data for testing."""
        # Use fixed random seed equivalent by providing deterministic data
        return {
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
    
    def test_identical_document_processing_determinism(self, scoring_system: ScoringSystem, deterministic_evaluation_data: Dict[str, List[Dict[str, Any]]]):
        """Test that processing the same document twice produces identical results."""
        # First evaluation run
        result1 = scoring_system.process_point_evaluation(1, deterministic_evaluation_data)
        
        # Second evaluation run with identical data
        result2 = scoring_system.process_point_evaluation(1, deepcopy(deterministic_evaluation_data))
        
        # Verify identical results
        assert result1.point_id == result2.point_id, "Point IDs must be identical"
        assert result1.final_score == result2.final_score, (
            f"Final scores differ: {result1.final_score} vs {result2.final_score}"
        )
        assert result1.total_questions == result2.total_questions, "Question counts must be identical"
        
        # Verify dimension-level determinism
        assert len(result1.dimension_scores) == len(result2.dimension_scores), "Dimension counts must match"
        
        for dim1, dim2 in zip(result1.dimension_scores, result2.dimension_scores):
            assert dim1.dimension_id == dim2.dimension_id, f"Dimension IDs differ: {dim1.dimension_id} vs {dim2.dimension_id}"
            assert dim1.weighted_average == dim2.weighted_average, (
                f"Dimension {dim1.dimension_id} averages differ: {dim1.weighted_average} vs {dim2.weighted_average}"
            )
            assert dim1.total_questions == dim2.total_questions, f"Dimension {dim1.dimension_id} question counts differ"
    
    def test_json_output_byte_level_consistency(self, scoring_system: ScoringSystem, deterministic_evaluation_data: Dict[str, List[Dict[str, Any]]]):
        """Test byte-level consistency of JSON outputs."""
        # Generate outputs
        outputs = []
        for run in range(3):
            result = scoring_system.process_point_evaluation(1, deepcopy(deterministic_evaluation_data))
            
            # Convert to deterministic JSON
            output_dict = {
                "point_id": result.point_id,
                "final_score": result.final_score,
                "total_questions": result.total_questions,
                "dimension_scores": [
                    {
                        "dimension_id": dim.dimension_id,
                        "weighted_average": dim.weighted_average,
                        "total_questions": dim.total_questions,
                        "question_responses": [
                            {
                                "question_id": qr.question_id,
                                "response": qr.response,
                                "base_score": qr.base_score,
                                "evidence_completeness": qr.evidence_completeness,
                                "page_reference_quality": qr.page_reference_quality,
                                "final_score": qr.final_score
                            }
                            for qr in sorted(dim.question_responses, key=lambda x: x.question_id)
                        ]
                    }
                    for dim in sorted(result.dimension_scores, key=lambda x: x.dimension_id)
                ]
            }
            
            # Generate deterministic JSON
            json_output = json.dumps(output_dict, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
            outputs.append(json_output)
        
        # Verify byte-level identity
        for i in range(1, len(outputs)):
            assert outputs[i] == outputs[0], f"JSON output run {i} differs from run 0"
        
        # Verify hash consistency
        hashes = [hashlib.sha256(output.encode('utf-8')).hexdigest() for output in outputs]
        for i in range(1, len(hashes)):
            assert hashes[i] == hashes[0], f"JSON hash run {i} differs from run 0"
    
    def test_deterministic_ordering_stability(self, scoring_system: ScoringSystem):
        """Test that ordering is deterministic and stable."""
        # Create evaluation data with shuffled input order
        base_data = {
            "DE-4": [{"question_id": f"DQ_01_DE4_{i:08d}", "response": "Sí", "evidence_completeness": 0.8, "page_reference_quality": 0.7} for i in range(11)],
            "DE-1": [{"question_id": f"DQ_01_DE1_{i:08d}", "response": "Parcial", "evidence_completeness": 0.6, "page_reference_quality": 0.5} for i in range(11)],
            "DE-3": [{"question_id": f"DQ_01_DE3_{i:08d}", "response": "No", "evidence_completeness": 0.4, "page_reference_quality": 0.3} for i in range(11)],
            "DE-2": [{"question_id": f"DQ_01_DE2_{i:08d}", "response": "NI", "evidence_completeness": 0.2, "page_reference_quality": 0.1} for i in range(11)]
        }
        
        # Process multiple times
        results = []
        for _ in range(3):
            result = scoring_system.process_point_evaluation(1, deepcopy(base_data))
            results.append(result)
        
        # Verify ordering is consistent
        for i in range(1, len(results)):
            # Check dimension ordering
            dim_ids_1 = [dim.dimension_id for dim in results[0].dimension_scores]
            dim_ids_i = [dim.dimension_id for dim in results[i].dimension_scores]
            assert dim_ids_1 == dim_ids_i, f"Dimension ordering differs between runs 0 and {i}"
            
            # Check question ordering within dimensions
            for dim1, dimi in zip(results[0].dimension_scores, results[i].dimension_scores):
                q_ids_1 = [qr.question_id for qr in dim1.question_responses]
                q_ids_i = [qr.question_id for qr in dimi.question_responses]
                assert q_ids_1 == q_ids_i, (
                    f"Question ordering differs in dimension {dim1.dimension_id} between runs 0 and {i}"
                )
    
    def test_state_isolation_between_runs(self, scoring_system: ScoringSystem):
        """Test that evaluation runs don't interfere with each other."""
        # Create different evaluation scenarios
        scenario_a = {
            "DE-1": [{"question_id": f"DQ_01_DE1_{i:08d}", "response": "Sí", "evidence_completeness": 1.0, "page_reference_quality": 1.0} for i in range(11)],
            "DE-2": [{"question_id": f"DQ_01_DE2_{i:08d}", "response": "Sí", "evidence_completeness": 1.0, "page_reference_quality": 1.0} for i in range(11)],
            "DE-3": [{"question_id": f"DQ_01_DE3_{i:08d}", "response": "Sí", "evidence_completeness": 1.0, "page_reference_quality": 1.0} for i in range(11)],
            "DE-4": [{"question_id": f"DQ_01_DE4_{i:08d}", "response": "Sí", "evidence_completeness": 1.0, "page_reference_quality": 1.0} for i in range(11)]
        }
        
        scenario_b = {
            "DE-1": [{"question_id": f"DQ_02_DE1_{i:08d}", "response": "No", "evidence_completeness": 0.0, "page_reference_quality": 0.0} for i in range(11)],
            "DE-2": [{"question_id": f"DQ_02_DE2_{i:08d}", "response": "No", "evidence_completeness": 0.0, "page_reference_quality": 0.0} for i in range(11)],
            "DE-3": [{"question_id": f"DQ_02_DE3_{i:08d}", "response": "No", "evidence_completeness": 0.0, "page_reference_quality": 0.0} for i in range(11)],
            "DE-4": [{"question_id": f"DQ_02_DE4_{i:08d}", "response": "No", "evidence_completeness": 0.0, "page_reference_quality": 0.0} for i in range(11)]
        }
        
        # Interleaved processing to test state isolation
        result_a1 = scoring_system.process_point_evaluation(1, scenario_a)
        result_b1 = scoring_system.process_point_evaluation(2, scenario_b)
        result_a2 = scoring_system.process_point_evaluation(1, scenario_a)
        result_b2 = scoring_system.process_point_evaluation(2, scenario_b)
        
        # Verify state isolation
        assert result_a1.final_score == result_a2.final_score, "Scenario A results affected by interleaved processing"
        assert result_b1.final_score == result_b2.final_score, "Scenario B results affected by interleaved processing"
        
        # Verify scenarios produce different results (sanity check)
        assert result_a1.final_score != result_b1.final_score, "Different scenarios should produce different results"
    
    def test_registry_determinism_across_instances(self):
        """Test that question registry is deterministic across instances."""
        # Create multiple registry instances
        registries = [get_decalogo_question_registry() for _ in range(3)]
        
        # Extract question data for comparison
        registry_data = []
        for registry in registries:
            questions = registry.get_all_questions()
            data = [(q.question_id, q.point_number, q.dimension_code, q.question_text) for q in questions]
            registry_data.append(sorted(data))  # Sort for deterministic comparison
        
        # Verify identical registries
        for i in range(1, len(registry_data)):
            assert registry_data[i] == registry_data[0], f"Registry instance {i} differs from instance 0"
        
        # Verify question count consistency
        for i, data in enumerate(registry_data):
            assert len(data) == 470, f"Registry instance {i} has {len(data)} questions, expected 470"
    
    def test_scoring_precision_determinism(self, scoring_system: ScoringSystem):
        """Test that scoring precision is deterministic."""
        # Test cases with known precision requirements
        test_cases = [
            ("Sí", 0.8, 0.7),      # Should produce consistent decimal result
            ("Parcial", 0.5, 0.5),  # Should produce 0.5 * multiplier
            ("No", 1.0, 1.0),       # Should produce 0.0 regardless of evidence
        ]
        
        # Multiple runs for each test case
        for response, completeness, reference_quality in test_cases:
            results = []
            for _ in range(5):
                base_score, multiplier, final_score = scoring_system.calculate_final_score(
                    response, completeness, reference_quality
                )
                results.append((float(base_score), float(multiplier), float(final_score)))
            
            # Verify all results are identical
            for i in range(1, len(results)):
                assert results[i] == results[0], (
                    f"Precision determinism failed for {response}, {completeness}, {reference_quality}: "
                    f"Run {i} {results[i]} != Run 0 {results[0]}"
                )
    
    def test_decimal_arithmetic_consistency(self, scoring_system: ScoringSystem):
        """Test that decimal arithmetic is consistent and deterministic."""
        from decimal import Decimal
        
        # Test arithmetic operations
        test_operations = [
            (Decimal("0.1") + Decimal("0.2"), "0.3000"),  # Addition
            (Decimal("1.0") / Decimal("3.0"), "0.3333"),  # Division with rounding
            (Decimal("0.5") * Decimal("1.2"), "0.6000"),  # Multiplication
        ]
        
        for operation_result, expected_str in test_operations:
            rounded_result = scoring_system._round_decimal(operation_result)
            assert str(rounded_result) == expected_str, (
                f"Decimal operation {operation_result} rounded to {rounded_result}, expected {expected_str}"
            )
    
    def test_multiple_document_determinism(self, scoring_system: ScoringSystem):
        """Test determinism across multiple different documents."""
        # Create multiple document scenarios
        documents = []
        for doc_id in range(3):
            evaluation_data = {
                f"DE-{dim_num}": [
                    {
                        "question_id": f"DQ_{doc_id:02d}_DE{dim_num}_{i:08d}",
                        "response": ["Sí", "Parcial", "No", "NI"][i % 4],
                        "evidence_completeness": round(0.2 + (doc_id * 0.1 + i * 0.05) % 0.8, 3),
                        "page_reference_quality": round(0.1 + (doc_id * 0.2 + i * 0.03) % 0.9, 3)
                    }
                    for i in range(11)
                ]
                for dim_num in range(1, 5)
            }
            documents.append(evaluation_data)
        
        # Process each document multiple times
        document_results = []
        for doc_idx, doc_data in enumerate(documents):
            doc_runs = []
            for run in range(3):
                result = scoring_system.process_point_evaluation(doc_idx + 1, deepcopy(doc_data))
                doc_runs.append((result.final_score, result.total_questions))
            
            # Verify determinism within document
            for run in range(1, len(doc_runs)):
                assert doc_runs[run] == doc_runs[0], (
                    f"Document {doc_idx} run {run} differs from run 0: {doc_runs[run]} vs {doc_runs[0]}"
                )
            
            document_results.append(doc_runs[0])
        
        # Verify different documents produce different results (sanity check)
        unique_results = set(document_results)
        assert len(unique_results) == len(documents), "Different documents should produce different results"
    
    # Negative test cases for determinism validation
    
    def test_non_deterministic_inputs_handling(self, scoring_system: ScoringSystem):
        """Test that non-deterministic inputs are handled deterministically."""
        # Simulate inputs with potential non-determinism
        unstable_data = {
            "DE-1": [
                {
                    "question_id": f"DQ_01_DE1_{i:08d}",
                    "response": "Sí",
                    # Floating point precision issues
                    "evidence_completeness": 0.1 + 0.2 + 0.3 + 0.4 - 1.0,  # Should be 0.0 but might have precision error
                    "page_reference_quality": 1.0 / 3.0 * 3.0 - 1.0        # Should be 0.0 but might have precision error
                }
                for i in range(11)
            ],
            "DE-2": [
                {
                    "question_id": f"DQ_01_DE2_{i:08d}",
                    "response": "Parcial",
                    "evidence_completeness": 0.7,
                    "page_reference_quality": 0.5
                }
                for i in range(11)
            ],
            "DE-3": [
                {
                    "question_id": f"DQ_01_DE3_{i:08d}",
                    "response": "No",
                    "evidence_completeness": 0.3,
                    "page_reference_quality": 0.4
                }
                for i in range(11)
            ],
            "DE-4": [
                {
                    "question_id": f"DQ_01_DE4_{i:08d}",
                    "response": "NI",
                    "evidence_completeness": 0.1,
                    "page_reference_quality": 0.2
                }
                for i in range(11)
            ]
        }
        
        # Multiple runs should produce identical results despite input instability
        results = []
        for _ in range(3):
            result = scoring_system.process_point_evaluation(1, deepcopy(unstable_data))
            results.append(result.final_score)
        
        # All results should be identical
        for i in range(1, len(results)):
            assert results[i] == results[0], (
                f"Non-deterministic input handling failed: run {i} score {results[i]} != run 0 {results[0]}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])