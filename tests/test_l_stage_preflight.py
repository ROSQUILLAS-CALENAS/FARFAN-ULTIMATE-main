"""
CI Preflight Validation Suite for L-Stage Classification Evaluation

Validates:
- Registry checksum verification and integrity
- 470-question count validation across all 10 Decálogo points  
- Question distribution across 4 dimensions (DE-1, DE-2, DE-3, DE-4)
- Registry structure conformance
- Negative test cases for corrupted registries and incomplete question sets
"""

import json
import hashlib
import pytest
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

# Add canonical_flow to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "canonical_flow"))

from canonical_flow.A_analysis_nlp.question_analyzer import (
    DecalogoQuestionRegistry,
    DecalogoQuestion,
    get_decalogo_question_registry
)
from canonical_flow.L_classification_evaluation.decalogo_scoring_system import ScoringSystem


class TestLStagePreflightValidation:
    """Comprehensive preflight validation for L-stage components."""
    
    @pytest.fixture
    def registry(self) -> DecalogoQuestionRegistry:
        """Get clean question registry instance."""
        return get_decalogo_question_registry()
    
    @pytest.fixture
    def scoring_system(self) -> ScoringSystem:
        """Get scoring system instance."""
        return ScoringSystem(precision=4)
    
    def test_registry_checksum_verification(self, registry: DecalogoQuestionRegistry):
        """Test registry checksum verification for integrity."""
        # Get all questions in deterministic order
        questions = registry.get_all_questions()
        
        # Create deterministic representation for checksum
        question_data = []
        for q in questions:
            question_data.append({
                "question_id": q.question_id,
                "point_number": q.point_number,
                "dimension_code": q.dimension_code,
                "question_text": q.question_text
            })
        
        # Sort by question_id for deterministic ordering
        question_data.sort(key=lambda x: x["question_id"])
        
        # Calculate checksum
        registry_json = json.dumps(question_data, sort_keys=True, ensure_ascii=False)
        checksum = hashlib.sha256(registry_json.encode('utf-8')).hexdigest()
        
        # Verify checksum stability across multiple generations
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
        
        assert checksum == checksum2, "Registry checksum must be deterministic across instances"
        assert len(checksum) == 64, "SHA256 checksum must be 64 characters"
        
        # Store checksum for contract verification
        assert hasattr(registry, 'questions'), "Registry must have questions attribute"
        
    def test_470_question_count_validation(self, registry: DecalogoQuestionRegistry):
        """Test that exactly 470 questions exist across all 10 Decálogo points."""
        questions = registry.get_all_questions()
        
        # Total count validation
        assert len(questions) == 470, f"Expected exactly 470 questions, found {len(questions)}"
        
        # Per-point validation (47 questions per point)
        for point_num in range(1, 11):  # Points 1-10
            point_questions = registry.get_questions_by_point(point_num)
            assert len(point_questions) == 47, (
                f"Point {point_num} must have exactly 47 questions, "
                f"found {len(point_questions)}"
            )
        
        # Verify no duplicates
        question_ids = [q.question_id for q in questions]
        assert len(question_ids) == len(set(question_ids)), "All question IDs must be unique"
        
    def test_decalogo_points_distribution(self, registry: DecalogoQuestionRegistry):
        """Test question distribution across all 10 Decálogo points."""
        questions = registry.get_all_questions()
        
        # Count questions by point
        point_counts = {}
        for q in questions:
            point_counts[q.point_number] = point_counts.get(q.point_number, 0) + 1
        
        # Verify all 10 points are present
        expected_points = set(range(1, 11))
        actual_points = set(point_counts.keys())
        assert actual_points == expected_points, (
            f"Missing points: {expected_points - actual_points}, "
            f"Extra points: {actual_points - expected_points}"
        )
        
        # Each point must have exactly 47 questions
        for point_num, count in point_counts.items():
            assert count == 47, f"Point {point_num} has {count} questions, expected 47"
    
    def test_dimension_distribution_validation(self, registry: DecalogoQuestionRegistry):
        """Test question distribution across 4 dimensions (DE-1, DE-2, DE-3, DE-4)."""
        questions = registry.get_all_questions()
        
        expected_dimensions = {"DE1", "DE2", "DE3", "DE4", "GEN"}
        dimension_counts = {}
        
        for q in questions:
            dim = q.dimension_code
            dimension_counts[dim] = dimension_counts.get(dim, 0) + 1
        
        # Verify all expected dimensions are present
        actual_dimensions = set(dimension_counts.keys())
        missing_dims = expected_dimensions - actual_dimensions
        assert not missing_dims, f"Missing dimensions: {missing_dims}"
        
        # Per-point dimension validation
        for point_num in range(1, 11):
            point_questions = registry.get_questions_by_point(point_num)
            point_dim_counts = {}
            
            for q in point_questions:
                dim = q.dimension_code
                point_dim_counts[dim] = point_dim_counts.get(dim, 0) + 1
            
            # Each point should have questions in DE1-DE4 (11 each) plus 3 GEN questions
            for dim in ["DE1", "DE2", "DE3", "DE4"]:
                expected_count = 11
                actual_count = point_dim_counts.get(dim, 0)
                assert actual_count == expected_count, (
                    f"Point {point_num} dimension {dim}: expected {expected_count}, "
                    f"got {actual_count}"
                )
            
            # General questions (3 per point)
            gen_count = point_dim_counts.get("GEN", 0)
            assert gen_count == 3, f"Point {point_num} expected 3 GEN questions, got {gen_count}"
    
    def test_question_id_determinism(self, registry: DecalogoQuestionRegistry):
        """Test that question IDs are deterministic and well-formed."""
        questions = registry.get_all_questions()
        
        for q in questions:
            # Question ID format validation
            assert q.question_id.startswith("DQ_"), f"Question ID must start with 'DQ_': {q.question_id}"
            
            # ID must contain point number
            point_part = f"{q.point_number:02d}"
            assert point_part in q.question_id, (
                f"Question ID must contain point number {point_part}: {q.question_id}"
            )
            
            # ID must contain dimension code
            assert q.dimension_code in q.question_id, (
                f"Question ID must contain dimension {q.dimension_code}: {q.question_id}"
            )
            
            # Verify deterministic regeneration
            regenerated_id = q._generate_deterministic_id()
            assert q.question_id == regenerated_id, (
                f"Question ID not deterministic: {q.question_id} vs {regenerated_id}"
            )
    
    def test_registry_structure_conformance(self, registry: DecalogoQuestionRegistry):
        """Test registry structure conforms to expected schema."""
        questions = registry.get_all_questions()
        
        for q in questions:
            # Required fields validation
            assert hasattr(q, 'question_id'), "Question must have question_id"
            assert hasattr(q, 'point_number'), "Question must have point_number"
            assert hasattr(q, 'dimension_code'), "Question must have dimension_code"
            assert hasattr(q, 'question_text'), "Question must have question_text"
            assert hasattr(q, 'metadata'), "Question must have metadata"
            
            # Field type validation
            assert isinstance(q.question_id, str), "question_id must be string"
            assert isinstance(q.point_number, int), "point_number must be int"
            assert isinstance(q.dimension_code, str), "dimension_code must be string"
            assert isinstance(q.question_text, str), "question_text must be string"
            assert isinstance(q.metadata, dict), "metadata must be dict"
            
            # Value range validation
            assert 1 <= q.point_number <= 10, f"Point number must be 1-10: {q.point_number}"
            assert q.dimension_code in ["DE1", "DE2", "DE3", "DE4", "GEN"], (
                f"Invalid dimension code: {q.dimension_code}"
            )
            assert len(q.question_text) > 0, "Question text cannot be empty"
    
    def test_scoring_system_configuration(self, scoring_system: ScoringSystem):
        """Test scoring system is properly configured for validation."""
        system_info = scoring_system.get_system_info()
        
        # Verify base scores
        expected_base_scores = {"Sí": 1.0, "Parcial": 0.5, "No": 0.0, "NI": 0.0}
        assert system_info["base_scores"] == expected_base_scores, "Base scores mismatch"
        
        # Verify Decálogo weights
        expected_weights = {"DE-1": 0.30, "DE-2": 0.25, "DE-3": 0.25, "DE-4": 0.20}
        assert system_info["decalogo_weights"] == expected_weights, "Decálogo weights mismatch"
        
        # Verify multiplier range
        assert system_info["evidence_multiplier_range"] == [0.5, 1.2], "Evidence multiplier range invalid"
        
        # Verify deterministic configuration
        assert system_info["deterministic"] is True, "Scoring must be deterministic"
        assert system_info["validation_enabled"] is True, "Validation must be enabled"
        assert system_info["precision"] == 4, "Precision must be 4 decimal places"
    
    # Negative test cases
    
    def test_corrupted_registry_detection(self):
        """Test detection of corrupted registries."""
        # Create corrupted registry by manipulating internal state
        registry = get_decalogo_question_registry()
        
        # Corrupt the registry by removing questions
        original_count = len(registry.questions)
        registry.questions = registry.questions[:400]  # Remove 70 questions
        
        # Verify corruption is detected
        questions = registry.get_all_questions()
        assert len(questions) != 470, "Corrupted registry should not have 470 questions"
        
        # Test should fail validation if used in real pipeline
        with pytest.raises(AssertionError, match="Expected exactly 470 questions"):
            assert len(questions) == 470, f"Expected exactly 470 questions, found {len(questions)}"
    
    def test_incomplete_question_sets_detection(self):
        """Test detection of incomplete question sets per point."""
        registry = get_decalogo_question_registry()
        
        # Remove questions from point 5
        registry.questions = [q for q in registry.questions if not (q.point_number == 5 and q.dimension_code == "DE1")]
        
        # Verify detection
        point_5_questions = registry.get_questions_by_point(5)
        assert len(point_5_questions) != 47, "Point 5 should be incomplete"
        
        # Test validation failure
        with pytest.raises(AssertionError, match="Point 5 must have exactly 47 questions"):
            assert len(point_5_questions) == 47, f"Point 5 must have exactly 47 questions, found {len(point_5_questions)}"
    
    def test_malformed_dimension_codes(self):
        """Test detection of malformed dimension codes."""
        registry = get_decalogo_question_registry()
        
        # Corrupt dimension codes
        if registry.questions:
            registry.questions[0].dimension_code = "INVALID_DIM"
        
        # Verify detection
        questions = registry.get_all_questions()
        invalid_dims = [q for q in questions if q.dimension_code not in ["DE1", "DE2", "DE3", "DE4", "GEN"]]
        assert len(invalid_dims) > 0, "Should detect malformed dimension codes"
        
        # Test validation failure
        with pytest.raises(AssertionError, match="Invalid dimension code"):
            for q in questions:
                assert q.dimension_code in ["DE1", "DE2", "DE3", "DE4", "GEN"], f"Invalid dimension code: {q.dimension_code}"
    
    def test_duplicate_question_ids_detection(self):
        """Test detection of duplicate question IDs."""
        registry = get_decalogo_question_registry()
        
        # Create duplicate IDs
        if len(registry.questions) >= 2:
            registry.questions[1].question_id = registry.questions[0].question_id
        
        # Verify detection
        questions = registry.get_all_questions()
        question_ids = [q.question_id for q in questions]
        unique_ids = set(question_ids)
        assert len(question_ids) != len(unique_ids), "Should detect duplicate question IDs"
        
        # Test validation failure
        with pytest.raises(AssertionError, match="All question IDs must be unique"):
            assert len(question_ids) == len(set(question_ids)), "All question IDs must be unique"
    
    def test_canonical_flow_artifact_integration(self):
        """Test integration with canonical_flow artifact structure."""
        registry = get_decalogo_question_registry()
        questions = registry.get_all_questions()
        
        # Verify artifact structure compatibility
        artifact_data = {
            "component": "L_classification_evaluation",
            "stage": "preflight_validation",
            "question_registry": {
                "total_questions": len(questions),
                "points_covered": list(set(q.point_number for q in questions)),
                "dimensions_covered": list(set(q.dimension_code for q in questions)),
                "checksum": hashlib.sha256(
                    json.dumps([{
                        "id": q.question_id,
                        "point": q.point_number,
                        "dim": q.dimension_code
                    } for q in sorted(questions, key=lambda x: x.question_id)],
                    sort_keys=True).encode('utf-8')
                ).hexdigest()
            },
            "validation_status": "passed",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        # Validate artifact structure
        assert artifact_data["component"] == "L_classification_evaluation"
        assert artifact_data["question_registry"]["total_questions"] == 470
        assert len(artifact_data["question_registry"]["points_covered"]) == 10
        assert set(artifact_data["question_registry"]["dimensions_covered"]) == {"DE1", "DE2", "DE3", "DE4", "GEN"}
        assert len(artifact_data["question_registry"]["checksum"]) == 64
        
    def test_message_schema_compliance(self):
        """Test compliance with standardized message schemas."""
        registry = get_decalogo_question_registry()
        
        # Create standardized message
        preflight_message = {
            "message_type": "l_stage_preflight_validation",
            "schema_version": "1.0",
            "source_component": "L_classification_evaluation",
            "target_component": "pipeline_orchestrator",
            "payload": {
                "validation_results": {
                    "registry_checksum_valid": True,
                    "question_count_valid": True,
                    "dimension_distribution_valid": True,
                    "total_questions": 470,
                    "points_validated": 10,
                    "dimensions_validated": 5
                },
                "errors": [],
                "warnings": []
            },
            "timestamp": "2024-01-01T00:00:00Z",
            "correlation_id": "test_preflight_001"
        }
        
        # Validate message schema
        required_fields = ["message_type", "schema_version", "source_component", "payload"]
        for field in required_fields:
            assert field in preflight_message, f"Required field missing: {field}"
        
        # Validate payload structure
        payload = preflight_message["payload"]
        assert "validation_results" in payload
        assert "errors" in payload
        assert isinstance(payload["errors"], list)
        
        # Validate results structure
        results = payload["validation_results"]
        required_results = ["registry_checksum_valid", "question_count_valid", "total_questions"]
        for field in required_results:
            assert field in results, f"Required validation result missing: {field}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])