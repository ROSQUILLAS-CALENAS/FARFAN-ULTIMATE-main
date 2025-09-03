"""
Test suite for comprehensive API contract schema definitions.
Tests validation decorators, deterministic ID generation, and unknown response handling.
"""

import pytest
import json
# # # from typing import Dict, Any  # Module not found  # Module not found  # Module not found
# # # from pydantic import ValidationError  # Module not found  # Module not found  # Module not found

# # # from .schemas import (  # Module not found  # Module not found  # Module not found
    QuestionEvalInput, DimensionEvalOutput, PointEvalOutput, StageMeta,
    ResponseValue, DimensionId, validate_input_schema, validate_output_schema,
    validate_both_schemas, reject_unknown_responses, ValidationResult,
    DeterministicSortingMixin
)


class TestResponseValueEnum:
    """Test enum-constrained response values and synonym handling."""
    
    def test_valid_response_values(self):
        """Test all valid response values are accepted."""
        assert ResponseValue.SI == "Sí"
        assert ResponseValue.PARCIAL == "Parcial"
        assert ResponseValue.NO == "No"
        assert ResponseValue.NI == "NI"
    
    def test_synonym_mapping(self):
        """Test documented synonyms are properly mapped."""
        assert ResponseValue.from_synonym("yes") == ResponseValue.SI
        assert ResponseValue.from_synonym("si") == ResponseValue.SI
        assert ResponseValue.from_synonym("sí") == ResponseValue.SI
        assert ResponseValue.from_synonym("partial") == ResponseValue.PARCIAL
        assert ResponseValue.from_synonym("parcialmente") == ResponseValue.PARCIAL
        assert ResponseValue.from_synonym("no") == ResponseValue.NO
        assert ResponseValue.from_synonym("ni") == ResponseValue.NI
        assert ResponseValue.from_synonym("no_info") == ResponseValue.NI
        assert ResponseValue.from_synonym("missing") == ResponseValue.NI
    
    def test_unknown_response_rejection(self):
        """Test that unknown responses are explicitly rejected."""
        with pytest.raises(ValueError, match="Unrecognized response value"):
            ResponseValue.from_synonym("maybe")
        
        with pytest.raises(ValueError, match="Unrecognized response value"):
            ResponseValue.from_synonym("unknown")
        
        with pytest.raises(ValueError, match="Unrecognized response value"):
            reject_unknown_responses("invalid")


class TestQuestionEvalInput:
    """Test QuestionEvalInput schema validation."""
    
    def test_valid_input(self):
        """Test valid input data passes validation."""
        valid_data = {
            "doc_id": "test_doc_123",
            "page_num": 5,
            "question_id": "Q1.1",
            "dimension_id": "DE-1",
            "response": "Sí",
            "evidence_completeness": 0.8,
            "page_reference_quality": 0.9,
            "evaluator_id": "eval_001"
        }
        
        input_obj = QuestionEvalInput.model_validate(valid_data)
        assert input_obj.doc_id == "test_doc_123"
        assert input_obj.page_num == 5
        assert input_obj.response == ResponseValue.SI
        assert input_obj.dimension_id == DimensionId.DE_1
    
    def test_required_fields_validation(self):
        """Test that required doc_id and page_num fields are enforced."""
        # Missing doc_id
        with pytest.raises(ValidationError):
            QuestionEvalInput.model_validate({
                "page_num": 1,
                "question_id": "Q1",
                "dimension_id": "DE-1",
                "response": "Sí"
            })
        
        # Missing page_num
        with pytest.raises(ValidationError):
            QuestionEvalInput.model_validate({
                "doc_id": "test_doc",
                "question_id": "Q1",
                "dimension_id": "DE-1", 
                "response": "Sí"
            })
    
    def test_response_synonym_validation(self):
        """Test response field accepts synonyms and converts them."""
        data = {
            "doc_id": "test_doc",
            "page_num": 1,
            "question_id": "Q1",
            "dimension_id": "DE-1",
            "response": "yes"  # Synonym
        }
        
        input_obj = QuestionEvalInput.model_validate(data)
        assert input_obj.response == ResponseValue.SI
    
    def test_deterministic_id_generation(self):
        """Test that deterministic IDs are consistent and unique."""
        data1 = {
            "doc_id": "doc_A",
            "page_num": 1,
            "question_id": "Q1",
            "dimension_id": "DE-1",
            "response": "Sí"
        }
        
        data2 = {
            "doc_id": "doc_A", 
            "page_num": 1,
            "question_id": "Q1",
            "dimension_id": "DE-1",
            "response": "No"  # Different response
        }
        
        data3 = {
            "doc_id": "doc_B",  # Different doc
            "page_num": 1,
            "question_id": "Q1", 
            "dimension_id": "DE-1",
            "response": "Sí"
        }
        
        input1 = QuestionEvalInput.model_validate(data1)
        input2 = QuestionEvalInput.model_validate(data2)
        input3 = QuestionEvalInput.model_validate(data3)
        
        id1 = input1.generate_deterministic_id()
        id2 = input2.generate_deterministic_id()
        id3 = input3.generate_deterministic_id()
        
        # Same core fields should generate same ID regardless of response
        assert id1 == id2
        # Different doc should generate different ID
        assert id1 != id3
        
        # IDs should be consistent across multiple calls
        assert input1.generate_deterministic_id() == id1


class TestDimensionEvalOutput:
    """Test DimensionEvalOutput schema validation."""
    
    def test_valid_output(self):
        """Test valid dimension output passes validation."""
        valid_data = {
            "dimension_id": "DE-1",
            "doc_id": "test_doc",
            "weighted_average": 0.85,
            "base_score_average": 0.75,
            "evidence_quality_average": 0.90,
            "question_count": 12,
            "question_ids": ["Q1.1", "Q1.2", "Q1.3"],
            "response_distribution": {
                "Sí": 8,
                "Parcial": 3,
                "No": 1,
                "NI": 0
            },
            "validation_passed": True
        }
        
        output_obj = DimensionEvalOutput.model_validate(valid_data)
        assert output_obj.dimension_id == DimensionId.DE_1
        assert output_obj.weighted_average == 0.85
        assert output_obj.validation_passed is True
    
    def test_sorted_dict_output(self):
        """Test deterministic field ordering in dictionary output."""
        data = {
            "dimension_id": "DE-2",
            "doc_id": "test_doc",
            "weighted_average": 0.7,
            "base_score_average": 0.6,
            "evidence_quality_average": 0.8,
            "question_count": 10,
            "question_ids": ["Q2.1", "Q2.2"],
            "response_distribution": {
                "Sí": 5,
                "No": 5
            },
            "validation_passed": True
        }
        
        output_obj = DimensionEvalOutput.model_validate(data)
        sorted_dict = output_obj.to_sorted_dict()
        
        # Check that keys are sorted alphabetically
        keys = list(sorted_dict.keys())
        assert keys == sorted(keys)


class TestPointEvalOutput:
    """Test PointEvalOutput schema validation."""
    
    def test_valid_point_output(self):
        """Test valid point output with all dimensions."""
        valid_data = {
            "point_id": 5,
            "doc_id": "test_doc",
            "final_score": 0.82,
            "normalized_score": 0.82,
            "dimension_scores": {
                "DE-1": 0.85,
                "DE-2": 0.80,
                "DE-3": 0.78,
                "DE-4": 0.85
            },
            "dimension_weights": {
                "DE-1": 0.30,
                "DE-2": 0.25,
                "DE-3": 0.25,
                "DE-4": 0.20
            },
            "total_questions": 47,
            "overall_evidence_quality": 0.88,
            "validation_passed": True
        }
        
        output_obj = PointEvalOutput.model_validate(valid_data)
        assert output_obj.point_id == 5
        assert output_obj.final_score == 0.82
        assert len(output_obj.dimension_scores) == 4
    
    def test_dimension_completeness_validation(self):
        """Test that all required dimensions must be present."""
        incomplete_data = {
            "point_id": 1,
            "doc_id": "test_doc",
            "final_score": 0.8,
            "normalized_score": 0.8,
            "dimension_scores": {
                "DE-1": 0.8,
                "DE-2": 0.8
                # Missing DE-3 and DE-4
            },
            "dimension_weights": {
                "DE-1": 0.5,
                "DE-2": 0.5
            },
            "total_questions": 47,
            "overall_evidence_quality": 0.8,
            "validation_passed": True
        }
        
        with pytest.raises(ValidationError, match="Missing dimensions"):
            PointEvalOutput.model_validate(incomplete_data)


class TestStageMeta:
    """Test StageMeta schema validation."""
    
    def test_valid_stage_meta(self):
        """Test valid stage metadata passes validation."""
        valid_data = {
            "stage_name": "question_evaluation",
            "stage_version": "1.2.3",
            "doc_id": "test_doc",
            "processing_timestamp": "2024-01-15T10:30:00Z",
            "input_schema_version": "1.0.0",
            "output_schema_version": "1.0.0",
            "processing_duration_ms": 1500,
            "memory_usage_mb": 256.5
        }
        
        meta_obj = StageMeta.model_validate(valid_data)
        assert meta_obj.stage_name == "question_evaluation"
        assert meta_obj.stage_version == "1.2.3"
        assert meta_obj.processing_duration_ms == 1500
    
    def test_version_pattern_validation(self):
        """Test semantic version pattern validation."""
        invalid_version_data = {
            "stage_name": "test_stage",
            "stage_version": "1.2",  # Invalid: not semantic version
            "doc_id": "test_doc",
            "processing_timestamp": "2024-01-15T10:30:00Z",
            "input_schema_version": "1.0.0",
            "output_schema_version": "1.0.0"
        }
        
        with pytest.raises(ValidationError):
            StageMeta.model_validate(invalid_version_data)


class TestValidationDecorators:
    """Test schema validation decorators."""
    
    def test_input_validation_decorator(self):
        """Test input validation decorator functionality."""
        
        class MockProcessor:
            @validate_input_schema(QuestionEvalInput)
            def process(self, data):
                return {
                    "processed": True,
                    "doc_id": data.doc_id,
                    "response": data.response.value
                }
        
        processor = MockProcessor()
        
        # Valid input should work
        valid_input = {
            "doc_id": "test_doc",
            "page_num": 1,
            "question_id": "Q1",
            "dimension_id": "DE-1",
            "response": "Sí"
        }
        
        result = processor.process(valid_input)
        assert result["processed"] is True
        assert result["doc_id"] == "test_doc"
        assert result["response"] == "Sí"
        
        # Invalid input should raise detailed error
        invalid_input = {
            "doc_id": "",  # Invalid: too short
            "page_num": 0,  # Invalid: must be >= 1
            "response": "invalid"  # Invalid: unknown response
        }
        
        with pytest.raises(ValidationError, match="Malformed data rejected"):
            processor.process(invalid_input)
    
    def test_output_validation_decorator(self):
        """Test output validation decorator functionality."""
        
        class MockProcessor:
            @validate_output_schema(DimensionEvalOutput)
            def process(self):
                return {
                    "dimension_id": "DE-1",
                    "doc_id": "test_doc",
                    "weighted_average": 0.8,
                    "base_score_average": 0.7,
                    "evidence_quality_average": 0.9,
                    "question_count": 10,
                    "question_ids": ["Q1", "Q2"],
                    "response_distribution": {"Sí": 8, "No": 2},
                    "validation_passed": True
                }
        
        processor = MockProcessor()
        result = processor.process()
        
        # Should return sorted dictionary
        assert isinstance(result, dict)
        assert "dimension_id" in result
        assert result["validation_passed"] is True
        
        # Keys should be sorted
        keys = list(result.keys())
        assert keys == sorted(keys)
    
    def test_both_schemas_decorator(self):
        """Test combined input/output validation decorator."""
        
        @validate_both_schemas(QuestionEvalInput, DimensionEvalOutput) 
        def mock_process(data):
            return {
                "dimension_id": data.dimension_id.value,
                "doc_id": data.doc_id,
                "weighted_average": 0.8,
                "base_score_average": 0.7,
                "evidence_quality_average": 0.9,
                "question_count": 1,
                "question_ids": [data.question_id],
                "response_distribution": {data.response.value: 1},
                "validation_passed": True
            }
        
        valid_input = {
            "doc_id": "test_doc",
            "page_num": 1,
            "question_id": "Q1",
            "dimension_id": "DE-1",
            "response": "Sí"
        }
        
        result = mock_process(valid_input)
        assert isinstance(result, dict)
        assert result["dimension_id"] == "DE-1"


class TestDeterministicSorting:
    """Test deterministic sorting utilities."""
    
    def test_deterministic_json_serialization(self):
        """Test consistent JSON serialization across instances."""
        
        class TestSchema(DeterministicSortingMixin):
            def __init__(self, data):
                self.data = data
            
            def model_dump(self):
                return self.data
        
        data1 = {"z_field": "last", "a_field": "first", "m_field": "middle"}
        data2 = {"a_field": "first", "m_field": "middle", "z_field": "last"}
        
        schema1 = TestSchema(data1)
        schema2 = TestSchema(data2)
        
        json1 = schema1.to_deterministic_json()
        json2 = schema2.to_deterministic_json()
        
        # Should produce identical JSON despite different input ordering
        assert json1 == json2
        
        # JSON should have sorted keys
        parsed = json.loads(json1)
        keys = list(parsed.keys())
        assert keys == sorted(keys)
    
    def test_nested_dict_sorting(self):
        """Test that nested dictionaries are also sorted."""
        
        class TestSchema(DeterministicSortingMixin):
            def model_dump(self):
                return {
                    "outer_z": {"nested_z": 1, "nested_a": 2},
                    "outer_a": {"nested_m": 3, "nested_b": 4}
                }
        
        schema = TestSchema()
        json_output = schema.to_deterministic_json()
        parsed = json.loads(json_output)
        
        # Check outer keys are sorted
        assert list(parsed.keys()) == ["outer_a", "outer_z"]
        
        # Check nested keys are sorted
        assert list(parsed["outer_a"].keys()) == ["nested_b", "nested_m"]
        assert list(parsed["outer_z"].keys()) == ["nested_a", "nested_z"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])