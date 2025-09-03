"""
Test suite for schema validation contracts
"""

import pytest
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, Any  # Module not found  # Module not found  # Module not found

# # # from contracts.schemas import (  # Module not found  # Module not found  # Module not found
    QuestionEvalInput,
    DimensionEvalOutput,
    PointEvalOutput,
    StageMeta,
    ComplianceStatus,
    ConfidenceLevel,
    ProcessingStatus,
    SchemaValidationError,
    validate_input_schema,
    validate_output_schema,
    validate_process_schemas,
    enforce_required_fields,
    create_stage_meta,
    validate_pipeline_data,
)


class TestSchemaValidation:
    """Test schema validation functionality"""
    
    def test_question_eval_input_valid(self):
        """Test valid QuestionEvalInput creation"""
        data = {
            "doc_id": "test_doc_001",
            "page_num": 5,
            "question_text": "How does this document address human rights?",
            "context": {"category": "human_rights"},
            "evaluation_criteria": ["relevance", "completeness"],
            "priority": 3
        }
        
        input_obj = QuestionEvalInput(**data)
        assert input_obj.doc_id == "test_doc_001"
        assert input_obj.page_num == 5
        assert input_obj.question_text == "How does this document address human rights?"
        assert len(input_obj.evaluation_criteria) == 2
        
        # Test deterministic ID generation
        id1 = input_obj.get_deterministic_id()
        id2 = input_obj.get_deterministic_id()
        assert id1 == id2
        assert id1.startswith("qe_")
    
    def test_question_eval_input_invalid(self):
        """Test invalid QuestionEvalInput validation"""
        # Missing required fields
        with pytest.raises(ValueError):
            QuestionEvalInput(question_text="test")
        
        # Empty doc_id
        with pytest.raises(ValueError):
            QuestionEvalInput(doc_id="", page_num=1, question_text="test")
        
        # Invalid page_num
        with pytest.raises(ValueError):
            QuestionEvalInput(doc_id="test", page_num=0, question_text="test")
        
        # Empty question_text
        with pytest.raises(ValueError):
            QuestionEvalInput(doc_id="test", page_num=1, question_text="")
    
    def test_dimension_eval_output_valid(self):
        """Test valid DimensionEvalOutput creation"""
        data = {
            "doc_id": "test_doc_001",
            "page_num": 5,
            "dimension_id": "DE1",
            "dimension_name": "Institutional Dimension",
            "score": 0.75,
            "compliance_status": ComplianceStatus.CUMPLE,
            "confidence_level": ConfidenceLevel.HIGH,
            "evidence_count": 8,
            "sub_scores": {"governance": 0.8, "transparency": 0.7},
            "recommendations": ["Improve documentation", "Add more evidence"]
        }
        
        output_obj = DimensionEvalOutput(**data)
        assert output_obj.dimension_id == "DE1"
        assert output_obj.score == 0.75
        assert output_obj.compliance_status == ComplianceStatus.CUMPLE
        
        # Test deterministic ID generation
        id1 = output_obj.get_deterministic_id()
        assert id1.startswith("de_")
    
    def test_dimension_eval_output_invalid(self):
        """Test invalid DimensionEvalOutput validation"""
        base_data = {
            "doc_id": "test_doc",
            "page_num": 1,
            "dimension_name": "Test Dimension",
            "score": 0.5,
            "compliance_status": ComplianceStatus.CUMPLE_PARCIAL,
            "confidence_level": ConfidenceLevel.MEDIUM,
            "evidence_count": 3
        }
        
        # Invalid dimension_id
        with pytest.raises(ValueError):
            DimensionEvalOutput(**base_data, dimension_id="DE5")
        
        # Invalid score range
        with pytest.raises(ValueError):
            DimensionEvalOutput(**base_data, dimension_id="DE1", score=1.5)
    
    def test_point_eval_output_valid(self):
        """Test valid PointEvalOutput creation"""
        data = {
            "doc_id": "test_doc_001",
            "page_num": 5,
            "point_id": "P7",
            "point_title": "Desarrollo económico inclusivo",
            "score": 0.65,
            "compliance_status": ComplianceStatus.CUMPLE_PARCIAL,
            "confidence_level": ConfidenceLevel.MEDIUM,
            "evidence_count": 4,
            "dimension_alignment": "DE3",
            "key_findings": ["Economic growth noted", "Inclusion gaps exist"],
            "gap_analysis": ["Need more inclusive policies"]
        }
        
        output_obj = PointEvalOutput(**data)
        assert output_obj.point_id == "P7"
        assert output_obj.score == 0.65
        assert output_obj.dimension_alignment == "DE3"
        
        # Test deterministic ID generation
        id1 = output_obj.get_deterministic_id()
        assert id1.startswith("pe_")
    
    def test_point_eval_output_invalid(self):
        """Test invalid PointEvalOutput validation"""
        base_data = {
            "doc_id": "test_doc",
            "page_num": 1,
            "point_title": "Test Point",
            "score": 0.5,
            "compliance_status": ComplianceStatus.CUMPLE_PARCIAL,
            "confidence_level": ConfidenceLevel.MEDIUM,
            "evidence_count": 3
        }
        
        # Invalid point_id
        with pytest.raises(ValueError):
            PointEvalOutput(**base_data, point_id="P11")
    
    def test_stage_meta_valid(self):
        """Test valid StageMeta creation"""
        data = {
            "stage_name": "classification_engine",
            "stage_version": "2.1.0",
            "processing_status": ProcessingStatus.SUCCESS,
            "execution_time_ms": 250.5,
            "performance_metrics": {"accuracy": 0.95, "recall": 0.88},
            "resource_usage": {"memory_mb": 128, "cpu_percent": 45}
        }
        
        meta_obj = StageMeta(**data)
        assert meta_obj.stage_name == "classification_engine"
        assert meta_obj.execution_time_ms == 250.5
        assert meta_obj.performance_metrics["accuracy"] == 0.95
    
    def test_schema_validation_decorators(self):
        """Test schema validation decorators"""
        
        @validate_input_schema(QuestionEvalInput)
        def process_question(data):
            return {"processed": True, "question": data.question_text}
        
        # Valid input
        valid_data = {
            "doc_id": "test_doc",
            "page_num": 1,
            "question_text": "Test question?"
        }
        result = process_question(valid_data)
        assert result["processed"] is True
        
        # Invalid input should raise SchemaValidationError
        invalid_data = {"question_text": "Missing required fields"}
        with pytest.raises(SchemaValidationError):
            process_question(invalid_data)
    
    def test_enforce_required_fields_decorator(self):
        """Test enforce_required_fields decorator"""
        
        @enforce_required_fields("doc_id", "page_num")
        def process_data(data):
            return {"status": "processed"}
        
        # Valid data
        valid_data = {"doc_id": "test", "page_num": 1, "extra": "data"}
        result = process_data(valid_data)
        assert result["status"] == "processed"
        
        # Missing required field
        invalid_data = {"page_num": 1}
        with pytest.raises(SchemaValidationError):
            process_data(invalid_data)
    
    def test_create_stage_meta_utility(self):
        """Test create_stage_meta utility function"""
        meta = create_stage_meta(
            stage_name="test_stage",
            processing_status=ProcessingStatus.SUCCESS,
            stage_version="1.0",
            execution_time_ms=100.0
        )
        
        assert isinstance(meta, StageMeta)
        assert meta.stage_name == "test_stage"
        assert meta.processing_status == ProcessingStatus.SUCCESS
        assert meta.execution_time_ms == 100.0
    
    def test_validate_pipeline_data(self):
        """Test validate_pipeline_data utility"""
        schemas = {
            "input": QuestionEvalInput,
            "output": DimensionEvalOutput
        }
        
        data = {
            "input": {
                "doc_id": "test",
                "page_num": 1,
                "question_text": "Test?"
            },
            "output": {
                "doc_id": "test",
                "page_num": 1,
                "dimension_id": "DE1",
                "dimension_name": "Test Dimension",
                "score": 0.8,
                "compliance_status": "CUMPLE",
                "confidence_level": "high",
                "evidence_count": 5
            }
        }
        
        validated = validate_pipeline_data(data, schemas)
        assert isinstance(validated["input"], QuestionEvalInput)
        assert isinstance(validated["output"], DimensionEvalOutput)
    
    def test_strict_unknown_field_rejection(self):
        """Test that unknown fields are strictly rejected"""
        with pytest.raises(ValueError):  # Pydantic validation error
            QuestionEvalInput(
                doc_id="test",
                page_num=1,
                question_text="test",
                unknown_field="should_fail"
            )
    
    def test_deterministic_id_stability(self):
        """Test that deterministic IDs are stable across instances"""
        data = {
            "doc_id": "test_doc",
            "page_num": 1,
            "question_text": "Test question",
            "context": {"key": "value"},
            "priority": 5
        }
        
        obj1 = QuestionEvalInput(**data)
        obj2 = QuestionEvalInput(**data)
        
        assert obj1.get_deterministic_id() == obj2.get_deterministic_id()
        
        # Different data should produce different IDs
        data2 = data.copy()
        data2["question_text"] = "Different question"
        obj3 = QuestionEvalInput(**data2)
        
        assert obj1.get_deterministic_id() != obj3.get_deterministic_id()


if __name__ == "__main__":
    pytest.main([__file__])