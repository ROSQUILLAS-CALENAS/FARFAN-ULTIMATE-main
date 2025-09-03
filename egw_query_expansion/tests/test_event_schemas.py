"""
Tests for event schema definitions
"""

import pytest
import time
from unittest.mock import Mock

from egw_query_expansion.core.event_schemas import (
    PipelineStage, ValidationOutcome, OperationType, PipelineContext,
    ValidationPayload, ValidationResult, StageStartedEvent, StageCompletedEvent,
    StageFailedEvent, ValidationRequestedEvent, ValidationCompletedEvent,
    DataTransformEvent, ErrorEvent, PerformanceMetricEvent,
    create_stage_started_event, create_validation_request, create_error_event,
    create_performance_event
)


class TestPipelineContext:
    """Test cases for PipelineContext"""
    
    def test_context_creation(self):
        """Test pipeline context creation"""
        context = PipelineContext(query="test query", document_id="doc123")
        
        assert context.query == "test query"
        assert context.document_id == "doc123"
        assert isinstance(context.metadata, dict)
        assert isinstance(context.processing_history, list)
        assert isinstance(context.performance_metrics, dict)
    
    def test_context_with_metadata(self):
        """Test context creation with metadata"""
        metadata = {"source": "test", "priority": "high"}
        context = PipelineContext(
            query="test", 
            metadata=metadata
        )
        
        assert context.metadata == metadata
    
    def test_processing_history_tracking(self):
        """Test processing history tracking"""
        context = PipelineContext(query="test")
        
        context.processing_history.append("ingestion")
        context.processing_history.append("analysis")
        
        assert len(context.processing_history) == 2
        assert context.processing_history == ["ingestion", "analysis"]


class TestValidationPayload:
    """Test cases for ValidationPayload"""
    
    def test_payload_creation(self):
        """Test validation payload creation"""
        payload = ValidationPayload(
            validator_id="test_validator",
            validation_type="structure_check",
            data_to_validate={"test": "data"}
        )
        
        assert payload.validator_id == "test_validator"
        assert payload.validation_type == "structure_check"
        assert payload.data_to_validate == {"test": "data"}
        assert payload.required_confidence == 0.8
    
    def test_payload_with_context(self):
        """Test payload with pipeline context"""
        context = PipelineContext(query="test")
        payload = ValidationPayload(
            validator_id="test_validator",
            validation_type="test",
            data_to_validate="data",
            context=context
        )
        
        assert payload.context == context


class TestValidationResult:
    """Test cases for ValidationResult"""
    
    def test_result_creation(self):
        """Test validation result creation"""
        result = ValidationResult(
            validator_id="test_validator",
            outcome=ValidationOutcome.PASSED,
            confidence_score=0.95
        )
        
        assert result.validator_id == "test_validator"
        assert result.outcome == ValidationOutcome.PASSED
        assert result.confidence_score == 0.95
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.details, dict)
    
    def test_result_with_errors_and_warnings(self):
        """Test result with errors and warnings"""
        result = ValidationResult(
            validator_id="test_validator",
            outcome=ValidationOutcome.FAILED,
            confidence_score=0.3,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"]
        )
        
        assert len(result.errors) == 2
        assert len(result.warnings) == 1
        assert result.errors[0] == "Error 1"


class TestStageEvents:
    """Test cases for pipeline stage events"""
    
    def test_stage_started_event(self):
        """Test stage started event"""
        context = PipelineContext(query="test")
        event = StageStartedEvent(
            stage=PipelineStage.INGESTION,
            context=context,
            operation_type=OperationType.TRANSFORM
        )
        
        assert event.event_type == "pipeline.stage.started"
        assert event.stage == PipelineStage.INGESTION
        assert event.context == context
        assert event.operation_type == OperationType.TRANSFORM
    
    def test_stage_completed_event(self):
        """Test stage completed event"""
        context = PipelineContext(query="test")
        results = {"processed": True}
        
        event = StageCompletedEvent(
            stage=PipelineStage.ANALYSIS,
            context=context,
            results=results,
            success=True
        )
        
        assert event.event_type == "pipeline.stage.completed"
        assert event.stage == PipelineStage.ANALYSIS
        assert event.results == results
        assert event.success is True
    
    def test_stage_failed_event(self):
        """Test stage failed event"""
        context = PipelineContext(query="test")
        exception_details = {"exception_type": "ValueError", "line": 42}
        
        event = StageFailedEvent(
            stage=PipelineStage.CLASSIFICATION,
            context=context,
            error="Processing failed",
            exception_details=exception_details
        )
        
        assert event.event_type == "pipeline.stage.failed"
        assert event.stage == PipelineStage.CLASSIFICATION
        assert event.error == "Processing failed"
        assert event.exception_details == exception_details


class TestValidationEvents:
    """Test cases for validation events"""
    
    def test_validation_requested_event(self):
        """Test validation requested event"""
        payload = ValidationPayload(
            validator_id="test_validator",
            validation_type="test",
            data_to_validate="data"
        )
        
        event = ValidationRequestedEvent(payload)
        
        assert event.event_type == "validation.requested"
        assert event.payload == payload
    
    def test_validation_completed_event(self):
        """Test validation completed event"""
        result = ValidationResult(
            validator_id="test_validator",
            outcome=ValidationOutcome.PASSED,
            confidence_score=0.9
        )
        
        payload = ValidationPayload(
            validator_id="test_validator",
            validation_type="test",
            data_to_validate="data"
        )
        
        event = ValidationCompletedEvent(result, payload)
        
        assert event.event_type == "validation.completed"
        assert event.result == result
        assert event.original_request == payload


class TestDataTransformEvent:
    """Test cases for data transform events"""
    
    def test_data_transform_event(self):
        """Test data transform event"""
        data = {"key": "value"}
        
        event = DataTransformEvent(
            source_stage=PipelineStage.INGESTION,
            target_stage=PipelineStage.ANALYSIS,
            data=data,
            transformation_type="normalize"
        )
        
        assert event.event_type == "pipeline.data.transform"
        assert event.source_stage == PipelineStage.INGESTION
        assert event.target_stage == PipelineStage.ANALYSIS
        assert event.transform_data == data
        assert event.transformation_type == "normalize"


class TestErrorEvent:
    """Test cases for error events"""
    
    def test_error_event(self):
        """Test error event creation"""
        context = {"component": "validator", "line": 100}
        
        event = ErrorEvent(
            error_type="validation_error",
            error_message="Validation failed",
            stage=PipelineStage.CLASSIFICATION,
            error_context=context,
            severity="critical"
        )
        
        assert event.event_type == "pipeline.error"
        assert event.error_type == "validation_error"
        assert event.error_message == "Validation failed"
        assert event.stage == PipelineStage.CLASSIFICATION
        assert event.error_context == context
        assert event.severity == "critical"


class TestPerformanceMetricEvent:
    """Test cases for performance metric events"""
    
    def test_performance_event(self):
        """Test performance metric event"""
        additional_metrics = {"memory_mb": 256.0, "cpu_percent": 45.0}
        
        event = PerformanceMetricEvent(
            stage=PipelineStage.SYNTHESIS,
            metric_name="processing_time",
            metric_value=1.25,
            metric_unit="seconds",
            additional_metrics=additional_metrics
        )
        
        assert event.event_type == "pipeline.performance"
        assert event.stage == PipelineStage.SYNTHESIS
        assert event.metric_name == "processing_time"
        assert event.metric_value == 1.25
        assert event.metric_unit == "seconds"
        assert event.additional_metrics == additional_metrics


class TestEventCreationHelpers:
    """Test cases for event creation helper functions"""
    
    def test_create_stage_started_event(self):
        """Test stage started event creation helper"""
        context = PipelineContext(query="test")
        
        event = create_stage_started_event(
            stage=PipelineStage.INGESTION,
            context=context,
            source_id="orchestrator"
        )
        
        assert event.event_type == "pipeline.stage.started"
        assert event.stage == PipelineStage.INGESTION
        assert event.metadata.source_id == "orchestrator"
    
    def test_create_validation_request(self):
        """Test validation request creation helper"""
        context = PipelineContext(query="test")
        
        event = create_validation_request(
            validator_id="constraint_validator",
            validation_type="dimension_check",
            data={"dimension": "quality"},
            context=context,
            source_id="orchestrator"
        )
        
        assert event.event_type == "validation.requested"
        assert event.payload.validator_id == "constraint_validator"
        assert event.payload.validation_type == "dimension_check"
        assert event.payload.context == context
        assert event.metadata.source_id == "orchestrator"
    
    def test_create_error_event(self):
        """Test error event creation helper"""
        event = create_error_event(
            error_type="processing_error",
            message="Something went wrong",
            stage=PipelineStage.ANALYSIS,
            source_id="processor"
        )
        
        assert event.event_type == "pipeline.error"
        assert event.error_type == "processing_error"
        assert event.error_message == "Something went wrong"
        assert event.stage == PipelineStage.ANALYSIS
        assert event.metadata.source_id == "processor"
    
    def test_create_performance_event(self):
        """Test performance event creation helper"""
        event = create_performance_event(
            stage=PipelineStage.AGGREGATION,
            metric_name="throughput",
            metric_value=150.5,
            source_id="aggregator"
        )
        
        assert event.event_type == "pipeline.performance"
        assert event.stage == PipelineStage.AGGREGATION
        assert event.metric_name == "throughput"
        assert event.metric_value == 150.5
        assert event.metadata.source_id == "aggregator"


class TestEnumValues:
    """Test cases for enum definitions"""
    
    def test_pipeline_stage_enum(self):
        """Test pipeline stage enum values"""
        assert PipelineStage.INGESTION.value == "ingestion"
        assert PipelineStage.ANALYSIS.value == "analysis"
        assert PipelineStage.KNOWLEDGE_EXTRACTION.value == "knowledge_extraction"
        assert PipelineStage.CLASSIFICATION.value == "classification"
        assert PipelineStage.AGGREGATION.value == "aggregation"
    
    def test_validation_outcome_enum(self):
        """Test validation outcome enum values"""
        assert ValidationOutcome.PASSED.value == "passed"
        assert ValidationOutcome.FAILED.value == "failed"
        assert ValidationOutcome.WARNING.value == "warning"
        assert ValidationOutcome.PARTIAL.value == "partial"
        assert ValidationOutcome.SKIPPED.value == "skipped"
    
    def test_operation_type_enum(self):
        """Test operation type enum values"""
        assert OperationType.TRANSFORM.value == "transform"
        assert OperationType.VALIDATE.value == "validate"
        assert OperationType.AGGREGATE.value == "aggregate"
        assert OperationType.ROUTE.value == "route"
        assert OperationType.ENHANCE.value == "enhance"


if __name__ == "__main__":
    pytest.main([__file__])