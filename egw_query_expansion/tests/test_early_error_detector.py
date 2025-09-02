"""
Tests for Early Error Detection System
"""

import json
import pytest
import time
from unittest.mock import Mock, patch
from typing import Any, Dict, List

from egw_query_expansion.core.early_error_detector import (
    EarlyErrorDetector, 
    ValidationError, 
    ValidationResult,
    ValidationSeverity,
    ValidationErrorType,
    StageSchema,
    LoggingHook,
    WebhookHook,
    validate_stage
)


class TestValidationError:
    """Test ValidationError class"""
    
    def test_validation_error_creation(self):
        error = ValidationError(
            error_type=ValidationErrorType.MISSING_FIELD,
            severity=ValidationSeverity.ERROR,
            message="Test error",
            field_path="test.field",
            stage_name="test_stage"
        )
        
        assert error.error_type == ValidationErrorType.MISSING_FIELD
        assert error.severity == ValidationSeverity.ERROR
        assert error.message == "Test error"
        assert error.field_path == "test.field"
        assert error.stage_name == "test_stage"
    
    def test_validation_error_to_dict(self):
        error = ValidationError(
            error_type=ValidationErrorType.TYPE_MISMATCH,
            severity=ValidationSeverity.WARNING,
            message="Type mismatch",
            expected_value="string",
            actual_value="int"
        )
        
        error_dict = error.to_dict()
        assert error_dict["error_type"] == "type_mismatch"
        assert error_dict["severity"] == "warning"
        assert error_dict["message"] == "Type mismatch"
        assert error_dict["expected_value"] == "string"
        assert error_dict["actual_value"] == "int"


class TestValidationResult:
    """Test ValidationResult class"""
    
    def test_valid_result(self):
        result = ValidationResult(is_valid=True)
        assert result.is_valid
        assert not result.has_errors
        assert not result.has_warnings
        assert len(result.critical_errors) == 0
    
    def test_result_with_errors(self):
        error = ValidationError(
            error_type=ValidationErrorType.MISSING_FIELD,
            severity=ValidationSeverity.ERROR,
            message="Missing field"
        )
        
        result = ValidationResult(is_valid=False, errors=[error])
        assert not result.is_valid
        assert result.has_errors
        assert len(result.errors) == 1
    
    def test_critical_errors_filter(self):
        critical_error = ValidationError(
            error_type=ValidationErrorType.SCHEMA_VIOLATION,
            severity=ValidationSeverity.CRITICAL,
            message="Critical error"
        )
        
        normal_error = ValidationError(
            error_type=ValidationErrorType.MISSING_FIELD,
            severity=ValidationSeverity.ERROR,
            message="Normal error"
        )
        
        result = ValidationResult(
            is_valid=False, 
            errors=[critical_error, normal_error]
        )
        
        assert len(result.critical_errors) == 1
        assert result.critical_errors[0].severity == ValidationSeverity.CRITICAL


class TestStageSchema:
    """Test StageSchema class"""
    
    def test_stage_schema_creation(self):
        schema = StageSchema(
            stage_name="test_stage",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            required_fields={"field1", "field2"},
            optional_fields={"field3"}
        )
        
        assert schema.stage_name == "test_stage"
        assert schema.required_fields == {"field1", "field2"}
        assert schema.optional_fields == {"field3"}


class TestLoggingHook:
    """Test LoggingHook class"""
    
    def test_logging_hook_creation(self):
        hook = LoggingHook()
        assert hook.logger is not None
    
    def test_logging_hook_with_custom_logger(self):
        import logging
        custom_logger = logging.getLogger("custom")
        hook = LoggingHook(custom_logger)
        assert hook.logger == custom_logger
    
    @patch('logging.Logger.info')
    def test_on_validation_start_logging(self, mock_log):
        hook = LoggingHook()
        hook.on_validation_start("test_stage", {"data": "test"})
        mock_log.assert_called_once_with("Starting validation for stage: test_stage")
    
    @patch('logging.Logger.log')
    def test_on_validation_error_logging(self, mock_log):
        hook = LoggingHook()
        error = ValidationError(
            error_type=ValidationErrorType.MISSING_FIELD,
            severity=ValidationSeverity.ERROR,
            message="Test error",
            stage_name="test_stage"
        )
        
        hook.on_validation_error(error)
        mock_log.assert_called_once()


class TestWebhookHook:
    """Test WebhookHook class"""
    
    def test_webhook_hook_creation(self):
        hook = WebhookHook("https://example.com/webhook")
        assert hook.webhook_url == "https://example.com/webhook"
        assert hook.auth_token is None
    
    def test_webhook_hook_with_auth(self):
        hook = WebhookHook("https://example.com/webhook", "token123")
        assert hook.auth_token == "token123"
    
    def test_invalid_webhook_url(self):
        with pytest.raises(ValueError):
            WebhookHook("invalid-url")
    
    @patch('urllib.request.urlopen')
    def test_send_notification_success(self, mock_urlopen):
        mock_response = Mock()
        mock_response.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        hook = WebhookHook("https://example.com/webhook")
        error = ValidationError(
            error_type=ValidationErrorType.MISSING_FIELD,
            severity=ValidationSeverity.ERROR,
            message="Test error"
        )
        
        hook.on_validation_error(error)
        mock_urlopen.assert_called_once()
    
    def test_webhook_hook_ignores_warnings(self):
        with patch('urllib.request.urlopen') as mock_urlopen:
            hook = WebhookHook("https://example.com/webhook")
            error = ValidationError(
                error_type=ValidationErrorType.MISSING_FIELD,
                severity=ValidationSeverity.WARNING,
                message="Test warning"
            )
            
            hook.on_validation_error(error)
            mock_urlopen.assert_not_called()


class TestEarlyErrorDetector:
    """Test EarlyErrorDetector class"""
    
    def setup_method(self):
        self.detector = EarlyErrorDetector()
        self.test_schema = StageSchema(
            stage_name="test_stage",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer", "minimum": 0}
                },
                "required": ["name"]
            },
            output_schema={"type": "object"},
            required_fields={"name"},
            optional_fields={"description"},
            value_ranges={
                "age": {"min": 0, "max": 150},
                "status": {"allowed": ["active", "inactive"]}
            }
        )
    
    def test_detector_creation(self):
        assert len(self.detector.schemas) == 0
        assert len(self.detector.hooks) == 0
        assert len(self.detector.stage_artifacts) == 0
    
    def test_register_schema(self):
        self.detector.register_schema(self.test_schema)
        assert "test_stage" in self.detector.schemas
        assert self.detector.schemas["test_stage"] == self.test_schema
    
    def test_add_remove_hooks(self):
        hook = LoggingHook()
        self.detector.add_hook(hook)
        assert hook in self.detector.hooks
        
        self.detector.remove_hook(hook)
        assert hook not in self.detector.hooks
    
    def test_validate_stage_input_no_schema(self):
        result = self.detector.validate_stage_input("unknown_stage", {})
        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].error_type == ValidationErrorType.SCHEMA_VIOLATION
        assert result.errors[0].severity == ValidationSeverity.CRITICAL
    
    def test_validate_stage_input_success(self):
        self.detector.register_schema(self.test_schema)
        data = {"name": "John", "age": 25}
        
        result = self.detector.validate_stage_input("test_stage", data)
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_stage_input_missing_required_field(self):
        self.detector.register_schema(self.test_schema)
        data = {"age": 25}  # Missing required "name" field
        
        result = self.detector.validate_stage_input("test_stage", data)
        assert not result.is_valid
        assert len(result.errors) >= 1
        
        # Check for missing field error
        missing_field_errors = [
            e for e in result.errors 
            if e.error_type == ValidationErrorType.MISSING_FIELD
        ]
        assert len(missing_field_errors) == 1
        assert "name" in missing_field_errors[0].message
    
    def test_validate_stage_input_value_range_violation(self):
        self.detector.register_schema(self.test_schema)
        data = {"name": "John", "age": 200}  # Age exceeds maximum
        
        result = self.detector.validate_stage_input("test_stage", data)
        assert not result.is_valid
        
        range_errors = [
            e for e in result.errors 
            if e.error_type == ValidationErrorType.RANGE_VIOLATION
        ]
        assert len(range_errors) >= 1
    
    def test_validate_stage_input_allowed_values_violation(self):
        schema = StageSchema(
            stage_name="status_stage",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            required_fields=set(),
            value_ranges={
                "status": {"allowed": ["active", "inactive"]}
            }
        )
        self.detector.register_schema(schema)
        
        data = {"status": "unknown"}  # Invalid status value
        
        result = self.detector.validate_stage_input("status_stage", data)
        assert not result.is_valid
        
        range_errors = [
            e for e in result.errors 
            if e.error_type == ValidationErrorType.RANGE_VIOLATION
        ]
        assert len(range_errors) >= 1
    
    def test_validate_artifacts_missing(self):
        schema = StageSchema(
            stage_name="artifact_stage",
            input_schema={},
            output_schema={},
            required_fields=set(),
            artifact_patterns=[".*\\.json$", ".*\\.txt$"]
        )
        self.detector.register_schema(schema)
        
        data = {}  # No artifacts
        
        result = self.detector.validate_stage_input("artifact_stage", data)
        assert not result.is_valid
        
        artifact_errors = [
            e for e in result.errors 
            if e.error_type == ValidationErrorType.ARTIFACT_MISSING
        ]
        assert len(artifact_errors) >= 1
    
    def test_validate_artifacts_naming_convention(self):
        schema = StageSchema(
            stage_name="naming_stage",
            input_schema={},
            output_schema={},
            required_fields=set(),
            artifact_patterns=["data_.*\\.json$"]
        )
        self.detector.register_schema(schema)
        
        data = {"artifacts": ["wrong_name.json", "other.txt"]}
        
        result = self.detector.validate_stage_input("naming_stage", data)
        # Should have warnings about naming convention
        naming_errors = [
            e for e in (result.errors + result.warnings)
            if e.error_type == ValidationErrorType.NAMING_CONVENTION
        ]
        assert len(naming_errors) >= 1
    
    def test_validate_metadata_requirements(self):
        schema = StageSchema(
            stage_name="metadata_stage",
            input_schema={},
            output_schema={},
            required_fields=set(),
            metadata_requirements={
                "version": {"required": True, "type": str},
                "timestamp": {"required": False, "type": float}
            }
        )
        self.detector.register_schema(schema)
        
        # Missing required metadata
        data = {"metadata": {}}
        result = self.detector.validate_stage_input("metadata_stage", data)
        assert not result.is_valid
        
        metadata_errors = [
            e for e in result.errors 
            if e.error_type == ValidationErrorType.METADATA_INVALID
        ]
        assert len(metadata_errors) >= 1
    
    def test_validate_metadata_type_mismatch(self):
        schema = StageSchema(
            stage_name="type_stage",
            input_schema={},
            output_schema={},
            required_fields=set(),
            metadata_requirements={
                "version": {"required": True, "type": str}
            }
        )
        self.detector.register_schema(schema)
        
        # Wrong type for version
        data = {"metadata": {"version": 123}}  # Should be string
        result = self.detector.validate_stage_input("type_stage", data)
        assert not result.is_valid
        
        type_errors = [
            e for e in result.errors 
            if e.error_type == ValidationErrorType.TYPE_MISMATCH
        ]
        assert len(type_errors) >= 1
    
    def test_validate_referential_integrity(self):
        # Set up previous stage artifacts
        self.detector.stage_artifacts["previous_stage"] = {
            "output_data": "test",
            "results": [1, 2, 3]
        }
        
        schema = StageSchema(
            stage_name="dependent_stage",
            input_schema={},
            output_schema={},
            required_fields=set()
        )
        self.detector.register_schema(schema)
        
        # Reference non-existent field
        data = {
            "references": {
                "previous_stage": ["missing_field"]
            }
        }
        
        result = self.detector.validate_stage_input("dependent_stage", data)
        assert not result.is_valid
        
        integrity_errors = [
            e for e in result.errors 
            if e.error_type == ValidationErrorType.REFERENTIAL_INTEGRITY
        ]
        assert len(integrity_errors) >= 1
    
    def test_validate_referential_integrity_missing_stage(self):
        schema = StageSchema(
            stage_name="dependent_stage",
            input_schema={},
            output_schema={},
            required_fields=set()
        )
        self.detector.register_schema(schema)
        
        # Reference non-existent stage
        data = {
            "references": {
                "missing_stage": ["some_field"]
            }
        }
        
        result = self.detector.validate_stage_input("dependent_stage", data)
        assert not result.is_valid
        
        integrity_errors = [
            e for e in result.errors 
            if e.error_type == ValidationErrorType.REFERENTIAL_INTEGRITY
        ]
        assert len(integrity_errors) >= 1
    
    def test_custom_validators(self):
        def custom_validator(data: Dict[str, Any], stage_name: str) -> List[ValidationError]:
            errors = []
            if "custom_field" in data and data["custom_field"] == "invalid":
                errors.append(ValidationError(
                    error_type=ValidationErrorType.SCHEMA_VIOLATION,
                    severity=ValidationSeverity.ERROR,
                    message="Custom validation failed",
                    stage_name=stage_name
                ))
            return errors
        
        schema = StageSchema(
            stage_name="custom_stage",
            input_schema={},
            output_schema={},
            required_fields=set(),
            custom_validators=[custom_validator]
        )
        self.detector.register_schema(schema)
        
        # Valid data
        result = self.detector.validate_stage_input("custom_stage", {"custom_field": "valid"})
        assert result.is_valid
        
        # Invalid data
        result = self.detector.validate_stage_input("custom_stage", {"custom_field": "invalid"})
        assert not result.is_valid
        assert len(result.errors) >= 1
    
    def test_validate_stage_output_stores_artifacts(self):
        self.detector.register_schema(self.test_schema)
        data = {"name": "John", "output_data": "test"}
        
        result = self.detector.validate_stage_output("test_stage", data)
        assert result.is_valid
        assert "test_stage" in self.detector.stage_artifacts
        assert self.detector.stage_artifacts["test_stage"] == data
    
    def test_clear_stage_artifacts(self):
        self.detector.stage_artifacts["stage1"] = {"data": "test1"}
        self.detector.stage_artifacts["stage2"] = {"data": "test2"}
        
        # Clear specific stage
        self.detector.clear_stage_artifacts("stage1")
        assert "stage1" not in self.detector.stage_artifacts
        assert "stage2" in self.detector.stage_artifacts
        
        # Clear all
        self.detector.clear_stage_artifacts()
        assert len(self.detector.stage_artifacts) == 0
    
    def test_get_validation_summary(self):
        self.detector.register_schema(self.test_schema)
        self.detector.add_hook(LoggingHook())
        self.detector.stage_artifacts["test"] = {"data": "test"}
        
        summary = self.detector.get_validation_summary()
        assert "test_stage" in summary["registered_schemas"]
        assert "test" in summary["stored_artifacts"]
        assert summary["active_hooks"] == 1
    
    def test_get_nested_value(self):
        data = {
            "level1": {
                "level2": {
                    "value": "test"
                }
            }
        }
        
        # Test existing path
        value = self.detector._get_nested_value(data, "level1.level2.value")
        assert value == "test"
        
        # Test non-existent path
        value = self.detector._get_nested_value(data, "level1.missing.value")
        assert value is None


class TestValidateStageDecorator:
    """Test validate_stage decorator"""
    
    def test_validate_stage_decorator_success(self):
        detector = EarlyErrorDetector()
        schema = StageSchema(
            stage_name="decorated_stage",
            input_schema={},
            output_schema={},
            required_fields=set()
        )
        detector.register_schema(schema)
        
        @validate_stage("decorated_stage", detector)
        def test_function(x: int) -> int:
            return x * 2
        
        # Should succeed
        result = test_function(5)
        assert result == 10
    
    def test_validate_stage_decorator_input_failure(self):
        detector = EarlyErrorDetector()
        # No schema registered - should fail
        
        @validate_stage("missing_stage", detector)
        def test_function(x: int) -> int:
            return x * 2
        
        # Should raise ValueError due to validation failure
        with pytest.raises(ValueError) as exc_info:
            test_function(5)
        
        assert "input validation failed" in str(exc_info.value)


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_complete_pipeline_validation(self):
        detector = EarlyErrorDetector()
        
        # Add logging hook
        logging_hook = LoggingHook()
        detector.add_hook(logging_hook)
        
        # Define schemas for a 3-stage pipeline
        stage1_schema = StageSchema(
            stage_name="data_ingestion",
            input_schema={
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "format": {"type": "string"}
                },
                "required": ["source"]
            },
            output_schema={},
            required_fields={"source"},
            value_ranges={
                "format": {"allowed": ["json", "csv", "xml"]}
            },
            metadata_requirements={
                "ingestion_time": {"required": True, "type": float}
            }
        )
        
        stage2_schema = StageSchema(
            stage_name="data_processing",
            input_schema={},
            output_schema={},
            required_fields={"processed_data"},
            artifact_patterns=["processed_.*\\.json$"]
        )
        
        stage3_schema = StageSchema(
            stage_name="data_output",
            input_schema={},
            output_schema={},
            required_fields={"final_result"}
        )
        
        # Register schemas
        detector.register_schema(stage1_schema)
        detector.register_schema(stage2_schema)
        detector.register_schema(stage3_schema)
        
        # Stage 1: Data Ingestion
        stage1_input = {
            "source": "database",
            "format": "json",
            "metadata": {
                "ingestion_time": time.time()
            }
        }
        
        result1 = detector.validate_stage_input("data_ingestion", stage1_input)
        assert result1.is_valid
        
        stage1_output = {
            "raw_data": [1, 2, 3, 4, 5],
            "metadata": stage1_input["metadata"]
        }
        
        result1_output = detector.validate_stage_output("data_ingestion", stage1_output)
        assert result1_output.is_valid
        
        # Stage 2: Data Processing (with referential integrity)
        stage2_input = {
            "processed_data": [2, 4, 6, 8, 10],
            "artifacts": ["processed_data.json"],
            "references": {
                "data_ingestion": ["raw_data", "metadata"]
            }
        }
        
        result2 = detector.validate_stage_input("data_processing", stage2_input)
        assert result2.is_valid
        
        stage2_output = {
            "processed_data": stage2_input["processed_data"],
            "processing_metadata": {"algorithm": "double"}
        }
        
        result2_output = detector.validate_stage_output("data_processing", stage2_output)
        assert result2_output.is_valid
        
        # Stage 3: Data Output
        stage3_input = {
            "final_result": {"sum": 30, "average": 6.0},
            "references": {
                "data_processing": ["processed_data"]
            }
        }
        
        result3 = detector.validate_stage_input("data_output", stage3_input)
        assert result3.is_valid
        
        # Verify all stages have stored artifacts
        summary = detector.get_validation_summary()
        assert len(summary["stored_artifacts"]) == 2  # stage 1 and 2 outputs stored
        assert "data_ingestion" in summary["stored_artifacts"]
        assert "data_processing" in summary["stored_artifacts"]


if __name__ == "__main__":
    pytest.main([__file__])