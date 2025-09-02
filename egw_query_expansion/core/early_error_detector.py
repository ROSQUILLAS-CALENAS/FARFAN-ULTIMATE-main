"""
Early Error Detection System for EGW Query Expansion Pipeline

Implements comprehensive validation and monitoring at pipeline stage boundaries
to detect schema compliance issues, data integrity problems, and referential
integrity violations before expensive processing operations commence.

Features:
- Schema validation against predefined JSON schemas
- Type checking and value range validation
- Real-time monitoring hooks with logging and webhook notifications
- Stage-specific artifact validation
- Metadata integrity checks
- Referential integrity validation between pipeline stages
"""

import json
import logging
import re
import time
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union
from urllib.parse import urlparse

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


class ValidationSeverity(Enum):
    """Severity levels for validation errors"""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


class ValidationErrorType(Enum):
    """Types of validation errors"""
    SCHEMA_VIOLATION = "schema_violation"
    TYPE_MISMATCH = "type_mismatch"
    RANGE_VIOLATION = "range_violation"
    MISSING_FIELD = "missing_field"
    INVALID_FORMAT = "invalid_format"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    ARTIFACT_MISSING = "artifact_missing"
    METADATA_INVALID = "metadata_invalid"
    NAMING_CONVENTION = "naming_convention"


@dataclass
class ValidationError:
    """Container for validation error details"""
    error_type: ValidationErrorType
    severity: ValidationSeverity
    message: str
    field_path: str = ""
    expected_value: Any = None
    actual_value: Any = None
    stage_name: str = ""
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization"""
        return {
            "error_type": self.error_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "field_path": self.field_path,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "stage_name": self.stage_name,
            "timestamp": self.timestamp,
            "context": self.context
        }


@dataclass
class ValidationResult:
    """Result of validation check"""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    @property
    def critical_errors(self) -> List[ValidationError]:
        return [e for e in self.errors if e.severity == ValidationSeverity.CRITICAL]


@dataclass
class StageSchema:
    """Schema definition for a pipeline stage"""
    stage_name: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    required_fields: Set[str]
    optional_fields: Set[str] = field(default_factory=set)
    value_ranges: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    custom_validators: List[Callable] = field(default_factory=list)
    artifact_patterns: List[str] = field(default_factory=list)
    metadata_requirements: Dict[str, Any] = field(default_factory=dict)


class ValidationHook(ABC):
    """Abstract base class for validation hooks"""
    
    @abstractmethod
    def on_validation_start(self, stage_name: str, data: Any) -> None:
        """Called when validation starts"""
        pass

    @abstractmethod  
    def on_validation_error(self, error: ValidationError) -> None:
        """Called when validation error occurs"""
        pass

    @abstractmethod
    def on_validation_complete(self, result: ValidationResult) -> None:
        """Called when validation completes"""
        pass


class LoggingHook(ValidationHook):
    """Hook that logs validation events"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def on_validation_start(self, stage_name: str, data: Any) -> None:
        self.logger.info(f"Starting validation for stage: {stage_name}")
    
    def on_validation_error(self, error: ValidationError) -> None:
        log_level = {
            ValidationSeverity.INFO: logging.INFO,
            ValidationSeverity.WARNING: logging.WARNING,
            ValidationSeverity.ERROR: logging.ERROR,
            ValidationSeverity.CRITICAL: logging.CRITICAL
        }.get(error.severity, logging.ERROR)
        
        self.logger.log(log_level, f"Validation error in {error.stage_name}: {error.message}")
    
    def on_validation_complete(self, result: ValidationResult) -> None:
        if result.is_valid:
            self.logger.info("Validation completed successfully")
        else:
            self.logger.error(f"Validation failed with {len(result.errors)} errors")


class WebhookHook(ValidationHook):
    """Hook that sends notifications via webhooks"""
    
    def __init__(self, webhook_url: str, auth_token: Optional[str] = None):
        self.webhook_url = webhook_url
        self.auth_token = auth_token
        self._validate_url()
    
    def _validate_url(self) -> None:
        """Validate webhook URL format"""
        try:
            result = urlparse(self.webhook_url)
            if not all([result.scheme, result.netloc]):
                raise ValueError("Invalid webhook URL")
        except Exception as e:
            raise ValueError(f"Invalid webhook URL: {e}")
    
    def _send_notification(self, payload: Dict[str, Any]) -> None:
        """Send notification to webhook"""
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                self.webhook_url, 
                data=data, 
                headers={'Content-Type': 'application/json'}
            )
            
            if self.auth_token:
                req.add_header('Authorization', f'Bearer {self.auth_token}')
            
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status >= 400:
                    logging.warning(f"Webhook notification failed: {response.status}")
        except Exception as e:
            logging.error(f"Failed to send webhook notification: {e}")
    
    def on_validation_start(self, stage_name: str, data: Any) -> None:
        pass  # Don't send notifications for start events
    
    def on_validation_error(self, error: ValidationError) -> None:
        if error.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            payload = {
                "event_type": "validation_error",
                "timestamp": datetime.now().isoformat(),
                "error": error.to_dict()
            }
            self._send_notification(payload)
    
    def on_validation_complete(self, result: ValidationResult) -> None:
        if not result.is_valid and result.critical_errors:
            payload = {
                "event_type": "validation_failed",
                "timestamp": datetime.now().isoformat(),
                "error_count": len(result.errors),
                "warning_count": len(result.warnings),
                "critical_errors": [e.to_dict() for e in result.critical_errors]
            }
            self._send_notification(payload)


class EarlyErrorDetector:
    """
    Main validation class that checks data integrity and schema compliance
    at pipeline stage boundaries with real-time monitoring capabilities.
    """
    
    def __init__(self):
        self.schemas: Dict[str, StageSchema] = {}
        self.hooks: List[ValidationHook] = []
        self.logger = logging.getLogger(__name__)
        self.stage_artifacts: Dict[str, Dict[str, Any]] = {}
        self._compiled_schemas: Dict[str, Any] = {}
    
    def register_schema(self, schema: StageSchema) -> None:
        """Register a schema for a pipeline stage"""
        self.schemas[schema.stage_name] = schema
        
        # Pre-compile JSON schemas if available
        if HAS_JSONSCHEMA:
            try:
                if schema.input_schema:
                    jsonschema.Draft7Validator.check_schema(schema.input_schema)
                    self._compiled_schemas[f"{schema.stage_name}_input"] = schema.input_schema
                
                if schema.output_schema:
                    jsonschema.Draft7Validator.check_schema(schema.output_schema)
                    self._compiled_schemas[f"{schema.stage_name}_output"] = schema.output_schema
            except Exception as e:
                self.logger.warning(f"Failed to compile schema for {schema.stage_name}: {e}")
    
    def add_hook(self, hook: ValidationHook) -> None:
        """Add a validation hook for monitoring"""
        self.hooks.append(hook)
    
    def remove_hook(self, hook: ValidationHook) -> None:
        """Remove a validation hook"""
        if hook in self.hooks:
            self.hooks.remove(hook)
    
    def _notify_hooks_start(self, stage_name: str, data: Any) -> None:
        """Notify all hooks that validation is starting"""
        for hook in self.hooks:
            try:
                hook.on_validation_start(stage_name, data)
            except Exception as e:
                self.logger.error(f"Hook error on validation start: {e}")
    
    def _notify_hooks_error(self, error: ValidationError) -> None:
        """Notify all hooks of validation error"""
        for hook in self.hooks:
            try:
                hook.on_validation_error(error)
            except Exception as e:
                self.logger.error(f"Hook error on validation error: {e}")
    
    def _notify_hooks_complete(self, result: ValidationResult) -> None:
        """Notify all hooks that validation is complete"""
        for hook in self.hooks:
            try:
                hook.on_validation_complete(result)
            except Exception as e:
                self.logger.error(f"Hook error on validation complete: {e}")
    
    def _validate_json_schema(self, data: Any, schema: Dict[str, Any], 
                            stage_name: str, schema_type: str) -> List[ValidationError]:
        """Validate data against JSON schema"""
        errors = []
        
        if not HAS_JSONSCHEMA:
            return errors
        
        try:
            validator = jsonschema.Draft7Validator(schema)
            for validation_error in validator.iter_errors(data):
                error = ValidationError(
                    error_type=ValidationErrorType.SCHEMA_VIOLATION,
                    severity=ValidationSeverity.ERROR,
                    message=validation_error.message,
                    field_path=".".join(str(p) for p in validation_error.absolute_path),
                    stage_name=stage_name,
                    context={"schema_type": schema_type}
                )
                errors.append(error)
                self._notify_hooks_error(error)
        except Exception as e:
            error = ValidationError(
                error_type=ValidationErrorType.SCHEMA_VIOLATION,
                severity=ValidationSeverity.ERROR,
                message=f"Schema validation failed: {e}",
                stage_name=stage_name,
                context={"schema_type": schema_type}
            )
            errors.append(error)
            self._notify_hooks_error(error)
        
        return errors
    
    def _validate_required_fields(self, data: Dict[str, Any], required_fields: Set[str],
                                stage_name: str) -> List[ValidationError]:
        """Validate that all required fields are present"""
        errors = []
        
        for field in required_fields:
            if field not in data:
                error = ValidationError(
                    error_type=ValidationErrorType.MISSING_FIELD,
                    severity=ValidationSeverity.ERROR,
                    message=f"Required field '{field}' is missing",
                    field_path=field,
                    stage_name=stage_name
                )
                errors.append(error)
                self._notify_hooks_error(error)
        
        return errors
    
    def _validate_value_ranges(self, data: Dict[str, Any], ranges: Dict[str, Dict[str, Any]],
                             stage_name: str) -> List[ValidationError]:
        """Validate that field values are within specified ranges"""
        errors = []
        
        for field_path, range_spec in ranges.items():
            try:
                value = self._get_nested_value(data, field_path)
                if value is None:
                    continue
                
                # Check numeric ranges
                if 'min' in range_spec and value < range_spec['min']:
                    error = ValidationError(
                        error_type=ValidationErrorType.RANGE_VIOLATION,
                        severity=ValidationSeverity.ERROR,
                        message=f"Value {value} below minimum {range_spec['min']}",
                        field_path=field_path,
                        expected_value=f">= {range_spec['min']}",
                        actual_value=value,
                        stage_name=stage_name
                    )
                    errors.append(error)
                    self._notify_hooks_error(error)
                
                if 'max' in range_spec and value > range_spec['max']:
                    error = ValidationError(
                        error_type=ValidationErrorType.RANGE_VIOLATION,
                        severity=ValidationSeverity.ERROR,
                        message=f"Value {value} above maximum {range_spec['max']}",
                        field_path=field_path,
                        expected_value=f"<= {range_spec['max']}",
                        actual_value=value,
                        stage_name=stage_name
                    )
                    errors.append(error)
                    self._notify_hooks_error(error)
                
                # Check allowed values
                if 'allowed' in range_spec and value not in range_spec['allowed']:
                    error = ValidationError(
                        error_type=ValidationErrorType.RANGE_VIOLATION,
                        severity=ValidationSeverity.ERROR,
                        message=f"Value '{value}' not in allowed values",
                        field_path=field_path,
                        expected_value=range_spec['allowed'],
                        actual_value=value,
                        stage_name=stage_name
                    )
                    errors.append(error)
                    self._notify_hooks_error(error)
                
            except Exception as e:
                error = ValidationError(
                    error_type=ValidationErrorType.RANGE_VIOLATION,
                    severity=ValidationSeverity.WARNING,
                    message=f"Could not validate range for {field_path}: {e}",
                    field_path=field_path,
                    stage_name=stage_name
                )
                errors.append(error)
        
        return errors
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = path.split('.')
        value = data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _validate_artifacts(self, stage_name: str, data: Dict[str, Any]) -> List[ValidationError]:
        """Validate stage artifacts exist and follow naming conventions"""
        errors = []
        
        if stage_name not in self.schemas:
            return errors
        
        schema = self.schemas[stage_name]
        
        # Check artifact patterns
        for pattern in schema.artifact_patterns:
            if 'artifacts' not in data:
                error = ValidationError(
                    error_type=ValidationErrorType.ARTIFACT_MISSING,
                    severity=ValidationSeverity.ERROR,
                    message=f"No artifacts found for pattern: {pattern}",
                    stage_name=stage_name,
                    context={"pattern": pattern}
                )
                errors.append(error)
                self._notify_hooks_error(error)
                continue
            
            artifacts = data.get('artifacts', [])
            matching_artifacts = [a for a in artifacts if re.match(pattern, str(a))]
            
            if not matching_artifacts:
                error = ValidationError(
                    error_type=ValidationErrorType.NAMING_CONVENTION,
                    severity=ValidationSeverity.WARNING,
                    message=f"No artifacts match pattern: {pattern}",
                    stage_name=stage_name,
                    context={"pattern": pattern, "artifacts": artifacts}
                )
                errors.append(error)
        
        return errors
    
    def _validate_metadata(self, data: Dict[str, Any], requirements: Dict[str, Any],
                          stage_name: str) -> List[ValidationError]:
        """Validate metadata requirements"""
        errors = []
        
        metadata = data.get('metadata', {})
        
        for field, requirement in requirements.items():
            if field not in metadata:
                severity = ValidationSeverity.ERROR if requirement.get('required', False) else ValidationSeverity.WARNING
                error = ValidationError(
                    error_type=ValidationErrorType.METADATA_INVALID,
                    severity=severity,
                    message=f"Metadata field '{field}' is missing",
                    field_path=f"metadata.{field}",
                    stage_name=stage_name
                )
                errors.append(error)
                continue
            
            value = metadata[field]
            expected_type = requirement.get('type')
            
            if expected_type and not isinstance(value, expected_type):
                error = ValidationError(
                    error_type=ValidationErrorType.TYPE_MISMATCH,
                    severity=ValidationSeverity.ERROR,
                    message=f"Metadata field '{field}' has wrong type",
                    field_path=f"metadata.{field}",
                    expected_value=expected_type.__name__,
                    actual_value=type(value).__name__,
                    stage_name=stage_name
                )
                errors.append(error)
        
        return errors
    
    def _validate_referential_integrity(self, stage_name: str, data: Dict[str, Any]) -> List[ValidationError]:
        """Validate referential integrity with upstream stages"""
        errors = []
        
        # Check if this stage references outputs from previous stages
        references = data.get('references', {})
        
        for ref_stage, ref_fields in references.items():
            if ref_stage not in self.stage_artifacts:
                error = ValidationError(
                    error_type=ValidationErrorType.REFERENTIAL_INTEGRITY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Referenced stage '{ref_stage}' has no artifacts",
                    stage_name=stage_name,
                    context={"referenced_stage": ref_stage}
                )
                errors.append(error)
                continue
            
            stage_artifacts = self.stage_artifacts[ref_stage]
            
            for field in ref_fields:
                if field not in stage_artifacts:
                    error = ValidationError(
                        error_type=ValidationErrorType.REFERENTIAL_INTEGRITY,
                        severity=ValidationSeverity.ERROR,
                        message=f"Referenced field '{field}' not found in stage '{ref_stage}'",
                        field_path=f"references.{ref_stage}.{field}",
                        stage_name=stage_name,
                        context={"referenced_stage": ref_stage, "referenced_field": field}
                    )
                    errors.append(error)
        
        return errors
    
    def validate_stage_input(self, stage_name: str, data: Any) -> ValidationResult:
        """Validate input data for a pipeline stage"""
        self._notify_hooks_start(stage_name, data)
        
        errors = []
        warnings = []
        
        if stage_name not in self.schemas:
            error = ValidationError(
                error_type=ValidationErrorType.SCHEMA_VIOLATION,
                severity=ValidationSeverity.CRITICAL,
                message=f"No schema registered for stage: {stage_name}",
                stage_name=stage_name
            )
            errors.append(error)
            self._notify_hooks_error(error)
            result = ValidationResult(is_valid=False, errors=errors)
            self._notify_hooks_complete(result)
            return result
        
        schema = self.schemas[stage_name]
        
        # Convert data to dict if needed
        if not isinstance(data, dict):
            try:
                data = {"input": data}
            except Exception:
                error = ValidationError(
                    error_type=ValidationErrorType.TYPE_MISMATCH,
                    severity=ValidationSeverity.ERROR,
                    message="Input data must be convertible to dictionary",
                    stage_name=stage_name
                )
                errors.append(error)
                self._notify_hooks_error(error)
        
        # JSON Schema validation
        if schema.input_schema:
            schema_key = f"{stage_name}_input"
            if schema_key in self._compiled_schemas:
                errors.extend(self._validate_json_schema(
                    data, self._compiled_schemas[schema_key], stage_name, "input"
                ))
        
        # Required fields validation
        if isinstance(data, dict):
            errors.extend(self._validate_required_fields(data, schema.required_fields, stage_name))
            
            # Value ranges validation
            errors.extend(self._validate_value_ranges(data, schema.value_ranges, stage_name))
            
            # Artifact validation
            errors.extend(self._validate_artifacts(stage_name, data))
            
            # Metadata validation
            if schema.metadata_requirements:
                errors.extend(self._validate_metadata(data, schema.metadata_requirements, stage_name))
            
            # Referential integrity validation
            errors.extend(self._validate_referential_integrity(stage_name, data))
        
        # Run custom validators
        for validator in schema.custom_validators:
            try:
                custom_errors = validator(data, stage_name)
                if custom_errors:
                    errors.extend(custom_errors)
            except Exception as e:
                error = ValidationError(
                    error_type=ValidationErrorType.SCHEMA_VIOLATION,
                    severity=ValidationSeverity.WARNING,
                    message=f"Custom validator failed: {e}",
                    stage_name=stage_name
                )
                warnings.append(error)
        
        # Separate errors and warnings
        actual_errors = [e for e in errors if e.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
        actual_warnings = [e for e in errors if e.severity == ValidationSeverity.WARNING] + warnings
        
        result = ValidationResult(
            is_valid=len(actual_errors) == 0,
            errors=actual_errors,
            warnings=actual_warnings,
            metadata={"stage_name": stage_name, "validation_timestamp": time.time()}
        )
        
        self._notify_hooks_complete(result)
        return result
    
    def validate_stage_output(self, stage_name: str, data: Any) -> ValidationResult:
        """Validate output data from a pipeline stage and store artifacts"""
        result = self.validate_stage_input(stage_name, data)  # Reuse validation logic
        
        # Store stage artifacts for referential integrity checks
        if result.is_valid and isinstance(data, dict):
            self.stage_artifacts[stage_name] = data.copy()
        
        return result
    
    def clear_stage_artifacts(self, stage_name: Optional[str] = None) -> None:
        """Clear stored stage artifacts"""
        if stage_name:
            self.stage_artifacts.pop(stage_name, None)
        else:
            self.stage_artifacts.clear()
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation state"""
        return {
            "registered_schemas": list(self.schemas.keys()),
            "stored_artifacts": list(self.stage_artifacts.keys()),
            "active_hooks": len(self.hooks),
            "compiled_schemas": list(self._compiled_schemas.keys())
        }


def validate_stage(stage_name: str, detector: Optional[EarlyErrorDetector] = None):
    """Decorator for automatic stage validation"""
    if detector is None:
        detector = EarlyErrorDetector()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate input
            input_data = {"args": args, "kwargs": kwargs}
            input_result = detector.validate_stage_input(stage_name, input_data)
            
            if not input_result.is_valid:
                raise ValueError(f"Stage {stage_name} input validation failed: {input_result.errors}")
            
            # Execute function
            output = func(*args, **kwargs)
            
            # Validate output
            output_result = detector.validate_stage_output(stage_name, {"output": output})
            
            if not output_result.is_valid:
                raise ValueError(f"Stage {stage_name} output validation failed: {output_result.errors}")
            
            return output
        
        return wrapper
    return decorator