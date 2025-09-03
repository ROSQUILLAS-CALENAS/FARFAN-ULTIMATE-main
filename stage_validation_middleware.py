"""
Stage Validation Middleware for Pipeline Execution

This middleware intercepts pipeline execution at each stage boundary to validate 
input data before processing begins. It provides comprehensive validation including:

- Data integrity checks using checksums
- Schema compliance against stage-specific contracts  
- Required field presence and format verification
- Early failure detection with immediate pipeline halt
- Detailed logging of validation results and failures

Author: Pipeline Validation System
Date: December 2024
"""

import hashlib
import json
import logging
import time
# # # from abc import ABC, abstractmethod  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple  # Module not found  # Module not found  # Module not found


# Configure logging
logger = logging.getLogger(__name__)


class ValidationStatus(str, Enum):
    """Status codes for validation results."""
    
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class ValidationError(Exception):
    """Exception raised when critical validation fails."""
    pass


class SchemaValidationError(ValidationError):
    """Exception raised when schema validation fails."""
    pass


class IntegrityValidationError(ValidationError):
    """Exception raised when data integrity checks fail."""
    pass


@dataclass
class ValidationRule:
    """Defines a validation rule with its configuration."""
    
    name: str
    description: str
    validator_func: Callable[[Any], bool]
    required: bool = True
    error_message: str = ""
    warning_message: str = ""


@dataclass
class FieldSpec:
    """Specification for a required field in stage input data."""
    
    name: str
    field_type: type
    required: bool = True
    default_value: Any = None
    validation_pattern: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    allowed_values: Optional[List[Any]] = None


@dataclass
class StageSchema:
    """Schema definition for a pipeline stage."""
    
    stage_name: str
    version: str
    required_fields: List[FieldSpec]
    optional_fields: List[FieldSpec] = field(default_factory=list)
    validation_rules: List[ValidationRule] = field(default_factory=list)
    checksum_fields: List[str] = field(default_factory=list)
    
    
@dataclass
class ValidationResult:
    """Result of a validation check."""
    
    rule_name: str
    status: ValidationStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StageValidationReport:
    """Comprehensive validation report for a stage."""
    
    stage_name: str
    input_data_hash: str
    validation_results: List[ValidationResult] = field(default_factory=list)
    schema_violations: List[str] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    integrity_failures: List[str] = field(default_factory=list)
    overall_status: ValidationStatus = ValidationStatus.PASSED
    total_validation_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def has_failures(self) -> bool:
        """Check if any critical validation failed."""
        return (
            self.overall_status == ValidationStatus.FAILED or
            len(self.schema_violations) > 0 or
            len(self.missing_fields) > 0 or
            len(self.integrity_failures) > 0
        )
    
    def get_failure_summary(self) -> str:
        """Get a summary of all validation failures."""
        failures = []
        
        if self.schema_violations:
            failures.append(f"Schema violations: {', '.join(self.schema_violations)}")
        
        if self.missing_fields:
            failures.append(f"Missing fields: {', '.join(self.missing_fields)}")
            
        if self.integrity_failures:
            failures.append(f"Integrity failures: {', '.join(self.integrity_failures)}")
        
        failed_validations = [
            result.rule_name for result in self.validation_results 
            if result.status == ValidationStatus.FAILED
        ]
        
        if failed_validations:
            failures.append(f"Failed validations: {', '.join(failed_validations)}")
        
        return "; ".join(failures) if failures else "No failures detected"


class IntegrityChecker:
    """Handles data integrity verification using checksums."""
    
    @staticmethod
    def calculate_checksum(data: Any, algorithm: str = "sha256") -> str:
        """Calculate checksum for data using specified algorithm."""
        if isinstance(data, dict):
            # Sort keys for consistent hashing
            data_str = json.dumps(data, sort_keys=True, default=str)
        elif isinstance(data, (list, tuple)):
            data_str = json.dumps(sorted(data) if all(isinstance(x, (str, int, float)) for x in data) else list(data), default=str)
        else:
            data_str = str(data)
        
        data_bytes = data_str.encode('utf-8')
        
        if algorithm == "sha256":
            return hashlib.sha256(data_bytes).hexdigest()
        elif algorithm == "md5":
            return hashlib.md5(data_bytes).hexdigest()
        elif algorithm == "sha1":
            return hashlib.sha1(data_bytes).hexdigest()
        else:
            raise ValueError(f"Unsupported checksum algorithm: {algorithm}")
    
    @staticmethod
    def verify_checksum(data: Any, expected_checksum: str, algorithm: str = "sha256") -> bool:
        """Verify data integrity against expected checksum."""
        calculated_checksum = IntegrityChecker.calculate_checksum(data, algorithm)
        return calculated_checksum == expected_checksum
    
    @staticmethod
    def validate_field_integrity(data: Dict[str, Any], checksum_fields: List[str]) -> List[str]:
        """Validate integrity of specific fields using stored checksums."""
        integrity_failures = []
        
        for field_name in checksum_fields:
            if field_name not in data:
                integrity_failures.append(f"Missing checksum field: {field_name}")
                continue
            
            # Look for corresponding data field (remove _checksum suffix if present)
            data_field = field_name.replace('_checksum', '')
            if data_field not in data:
                integrity_failures.append(f"Missing data field for checksum: {data_field}")
                continue
            
            expected_checksum = data[field_name]
            actual_data = data[data_field]
            
            if not IntegrityChecker.verify_checksum(actual_data, expected_checksum):
                integrity_failures.append(f"Integrity check failed for field: {data_field}")
        
        return integrity_failures


class SchemaValidator:
    """Validates data against stage-specific schemas."""
    
    def __init__(self, schema: StageSchema):
        self.schema = schema
    
    def validate_field_spec(self, field_spec: FieldSpec, value: Any) -> List[str]:
        """Validate a single field against its specification."""
        errors = []
        
        # Type validation
        if not isinstance(value, field_spec.field_type):
            errors.append(f"Field '{field_spec.name}' expected {field_spec.field_type.__name__}, got {type(value).__name__}")
            return errors  # Don't continue if type is wrong
        
        # String-specific validations
        if field_spec.field_type == str and isinstance(value, str):
            if field_spec.min_length is not None and len(value) < field_spec.min_length:
                errors.append(f"Field '{field_spec.name}' too short (minimum {field_spec.min_length} chars)")
            
            if field_spec.max_length is not None and len(value) > field_spec.max_length:
                errors.append(f"Field '{field_spec.name}' too long (maximum {field_spec.max_length} chars)")
            
            if field_spec.validation_pattern:
                import re
                if not re.match(field_spec.validation_pattern, value):
                    errors.append(f"Field '{field_spec.name}' doesn't match required pattern")
        
        # Allowed values validation
        if field_spec.allowed_values and value not in field_spec.allowed_values:
            errors.append(f"Field '{field_spec.name}' value '{value}' not in allowed values: {field_spec.allowed_values}")
        
        return errors
    
    def validate_schema(self, data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate data against schema returning (schema_violations, missing_fields)."""
        schema_violations = []
        missing_fields = []
        
        # Check required fields
        for field_spec in self.schema.required_fields:
            if field_spec.name not in data:
                missing_fields.append(field_spec.name)
                continue
            
            field_errors = self.validate_field_spec(field_spec, data[field_spec.name])
            schema_violations.extend(field_errors)
        
        # Check optional fields if present
        for field_spec in self.schema.optional_fields:
            if field_spec.name in data:
                field_errors = self.validate_field_spec(field_spec, data[field_spec.name])
                schema_violations.extend(field_errors)
        
        return schema_violations, missing_fields


class StageValidationMiddleware:
    """
    Middleware that intercepts pipeline execution at stage boundaries to validate input data.
    
    Features:
    - Data integrity checks using checksums
    - Schema compliance validation against stage-specific contracts
    - Required field verification with format checking
    - Early failure detection with immediate pipeline halt
    - Comprehensive logging with detailed error messages
    """
    
    def __init__(self, 
                 halt_on_failure: bool = True,
                 log_level: int = logging.INFO,
                 validation_timeout_seconds: float = 30.0):
        """
        Initialize the stage validation middleware.
        
        Args:
            halt_on_failure: Whether to halt pipeline execution on validation failure
            log_level: Logging level for validation messages
            validation_timeout_seconds: Maximum time allowed for validation
        """
        self.halt_on_failure = halt_on_failure
        self.validation_timeout_seconds = validation_timeout_seconds
        self.stage_schemas: Dict[str, StageSchema] = {}
        self.validation_history: List[StageValidationReport] = []
        
        # Configure logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)
        
        # Add handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def register_stage_schema(self, schema: StageSchema) -> None:
        """Register a schema for a specific pipeline stage."""
        self.stage_schemas[schema.stage_name] = schema
        self.logger.info(f"Registered schema for stage: {schema.stage_name} (version {schema.version})")
    
    def validate_stage_input(self, 
                           stage_name: str, 
                           input_data: Any, 
                           context: Optional[Dict[str, Any]] = None) -> StageValidationReport:
        """
        Validate input data for a specific pipeline stage.
        
        Args:
            stage_name: Name of the pipeline stage
            input_data: Data to validate
            context: Optional context for validation
        
        Returns:
            StageValidationReport with validation results
            
        Raises:
            ValidationError: If critical validation fails and halt_on_failure is True
        """
        validation_start_time = time.time()
        
        # Calculate input data hash for integrity tracking
        input_hash = IntegrityChecker.calculate_checksum(input_data)
        
        # Initialize validation report
        report = StageValidationReport(
            stage_name=stage_name,
            input_data_hash=input_hash
        )
        
        self.logger.info(f"Starting validation for stage '{stage_name}' (data hash: {input_hash[:16]}...)")
        
        try:
            # Get schema for stage
            schema = self.stage_schemas.get(stage_name)
            if not schema:
                self.logger.warning(f"No schema registered for stage: {stage_name}")
                report.validation_results.append(ValidationResult(
                    rule_name="schema_availability",
                    status=ValidationStatus.WARNING,
                    message=f"No schema registered for stage: {stage_name}"
                ))
            else:
                # Perform schema validation
                self._validate_schema_compliance(report, schema, input_data)
                
                # Perform integrity validation
                self._validate_data_integrity(report, schema, input_data)
                
                # Execute custom validation rules
                self._execute_validation_rules(report, schema, input_data, context)
            
            # Calculate overall status
            self._determine_overall_status(report)
            
        except Exception as e:
            self.logger.error(f"Unexpected error during validation for stage '{stage_name}': {str(e)}")
            report.overall_status = ValidationStatus.FAILED
            report.schema_violations.append(f"Validation error: {str(e)}")
        
        finally:
            # Calculate total validation time
            validation_end_time = time.time()
            report.total_validation_time_ms = (validation_end_time - validation_start_time) * 1000
            
            # Store validation report
            self.validation_history.append(report)
            
            # Log validation results
            self._log_validation_results(report)
        
        # Handle validation failures
        if report.has_failures() and self.halt_on_failure:
            error_msg = f"Critical validation failure for stage '{stage_name}': {report.get_failure_summary()}"
            self.logger.error(error_msg)
            raise ValidationError(error_msg)
        
        return report
    
    def _validate_schema_compliance(self, 
                                  report: StageValidationReport, 
                                  schema: StageSchema, 
                                  input_data: Any) -> None:
        """Validate data against schema compliance."""
        validation_start = time.time()
        
        if not isinstance(input_data, dict):
            report.schema_violations.append(f"Input data must be a dictionary, got {type(input_data).__name__}")
            return
        
        validator = SchemaValidator(schema)
        schema_violations, missing_fields = validator.validate_schema(input_data)
        
        report.schema_violations.extend(schema_violations)
        report.missing_fields.extend(missing_fields)
        
        execution_time = (time.time() - validation_start) * 1000
        
        status = ValidationStatus.FAILED if (schema_violations or missing_fields) else ValidationStatus.PASSED
        report.validation_results.append(ValidationResult(
            rule_name="schema_compliance",
            status=status,
            message=f"Schema compliance check completed with {len(schema_violations)} violations, {len(missing_fields)} missing fields",
            details={
                "violations": schema_violations,
                "missing_fields": missing_fields
            },
            execution_time_ms=execution_time
        ))
    
    def _validate_data_integrity(self, 
                               report: StageValidationReport, 
                               schema: StageSchema, 
                               input_data: Dict[str, Any]) -> None:
        """Validate data integrity using checksums."""
        validation_start = time.time()
        
        if schema.checksum_fields:
            integrity_failures = IntegrityChecker.validate_field_integrity(
                input_data, schema.checksum_fields
            )
            report.integrity_failures.extend(integrity_failures)
            
            execution_time = (time.time() - validation_start) * 1000
            
            status = ValidationStatus.FAILED if integrity_failures else ValidationStatus.PASSED
            report.validation_results.append(ValidationResult(
                rule_name="data_integrity",
                status=status,
                message=f"Data integrity check completed with {len(integrity_failures)} failures",
                details={"failures": integrity_failures},
                execution_time_ms=execution_time
            ))
    
    def _execute_validation_rules(self, 
                                report: StageValidationReport, 
                                schema: StageSchema, 
                                input_data: Any, 
                                context: Optional[Dict[str, Any]]) -> None:
        """Execute custom validation rules."""
        for rule in schema.validation_rules:
            validation_start = time.time()
            
            try:
                # Execute validation rule
                is_valid = rule.validator_func(input_data)
                
                execution_time = (time.time() - validation_start) * 1000
                
                if is_valid:
                    status = ValidationStatus.PASSED
                    message = f"Validation rule '{rule.name}' passed"
                else:
                    status = ValidationStatus.FAILED if rule.required else ValidationStatus.WARNING
                    message = rule.error_message or f"Validation rule '{rule.name}' failed"
                    if not rule.required and rule.warning_message:
                        message = rule.warning_message
                
                report.validation_results.append(ValidationResult(
                    rule_name=rule.name,
                    status=status,
                    message=message,
                    details={"description": rule.description},
                    execution_time_ms=execution_time
                ))
                
            except Exception as e:
                execution_time = (time.time() - validation_start) * 1000
                
                report.validation_results.append(ValidationResult(
                    rule_name=rule.name,
                    status=ValidationStatus.FAILED,
                    message=f"Validation rule '{rule.name}' threw exception: {str(e)}",
                    details={"exception": str(e)},
                    execution_time_ms=execution_time
                ))
    
    def _determine_overall_status(self, report: StageValidationReport) -> None:
        """Determine overall validation status based on all checks."""
        if report.has_failures():
            report.overall_status = ValidationStatus.FAILED
        elif any(result.status == ValidationStatus.WARNING for result in report.validation_results):
            report.overall_status = ValidationStatus.WARNING
        else:
            report.overall_status = ValidationStatus.PASSED
    
    def _log_validation_results(self, report: StageValidationReport) -> None:
        """Log detailed validation results."""
        status_symbol = {
            ValidationStatus.PASSED: "✅",
            ValidationStatus.WARNING: "⚠️",
            ValidationStatus.FAILED: "❌",
            ValidationStatus.SKIPPED: "⏭️"
        }
        
        symbol = status_symbol.get(report.overall_status, "❓")
        
        self.logger.info(f"{symbol} Stage '{report.stage_name}' validation completed: {report.overall_status.value}")
        self.logger.info(f"   └── Validation time: {report.total_validation_time_ms:.2f}ms")
        self.logger.info(f"   └── Input data hash: {report.input_data_hash}")
        
        # Log specific validation results
        for result in report.validation_results:
            result_symbol = status_symbol.get(result.status, "❓")
            self.logger.info(f"   └── {result_symbol} {result.rule_name}: {result.message} ({result.execution_time_ms:.2f}ms)")
        
        # Log schema violations
        if report.schema_violations:
            self.logger.error(f"   └── Schema violations detected:")
            for violation in report.schema_violations:
                self.logger.error(f"       • {violation}")
        
        # Log missing fields
        if report.missing_fields:
            self.logger.error(f"   └── Missing required fields:")
            for field in report.missing_fields:
                self.logger.error(f"       • {field}")
        
        # Log integrity failures
        if report.integrity_failures:
            self.logger.error(f"   └── Data integrity failures:")
            for failure in report.integrity_failures:
                self.logger.error(f"       • {failure}")
        
        # Log failure summary if any failures occurred
        if report.has_failures():
            self.logger.error(f"   └── VALIDATION FAILED: {report.get_failure_summary()}")
    
    def intercept_stage_execution(self, 
                                stage_name: str, 
                                stage_executor: Callable[[Any], Any], 
                                input_data: Any, 
                                context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Intercept and validate stage execution.
        
        Args:
            stage_name: Name of the pipeline stage
            stage_executor: Function that executes the stage
            input_data: Input data for the stage
            context: Optional execution context
        
        Returns:
            Result of stage execution
            
        Raises:
            ValidationError: If validation fails and halt_on_failure is True
        """
        self.logger.info(f"Intercepting execution for stage: {stage_name}")
        
        # Validate input data before stage execution
        validation_report = self.validate_stage_input(stage_name, input_data, context)
        
        # Add validation report to context
        if context is None:
            context = {}
        context['validation_report'] = validation_report
        
        # Execute the stage if validation passed or warnings only
        if validation_report.overall_status in [ValidationStatus.PASSED, ValidationStatus.WARNING]:
            self.logger.info(f"Proceeding with stage execution: {stage_name}")
            return stage_executor(input_data)
        else:
            # This should only happen if halt_on_failure is False
            self.logger.warning(f"Stage execution may be unreliable due to validation failures: {stage_name}")
            return stage_executor(input_data)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of all validation activity."""
        total_validations = len(self.validation_history)
        passed = sum(1 for report in self.validation_history if report.overall_status == ValidationStatus.PASSED)
        warnings = sum(1 for report in self.validation_history if report.overall_status == ValidationStatus.WARNING)
        failed = sum(1 for report in self.validation_history if report.overall_status == ValidationStatus.FAILED)
        
        avg_validation_time = sum(report.total_validation_time_ms for report in self.validation_history) / max(1, total_validations)
        
        return {
            "total_validations": total_validations,
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
            "success_rate": (passed / max(1, total_validations)) * 100,
            "average_validation_time_ms": avg_validation_time,
            "registered_schemas": list(self.stage_schemas.keys()),
            "halt_on_failure": self.halt_on_failure
        }


# Helper functions for creating common validation rules and schemas

def create_data_completeness_rule(min_fields: int = 1) -> ValidationRule:
    """Create a validation rule that checks for data completeness."""
    def validator(data):
        if isinstance(data, dict):
            return len(data) >= min_fields
        elif isinstance(data, (list, tuple)):
            return len(data) >= min_fields
        else:
            return data is not None
    
    return ValidationRule(
        name="data_completeness",
        description=f"Ensures data contains at least {min_fields} fields/items",
        validator_func=validator,
        required=True,
        error_message=f"Data must contain at least {min_fields} fields/items"
    )


def create_non_empty_string_rule(field_name: str) -> ValidationRule:
    """Create a validation rule that checks for non-empty strings."""
    def validator(data):
        if isinstance(data, dict) and field_name in data:
            value = data[field_name]
            return isinstance(value, str) and len(value.strip()) > 0
        return False
    
    return ValidationRule(
        name=f"non_empty_{field_name}",
        description=f"Ensures {field_name} is a non-empty string",
        validator_func=validator,
        required=True,
        error_message=f"Field '{field_name}' must be a non-empty string"
    )


def create_basic_pipeline_schema(stage_name: str) -> StageSchema:
    """Create a basic schema for pipeline stages."""
    return StageSchema(
        stage_name=stage_name,
        version="1.0",
        required_fields=[
            FieldSpec(name="data", field_type=dict),
            FieldSpec(name="stage_id", field_type=str),
            FieldSpec(name="timestamp", field_type=str)
        ],
        optional_fields=[
            FieldSpec(name="metadata", field_type=dict),
            FieldSpec(name="context", field_type=dict)
        ],
        validation_rules=[
            create_data_completeness_rule(min_fields=3),
            create_non_empty_string_rule("stage_id")
        ],
        checksum_fields=["data_checksum"]
    )


# Example usage and testing functions

def create_sample_middleware() -> StageValidationMiddleware:
    """Create a sample middleware instance for testing."""
    middleware = StageValidationMiddleware(halt_on_failure=True)
    
    # Register sample schemas
    for stage in ["ingestion", "analysis", "synthesis"]:
        schema = create_basic_pipeline_schema(stage)
        middleware.register_stage_schema(schema)
    
    return middleware


def demo_validation_middleware():
    """Demonstrate the stage validation middleware."""
    print("=== Stage Validation Middleware Demo ===")
    
    # Create middleware
    middleware = create_sample_middleware()
    
    # Valid data sample
    valid_data = {
        "data": {"content": "sample content", "source": "test"},
        "stage_id": "ingestion_001",
        "timestamp": "2024-12-20T10:00:00Z",
        "metadata": {"version": "1.0"},
        "data_checksum": IntegrityChecker.calculate_checksum({"content": "sample content", "source": "test"})
    }
    
    # Test valid data
    print("\n1. Testing valid data:")
    try:
        report = middleware.validate_stage_input("ingestion", valid_data)
        print(f"   Status: {report.overall_status}")
        print(f"   Validation time: {report.total_validation_time_ms:.2f}ms")
    except ValidationError as e:
        print(f"   Validation failed: {e}")
    
    # Invalid data sample
    invalid_data = {
        "data": {},  # Empty data
        "stage_id": "",  # Empty string
        # Missing timestamp
        "data_checksum": "invalid_checksum"
    }
    
    # Test invalid data
    print("\n2. Testing invalid data:")
    try:
        report = middleware.validate_stage_input("ingestion", invalid_data)
        print(f"   Status: {report.overall_status}")
    except ValidationError as e:
        print(f"   Validation failed: {e}")
    
    # Print summary
    print("\n3. Validation summary:")
    summary = middleware.get_validation_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demo
    demo_validation_middleware()