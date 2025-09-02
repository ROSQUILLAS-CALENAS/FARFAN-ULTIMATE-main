"""
Contracts Package
================

This package provides standardized schemas and validation for pipeline components.
"""

from .schemas import (
    # Enums
    SchemaVersion, ProcessingStatus, EvaluationLevel, ComplianceStatus, ConfidenceLevel,
    # Schemas
    BaseSchema, RequiredDocumentFields, QuestionEvalInput, DimensionEvalOutput, 
    PointEvalOutput, StageMeta,
    # Decorators
    validate_input_schema, validate_output_schema, validate_process_schemas, enforce_required_fields,
    # Registry
    SchemaRegistry, registry,
    # Utilities
    create_stage_meta, validate_pipeline_data,
    # Exceptions
    SchemaValidationError,
)

__all__ = [
    # Enums
    'SchemaVersion', 'ProcessingStatus', 'EvaluationLevel', 'ComplianceStatus', 'ConfidenceLevel',
    # Schemas
    'BaseSchema', 'RequiredDocumentFields', 'QuestionEvalInput', 'DimensionEvalOutput', 
    'PointEvalOutput', 'StageMeta',
    # Decorators
    'validate_input_schema', 'validate_output_schema', 'validate_process_schemas', 'enforce_required_fields',
    # Registry
    'SchemaRegistry', 'registry',
    # Utilities
    'create_stage_meta', 'validate_pipeline_data',
    # Exceptions
    'SchemaValidationError',
]