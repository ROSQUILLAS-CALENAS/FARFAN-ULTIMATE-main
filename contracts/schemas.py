"""
Standardized Message Schema Classes
===================================

This module provides Pydantic validation schemas for pipeline stage contracts,
ensuring data consistency and API compliance across all pipeline components.

Key Features:
- Required doc_id and page_num fields on all data structures
- Strict validation rejecting unknown response values
- Deterministic ID generation using stable field sorting
- Schema versioning support
- Runtime contract enforcement decorators
"""

import hashlib
import inspect
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Union
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field, validator, root_validator
from pydantic import __version__ as pydantic_version

# Check Pydantic version for compatibility
PYDANTIC_V2 = pydantic_version.startswith("2.")

if PYDANTIC_V2:
    from pydantic import field_validator, model_validator
else:
    # For Pydantic v1 compatibility
    field_validator = validator
    model_validator = root_validator


class SchemaVersion(str, Enum):
    """Schema versioning enumeration"""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"


class ProcessingStatus(str, Enum):
    """Standard processing status values"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    PENDING = "pending"
    SKIPPED = "skipped"


class EvaluationLevel(str, Enum):
    """Standard evaluation levels"""
    DIMENSION = "dimension"
    POINT = "point"
    STAGE = "stage"
    GLOBAL = "global"


class ComplianceStatus(str, Enum):
    """Standard compliance status values"""
    CUMPLE = "CUMPLE"
    CUMPLE_PARCIAL = "CUMPLE_PARCIAL"
    NO_CUMPLE = "NO_CUMPLE"
    NO_EVALUADO = "NO_EVALUADO"


class ConfidenceLevel(str, Enum):
    """Standard confidence levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


# Base Schema Mixins
class BaseSchema(BaseModel):
    """Base schema with common fields and methods"""
    
    schema_version: SchemaVersion = Field(default=SchemaVersion.V2_0, description="Schema version")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    class Config:
        # Pydantic v1/v2 compatibility
        if PYDANTIC_V2:
            extra = "forbid"  # Strict rejection of unknown fields
            str_strip_whitespace = True
            validate_assignment = True
        else:
            extra = "forbid"
            anystr_strip_whitespace = True
            validate_assignment = True
            
    def generate_deterministic_id(self, prefix: str = "") -> str:
        """Generate deterministic ID from stable field sorting"""
        # Get all fields and sort them for deterministic ordering
        field_dict = self.dict(exclude={'created_at', 'schema_version'})
        sorted_items = sorted(field_dict.items(), key=lambda x: str(x[0]))
        
        # Create hash from sorted field representation
        hash_input = f"{prefix}:{sorted_items}"
        return f"{prefix}{hashlib.sha256(hash_input.encode()).hexdigest()[:16]}"


class RequiredDocumentFields(BaseModel):
    """Mixin for required document identification fields"""
    
    doc_id: str = Field(..., min_length=1, description="Required document identifier")
    page_num: int = Field(..., ge=1, description="Required page number (1-indexed)")
    
    @field_validator('doc_id') if PYDANTIC_V2 else validator('doc_id')
    @classmethod
    def validate_doc_id(cls, v):
        if not v or not v.strip():
            raise ValueError("doc_id cannot be empty or whitespace-only")
        return v.strip()
    
    @field_validator('page_num') if PYDANTIC_V2 else validator('page_num')
    @classmethod
    def validate_page_num(cls, v):
        if v < 1:
            raise ValueError("page_num must be >= 1")
        return v


# Main Schema Classes
class QuestionEvalInput(BaseSchema, RequiredDocumentFields):
    """Input schema for question evaluation pipeline stages"""
    
    question_text: str = Field(..., min_length=1, description="Question text to evaluate")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    evaluation_criteria: Optional[List[str]] = Field(default_factory=list, description="Evaluation criteria")
    priority: Optional[int] = Field(default=1, ge=1, le=10, description="Processing priority")
    
    @field_validator('question_text') if PYDANTIC_V2 else validator('question_text')
    @classmethod
    def validate_question_text(cls, v):
        if not v or not v.strip():
            raise ValueError("question_text cannot be empty")
        return v.strip()
    
    def get_deterministic_id(self) -> str:
        """Generate deterministic ID for this question evaluation input"""
        return self.generate_deterministic_id("qe_")


class DimensionEvalOutput(BaseSchema, RequiredDocumentFields):
    """Output schema for dimension evaluation results"""
    
    dimension_id: str = Field(..., regex=r'^DE[1-4]$', description="Dimension identifier (DE1-DE4)")
    dimension_name: str = Field(..., min_length=1, description="Human readable dimension name")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized dimension score [0,1]")
    compliance_status: ComplianceStatus = Field(..., description="Compliance status")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence in the evaluation")
    evidence_count: int = Field(..., ge=0, description="Number of evidence items used")
    sub_scores: Optional[Dict[str, float]] = Field(default_factory=dict, description="Sub-dimension scores")
    recommendations: Optional[List[str]] = Field(default_factory=list, description="Improvement recommendations")
    processing_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Processing metadata")
    
    @field_validator('dimension_id') if PYDANTIC_V2 else validator('dimension_id')
    @classmethod
    def validate_dimension_id(cls, v):
        valid_dimensions = {'DE1', 'DE2', 'DE3', 'DE4'}
        if v not in valid_dimensions:
            raise ValueError(f"dimension_id must be one of {valid_dimensions}")
        return v
    
    @field_validator('score') if PYDANTIC_V2 else validator('score')
    @classmethod
    def validate_score(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("score must be between 0.0 and 1.0")
        return v
    
    def get_deterministic_id(self) -> str:
        """Generate deterministic ID for this dimension evaluation"""
        return self.generate_deterministic_id("de_")


class PointEvalOutput(BaseSchema, RequiredDocumentFields):
    """Output schema for Decálogo point evaluation results"""
    
    point_id: str = Field(..., regex=r'^P([1-9]|10)$', description="Decálogo point identifier (P1-P10)")
    point_title: str = Field(..., min_length=1, description="Human rights point title")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized point score [0,1]")
    compliance_status: ComplianceStatus = Field(..., description="Compliance status")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence in the evaluation")
    evidence_count: int = Field(..., ge=0, description="Number of evidence items used")
    dimension_alignment: Optional[str] = Field(default=None, regex=r'^DE[1-4]$', description="Primary dimension alignment")
    key_findings: Optional[List[str]] = Field(default_factory=list, description="Key evaluation findings")
    gap_analysis: Optional[List[str]] = Field(default_factory=list, description="Identified gaps")
    processing_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Processing metadata")
    
    @field_validator('point_id') if PYDANTIC_V2 else validator('point_id')
    @classmethod
    def validate_point_id(cls, v):
        valid_points = {f'P{i}' for i in range(1, 11)}
        if v not in valid_points:
            raise ValueError(f"point_id must be one of {valid_points}")
        return v
    
    @field_validator('score') if PYDANTIC_V2 else validator('score')
    @classmethod
    def validate_score(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("score must be between 0.0 and 1.0")
        return v
    
    def get_deterministic_id(self) -> str:
        """Generate deterministic ID for this point evaluation"""
        return self.generate_deterministic_id("pe_")


class StageMeta(BaseSchema):
    """Metadata schema for pipeline stage processing"""
    
    stage_name: str = Field(..., min_length=1, description="Pipeline stage name")
    stage_version: str = Field(..., min_length=1, description="Stage implementation version")
    processing_status: ProcessingStatus = Field(..., description="Processing status")
    execution_time_ms: Optional[float] = Field(default=None, ge=0, description="Execution time in milliseconds")
    input_schema_version: Optional[str] = Field(default=None, description="Input schema version used")
    output_schema_version: Optional[str] = Field(default=None, description="Output schema version produced")
    error_details: Optional[Dict[str, Any]] = Field(default=None, description="Error details if failed")
    performance_metrics: Optional[Dict[str, float]] = Field(default_factory=dict, description="Performance metrics")
    resource_usage: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Resource usage stats")
    
    @field_validator('stage_name') if PYDANTIC_V2 else validator('stage_name')
    @classmethod
    def validate_stage_name(cls, v):
        if not v or not v.strip():
            raise ValueError("stage_name cannot be empty")
        return v.strip()
    
    @field_validator('execution_time_ms') if PYDANTIC_V2 else validator('execution_time_ms')
    @classmethod
    def validate_execution_time(cls, v):
        if v is not None and v < 0:
            raise ValueError("execution_time_ms must be >= 0")
        return v
    
    def get_deterministic_id(self) -> str:
        """Generate deterministic ID for this stage metadata"""
        return self.generate_deterministic_id("sm_")


# Schema Validation Decorators
class SchemaValidationError(Exception):
    """Custom exception for schema validation errors"""
    pass


def validate_input_schema(input_schema: type):
    """Decorator to validate input data against schema"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Find data parameter in function signature
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            # Try to find data parameter
            data = None
            if 'data' in kwargs:
                data = kwargs['data']
            elif len(args) >= 2 and 'data' in params:  # Assuming self, data pattern
                data = args[1]
            elif len(args) >= 1 and params[0] == 'data':  # Function with data as first param
                data = args[0]
            
            if data is not None:
                try:
                    # Validate input data
                    if not isinstance(data, input_schema):
                        validated_data = input_schema(**data) if isinstance(data, dict) else input_schema.parse_obj(data)
                        # Replace data with validated version
                        if 'data' in kwargs:
                            kwargs['data'] = validated_data
                        elif len(args) >= 2 and 'data' in params:
                            args = list(args)
                            args[1] = validated_data
                            args = tuple(args)
                        elif len(args) >= 1 and params[0] == 'data':
                            args = list(args)
                            args[0] = validated_data
                            args = tuple(args)
                except Exception as e:
                    raise SchemaValidationError(f"Input validation failed: {e}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_output_schema(output_schema: type):
    """Decorator to validate output data against schema"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Validate output
            try:
                if not isinstance(result, output_schema):
                    validated_result = output_schema(**result) if isinstance(result, dict) else output_schema.parse_obj(result)
                    return validated_result
                return result
            except Exception as e:
                raise SchemaValidationError(f"Output validation failed: {e}")
        
        return wrapper
    return decorator


def validate_process_schemas(input_schema: type = None, output_schema: type = None):
    """Decorator to validate both input and output schemas for process methods"""
    def decorator(func: Callable) -> Callable:
        # Apply input validation if specified
        if input_schema:
            func = validate_input_schema(input_schema)(func)
        
        # Apply output validation if specified
        if output_schema:
            func = validate_output_schema(output_schema)(func)
        
        return func
    return decorator


def enforce_required_fields(*required_fields: str):
    """Decorator to enforce required fields in data parameter"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Find data parameter
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            data = None
            if 'data' in kwargs:
                data = kwargs['data']
            elif len(args) >= 2 and 'data' in params:
                data = args[1]
            elif len(args) >= 1 and params[0] == 'data':
                data = args[0]
            
            if data is not None and isinstance(data, dict):
                missing_fields = []
                for field in required_fields:
                    if field not in data or data[field] is None or data[field] == "":
                        missing_fields.append(field)
                
                if missing_fields:
                    raise SchemaValidationError(f"Missing required fields: {', '.join(missing_fields)}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Schema Registry for Version Management
class SchemaRegistry:
    """Registry for managing schema versions and compatibility"""
    
    def __init__(self):
        self._schemas = {}
        self._compatibility_matrix = {}
    
    def register_schema(self, name: str, version: SchemaVersion, schema_class: type):
        """Register a schema class with version"""
        key = f"{name}:{version.value}"
        self._schemas[key] = schema_class
    
    def get_schema(self, name: str, version: SchemaVersion) -> type:
        """Get schema class by name and version"""
        key = f"{name}:{version.value}"
        return self._schemas.get(key)
    
    def register_compatibility(self, from_version: SchemaVersion, to_version: SchemaVersion, compatible: bool = True):
        """Register version compatibility"""
        self._compatibility_matrix[(from_version.value, to_version.value)] = compatible
    
    def is_compatible(self, from_version: SchemaVersion, to_version: SchemaVersion) -> bool:
        """Check if versions are compatible"""
        return self._compatibility_matrix.get((from_version.value, to_version.value), False)


# Global schema registry instance
registry = SchemaRegistry()

# Register default schemas
registry.register_schema("QuestionEvalInput", SchemaVersion.V2_0, QuestionEvalInput)
registry.register_schema("DimensionEvalOutput", SchemaVersion.V2_0, DimensionEvalOutput)
registry.register_schema("PointEvalOutput", SchemaVersion.V2_0, PointEvalOutput)
registry.register_schema("StageMeta", SchemaVersion.V2_0, StageMeta)

# Register compatibility rules
registry.register_compatibility(SchemaVersion.V1_0, SchemaVersion.V1_1, True)
registry.register_compatibility(SchemaVersion.V1_1, SchemaVersion.V2_0, True)


# Utility Functions
def create_stage_meta(stage_name: str, processing_status: ProcessingStatus, **kwargs) -> StageMeta:
    """Utility function to create StageMeta instances"""
    return StageMeta(
        stage_name=stage_name,
        stage_version=kwargs.get('stage_version', '1.0'),
        processing_status=processing_status,
        **{k: v for k, v in kwargs.items() if k != 'stage_version'}
    )


def validate_pipeline_data(data: Dict[str, Any], required_schemas: Dict[str, type]) -> Dict[str, Any]:
    """Validate multiple data items against their respective schemas"""
    validated = {}
    errors = []
    
    for key, schema_class in required_schemas.items():
        if key in data:
            try:
                validated[key] = schema_class(**data[key]) if isinstance(data[key], dict) else schema_class.parse_obj(data[key])
            except Exception as e:
                errors.append(f"{key}: {e}")
        else:
            errors.append(f"Missing required data key: {key}")
    
    if errors:
        raise SchemaValidationError(f"Pipeline data validation failed: {'; '.join(errors)}")
    
    return validated


# Export all public classes and functions
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