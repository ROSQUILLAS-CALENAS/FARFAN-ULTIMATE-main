"""
Comprehensive API Contract Schema Definitions
Provides strict field validation, deterministic ID generation, and schema validation decorators
for the classification evaluation calibration system.
"""

import json
import hashlib
# # # from typing import Dict, Any, List, Optional, Union, Callable  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from functools import wraps  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass  # Module not found  # Module not found  # Module not found

# # # from pydantic import BaseModel, Field, validator, ValidationError  # Module not found  # Module not found  # Module not found
# # # from pydantic.config import ConfigDict  # Module not found  # Module not found  # Module not found


class ResponseValue(str, Enum):
    """Enum-constrained response values for question evaluations."""
    SI = "Sí"
    PARCIAL = "Parcial"
    NO = "No"
    NI = "NI"  # No Information
    
    @classmethod
    def from_synonym(cls, value: str) -> "ResponseValue":
        """Map documented synonyms to canonical values."""
        synonym_map = {
            "yes": cls.SI,
            "si": cls.SI,
            "sí": cls.SI,
            "partial": cls.PARCIAL,
            "parcialmente": cls.PARCIAL,
            "no": cls.NO,
            "ni": cls.NI,
            "no_info": cls.NI,
            "no_information": cls.NI,
            "missing": cls.NI,
        }
        
        normalized = value.lower().strip()
        if normalized in synonym_map:
            return synonym_map[normalized]
        
        # Try direct enum match
        try:
            return cls(value)
        except ValueError:
            raise ValueError(
                f"Unrecognized response value '{value}'. "
                f"Valid values: {list(cls.__members__.values())} or synonyms: {list(synonym_map.keys())}"
            )


class DimensionId(str, Enum):
    """Enum for valid dimension identifiers."""
    DE_1 = "DE-1"
    DE_2 = "DE-2" 
    DE_3 = "DE-3"
    DE_4 = "DE-4"


class QuestionEvalInput(BaseModel):
    """Input schema for individual question evaluation with strict validation."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        frozen=True
    )
    
    # Required identification fields
    doc_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Required document identifier for traceability"
    )
    page_num: int = Field(
        ...,
        ge=1,
        description="Required page number (1-indexed) for evidence location"
    )
    question_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Unique question identifier"
    )
    dimension_id: DimensionId = Field(
        ...,
        description="Dimension classification (DE-1, DE-2, DE-3, DE-4)"
    )
    
    # Response data
    response: ResponseValue = Field(
        ...,
        description="Evaluated response value with enum constraints"
    )
    evidence_text: Optional[str] = Field(
        None,
        max_length=2048,
# # #         description="Supporting evidence text from document"  # Module not found  # Module not found  # Module not found
    )
    evidence_completeness: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Evidence completeness score (0.0 to 1.0)"
    )
    page_reference_quality: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Page reference quality score (0.0 to 1.0)"
    )
    
    # Metadata
    evaluator_id: Optional[str] = Field(
        None,
        max_length=128,
        description="Optional evaluator identifier for audit trail"
    )
    evaluation_timestamp: Optional[str] = Field(
        None,
        description="ISO 8601 timestamp of evaluation"
    )
    
    @validator('response', pre=True)
    def validate_response(cls, v):
        """Validate and normalize response values, including synonyms."""
        if isinstance(v, str):
            return ResponseValue.from_synonym(v)
        return v
    
    def generate_deterministic_id(self) -> str:
# # #         """Generate deterministic ID from sorted field values."""  # Module not found  # Module not found  # Module not found
        # Create stable field ordering for consistent hashing
        id_fields = {
            "doc_id": self.doc_id,
            "page_num": self.page_num,
            "question_id": self.question_id,
            "dimension_id": self.dimension_id.value
        }
        
        # Sort alphabetically for determinism
        sorted_content = json.dumps(id_fields, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(sorted_content.encode('utf-8')).hexdigest()[:16]


class DimensionEvalOutput(BaseModel):
    """Output schema for dimension-level evaluation results."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=True
    )
    
    # Required identification
    dimension_id: DimensionId = Field(
        ...,
        description="Dimension identifier"
    )
    doc_id: str = Field(
        ...,
        min_length=1,
        description="Source document identifier"
    )
    
    # Aggregated scores
    weighted_average: float = Field(
        ...,
        ge=0.0,
        le=1.2,  # Max with evidence multiplier
        description="Weighted average score for dimension"
    )
    base_score_average: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average base score without evidence multipliers"
    )
    evidence_quality_average: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average evidence quality across questions"
    )
    
    # Question details
    question_count: int = Field(
        ...,
        ge=1,
        description="Number of questions evaluated"
    )
    question_ids: List[str] = Field(
        ...,
        min_items=1,
        description="List of evaluated question identifiers"
    )
    
    # Response distribution
    response_distribution: Dict[ResponseValue, int] = Field(
        ...,
        description="Count of each response type"
    )
    
    # Validation metadata
    validation_passed: bool = Field(
        ...,
        description="Whether all validation checks passed"
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="List of validation error messages"
    )
    
    def generate_deterministic_id(self) -> str:
# # #         """Generate deterministic ID from dimension and document."""  # Module not found  # Module not found  # Module not found
        id_content = f"{self.dimension_id.value}:{self.doc_id}"
        return hashlib.sha256(id_content.encode('utf-8')).hexdigest()[:16]
    
    def to_sorted_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with deterministic field ordering."""
        data = self.model_dump()
        return dict(sorted(data.items()))


class PointEvalOutput(BaseModel):
    """Output schema for point-level evaluation results."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=True
    )
    
    # Required identification  
    point_id: int = Field(
        ...,
        ge=1,
        le=11,
        description="Decálogo point identifier (1-11)"
    )
    doc_id: str = Field(
        ...,
        min_length=1,
        description="Source document identifier"
    )
    
    # Final scores
    final_score: float = Field(
        ...,
        ge=0.0,
        le=1.2,
        description="Final composed point score"
    )
    normalized_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized score (0.0 to 1.0)"
    )
    
    # Dimension breakdown
    dimension_scores: Dict[DimensionId, float] = Field(
        ...,
        description="Score breakdown by dimension"
    )
    dimension_weights: Dict[DimensionId, float] = Field(
        ...,
        description="Weights used for dimension composition"
    )
    
    # Aggregated statistics
    total_questions: int = Field(
        ...,
        ge=1,
        description="Total questions evaluated across all dimensions"
    )
    overall_evidence_quality: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall evidence quality score"
    )
    
    # Validation metadata
    expected_question_count: int = Field(
        47,
        description="Expected number of questions per point"
    )
    validation_passed: bool = Field(
        ...,
        description="Whether validation checks passed"
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="List of validation error messages"
    )
    
    @validator('dimension_scores', 'dimension_weights')
    def validate_dimensions_complete(cls, v):
        """Ensure all required dimensions are present."""
        required_dims = {DimensionId.DE_1, DimensionId.DE_2, DimensionId.DE_3, DimensionId.DE_4}
        provided_dims = set(v.keys())
        
        if provided_dims != required_dims:
            missing = required_dims - provided_dims
            extra = provided_dims - required_dims
            errors = []
            if missing:
                errors.append(f"Missing dimensions: {[d.value for d in missing]}")
            if extra:
                errors.append(f"Extra dimensions: {[d.value for d in extra]}")
            raise ValueError("; ".join(errors))
        
        return v
    
    def generate_deterministic_id(self) -> str:
# # #         """Generate deterministic ID from point and document."""  # Module not found  # Module not found  # Module not found
        id_content = f"point_{self.point_id}:{self.doc_id}"
        return hashlib.sha256(id_content.encode('utf-8')).hexdigest()[:16]
    
    def to_sorted_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with deterministic field ordering."""
        data = self.model_dump()
        # Ensure nested dicts are also sorted
        if 'dimension_scores' in data:
            data['dimension_scores'] = dict(sorted(data['dimension_scores'].items()))
        if 'dimension_weights' in data:
            data['dimension_weights'] = dict(sorted(data['dimension_weights'].items()))
        return dict(sorted(data.items()))


class StageMeta(BaseModel):
    """Metadata schema for evaluation stage information."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=True
    )
    
    # Stage identification
    stage_name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Name of the evaluation stage"
    )
    stage_version: str = Field(
        ...,
        pattern=r'^\d+\.\d+\.\d+$',
        description="Semantic version of the stage (e.g., '1.0.0')"
    )
    
    # Processing metadata
    doc_id: str = Field(
        ...,
        min_length=1,
        description="Document identifier for this stage"
    )
    processing_timestamp: str = Field(
        ...,
        description="ISO 8601 timestamp of stage processing"
    )
    
    # Configuration
    configuration: Dict[str, Any] = Field(
        default_factory=dict,
        description="Stage-specific configuration parameters"
    )
    input_schema_version: str = Field(
        ...,
        description="Version of input schema used"
    )
    output_schema_version: str = Field(
        ...,
        description="Version of output schema produced"
    )
    
    # Quality metrics
    processing_duration_ms: Optional[int] = Field(
        None,
        ge=0,
        description="Processing duration in milliseconds"
    )
    memory_usage_mb: Optional[float] = Field(
        None,
        ge=0.0,
        description="Peak memory usage in MB"
    )
    
    def generate_deterministic_id(self) -> str:
# # #         """Generate deterministic ID from stage and document."""  # Module not found  # Module not found  # Module not found
        id_content = f"{self.stage_name}:{self.stage_version}:{self.doc_id}"
        return hashlib.sha256(id_content.encode('utf-8')).hexdigest()[:16]


@dataclass
class ValidationResult:
    """Result of schema validation with detailed error information."""
    is_valid: bool
    errors: List[str]
    validated_data: Optional[Any] = None
    schema_version: Optional[str] = None
    
    def raise_if_invalid(self):
        """Raise ValidationError if validation failed."""
        if not self.is_valid:
            raise ValidationError(self.errors)


def validate_input_schema(schema_class: BaseModel, strict: bool = True):
    """
    Decorator to validate inputs against schema before processing.
    Automatically rejects malformed data with detailed error messages.
    
    Args:
        schema_class: Pydantic schema class for validation
        strict: Whether to reject unrecognized fields (default: True)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Extract data parameter (first positional or named 'data')
            data = None
            if args:
                data = args[0]
# # #                 args = args[1:]  # Remove data from args  # Module not found  # Module not found  # Module not found
            elif 'data' in kwargs:
                data = kwargs.pop('data')
            
            if data is None:
                # Allow process methods to work without data parameter
                return func(self, *args, **kwargs)
            
            # Validate input data
            try:
                validated_data = schema_class.model_validate(data)
                # Replace data with validated instance
                return func(self, validated_data, *args, **kwargs)
                
            except ValidationError as e:
                # Format detailed error message for malformed data rejection
                error_details = []
                for error in e.errors():
                    field_path = " -> ".join(str(x) for x in error["loc"])
                    error_details.append(f"Field '{field_path}': {error['msg']}")
                
                detailed_msg = (
                    f"Input validation failed for {func.__name__} - "
                    f"Malformed data rejected with errors: {'; '.join(error_details)}"
                )
# # #                 raise ValidationError([detailed_msg]) from e  # Module not found  # Module not found  # Module not found
                
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper._schema_validated = True
        wrapper._input_schema = schema_class
        return wrapper
    return decorator


def validate_output_schema(schema_class: BaseModel, strict: bool = True):
    """
    Decorator to validate outputs against schema after processing.
    Ensures deterministic ID generation through stable field ordering.
    
    Args:
        schema_class: Pydantic schema class for validation
        strict: Whether to reject unrecognized fields (default: True)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Skip validation if result is already a schema instance
            if isinstance(result, BaseModel):
                if hasattr(result, 'to_sorted_dict'):
                    return result.to_sorted_dict()
                return result.model_dump()
            
            # Validate output data
            try:
                validated_result = schema_class.model_validate(result)
                # Return validated instance with sorted fields for determinism
                if hasattr(validated_result, 'to_sorted_dict'):
                    return validated_result.to_sorted_dict()
                return validated_result.model_dump()
                
            except ValidationError as e:
                # Format detailed error message for output validation failure
                error_details = []
                for error in e.errors():
                    field_path = " -> ".join(str(x) for x in error["loc"])
                    error_details.append(f"Field '{field_path}': {error['msg']}")
                
                detailed_msg = (
                    f"Output validation failed for {func.__name__} - "
                    f"Schema compliance error: {'; '.join(error_details)}"
                )
# # #                 raise ValidationError([detailed_msg]) from e  # Module not found  # Module not found  # Module not found
                
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper._schema_validated = True
        wrapper._output_schema = schema_class
        return wrapper
    return decorator


def validate_both_schemas(input_schema: BaseModel, output_schema: BaseModel):
    """
    Decorator to validate both inputs and outputs against schemas.
    
    Args:
        input_schema: Pydantic schema for input validation
        output_schema: Pydantic schema for output validation
    """
    def decorator(func: Callable) -> Callable:
        # Apply output validation first, then input validation
        return validate_input_schema(input_schema)(
            validate_output_schema(output_schema)(func)
        )
    return decorator


class DeterministicSortingMixin:
    """Mixin providing deterministic sorting utilities for consistent JSON serialization."""
    
    def to_deterministic_json(self) -> str:
        """Convert to JSON with deterministic field ordering."""
        if hasattr(self, 'to_sorted_dict'):
            data = self.to_sorted_dict()
        else:
            data = self.model_dump()
            data = self._deep_sort_dict(data)
        
        return json.dumps(data, sort_keys=True, separators=(',', ':'))
    
    def _deep_sort_dict(self, obj: Any) -> Any:
        """Recursively sort dictionary keys for deterministic serialization."""
        if isinstance(obj, dict):
            return {k: self._deep_sort_dict(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, list):
            return [self._deep_sort_dict(item) for item in obj]
        else:
            return obj


# Apply sorting mixin to schema classes
class QuestionEvalInputSorted(QuestionEvalInput, DeterministicSortingMixin):
    """QuestionEvalInput with deterministic sorting capabilities."""
    pass


class DimensionEvalOutputSorted(DimensionEvalOutput, DeterministicSortingMixin):
    """DimensionEvalOutput with deterministic sorting capabilities."""
    pass


class PointEvalOutputSorted(PointEvalOutput, DeterministicSortingMixin):
    """PointEvalOutput with deterministic sorting capabilities."""
    pass


class StageMetaSorted(StageMeta, DeterministicSortingMixin):
    """StageMeta with deterministic sorting capabilities."""
    pass


def reject_unknown_responses(response_value: str) -> ResponseValue:
    """
    Explicitly reject unrecognized response values unless they map to documented synonyms.
    
    Args:
        response_value: Raw response string to validate
        
    Returns:
        Validated ResponseValue enum
        
    Raises:
        ValueError: If response is not recognized and has no synonym mapping
    """
    try:
        return ResponseValue.from_synonym(response_value)
    except ValueError as e:
        # Re-raise with additional context about rejection policy
        raise ValueError(
            f"Response validation failed: {str(e)}. "
            "Unknown responses are explicitly rejected to ensure data integrity. "
            "Please use documented response values or add synonym mappings if needed."
# # #         ) from e  # Module not found  # Module not found  # Module not found


# Export all schema classes and utilities
__all__ = [
    'ResponseValue',
    'DimensionId', 
    'QuestionEvalInput',
    'DimensionEvalOutput',
    'PointEvalOutput',
    'StageMeta',
    'ValidationResult',
    'validate_input_schema',
    'validate_output_schema',
    'validate_both_schemas',
    'DeterministicSortingMixin',
    'QuestionEvalInputSorted',
    'DimensionEvalOutputSorted', 
    'PointEvalOutputSorted',
    'StageMetaSorted',
    'reject_unknown_responses'
]