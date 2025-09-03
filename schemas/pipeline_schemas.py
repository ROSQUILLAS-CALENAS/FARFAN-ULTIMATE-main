"""
Pipeline Stage Schema Definitions

This module defines the expected schemas for data passed between critical pipeline
stages to ensure data integrity and consistency across handoffs.
"""

# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Union  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
import json


class StageType(str, Enum):
    """Pipeline stage types"""
    INGESTION_PREPARATION = "I_ingestion_preparation"
    CONTEXT_ESTABLISHMENT = "C_context_establishment"  
    KNOWLEDGE_EXTRACTION = "K_knowledge_extraction"
    ANALYSIS_NLP = "A_analysis_nlp"
    CLASSIFICATION_EVALUATION = "L_classification_evaluation"
    SEARCH_RETRIEVAL = "S_search_retrieval"


class DataIntegrityLevel(str, Enum):
    """Data integrity validation levels"""
    STRICT = "strict"        # All fields required, strict type checking
    MODERATE = "moderate"    # Required fields must exist, flexible types
    RELAXED = "relaxed"      # Basic validation only


@dataclass
class DocumentMetadata:
    """Core document metadata required across all stages"""
    doc_id: str
    source_path: str
    content_hash: str
    created_timestamp: datetime
    stage_processed: str
    processing_version: str = "1.0"
    content_length: int = 0
    language: Optional[str] = None
    document_type: Optional[str] = None


@dataclass
class ProcessingMetrics:
    """Processing metrics for stage performance tracking"""
    stage_id: str
    start_time: datetime
    end_time: datetime
    processing_duration_seconds: float
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    error_count: int = 0
    warning_count: int = 0


@dataclass
class ValidationError:
    """Validation error details"""
    field_name: str
    error_type: str
    error_message: str
    severity: str = "error"  # error, warning, info
    suggested_fix: Optional[str] = None


@dataclass
class IngestionOutput:
    """Schema for I_ingestion_preparation stage output"""
    metadata: DocumentMetadata
    extracted_text: str
    document_structure: Dict[str, Any]
    page_count: int
    extraction_confidence: float
    raw_features: Dict[str, Any] = field(default_factory=dict)
    processing_notes: List[str] = field(default_factory=list)
    
    # Required for handoff to C_context_establishment
    content_blocks: List[Dict[str, Any]] = field(default_factory=list)
    semantic_markers: List[str] = field(default_factory=list)


@dataclass
class ContextOutput:
    """Schema for C_context_establishment stage output"""
    metadata: DocumentMetadata
    context_graph: Dict[str, Any]
    entity_references: List[Dict[str, Any]]
    relationship_mappings: Dict[str, Any]
    contextual_embeddings: Optional[List[float]] = None
    context_confidence: float = 0.0
    
# # #     # Must preserve from ingestion  # Module not found  # Module not found  # Module not found
    extracted_text: str = ""
    document_structure: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class KnowledgeOutput:
    """Schema for K_knowledge_extraction stage output"""
    metadata: DocumentMetadata
    knowledge_entities: List[Dict[str, Any]]
    concept_relations: Dict[str, Any]
    factual_claims: List[Dict[str, Any]]
    knowledge_graph: Dict[str, Any] = field(default_factory=dict)
    extraction_accuracy: float = 0.0
    
    # Required for handoff to A_analysis_nlp
    linguistic_features: Dict[str, Any] = field(default_factory=dict)
    semantic_vectors: List[float] = field(default_factory=list)


@dataclass
class AnalysisOutput:
    """Schema for A_analysis_nlp stage output"""
    metadata: DocumentMetadata
    sentiment_analysis: Dict[str, Any]
    topic_classification: List[Dict[str, Any]]
    named_entities: List[Dict[str, Any]]
    linguistic_patterns: Dict[str, Any]
    analysis_confidence: float = 0.0
    
# # #     # Must preserve from knowledge extraction  # Module not found  # Module not found  # Module not found
    knowledge_entities: List[Dict[str, Any]] = field(default_factory=list)
    concept_relations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassificationOutput:
    """Schema for L_classification_evaluation stage output"""
    metadata: DocumentMetadata
    classification_labels: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    evaluation_metrics: Dict[str, float]
    model_predictions: List[Dict[str, Any]]
    
    # Required for handoff to S_search_retrieval
    search_keywords: List[str] = field(default_factory=list)
    retrieval_vectors: List[float] = field(default_factory=list)
    relevance_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalOutput:
    """Schema for S_search_retrieval stage output"""
    metadata: DocumentMetadata
    search_results: List[Dict[str, Any]]
    retrieval_rankings: List[Dict[str, Any]]
    similarity_scores: Dict[str, float]
    retrieved_documents: List[str] = field(default_factory=list)
    
    # Must preserve classification data
    classification_labels: List[Dict[str, Any]] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)


# Schema mappings for critical handoff points
HANDOFF_SCHEMAS = {
    (StageType.INGESTION_PREPARATION, StageType.CONTEXT_ESTABLISHMENT): {
        "input_schema": IngestionOutput,
        "output_schema": ContextOutput,
        "required_fields": ["metadata", "extracted_text", "document_structure", "content_blocks"],
        "cross_stage_dependencies": ["doc_id", "content_hash"]
    },
    (StageType.KNOWLEDGE_EXTRACTION, StageType.ANALYSIS_NLP): {
        "input_schema": KnowledgeOutput, 
        "output_schema": AnalysisOutput,
        "required_fields": ["metadata", "knowledge_entities", "linguistic_features", "semantic_vectors"],
        "cross_stage_dependencies": ["doc_id", "knowledge_entities"]
    },
    (StageType.CLASSIFICATION_EVALUATION, StageType.SEARCH_RETRIEVAL): {
        "input_schema": ClassificationOutput,
        "output_schema": RetrievalOutput, 
        "required_fields": ["metadata", "classification_labels", "search_keywords", "retrieval_vectors"],
        "cross_stage_dependencies": ["doc_id", "classification_labels", "confidence_scores"]
    }
}


@dataclass
class StageHandoffValidationResult:
    """Result of stage handoff validation"""
    source_stage: StageType
    target_stage: StageType
    is_valid: bool
    validation_errors: List[ValidationError] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    schema_violations: List[str] = field(default_factory=list)
    cross_stage_dependency_errors: List[str] = field(default_factory=list)
    validation_timestamp: datetime = field(default_factory=datetime.now)
    data_integrity_level: DataIntegrityLevel = DataIntegrityLevel.STRICT


def serialize_stage_output(output_obj: Union[IngestionOutput, ContextOutput, KnowledgeOutput, 
                                           AnalysisOutput, ClassificationOutput, RetrievalOutput]) -> Dict[str, Any]:
    """Serialize stage output objects to JSON-compatible dictionaries"""
    if hasattr(output_obj, '__dict__'):
        result = {}
        for key, value in output_obj.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif hasattr(value, '__dict__'):  # Nested dataclass
                result[key] = serialize_stage_output(value)
            else:
                result[key] = value
        return result
    return output_obj


def deserialize_stage_output(data: Dict[str, Any], output_type: type) -> Any:
    """Deserialize dictionary back to stage output objects"""
    # Handle datetime fields
    for key, value in data.items():
        if key.endswith('_timestamp') and isinstance(value, str):
            try:
                data[key] = datetime.fromisoformat(value)
            except ValueError:
                pass  # Keep as string if parsing fails
    
    # Handle metadata separately if it exists
    if 'metadata' in data and isinstance(data['metadata'], dict):
        metadata_dict = data['metadata']
        # Convert timestamp fields in metadata
        for ts_field in ['created_timestamp']:
            if ts_field in metadata_dict and isinstance(metadata_dict[ts_field], str):
                try:
                    metadata_dict[ts_field] = datetime.fromisoformat(metadata_dict[ts_field])
                except ValueError:
                    pass
        data['metadata'] = DocumentMetadata(**metadata_dict)
    
    return output_type(**data)