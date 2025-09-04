"""
Data Transfer Objects for Anti-Corruption Layer

These DTOs define the interface contracts between retrieval and analysis phases.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass 
class RetrievalOutputDTO:
    """Standard output format from retrieval components"""
    query_id: str
    retrieved_chunks: List[Dict[str, Any]]
    similarity_scores: List[float]
    retrieval_metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    schema_version: str = "1.0.0"


@dataclass
class AnalysisInputDTO:
    """Standard input format for analysis components"""
    document_chunks: List[Dict[str, Any]]
    context: Dict[str, Any]
    processing_metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    schema_version: str = "1.0.0"


@dataclass
class SchemaMismatchEvent:
    """Event data for schema mismatches"""
    source_schema: str
    target_schema: str
    source_data: Dict[str, Any]
    mismatch_details: List[str]
    adapter_id: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass 
class LineageEvent:
    """Event data for dependency tracking"""
    component_id: str
    operation_type: str
    input_schema: str
    output_schema: str
    dependencies: List[str]
    violation_type: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)