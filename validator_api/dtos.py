"""
Data Transfer Objects for Validation Operations

Pure data structures for passing information between validation components.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum


class ValidationError(Exception):
    """Exception raised during validation operations."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


@dataclass
class ValidationRequest:
    """Request for validation operations."""
    data: Any
    validation_type: str
    rules: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationResponse:
    """Response from validation operations."""
    request_id: Optional[str]
    status: str
    success: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    confidence_score: Optional[float] = None
    processing_time_ms: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceMetadata:
    """Metadata for evidence items."""
    document_id: str
    title: str = ""
    author: str = ""
    publication_date: Optional[str] = None
    page_number: Optional[int] = None
    section_header: Optional[str] = None
    document_type: str = "document"
    url: Optional[str] = None
    confidence_level: Optional[str] = None
    source_hash: Optional[str] = None


@dataclass
class EvidenceItem:
    """Individual evidence item."""
    evidence_id: str
    text: str
    metadata: EvidenceMetadata
    context_before: str = ""
    context_after: str = ""
    start_position: int = 0
    end_position: int = 0
    evidence_type: str = "direct_quote"
    quality_score: float = 0.0
    processing_notes: List[str] = field(default_factory=list)


@dataclass
class EvidenceProcessingRequest:
    """Request for evidence processing operations."""
    raw_evidence: Any
    processing_type: str = "standard"
    validation_rules: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EvidenceProcessingResponse:
    """Response from evidence processing operations."""
    request_id: Optional[str]
    success: bool
    processed_evidence: List[EvidenceItem] = field(default_factory=list)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_evidence_count(self) -> int:
        """Get total number of processed evidence items."""
        return len(self.processed_evidence)
    
    def get_successful_evidence(self) -> List[EvidenceItem]:
        """Get evidence items that were successfully processed."""
        return [item for item in self.processed_evidence if item.quality_score > 0]


@dataclass
class BatchValidationRequest:
    """Request for batch validation of multiple items."""
    items: List[ValidationRequest]
    batch_id: Optional[str] = None
    parallel_processing: bool = True
    max_workers: int = 4
    fail_fast: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BatchValidationResponse:
    """Response from batch validation operations."""
    batch_id: Optional[str]
    results: List[ValidationResponse] = field(default_factory=list)
    total_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    processing_time_ms: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.successful_items / self.total_items) * 100.0


@dataclass
class ValidationConfiguration:
    """Configuration for validation operations."""
    rules: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)
    error_handling: Dict[str, str] = field(default_factory=dict)
    output_format: str = "detailed"
    enable_warnings: bool = True
    strict_mode: bool = False
    timeout_seconds: Optional[int] = None


@dataclass
class ProcessorConfiguration:
    """Configuration for evidence processing operations."""
    processing_rules: Dict[str, Any] = field(default_factory=dict)
    quality_thresholds: Dict[str, float] = field(default_factory=dict)
    metadata_extraction: Dict[str, bool] = field(default_factory=dict)
    output_format: str = "structured"
    enable_validation: bool = True
    max_evidence_per_batch: int = 100
    context_window_size: int = 200
