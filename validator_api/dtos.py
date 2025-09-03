"""
Data Transfer Objects for validator operations

These DTOs provide concrete implementations of the validation data structures
without importing any pipeline components.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium" 
    LOW = "low"
    INFO = "info"


class ValidationCategory(str, Enum):
    """Categories of validation checks"""
    FACTUAL_ACCURACY = "factual_accuracy"
    LOGICAL_CONSISTENCY = "logical_consistency"
    SOURCE_RELIABILITY = "source_reliability"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    BIAS_DETECTION = "bias_detection"


class DNPAlignmentCategory(str, Enum):
    """DNP alignment categories"""
    CONSTITUTIONAL = "constitutional"
    REGULATORY = "regulatory"
    PROCEDURAL = "procedural"
    ETHICAL = "ethical"
    TECHNICAL = "technical"


@dataclass(frozen=True)
class ValidationMetrics:
    """Metrics for validation operations"""
    
    accuracy_score: float = 0.0
    confidence_score: float = 0.0
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    relevance_score: float = 0.0
    processing_time_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "accuracy_score": self.accuracy_score,
            "confidence_score": self.confidence_score,
            "completeness_score": self.completeness_score,
            "consistency_score": self.consistency_score,
            "relevance_score": self.relevance_score,
            "processing_time_ms": self.processing_time_ms
        }


@dataclass(frozen=True)
class EvidenceItem:
    """Structured evidence item"""
    
    id: str
    content: str
    source: str
    evidence_type: str
    confidence_level: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            object.__setattr__(self, 'created_at', datetime.utcnow())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "evidence_type": self.evidence_type,
            "confidence_level": self.confidence_level,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvidenceItem':
        """Create from dictionary representation"""
        created_at = None
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])
        
        return cls(
            id=data["id"],
            content=data["content"],
            source=data["source"],
            evidence_type=data["evidence_type"],
            confidence_level=data.get("confidence_level", 0.0),
            metadata=data.get("metadata", {}),
            created_at=created_at
        )


@dataclass
class ValidationRequest:
    """Request for validation operations"""
    
    evidence_text: str
    context: str = ""
    validation_type: str = "comprehensive"
    validation_categories: List[ValidationCategory] = field(default_factory=list)
    dnp_alignment_categories: List[DNPAlignmentCategory] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.request_id is None:
            # Generate deterministic ID based on content
            content_hash = hashlib.sha256(
                f"{self.evidence_text}{self.context}{self.validation_type}".encode()
            ).hexdigest()[:12]
            self.request_id = f"req_{content_hash}"
        
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        
        # Set default validation categories if none provided
        if not self.validation_categories:
            self.validation_categories = [
                ValidationCategory.FACTUAL_ACCURACY,
                ValidationCategory.LOGICAL_CONSISTENCY,
                ValidationCategory.SOURCE_RELIABILITY
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "evidence_text": self.evidence_text,
            "context": self.context,
            "validation_type": self.validation_type,
            "validation_categories": [cat.value for cat in self.validation_categories],
            "dnp_alignment_categories": [cat.value for cat in self.dnp_alignment_categories],
            "config": self.config,
            "request_id": self.request_id,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class ValidationResult:
    """Result of a validation operation"""
    
    is_valid: bool
    severity: ValidationSeverity
    messages: List[str] = field(default_factory=list)
    category: Optional[ValidationCategory] = None
    dnp_alignment_category: Optional[DNPAlignmentCategory] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "is_valid": self.is_valid,
            "severity": self.severity.value,
            "messages": self.messages,
            "category": self.category.value if self.category else None,
            "dnp_alignment_category": self.dnp_alignment_category.value if self.dnp_alignment_category else None,
            "metadata": self.metadata
        }


@dataclass
class ValidationResponse:
    """Response from validation operations"""
    
    request_id: str
    validation_results: List[ValidationResult] = field(default_factory=list)
    overall_validation_result: Optional[ValidationResult] = None
    confidence_score: float = 0.0
    metrics: Optional[ValidationMetrics] = None
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        
        # Calculate overall validation if not provided
        if self.overall_validation_result is None and self.validation_results:
            self._calculate_overall_validation()
    
    def _calculate_overall_validation(self):
        """Calculate overall validation result from individual results"""
        if not self.validation_results:
            return
        
        # Overall is valid only if all validations pass
        overall_valid = all(result.is_valid for result in self.validation_results)
        
        # Get highest severity level
        severities = [result.severity for result in self.validation_results]
        severity_order = [
            ValidationSeverity.CRITICAL,
            ValidationSeverity.HIGH,
            ValidationSeverity.MEDIUM,
            ValidationSeverity.LOW,
            ValidationSeverity.INFO
        ]
        
        overall_severity = ValidationSeverity.INFO
        for sev in severity_order:
            if sev in severities:
                overall_severity = sev
                break
        
        # Combine all messages
        all_messages = []
        for result in self.validation_results:
            all_messages.extend(result.messages)
        
        self.overall_validation_result = ValidationResult(
            is_valid=overall_valid,
            severity=overall_severity,
            messages=all_messages,
            metadata={"num_validations": len(self.validation_results)}
        )
    
    @property
    def is_valid(self) -> bool:
        """Whether the overall validation passed"""
        if self.overall_validation_result:
            return self.overall_validation_result.is_valid
        return len(self.validation_results) > 0 and all(r.is_valid for r in self.validation_results)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "request_id": self.request_id,
            "validation_results": [result.to_dict() for result in self.validation_results],
            "overall_validation_result": self.overall_validation_result.to_dict() if self.overall_validation_result else None,
            "confidence_score": self.confidence_score,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "processing_metadata": self.processing_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_valid": self.is_valid
        }