"""
Validator API Package - Abstract Interfaces and Data Transfer Objects

This package defines the contract for validator operations without importing
any pipeline components, ensuring clean dependency separation.
"""

from .interfaces import (
    IValidator,
    IEvidenceProcessor,
    IValidationResult,
    IEvidenceValidationRequest,
    IEvidenceValidationResponse
)

from .dtos import (
    ValidationRequest,
    ValidationResponse,
    EvidenceItem,
    ValidationMetrics,
    ValidationSeverity,
    ValidationCategory,
    DNPAlignmentCategory
)

__all__ = [
    "IValidator",
    "IEvidenceProcessor", 
    "IValidationResult",
    "IEvidenceValidationRequest",
    "IEvidenceValidationResponse",
    "ValidationRequest",
    "ValidationResponse",
    "EvidenceItem",
    "ValidationMetrics",
    "ValidationSeverity",
    "ValidationCategory",
    "DNPAlignmentCategory"
]