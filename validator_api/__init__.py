"""
Validator API Package

Provides abstract base classes and data transfer objects that define 
interfaces for validation operations without importing any concrete 
pipeline components.
"""

from .validation_interfaces import (
    ValidationResult,
    ValidatorPort,
    EvidenceProcessorPort,
    ValidationService,
)

from .dtos import (
    ValidationRequest,
    ValidationResponse,
    EvidenceProcessingRequest, 
    EvidenceProcessingResponse,
    ValidationError,
    ValidationStatus,
)

__all__ = [
    # Interfaces
    'ValidationResult',
    'ValidatorPort',
    'EvidenceProcessorPort', 
    'ValidationService',
    
    # DTOs
    'ValidationRequest',
    'ValidationResponse',
    'EvidenceProcessingRequest',
    'EvidenceProcessingResponse',
    'ValidationError',
    'ValidationStatus',
]