"""
Abstract Base Classes for Validation Operations

Defines clean interfaces for validation without importing concrete components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass


class ValidationStatus(Enum):
    """Status of validation operations."""
    PASSED = "passed"
    WARNING = "warning" 
    FAILED = "failed"
    ERROR = "error"
    PENDING = "pending"


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    status: ValidationStatus
    message: str
    details: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    confidence_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def is_success(self) -> bool:
        """Check if validation passed."""
        return self.status == ValidationStatus.PASSED
    
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return len(self.warnings) > 0


class ValidatorPort(ABC):
    """Port interface for validators."""
    
    @abstractmethod
    def validate(self, request: 'ValidationRequest') -> ValidationResult:
        """
        Validate input data according to defined rules.
        
        Args:
            request: Validation request containing data to validate
            
        Returns:
            ValidationResult with status and details
        """
        pass
    
    @abstractmethod
    def get_validation_rules(self) -> Dict[str, Any]:
        """
        Get the validation rules used by this validator.
        
        Returns:
            Dictionary describing validation rules
        """
        pass
    
    @abstractmethod
    def supports_data_type(self, data_type: str) -> bool:
        """
        Check if this validator supports a given data type.
        
        Args:
            data_type: Type of data to check support for
            
        Returns:
            True if supported, False otherwise
        """
        pass


class EvidenceProcessorPort(ABC):
    """Port interface for evidence processors."""
    
    @abstractmethod
    def process_evidence(self, request: 'EvidenceProcessingRequest') -> 'EvidenceProcessingResponse':
        """
        Process raw evidence into structured format.
        
        Args:
            request: Evidence processing request
            
        Returns:
            EvidenceProcessingResponse with processed evidence
        """
        pass
    
    @abstractmethod
    def validate_evidence_structure(self, evidence: Dict[str, Any]) -> ValidationResult:
        """
        Validate the structure of evidence data.
        
        Args:
            evidence: Evidence data to validate
            
        Returns:
            ValidationResult indicating if structure is valid
        """
        pass
    
    @abstractmethod
    def extract_metadata(self, raw_data: Any) -> Dict[str, Any]:
        """
        Extract metadata from raw evidence data.
        
        Args:
            raw_data: Raw evidence data
            
        Returns:
            Dictionary containing extracted metadata
        """
        pass


class ValidationService(ABC):
    """Service interface that orchestrates validation operations."""
    
    @abstractmethod
    def register_validator(self, name: str, validator: ValidatorPort) -> None:
        """
        Register a validator with the service.
        
        Args:
            name: Name to register validator under
            validator: Validator instance implementing ValidatorPort
        """
        pass
    
    @abstractmethod
    def register_evidence_processor(self, name: str, processor: EvidenceProcessorPort) -> None:
        """
        Register an evidence processor with the service.
        
        Args:
            name: Name to register processor under
            processor: Evidence processor implementing EvidenceProcessorPort
        """
        pass
    
    @abstractmethod
    def validate_with_pipeline(
        self, 
        data: Any, 
        validator_names: List[str]
    ) -> Dict[str, ValidationResult]:
        """
        Run validation through a pipeline of validators.
        
        Args:
            data: Data to validate
            validator_names: List of validator names to run
            
        Returns:
            Dictionary mapping validator names to their results
        """
        pass
    
    @abstractmethod
    def process_and_validate_evidence(
        self, 
        raw_evidence: Any,
        processor_name: str,
        validator_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process evidence and optionally validate it.
        
        Args:
            raw_evidence: Raw evidence data to process
            processor_name: Name of evidence processor to use
            validator_names: Optional list of validators to run on processed evidence
            
        Returns:
            Dictionary containing processing results and validation results
        """
        pass