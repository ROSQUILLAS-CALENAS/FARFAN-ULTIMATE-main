"""
Abstract interfaces for validator operations

These interfaces define the contract for validation operations without
importing any concrete implementations or pipeline components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Protocol


class IValidationResult(Protocol):
    """Protocol for validation results"""
    
    @property
    def is_valid(self) -> bool:
        """Whether the validation passed"""
        ...
    
    @property
    def severity(self) -> str:
        """Severity level of validation issues"""
        ...
    
    @property
    def messages(self) -> List[str]:
        """List of validation messages"""
        ...
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Additional metadata about the validation"""
        ...


class IEvidenceValidationRequest(Protocol):
    """Protocol for evidence validation requests"""
    
    @property
    def evidence_text(self) -> str:
        """Text content to validate"""
        ...
    
    @property
    def context(self) -> str:
        """Context for validation"""
        ...
    
    @property
    def validation_type(self) -> str:
        """Type of validation to perform"""
        ...


class IEvidenceValidationResponse(Protocol):
    """Protocol for evidence validation responses"""
    
    @property
    def validation_result(self) -> IValidationResult:
        """The validation result"""
        ...
    
    @property
    def confidence_score(self) -> float:
        """Confidence score for the validation"""
        ...
    
    @property
    def processing_metadata(self) -> Dict[str, Any]:
        """Metadata about the processing"""
        ...


class IValidator(ABC):
    """Abstract base class for all validators"""
    
    @abstractmethod
    def validate(self, request: IEvidenceValidationRequest) -> IEvidenceValidationResponse:
        """
        Validate evidence according to the specified criteria
        
        Args:
            request: Validation request containing evidence and parameters
            
        Returns:
            Validation response with results and metadata
        """
        pass
    
    @abstractmethod
    def get_supported_validation_types(self) -> List[str]:
        """
        Get list of validation types supported by this validator
        
        Returns:
            List of supported validation type strings
        """
        pass
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the validator with the given parameters
        
        Args:
            config: Configuration parameters
        """
        pass


class IEvidenceProcessor(ABC):
    """Abstract base class for evidence processors"""
    
    @abstractmethod
    def process_evidence(self, evidence_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process raw evidence data into structured format
        
        Args:
            evidence_data: Raw evidence data to process
            
        Returns:
            List of processed evidence items
        """
        pass
    
    @abstractmethod
    def extract_features(self, evidence_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from a single evidence item
        
        Args:
            evidence_item: Evidence item to extract features from
            
        Returns:
            Dictionary of extracted features
        """
        pass
    
    @abstractmethod
    def validate_evidence_structure(self, evidence_item: Dict[str, Any]) -> bool:
        """
        Validate the structure of an evidence item
        
        Args:
            evidence_item: Evidence item to validate
            
        Returns:
            True if structure is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_evidence_metadata(self, evidence_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get metadata for an evidence item
        
        Args:
            evidence_item: Evidence item to get metadata for
            
        Returns:
            Dictionary of metadata
        """
        pass


class IValidatorFactory(ABC):
    """Abstract factory for creating validators"""
    
    @abstractmethod
    def create_validator(self, validator_type: str, config: Optional[Dict[str, Any]] = None) -> IValidator:
        """
        Create a validator of the specified type
        
        Args:
            validator_type: Type of validator to create
            config: Optional configuration parameters
            
        Returns:
            Configured validator instance
        """
        pass
    
    @abstractmethod
    def get_available_validator_types(self) -> List[str]:
        """
        Get list of available validator types
        
        Returns:
            List of available validator type strings
        """
        pass


class IEvidenceProcessorFactory(ABC):
    """Abstract factory for creating evidence processors"""
    
    @abstractmethod
    def create_processor(self, processor_type: str, config: Optional[Dict[str, Any]] = None) -> IEvidenceProcessor:
        """
        Create an evidence processor of the specified type
        
        Args:
            processor_type: Type of processor to create
            config: Optional configuration parameters
            
        Returns:
            Configured processor instance
        """
        pass
    
    @abstractmethod
    def get_available_processor_types(self) -> List[str]:
        """
        Get list of available processor types
        
        Returns:
            List of available processor type strings
        """
        pass