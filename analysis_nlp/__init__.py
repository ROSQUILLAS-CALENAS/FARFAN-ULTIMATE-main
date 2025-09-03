"""
Analysis NLP Contract Module

This module defines the standardized contract interface for all analysis_nlp components,
including base classes, error handling, and artifact path conventions.
"""

# # # from abc import ABC, abstractmethod  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, Any, Optional, Union  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
import json


# Error Code Constants
class ErrorCodes:
    """Standardized error codes for analysis_nlp components"""
    VALIDATION_FAILED = "VALIDATION_FAILED"
    PROCESSING_TIMEOUT = "PROCESSING_TIMEOUT"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    INVALID_PDF_PATH = "INVALID_PDF_PATH"
    INVALID_DOCUMENT_ID = "INVALID_DOCUMENT_ID"
    INVALID_OUTPUT_DIR = "INVALID_OUTPUT_DIR"
    ARTIFACT_GENERATION_FAILED = "ARTIFACT_GENERATION_FAILED"
    INSUFFICIENT_RESOURCES = "INSUFFICIENT_RESOURCES"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"


# Custom Exception Classes
class AnalysisError(Exception):
    """Base exception class for analysis_nlp components"""
    def __init__(self, message: str, error_code: str, component: str = None):
        super().__init__(message)
        self.error_code = error_code
        self.component = component
        self.message = message

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format for consistent error reporting"""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "component": self.component
        }


class ValidationError(AnalysisError):
    """Exception raised when input validation fails"""
    def __init__(self, message: str, component: str = None, validation_details: Dict[str, Any] = None):
        super().__init__(message, ErrorCodes.VALIDATION_FAILED, component)
        self.validation_details = validation_details or {}

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["validation_details"] = self.validation_details
        return result


class ProcessingError(AnalysisError):
    """Exception raised during processing operations"""
    def __init__(self, message: str, component: str = None, processing_stage: str = None):
        super().__init__(message, ErrorCodes.PROCESSING_ERROR, component)
        self.processing_stage = processing_stage

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["processing_stage"] = self.processing_stage
        return result


class TimeoutError(AnalysisError):
    """Exception raised when processing times out"""
    def __init__(self, message: str, component: str = None, timeout_seconds: float = None):
        super().__init__(message, ErrorCodes.PROCESSING_TIMEOUT, component)
        self.timeout_seconds = timeout_seconds

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["timeout_seconds"] = self.timeout_seconds
        return result


# Component Suffix Mapping for the 9 analysis_nlp components
COMPONENT_SUFFIXES = {
    "adaptive_analyzer": "_adaptive",
    "question_analyzer": "_question", 
    "implementacion_mapeo": "_implementacion",
    "evidence_processor": "_evidence",
    "evaluation_driven_processor": "_evaluation",
    "dnp_alignment_adapter": "_dnp",
    "extractor_evidencias_contextual": "_extractor",
    "evidence_validation_model": "_validation",
    "metadata": "_metadata"  # Generic suffix for any additional components
}


class ArtifactPathBuilder:
    """Utility class for building standardized artifact paths"""
    
    @staticmethod
    def build_path(output_dir: str, document_id: str, component_suffix: str) -> str:
        """
        Build standardized artifact path following the pattern:
        {output_dir}/{document_id}_{component_suffix}.json
        
        Args:
            output_dir: Output directory path
            document_id: Unique document identifier
            component_suffix: Component-specific suffix (e.g., '_adaptive', '_question')
            
        Returns:
            Standardized artifact file path
        """
        if not component_suffix.startswith("_"):
            component_suffix = f"_{component_suffix}"
            
        filename = f"{document_id}{component_suffix}.json"
        return str(Path(output_dir) / filename)
    
    @staticmethod
    def get_suffix_for_component(component_name: str) -> str:
        """Get the standardized suffix for a component name"""
        return COMPONENT_SUFFIXES.get(component_name, "_metadata")


class ProcessingResult:
    """Standardized return schema for analysis_nlp components"""
    
    def __init__(self, 
                 status: str,
                 artifacts: Dict[str, str] = None,
                 error_details: Dict[str, Any] = None,
                 processing_metadata: Dict[str, Any] = None):
        """
        Initialize processing result
        
        Args:
            status: Processing status ('success', 'failed', 'partial')
            artifacts: Dictionary mapping artifact names to file paths
            error_details: Error information if status is 'failed' or 'partial'
            processing_metadata: Additional metadata about the processing
        """
        self.status = status
        self.artifacts = artifacts or {}
        self.error_details = error_details or {}
        self.processing_metadata = processing_metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format"""
        return {
            "status": self.status,
            "artifacts": self.artifacts,
            "error_details": self.error_details,
            "processing_metadata": self.processing_metadata
        }
    
    def to_json(self) -> str:
        """Convert result to JSON string"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class BaseAnalysisContract(ABC):
    """
    Abstract base class defining the standardized contract for analysis_nlp components.
    
# # #     All analysis_nlp components must inherit from this class and implement the required  # Module not found  # Module not found  # Module not found
    abstract methods to ensure consistent interface and behavior across the system.
    """
    
    def __init__(self, component_name: str):
        """
        Initialize the analysis component
        
        Args:
            component_name: Name of the component (used for error reporting and artifacts)
        """
        self.component_name = component_name
        self.component_suffix = ArtifactPathBuilder.get_suffix_for_component(component_name)
    
    def process(self, 
                pdf_path: str, 
                document_id: str, 
                output_dir: str,
                **kwargs) -> ProcessingResult:
        """
        Main processing method with standardized signature.
        
        Args:
            pdf_path: Path to the PDF file to be processed
            document_id: Unique identifier for the document
            output_dir: Directory where artifacts will be stored
            **kwargs: Additional component-specific parameters
            
        Returns:
            ProcessingResult with status, artifacts, and error details
            
        Raises:
            ValidationError: If input validation fails
            ProcessingError: If processing encounters an error
            TimeoutError: If processing times out
        """
        try:
            # Step 1: Validate inputs
            self.validate_inputs(pdf_path, document_id, output_dir, **kwargs)
            
            # Step 2: Generate artifacts
            artifacts = self.generate_artifacts(pdf_path, document_id, output_dir, **kwargs)
            
            # Step 3: Return success result
            return ProcessingResult(
                status="success",
                artifacts=artifacts,
                processing_metadata={
                    "component": self.component_name,
                    "component_suffix": self.component_suffix
                }
            )
            
        except (ValidationError, ProcessingError, TimeoutError) as e:
            # Handle known exceptions
            return ProcessingResult(
                status="failed",
                error_details=e.to_dict(),
                processing_metadata={
                    "component": self.component_name,
                    "component_suffix": self.component_suffix
                }
            )
            
        except Exception as e:
            # Handle unexpected exceptions
            error = ProcessingError(
                f"Unexpected error in {self.component_name}: {str(e)}",
                component=self.component_name
            )
            return ProcessingResult(
                status="failed",
                error_details=error.to_dict(),
                processing_metadata={
                    "component": self.component_name,
                    "component_suffix": self.component_suffix
                }
            )
    
    @abstractmethod
    def validate_inputs(self, 
                       pdf_path: str, 
                       document_id: str, 
                       output_dir: str,
                       **kwargs) -> None:
        """
        Validate input parameters before processing.
        
        Args:
            pdf_path: Path to the PDF file
            document_id: Document identifier
            output_dir: Output directory path
            **kwargs: Additional component-specific parameters
            
        Raises:
            ValidationError: If any input validation fails
        """
        pass
    
    @abstractmethod
    def generate_artifacts(self, 
                          pdf_path: str, 
                          document_id: str, 
                          output_dir: str,
                          **kwargs) -> Dict[str, str]:
        """
        Generate component-specific artifacts.
        
        Args:
            pdf_path: Path to the PDF file
            document_id: Document identifier  
            output_dir: Output directory path
            **kwargs: Additional component-specific parameters
            
        Returns:
            Dictionary mapping artifact names to file paths
            
        Raises:
            ProcessingError: If artifact generation fails
        """
        pass
    
    def build_artifact_path(self, output_dir: str, document_id: str, suffix_override: str = None) -> str:
        """
        Build standardized artifact path for this component.
        
        Args:
            output_dir: Output directory path
            document_id: Document identifier
            suffix_override: Optional suffix override (defaults to component suffix)
            
        Returns:
            Standardized artifact file path
        """
        suffix = suffix_override or self.component_suffix
        return ArtifactPathBuilder.build_path(output_dir, document_id, suffix)
    
    def save_artifact(self, data: Dict[str, Any], artifact_path: str) -> None:
        """
        Save artifact data to file with error handling.
        
        Args:
            data: Artifact data to save
            artifact_path: Path where artifact will be saved
            
        Raises:
            ProcessingError: If artifact saving fails
        """
        try:
            # Ensure output directory exists
            Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save artifact as JSON
            with open(artifact_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            raise ProcessingError(
                f"Failed to save artifact to {artifact_path}: {str(e)}",
                component=self.component_name,
                processing_stage="artifact_saving"
            )
    
    def validate_common_inputs(self, pdf_path: str, document_id: str, output_dir: str) -> None:
        """
        Common input validation logic that can be used by all components.
        
        Args:
            pdf_path: Path to PDF file
            document_id: Document identifier
            output_dir: Output directory path
            
        Raises:
            ValidationError: If any common validation fails
        """
        validation_errors = {}
        
        # Validate PDF path
        if not pdf_path or not isinstance(pdf_path, str):
            validation_errors["pdf_path"] = "PDF path must be a non-empty string"
        elif not Path(pdf_path).exists():
            validation_errors["pdf_path"] = f"PDF file does not exist: {pdf_path}"
        elif not pdf_path.lower().endswith('.pdf'):
            validation_errors["pdf_path"] = f"File must be a PDF: {pdf_path}"
        
        # Validate document ID
        if not document_id or not isinstance(document_id, str):
            validation_errors["document_id"] = "Document ID must be a non-empty string"
        elif len(document_id.strip()) == 0:
            validation_errors["document_id"] = "Document ID cannot be empty or whitespace"
        
        # Validate output directory
        if not output_dir or not isinstance(output_dir, str):
            validation_errors["output_dir"] = "Output directory must be a non-empty string"
        else:
            # Try to create output directory if it doesn't exist
            try:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                validation_errors["output_dir"] = f"Cannot create output directory {output_dir}: {str(e)}"
        
        if validation_errors:
            raise ValidationError(
                "Input validation failed",
                component=self.component_name,
                validation_details=validation_errors
            )


# Export all public classes and constants
__all__ = [
    'BaseAnalysisContract',
    'ProcessingResult', 
    'ArtifactPathBuilder',
    'AnalysisError',
    'ValidationError',
    'ProcessingError',
    'TimeoutError',
    'ErrorCodes',
    'COMPONENT_SUFFIXES'
]