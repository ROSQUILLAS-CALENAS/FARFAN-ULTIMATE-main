"""
Refactored Validator Module using Dependency Injection

This module demonstrates how to use the new validator_api interfaces 
with dependency injection instead of direct imports from pipeline components.
"""

from typing import Dict, List, Any, Optional
import logging

# Import from validator_api package - clean interfaces only
from validator_api import (
    ValidationService,
    ValidatorPort,
    ValidationRequest,
    ValidationResponse,
    ValidationStatus
)

# Import concrete implementations
from validator_impl import PDTValidator, ValidationServiceImpl

logger = logging.getLogger(__name__)


class ValidatorOrchestrator:
    """
    Orchestrates validation operations using dependency injection.
    
    This replaces direct imports and coupling with pipeline components.
    """
    
    def __init__(self):
        self.validation_service: ValidationService = ValidationServiceImpl()
        self._setup_validators()
    
    def _setup_validators(self):
        """Setup and register validators with the service."""
        # Register PDT validator
        pdt_validator = PDTValidator()
        self.validation_service.register_validator("pdt", pdt_validator)
        
        logger.info("Validators registered successfully")
    
    def validate_document(
        self, 
        document_data: Any, 
        document_type: str = "pdt",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate a document using the appropriate validator.
        
        Args:
            document_data: Document data to validate
            document_type: Type of document (determines which validator to use)
            context: Optional context information
            
        Returns:
            Dictionary containing validation results
        """
        try:
            # Use the validation service instead of direct validator calls
            result = self.validation_service.validate_single(
                data=document_data,
                validator_name=document_type,
                context=context
            )
            
            return {
                "success": result.is_success(),
                "status": result.status.value,
                "message": result.message,
                "details": result.details,
                "errors": result.errors,
                "warnings": result.warnings,
                "confidence_score": result.confidence_score,
                "metadata": result.metadata
            }
            
        except Exception as e:
            logger.error(f"Document validation failed: {str(e)}")
            return {
                "success": False,
                "status": "error",
                "message": f"Validation failed: {str(e)}",
                "details": {},
                "errors": [str(e)],
                "warnings": [],
                "confidence_score": 0.0,
                "metadata": {}
            }
    
    def validate_batch(
        self, 
        documents: List[Dict[str, Any]], 
        document_type: str = "pdt"
    ) -> List[Dict[str, Any]]:
        """
        Validate a batch of documents.
        
        Args:
            documents: List of documents to validate
            document_type: Type of documents
            
        Returns:
            List of validation results for each document
        """
        results = []
        
        for i, doc in enumerate(documents):
            try:
                result = self.validate_document(
                    document_data=doc,
                    document_type=document_type,
                    context={"batch_index": i, "batch_size": len(documents)}
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Batch validation failed for document {i}: {str(e)}")
                results.append({
                    "success": False,
                    "status": "error",
                    "message": f"Batch validation failed: {str(e)}",
                    "details": {"batch_index": i},
                    "errors": [str(e)],
                    "warnings": [],
                    "confidence_score": 0.0,
                    "metadata": {}
                })
        
        return results
    
    def validate_with_multiple_validators(
        self, 
        data: Any, 
        validator_names: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run validation through multiple validators.
        
        Args:
            data: Data to validate
            validator_names: List of validator names to use
            
        Returns:
            Dictionary mapping validator names to their results
        """
        try:
            # Use the validation service's pipeline functionality
            validation_results = self.validation_service.validate_with_pipeline(
                data=data,
                validator_names=validator_names
            )
            
            # Convert results to standard format
            formatted_results = {}
            for name, result in validation_results.items():
                formatted_results[name] = {
                    "success": result.is_success(),
                    "status": result.status.value,
                    "message": result.message,
                    "details": result.details,
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "confidence_score": result.confidence_score,
                    "metadata": result.metadata
                }
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Multi-validator validation failed: {str(e)}")
            return {
                validator_name: {
                    "success": False,
                    "status": "error", 
                    "message": f"Multi-validation failed: {str(e)}",
                    "details": {},
                    "errors": [str(e)],
                    "warnings": [],
                    "confidence_score": 0.0,
                    "metadata": {}
                } for validator_name in validator_names
            }
    
    def get_validator_info(self) -> Dict[str, Any]:
        """Get information about available validators."""
        try:
            return {
                "registered_validators": self.validation_service.get_registered_validators(),
                "health_status": self.validation_service.health_check()
            }
        except Exception as e:
            logger.error(f"Failed to get validator info: {str(e)}")
            return {
                "error": str(e),
                "registered_validators": {},
                "health_status": {"status": "error", "error": str(e)}
            }
    
    def add_custom_validator(self, name: str, validator: ValidatorPort) -> bool:
        """
        Add a custom validator to the service.
        
        Args:
            name: Name to register validator under
            validator: Validator implementing ValidatorPort
            
        Returns:
            True if successfully registered, False otherwise
        """
        try:
            self.validation_service.register_validator(name, validator)
            logger.info(f"Successfully registered custom validator '{name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to register custom validator '{name}': {str(e)}")
            return False


# Convenience functions for backwards compatibility
def validate_pdt_document(document_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to validate a PDT document.
    
    Args:
        document_data: PDT document data to validate
        context: Optional context information
        
    Returns:
        Validation result dictionary
    """
    orchestrator = ValidatorOrchestrator()
    return orchestrator.validate_document(
        document_data=document_data,
        document_type="pdt",
        context=context
    )


def create_validation_service() -> ValidationService:
    """
    Factory function to create a validation service with default validators.
    
    Returns:
        Configured ValidationService instance
    """
    service = ValidationServiceImpl()
    
    # Register default validators
    service.register_validator("pdt", PDTValidator())
    
    return service


# Example usage for migration from old code
if __name__ == "__main__":
    # Example of how to use the refactored validator
    orchestrator = ValidatorOrchestrator()
    
    # Sample document data
    sample_document = {
        "blocks": [
            {
                "section_type": "diagnostico",
                "text": "Este es un diagnóstico situacional detallado que analiza las condiciones actuales...",
                "confidence": 0.9
            },
            {
                "section_type": "programas", 
                "text": "Los programas propuestos incluyen iniciativas específicas para mejorar...",
                "confidence": 0.8
            }
        ]
    }
    
    # Validate document
    result = orchestrator.validate_document(
        document_data=sample_document,
        document_type="pdt"
    )
    
    print("Validation Result:")
    print(f"Success: {result['success']}")
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    print(f"Confidence Score: {result['confidence_score']}")
    
    if result['warnings']:
        print(f"Warnings: {result['warnings']}")
    
    if result['errors']:
        print(f"Errors: {result['errors']}")