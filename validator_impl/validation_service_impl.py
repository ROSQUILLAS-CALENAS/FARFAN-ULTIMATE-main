"""
Validation Service Implementation

Concrete implementation of ValidationService that orchestrates validation 
operations using dependency injection. Only depends on validator_api interfaces.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Import only from validator_api - no pipeline dependencies
from validator_api.validation_interfaces import (
    ValidationService,
    ValidatorPort,
    EvidenceProcessorPort,
    ValidationResult,
    ValidationStatus
)
from validator_api.dtos import (
    ValidationRequest,
    ValidationResponse,
    EvidenceProcessingRequest,
    ValidationError
)

logger = logging.getLogger(__name__)


class ValidationServiceImpl(ValidationService):
    """Service that orchestrates validation operations through dependency injection."""
    
    def __init__(self):
        self._validators: Dict[str, ValidatorPort] = {}
        self._evidence_processors: Dict[str, EvidenceProcessorPort] = {}
        self._default_timeout = 30  # seconds
        self._max_parallel_workers = 4
    
    def register_validator(self, name: str, validator: ValidatorPort) -> None:
        """
        Register a validator with the service.
        
        Args:
            name: Name to register validator under
            validator: Validator instance implementing ValidatorPort
        """
        if not isinstance(validator, ValidatorPort):
            raise ValidationError(
                f"Validator must implement ValidatorPort interface",
                error_code="INVALID_VALIDATOR_TYPE"
            )
        
        self._validators[name] = validator
        logger.info(f"Registered validator '{name}' of type {type(validator).__name__}")
    
    def register_evidence_processor(self, name: str, processor: EvidenceProcessorPort) -> None:
        """
        Register an evidence processor with the service.
        
        Args:
            name: Name to register processor under
            processor: Evidence processor implementing EvidenceProcessorPort
        """
        if not isinstance(processor, EvidenceProcessorPort):
            raise ValidationError(
                f"Evidence processor must implement EvidenceProcessorPort interface",
                error_code="INVALID_PROCESSOR_TYPE"
            )
        
        self._evidence_processors[name] = processor
        logger.info(f"Registered evidence processor '{name}' of type {type(processor).__name__}")
    
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
        results = {}
        
        for validator_name in validator_names:
            try:
                if validator_name not in self._validators:
                    results[validator_name] = ValidationResult(
                        status=ValidationStatus.ERROR,
                        message=f"Validator '{validator_name}' not found",
                        details={"available_validators": list(self._validators.keys())},
                        errors=[f"Validator '{validator_name}' is not registered"],
                        warnings=[]
                    )
                    continue
                
                validator = self._validators[validator_name]
                
                # Create validation request
                request = ValidationRequest(
                    data=data,
                    validation_type=validator_name,
                    request_id=f"pipeline_{validator_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                # Perform validation
                result = validator.validate(request)
                results[validator_name] = result
                
                # Log result
                logger.info(f"Validator '{validator_name}' completed with status: {result.status}")
                
            except Exception as e:
                logger.error(f"Error running validator '{validator_name}': {str(e)}")
                results[validator_name] = ValidationResult(
                    status=ValidationStatus.ERROR,
                    message=f"Validation failed with exception: {str(e)}",
                    details={"exception_type": type(e).__name__},
                    errors=[str(e)],
                    warnings=[]
                )
        
        return results
    
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
        start_time = datetime.now()
        result = {
            "processing_result": None,
            "validation_results": {},
            "success": False,
            "errors": [],
            "warnings": [],
            "processing_time_ms": 0.0,
            "timestamp": start_time.isoformat()
        }
        
        try:
            # Check if processor exists
            if processor_name not in self._evidence_processors:
                error_msg = f"Evidence processor '{processor_name}' not found"
                result["errors"].append(error_msg)
                result["processing_result"] = {
                    "error": error_msg,
                    "available_processors": list(self._evidence_processors.keys())
                }
                return result
            
            processor = self._evidence_processors[processor_name]
            
            # Create processing request
            processing_request = EvidenceProcessingRequest(
                raw_evidence=raw_evidence,
                processing_type="standard",
                request_id=f"process_{processor_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Process evidence
            processing_response = processor.process_evidence(processing_request)
            result["processing_result"] = {
                "success": processing_response.success,
                "evidence_count": processing_response.get_evidence_count(),
                "processing_metadata": processing_response.processing_metadata,
                "errors": processing_response.errors,
                "warnings": processing_response.warnings,
                "evidence_items": [
                    {
                        "evidence_id": item.evidence_id,
                        "text": item.text,
                        "quality_score": item.quality_score,
                        "evidence_type": item.evidence_type,
                        "metadata": {
                            "document_id": item.metadata.document_id,
                            "title": item.metadata.title,
                            "author": item.metadata.author
                        }
                    } for item in processing_response.processed_evidence
                ]
            }
            
            # Add processing errors/warnings to result
            result["errors"].extend(processing_response.errors)
            result["warnings"].extend(processing_response.warnings)
            
            # Run validation if requested and processing was successful
            if validator_names and processing_response.success and processing_response.processed_evidence:
                validation_data = {
                    "processed_evidence": processing_response.processed_evidence,
                    "processing_metadata": processing_response.processing_metadata
                }
                
                validation_results = self.validate_with_pipeline(validation_data, validator_names)
                result["validation_results"] = {
                    name: {
                        "status": validation_result.status.value,
                        "message": validation_result.message,
                        "success": validation_result.is_success(),
                        "warnings_count": len(validation_result.warnings),
                        "errors_count": len(validation_result.errors),
                        "confidence_score": validation_result.confidence_score
                    } for name, validation_result in validation_results.items()
                }
                
                # Aggregate validation errors/warnings
                for validation_result in validation_results.values():
                    result["errors"].extend(validation_result.errors)
                    result["warnings"].extend(validation_result.warnings)
            
            # Determine overall success
            result["success"] = (
                processing_response.success and 
                len(result["errors"]) == 0 and
                (not validator_names or all(
                    vr.is_success() or vr.status == ValidationStatus.WARNING 
                    for vr in validation_results.values()
                ) if validator_names else True)
            )
            
        except Exception as e:
            logger.error(f"Error in process_and_validate_evidence: {str(e)}")
            result["errors"].append(str(e))
            result["processing_result"] = {"error": str(e)}
        
        finally:
            # Calculate processing time
            result["processing_time_ms"] = (datetime.now() - start_time).total_seconds() * 1000
        
        return result
    
    def get_registered_validators(self) -> Dict[str, Dict[str, Any]]:
        """Get information about registered validators."""
        validator_info = {}
        
        for name, validator in self._validators.items():
            try:
                validator_info[name] = {
                    "name": name,
                    "type": type(validator).__name__,
                    "validation_rules": validator.get_validation_rules(),
                    "supported_data_types": self._get_supported_data_types(validator)
                }
            except Exception as e:
                validator_info[name] = {
                    "name": name,
                    "type": type(validator).__name__,
                    "error": f"Failed to get validator info: {str(e)}"
                }
        
        return validator_info
    
    def get_registered_processors(self) -> Dict[str, Dict[str, Any]]:
        """Get information about registered evidence processors."""
        processor_info = {}
        
        for name, processor in self._evidence_processors.items():
            processor_info[name] = {
                "name": name,
                "type": type(processor).__name__,
                "description": f"Evidence processor for {name}"
            }
        
        return processor_info
    
    def validate_single(
        self, 
        data: Any, 
        validator_name: str,
        validation_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Run a single validator on data.
        
        Args:
            data: Data to validate
            validator_name: Name of validator to use
            validation_type: Optional validation type override
            context: Optional context for validation
            
        Returns:
            ValidationResult from the validator
        """
        if validator_name not in self._validators:
            return ValidationResult(
                status=ValidationStatus.ERROR,
                message=f"Validator '{validator_name}' not found",
                details={"available_validators": list(self._validators.keys())},
                errors=[f"Validator '{validator_name}' is not registered"],
                warnings=[]
            )
        
        try:
            validator = self._validators[validator_name]
            
            # Create validation request
            request = ValidationRequest(
                data=data,
                validation_type=validation_type or validator_name,
                context=context,
                request_id=f"single_{validator_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            return validator.validate(request)
            
        except Exception as e:
            logger.error(f"Error in validate_single with validator '{validator_name}': {str(e)}")
            return ValidationResult(
                status=ValidationStatus.ERROR,
                message=f"Validation failed with exception: {str(e)}",
                details={"exception_type": type(e).__name__},
                errors=[str(e)],
                warnings=[]
            )
    
    def _get_supported_data_types(self, validator: ValidatorPort) -> List[str]:
        """Get supported data types for a validator."""
        try:
            # Test common data types
            common_types = ["document", "text", "pdf", "json", "structured", "pdt"]
            supported = []
            
            for data_type in common_types:
                if validator.supports_data_type(data_type):
                    supported.append(data_type)
            
            return supported
            
        except Exception:
            return ["unknown"]
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all registered components."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "validators": {},
            "processors": {},
            "errors": [],
            "warnings": []
        }
        
        # Check validators
        for name, validator in self._validators.items():
            try:
                # Test with simple data
                test_request = ValidationRequest(
                    data="health check test",
                    validation_type="health_check"
                )
                
                result = validator.validate(test_request)
                health_status["validators"][name] = {
                    "status": "healthy",
                    "type": type(validator).__name__
                }
                
            except Exception as e:
                health_status["validators"][name] = {
                    "status": "error",
                    "error": str(e),
                    "type": type(validator).__name__
                }
                health_status["errors"].append(f"Validator '{name}' health check failed: {str(e)}")
        
        # Check processors
        for name, processor in self._evidence_processors.items():
            try:
                # Test with simple data
                test_request = EvidenceProcessingRequest(
                    raw_evidence="health check test",
                    processing_type="health_check"
                )
                
                result = processor.process_evidence(test_request)
                health_status["processors"][name] = {
                    "status": "healthy",
                    "type": type(processor).__name__
                }
                
            except Exception as e:
                health_status["processors"][name] = {
                    "status": "error",
                    "error": str(e),
                    "type": type(processor).__name__
                }
                health_status["errors"].append(f"Processor '{name}' health check failed: {str(e)}")
        
        # Set overall status
        if health_status["errors"]:
            health_status["status"] = "unhealthy"
        elif health_status["warnings"]:
            health_status["status"] = "warning"
        
        return health_status