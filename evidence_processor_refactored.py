"""
Refactored Evidence Processor Module using Dependency Injection

This module demonstrates how to use the new validator_api interfaces 
for evidence processing with dependency injection instead of direct 
imports from pipeline components.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# Import from validator_api package - clean interfaces only
from validator_api import (
    ValidationService,
    EvidenceProcessorPort,
    EvidenceProcessingRequest,
    EvidenceProcessingResponse,
    EvidenceItem
)

# Import concrete implementations
from validator_impl import EvidenceProcessorImpl, ValidationServiceImpl

logger = logging.getLogger(__name__)


class EvidenceProcessingOrchestrator:
    """
    Orchestrates evidence processing operations using dependency injection.
    
    This replaces direct imports and coupling with pipeline components.
    """
    
    def __init__(self):
        self.validation_service: ValidationService = ValidationServiceImpl()
        self._setup_processors()
    
    def _setup_processors(self):
        """Setup and register evidence processors with the service."""
        # Register default evidence processor
        evidence_processor = EvidenceProcessorImpl()
        self.validation_service.register_evidence_processor("default", evidence_processor)
        
        logger.info("Evidence processors registered successfully")
    
    def process_evidence(
        self, 
        raw_evidence: Any, 
        processor_type: str = "default",
        validation_enabled: bool = False,
        validator_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process raw evidence into structured format.
        
        Args:
            raw_evidence: Raw evidence data to process
            processor_type: Type of processor to use
            validation_enabled: Whether to run validation on processed evidence
            validator_names: List of validators to run if validation is enabled
            
        Returns:
            Dictionary containing processing results and optional validation results
        """
        try:
            # Use the validation service's process_and_validate_evidence method
            result = self.validation_service.process_and_validate_evidence(
                raw_evidence=raw_evidence,
                processor_name=processor_type,
                validator_names=validator_names if validation_enabled else None
            )
            
            return {
                "success": result["success"],
                "processing_result": result["processing_result"],
                "validation_results": result["validation_results"],
                "errors": result["errors"],
                "warnings": result["warnings"],
                "processing_time_ms": result["processing_time_ms"],
                "timestamp": result["timestamp"]
            }
            
        except Exception as e:
            logger.error(f"Evidence processing failed: {str(e)}")
            return {
                "success": False,
                "processing_result": None,
                "validation_results": {},
                "errors": [str(e)],
                "warnings": [],
                "processing_time_ms": 0.0,
                "timestamp": datetime.now().isoformat()
            }
    
    def process_evidence_batch(
        self, 
        evidence_batch: List[Any], 
        processor_type: str = "default",
        max_workers: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of evidence items.
        
        Args:
            evidence_batch: List of raw evidence items to process
            processor_type: Type of processor to use
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of processing results for each evidence item
        """
        results = []
        
        for i, evidence_item in enumerate(evidence_batch):
            try:
                result = self.process_evidence(
                    raw_evidence=evidence_item,
                    processor_type=processor_type
                )
                
                # Add batch metadata
                result["batch_metadata"] = {
                    "batch_index": i,
                    "batch_size": len(evidence_batch)
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Batch processing failed for evidence {i}: {str(e)}")
                results.append({
                    "success": False,
                    "processing_result": None,
                    "validation_results": {},
                    "errors": [str(e)],
                    "warnings": [],
                    "processing_time_ms": 0.0,
                    "timestamp": datetime.now().isoformat(),
                    "batch_metadata": {
                        "batch_index": i,
                        "batch_size": len(evidence_batch)
                    }
                })
        
        return results
    
    def extract_evidence_metadata(
        self, 
        raw_evidence: Any, 
        processor_type: str = "default"
    ) -> Dict[str, Any]:
        """
        Extract metadata from raw evidence without full processing.
        
        Args:
            raw_evidence: Raw evidence data
            processor_type: Type of processor to use
            
        Returns:
            Dictionary containing extracted metadata
        """
        try:
            # Get the processor directly for metadata extraction
            if processor_type not in self.validation_service._evidence_processors:
                return {
                    "error": f"Processor '{processor_type}' not found",
                    "available_processors": list(self.validation_service._evidence_processors.keys())
                }
            
            processor = self.validation_service._evidence_processors[processor_type]
            metadata = processor.extract_metadata(raw_evidence)
            
            return {
                "success": True,
                "metadata": metadata,
                "processor_type": processor_type,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "metadata": {},
                "processor_type": processor_type,
                "timestamp": datetime.now().isoformat()
            }
    
    def validate_evidence_structure(
        self, 
        evidence_data: Dict[str, Any], 
        processor_type: str = "default"
    ) -> Dict[str, Any]:
        """
        Validate the structure of evidence data.
        
        Args:
            evidence_data: Evidence data to validate
            processor_type: Type of processor to use for validation
            
        Returns:
            Dictionary containing validation results
        """
        try:
            # Get the processor for structure validation
            if processor_type not in self.validation_service._evidence_processors:
                return {
                    "success": False,
                    "error": f"Processor '{processor_type}' not found",
                    "available_processors": list(self.validation_service._evidence_processors.keys())
                }
            
            processor = self.validation_service._evidence_processors[processor_type]
            validation_result = processor.validate_evidence_structure(evidence_data)
            
            return {
                "success": validation_result.is_success(),
                "status": validation_result.status.value,
                "message": validation_result.message,
                "details": validation_result.details,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "confidence_score": validation_result.confidence_score
            }
            
        except Exception as e:
            logger.error(f"Evidence structure validation failed: {str(e)}")
            return {
                "success": False,
                "status": "error",
                "message": f"Structure validation failed: {str(e)}",
                "details": {},
                "errors": [str(e)],
                "warnings": [],
                "confidence_score": 0.0
            }
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get information about available evidence processors."""
        try:
            return {
                "registered_processors": self.validation_service.get_registered_processors(),
                "health_status": self.validation_service.health_check()
            }
        except Exception as e:
            logger.error(f"Failed to get processor info: {str(e)}")
            return {
                "error": str(e),
                "registered_processors": {},
                "health_status": {"status": "error", "error": str(e)}
            }
    
    def add_custom_processor(self, name: str, processor: EvidenceProcessorPort) -> bool:
        """
        Add a custom evidence processor to the service.
        
        Args:
            name: Name to register processor under
            processor: Processor implementing EvidenceProcessorPort
            
        Returns:
            True if successfully registered, False otherwise
        """
        try:
            self.validation_service.register_evidence_processor(name, processor)
            logger.info(f"Successfully registered custom processor '{name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to register custom processor '{name}': {str(e)}")
            return False
    
    def generate_processing_report(
        self, 
        processing_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a summary report from processing results.
        
        Args:
            processing_results: List of processing results
            
        Returns:
            Dictionary containing summary report
        """
        try:
            total_items = len(processing_results)
            successful_items = sum(1 for r in processing_results if r.get("success", False))
            failed_items = total_items - successful_items
            
            # Calculate evidence counts
            total_evidence = 0
            high_quality_evidence = 0
            
            for result in processing_results:
                if result.get("processing_result") and result["processing_result"].get("evidence_items"):
                    items = result["processing_result"]["evidence_items"]
                    total_evidence += len(items)
                    high_quality_evidence += sum(1 for item in items if item.get("quality_score", 0) >= 0.8)
            
            # Calculate processing times
            processing_times = [r.get("processing_time_ms", 0) for r in processing_results]
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            return {
                "summary": {
                    "total_items_processed": total_items,
                    "successful_items": successful_items,
                    "failed_items": failed_items,
                    "success_rate": (successful_items / total_items * 100) if total_items > 0 else 0,
                    "total_evidence_extracted": total_evidence,
                    "high_quality_evidence": high_quality_evidence,
                    "quality_rate": (high_quality_evidence / total_evidence * 100) if total_evidence > 0 else 0,
                    "average_processing_time_ms": avg_processing_time
                },
                "errors": [
                    error for result in processing_results 
                    for error in result.get("errors", [])
                ],
                "warnings": [
                    warning for result in processing_results 
                    for warning in result.get("warnings", [])
                ],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate processing report: {str(e)}")
            return {
                "summary": {"error": str(e)},
                "errors": [str(e)],
                "warnings": [],
                "timestamp": datetime.now().isoformat()
            }


# Convenience functions for backwards compatibility
def process_evidence_simple(raw_evidence: Any) -> Dict[str, Any]:
    """
    Convenience function to process evidence with default settings.
    
    Args:
        raw_evidence: Raw evidence data to process
        
    Returns:
        Processing result dictionary
    """
    orchestrator = EvidenceProcessingOrchestrator()
    return orchestrator.process_evidence(raw_evidence=raw_evidence)


def create_evidence_processing_service() -> ValidationService:
    """
    Factory function to create a validation service with default evidence processors.
    
    Returns:
        Configured ValidationService instance
    """
    service = ValidationServiceImpl()
    
    # Register default evidence processors
    service.register_evidence_processor("default", EvidenceProcessorImpl())
    
    return service


# Example usage for migration from old code
if __name__ == "__main__":
    # Example of how to use the refactored evidence processor
    orchestrator = EvidenceProcessingOrchestrator()
    
    # Sample raw evidence data
    sample_evidence = [
        {
            "text": "Esta es una evidencia directa que demuestra la efectividad del programa implementado.",
            "metadata": {
                "document_id": "doc_001",
                "title": "Informe de Evaluación",
                "author": "Equipo de Monitoreo",
                "page_number": 15
            }
        },
        {
            "text": "Los datos estadísticos muestran un incremento del 25% en los indicadores clave.",
            "metadata": {
                "document_id": "doc_002", 
                "title": "Reporte Estadístico",
                "author": "Departamento de Análisis"
            }
        }
    ]
    
    # Process evidence
    result = orchestrator.process_evidence(
        raw_evidence=sample_evidence,
        processor_type="default"
    )
    
    print("Evidence Processing Result:")
    print(f"Success: {result['success']}")
    
    if result['processing_result']:
        processing_info = result['processing_result']
        print(f"Evidence Count: {processing_info.get('evidence_count', 0)}")
        print(f"Processing Errors: {len(processing_info.get('errors', []))}")
        print(f"Processing Warnings: {len(processing_info.get('warnings', []))}")
    
    print(f"Processing Time: {result['processing_time_ms']}ms")
    
    # Extract metadata example
    metadata_result = orchestrator.extract_evidence_metadata(sample_evidence[0])
    print("\nMetadata Extraction Result:")
    print(f"Success: {metadata_result['success']}")
    if metadata_result['success']:
        print(f"Extracted Metadata: {metadata_result['metadata']}")