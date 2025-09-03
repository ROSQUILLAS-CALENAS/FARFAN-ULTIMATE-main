"""
Refactored Pipeline Orchestrator using Event Bus System
======================================================

This demonstrates how to refactor existing orchestrator modules to use the event bus
system instead of direct imports. The orchestrator publishes events for pipeline
stages and subscribes to validation events.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from uuid import uuid4

from event_bus import EventBus
from event_schemas import EventType
from event_driven_orchestrator import EventDrivenOrchestrator
from event_driven_validator import ValidatorRegistry


logger = logging.getLogger(__name__)


class RefactoredPipelineOrchestrator(EventDrivenOrchestrator):
    """
    Refactored version of existing pipeline orchestrator that uses events
    instead of direct imports to pipeline stages.
    """
    
    def __init__(self):
        # Create event bus
        self.event_bus = EventBus(
            max_event_history=2000,
            max_handler_errors=5,
            enable_event_logging=True
        )
        
        # Initialize parent with event bus
        super().__init__(
            event_bus=self.event_bus,
            orchestrator_name="RefactoredPipelineOrchestrator"
        )
        
        # Create and register validators
        self.validator_registry = ValidatorRegistry(self.event_bus)
        self.validator_registry.create_default_validators()
        
        # Register stage handlers (these replace direct imports)
        self._register_stage_handlers()
        
        logger.info("RefactoredPipelineOrchestrator initialized with event bus")
    
    def _register_stage_handlers(self):
        """Register handlers for pipeline stages (replaces direct imports)"""
        
        # Instead of importing analysis_nlp modules directly, we register handlers
        self.register_stage_handler("document_ingestion", self._handle_document_ingestion)
        self.register_stage_handler("text_extraction", self._handle_text_extraction)
        self.register_stage_handler("structure_analysis", self._handle_structure_analysis)
        self.register_stage_handler("content_validation", self._handle_content_validation)
        self.register_stage_handler("quality_assessment", self._handle_quality_assessment)
        self.register_stage_handler("final_packaging", self._handle_final_packaging)
    
    # Stage handlers (these replace direct calls to imported modules)
    
    def _handle_document_ingestion(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle document ingestion stage"""
        logger.info("Processing document ingestion stage")
        
        document_path = data.get('document_path')
        if not document_path:
            raise ValueError("Document path not provided")
        
        # Simulate document ingestion processing
        # In real implementation, this would call the actual ingestion logic
        # without importing it directly
        
        result = {
            'document_id': str(uuid4()),
            'document_path': document_path,
            'document_type': data.get('document_type', 'PDF'),
            'ingestion_timestamp': datetime.utcnow().isoformat(),
            'file_size_bytes': data.get('file_size', 0),
            'pages_count': data.get('pages', 0)
        }
        
        logger.info(f"Document ingestion completed: {result['document_id']}")
        return result
    
    def _handle_text_extraction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle text extraction stage"""
        logger.info("Processing text extraction stage")
        
        document_id = data.get('document_id')
        if not document_id:
            raise ValueError("Document ID not provided")
        
        # Simulate text extraction
        # This would trigger OCR, table extraction, etc. through events
        # rather than direct imports
        
        result = {
            'extracted_text': data.get('mock_text', 'Sample extracted text content'),
            'text_length': len(data.get('mock_text', 'Sample text')),
            'ocr_confidence': 0.95,
            'extraction_method': 'mixed_ocr_text',
            'language_detected': 'spanish',
            'tables_found': data.get('tables_count', 2)
        }
        
        # Request validation through events instead of direct call
        self._request_extraction_validation(document_id, result)
        
        logger.info(f"Text extraction completed for document: {document_id}")
        return result
    
    def _handle_structure_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle document structure analysis stage"""
        logger.info("Processing structure analysis stage")
        
        extracted_text = data.get('extracted_text', '')
        if not extracted_text:
            logger.warning("No extracted text available for structure analysis")
        
        # Simulate structure analysis
        # This would identify sections, headers, etc. through events
        
        result = {
            'sections_identified': [
                {'name': 'DIAGNOSTICO', 'start': 0, 'end': 1000, 'confidence': 0.9},
                {'name': 'PROGRAMAS', 'start': 1000, 'end': 2500, 'confidence': 0.85},
                {'name': 'PRESUPUESTO', 'start': 2500, 'end': 3200, 'confidence': 0.88},
                {'name': 'METAS', 'start': 3200, 'end': 4000, 'confidence': 0.92}
            ],
            'structure_confidence': 0.88,
            'coherence_score': 0.82,
            'completeness_index': 0.75
        }
        
        # Request structure validation through events
        document_id = data.get('document_id')
        self._request_structure_validation(document_id, result)
        
        logger.info(f"Structure analysis completed for document: {document_id}")
        return result
    
    def _handle_content_validation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle content validation stage"""
        logger.info("Processing content validation stage")
        
        # This stage coordinates validation requests rather than performing them directly
        document_id = data.get('document_id')
        sections = data.get('sections_identified', [])
        
        # Request comprehensive validation through events
        validation_requests = []
        
        # Request PDT document validation
        validation_id_1 = self._request_pdt_validation(document_id, data)
        validation_requests.append(validation_id_1)
        
        # Request schema validation
        validation_id_2 = self._request_schema_validation(document_id, data)
        validation_requests.append(validation_id_2)
        
        result = {
            'validation_requests_issued': len(validation_requests),
            'validation_request_ids': validation_requests,
            'validation_coordinator': self.orchestrator_name,
            'sections_to_validate': len(sections)
        }
        
        logger.info(f"Content validation coordinated for document: {document_id}")
        return result
    
    def _handle_quality_assessment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle quality assessment stage"""
        logger.info("Processing quality assessment stage")
        
        document_id = data.get('document_id')
        
        # Simulate quality assessment
        quality_metrics = {
            'overall_quality_score': 0.82,
            'structural_metrics': {
                'coherence': 0.78,
                'completeness': 0.85,
                'consistency': 0.80
            },
            'content_metrics': {
                'relevance': 0.88,
                'accuracy': 0.75,
                'clarity': 0.82
            },
            'compliance_metrics': {
                'dnp_standards': 0.90,
                'legal_requirements': 0.85,
                'format_compliance': 0.92
            }
        }
        
        # Request quality validation through events
        validation_id = self._request_quality_validation(document_id, quality_metrics)
        
        result = {
            'quality_metrics': quality_metrics,
            'quality_validation_requested': validation_id,
            'assessment_timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Quality assessment completed for document: {document_id}")
        return result
    
    def _handle_final_packaging(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle final packaging stage"""
        logger.info("Processing final packaging stage")
        
        document_id = data.get('document_id')
        
        # Package all results
        result = {
            'package_id': str(uuid4()),
            'document_id': document_id,
            'packaging_timestamp': datetime.utcnow().isoformat(),
            'components_packaged': [
                'extracted_text', 'structure_analysis', 'validation_results',
                'quality_metrics', 'compliance_report'
            ],
            'package_status': 'completed',
            'output_format': 'json',
            'total_processing_time': 'calculated_dynamically'
        }
        
        logger.info(f"Final packaging completed: {result['package_id']}")
        return result
    
    # Validation request helpers (replaces direct validator calls)
    
    def _request_extraction_validation(self, document_id: str, extraction_result: Dict[str, Any]) -> str:
        """Request validation of text extraction results"""
        pdt_validator = self.validator_registry.get_validator("PDTDocumentValidator")
        if pdt_validator:
            return pdt_validator.request_validation(
                validation_target="text_extraction_quality",
                input_data={
                    'document_id': document_id,
                    'extraction_result': extraction_result
                },
                validation_rules=["check_extraction_completeness", "verify_ocr_confidence"],
                correlation_id=document_id
            )
        return None
    
    def _request_structure_validation(self, document_id: str, structure_result: Dict[str, Any]) -> str:
        """Request validation of document structure analysis"""
        pdt_validator = self.validator_registry.get_validator("PDTDocumentValidator")
        if pdt_validator:
            return pdt_validator.request_validation(
                validation_target="document_structure",
                input_data={
                    'document_id': document_id,
                    'sections': structure_result.get('sections_identified', []),
                    'structure_metrics': structure_result
                },
                validation_rules=["check_mandatory_sections", "validate_section_order"],
                correlation_id=document_id
            )
        return None
    
    def _request_pdt_validation(self, document_id: str, document_data: Dict[str, Any]) -> str:
        """Request comprehensive PDT document validation"""
        pdt_validator = self.validator_registry.get_validator("PDTDocumentValidator")
        if pdt_validator:
            return pdt_validator.request_validation(
                validation_target="pdt_document_compliance",
                input_data={
                    'document_id': document_id,
                    'sections': document_data.get('sections_identified', []),
                    'quality_metrics': document_data.get('quality_metrics', {}),
                    'budget_data': document_data.get('budget_data', {}),
                    'goals': document_data.get('goals', []),
                    'programs': document_data.get('programs', []),
                    'legal_references': document_data.get('legal_references', [])
                },
                validation_rules=[
                    "check_budget_consistency",
                    "check_goal_alignment", 
                    "check_legal_compliance"
                ],
                correlation_id=document_id
            )
        return None
    
    def _request_schema_validation(self, document_id: str, document_data: Dict[str, Any]) -> str:
        """Request schema validation"""
        schema_validator = self.validator_registry.get_validator("SchemaValidator")
        if schema_validator:
            return schema_validator.request_validation(
                validation_target="document_schema_compliance",
                input_data={
                    'document_id': document_id,
                    'schema_type': 'pdt_document',
                    **document_data
                },
                validation_rules=["pdt_document"],
                correlation_id=document_id
            )
        return None
    
    def _request_quality_validation(self, document_id: str, quality_metrics: Dict[str, Any]) -> str:
        """Request quality validation"""
        qa_validator = self.validator_registry.get_validator("QualityAssuranceValidator")
        if qa_validator:
            return qa_validator.request_validation(
                validation_target="document_quality_metrics",
                input_data={
                    'document_id': document_id,
                    'overall_quality_score': quality_metrics.get('overall_quality_score'),
                    'structural_metrics': quality_metrics.get('structural_metrics', {}),
                    'completeness_score': quality_metrics.get('structural_metrics', {}).get('completeness', 0.0)
                },
                validation_rules=["assess_overall_quality", "check_structural_coherence"],
                correlation_id=document_id
            )
        return None
    
    # Public API methods
    
    def execute_pipeline(self, input_data: Dict[str, Any]) -> str:
        """
        Execute the complete pipeline using event-driven architecture.
        
        Args:
            input_data: Pipeline input data including document path
            
        Returns:
            Execution ID for tracking
        """
        pipeline_config = {
            'stages': [
                'document_ingestion',
                'text_extraction', 
                'structure_analysis',
                'content_validation',
                'quality_assessment',
                'final_packaging'
            ],
            'failure_strategy': 'stop_on_failure',
            'enable_validation': True,
            'quality_thresholds': {
                'min_quality_score': 0.7,
                'min_completeness': 0.8
            }
        }
        
        execution_id = self.start_pipeline_execution(
            pipeline_config=pipeline_config,
            input_data=input_data
        )
        
        logger.info(f"Pipeline execution started with ID: {execution_id}")
        return execution_id
    
    def get_pipeline_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed pipeline execution status"""
        status = self.get_execution_status(execution_id)
        if not status:
            return None
        
        # Add validator statistics
        status['validator_statistics'] = self.validator_registry.get_all_statistics()
        
        # Add event bus statistics
        status['event_bus_statistics'] = self.event_bus.get_stats()
        
        return status
    
    def shutdown(self):
        """Shutdown the orchestrator and all components"""
        logger.info("Shutting down RefactoredPipelineOrchestrator")
        
        # Shutdown validators
        self.validator_registry.shutdown_all()
        
        # Shutdown parent orchestrator
        super().shutdown()
        
        # Shutdown event bus
        self.event_bus.shutdown()


# ============================================================================
# EXAMPLE USAGE AND INTEGRATION
# ============================================================================

def demonstrate_refactored_orchestrator():
    """Demonstrate the refactored orchestrator in action"""
    
    # Create orchestrator (automatically sets up event bus and validators)
    orchestrator = RefactoredPipelineOrchestrator()
    
    try:
        # Execute pipeline with sample input
        input_data = {
            'document_path': '/path/to/sample_pdt.pdf',
            'document_type': 'PDT',
            'pages': 45,
            'file_size': 2048000,
            'mock_text': 'DIAGNOSTICO: Este es el diagnóstico situacional...',
            'tables_count': 3
        }
        
        execution_id = orchestrator.execute_pipeline(input_data)
        print(f"Pipeline execution started: {execution_id}")
        
        # Monitor execution status
        import time
        for i in range(5):
            status = orchestrator.get_pipeline_status(execution_id)
            if status:
                print(f"Progress: {status['progress_percentage']:.1f}% - Stage: {status['current_stage']}")
                if status['is_complete']:
                    print("Pipeline execution completed!")
                    break
            time.sleep(1)
        
        # Show final statistics
        final_status = orchestrator.get_pipeline_status(execution_id)
        if final_status:
            print("\nFinal Execution Status:")
            print(f"Completed stages: {final_status['completed_stages']}")
            print(f"Failed stages: {final_status['failed_stages']}")
            
            print("\nValidator Statistics:")
            for validator_name, stats in final_status['validator_statistics'].items():
                print(f"  {validator_name}: {stats['validations_performed']} validations, "
                      f"{stats['success_rate_percentage']:.1f}% success rate")
            
            print("\nEvent Bus Statistics:")
            bus_stats = final_status['event_bus_statistics']
            print(f"  Events published: {bus_stats['events_published']}")
            print(f"  Events delivered: {bus_stats['events_delivered']}")
            print(f"  Success rate: {bus_stats['success_rate']:.1f}%")
    
    finally:
        # Clean shutdown
        orchestrator.shutdown()


if __name__ == "__main__":
    demonstrate_refactored_orchestrator()