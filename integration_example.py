"""
Integration Example: Using Refactored Validator System with Dependency Injection

This example demonstrates how to use the new validator_api and validator_impl
packages with the refactored pipeline orchestrator.
"""

import logging
from typing import Dict, Any

# Import validator API interfaces and DTOs
from validator_api.interfaces import IValidatorFactory, IEvidenceProcessorFactory
from validator_api.dtos import ValidationCategory, DNPAlignmentCategory

# Import concrete implementations
from validator_impl.factories import (
    ValidatorFactory,
    EvidenceProcessorFactory,
    DependencyInjectionContainer,
    get_global_container,
    configure_global_defaults
)

# Import refactored orchestrator
from canonical_flow.A_analysis_nlp.refactored_stage_orchestrator import (
    create_analysis_orchestrator,
    AnalysisNLPOrchestrator
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_basic_usage():
    """Demonstrate basic usage with manual dependency injection"""
    
    print("=== Basic Usage Example ===")
    
    # Create factories manually
    validator_factory = ValidatorFactory()
    processor_factory = EvidenceProcessorFactory()
    
    # Create orchestrator with dependency injection
    orchestrator = create_analysis_orchestrator(
        validator_factory=validator_factory,
        processor_factory=processor_factory
    )
    
    # Process a sample analysis request
    results = orchestrator.process_analysis_request(
        text_content="El Plan Nacional de Desarrollo establece las directrices para el crecimiento económico sostenible. Según estudios del DANE, el PIB creció 3.2% en 2023.",
        context="Evaluación de políticas públicas colombianas",
        validation_categories=[
            ValidationCategory.FACTUAL_ACCURACY,
            ValidationCategory.SOURCE_RELIABILITY,
            ValidationCategory.LOGICAL_CONSISTENCY
        ],
        dnp_categories=[
            DNPAlignmentCategory.REGULATORY,
            DNPAlignmentCategory.PROCEDURAL
        ]
    )
    
    print(f"Analysis completed with status: {results['status']}")
    print(f"Overall validation: {'PASSED' if results.get('validation', {}).get('overall', {}).get('is_valid') else 'FAILED'}")
    print(f"Confidence score: {results.get('scores', {}).get('overall_confidence', 0.0):.2f}")
    print()


def demonstrate_container_usage():
    """Demonstrate usage with dependency injection container"""
    
    print("=== Container Usage Example ===")
    
    # Get global container and configure defaults
    container = get_global_container()
    configure_global_defaults()
    
    # Add custom configuration
    container.register_configuration("comprehensive", {
        "enable_all_categories": True,
        "confidence_threshold": 0.8,
        "strict_mode": True
    })
    
    container.register_configuration("dnp", {
        "enable_legal_detection": True,
        "strict_compliance": True,
        "include_constitutional_checks": True
    })
    
    # Create orchestrator using container
    orchestrator = create_analysis_orchestrator(
        validator_factory=container.get_validator_factory(),
        processor_factory=container.get_processor_factory(),
        config={
            'validators': {
                'comprehensive': {'strict_mode': True},
                'dnp_alignment': {'strict_compliance': True}
            }
        }
    )
    
    # Test with evidence processing
    evidence_data = {
        'items': [
            {
                'content': 'La Constitución Política de Colombia establece en su artículo 339 que el Plan Nacional de Desarrollo debe ser aprobado por el Congreso.',
                'source': 'Constitución Política de Colombia',
                'type': 'legal_reference'
            },
            {
                'content': 'Según decreto 1082 de 2015, los municipios deben elaborar sus PDT siguiendo los lineamientos del DNP.',
                'source': 'Decreto 1082 de 2015',
                'type': 'regulatory_text'
            }
        ]
    }
    
    results = orchestrator.process_analysis_request(
        text_content="Los Planes de Desarrollo Territorial (PDT) deben alinearse con el Plan Nacional de Desarrollo según la normativa vigente.",
        context="Marco normativo para PDT municipales",
        validation_categories=[ValidationCategory.LOGICAL_CONSISTENCY, ValidationCategory.COMPLETENESS],
        dnp_categories=[DNPAlignmentCategory.CONSTITUTIONAL, DNPAlignmentCategory.REGULATORY],
        evidence_data=evidence_data
    )
    
    print(f"Analysis with evidence completed: {results['status']}")
    print(f"Evidence items processed: {results.get('evidence_processing', {}).get('items_processed', 0)}")
    print(f"DNP compliance score: {results.get('scores', {}).get('dnp_compliance', 0.0):.2f}")
    print()


def demonstrate_validator_factory_usage():
    """Demonstrate direct validator factory usage"""
    
    print("=== Direct Validator Factory Usage ===")
    
    # Create factories
    validator_factory = ValidatorFactory()
    processor_factory = EvidenceProcessorFactory()
    
    # Show available types
    print("Available validator types:", validator_factory.get_available_validator_types())
    print("Available processor types:", processor_factory.get_available_processor_types())
    
    # Create specific validators
    comprehensive_validator = validator_factory.create_validator('comprehensive')
    dnp_validator = validator_factory.create_validator('dnp_alignment')
    
    # Create specific processors
    default_processor = processor_factory.create_processor('default')
    dnp_processor = processor_factory.create_processor('dnp')
    
    # Test validator directly
    from validator_api.dtos import ValidationRequest
    
    request = ValidationRequest(
        evidence_text="Los recursos del Sistema General de Participaciones se distribuyen según la Ley 715 de 2001.",
        context="Financiación municipal",
        validation_categories=[ValidationCategory.FACTUAL_ACCURACY, ValidationCategory.SOURCE_RELIABILITY]
    )
    
    response = comprehensive_validator.validate(request)
    print(f"Direct validation result: {'VALID' if response.is_valid else 'INVALID'}")
    print(f"Confidence: {response.confidence_score:.2f}")
    
    # Test processor directly
    evidence_data = {
        'content': 'La Ley 715 de 2001 regula la distribución de recursos del SGP a municipios y departamentos.',
        'source': 'Ley 715 de 2001',
        'type': 'legal_reference'
    }
    
    processed = dnp_processor.process_evidence(evidence_data)
    print(f"Processed evidence items: {len(processed)}")
    if processed:
        print(f"Evidence type classified as: {processed[0].get('evidence_type', 'unknown')}")
    print()


def demonstrate_factory_registration():
    """Demonstrate custom validator registration"""
    
    print("=== Custom Validator Registration ===")
    
    # Create a custom validator class
    from validator_api.interfaces import IValidator
    from validator_api.dtos import ValidationResponse, ValidationResult, ValidationSeverity
    
    class CustomValidator(IValidator):
        def __init__(self, config=None):
            self.config = config or {}
        
        def validate(self, request):
            # Simple custom validation logic
            is_valid = len(request.evidence_text) > 50 and 'custom' in request.evidence_text.lower()
            
            result = ValidationResult(
                is_valid=is_valid,
                severity=ValidationSeverity.INFO if is_valid else ValidationSeverity.MEDIUM,
                messages=["Custom validation passed" if is_valid else "Custom validation failed - missing 'custom' keyword or too short"]
            )
            
            return ValidationResponse(
                request_id=getattr(request, 'request_id', 'custom_req'),
                validation_results=[result],
                confidence_score=1.0 if is_valid else 0.5
            )
        
        def get_supported_validation_types(self):
            return ['custom']
        
        def configure(self, config):
            self.config.update(config)
    
    # Register custom validator
    validator_factory = ValidatorFactory()
    validator_factory.register_validator('custom', CustomValidator)
    
    print("Registered custom validator")
    print("Available types now:", validator_factory.get_available_validator_types())
    
    # Test custom validator
    custom_validator = validator_factory.create_validator('custom')
    from validator_api.dtos import ValidationRequest
    
    request = ValidationRequest(
        evidence_text="This is a custom validation test that should pass because it contains the custom keyword.",
        validation_type='custom'
    )
    
    response = custom_validator.validate(request)
    print(f"Custom validation result: {'PASSED' if response.is_valid else 'FAILED'}")
    print()


def demonstrate_error_handling():
    """Demonstrate error handling in the refactored system"""
    
    print("=== Error Handling Example ===")
    
    container = get_global_container()
    orchestrator = create_analysis_orchestrator(
        validator_factory=container.get_validator_factory(),
        processor_factory=container.get_processor_factory()
    )
    
    # Test with invalid input
    try:
        results = orchestrator.process_analysis_request(
            text_content="",  # Empty content
            context="Test context"
        )
        print(f"Empty content handling: {results['status']}")
        print(f"Error message: {results.get('error', 'No error')}")
        
    except Exception as e:
        print(f"Exception caught: {e}")
    
    # Test with malformed evidence data
    try:
        malformed_evidence = {
            'items': [
                {'content': None},  # Invalid content
                {'invalid_field': 'test'}  # Missing required fields
            ]
        }
        
        results = orchestrator.process_analysis_request(
            text_content="Test content for malformed evidence",
            evidence_data=malformed_evidence
        )
        
        evidence_processing = results.get('evidence_processing', {})
        print(f"Malformed evidence handling: {evidence_processing.get('processing_successful', False)}")
        print(f"Items processed: {evidence_processing.get('items_processed', 0)}")
        
    except Exception as e:
        print(f"Exception in evidence processing: {e}")
    
    print()


def main():
    """Run all demonstration examples"""
    
    print("Validator System Integration Examples")
    print("=" * 50)
    
    try:
        demonstrate_basic_usage()
        demonstrate_container_usage()
        demonstrate_validator_factory_usage()
        demonstrate_factory_registration()
        demonstrate_error_handling()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        print(f"Error during execution: {e}")


if __name__ == "__main__":
    main()