#!/usr/bin/env python3
"""
Test script for the refactored validator system

This script tests the validator_api and validator_impl packages
to ensure they work correctly with dependency injection.
"""

import sys
import traceback
from typing import Dict, Any, List

def test_basic_imports():
    """Test that all basic imports work"""
    print("Testing basic imports...")
    
    try:
        # Test validator_api imports
        from validator_api.interfaces import (
            IValidator, IEvidenceProcessor, IValidatorFactory, IEvidenceProcessorFactory
        )
        from validator_api.dtos import (
            ValidationRequest, ValidationResponse, EvidenceItem,
            ValidationSeverity, ValidationCategory, DNPAlignmentCategory
        )
        print("✓ validator_api imports successful")
        
        # Test validator_impl imports
        from validator_impl.validators import (
            ComprehensiveValidator, DNPAlignmentValidator, EvidenceValidator
        )
        from validator_impl.evidence_processors import (
            DefaultEvidenceProcessor, DNPEvidenceProcessor
        )
        from validator_impl.factories import (
            ValidatorFactory, EvidenceProcessorFactory, DependencyInjectionContainer
        )
        print("✓ validator_impl imports successful")
        
        # Test canonical_flow imports
        from canonical_flow.A_analysis_nlp.refactored_stage_orchestrator import (
            AnalysisNLPOrchestrator, create_analysis_orchestrator
        )
        print("✓ refactored orchestrator imports successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_validator_creation():
    """Test validator creation through factory"""
    print("\nTesting validator creation...")
    
    try:
        from validator_impl.factories import ValidatorFactory
        
        factory = ValidatorFactory()
        
        # Test available types
        available_types = factory.get_available_validator_types()
        print(f"✓ Available validator types: {available_types}")
        
        # Test creating different validators
        validators_to_test = ['comprehensive', 'dnp_alignment', 'evidence']
        
        for validator_type in validators_to_test:
            try:
                validator = factory.create_validator(validator_type)
                supported_types = validator.get_supported_validation_types()
                print(f"✓ Created {validator_type} validator (supports: {supported_types})")
            except Exception as e:
                print(f"✗ Failed to create {validator_type} validator: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Validator creation failed: {e}")
        traceback.print_exc()
        return False


def test_processor_creation():
    """Test evidence processor creation through factory"""
    print("\nTesting evidence processor creation...")
    
    try:
        from validator_impl.factories import EvidenceProcessorFactory
        
        factory = EvidenceProcessorFactory()
        
        # Test available types
        available_types = factory.get_available_processor_types()
        print(f"✓ Available processor types: {available_types}")
        
        # Test creating different processors
        processors_to_test = ['default', 'dnp']
        
        for processor_type in processors_to_test:
            try:
                processor = factory.create_processor(processor_type)
                print(f"✓ Created {processor_type} processor")
                
                # Test basic functionality
                test_evidence = {
                    'content': 'Test evidence content for validation',
                    'source': 'test_source',
                    'type': 'general'
                }
                
                processed = processor.process_evidence(test_evidence)
                print(f"  → Processed {len(processed)} evidence items")
                
                if processed:
                    is_valid = processor.validate_evidence_structure(processed[0])
                    print(f"  → Evidence structure valid: {is_valid}")
                
            except Exception as e:
                print(f"✗ Failed to create/test {processor_type} processor: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Processor creation failed: {e}")
        traceback.print_exc()
        return False


def test_validation_workflow():
    """Test complete validation workflow"""
    print("\nTesting validation workflow...")
    
    try:
        from validator_impl.factories import ValidatorFactory
        from validator_api.dtos import ValidationRequest, ValidationCategory
        
        factory = ValidatorFactory()
        validator = factory.create_validator('comprehensive')
        
        # Create test request
        request = ValidationRequest(
            evidence_text="El Plan Nacional de Desarrollo establece las directrices para el desarrollo sostenible del país.",
            context="Política pública colombiana",
            validation_categories=[ValidationCategory.FACTUAL_ACCURACY, ValidationCategory.LOGICAL_CONSISTENCY]
        )
        
        # Perform validation
        response = validator.validate(request)
        
        print(f"✓ Validation completed")
        print(f"  → Request ID: {response.request_id}")
        print(f"  → Overall valid: {response.is_valid}")
        print(f"  → Confidence score: {response.confidence_score:.2f}")
        print(f"  → Number of validation results: {len(response.validation_results)}")
        
        # Test DNP validator
        dnp_validator = factory.create_validator('dnp_alignment')
        dnp_response = dnp_validator.validate(request)
        
        print(f"✓ DNP validation completed")
        print(f"  → Overall valid: {dnp_response.is_valid}")
        print(f"  → Confidence score: {dnp_response.confidence_score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation workflow failed: {e}")
        traceback.print_exc()
        return False


def test_dependency_injection():
    """Test dependency injection container"""
    print("\nTesting dependency injection...")
    
    try:
        from validator_impl.factories import DependencyInjectionContainer, get_global_container, configure_global_defaults
        
        # Test global container
        container = get_global_container()
        configure_global_defaults()
        print("✓ Global container configured")
        
        # Test getting components through container
        validator_factory = container.get_validator_factory()
        processor_factory = container.get_processor_factory()
        
        print("✓ Factories obtained from container")
        
        # Test singleton pattern
        validator1 = container.get_validator('comprehensive', singleton=True)
        validator2 = container.get_validator('comprehensive', singleton=True)
        
        print(f"✓ Singleton pattern test: {validator1 is validator2}")
        
        # Test non-singleton
        validator3 = container.get_validator('comprehensive', singleton=False)
        validator4 = container.get_validator('comprehensive', singleton=False)
        
        print(f"✓ Non-singleton pattern test: {validator3 is not validator4}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dependency injection failed: {e}")
        traceback.print_exc()
        return False


def test_orchestrator_integration():
    """Test the refactored orchestrator with dependency injection"""
    print("\nTesting orchestrator integration...")
    
    try:
        from validator_impl.factories import get_global_container, configure_global_defaults
        from canonical_flow.A_analysis_nlp.refactored_stage_orchestrator import create_analysis_orchestrator
        from validator_api.dtos import ValidationCategory, DNPAlignmentCategory
        
        # Setup container
        container = get_global_container()
        configure_global_defaults()
        
        # Create orchestrator
        orchestrator = create_analysis_orchestrator(
            validator_factory=container.get_validator_factory(),
            processor_factory=container.get_processor_factory()
        )
        
        print("✓ Orchestrator created successfully")
        
        # Test analysis request
        results = orchestrator.process_analysis_request(
            text_content="Los municipios deben elaborar sus Planes de Desarrollo Territorial siguiendo los lineamientos del Departamento Nacional de Planeación.",
            context="Marco normativo municipal",
            validation_categories=[ValidationCategory.FACTUAL_ACCURACY, ValidationCategory.SOURCE_RELIABILITY],
            dnp_categories=[DNPAlignmentCategory.REGULATORY, DNPAlignmentCategory.PROCEDURAL]
        )
        
        print(f"✓ Analysis completed: {results['status']}")
        print(f"  → Request ID: {results.get('request_id', 'unknown')}")
        print(f"  → Processing time: {results.get('processing_time_ms', 0)}ms")
        
        # Check validation results
        validation = results.get('validation', {})
        overall = validation.get('overall', {})
        print(f"  → Overall validation: {'PASSED' if overall.get('is_valid') else 'FAILED'}")
        print(f"  → Confidence: {overall.get('confidence_score', 0.0):.2f}")
        
        # Test with evidence
        evidence_data = {
            'items': [
                {
                    'content': 'La Ley 152 de 1994 establece el marco para los planes de desarrollo territorial.',
                    'source': 'Ley 152 de 1994',
                    'type': 'legal_reference'
                }
            ]
        }
        
        results_with_evidence = orchestrator.process_analysis_request(
            text_content="Los PDT deben seguir la normativa vigente.",
            evidence_data=evidence_data
        )
        
        evidence_processing = results_with_evidence.get('evidence_processing', {})
        print(f"✓ Evidence processing: {evidence_processing.get('processing_successful', False)}")
        print(f"  → Items processed: {evidence_processing.get('items_processed', 0)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Orchestrator integration failed: {e}")
        traceback.print_exc()
        return False


def test_interface_compliance():
    """Test that implementations comply with interfaces"""
    print("\nTesting interface compliance...")
    
    try:
        from validator_api.interfaces import IValidator, IEvidenceProcessor
        from validator_impl.validators import ComprehensiveValidator
        from validator_impl.evidence_processors import DefaultEvidenceProcessor
        
        # Test validator interface compliance
        validator = ComprehensiveValidator()
        assert hasattr(validator, 'validate'), "Validator missing validate method"
        assert hasattr(validator, 'get_supported_validation_types'), "Validator missing get_supported_validation_types method"
        assert hasattr(validator, 'configure'), "Validator missing configure method"
        print("✓ Validator interface compliance verified")
        
        # Test processor interface compliance
        processor = DefaultEvidenceProcessor()
        assert hasattr(processor, 'process_evidence'), "Processor missing process_evidence method"
        assert hasattr(processor, 'extract_features'), "Processor missing extract_features method"
        assert hasattr(processor, 'validate_evidence_structure'), "Processor missing validate_evidence_structure method"
        assert hasattr(processor, 'get_evidence_metadata'), "Processor missing get_evidence_metadata method"
        print("✓ Processor interface compliance verified")
        
        return True
        
    except Exception as e:
        print(f"✗ Interface compliance failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests() -> bool:
    """Run all tests and return overall success"""
    print("=" * 60)
    print("VALIDATOR SYSTEM REFACTOR TESTS")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_validator_creation,
        test_processor_creation,
        test_validation_workflow,
        test_dependency_injection,
        test_orchestrator_integration,
        test_interface_compliance
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, test in enumerate(tests):
        status = "✓ PASS" if results[i] else "✗ FAIL"
        print(f"{status} - {test.__name__}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The refactored validator system is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the output above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)