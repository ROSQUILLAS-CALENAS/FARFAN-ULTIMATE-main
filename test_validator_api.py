"""
Test script for the new validator_api port/adapter architecture
"""

import sys
import traceback

def test_validator_api():
    """Test the validator_api interfaces and implementations."""
    print("Testing validator_api port/adapter architecture...\n")
    
    # Test 1: Import validator_api interfaces
    print("1. Testing validator_api imports...")
    try:
        from validator_api import (
            ValidationResult, ValidatorPort, EvidenceProcessorPort,
            ValidationService, ValidationRequest, ValidationResponse,
            EvidenceProcessingRequest, EvidenceProcessingResponse
        )
        print("   ✓ Successfully imported validator_api interfaces")
    except Exception as e:
        print(f"   ✗ Failed to import validator_api interfaces: {e}")
        return False
    
    # Test 2: Import validator_impl concrete implementations
    print("\n2. Testing validator_impl imports...")
    try:
        from validator_impl import PDTValidator, EvidenceProcessorImpl, ValidationServiceImpl
        print("   ✓ Successfully imported validator_impl implementations")
    except Exception as e:
        print(f"   ✗ Failed to import validator_impl implementations: {e}")
        return False
    
    # Test 3: Create and configure validation service
    print("\n3. Testing ValidationService setup...")
    try:
        service = ValidationServiceImpl()
        pdt_validator = PDTValidator()
        evidence_processor = EvidenceProcessorImpl()
        
        service.register_validator("pdt", pdt_validator)
        service.register_evidence_processor("default", evidence_processor)
        print("   ✓ Successfully created and configured ValidationService")
    except Exception as e:
        print(f"   ✗ Failed to setup ValidationService: {e}")
        return False
    
    # Test 4: Test PDT validation
    print("\n4. Testing PDT validation...")
    try:
        sample_document = {
            "blocks": [
                {
                    "section_type": "diagnostico",
                    "text": "Este es un diagnóstico situacional detallado que analiza las condiciones actuales del territorio y establece una línea base comprehensiva para el desarrollo de estrategias de intervención efectivas y sostenibles.",
                    "confidence": 0.9
                },
                {
                    "section_type": "programas",
                    "text": "Los programas propuestos incluyen iniciativas específicas para mejorar la calidad de vida de la población objetivo a través de intervenciones coordinadas en sectores clave como educación, salud, infraestructura y desarrollo económico local.",
                    "confidence": 0.8
                }
            ]
        }
        
        request = ValidationRequest(
            data=sample_document,
            validation_type="pdt"
        )
        
        result = pdt_validator.validate(request)
        print(f"   ✓ PDT validation completed with status: {result.status.value}")
        print(f"   ✓ Validation message: {result.message[:100]}...")
    except Exception as e:
        print(f"   ✗ PDT validation failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 5: Test evidence processing
    print("\n5. Testing evidence processing...")
    try:
        sample_evidence = [
            {
                "text": "Esta es una evidencia directa que demuestra la efectividad del programa implementado en la región.",
                "metadata": {
                    "document_id": "doc_001",
                    "title": "Informe de Evaluación",
                    "author": "Equipo de Monitoreo"
                }
            }
        ]
        
        processing_request = EvidenceProcessingRequest(
            raw_evidence=sample_evidence,
            processing_type="standard"
        )
        
        processing_result = evidence_processor.process_evidence(processing_request)
        print(f"   ✓ Evidence processing completed successfully: {processing_result.success}")
        print(f"   ✓ Processed {processing_result.get_evidence_count()} evidence items")
    except Exception as e:
        print(f"   ✗ Evidence processing failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 6: Test validation service pipeline
    print("\n6. Testing validation pipeline...")
    try:
        pipeline_result = service.validate_with_pipeline(
            data=sample_document,
            validator_names=["pdt"]
        )
        
        print(f"   ✓ Pipeline validation completed for {len(pipeline_result)} validators")
        for validator_name, result in pipeline_result.items():
            print(f"   ✓ {validator_name}: {result.status.value}")
    except Exception as e:
        print(f"   ✗ Pipeline validation failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 7: Test process_and_validate_evidence
    print("\n7. Testing integrated evidence processing and validation...")
    try:
        integrated_result = service.process_and_validate_evidence(
            raw_evidence=sample_evidence,
            processor_name="default",
            validator_names=None  # No validation for now
        )
        
        print(f"   ✓ Integrated processing completed successfully: {integrated_result['success']}")
        if integrated_result['processing_result']:
            evidence_count = integrated_result['processing_result'].get('evidence_count', 0)
            print(f"   ✓ Processed {evidence_count} evidence items")
    except Exception as e:
        print(f"   ✗ Integrated processing failed: {e}")
        traceback.print_exc()
        return False
    
    # Test 8: Test refactored orchestrators
    print("\n8. Testing refactored orchestrators...")
    try:
        from validator_refactored import ValidatorOrchestrator
        from evidence_processor_refactored import EvidenceProcessingOrchestrator
        
        validator_orchestrator = ValidatorOrchestrator()
        evidence_orchestrator = EvidenceProcessingOrchestrator()
        
        # Test validator orchestrator
        validation_result = validator_orchestrator.validate_document(
            document_data=sample_document,
            document_type="pdt"
        )
        print(f"   ✓ Validator orchestrator worked: {validation_result['success']}")
        
        # Test evidence orchestrator
        evidence_result = evidence_orchestrator.process_evidence(
            raw_evidence=sample_evidence
        )
        print(f"   ✓ Evidence orchestrator worked: {evidence_result['success']}")
        
    except Exception as e:
        print(f"   ✗ Refactored orchestrators failed: {e}")
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("✅ All tests passed! Port/adapter architecture is working correctly.")
    print("✅ Dependency injection successfully decouples validation from pipeline components.")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = test_validator_api()
    sys.exit(0 if success else 1)