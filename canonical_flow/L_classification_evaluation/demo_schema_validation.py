"""
Demo script for Schema Validation and Registry Integration
Shows practical usage of Pydantic schemas, checksum validation, and artifact generation
in the L_classification_evaluation stage.
"""

import json
from datetime import datetime
from pprint import pprint

from schemas import (
    QuestionEvalInput,
    DimensionEvalOutput, 
    PointEvalOutput,
    StageMeta,
    RegistryChecksumValidator,
    validate_artifact_json,
    ResponseType,
    DimensionType,
    ComplianceLevel
)
from question_registry import DecalogoQuestionRegistry, get_default_registry
from decalogo_scoring_system import ScoringSystem


def demo_question_input_validation():
    """Demonstrate QuestionEvalInput schema validation"""
    print("=== QUESTION INPUT VALIDATION DEMO ===")
    
    registry = get_default_registry()
    checksum = registry.get_checksum()
    
    # Valid question input
    valid_data = {
        "question_id": "DE1_Q1",
        "dimension": "DE-1", 
        "response": "Sí",
        "evidence_completeness": 0.85,
        "page_reference_quality": 0.90,
        "registry_checksum": checksum,
        "citation_pages": [12, 15, 18],
        "evidence_snippets": [
            "El PDT establece claramente los mecanismos...",
            "Se evidencia la implementación de procesos..."
        ],
        "confidence_level": 0.95
    }
    
    try:
        validated_input = QuestionEvalInput(**valid_data)
        print("✓ Valid input validated successfully:")
        print(f"  Question: {validated_input.question_id}")
        print(f"  Dimension: {validated_input.dimension}")
        print(f"  Response: {validated_input.response}")
        print(f"  Evidence Quality: {validated_input.evidence_completeness}")
        print(f"  Schema Version: {validated_input.schema_version}")
        print()
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        print()
    
    # Invalid input - misaligned dimension
    print("Testing invalid input (dimension mismatch):")
    invalid_data = valid_data.copy()
    invalid_data["question_id"] = "DE1_Q1"  # DE1 question
    invalid_data["dimension"] = "DE-2"      # But DE-2 dimension
    
    try:
        QuestionEvalInput(**invalid_data)
        print("✗ Should have failed validation!")
    except Exception as e:
        print(f"✓ Correctly rejected invalid input: {e}")
    
    print()


def demo_registry_checksum_validation():
    """Demonstrate registry checksum computation and validation"""
    print("=== REGISTRY CHECKSUM VALIDATION DEMO ===")
    
    registry = get_default_registry()
    
    # Get original checksum
    original_checksum = registry.get_checksum()
    print(f"Original registry checksum: {original_checksum[:16]}...")
    
    # Validate current registry data
    registry_data = registry.get_registry_data()
    is_valid = RegistryChecksumValidator.validate_registry_checksum(
        original_checksum, registry_data
    )
    print(f"Checksum validation: {'✓ PASS' if is_valid else '✗ FAIL'}")
    
    # Simulate registry modification
    modified_data = registry_data.copy()
    modified_data['metadata']['version'] = '2.0.0'  # Modify metadata
    
    new_checksum = RegistryChecksumValidator.compute_registry_checksum(modified_data)
    print(f"Modified registry checksum: {new_checksum[:16]}...")
    
    # Validation should fail
    is_still_valid = RegistryChecksumValidator.validate_registry_checksum(
        original_checksum, modified_data
    )
    print(f"Validation after modification: {'✓ PASS' if is_still_valid else '✗ FAIL (expected)'}")
    
    # Show registry stats
    stats = registry.get_registry_stats()
    print(f"\nRegistry Statistics:")
    print(f"  Total Questions: {stats['total_questions']}")
    print(f"  Total Dimensions: {stats['total_dimensions']}")
    print(f"  Last Modified: {stats['last_modified']}")
    
    print()


def demo_artifact_generation_with_validation():
    """Demonstrate artifact generation with schema validation"""
    print("=== ARTIFACT GENERATION WITH VALIDATION DEMO ===")
    
    registry = get_default_registry()
    scoring_system = ScoringSystem(question_registry=registry)
    checksum = registry.get_checksum()
    
    # Sample evaluation data  
    evaluation_data = {
        "DE-1": [
            {
                "question_id": "DE1_Q1",
                "response": "Sí",
                "evidence_completeness": 0.9,
                "page_reference_quality": 1.0
            },
            {
                "question_id": "DE1_Q2", 
                "response": "Parcial",
                "evidence_completeness": 0.6,
                "page_reference_quality": 0.7
            }
        ],
        "DE-2": [
            {
                "question_id": "DE2_Q1",
                "response": "Sí",
                "evidence_completeness": 0.8,
                "page_reference_quality": 0.9
            }
        ],
        "DE-3": [
            {
                "question_id": "DE3_Q1",
                "response": "No",
                "evidence_completeness": 0.0,
                "page_reference_quality": 0.0
            }
        ],
        "DE-4": [
            {
                "question_id": "DE4_Q1",
                "response": "NI", 
                "evidence_completeness": 0.0,
                "page_reference_quality": 0.0
            }
        ]
    }
    
    # Generate point evaluation artifact (with schema validation)
    print("Generating validated point evaluation artifact...")
    try:
        point_artifact = scoring_system.process_point_evaluation(
            1, evaluation_data, checksum
        )
        
        print("✓ Point evaluation artifact generated and validated:")
        print(f"  Point ID: {point_artifact['point_id']}")
        print(f"  Final Score: {point_artifact['final_score']:.4f}")
        print(f"  Compliance Level: {point_artifact['compliance_level']}")
        print(f"  Total Questions: {point_artifact['total_questions']}")
        print(f"  Schema Validated: {point_artifact['_schema_metadata']['is_validated']}")
        print(f"  Validation Timestamp: {point_artifact['_schema_metadata']['validation_timestamp']}")
        
    except Exception as e:
        print(f"✗ Artifact generation failed: {e}")
    
    print()


def demo_stage_metadata_generation():
    """Demonstrate stage metadata generation"""
    print("=== STAGE METADATA GENERATION DEMO ===")
    
    registry = get_default_registry()
    
    # Create stage metadata using decorator
    @validate_artifact_json(StageMeta)
    def generate_stage_metadata(execution_context=None):
        """Generate stage execution metadata"""
        context = execution_context or {}
        
        return {
            "stage_id": "L_classification_evaluation",
            "version": "1.0.0",
            "municipality_code": context.get("municipality_code", "05001"),
            "pdt_id": context.get("pdt_id", "PDT_2024_001"),
            "evaluation_mode": "schema_validated",
            "points_processed": 10,
            "questions_processed": 47,
            "success_rate": 0.96,
            "average_confidence": 0.87,
            "validation_errors": [],
            "registry_checksum": registry.get_checksum(),
            "peak_memory_mb": 128.5,
            "cpu_usage_percent": 15.2
        }
    
    try:
        # Generate metadata
        execution_context = {
            "municipality_code": "05001",
            "pdt_id": "PDT_MEDELLIN_2024", 
            "points_processed": 10,
            "questions_processed": 470
        }
        
        metadata_artifact = generate_stage_metadata(execution_context)
        
        print("✓ Stage metadata artifact generated and validated:")
        print(f"  Execution ID: {metadata_artifact['execution_id']}")
        print(f"  Municipality: {metadata_artifact['municipality_code']}")
        print(f"  PDT ID: {metadata_artifact['pdt_id']}")
        print(f"  Success Rate: {metadata_artifact['success_rate']:.2%}")
        print(f"  Schema Validated: {metadata_artifact['_schema_metadata']['is_validated']}")
        
        # Mark completion
        stage_meta = StageMeta(**{k: v for k, v in metadata_artifact.items() if not k.startswith('_')})
        stage_meta.mark_completion()
        print(f"  Processing Duration: {stage_meta.processing_duration_ms}ms")
        
    except Exception as e:
        print(f"✗ Metadata generation failed: {e}")
    
    print()


def demo_ci_testing_integration():
    """Demonstrate CI testing integration with schema validation"""
    print("=== CI TESTING INTEGRATION DEMO ===")
    
    # Simulate CI test that validates artifact schemas
    def validate_all_artifacts():
        """CI test function that validates all artifact types"""
        test_results = []
        
        # Test QuestionEvalInput validation
        try:
            test_data = {
                "question_id": "DE1_Q1",
                "dimension": "DE-1",
                "response": "Sí", 
                "evidence_completeness": 0.8,
                "page_reference_quality": 0.9,
                "registry_checksum": "test_checksum"
            }
            QuestionEvalInput(**test_data)
            test_results.append(("QuestionEvalInput", True, None))
        except Exception as e:
            test_results.append(("QuestionEvalInput", False, str(e)))
        
        # Test PointEvalOutput validation
        try:
            test_data = {
                "point_id": 1,
                "final_score": 0.78,
                "compliance_level": "CUMPLE",
                "dimension_scores": {
                    DimensionType.DE_1: 0.80,
                    DimensionType.DE_2: 0.75,
                    DimensionType.DE_3: 0.82, 
                    DimensionType.DE_4: 0.70
                },
                "dimension_weights": {
                    DimensionType.DE_1: 0.30,
                    DimensionType.DE_2: 0.25,
                    DimensionType.DE_3: 0.25,
                    DimensionType.DE_4: 0.20  
                },
                "dimension_compliance": {
                    DimensionType.DE_1: "CUMPLE",
                    DimensionType.DE_2: "CUMPLE_PARCIAL",
                    DimensionType.DE_3: "CUMPLE",
                    DimensionType.DE_4: "CUMPLE_PARCIAL"
                },
                "total_questions": 47,
                "expected_questions": 47,
                "completeness_ratio": 1.0,
                "overall_evidence_quality": 0.80,
                "citation_density": 0.85,
                "assessment_robustness": 0.90,
                "registry_checksum": "test_checksum",
                "processing_duration_ms": 150
            }
            PointEvalOutput(**test_data)
            test_results.append(("PointEvalOutput", True, None))
        except Exception as e:
            test_results.append(("PointEvalOutput", False, str(e)))
        
        return test_results
    
    # Run CI tests
    results = validate_all_artifacts()
    
    print("CI Schema Validation Test Results:")
    total_tests = len(results)
    passed_tests = sum(1 for _, passed, _ in results if passed)
    
    for schema_name, passed, error in results:
        status = "✓ PASS" if passed else f"✗ FAIL: {error}"
        print(f"  {schema_name}: {status}")
    
    print(f"\nTest Summary: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("✅ All schema validation tests passed - ready for deployment")
    else:
        print("❌ Some schema validation tests failed - review before deployment")
    
    print()


if __name__ == "__main__":
    print("L_classification_evaluation Schema Validation Demo")
    print("=" * 55)
    print()
    
    demo_question_input_validation()
    demo_registry_checksum_validation() 
    demo_artifact_generation_with_validation()
    demo_stage_metadata_generation()
    demo_ci_testing_integration()
    
    print("🎉 Demo completed successfully!")
    print("Schema validation system is ready for production use.")