"""
Demo script for L-Classification Stage Orchestrator
Shows practical usage of the standardized process() API with complete
artifact generation and error handling.
"""

import json
import tempfile
from pathlib import Path
from pprint import pprint

from stage_orchestrator import LClassificationStageOrchestrator, process


def create_sample_data():
    """Create sample classification input data for demonstration"""
    sample_data = {
        "DE-1": [
            {
                "question_id": "DE1_Q1_productos",
                "question_text": "¿Se identifican claramente los productos o entregables del proyecto?",
                "response": "Sí",
                "evidence_completeness": 0.95,
                "page_reference_quality": 1.0,
                "evidence_text": "El documento presenta una tabla detallada con todos los productos esperados en la sección 3.1, incluyendo cronograma y responsables.",
                "page_references": ["p. 15-17", "Anexo A"]
            },
            {
                "question_id": "DE1_Q2_cronograma",
                "question_text": "¿Existe un cronograma claro para la entrega de productos?",
                "response": "Parcial",
                "evidence_completeness": 0.7,
                "page_reference_quality": 0.8,
                "evidence_text": "Se menciona un cronograma general pero faltan algunos hitos específicos.",
                "page_references": ["p. 12"]
            },
            {
                "question_id": "DE1_Q3_presupuesto",
                "question_text": "¿Se especifica el presupuesto asociado a cada producto?",
                "response": "No",
                "evidence_completeness": 0.1,
                "page_reference_quality": 0.2,
                "evidence_text": "No se encontró información presupuestal específica por producto.",
                "page_references": []
            }
        ],
        "DE-2": [
            {
                "question_id": "DE2_Q1_diagnostico",
                "question_text": "¿Se presenta un diagnóstico integral del territorio?",
                "response": "Sí",
                "evidence_completeness": 0.9,
                "page_reference_quality": 0.95,
                "evidence_text": "Diagnóstico comprehensivo en el capítulo 2 con análisis socioeconómico, ambiental y institucional.",
                "page_references": ["p. 25-45", "Mapas 1-5"]
            },
            {
                "question_id": "DE2_Q2_problematica",
                "question_text": "¿Se identifican claramente las problemáticas centrales?",
                "response": "Parcial",
                "evidence_completeness": 0.6,
                "page_reference_quality": 0.7,
                "evidence_text": "Algunas problemáticas están bien identificadas pero falta priorización.",
                "page_references": ["p. 30-33"]
            }
        ],
        "DE-3": [
            {
                "question_id": "DE3_Q1_seguimiento",
                "question_text": "¿Se definen indicadores de seguimiento?",
                "response": "Sí",
                "evidence_completeness": 0.8,
                "page_reference_quality": 0.85,
                "evidence_text": "Marco de indicadores presentado en sección 5 con metodología de seguimiento.",
                "page_references": ["p. 78-82"]
            },
            {
                "question_id": "DE3_Q2_frecuencia",
                "question_text": "¿Se especifica la frecuencia de medición?",
                "response": "NI",
                "evidence_completeness": 0.0,
                "page_reference_quality": 0.0,
                "evidence_text": "No se encontró información sobre frecuencia de medición.",
                "page_references": []
            }
        ],
        "DE-4": [
            {
                "question_id": "DE4_Q1_evaluacion",
                "question_text": "¿Se establecen mecanismos de evaluación?",
                "response": "Parcial",
                "evidence_completeness": 0.5,
                "page_reference_quality": 0.6,
                "evidence_text": "Menciona evaluación pero sin detallar metodología específica.",
                "page_references": ["p. 85"]
            }
        ]
    }
    return sample_data


def demo_basic_orchestration():
    """Demonstrate basic orchestration with single point"""
    print("=== BASIC ORCHESTRATION DEMO ===")
    
    # Create temporary directory and sample data
    temp_dir = Path(tempfile.mkdtemp())
    sample_data = create_sample_data()
    
    # Create input file
    input_file = temp_dir / "classification_input/municipality_1/P1_questions.json"
    input_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created sample input file: {input_file}")
    
    # Initialize orchestrator
    orchestrator = LClassificationStageOrchestrator(precision=4)
    
    # Process using direct orchestrator
    print("\nProcessing single point...")
    results = orchestrator.process(
        data=str(input_file),
        context={"demo_mode": True}
    )
    
    # Display key results
    print(f"\nExecution ID: {results['execution_metadata']['execution_id']}")
    print(f"Processing Status: {results['status_report']['processing_summary']}")
    
    # Show successful points
    if results['status_report']['successful_points']:
        point_id = results['status_report']['successful_points'][0]
        print(f"\nSuccessfully processed point {point_id}")
        
        # Show point summary
        point_key = f"P{point_id:02d}"
        if point_key in results['artifacts']['point_summaries']:
            summary = results['artifacts']['point_summaries'][point_key]
            print(f"Final Score: {summary['final_score']:.4f}")
            print(f"Total Questions: {summary['total_questions']}")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("\nBasic demo completed.\n")


def demo_multiple_points():
    """Demonstrate processing multiple points with different data"""
    print("=== MULTIPLE POINTS DEMO ===")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create different scenarios for different points
    scenarios = {
        1: create_sample_data(),  # Good data
        2: {  # Minimal data
            "DE-1": [{"question_id": "DE1_Q1", "response": "Parcial", "evidence_completeness": 0.3, "page_reference_quality": 0.4}],
            "DE-2": [{"question_id": "DE2_Q1", "response": "No", "evidence_completeness": 0.0, "page_reference_quality": 0.0}],
            "DE-3": [{"question_id": "DE3_Q1", "response": "NI", "evidence_completeness": 0.0, "page_reference_quality": 0.0}],
            "DE-4": [{"question_id": "DE4_Q1", "response": "Sí", "evidence_completeness": 0.8, "page_reference_quality": 0.9}]
        },
        3: {  # High quality data
            "DE-1": [{"question_id": "DE1_Q1", "response": "Sí", "evidence_completeness": 1.0, "page_reference_quality": 1.0}],
            "DE-2": [{"question_id": "DE2_Q1", "response": "Sí", "evidence_completeness": 0.95, "page_reference_quality": 0.98}],
            "DE-3": [{"question_id": "DE3_Q1", "response": "Sí", "evidence_completeness": 0.9, "page_reference_quality": 0.92}],
            "DE-4": [{"question_id": "DE4_Q1", "response": "Parcial", "evidence_completeness": 0.7, "page_reference_quality": 0.8}]
        }
    }
    
    # Create input files
    input_dir = temp_dir / "classification_input/demo_municipality"
    for point_id, data in scenarios.items():
        input_file = input_dir / f"P{point_id}_questions.json"
        input_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Created {len(scenarios)} input files in {input_dir}")
    
    # Process using standalone function
    print("\nProcessing multiple points...")
    results = process(
        data=str(input_dir),
        context={"batch_mode": True}
    )
    
    # Display results summary
    summary = results['status_report']['processing_summary']
    print(f"\nProcessing Summary:")
    print(f"  Total Points: {summary['total_points']}")
    print(f"  Successful: {summary['successful_points']}")
    print(f"  Failed: {summary['failed_points']}")
    print(f"  Success Rate: {summary['success_rate']:.2%}")
    
    # Show individual point scores
    print(f"\nIndividual Point Scores:")
    for point_key, summary in results['artifacts']['point_summaries'].items():
        print(f"  {point_key}: {summary['final_score']:.4f} (Questions: {summary['total_questions']})")
    
    # Show points index
    print(f"\nPoints Index:")
    for point_key, index_info in results['artifacts']['points_index'].items():
        artifacts_count = len(index_info['artifacts_available'])
        print(f"  {point_key}: {index_info['status']} ({artifacts_count} artifacts)")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("\nMultiple points demo completed.\n")


def demo_error_handling():
    """Demonstrate error handling and isolation"""
    print("=== ERROR HANDLING DEMO ===")
    
    temp_dir = Path(tempfile.mkdtemp())
    input_dir = temp_dir / "classification_input/error_demo"
    
    # Create valid and invalid input files
    valid_data = create_sample_data()
    invalid_data = {"invalid_structure": "bad_data"}
    
    # Valid file
    valid_file = input_dir / "P1_questions.json"
    valid_file.parent.mkdir(parents=True, exist_ok=True)
    with open(valid_file, 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, indent=2)
    
    # Invalid file (will cause processing error)
    invalid_file = input_dir / "P2_questions.json"
    with open(invalid_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_data, f, indent=2)
    
    # Another valid file
    another_valid_file = input_dir / "P3_questions.json"
    with open(another_valid_file, 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, indent=2)
    
    print(f"Created mixed valid/invalid files in {input_dir}")
    
    # Process with error handling
    print("\nProcessing with error isolation...")
    results = process(data=str(input_dir))
    
    # Show results
    print(f"\nError Handling Results:")
    print(f"  Successful Points: {results['status_report']['successful_points']}")
    print(f"  Failed Points: {len(results['status_report']['failed_points'])}")
    
    # Show failure details
    if results['status_report']['failed_points']:
        print(f"\nFailure Details:")
        for failure in results['status_report']['failed_points']:
            print(f"  Point {failure['point_id']}: {failure['error']}")
    
    # Show that successful points still have artifacts
    successful_artifacts = sum(1 for key in results['artifacts']['point_summaries'].keys())
    print(f"\nGenerated {successful_artifacts} successful point summaries despite errors")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("\nError handling demo completed.\n")


def demo_deterministic_serialization():
    """Demonstrate deterministic JSON serialization"""
    print("=== DETERMINISTIC SERIALIZATION DEMO ===")
    
    temp_dir = Path(tempfile.mkdtemp())
    sample_data = create_sample_data()
    
    # Create input file
    input_file = temp_dir / "classification_input/serialization_test/P1_questions.json"
    input_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2)
    
    # Process multiple times
    orchestrator = LClassificationStageOrchestrator()
    
    print("Running multiple processing iterations...")
    json_outputs = []
    for i in range(3):
        results = orchestrator.process(data=str(input_file))
        
        # Remove non-deterministic fields for comparison
        clean_results = results.copy()
        clean_results['execution_metadata']['execution_id'] = 'DETERMINISTIC_ID'
        clean_results['execution_metadata']['start_time'] = 'DETERMINISTIC_TIME'
        clean_results['execution_metadata']['end_time'] = 'DETERMINISTIC_TIME'
        clean_results['audit_log'] = []  # Remove timestamps
        
        json_str = orchestrator.serialize_results(clean_results)
        json_outputs.append(json_str)
    
    # Check deterministic behavior
    all_identical = all(output == json_outputs[0] for output in json_outputs)
    print(f"All JSON outputs identical: {all_identical}")
    
    if all_identical:
        print("✓ Deterministic serialization confirmed")
        
        # Show JSON format sample
        print(f"\nJSON format sample (first 500 characters):")
        print(json_outputs[0][:500] + "...")
    else:
        print("✗ Non-deterministic behavior detected")
        
        # Find differences
        for i in range(1, len(json_outputs)):
            if json_outputs[i] != json_outputs[0]:
                print(f"Difference found in iteration {i}")
    
    # Test file output
    output_file = temp_dir / "deterministic_results.json"
    orchestrator.serialize_results(results, output_file)
    
    print(f"\nResults saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size:,} bytes")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("\nSerialization demo completed.\n")


def demo_artifact_types():
    """Demonstrate all artifact types generated"""
    print("=== ARTIFACT TYPES DEMO ===")
    
    temp_dir = Path(tempfile.mkdtemp())
    sample_data = create_sample_data()
    
    # Create input file
    input_file = temp_dir / "classification_input/artifacts_demo/P5_questions.json"
    input_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2)
    
    # Process
    results = process(data=str(input_file))
    
    # Show all artifact types
    artifacts = results['artifacts']
    print(f"Generated Artifact Types:")
    
    for artifact_type, artifact_data in artifacts.items():
        count = len(artifact_data) if isinstance(artifact_data, dict) else 1
        print(f"  {artifact_type}: {count} items")
        
        # Show sample structure for each type
        if artifact_data:
            if isinstance(artifact_data, dict) and artifact_data:
                first_key = list(artifact_data.keys())[0]
                sample_item = artifact_data[first_key]
                
                print(f"    Sample structure: {list(sample_item.keys()) if isinstance(sample_item, dict) else type(sample_item)}")
    
    # Show specific artifact details
    print(f"\nDetailed Artifact Examples:")
    
    # Point Summary
    if 'point_summaries' in artifacts and artifacts['point_summaries']:
        first_summary = list(artifacts['point_summaries'].values())[0]
        print(f"\nPoint Summary:")
        for key, value in first_summary.items():
            print(f"  {key}: {value}")
    
    # Guard Report
    if 'guard_reports' in artifacts and artifacts['guard_reports']:
        first_guard = list(artifacts['guard_reports'].values())[0]
        print(f"\nGuard Report:")
        print(f"  Guard Status: {first_guard['guard_status']}")
        print(f"  Validation Checks: {first_guard['validation_checks']}")
    
    # Composition Trace
    if 'composition_traces' in artifacts and artifacts['composition_traces']:
        first_trace = list(artifacts['composition_traces'].values())[0]
        print(f"\nComposition Trace:")
        print(f"  Component Order: {first_trace['component_execution_order']}")
        print(f"  Dimension Weights: {first_trace['dimension_weights']}")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("\nArtifact types demo completed.\n")


def demo_orchestrator_info():
    """Display orchestrator configuration information"""
    print("=== ORCHESTRATOR CONFIGURATION ===")
    
    orchestrator = LClassificationStageOrchestrator()
    info = orchestrator.get_orchestrator_info()
    
    print("Orchestrator Information:")
    pprint(info, indent=2)
    print()


if __name__ == "__main__":
    print("L-Classification Stage Orchestrator Demo")
    print("=" * 50)
    print()
    
    # Run all demos
    demo_orchestrator_info()
    demo_basic_orchestration()
    demo_multiple_points()
    demo_error_handling()
    demo_deterministic_serialization()
    demo_artifact_types()
    
    print("=" * 50)
    print("All demos completed successfully!")
    print()
    print("Key Features Demonstrated:")
    print("✓ Standardized process() API contract")
    print("✓ Complete artifact generation")
    print("✓ Error isolation between points")
    print("✓ Deterministic JSON serialization")
    print("✓ Comprehensive audit logging")
    print("✓ All L-stage component coordination")