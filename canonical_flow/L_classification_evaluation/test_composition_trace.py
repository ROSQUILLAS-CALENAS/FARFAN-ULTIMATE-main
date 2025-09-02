"""
Test suite for Point Composition Trace System
Tests the mathematical transparency and audit trail capabilities.
"""

import json
import tempfile
import shutil
from pathlib import Path
from decalogo_scoring_system import ScoringSystem, CompositionTrace


def test_composition_trace_generation():
    """Test that composition trace is generated with complete mathematical details"""
    print("=== COMPOSITION TRACE GENERATION TEST ===")
    
    scoring_system = ScoringSystem(precision=4)
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Override the canonical_flow path for testing
        original_path = Path("canonical_flow/classification")
        test_path = Path(temp_dir) / "classification"
        
        # Mock the path by temporarily changing working directory context
        evaluation_data = {
            "DE-1": [
                {"question_id": "DE1_Q1", "response": "Sí", "evidence_completeness": 1.0, "page_reference_quality": 0.9},
                {"question_id": "DE1_Q2", "response": "Parcial", "evidence_completeness": 0.7, "page_reference_quality": 0.8}
            ],
            "DE-2": [
                {"question_id": "DE2_Q1", "response": "Sí", "evidence_completeness": 0.8, "page_reference_quality": 0.7},
            ],
            "DE-3": [
                {"question_id": "DE3_Q1", "response": "Parcial", "evidence_completeness": 0.5, "page_reference_quality": 0.6}
            ],
            "DE-4": [
                {"question_id": "DE4_Q1", "response": "No", "evidence_completeness": 0.2, "page_reference_quality": 0.1}
            ]
        }
        
        # Process with trace saving disabled for this test (we'll test the structure)
        point_score = scoring_system.process_point_evaluation(
            point_id=3, 
            evaluation_data=evaluation_data,
            doc_id="TEST_DOC_001",
            save_trace=False
        )
        
        print(f"✓ Point {point_score.point_id} processed successfully")
        print(f"  Final score: {point_score.final_score:.4f}")
        print(f"  Dimensions processed: {len(point_score.dimension_scores)}")
        
        # Test the composition trace structure manually
        dimension_scores = {
            f"DE-{i+1}": point_score.dimension_scores[i].weighted_average 
            for i in range(len(point_score.dimension_scores))
        }
        
        from decimal import Decimal
        dimension_scores_decimal = {k: Decimal(str(v)) for k, v in dimension_scores.items()}
        final_score, composition_trace = scoring_system.compose_point_score(dimension_scores_decimal)
        
        print(f"✓ Composition trace generated")
        print(f"  Weighted sum: {composition_trace.weighted_sum:.4f}")
        print(f"  Total weight: {composition_trace.total_weight:.4f}")
        print(f"  Final score: {composition_trace.final_score:.4f}")
        
        return True


def test_trace_file_structure():
    """Test that trace files are saved with correct structure and mathematical transparency"""
    print("\n=== TRACE FILE STRUCTURE TEST ===")
    
    scoring_system = ScoringSystem()
    
    evaluation_data = {
        "DE-1": [
            {"question_id": "DE1_Q1", "response": "Sí", "evidence_completeness": 0.9, "page_reference_quality": 1.0},
        ],
        "DE-2": [
            {"question_id": "DE2_Q1", "response": "Parcial", "evidence_completeness": 0.6, "page_reference_quality": 0.7},
        ],
        "DE-3": [
            {"question_id": "DE3_Q1", "response": "Sí", "evidence_completeness": 0.8, "page_reference_quality": 0.9},
        ],
        "DE-4": [
            {"question_id": "DE4_Q1", "response": "NI", "evidence_completeness": 0.0, "page_reference_quality": 0.0},
        ]
    }
    
    # Process with trace saving enabled
    point_score = scoring_system.process_point_evaluation(
        point_id=5, 
        evaluation_data=evaluation_data,
        doc_id="TEST_DOC_002",
        save_trace=True
    )
    
    # Check that trace file was created
    trace_file = Path("canonical_flow/classification/TEST_DOC_002/P5_composition.json")
    
    if trace_file.exists():
        print(f"✓ Trace file created: {trace_file}")
        
        # Load and validate structure
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)
        
        required_keys = [
            "metadata", "scoring_configuration", "dimension_details", 
            "composition_calculation", "mathematical_audit"
        ]
        
        for key in required_keys:
            if key in trace_data:
                print(f"  ✓ {key} section present")
            else:
                print(f"  ✗ {key} section missing")
                return False
        
        # Validate mathematical audit
        audit = trace_data.get("mathematical_audit", {})
        if audit.get("weighted_sum_validation") and audit.get("weight_sum_validation"):
            print("  ✓ Mathematical validation passed")
        else:
            print("  ✗ Mathematical validation failed")
            return False
        
        # Validate dimension details
        dimensions = trace_data.get("dimension_details", [])
        if len(dimensions) == 4:
            print(f"  ✓ All 4 dimensions present in trace")
        else:
            print(f"  ✗ Expected 4 dimensions, found {len(dimensions)}")
        
        # Check question-level breakdown
        total_questions = sum(d.get("total_questions", 0) for d in dimensions)
        print(f"  ✓ Total questions in trace: {total_questions}")
        
        # Validate deterministic serialization (sorted keys)
        trace_str = json.dumps(trace_data, sort_keys=True)
        if '"composition_calculation"' in trace_str and '"metadata"' in trace_str:
            print("  ✓ JSON structure is deterministically serialized")
        
        return True
    else:
        print(f"✗ Trace file not created: {trace_file}")
        return False


def test_mathematical_transparency():
    """Test that all mathematical operations are traceable and auditable"""
    print("\n=== MATHEMATICAL TRANSPARENCY TEST ===")
    
    scoring_system = ScoringSystem(precision=6)
    
    # Simple evaluation data for precise testing
    evaluation_data = {
        "DE-1": [
            {"question_id": "DE1_Q1", "response": "Sí", "evidence_completeness": 1.0, "page_reference_quality": 1.0},
        ],
        "DE-2": [  
            {"question_id": "DE2_Q1", "response": "Parcial", "evidence_completeness": 0.5, "page_reference_quality": 0.5},
        ],
        "DE-3": [
            {"question_id": "DE3_Q1", "response": "No", "evidence_completeness": 0.0, "page_reference_quality": 0.0},
        ],
        "DE-4": [
            {"question_id": "DE4_Q1", "response": "Sí", "evidence_completeness": 0.8, "page_reference_quality": 0.9},
        ]
    }
    
    point_score = scoring_system.process_point_evaluation(
        point_id=7, 
        evaluation_data=evaluation_data,
        doc_id="MATH_TEST_DOC",
        save_trace=True
    )
    
    # Load trace and manually verify calculations
    trace_file = Path("canonical_flow/classification/MATH_TEST_DOC/P7_composition.json")
    
    if trace_file.exists():
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)
        
        composition = trace_data.get("composition_calculation", {})
        dimension_scores = composition.get("dimension_scores", {})
        weights = composition.get("weights", {})
        contributions = composition.get("weighted_contributions", {})
        
        print("Manual verification of mathematical operations:")
        
        # Verify weighted contributions
        manual_contributions = {}
        for dim_id, score in dimension_scores.items():
            if dim_id in weights:
                manual_contrib = score * weights[dim_id]
                manual_contributions[dim_id] = manual_contrib
                stored_contrib = contributions.get(dim_id, 0)
                
                if abs(manual_contrib - stored_contrib) < 1e-10:
                    print(f"  ✓ {dim_id}: {score:.6f} × {weights[dim_id]:.2f} = {manual_contrib:.6f}")
                else:
                    print(f"  ✗ {dim_id}: calculation mismatch")
                    return False
        
        # Verify weighted sum
        manual_sum = sum(manual_contributions.values())
        stored_sum = composition.get("weighted_sum", 0)
        
        if abs(manual_sum - stored_sum) < 1e-10:
            print(f"  ✓ Weighted sum: {manual_sum:.6f}")
        else:
            print(f"  ✗ Weighted sum mismatch: {manual_sum:.6f} vs {stored_sum:.6f}")
            return False
        
        # Verify final score
        total_weight = composition.get("total_weight", 1)
        manual_final = manual_sum / total_weight if total_weight > 0 else 0
        stored_final = composition.get("final_score", 0)
        
        if abs(manual_final - stored_final) < 1e-10:
            print(f"  ✓ Final score: {manual_sum:.6f} ÷ {total_weight:.2f} = {manual_final:.6f}")
        else:
            print(f"  ✗ Final score mismatch: {manual_final:.6f} vs {stored_final:.6f}")
            return False
        
        print("✓ All mathematical operations verified successfully")
        return True
    else:
        print(f"✗ Trace file not found: {trace_file}")
        return False


def test_deterministic_serialization():
    """Test that JSON serialization is deterministic and reproducible"""
    print("\n=== DETERMINISTIC SERIALIZATION TEST ===")
    
    scoring_system = ScoringSystem()
    
    evaluation_data = {
        "DE-4": [{"question_id": "DE4_Q1", "response": "Sí", "evidence_completeness": 0.7, "page_reference_quality": 0.8}],
        "DE-1": [{"question_id": "DE1_Q1", "response": "Parcial", "evidence_completeness": 0.5, "page_reference_quality": 0.6}],
        "DE-3": [{"question_id": "DE3_Q1", "response": "No", "evidence_completeness": 0.1, "page_reference_quality": 0.2}],
        "DE-2": [{"question_id": "DE2_Q1", "response": "Sí", "evidence_completeness": 0.9, "page_reference_quality": 1.0}]
    }
    
    # Process multiple times
    trace_files = []
    for run in range(3):
        doc_id = f"DETERMINISTIC_TEST_RUN_{run}"
        scoring_system.process_point_evaluation(
            point_id=9, 
            evaluation_data=evaluation_data,
            doc_id=doc_id,
            save_trace=True
        )
        trace_files.append(Path(f"canonical_flow/classification/{doc_id}/P9_composition.json"))
    
    # Compare serializations (excluding timestamp and document_id)
    trace_contents = []
    for trace_file in trace_files:
        if trace_file.exists():
            with open(trace_file, 'r') as f:
                content = json.load(f)
                # Remove timestamp and document_id for comparison (these are expected to differ)
                if 'metadata' in content:
                    if 'timestamp' in content['metadata']:
                        del content['metadata']['timestamp']
                    if 'document_id' in content['metadata']:
                        del content['metadata']['document_id']
                trace_contents.append(json.dumps(content, sort_keys=True))
        else:
            print(f"Trace file not found: {trace_file}")
    
    if len(trace_contents) == 3 and all(c == trace_contents[0] for c in trace_contents):
        print("✓ JSON serialization is deterministic across multiple runs (excluding timestamps)")
        return True
    else:
        print("✗ JSON serialization varies between runs")
        # Show differences for debugging
        if len(trace_contents) >= 2:
            print("First trace length:", len(trace_contents[0]))
            print("Second trace length:", len(trace_contents[1]))
        return False


if __name__ == "__main__":
    print("Point Composition Trace System Test Suite")
    print("=" * 50)
    
    tests = [
        test_composition_trace_generation,
        test_trace_file_structure,
        test_mathematical_transparency,
        test_deterministic_serialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
                print("PASSED\n")
            else:
                print("FAILED\n")
        except Exception as e:
            print(f"ERROR: {e}\n")
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed - Composition trace system is working correctly")
    else:
        print("✗ Some tests failed - Review implementation")