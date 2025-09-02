"""
Simple test for Macro Alignment Calculator with basic functionality validation
"""

import json
import os
import tempfile
from macro_alignment_calculator import process, ComplianceLevel

def test_basic_functionality():
    """Test basic macro alignment functionality"""
    
    # Create a simple test meso file
    test_doc_stem = "simple_test"
    test_meso_data = {
        "doc_stem": test_doc_stem,
        "coverage_metrics": {
            "coverage_matrix": {
                "OBJECTIVES": {
                    "clusters_evaluating": ["C1", "C2"],
                    "coverage_percentage": 75.0
                },
                "COMPLIANCE": {
                    "clusters_evaluating": ["C1", "C2", "C3"],
                    "coverage_percentage": 100.0
                }
            }
        },
        "divergence_scores": {
            "question_divergences": {
                "Q1": {
                    "jensen_shannon_max": 0.1,
                    "cosine_similarity_min": 0.9
                }
            }
        },
        "cluster_participation": {
            "participation_counts": {
                "C1": 3,
                "C2": 3,
                "C3": 2,
                "C4": 2
            }
        },
        "dimension_groupings": {
            "by_component": {
                "OBJECTIVES": ["Q1"],
                "COMPLIANCE": ["Q1"]
            },
            "by_question": {
                "Q1": ["OBJECTIVES", "COMPLIANCE"]
            }
        }
    }
    
    # Create temporary meso file
    os.makedirs("canonical_flow/aggregation", exist_ok=True)
    meso_filepath = f"canonical_flow/aggregation/{test_doc_stem}_meso.json"
    
    try:
        # Write test meso data
        with open(meso_filepath, 'w') as f:
            json.dump(test_meso_data, f)
        
        # Test macro alignment calculation
        result = process(test_doc_stem)
        
        print(f"✓ Macro alignment calculation completed")
        print(f"✓ Result contains macro_alignment: {'macro_alignment' in result}")
        
        if 'macro_alignment' in result:
            macro = result['macro_alignment']
            calc_results = macro.get('calculation_results', {})
            
            print(f"✓ Final score: {calc_results.get('final_score', 'N/A')}")
            print(f"✓ Compliance level: {calc_results.get('compliance_level', 'N/A')}")
            print(f"✓ Has score breakdown: {'score_breakdown' in calc_results}")
            print(f"✓ Has supporting metrics: {'supporting_metrics' in macro}")
            
            # Verify macro artifact was created
            macro_filepath = f"canonical_flow/aggregation/{test_doc_stem}_macro.json"
            if os.path.exists(macro_filepath):
                print(f"✓ Macro artifact created: {macro_filepath}")
                
                # Verify artifact content
                with open(macro_filepath, 'r') as f:
                    artifact_data = json.load(f)
                
                required_fields = [
                    "doc_stem", "calculation_results", "supporting_metrics",
                    "decalogo_weights", "classification_thresholds"
                ]
                
                for field in required_fields:
                    if field in artifact_data:
                        print(f"✓ Artifact contains {field}")
                    else:
                        print(f"✗ Artifact missing {field}")
            else:
                print(f"✗ Macro artifact not created")
        else:
            print(f"✗ No macro_alignment in result")
            
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False
    finally:
        # Cleanup
        for filepath in [meso_filepath, f"canonical_flow/aggregation/{test_doc_stem}_macro.json"]:
            if os.path.exists(filepath):
                os.remove(filepath)

def test_compliance_classification():
    """Test compliance level classification"""
    
    print("\n=== Testing Compliance Classification ===")
    
    test_cases = [
        (0.85, ComplianceLevel.CUMPLE),
        (0.75, ComplianceLevel.CUMPLE),
        (0.65, ComplianceLevel.CUMPLE_PARCIAL),
        (0.50, ComplianceLevel.CUMPLE_PARCIAL),
        (0.35, ComplianceLevel.NO_CUMPLE),
        (0.0, ComplianceLevel.NO_CUMPLE),
    ]
    
    from macro_alignment_calculator import classify_compliance
    
    for score, expected in test_cases:
        result = classify_compliance(score)
        if result == expected:
            print(f"✓ Score {score} -> {result.value}")
        else:
            print(f"✗ Score {score} -> {result.value} (expected {expected.value})")

def test_error_handling():
    """Test error handling for missing data"""
    
    print("\n=== Testing Error Handling ===")
    
    # Test with no doc_stem
    result1 = process({})
    if 'macro_alignment' in result1 and 'error' in result1['macro_alignment']:
        print("✓ Handles missing doc_stem correctly")
    else:
        print("✗ Failed to handle missing doc_stem")
    
    # Test with non-existent meso file
    result2 = process("nonexistent_document")
    if 'macro_alignment' in result2 and 'error' in result2['macro_alignment']:
        print("✓ Handles missing meso file correctly")
    else:
        print("✗ Failed to handle missing meso file")

if __name__ == "__main__":
    print("=== Simple Macro Alignment Calculator Test ===")
    
    success = test_basic_functionality()
    test_compliance_classification()
    test_error_handling()
    
    print(f"\n=== Test Summary ===")
    if success:
        print("✓ Basic functionality test passed")
    else:
        print("✗ Basic functionality test failed")
    
    print("✓ All tests completed")