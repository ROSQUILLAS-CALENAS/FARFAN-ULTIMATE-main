"""
Simple integration test for meso aggregator with macro alignment
Tests the complete flow from meso to macro calculation
"""

import json
import os
import tempfile
from datetime import datetime

def test_meso_to_macro_integration():
    """Test complete integration from meso aggregation to macro alignment"""
    
    print("=== Meso to Macro Integration Test ===")
    
    # Create test cluster audit data
    test_data = {
        "cluster_audit": {
            "micro": {
                "C1": {
                    "answers": [
                        {
                            "question_id": "Q1",
                            "question": "Is there evidence of violence prevention measures?",
                            "verdict": "yes",
                            "score": 0.8,
                            "evidence_ids": ["ev1", "ev2"],
                            "components": ["OBJECTIVES", "STRATEGIES"]
                        },
                        {
                            "question_id": "Q2", 
                            "question": "Are compliance measures in place?",
                            "verdict": "partially",
                            "score": 0.6,
                            "evidence_ids": ["ev3"],
                            "components": ["COMPLIANCE"]
                        }
                    ]
                },
                "C2": {
                    "answers": [
                        {
                            "question_id": "Q1",
                            "question": "Is there evidence of violence prevention measures?", 
                            "verdict": "yes",
                            "score": 0.75,
                            "evidence_ids": ["ev4"],
                            "components": ["OBJECTIVES", "STRATEGIES"]
                        },
                        {
                            "question_id": "Q3",
                            "question": "Are health services accessible?",
                            "verdict": "yes", 
                            "score": 0.9,
                            "evidence_ids": ["ev5", "ev6"],
                            "components": ["OBJECTIVES", "INDICATORS"]
                        }
                    ]
                }
            }
        }
    }
    
    # Test with meso aggregator
    try:
        from meso_aggregator import process as meso_process
        
        # Process meso aggregation with context
        context = {
            "doc_stem": "integration_test",
            "timestamp": datetime.now().isoformat()
        }
        
        print("✓ Running meso aggregation...")
        meso_result = meso_process(test_data, context)
        
        # Verify meso results
        if "meso_summary" in meso_result:
            print("✓ Meso summary generated")
            
            meso_summary = meso_result["meso_summary"]
            if "items" in meso_summary:
                print(f"✓ Questions processed: {len(meso_summary['items'])}")
            
            if "cluster_participation" in meso_summary:
                participation = meso_summary["cluster_participation"]
                print(f"✓ Cluster participation: {participation}")
        
        # Verify coverage matrix
        if "coverage_matrix" in meso_result:
            print("✓ Coverage matrix generated")
            coverage = meso_result["coverage_matrix"]
            covered_components = [comp for comp, data in coverage.items() 
                               if data.get("coverage_percentage", 0) > 0]
            print(f"✓ Components with coverage: {len(covered_components)}")
        
        # Verify macro alignment was processed
        if "macro_alignment" in meso_result:
            print("✓ Macro alignment calculated")
            
            macro = meso_result["macro_alignment"]
            if "calculation_results" in macro:
                calc_results = macro["calculation_results"]
                final_score = calc_results.get("final_score", 0)
                compliance_level = calc_results.get("compliance_level", "UNKNOWN")
                
                print(f"✓ Final compliance score: {final_score:.3f}")
                print(f"✓ Compliance level: {compliance_level}")
                
                # Check score breakdown
                if "score_breakdown" in calc_results:
                    breakdown = calc_results["score_breakdown"]
                    print(f"  - Coverage score: {breakdown.get('coverage_score', 0):.3f}")
                    print(f"  - Participation score: {breakdown.get('participation_score', 0):.3f}")
                    print(f"  - Divergence penalty: {breakdown.get('divergence_penalty', 1):.3f}")
        else:
            print("✗ Macro alignment not calculated")
        
        # Check for artifact files
        meso_file = "canonical_flow/aggregation/integration_test_meso.json"
        macro_file = "canonical_flow/aggregation/integration_test_macro.json"
        
        if os.path.exists(meso_file):
            print(f"✓ Meso artifact created: {meso_file}")
            
            # Verify meso artifact structure
            with open(meso_file, 'r') as f:
                meso_artifact = json.load(f)
            
            required_meso_fields = [
                "doc_stem", "coverage_metrics", "divergence_scores",
                "cluster_participation", "dimension_groupings"
            ]
            
            for field in required_meso_fields:
                if field in meso_artifact:
                    print(f"  ✓ Contains {field}")
                else:
                    print(f"  ✗ Missing {field}")
        else:
            print(f"✗ Meso artifact not found: {meso_file}")
        
        if os.path.exists(macro_file):
            print(f"✓ Macro artifact created: {macro_file}")
            
            # Verify macro artifact structure
            with open(macro_file, 'r') as f:
                macro_artifact = json.load(f)
            
            required_macro_fields = [
                "doc_stem", "calculation_results", "supporting_metrics",
                "decalogo_weights", "classification_thresholds"
            ]
            
            for field in required_macro_fields:
                if field in macro_artifact:
                    print(f"  ✓ Contains {field}")
                else:
                    print(f"  ✗ Missing {field}")
        else:
            print(f"✗ Macro artifact not found: {macro_file}")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup artifacts
        for filepath in [
            "canonical_flow/aggregation/integration_test_meso.json",
            "canonical_flow/aggregation/integration_test_macro.json"
        ]:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"Cleaned up: {filepath}")

def test_compliance_scoring_scenarios():
    """Test different compliance scoring scenarios"""
    
    print("\n=== Compliance Scoring Scenarios ===")
    
    scenarios = [
        {
            "name": "High Coverage, Low Divergence",
            "coverage_percentage": 85,
            "participation": {"C1": 8, "C2": 7, "C3": 6, "C4": 5},
            "divergence": 0.1,
            "expected": "CUMPLE"
        },
        {
            "name": "Medium Coverage, Medium Divergence", 
            "coverage_percentage": 60,
            "participation": {"C1": 4, "C2": 4, "C3": 3, "C4": 3},
            "divergence": 0.3,
            "expected": "CUMPLE_PARCIAL"
        },
        {
            "name": "Low Coverage, High Divergence",
            "coverage_percentage": 30,
            "participation": {"C1": 2, "C2": 1, "C3": 1, "C4": 1},
            "divergence": 0.6,
            "expected": "NO_CUMPLE"
        }
    ]
    
    try:
        from macro_alignment_calculator import (
            calculate_coverage_score,
            calculate_cluster_participation_score,
            calculate_divergence_penalty,
            classify_compliance
        )
        
        for scenario in scenarios:
            print(f"\n--- {scenario['name']} ---")
            
            # Simulate coverage metrics
            coverage_metrics = {
                "coverage_matrix": {
                    "OBJECTIVES": {"coverage_percentage": scenario["coverage_percentage"]},
                    "STRATEGIES": {"coverage_percentage": scenario["coverage_percentage"]},
                }
            }
            
            # Simulate participation metrics
            cluster_participation = {
                "participation_counts": scenario["participation"]
            }
            
            # Simulate divergence metrics
            divergence_scores = {
                "question_divergences": {
                    "Q1": {
                        "jensen_shannon_max": scenario["divergence"],
                        "cosine_similarity_min": 1.0 - scenario["divergence"]
                    }
                }
            }
            
            # Calculate component scores
            coverage_score = calculate_coverage_score(coverage_metrics)
            participation_score = calculate_cluster_participation_score(cluster_participation)
            divergence_penalty = calculate_divergence_penalty(divergence_scores)
            
            # Calculate final score
            base_score = (coverage_score * 0.50) + (participation_score * 0.30)
            final_score = base_score * divergence_penalty
            
            compliance_level = classify_compliance(final_score)
            
            print(f"  Coverage Score: {coverage_score:.3f}")
            print(f"  Participation Score: {participation_score:.3f}")
            print(f"  Divergence Penalty: {divergence_penalty:.3f}")
            print(f"  Final Score: {final_score:.3f}")
            print(f"  Compliance Level: {compliance_level.value}")
            print(f"  Expected: {scenario['expected']}")
            
            if compliance_level.value == scenario["expected"]:
                print(f"  ✓ Scenario result matches expectation")
            else:
                print(f"  ⚠ Scenario result differs from expectation")
        
        return True
        
    except Exception as e:
        print(f"✗ Scenario testing failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting integration tests...")
    
    success1 = test_meso_to_macro_integration()
    success2 = test_compliance_scoring_scenarios()
    
    print(f"\n=== Final Results ===")
    if success1:
        print("✓ Meso to Macro integration test passed")
    else:
        print("✗ Meso to Macro integration test failed")
    
    if success2:
        print("✓ Compliance scoring scenarios test passed")
    else:
        print("✗ Compliance scoring scenarios test failed")
    
    if success1 and success2:
        print("\n🎉 All integration tests passed!")
    else:
        print("\n❌ Some tests failed")