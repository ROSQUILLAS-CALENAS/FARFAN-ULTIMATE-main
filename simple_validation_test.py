#!/usr/bin/env python3

# Simple validation test for deterministic_shield module

try:
# # #     from deterministic_shield import EvaluationReport  # Module not found  # Module not found  # Module not found
    print("✓ Successfully imported EvaluationReport")
    
    # Create instance
    report = EvaluationReport(satisfied=True, results=[], first_failure=None)
    print("✓ Successfully created EvaluationReport instance")
    
    # Test simple validation
    test_data = {
        "points": {
            "point_1": {
                "final_score": 0.8,
                "dimensions": {
                    "DE-1": {
                        "weighted_average": 0.8,
                        "questions": {f"Q{i}": {"final_score": 0.8} for i in range(1, 48)}
                    }
                }
            }
        }
    }
    
    result = report.validate_stage_rules(test_data, "test_simple")
    print(f"✓ Validation completed with status: {result['validation_summary']['overall_status']}")
    print(f"✓ Rules checked: {result['validation_summary']['total_rules_checked']}")
    print(f"✓ Fingerprint generated: {result['canonical_fingerprint'][:16]}...")
    
    print("\n🎉 All tests passed - module is working correctly!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()