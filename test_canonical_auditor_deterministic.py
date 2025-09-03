#!/usr/bin/env python3
"""
Test script for the deterministic flag functionality in canonical_output_auditor.py
"""

import sys
import os
sys.path.insert(0, '.')

import canonical_output_auditor


def test_deterministic_flag():
    """Test that the deterministic flag works correctly"""
    
    # Test data
    test_data = {
        "cluster_audit": {
            "present": ["C1", "C2", "C3", "C4"],
            "complete": True,
            "non_redundant": True,
            "micro": {
                "C1": {
                    "answers": [
                        {"question_id": "q1", "evidence_ids": ["e1", "e2"]},
                        {"question_id": "q2", "evidence_ids": ["e3"]}
                    ],
                    "evidence_linked": True
                },
                "C2": {
                    "answers": [
                        {"question_id": "q3", "evidence_ids": ["e4"]}
                    ],
                    "evidence_linked": True
                }
            }
        },
        "meso_summary": {
            "divergence_stats": {"max": 0.3, "avg": 0.2, "count": 10},
            "items": {
                "item1": {"evidence_coverage": 5},
                "item2": {"evidence_coverage": 3}
            }
        },
        "evidence": {
            "q1": [{"id": "e1"}, {"id": "e2"}],
            "q2": [{"id": "e3"}],
            "q3": [{"id": "e4"}]
        }
    }
    
    print("Testing deterministic flag functionality...")
    
    # Test without deterministic flag (backward compatibility)
    result1 = canonical_output_auditor.process(test_data, {})
    print("✓ Process works without deterministic flag")
    
    # Test with deterministic=False
    result2 = canonical_output_auditor.process(test_data, {'deterministic': False})
    print("✓ Process works with deterministic=False")
    
    # Test with deterministic=True
    result3 = canonical_output_auditor.process(test_data, {'deterministic': True})
    print("✓ Process works with deterministic=True")
    
    # Test multiple runs with deterministic=True should be identical
    result4 = canonical_output_auditor.process(test_data, {'deterministic': True})
    result5 = canonical_output_auditor.process(test_data, {'deterministic': True})
    
    # Check timestamps are deterministic
    if result4['canonical_audit']['timestamp'] == result5['canonical_audit']['timestamp']:
        print("✓ Deterministic timestamps work")
    else:
        print("✗ Deterministic timestamps failed")
        return False
    
    # Check macro synthesis timestamps
    if result4['macro_synthesis']['timestamp'] == result5['macro_synthesis']['timestamp']:
        print("✓ Deterministic macro synthesis timestamps work")
    else:
        print("✗ Deterministic macro synthesis timestamps failed")
        return False
    
    # Check hashes are deterministic
    hash1 = result4['canonical_audit']['replicability']['cluster_audit_hash']
    hash2 = result5['canonical_audit']['replicability']['cluster_audit_hash']
    if hash1 == hash2:
        print("✓ Deterministic hashing works")
    else:
        print("✗ Deterministic hashing failed")
        return False
    
    # Test that non-deterministic runs have different timestamps
    import time
    time.sleep(0.1)  # Small delay
    result6 = canonical_output_auditor.process(test_data, {'deterministic': False})
    
    if result3['canonical_audit']['timestamp'] != result6['canonical_audit']['timestamp']:
        print("✓ Non-deterministic timestamps are different")
    else:
        print("✗ Non-deterministic timestamps are the same (unexpected)")
        return False
    
    print("All tests passed! ✓")
    return True


if __name__ == "__main__":
    success = test_deterministic_flag()
    sys.exit(0 if success else 1)