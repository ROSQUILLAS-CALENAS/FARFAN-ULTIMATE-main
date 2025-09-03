#!/usr/bin/env python3
"""
Test integration of deterministic hashing in canonical_output_auditor.
"""

import sys
sys.path.insert(0, '.')

def test_canonical_auditor_deterministic():
    """Test that canonical_output_auditor uses deterministic hashing."""
    
    # Test data
    test_data = {
        'cluster_audit': {
            'complete': True,
            'non_redundant': True,
            'micro': {'C1': {'answers': []}}
        },
        'meso_summary': {'divergence_stats': {'max': 0.1, 'avg': 0.05}},
        'macro_synthesis': {'alignment_score': 0.9}
    }
    
    # Import and test deterministic behavior
    import importlib.util
    spec = importlib.util.spec_from_file_location('canonical_output_auditor', 'canonical_output_auditor.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Run twice with deterministic mode
    result1 = module.process(test_data, {'deterministic': True})
    result2 = module.process(test_data, {'deterministic': True})
    
    # Check deterministic results
    audit1 = result1.get('canonical_audit', {})
    audit2 = result2.get('canonical_audit', {})
    
    repl1 = audit1.get('replicability', {})
    repl2 = audit2.get('replicability', {})
    
    print('Testing canonical_output_auditor deterministic behavior:')
    print(f'  Cluster audit hashes match: {repl1.get("cluster_audit_hash") == repl2.get("cluster_audit_hash")}')
    print(f'  Meso summary hashes match: {repl1.get("meso_summary_hash") == repl2.get("meso_summary_hash")}')
    print(f'  Macro synthesis hashes match: {repl1.get("macro_synthesis_hash") == repl2.get("macro_synthesis_hash")}')
    
    # Verify hashes are not None and are consistent
    cluster_hash_match = repl1.get("cluster_audit_hash") == repl2.get("cluster_audit_hash")
    meso_hash_match = repl1.get("meso_summary_hash") == repl2.get("meso_summary_hash") 
    macro_hash_match = repl1.get("macro_synthesis_hash") == repl2.get("macro_synthesis_hash")
    
    all_match = cluster_hash_match and meso_hash_match and macro_hash_match
    
    if all_match and repl1.get("cluster_audit_hash") is not None:
        print('✓ canonical_output_auditor successfully using deterministic hashing')
        return True
    else:
        print('❌ canonical_output_auditor deterministic hashing failed')
        print(f'  Hash 1: {repl1}')
        print(f'  Hash 2: {repl2}')
        return False

if __name__ == "__main__":
    success = test_canonical_auditor_deterministic()
    sys.exit(0 if success else 1)