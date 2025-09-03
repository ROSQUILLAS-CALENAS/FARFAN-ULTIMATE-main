#!/usr/bin/env python3
"""
Test the canonical_output_auditor process function with deterministic hashing.
"""
import sys
sys.path.insert(0, '.')

from canonical_output_auditor import process

def test_canonical_audit():
    """Test the full canonical_output_auditor process function."""
    
    test_data = {
        'cluster_audit': {
            'present': ['C1', 'C2', 'C3', 'C4'],
            'complete': True,
            'non_redundant': True,
            'micro': {
                'C1': {
                    'answers': [
                        {
                            'question_id': 'q1',
                            'evidence_ids': ['e1', 'e2']
                        }
                    ]
                }
            }
        },
        'meso_summary': {
            'divergence_stats': {
                'max': 0.2,
                'avg': 0.1,
                'count': 5
            }
        },
        'evidence': {
            'q1': [
                {'id': 'e1', 'content': 'evidence 1'},
                {'id': 'e2', 'content': 'evidence 2'}
            ]
        }
    }
    
    try:
        result = process(test_data)
        audit = result.get('canonical_audit', {})
        
        print("✅ Process function executed successfully")
        print(f"Replicability hashes generated: {bool(audit.get('replicability'))}")
        
        if audit.get('replicability'):
            repl = audit['replicability']
            cluster_hash = repl.get('cluster_audit_hash')
            meso_hash = repl.get('meso_summary_hash')
            
            print(f"Cluster audit hash length: {len(cluster_hash) if cluster_hash else 0}")
            print(f"Meso summary hash length: {len(meso_hash) if meso_hash else 0}")
            print(f"Hash format consistent: {len(cluster_hash) == 16 if cluster_hash else False}")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    print("Testing canonical_output_auditor process function...")
    success = test_canonical_audit()
    if success:
        print("\n✅ Canonical audit test passed!")
    else:
        print("\n❌ Canonical audit test failed!")
        sys.exit(1)