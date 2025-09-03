#!/usr/bin/env python3
"""
Manual property tests for deterministic hashing (simulating Hypothesis tests)
"""

import sys
import os
sys.path.insert(0, '.')

def run_property_tests():
    """Run manual property tests for deterministic hashing."""
    
    # Import directly to avoid package issues
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("deterministic_hashing", 
                                                 "egw_query_expansion/core/deterministic_hashing.py")
    dh_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dh_module)
    
    hash_context = dh_module.hash_context
    hash_synthesis = dh_module.hash_synthesis
    combine_hashes = dh_module.combine_hashes

    print('Running Hypothesis-like property tests...')

    # Test 1: Determinism property
    test_context = {'question': 'test', 'data': {'nested': {'key': 'value'}}}
    hashes = [hash_context(test_context) for _ in range(10)]
    determinism_ok = all(h == hashes[0] for h in hashes)
    print(f'✓ Context hash determinism: {determinism_ok}')

    # Test 2: Different data produces different hashes
    different_context = {'question': 'different', 'data': {'nested': {'key': 'value'}}}
    different_hash = hash_context(different_context)
    different_ok = different_hash != hashes[0]
    print(f'✓ Different data -> different hashes: {different_ok}')

    # Test 3: Order independence 
    dict1 = {'z': 3, 'a': 1, 'b': 2}
    dict2 = {'a': 1, 'b': 2, 'z': 3}
    hash1 = hash_context(dict1)
    hash2 = hash_context(dict2) 
    order_independence_ok = hash1 == hash2
    print(f'✓ Key order independence: {order_independence_ok}')

    # Test 4: Synthesis hashing
    synthesis1 = {'answer': 'yes', 'confidence': 0.8, 'evidence': [{'id': 1}, {'id': 2}]}
    synthesis2 = {'evidence': [{'id': 1}, {'id': 2}], 'answer': 'yes', 'confidence': 0.8}
    synth_hash1 = hash_synthesis(synthesis1)
    synth_hash2 = hash_synthesis(synthesis2)
    synthesis_order_ok = synth_hash1 == synth_hash2
    print(f'✓ Synthesis order independence: {synthesis_order_ok}')

    # Test 5: Hash combination commutativity
    h1, h2, h3 = 'abc', 'def', 'ghi'
    combined1 = combine_hashes(h1, h2, h3)
    combined2 = combine_hashes(h3, h1, h2)
    combined3 = combine_hashes(h2, h3, h1)
    combine_commutativity_ok = combined1 == combined2 == combined3
    print(f'✓ Combine commutativity: {combine_commutativity_ok}')

    # Test 6: Nested structure consistency
    nested1 = {
        'level1': {
            'level2': [{'a': 1, 'b': 2}, {'c': 3}],
            'sets': {4, 5, 6}
        }
    }
    nested2 = {
        'level1': {
            'sets': {6, 4, 5},  # Different order
            'level2': [{'b': 2, 'a': 1}, {'c': 3}]  # Different key order
        }
    }
    nested_hash1 = hash_context(nested1)
    nested_hash2 = hash_context(nested2)
    nested_consistency_ok = nested_hash1 == nested_hash2
    print(f'✓ Nested structure consistency: {nested_consistency_ok}')
    
    # Test 7: Empty and special values
    empty_context = {}
    none_context = {'key': None}
    empty_hash1 = hash_context(empty_context)
    empty_hash2 = hash_context(empty_context)
    none_hash1 = hash_context(none_context)
    none_hash2 = hash_context(none_context)
    special_values_ok = (empty_hash1 == empty_hash2) and (none_hash1 == none_hash2)
    print(f'✓ Special values consistency: {special_values_ok}')

    all_tests_passed = all([
        determinism_ok,
        different_ok,
        order_independence_ok,
        synthesis_order_ok,
        combine_commutativity_ok,
        nested_consistency_ok,
        special_values_ok
    ])

    if all_tests_passed:
        print('\n🎉 All property tests passed!')
        return True
    else:
        print('\n❌ Some property tests failed!')
        return False

if __name__ == "__main__":
    success = run_property_tests()
    sys.exit(0 if success else 1)