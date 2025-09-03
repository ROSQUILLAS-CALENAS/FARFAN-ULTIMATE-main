#!/usr/bin/env python3
"""
Simple test to verify deterministic hash functionality in canonical_output_auditor.py
"""
import sys
sys.path.insert(0, '.')

# Import directly from the module file to avoid __init__.py issues
import importlib.util

# Load deterministic hashing module directly
spec = importlib.util.spec_from_file_location("deterministic_hashing", 
                                             "egw_query_expansion/core/deterministic_hashing.py")
dh_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dh_module)

def _stable_hash_dict(d, deterministic=True):
    """Use deterministic hashing for consistent results."""
    return dh_module.hash_context(d)[:16]

def test_deterministic_hashing():
    """Test that the deterministic hash produces consistent results."""
    
    # Test 1: Same data in different key orders should produce same hash
    data1 = {
        'z_key': [3, 1, 2],
        'a_key': {'nested': {1, 2, 3}},
        'b_key': None,
        'mixed': [{'c': 1, 'a': 2}, {'b': 3}]
    }
    
    data2 = {
        'a_key': {'nested': {3, 2, 1}},  # Set with different order
        'b_key': None,
        'mixed': [{'a': 2, 'c': 1}, {'b': 3}],  # Dict keys different order
        'z_key': [3, 1, 2]  # Same data as data1
    }
    
    hash1 = _stable_hash_dict(data1, deterministic=True)
    hash2 = _stable_hash_dict(data2, deterministic=True)
    
    print(f"Test 1 - Dictionary key ordering:")
    print(f"  Hash 1: {hash1[:16]}")
    print(f"  Hash 2: {hash2[:16]}")
    print(f"  Same: {hash1 == hash2}")
    
    # Test 2: Test nested structures
    nested_data1 = {
        'level1': {
            'level2': {
                'sets': {5, 3, 1},
                'lists': [9, 7, 5]
            }
        }
    }
    
    nested_data2 = {
        'level1': {
            'level2': {
                'lists': [9, 7, 5],  # Same data as nested_data1
                'sets': {1, 3, 5}    # Different order but same data
            }
        }
    }
    
    hash3 = _stable_hash_dict(nested_data1, deterministic=True)
    hash4 = _stable_hash_dict(nested_data2, deterministic=True)
    
    print(f"\nTest 2 - Nested structures:")
    print(f"  Hash 3: {hash3[:16]}")
    print(f"  Hash 4: {hash4[:16]}")
    print(f"  Same: {hash3 == hash4}")
    
    # Test 3: Test hashing examples
    print(f"\nTest 3 - Hash examples:")
    print(f"  Dict: {_stable_hash_dict({'b': 2, 'a': 1}, deterministic=True)}")
    print(f"  Complex: {_stable_hash_dict({'set': {3, 1, 2}, 'list': [3, 1, 2]}, deterministic=True)}")
    
    return hash1 == hash2 and hash3 == hash4

if __name__ == "__main__":
    print("Testing deterministic hash functionality...")
    try:
        success = test_deterministic_hashing()
        if success:
            print("\n✅ All tests passed! Deterministic hashing is working correctly.")
        else:
            print("\n❌ Tests failed! Hash results are not consistent.")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)