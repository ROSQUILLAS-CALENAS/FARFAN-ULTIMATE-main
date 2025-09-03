#!/usr/bin/env python3
"""
Simple test to verify deterministic hash functionality in canonical_output_auditor.py
"""
import sys
sys.path.insert(0, '.')

from canonical_output_auditor import deterministic_hash, _deterministic_serialize

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
        'z_key': [1, 2, 3]  # List with different order
    }
    
    hash1 = deterministic_hash(data1)
    hash2 = deterministic_hash(data2)
    
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
                'lists': [5, 7, 9],  # Different order
                'sets': {1, 3, 5}    # Different order
            }
        }
    }
    
    hash3 = deterministic_hash(nested_data1)
    hash4 = deterministic_hash(nested_data2)
    
    print(f"\nTest 2 - Nested structures:")
    print(f"  Hash 3: {hash3[:16]}")
    print(f"  Hash 4: {hash4[:16]}")
    print(f"  Same: {hash3 == hash4}")
    
    # Test 3: Test serialization directly
    print(f"\nTest 3 - Serialization examples:")
    print(f"  Dict: {_deterministic_serialize({'b': 2, 'a': 1})}")
    print(f"  Set:  {_deterministic_serialize({3, 1, 2})}")
    print(f"  List: {_deterministic_serialize([3, 1, 2])}")
    
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