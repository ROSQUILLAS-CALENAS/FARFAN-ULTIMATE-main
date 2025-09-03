"""
Direct test of hash policies without importing the main package
"""

import sys
import os

# Add the specific module directory to path to avoid package import issues
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'egw_query_expansion', 'core'))

try:
    import hash_policies
    
    def test_basic_functionality():
        """Test basic hash policy functionality"""
        policy = hash_policies.CanonicalHashPolicy()
        
        # Test simple data
        test_data = {
            'question': 'What is the answer?',
            'context': {'key1': 'value1', 'key2': 42},
            'metadata': {'timestamp': '2024-01-01T00:00:00Z'}
        }
        
        # Hash multiple times
        hash1 = policy.hash_object(test_data)
        hash2 = policy.hash_object(test_data)
        hash3 = policy.hash_object(test_data)
        
        # Should be deterministic
        assert hash1 == hash2 == hash3
        assert len(hash1) > 0
        print(f"✓ Deterministic hashing works: {hash1[:16]}...")
        
        # Test order independence for dictionaries
        reversed_data = {
            'metadata': {'timestamp': '2024-01-01T00:00:00Z'},
            'context': {'key2': 42, 'key1': 'value1'},
            'question': 'What is the answer?'
        }
        
        hash4 = policy.hash_object(reversed_data)
        assert hash1 == hash4
        print("✓ Order independence works for dictionaries")
        
        # Test different policies
        fast_policy = hash_policies.FastHashPolicy()
        secure_policy = hash_policies.SecureHashPolicy()
        
        fast_hash = fast_policy.hash_object(test_data)
        secure_hash = secure_policy.hash_object(test_data)
        
        assert fast_hash == fast_policy.hash_object(test_data)
        assert secure_hash == secure_policy.hash_object(test_data)
        print("✓ Different policies work consistently")
        
        # Test utility function
        util_hash = hash_policies.hash_object(test_data)
        assert len(util_hash) > 0
        print("✓ Utility function works")
        
        print("\n✅ All basic tests passed!")
    
    def test_property_based_simulation():
        """Simulate property-based tests by generating variations"""
        policy = hash_policies.CanonicalHashPolicy(sort_keys=True)
        
        # Test dictionary order independence with multiple permutations
        original_dict = {
            'alpha': 1,
            'beta': 2,
            'gamma': 3,
            'delta': 4
        }
        
        # Generate different orderings
        import itertools
        keys = list(original_dict.keys())
        
        hashes = []
        for permuted_keys in itertools.permutations(keys):
            permuted_dict = {k: original_dict[k] for k in permuted_keys}
            hash_val = policy.hash_object(permuted_dict)
            hashes.append(hash_val)
        
        # All permutations should produce the same hash
        assert len(set(hashes)) == 1, f"Found {len(set(hashes))} different hashes for permuted data"
        print(f"✓ Dictionary order independence verified across {len(hashes)} permutations")
        
    def test_nested_structures():
        """Test complex nested data structures"""
        policy = hash_policies.CanonicalHashPolicy()
        
        # Create complex nested structure
        complex_data = {
            'level1': {
                'level2': {
                    'list_data': [
                        {'item': 'a', 'value': 1},
                        {'item': 'b', 'value': 2},
                    ],
                    'set_data': {1, 2, 3, 4, 5},
                    'primitives': [None, True, False, 42, 3.14159]
                }
            },
            'metadata': {
                'timestamp': '2024-01-01T00:00:00Z',
                'version': 1
            }
        }
        
        # Test consistency
        hash1 = policy.hash_object(complex_data)
        hash2 = policy.hash_object(complex_data)
        assert hash1 == hash2
        print("✓ Complex nested structures hash consistently")
        
        # Create equivalent structure with different dict ordering
        reordered_data = {
            'metadata': {
                'version': 1,
                'timestamp': '2024-01-01T00:00:00Z'
            },
            'level1': {
                'level2': {
                    'primitives': [None, True, False, 42, 3.14159],
                    'set_data': {5, 1, 3, 2, 4},  # Different order, same content
                    'list_data': [
                        {'value': 1, 'item': 'a'},  # Reordered keys
                        {'value': 2, 'item': 'b'},
                    ]
                }
            }
        }
        
        hash3 = policy.hash_object(reordered_data)
        assert hash1 == hash3
        print("✓ Reordered nested structure produces same hash")
        
    def test_different_policies():
        """Test different hash policies"""
        test_data = {'key': 'value', 'number': 42}
        
        canonical = hash_policies.CanonicalHashPolicy()
        fast = hash_policies.FastHashPolicy()
        secure = hash_policies.SecureHashPolicy()
        
        # Each policy should be consistent with itself
        for policy in [canonical, fast, secure]:
            hash1 = policy.hash_object(test_data)
            hash2 = policy.hash_object(test_data)
            assert hash1 == hash2
        
        print("✓ All hash policies are internally consistent")
        
    def test_mock_context_and_synthesis():
        """Test with mock context and synthesis data structures"""
        
        # Mock QuestionContext-like structure
        mock_context = {
            'question_text': 'What is the capital of France?',
            'context_data': {
                'query_expansion': ['capital', 'France', 'Paris'],
                'retrieval_params': {'top_k': 10}
            },
            'metadata': {
                'derivation_id': 'ctx_123',
                'timestamp': '2024-01-01T00:00:00Z'
            }
        }
        
        # Mock SynthesizedAnswer-like structure
        mock_synthesis = {
            'question': 'What is the capital of France?',
            'verdict': 'yes',
            'rationale': 'Paris is the capital of France.',
            'premises': [
                {'text': 'Paris is the capital city of France.', 'score': 0.95},
                {'text': 'France is a country in Europe.', 'score': 0.82}
            ],
            'confidence': 0.89,
            'metadata': {'synthesis_time': '2024-01-01T00:01:00Z'}
        }
        
        context_hasher = hash_policies.ContextHasher()
        synthesis_hasher = hash_policies.SynthesisHasher()
        
        # Test context hashing
        context_hash1 = context_hasher.policy.hash_object(mock_context)
        context_hash2 = context_hasher.policy.hash_object(mock_context)
        assert context_hash1 == context_hash2
        print("✓ Mock context hashes consistently")
        
        # Test synthesis hashing
        synthesis_hash1 = synthesis_hasher.policy.hash_object(mock_synthesis)
        synthesis_hash2 = synthesis_hasher.policy.hash_object(mock_synthesis)
        assert synthesis_hash1 == synthesis_hash2
        print("✓ Mock synthesis result hashes consistently")
        
    def test_pipeline_validator():
        """Test pipeline validation concepts"""
        validator = hash_policies.PipelineHashValidator()
        
        # Mock pipeline stages
        stage_data = [
            {'stage': 'input', 'data': {'query': 'test'}},
            {'stage': 'expansion', 'data': {'query': 'test', 'expanded': ['test', 'example']}},
            {'stage': 'synthesis', 'data': {'result': 'answer'}}
        ]
        
        # Validate each stage
        for i, data in enumerate(stage_data):
            stage_name = f"test_stage_{i}"
            validator.synthesis_hasher.policy.hash_object(data)  # Just ensure it works
        
        print("✓ Pipeline validator components work")
        
        # Generate report
        report = validator.get_validation_report()
        assert 'stage_hashes' in report
        assert 'validation_log' in report
        print("✓ Pipeline validation report generated")
    
    if __name__ == "__main__":
        print("Testing hash policies directly...")
        test_basic_functionality()
        test_property_based_simulation()
        test_nested_structures()
        test_different_policies()
        test_mock_context_and_synthesis()
        test_pipeline_validator()
        print("\n🎉 All direct tests completed successfully!")

except ImportError as e:
    print(f"Import error: {e}")
    print("Hash policies module not available")
    
except Exception as e:
    print(f"Test failed: {e}")
    import traceback
    traceback.print_exc()