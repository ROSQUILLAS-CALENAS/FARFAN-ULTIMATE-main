"""
Property-based tests for deterministic hashing system

Uses Hypothesis to generate random permutations of context data and synthesis 
results to verify that hash functions produce identical outputs regardless of 
input order or structure variations.

Test categories:
- Hash determinism and consistency
- Order independence for dictionaries and lists
- Type normalization and canonicalization
- Circular reference handling
- Pipeline validation and integration
"""

import json
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import pytest
from hypothesis import assume, given, settings, strategies as st

from ..core.hash_policies import (
    CanonicalHashPolicy,
    ContextHasher,
    DEFAULT_HASH_POLICY,
    FastHashPolicy,
    PipelineHashValidator,
    SecureHashPolicy,
    SynthesisHasher,
    hash_object,
)


# Test data structures
@dataclass
class MockEvidence:
    text: str
    source: str
    confidence: float = 0.8


@dataclass
class MockContext:
    question_text: str
    context_data: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'question_text': self.question_text,
            'context_data': self.context_data,
            'metadata': self.metadata
        }


@dataclass
class MockSynthesisResult:
    question: str
    answer: str
    evidence: List[MockEvidence]
    confidence: float
    metadata: Dict[str, Any]


# Hypothesis strategies for generating test data
@st.composite
def text_strategy(draw):
    """Generate random text strings"""
    return draw(st.text(min_size=1, max_size=100))


@st.composite
def evidence_strategy(draw):
    """Generate MockEvidence objects"""
    return MockEvidence(
        text=draw(text_strategy()),
        source=draw(text_strategy()),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0, width=16))
    )


@st.composite
def context_data_strategy(draw):
    """Generate nested context data dictionaries"""
    return draw(st.recursive(
        st.one_of(
            st.integers(),
            st.floats(width=16),
            st.booleans(),
            st.text(),
            st.none(),
        ),
        lambda children: st.one_of(
            st.dictionaries(st.text(), children, max_size=5),
            st.lists(children, max_size=5),
        ),
        max_leaves=10
    ))


@st.composite
def mock_context_strategy(draw):
    """Generate MockContext objects"""
    return MockContext(
        question_text=draw(text_strategy()),
        context_data=draw(context_data_strategy()),
        metadata={
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'version': draw(st.integers(min_value=1, max_value=100))
        }
    )


@st.composite
def mock_synthesis_strategy(draw):
    """Generate MockSynthesisResult objects"""
    return MockSynthesisResult(
        question=draw(text_strategy()),
        answer=draw(text_strategy()),
        evidence=draw(st.lists(evidence_strategy(), min_size=1, max_size=5)),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0, width=16)),
        metadata=draw(context_data_strategy())
    )


class TestHashPolicies:
    """Test suite for hash policy implementations"""
    
    @given(st.data())
    @settings(max_examples=50)
    def test_canonical_policy_determinism(self, data):
        """Test that CanonicalHashPolicy produces consistent hashes"""
        policy = CanonicalHashPolicy()
        
        # Generate random object
        obj = data.draw(context_data_strategy())
        
        # Hash multiple times
        hash1 = policy.hash_object(obj)
        hash2 = policy.hash_object(obj)
        hash3 = policy.hash_object(obj)
        
        # All hashes should be identical
        assert hash1 == hash2 == hash3
        assert isinstance(hash1, str)
        assert len(hash1) > 0
    
    @given(st.dictionaries(st.text(), st.integers(), min_size=2, max_size=10))
    @settings(max_examples=50)
    def test_dict_order_independence(self, original_dict):
        """Test that dictionary key ordering doesn't affect hash"""
        policy = CanonicalHashPolicy(sort_keys=True)
        
        # Create different orderings of the same dictionary
        keys = list(original_dict.keys())
        assume(len(keys) >= 2)
        
        # Original dict
        hash1 = policy.hash_object(original_dict)
        
        # Reverse key order
        reversed_dict = {k: original_dict[k] for k in reversed(keys)}
        hash2 = policy.hash_object(reversed_dict)
        
        # Random permutation using OrderedDict
        import random
        shuffled_keys = keys.copy()
        random.shuffle(shuffled_keys)
        shuffled_dict = OrderedDict((k, original_dict[k]) for k in shuffled_keys)
        hash3 = policy.hash_object(shuffled_dict)
        
        # All hashes should be identical
        assert hash1 == hash2 == hash3
    
    @given(st.lists(st.integers(), min_size=2, max_size=10))
    @settings(max_examples=50)
    def test_list_order_sensitivity(self, original_list):
        """Test that list ordering DOES affect hash (as expected)"""
        assume(len(set(original_list)) > 1)  # Ensure list has distinct elements
        
        policy = CanonicalHashPolicy()
        
        # Original list
        hash1 = policy.hash_object(original_list)
        
        # Reversed list
        reversed_list = list(reversed(original_list))
        hash2 = policy.hash_object(reversed_list)
        
        # Hashes should be different if order is different
        if original_list != reversed_list:
            assert hash1 != hash2
    
    @given(st.sets(st.integers(), min_size=2, max_size=10))
    @settings(max_examples=50)
    def test_set_order_independence(self, original_set):
        """Test that set element ordering doesn't affect hash"""
        policy = CanonicalHashPolicy()
        
        # Convert to list and back to set multiple times to test ordering
        list1 = list(original_set)
        set1 = set(list1)
        hash1 = policy.hash_object(set1)
        
        list2 = list(reversed(list1))
        set2 = set(list2)
        hash2 = policy.hash_object(set2)
        
        # Hashes should be identical (sets are unordered)
        assert hash1 == hash2
    
    @given(mock_context_strategy())
    @settings(max_examples=30)
    def test_nested_structure_consistency(self, context):
        """Test hash consistency for nested data structures"""
        policy = CanonicalHashPolicy()
        
        # Hash original object
        hash1 = policy.hash_object(context)
        
        # Create equivalent structure through different construction
        equivalent_context = MockContext(
            question_text=context.question_text,
            context_data=context.context_data.copy(),
            metadata=context.metadata.copy()
        )
        hash2 = policy.hash_object(equivalent_context)
        
        # Hashes should be identical
        assert hash1 == hash2
    
    def test_circular_reference_handling(self):
        """Test that circular references are handled gracefully"""
        policy = CanonicalHashPolicy(handle_circular_refs=True)
        
        # Create circular reference
        obj1 = {'name': 'obj1'}
        obj2 = {'name': 'obj2', 'ref': obj1}
        obj1['ref'] = obj2
        
        # Should not raise exception and should produce consistent hash
        hash1 = policy.hash_object(obj1)
        hash2 = policy.hash_object(obj1)
        
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0
    
    @given(mock_context_strategy())
    @settings(max_examples=30)
    def test_policy_comparison(self, context):
        """Test different hash policies on same data"""
        canonical = CanonicalHashPolicy()
        fast = FastHashPolicy()
        secure = SecureHashPolicy()
        
        # All should produce valid hashes
        canonical_hash = canonical.hash_object(context)
        fast_hash = fast.hash_object(context)
        secure_hash = secure.hash_object(context)
        
        assert isinstance(canonical_hash, str)
        assert isinstance(fast_hash, str) 
        assert isinstance(secure_hash, str)
        
        # Each should be consistent with itself
        assert canonical_hash == canonical.hash_object(context)
        assert fast_hash == fast.hash_object(context)
        assert secure_hash == secure.hash_object(context)
    
    @given(st.data())
    @settings(max_examples=30)
    def test_type_normalization(self, data):
        """Test that type normalization produces consistent results"""
        policy = CanonicalHashPolicy(normalize_types=True)
        
        # Create objects with same data but different types
        dict_obj = {'value': 42, 'name': 'test'}
        mock_obj = MockContext(
            question_text='test',
            context_data={'value': 42},
            metadata={'name': 'test'}
        )
        
        # With type normalization, these should hash differently
        hash1 = policy.hash_object(dict_obj)
        hash2 = policy.hash_object(mock_obj)
        
        assert hash1 != hash2  # Different types should have different hashes


class TestContextHasher:
    """Test suite for ContextHasher functionality"""
    
    def test_context_hasher_initialization(self):
        """Test ContextHasher initialization and policy assignment"""
        # Default policy
        hasher1 = ContextHasher()
        assert hasher1.policy is not None
        
        # Custom policy
        policy = FastHashPolicy()
        hasher2 = ContextHasher(policy)
        assert hasher2.policy is policy
    
    @given(mock_synthesis_strategy())
    @settings(max_examples=30)
    def test_synthesis_result_hashing(self, synthesis_result):
        """Test synthesis result hashing consistency"""
        hasher = SynthesisHasher()
        
        # Hash multiple times
        hash1 = hasher.hash_synthesis_result(synthesis_result)
        hash2 = hasher.hash_synthesis_result(synthesis_result)
        
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0


class TestPipelineHashValidator:
    """Test suite for PipelineHashValidator"""
    
    def test_validator_initialization(self):
        """Test PipelineHashValidator initialization"""
        validator = PipelineHashValidator()
        
        assert validator.context_hasher is not None
        assert validator.synthesis_hasher is not None
        assert isinstance(validator.stage_hashes, dict)
        assert isinstance(validator.validation_log, list)
    
    @given(mock_synthesis_strategy())
    @settings(max_examples=20)
    def test_synthesis_validation(self, synthesis_result):
        """Test synthesis result validation"""
        validator = PipelineHashValidator()
        
        # Validate synthesis result
        result = validator.validate_synthesis_consistency(
            'test_stage', 
            synthesis_result
        )
        
        assert isinstance(result, bool)
        assert len(validator.validation_log) > 0
        assert 'test_stage' in validator.stage_hashes
    
    def test_validation_report_generation(self):
        """Test validation report generation"""
        validator = PipelineHashValidator()
        
        # Add some validation data
        mock_result = MockSynthesisResult(
            question='test',
            answer='test answer',
            evidence=[MockEvidence('evidence', 'source')],
            confidence=0.8,
            metadata={}
        )
        
        validator.validate_synthesis_consistency('stage1', mock_result)
        
        # Generate report
        report = validator.get_validation_report()
        
        assert 'stage_hashes' in report
        assert 'validation_log' in report
        assert 'total_validations' in report
        assert 'passed_validations' in report
        assert 'failed_validations' in report
        assert 'report_timestamp' in report
        
        assert report['total_validations'] >= 1
        assert isinstance(report['passed_validations'], int)
        assert isinstance(report['failed_validations'], int)


class TestHashUtilityFunctions:
    """Test suite for utility functions"""
    
    @given(context_data_strategy())
    @settings(max_examples=30)
    def test_hash_object_function(self, obj):
        """Test hash_object utility function"""
        # Default policy
        hash1 = hash_object(obj)
        hash2 = hash_object(obj)
        
        assert hash1 == hash2
        assert isinstance(hash1, str)
        
        # Custom policy
        policy = FastHashPolicy()
        hash3 = hash_object(obj, policy)
        hash4 = hash_object(obj, policy)
        
        assert hash3 == hash4
        assert isinstance(hash3, str)
    
    def test_hash_special_types(self):
        """Test hashing of special Python types"""
        policy = CanonicalHashPolicy()
        
        # Test datetime
        dt = datetime.now(timezone.utc)
        hash1 = policy.hash_object(dt)
        hash2 = policy.hash_object(dt)
        assert hash1 == hash2
        
        # Test UUID
        uid = uuid.uuid4()
        hash3 = policy.hash_object(uid)
        hash4 = policy.hash_object(uid)
        assert hash3 == hash4
        
        # Test bytes
        data = b'test data'
        hash5 = policy.hash_object(data)
        hash6 = policy.hash_object(data)
        assert hash5 == hash6


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_max_depth_exceeded(self):
        """Test maximum depth protection"""
        policy = CanonicalHashPolicy(max_depth=2)
        
        # Create deeply nested structure
        nested = {'level1': {'level2': {'level3': {'level4': 'deep'}}}}
        
        # Should raise ValueError due to depth limit
        with pytest.raises(ValueError, match="Maximum serialization depth"):
            policy.serialize(nested)
    
    def test_empty_containers(self):
        """Test hashing of empty containers"""
        policy = CanonicalHashPolicy()
        
        empty_dict = {}
        empty_list = []
        empty_set = set()
        
        # All should hash successfully
        hash1 = policy.hash_object(empty_dict)
        hash2 = policy.hash_object(empty_list) 
        hash3 = policy.hash_object(empty_set)
        
        assert isinstance(hash1, str)
        assert isinstance(hash2, str)
        assert isinstance(hash3, str)
        
        # Empty list and set should hash the same (both become empty arrays)
        assert hash2 == hash3
        
        # Empty dict should be different
        assert hash1 != hash2
    
    def test_none_and_boolean_handling(self):
        """Test proper handling of None and boolean values"""
        policy = CanonicalHashPolicy()
        
        values = [None, True, False]
        hashes = [policy.hash_object(v) for v in values]
        
        # All should be different
        assert len(set(hashes)) == len(values)
        
        # Should be consistent
        for value in values:
            hash1 = policy.hash_object(value)
            hash2 = policy.hash_object(value)
            assert hash1 == hash2


# Property-based integration tests
class TestPropertyBasedIntegration:
    """Property-based tests for full integration scenarios"""
    
    @given(
        st.lists(
            mock_context_strategy(),
            min_size=2,
            max_size=5
        )
    )
    @settings(max_examples=20)
    def test_pipeline_consistency_across_contexts(self, contexts):
        """Test that pipeline validation works across multiple contexts"""
        validator = PipelineHashValidator()
        
        # Process each context through the pipeline
        for i, context in enumerate(contexts):
            stage_name = f"stage_{i}"
            result = validator.validate_synthesis_consistency(
                stage_name,
                context,
                context  # Use context as both synthesis result and context
            )
            assert isinstance(result, bool)
        
        # Generate final report
        report = validator.get_validation_report()
        assert report['total_validations'] == len(contexts) * 2  # Each context generates 2 validations
        
    @given(
        mock_context_strategy(),
        st.lists(st.text(), min_size=1, max_size=10)  # Random stage names
    )
    @settings(max_examples=15)
    def test_hash_stability_across_stages(self, context, stage_names):
        """Test that context hashes remain stable across pipeline stages"""
        hasher = ContextHasher()
        
        # Get initial hash
        initial_hash = hasher.hash_context_content(context)
        
        # Process through multiple stages (simulated)
        stage_hashes = []
        for stage in stage_names:
            # Hash should remain the same if context doesn't change
            current_hash = hasher.hash_context_content(context)
            stage_hashes.append(current_hash)
        
        # All hashes should be identical
        assert all(h == initial_hash for h in stage_hashes)
    
    @given(
        context_data_strategy(),
        st.integers(min_value=1, max_value=5)  # Number of permutations
    )
    @settings(max_examples=20)
    def test_permutation_invariance(self, data, num_permutations):
        """Test that hash is invariant under data permutations"""
        assume(isinstance(data, dict) and len(data) >= 2)
        
        policy = CanonicalHashPolicy(sort_keys=True)
        original_hash = policy.hash_object(data)
        
        # Create permutations by rebuilding dict in different orders
        import random
        keys = list(data.keys())
        
        for _ in range(num_permutations):
            random.shuffle(keys)
            permuted_data = {k: data[k] for k in keys}
            permuted_hash = policy.hash_object(permuted_data)
            assert permuted_hash == original_hash


if __name__ == "__main__":
    pytest.main([__file__])