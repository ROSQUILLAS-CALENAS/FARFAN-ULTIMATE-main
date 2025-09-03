"""
Property-based tests for deterministic hashing module using Hypothesis.

Tests that hash_context() and hash_synthesis() functions produce identical
outputs when called multiple times with equivalent data structures, and
that combining operations remain associative regardless of input order.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.strategies import composite
from typing import Any, Dict, List, Union

from ..core.deterministic_hashing import (
    hash_context,
    hash_synthesis, 
    hash_data_structure,
    combine_hashes,
    verify_hash_consistency,
    _canonicalize_data,
    _serialize_canonical
)


# Custom strategies for generating test data
@composite
def nested_dict_strategy(draw, max_depth=3, current_depth=0):
    """Generate nested dictionaries with controlled depth."""
    if current_depth >= max_depth:
        # At max depth, use simple values
        return draw(st.dictionaries(
            st.text(min_size=1, max_size=10),
            st.one_of(st.integers(), st.floats(allow_nan=False), st.text(), st.booleans())
        ))
    
    return draw(st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.one_of(
            st.integers(),
            st.floats(allow_nan=False),
            st.text(),
            st.booleans(),
            st.lists(st.integers(), max_size=5),
            st.sets(st.integers(), max_size=5),
            nested_dict_strategy(max_depth=max_depth, current_depth=current_depth + 1)
        ),
        min_size=1,
        max_size=5
    ))


@composite
def context_strategy(draw):
    """Generate context-like dictionaries."""
    base_dict = draw(nested_dict_strategy(max_depth=2))
    
    # Add typical context fields
    context_fields = draw(st.dictionaries(
        st.sampled_from(['question', 'metadata', 'expansion', 'derivation_id']),
        st.one_of(st.text(), st.dictionaries(st.text(), st.integers())),
        min_size=1,
        max_size=3
    ))
    
    base_dict.update(context_fields)
    return base_dict


@composite
def synthesis_strategy(draw):
    """Generate synthesis-like dictionaries."""
    base_dict = draw(nested_dict_strategy(max_depth=2))
    
    # Add typical synthesis fields
    synthesis_fields = draw(st.dictionaries(
        st.sampled_from(['answer', 'evidence', 'confidence', 'citations', 'verdict']),
        st.one_of(
            st.text(),
            st.lists(st.dictionaries(st.text(), st.text())),
            st.floats(min_value=0, max_value=1, allow_nan=False)
        ),
        min_size=1,
        max_size=4
    ))
    
    base_dict.update(synthesis_fields)
    return base_dict


class TestDeterministicHashing:
    """Test suite for deterministic hashing functions."""

    @given(context_strategy())
    @settings(max_examples=100, deadline=None)
    def test_hash_context_deterministic(self, context_data):
        """Test that hash_context produces identical outputs for same input."""
        hash1 = hash_context(context_data)
        hash2 = hash_context(context_data)
        hash3 = hash_context(context_data)
        
        assert hash1 == hash2 == hash3
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA-256 produces 64-character hex strings

    @given(synthesis_strategy())
    @settings(max_examples=100, deadline=None)
    def test_hash_synthesis_deterministic(self, synthesis_data):
        """Test that hash_synthesis produces identical outputs for same input."""
        hash1 = hash_synthesis(synthesis_data)
        hash2 = hash_synthesis(synthesis_data)
        hash3 = hash_synthesis(synthesis_data)
        
        assert hash1 == hash2 == hash3
        assert isinstance(hash1, str)
        assert len(hash1) == 64

    @given(context_strategy(), context_strategy())
    @settings(max_examples=50, deadline=None)
    def test_different_contexts_different_hashes(self, context1, context2):
        """Test that different contexts produce different hashes."""
        assume(context1 != context2)
        
        hash1 = hash_context(context1)
        hash2 = hash_context(context2)
        
        assert hash1 != hash2

    @given(synthesis_strategy(), synthesis_strategy())
    @settings(max_examples=50, deadline=None)
    def test_different_synthesis_different_hashes(self, synthesis1, synthesis2):
        """Test that different synthesis data produces different hashes."""
        assume(synthesis1 != synthesis2)
        
        hash1 = hash_synthesis(synthesis1)
        hash2 = hash_synthesis(synthesis2)
        
        assert hash1 != hash2

    @given(st.dictionaries(st.text(), st.integers()))
    @settings(max_examples=50, deadline=None)
    def test_key_order_independence(self, data):
        """Test that dictionary key order doesn't affect hash."""
        assume(len(data) > 1)
        
        # Create equivalent dict with different insertion order
        keys = list(data.keys())
        reversed_dict = {k: data[k] for k in reversed(keys)}
        
        hash1 = hash_data_structure(data)
        hash2 = hash_data_structure(reversed_dict)
        
        assert hash1 == hash2

    @given(st.lists(st.text(), min_size=2, max_size=10, unique=True))
    @settings(max_examples=50, deadline=None)
    def test_combine_hashes_associative(self, hash_strings):
        """Test that hash combination is associative."""
        assume(len(hash_strings) >= 3)
        
        h1, h2, h3 = hash_strings[:3]
        
        # Test associativity: (h1 + h2) + h3 == h1 + (h2 + h3)
        left_assoc = combine_hashes(combine_hashes(h1, h2), h3)
        right_assoc = combine_hashes(h1, combine_hashes(h2, h3))
        
        assert left_assoc == right_assoc

    @given(st.lists(st.text(), min_size=2, max_size=10, unique=True))
    @settings(max_examples=50, deadline=None)
    def test_combine_hashes_order_independence(self, hash_strings):
        """Test that combine_hashes is order-independent."""
        assume(len(hash_strings) >= 2)
        
        # Combine in original order
        combined1 = combine_hashes(*hash_strings)
        
        # Combine in reversed order
        combined2 = combine_hashes(*reversed(hash_strings))
        
        assert combined1 == combined2

    @given(nested_dict_strategy())
    @settings(max_examples=50, deadline=None)
    def test_nested_structure_handling(self, nested_data):
        """Test that nested structures are handled consistently."""
        hash1 = hash_data_structure(nested_data)
        hash2 = hash_data_structure(nested_data)
        
        assert hash1 == hash2
        
        # Test that canonicalization is stable
        canonical1 = _canonicalize_data(nested_data)
        canonical2 = _canonicalize_data(nested_data)
        
        assert canonical1 == canonical2

    @given(st.dictionaries(
        st.text(),
        st.one_of(
            st.lists(st.integers()),
            st.sets(st.integers()),
            st.tuples(st.integers(), st.integers())
        )
    ))
    @settings(max_examples=30, deadline=None)
    def test_collection_type_normalization(self, data):
        """Test that different collection types are normalized consistently."""
        # Test multiple times to ensure determinism
        hashes = [hash_data_structure(data) for _ in range(3)]
        
        assert all(h == hashes[0] for h in hashes)

    @given(context_strategy())
    @settings(max_examples=30, deadline=None)
    def test_hash_verification(self, context_data):
        """Test hash verification functionality."""
        expected_hash = hash_context(context_data)
        
        # Verification should succeed
        assert verify_hash_consistency(context_data, expected_hash)
        
        # Verification should fail with wrong hash
        wrong_hash = "a" * 64
        assert not verify_hash_consistency(context_data, wrong_hash)

    def test_empty_combine_hashes(self):
        """Test combining empty hash list."""
        result = combine_hashes()
        assert isinstance(result, str)
        assert len(result) == 64

    def test_single_hash_combine(self):
        """Test combining single hash."""
        single_hash = "abc123def456"
        result = combine_hashes(single_hash)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_invalid_input_types(self):
        """Test error handling for invalid input types."""
        with pytest.raises(TypeError):
            hash_context("not a dict")
            
        with pytest.raises(TypeError):
            hash_synthesis(["not", "a", "dict"])

    @given(st.dictionaries(
        st.text(),
        st.recursive(
            st.one_of(st.integers(), st.text(), st.booleans()),
            lambda children: st.dictionaries(st.text(), children) | st.lists(children),
            max_leaves=10
        )
    ))
    @settings(max_examples=30, deadline=None)
    def test_deep_recursive_structures(self, recursive_data):
        """Test handling of deeply recursive data structures."""
        # Should not raise exceptions
        hash_result = hash_data_structure(recursive_data)
        
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64
        
        # Should be deterministic
        hash_result2 = hash_data_structure(recursive_data)
        assert hash_result == hash_result2

    @given(st.dictionaries(st.text(), st.text(), min_size=1))
    @settings(max_examples=20, deadline=None)
    def test_serialization_stability(self, data):
        """Test that serialization is stable across multiple calls."""
        serialized1 = _serialize_canonical(data)
        serialized2 = _serialize_canonical(data)
        serialized3 = _serialize_canonical(data)
        
        assert serialized1 == serialized2 == serialized3
        
        # Should be valid JSON
        import json
        parsed = json.loads(serialized1)
        assert isinstance(parsed, dict)

    def test_edge_cases(self):
        """Test specific edge cases."""
        # Empty dictionaries
        empty_context_hash1 = hash_context({})
        empty_context_hash2 = hash_context({})
        assert empty_context_hash1 == empty_context_hash2
        
        # None values
        none_data = {"key1": None, "key2": "value"}
        hash1 = hash_data_structure(none_data)
        hash2 = hash_data_structure(none_data)
        assert hash1 == hash2
        
        # Mixed nested types
        complex_data = {
            "list": [1, 2, {"nested": "value"}],
            "tuple": (1, 2, 3),
            "set": {1, 2, 3},
            "nested_dict": {
                "inner": {"deep": ["very", "deep", {"structure": True}]}
            }
        }
        hash1 = hash_data_structure(complex_data)
        hash2 = hash_data_structure(complex_data)
        assert hash1 == hash2