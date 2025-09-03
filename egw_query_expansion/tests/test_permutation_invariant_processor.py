"""
Test suite for Permutation Invariant Processor.

Tests the implementation of generalized Deep Sets theory with invariance verification,
numerical consistency, and multiset semantics preservation.
"""

# # # from collections import Counter  # Module not found  # Module not found  # Module not found
# # # from typing import List, Tuple  # Module not found  # Module not found  # Module not found

import numpy as np
import pytest
import torch
import torch.nn as nn

# # # from egw_query_expansion.core.permutation_invariant_processor import (  # Module not found  # Module not found  # Module not found
    AggregationType,
    DeepSetBlock,
    DeterministicPooler,
    EquivariantAttention,
    MultisetSerializer,
    MultisetStats,
    PermutationInvariantProcessor,
)


class TestDeterministicPooler:
    """Test deterministic pooling operations."""

    def test_sum_pooling_consistency(self):
        """Test sum pooling produces consistent results."""
        x = torch.randn(4, 10, 64)

        # Multiple computations should be identical
        result1 = DeterministicPooler.sum_pool(x, dim=1)
        result2 = DeterministicPooler.sum_pool(x, dim=1)

        assert torch.allclose(result1, result2, atol=1e-8)
        assert result1.shape == (4, 64)

    def test_mean_pooling_consistency(self):
        """Test mean pooling produces consistent results."""
        x = torch.randn(4, 10, 64)

        result1 = DeterministicPooler.mean_pool(x, dim=1)
        result2 = DeterministicPooler.mean_pool(x, dim=1)

        assert torch.allclose(result1, result2, atol=1e-8)
        assert result1.shape == (4, 64)

    def test_log_sum_exp_stability(self):
        """Test log-sum-exp pooling numerical stability."""
        # Test with large values that could cause overflow
        x = torch.tensor([[[100.0, 200.0, 300.0]]]).float()

        result = DeterministicPooler.log_sum_exp_pool(x, dim=1)

        # Should not be inf or nan
        assert torch.isfinite(result).all()
        assert result.shape == (1, 3)

    def test_pooler_retrieval(self):
        """Test pooler function retrieval by type."""
        sum_pooler = DeterministicPooler.get_pooler(AggregationType.SUM)
        mean_pooler = DeterministicPooler.get_pooler(AggregationType.MEAN)

        x = torch.randn(2, 5, 3)
        sum_result = sum_pooler(x, dim=1)
        mean_result = mean_pooler(x, dim=1)

        assert sum_result.shape == (2, 3)
        assert mean_result.shape == (2, 3)
        assert not torch.allclose(sum_result, mean_result)


class TestMultisetSerializer:
    """Test multiset serialization and deserialization."""

    def test_serialize_simple_multiset(self):
        """Test serialization of simple multiset."""
        elements = ["a", "b", "a", "c"]
        result = MultisetSerializer.serialize_multiset(elements)

        assert isinstance(result, str)
        assert "multiset" in result
        assert "('a', 2)" in result
        assert "('b', 1)" in result
        assert "('c', 1)" in result

    def test_serialize_with_explicit_multiplicities(self):
        """Test serialization with explicit multiplicities."""
        elements = ["x", "y", "z"]
        multiplicities = [3, 1, 2]

        result = MultisetSerializer.serialize_multiset(elements, multiplicities)

        assert "('x', 3)" in result
        assert "('y', 1)" in result
        assert "('z', 2)" in result

    def test_canonical_ordering(self):
        """Test that serialization produces canonical ordering."""
        elements1 = ["c", "a", "b"]
        elements2 = ["a", "b", "c"]

        result1 = MultisetSerializer.serialize_multiset(elements1)
        result2 = MultisetSerializer.serialize_multiset(elements2)

        # Should be identical due to canonical ordering
        assert result1 == result2

    def test_deserialize_roundtrip(self):
        """Test serialization -> deserialization roundtrip."""
        elements = ["a", "b", "a", "c", "b", "a"]

        serialized = MultisetSerializer.serialize_multiset(elements)
        (
            deserialized_elements,
            deserialized_mults,
        ) = MultisetSerializer.deserialize_multiset(serialized)

        # Check counts are preserved
        original_counter = Counter(elements)
        deserialized_counter = Counter()
        for elem, mult in zip(deserialized_elements, deserialized_mults):
            deserialized_counter[elem] = mult

        assert original_counter == deserialized_counter


class TestEquivariantAttention:
    """Test equivariant attention mechanism."""

    def test_attention_equivariance(self):
        """Test that attention maintains equivariance."""
        d_model = 64
        seq_len = 10
        batch_size = 2

        attention = EquivariantAttention(d_model)
        x = torch.randn(batch_size, seq_len, d_model)

        # Process original
        output1 = attention(x)

        # Process permuted version
        perm = torch.randperm(seq_len)
        x_perm = x[:, perm, :]
        output2 = attention(x_perm)

        # Un-permute output2 for comparison
        inv_perm = torch.argsort(perm)
        output2_unperm = output2[:, inv_perm, :]

        # Should be approximately equal (attention is equivariant)
        assert torch.allclose(output1, output2_unperm, atol=1e-5)

    def test_attention_output_shape(self):
        """Test attention output shape preservation."""
        d_model = 128
        seq_len = 15
        batch_size = 3

        attention = EquivariantAttention(d_model, n_heads=8)
        x = torch.randn(batch_size, seq_len, d_model)

        output = attention(x)

        assert output.shape == x.shape


class TestDeepSetBlock:
    """Test Deep Set processing blocks."""

    def test_deep_set_invariance(self):
        """Test that Deep Set block maintains invariance."""
        input_dim = 32
        hidden_dim = 64
        output_dim = 16

        block = DeepSetBlock(input_dim, hidden_dim, output_dim, AggregationType.SUM)

        # Create test input
        batch_size, set_size = 2, 8
        x = torch.randn(batch_size, set_size, input_dim)

        # Process original
        output1 = block(x)

        # Process permuted version
        perm = torch.randperm(set_size)
        x_perm = x[:, perm, :]
        output2 = block(x_perm)

        # Should be identical (invariant)
        assert torch.allclose(output1, output2, atol=1e-6)

    def test_deep_set_with_multiplicities(self):
        """Test Deep Set block with multiplicities."""
        input_dim = 16
        hidden_dim = 32
        output_dim = 8

        block = DeepSetBlock(input_dim, hidden_dim, output_dim, AggregationType.SUM)

        batch_size, set_size = 1, 5
        x = torch.randn(batch_size, set_size, input_dim)
        multiplicities = torch.tensor([[2.0, 1.0, 3.0, 1.0, 2.0]])

        output_with_mult = block(x, multiplicities)
        output_without_mult = block(x)

        # Should be different when multiplicities are applied
        assert not torch.allclose(output_with_mult, output_without_mult, atol=1e-5)
        assert output_with_mult.shape == (batch_size, output_dim)

    def test_aggregation_types(self):
        """Test different aggregation types in Deep Set blocks."""
        input_dim, hidden_dim, output_dim = 8, 16, 4
        batch_size, set_size = 1, 3

        x = torch.randn(batch_size, set_size, input_dim)

        # Test different aggregation types
        agg_types = [AggregationType.SUM, AggregationType.MEAN, AggregationType.MAX]
        outputs = {}

        for agg_type in agg_types:
            block = DeepSetBlock(input_dim, hidden_dim, output_dim, agg_type)
            outputs[agg_type] = block(x)

        # Different aggregations should produce different results
        assert not torch.allclose(
            outputs[AggregationType.SUM], outputs[AggregationType.MEAN]
        )
        assert not torch.allclose(
            outputs[AggregationType.SUM], outputs[AggregationType.MAX]
        )


class TestPermutationInvariantProcessor:
    """Test main permutation invariant processor."""

    def test_processor_invariance(self):
        """Test that processor maintains permutation invariance."""
        input_dim = 32
        hidden_dim = 64
        output_dim = 16

        processor = PermutationInvariantProcessor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=2,
            enable_verification=False,  # We'll test manually
        )

        batch_size, set_size = 2, 10
        x = torch.randn(batch_size, set_size, input_dim)

        # Process original
        output1 = processor(x, verify_invariance=False)

        # Process multiple permutations
        for _ in range(5):
            perm = torch.randperm(set_size)
            x_perm = x[:, perm, :]
            output_perm = processor(x_perm, verify_invariance=False)

            # Should be invariant
            assert torch.allclose(
                output1, output_perm, atol=1e-5
            ), "Invariance violation detected"

        assert output1.shape == (batch_size, output_dim)

    def test_multiset_processing(self):
        """Test processing of explicit multisets."""
        input_dim = 16
        processor = PermutationInvariantProcessor(
            input_dim=input_dim, hidden_dim=32, output_dim=8, enable_verification=False
        )

        # Create multiset
        elements = ["item1", "item2", "item1", "item3"]
        embeddings = torch.randn(4, input_dim)
        multiplicities = [2, 1, 2, 1]  # Corresponds to elements

        output, stats = processor.process_multiset(elements, embeddings, multiplicities)

        assert output.shape == (8,)  # output_dim
        assert stats.cardinality == sum(multiplicities)
        assert stats.aggregation_type == AggregationType.SUM
        assert isinstance(stats.platform_hash, str)

    def test_invariance_verification(self):
        """Test built-in invariance verification."""
        processor = PermutationInvariantProcessor(
            input_dim=16, hidden_dim=32, output_dim=8, enable_verification=True
        )

        x = torch.randn(1, 5, 16)

        # Should pass verification
        output = processor(x)

        verification_summary = processor.get_verification_summary()
        assert verification_summary["total_verifications"] > 0
        assert verification_summary["pass_rate"] == 1.0
        assert verification_summary["status"] == "all_passed"

    def test_statistics_tracking(self):
        """Test statistics tracking and reset."""
        processor = PermutationInvariantProcessor(
            input_dim=8, hidden_dim=16, output_dim=4
        )

        x1 = torch.randn(1, 3, 8)
        x2 = torch.randn(1, 5, 8)

        processor(x1, batch_id="batch1", verify_invariance=False)
        processor(x2, batch_id="batch2", verify_invariance=False)

        assert len(processor.processing_stats) == 2
        assert "batch1" in processor.processing_stats
        assert "batch2" in processor.processing_stats

        stats1 = processor.processing_stats["batch1"]
        stats2 = processor.processing_stats["batch2"]

        assert stats1.cardinality == 3
        assert stats2.cardinality == 5

        # Test reset
        processor.reset_statistics()
        assert len(processor.processing_stats) == 0
        assert len(processor.verification_results) == 0

    def test_different_aggregation_types(self):
        """Test processor with different aggregation types."""
        input_dim = 12
        x = torch.randn(1, 4, input_dim)

        aggregation_types = [
            AggregationType.SUM,
            AggregationType.MEAN,
            AggregationType.LOG_SUM_EXP,
        ]

        outputs = {}
        for agg_type in aggregation_types:
            processor = PermutationInvariantProcessor(
                input_dim=input_dim,
                hidden_dim=24,
                output_dim=6,
                aggregation=agg_type,
                enable_verification=False,
            )
            outputs[agg_type] = processor(x, verify_invariance=False)

        # Different aggregations should produce different results
        assert not torch.allclose(
            outputs[AggregationType.SUM], outputs[AggregationType.MEAN]
        )
        assert not torch.allclose(
            outputs[AggregationType.SUM], outputs[AggregationType.LOG_SUM_EXP]
        )

    def test_numerical_consistency(self):
        """Test numerical consistency across multiple runs."""
        torch.manual_seed(42)  # Set seed for reproducibility

        processor = PermutationInvariantProcessor(
            input_dim=16, hidden_dim=32, output_dim=8, enable_verification=False
        )

        x = torch.randn(2, 6, 16)

        # Multiple runs should be identical
        output1 = processor(x, verify_invariance=False)
        output2 = processor(x, verify_invariance=False)

        assert torch.allclose(output1, output2, atol=1e-8)

    def test_multiplicities_invariance(self):
        """Test that multiplicities maintain invariance."""
        processor = PermutationInvariantProcessor(
            input_dim=8, hidden_dim=16, output_dim=4, enable_verification=False
        )

        batch_size, set_size = 1, 4
        x = torch.randn(batch_size, set_size, 8)
        multiplicities = torch.tensor([[2.0, 1.0, 3.0, 1.0]])

        # Process original
        output1 = processor(x, multiplicities, verify_invariance=False)

        # Process permuted
        perm = torch.randperm(set_size)
        x_perm = x[:, perm, :]
        mult_perm = multiplicities[:, perm]
        output2 = processor(x_perm, mult_perm, verify_invariance=False)

        # Should be invariant even with multiplicities
        assert torch.allclose(output1, output2, atol=1e-6)


class TestIntegration:
    """Integration tests for the complete system."""

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Setup
        processor = PermutationInvariantProcessor(
            input_dim=32,
            hidden_dim=64,
            output_dim=16,
            n_layers=3,
            n_attention_layers=2,
            aggregation=AggregationType.MEAN,
            enable_verification=True,
        )

        # Create realistic multiset data
        elements = ["doc1", "doc2", "doc1", "doc3", "doc2", "doc1"]
        embeddings = torch.randn(6, 32)

        # Process as explicit multiset
        output, stats = processor.process_multiset(elements, embeddings)

        # Verify results
        assert output.shape == (16,)
        assert stats.cardinality == 6
        assert stats.unique_elements == 6
        assert stats.aggregation_type == AggregationType.MEAN

        # Check verification passed
        summary = processor.get_verification_summary()
        assert summary["pass_rate"] == 1.0

        # Test serialization consistency
        serialized1 = MultisetSerializer.serialize_multiset(elements)
        serialized2 = MultisetSerializer.serialize_multiset(
            ["doc2", "doc1", "doc3", "doc1", "doc1", "doc2"]
        )
        assert serialized1 == serialized2  # Same multiset, different order

    def test_stress_test_large_sets(self):
        """Stress test with larger sets."""
        processor = PermutationInvariantProcessor(
            input_dim=64,
            hidden_dim=128,
            output_dim=32,
            n_layers=2,
            enable_verification=False,  # Skip for performance
        )

        # Large set
        batch_size, set_size = 4, 100
        x = torch.randn(batch_size, set_size, 64)

        output = processor(x, verify_invariance=False)

        assert output.shape == (batch_size, 32)
        assert torch.isfinite(output).all()

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        processor = PermutationInvariantProcessor(
            input_dim=4, hidden_dim=8, output_dim=2, enable_verification=False
        )

        # Single element set
        x_single = torch.randn(1, 1, 4)
        output_single = processor(x_single, verify_invariance=False)
        assert output_single.shape == (1, 2)

        # Empty-like set (all zeros)
        x_zeros = torch.zeros(1, 3, 4)
        output_zeros = processor(x_zeros, verify_invariance=False)
        assert output_zeros.shape == (1, 2)
        assert torch.isfinite(output_zeros).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
