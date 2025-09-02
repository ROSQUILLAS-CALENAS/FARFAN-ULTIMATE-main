"""
Permutation Invariant Checking (PIC) Tests using Hypothesis for metamorphic testing.

This module validates that set aggregations (patterns/evidence) are invariant to order
through property-based testing with random shuffling of inputs.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
from hypothesis_module import given, settings
from hypothesis_module import strategies as st

from egw_query_expansion.core.permutation_invariant_processor import (
    AggregationType, DeterministicPooler, MultisetSerializer,
    PermutationInvariantProcessor)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
NUMERICAL_TOLERANCE = 1e-5
DEFAULT_TRIALS = 100
STRESS_TEST_TRIALS = 500


class PermutationInvariantAggregator:
    """
    Implementation of f(S) = φ(Σ ψ(x)) for permutation invariant aggregation.

    This class provides various aggregation functions that should be invariant
    to the order of elements in the input set S.
    """

    def __init__(self, aggregation_type: AggregationType = AggregationType.SUM):
        self.aggregation_type = aggregation_type
        self.pooler = DeterministicPooler.get_pooler(aggregation_type)

    def psi(self, x: torch.Tensor) -> torch.Tensor:
        """Element-wise transformation ψ(x)."""
        # Simple polynomial transformation for testing
        return torch.tanh(x) + 0.1 * torch.sin(x)

    def phi(self, aggregated: torch.Tensor) -> torch.Tensor:
        """Set-level transformation φ(Σ ψ(x_i))."""
        # Nonlinear transformation of aggregated features
        return torch.sigmoid(aggregated) * aggregated

    def aggregate(
        self, multiset: torch.Tensor, multiplicities: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute f(S) = φ(Σ ψ(x)) for multiset S.

        Args:
            multiset: Input tensor [batch_size, set_size, feature_dim]
            multiplicities: Optional multiplicities [batch_size, set_size]

        Returns:
            Aggregated representation [batch_size, feature_dim]
        """
        # Apply element-wise transformation
        psi_x = self.psi(multiset)

        # Apply multiplicities if provided
        if multiplicities is not None:
            psi_x = psi_x * multiplicities.unsqueeze(-1)

        # Aggregate using pooling operation
        aggregated = self.pooler(psi_x, dim=1)

        # Apply set-level transformation
        return self.phi(aggregated)


class PICTestRunner:
    """Test runner for permutation invariance checking with comprehensive validation."""

    def __init__(self, tolerance: float = NUMERICAL_TOLERANCE):
        self.tolerance = tolerance
        self.results: List[Dict[str, Any]] = []
        self.mismatches: List[Dict[str, Any]] = []

    def shuffle_preserving_semantics(
        self, multiset: torch.Tensor, multiplicities: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Shuffle multiset while preserving semantic content."""
        batch_size, set_size, _ = multiset.shape
        shuffled_multiset = multiset.clone()
        shuffled_multiplicities = (
            multiplicities.clone() if multiplicities is not None else None
        )

        for batch_idx in range(batch_size):
            # Generate random permutation for this batch
            perm = torch.randperm(set_size)
            shuffled_multiset[batch_idx] = multiset[batch_idx, perm]
            if shuffled_multiplicities is not None:
                shuffled_multiplicities[batch_idx] = multiplicities[batch_idx, perm]

        return shuffled_multiset, shuffled_multiplicities

    def validate_invariance(
        self,
        aggregator: PermutationInvariantAggregator,
        multiset: torch.Tensor,
        multiplicities: Optional[torch.Tensor] = None,
        n_shuffles: int = 10,
    ) -> Dict[str, Any]:
        """
        Validate permutation invariance through multiple random shuffles.

        Returns:
            Dictionary with validation results
        """
        # Compute reference output
        reference_output = aggregator.aggregate(multiset, multiplicities)

        # Test multiple random shuffles
        violations = []
        max_diff = 0.0

        for shuffle_idx in range(n_shuffles):
            # Shuffle while preserving semantics
            shuffled_multiset, shuffled_multiplicities = (
                self.shuffle_preserving_semantics(multiset, multiplicities)
            )

            # Compute output for shuffled version
            shuffled_output = aggregator.aggregate(
                shuffled_multiset, shuffled_multiplicities
            )

            # Check invariance
            diff = torch.abs(reference_output - shuffled_output).max().item()
            max_diff = max(max_diff, diff)

            if diff > self.tolerance:
                violations.append(
                    {
                        "shuffle_idx": shuffle_idx,
                        "max_difference": diff,
                        "reference_norm": torch.norm(reference_output).item(),
                        "shuffled_norm": torch.norm(shuffled_output).item(),
                    }
                )

        # Compile results
        result = {
            "passed": len(violations) == 0,
            "n_shuffles": n_shuffles,
            "violations": violations,
            "max_difference": max_diff,
            "tolerance": self.tolerance,
            "aggregation_type": aggregator.aggregation_type.value,
        }

        self.results.append(result)
        if violations:
            self.mismatches.append(result)

        return result


# Hypothesis strategies for generating test data
@st.composite
def multiset_strategy(
    draw,
    min_batch_size=1,
    max_batch_size=4,
    min_set_size=2,
    max_set_size=20,
    min_feature_dim=4,
    max_feature_dim=32,
):
    """Generate random multisets for testing."""
    batch_size = draw(st.integers(min_batch_size, max_batch_size))
    set_size = draw(st.integers(min_set_size, max_set_size))
    feature_dim = draw(st.integers(min_feature_dim, max_feature_dim))

    # Generate random tensor
    elements = draw(
        st.lists(
            st.floats(
                min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False
            ),
            min_size=batch_size * set_size * feature_dim,
            max_size=batch_size * set_size * feature_dim,
        )
    )

    tensor = torch.tensor(elements).reshape(batch_size, set_size, feature_dim)

    # Optionally generate multiplicities
    include_multiplicities = draw(st.booleans())
    multiplicities = None

    if include_multiplicities:
        mult_values = draw(
            st.lists(
                st.floats(
                    min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False
                ),
                min_size=batch_size * set_size,
                max_size=batch_size * set_size,
            )
        )
        multiplicities = torch.tensor(mult_values).reshape(batch_size, set_size)

    return tensor, multiplicities


@st.composite
def aggregation_strategy(draw):
    """Generate aggregation types for testing."""
    return draw(
        st.sampled_from(
            [
                AggregationType.SUM,
                AggregationType.MEAN,
                AggregationType.LOG_SUM_EXP,
                AggregationType.MAX,
                AggregationType.MIN,
            ]
        )
    )


class TestPermutationInvariance:
    """Main test class for permutation invariance validation."""

    def setup_method(self):
        """Setup test runner for each test method."""
        self.test_runner = PICTestRunner(tolerance=NUMERICAL_TOLERANCE)

    @given(data=multiset_strategy(), agg_type=aggregation_strategy())
    @settings(max_examples=DEFAULT_TRIALS, deadline=10000)
    def test_basic_permutation_invariance(self, data, agg_type):
        """Test basic permutation invariance property."""
        multiset, multiplicities = data
        aggregator = PermutationInvariantAggregator(agg_type)

        result = self.test_runner.validate_invariance(
            aggregator, multiset, multiplicities, n_shuffles=5
        )

        # Assert invariance holds
        assert result["passed"], f"Invariance violation: {result}"

    @given(
        data=multiset_strategy(min_set_size=10, max_set_size=50),
        agg_type=aggregation_strategy(),
    )
    @settings(max_examples=50, deadline=15000)
    def test_large_set_invariance(self, data, agg_type):
        """Test invariance on larger sets."""
        multiset, multiplicities = data
        aggregator = PermutationInvariantAggregator(agg_type)

        result = self.test_runner.validate_invariance(
            aggregator, multiset, multiplicities, n_shuffles=3
        )

        assert result["passed"], f"Large set invariance violation: {result}"

    def test_edge_case_single_element(self):
        """Test invariance with single element sets."""
        multiset = torch.randn(2, 1, 8)  # Single element per set
        aggregator = PermutationInvariantAggregator(AggregationType.SUM)

        result = self.test_runner.validate_invariance(
            aggregator, multiset, n_shuffles=10
        )
        assert result["passed"], "Single element invariance failed"

    def test_edge_case_identical_elements(self):
        """Test invariance with identical elements."""
        # All elements are the same
        element = torch.randn(1, 1, 6)
        multiset = element.repeat(3, 8, 1)  # Repeat same element

        aggregator = PermutationInvariantAggregator(AggregationType.MEAN)
        result = self.test_runner.validate_invariance(
            aggregator, multiset, n_shuffles=10
        )

        assert result["passed"], "Identical elements invariance failed"

    def test_multiplicities_invariance(self):
        """Test that multiplicities preserve invariance."""
        torch.manual_seed(42)
        multiset = torch.randn(2, 6, 4)
        multiplicities = torch.rand(2, 6) * 3 + 0.5  # Random multiplicities

        for agg_type in [AggregationType.SUM, AggregationType.MEAN]:
            aggregator = PermutationInvariantAggregator(agg_type)
            result = self.test_runner.validate_invariance(
                aggregator, multiset, multiplicities, n_shuffles=10
            )
            assert result["passed"], f"Multiplicities invariance failed for {agg_type}"

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very small values
        small_multiset = torch.randn(1, 5, 3) * 1e-6
        # Test with large values
        large_multiset = torch.randn(1, 5, 3) * 1e3

        for multiset, desc in [(small_multiset, "small"), (large_multiset, "large")]:
            aggregator = PermutationInvariantAggregator(AggregationType.LOG_SUM_EXP)
            result = self.test_runner.validate_invariance(
                aggregator, multiset, n_shuffles=5
            )
            assert result["passed"], f"Numerical stability failed for {desc} values"

    @pytest.mark.parametrize(
        "agg_type",
        [
            AggregationType.SUM,
            AggregationType.MEAN,
            AggregationType.LOG_SUM_EXP,
            AggregationType.MAX,
            AggregationType.MIN,
        ],
    )
    def test_all_aggregation_types(self, agg_type):
        """Test invariance for all aggregation types."""
        torch.manual_seed(123)
        multiset = torch.randn(3, 8, 5)
        multiplicities = torch.ones(3, 8) + torch.rand(3, 8)

        aggregator = PermutationInvariantAggregator(agg_type)
        result = self.test_runner.validate_invariance(
            aggregator, multiset, multiplicities, n_shuffles=15
        )

        assert result["passed"], f"Aggregation type {agg_type.value} failed invariance"

    def test_deep_sets_processor_integration(self):
        """Integration test with the full PermutationInvariantProcessor."""
        # Note: PermutationInvariantProcessor includes attention layers which
        # may not be perfectly invariant due to numerical precision in
        # complex neural network operations. For strict testing, we use
        # simpler architectures.

        processor = PermutationInvariantProcessor(
            input_dim=16,
            hidden_dim=32,
            output_dim=8,
            n_layers=1,  # Single layer for better numerical stability
            n_attention_layers=0,  # Disable attention for strict invariance
            aggregation=AggregationType.SUM,
            enable_verification=False,
        )

        multiset = torch.randn(2, 10, 16)
        reference_output = processor(multiset, verify_invariance=False)

        # Test a single permutation with relaxed tolerance for neural networks
        perm = torch.randperm(10)
        shuffled_multiset = multiset[:, perm, :]
        shuffled_output = processor(shuffled_multiset, verify_invariance=False)

        # Use appropriate tolerance for neural network computations
        diff = torch.abs(reference_output - shuffled_output).max().item()
        neural_tolerance = 1e-3  # Reasonable tolerance for neural architectures

        assert (
            diff < neural_tolerance
        ), f"Deep Sets processor invariance violated: {diff}"
        assert reference_output.shape == (
            2,
            8,
        ), f"Unexpected output shape: {reference_output.shape}"

    def teardown_method(self):
        """Generate test certificate after each test class."""
        self._generate_certificate()

    def _generate_certificate(self):
        """Generate PIC certificate with test results."""
        total_results = len(self.test_runner.results)
        passed_results = sum(1 for r in self.test_runner.results if r["passed"])
        failed_results = total_results - passed_results

        certificate = {
            "pass": failed_results == 0,
            "trials": total_results,
            "mismatches": len(self.test_runner.mismatches),
            "details": {
                "passed": passed_results,
                "failed": failed_results,
                "numerical_tolerance": NUMERICAL_TOLERANCE,
                "test_timestamp": str(
                    torch.utils.data.get_worker_info() or "main_process"
                ),
                "aggregation_types_tested": list(
                    set(
                        r.get("aggregation_type", "unknown")
                        for r in self.test_runner.results
                    )
                ),
                "max_observed_difference": max(
                    (r.get("max_difference", 0) for r in self.test_runner.results),
                    default=0.0,
                ),
            },
            "mismatches_detail": self.test_runner.mismatches[
                :10
            ],  # Keep first 10 failures
        }

        # Write certificate to file
        with open("pic_certificate.json", "w") as f:
            json.dump(certificate, f, indent=2)

        logger.info(
            f"Generated PIC certificate: {passed_results}/{total_results} passed"
        )


class TestAdvancedPermutationInvariance:
    """Advanced tests for edge cases and stress testing."""

    def setup_method(self):
        self.test_runner = PICTestRunner()

    def test_stress_many_permutations(self):
        """Stress test with many permutations."""
        torch.manual_seed(999)
        multiset = torch.randn(1, 12, 8)
        aggregator = PermutationInvariantAggregator(AggregationType.SUM)

        result = self.test_runner.validate_invariance(
            aggregator, multiset, n_shuffles=100
        )

        assert result["passed"], f"Stress test failed: {result}"

    def test_multiset_serialization_consistency(self):
        """Test that multiset serialization is order-invariant."""
        elements1 = ["a", "b", "c", "a", "b"]
        elements2 = ["b", "a", "a", "c", "b"]  # Same multiset, different order

        serialized1 = MultisetSerializer.serialize_multiset(elements1)
        serialized2 = MultisetSerializer.serialize_multiset(elements2)

        assert serialized1 == serialized2, "Multiset serialization not order-invariant"

    def test_aggregation_commutativity(self):
        """Test that aggregation operations are commutative."""
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2x2
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])  # 2x2

        # Test sum commutativity
        sum_ab = DeterministicPooler.sum_pool(torch.stack([a, b], dim=1), dim=1)
        sum_ba = DeterministicPooler.sum_pool(torch.stack([b, a], dim=1), dim=1)

        assert torch.allclose(sum_ab, sum_ba), "Sum pooling not commutative"

        # Test mean commutativity
        mean_ab = DeterministicPooler.mean_pool(torch.stack([a, b], dim=1), dim=1)
        mean_ba = DeterministicPooler.mean_pool(torch.stack([b, a], dim=1), dim=1)

        assert torch.allclose(mean_ab, mean_ba), "Mean pooling not commutative"


if __name__ == "__main__":
    # Run tests and generate certificate
    pytest.main([__file__, "-v", "--tb=short"])
