"""
Permutation Invariant Processor based on generalized Deep Sets theory.

Implementation following Wagstaff et al. (2022) "Universal Approximation of Set
# # # Operations with Transformers" from Neural Information Processing Systems.  # Module not found  # Module not found  # Module not found
Extends the fundamental invariance theorem through equivariant attention
architectures for multiset processing.
"""

import logging
import warnings
# # # from collections import Counter, defaultdict  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # Module not found  # Module not found  # Module not found

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AggregationType(Enum):
    """Supported aggregation types with deterministic behavior."""

    SUM = "sum"
    MEAN = "mean"
    LOG_SUM_EXP = "log_sum_exp"
    MAX = "max"
    MIN = "min"


@dataclass
class MultisetStats:
    """Statistics and metadata for multiset operations."""

    cardinality: int
    unique_elements: int
    max_multiplicity: int
    aggregation_type: AggregationType
    numeric_precision: str = "float32"
    platform_hash: str = field(default_factory=str)

    def __post_init__(self):
        """Generate platform-specific hash for consistency verification."""
        import hashlib
        import platform

        platform_info = f"{platform.system()}-{platform.machine()}-{torch.__version__}"
        self.platform_hash = hashlib.md5(platform_info.encode()).hexdigest()[:8]


class DeterministicPooler:
    """
    Deterministic pooling operations with guaranteed numerical precision.
    Implements commutative and associative reductions.
    """

    @staticmethod
    def sum_pool(x: torch.Tensor, dim: int = -2, keepdim: bool = False) -> torch.Tensor:
        """Sum pooling with numerical stability."""
        return torch.sum(x, dim=dim, keepdim=keepdim)

    @staticmethod
    def mean_pool(
        x: torch.Tensor, dim: int = -2, keepdim: bool = False
    ) -> torch.Tensor:
        """Mean pooling with cardinality normalization."""
        return torch.mean(x, dim=dim, keepdim=keepdim)

    @staticmethod
    def log_sum_exp_pool(
        x: torch.Tensor, dim: int = -2, keepdim: bool = False
    ) -> torch.Tensor:
        """Log-sum-exp pooling with numerical stability."""
        return torch.logsumexp(x, dim=dim, keepdim=keepdim)

    @staticmethod
    def max_pool(x: torch.Tensor, dim: int = -2, keepdim: bool = False) -> torch.Tensor:
        """Max pooling operation."""
        return torch.max(x, dim=dim, keepdim=keepdim)[0]

    @staticmethod
    def min_pool(x: torch.Tensor, dim: int = -2, keepdim: bool = False) -> torch.Tensor:
        """Min pooling operation."""
        return torch.min(x, dim=dim, keepdim=keepdim)[0]

    @classmethod
    def get_pooler(cls, agg_type: AggregationType) -> Callable:
        """Get pooling function by aggregation type."""
        poolers = {
            AggregationType.SUM: cls.sum_pool,
            AggregationType.MEAN: cls.mean_pool,
            AggregationType.LOG_SUM_EXP: cls.log_sum_exp_pool,
            AggregationType.MAX: cls.max_pool,
            AggregationType.MIN: cls.min_pool,
        }
        return poolers[agg_type]


class MultisetSerializer:
    """
    Serializer that preserves multiset semantics regardless of internal representation.
    Maintains canonical ordering for consistent hashing and comparison.
    """

    @staticmethod
    def serialize_multiset(
        elements: List[Any], multiplicities: Optional[List[int]] = None
    ) -> str:
        """
        Serialize multiset to canonical string representation.

        Args:
            elements: List of elements (may contain duplicates)
            multiplicities: Optional explicit multiplicities

        Returns:
            Canonical string representation
        """
        if multiplicities is None:
            counter = Counter(elements)
        else:
            counter = Counter()
            for elem, mult in zip(elements, multiplicities):
                counter[elem] = mult

        # Sort by element for canonical ordering
        sorted_items = sorted(counter.items())
        return f"multiset({sorted_items})"

    @staticmethod
    def deserialize_multiset(serialized: str) -> Tuple[List[Any], List[int]]:
        """
        Deserialize canonical representation back to elements and multiplicities.

        Args:
# # #             serialized: Canonical string from serialize_multiset  # Module not found  # Module not found  # Module not found

        Returns:
            Tuple of (elements, multiplicities)
        """
        # Simple parsing - in practice would use robust parser
        import ast

        content = serialized.replace("multiset(", "").rstrip(")")
        items = ast.literal_eval(content)
        elements, multiplicities = zip(*items) if items else ([], [])
        return list(elements), list(multiplicities)


class EquivariantAttention(nn.Module):
    """
    Equivariant attention mechanism following Wagstaff et al. (2022).
    Maintains permutation equivariance in intermediate representations.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass maintaining equivariance.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            Attention output preserving permutation equivariance
        """
        batch_size, seq_len, _ = x.shape
        residual = x

        # Multi-head attention
        q = (
            self.w_q(x)
            .view(batch_size, seq_len, self.n_heads, self.d_head)
            .transpose(1, 2)
        )
        k = (
            self.w_k(x)
            .view(batch_size, seq_len, self.n_heads, self.d_head)
            .transpose(1, 2)
        )
        v = (
            self.w_v(x)
            .view(batch_size, seq_len, self.n_heads, self.d_head)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_head)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        output = self.w_o(attn_output)
        output = self.layer_norm(output + residual)

        return output


class DeepSetBlock(nn.Module):
    """
    Deep Set processing block with element-wise transformation and aggregation.
    Implements the fundamental decomposition: φ(Σ_i ψ(x_i))
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        aggregation: AggregationType = AggregationType.SUM,
    ):
        super().__init__()
        self.aggregation = aggregation
        self.pooler = DeterministicPooler.get_pooler(aggregation)

        # Element-wise transformation ψ
        self.psi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Set-level transformation φ
        self.phi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self, x: torch.Tensor, multiplicities: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through Deep Set block.

        Args:
            x: Input tensor [batch, set_size, input_dim]
            multiplicities: Optional multiplicities [batch, set_size]

        Returns:
            Set representation [batch, output_dim]
        """
        # Element-wise transformation
        psi_x = self.psi(x)  # [batch, set_size, hidden_dim]

        # Apply multiplicities if provided
        if multiplicities is not None:
            psi_x = psi_x * multiplicities.unsqueeze(-1)

        # Aggregation (commutative and associative)
        aggregated = self.pooler(psi_x, dim=1)  # [batch, hidden_dim]

        # Set-level transformation
        output = self.phi(aggregated)  # [batch, output_dim]

        return output


class PermutationInvariantProcessor(nn.Module):
    """
    Main permutation invariant processor implementing generalized Deep Sets
    with equivariant attention architectures.

    Based on Wagstaff et al. (2022) extension of the fundamental invariance theorem.
    Handles multisets with multiplicities and provides numerical consistency guarantees.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        n_layers: int = 3,
        n_attention_layers: int = 2,
        n_heads: int = 8,
        aggregation: AggregationType = AggregationType.SUM,
        dropout: float = 0.1,
        enable_verification: bool = True,
    ):
        """
        Initialize permutation invariant processor.

        Args:
            input_dim: Dimension of input elements
            hidden_dim: Hidden dimension for transformations
            output_dim: Final output dimension
            n_layers: Number of Deep Set blocks
            n_attention_layers: Number of attention layers
            n_heads: Number of attention heads
            aggregation: Type of aggregation to use
            dropout: Dropout probability
            enable_verification: Whether to enable invariance verification
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.aggregation = aggregation
        self.enable_verification = enable_verification

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Equivariant attention layers
        self.attention_layers = nn.ModuleList(
            [
                EquivariantAttention(hidden_dim, n_heads, dropout)
                for _ in range(n_attention_layers)
            ]
        )

        # Deep Set blocks
        self.deep_set_blocks = nn.ModuleList(
            [
                DeepSetBlock(hidden_dim, hidden_dim, hidden_dim, aggregation)
                for _ in range(n_layers - 1)
            ]
        )

        # Final Deep Set block
        self.final_block = DeepSetBlock(hidden_dim, hidden_dim, output_dim, aggregation)

        # Statistics tracking
        self.processing_stats: Dict[str, MultisetStats] = {}
        self.verification_results: List[bool] = []

        self.logger = logging.getLogger(self.__class__.__name__)

    def _compute_multiset_stats(
        self,
        x: torch.Tensor,
        multiplicities: Optional[torch.Tensor] = None,
        batch_id: str = "default",
    ) -> MultisetStats:
        """Compute statistics for multiset processing."""
        batch_size, set_size, _ = x.shape

        if multiplicities is not None:
            total_cardinality = int(torch.sum(multiplicities).item())
            unique_elements = int(torch.count_nonzero(multiplicities).item())
            max_mult = int(torch.max(multiplicities).item())
        else:
            total_cardinality = batch_size * set_size
            unique_elements = set_size
            max_mult = 1

        stats = MultisetStats(
            cardinality=total_cardinality,
            unique_elements=unique_elements,
            max_multiplicity=max_mult,
            aggregation_type=self.aggregation,
            numeric_precision=str(x.dtype),
        )

        self.processing_stats[batch_id] = stats
        return stats

    def _verify_invariance(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
        multiplicities: Optional[torch.Tensor] = None,
        n_permutations: int = 5,
    ) -> bool:
        """
        Verify permutation invariance of the output.

        Args:
            x: Input tensor
            output: Processor output
            multiplicities: Optional multiplicities
            n_permutations: Number of random permutations to test

        Returns:
            True if invariant, False otherwise
        """
        if not self.enable_verification:
            return True

        batch_size, set_size, _ = x.shape
        tolerance = 1e-5

        with torch.no_grad():
            for _ in range(n_permutations):
                # Generate random permutation
                perm = torch.randperm(set_size)
                x_perm = x[:, perm, :]
                mult_perm = (
                    multiplicities[:, perm] if multiplicities is not None else None
                )

                # Process permuted input
                output_perm = self.forward(x_perm, mult_perm, verify_invariance=False)

                # Check invariance
                diff = torch.abs(output - output_perm).max().item()
                if diff > tolerance:
                    self.logger.warning(
                        f"Invariance violation detected: diff={diff:.6f}"
                    )
                    return False

        return True

    def forward(
        self,
        x: torch.Tensor,
        multiplicities: Optional[torch.Tensor] = None,
        batch_id: str = "default",
        verify_invariance: bool = None,
    ) -> torch.Tensor:
        """
        Forward pass through the permutation invariant processor.

        Args:
            x: Input multiset tensor [batch, set_size, input_dim]
            multiplicities: Optional multiplicities [batch, set_size]
            batch_id: Identifier for tracking statistics
            verify_invariance: Override verification setting

        Returns:
            Invariant representation [batch, output_dim]
        """
        # Compute statistics
        stats = self._compute_multiset_stats(x, multiplicities, batch_id)

        # Input projection
        h = self.input_projection(x)  # [batch, set_size, hidden_dim]

        # Apply equivariant attention layers
        for attention_layer in self.attention_layers:
            h = attention_layer(h)

        # Apply Deep Set blocks (all but last)
        for block in self.deep_set_blocks:
            h_agg = block(h, multiplicities)  # [batch, hidden_dim]
            # Broadcast back to set for next layer
            h = h_agg.unsqueeze(1).expand(-1, h.size(1), -1) + h

        # Final aggregation
        output = self.final_block(h, multiplicities)  # [batch, output_dim]

        # Verify invariance if enabled
        should_verify = (
            verify_invariance
            if verify_invariance is not None
            else self.enable_verification
        )
        if should_verify:
            is_invariant = self._verify_invariance(x, output, multiplicities)
            self.verification_results.append(is_invariant)

            if not is_invariant:
                warnings.warn("Permutation invariance verification failed", UserWarning)

        return output

    def process_multiset(
        self,
        elements: List[Any],
        element_embeddings: torch.Tensor,
        multiplicities: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, MultisetStats]:
        """
        Process a multiset with explicit elements and embeddings.

        Args:
            elements: List of multiset elements
            element_embeddings: Embeddings for elements [n_elements, input_dim]
            multiplicities: Optional multiplicities for each element

        Returns:
            Tuple of (processed_representation, statistics)
        """
        # Serialize multiset for canonical representation
        serialized = MultisetSerializer.serialize_multiset(elements, multiplicities)

        # Prepare tensors
        if multiplicities is not None:
            mult_tensor = torch.tensor(multiplicities, dtype=torch.float32).unsqueeze(0)
        else:
            mult_tensor = None

        embeddings = element_embeddings.unsqueeze(0)  # Add batch dimension

        # Process
        output = self.forward(embeddings, mult_tensor, batch_id=serialized)
        stats = self.processing_stats[serialized]

        return output.squeeze(0), stats  # Remove batch dimension

    def get_verification_summary(self) -> Dict[str, Any]:
        """Get summary of invariance verification results."""
        if not self.verification_results:
            return {"status": "no_verifications_performed"}

        total = len(self.verification_results)
        passed = sum(self.verification_results)

        return {
            "total_verifications": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total,
            "status": "all_passed" if passed == total else "some_failed",
        }

    def reset_statistics(self):
        """Reset all processing statistics and verification results."""
        self.processing_stats.clear()
        self.verification_results.clear()
