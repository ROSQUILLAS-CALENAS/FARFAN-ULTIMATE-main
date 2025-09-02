"""
Entropic Gromov-Wasserstein Alignment Module

Implements EGW optimal transport for aligning pattern graphs to corpus/topic graphs
and DNP vocabulary, creating fused lexical+semantic costs.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import ot
import torch
from scipy.spatial.distance import cdist
# Optional sklearn cosine_distances with fallback
try:
    from sklearn.metrics.pairwise import cosine_distances  # type: ignore
except Exception:
    def cosine_distances(A, B):
        import numpy as np
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
        # distance = 1 - cosine_similarity
        return 1.0 - (A_norm @ B_norm.T)


class GromovWassersteinAligner:
    """
    Entropic Gromov-Wasserstein aligner for query pattern alignment.

    Aligns pattern graphs to corpus/topic graphs and DNP vocabulary using
    entropic regularized optimal transport with stability tracking.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        lambda_reg: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
        device: str = "cpu",
    ):
        """
        Initialize GW aligner.

        Args:
            epsilon: Entropic regularization parameter
            lambda_reg: Regularization for GW term
            max_iter: Maximum iterations for Sinkhorn
            tol: Tolerance for convergence
            device: Computation device
        """
        self.epsilon = epsilon
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol
        self.device = device

        # Stability tracking
        self.stability_log = []

        self.logger = logging.getLogger(__name__)

    def compute_structure_cost(
        self,
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        metric: str = "euclidean",
    ) -> np.ndarray:
        """
        Compute structural cost matrix between embeddings.

        Args:
            source_embeddings: Source embedding matrix [n_source, dim]
            target_embeddings: Target embedding matrix [n_target, dim]
            metric: Distance metric

        Returns:
            Cost matrix [n_source, n_target]
        """
        if metric == "cosine":
            return cosine_distances(source_embeddings, target_embeddings)
        else:
            return cdist(source_embeddings, target_embeddings, metric=metric)

    def compute_gw_cost(self, C1: np.ndarray, C2: np.ndarray, T: np.ndarray) -> float:
        """
        Compute Gromov-Wasserstein cost for current transport plan T.

        Args:
            C1: Source structure cost matrix [n1, n1]
            C2: Target structure cost matrix [n2, n2]
            T: Transport plan [n1, n2]

        Returns:
            GW cost value
        """
        # GW cost: sum_{i,j,k,l} |C1[i,k] - C2[j,l]|^2 * T[i,j] * T[k,l]
        gw_cost = 0.0
        n1, n2 = T.shape

        for i in range(n1):
            for j in range(n2):
                for k in range(n1):
                    for l in range(n2):
                        gw_cost += ((C1[i, k] - C2[j, l]) ** 2) * T[i, j] * T[k, l]

        return gw_cost

    def entropic_gw_alignment(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
        source_weights: Optional[np.ndarray] = None,
        target_weights: Optional[np.ndarray] = None,
        fused_cost: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Perform entropic Gromov-Wasserstein alignment.

        Args:
            source_features: Source feature matrix [n_source, dim]
            target_features: Target feature matrix [n_target, dim]
            source_weights: Source distribution weights
            target_weights: Target distribution weights
            fused_cost: Additional cost matrix for fusion

        Returns:
            Transport plan matrix and alignment info
        """
        global current_cost
        n_source, n_target = source_features.shape[0], target_features.shape[0]

        # Default uniform distributions
        if source_weights is None:
            source_weights = np.ones(n_source) / n_source
        if target_weights is None:
            target_weights = np.ones(n_target) / n_target

        # Compute structure costs
        C1 = self.compute_structure_cost(source_features, source_features)
        C2 = self.compute_structure_cost(target_features, target_features)

        # Initial transport plan
        T = np.outer(source_weights, target_weights)

        # Entropic GW with Sinkhorn iterations
        prev_cost = float("inf")

        alignment_info = {
            "converged": False,
            "iterations": 0,
            "final_cost": 0.0,
            "stability_epsilon": self.epsilon,
            "stability_lambda": self.lambda_reg,
        }

        for iteration in range(self.max_iter):
            # Compute GW gradient
            gw_grad = self.compute_gw_gradient(C1, C2, T)

            # Add fused cost if provided
            if fused_cost is not None:
                total_cost = gw_grad + fused_cost
            else:
                total_cost = gw_grad

            # Entropic regularization step (Sinkhorn)
            T = self.sinkhorn_step(total_cost, source_weights, target_weights)

            # Compute current cost
            current_cost = self.compute_gw_cost(C1, C2, T)

            # Stability check
            cost_change = abs(current_cost - prev_cost)
            if cost_change < self.tol:
                alignment_info["converged"] = True
                break

            prev_cost = current_cost
            alignment_info["iterations"] = iteration + 1

        alignment_info["final_cost"] = current_cost

        # Log stability metrics
        stability_metrics = {
            "epsilon": self.epsilon,
            "lambda": self.lambda_reg,
            "final_cost": current_cost,
            "iterations": alignment_info["iterations"],
            "converged": alignment_info["converged"],
        }
        self.stability_log.append(stability_metrics)

        self.logger.info(
            f"EGW alignment completed: cost={current_cost:.6f}, "
            f"iterations={alignment_info['iterations']}, "
            f"converged={alignment_info['converged']}"
        )

        return T, alignment_info

    def compute_gw_gradient(
        self, C1: np.ndarray, C2: np.ndarray, T: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient of GW cost w.r.t. transport plan.

        Args:
            C1: Source cost matrix [n1, n1]
            C2: Target cost matrix [n2, n2]
            T: Current transport plan [n1, n2]

        Returns:
            Gradient matrix [n1, n2]
        """
        n1, n2 = T.shape
        grad = np.zeros((n1, n2))

        # Compute gradient: 2 * sum_k,l (C1[i,k] - C2[j,l]) * T[k,l]
        for i in range(n1):
            for j in range(n2):
                grad_val = 0.0
                for k in range(n1):
                    for l in range(n2):
                        grad_val += (C1[i, k] - C2[j, l]) * T[k, l]
                grad[i, j] = 2 * grad_val

        return grad

    def sinkhorn_step(
        self,
        cost_matrix: np.ndarray,
        source_weights: np.ndarray,
        target_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Single Sinkhorn iteration for entropic regularization.

        Args:
            cost_matrix: Cost matrix [n1, n2]
            source_weights: Source marginal [n1]
            target_weights: Target marginal [n2]

        Returns:
            Updated transport plan
        """
        # Exponential transformation with entropic regularization
        K = np.exp(-cost_matrix / self.epsilon)

        # Sinkhorn scaling
        u = np.ones(len(source_weights))
        v = np.ones(len(target_weights))

        # Iterate until convergence (simplified single step here)
        for _ in range(10):  # Inner Sinkhorn iterations
            u = source_weights / (K @ v + 1e-8)
            v = target_weights / (K.T @ u + 1e-8)

        T = np.diag(u) @ K @ np.diag(v)
        return T

    def create_fused_cost(
        self, lexical_cost: np.ndarray, semantic_cost: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create fused cost matrix combining lexical and semantic costs.

        Args:
            lexical_cost: Lexical similarity cost matrix
            semantic_cost: Semantic similarity cost matrix
            alpha: Balance parameter (0=lexical, 1=semantic)

        Returns:
            Fused cost matrix
        """
        # Normalize costs to [0, 1]
        lexical_norm = (lexical_cost - lexical_cost.min()) / (
            lexical_cost.max() - lexical_cost.min() + 1e-8
        )
        semantic_norm = (semantic_cost - semantic_cost.min()) / (
            semantic_cost.max() - semantic_cost.min() + 1e-8
        )

        fused = (1 - alpha) * lexical_norm + alpha * semantic_norm
        return fused

    def align_pattern_to_corpus(
        self,
        pattern_features: np.ndarray,
        corpus_features: np.ndarray,
        lexical_similarity: Optional[np.ndarray] = None,
        alpha: float = 0.5,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Align pattern graph to corpus facets using EGW.

        Args:
            pattern_features: Pattern embedding features [n_patterns, dim]
            corpus_features: Corpus embedding features [n_corpus, dim]
            lexical_similarity: Optional lexical similarity matrix
            alpha: Balance between lexical/semantic costs

        Returns:
            Transport plan and alignment information
        """
        # Compute semantic cost
        semantic_cost = self.compute_structure_cost(
            pattern_features, corpus_features, metric="cosine"
        )

        # Create fused cost if lexical similarity provided
        fused_cost = None
        if lexical_similarity is not None:
            fused_cost = self.create_fused_cost(
                lexical_similarity, semantic_cost, alpha
            )

        # Perform EGW alignment
        transport_plan, info = self.entropic_gw_alignment(
            pattern_features, corpus_features, fused_cost=fused_cost
        )

        return transport_plan, info

    def get_stability_metrics(self) -> List[Dict]:
        """Return logged stability metrics."""
        return self.stability_log

    def max_transport_radius(
        self, transport_plan: np.ndarray, features: np.ndarray
    ) -> float:
        """
        Compute maximum transport radius for stability constraint.

        Args:
            transport_plan: Transport plan matrix [n1, n2]
            features: Feature matrix for distance computation

        Returns:
            Maximum transport radius
        """
        max_radius = 0.0
        n1, n2 = transport_plan.shape

        for i in range(n1):
            for j in range(n2):
                if transport_plan[i, j] > 1e-6:  # Non-negligible transport
                    # Use transport plan weight as proxy for radius
                    radius = transport_plan[i, j] * np.linalg.norm(
                        features[i] if i < len(features) else features[0]
                    )
                    max_radius = max(max_radius, radius)

        return max_radius
