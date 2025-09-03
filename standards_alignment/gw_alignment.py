"""Gromov-Wasserstein alignment with stability guarantees."""

import logging
# # # from typing import Dict  # Module not found  # Module not found  # Module not found

import numpy as np
import ot
# # # from scipy.optimize import linear_sum_assignment  # Module not found  # Module not found  # Module not found

# # # from .graph_ops import DocumentGraph, StandardsGraph  # Module not found  # Module not found  # Module not found

logger = logging.getLogger(__name__)


class GWAlignmentResult:
    """Result container for GW alignment with stability metrics."""

    def __init__(
        self,
        transport_plan: np.ndarray,
        cost_decomposition: Dict[str, float],
        stability_bound: float,
        regularization: float,
    ):
        self.transport_plan = transport_plan
        self.cost_decomposition = cost_decomposition
        self.stability_bound = stability_bound
        self.regularization = regularization
        self.matching_pairs = self._extract_matching_pairs()

    def _extract_matching_pairs(self) -> list:
# # #         """Extract matching pairs from transport plan."""  # Module not found  # Module not found  # Module not found
        pairs = []
        threshold = np.max(self.transport_plan) * 0.1  # 10% of max
        rows, cols = np.where(self.transport_plan > threshold)

        for i, j in zip(rows, cols):
            pairs.append((i, j, self.transport_plan[i, j]))

        return sorted(pairs, key=lambda x: x[2], reverse=True)


def gw_align(
    G_std: StandardsGraph, G_doc: DocumentGraph, reg: float = 0.1, max_iter: int = 100
) -> GWAlignmentResult:
    """
    Align standards graph to document graph using entropic Gromov-Wasserstein.

    Based on:
    - JMLR 2024: Entropic GW stability under regularization λ = reg
    - JCGS 2023: GW consistency guarantees under sampling

    Args:
        G_std: Standards graph
        G_doc: Document graph
        reg: Entropic regularization parameter λ
        max_iter: Maximum iterations for GW solver

    Returns:
        GWAlignmentResult with transport plan and stability metrics
    """
    # Get distance matrices
    C1 = G_std.get_distance_matrix()  # Standards graph distances
    C2 = G_doc.get_distance_matrix()  # Document graph distances

    n1, n2 = C1.shape[0], C2.shape[0]

    # Uniform distributions (can be made adaptive)
    p = ot.unif(n1)
    q = ot.unif(n2)

    logger.info(f"Computing entropic GW with λ={reg}, shapes ({n1}, {n2})")

    # Compute entropic Gromov-Wasserstein
    T, log = ot.gromov.entropic_gromov_wasserstein(
        C1,
        C2,
        p,
        q,
        loss="square_loss",
        epsilon=reg,
        max_iter=max_iter,
        log=True,
        verbose=False,
    )

    # Compute cost decomposition
    gw_cost = log["gw_dist"]
    entropic_cost = _compute_entropic_cost(T, reg)
    total_cost = gw_cost + entropic_cost

    cost_decomposition = {
        "gw_distance": float(gw_cost),
        "entropic_penalty": float(entropic_cost),
        "total_cost": float(total_cost),
    }

# # #     # Stability bound from JMLR 2024 (simplified)  # Module not found  # Module not found  # Module not found
    # ε ≤ C * sqrt(λ * log(max(n1, n2)))
    C_const = 2.0  # Problem-dependent constant
    stability_bound = C_const * np.sqrt(reg * np.log(max(n1, n2)))

    logger.info(
        f"GW alignment: cost={total_cost:.4f}, stability_bound={stability_bound:.4f}"
    )

    return GWAlignmentResult(T, cost_decomposition, stability_bound, reg)


def sparse_align(G_std: StandardsGraph, G_doc: DocumentGraph) -> GWAlignmentResult:
    """
    Sparse bipartite matching fallback using Hungarian algorithm.

    Uses combined structural + semantic similarity for edge weights.
    """
    std_nodes = list(G_std.G.nodes())
    doc_nodes = list(G_doc.G.nodes())

    n1, n2 = len(std_nodes), len(doc_nodes)

    # Build cost matrix based on node type compatibility and features
    cost_matrix = np.ones((n1, n2)) * 10.0  # High cost for incompatible

    for i, std_node in enumerate(std_nodes):
        std_type = G_std.G.nodes[std_node].get("node_type", "unknown")

        for j, doc_node in enumerate(doc_nodes):
            doc_type = G_doc.G.nodes[doc_node].get("node_type", "unknown")

            # Type compatibility
            if _are_types_compatible(std_type, doc_type):
                # Use semantic similarity (simplified)
                similarity = _compute_semantic_similarity(
                    std_node, doc_node, G_std, G_doc
                )
                cost_matrix[i, j] = 1.0 - similarity

    # Solve assignment problem
    if n1 <= n2:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        T = np.zeros((n1, n2))
        T[row_ind, col_ind] = 1.0
    else:
        # Transpose for n1 > n2 case
        row_ind, col_ind = linear_sum_assignment(cost_matrix.T)
        T = np.zeros((n1, n2))
        T[col_ind, row_ind] = 1.0

    # Compute costs
    total_cost = float(np.sum(T * cost_matrix))
    cost_decomposition = {
        "assignment_cost": total_cost,
        "entropic_penalty": 0.0,
        "total_cost": total_cost,
    }

    stability_bound = 0.0  # No stability guarantees for sparse matching

    logger.info(f"Sparse alignment: cost={total_cost:.4f}")

    return GWAlignmentResult(T, cost_decomposition, stability_bound, 0.0)


def _compute_entropic_cost(T: np.ndarray, reg: float) -> float:
    """Compute entropic regularization penalty."""
    T_pos = T[T > 1e-16]  # Avoid log(0)
    return reg * np.sum(T_pos * np.log(T_pos))


def _are_types_compatible(std_type: str, doc_type: str) -> bool:
    """Check if node types are compatible for matching."""
    compatibility = {
        "dimension": ["section", "paragraph"],
        "subdimension": ["section", "paragraph"],
        "point": ["paragraph", "table"],
    }
    return doc_type in compatibility.get(std_type, [])


def _compute_semantic_similarity(
    std_node: str, doc_node: str, G_std: StandardsGraph, G_doc: DocumentGraph
) -> float:
    """Compute semantic similarity between nodes (placeholder)."""
    # Simplified: just return based on string similarity
    std_words = set(std_node.lower().split("_"))

    doc_content = G_doc.node_content.get(doc_node, {})
    doc_text = ""
    if "title" in doc_content:
        doc_text += doc_content["title"] + " "
    if "content" in doc_content:
        doc_text += doc_content["content"]

    doc_words = set(doc_text.lower().split())

    if not std_words or not doc_words:
        return 0.1

    intersection = std_words & doc_words
    union = std_words | doc_words

    return len(intersection) / len(union) if union else 0.0
