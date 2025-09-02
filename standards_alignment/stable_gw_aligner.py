"""
Stable Entropic Gromov-Wasserstein Aligner

Implements stable optimal transport alignment based on Peyré et al. (2020)
"Computational Optimal Transport" with frozen hyperparameters, deterministic
convergence, explicit unmatched mass handling, and full auditability.
"""

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StableGWConfig:
    """Frozen hyperparameters for stable alignment."""

    epsilon: float = 0.05  # Entropic regularization (frozen)
    lambda_struct: float = 0.1  # Structural weight (frozen)
    lambda_content: float = 0.9  # Content weight (frozen)
    max_iter: int = 1000
    convergence_threshold: int = 10  # Deterministic: iterations without improvement
    sinkhorn_max_iter: int = 100
    min_transport_mass: float = 1e-8  # Minimum mass for coupling
    coverage_threshold: float = 0.85  # Coverage gap detection threshold


@dataclass
class TransportPlan:
    """Complete transport plan with serialization support."""

    coupling_matrix: np.ndarray
    source_marginals: np.ndarray
    target_marginals: np.ndarray
    cost_decomposition: Dict[str, float]
    convergence_info: Dict[str, Any]
    unmatched_mass_info: Dict[str, float]
    stability_metrics: Dict[str, float]

    def save(self, filepath: Path) -> None:
        """Serialize complete transport plan."""
        data = {
            "coupling_matrix": self.coupling_matrix.tolist(),
            "source_marginals": self.source_marginals.tolist(),
            "target_marginals": self.target_marginals.tolist(),
            "cost_decomposition": self.cost_decomposition,
            "convergence_info": self.convergence_info,
            "unmatched_mass_info": self.unmatched_mass_info,
            "stability_metrics": self.stability_metrics,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "TransportPlan":
        """Deserialize transport plan."""
        with open(filepath, "r") as f:
            data = json.load(f)

        return cls(
            coupling_matrix=np.array(data["coupling_matrix"]),
            source_marginals=np.array(data["source_marginals"]),
            target_marginals=np.array(data["target_marginals"]),
            cost_decomposition=data["cost_decomposition"],
            convergence_info=data["convergence_info"],
            unmatched_mass_info=data["unmatched_mass_info"],
            stability_metrics=data["stability_metrics"],
        )


class StableEGWAligner:
    """
    Stable Entropic Gromov-Wasserstein Aligner with reproducibility guarantees.

    Features:
    - Frozen hyperparameters for stability
    - Deterministic convergence criteria
    - Explicit unmatched mass quantification
    - Complete auditability through serialization
    - Identical results for identical inputs
    """

    def __init__(self, config: StableGWConfig = None):
        """Initialize with frozen configuration."""
        self.config = config or StableGWConfig()
        self.alignment_cache = {}  # Cache for reproducibility

    def _compute_graph_hash(self, graph: nx.Graph) -> str:
        """Compute deterministic hash of graph structure and content."""
        # Sort nodes and edges for deterministic ordering
        nodes = sorted(graph.nodes(data=True), key=lambda x: str(x[0]))
        edges = sorted(
            graph.edges(data=True),
            key=lambda x: (min(str(x[0]), str(x[1])), max(str(x[0]), str(x[1]))),
        )

        graph_data = {
            "nodes": [
                (str(node), sorted(data.items()) if data else [])
                for node, data in nodes
            ],
            "edges": [
                (
                    min(str(u), str(v)),
                    max(str(u), str(v)),
                    sorted(data.items()) if data else [],
                )
                for u, v, data in edges
            ],
        }

        graph_str = json.dumps(graph_data, sort_keys=True)
        return hashlib.sha256(graph_str.encode()).hexdigest()

    def _compute_structure_matrix(
        self, graph: nx.Graph
    ) -> Tuple[np.ndarray, List[str]]:
        """Compute structure cost matrix preserving topological information."""
        nodes = sorted(graph.nodes())
        n = len(nodes)

        # Compute shortest path distances
        distances = dict(nx.all_pairs_shortest_path_length(graph))

        # Create structure matrix
        C = np.zeros((n, n))
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if node_j in distances.get(node_i, {}):
                    C[i, j] = distances[node_i][node_j]
                else:
                    C[i, j] = n  # Disconnected nodes get maximum distance

        return C, nodes

    def _compute_content_matrix(
        self,
        source_graph: nx.Graph,
        target_graph: nx.Graph,
        source_nodes: List[str],
        target_nodes: List[str],
    ) -> np.ndarray:
        """Compute content similarity matrix."""
        n_source = len(source_nodes)
        n_target = len(target_nodes)

        content_matrix = np.zeros((n_source, n_target))

        for i, src_node in enumerate(source_nodes):
            src_data = source_graph.nodes[src_node]
            src_content = self._extract_node_content(src_data)

            for j, tgt_node in enumerate(target_nodes):
                tgt_data = target_graph.nodes[tgt_node]
                tgt_content = self._extract_node_content(tgt_data)

                # Compute content similarity (1 - normalized distance)
                similarity = self._content_similarity(src_content, tgt_content)
                content_matrix[i, j] = 1.0 - similarity

        return content_matrix

    def _extract_node_content(self, node_data: Dict) -> str:
        """Extract textual content from node data."""
        content_fields = ["title", "content", "text", "label", "description"]
        content = ""

        for field in content_fields:
            if field in node_data:
                content += str(node_data[field]) + " "

        return content.strip().lower()

    def _content_similarity(self, content1: str, content2: str) -> float:
        """Compute content similarity using Jaccard similarity."""
        if not content1 or not content2:
            return 0.0

        words1 = set(content1.split())
        words2 = set(content2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def _sinkhorn_stabilized(
        self,
        cost_matrix: np.ndarray,
        source_weights: np.ndarray,
        target_weights: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        """Stabilized Sinkhorn algorithm with deterministic convergence."""
        global max_error, iteration
        epsilon = self.config.epsilon
        max_iter = self.config.sinkhorn_max_iter

        n_source, n_target = cost_matrix.shape

        # Log-domain stabilization
        K = np.exp(-cost_matrix / epsilon)

        # Initialize dual variables
        u = np.ones(n_source)
        v = np.ones(n_target)

        # Track convergence
        convergence_history = []

        for iteration in range(max_iter):
            u_prev = u.copy()
            v_prev = v.copy()

            # Sinkhorn updates
            u = source_weights / (K @ v + self.config.min_transport_mass)
            v = target_weights / (K.T @ u + self.config.min_transport_mass)

            # Compute marginal errors
            marginal_error_u = np.max(np.abs(u - u_prev))
            marginal_error_v = np.max(np.abs(v - v_prev))
            max_error = max(marginal_error_u, marginal_error_v)

            convergence_history.append(
                {
                    "iteration": iteration,
                    "marginal_error": max_error,
                    "u_norm": np.linalg.norm(u),
                    "v_norm": np.linalg.norm(v),
                }
            )

            # Deterministic convergence: check last N iterations
            if len(convergence_history) >= self.config.convergence_threshold:
                recent_errors = [
                    h["marginal_error"]
                    for h in convergence_history[-self.config.convergence_threshold :]
                ]

                if all(error < 1e-6 for error in recent_errors):
                    break

        # Compute final transport plan
        T = np.diag(u) @ K @ np.diag(v)

        convergence_info = {
            "sinkhorn_iterations": iteration + 1,
            "final_marginal_error": max_error,
            "convergence_history": convergence_history,
            "converged": iteration < max_iter - 1,
        }

        return T, convergence_info

    def _compute_gw_gradient(
        self, C_source: np.ndarray, C_target: np.ndarray, T: np.ndarray
    ) -> np.ndarray:
        """Compute Gromov-Wasserstein gradient tensor contraction."""
        n_source, n_target = T.shape

        # Efficient tensor contraction for GW gradient
        # grad[i,j] = 2 * sum_{k,l} (C_source[i,k] - C_target[j,l])^2 * T[k,l]

        # Pre-compute squared matrices
        C_source_sq = C_source**2
        C_target_sq = C_target**2

        # Compute gradient components
        grad = np.zeros((n_source, n_target))

        # First term: 2 * C_source^2 @ T @ ones
        grad += 2 * (C_source_sq @ T @ np.ones((n_target, 1))).reshape(n_source, 1)

        # Second term: -4 * C_source @ T @ C_target^T
        grad -= 4 * C_source @ T @ C_target.T

        # Third term: 2 * ones^T @ T @ C_target^2
        grad += 2 * (np.ones((1, n_source)) @ T @ C_target_sq).reshape(1, n_target)

        return grad

    def _quantify_unmatched_mass(self, T: np.ndarray) -> Dict[str, float]:
        """Quantify unmatched mass and coverage gaps."""
        source_coverage = np.sum(T, axis=1)  # Mass assigned from each source
        target_coverage = np.sum(T, axis=0)  # Mass received by each target

        # Unmatched source mass
        unmatched_source = np.sum(1.0 - source_coverage)

        # Unmatched target mass
        unmatched_target = np.sum(1.0 - target_coverage)

        # Coverage statistics
        source_coverage_ratio = np.mean(source_coverage)
        target_coverage_ratio = np.mean(target_coverage)

        # Gap detection
        coverage_gap = 1.0 - min(source_coverage_ratio, target_coverage_ratio)
        coverage_adequate = coverage_gap < (1.0 - self.config.coverage_threshold)

        return {
            "unmatched_source_mass": float(unmatched_source),
            "unmatched_target_mass": float(unmatched_target),
            "source_coverage_ratio": float(source_coverage_ratio),
            "target_coverage_ratio": float(target_coverage_ratio),
            "coverage_gap": float(coverage_gap),
            "coverage_adequate": bool(coverage_adequate),
            "min_coverage_achieved": float(
                min(source_coverage_ratio, target_coverage_ratio)
            ),
        }

    def align_graphs(
        self,
        standards_graph: nx.Graph,
        document_graph: nx.Graph,
        cache_key: Optional[str] = None,
    ) -> TransportPlan:
        """
        Align standards graph to document graph with stability guarantees.

        Args:
            standards_graph: Source standards graph
            document_graph: Target document graph
            cache_key: Optional cache key for reproducibility

        Returns:
            Complete transport plan with full auditability
        """
        # Generate cache key for reproducibility
        global sinkhorn_info, transport_change, iteration
        if cache_key is None:
            source_hash = self._compute_graph_hash(standards_graph)
            target_hash = self._compute_graph_hash(document_graph)
            cache_key = f"{source_hash}_{target_hash}"

        # Check cache for identical inputs
        if cache_key in self.alignment_cache:
            logger.info(f"Returning cached alignment for key: {cache_key}")
            return self.alignment_cache[cache_key]

        logger.info(
            f"Computing stable EGW alignment for graphs: "
            f"standards={len(standards_graph.nodes)} nodes, "
            f"documents={len(document_graph.nodes)} nodes"
        )

        # Compute structure matrices
        C_source, source_nodes = self._compute_structure_matrix(standards_graph)
        C_target, target_nodes = self._compute_structure_matrix(document_graph)

        # Compute content similarity matrix
        M_content = self._compute_content_matrix(
            standards_graph, document_graph, source_nodes, target_nodes
        )

        n_source, n_target = len(source_nodes), len(target_nodes)

        # Initialize uniform distributions
        source_weights = np.ones(n_source) / n_source
        target_weights = np.ones(n_target) / n_target

        # Initialize transport plan
        T = np.outer(source_weights, target_weights)

        # Track costs and convergence
        cost_history = []
        convergence_plateau = 0

        for iteration in range(self.config.max_iter):
            T_prev = T.copy()

            # Compute Gromov-Wasserstein gradient
            gw_gradient = self._compute_gw_gradient(C_source, C_target, T)

            # Combine structural and content costs
            combined_cost = (
                self.config.lambda_struct * gw_gradient
                + self.config.lambda_content * M_content
            )

            # Sinkhorn step with stabilization
            T, sinkhorn_info = self._sinkhorn_stabilized(
                combined_cost, source_weights, target_weights
            )

            # Compute current costs
            gw_cost = np.sum(gw_gradient * T)
            content_cost = np.sum(M_content * T)
            entropic_cost = self.config.epsilon * np.sum(
                T * np.log(T + self.config.min_transport_mass)
            )
            total_cost = gw_cost + content_cost + entropic_cost

            cost_info = {
                "iteration": iteration,
                "gw_cost": float(gw_cost),
                "content_cost": float(content_cost),
                "entropic_cost": float(entropic_cost),
                "total_cost": float(total_cost),
            }
            cost_history.append(cost_info)

            # Deterministic convergence check
            transport_change = np.max(np.abs(T - T_prev))

            if len(cost_history) >= 2:
                cost_change = abs(
                    cost_history[-1]["total_cost"] - cost_history[-2]["total_cost"]
                )
                if cost_change < 1e-8:
                    convergence_plateau += 1
                else:
                    convergence_plateau = 0

                if convergence_plateau >= self.config.convergence_threshold:
                    break

        # Final cost decomposition
        final_costs = (
            cost_history[-1]
            if cost_history
            else {
                "gw_cost": 0.0,
                "content_cost": 0.0,
                "entropic_cost": 0.0,
                "total_cost": 0.0,
            }
        )

        # Convergence information
        convergence_info = {
            "egw_iterations": iteration + 1,
            "final_transport_change": float(transport_change)
            if "transport_change" in locals()
            else 0.0,
            "convergence_plateau_length": convergence_plateau,
            "converged": convergence_plateau >= self.config.convergence_threshold,
            "cost_history": cost_history,
            "sinkhorn_info": sinkhorn_info if "sinkhorn_info" in locals() else {},
        }

        # Quantify unmatched mass
        unmatched_info = self._quantify_unmatched_mass(T)

        # Stability metrics
        stability_metrics = {
            "epsilon": self.config.epsilon,
            "lambda_struct": self.config.lambda_struct,
            "lambda_content": self.config.lambda_content,
            "max_transport_value": float(np.max(T)),
            "min_transport_value": float(np.min(T)),
            "transport_entropy": float(
                -np.sum(T * np.log(T + self.config.min_transport_mass))
            ),
            "cache_key": cache_key,
        }

        # Create transport plan
        transport_plan = TransportPlan(
            coupling_matrix=T,
            source_marginals=source_weights,
            target_marginals=target_weights,
            cost_decomposition=final_costs,
            convergence_info=convergence_info,
            unmatched_mass_info=unmatched_info,
            stability_metrics=stability_metrics,
        )

        # Cache for reproducibility
        self.alignment_cache[cache_key] = transport_plan

        logger.info(
            f"EGW alignment completed: "
            f"total_cost={final_costs['total_cost']:.6f}, "
            f"iterations={convergence_info['egw_iterations']}, "
            f"converged={convergence_info['converged']}, "
            f"coverage_gap={unmatched_info['coverage_gap']:.3f}"
        )

        return transport_plan

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get alignment cache statistics."""
        return {
            "cache_size": len(self.alignment_cache),
            "cache_keys": list(self.alignment_cache.keys()),
        }
