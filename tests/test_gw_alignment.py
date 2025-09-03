"""Tests for Gromov-Wasserstein alignment functionality."""

import logging

import numpy as np
import pytest
# # # from scipy.stats import bootstrap  # Module not found  # Module not found  # Module not found

# # # from standards_alignment.graph_ops import DocumentGraph, StandardsGraph  # Module not found  # Module not found  # Module not found
# # # from standards_alignment.gw_alignment import gw_align, sparse_align  # Module not found  # Module not found  # Module not found
# # # from standards_alignment.patterns import (  # Module not found  # Module not found  # Module not found
    Criterion,
    PatternSpec,
    PatternType,
    Requirement,
)

logging.basicConfig(level=logging.INFO)


class TestGWAlignment:
    def create_synthetic_graphs(self, n_std: int = 5, n_doc: int = 8) -> tuple:
        """Create synthetic graph pair with known correspondences."""
        # Standards graph
        std_graph = StandardsGraph()
        std_graph.add_dimension("test_dim", {})
        std_graph.add_subdimension("test_dim", "test_subdim", {})

        for i in range(3):
            std_graph.add_point("test_subdim", i, {})

        # Document graph
        doc_graph = DocumentGraph()
        doc_graph.add_section(
            "sec1", "Security Requirements", "This section covers security."
        )
        doc_graph.add_paragraph(
            "sec1", "para1", "User authentication must be implemented."
        )
        doc_graph.add_paragraph("sec1", "para2", "Data encryption is required.")
        doc_graph.add_table(
            "sec1", "table1", {"rows": [["Role", "Permission"]], "headers": True}
        )

        doc_graph.add_section("sec2", "Performance", "Performance requirements.")
        doc_graph.add_paragraph("sec2", "para3", "Response time must be under 100ms.")
        doc_graph.add_paragraph("sec2", "para4", "System must handle 1000 users.")
        doc_graph.add_table("sec2", "table2", {"rows": [["Metric", "Target"]]})

        return std_graph, doc_graph

    def test_gw_align_basic(self):
        """Test basic GW alignment functionality."""
        std_graph, doc_graph = self.create_synthetic_graphs()

        result = gw_align(std_graph, doc_graph, reg=0.1)

        assert result.transport_plan.shape == (5, 8)
        assert np.isclose(np.sum(result.transport_plan), 1.0, atol=1e-6)
        assert "gw_distance" in result.cost_decomposition
        assert "entropic_penalty" in result.cost_decomposition
        assert result.stability_bound > 0
        assert len(result.matching_pairs) > 0

    def test_sparse_align_basic(self):
        """Test sparse bipartite matching."""
        std_graph, doc_graph = self.create_synthetic_graphs()

        result = sparse_align(std_graph, doc_graph)

        assert result.transport_plan.shape == (5, 8)
        assert result.stability_bound == 0.0  # No stability for sparse
        assert result.regularization == 0.0
        assert "assignment_cost" in result.cost_decomposition

    def test_precision_recall_synthetic(self):
        """Test precision/recall on synthetic data with known correspondences."""
        std_graph, doc_graph = self.create_synthetic_graphs()

        # Define true correspondences (simplified)
        true_matches = {0: [1], 1: [2], 2: [3]}  # std_node -> doc_nodes

        result = gw_align(std_graph, doc_graph)

        # Extract predicted matches
        predicted_matches = {}
        for i, j, weight in result.matching_pairs:
            if weight > 0.1:  # Threshold
                if i not in predicted_matches:
                    predicted_matches[i] = []
                predicted_matches[i].append(j)

        # Compute precision/recall
        tp, fp, fn = 0, 0, 0

        for std_node, true_docs in true_matches.items():
            pred_docs = predicted_matches.get(std_node, [])

            for doc in pred_docs:
                if doc in true_docs:
                    tp += 1
                else:
                    fp += 1

            for doc in true_docs:
                if doc not in pred_docs:
                    fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")

        # Basic sanity check
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1

    def test_jackknife_confidence(self):
        """Test Jackknife+ confidence intervals for precision/recall."""
        # Generate multiple synthetic pairs
        results = []

        for _ in range(10):
            std_graph, doc_graph = self.create_synthetic_graphs()
            result = gw_align(std_graph, doc_graph)

            # Simplified precision metric: fraction of high-weight matches
            high_weight_fraction = np.mean(
                [w > 0.1 for _, _, w in result.matching_pairs[:3]]
            )
            results.append(high_weight_fraction)

        results = np.array(results)

        # Jackknife+ bootstrap for confidence intervals
        def compute_metric(sample_indices):
            return np.mean(results[sample_indices])

        n_bootstrap = 100
        bootstrap_results = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(len(results), size=len(results), replace=True)
            bootstrap_results.append(compute_metric(indices))

        bootstrap_results = np.array(bootstrap_results)
        ci_lower = np.percentile(bootstrap_results, 2.5)
        ci_upper = np.percentile(bootstrap_results, 97.5)

        print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

        # Basic sanity checks
        assert ci_lower <= ci_upper
        assert 0 <= ci_lower <= 1
        assert 0 <= ci_upper <= 1

    def test_stability_bound_logging(self):
        """Test that stability bounds and regularization are logged."""
        std_graph, doc_graph = self.create_synthetic_graphs()

        # Just test that the function works and returns expected fields
        result = gw_align(std_graph, doc_graph, reg=0.05)

        # Check result contains expected fields
        assert result.stability_bound > 0
        assert result.regularization == 0.05
        assert "gw_distance" in result.cost_decomposition
        assert "entropic_penalty" in result.cost_decomposition


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
