"""Tests for stable EGW aligner."""

import tempfile
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

import networkx as nx
import numpy as np
import pytest

# # # from standards_alignment.stable_gw_aligner import (  # Module not found  # Module not found  # Module not found
    StableEGWAligner,
    StableGWConfig,
    TransportPlan,
)


@pytest.fixture
def simple_standards_graph():
    """Create a simple standards graph for testing."""
    G = nx.Graph()
    G.add_node(
        "dim1",
        node_type="dimension",
        title="Dimension 1",
        content="First dimension content",
    )
    G.add_node(
        "subdim1",
        node_type="subdimension",
        title="Subdimension 1",
        content="First subdimension content",
    )
    G.add_node(
        "point1", node_type="point", title="Point 1", content="First point content"
    )

    G.add_edges_from([("dim1", "subdim1"), ("subdim1", "point1")])
    return G


@pytest.fixture
def simple_document_graph():
    """Create a simple document graph for testing."""
    G = nx.Graph()
    G.add_node(
        "sec1",
        node_type="section",
        title="Section 1",
        content="Section content about dimensions",
    )
    G.add_node(
        "para1",
        node_type="paragraph",
        title="Paragraph 1",
        content="Paragraph content about first point",
    )
    G.add_node(
        "table1", node_type="table", title="Table 1", content="Table with point data"
    )

    G.add_edges_from([("sec1", "para1"), ("para1", "table1")])
    return G


@pytest.fixture
def stable_config():
    """Create stable configuration for testing."""
    return StableGWConfig(
        epsilon=0.1,
        lambda_struct=0.3,
        lambda_content=0.7,
        max_iter=50,
        convergence_threshold=5,
    )


class TestStableGWConfig:
    """Test stable configuration."""

    def test_frozen_config(self):
        """Test that config is frozen and immutable."""
        config = StableGWConfig(epsilon=0.05)

        with pytest.raises(AttributeError):
            config.epsilon = 0.1  # Should raise error as dataclass is frozen


class TestTransportPlan:
    """Test transport plan serialization."""

    def test_transport_plan_serialization(
        self, simple_standards_graph, simple_document_graph
    ):
        """Test complete serialization cycle."""
        aligner = StableEGWAligner(StableGWConfig(max_iter=10))
        plan = aligner.align_graphs(simple_standards_graph, simple_document_graph)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "transport_plan.json"

            # Save plan
            plan.save(filepath)
            assert filepath.exists()

            # Load plan
            loaded_plan = TransportPlan.load(filepath)

            # Verify identical data
            np.testing.assert_array_equal(
                plan.coupling_matrix, loaded_plan.coupling_matrix
            )
            np.testing.assert_array_equal(
                plan.source_marginals, loaded_plan.source_marginals
            )
            np.testing.assert_array_equal(
                plan.target_marginals, loaded_plan.target_marginals
            )
            assert plan.cost_decomposition == loaded_plan.cost_decomposition
            assert plan.stability_metrics == loaded_plan.stability_metrics


class TestStableEGWAligner:
    """Test stable EGW aligner."""

    def test_deterministic_alignment(
        self, simple_standards_graph, simple_document_graph, stable_config
    ):
        """Test that identical inputs produce identical outputs."""
        aligner = StableEGWAligner(stable_config)

        # Run alignment twice
        plan1 = aligner.align_graphs(simple_standards_graph, simple_document_graph)
        plan2 = aligner.align_graphs(simple_standards_graph, simple_document_graph)

        # Should be identical (cached)
        assert np.array_equal(plan1.coupling_matrix, plan2.coupling_matrix)
        assert plan1.cost_decomposition == plan2.cost_decomposition

    def test_reproducibility_with_different_aligner_instances(
        self, simple_standards_graph, simple_document_graph, stable_config
    ):
        """Test reproducibility across different aligner instances."""
        aligner1 = StableEGWAligner(stable_config)
        aligner2 = StableEGWAligner(stable_config)

        plan1 = aligner1.align_graphs(simple_standards_graph, simple_document_graph)
        plan2 = aligner2.align_graphs(simple_standards_graph, simple_document_graph)

        # Should produce very similar results (within numerical tolerance)
        np.testing.assert_array_almost_equal(
            plan1.coupling_matrix, plan2.coupling_matrix, decimal=6
        )

    def test_unmatched_mass_quantification(
        self, simple_standards_graph, simple_document_graph
    ):
        """Test explicit unmatched mass handling."""
        aligner = StableEGWAligner(StableGWConfig(max_iter=10))
        plan = aligner.align_graphs(simple_standards_graph, simple_document_graph)

        # Check unmatched mass info is populated
        assert "unmatched_source_mass" in plan.unmatched_mass_info
        assert "unmatched_target_mass" in plan.unmatched_mass_info
        assert "coverage_gap" in plan.unmatched_mass_info
        assert "coverage_adequate" in plan.unmatched_mass_info

        # Values should be reasonable
        assert 0 <= plan.unmatched_mass_info["coverage_gap"] <= 1

    def test_cost_decomposition_auditability(
        self, simple_standards_graph, simple_document_graph
    ):
        """Test complete cost decomposition for auditability."""
        aligner = StableEGWAligner(StableGWConfig(max_iter=10))
        plan = aligner.align_graphs(simple_standards_graph, simple_document_graph)

        # Check cost decomposition completeness
        required_costs = ["gw_cost", "content_cost", "entropic_cost", "total_cost"]
        for cost_type in required_costs:
            assert cost_type in plan.cost_decomposition
            assert isinstance(plan.cost_decomposition[cost_type], (int, float))

    def test_stability_metrics(
        self, simple_standards_graph, simple_document_graph, stable_config
    ):
        """Test stability metrics are tracked."""
        aligner = StableEGWAligner(stable_config)
        plan = aligner.align_graphs(simple_standards_graph, simple_document_graph)

        # Check stability metrics
        required_metrics = ["epsilon", "lambda_struct", "lambda_content", "cache_key"]
        for metric in required_metrics:
            assert metric in plan.stability_metrics

        # Verify frozen hyperparameters are recorded
        assert plan.stability_metrics["epsilon"] == stable_config.epsilon
        assert plan.stability_metrics["lambda_struct"] == stable_config.lambda_struct
        assert plan.stability_metrics["lambda_content"] == stable_config.lambda_content

    def test_convergence_tracking(self, simple_standards_graph, simple_document_graph):
        """Test deterministic convergence tracking."""
        aligner = StableEGWAligner(StableGWConfig(max_iter=20, convergence_threshold=3))
        plan = aligner.align_graphs(simple_standards_graph, simple_document_graph)

        # Check convergence info
        assert "converged" in plan.convergence_info
        assert "egw_iterations" in plan.convergence_info
        assert isinstance(plan.convergence_info["converged"], bool)
        assert plan.convergence_info["egw_iterations"] > 0

    def test_graph_hashing_consistency(self):
        """Test that graph hashing is deterministic."""
        G1 = nx.Graph()
        G1.add_node("a", content="test1")
        G1.add_node("b", content="test2")
        G1.add_edge("a", "b")

        # Same graph with different node addition order
        G2 = nx.Graph()
        G2.add_node("b", content="test2")
        G2.add_node("a", content="test1")
        G2.add_edge("a", "b")

        aligner = StableEGWAligner()
        hash1 = aligner._compute_graph_hash(G1)
        hash2 = aligner._compute_graph_hash(G2)

        assert hash1 == hash2  # Should be identical despite different creation order

    def test_transport_plan_properties(
        self, simple_standards_graph, simple_document_graph
    ):
        """Test transport plan mathematical properties."""
        aligner = StableEGWAligner(StableGWConfig(max_iter=10))
        plan = aligner.align_graphs(simple_standards_graph, simple_document_graph)

        T = plan.coupling_matrix

        # Check non-negativity
        assert np.all(T >= 0)

        # Check marginal constraints (approximately)
        source_marginals = np.sum(T, axis=1)
        target_marginals = np.sum(T, axis=0)

        # Should approximately sum to uniform distributions
        expected_source = np.ones(len(simple_standards_graph.nodes)) / len(
            simple_standards_graph.nodes
        )
        expected_target = np.ones(len(simple_document_graph.nodes)) / len(
            simple_document_graph.nodes
        )

        np.testing.assert_array_almost_equal(
            source_marginals, expected_source, decimal=3
        )
        np.testing.assert_array_almost_equal(
            target_marginals, expected_target, decimal=3
        )

    def test_cache_functionality(self, simple_standards_graph, simple_document_graph):
        """Test alignment caching system."""
        aligner = StableEGWAligner(StableGWConfig(max_iter=5))

        # Initial cache should be empty
        stats = aligner.get_cache_statistics()
        assert stats["cache_size"] == 0

        # Run alignment
        plan = aligner.align_graphs(simple_standards_graph, simple_document_graph)

        # Cache should now have one entry
        stats = aligner.get_cache_statistics()
        assert stats["cache_size"] == 1
        assert len(stats["cache_keys"]) == 1
