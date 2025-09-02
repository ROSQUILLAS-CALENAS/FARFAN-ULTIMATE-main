"""
Tests for the Deterministic Router system
"""

import hashlib
import json
from unittest.mock import patch

import numpy as np
import pytest

from egw_query_expansion.core.deterministic_router import (
    ActionType,
    ConvexProjector,
    DeterministicRouter,
    ImmutableConfig,
    LexicographicComparator,
    RoutingContext,
    SeedDerivation,
    create_deterministic_router,
)


class TestImmutableConfig:
    """Test the immutable configuration system"""

    def test_config_immutability(self):
        """Test that configuration is immutable after creation"""
        config = ImmutableConfig(projection_tolerance=1e-5)

        # Should not be able to modify
        with pytest.raises(Exception):
            config.projection_tolerance = 1e-4

    def test_config_hashing(self):
        """Test cryptographic hashing of configuration"""
        config1 = ImmutableConfig(projection_tolerance=1e-6)
        config2 = ImmutableConfig(projection_tolerance=1e-6)
        config3 = ImmutableConfig(projection_tolerance=1e-5)

        # Same configs should have same hash
        assert config1.config_hash == config2.config_hash

        # Different configs should have different hashes
        assert config1.config_hash != config3.config_hash

        # Hash should be 64-character SHA256
        assert len(config1.config_hash) == 64
        assert all(c in "0123456789abcdef" for c in config1.config_hash)


class TestRoutingContext:
    """Test routing context creation and immutability"""

    def test_context_creation(self):
        """Test context creation from query data"""
        query = "test query"
        embedding = np.array([0.1, 0.2, 0.3])

        context = RoutingContext.from_query(
            query, embedding, corpus_size=1000, mode="hybrid"
        )

        assert context.query_hash == hashlib.sha256(query.encode()).hexdigest()
        assert context.query_embedding == (0.1, 0.2, 0.3)
        assert context.corpus_size == 1000
        assert context.retrieval_mode == "hybrid"

    def test_context_immutability(self):
        """Test that context is immutable"""
        embedding = np.array([0.1, 0.2, 0.3])
        context = RoutingContext.from_query("test", embedding, 1000, "hybrid")

        # Should not be able to modify
        with pytest.raises(Exception):
            context.corpus_size = 2000


class TestConvexProjector:
    """Test convex projection algorithms"""

    def setup_method(self):
        """Setup test projector"""
        config = ImmutableConfig()
        self.projector = ConvexProjector(config)

    def test_simplex_projection(self):
        """Test projection to probability simplex"""
        # Test case 1: Already valid
        point = np.array([0.3, 0.4, 0.3])
        projected = self.projector.project_to_simplex(point)

        assert np.allclose(projected.sum(), 1.0)
        assert np.all(projected >= 0)

        # Test case 2: Needs scaling
        point = np.array([1.0, 2.0, 3.0])
        projected = self.projector.project_to_simplex(point)

        assert np.allclose(projected.sum(), 1.0)
        assert np.all(projected >= 0)

        # Test case 3: Has negative values
        point = np.array([-0.5, 1.0, 0.8])
        projected = self.projector.project_to_simplex(point)

        assert np.allclose(projected.sum(), 1.0)
        assert np.all(projected >= 0)

    def test_box_projection(self):
        """Test projection to box constraints"""
        point = np.array([-1.0, 0.5, 2.0])
        bounds = (0.0, 1.0)

        projected = self.projector.project_to_box(point, bounds)

        assert np.allclose(projected, [0.0, 0.5, 1.0])

    def test_convergence_metric(self):
        """Test convergence metric computation"""
        current = np.array([1.0, 2.0, 3.0])
        previous = np.array([1.1, 1.9, 3.1])

        metric = self.projector.compute_convergence_metric(current, previous)
        expected = np.linalg.norm(current - previous)

        assert np.allclose(metric, expected)


class TestSeedDerivation:
    """Test deterministic seed derivation"""

    def test_seed_determinism(self):
        """Test that same inputs produce same seeds"""
        embedding = np.array([0.1, 0.2, 0.3])
        context = RoutingContext.from_query("test", embedding, 1000, "hybrid")

        seed1 = SeedDerivation.derive_seed(context, 1, "module1", "hash1")
        seed2 = SeedDerivation.derive_seed(context, 1, "module1", "hash1")

        assert seed1 == seed2

    def test_seed_variation(self):
        """Test that different inputs produce different seeds"""
        embedding = np.array([0.1, 0.2, 0.3])
        context = RoutingContext.from_query("test", embedding, 1000, "hybrid")

        seed1 = SeedDerivation.derive_seed(context, 1, "module1", "hash1")
        seed2 = SeedDerivation.derive_seed(context, 2, "module1", "hash1")
        seed3 = SeedDerivation.derive_seed(context, 1, "module2", "hash1")

        assert seed1 != seed2
        assert seed1 != seed3
        assert seed2 != seed3

    def test_traceability_id_uniqueness(self):
        """Test traceability ID uniqueness"""
        embedding = np.array([0.1, 0.2, 0.3])
        context = RoutingContext.from_query("test", embedding, 1000, "hybrid")

        # Even same context and step should produce different IDs due to timestamp
        id1 = SeedDerivation.generate_traceability_id(context, 1)
        id2 = SeedDerivation.generate_traceability_id(context, 1)

        assert id1 != id2
        assert len(id1) == 16
        assert len(id2) == 16


class TestLexicographicComparator:
    """Test lexicographic comparison for tie breaking"""

    def test_action_priority_ordering(self):
        """Test that actions are ordered by priority"""
        comp = LexicographicComparator()

        result = comp.compare_actions(
            ActionType.ROUTE_TO_SPARSE, ActionType.ROUTE_TO_DENSE, "hash1", "hash2"
        )

        assert result == -1  # SPARSE has lower priority (0) than DENSE (1)

    def test_tie_breaker_by_hash(self):
        """Test tie breaking using context hashes"""
        comp = LexicographicComparator()

        # Same action, different hashes
        result = comp.compare_actions(
            ActionType.ROUTE_TO_SPARSE, ActionType.ROUTE_TO_SPARSE, "aaaa", "bbbb"
        )

        assert result == -1  # "aaaa" < "bbbb"

        # Same action, same hash
        result = comp.compare_actions(
            ActionType.ROUTE_TO_SPARSE, ActionType.ROUTE_TO_SPARSE, "hash1", "hash1"
        )

        assert result == 0  # Equal


class TestDeterministicRouter:
    """Test the main deterministic router"""

    def setup_method(self):
        """Setup test router"""
        self.config = ImmutableConfig(max_iterations=5)
        self.router = DeterministicRouter(self.config)

    def test_deterministic_routing(self):
        """Test that same input produces same route"""
        embedding = np.array([0.1, 0.2, 0.3, 0.4])
        context = RoutingContext.from_query("test query", embedding, 1000, "hybrid")

        # Route multiple times
        decisions1 = self.router.route_query(context)
        self.router.decisions.clear()  # Clear for clean second run
        decisions2 = self.router.route_query(context)

        # Should get identical sequences
        actions1 = [d.action for d in decisions1]
        actions2 = [d.action for d in decisions2]

        assert actions1 == actions2

    def test_different_contexts_different_routes(self):
        """Test that different contexts produce different routes"""
        embedding1 = np.array([0.1, 0.2, 0.3, 0.4])
        embedding2 = np.array([0.4, 0.3, 0.2, 0.1])

        context1 = RoutingContext.from_query("query1", embedding1, 1000, "sparse")
        context2 = RoutingContext.from_query("query2", embedding2, 2000, "dense")

        decisions1 = self.router.route_query(context1)
        decisions2 = self.router.route_query(context2)

        # Should get different sequences
        actions1 = [d.action for d in decisions1]
        actions2 = [d.action for d in decisions2]

        assert actions1 != actions2

    def test_path_reconstruction(self):
        """Test complete path reconstruction from logs"""
        embedding = np.array([0.1, 0.2, 0.3, 0.4])
        context = RoutingContext.from_query("test query", embedding, 1000, "hybrid")

        decisions = self.router.route_query(context)

        # Reconstruct path
        reconstructed = self.router.reconstruct_path(context.query_hash)

        assert len(reconstructed) == len(decisions)

        # Check all decisions are recorded with proper fields
        for i, record in enumerate(reconstructed):
            assert record["step_id"] == i
            assert record["context_hash"] == context.query_hash
            assert "action" in record
            assert "projection_point" in record
            assert "justification" in record
            assert "traceability_id" in record

    def test_determinism_verification(self):
        """Test the built-in determinism verification"""
        embedding = np.array([0.1, 0.2, 0.3, 0.4])
        context = RoutingContext.from_query("test query", embedding, 1000, "hybrid")

        is_deterministic = self.router.verify_determinism(context, num_trials=3)

        assert is_deterministic is True

    def test_routing_statistics(self):
        """Test routing statistics collection"""
        embedding = np.array([0.1, 0.2, 0.3, 0.4])
        context = RoutingContext.from_query("test query", embedding, 1000, "hybrid")

        decisions = self.router.route_query(context)
        stats = self.router.get_routing_statistics()

        assert "total_decisions" in stats
        assert stats["total_decisions"] == len(decisions)
        assert "action_distribution" in stats
        assert "average_convergence" in stats
        assert "tie_breaker_usage" in stats

    def test_cached_weight_computation(self):
        """Test that weight computation is properly cached"""
        embedding = np.array([0.1, 0.2, 0.3, 0.4])
        context = RoutingContext.from_query("test query", embedding, 1000, "hybrid")

        # Should be able to call multiple times with same result
        weights1 = self.router._compute_routing_weights(
            context.query_hash, context.corpus_size, context.retrieval_mode
        )
        weights2 = self.router._compute_routing_weights(
            context.query_hash, context.corpus_size, context.retrieval_mode
        )

        assert weights1 == weights2


class TestFactoryFunction:
    """Test the factory function for router creation"""

    def test_default_creation(self):
        """Test creating router with default config"""
        router = create_deterministic_router()

        assert isinstance(router, DeterministicRouter)
        assert isinstance(router.config, ImmutableConfig)

    def test_custom_config_creation(self):
        """Test creating router with custom config"""
        config_dict = {"projection_tolerance": 1e-5, "max_iterations": 500}

        router = create_deterministic_router(config_dict)

        assert router.config.projection_tolerance == 1e-5
        assert router.config.max_iterations == 500


# Integration tests
class TestIntegration:
    """Integration tests for the complete system"""

    def test_end_to_end_routing(self):
        """Test complete end-to-end routing workflow"""
        # Create router
        router = create_deterministic_router(
            {"max_iterations": 3, "convergence_threshold": 1e-6}
        )

        # Create context
        query = "What is machine learning?"
        embedding = np.random.rand(128)  # Typical embedding size
        context = RoutingContext.from_query(query, embedding, 50000, "hybrid")

        # Route query
        decisions = router.route_query(context)

        # Verify results
        assert len(decisions) > 0
        assert all(isinstance(d.action, ActionType) for d in decisions)
        assert decisions[-1].action == ActionType.VALIDATE_OUTPUT

        # Verify determinism
        assert router.verify_determinism(context, num_trials=2)

        # Verify path reconstruction
        reconstructed = router.reconstruct_path(context.query_hash)
        assert len(reconstructed) == len(decisions)

        # Verify statistics
        stats = router.get_routing_statistics()
        assert stats["total_decisions"] > 0

    def test_multiple_query_routing(self):
        """Test routing multiple different queries"""
        router = create_deterministic_router({"max_iterations": 3})

        queries = [
            ("machine learning basics", "sparse"),
            ("deep neural networks", "dense"),
            ("natural language processing", "colbert"),
            ("computer vision applications", "hybrid"),
        ]

        all_decisions = []

        for query, mode in queries:
            embedding = np.random.rand(64)
            context = RoutingContext.from_query(query, embedding, 10000, mode)
            decisions = router.route_query(context)
            all_decisions.extend(decisions)

        # Verify each query has unique routing decisions
        query_hashes = set(d.context_hash for d in all_decisions)
        assert len(query_hashes) == len(queries)

        # Verify statistics cover all decisions
        stats = router.get_routing_statistics()
        assert stats["total_decisions"] == len(all_decisions)
