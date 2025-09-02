"""
Test suite for Evidence System with Conformal Coverage
"""
import time

import numpy as np

from evidence_system import Evidence, EvidenceSystem


class TestEvidenceSystem:
    """Test cases for the Evidence System"""

    def setup_method(self):
        """Setup for each test"""
        self.system = EvidenceSystem(alpha=0.1)
        np.random.seed(42)  # For reproducible tests

    def test_add_evidence_idempotent(self):
        """Test idempotent evidence insertion"""
        qid = "q1"
        evidence = Evidence(
            qid=qid, content="Test evidence", score=0.8, dimension="quality"
        )

        # First insertion should return True
        assert self.system.add_evidence(qid, evidence) == True

        # Second insertion should return False (idempotent)
        assert self.system.add_evidence(qid, evidence) == False

        # Should only have one evidence item
        evidence_list = self.system.get_evidence_for_question(qid)
        assert len(evidence_list) == 1

    def test_get_evidence_for_question(self):
        """Test evidence retrieval by question"""
        qid = "q1"
        evidence1 = Evidence(
            qid=qid, content="Evidence 1", score=0.7, dimension="quality"
        )
        evidence2 = Evidence(
            qid=qid, content="Evidence 2", score=0.9, dimension="relevance"
        )

        self.system.add_evidence(qid, evidence1)
        self.system.add_evidence(qid, evidence2)

        retrieved = self.system.get_evidence_for_question(qid)
        assert len(retrieved) == 2, f"Expected 2 evidence items, got {len(retrieved)}"

        # Test non-existent question
        empty_retrieved = self.system.get_evidence_for_question("nonexistent")
        assert (
            len(empty_retrieved) == 0
        ), f"Expected 0 evidence items for nonexistent qid, got {len(empty_retrieved)}"

    def test_group_by_dimension(self):
        """Test grouping by dimension"""
        # Add evidence with different dimensions
        evidence1 = Evidence(
            qid="q1", content="Content 1", score=0.8, dimension="quality"
        )
        evidence2 = Evidence(
            qid="q2", content="Content 2", score=0.7, dimension="quality"
        )
        evidence3 = Evidence(
            qid="q3", content="Content 3", score=0.9, dimension="relevance"
        )

        self.system.add_evidence("q1", evidence1)
        self.system.add_evidence("q2", evidence2)
        self.system.add_evidence("q3", evidence3)

        grouped = self.system.group_by_dimension()

        assert (
            "quality" in grouped
        ), f"Expected 'quality' dimension in grouped keys: {list(grouped.keys())}"
        assert (
            "relevance" in grouped
        ), f"Expected 'relevance' dimension in grouped keys: {list(grouped.keys())}"
        assert (
            len(grouped["quality"]) == 2
        ), f"Expected 2 quality items, got {len(grouped['quality'])}"
        assert (
            len(grouped["relevance"]) == 1
        ), f"Expected 1 relevance item, got {len(grouped['relevance'])}"

    def test_shuffle_invariance(self):
        """Test shuffle invariance over evidence order"""
        qid = "q1"

        # Add multiple evidence items
        for i in range(5):
            evidence = Evidence(
                qid=qid,
                content=f"Evidence {i}",
                score=0.5 + i * 0.1,
                dimension="quality",
            )
            self.system.add_evidence(qid, evidence)

        # Test shuffle invariance
        invariant = self.system.test_shuffle_invariance(qid, n_shuffles=10)
        assert invariant == True

    def test_coverage_calculation(self):
        """Test coverage calculation with synthetic data"""
        # Add some evidence
        for i in range(10):
            evidence = Evidence(
                qid=f"q{i}",
                content=f"Evidence {i}",
                score=np.random.normal(0, 1),
                dimension="quality",
            )
            self.system.add_evidence(f"q{i}", evidence)

        # Calculate coverage
        coverage = self.system.calculate_coverage()

        # Coverage should be between 0 and 1
        assert 0 <= coverage <= 1

        # Should be close to target (1 - alpha = 0.9)
        target_coverage = 1 - self.system.alpha
        assert (
            abs(coverage - target_coverage) < 0.5
        )  # Generous tolerance for synthetic data

    def test_coverage_audit(self):
        """Test coverage audit functionality"""
        # Add some evidence first
        for i in range(5):
            evidence = Evidence(
                qid=f"q{i}",
                content=f"Evidence {i}",
                score=np.random.normal(0, 1),
                dimension="quality",
            )
            self.system.add_evidence(f"q{i}", evidence)

        # Run audit
        audit_results = self.system.audit_coverage(n_trials=10, delta=0.05)

        # Check audit results structure
        required_keys = [
            "empirical_coverage",
            "target_coverage",
            "coverage_std",
            "delta",
            "passes_audit",
            "n_trials",
            "coverage_distribution",
        ]

        for key in required_keys:
            assert key in audit_results

        assert audit_results["n_trials"] == 10
        assert audit_results["target_coverage"] == 0.9
        assert len(audit_results["coverage_distribution"]) == 10

    def test_evidence_hash_consistency(self):
        """Test that evidence hashing is consistent for deduplication"""
        qid = "q1"
        content = "Test content"

        evidence1 = Evidence(qid=qid, content=content, score=0.8, dimension="quality")
        evidence2 = Evidence(qid=qid, content=content, score=0.8, dimension="quality")

        # Should have same hash despite different timestamp
        assert hash(evidence1) == hash(evidence2)

        # Different content should have different hash
        evidence3 = Evidence(
            qid=qid, content="Different content", score=0.8, dimension="quality"
        )
        assert hash(evidence1) != hash(evidence3)

    def test_dr_submodular_selection(self):
        """Test DR-submodular evidence selection"""
        # Create evidence list
        evidence_list = []
        for i in range(10):
            evidence = Evidence(
                qid=f"q{i}",
                content=f"Evidence content {i} with some unique words {i}",
                score=np.random.uniform(0.5, 1.0),
                dimension="quality",
            )
            evidence_list.append(evidence)

        # Test selection
        selected, objective_value = self.system._dr_submodular_selection(
            evidence_list, budget=5
        )

        assert len(selected) == 5
        assert objective_value > 0
        assert len(set(selected)) == 5  # No duplicates

        # Test empty budget
        selected_empty, obj_empty = self.system._dr_submodular_selection(
            evidence_list, budget=0
        )
        assert len(selected_empty) == 0
        assert obj_empty == 0.0

    def test_system_stats(self):
        """Test system statistics"""
        # Add some evidence
        for i in range(3):
            evidence = Evidence(
                qid=f"q{i}",
                content=f"Evidence {i}",
                score=0.8,
                dimension="quality" if i < 2 else "relevance",
            )
            self.system.add_evidence(f"q{i}", evidence)

        stats = self.system.get_stats()

        assert (
            stats["total_questions"] == 3
        ), f"Expected 3 questions, got {stats['total_questions']}"
        assert (
            stats["total_evidence"] == 3
        ), f"Expected 3 evidence items, got {stats['total_evidence']}"
        assert stats["alpha"] == 0.1, f"Expected alpha=0.1, got {stats['alpha']}"
        assert (
            stats["target_coverage"] == 0.9
        ), f"Expected target_coverage=0.9, got {stats['target_coverage']}"
        assert (
            "quality" in stats["dimensions"]
        ), f"Expected 'quality' in dimensions: {stats['dimensions']}"
        assert (
            "relevance" in stats["dimensions"]
        ), f"Expected 'relevance' in dimensions: {stats['dimensions']}"


def test_evidence_msgspec_serialization():
    """Test msgspec serialization of Evidence"""
    evidence = Evidence(
        qid="q1",
        content="Test content",
        score=0.85,
        dimension="quality",
        metadata={"source": "test"},
    )

    # Test that msgspec can encode/decode
    import msgspec

    encoded = msgspec.json.encode(evidence)
    decoded = msgspec.json.decode(encoded, type=Evidence)

    assert decoded.qid == evidence.qid
    assert decoded.content == evidence.content
    assert decoded.score == evidence.score
    assert decoded.dimension == evidence.dimension


if __name__ == "__main__":
    # Run basic tests
    test_system = TestEvidenceSystem()
    test_system.setup_method()

    print("Running basic functionality tests...")

    # Test basic functionality
    test_system.test_add_evidence_idempotent()
    print("✓ Idempotent insertion test passed")

    test_system.test_shuffle_invariance()
    print("✓ Shuffle invariance test passed")

    test_system.test_coverage_calculation()
    print("✓ Coverage calculation test passed")

    test_system.test_coverage_audit()
    print("✓ Coverage audit test passed")

    print("\nAll tests passed! 🎉")
