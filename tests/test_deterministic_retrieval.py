"""
Tests for Deterministic Hybrid Retrieval System
"""

import numpy as np
import pytest

# # # from deterministic_hybrid_retrieval import (  # Module not found  # Module not found  # Module not found
    DeterministicHybridRetriever,
    DNPConstraint,
    DocumentHash,
    IndexSnapshot,
    OptimalRankingFusion,
    QueryExpansion,
)


class TestDeterministicHybridRetrieval:
    def setup_method(self):
        """Setup test fixtures"""
        self.retriever = DeterministicHybridRetriever(
            embedding_model_name="intfloat/e5-base-v2", device="cpu", seed=42
        )

        self.test_documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing deals with text understanding.",
            "Computer vision focuses on image and video analysis.",
            "Reinforcement learning learns through trial and error.",
        ]

        self.test_doc_ids = [f"doc_{i}" for i in range(len(self.test_documents))]

    def test_document_hash_creation(self):
        """Test deterministic document hashing"""
        doc_hash1 = DocumentHash.from_content("test content", "doc1")
        doc_hash2 = DocumentHash.from_content("test content", "doc2")  # Different ID
        doc_hash3 = DocumentHash.from_content(
            "test content", "doc1"
        )  # Same content and ID

        # Same content should produce same hash regardless of doc_id for content_hash
        assert doc_hash1.content_hash == doc_hash2.content_hash
        assert doc_hash1.content_hash == doc_hash3.content_hash

        # But different doc_ids should be preserved
        assert doc_hash1.doc_id != doc_hash2.doc_id
        assert doc_hash1.doc_id == doc_hash3.doc_id

    def test_index_snapshot_creation(self):
        """Test snapshot creation and determinism"""
        snapshot1 = IndexSnapshot.create(self.test_documents, 384, 5000)
        snapshot2 = IndexSnapshot.create(self.test_documents, 384, 5000)

        # Same documents should produce same snapshot
        assert snapshot1.documents_hash == snapshot2.documents_hash
        assert snapshot1.snapshot_id == snapshot2.snapshot_id

        # Different documents should produce different snapshots
        different_docs = self.test_documents + ["Additional document"]
        snapshot3 = IndexSnapshot.create(different_docs, 384, 5000)
        assert snapshot1.snapshot_id != snapshot3.snapshot_id

    def test_query_expansion(self):
        """Test deterministic query expansion"""
        expansion_config = QueryExpansion(
            expansion_weights=(0.8, 0.6, 0.4), expansion_radius=0.5, max_expansions=3
        )

        query = "machine learning algorithms"

        # Multiple calls should produce same expansions
        expansions1 = expansion_config.expand_query(query, seed=42)
        expansions2 = expansion_config.expand_query(query, seed=42)

        assert expansions1 == expansions2
        assert len(expansions1) <= expansion_config.max_expansions
        assert query in expansions1  # Original query should be included

    def test_dnp_constraints(self):
        """Test DNP constraint filtering"""
        constraint = DNPConstraint(field="category", operator="eq", value="private")

        # Test document that should be filtered
        private_metadata = {"category": "private", "author": "test"}
        assert constraint.applies_to_document(private_metadata) == True

        # Test document that should not be filtered
        public_metadata = {"category": "public", "author": "test"}
        assert constraint.applies_to_document(public_metadata) == False

        # Test document without the field
        no_category_metadata = {"author": "test"}
        assert constraint.applies_to_document(no_category_metadata) == False

    def test_optimal_ranking_fusion(self):
        """Test optimal ranking fusion determinism"""
        fusion = OptimalRankingFusion(alpha=0.4, beta=0.4)

        # Create test results
        doc_hashes = [
            DocumentHash.from_content(doc, f"doc_{i}")
            for i, doc in enumerate(self.test_documents)
        ]

        sparse_results = [(doc_hashes[0], 0.8), (doc_hashes[1], 0.6)]
        dense_results = [(doc_hashes[0], 0.7), (doc_hashes[2], 0.9)]
        projection_results = [(doc_hashes[1], 0.5), (doc_hashes[2], 0.3)]

        # Test fusion multiple times
        fusion1 = fusion.fuse_rankings(
            sparse_results, dense_results, projection_results
        )
        fusion2 = fusion.fuse_rankings(
            sparse_results, dense_results, projection_results
        )

        # Should produce identical results
        assert len(fusion1) == len(fusion2)
        for f1, f2 in zip(fusion1, fusion2):
            assert f1.doc_hash == f2.doc_hash
            assert abs(f1.fusion_score - f2.fusion_score) < 1e-10
            assert f1.rank == f2.rank

    def test_document_addition_and_deduplication(self):
        """Test document addition and deduplication"""
        # Add initial documents
        snapshot1 = self.retriever.add_documents(self.test_documents, self.test_doc_ids)

        # Add same documents again (should deduplicate)
        snapshot2 = self.retriever.add_documents(self.test_documents, self.test_doc_ids)

        # Snapshot should be same (no new unique documents)
        assert snapshot1.snapshot_id == snapshot2.snapshot_id
        assert len(self.retriever.documents) == len(self.test_documents)

        # Add new document
        new_docs = ["This is a completely new document."]
        new_ids = ["new_doc_1"]
        snapshot3 = self.retriever.add_documents(new_docs, new_ids)

        # Should create new snapshot
        assert snapshot3.snapshot_id != snapshot1.snapshot_id
        assert len(self.retriever.documents) == len(self.test_documents) + 1

    def test_search_determinism(self):
        """Test that search results are deterministic"""
        # Add documents
        self.retriever.add_documents(self.test_documents, self.test_doc_ids)

        query = "machine learning neural networks"

        # Perform multiple searches
        results = []
        for _ in range(3):
            result = self.retriever.search(query, top_k=3)
            results.append(result)

        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result.doc_hashes == first_result.doc_hashes
            assert result.scores == first_result.scores
            assert result.method == first_result.method
            assert result.snapshot_id == first_result.snapshot_id

    def test_search_with_expansion(self):
        """Test search with query expansion"""
        # Add documents
        self.retriever.add_documents(self.test_documents, self.test_doc_ids)

        expansion_config = QueryExpansion(
            expansion_weights=(0.8, 0.6), expansion_radius=0.5, max_expansions=2
        )

        query = "machine learning"

        # Search with expansion
        result1 = self.retriever.search(
            query, top_k=3, expansion_config=expansion_config
        )
        result2 = self.retriever.search(
            query, top_k=3, expansion_config=expansion_config
        )

        # Results should be deterministic
        assert result1.doc_hashes == result2.doc_hashes
        assert result1.scores == result2.scores

    def test_search_with_dnp_constraints(self):
        """Test search with DNP constraints"""
        # Add documents with metadata
        metadata = [
            {"category": "public", "level": "beginner"},
            {"category": "private", "level": "advanced"},
            {"category": "public", "level": "intermediate"},
            {"category": "public", "level": "beginner"},
            {"category": "private", "level": "expert"},
        ]

        self.retriever.add_documents(self.test_documents, self.test_doc_ids, metadata)

        # Create DNP constraint to filter out private documents
        dnp_constraint = DNPConstraint(field="category", operator="eq", value="private")

        query = "machine learning"

        # Search without constraints
        result_no_constraints = self.retriever.search(query, top_k=5)

        # Search with constraints
        result_with_constraints = self.retriever.search(
            query, top_k=5, dnp_constraints=[dnp_constraint]
        )

        # Should have fewer results due to filtering
        assert len(result_with_constraints.doc_hashes) <= len(
            result_no_constraints.doc_hashes
        )

        # Verify no private documents in constrained results
        for doc_hash in result_with_constraints.doc_hashes:
            doc_metadata = self.retriever.doc_metadata.get(doc_hash.content_hash, {})
            assert doc_metadata.get("category") != "private"

    def test_verify_determinism_method(self):
        """Test the built-in determinism verification"""
        # Add documents
        self.retriever.add_documents(self.test_documents, self.test_doc_ids)

        # Verify determinism
        is_deterministic = self.retriever.verify_determinism(
            "machine learning", num_trials=5
        )
        assert is_deterministic == True

    def test_reproducibility_info(self):
        """Test reproducibility information extraction"""
        # Add documents
        self.retriever.add_documents(self.test_documents, self.test_doc_ids)

        info = self.retriever.get_reproducibility_info()

        # Check required fields are present
        assert "seed" in info
        assert "model_name" in info
        assert "fusion_weights" in info
        assert "current_snapshot" in info
        assert "num_documents" in info

        # Check values
        assert info["seed"] == 42
        assert info["model_name"] == "intfloat/e5-base-v2"
        assert info["num_documents"] == len(self.test_documents)

    def test_empty_index_search(self):
        """Test search on empty index"""
        with pytest.raises(ValueError, match="No documents indexed"):
            self.retriever.search("test query")

    def test_score_normalization(self):
        """Test that score normalization is deterministic"""
        fusion = OptimalRankingFusion()

        # Test with known scores
        scores = [0.9, 0.7, 0.5, 0.3, 0.1]
        normalized1 = fusion._normalize_scores(scores)
        normalized2 = fusion._normalize_scores(scores)

        # Should be identical
        assert normalized1 == normalized2

        # Should be in [0, 1] range
        assert all(0 <= score <= 1 for score in normalized1)
        assert normalized1[0] == 1.0  # Max should be 1
        assert normalized1[-1] == 0.0  # Min should be 0

    def test_learned_projection_determinism(self):
        """Test that learned projections are deterministic"""
# # #         from deterministic_hybrid_retrieval import LearnedProjection  # Module not found  # Module not found  # Module not found

        # Create two projections with same seed
        proj1 = LearnedProjection(384, 100, seed=42)
        proj2 = LearnedProjection(384, 100, seed=42)

        # Test input
        test_input = np.random.randn(384).astype(np.float32)

        # Project with both
        output1 = proj1.project(test_input)
        output2 = proj2.project(test_input)

        # Should be identical
        np.testing.assert_array_equal(output1, output2)


if __name__ == "__main__":
    # Run a simple test
    test_instance = TestDeterministicHybridRetrieval()
    test_instance.setup_method()
    test_instance.test_document_hash_creation()
    print("Basic test passed!")
