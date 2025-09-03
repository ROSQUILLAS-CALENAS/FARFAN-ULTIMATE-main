"""
Comprehensive tests for tamper-evident lineage tracking system
"""
import json
import os
import tempfile
import time
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

import pytest

# # # from lineage_tracker import (  # Module not found  # Module not found  # Module not found
    AuditTrail,
    ConsistencyProof,
    EventType,
    InclusionProof,
    LineageEvent,
    LineageTracker,
    MerkleTree,
)


class TestMerkleTree:
    """Test Merkle tree implementation"""

    def test_single_leaf(self):
        tree = MerkleTree(["hash1"])
        assert tree.root_hash == "hash1"

    def test_two_leaves(self):
        tree = MerkleTree(["hash1", "hash2"])
        assert tree.root_hash != "hash1"
        assert tree.root_hash != "hash2"
        assert len(tree.root_hash) > 0

    def test_inclusion_proof(self):
        hashes = ["leaf1", "leaf2", "leaf3", "leaf4"]
        tree = MerkleTree(hashes)

        # Test inclusion proof for first leaf
        proof = tree.get_inclusion_proof("leaf1")
        assert isinstance(proof, list)

        # Verify proof manually
        inclusion_proof = InclusionProof(
            leaf_hash="leaf1",
            merkle_root=tree.root_hash,
            proof_path=proof,
            trace_id="test",
            event_id="test_event",
        )
        assert inclusion_proof.verify()

    def test_empty_tree(self):
        tree = MerkleTree([])
        assert tree.root_hash == ""


class TestLineageEvent:
    """Test LineageEvent functionality"""

    def test_event_creation(self):
        event = LineageEvent(
            trace_id="trace_123",
            event_id="event_456",
            event_type=EventType.EVIDENCE_ADDED,
            timestamp=time.time(),
            previous_hash="prev_hash",
            data_hash="data_hash",
            metadata={"source": "test"},
        )
        assert event.trace_id == "trace_123"
        assert event.event_type == EventType.EVIDENCE_ADDED

    def test_hash_computation(self):
        event = LineageEvent(
            trace_id="trace_123",
            event_id="event_456",
            event_type=EventType.EVIDENCE_ADDED,
            timestamp=1234567890.0,
            previous_hash="prev_hash",
            data_hash="data_hash",
            metadata={"key": "value"},
        )

        hash1 = event.compute_hash()
        hash2 = event.compute_hash()
        assert hash1 == hash2  # Deterministic
        assert len(hash1) > 0

    def test_hash_changes_with_data(self):
        event1 = LineageEvent(
            trace_id="trace_123",
            event_id="event_456",
            event_type=EventType.EVIDENCE_ADDED,
            timestamp=1234567890.0,
            previous_hash="prev_hash",
            data_hash="data_hash1",
            metadata={},
        )

        event2 = LineageEvent(
            trace_id="trace_123",
            event_id="event_456",
            event_type=EventType.EVIDENCE_ADDED,
            timestamp=1234567890.0,
            previous_hash="prev_hash",
            data_hash="data_hash2",  # Different data
            metadata={},
        )

        assert event1.compute_hash() != event2.compute_hash()


class TestLineageTracker:
    """Test LineageTracker main functionality"""

    @pytest.fixture
    def temp_tracker(self):
        """Create temporary tracker for testing"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name

        tracker = LineageTracker(db_path)
        yield tracker

        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    def test_create_trace(self, temp_tracker):
        """Test trace creation"""
        trace_id = temp_tracker.create_trace("Q001", {"domain": "test"})

        assert trace_id.startswith("trace_")
        assert len(trace_id) > 20  # Reasonable length

        # Verify trace exists
        summary = temp_tracker.get_trace_summary(trace_id)
        assert summary["exists"]
        assert summary["event_count"] >= 1  # At least creation event

    def test_log_evidence(self, temp_tracker):
        """Test evidence logging"""
        trace_id = temp_tracker.create_trace("Q001")
        event_id = temp_tracker.log_evidence_source(
            trace_id, "test evidence", "test_source"
        )

        assert event_id.startswith("event_")

        # Verify evidence in trace
        audit_trail = temp_tracker.generate_audit_trail(trace_id)
        evidence_events = [
            e for e in audit_trail.events if e.event_type == EventType.EVIDENCE_ADDED
        ]
        assert len(evidence_events) >= 1
        assert evidence_events[0].metadata["source"] == "test_source"

    def test_log_processing_step(self, temp_tracker):
        """Test processing step logging"""
        trace_id = temp_tracker.create_trace("Q001")
        event_id = temp_tracker.log_processing_step(
            trace_id, "test_step", {"result": "success"}
        )

        assert event_id.startswith("event_")

        # Verify step in trace
        audit_trail = temp_tracker.generate_audit_trail(trace_id)
        step_events = [
            e for e in audit_trail.events if e.event_type == EventType.PROCESSING_STEP
        ]
        assert len(step_events) >= 1
        assert step_events[0].metadata["step_name"] == "test_step"

    def test_ratchet_chaining(self, temp_tracker):
        """Test cryptographic ratchet chaining"""
        trace_id = temp_tracker.create_trace("Q001")

        # Add multiple events
        temp_tracker.log_evidence_source(trace_id, "evidence1", "source1")
        temp_tracker.log_evidence_source(trace_id, "evidence2", "source2")
        temp_tracker.log_processing_step(trace_id, "step1", "result1")

        # Verify chain integrity
        audit_trail = temp_tracker.generate_audit_trail(trace_id)
        events = sorted(audit_trail.events, key=lambda e: e.timestamp)

        # Check that each event references previous hash
        previous_hash = ""
        for event in events:
            assert event.previous_hash == previous_hash
            previous_hash = event.compute_hash()

    def test_tamper_detection(self, temp_tracker):
        """Test tamper detection through hash verification"""
        trace_id = temp_tracker.create_trace("Q001")
        temp_tracker.log_evidence_source(trace_id, "evidence", "source")

        # Normal verification should pass
        integrity_results = temp_tracker.verify_trace_integrity(trace_id)
        assert not integrity_results["tamper_detected"]
        assert integrity_results["ratchet_integrity"]

        # Manually corrupt database to test tamper detection
        import sqlite3

        with sqlite3.connect(temp_tracker.db_path) as conn:
            conn.execute(
                """
                UPDATE events
                SET event_hash = 'corrupted_hash'
                WHERE trace_id = ?
                LIMIT 1
            """,
                (trace_id,),
            )

        # Clear cache to force reload
        temp_tracker._merkle_cache.clear()

        # Verification should now detect tampering
        integrity_results = temp_tracker.verify_trace_integrity(trace_id)
        assert integrity_results["tamper_detected"]
        assert len(integrity_results["failed_checks"]) > 0

    def test_audit_trail_generation(self, temp_tracker):
        """Test complete audit trail generation"""
        trace_id = temp_tracker.create_trace("Q001", {"test": True})

        # Add various events
        temp_tracker.log_evidence_source(trace_id, "evidence1", "source1")
        temp_tracker.log_processing_step(trace_id, "step1", "result1")
        temp_tracker.log_processing_step(trace_id, "step2", "result2")

        # Generate audit trail
        audit_trail = temp_tracker.generate_audit_trail(trace_id)

        assert audit_trail.trace_id == trace_id
        assert audit_trail.total_events >= 4  # Creation + 3 added events
        assert len(audit_trail.merkle_root) > 0
        assert len(audit_trail.events) == audit_trail.total_events
        assert len(audit_trail.inclusion_proofs) == audit_trail.total_events

    def test_inclusion_proofs(self, temp_tracker):
        """Test Merkle inclusion proof generation and verification"""
        trace_id = temp_tracker.create_trace("Q001")

        # Add multiple events for non-trivial tree
        event_ids = []
        for i in range(5):
            event_id = temp_tracker.log_evidence_source(
                trace_id, f"evidence_{i}", f"source_{i}"
            )
            event_ids.append(event_id)

        # Generate audit trail
        audit_trail = temp_tracker.generate_audit_trail(trace_id)

        # Verify all inclusion proofs
        for event_id, proof in audit_trail.inclusion_proofs.items():
            assert proof.verify(), f"Inclusion proof failed for {event_id}"
            assert proof.merkle_root == audit_trail.merkle_root
            assert proof.trace_id == trace_id

    def test_consistency_proofs(self, temp_tracker):
        """Test cross-snapshot consistency proofs"""
        trace_id = temp_tracker.create_trace("Q001")

        # Add events to create multiple snapshots
        temp_tracker.log_evidence_source(trace_id, "evidence1", "source1")
        temp_tracker.log_processing_step(trace_id, "step1", "result1")

        audit_trail = temp_tracker.generate_audit_trail(trace_id)

        # Should have consistency proofs
        assert len(audit_trail.consistency_proofs) > 0

        # Verify consistency proofs
        for proof in audit_trail.consistency_proofs:
            assert proof.verify(), "Consistency proof verification failed"

    def test_concurrent_access(self, temp_tracker):
        """Test thread safety with concurrent access"""
        import threading

        trace_id = temp_tracker.create_trace("Q001")
        errors = []

        def add_events(thread_id):
            try:
                for i in range(10):
                    temp_tracker.log_evidence_source(
                        trace_id, f"evidence_{thread_id}_{i}", f"source_{thread_id}"
                    )
            except Exception as e:
                errors.append(e)

        # Run concurrent threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=add_events, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Check for errors
        assert len(errors) == 0, f"Concurrent access errors: {errors}"

        # Verify trace integrity
        integrity_results = temp_tracker.verify_trace_integrity(trace_id)
        assert not integrity_results["tamper_detected"]
        assert integrity_results["ratchet_integrity"]


class TestIntegration:
    """Test integration with existing systems"""

    def test_evidence_system_integration(self, temp_tracker):
        """Test integration with EvidenceSystem"""
# # #         from evidence_system import Evidence, EvidenceSystem  # Module not found  # Module not found  # Module not found
# # #         from lineage_tracker import create_evidence_lineage_adapter  # Module not found  # Module not found  # Module not found

        evidence_system = EvidenceSystem()
        adapter = create_evidence_lineage_adapter(evidence_system, temp_tracker)

        # Test evidence tracking
        evidence = Evidence(
            qid="Q001",
            content="Test evidence content",
            score=0.85,
            dimension="test_dim",
        )

        # This would be called during evidence addition
        trace_id = temp_tracker.create_trace("Q001")
        event_id = adapter["track_evidence"]("Q001", evidence)

        assert event_id.startswith("event_")

        # Verify lineage was recorded
        summary = temp_tracker.get_trace_summary(trace_id)
        assert summary["exists"]

    def test_pipeline_tracking(self, temp_tracker):
        """Test complete pipeline tracking"""
# # #         from lineage_tracker import create_evidence_lineage_adapter  # Module not found  # Module not found  # Module not found

        adapter = create_evidence_lineage_adapter(None, temp_tracker)

        # Simulate processing pipeline
        pipeline_steps = [
            ("evidence_collection", {"count": 5, "sources": ["pdf", "api"]}),
            ("conformal_prediction", {"coverage": 0.95, "alpha": 0.05}),
            ("final_scoring", {"score": 0.87, "confidence": 0.92}),
        ]

        trace_id = adapter["track_pipeline"]("Q001", pipeline_steps)

        # Verify complete pipeline was recorded
        audit_trail = temp_tracker.generate_audit_trail(trace_id)

        processing_events = [
            e for e in audit_trail.events if e.event_type == EventType.PROCESSING_STEP
        ]
        assert len(processing_events) == 3

        # Verify step names
        step_names = [e.metadata["step_name"] for e in processing_events]
        expected_steps = [
            "evidence_collection",
            "conformal_prediction",
            "final_scoring",
        ]
        assert all(step in step_names for step in expected_steps)


class TestPerformance:
    """Performance and scalability tests"""

    @pytest.fixture
    def temp_tracker(self):
        """Create temporary tracker for testing"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name

        tracker = LineageTracker(db_path)
        yield tracker

        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    def test_large_trace_performance(self, temp_tracker):
        """Test performance with large trace"""
        trace_id = temp_tracker.create_trace("Q001")

        # Add many events
        n_events = 100
        start_time = time.time()

        for i in range(n_events):
            temp_tracker.log_evidence_source(trace_id, f"evidence_{i}", f"source_{i}")

        addition_time = time.time() - start_time
        assert addition_time < 30  # Should complete within 30 seconds

        # Test audit trail generation
        start_time = time.time()
        audit_trail = temp_tracker.generate_audit_trail(trace_id)
        audit_time = time.time() - start_time

        assert audit_time < 10  # Should generate audit trail quickly
        assert audit_trail.total_events == n_events + 1  # +1 for creation event

    def test_inclusion_proof_complexity(self, temp_tracker):
        """Test O(log n) inclusion proof complexity"""
        trace_id = temp_tracker.create_trace("Q001")

        # Add events in powers of 2 to test logarithmic scaling
        for n in [16, 32, 64]:
            # Clear and rebuild
            temp_tracker._merkle_cache.clear()

            for i in range(n):
                temp_tracker.log_evidence_source(trace_id, f"evidence_{i}", "source")

            # Measure inclusion proof generation time
            start_time = time.time()
            audit_trail = temp_tracker.generate_audit_trail(trace_id)

            # Verify one inclusion proof
            first_event_id = list(audit_trail.inclusion_proofs.keys())[0]
            proof = audit_trail.inclusion_proofs[first_event_id]
            proof_verified = proof.verify()

            proof_time = time.time() - start_time

            assert proof_verified
            # Proof path length should be approximately log2(n)
            expected_path_length = len(proof.proof_path)
            assert expected_path_length <= (n.bit_length() + 1)


if __name__ == "__main__":
    # Run basic functionality test
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        db_path = tmp.name

    try:
        tracker = LineageTracker(db_path)

        # Basic functionality test
        trace_id = tracker.create_trace("Q001", {"test": True})
        tracker.log_evidence_source(trace_id, "evidence", "source")
        tracker.log_processing_step(trace_id, "step", "result")

        audit_trail = tracker.generate_audit_trail(trace_id)
        integrity_results = tracker.verify_trace_integrity(trace_id)

        print(f"✓ Basic functionality test passed")
        print(f"  Trace: {trace_id}")
        print(f"  Events: {audit_trail.total_events}")
        print(f"  Merkle root: {audit_trail.merkle_root[:16]}...")
        print(
            f"  Integrity: {'PASS' if not integrity_results['tamper_detected'] else 'FAIL'}"
        )

        # Test inclusion proofs
        all_proofs_valid = all(
            proof.verify() for proof in audit_trail.inclusion_proofs.values()
        )
        print(f"  Inclusion proofs: {'PASS' if all_proofs_valid else 'FAIL'}")

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)
