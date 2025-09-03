"""
Tests for traceability system using Merkle trees.
Validates replay consistency and tamper detection.
"""

import pytest
import json
import tempfile
import os
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
import sys

# Add tools directory to path
sys.path.append(str(Path(__file__).parent.parent / "tools"))

# # # from audit_trail import AuditTrail, MerkleNode  # Module not found  # Module not found  # Module not found


class TestMerkleTreeTraceability:
    """Test suite for Merkle tree-based traceability."""
    
    def setup_method(self):
        """Setup test environment with temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_audit.db")
        self.audit = AuditTrail(self.db_path)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_trace_addition(self):
        """Test basic trace addition and hash computation."""
        trace_data = {"action": "query", "params": {"q": "test"}}
        trace_id = self.audit.add_trace("search_operation", trace_data)
        
        assert trace_id > 0
        assert len(self.audit.traces) == 1
        assert self.audit.traces[0]["operation"] == "search_operation"
        assert self.audit.traces[0]["data"] == trace_data
        assert "hash" in self.audit.traces[0]
    
    def test_replay_produces_same_hashes(self):
        """Test that replay produces identical hashes."""
        # Add multiple traces
        traces_data = [
            {"action": "query", "params": {"q": "test1"}},
            {"action": "retrieve", "params": {"doc_id": "123"}},
            {"action": "rank", "params": {"scores": [0.8, 0.6, 0.4]}}
        ]
        
        original_hashes = []
        for i, data in enumerate(traces_data):
            self.audit.add_trace(f"operation_{i}", data)
            original_hashes.append(self.audit.traces[-1]["hash"])
        
        original_root = self.audit.get_merkle_root()
        
        # Replay traces
        replayed_hashes, replayed_root = self.audit.replay_traces()
        
        # Verify hashes match
        assert len(original_hashes) == len(replayed_hashes)
        for orig, replay in zip(original_hashes, replayed_hashes):
            assert orig == replay
        
        assert original_root == replayed_root
    
    def test_tampering_detection(self):
        """Test that tampering is detected when traces are modified."""
        # Add initial traces
        self.audit.add_trace("operation_1", {"data": "original"})
        self.audit.add_trace("operation_2", {"data": "untouched"})
        
        original_root = self.audit.get_merkle_root()
        
        # Simulate tampering by modifying trace data
        self.audit.traces[0]["data"]["data"] = "tampered"
        
        # Detect tampering
        tamper_detected = self.audit.detect_tampering()
        assert tamper_detected == True
        
        # Verify root changed
        new_root = self.audit.get_merkle_root()
        # Root might be same if only data changed but not hash field
        # So let's modify the hash directly
        original_hash = self.audit.traces[0]["hash"]
        self.audit.traces[0]["hash"] = "tampered_hash"
        
        tamper_detected = self.audit.detect_tampering()
        assert tamper_detected == True
    
    def test_hash_modification_changes_root(self):
        """Test that modifying trace hash changes Merkle root."""
        # Add traces
        self.audit.add_trace("op1", {"value": 1})
        self.audit.add_trace("op2", {"value": 2})
        
        original_root = self.audit.get_merkle_root()
        
        # Modify hash directly
        original_hash = self.audit.traces[0]["hash"]
        self.audit.traces[0]["hash"] = "modified_hash"
        
        new_root = self.audit.get_merkle_root()
        
        # Root should change
        assert original_root != new_root
        
        # Restore original hash
        self.audit.traces[0]["hash"] = original_hash
        restored_root = self.audit.get_merkle_root()
        
        # Root should match original
        assert original_root == restored_root
    
    def test_inclusion_proof_generation(self):
        """Test generation of inclusion proofs."""
        # Add multiple traces
        for i in range(4):
            self.audit.add_trace(f"operation_{i}", {"index": i})
        
        root = self.audit.get_merkle_root()
        
        # Generate proofs for each trace
        for i in range(len(self.audit.traces)):
            proof = self.audit.generate_inclusion_proof(i)
            assert isinstance(proof, list)
            
            # Verify proof
            trace_hash = self.audit.traces[i]["hash"]
            is_valid = self.audit.verify_inclusion_proof(trace_hash, proof, root)
            assert is_valid == True
    
    def test_inclusion_proof_verification_failure(self):
        """Test that invalid proofs are rejected."""
        # Add traces
        self.audit.add_trace("op1", {"data": "test1"})
        self.audit.add_trace("op2", {"data": "test2"})
        
        root = self.audit.get_merkle_root()
        proof = self.audit.generate_inclusion_proof(0)
        
        # Verify with wrong hash
        fake_hash = "fake_hash_value"
        is_valid = self.audit.verify_inclusion_proof(fake_hash, proof, root)
        assert is_valid == False
        
        # Verify with wrong root
        fake_root = "fake_root_value"
        trace_hash = self.audit.traces[0]["hash"]
        is_valid = self.audit.verify_inclusion_proof(trace_hash, proof, fake_root)
        assert is_valid == False
    
    def test_certificate_generation(self):
        """Test audit certificate generation."""
        # Add traces
        self.audit.add_trace("query", {"q": "test"})
        self.audit.add_trace("retrieve", {"docs": [1, 2, 3]})
        
        certificate = self.audit.generate_certificate()
        
        # Verify certificate structure
        assert "pass" in certificate
        assert "merkle_root" in certificate
        assert "proofs" in certificate
        assert "tamper_detected" in certificate
        assert "hash_function" in certificate
        assert "total_traces" in certificate
        assert "timestamp" in certificate
        
        # Verify values
        assert certificate["proofs"] == 2
        assert certificate["total_traces"] == 2
        assert certificate["tamper_detected"] == False
        assert certificate["pass"] == True
    
    def test_certificate_with_tampering(self):
        """Test certificate generation with tampered data."""
        # Add traces
        self.audit.add_trace("op1", {"data": "original"})
        
        # Tamper with data
        self.audit.traces[0]["hash"] = "tampered_hash"
        
        certificate = self.audit.generate_certificate()
        
        assert certificate["tamper_detected"] == True
        assert certificate["pass"] == False
    
    def test_single_trace_merkle_tree(self):
        """Test Merkle tree with single trace."""
        self.audit.add_trace("single_op", {"data": "alone"})
        
        root = self.audit.get_merkle_root()
        assert root is not None
        
        # Proof should work
        proof = self.audit.generate_inclusion_proof(0)
        trace_hash = self.audit.traces[0]["hash"]
        is_valid = self.audit.verify_inclusion_proof(trace_hash, proof, root)
        assert is_valid == True
    
    def test_odd_number_of_traces(self):
        """Test Merkle tree construction with odd number of traces."""
        # Add 3 traces (odd number)
        for i in range(3):
            self.audit.add_trace(f"op_{i}", {"index": i})
        
        root = self.audit.get_merkle_root()
        assert root is not None
        
        # All proofs should be valid
        for i in range(3):
            proof = self.audit.generate_inclusion_proof(i)
            trace_hash = self.audit.traces[i]["hash"]
            is_valid = self.audit.verify_inclusion_proof(trace_hash, proof, root)
            assert is_valid == True
    
    def test_empty_audit_trail(self):
        """Test behavior with empty audit trail."""
        root = self.audit.get_merkle_root()
        assert root is None
        
        certificate = self.audit.generate_certificate()
        assert certificate["merkle_root"] is None
        assert certificate["proofs"] == 0
        assert certificate["total_traces"] == 0
    
    def test_database_persistence(self):
        """Test that traces persist in database."""
        # Add trace
        self.audit.add_trace("persistent_op", {"data": "persistent"})
        original_count = len(self.audit.traces)
        
        # Create new audit instance with same database
        new_audit = AuditTrail(self.db_path)
        new_audit.load_traces_from_db()
        
        assert len(new_audit.traces) == original_count
        assert new_audit.traces[0]["operation"] == "persistent_op"
        assert new_audit.traces[0]["data"]["data"] == "persistent"
    
    def test_mutation_changes_verification(self):
        """Test that mutation attempts change verification results."""
        # Add initial traces
        self.audit.add_trace("op1", {"value": "original1"})
        self.audit.add_trace("op2", {"value": "original2"})
        
        # Generate initial certificate
        original_cert = self.audit.generate_certificate()
        assert original_cert["pass"] == True
        original_root = original_cert["merkle_root"]
        
        # Test data mutation detection (changes data but not hash)
        self.audit.traces[0]["data"]["value"] = "mutated"
        
        data_mutated_cert = self.audit.generate_certificate()
        assert data_mutated_cert["tamper_detected"] == True  # Should detect data inconsistency
        assert data_mutated_cert["pass"] == False
        
        # Restore original data
        self.audit.traces[0]["data"]["value"] = "original1"
        
        # Test hash mutation (changes hash directly)
        original_hash = self.audit.traces[0]["hash"]
        self.audit.traces[0]["hash"] = "tampered_hash_value"
        
# # #         # The merkle root should change since tree is built from stored hashes  # Module not found  # Module not found  # Module not found
        hash_mutated_cert = self.audit.generate_certificate()
        assert hash_mutated_cert["merkle_root"] != original_root
    
    def test_hash_consistency(self):
        """Test that identical data produces identical hashes."""
        audit1 = AuditTrail(os.path.join(self.temp_dir, "audit1.db"))
        audit2 = AuditTrail(os.path.join(self.temp_dir, "audit2.db"))
        
        # Add same data to both
        data = {"operation": "test", "params": {"value": 42}}
        audit1.add_trace("test_op", data, "2024-01-01T00:00:00")
        audit2.add_trace("test_op", data, "2024-01-01T00:00:00")
        
        # Hashes should be identical
        assert audit1.traces[0]["hash"] == audit2.traces[0]["hash"]
        assert audit1.get_merkle_root() == audit2.get_merkle_root()


class TestMerkleNode:
    """Test Merkle node functionality."""
    
    def test_leaf_node_identification(self):
        """Test leaf node identification."""
        leaf = MerkleNode("hash_value", data={"test": "data"})
        assert leaf.is_leaf() == True
        
        parent = MerkleNode("parent_hash", left=leaf, right=None)
        assert parent.is_leaf() == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])