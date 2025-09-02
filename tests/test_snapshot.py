"""
Forensics test for cryptographic snapshot validation (SC — Snapshot Contract).

Tests:
1. Reproducibility: Same σ ⇒ identical output digests
2. Refusal behavior: Missing σ ⇒ typed failure without side effects
"""
import hashlib
import json
import os
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

# Add project root to path for canonical imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import blake3
except ImportError:
    # Fallback if blake3 not available
    blake3 = None

from snapshot_manager import (
    get_current_snapshot_id,
    mount_snapshot,
    resolve_snapshot
)


class SnapshotRefusal(Exception):
    """Typed refusal for missing or invalid snapshot."""
    pass


def blake3_hash(data: str) -> str:
    """Generate BLAKE3 hash for forensic integrity."""
    if blake3 is None:
        # Fallback to SHA256 if blake3 not available
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    return blake3.blake3(data.encode('utf-8')).hexdigest()


def sha256_hash(data: str) -> str:
    """Generate SHA-256 hash for forensic integrity."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


class MockProcessor:
    """Mock processor for testing snapshot behavior."""
    
    def __init__(self, use_network=False):
        self.use_network = use_network
        self.side_effects = []
        
    def process(self, data: dict, snapshot_id: str = None) -> dict:
        """Process data with snapshot dependency."""
        if snapshot_id is None:
            raise SnapshotRefusal("Missing required snapshot σ")
            
        if self.use_network:
            raise SnapshotRefusal("Network operations forbidden under snapshot constraint")
            
        # Record side effect for testing (only after validation passes)
        self.side_effects.append(f"process_called_{snapshot_id[:8]}")
            
        # Deterministic processing based on snapshot
        output = {
            "processed": data,
            "snapshot_hash": snapshot_id,
            "deterministic_value": blake3_hash(json.dumps(data, sort_keys=True))
        }
        return output


class TestSnapshotForensics(unittest.TestCase):
    
    def setUp(self):
        """Reset snapshot state for each test."""
        # Clear any existing snapshot
        import snapshot_manager
        with snapshot_manager._lock:
            snapshot_manager._current_snapshot_id = None
            snapshot_manager._registry.clear()
    
    def test_repeated_execution_identical_digests(self):
        """Test that same σ produces identical output digests."""
        # Create deterministic test data
        test_state = {
            "corpus": {"documents": ["doc1", "doc2"], "metadata": {"version": 1}},
            "indices": {"embedding_dim": 768, "index_type": "faiss"},
            "standards": {"compliance": "v2.1", "schema": "json"}
        }
        
        # Mount snapshot
        sigma = mount_snapshot(test_state)
        
        # Create processor
        processor = MockProcessor(use_network=False)
        
        # Test data
        input_data = {"query": "test query", "params": {"top_k": 5}}
        
        # Execute multiple times
        results = []
        for i in range(3):
            output = processor.process(input_data, snapshot_id=sigma)
            output_json = json.dumps(output, sort_keys=True, separators=(',', ':'))
            digest = blake3_hash(output_json)
            results.append(digest)
        
        # All digests must be identical
        self.assertEqual(len(set(results)), 1, "Repeated execution with same σ must produce identical digests")
        
        # Verify snapshot immutability
        snapshot_data = resolve_snapshot(sigma)
        self.assertIn("frozen_json", snapshot_data)
        
        # Re-resolve must be byte-identical
        snapshot_data2 = resolve_snapshot(sigma)
        self.assertEqual(snapshot_data["frozen_json"], snapshot_data2["frozen_json"])
    
    def test_missing_sigma_typed_refusal(self):
        """Test that missing σ causes typed refusal without side effects."""
        processor = MockProcessor(use_network=False)
        input_data = {"query": "test query"}
        
        # Ensure no side effects recorded before failure
        initial_effects = len(processor.side_effects)
        
        # Attempt processing without snapshot
        with self.assertRaises(SnapshotRefusal) as cm:
            processor.process(input_data, snapshot_id=None)
        
        # Verify typed refusal message
        self.assertIn("Missing required snapshot σ", str(cm.exception))
        
        # Verify no side effects occurred
        self.assertEqual(len(processor.side_effects), initial_effects)
        
        # Verify system state unchanged
        self.assertIsNone(get_current_snapshot_id())
    
    def test_network_operations_refused(self):
        """Test that network operations are refused under snapshot constraint."""
        test_state = {"corpus": {}, "indices": {}, "standards": {}}
        sigma = mount_snapshot(test_state)
        
        processor = MockProcessor(use_network=True)
        input_data = {"query": "test query"}
        
        initial_effects = len(processor.side_effects)
        
        with self.assertRaises(SnapshotRefusal) as cm:
            processor.process(input_data, snapshot_id=sigma)
        
        self.assertIn("Network operations forbidden", str(cm.exception))
        self.assertEqual(len(processor.side_effects), initial_effects)
    
    def test_snapshot_immutability(self):
        """Test that snapshots cannot be mutated after creation."""
        test_state = {
            "corpus": {"docs": ["a", "b"]},
            "indices": {"dim": 128},
            "standards": {"v": 1}
        }
        
        sigma = mount_snapshot(test_state)
        snapshot_data = resolve_snapshot(sigma)
        
        # Attempt to modify resolved data (should not affect original)
        frozen_copy = json.loads(snapshot_data["frozen_json"])
        frozen_copy["corpus"]["docs"].append("c")
        
        # Re-resolve should be unchanged
        snapshot_data2 = resolve_snapshot(sigma)
        self.assertEqual(snapshot_data["frozen_json"], snapshot_data2["frozen_json"])
    
    def test_checksum_verification(self):
        """Test cryptographic integrity verification."""
        test_state = {
            "corpus": {"test": "data"},
            "indices": {"type": "dense"},
            "standards": {"version": "1.0"}
        }
        
        sigma = mount_snapshot(test_state)
        
        # Manually verify checksum computation
        from snapshot_manager import compute_snapshot_id
        
        expected_corpus = sha256_hash(json.dumps({"test": "data"}, sort_keys=True, separators=(',', ':')))
        expected_indices = sha256_hash(json.dumps({"type": "dense"}, sort_keys=True, separators=(',', ':')))
        expected_standards = sha256_hash(json.dumps({"version": "1.0"}, sort_keys=True, separators=(',', ':')))
        
        expected_sigma = compute_snapshot_id(
            corpus_digest=expected_corpus,
            index_digest=expected_indices, 
            standards_digest=expected_standards
        )
        
        self.assertEqual(sigma, expected_sigma)


if __name__ == "__main__":
    unittest.main()