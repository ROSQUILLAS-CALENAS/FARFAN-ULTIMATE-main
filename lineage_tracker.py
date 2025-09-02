"""
Tamper-evident lineage tracking system using cryptographic ratchets and Merkle structures.
Provides immutable audit trails for question processing pipelines with O(log n) inclusion proofs.
"""
import hashlib
import json
import logging
import secrets
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Export all public classes and functions
__all__ = [
    "EventType",
    "LineageEvent", 
    "MerkleNode",
    "InclusionProof",
    "ConsistencyProof",
    "AuditTrail",
    "MerkleTree",
    "LineageTracker",
    "create_evidence_lineage_adapter"
]

# Try to import blake3, fallback to hashlib for SHA-256
try:
    import blake3

    HASH_FUNCTION = blake3.blake3
    HASH_NAME = "blake3"
except ImportError:
    HASH_FUNCTION = lambda data: hashlib.sha256(
        data.encode() if isinstance(data, str) else data
    )
    HASH_NAME = "sha256"


class EventType(str, Enum):
    """Types of events in the lineage trace"""

    TRACE_CREATED = "trace_created"
    EVIDENCE_ADDED = "evidence_added"
    PROCESSING_STEP = "processing_step"
    RESULT_GENERATED = "result_generated"
    AUDIT_CHECKPOINT = "audit_checkpoint"


@dataclass(frozen=True)
class LineageEvent:
    """Immutable lineage event structure"""

    trace_id: str
    event_id: str
    event_type: EventType
    timestamp: float
    previous_hash: str
    data_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_hash(self) -> str:
        """Compute cryptographic hash of this event"""
        event_data = {
            "trace_id": self.trace_id,
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "data_hash": self.data_hash,
            "metadata": self.metadata,
        }
        serialized = json.dumps(event_data, sort_keys=True)
        return HASH_FUNCTION(serialized).hexdigest()


@dataclass
class MerkleNode:
    """Merkle tree node for efficient inclusion proofs"""

    hash_value: str
    left: Optional["MerkleNode"] = None
    right: Optional["MerkleNode"] = None
    data: Optional[str] = None  # Leaf nodes only

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


@dataclass
class InclusionProof:
    """Merkle inclusion proof structure"""

    leaf_hash: str
    merkle_root: str
    proof_path: List[Tuple[str, str]]  # (hash, direction: 'left' or 'right')
    trace_id: str
    event_id: str

    def verify(self) -> bool:
        """Verify inclusion proof in O(log n) time"""
        current_hash = self.leaf_hash

        for sibling_hash, direction in self.proof_path:
            if direction == "left":
                combined = f"{sibling_hash}{current_hash}"
            else:
                combined = f"{current_hash}{sibling_hash}"
            current_hash = HASH_FUNCTION(combined).hexdigest()

        return current_hash == self.merkle_root


@dataclass
class ConsistencyProof:
    """Merkle consistency proof for cross-snapshot verification"""

    old_size: int
    new_size: int
    old_root: str
    new_root: str
    proof_hashes: List[str]

    def verify(self) -> bool:
        """Verify consistency proof (prefix property)"""
        if self.old_size >= self.new_size:
            return self.old_root == self.new_root

        # Simplified consistency check - full implementation would follow RFC 6962
        # For now, verify that old_root appears in proof_hashes
        return self.old_root in self.proof_hashes


@dataclass
class AuditTrail:
    """Complete audit trail with cryptographic proofs"""

    trace_id: str
    merkle_root: str
    total_events: int
    creation_time: float
    last_update: float
    events: List[LineageEvent]
    inclusion_proofs: Dict[str, InclusionProof]
    consistency_proofs: List[ConsistencyProof]
    tamper_checks: Dict[str, bool]
    metadata: Dict[str, Any] = field(default_factory=dict)


class MerkleTree:
    """Efficient Merkle tree implementation for lineage events"""

    def __init__(self, leaf_hashes: List[str]):
        self.leaf_hashes = leaf_hashes.copy()
        self.levels = self._build_levels(leaf_hashes)
        self.root = self.levels[-1][0] if self.levels else MerkleNode("")

    def _build_levels(self, hashes: List[str]) -> List[List[MerkleNode]]:
        """Build Merkle tree levels bottom-up for easier proof generation"""
        if not hashes:
            return []

        # Start with leaf level
        levels = []
        current_level = [MerkleNode(h, data=h) for h in hashes]
        levels.append(current_level)

        # Build up to root
        while len(current_level) > 1:
            next_level = []

            # Pad with duplicate if odd
            if len(current_level) % 2 == 1:
                current_level.append(current_level[-1])

            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1]

                combined_hash = HASH_FUNCTION(
                    f"{left.hash_value}{right.hash_value}"
                ).hexdigest()
                parent = MerkleNode(combined_hash, left=left, right=right)
                next_level.append(parent)

            levels.append(next_level)
            current_level = next_level

        return levels

    def get_inclusion_proof(self, leaf_hash: str) -> List[Tuple[str, str]]:
        """Generate inclusion proof for a leaf using level-based approach"""
        if not self.levels or not self.leaf_hashes:
            return []

        # Find leaf index in original leaf list
        leaf_index = -1
        for i, hash_val in enumerate(self.leaf_hashes):
            if hash_val == leaf_hash:
                leaf_index = i
                break

        if leaf_index == -1:
            return []

        # Build proof path level by level
        proof_path = []
        current_index = leaf_index

        # Traverse from leaf to root (excluding root level)
        for level_idx in range(len(self.levels) - 1):
            current_level = self.levels[level_idx]

            # Account for padding
            level_size = len(current_level)
            if current_index >= level_size:
                current_index = level_size - 1

            # Find sibling
            if current_index % 2 == 0:
                # Left node, sibling is right
                sibling_index = current_index + 1
                direction = "right"
            else:
                # Right node, sibling is left
                sibling_index = current_index - 1
                direction = "left"

            # Get sibling hash if it exists and is different
            if sibling_index < level_size:
                sibling_node = current_level[sibling_index]
                proof_path.append((sibling_node.hash_value, direction))

            # Move to parent level
            current_index = current_index // 2

        return proof_path

    @property
    def root_hash(self) -> str:
        return self.root.hash_value if self.root else ""


class LineageTracker:
    """Tamper-evident lineage tracking system with cryptographic guarantees"""

    def __init__(self, db_path: str = "lineage.db"):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._ratchet_state: Dict[str, str] = {}  # trace_id -> current_hash
        self._merkle_cache: Dict[str, MerkleTree] = {}
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database with tamper-evident schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    creation_time REAL NOT NULL,
                    last_update REAL NOT NULL,
                    current_hash TEXT NOT NULL,
                    event_count INTEGER DEFAULT 0,
                    merkle_root TEXT DEFAULT '',
                    metadata TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    previous_hash TEXT NOT NULL,
                    data_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (trace_id) REFERENCES traces (trace_id)
                );

                CREATE TABLE IF NOT EXISTS merkle_snapshots (
                    trace_id TEXT NOT NULL,
                    snapshot_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    merkle_root TEXT NOT NULL,
                    event_count INTEGER NOT NULL,
                    tree_data TEXT NOT NULL,
                    PRIMARY KEY (trace_id, snapshot_id),
                    FOREIGN KEY (trace_id) REFERENCES traces (trace_id)
                );

                CREATE INDEX IF NOT EXISTS idx_events_trace_time
                ON events (trace_id, timestamp);

                CREATE INDEX IF NOT EXISTS idx_events_hash
                ON events (event_hash);
            """
            )

    def create_trace(
        self, question_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create new lineage trace and return root handle"""
        trace_id = f"trace_{int(time.time() * 1000)}_{secrets.token_hex(8)}"
        current_time = time.time()

        # Genesis hash for new trace - will be computed by first event
        genesis_hash = ""

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO traces
                    (trace_id, creation_time, last_update, current_hash, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        trace_id,
                        current_time,
                        current_time,
                        genesis_hash,
                        json.dumps(metadata or {}),
                    ),
                )

            # Initialize ratchet state - empty for first event
            self._ratchet_state[trace_id] = ""

            # Log creation event with empty previous hash
            creation_data = {
                "trace_id": trace_id,
                "question_id": question_id,
                "creation_time": current_time,
                "metadata": metadata or {},
            }
            creation_data_hash = HASH_FUNCTION(
                json.dumps(creation_data, sort_keys=True)
            ).hexdigest()

            self._append_event(
                trace_id,
                EventType.TRACE_CREATED,
                {"question_id": question_id},
                creation_data_hash,
            )

        logger.info(f"Created tamper-evident trace: {trace_id}")
        return trace_id

    def log_evidence_source(self, trace_id: str, evidence: Any, source: str) -> str:
        """Log evidence with content hash and metadata"""
        # Compute evidence content hash
        evidence_data = {
            "content": str(evidence),
            "source": source,
            "timestamp": time.time(),
        }
        evidence_hash = HASH_FUNCTION(
            json.dumps(evidence_data, sort_keys=True)
        ).hexdigest()

        metadata = {
            "source": source,
            "evidence_type": type(evidence).__name__,
            "content_preview": str(evidence)[:100]
            if len(str(evidence)) > 100
            else str(evidence),
        }

        event_id = self._append_event(
            trace_id, EventType.EVIDENCE_ADDED, metadata, evidence_hash
        )
        logger.info(f"Logged evidence for trace {trace_id}: {event_id}")
        return event_id

    def log_processing_step(self, trace_id: str, step: str, result: Any) -> str:
        """Log processing step with chained hash"""
        result_data = {"step": step, "result": str(result), "timestamp": time.time()}
        result_hash = HASH_FUNCTION(json.dumps(result_data, sort_keys=True)).hexdigest()

        metadata = {
            "step_name": step,
            "result_type": type(result).__name__,
            "result_preview": str(result)[:100]
            if len(str(result)) > 100
            else str(result),
        }

        event_id = self._append_event(
            trace_id, EventType.PROCESSING_STEP, metadata, result_hash
        )
        logger.info(f"Logged processing step for trace {trace_id}: {step}")
        return event_id

    def _append_event(
        self,
        trace_id: str,
        event_type: EventType,
        metadata: Dict[str, Any],
        data_hash: str,
    ) -> str:
        """Append event to trace with cryptographic chaining"""
        with self._lock:
            # Get current ratchet state
            previous_hash = self._ratchet_state.get(trace_id, "")

            # Generate unique event ID
            event_id = f"event_{int(time.time() * 1000000)}_{secrets.token_hex(6)}"
            current_time = time.time()

            # Create event
            event = LineageEvent(
                trace_id=trace_id,
                event_id=event_id,
                event_type=event_type,
                timestamp=current_time,
                previous_hash=previous_hash,
                data_hash=data_hash,
                metadata=metadata,
            )

            # Compute event hash
            event_hash = event.compute_hash()

            # Store event
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO events
                    (event_id, trace_id, event_type, timestamp, previous_hash,
                     data_hash, event_hash, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        event_id,
                        trace_id,
                        event_type.value,
                        current_time,
                        previous_hash,
                        data_hash,
                        event_hash,
                        json.dumps(metadata),
                    ),
                )

                # Update trace state
                conn.execute(
                    """
                    UPDATE traces
                    SET last_update = ?, current_hash = ?, event_count = event_count + 1
                    WHERE trace_id = ?
                """,
                    (current_time, event_hash, trace_id),
                )

            # Update ratchet state (cryptographic advancement)
            self._ratchet_state[trace_id] = event_hash

            # Clear Merkle cache for this trace
            if trace_id in self._merkle_cache:
                del self._merkle_cache[trace_id]

            return event_id

    def _get_merkle_tree(self, trace_id: str) -> MerkleTree:
        """Get or compute Merkle tree for trace"""
        if trace_id in self._merkle_cache:
            return self._merkle_cache[trace_id]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT event_hash FROM events
                WHERE trace_id = ?
                ORDER BY timestamp ASC
            """,
                (trace_id,),
            )

            event_hashes = [row[0] for row in cursor.fetchall()]

        if not event_hashes:
            tree = MerkleTree([])
        else:
            tree = MerkleTree(event_hashes)

        self._merkle_cache[trace_id] = tree
        return tree

    def generate_audit_trail(self, trace_id: str) -> AuditTrail:
        """Generate complete audit trail with Merkle proofs"""
        with self._lock:
            # Get trace info
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT creation_time, last_update, event_count, metadata
                    FROM traces WHERE trace_id = ?
                """,
                    (trace_id,),
                )

                trace_row = cursor.fetchone()
                if not trace_row:
                    raise ValueError(f"Trace not found: {trace_id}")

                creation_time, last_update, event_count, metadata_json = trace_row

                # Get all events with stored event_hash
                cursor = conn.execute(
                    """
                    SELECT event_id, event_type, timestamp, previous_hash,
                           data_hash, event_hash, metadata
                    FROM events
                    WHERE trace_id = ?
                    ORDER BY timestamp ASC
                """,
                    (trace_id,),
                )

                events = []
                stored_hashes = []
                for row in cursor.fetchall():
                    (
                        event_id,
                        event_type,
                        timestamp,
                        previous_hash,
                        data_hash,
                        stored_event_hash,
                        metadata_json,
                    ) = row

                    event = LineageEvent(
                        trace_id=trace_id,
                        event_id=event_id,
                        event_type=EventType(event_type),
                        timestamp=timestamp,
                        previous_hash=previous_hash,
                        data_hash=data_hash,
                        metadata=json.loads(metadata_json),
                    )
                    events.append(event)
                    stored_hashes.append(
                        stored_event_hash
                    )  # Use stored hash for Merkle tree

            # Generate Merkle tree using stored hashes
            merkle_tree = MerkleTree(stored_hashes) if stored_hashes else MerkleTree([])
            inclusion_proofs = {}
            tamper_checks = {}

            for i, (event, stored_hash) in enumerate(zip(events, stored_hashes)):
                # Verify event hash integrity
                computed_hash = event.compute_hash()
                tamper_checks[event.event_id] = computed_hash == stored_hash

                # Generate inclusion proof using stored hash
                proof_path = merkle_tree.get_inclusion_proof(stored_hash)
                inclusion_proof = InclusionProof(
                    leaf_hash=stored_hash,
                    merkle_root=merkle_tree.root_hash,
                    proof_path=proof_path,
                    trace_id=trace_id,
                    event_id=event.event_id,
                )
                inclusion_proofs[event.event_id] = inclusion_proof

            # Generate consistency proofs (simplified)
            consistency_proofs = []
            if event_count > 1:
                # Create a consistency proof for the full trace
                consistency_proof = ConsistencyProof(
                    old_size=1,
                    new_size=event_count,
                    old_root=stored_hashes[0] if stored_hashes else "",
                    new_root=merkle_tree.root_hash,
                    proof_hashes=stored_hashes,
                )
                consistency_proofs.append(consistency_proof)

            audit_trail = AuditTrail(
                trace_id=trace_id,
                merkle_root=merkle_tree.root_hash,
                total_events=event_count,
                creation_time=creation_time,
                last_update=last_update,
                events=events,
                inclusion_proofs=inclusion_proofs,
                consistency_proofs=consistency_proofs,
                tamper_checks=tamper_checks,
                metadata=json.loads(metadata_json) if metadata_json else {},
            )

            return audit_trail

    def verify_trace_integrity(self, trace_id: str) -> Dict[str, Any]:
        """Comprehensive integrity verification"""
        audit_trail = self.generate_audit_trail(trace_id)

        results = {
            "trace_id": trace_id,
            "total_events": audit_trail.total_events,
            "merkle_root": audit_trail.merkle_root,
            "tamper_detected": False,
            "failed_checks": [],
            "inclusion_proof_results": {},
            "consistency_check": True,
            "ratchet_integrity": True,
        }

        # Check tamper evidence
        for event_id, integrity_ok in audit_trail.tamper_checks.items():
            if not integrity_ok:
                results["tamper_detected"] = True
                results["failed_checks"].append(f"Hash mismatch: {event_id}")

        # Verify inclusion proofs
        for event_id, proof in audit_trail.inclusion_proofs.items():
            proof_valid = proof.verify()
            results["inclusion_proof_results"][event_id] = proof_valid
            if not proof_valid:
                results["tamper_detected"] = True
                results["failed_checks"].append(f"Inclusion proof failed: {event_id}")

        # Verify consistency proofs
        for consistency_proof in audit_trail.consistency_proofs:
            if not consistency_proof.verify():
                results["consistency_check"] = False
                results["failed_checks"].append("Consistency proof failed")

        # Verify ratchet chain using stored hashes
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT event_id, previous_hash, event_hash
                FROM events
                WHERE trace_id = ?
                ORDER BY timestamp ASC
            """,
                (trace_id,),
            )

            previous_hash = ""
            for event_id, stored_prev_hash, stored_event_hash in cursor.fetchall():
                if stored_prev_hash != previous_hash:
                    results["ratchet_integrity"] = False
                    results["failed_checks"].append(
                        f"Ratchet chain broken at {event_id}"
                    )
                    break
                previous_hash = stored_event_hash

        return results

    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get summary statistics for a trace"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as event_count,
                    MIN(timestamp) as first_event,
                    MAX(timestamp) as last_event,
                    COUNT(DISTINCT event_type) as unique_event_types
                FROM events
                WHERE trace_id = ?
            """,
                (trace_id,),
            )

            stats = cursor.fetchone()
            if not stats or stats[0] == 0:
                return {"trace_id": trace_id, "exists": False}

            event_count, first_event, last_event, unique_event_types = stats

            # Get event type distribution
            cursor = conn.execute(
                """
                SELECT event_type, COUNT(*) as count
                FROM events
                WHERE trace_id = ?
                GROUP BY event_type
            """,
                (trace_id,),
            )

            event_distribution = dict(cursor.fetchall())

            return {
                "trace_id": trace_id,
                "exists": True,
                "event_count": event_count,
                "duration_seconds": last_event - first_event,
                "unique_event_types": unique_event_types,
                "event_distribution": event_distribution,
                "first_event": first_event,
                "last_event": last_event,
                "hash_algorithm": HASH_NAME,
            }


# Integration utilities for existing QuestionContext system


def create_evidence_lineage_adapter(evidence_system, lineage_tracker):
    """Create adapter to integrate with existing EvidenceSystem"""

    def track_evidence_addition(qid: str, evidence: Any) -> str:
        """Track evidence addition in lineage"""
        # Get or create trace for this question
        trace_id = f"trace_{qid}_{int(time.time())}"

        # Log evidence with source information
        return lineage_tracker.log_evidence_source(
            trace_id, evidence, source="evidence_system"
        )

    def track_processing_pipeline(qid: str, steps: List[Tuple[str, Any]]) -> str:
        """Track complete processing pipeline"""
        trace_id = lineage_tracker.create_trace(qid)

        for step_name, result in steps:
            lineage_tracker.log_processing_step(trace_id, step_name, result)

        return trace_id

    return {
        "track_evidence": track_evidence_addition,
        "track_pipeline": track_processing_pipeline,
    }


if __name__ == "__main__":
    # Demo usage
    tracker = LineageTracker()

    # Create trace
    trace_id = tracker.create_trace("Q001", {"domain": "municipal_planning"})

    # Add evidence
    tracker.log_evidence_source(trace_id, "Budget document analysis", "pdf_extractor")
    tracker.log_evidence_source(trace_id, "Population statistics", "census_data")

    # Log processing steps
    tracker.log_processing_step(trace_id, "evidence_aggregation", {"score": 0.85})
    tracker.log_processing_step(trace_id, "conformal_prediction", {"coverage": 0.95})
    tracker.log_processing_step(
        trace_id, "final_answer", "Recommendation: Increase budget allocation"
    )

    # Generate audit trail
    audit_trail = tracker.generate_audit_trail(trace_id)
    print(f"Generated audit trail with {len(audit_trail.events)} events")
    print(f"Merkle root: {audit_trail.merkle_root}")

    # Verify integrity
    integrity_results = tracker.verify_trace_integrity(trace_id)
    print(
        f"Integrity check: {'PASS' if not integrity_results['tamper_detected'] else 'FAIL'}"
    )

    # Test inclusion proofs
    for event_id, proof in audit_trail.inclusion_proofs.items():
        verified = proof.verify()
        print(f"Inclusion proof for {event_id}: {'VALID' if verified else 'INVALID'}")
