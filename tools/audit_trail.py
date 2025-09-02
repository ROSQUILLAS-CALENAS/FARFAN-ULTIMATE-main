"""
Audit trail system using Merkle trees for traceability and tamper detection.
Generates inclusion proofs and validates data integrity.
"""

import hashlib
import json
import sqlite3
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging

try:
    import blake3
    HASH_FUNCTION = blake3.blake3
    HASH_NAME = "blake3"
except ImportError:
    HASH_FUNCTION = hashlib.sha256
    HASH_NAME = "sha256"

logger = logging.getLogger(__name__)


class MerkleNode:
    """Represents a node in the Merkle tree."""
    
    def __init__(self, hash_value: str, left: Optional['MerkleNode'] = None, 
                 right: Optional['MerkleNode'] = None, data: Optional[Any] = None):
        self.hash_value = hash_value
        self.left = left
        self.right = right
        self.data = data
        
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class AuditTrail:
    """Merkle tree-based audit trail for tracking operations and detecting tampering."""
    
    def __init__(self, db_path: str = "audit_trail.db"):
        self.db_path = db_path
        self.traces: List[Dict[str, Any]] = []
        self._init_database()
        
    def _init_database(self) -> None:
        """Initialize SQLite database for audit trail storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    data TEXT NOT NULL,
                    hash_value TEXT NOT NULL,
                    merkle_root TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS merkle_proofs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id INTEGER,
                    proof_path TEXT NOT NULL,
                    FOREIGN KEY (trace_id) REFERENCES audit_traces(id)
                )
            """)
    
    def _compute_hash(self, data: Any) -> str:
        """Compute hash of data using configured hash function."""
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, dict):
            data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
        else:
            data_bytes = str(data).encode('utf-8')
            
        if HASH_NAME == "blake3":
            return HASH_FUNCTION(data_bytes).hexdigest()
        else:
            return HASH_FUNCTION(data_bytes).hexdigest()
    
    def add_trace(self, operation: str, data: Dict[str, Any], timestamp: str = None) -> int:
        """Add a new trace entry to the audit trail."""
        import datetime
        if timestamp is None:
            timestamp = datetime.datetime.now().isoformat()
            
        trace = {
            "timestamp": timestamp,
            "operation": operation,
            "data": data
        }
        
        hash_value = self._compute_hash(trace)
        trace["hash"] = hash_value
        self.traces.append(trace)
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO audit_traces (timestamp, operation, data, hash_value)
                VALUES (?, ?, ?, ?)
            """, (timestamp, operation, json.dumps(data), hash_value))
            trace_id = cursor.lastrowid
            
        logger.info(f"Added trace {trace_id}: {operation}")
        return trace_id
    
    def build_merkle_tree(self) -> Optional[MerkleNode]:
        """Build Merkle tree from all traces."""
        if not self.traces:
            return None
            
        # Create leaf nodes from trace hashes
        nodes = [MerkleNode(trace["hash"], data=trace) for trace in self.traces]
        
        # Build tree bottom-up
        while len(nodes) > 1:
            next_level = []
            
            # Process pairs of nodes
            for i in range(0, len(nodes), 2):
                left = nodes[i]
                right = nodes[i + 1] if i + 1 < len(nodes) else None
                
                if right is None:
                    # Odd number of nodes, duplicate the last one
                    combined_hash = self._compute_hash(left.hash_value + left.hash_value)
                    parent = MerkleNode(combined_hash, left, left)
                else:
                    combined_hash = self._compute_hash(left.hash_value + right.hash_value)
                    parent = MerkleNode(combined_hash, left, right)
                    
                next_level.append(parent)
            
            nodes = next_level
            
        return nodes[0] if nodes else None
    
    def get_merkle_root(self) -> Optional[str]:
        """Get the Merkle root hash of all traces."""
        tree = self.build_merkle_tree()
        return tree.hash_value if tree else None
    
    def generate_inclusion_proof(self, trace_index: int) -> List[Dict[str, Any]]:
        """Generate inclusion proof for a specific trace."""
        if trace_index >= len(self.traces):
            raise ValueError(f"Trace index {trace_index} out of bounds")
            
        tree = self.build_merkle_tree()
        if not tree:
            return []
            
        # Find path from leaf to root
        proof_path = []
        current_nodes = [MerkleNode(trace["hash"], data=trace) for trace in self.traces]
        target_hash = self.traces[trace_index]["hash"]
        
        while len(current_nodes) > 1:
            next_level = []
            target_found = False
            
            for i in range(0, len(current_nodes), 2):
                left = current_nodes[i]
                right = current_nodes[i + 1] if i + 1 < len(current_nodes) else None
                
                # Check if target is in this pair
                if left.hash_value == target_hash or (right and right.hash_value == target_hash):
                    sibling = right if left.hash_value == target_hash else left
                    proof_path.append({
                        "hash": sibling.hash_value if sibling else left.hash_value,
                        "position": "right" if left.hash_value == target_hash else "left"
                    })
                    target_found = True
                    
                    if right is None:
                        combined_hash = self._compute_hash(left.hash_value + left.hash_value)
                        parent = MerkleNode(combined_hash, left, left)
                    else:
                        combined_hash = self._compute_hash(left.hash_value + right.hash_value)
                        parent = MerkleNode(combined_hash, left, right)
                    target_hash = parent.hash_value
                else:
                    if right is None:
                        combined_hash = self._compute_hash(left.hash_value + left.hash_value)
                        parent = MerkleNode(combined_hash, left, left)
                    else:
                        combined_hash = self._compute_hash(left.hash_value + right.hash_value)
                        parent = MerkleNode(combined_hash, left, right)
                        
                next_level.append(parent)
            
            current_nodes = next_level
            
        return proof_path
    
    def verify_inclusion_proof(self, trace_hash: str, proof_path: List[Dict[str, Any]], 
                             expected_root: str) -> bool:
        """Verify that a trace is included in the Merkle tree."""
        current_hash = trace_hash
        
        for proof_step in proof_path:
            sibling_hash = proof_step["hash"]
            position = proof_step["position"]
            
            if position == "left":
                current_hash = self._compute_hash(sibling_hash + current_hash)
            else:
                current_hash = self._compute_hash(current_hash + sibling_hash)
                
        return current_hash == expected_root
    
    def replay_traces(self) -> Tuple[List[str], str]:
        """Replay all traces and return individual hashes and final root."""
        hashes = []
        for trace in self.traces:
            # Recompute hash for verification
            trace_copy = {k: v for k, v in trace.items() if k != "hash"}
            computed_hash = self._compute_hash(trace_copy)
            hashes.append(computed_hash)
            
        root = self.get_merkle_root()
        return hashes, root
    
    def detect_tampering(self, expected_root: str = None) -> bool:
        """Detect if traces have been tampered with."""
        current_root = self.get_merkle_root()
        
        if expected_root:
            return current_root != expected_root
            
        # Verify internal consistency
        for i, trace in enumerate(self.traces):
            trace_copy = {k: v for k, v in trace.items() if k != "hash"}
            computed_hash = self._compute_hash(trace_copy)
            if computed_hash != trace["hash"]:
                logger.warning(f"Tampering detected in trace {i}")
                return True
                
        return False
    
    def generate_certificate(self) -> Dict[str, Any]:
        """Generate audit certificate with pass/fail status and proofs."""
        merkle_root = self.get_merkle_root()
        tamper_detected = self.detect_tampering()
        
        # Generate proofs for all traces
        proofs = []
        for i in range(len(self.traces)):
            try:
                proof = self.generate_inclusion_proof(i)
                proofs.append({
                    "trace_index": i,
                    "proof_path": proof,
                    "verified": self.verify_inclusion_proof(
                        self.traces[i]["hash"], proof, merkle_root
                    ) if merkle_root else False
                })
            except Exception as e:
                logger.error(f"Failed to generate proof for trace {i}: {e}")
        
        certificate = {
            "pass": not tamper_detected and all(p["verified"] for p in proofs),
            "merkle_root": merkle_root,
            "proofs": len(proofs),
            "tamper_detected": tamper_detected,
            "hash_function": HASH_NAME,
            "total_traces": len(self.traces),
            "timestamp": __import__("datetime").datetime.now().isoformat()
        }
        
        return certificate
    
    def save_certificate(self, filename: str = "tc_certificate.json") -> None:
        """Save audit certificate to JSON file."""
        certificate = self.generate_certificate()
        with open(filename, 'w') as f:
            json.dump(certificate, f, indent=2)
        logger.info(f"Certificate saved to {filename}")
    
    def load_traces_from_db(self) -> None:
        """Load existing traces from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT timestamp, operation, data, hash_value
                FROM audit_traces ORDER BY id
            """)
            
            self.traces = []
            for row in cursor.fetchall():
                trace = {
                    "timestamp": row[0],
                    "operation": row[1],
                    "data": json.loads(row[2]),
                    "hash": row[3]
                }
                self.traces.append(trace)


def main():
    """CLI interface for audit trail operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Audit Trail Merkle Tree System")
    parser.add_argument("--db", default="audit_trail.db", help="Database path")
    parser.add_argument("--operation", choices=["add", "verify", "certificate"], 
                       help="Operation to perform")
    parser.add_argument("--trace-data", help="JSON data for new trace")
    parser.add_argument("--trace-operation", help="Operation name for new trace")
    parser.add_argument("--output", default="tc_certificate.json", 
                       help="Certificate output file")
    
    args = parser.parse_args()
    
    audit = AuditTrail(args.db)
    audit.load_traces_from_db()
    
    if args.operation == "add" and args.trace_data and args.trace_operation:
        data = json.loads(args.trace_data)
        trace_id = audit.add_trace(args.trace_operation, data)
        print(f"Added trace {trace_id}")
        
    elif args.operation == "verify":
        tamper_detected = audit.detect_tampering()
        print(f"Tampering detected: {tamper_detected}")
        
    elif args.operation == "certificate":
        audit.save_certificate(args.output)
        print(f"Certificate saved to {args.output}")


if __name__ == "__main__":
    main()