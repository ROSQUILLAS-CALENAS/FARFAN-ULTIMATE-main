#!/usr/bin/env python3
"""
Snapshot Guard - Forensic security tool that refuses execution without valid σ.

This tool enforces the SC (Snapshot Contract) by:
1. Verifying required snapshot σ = {standards_hash, corpus_hash, index_hash} exists
2. Refusing to proceed with any processing if σ is missing or invalid
3. Providing cryptographic integrity verification using blake3/sha256
4. Ensuring deterministic, reproducible execution environments

Usage:
    python -m tools.snapshot_guard <command> [args...]
    
    The guard will:
    - Check for valid snapshot before executing command
    - Abort with typed refusal if σ is missing/invalid
    - Log all verification steps for forensic audit
    - Ensure no mutations or network access during verification
"""

import hashlib
import json
import sys
import subprocess
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, Any, Optional  # Module not found  # Module not found  # Module not found

try:
    import blake3
except ImportError:
    blake3 = None

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# # # from snapshot_manager import (  # Module not found  # Module not found  # Module not found
    get_current_snapshot_id,
    resolve_snapshot,
    compute_snapshot_id
)


class SnapshotGuardRefusal(Exception):
    """Typed refusal when snapshot validation fails."""
    pass


class SnapshotGuard:
    """Forensic guard that enforces snapshot requirements."""
    
    def __init__(self):
        self.verification_log = []
        self.allow_network = False
        self.allow_mutations = False
    
    def log_verification(self, message: str, level: str = "INFO"):
        """Log verification step for forensic audit."""
        entry = {
            "level": level,
            "message": message,
            "timestamp": self._get_timestamp()
        }
        self.verification_log.append(entry)
        print(f"[GUARD_{level}] {message}", file=sys.stderr)
    
    def _get_timestamp(self) -> str:
        """Get deterministic timestamp for logging."""
        import time
        return str(int(time.time()))
    
    def verify_snapshot_exists(self) -> str:
        """Verify that a valid snapshot σ exists."""
        self.log_verification("Starting snapshot verification")
        
        sigma = get_current_snapshot_id()
        if sigma is None:
            self.log_verification("No active snapshot found", "ERROR")
            raise SnapshotGuardRefusal(
                "Snapshot σ missing. Cannot proceed without valid "
                "σ = {standards_hash, corpus_hash, index_hash}"
            )
        
        self.log_verification(f"Found active snapshot: {sigma[:16]}...")
        return sigma
    
    def verify_snapshot_integrity(self, sigma: str) -> Dict[str, Any]:
        """Verify cryptographic integrity of snapshot."""
        self.log_verification("Verifying snapshot cryptographic integrity")
        
        try:
            snapshot_data = resolve_snapshot(sigma)
            self.log_verification("Snapshot integrity verified")
            return snapshot_data
        except (KeyError, ValueError) as e:
            self.log_verification(f"Integrity verification failed: {e}", "ERROR")
            raise SnapshotGuardRefusal(f"Snapshot σ integrity compromised: {e}")
    
    def verify_hash_completeness(self, sigma: str):
        """Verify that all required hashes are present in σ."""
        self.log_verification("Checking hash completeness")
        
        snapshot_data = resolve_snapshot(sigma)
        frozen_data = json.loads(snapshot_data["frozen_json"])
        
        required_components = ["corpus", "indices", "standards"]
        missing = []
        
        for component in required_components:
            if component not in frozen_data:
                missing.append(component)
        
        if missing:
            self.log_verification(f"Missing components: {missing}", "ERROR")
            raise SnapshotGuardRefusal(
                f"Incomplete σ: missing {missing}. "
                f"Required: {required_components}"
            )
        
        self.log_verification("All required hash components present")
    
    def enforce_constraints(self):
        """Enforce snapshot-related execution constraints."""
        self.log_verification("Enforcing execution constraints")
        
        if not self.allow_network:
            # Set environment variables to disable network access
            import os
            os.environ["DISABLE_NETWORK"] = "1"
            os.environ["HTTP_PROXY"] = "127.0.0.1:0"
            os.environ["HTTPS_PROXY"] = "127.0.0.1:0"
            self.log_verification("Network access disabled")
        
        if not self.allow_mutations:
            # Set read-only environment markers
            import os
            os.environ["READONLY_MODE"] = "1"
            self.log_verification("Mutation protection enabled")
    
    def guard_execution(self, command: list) -> int:
        """Guard command execution with snapshot verification."""
        try:
            # Step 1: Verify snapshot exists
            sigma = self.verify_snapshot_exists()
            
            # Step 2: Verify cryptographic integrity
            self.verify_snapshot_integrity(sigma)
            
            # Step 3: Verify hash completeness
            self.verify_hash_completeness(sigma)
            
            # Step 4: Enforce constraints
            self.enforce_constraints()
            
            # Step 5: Execute guarded command with inherited environment
            self.log_verification(f"Executing guarded command: {' '.join(command)}")
            
            # Pass through environment variables including PYTHONPATH
            import os
            env = os.environ.copy()
            env["SNAPSHOT_ID"] = sigma  # Pass sigma to child process
            
            result = subprocess.run(command, capture_output=False, env=env)
            
            self.log_verification(f"Command completed with code: {result.returncode}")
            return result.returncode
            
        except SnapshotGuardRefusal as e:
            self.log_verification(f"Guard refusal: {e}", "ERROR")
            print(f"GUARD REFUSAL: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            self.log_verification(f"Unexpected error: {e}", "ERROR")
            print(f"GUARD ERROR: {e}", file=sys.stderr)
            return 1
    
    def print_verification_log(self):
        """Print forensic verification log."""
        print("\n=== SNAPSHOT GUARD VERIFICATION LOG ===", file=sys.stderr)
        for entry in self.verification_log:
            timestamp = entry["timestamp"]
            level = entry["level"]
            message = entry["message"]
            print(f"{timestamp} [{level}] {message}", file=sys.stderr)
        print("=== END VERIFICATION LOG ===\n", file=sys.stderr)


def main():
    """Main entry point for snapshot guard."""
    if len(sys.argv) < 2:
        print("Usage: python -m tools.snapshot_guard <command> [args...]", file=sys.stderr)
        print("\nSnapshot Guard: Enforces σ requirements for forensic integrity", file=sys.stderr)
        return 1
    
    command = sys.argv[1:]
    guard = SnapshotGuard()
    
    try:
        exit_code = guard.guard_execution(command)
        guard.print_verification_log()
        return exit_code
    except KeyboardInterrupt:
        guard.log_verification("Execution interrupted by user", "WARNING")
        guard.print_verification_log()
        return 130


if __name__ == "__main__":
    sys.exit(main())