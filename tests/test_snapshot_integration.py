#!/usr/bin/env python3
"""
Integration test for complete snapshot forensics system.

Tests the full workflow:
1. Mount a snapshot
2. Run guarded execution  
3. Generate certificate
4. Verify forensic integrity
"""

import subprocess
import sys
import json
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

# Add current dir to path
sys.path.insert(0, str(Path(__file__).parent))

# # # from snapshot_manager import mount_snapshot, get_current_snapshot_id  # Module not found  # Module not found  # Module not found
# # # from tools.certificate_generator import CertificateGenerator  # Module not found  # Module not found  # Module not found


def test_integration():
    """Test complete forensic workflow."""
    print("=== SNAPSHOT FORENSICS INTEGRATION TEST ===")
    
    # Step 1: Mount snapshot
    print("1. Mounting snapshot...")
    test_state = {
        "corpus": {"documents": ["doc1", "doc2"], "version": "1.0"},
        "indices": {"embedding_dim": 384, "type": "dense"},
        "standards": {"compliance": "v2.1", "schema": "json"}
    }
    
    sigma = mount_snapshot(test_state)
    print(f"   Mounted σ: {sigma[:16]}...")
    
    # Step 2: Test snapshot guard with valid snapshot
    print("2. Testing snapshot guard with valid σ...")
    result = subprocess.run([
        "python3", "tools/snapshot_guard.py", "echo", "Guarded execution successful"
    ], env={"PYTHONPATH": "."}, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("   ✓ Guard allowed execution")
    else:
        print("   ✗ Guard blocked execution")
        print("   STDERR:", result.stderr)
    
    # Step 3: Generate certificate
    print("3. Generating SC certificate...")
    generator = CertificateGenerator()
    
    def test_func(value: str):
        return {"processed": value, "length": len(value)}
    
    certificate = generator.generate_certificate(
        test_func=test_func,
        inputs={"value": "test_data"}
    )
    
    generator.save_certificate("test_sc_certificate.json")
    
    # Step 4: Verify certificate contents
    print("4. Verifying certificate...")
    print(f"   Pass Status: {certificate['pass']}")
    print(f"   Replay Equal: {certificate['replay_equal']}")
    print(f"   Standards Hash: {certificate['sigma']['standards_hash'][:16]}...")
    print(f"   Corpus Hash: {certificate['sigma']['corpus_hash'][:16]}...")
    print(f"   Index Hash: {certificate['sigma']['index_hash'][:16]}...")
    
    if certificate['pass']:
        print("✓ INTEGRATION TEST PASSED")
        return True
    else:
        print("✗ INTEGRATION TEST FAILED")
        return False


if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)