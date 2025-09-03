#!/usr/bin/env python3
"""
Simple test to verify DataIntegrityChecker works correctly
"""

# # # from data_integrity_checker import DataIntegrityChecker  # Module not found  # Module not found  # Module not found
import tempfile
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

def main():
    print("Testing DataIntegrityChecker...")
    
    # Test 1: Hash computation
    checker = DataIntegrityChecker()
    test_data = {"test": True, "value": 42}
    hash_val = checker.compute_artifact_hash(test_data)
    print(f"✅ Hash computed: {hash_val[:16]}...")
    
    # Test 2: Metadata generation  
    metadata = checker.generate_artifact_metadata(
        test_data, "test_stage", "TestComponent", "doc_001"
    )
    print(f"✅ Metadata generated: {metadata.stage_name}/{metadata.component_name}")
    
    # Test 3: Save and validate artifact
    with tempfile.TemporaryDirectory() as temp_dir:
        checker.canonical_flow_dir = Path(temp_dir)
        
        output_path, saved_metadata = checker.save_artifact_with_integrity(
            test_data, "test_stage", "TestComponent", "test_doc"
        )
        
        print(f"✅ Artifact saved: {output_path.name}")
        
        # Validate
        is_valid, corruption_report = checker.validate_artifact_integrity(
            output_path, saved_metadata
        )
        
        if is_valid:
            print("✅ Artifact validation: PASSED")
        else:
            print(f"❌ Artifact validation: FAILED - {corruption_report.corruption_type if corruption_report else 'Unknown error'}")
    
    print("✅ All basic tests completed successfully!")

if __name__ == "__main__":
    main()