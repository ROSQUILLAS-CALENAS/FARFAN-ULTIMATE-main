#!/usr/bin/env python3
"""Run integrity tests safely"""

import sys
import traceback

def run_simple_test():
    """Run the simple integrity test"""
    try:
# # #         from data_integrity_checker import DataIntegrityChecker  # Module not found  # Module not found  # Module not found
        import tempfile
# # #         from pathlib import Path  # Module not found  # Module not found  # Module not found
        
        print("🔧 Testing DataIntegrityChecker...")
        
        # Test 1: Hash computation
        checker = DataIntegrityChecker()
        test_data = {"test": True, "value": 42}
        hash_val = checker.compute_artifact_hash(test_data)
        print(f"✅ Hash computed: {hash_val[:16]}...")
        assert len(hash_val) == 64, "Hash should be 64 characters"
        
        # Test 2: Metadata generation  
        metadata = checker.generate_artifact_metadata(
            test_data, "test_stage", "TestComponent", "doc_001"
        )
        print(f"✅ Metadata generated: {metadata.stage_name}/{metadata.component_name}")
        assert metadata.stage_name == "test_stage"
        assert metadata.component_name == "TestComponent"
        
        # Test 3: Save and validate artifact
        with tempfile.TemporaryDirectory() as temp_dir:
            checker.canonical_flow_dir = Path(temp_dir)
            
            output_path, saved_metadata = checker.save_artifact_with_integrity(
                test_data, "test_stage", "TestComponent", "test_doc"
            )
            
            print(f"✅ Artifact saved: {output_path.name}")
            assert output_path.exists(), "Artifact file should exist"
            
            # Validate
            is_valid, corruption_report = checker.validate_artifact_integrity(
                output_path, saved_metadata
            )
            
            if is_valid:
                print("✅ Artifact validation: PASSED")
            else:
                print(f"❌ Artifact validation: FAILED - {corruption_report.corruption_type if corruption_report else 'Unknown error'}")
                if corruption_report:
                    print(f"   Details: {corruption_report.__dict__}")
                return False
        
        print("✅ All basic tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        traceback.print_exc()
        return False

def run_corruption_test():
    """Test corruption detection"""
    try:
# # #         from data_integrity_checker import DataIntegrityChecker, CorruptionType  # Module not found  # Module not found  # Module not found
        import tempfile
        import json
# # #         from pathlib import Path  # Module not found  # Module not found  # Module not found
        
        print("\n🔧 Testing corruption detection...")
        
        checker = DataIntegrityChecker()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checker.canonical_flow_dir = Path(temp_dir)
            
            # Create valid artifact
            test_data = {"original": True, "value": 100}
            output_path, metadata = checker.save_artifact_with_integrity(
                test_data, "test_stage", "TestComponent", "corruption_test"
            )
            
            print("✅ Valid artifact created")
            
            # Corrupt the artifact by modifying the data portion
            with open(output_path, 'r') as f:
                artifact_data = json.load(f)
            
            # Modify the data
            artifact_data['data']['corrupted'] = True
            artifact_data['data']['value'] = 999
            
            with open(output_path, 'w') as f:
                json.dump(artifact_data, f, indent=2)
            
            print("🔧 Artifact corrupted")
            
            # Validate and expect corruption
            is_valid, corruption_report = checker.validate_artifact_integrity(
                output_path, metadata
            )
            
            if not is_valid:
                print("✅ Corruption detected successfully")
                print(f"   Type: {corruption_report.corruption_type}")
                assert corruption_report.corruption_type == CorruptionType.HASH_MISMATCH
                return True
            else:
                print("❌ Failed to detect corruption")
                return False
                
    except Exception as e:
        print(f"❌ Corruption test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Data Integrity Checker Tests")
    print("=" * 60)
    
    success_count = 0
    total_tests = 2
    
    # Run simple test
    if run_simple_test():
        success_count += 1
    
    # Run corruption test
    if run_corruption_test():
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("✅ All tests PASSED")
        return True
    else:
        print("❌ Some tests FAILED") 
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)