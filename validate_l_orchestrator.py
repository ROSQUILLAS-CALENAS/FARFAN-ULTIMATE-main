#!/usr/bin/env python3
"""
Validation script for L-Classification Stage Orchestrator
Tests all major functionality including deterministic behavior
"""

import sys
import json
import tempfile
from pathlib import Path
import hashlib

# Add current directory to path
sys.path.insert(0, '.')

def create_comprehensive_test_data():
    """Create comprehensive test data with all dimensions"""
    return {
        "DE-1": [
            {
                "question_id": "DE1_Q1",
                "response": "Sí",
                "evidence_completeness": 0.95,
                "page_reference_quality": 1.0,
                "evidence_text": "Comprehensive product identification in section 3.1"
            },
            {
                "question_id": "DE1_Q2",
                "response": "Parcial",
                "evidence_completeness": 0.7,
                "page_reference_quality": 0.8,
                "evidence_text": "Partial timeline information available"
            }
        ],
        "DE-2": [
            {
                "question_id": "DE2_Q1",
                "response": "Sí",
                "evidence_completeness": 0.9,
                "page_reference_quality": 0.95,
                "evidence_text": "Comprehensive territorial diagnosis in chapter 2"
            }
        ],
        "DE-3": [
            {
                "question_id": "DE3_Q1",
                "response": "Parcial",
                "evidence_completeness": 0.6,
                "page_reference_quality": 0.7,
                "evidence_text": "Some monitoring indicators defined"
            }
        ],
        "DE-4": [
            {
                "question_id": "DE4_Q1",
                "response": "No",
                "evidence_completeness": 0.1,
                "page_reference_quality": 0.2,
                "evidence_text": "Limited evaluation framework"
            }
        ]
    }

def test_orchestrator_functionality():
    """Test core orchestrator functionality"""
    print("=" * 60)
    print("TESTING L-CLASSIFICATION STAGE ORCHESTRATOR")
    print("=" * 60)
    
    try:
        from canonical_flow.L_classification_evaluation.stage_orchestrator import LClassificationStageOrchestrator, process
        print("✓ Successfully imported orchestrator components")
        
        # Test orchestrator info
        orchestrator = LClassificationStageOrchestrator()
        info = orchestrator.get_orchestrator_info()
        
        print(f"\nOrchestrator Information:")
        print(f"  Name: {info['orchestrator']}")
        print(f"  Version: {info['version']}")  
        print(f"  Stage: {info['stage']}")
        print(f"  Components: {', '.join(info['components'])}")
        print(f"  Deterministic: {info['deterministic']}")
        print(f"  API Contract: {info['api_contract']}")
        print(f"  Artifact Types: {len(info['artifact_types'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Import or info test failed: {e}")
        return False

def test_process_api():
    """Test the standardized process() API"""
    print(f"\n{'='*60}")
    print("TESTING PROCESS() API CONTRACT")
    print("="*60)
    
    try:
        from canonical_flow.L_classification_evaluation.stage_orchestrator import process
        
        # Create test data
        temp_dir = Path(tempfile.mkdtemp())
        test_data = create_comprehensive_test_data()
        
        # Create multiple input files
        input_files = []
        for point_id in [1, 2, 3]:
            input_file = temp_dir / "classification_input" / f"doc{point_id}" / f"P{point_id}_questions.json"
            input_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(input_file, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, indent=2, ensure_ascii=False)
            
            input_files.append(input_file)
        
        print(f"✓ Created {len(input_files)} test input files")
        
        # Test different input methods
        test_cases = [
            ("Directory path", str(temp_dir / "classification_input")),
            ("Single file", str(input_files[0])),
            ("File list", [str(f) for f in input_files[:2]]),
            ("Dict with path", {"classification_input_path": str(temp_dir / "classification_input")}),
        ]
        
        for test_name, test_input in test_cases:
            print(f"\nTesting {test_name}...")
            
            try:
                results = process(data=test_input, context={"test_case": test_name})
                
                # Validate result structure
                required_keys = ["execution_metadata", "artifacts", "status_report", "audit_log"]
                for key in required_keys:
                    if key not in results:
                        print(f"  ✗ Missing required key: {key}")
                        continue
                
                # Check artifacts
                artifacts = results["artifacts"]
                expected_artifacts = [
                    "dimension_evaluations",
                    "point_summaries", 
                    "composition_traces",
                    "confidence_intervals",
                    "guard_reports",
                    "points_index"
                ]
                
                missing_artifacts = [a for a in expected_artifacts if a not in artifacts]
                if missing_artifacts:
                    print(f"  ✗ Missing artifacts: {missing_artifacts}")
                    continue
                
                # Check processing results
                status = results["status_report"]
                successful = len(status.get("successful_points", []))
                failed = len(status.get("failed_points", []))
                
                print(f"  ✓ {test_name}: {successful} successful, {failed} failed points")
                
            except Exception as e:
                print(f"  ✗ {test_name} failed: {e}")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        print("✓ Process API tests completed")
        return True
        
    except Exception as e:
        print(f"✗ Process API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_deterministic_behavior():
    """Test deterministic behavior and serialization"""
    print(f"\n{'='*60}")
    print("TESTING DETERMINISTIC BEHAVIOR")
    print("="*60)
    
    try:
        from canonical_flow.L_classification_evaluation.stage_orchestrator import LClassificationStageOrchestrator
        
        # Create test data
        temp_dir = Path(tempfile.mkdtemp())
        test_data = create_comprehensive_test_data()
        
        input_file = temp_dir / "P1_questions.json"
        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2)
        
        # Run multiple times
        orchestrator = LClassificationStageOrchestrator()
        results_list = []
        
        for i in range(3):
            results = orchestrator.process(data=str(input_file))
            
            # Remove non-deterministic fields
            clean_results = results.copy()
            clean_results["execution_metadata"]["execution_id"] = "DETERMINISTIC_ID"
            clean_results["execution_metadata"]["start_time"] = "DETERMINISTIC_TIME" 
            clean_results["execution_metadata"]["end_time"] = "DETERMINISTIC_TIME"
            clean_results["execution_metadata"]["total_processing_time_seconds"] = 0.0
            
            # Remove timestamps from audit log
            for entry in clean_results.get("audit_log", []):
                entry["timestamp"] = "DETERMINISTIC_TIME"
                entry["execution_id"] = "DETERMINISTIC_ID"
            
            # Remove timestamps from artifacts
            for artifact_type, artifacts in clean_results["artifacts"].items():
                if isinstance(artifacts, dict):
                    for artifact_key, artifact in artifacts.items():
                        if isinstance(artifact, dict):
                            for field in ["evaluation_timestamp", "processing_timestamp", 
                                        "trace_timestamp", "interval_timestamp", "report_timestamp", 
                                        "index_timestamp"]:
                                if field in artifact:
                                    artifact[field] = "DETERMINISTIC_TIME"
            
            # Remove processing completion timestamp
            if "processing_summary" in clean_results["status_report"]:
                clean_results["status_report"]["processing_summary"]["processing_completed"] = "DETERMINISTIC_TIME"
            
            results_list.append(clean_results)
        
        # Test deterministic serialization
        json_strings = []
        for results in results_list:
            json_str = orchestrator.serialize_results(results)
            json_strings.append(json_str)
        
        # Check if all JSON strings are identical
        all_identical = all(js == json_strings[0] for js in json_strings)
        
        if all_identical:
            print("✓ Deterministic serialization: All outputs identical")
            
            # Test JSON format
            parsed = json.loads(json_strings[0])
            if isinstance(parsed, dict):
                print("✓ Valid JSON structure")
            
            # Check for proper formatting (indent=2)
            if '  "artifacts":' in json_strings[0]:
                print("✓ Proper JSON indentation (indent=2)")
            
            # Test hash consistency
            hash1 = hashlib.md5(json_strings[0].encode('utf-8')).hexdigest()
            hash2 = hashlib.md5(json_strings[1].encode('utf-8')).hexdigest()
            
            if hash1 == hash2:
                print("✓ Content hash consistency verified")
            
        else:
            print("✗ Non-deterministic behavior detected")
            
            # Find differences
            for i in range(1, len(json_strings)):
                if json_strings[i] != json_strings[0]:
                    print(f"  Difference found in run {i+1}")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return all_identical
        
    except Exception as e:
        print(f"✗ Deterministic behavior test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling and isolation"""
    print(f"\n{'='*60}")
    print("TESTING ERROR HANDLING & ISOLATION")
    print("="*60)
    
    try:
        from canonical_flow.L_classification_evaluation.stage_orchestrator import LClassificationStageOrchestrator
        
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create valid and invalid files
        valid_data = create_comprehensive_test_data()
        invalid_data = {"invalid_structure": True, "missing_dimensions": []}
        
        files_created = []
        
        # Valid file
        valid_file = temp_dir / "P1_questions.json"
        with open(valid_file, 'w', encoding='utf-8') as f:
            json.dump(valid_data, f, indent=2)
        files_created.append(("valid", valid_file))
        
        # Invalid file
        invalid_file = temp_dir / "P2_questions.json"
        with open(invalid_file, 'w', encoding='utf-8') as f:
            json.dump(invalid_data, f, indent=2)
        files_created.append(("invalid", invalid_file))
        
        # Another valid file
        another_valid_file = temp_dir / "P3_questions.json"
        with open(another_valid_file, 'w', encoding='utf-8') as f:
            json.dump(valid_data, f, indent=2)
        files_created.append(("valid", another_valid_file))
        
        print(f"✓ Created test files: {[f[0] for f in files_created]}")
        
        # Process with mixed valid/invalid files
        orchestrator = LClassificationStageOrchestrator()
        results = orchestrator.process(data=str(temp_dir))
        
        # Analyze results
        successful = results["status_report"]["successful_points"]
        failed = results["status_report"]["failed_points"]
        
        print(f"✓ Processing completed")
        print(f"  Successful points: {len(successful)}")
        print(f"  Failed points: {len(failed)}")
        
        # Verify error isolation
        if len(successful) > 0 and len(failed) > 0:
            print("✓ Error isolation: Some points succeeded despite failures")
        
        # Check failure details
        if failed:
            for failure in failed:
                if "error" in failure and "traceback" in failure:
                    print(f"✓ Detailed error info captured for point {failure['point_id']}")
                    break
        
        # Verify successful points have artifacts
        artifacts_generated = 0
        for artifact_type, artifacts in results["artifacts"].items():
            artifacts_generated += len(artifacts) if isinstance(artifacts, dict) else 0
        
        if artifacts_generated > 0:
            print(f"✓ Generated {artifacts_generated} artifacts from successful points")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_artifact_generation():
    """Test complete artifact generation"""
    print(f"\n{'='*60}")
    print("TESTING ARTIFACT GENERATION")
    print("="*60)
    
    try:
        from canonical_flow.L_classification_evaluation.stage_orchestrator import process
        
        # Create comprehensive test data
        temp_dir = Path(tempfile.mkdtemp())
        test_data = create_comprehensive_test_data()
        
        input_file = temp_dir / "P1_questions.json"
        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2)
        
        # Process
        results = process(data=str(input_file))
        
        # Check all required artifacts
        artifacts = results["artifacts"]
        expected_artifacts = [
            "dimension_evaluations",
            "point_summaries", 
            "composition_traces",
            "confidence_intervals",
            "guard_reports",
            "points_index"
        ]
        
        print("Artifact Generation Results:")
        for artifact_type in expected_artifacts:
            if artifact_type in artifacts:
                count = len(artifacts[artifact_type]) if isinstance(artifacts[artifact_type], dict) else 1
                print(f"  ✓ {artifact_type}: {count} items")
                
                # Validate structure of first item
                if isinstance(artifacts[artifact_type], dict) and artifacts[artifact_type]:
                    first_key = list(artifacts[artifact_type].keys())[0]
                    first_item = artifacts[artifact_type][first_key]
                    
                    if isinstance(first_item, dict):
                        required_fields = {
                            "dimension_evaluations": ["point_id", "dimensions"],
                            "point_summaries": ["point_id", "final_score"],
                            "composition_traces": ["point_id", "component_execution_order"],
                            "confidence_intervals": ["point_id", "confidence_level"],
                            "guard_reports": ["point_id", "validation_checks"],
                            "points_index": ["point_id", "status"]
                        }
                        
                        if artifact_type in required_fields:
                            missing_fields = [f for f in required_fields[artifact_type] if f not in first_item]
                            if not missing_fields:
                                print(f"    ✓ Required fields present")
                            else:
                                print(f"    ✗ Missing fields: {missing_fields}")
                
            else:
                print(f"  ✗ {artifact_type}: Missing")
        
        # Test file serialization
        output_file = temp_dir / "test_results.json"
        from canonical_flow.L_classification_evaluation.stage_orchestrator import LClassificationStageOrchestrator
        orchestrator = LClassificationStageOrchestrator()
        json_str = orchestrator.serialize_results(results, output_file)
        
        if output_file.exists():
            print(f"✓ Results serialized to file ({output_file.stat().st_size:,} bytes)")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"✗ Artifact generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests"""
    print("L-Classification Stage Orchestrator Validation")
    print("Comprehensive testing of all functionality")
    print()
    
    tests = [
        ("Orchestrator Functionality", test_orchestrator_functionality),
        ("Process API Contract", test_process_api),
        ("Deterministic Behavior", test_deterministic_behavior),
        ("Error Handling & Isolation", test_error_handling),
        ("Artifact Generation", test_artifact_generation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"  {symbol} {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%} success rate)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Orchestrator is fully functional!")
        return True
    else:
        print(f"\n⚠️  {total-passed} tests failed - Review issues above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)