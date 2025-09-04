#!/usr/bin/env python3
"""
Test script for StageTransitionController module
"""

import json
import os
import tempfile
from pathlib import Path

def test_stage_transition_controller():
    """Test basic functionality of the stage transition controller"""
    
    try:
        # Import the module
        from stage_transition_controller import (
            StageTransitionController, 
            StageState, 
            ValidationResult,
            create_controller,
            validate_pipeline_sequence
        )
        print("✓ Module imported successfully")
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = os.path.join(temp_dir, "artifacts")
            log_file = os.path.join(temp_dir, "test_transitions.log")
            
            # Create controller
            controller = create_controller(artifact_dir, log_file)
            print("✓ Controller created successfully")
            
            # Test initial state
            assert controller.get_current_stage() is None
            print("✓ Initial state is None")
            
            # Test valid first transition
            result = controller.validate_transition(None, StageState.INGESTION_PREPARATION, require_artifacts=False)
            assert result.result == ValidationResult.SUCCESS
            print("✓ Valid first transition (None → I)")
            
            # Test invalid transition (skipping stages)
            result = controller.validate_transition(StageState.INGESTION_PREPARATION, StageState.KNOWLEDGE_EXTRACTION, require_artifacts=False)
            assert result.result == ValidationResult.INVALID_TRANSITION
            assert len(result.remediation_suggestions) > 0
            print("✓ Invalid transition detected (I → K)")
            
            # Test valid sequential transitions
            valid_sequence = [
                (StageState.INGESTION_PREPARATION, StageState.CONTEXT_CONSTRUCTION),
                (StageState.CONTEXT_CONSTRUCTION, StageState.KNOWLEDGE_EXTRACTION),
                (StageState.KNOWLEDGE_EXTRACTION, StageState.ANALYSIS_NLP)
            ]
            
            current = StageState.INGESTION_PREPARATION
            for from_stage, to_stage in valid_sequence:
                result = controller.validate_transition(from_stage, to_stage, require_artifacts=False)
                assert result.result == ValidationResult.SUCCESS
                current = to_stage
            print("✓ Valid sequence transitions (I→X→K→A)")
            
            # Test artifact registration
            test_file = os.path.join(temp_dir, "test_artifact.txt")
            with open(test_file, 'w') as f:
                f.write("Test artifact content")
            
            artifact = controller.register_artifact(StageState.ANALYSIS_NLP, test_file, {"type": "test"})
            assert artifact.verify_existence()
            assert artifact.verify_integrity()
            print("✓ Artifact registration and verification")
            
            # Test compliance report
            report = controller.get_compliance_report()
            assert "timestamp" in report
            assert "current_stage" in report
            assert "artifact_summary" in report
            assert "import_violations" in report
            print("✓ Compliance report generation")
            
            # Test report export
            report_file = os.path.join(temp_dir, "test_report.json")
            controller.export_compliance_report(report_file)
            assert os.path.exists(report_file)
            
            with open(report_file) as f:
                exported_report = json.load(f)
                assert exported_report["total_transitions"] > 0
            print("✓ Report export functionality")
            
            # Test pipeline sequence validation
            canonical_stages = [
                StageState.INGESTION_PREPARATION,
                StageState.CONTEXT_CONSTRUCTION,
                StageState.KNOWLEDGE_EXTRACTION,
                StageState.ANALYSIS_NLP,
                StageState.CLASSIFICATION_EVALUATION,
                StageState.SEARCH_RETRIEVAL,
                StageState.ORCHESTRATION_CONTROL,
                StageState.AGGREGATION_REPORTING,
                StageState.INTEGRATION_STORAGE,
                StageState.SYNTHESIS_OUTPUT
            ]
            
            # Reset controller for clean test
            controller.reset_state()
            results = validate_pipeline_sequence(controller, canonical_stages, require_artifacts=False)
            
            assert len(results) == len(canonical_stages)
            # Check that all transitions were successful
            failed_results = [r for r in results if r.result != ValidationResult.SUCCESS]
            if failed_results:
                print(f"Failed transitions: {[(r.from_stage, r.to_stage, r.message) for r in failed_results]}")
                assert False, f"Expected all transitions to succeed, but {len(failed_results)} failed"
            
            print("✓ Complete canonical sequence validation")
            
        print("\n=== ALL TESTS PASSED ===")
        print("StageTransitionController module is functioning correctly")
        
        # Print feature summary
        print("\nImplemented Features:")
        print("- Finite State Automaton with canonical I→X→K→A→L→R→O→G→T→S sequence")
        print("- Artifact verification with SHA-256 checksums")
        print("- Hash continuity detection across processing chain")
        print("- Runtime import monitoring for backward dependency detection")
        print("- Comprehensive logging with structured output")
        print("- Detailed error messages with remediation suggestions")
        print("- Compliance reporting for debugging and auditing")
        print("- Context manager for stage execution monitoring")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_stage_transition_controller()
    exit(0 if success else 1)