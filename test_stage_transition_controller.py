#!/usr/bin/env python3
"""
Test script for Stage Transition Controller
"""

def test_basic_functionality():
    """Test basic controller functionality"""
    try:
        import stage_transition_controller as stc
        
        print("✓ Module imported successfully")
        
        # Test enum
        stages = stc.PipelineStage.get_canonical_sequence()
        assert len(stages) == 10, f"Expected 10 stages, got {len(stages)}"
        print(f"✓ Canonical sequence has {len(stages)} stages")
        
        # Test controller creation
        controller = stc.StageTransitionController()
        print("✓ Controller instance created")
        
        # Test initial state
        current = controller.get_current_stage()
        assert current is None, f"Expected None, got {current}"
        print("✓ Initial state is None")
        
        # Test valid next stages
        next_stages = controller.get_valid_next_stages()
        expected = {stc.PipelineStage.I_INGESTION_PREPARATION}
        assert next_stages == expected, f"Expected {expected}, got {next_stages}"
        print("✓ Valid next stages correct")
        
        # Test artifact creation
        artifact = stc.create_stage_artifact("test", {"data": "value"})
        assert artifact.name == "test"
        assert len(artifact.hash_value) == 64  # SHA256 hex
        print("✓ Artifact creation works")
        
        # Test transition
        result = controller.transition_to_stage(
            stc.PipelineStage.I_INGESTION_PREPARATION, 
            {"test_artifact": artifact}
        )
        assert result.success, f"Transition failed: {result.message}"
        print("✓ First transition successful")
        
        # Test current stage updated
        current = controller.get_current_stage()
        assert current == stc.PipelineStage.I_INGESTION_PREPARATION
        print("✓ Current stage updated correctly")
        
        # Test invalid transition
        result = controller.transition_to_stage(
            stc.PipelineStage.K_KNOWLEDGE_EXTRACTION  # Skip X
        )
        assert not result.success, "Invalid transition should fail"
        print("✓ Invalid transition correctly rejected")
        
        # Test report generation
        report = controller.generate_transition_report()
        assert "controller_status" in report
        assert "stage_history" in report
        print("✓ Report generation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Stage Transition Controller...")
    success = test_basic_functionality()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Tests failed!")
        exit(1)