#!/usr/bin/env python3
"""
Simple test for recovery_scripts module
"""

def test_recovery_scripts_import():
    """Test that recovery_scripts can be imported successfully"""
    try:
        import recovery_scripts
        print("✓ Successfully imported recovery_scripts module")
        
        # Test that main functions exist
        assert hasattr(recovery_scripts, 'detect_partial_installation')
        assert hasattr(recovery_scripts, 'clean_corrupted_environment')  
        assert hasattr(recovery_scripts, 'reinstall_dependencies_ordered')
        assert hasattr(recovery_scripts, 'run_recovery_workflow')
        print("✓ All required functions present")
        
        # Test that functions are callable
        assert callable(recovery_scripts.detect_partial_installation)
        assert callable(recovery_scripts.clean_corrupted_environment)
        assert callable(recovery_scripts.reinstall_dependencies_ordered)
        assert callable(recovery_scripts.run_recovery_workflow)
        print("✓ All functions are callable")
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import recovery_scripts: {e}")
        return False
    except AssertionError as e:
        print(f"✗ Module structure validation failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_recovery_scripts_import()
    exit(0 if success else 1)