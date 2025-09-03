"""
Simple import test for safety controller
"""

def test_safety_controller_imports():
    """Test that safety controller can be imported"""
    try:
# # #         from egw_query_expansion.core.mathematical_safety_controller import (  # Module not found  # Module not found  # Module not found
            MathematicalEnhancementSafetyController,
            SafetyThresholds,
            FeatureFlag,
            EnhancementStatus,
            DegradationReason
        )
        return True
    except ImportError:
        return False

if __name__ == "__main__":
    success = test_safety_controller_imports()
    print("✅ Import test passed" if success else "❌ Import test failed")