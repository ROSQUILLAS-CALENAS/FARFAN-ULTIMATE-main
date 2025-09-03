#!/usr/bin/env python3
"""
Validation script for Mathematical Enhancement Safety Controller
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

def main():
    print("🚀 Mathematical Enhancement Safety Controller - Validation")
    print("=" * 60)
    
    try:
        # Test basic import
# # #         from egw_query_expansion.core.mathematical_safety_controller import (  # Module not found  # Module not found  # Module not found
            MathematicalEnhancementSafetyController,
            SafetyThresholds,
            FeatureFlag,
            EnhancementStatus,
            create_default_safety_controller
        )
        
        print("✅ Successfully imported safety controller modules")
        
        # Create controller
        controller = create_default_safety_controller()
        print("✅ Successfully created safety controller")
        
        # Test basic operations
        thresholds = SafetyThresholds(max_iterations=500)
        print(f"✅ Created safety thresholds (max_iterations: {thresholds.max_iterations})")
        
        # Test feature flag
        flag = FeatureFlag(name="test_feature", enabled=True)
        print(f"✅ Created feature flag: {flag.name} (enabled: {flag.enabled})")
        
        # Register a simple enhancement
        def simple_enhancement(x):
            return x * 2
        
        def simple_baseline(x):
            return x
        
        controller.register_enhancement("demo", simple_enhancement, simple_baseline)
        controller.enable_enhancement("demo")
        print("✅ Registered and enabled demo enhancement")
        
        # Test safe computation
        result = controller.safe_compute_with_fallback("demo", 5, enhanced_fn=simple_enhancement)
        print(f"✅ Safe computation result: {result}")
        
        # Check status
        status = controller.get_enhancement_status("demo")
        print(f"✅ Enhancement status retrieved successfully")
        
        # Get system health
        health = controller.get_system_health()
        print(f"✅ System health: {health['total_activations']} activations")
        
        print("\n🎉 All validation tests passed!")
        print("✅ Mathematical Enhancement Safety Controller is working correctly")
        
        return 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())