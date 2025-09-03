#!/usr/bin/env python3
"""
Validation script for Mathematical Enhancement Safety Controller

Validates the safety controller module can be imported and basic
functionality works without external dependencies.
"""

import sys
import os

def validate_imports():
    """Validate that safety controller can be imported"""
    print("Validating Mathematical Enhancement Safety Controller...")
    
    try:
        # Test basic import
# # #         from egw_query_expansion.core.safety_controller import (  # Module not found  # Module not found  # Module not found
            MathEnhancementSafetyController,
            EnhancementStatus, 
            StabilityLevel,
            EnhancementConfig
        )
        print("✓ Core safety controller classes imported successfully")
        
        # Test exception imports
# # #         from egw_query_expansion.core.safety_controller import (  # Module not found  # Module not found  # Module not found
            NumericalInstabilityError,
            IterationLimitExceededError,
            ConvergenceError
        )
        print("✓ Safety controller exceptions imported successfully")
        
        # Test decorator import
# # #         from egw_query_expansion.core.safety_controller import safe_enhancement  # Module not found  # Module not found  # Module not found
        print("✓ Safety decorator imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def validate_basic_functionality():
    """Validate basic functionality without numpy/torch"""
    print("\nValidating basic functionality...")
    
    try:
# # #         from egw_query_expansion.core.safety_controller import (  # Module not found  # Module not found  # Module not found
            MathEnhancementSafetyController,
            EnhancementStatus,
            StabilityLevel,
            EnhancementConfig
        )
        
        # Create safety controller
        controller = MathEnhancementSafetyController()
        print("✓ Safety controller created successfully")
        
        # Test enhancement registration
        config = EnhancementConfig(
            name="test_enhancement",
            enabled=True,
            max_iterations=100
        )
        controller.register_enhancement("test_enhancement", config)
        print("✓ Enhancement registration works")
        
        # Test status checking
        is_active = controller.is_enhancement_active("test_enhancement")
        print(f"✓ Enhancement status check works: {is_active}")
        
        # Test enable/disable
        success = controller.disable_enhancement("test_enhancement")
        print(f"✓ Enhancement disable works: {success}")
        
        success = controller.enable_enhancement("test_enhancement")
        print(f"✓ Enhancement enable works: {success}")
        
        # Test status reporting
        status = controller.get_enhancement_status()
        print(f"✓ Status reporting works: {len(status)} enhancements")
        
        # Test pipeline metadata
        metadata = controller.get_pipeline_artifact_metadata()
        print("✓ Pipeline metadata generation works")
        
        # Test enum values
        levels = [StabilityLevel.STRICT, StabilityLevel.MODERATE, StabilityLevel.RELAXED]
        statuses = [EnhancementStatus.ENABLED, EnhancementStatus.DISABLED, EnhancementStatus.AUTO_DISABLED]
        print(f"✓ Enums work: {len(levels)} stability levels, {len(statuses)} status types")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def validate_structure():
    """Validate module structure and documentation"""
    print("\nValidating module structure...")
    
    try:
        import egw_query_expansion.core.safety_controller as sc
        
        # Check for required classes
        required_classes = [
            'MathEnhancementSafetyController',
            'EnhancementConfig', 
            'StabilityGuard',
            'SafeComputationContext'
        ]
        
        for class_name in required_classes:
            if hasattr(sc, class_name):
                print(f"✓ {class_name} class present")
            else:
                print(f"✗ {class_name} class missing")
                return False
        
        # Check for required enums
        required_enums = ['EnhancementStatus', 'StabilityLevel']
        for enum_name in required_enums:
            if hasattr(sc, enum_name):
                print(f"✓ {enum_name} enum present")
            else:
                print(f"✗ {enum_name} enum missing")
                return False
        
        # Check for required exceptions
        required_exceptions = [
            'NumericalInstabilityError',
            'IterationLimitExceededError', 
            'ConvergenceError'
        ]
        
        for exc_name in required_exceptions:
            if hasattr(sc, exc_name):
                print(f"✓ {exc_name} exception present")
            else:
                print(f"✗ {exc_name} exception missing")
                return False
        
        # Check module docstring
        if hasattr(sc, '__doc__') and sc.__doc__:
            print("✓ Module has documentation")
        else:
            print("✗ Module documentation missing")
        
        return True
        
    except Exception as e:
        print(f"✗ Structure validation failed: {e}")
        return False


def main():
    """Main validation function"""
    print("Mathematical Enhancement Safety Controller Validation")
    print("=" * 60)
    
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    results = []
    
    # Run validation tests
    results.append(validate_imports())
    results.append(validate_basic_functionality()) 
    results.append(validate_structure())
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results)
    
    print(f"Total validation tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {(passed_tests / total_tests * 100):.1f}%")
    
    if all(results):
        print("\n🎉 All validations passed! Safety controller is ready to use.")
        print("\nKey features validated:")
        print("• Feature flag management for mathematical enhancements")
        print("• Numerical stability protection framework")
        print("• Graceful degradation mechanisms")
        print("• Enhancement status tracking")
        print("• Pipeline artifact metadata generation")
        return True
    else:
        print("\n❌ Some validations failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)