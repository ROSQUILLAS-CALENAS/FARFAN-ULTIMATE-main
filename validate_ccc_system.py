#!/usr/bin/env python3
"""
Validation script for the Continuous Canonical Compliance (CCC) system.
Tests the validator and CI integration components.
"""

import sys
from pathlib import Path

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent / "tools"))

def test_ccc_validator():
    """Test the CCC validator functionality."""
    try:
        from ccc_validator import CCCValidator, ValidationResult, ComponentInfo
        print("✅ CCC Validator imports successfully")
        
        # Test validator initialization
        validator = CCCValidator(Path("."))
        print("✅ CCC Validator initializes successfully")
        
        # Test basic structures
        result = ValidationResult("test", True, "Test message")
        print(f"✅ ValidationResult: {result.gate} - {result.message}")
        
        component = ComponentInfo("test_path", "I", "01I")
        print(f"✅ ComponentInfo: {component.path} - {component.phase}")
        
        return True
    except Exception as e:
        print(f"❌ CCC Validator test failed: {e}")
        return False

def test_ci_integration():
    """Test the CI integration functionality."""
    try:
        from ci_ccc_integration import CCCCIIntegration
        print("✅ CI Integration imports successfully")
        
        # Test CI integration initialization
        ci = CCCCIIntegration()
        print("✅ CI Integration initializes successfully")
        
        return True
    except Exception as e:
        print(f"❌ CI Integration test failed: {e}")
        return False

def test_phase_mapping():
    """Test canonical phase mapping."""
    try:
        from ccc_validator import CANONICAL_PHASES, PHASE_ORDER
        
        print("📊 Canonical Phase Mapping:")
        for phase, (name, order) in CANONICAL_PHASES.items():
            print(f"   {phase}: {name} (order: {order})")
        
        print(f"📋 Phase Order: {' → '.join(PHASE_ORDER)}")
        
        # Verify order consistency
        for i, phase in enumerate(PHASE_ORDER):
            expected_order = CANONICAL_PHASES[phase][1]
            if i != expected_order:
                print(f"❌ Phase order mismatch for {phase}: expected {i}, got {expected_order}")
                return False
        
        print("✅ Phase mapping is consistent")
        return True
    except Exception as e:
        print(f"❌ Phase mapping test failed: {e}")
        return False

def test_file_structure():
    """Test that required files exist."""
    required_files = [
        "tools/ccc_validator.py",
        "tools/ci_ccc_integration.py",
        ".github/workflows/ccc_validation.yml",
        "canonical_flow/index.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def main():
    """Run all validation tests."""
    print("🔍 Validating Continuous Canonical Compliance (CCC) System")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Phase Mapping", test_phase_mapping),
        ("CCC Validator", test_ccc_validator),
        ("CI Integration", test_ci_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}:")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} test ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Summary:")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   Total:  {passed + failed}")
    
    if failed == 0:
        print("🎉 All CCC system tests PASSED!")
        return 0
    else:
        print("🚨 Some CCC system tests FAILED!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)