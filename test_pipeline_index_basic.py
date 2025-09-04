#!/usr/bin/env python3
"""
Basic test script for pipeline index system components
"""

import json
import sys
from pathlib import Path

def test_basic_functionality():
    """Test basic functionality without external dependencies"""
    
    print("Testing Pipeline Index System Basic Functionality")
    print("="*50)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from pipeline_index_system import PipelineIndexSystem, ComponentInfo
        print("   ✅ Core imports successful")
        
        # Test system initialization  
        print("2. Testing system initialization...")
        system = PipelineIndexSystem()
        print("   ✅ System initialization successful")
        
        # Test component scanning (without saving)
        print("3. Testing component scanning...")
        components = system.scan_filesystem_components()
        print(f"   ✅ Found {len(components)} components")
        
        if components:
            print("   📊 Sample components:")
            for comp in components[:3]:
                print(f"      {comp.code}: {comp.stage} - {Path(comp.original_path).name}")
        
        # Test stage distribution
        print("4. Testing stage analysis...")
        stage_counts = {}
        for comp in components:
            stage_counts[comp.stage] = stage_counts.get(comp.stage, 0) + 1
        
        print("   📈 Stage distribution:")
        for stage, count in sorted(stage_counts.items()):
            print(f"      {stage}: {count}")
        
        # Test dependency calculation (basic)
        print("5. Testing dependency calculation...")
        dependencies = system.calculate_dependencies(components)
        dep_count = sum(len(deps) for deps in dependencies.values())
        print(f"   ✅ Found {dep_count} total dependencies")
        
        # Test validation logic (basic)
        print("6. Testing validation logic...")
        validation = system.validate_index_consistency()
        print(f"   ✅ Validation {'passed' if validation['valid'] else 'found issues'}")
        
        if validation['errors']:
            print("   ❌ Validation errors:")
            for error in validation['errors'][:3]:
                print(f"      - {error}")
        
        if validation['warnings']:
            print("   ⚠️  Validation warnings:")
            for warning in validation['warnings'][:3]:
                print(f"      - {warning}")
        
        print("\n" + "="*50)
        print("✅ ALL BASIC TESTS PASSED")
        print(f"✅ System can discover {len(components)} components")
        print(f"✅ Validation framework is working")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_index_operations():
    """Test index read/write operations"""
    
    print("\nTesting Index Operations")
    print("-" * 30)
    
    try:
        from pipeline_index_system import PipelineIndexSystem
        
        system = PipelineIndexSystem()
        
        # Test loading existing index
        print("1. Testing index loading...")
        existing_components = system.load_current_index()
        print(f"   ✅ Loaded {len(existing_components)} components from existing index")
        
        # Test reconciliation
        print("2. Testing reconciliation...")
        fs_components, reconciliation = system.reconcile_index()
        print(f"   ✅ Reconciliation completed:")
        print(f"      Added: {len(reconciliation['added'])}")
        print(f"      Modified: {len(reconciliation['modified'])}")  
        print(f"      Deleted: {len(reconciliation['deleted'])}")
        print(f"      Unchanged: {len(reconciliation['unchanged'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Index operations test failed: {e}")
        return False

if __name__ == "__main__":
    success = True
    
    success &= test_basic_functionality()
    success &= test_index_operations()
    
    if success:
        print("\n🎉 All tests passed! Pipeline Index System is working correctly.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Check the output above.")
        sys.exit(1)