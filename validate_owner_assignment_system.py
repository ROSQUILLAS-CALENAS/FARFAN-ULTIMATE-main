#!/usr/bin/env python3
"""
Validation script for Owner Assignment System
Tests core functionality without external dependencies
"""

import tempfile
import json
import sqlite3
from pathlib import Path
from datetime import datetime

def test_basic_functionality():
    """Test basic functionality of the owner assignment system"""
    print("🔍 Testing Owner Assignment System...")
    
    # Test imports
    try:
        from canonical_flow.owner_assignment_system import (
            OwnerAssignmentSystem,
            ComponentRegistryService,
            ContributorInfo,
            ComponentOwnership,
            create_owner_assignment_api
        )
        print("✅ Imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test database initialization
    try:
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            owner_system = OwnerAssignmentSystem(db_path=temp_db.name)
            
        # Verify tables exist
        with sqlite3.connect(temp_db.name) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master WHERE type='table'
            """)
            tables = {row[0] for row in cursor.fetchall()}
            
        expected_tables = {
            'component_ownership',
            'ownership_history', 
            'team_mappings'
        }
        
        if expected_tables.issubset(tables):
            print("✅ Database initialization successful")
        else:
            print(f"❌ Missing tables: {expected_tables - tables}")
            return False
            
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False
    
    # Test team mappings
    try:
        with sqlite3.connect(temp_db.name) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM team_mappings")
            count = cursor.fetchone()[0]
            
        if count > 0:
            print("✅ Team mappings initialized successfully")
        else:
            print("❌ No team mappings found")
            return False
            
    except Exception as e:
        print(f"❌ Team mappings test failed: {e}")
        return False
    
    # Test ownership operations
    try:
        ownership = ComponentOwnership(
            component_path="test/component.py",
            primary_owner="test@example.com",
            secondary_owners=["secondary@example.com"],
            team="test_team",
            confidence_score=0.85,
            last_updated=datetime.now(),
            manual_override=False
        )
        
        # Save ownership
        owner_system.save_ownership(ownership)
        
        # Retrieve ownership
        retrieved = owner_system.get_ownership("test/component.py")
        
        if retrieved and retrieved.primary_owner == "test@example.com":
            print("✅ Ownership save/retrieve successful")
        else:
            print("❌ Ownership save/retrieve failed")
            return False
            
    except Exception as e:
        print(f"❌ Ownership operations failed: {e}")
        return False
    
    # Test API creation
    try:
        api = create_owner_assignment_api()
        
        expected_endpoints = [
            'assign_ownership',
            'reassign_ownership', 
            'get_ownership_info',
            'get_dashboard_data'
        ]
        
        missing_endpoints = []
        for endpoint in expected_endpoints:
            if endpoint not in api or not callable(api[endpoint]):
                missing_endpoints.append(endpoint)
        
        if not missing_endpoints:
            print("✅ API creation successful")
        else:
            print(f"❌ Missing API endpoints: {missing_endpoints}")
            return False
            
    except Exception as e:
        print(f"❌ API creation failed: {e}")
        return False
    
    # Test governance UI integration
    try:
        from canonical_flow.governance_ui_integration import (
            GovernanceUIIntegration,
            create_governance_api_routes
        )
        
        governance = GovernanceUIIntegration(owner_system)
        api_routes = create_governance_api_routes()
        
        if governance and api_routes:
            print("✅ Governance UI integration successful")
        else:
            print("❌ Governance UI integration failed")
            return False
            
    except Exception as e:
        print(f"❌ Governance UI integration failed: {e}")
        return False
    
    print("🎉 All tests passed!")
    return True

def test_bridge_registry_integration():
    """Test integration with bridge registry"""
    print("\n🔍 Testing Bridge Registry Integration...")
    
    try:
        from canonical_flow.bridge_registry import get_bridge_registry
        from canonical_flow.owner_assignment_system import OwnerAssignmentSystem
        
        owner_system = OwnerAssignmentSystem()
        bridge_registry = get_bridge_registry()
        
        # Verify bridge registry is accessible
        if bridge_registry:
            print("✅ Bridge registry integration successful")
            
            # Test team mapping based on bridge phases
            registry_info = bridge_registry.get_registry_info()
            if 'bridge_summary' in registry_info:
                print("✅ Bridge registry provides team mapping data")
            else:
                print("⚠️ Bridge registry missing team mapping data")
                
        else:
            print("❌ Bridge registry not accessible")
            return False
            
    except Exception as e:
        print(f"❌ Bridge registry integration failed: {e}")
        return False
        
    return True

def test_component_discovery():
    """Test component discovery functionality"""
    print("\n🔍 Testing Component Discovery...")
    
    try:
        from canonical_flow.owner_assignment_system import ComponentRegistryService, OwnerAssignmentSystem
        
        # Create temporary test files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test component file
            test_component = temp_path / "test_component.py"
            test_component.write_text('''
"""Test component"""
__phase__ = "A"
__code__ = "01A"  
__stage_order__ = 4

class TestProcessor:
    def process(self, data):
        return data
''')
            
            owner_system = OwnerAssignmentSystem()
            registry_service = ComponentRegistryService(owner_system)
            
            # Test component file detection
            is_component = registry_service._is_component_file(test_component)
            
            if is_component:
                print("✅ Component discovery successful")
            else:
                print("❌ Component discovery failed")
                return False
                
    except Exception as e:
        print(f"❌ Component discovery failed: {e}")
        return False
        
    return True

def main():
    """Run all validation tests"""
    print("=" * 60)
    print("🚀 Owner Assignment System Validation")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_basic_functionality()
    all_passed &= test_bridge_registry_integration()
    all_passed &= test_component_discovery()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("Owner Assignment System is ready for use.")
    else:
        print("❌ SOME VALIDATION TESTS FAILED!")
        print("Please check the errors above.")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)