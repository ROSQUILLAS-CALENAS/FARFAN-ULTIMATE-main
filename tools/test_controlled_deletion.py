#!/usr/bin/env python3
"""
Test script for the Controlled Deletion System

This script demonstrates the functionality of the controlled deletion system
without requiring external dependencies.
"""

import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# Import our modules
try:
    from controlled_deletion_system import ControlledDeletionManager, EmbargoRecord
    from import_linter_config import CustomASTImportChecker
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to run from the tools directory")
    exit(1)


def test_basic_embargo_functionality():
    """Test basic embargo functionality"""
    print("🧪 Testing basic embargo functionality...")
    
    # Create a temporary test directory
    test_dir = Path("test_duplicate_dir")
    test_dir.mkdir(exist_ok=True)
    
    # Create a test Python file
    test_file = test_dir / "test_module.py"
    test_file.write_text("""
def test_function():
    print("This is a test function")

class TestClass:
    pass
""")
    
    try:
        # Initialize the manager
        manager = ControlledDeletionManager()
        
        # Test embargo
        success = manager.embargo_directory(
            str(test_dir),
            "Test duplicate directory",
            grace_period_days=7
        )
        
        assert success, "Failed to embargo directory"
        print("✅ Successfully embargoed directory")
        
        # Test listing embargoed directories
        assert str(test_dir) in manager.embargo_registry
        record = manager.embargo_registry[str(test_dir)]
        assert record.reason == "Test duplicate directory"
        assert record.grace_period_days == 7
        print("✅ Embargo record created correctly")
        
        # Test scanning
        scan_results = manager.scan_embargoed_directories()
        assert str(test_dir) in scan_results
        print("✅ Scan completed successfully")
        
        # Test summary report
        summary = manager.generate_summary_report()
        assert summary['total_embargoed'] >= 1
        print("✅ Summary report generated")
        
        # Test import checking
        violations = manager.check_deprecated_imports()
        print(f"✅ Import check completed, found {len(violations)} violations")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)
        # Remove from registry
        if str(test_dir) in manager.embargo_registry:
            del manager.embargo_registry[str(test_dir)]
            manager._save_embargo_registry()


def test_ast_import_analysis():
    """Test AST-based import analysis"""
    print("\n🧪 Testing AST import analysis...")
    
    # Create temporary files for testing
    test_dir = Path("test_project")
    test_dir.mkdir(exist_ok=True)
    
    # Create a source file that imports from embargoed directory
    source_file = test_dir / "source.py"
    source_file.write_text("""
import analysis_nlp.processor
from retrieval_engine import search
from canonical_flow.A_analysis_nlp import canonical_processor
""")
    
    # Create embargoed directory structure
    embargoed_dir = test_dir / "analysis_nlp"
    embargoed_dir.mkdir(exist_ok=True)
    (embargoed_dir / "processor.py").write_text("# Embargoed module")
    
    try:
        # Setup manager and embargo the directory
        manager = ControlledDeletionManager()
        manager.embargo_directory(str(embargoed_dir), "Test embargo", 7)
        
        # Test AST checker
        ast_checker = CustomASTImportChecker(manager)
        violations = ast_checker.check_file(source_file)
        
        print(f"✅ Found {len(violations)} import violations")
        
        # Should find violation for analysis_nlp import
        embargoed_violations = [v for v in violations if 'analysis_nlp' in v.get('module', '')]
        assert len(embargoed_violations) > 0, "Should detect embargoed import"
        print("✅ Correctly detected embargoed imports")
        
        return True
        
    except Exception as e:
        print(f"❌ AST test failed: {e}")
        return False
    
    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)


def test_deprecation_warning_injection():
    """Test deprecation warning injection"""
    print("\n🧪 Testing deprecation warning injection...")
    
    test_file = Path("test_warning_injection.py")
    test_file.write_text("""
import os
import sys

def main():
    print("Hello world")

if __name__ == '__main__':
    main()
""")
    
    try:
        from controlled_deletion_system import DeprecationWarningInjector
        
        injector = DeprecationWarningInjector()
        expiry_date = datetime.now() + timedelta(days=30)
        
        success = injector.inject_warning(
            test_file,
            expiry_date,
            "canonical_flow/example"
        )
        
        assert success, "Failed to inject warning"
        print("✅ Successfully injected deprecation warning")
        
        # Verify warning was injected
        content = test_file.read_text()
        assert "This module is deprecated" in content, "Warning not found in file"
        assert "canonical_flow/example" in content, "Canonical path not in warning"
        print("✅ Warning content verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Warning injection test failed: {e}")
        return False
    
    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()


def test_embargo_record_functionality():
    """Test EmbargoRecord functionality"""
    print("\n🧪 Testing EmbargoRecord functionality...")
    
    try:
        # Create test record
        record = EmbargoRecord(
            directory="test_dir",
            embargo_date=datetime.now(),
            grace_period_days=30,
            reason="Test embargo"
        )
        
        # Test expiry calculation
        expected_expiry = record.embargo_date + timedelta(days=30)
        assert record.expiry_date.date() == expected_expiry.date()
        print("✅ Expiry date calculation correct")
        
        # Test days remaining
        days_remaining = record.days_remaining
        assert 29 <= days_remaining <= 30, f"Days remaining should be ~30, got {days_remaining}"
        print("✅ Days remaining calculation correct")
        
        # Test expired status
        old_record = EmbargoRecord(
            directory="old_dir",
            embargo_date=datetime.now() - timedelta(days=40),
            grace_period_days=30,
            reason="Old test"
        )
        assert old_record.is_expired, "Old record should be expired"
        print("✅ Expiry status calculation correct")
        
        return True
        
    except Exception as e:
        print(f"❌ EmbargoRecord test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Starting Controlled Deletion System Tests\n")
    
    tests = [
        test_embargo_record_functionality,
        test_deprecation_warning_injection,
        test_basic_embargo_functionality,
        test_ast_import_analysis,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ PASSED\n")
            else:
                failed += 1
                print("❌ FAILED\n")
        except Exception as e:
            failed += 1
            print(f"❌ FAILED with exception: {e}\n")
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"⚠️  {failed} tests failed")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)