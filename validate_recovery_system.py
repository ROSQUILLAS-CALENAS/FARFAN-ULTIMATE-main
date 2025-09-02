#!/usr/bin/env python3
"""
Recovery System Validation Script

This script validates the document recovery system functionality
without requiring external dependencies like Redis or numpy.
"""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock


def test_recovery_imports():
    """Test recovery system imports"""
    print("🔍 Testing recovery system imports...")
    
    try:
        from recovery_system import (
            FailedDocumentRecord, 
            FailedDocumentsTracker, 
            DocumentRecoveryManager,
            RecoveryMetrics
        )
        print("✅ Recovery system imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False


def test_failed_document_record():
    """Test FailedDocumentRecord functionality"""
    print("🧪 Testing FailedDocumentRecord...")
    
    try:
        from recovery_system import FailedDocumentRecord
        
        # Create a test record
        record = FailedDocumentRecord(
            document_path="/test/document.pdf",
            task_id="test-task-123",
            error_message="Import error: module not found",
            failure_timestamp=time.time(),
            retry_count=1,
            query="test query",
            worker_id="worker-1",
            metadata={"test": True},
            recovery_attempts=[time.time() - 3600]
        )
        
        # Test serialization
        data = record.to_dict()
        recovered = FailedDocumentRecord.from_dict(data)
        
        assert recovered.document_path == record.document_path
        assert recovered.task_id == record.task_id
        assert recovered.retry_count == record.retry_count
        
        # Test age calculation
        age_hours = record.get_failure_age_hours()
        assert age_hours >= 0
        
        # Test retry logic
        should_retry = record.should_retry(max_retry_count=3, min_retry_interval_hours=0.5)
        assert isinstance(should_retry, bool)
        
        print("✅ FailedDocumentRecord tests passed")
        return True
        
    except Exception as e:
        print(f"❌ FailedDocumentRecord test failed: {e}")
        return False


def test_recovery_metrics():
    """Test RecoveryMetrics functionality"""
    print("📊 Testing RecoveryMetrics...")
    
    try:
        from recovery_system import RecoveryMetrics
        
        metrics = RecoveryMetrics(
            total_failed_documents=10,
            documents_eligible_for_retry=5,
            recovery_attempts=20,
            successful_recoveries=15,
            failed_recoveries=5,
            recovery_success_rate=0.75,
            average_recovery_time=5.5,
            last_recovery_run=time.time()
        )
        
        # Test serialization
        data = metrics.to_dict()
        assert data['total_failed_documents'] == 10
        assert data['recovery_success_rate'] == 0.75
        
        print("✅ RecoveryMetrics tests passed")
        return True
        
    except Exception as e:
        print(f"❌ RecoveryMetrics test failed: {e}")
        return False


async def test_failed_documents_tracker():
    """Test FailedDocumentsTracker with mocked Redis"""
    print("🔧 Testing FailedDocumentsTracker...")
    
    try:
        from recovery_system import FailedDocumentsTracker, FailedDocumentRecord
        
        # Mock Redis client
        mock_redis = MagicMock()
        mock_redis.hdel = AsyncMock(return_value=1)
        mock_redis.hgetall = AsyncMock(return_value={})
        mock_redis.hget = AsyncMock(return_value=None)
        mock_redis.hset = AsyncMock(return_value=1)
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.set = AsyncMock(return_value=True)
        
        tracker = FailedDocumentsTracker(mock_redis, retention_days=7)
        
        # Test tracking a failed document
        await tracker.track_failed_document(
            document_path="/test/doc.pdf",
            task_id="test-123",
            error_message="Test error",
            query="test query",
            worker_id="worker-1",
            metadata={"test": True}
        )
        
        # Test removing a failed document
        success = await tracker.remove_failed_document("/test/doc.pdf")
        assert success == True
        
        # Test getting failed documents (empty result)
        failed_docs = await tracker.get_failed_documents()
        assert isinstance(failed_docs, list)
        
        # Test cleanup
        removed_count = await tracker.cleanup_old_records()
        assert isinstance(removed_count, int)
        
        # Test metrics
        metrics = await tracker.get_recovery_metrics()
        assert hasattr(metrics, 'total_failed_documents')
        
        print("✅ FailedDocumentsTracker tests passed")
        return True
        
    except Exception as e:
        print(f"❌ FailedDocumentsTracker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_document_recovery_manager():
    """Test DocumentRecoveryManager with mocked components"""
    print("🔄 Testing DocumentRecoveryManager...")
    
    try:
        from recovery_system import DocumentRecoveryManager, FailedDocumentsTracker
        
        # Mock distributed processor
        mock_processor = MagicMock()
        mock_processor._submit_tasks = AsyncMock()
        mock_processor.redis_client = MagicMock()
        mock_processor.redis_client.get = AsyncMock(return_value=None)
        
        # Mock tracker
        mock_tracker = MagicMock()
        mock_tracker.cleanup_old_records = AsyncMock(return_value=0)
        mock_tracker.get_failed_documents = AsyncMock(return_value=[])
        mock_tracker.update_recovery_metrics = AsyncMock()
        mock_tracker.get_recovery_metrics = AsyncMock()
        
        config = {
            'max_retry_count': 3,
            'min_retry_interval_hours': 1.0,
            'recovery_batch_size': 10,
            'enable_periodic_recovery': False
        }
        
        recovery_manager = DocumentRecoveryManager(
            distributed_processor=mock_processor,
            failed_docs_tracker=mock_tracker,
            config=config
        )
        
        # Test initialization
        await recovery_manager.initialize()
        
        # Test recovery attempt (no documents to recover)
        result = await recovery_manager.attempt_recovery()
        
        assert isinstance(result, dict)
        assert 'attempted_documents' in result
        assert result['attempted_documents'] == 0
        
        # Test recovery status
        status = await recovery_manager.get_recovery_status()
        assert isinstance(status, dict)
        
        print("✅ DocumentRecoveryManager tests passed")
        return True
        
    except Exception as e:
        print(f"❌ DocumentRecoveryManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_categorization():
    """Test error categorization functionality"""
    print("🔍 Testing error categorization...")
    
    try:
        from recovery_system import DocumentRecoveryManager
        
        # Create a mock recovery manager to test categorization
        config = {}
        recovery_manager = DocumentRecoveryManager(None, None, config)
        
        # Test various error types
        test_cases = [
            ("ModuleNotFoundError: No module named 'numpy'", "Import/Module Error"),
            ("MemoryError: Unable to allocate array", "Memory Error"),
            ("TimeoutError: Operation timed out", "Timeout Error"),
            ("FileNotFoundError: No such file or directory", "File Not Found"),
            ("PermissionError: Access denied", "Permission Error"),
            ("ConnectionError: Failed to connect", "Network Error"),
            ("Some other random error", "Other Error")
        ]
        
        for error_msg, expected_category in test_cases:
            category = recovery_manager._categorize_error(error_msg)
            assert category == expected_category, f"Expected {expected_category}, got {category} for {error_msg}"
        
        print("✅ Error categorization tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Error categorization test failed: {e}")
        return False


async def test_integration_with_main():
    """Test integration with main.py recovery command"""
    print("🔗 Testing integration with main.py...")
    
    try:
        # Test that recovery system can be imported in main context
        import sys
        from pathlib import Path
        
        # Add current directory to path (simulate main.py behavior)
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        
        # Import recovery system as main.py would
        from recovery_system import run_document_recovery
        
        # Test that the standalone recovery function exists and is callable
        assert callable(run_document_recovery)
        
        print("✅ Integration with main.py tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


async def run_all_tests():
    """Run all recovery system tests"""
    print("🚀 Recovery System Validation")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_recovery_imports()),
        ("FailedDocumentRecord", test_failed_document_record()),
        ("RecoveryMetrics", test_recovery_metrics()),
        ("FailedDocumentsTracker", await test_failed_documents_tracker()),
        ("DocumentRecoveryManager", await test_document_recovery_manager()),
        ("Error Categorization", test_error_categorization()),
        ("Main Integration", await test_integration_with_main())
    ]
    
    passed = 0
    failed = 0
    
    print("\n📊 Test Results:")
    print("=" * 30)
    
    for test_name, result in tests:
        if result:
            print(f"✅ {test_name}")
            passed += 1
        else:
            print(f"❌ {test_name}")
            failed += 1
    
    print(f"\n📈 Summary: {passed}/{len(tests)} tests passed")
    
    if failed == 0:
        print("🎉 All recovery system tests passed!")
        print("\nRecovery system is ready for use:")
        print("  - Use 'python main.py --recover' for manual recovery")
        print("  - Recovery is automatically integrated into distributed processing")
        print("  - Check README_RECOVERY_SYSTEM.md for detailed documentation")
    else:
        print(f"⚠️  {failed} tests failed. Please check the implementation.")
    
    return failed == 0


if __name__ == "__main__":
    # Run validation
    asyncio.run(run_all_tests())