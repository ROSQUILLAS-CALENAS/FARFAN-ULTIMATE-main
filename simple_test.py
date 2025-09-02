#!/usr/bin/env python3
"""Simple test to verify our error handling implementation works."""

import sys
import traceback

def test_basic_functionality():
    """Test basic functionality of error handling components."""
    try:
        print("Testing knowledge extraction error handling...")
        
        # Import and test error handler
        from knowledge_extraction_error_handler import KnowledgeExtractionErrorHandler
        print("✓ KnowledgeExtractionErrorHandler imported successfully")
        
        handler = KnowledgeExtractionErrorHandler(default_timeout=5.0)
        print("✓ Error handler initialized")
        
        # Test timeout decorator
        @handler.timeout_decorator(timeout=1.0, component_name="test_component")
        def test_function(data):
            return {"result": "success", "data": data}
        
        result = test_function({"test": "data"})
        print("✓ Timeout decorator works")
        
        # Import and test pipeline
        from knowledge_extraction_error_handler import KnowledgeExtractionPipeline
        print("✓ KnowledgeExtractionPipeline imported successfully")
        
        pipeline = KnowledgeExtractionPipeline(default_timeout=5.0)
        print("✓ Pipeline initialized")
        
        # Test pipeline functionality
        chunks = [{"chunk_id": "test_chunk", "text": "Test document with Important Terms"}]
        results = pipeline.process_knowledge_extraction(chunks)
        print("✓ Pipeline processes chunks successfully")
        
        # Test orchestrator integration  
        from comprehensive_pipeline_orchestrator import ComprehensivePipelineOrchestrator
        print("✓ ComprehensivePipelineOrchestrator imported successfully")
        
        orchestrator = ComprehensivePipelineOrchestrator()
        print("✓ Orchestrator initialized")
        
        test_data = {
            'chunk_id': 'test_chunk',
            'chunk_data': {
                'pages': [{'text': 'Test document with Important Terms'}]
            }
        }
        
        result = orchestrator._run_knowledge_extraction(test_data)
        print("✓ Orchestrator knowledge extraction works")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return False

def test_error_scenarios():
    """Test error handling scenarios."""
    try:
        print("\nTesting error scenarios...")
        
        from knowledge_extraction_error_handler import KnowledgeExtractionErrorHandler
        
        handler = KnowledgeExtractionErrorHandler(default_timeout=2.0)
        
        # Test timeout scenario
        @handler.timeout_decorator(timeout=0.5, component_name="timeout_test")
        def slow_function(data):
            import time
            time.sleep(1.0)  # Will timeout
            return {"result": "should_not_reach"}
        
        result = slow_function({"chunk_id": "timeout_chunk"})
        
        if "fallback_reason" in result:
            print("✓ Timeout handling works")
        else:
            print("❌ Timeout handling failed")
            return False
        
        # Test exception scenario
        @handler.timeout_decorator(timeout=2.0, component_name="exception_test")
        def failing_function(data):
            raise ValueError("Test exception")
        
        result = failing_function({"chunk_id": "exception_chunk"})
        
        if "fallback_reason" in result:
            print("✓ Exception handling works")
        else:
            print("❌ Exception handling failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error scenario test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Knowledge Extraction Error Handling Implementation")
    print("=" * 60)
    
    basic_test = test_basic_functionality()
    error_test = test_error_scenarios()
    
    print("\n📊 Test Summary")
    print("=" * 30)
    
    if basic_test:
        print("✅ Basic functionality tests: PASSED")
    else:
        print("❌ Basic functionality tests: FAILED")
    
    if error_test:
        print("✅ Error scenario tests: PASSED")
    else:
        print("❌ Error scenario tests: FAILED")
    
    if basic_test and error_test:
        print("\n🎉 All tests passed! Error handling implementation is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())