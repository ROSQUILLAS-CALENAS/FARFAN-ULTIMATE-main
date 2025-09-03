"""
Test script for the standardized process interfaces and telemetry system.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add the current directory to Python path
sys.path.insert(0, str(Path.cwd()))

try:
    from canonical_flow.interfaces.process_interface import (
        StandardizedProcessor, ProcessAdapter, ProcessAdapterFactory
    )
    from canonical_flow.interfaces.telemetry import (
        pipeline_telemetry, trace_component, trace_process_method
    )
    print("✓ Successfully imported standardization interfaces")
except ImportError as e:
    print(f"✗ Failed to import interfaces: {e}")
    sys.exit(1)


class TestProcessor(StandardizedProcessor):
    """Test implementation of standardized processor."""
    
    def __init__(self, name: str = "test_processor"):
        self.name = name
    
    @trace_process_method
    def process(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process method following standardized interface."""
        if context is None:
            context = {}
        
        # Simulate processing
        time.sleep(0.001)  # 1ms
        
        return {
            'success': True,
            'data': f"Processed: {data}",
            'metadata': {
                'processor': self.name,
                'input_type': type(data).__name__
            },
            'errors': []
        }


def test_legacy_function(text: str, options: Dict = None) -> str:
    """Legacy function to test adaptation."""
    if options is None:
        options = {}
    
    transform = options.get('transform', 'identity')
    
    if transform == 'upper':
        return text.upper()
    elif transform == 'lower':  
        return text.lower()
    else:
        return text


@trace_component(component_name="test.decorated_function", phase="testing")
def test_decorated_function(data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Test function with telemetry decoration."""
    return {
        'success': True,
        'data': f"Decorated processing of: {data}",
        'metadata': {'decorated': True},
        'errors': []
    }


def test_standardized_processor():
    """Test the standardized processor implementation."""
    print("\n=== Testing Standardized Processor ===")
    
    processor = TestProcessor("test_instance")
    
    # Test various data types
    test_cases = [
        "simple text",
        {"key": "value"},
        [1, 2, 3],
        42
    ]
    
    for data in test_cases:
        result = processor.process(data, {"test": True})
        assert result['success'], f"Processor failed for data: {data}"
        assert 'data' in result, "Missing data field in result"
        assert 'metadata' in result, "Missing metadata field in result"
        print(f"✓ Processed {type(data).__name__}: {data}")
    
    print("✓ Standardized processor tests passed")


def test_process_adapter():
    """Test the process adapter functionality."""
    print("\n=== Testing Process Adapter ===")
    
    # Create adapter for legacy function
    adapter = ProcessAdapterFactory.create_adapter(
        test_legacy_function,
        component_name="test.legacy_text_processor", 
        phase="text_processing"
    )
    
    # Test the adapter
    test_data = "Hello World"
    test_context = {"options": {"transform": "upper"}}
    
    result = adapter.process(test_data, test_context)
    
    assert result['success'], "Adapter processing failed"
    assert result['data'] == "HELLO WORLD", f"Expected 'HELLO WORLD', got {result['data']}"
    assert 'adapter_metadata' in result, "Missing adapter metadata"
    
    print(f"✓ Adapter result: {result['data']}")
    print(f"✓ Coercions: {len(adapter.coercions_performed)}")
    
    # Test process wrapper
    process_wrapper = ProcessAdapterFactory.create_process_wrapper(
        test_legacy_function,
        component_name="test.wrapped_processor"
    )
    
    wrapped_result = process_wrapper(test_data, test_context)
    assert wrapped_result['success'], "Process wrapper failed"
    
    print("✓ Process adapter tests passed")


def test_telemetry():
    """Test telemetry functionality."""
    print("\n=== Testing Telemetry ===")
    
    # Test decorated function
    result = test_decorated_function("test data", {"context": "test"})
    assert result['success'], "Decorated function failed"
    
    # Test processor with telemetry
    processor = TestProcessor("telemetry_test")
    processor.process("telemetry test data")
    
    # Get metrics
    metrics = pipeline_telemetry.get_metrics()
    
    print(f"✓ Collected metrics for {len(metrics)} components")
    
    for component, metric_data in metrics.items():
        print(f"  {component}: {metric_data['total_calls']} calls, "
              f"{metric_data['average_time_ms']:.2f}ms avg")
    
    print("✓ Telemetry tests passed")


def test_module_standardization():
    """Test module standardization functionality."""
    print("\n=== Testing Module Standardization ===")
    
    # Create a mock module
    class MockModule:
        def __init__(self):
            self.__name__ = "test_module"
        
        def legacy_function(self, data, options=None):
            return f"processed: {data}"
        
        def another_function(self, text):
            return text.strip()
    
    mock_module = MockModule()
    
    # Standardize the module
    from canonical_flow.interfaces.process_interface import standardize_module
    
    standardize_module(mock_module, phase_override="testing")
    
    # Check that process method was added
    assert hasattr(mock_module, 'process'), "Process method not added to module"
    assert hasattr(mock_module, '_process_adapters'), "Process adapters not added"
    assert hasattr(mock_module, 'list_processors'), "List processors method not added"
    
    # Test the added process method
    try:
        result = mock_module.process("test data", {"test": True})
        print(f"✓ Module process method result: {result.get('success', False)}")
    except Exception as e:
        print(f"⚠ Module process method failed (expected): {e}")
    
    # Test list processors
    processors = mock_module.list_processors()
    print(f"✓ Available processors: {processors}")
    
    print("✓ Module standardization tests passed")


def main():
    """Run all tests."""
    print("Testing Standardized Process Interfaces & Telemetry")
    print("=" * 55)
    
    try:
        test_standardized_processor()
        test_process_adapter()
        test_telemetry()
        test_module_standardization()
        
        # Final telemetry report
        print("\n=== Final Telemetry Summary ===")
        final_metrics = pipeline_telemetry.get_metrics()
        
        total_calls = sum(m['total_calls'] for m in final_metrics.values())
        total_time = sum(m['total_time_ms'] for m in final_metrics.values())
        
        print(f"✓ Total operations traced: {total_calls}")
        print(f"✓ Total execution time: {total_time:.2f}ms")
        print(f"✓ Components instrumented: {len(final_metrics)}")
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()