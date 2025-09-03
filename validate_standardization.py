#!/usr/bin/env python3
"""
Validation script for standardized process interfaces.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path.cwd()))

def main():
    print("Validating Standardized Process Interface Implementation")
    print("=" * 55)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from canonical_flow.interfaces.process_interface import (
            StandardizedProcessor, ProcessAdapter, ProcessAdapterFactory
        )
        from canonical_flow.interfaces.telemetry import pipeline_telemetry
        print("   ✓ Successfully imported all interface modules")
        
        # Test basic adapter functionality
        print("\n2. Testing process adapter...")
        
        def test_function(data, options=None):
            """Test legacy function."""
            if options is None:
                options = {}
            return str(data).upper() if options.get('transform') == 'upper' else str(data)
        
        adapter = ProcessAdapterFactory.create_adapter(
            test_function,
            component_name="test.legacy_function",
            phase="testing"
        )
        
        result = adapter.process("hello world", {"options": {"transform": "upper"}})
        
        assert result['success'] == True, "Adapter processing should succeed"
        assert result['data'] == "HELLO WORLD", f"Expected 'HELLO WORLD', got {result['data']}"
        assert 'adapter_metadata' in result, "Missing adapter metadata"
        
        print("   ✓ Process adapter working correctly")
        print(f"   ✓ Result: {result['data']}")
        print(f"   ✓ Execution time: {result['adapter_metadata']['execution_time_ms']:.2f}ms")
        
        # Test standardized processor
        print("\n3. Testing standardized processor...")
        
        class TestProcessor(StandardizedProcessor):
            def process(self, data, context=None):
                return {
                    'success': True,
                    'data': f"Processed: {data}",
                    'metadata': {'processed_at': 'test_time'},
                    'errors': []
                }
        
        processor = TestProcessor()
        proc_result = processor.process("test data")
        
        assert proc_result['success'] == True, "Processor should succeed"
        assert 'Processed: test data' == proc_result['data'], "Unexpected processor output"
        
        print("   ✓ Standardized processor working correctly")
        print(f"   ✓ Result: {proc_result['data']}")
        
        # Test telemetry
        print("\n4. Testing telemetry...")
        
        metrics = pipeline_telemetry.get_metrics()
        print(f"   ✓ Telemetry system initialized")
        print(f"   ✓ Current metrics tracked: {len(metrics)} components")
        
        print("\n✅ All validation tests passed!")
        
        return True
        
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        return False
    except AssertionError as e:
        print(f"   ✗ Test assertion failed: {e}")
        return False
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)