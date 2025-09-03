"""
Demonstration script for the standardized process interface and telemetry system.

Shows how to instrument existing modules and use the standardized interface.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from canonical_flow.interfaces import (
    StandardizedProcessor,
    ProcessAdapter,
    ProcessAdapterFactory,
    pipeline_telemetry,
    PipelineTelemetry
)
from canonical_flow.interfaces.auto_instrumenter import instrument_canonical_flow
from canonical_flow.interfaces.module_inspector import inspect_canonical_flow_modules
from canonical_flow.interfaces.telemetry import trace_component, trace_process_method


class DemoProcessor(StandardizedProcessor):
    """
    Example processor implementing the standardized interface.
    """
    
    def __init__(self, name: str = "demo_processor"):
        self.name = name
    
    @trace_process_method
    def process(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process data with context."""
        if context is None:
            context = {}
        
        # Simulate some processing
        time.sleep(0.001)  # 1ms processing time
        
        processed_data = {
            'original_data': data,
            'processed_by': self.name,
            'processing_timestamp': time.time(),
            'context_keys': list(context.keys()) if context else []
        }
        
        return {
            'success': True,
            'data': processed_data,
            'metadata': {
                'processor': self.name,
                'processing_time': 0.001,
                'data_type': type(data).__name__
            },
            'errors': []
        }


def demo_legacy_function(input_text: str, options: Dict[str, Any] = None) -> str:
    """
    Example legacy function that doesn't follow the standardized interface.
    """
    if options is None:
        options = {}
    
    transform = options.get('transform', 'upper')
    
    if transform == 'upper':
        return input_text.upper()
    elif transform == 'lower':
        return input_text.lower()
    else:
        return input_text


@trace_component(component_name="demo.enhanced_function", phase="analysis")
def demo_enhanced_function(data: Any, context: Dict[str, Any] = None) -> str:
    """
    Example function already following standardized patterns.
    """
    if context is None:
        context = {}
    
    operation = context.get('operation', 'identity')
    
    if operation == 'stringify':
        return str(data)
    elif operation == 'length':
        return len(str(data))
    else:
        return data


def demonstrate_adapters():
    """Demonstrate process adapters for legacy functions."""
    print("\n=== Process Adapter Demonstration ===")
    
    # Create adapter for legacy function
    adapter = ProcessAdapterFactory.create_adapter(
        demo_legacy_function,
        component_name="demo.legacy_text_processor",
        phase="text_processing"
    )
    
    # Test the adapter
    test_data = "Hello World"
    test_context = {"options": {"transform": "upper"}}
    
    print(f"Testing adapter with data: '{test_data}'")
    print(f"Context: {test_context}")
    
    result = adapter.process(test_data, test_context)
    
    print(f"Adapter result: {result}")
    print(f"Coercions performed: {adapter.coercions_performed}")
    
    # Create process wrapper
    process_wrapper = ProcessAdapterFactory.create_process_wrapper(
        demo_legacy_function,
        component_name="demo.wrapped_processor",
        phase="text_processing"
    )
    
    print(f"\nTesting process wrapper...")
    wrapped_result = process_wrapper(test_data, test_context)
    print(f"Wrapped result: {wrapped_result}")


def demonstrate_telemetry():
    """Demonstrate OpenTelemetry integration."""
    print("\n=== Telemetry Demonstration ===")
    
    # Create standardized processor
    processor = DemoProcessor("telemetry_demo")
    
    # Process some data
    test_cases = [
        ("simple string", {"operation": "test1"}),
        ({"key": "value"}, {"operation": "test2"}),
        ([1, 2, 3, 4, 5], {"operation": "test3"})
    ]
    
    for data, context in test_cases:
        print(f"Processing: {data}")
        result = processor.process(data, context)
        print(f"Success: {result['success']}")
    
    # Get telemetry metrics
    metrics = pipeline_telemetry.get_metrics()
    print(f"\nTelemetry metrics collected:")
    for component, metric_data in metrics.items():
        print(f"  {component}:")
        print(f"    Calls: {metric_data['total_calls']}")
        print(f"    Avg time: {metric_data['average_time_ms']:.2f}ms")
        print(f"    Total time: {metric_data['total_time_ms']:.2f}ms")


def demonstrate_instrumentation():
    """Demonstrate automatic module instrumentation."""
    print("\n=== Module Instrumentation Demonstration ===")
    
    # Inspect modules first
    print("Inspecting canonical flow modules...")
    inspection_report = inspect_canonical_flow_modules()
    
    compatibility_summary = inspection_report['compatibility_report']['compatibility_summary']
    print(f"Compatibility Summary:")
    print(f"  High compatibility: {compatibility_summary['high_compatibility']}")
    print(f"  Medium compatibility: {compatibility_summary['medium_compatibility']}")
    print(f"  Low compatibility: {compatibility_summary['low_compatibility']}")
    print(f"  Average score: {compatibility_summary['average_score']:.1f}")
    
    # Perform instrumentation
    print("\nInstrumenting canonical flow modules...")
    instrumentation_results = instrument_canonical_flow(
        pattern="**/I_ingestion_preparation/*.py",  # Limit to ingestion for demo
        force_reinstrument=True
    )
    
    print(f"Instrumentation Results:")
    print(f"  Discovered: {instrumentation_results['total_discovered']}")
    print(f"  Successful: {instrumentation_results['successful_instrumentations']}")
    print(f"  Failed: {instrumentation_results['failed_instrumentations']}")
    print(f"  Skipped: {instrumentation_results['skipped_instrumentations']}")
    
    if instrumentation_results['errors']:
        print(f"  Errors: {len(instrumentation_results['errors'])}")
        for error in instrumentation_results['errors'][:3]:  # Show first 3 errors
            print(f"    - {error.get('module_name', 'unknown')}: {error.get('error', 'unknown error')}")


def demonstrate_custom_telemetry():
    """Demonstrate custom telemetry usage."""
    print("\n=== Custom Telemetry Demonstration ===")
    
    # Create custom telemetry instance
    custom_telemetry = PipelineTelemetry("demo_service")
    
    # Use telemetry context manager
    with custom_telemetry.trace_operation(
        "custom_operation",
        "demo.custom_component", 
        "processing",
        {"input_size": 100, "operation_type": "demo"}
    ) as span:
        # Simulate work
        time.sleep(0.005)  # 5ms
        
        # Add custom attributes during execution
        span.set_attribute("processed_items", 42)
        span.set_attribute("algorithm", "demo_algorithm")
        
        print("Performed custom operation with telemetry")
    
    # Get metrics from custom instance
    custom_metrics = custom_telemetry.get_metrics()
    print(f"Custom telemetry metrics: {custom_metrics}")


def main():
    """Run all demonstrations."""
    print("Canonical Flow Process Interface & Telemetry Demonstration")
    print("=" * 60)
    
    try:
        demonstrate_adapters()
        demonstrate_telemetry()
        demonstrate_custom_telemetry()
        demonstrate_instrumentation()
        
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Final metrics report
        print("\n=== Final Telemetry Report ===")
        final_metrics = pipeline_telemetry.get_metrics()
        
        total_calls = sum(m['total_calls'] for m in final_metrics.values())
        total_time = sum(m['total_time_ms'] for m in final_metrics.values())
        
        print(f"Total operations traced: {total_calls}")
        print(f"Total execution time: {total_time:.2f}ms")
        print(f"Components instrumented: {len(final_metrics)}")
        
        if final_metrics:
            print("\nPer-component breakdown:")
            for component, metrics in final_metrics.items():
                print(f"  {component}: {metrics['total_calls']} calls, "
                      f"{metrics['average_time_ms']:.2f}ms avg")


if __name__ == "__main__":
    main()