"""
Validation script for OpenTelemetry tracing system
"""

import sys
import time
from typing import Dict, Any

def test_basic_imports():
    """Test that all tracing modules can be imported"""
    print("Testing imports...")
    
    try:
        from tracing import (
            PipelineTracer,
            trace_process,
            trace_edge,
            DependencyHeatmapVisualizer,
            BackEdgeDetector,
            TracingDashboard,
            get_pipeline_tracer,
            set_pipeline_tracer
        )
        print("✓ All core imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_tracer_functionality():
    """Test basic tracer functionality"""
    print("\nTesting tracer functionality...")
    
    try:
        from tracing import PipelineTracer, get_pipeline_tracer, set_pipeline_tracer
        
        # Create tracer
        tracer = PipelineTracer(service_name="validation-test")
        set_pipeline_tracer(tracer)
        
        # Test span creation
        span_id = tracer.create_edge_span(
            source_phase='I_ingestion_preparation',
            target_phase='X_context_construction',
            component_name='validation_test_component',
            data={"test": "data"},
            context={"validation": True}
        )
        
        assert span_id is not None
        assert span_id in tracer.active_spans
        print("✓ Span creation works")
        
        # Test span completion
        time.sleep(0.01)
        tracer.complete_edge_span(span_id, result={"success": True})
        
        assert span_id not in tracer.active_spans
        assert len(tracer.span_history) == 1
        print("✓ Span completion works")
        
        # Test metrics
        frequencies = tracer.get_edge_frequencies()
        assert frequencies['I_ingestion_preparation->X_context_construction'] == 1
        print("✓ Metrics collection works")
        
        return True
        
    except Exception as e:
        print(f"✗ Tracer functionality failed: {e}")
        return False


def test_decorators():
    """Test tracing decorators"""
    print("\nTesting tracing decorators...")
    
    try:
        from tracing import trace_process, get_pipeline_tracer
        
        @trace_process(
            source_phase='I_ingestion_preparation',
            target_phase='X_context_construction',
            component_name='decorated_test_function'
        )
        def test_function(data=None, context=None):
            time.sleep(0.01)
            return {"decorated": True}
            
        # Execute decorated function
        result = test_function({"input": "test"}, {"decorator": "test"})
        
        assert result["decorated"] is True
        print("✓ Function decoration works")
        
        # Check that span was recorded
        tracer = get_pipeline_tracer()
        spans = tracer.get_span_data(1)
        decorated_spans = [s for s in spans if s.component_name == 'decorated_test_function']
        assert len(decorated_spans) >= 1
        print("✓ Decorated function tracing works")
        
        return True
        
    except Exception as e:
        print(f"✗ Decorator test failed: {e}")
        return False


def test_back_edge_detection():
    """Test back-edge violation detection"""
    print("\nTesting back-edge detection...")
    
    try:
        from tracing import BackEdgeDetector, get_pipeline_tracer
        
        tracer = get_pipeline_tracer()
        detector = BackEdgeDetector()
        
        # Create a backward edge violation
        span_id = tracer.create_edge_span(
            source_phase='L_classification_evaluation',
            target_phase='I_ingestion_preparation',  # Backward!
            component_name='violating_test_component'
        )
        tracer.complete_edge_span(span_id)
        
        # Detect violations
        violations = detector.analyze_span_traces()
        
        # Should find the backward edge
        back_edge_violations = [
            v for v in violations 
            if v.violation_type == 'direct_back_edge'
        ]
        
        assert len(back_edge_violations) >= 1
        violation = back_edge_violations[0]
        assert violation.source_phase == 'L_classification_evaluation'
        assert violation.target_phase == 'I_ingestion_preparation'
        assert violation.severity == 'critical'
        print("✓ Back-edge detection works")
        
        # Test violation summary
        summary = detector.get_violation_summary()
        assert summary['total_violations'] >= 1
        assert 'by_type' in summary
        assert 'by_severity' in summary
        print("✓ Violation summary works")
        
        return True
        
    except Exception as e:
        print(f"✗ Back-edge detection failed: {e}")
        return False


def test_visualization():
    """Test heatmap visualization"""
    print("\nTesting visualization...")
    
    try:
        from tracing import DependencyHeatmapVisualizer, get_pipeline_tracer
        
        visualizer = DependencyHeatmapVisualizer()
        
        # Generate heatmap data
        data = visualizer.generate_heatmap_data()
        
        assert 'timestamp' in data
        assert 'edge_frequencies' in data
        assert 'canonical_phases' in data
        print("✓ Heatmap data generation works")
        
        # Generate HTML dashboard
        html = visualizer.generate_html_dashboard()
        
        assert isinstance(html, str)
        assert '<html>' in html
        assert 'Pipeline Dependency Heatmap' in html
        print("✓ HTML dashboard generation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False


def test_dashboard_creation():
    """Test dashboard server creation (without starting)"""
    print("\nTesting dashboard creation...")
    
    try:
        from tracing import TracingDashboard
        
        # Create dashboard (don't start server)
        dashboard = TracingDashboard(host='localhost', port=8081)
        
        assert dashboard.host == 'localhost'
        assert dashboard.port == 8081
        assert dashboard.visualizer is not None
        assert dashboard.detector is not None
        print("✓ Dashboard creation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Dashboard creation failed: {e}")
        return False


def test_integration_example():
    """Test a complete integration example"""
    print("\nTesting integration example...")
    
    try:
        from tracing import trace_process, get_pipeline_tracer, BackEdgeDetector
        
        # Create a simple traced component
        @trace_process(
            source_phase='I_ingestion_preparation',
            target_phase='X_context_construction',
            component_name='integration_example'
        )
        def example_component(data=None, context=None):
            time.sleep(0.01)
            return {"integration_success": True}
            
        # Execute the component
        result = example_component(
            data={"document": "example.pdf"},
            context={"run_id": "integration_test"}
        )
        
        assert result["integration_success"] is True
        print("✓ Component execution works")
        
        # Verify tracing worked
        tracer = get_pipeline_tracer()
        spans = tracer.get_span_data(1)
        integration_spans = [s for s in spans if s.component_name == 'integration_example']
        assert len(integration_spans) >= 1
        print("✓ Integration tracing works")
        
        # Test violation detection on the spans
        detector = BackEdgeDetector()
        violations = detector.analyze_span_traces()
        
        # Should not detect violations for forward edges
        back_edges = [v for v in violations if v.violation_type == 'direct_back_edge']
        forward_only = [v for v in back_edges if v.component_path == 'integration_example']
        assert len(forward_only) == 0
        print("✓ No false positive violations")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def main():
    """Run all validation tests"""
    print("=== OpenTelemetry Tracing System Validation ===\n")
    
    tests = [
        test_basic_imports,
        test_tracer_functionality,
        test_decorators,
        test_back_edge_detection,
        test_visualization,
        test_dashboard_creation,
        test_integration_example,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} crashed: {e}")
            failed += 1
            
    print(f"\n=== Validation Summary ===")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tracing system components are working correctly!")
        print("\nNext steps:")
        print("1. Run the demo: python -m tracing.demo")
        print("2. Try the example: python examples/tracing_example.py") 
        print("3. Integrate with your pipeline components")
        return True
    else:
        print(f"\n⚠️ {failed} tests failed. Please check the error messages above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)