"""
Demo script for OpenTelemetry tracing system
"""

import time
import threading
from typing import Dict, Any

from .otel_tracer import get_pipeline_tracer, set_pipeline_tracer, PipelineTracer
from .dashboard import TracingDashboard
from .integration import auto_instrument_pipeline
from .back_edge_detector import BackEdgeDetector


def simulate_pipeline_execution():
    """Simulate canonical pipeline execution with tracing"""
    print("Simulating canonical pipeline execution...")
    
    tracer = get_pipeline_tracer()
    
    # Simulate forward flow: I→X→K→A→L→R→O→G→T→S
    canonical_flow = [
        ('I_ingestion_preparation', 'X_context_construction', 'pdf_reader'),
        ('X_context_construction', 'K_knowledge_extraction', 'context_adapter'),
        ('K_knowledge_extraction', 'A_analysis_nlp', 'entity_extractor'),
        ('A_analysis_nlp', 'L_classification_evaluation', 'question_analyzer'), 
        ('L_classification_evaluation', 'R_search_retrieval', 'score_calculator'),
        ('R_search_retrieval', 'O_orchestration_control', 'hybrid_retriever'),
        ('O_orchestration_control', 'G_aggregation_reporting', 'core_orchestrator'),
        ('G_aggregation_reporting', 'T_integration_storage', 'report_compiler'),
        ('T_integration_storage', 'S_synthesis_output', 'metrics_collector'),
    ]
    
    # Execute normal forward flow
    for i, (source, target, component) in enumerate(canonical_flow):
        print(f"Step {i+1}: {source} -> {target} ({component})")
        
        # Create span
        span_id = tracer.create_edge_span(
            source_phase=source,
            target_phase=target,
            component_name=component,
            data={"step": i+1, "sample_data": f"data_{i}"},
            context={"pipeline_run": "demo", "timestamp": time.time()}
        )
        
        # Simulate processing time
        processing_time = 0.1 + (i * 0.05)  # Increasing complexity
        time.sleep(processing_time)
        
        # Complete span
        tracer.complete_edge_span(
            span_id, 
            result={"processed": True, "output_size": 100 + (i * 50)}
        )
        
    print("Forward flow completed")
    
    # Simulate some violations for testing
    print("Simulating dependency violations...")
    
    # Backward edge violation
    span_id = tracer.create_edge_span(
        source_phase='L_classification_evaluation',
        target_phase='K_knowledge_extraction', 
        component_name='score_calculator_backtrack',
        data={"violation": "backward_edge"},
        context={"type": "violation_test"}
    )
    time.sleep(0.1)
    tracer.complete_edge_span(span_id, result={"violation_detected": True})
    
    # Phase skip violation
    span_id = tracer.create_edge_span(
        source_phase='I_ingestion_preparation',
        target_phase='A_analysis_nlp',  # Skipping X and K phases
        component_name='direct_analyzer',
        data={"violation": "phase_skip"},
        context={"type": "violation_test"}
    )
    time.sleep(0.1)
    tracer.complete_edge_span(span_id, result={"phase_skip": True})
    
    print("Violation simulation completed")


def run_comprehensive_demo():
    """Run comprehensive demo of tracing system"""
    print("=== OpenTelemetry Pipeline Tracing Demo ===")
    
    # Initialize tracer with exporters (optional)
    tracer = PipelineTracer(
        service_name="canonical-pipeline-demo",
        # jaeger_endpoint="http://localhost:14268/api/traces",  # Uncomment if Jaeger available
        # otlp_endpoint="http://localhost:4317"  # Uncomment if OTLP collector available
    )
    set_pipeline_tracer(tracer)
    
    # Start dashboard
    dashboard = TracingDashboard(host='localhost', port=8080)
    dashboard.start()
    
    print(f"Dashboard available at: {dashboard.get_dashboard_url()}")
    print("Press Ctrl+C to stop the demo")
    
    try:
        # Auto-instrument pipeline components (if available)
        try:
            instrumentor = auto_instrument_pipeline()
            report = instrumentor.get_instrumentation_report()
            print(f"Auto-instrumented {report['total_instrumented']} components")
        except Exception as e:
            print(f"Auto-instrumentation warning: {e}")
            print("Continuing with manual simulation...")
            
        # Start continuous monitoring
        monitor_thread = dashboard.run_continuous_monitoring(check_interval=10)
        
        # Simulate pipeline execution multiple times
        for run in range(1, 6):
            print(f"\n--- Pipeline Run {run} ---")
            simulate_pipeline_execution()
            time.sleep(2)  # Brief pause between runs
            
        # Analyze violations
        detector = BackEdgeDetector()
        violations = detector.analyze_span_traces(time_window_minutes=5)
        
        print(f"\n--- Violation Analysis ---")
        print(f"Detected {len(violations)} violations:")
        
        for violation in violations:
            print(f"  - {violation.violation_type}: {violation.component_path}")
            print(f"    Path: {' -> '.join(violation.dependency_path)}")
            print(f"    Severity: {violation.severity}")
            print(f"    Description: {violation.description}")
            print()
            
        # Print metrics
        print(f"\n--- Metrics Summary ---")
        frequencies = tracer.get_edge_frequencies(time_window_minutes=5)
        print("Edge frequencies:")
        for edge, count in sorted(frequencies.items(), key=lambda x: x[1], reverse=True):
            print(f"  {edge}: {count}")
            
        latencies = tracer.get_latency_distributions(time_window_minutes=5)
        print("\nLatency statistics:")
        for edge, times in latencies.items():
            if times:
                avg_ms = (sum(times) / len(times)) * 1000
                print(f"  {edge}: {avg_ms:.2f}ms avg")
                
        # Keep dashboard running
        print(f"\nDashboard still running at: {dashboard.get_dashboard_url()}")
        print("View violations at: http://localhost:8080/violations")
        print("Press Enter to stop...")
        input()
        
    except KeyboardInterrupt:
        print("\nStopping demo...")
    finally:
        dashboard.stop()
        print("Demo completed")


def quick_test():
    """Quick test of tracing functionality"""
    print("Quick tracing test...")
    
    tracer = get_pipeline_tracer()
    
    # Simple trace
    span_id = tracer.create_edge_span(
        source_phase='I_ingestion_preparation',
        target_phase='X_context_construction',
        component_name='test_component',
        data={"test": True}
    )
    
    time.sleep(0.1)
    tracer.complete_edge_span(span_id, result={"success": True})
    
    # Check results
    spans = tracer.get_span_data(1)
    print(f"Recorded {len(spans)} spans")
    
    if spans:
        span = spans[0]
        duration = span.timing_end - span.timing_start if span.timing_end else 0
        print(f"Span: {span.source_phase} -> {span.target_phase}")
        print(f"Duration: {duration*1000:.2f}ms")
        print(f"Component: {span.component_name}")
        
    print("Quick test completed")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_test()
    else:
        run_comprehensive_demo()