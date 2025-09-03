"""
Example usage of OpenTelemetry tracing system for canonical pipeline
"""

import time
from typing import Dict, Any, Optional

# Import our tracing system
from tracing import (
    trace_process, 
    trace_edge,
    auto_trace_process,
    get_pipeline_tracer,
    TracingDashboard,
    auto_instrument_pipeline
)


# Example 1: Manual instrumentation of a component
class ExampleIngestionComponent:
    """Example component in I_ingestion_preparation phase"""
    
    @trace_process(
        source_phase='I_ingestion_preparation',
        target_phase='X_context_construction', 
        component_name='example_pdf_reader'
    )
    def process(self, data=None, context=None) -> Dict[str, Any]:
        """Process method with manual tracing instrumentation"""
        print(f"Processing in ingestion phase: {data}")
        
        # Simulate processing
        time.sleep(0.1)
        
        return {
            "processed_data": f"ingested_{data}",
            "phase": "I_ingestion_preparation",
            "status": "success"
        }


# Example 2: Using simplified edge tracing
class ExampleContextComponent:
    """Example component in X_context_construction phase"""
    
    @trace_edge('X_context_construction', 'K_knowledge_extraction')
    def process(self, data=None, context=None) -> Dict[str, Any]:
        """Process method with simplified edge tracing"""
        print(f"Building context: {data}")
        
        time.sleep(0.05)
        
        return {
            "context_data": f"context_{data}",
            "phase": "X_context_construction",
            "metadata": {"enriched": True}
        }


# Example 3: Auto-tracing (infers phase from module location)
class ExampleKnowledgeComponent:
    """Example component that uses auto-detection"""
    
    @auto_trace_process
    def process(self, data=None, context=None) -> Dict[str, Any]:
        """Process method with automatic tracing"""
        print(f"Extracting knowledge: {data}")
        
        time.sleep(0.2)
        
        return {
            "knowledge_graph": {"entities": ["example"], "relations": []},
            "phase": "K_knowledge_extraction",
            "extracted_concepts": 5
        }


# Example 4: Function-level tracing (not class-based)
@trace_process(
    source_phase='A_analysis_nlp',
    target_phase='L_classification_evaluation',
    component_name='example_analyzer'
)
def analyze_text(data=None, context=None) -> Dict[str, Any]:
    """Standalone function with tracing"""
    print(f"Analyzing text: {data}")
    
    time.sleep(0.15)
    
    return {
        "analysis_results": {"sentiment": "positive", "topics": ["example"]},
        "phase": "A_analysis_nlp",
        "confidence": 0.85
    }


# Example 5: Simulating backward dependency violation
@trace_process(
    source_phase='L_classification_evaluation', 
    target_phase='K_knowledge_extraction',  # VIOLATION: backward edge
    component_name='violating_component'
)
def violating_process(data=None, context=None) -> Dict[str, Any]:
    """Example of a component that creates a dependency violation"""
    print("WARNING: This creates a backward dependency violation!")
    
    time.sleep(0.05)
    
    return {
        "violation": True,
        "reason": "backward_edge_to_knowledge_phase"
    }


def run_tracing_example():
    """Run complete tracing example"""
    print("=== OpenTelemetry Pipeline Tracing Example ===")
    
    # Start dashboard server
    dashboard = TracingDashboard(host='localhost', port=8080)
    dashboard.start()
    
    print(f"Dashboard started: {dashboard.get_dashboard_url()}")
    
    try:
        # Create component instances
        ingestion = ExampleIngestionComponent()
        context_builder = ExampleContextComponent()
        knowledge_extractor = ExampleKnowledgeComponent()
        
        # Simulate pipeline execution
        print("\n--- Executing Pipeline ---")
        
        # Forward flow execution
        sample_data = "example_document.pdf"
        sample_context = {"run_id": "example_001", "timestamp": time.time()}
        
        # Step 1: Ingestion
        result1 = ingestion.process(sample_data, sample_context)
        
        # Step 2: Context construction
        result2 = context_builder.process(result1, sample_context)
        
        # Step 3: Knowledge extraction
        result3 = knowledge_extractor.process(result2, sample_context)
        
        # Step 4: Analysis
        result4 = analyze_text(result3, sample_context)
        
        print("Forward pipeline completed successfully")
        
        # Demonstrate violation detection
        print("\n--- Simulating Violations ---")
        
        # Create a violation (this will be detected by back-edge detector)
        violation_result = violating_process(result4, sample_context)
        
        # Wait a moment for spans to be recorded
        time.sleep(1)
        
        # Analyze traces for violations
        from tracing.back_edge_detector import BackEdgeDetector
        detector = BackEdgeDetector()
        violations = detector.analyze_span_traces(time_window_minutes=5)
        
        print(f"Detected {len(violations)} dependency violations:")
        for v in violations:
            print(f"  - {v.violation_type}: {v.component_path}")
            print(f"    Path: {' -> '.join(v.dependency_path)}")
            print(f"    Severity: {v.severity}")
            
        # Show metrics
        tracer = get_pipeline_tracer()
        frequencies = tracer.get_edge_frequencies(5)
        
        print(f"\n--- Edge Traversal Frequencies ---")
        for edge, count in frequencies.items():
            print(f"  {edge}: {count}")
            
        latencies = tracer.get_latency_distributions(5)
        print(f"\n--- Latency Statistics ---")
        for edge, times in latencies.items():
            if times:
                avg_ms = (sum(times) / len(times)) * 1000
                min_ms = min(times) * 1000  
                max_ms = max(times) * 1000
                print(f"  {edge}: {avg_ms:.1f}ms avg ({min_ms:.1f}-{max_ms:.1f}ms)")
        
        print(f"\nDashboards available:")
        print(f"  - Main: http://localhost:8080/dashboard")
        print(f"  - Violations: http://localhost:8080/violations")
        print(f"  - API: http://localhost:8080/api/heatmap")
        
        print("\nPress Enter to stop the example...")
        input()
        
    except KeyboardInterrupt:
        print("\nExample interrupted")
    finally:
        dashboard.stop()
        print("Example completed")


def run_auto_instrumentation_example():
    """Example of automatic pipeline instrumentation"""
    print("=== Auto-Instrumentation Example ===")
    
    try:
        # Attempt to auto-instrument the entire canonical pipeline
        instrumentor = auto_instrument_pipeline()
        report = instrumentor.get_instrumentation_report()
        
        print(f"Successfully auto-instrumented {report['total_instrumented']} components")
        print("Instrumentation by phase:")
        for phase, count in report['by_phase'].items():
            print(f"  {phase}: {count} components")
            
        print("\nInstrumented components:")
        for component in report['components']:
            print(f"  - {component}")
            
    except Exception as e:
        print(f"Auto-instrumentation failed: {e}")
        print("This is expected if canonical_flow modules are not available")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "auto":
        run_auto_instrumentation_example()
    else:
        run_tracing_example()