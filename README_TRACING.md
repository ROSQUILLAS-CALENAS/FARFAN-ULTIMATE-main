# OpenTelemetry Pipeline Tracing System

A comprehensive OpenTelemetry-based tracing system for monitoring canonical pipeline DAG execution, detecting dependency violations, and providing real-time visualization dashboards.

## Features

### 🔍 Edge Traversal Instrumentation
- **Automatic span creation/completion** for every `process(data, context)` interface
- **Rich metadata** including source phase, target phase, component name, and timing information
- **Error tracking** with exception details and stack traces
- **Data size metrics** for performance analysis

### 📊 Live Dependency Heatmap
- **Real-time HTML dashboards** with automatic refresh capabilities
- **Edge frequency visualization** showing pipeline execution patterns
- **Latency distribution analysis** with P95/P99 metrics
- **Phase transition pattern detection** for bottleneck identification

### ⚠️ Back-Edge Detection
- **Canonical ordering validation** against I→X→K→A→L→R→O→G→T→S sequence
- **Multiple violation types**: direct back-edges, cyclic dependencies, phase skips
- **Severity classification**: critical, warning, info levels
- **Component-specific violation paths** for immediate remediation

## Quick Start

### 1. Installation

Add OpenTelemetry dependencies to your environment:

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation \
            opentelemetry-exporter-jaeger opentelemetry-exporter-otlp-proto-grpc \
            opentelemetry-instrumentation-logging
```

### 2. Basic Usage

```python
from tracing import trace_process, TracingDashboard

# Instrument a component
@trace_process(
    source_phase='I_ingestion_preparation',
    target_phase='X_context_construction', 
    component_name='pdf_reader'
)
def process(data=None, context=None):
    # Your processing logic
    return {"processed": True}

# Start dashboard
dashboard = TracingDashboard()
dashboard.start()
print(f"Dashboard: {dashboard.get_dashboard_url()}")
```

### 3. Auto-Instrumentation

```python
from tracing import auto_instrument_pipeline

# Automatically instrument all canonical_flow components
instrumentor = auto_instrument_pipeline()
report = instrumentor.get_instrumentation_report()
print(f"Instrumented {report['total_instrumented']} components")
```

## Architecture

### Core Components

#### PipelineTracer
Central OpenTelemetry tracer managing span lifecycle:

```python
from tracing import get_pipeline_tracer

tracer = get_pipeline_tracer()

# Create edge span
span_id = tracer.create_edge_span(
    source_phase='I_ingestion_preparation',
    target_phase='X_context_construction',
    component_name='pdf_reader',
    data=document_data,
    context=processing_context
)

# Complete span
tracer.complete_edge_span(span_id, result=output_data)
```

#### Decorators
Multiple instrumentation approaches:

```python
# Manual specification
@trace_process('I_ingestion_preparation', 'X_context_construction', 'pdf_reader')
def process(data=None, context=None):
    return process_document(data)

# Simplified edge tracing
@trace_edge('I_ingestion_preparation', 'X_context_construction')  
def process(data=None, context=None):
    return process_document(data)

# Auto-detection from module path
@auto_trace_process
def process(data=None, context=None):
    return process_document(data)
```

#### Back-Edge Detector
Violation detection with multiple algorithms:

```python
from tracing import BackEdgeDetector

detector = BackEdgeDetector()
violations = detector.analyze_span_traces(time_window_minutes=60)

for violation in violations:
    print(f"{violation.violation_type}: {violation.component_path}")
    print(f"Path: {' -> '.join(violation.dependency_path)}")
    print(f"Severity: {violation.severity}")
```

#### Dashboard System
Live monitoring with multiple views:

```python
from tracing import TracingDashboard

dashboard = TracingDashboard(host='localhost', port=8080)
dashboard.start()

# Available endpoints:
# http://localhost:8080/dashboard - Main heatmap
# http://localhost:8080/violations - Violations view  
# http://localhost:8080/api/heatmap - JSON API
# http://localhost:8080/api/violations - Violations API
```

## Canonical Phase Ordering

The system enforces strict adherence to the canonical pipeline sequence:

```
I → X → K → A → L → R → O → G → T → S
```

| Phase | Full Name | Purpose |
|-------|-----------|---------|
| **I** | Ingestion Preparation | Document loading and initial processing |
| **X** | Context Construction | Context building and preparation |
| **K** | Knowledge Extraction | Entity/concept extraction and knowledge graphs |
| **A** | Analysis NLP | Text analysis and NLP processing |
| **L** | Classification Evaluation | Scoring and classification |
| **R** | Search Retrieval | Information retrieval and search |
| **O** | Orchestration Control | Workflow orchestration and control |
| **G** | Aggregation Reporting | Result aggregation and reporting |
| **T** | Integration Storage | Data integration and storage |
| **S** | Synthesis Output | Final output synthesis |

## Violation Types

### Direct Back-Edges (Critical)
Components that directly reference earlier phases:

```python
# VIOLATION: L → K (backward edge)
@trace_process('L_classification_evaluation', 'K_knowledge_extraction', 'bad_component')
def process(data=None, context=None):
    return reprocess_knowledge(data)  # Should not happen!
```

### Cyclic Dependencies (Critical)
Circular references between phases:

```
A → L → R → A  # Creates cycle
```

### Phase Ordering Violations (Warning)
Execution order not following canonical sequence:

```
I → K → A  # Skipped X phase
```

### Phase Skips (Info)
Missing intermediate phases in execution flow.

## Dashboard Features

### Main Dashboard (`/dashboard`)
- **Dependency heatmap** with color-coded frequency visualization
- **Latency statistics** table with percentile breakdowns  
- **Phase activity** counters and metrics
- **Error tracking** with component-specific counts
- **Canonical flow** visualization

### Violations Dashboard (`/violations`)
- **Real-time violation detection** with severity indicators
- **Violation type breakdowns** and component mapping
- **Historical violation tracking** with timestamps
- **Remediation guidance** with specific component paths

### JSON APIs
- `GET /api/heatmap?window=60` - Heatmap data for time window
- `GET /api/violations?window=60` - Violations data for time window
- `GET /health` - System health check

## Performance Considerations

### Span Collection
- **Asynchronous processing** with batched span exports
- **Memory-efficient** history management with configurable retention
- **Sampling support** for high-throughput environments

### Dashboard Rendering
- **Server-side generation** for fast loading
- **Configurable refresh intervals** (default 5 seconds)
- **Time window controls** for historical analysis

### Storage
- **In-memory span storage** for development/testing
- **External exporters** (Jaeger, OTLP) for production
- **Configurable retention policies** for long-term storage

## Integration Examples

### Existing Component Integration

```python
# Before: Untraced component
def process(data=None, context=None):
    return {"result": "processed"}

# After: Add single decorator
from tracing import trace_process

@trace_process('I_ingestion_preparation', 'X_context_construction', 'my_component')
def process(data=None, context=None):
    return {"result": "processed"}
```

### Full Pipeline Integration

```python
from tracing import auto_instrument_pipeline, TracingDashboard

# Auto-instrument entire pipeline
instrumentor = auto_instrument_pipeline()

# Start monitoring dashboard  
dashboard = TracingDashboard()
dashboard.start()

# Run continuous violation monitoring
monitor_thread = dashboard.run_continuous_monitoring()

# Your pipeline execution...
run_canonical_pipeline()

# Dashboard available at http://localhost:8080
```

### Custom Exporter Integration

```python
from tracing import PipelineTracer, set_pipeline_tracer

# Initialize with custom exporters
tracer = PipelineTracer(
    service_name="my-pipeline",
    jaeger_endpoint="http://localhost:14268/api/traces",
    otlp_endpoint="http://localhost:4317"
)
set_pipeline_tracer(tracer)

# All subsequent tracing will use these exporters
```

## Configuration

### Environment Variables

```bash
# OpenTelemetry configuration
export OTEL_SERVICE_NAME="canonical-pipeline"
export OTEL_EXPORTER_JAEGER_ENDPOINT="http://localhost:14268/api/traces"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# Dashboard configuration  
export TRACING_DASHBOARD_HOST="0.0.0.0"
export TRACING_DASHBOARD_PORT="8080"
export TRACING_REFRESH_INTERVAL="5"
```

### Programmatic Configuration

```python
from tracing import TracingDashboard, PipelineTracer

# Custom dashboard configuration
dashboard = TracingDashboard(
    host='0.0.0.0',
    port=8080,
    refresh_interval=10  # 10 second refresh
)

# Custom tracer configuration
tracer = PipelineTracer(
    service_name="production-pipeline",
    jaeger_endpoint="http://jaeger:14268/api/traces"
)
```

## Monitoring and Alerting

### Critical Violation Alerts

```python
from tracing import BackEdgeDetector

detector = BackEdgeDetector(log_level="CRITICAL")
violations = detector.analyze_span_traces()

critical_violations = [v for v in violations if v.severity == 'critical']
if critical_violations:
    # Send alert to monitoring system
    send_alert(f"CRITICAL: {len(critical_violations)} dependency violations detected")
```

### Performance Monitoring

```python
from tracing import get_pipeline_tracer

tracer = get_pipeline_tracer()
latencies = tracer.get_latency_distributions()

for edge, times in latencies.items():
    if times and max(times) > 5.0:  # 5 second threshold
        print(f"HIGH LATENCY WARNING: {edge} max={max(times):.2f}s")
```

### Health Checks

```bash
# Check dashboard health
curl http://localhost:8080/health

# Response:
{
    "status": "healthy",
    "active_spans": 5,
    "total_spans": 1234,
    "timestamp": 1640995200.0
}
```

## Troubleshooting

### Common Issues

1. **No spans appearing**: Verify OpenTelemetry exporters are configured
2. **Dashboard not loading**: Check port availability and firewall settings
3. **Auto-instrumentation failing**: Ensure canonical_flow modules are importable
4. **Performance impact**: Adjust sampling rates or disable in production

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from tracing import get_pipeline_tracer
tracer = get_pipeline_tracer()
# Detailed logging will now be available
```

### Manual Testing

```python
from tracing.demo import quick_test, run_comprehensive_demo

# Quick functionality test
quick_test()

# Full demo with violations
run_comprehensive_demo()
```

## Examples

See `examples/tracing_example.py` for comprehensive usage examples and `tracing/demo.py` for a full demonstration of all features.

Run the examples:

```bash
# Basic example
python examples/tracing_example.py

# Comprehensive demo
python -m tracing.demo

# Quick test
python -m tracing.demo quick
```