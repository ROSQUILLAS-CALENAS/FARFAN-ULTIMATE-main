# DAG Observability System

A comprehensive observability system for DAG (Directed Acyclic Graph) execution with OpenTelemetry tracing instrumentation, live dependency heatmaps, and circular dependency detection.

## Features

### 🔍 OpenTelemetry Tracing Instrumentation
- **Pipeline Edge Traversal**: Captures span data for each edge traversal in the DAG
- **Component Entry/Exit**: Tracks component lifecycle with detailed timing metrics
- **Phase Transitions**: Monitors transitions between different processing phases
- **Dependency Relationships**: Maps component dependencies with correlation IDs
- **Timing Metrics**: Comprehensive latency and performance measurements

### 📊 Span Export Configuration
- **OTLP Endpoint Support**: Direct export to OpenTelemetry collectors
- **JSON File Fallback**: Local development support with JSON file export
- **Multiple Export Formats**: Console output for debugging
- **Batch Processing**: Efficient span batching and export

### 🔥 Live Dependency Heatmap Generation
- **Edge Traversal Frequency**: Visualizes most frequently used paths
- **Latency Patterns**: Identifies high-latency components and edges
- **Hot Path Detection**: Real-time identification of performance bottlenecks
- **Component Load Distribution**: Monitors active executions and throughput
- **Interactive Visualizations**: Web-based dashboards with Plotly

### ⚠️ Residual Back-Edge Detection
- **Runtime Circular Dependency Monitoring**: Detects cycles during execution
- **Active Span Context Tracking**: Maintains execution stack state
- **Violation Detection**: Identifies when components invoke dependencies already in the stack
- **Alerting System**: Real-time alerts with detailed violation reports
- **Severity Classification**: High/Medium/Low severity based on cycle characteristics

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# The system requires OpenTelemetry packages:
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
```

## Quick Start

### Basic Usage

```python
from tracing.dag_observability import initialize_dag_observability
from tracing.dag_observability import trace_component_execution, trace_edge_traversal

# Initialize the observability system
obs_system = initialize_dag_observability(
    service_name="my_dag_pipeline",
    otlp_endpoint="http://localhost:4317",  # Optional OTLP endpoint
    json_fallback_path="tracing/spans.json",
    enable_heatmap=True,
    enable_circular_detection=True
)

# Trace component execution
async def process_data(data):
    with trace_component_execution("data_processor", "processing"):
        # Your component logic here
        result = await some_processing(data)
        return result

# Trace edge traversal
async def pipeline_step(data):
    with trace_edge_traversal("source_component", "target_component"):
        result = await process_data(data)
        return result
```

### Phase Transitions

```python
from tracing.dag_observability import trace_phase_transition

async def complex_component(data):
    with trace_component_execution("complex_processor"):
        # Phase 1: Initialization
        with trace_phase_transition("complex_processor", "init", "validation"):
            await initialize_resources()
        
        # Phase 2: Processing
        with trace_phase_transition("complex_processor", "validation", "processing"):
            result = await main_processing(data)
        
        # Phase 3: Cleanup
        with trace_phase_transition("complex_processor", "processing", "cleanup"):
            await cleanup_resources()
        
        return result
```

### Correlation IDs

```python
import uuid

async def pipeline_execution():
    correlation_id = str(uuid.uuid4())
    
    # All components in this execution will share the correlation ID
    with trace_component_execution("ingestion", correlation_id=correlation_id):
        data = await ingest_data()
    
    with trace_edge_traversal("ingestion", "processing", correlation_id):
        with trace_component_execution("processing", correlation_id=correlation_id):
            result = await process_data(data)
    
    return result
```

## Configuration Options

### Observability System Configuration

```python
obs_system = initialize_dag_observability(
    service_name="my_service",           # Service name for tracing
    otlp_endpoint="http://localhost:4317", # OTLP collector endpoint
    json_fallback_path="tracing/spans.json", # Local JSON export path
    enable_heatmap=True,                 # Enable live heatmap generation
    enable_circular_detection=True       # Enable circular dependency detection
)
```

### OTLP Endpoint Options

```python
# HTTP endpoint
otlp_endpoint="http://localhost:4318"

# gRPC endpoint  
otlp_endpoint="http://localhost:4317"

# Jaeger
otlp_endpoint="http://localhost:14268"

# No OTLP (JSON only)
otlp_endpoint=None
```

## Monitoring and Visualization

### Live Heatmap Data

```python
# Get current heatmap data
heatmap_data = obs_system.get_heatmap_data()

# Contains:
# - hot_paths: Most frequently traversed edges
# - high_latency_paths: Slowest edges
# - component_load: Component performance metrics
# - total_edges/components: System overview
```

### Circular Dependency Monitoring

```python
# Get current violations
violations = obs_system.get_circular_dependency_violations()

# Get dependency graph
dependency_graph = obs_system.get_dependency_graph()

# Get real-time alerts
alerts = obs_system.get_alerts()

# Resolve a violation
obs_system.resolve_violation("violation_id")
```

### Performance Dashboard

```python
from tracing.visualization import create_performance_dashboard

# Create comprehensive dashboard
dashboard_file = create_performance_dashboard(
    heatmap_data=heatmap_data,
    violations=violations,
    dependency_graph=dependency_graph,
    output_dir="dashboards"
)

print(f"Dashboard available at: {dashboard_file}")
```

## Running the Demo

A comprehensive demo is provided to showcase all features:

```bash
python tracing/integration_demo.py
```

The demo includes:
1. **Normal Pipeline Execution**: Process multiple items through a mock DAG
2. **Circular Dependency Detection**: Simulate and detect circular dependencies
3. **Performance Reports**: Generate heatmaps and visualizations
4. **Real-time Monitoring**: Continuous processing with live metrics

### Demo Output

The demo generates several outputs:
- `tracing/demo_spans.json` - Exported span data
- `tracing/demo_visualizations/` - Generated dashboards and reports
- `tracing/live_heatmap.json` - Real-time heatmap data
- `tracing/demo_final_report.json` - Comprehensive summary

## Architecture

### Core Components

1. **DAGObservabilitySystem**: Main coordinator class
2. **LiveHeatmapGenerator**: Real-time metrics and visualization
3. **CircularDependencyDetector**: Cycle detection and alerting
4. **JSONSpanExporter**: Custom span export functionality
5. **DependencyHeatmapVisualizer**: Interactive dashboard generation

### Data Flow

```
Component Execution → Span Creation → Metrics Update → Visualization
                                   ↓
Circular Detection ← Stack Tracking ← Context Management
                                   ↓  
Alerts/Violations → Dashboard Updates → Performance Reports
```

### Thread Safety

- All components are thread-safe with appropriate locking
- Background threads for metrics collection and visualization
- Queue-based alerting system for non-blocking notifications

## Integration with Existing Systems

### Pipeline Orchestrators

```python
# Integrate with existing pipeline orchestrator
class MyOrchestrator:
    def __init__(self):
        self.obs_system = initialize_dag_observability("my_pipeline")
    
    async def execute_component(self, component_name, data):
        with trace_component_execution(component_name):
            return await self.components[component_name].process(data)
    
    async def traverse_dependency(self, source, target, data):
        with trace_edge_traversal(source, target):
            return await self.execute_component(target, data)
```

### Workflow Engines

```python
# Airflow DAG integration example
from airflow import DAG
from airflow.operators.python import PythonOperator

def traced_task(**context):
    with trace_component_execution(context['task'].task_id):
        # Your task logic
        return process_data()

dag = DAG('observed_dag', ...)
task = PythonOperator(
    task_id='processing_task',
    python_callable=traced_task,
    dag=dag
)
```

## Performance Considerations

### Overhead
- Minimal overhead (~1-5ms per span)
- Async-friendly with non-blocking exports
- Configurable batch sizes and export intervals

### Memory Usage
- Bounded queues for metrics and alerts
- Configurable retention periods
- Automatic cleanup of old data

### Scalability
- Distributed tracing with correlation IDs
- Multiple exporter support
- Horizontal scaling with OTLP collectors

## Troubleshooting

### Common Issues

1. **Missing Spans**: Check OTLP endpoint connectivity
2. **High Memory Usage**: Reduce batch sizes or retention periods  
3. **False Circular Dependencies**: Verify component naming consistency
4. **Missing Visualizations**: Ensure Plotly dependencies are installed

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed tracing logs
obs_system = initialize_dag_observability(
    service_name="debug_pipeline",
    # ... other options
)
```

### Validation

```python
# Validate system health
alerts = obs_system.get_alerts()
if alerts:
    for alert in alerts:
        print(f"Alert: {alert}")

# Check metrics
heatmap = obs_system.get_heatmap_data()
print(f"Components tracked: {heatmap['total_components']}")
print(f"Edges tracked: {heatmap['total_edges']}")
```

## License

This DAG Observability System is part of the EGW Query Expansion project and follows the same licensing terms.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

For issues and feature requests, please use the project's issue tracker.