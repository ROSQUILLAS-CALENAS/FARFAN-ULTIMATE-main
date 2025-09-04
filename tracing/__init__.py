"""
Comprehensive DAG Observability System

This module provides OpenTelemetry-based tracing instrumentation for DAG execution
with live dependency heatmaps and circular dependency detection.
"""

# Import from existing tracing system
try:
    from .otel_tracer import (
        PipelineTracer,
        SpanMetadata,
        PhaseTransition,
        CANONICAL_PHASES
    )
    from .back_edge_detector import BackEdgeDetector, DependencyViolation
    from .dashboard import TracingDashboard
    LEGACY_TRACING_AVAILABLE = True
except ImportError:
    LEGACY_TRACING_AVAILABLE = False

# Import new DAG observability system
from .dag_observability import (
    initialize_dag_observability,
    get_dag_observability,
    trace_edge_traversal,
    trace_component_execution,
    trace_phase_transition,
    DAGObservabilitySystem,
    EdgeTraversalMetrics,
    ComponentMetrics,
    CircularDependencyViolation
)

from .visualization import (
    DependencyHeatmapVisualizer,
    CircularDependencyVisualizer,
    create_performance_dashboard
)

__all__ = [
    "initialize_dag_observability",
    "get_dag_observability", 
    "trace_edge_traversal",
    "trace_component_execution",
    "trace_phase_transition",
    "DAGObservabilitySystem",
    "EdgeTraversalMetrics",
    "ComponentMetrics", 
    "CircularDependencyViolation",
    "DependencyHeatmapVisualizer",
    "CircularDependencyVisualizer",
    "create_performance_dashboard"
]

# Add legacy exports if available
if LEGACY_TRACING_AVAILABLE:
    __all__.extend([
        'PipelineTracer',
        'SpanMetadata', 
        'PhaseTransition',
        'CANONICAL_PHASES',
        'BackEdgeDetector',
        'DependencyViolation',
        'TracingDashboard'
    ])