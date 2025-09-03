"""
OpenTelemetry tracing system for canonical pipeline DAG
"""

from .otel_tracer import (
    PipelineTracer,
    SpanMetadata,
    PhaseTransition,
    CANONICAL_PHASES
)
from .decorators import trace_process, trace_edge
from .visualization import DependencyHeatmapVisualizer
from .back_edge_detector import BackEdgeDetector, DependencyViolation
from .dashboard import TracingDashboard

__all__ = [
    'PipelineTracer',
    'SpanMetadata', 
    'PhaseTransition',
    'CANONICAL_PHASES',
    'trace_process',
    'trace_edge', 
    'DependencyHeatmapVisualizer',
    'BackEdgeDetector',
    'DependencyViolation',
    'TracingDashboard'
]