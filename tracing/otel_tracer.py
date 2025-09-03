"""
OpenTelemetry tracer for canonical pipeline DAG edge traversals
"""

import time
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from enum import Enum

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.trace import Status, StatusCode

# Canonical phase ordering: I→X→K→A→L→R→O→G→T→S
CANONICAL_PHASES = [
    'I_ingestion_preparation',
    'X_context_construction', 
    'K_knowledge_extraction',
    'A_analysis_nlp',
    'L_classification_evaluation',
    'R_search_retrieval',
    'O_orchestration_control',
    'G_aggregation_reporting', 
    'T_integration_storage',
    'S_synthesis_output'
]

PHASE_ORDER = {phase: idx for idx, phase in enumerate(CANONICAL_PHASES)}


@dataclass
class SpanMetadata:
    """Metadata for pipeline component spans"""
    source_phase: str
    target_phase: str
    component_name: str
    edge_type: str  # 'forward', 'backward', 'cross'
    timing_start: float
    timing_end: Optional[float] = None
    data_size: Optional[int] = None
    error: Optional[str] = None


@dataclass  
class PhaseTransition:
    """Represents a phase transition in the pipeline"""
    from_phase: str
    to_phase: str
    component: str
    timestamp: float
    duration: float
    success: bool
    metadata: Dict[str, Any]


class PipelineTracer:
    """OpenTelemetry tracer for canonical pipeline DAG"""
    
    def __init__(self, service_name: str = "canonical-pipeline", 
                 jaeger_endpoint: Optional[str] = None,
                 otlp_endpoint: Optional[str] = None):
        """Initialize the pipeline tracer"""
        
        # Configure resource
        resource = Resource.create({
            "service.name": service_name,
            "service.version": "1.0.0",
            "deployment.environment": "production"
        })
        
        # Set up tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)
        
        # Configure exporters
        if jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=14268,
            )
            tracer_provider.add_span_processor(
                BatchSpanProcessor(jaeger_exporter)
            )
            
        if otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                insecure=True
            )
            tracer_provider.add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )
        
        # Get tracer
        self.tracer = trace.get_tracer(__name__)
        
        # Initialize logging instrumentation
        LoggingInstrumentor().instrument()
        
        # Tracking collections
        self.active_spans: Dict[str, Any] = {}
        self.span_history: List[SpanMetadata] = []
        self.phase_transitions: List[PhaseTransition] = []
        
    def create_edge_span(self, source_phase: str, target_phase: str, 
                        component_name: str, data: Any = None, 
                        context: Optional[Dict] = None) -> str:
        """Create a span for edge traversal in pipeline DAG"""
        
        # Determine edge type
        edge_type = self._classify_edge(source_phase, target_phase)
        
        # Generate span ID
        span_id = f"{source_phase}->{target_phase}:{component_name}:{int(time.time() * 1000000)}"
        
        # Create span attributes
        attributes = {
            "pipeline.source_phase": source_phase,
            "pipeline.target_phase": target_phase, 
            "pipeline.component_name": component_name,
            "pipeline.edge_type": edge_type,
            "pipeline.data_size": len(str(data)) if data else 0,
        }
        
        if context:
            for key, value in context.items():
                if isinstance(value, (str, int, float, bool)):
                    attributes[f"context.{key}"] = value
                    
        # Start the span
        span = self.tracer.start_span(
            name=f"{source_phase}->{target_phase}",
            attributes=attributes
        )
        
        # Store span metadata
        metadata = SpanMetadata(
            source_phase=source_phase,
            target_phase=target_phase,
            component_name=component_name,
            edge_type=edge_type,
            timing_start=time.time(),
            data_size=len(str(data)) if data else 0
        )
        
        self.active_spans[span_id] = {
            'span': span,
            'metadata': metadata
        }
        
        return span_id
        
    def complete_edge_span(self, span_id: str, result: Any = None, 
                          error: Optional[Exception] = None):
        """Complete an edge traversal span"""
        
        if span_id not in self.active_spans:
            return
            
        span_info = self.active_spans[span_id]
        span = span_info['span']
        metadata = span_info['metadata']
        
        # Update timing
        metadata.timing_end = time.time()
        duration = metadata.timing_end - metadata.timing_start
        
        # Set span attributes
        span.set_attribute("pipeline.duration_ms", duration * 1000)
        if result:
            span.set_attribute("pipeline.result_size", len(str(result)))
            
        # Handle errors
        if error:
            metadata.error = str(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)
        else:
            span.set_status(Status(StatusCode.OK))
            
        # End span
        span.end()
        
        # Record phase transition
        transition = PhaseTransition(
            from_phase=metadata.source_phase,
            to_phase=metadata.target_phase,
            component=metadata.component_name,
            timestamp=metadata.timing_start,
            duration=duration,
            success=error is None,
            metadata={
                'edge_type': metadata.edge_type,
                'data_size': metadata.data_size,
                'error': metadata.error
            }
        )
        self.phase_transitions.append(transition)
        
        # Move to history
        self.span_history.append(metadata)
        del self.active_spans[span_id]
        
    def _classify_edge(self, source_phase: str, target_phase: str) -> str:
        """Classify edge type based on canonical phase ordering"""
        
        source_order = PHASE_ORDER.get(source_phase, -1)
        target_order = PHASE_ORDER.get(target_phase, -1)
        
        if source_order == -1 or target_order == -1:
            return 'unknown'
        elif target_order > source_order:
            return 'forward'
        elif target_order < source_order:
            return 'backward'  # Potential violation
        else:
            return 'self'
            
    def get_span_data(self, time_window_minutes: int = 60) -> List[SpanMetadata]:
        """Get span data within time window"""
        cutoff_time = time.time() - (time_window_minutes * 60)
        return [
            span for span in self.span_history 
            if span.timing_start >= cutoff_time
        ]
        
    def get_phase_transitions(self, time_window_minutes: int = 60) -> List[PhaseTransition]:
        """Get phase transitions within time window"""
        cutoff_time = time.time() - (time_window_minutes * 60)
        return [
            transition for transition in self.phase_transitions
            if transition.timestamp >= cutoff_time
        ]
        
    def get_edge_frequencies(self, time_window_minutes: int = 60) -> Dict[str, int]:
        """Get edge traversal frequencies"""
        spans = self.get_span_data(time_window_minutes)
        frequencies = {}
        
        for span in spans:
            edge_key = f"{span.source_phase}->{span.target_phase}"
            frequencies[edge_key] = frequencies.get(edge_key, 0) + 1
            
        return frequencies
        
    def get_latency_distributions(self, time_window_minutes: int = 60) -> Dict[str, List[float]]:
        """Get latency distributions by edge"""
        spans = self.get_span_data(time_window_minutes)
        distributions = {}
        
        for span in spans:
            if span.timing_end:
                edge_key = f"{span.source_phase}->{span.target_phase}"
                if edge_key not in distributions:
                    distributions[edge_key] = []
                duration = span.timing_end - span.timing_start
                distributions[edge_key].append(duration)
                
        return distributions


# Global tracer instance
_pipeline_tracer: Optional[PipelineTracer] = None


def get_pipeline_tracer() -> PipelineTracer:
    """Get or create global pipeline tracer"""
    global _pipeline_tracer
    if _pipeline_tracer is None:
        _pipeline_tracer = PipelineTracer()
    return _pipeline_tracer


def set_pipeline_tracer(tracer: PipelineTracer):
    """Set global pipeline tracer"""
    global _pipeline_tracer
    _pipeline_tracer = tracer