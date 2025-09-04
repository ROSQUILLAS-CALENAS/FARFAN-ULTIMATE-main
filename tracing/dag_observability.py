"""
DAG Observability System with OpenTelemetry Tracing

This module implements comprehensive observability for DAG execution including:
- OpenTelemetry tracing instrumentation for pipeline edges
- Span export configuration with OTLP and JSON fallback
- Live dependency heatmap generation
- Residual back-edge detection for circular dependency monitoring
"""

import json
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import logging
import queue

from opentelemetry import trace, context
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPOTLPSpanExporter

logger = logging.getLogger(__name__)


@dataclass
class EdgeTraversalMetrics:
    """Metrics for individual edge traversals in the DAG"""
    source_component: str
    target_component: str
    traversal_count: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    last_traversal: Optional[datetime] = None
    error_count: int = 0
    
    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(1, self.traversal_count)
    
    def add_traversal(self, latency_ms: float, error: bool = False):
        """Record a new traversal with timing information"""
        self.traversal_count += 1
        self.total_latency_ms += latency_ms
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        self.last_traversal = datetime.now()
        if error:
            self.error_count += 1


@dataclass
class ComponentMetrics:
    """Metrics for individual components in the DAG"""
    component_name: str
    entry_count: int = 0
    exit_count: int = 0
    total_execution_time_ms: float = 0.0
    error_count: int = 0
    active_executions: int = 0
    phase_transitions: Dict[str, int] = field(default_factory=dict)
    
    @property
    def avg_execution_time_ms(self) -> float:
        return self.total_execution_time_ms / max(1, self.exit_count)


@dataclass
class CircularDependencyViolation:
    """Details of a circular dependency violation detected at runtime"""
    violation_id: str
    timestamp: datetime
    initiating_component: str
    target_component: str
    execution_stack: List[str]
    span_context: Dict[str, Any]
    severity: str = "HIGH"
    resolved: bool = False


class JSONSpanExporter:
    """Custom span exporter that writes spans to JSON files"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
    
    def export(self, spans):
        """Export spans to JSON file"""
        with self._lock:
            try:
                spans_data = []
                for span in spans:
                    span_dict = {
                        "trace_id": hex(span.context.trace_id),
                        "span_id": hex(span.context.span_id),
                        "name": span.name,
                        "start_time": span.start_time,
                        "end_time": span.end_time,
                        "duration_ns": span.end_time - span.start_time if span.end_time else None,
                        "status": {
                            "status_code": span.status.status_code.name if span.status else None,
                            "description": span.status.description if span.status else None
                        },
                        "attributes": dict(span.attributes) if span.attributes else {},
                        "parent_span_id": hex(span.parent.span_id) if span.parent else None,
                        "resource": dict(span.resource.attributes) if span.resource else {}
                    }
                    spans_data.append(span_dict)
                
                # Append to file
                if self.file_path.exists():
                    with open(self.file_path, 'r') as f:
                        try:
                            existing_data = json.load(f)
                        except json.JSONDecodeError:
                            existing_data = []
                else:
                    existing_data = []
                
                existing_data.extend(spans_data)
                
                with open(self.file_path, 'w') as f:
                    json.dump(existing_data, f, indent=2, default=str)
                
                return True
            except Exception as e:
                logger.error(f"Failed to export spans to JSON: {e}")
                return False
    
    def shutdown(self):
        """Shutdown the exporter"""
        pass


class LiveHeatmapGenerator:
    """Generates live dependency heatmaps from span data"""
    
    def __init__(self, update_interval_seconds: int = 5):
        self.update_interval = update_interval_seconds
        self.edge_metrics: Dict[Tuple[str, str], EdgeTraversalMetrics] = {}
        self.component_metrics: Dict[str, ComponentMetrics] = {}
        self.running = False
        self.heatmap_thread = None
        self._lock = threading.Lock()
        
    def update_edge_metrics(self, source: str, target: str, latency_ms: float, error: bool = False):
        """Update metrics for an edge traversal"""
        with self._lock:
            key = (source, target)
            if key not in self.edge_metrics:
                self.edge_metrics[key] = EdgeTraversalMetrics(source, target)
            self.edge_metrics[key].add_traversal(latency_ms, error)
    
    def update_component_metrics(self, component: str, event_type: str, execution_time_ms: float = 0, 
                               phase: str = None, error: bool = False):
        """Update metrics for a component"""
        with self._lock:
            if component not in self.component_metrics:
                self.component_metrics[component] = ComponentMetrics(component)
            
            metrics = self.component_metrics[component]
            
            if event_type == "entry":
                metrics.entry_count += 1
                metrics.active_executions += 1
            elif event_type == "exit":
                metrics.exit_count += 1
                metrics.active_executions = max(0, metrics.active_executions - 1)
                metrics.total_execution_time_ms += execution_time_ms
            
            if phase:
                metrics.phase_transitions[phase] = metrics.phase_transitions.get(phase, 0) + 1
            
            if error:
                metrics.error_count += 1
    
    def start_live_updates(self):
        """Start live heatmap generation thread"""
        if self.running:
            return
        
        self.running = True
        self.heatmap_thread = threading.Thread(target=self._heatmap_update_loop)
        self.heatmap_thread.daemon = True
        self.heatmap_thread.start()
        logger.info("Live heatmap generation started")
    
    def stop_live_updates(self):
        """Stop live heatmap generation"""
        self.running = False
        if self.heatmap_thread:
            self.heatmap_thread.join(timeout=5)
        logger.info("Live heatmap generation stopped")
    
    def _heatmap_update_loop(self):
        """Background thread for updating heatmaps"""
        while self.running:
            try:
                self.generate_heatmap_data()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in heatmap update loop: {e}")
                time.sleep(self.update_interval)
    
    def generate_heatmap_data(self) -> Dict[str, Any]:
        """Generate current heatmap data"""
        with self._lock:
            # Calculate hot paths (most frequently traversed edges)
            hot_paths = sorted(
                self.edge_metrics.values(),
                key=lambda x: x.traversal_count,
                reverse=True
            )[:10]
            
            # Calculate high-latency paths
            high_latency_paths = sorted(
                self.edge_metrics.values(),
                key=lambda x: x.avg_latency_ms,
                reverse=True
            )[:10]
            
            # Calculate component load
            component_load = {
                name: {
                    "active_executions": metrics.active_executions,
                    "avg_execution_time": metrics.avg_execution_time_ms,
                    "error_rate": metrics.error_count / max(1, metrics.entry_count),
                    "throughput": metrics.entry_count
                }
                for name, metrics in self.component_metrics.items()
            }
            
            heatmap_data = {
                "timestamp": datetime.now().isoformat(),
                "hot_paths": [
                    {
                        "edge": f"{path.source_component} -> {path.target_component}",
                        "traversal_count": path.traversal_count,
                        "avg_latency_ms": path.avg_latency_ms,
                        "error_rate": path.error_count / max(1, path.traversal_count)
                    }
                    for path in hot_paths
                ],
                "high_latency_paths": [
                    {
                        "edge": f"{path.source_component} -> {path.target_component}",
                        "avg_latency_ms": path.avg_latency_ms,
                        "traversal_count": path.traversal_count
                    }
                    for path in high_latency_paths
                ],
                "component_load": component_load,
                "total_edges": len(self.edge_metrics),
                "total_components": len(self.component_metrics)
            }
            
            # Save to file for visualization
            heatmap_file = Path("tracing/live_heatmap.json")
            heatmap_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(heatmap_file, 'w') as f:
                    json.dump(heatmap_data, f, indent=2, default=str)
            except Exception as e:
                logger.error(f"Failed to save heatmap data: {e}")
            
            return heatmap_data


class CircularDependencyDetector:
    """Detects circular dependencies by tracking active span contexts"""
    
    def __init__(self, max_stack_depth: int = 100):
        self.max_stack_depth = max_stack_depth
        self.active_stacks: Dict[int, List[str]] = {}  # thread_id -> execution_stack
        self.violations: List[CircularDependencyViolation] = []
        self.component_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.Lock()
        self.alert_queue = queue.Queue()
        
    def enter_component(self, component_name: str, span_context: Optional[Any] = None) -> bool:
        """
        Record component entry and check for circular dependencies
        Returns False if circular dependency detected
        """
        thread_id = threading.get_ident()
        
        with self._lock:
            if thread_id not in self.active_stacks:
                self.active_stacks[thread_id] = []
            
            execution_stack = self.active_stacks[thread_id]
            
            # Check if component is already in the execution stack (circular dependency)
            if component_name in execution_stack:
                violation = self._create_violation(
                    component_name, execution_stack, span_context
                )
                self.violations.append(violation)
                self._queue_alert(violation)
                return False
            
            # Check stack depth
            if len(execution_stack) >= self.max_stack_depth:
                logger.warning(f"Maximum stack depth {self.max_stack_depth} reached")
                return False
            
            # Add component to stack
            execution_stack.append(component_name)
            
            # Record dependency relationship
            if len(execution_stack) > 1:
                parent_component = execution_stack[-2]
                self.component_dependencies[parent_component].add(component_name)
            
            return True
    
    def exit_component(self, component_name: str):
        """Record component exit"""
        thread_id = threading.get_ident()
        
        with self._lock:
            if thread_id in self.active_stacks:
                execution_stack = self.active_stacks[thread_id]
                if execution_stack and execution_stack[-1] == component_name:
                    execution_stack.pop()
                    
                    # Clean up empty stacks
                    if not execution_stack:
                        del self.active_stacks[thread_id]
    
    def _create_violation(self, component_name: str, execution_stack: List[str], 
                         span_context: Optional[Any]) -> CircularDependencyViolation:
        """Create a circular dependency violation record"""
        violation_id = f"circ_dep_{int(time.time() * 1000000)}"
        
        # Find the cycle in the stack
        cycle_start = execution_stack.index(component_name)
        cycle = execution_stack[cycle_start:] + [component_name]
        
        span_data = {}
        if span_context:
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                span_data = {
                    "trace_id": hex(current_span.context.trace_id),
                    "span_id": hex(current_span.context.span_id),
                    "span_name": current_span.name
                }
        
        return CircularDependencyViolation(
            violation_id=violation_id,
            timestamp=datetime.now(),
            initiating_component=component_name,
            target_component=execution_stack[-1] if execution_stack else "unknown",
            execution_stack=cycle,
            span_context=span_data,
            severity="HIGH" if len(cycle) > 3 else "MEDIUM"
        )
    
    def _queue_alert(self, violation: CircularDependencyViolation):
        """Queue an alert for a circular dependency violation"""
        try:
            alert_data = {
                "type": "circular_dependency",
                "violation_id": violation.violation_id,
                "timestamp": violation.timestamp.isoformat(),
                "severity": violation.severity,
                "message": f"Circular dependency detected: {' -> '.join(violation.execution_stack)}",
                "details": {
                    "initiating_component": violation.initiating_component,
                    "target_component": violation.target_component,
                    "cycle_length": len(violation.execution_stack),
                    "span_context": violation.span_context
                }
            }
            self.alert_queue.put(alert_data, timeout=1.0)
        except queue.Full:
            logger.warning("Alert queue is full, dropping circular dependency alert")
    
    def get_current_violations(self) -> List[CircularDependencyViolation]:
        """Get all current unresolved violations"""
        with self._lock:
            return [v for v in self.violations if not v.resolved]
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get the current dependency graph"""
        with self._lock:
            return {
                component: list(deps) 
                for component, deps in self.component_dependencies.items()
            }
    
    def resolve_violation(self, violation_id: str):
        """Mark a violation as resolved"""
        with self._lock:
            for violation in self.violations:
                if violation.violation_id == violation_id:
                    violation.resolved = True
                    break
    
    def get_alerts(self, timeout: float = 0.1) -> List[Dict[str, Any]]:
        """Get queued alerts"""
        alerts = []
        try:
            while True:
                alert = self.alert_queue.get(timeout=timeout)
                alerts.append(alert)
        except queue.Empty:
            pass
        return alerts


class DAGObservabilitySystem:
    """Main DAG observability system coordinating all components"""
    
    def __init__(self, 
                 service_name: str = "dag_pipeline",
                 otlp_endpoint: Optional[str] = None,
                 json_fallback_path: str = "tracing/spans.json",
                 enable_heatmap: bool = True,
                 enable_circular_detection: bool = True):
        
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self.json_fallback_path = json_fallback_path
        self.tracer_provider = None
        self.tracer = None
        
        # Initialize components
        self.heatmap_generator = LiveHeatmapGenerator() if enable_heatmap else None
        self.circular_detector = CircularDependencyDetector() if enable_circular_detection else None
        
        # Correlation ID tracking
        self.correlation_ids: Dict[str, str] = {}
        self._correlation_lock = threading.Lock()
        
        self._setup_tracing()
        
    def _setup_tracing(self):
        """Setup OpenTelemetry tracing with configured exporters"""
        # Create resource
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": "1.0.0",
            "deployment.environment": "production"
        })
        
        # Create tracer provider
        self.tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(self.tracer_provider)
        
        # Setup exporters
        exporters = []
        
        # Try OTLP exporter first
        if self.otlp_endpoint:
            try:
                if self.otlp_endpoint.startswith('http'):
                    otlp_exporter = HTTPOTLPSpanExporter(endpoint=f"{self.otlp_endpoint}/v1/traces")
                else:
                    otlp_exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint)
                exporters.append(otlp_exporter)
                logger.info(f"OTLP exporter configured for {self.otlp_endpoint}")
            except Exception as e:
                logger.warning(f"Failed to setup OTLP exporter: {e}")
        
        # JSON fallback exporter
        if self.json_fallback_path:
            json_exporter = JSONSpanExporter(self.json_fallback_path)
            exporters.append(json_exporter)
            logger.info(f"JSON exporter configured for {self.json_fallback_path}")
        
        # Console exporter for development
        exporters.append(ConsoleSpanExporter())
        
        # Add span processors
        for exporter in exporters:
            span_processor = BatchSpanProcessor(exporter)
            self.tracer_provider.add_span_processor(span_processor)
        
        # Get tracer
        self.tracer = trace.get_tracer(__name__)
        
    def start(self):
        """Start the observability system"""
        if self.heatmap_generator:
            self.heatmap_generator.start_live_updates()
        logger.info("DAG Observability System started")
        
    def stop(self):
        """Stop the observability system"""
        if self.heatmap_generator:
            self.heatmap_generator.stop_live_updates()
        
        # Shutdown tracer provider
        if self.tracer_provider:
            self.tracer_provider.shutdown()
        
        logger.info("DAG Observability System stopped")
    
    def set_correlation_id(self, component: str, correlation_id: str):
        """Set correlation ID for a component"""
        with self._correlation_lock:
            self.correlation_ids[component] = correlation_id
    
    def get_correlation_id(self, component: str) -> Optional[str]:
        """Get correlation ID for a component"""
        with self._correlation_lock:
            return self.correlation_ids.get(component)
    
    @contextmanager
    def trace_edge_traversal(self, source_component: str, target_component: str, 
                           correlation_id: Optional[str] = None):
        """Context manager for tracing edge traversal between components"""
        start_time = time.time()
        error_occurred = False
        
        # Set correlation ID if provided
        if correlation_id:
            self.set_correlation_id(target_component, correlation_id)
        
        # Get or use existing correlation ID
        corr_id = correlation_id or self.get_correlation_id(source_component)
        
        with self.tracer.start_as_current_span(
            f"edge_traversal:{source_component}->{target_component}",
            attributes={
                "dag.edge.source": source_component,
                "dag.edge.target": target_component,
                "dag.correlation_id": corr_id or "unknown"
            }
        ) as span:
            try:
                yield span
            except Exception as e:
                error_occurred = True
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            finally:
                # Record metrics
                latency_ms = (time.time() - start_time) * 1000
                
                if self.heatmap_generator:
                    self.heatmap_generator.update_edge_metrics(
                        source_component, target_component, latency_ms, error_occurred
                    )
                
                # Add timing attributes
                span.set_attribute("dag.edge.latency_ms", latency_ms)
                span.set_attribute("dag.edge.error", error_occurred)
    
    @contextmanager
    def trace_component_execution(self, component_name: str, phase: Optional[str] = None,
                                correlation_id: Optional[str] = None):
        """Context manager for tracing component execution"""
        start_time = time.time()
        error_occurred = False
        
        # Check for circular dependencies
        if self.circular_detector:
            current_span = trace.get_current_span()
            if not self.circular_detector.enter_component(component_name, current_span):
                raise RuntimeError(f"Circular dependency detected for component: {component_name}")
        
        # Set correlation ID if provided
        if correlation_id:
            self.set_correlation_id(component_name, correlation_id)
        
        # Get correlation ID
        corr_id = correlation_id or self.get_correlation_id(component_name)
        
        span_name = f"component:{component_name}"
        if phase:
            span_name += f":{phase}"
            
        with self.tracer.start_as_current_span(
            span_name,
            attributes={
                "dag.component.name": component_name,
                "dag.component.phase": phase or "execution",
                "dag.correlation_id": corr_id or "unknown"
            }
        ) as span:
            try:
                # Record component entry
                if self.heatmap_generator:
                    self.heatmap_generator.update_component_metrics(
                        component_name, "entry", phase=phase
                    )
                
                yield span
                
            except Exception as e:
                error_occurred = True
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            finally:
                # Calculate execution time
                execution_time_ms = (time.time() - start_time) * 1000
                
                # Record component exit
                if self.heatmap_generator:
                    self.heatmap_generator.update_component_metrics(
                        component_name, "exit", execution_time_ms, phase, error_occurred
                    )
                
                # Exit component in circular dependency detector
                if self.circular_detector:
                    self.circular_detector.exit_component(component_name)
                
                # Add timing attributes
                span.set_attribute("dag.component.execution_time_ms", execution_time_ms)
                span.set_attribute("dag.component.error", error_occurred)
    
    @contextmanager
    def trace_phase_transition(self, component_name: str, from_phase: str, to_phase: str):
        """Context manager for tracing phase transitions within components"""
        with self.tracer.start_as_current_span(
            f"phase_transition:{component_name}:{from_phase}->{to_phase}",
            attributes={
                "dag.component.name": component_name,
                "dag.phase.from": from_phase,
                "dag.phase.to": to_phase,
                "dag.correlation_id": self.get_correlation_id(component_name) or "unknown"
            }
        ) as span:
            yield span
    
    def get_heatmap_data(self) -> Optional[Dict[str, Any]]:
        """Get current heatmap data"""
        if self.heatmap_generator:
            return self.heatmap_generator.generate_heatmap_data()
        return None
    
    def get_circular_dependency_violations(self) -> List[CircularDependencyViolation]:
        """Get current circular dependency violations"""
        if self.circular_detector:
            return self.circular_detector.get_current_violations()
        return []
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get current dependency graph"""
        if self.circular_detector:
            return self.circular_detector.get_dependency_graph()
        return {}
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get queued alerts"""
        if self.circular_detector:
            return self.circular_detector.get_alerts()
        return []
    
    def resolve_violation(self, violation_id: str):
        """Resolve a circular dependency violation"""
        if self.circular_detector:
            self.circular_detector.resolve_violation(violation_id)


# Global observability system instance
_observability_system: Optional[DAGObservabilitySystem] = None


def initialize_dag_observability(
    service_name: str = "dag_pipeline",
    otlp_endpoint: Optional[str] = None,
    json_fallback_path: str = "tracing/spans.json",
    enable_heatmap: bool = True,
    enable_circular_detection: bool = True
) -> DAGObservabilitySystem:
    """Initialize global DAG observability system"""
    global _observability_system
    
    if _observability_system:
        _observability_system.stop()
    
    _observability_system = DAGObservabilitySystem(
        service_name=service_name,
        otlp_endpoint=otlp_endpoint,
        json_fallback_path=json_fallback_path,
        enable_heatmap=enable_heatmap,
        enable_circular_detection=enable_circular_detection
    )
    
    _observability_system.start()
    return _observability_system


def get_dag_observability() -> Optional[DAGObservabilitySystem]:
    """Get global DAG observability system instance"""
    return _observability_system


def trace_edge_traversal(source: str, target: str, correlation_id: Optional[str] = None):
    """Decorator/context manager for tracing edge traversal"""
    if _observability_system:
        return _observability_system.trace_edge_traversal(source, target, correlation_id)
    else:
        @contextmanager
        def dummy_context():
            yield None
        return dummy_context()


def trace_component_execution(component_name: str, phase: Optional[str] = None, 
                            correlation_id: Optional[str] = None):
    """Decorator/context manager for tracing component execution"""
    if _observability_system:
        return _observability_system.trace_component_execution(component_name, phase, correlation_id)
    else:
        @contextmanager
        def dummy_context():
            yield None
        return dummy_context()


def trace_phase_transition(component_name: str, from_phase: str, to_phase: str):
    """Decorator/context manager for tracing phase transitions"""
    if _observability_system:
        return _observability_system.trace_phase_transition(component_name, from_phase, to_phase)
    else:
        @contextmanager
        def dummy_context():
            yield None
        return dummy_context()


# Shutdown handler
import atexit

def _shutdown_observability():
    """Shutdown observability system on exit"""
    global _observability_system
    if _observability_system:
        _observability_system.stop()

atexit.register(_shutdown_observability)