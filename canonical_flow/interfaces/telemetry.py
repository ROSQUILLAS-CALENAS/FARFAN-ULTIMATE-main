"""
OpenTelemetry instrumentation for canonical flow pipeline modules.

Provides standardized telemetry collection for all pipeline components
with span creation, metrics capture, and execution tracking.
"""

import time
import logging
from typing import Any, Dict, Optional, Callable, List
from functools import wraps
from contextlib import contextmanager
from datetime import datetime, timezone

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    # Create mock classes for when OpenTelemetry is not available
    class MockSpan:
        def set_attribute(self, key, value): pass
        def set_status(self, status): pass
        def record_exception(self, exception): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    class MockTracer:
        def start_span(self, name, **kwargs): return MockSpan()
    
    # Mock trace module
    class MockTrace:
        Status = type('Status', (), {'OK': 'ok', 'ERROR': 'error'})()
        get_tracer = lambda name: MockTracer()
        set_tracer_provider = lambda provider: None

logger = logging.getLogger(__name__)


class PipelineTelemetry:
    """
    Centralized telemetry management for pipeline components.
    """
    
    def __init__(self, service_name: str = "canonical_pipeline"):
        self.service_name = service_name
        self.tracer = None
        self.metrics = {}
        self._setup_telemetry()
    
    def _setup_telemetry(self):
        """Initialize OpenTelemetry tracing."""
        if not OTEL_AVAILABLE:
            logger.warning("OpenTelemetry not available, using mock implementation")
            self.tracer = MockTrace.get_tracer(self.service_name)
            return
        
        try:
            # Create resource
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": "1.0.0"
            })
            
            # Set up tracer provider
            tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(tracer_provider)
            
            # Configure Jaeger exporter (optional)
            try:
                jaeger_exporter = JaegerExporter(
                    agent_host_name="localhost",
                    agent_port=6831,
                )
                span_processor = BatchSpanProcessor(jaeger_exporter)
                tracer_provider.add_span_processor(span_processor)
            except Exception as e:
                logger.debug(f"Jaeger exporter not configured: {e}")
            
            # Get tracer
            self.tracer = trace.get_tracer(self.service_name)
            
            # Set up logging instrumentation
            LoggingInstrumentor().instrument()
            
            logger.info(f"OpenTelemetry initialized for service: {self.service_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
            self.tracer = MockTrace.get_tracer(self.service_name)
    
    @contextmanager
    def trace_operation(self, 
                       operation_name: str,
                       component_name: str,
                       phase: str,
                       attributes: Optional[Dict[str, Any]] = None):
        """
        Context manager for tracing pipeline operations.
        
        Args:
            operation_name: Name of the operation being traced
            component_name: Name of the pipeline component
            phase: Processing phase classification
            attributes: Additional span attributes
        """
        span_name = f"{phase}.{component_name}.{operation_name}"
        
        with self.tracer.start_span(span_name) as span:
            # Set standard attributes
            span.set_attribute("component.name", component_name)
            span.set_attribute("component.phase", phase)
            span.set_attribute("operation.name", operation_name)
            span.set_attribute("service.name", self.service_name)
            
            # Set custom attributes
            if attributes:
                for key, value in attributes.items():
                    if isinstance(value, (str, int, float, bool)):
                        span.set_attribute(key, value)
                    else:
                        span.set_attribute(key, str(value))
            
            start_time = time.time()
            
            try:
                yield span
                
                # Record success
                span.set_attribute("operation.success", True)
                if OTEL_AVAILABLE:
                    span.set_status(trace.Status.OK)
                
            except Exception as e:
                # Record failure
                span.set_attribute("operation.success", False)
                span.set_attribute("error.message", str(e))
                span.set_attribute("error.type", type(e).__name__)
                
                if OTEL_AVAILABLE:
                    span.record_exception(e)
                    span.set_status(trace.Status.ERROR, str(e))
                
                logger.error(f"Error in {span_name}: {e}")
                raise
                
            finally:
                # Record execution time
                execution_time = time.time() - start_time
                span.set_attribute("execution.duration_ms", execution_time * 1000)
                
                # Update metrics
                self._update_metrics(component_name, phase, execution_time)
    
    def _update_metrics(self, component_name: str, phase: str, execution_time: float):
        """Update internal metrics tracking."""
        key = f"{phase}.{component_name}"
        
        if key not in self.metrics:
            self.metrics[key] = {
                'total_calls': 0,
                'total_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0,
                'errors': 0
            }
        
        metrics = self.metrics[key]
        metrics['total_calls'] += 1
        metrics['total_time'] += execution_time
        metrics['min_time'] = min(metrics['min_time'], execution_time)
        metrics['max_time'] = max(metrics['max_time'], execution_time)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics for all components."""
        aggregated = {}
        
        for key, metrics in self.metrics.items():
            avg_time = metrics['total_time'] / metrics['total_calls'] if metrics['total_calls'] > 0 else 0
            
            aggregated[key] = {
                'total_calls': metrics['total_calls'],
                'average_time_ms': avg_time * 1000,
                'min_time_ms': metrics['min_time'] * 1000 if metrics['min_time'] != float('inf') else 0,
                'max_time_ms': metrics['max_time'] * 1000,
                'total_time_ms': metrics['total_time'] * 1000,
                'error_count': metrics['errors']
            }
        
        return aggregated
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics.clear()


# Global telemetry instance
pipeline_telemetry = PipelineTelemetry()


def trace_component(component_name: Optional[str] = None,
                   phase: Optional[str] = None,
                   operation_name: Optional[str] = None,
                   telemetry_instance: Optional[PipelineTelemetry] = None):
    """
    Decorator for tracing component functions.
    
    Args:
        component_name: Component name (auto-detected if None)
        phase: Processing phase (auto-detected if None)
        operation_name: Operation name (uses function name if None)
        telemetry_instance: Telemetry instance to use
    """
    def decorator(func: Callable) -> Callable:
        # Auto-detect component name and phase
        nonlocal component_name, phase, operation_name
        
        if component_name is None:
            component_name = f"{func.__module__}.{func.__name__}"
        
        if phase is None:
            phase = _detect_phase_from_module(func.__module__)
        
        if operation_name is None:
            operation_name = func.__name__
            
        telemetry = telemetry_instance or pipeline_telemetry
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract attributes from function arguments
            attributes = _extract_attributes_from_args(func, args, kwargs)
            
            with telemetry.trace_operation(
                operation_name, component_name, phase, attributes
            ) as span:
                # Add coercion information if available
                if hasattr(func, '_adapter'):
                    adapter = func._adapter
                    span.set_attribute("adapter.coercions_count", len(adapter.coercions_performed))
                    
                    for i, coercion in enumerate(adapter.coercions_performed):
                        span.set_attribute(f"coercion.{i}.type", coercion.get('type', 'unknown'))
                        span.set_attribute(f"coercion.{i}.from", str(coercion.get('from', '')))
                        span.set_attribute(f"coercion.{i}.to", str(coercion.get('to', '')))
                
                result = func(*args, **kwargs)
                
                # Add result attributes
                if isinstance(result, dict):
                    span.set_attribute("result.success", result.get('success', True))
                    span.set_attribute("result.has_data", result.get('data') is not None)
                    span.set_attribute("result.error_count", len(result.get('errors', [])))
                
                return result
        
        return wrapper
    return decorator


def trace_process_method(func: Callable) -> Callable:
    """
    Decorator specifically for process(data, context) methods.
    Extracts detailed telemetry from the standardized interface.
    """
    @wraps(func)
    def wrapper(self_or_data, context_or_data=None, context=None):
        # Handle both static and instance method calls
        if hasattr(self_or_data, '__class__') and hasattr(self_or_data.__class__, func.__name__):
            # Instance method call
            instance = self_or_data
            data = context_or_data
            ctx = context
            component_name = f"{instance.__class__.__module__}.{instance.__class__.__name__}"
        else:
            # Static function call
            data = self_or_data
            ctx = context_or_data
            component_name = f"{func.__module__}.{func.__name__}"
        
        phase = _detect_phase_from_module(func.__module__)
        
        # Extract telemetry attributes
        attributes = {
            "data.type": type(data).__name__ if data is not None else "None",
            "data.size": _get_data_size(data),
            "context.provided": ctx is not None,
        }
        
        if ctx:
            attributes.update({
                "context.keys": list(ctx.keys()) if isinstance(ctx, dict) else [],
                "context.size": len(ctx) if isinstance(ctx, dict) else 0
            })
        
        with pipeline_telemetry.trace_operation(
            "process", component_name, phase, attributes
        ) as span:
            if hasattr(self_or_data, '__class__'):
                result = func(self_or_data, context_or_data, context)
            else:
                result = func(self_or_data, context_or_data)
            
            # Add result telemetry
            if isinstance(result, dict):
                span.set_attribute("result.success", result.get('success', True))
                span.set_attribute("result.has_data", result.get('data') is not None)
                span.set_attribute("result.error_count", len(result.get('errors', [])))
                
                # Add adapter metadata if available
                if 'adapter_metadata' in result:
                    metadata = result['adapter_metadata']
                    span.set_attribute("adapter.execution_time_ms", metadata.get('execution_time_ms', 0))
                    span.set_attribute("adapter.coercions_count", len(metadata.get('coercions_performed', [])))
                    span.set_attribute("adapter.original_function", metadata.get('original_function', ''))
            
            return result
    
    return wrapper


def _detect_phase_from_module(module_name: str) -> str:
    """Detect processing phase from module name."""
    if not module_name:
        return "unknown"
    
    module_lower = module_name.lower()
    
    phase_patterns = {
        'ingestion': ['ingestion', 'loader', 'reader', 'import'],
        'preparation': ['preparation', 'validator', 'normalizer'],
        'analysis': ['analysis', 'nlp', 'processor'],
        'retrieval': ['retrieval', 'search', 'index', 'query'],
        'knowledge': ['knowledge', 'extraction', 'graph'],
        'classification': ['classification', 'scoring', 'evaluation'],
        'orchestration': ['orchestration', 'control', 'workflow'],
        'synthesis': ['synthesis', 'output', 'formatter'],
        'integration': ['integration', 'storage', 'persistence'],
        'context': ['context', 'lineage', 'tracking'],
        'aggregation': ['aggregation', 'reporting', 'audit']
    }
    
    for phase, patterns in phase_patterns.items():
        if any(pattern in module_lower for pattern in patterns):
            return phase
    
    return "processing"


def _extract_attributes_from_args(func: Callable, args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Extract telemetry attributes from function arguments."""
    attributes = {}
    
    try:
        import inspect
        sig = inspect.signature(func)
        bound_args = sig.bind_partial(*args, **kwargs)
        
        for param_name, value in bound_args.arguments.items():
            if param_name in ['data', 'input_data', 'content']:
                attributes[f"input.{param_name}.type"] = type(value).__name__
                attributes[f"input.{param_name}.size"] = _get_data_size(value)
            elif param_name == 'context' and isinstance(value, dict):
                attributes["input.context.keys"] = list(value.keys())
                attributes["input.context.size"] = len(value)
    except Exception:
        pass  # Ignore signature extraction errors
    
    return attributes


def _get_data_size(data: Any) -> int:
    """Get size metric for data object."""
    if data is None:
        return 0
    elif isinstance(data, (str, bytes)):
        return len(data)
    elif isinstance(data, (list, tuple, dict)):
        return len(data)
    elif hasattr(data, '__len__'):
        try:
            return len(data)
        except:
            return 1
    else:
        return 1


class ComponentTelemetryMixin:
    """
    Mixin class to add telemetry capabilities to pipeline components.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._telemetry = pipeline_telemetry
        self._component_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        self._phase = _detect_phase_from_module(self.__class__.__module__)
    
    def trace_operation(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create a trace context for an operation."""
        return self._telemetry.trace_operation(
            operation_name, self._component_name, self._phase, attributes
        )
    
    def get_component_metrics(self) -> Dict[str, Any]:
        """Get metrics for this component."""
        all_metrics = self._telemetry.get_metrics()
        component_key = f"{self._phase}.{self._component_name}"
        return all_metrics.get(component_key, {})


def instrument_module_functions(module: Any, 
                               function_filter: Optional[Callable[[str, Callable], bool]] = None):
    """
    Instrument all suitable functions in a module with telemetry.
    
    Args:
        module: Module to instrument
        function_filter: Optional filter for which functions to instrument
    """
    import inspect
    
    instrumented_count = 0
    
    for name in dir(module):
        if name.startswith('_'):
            continue
            
        obj = getattr(module, name)
        if not callable(obj) or inspect.isclass(obj):
            continue
            
        # Apply filter if provided
        if function_filter and not function_filter(name, obj):
            continue
        
        # Skip if already instrumented
        if hasattr(obj, '__wrapped__'):
            continue
        
        # Instrument the function
        component_name = f"{module.__name__}.{name}"
        phase = _detect_phase_from_module(module.__name__)
        
        instrumented_func = trace_component(
            component_name=component_name,
            phase=phase,
            operation_name=name
        )(obj)
        
        # Replace function in module
        setattr(module, name, instrumented_func)
        instrumented_count += 1
    
    logger.info(f"Instrumented {instrumented_count} functions in module {module.__name__}")


# Convenience functions
def get_telemetry_instance() -> PipelineTelemetry:
    """Get the global telemetry instance."""
    return pipeline_telemetry


def get_pipeline_metrics() -> Dict[str, Any]:
    """Get metrics for all pipeline components."""
    return pipeline_telemetry.get_metrics()


def reset_pipeline_metrics():
    """Reset all pipeline metrics."""
    pipeline_telemetry.reset_metrics()