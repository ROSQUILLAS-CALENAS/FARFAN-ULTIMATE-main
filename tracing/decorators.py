"""
Decorators for instrumenting process(data, context) interfaces
"""

import functools
import inspect
from typing import Any, Dict, Optional, Callable

from .otel_tracer import get_pipeline_tracer


def trace_process(source_phase: str, target_phase: str, component_name: str):
    """
    Decorator to trace process(data, context) method calls with edge traversal spans
    
    Args:
        source_phase: Source phase in canonical pipeline
        target_phase: Target phase in canonical pipeline  
        component_name: Name of the component being traced
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_pipeline_tracer()
            
            # Extract data and context from arguments
            data = None
            context = None
            
            # Handle different function signatures
            if len(args) >= 1:
                data = args[0]
            if len(args) >= 2:
                context = args[1]
                
            # Check for keyword arguments
            if 'data' in kwargs:
                data = kwargs['data']
            if 'context' in kwargs:
                context = kwargs['context']
                
            # Create edge span
            span_id = tracer.create_edge_span(
                source_phase=source_phase,
                target_phase=target_phase,
                component_name=component_name,
                data=data,
                context=context
            )
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Complete span successfully
                tracer.complete_edge_span(span_id, result=result)
                
                return result
                
            except Exception as e:
                # Complete span with error
                tracer.complete_edge_span(span_id, error=e)
                raise
                
        return wrapper
    return decorator


def trace_edge(source_phase: str, target_phase: str):
    """
    Simplified decorator that infers component name from function/class
    """
    def decorator(func: Callable):
        # Try to infer component name
        component_name = func.__name__
        if hasattr(func, '__qualname__'):
            component_name = func.__qualname__
            
        return trace_process(source_phase, target_phase, component_name)(func)
    return decorator


def auto_trace_process(func: Callable):
    """
    Auto-detect phase transitions from function location and trace automatically
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Try to detect phases from module path
        module = inspect.getmodule(func)
        if not module:
            return func(*args, **kwargs)
            
        module_path = module.__name__
        
        # Extract phases from canonical_flow module paths
        source_phase = None
        target_phase = None
        
        # Parse module path like "canonical_flow.I_ingestion_preparation.component"
        path_parts = module_path.split('.')
        for part in path_parts:
            if part.startswith(('I_', 'X_', 'K_', 'A_', 'L_', 'R_', 'O_', 'G_', 'T_', 'S_')):
                source_phase = part
                # For now, assume target is next phase (can be refined)
                break
                
        if not source_phase:
            # Fall back to untraced execution
            return func(*args, **kwargs)
            
        # Default target to same phase if not determinable
        target_phase = source_phase
        
        component_name = f"{func.__qualname__}"
        
        tracer = get_pipeline_tracer()
        data = args[0] if args else kwargs.get('data')
        context = args[1] if len(args) > 1 else kwargs.get('context')
        
        span_id = tracer.create_edge_span(
            source_phase=source_phase,
            target_phase=target_phase,
            component_name=component_name,
            data=data,
            context=context
        )
        
        try:
            result = func(*args, **kwargs)
            tracer.complete_edge_span(span_id, result=result)
            return result
        except Exception as e:
            tracer.complete_edge_span(span_id, error=e)
            raise
            
    return wrapper