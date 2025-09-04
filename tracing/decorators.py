"""
Enhanced tracing decorators for DAG observability

Provides function and class decorators for automatic tracing integration.
"""

import functools
import inspect
from typing import Optional, Callable, Any
from .dag_observability import trace_component_execution, trace_edge_traversal, get_dag_observability


def traced_component(component_name: Optional[str] = None, 
                    phase: Optional[str] = None,
                    correlation_id_param: Optional[str] = None):
    """
    Decorator for automatic component execution tracing
    
    Args:
        component_name: Name of the component (defaults to function name)
        phase: Processing phase (defaults to 'execution')
        correlation_id_param: Parameter name containing correlation ID
    """
    def decorator(func):
        nonlocal component_name
        if component_name is None:
            component_name = func.__name__
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract correlation ID if specified
            correlation_id = None
            if correlation_id_param and correlation_id_param in kwargs:
                correlation_id = kwargs[correlation_id_param]
            
            with trace_component_execution(component_name, phase, correlation_id):
                return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Extract correlation ID if specified
            correlation_id = None
            if correlation_id_param and correlation_id_param in kwargs:
                correlation_id = kwargs[correlation_id_param]
                
            with trace_component_execution(component_name, phase, correlation_id):
                return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def traced_edge(source: str, target: Optional[str] = None,
               correlation_id_param: Optional[str] = None):
    """
    Decorator for automatic edge traversal tracing
    
    Args:
        source: Source component name
        target: Target component name (defaults to function name)
        correlation_id_param: Parameter name containing correlation ID
    """
    def decorator(func):
        nonlocal target
        if target is None:
            target = func.__name__
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract correlation ID if specified
            correlation_id = None
            if correlation_id_param and correlation_id_param in kwargs:
                correlation_id = kwargs[correlation_id_param]
            
            with trace_edge_traversal(source, target, correlation_id):
                return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Extract correlation ID if specified
            correlation_id = None
            if correlation_id_param and correlation_id_param in kwargs:
                correlation_id = kwargs[correlation_id_param]
                
            with trace_edge_traversal(source, target, correlation_id):
                return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def traced_class(component_name: Optional[str] = None):
    """
    Class decorator to automatically trace all public methods as components
    
    Args:
        component_name: Base component name (defaults to class name)
    """
    def decorator(cls):
        nonlocal component_name
        if component_name is None:
            component_name = cls.__name__
        
        # Trace all public methods
        for attr_name in dir(cls):
            if not attr_name.startswith('_'):
                attr = getattr(cls, attr_name)
                if callable(attr):
                    method_component_name = f"{component_name}.{attr_name}"
                    traced_attr = traced_component(method_component_name)(attr)
                    setattr(cls, attr_name, traced_attr)
        
        return cls
    
    return decorator


# Legacy compatibility
def trace(func):
    """Legacy trace decorator - now provides full observability"""
    return traced_component()(func)
