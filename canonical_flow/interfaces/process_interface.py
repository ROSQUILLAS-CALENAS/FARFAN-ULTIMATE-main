"""
Standardized process interface for canonical flow modules.

This module provides a unified `process(data, context) -> Dict[str, Any]` interface
with automatic adapters for existing module functions.
"""

import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable, Union, List, Tuple
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)


class StandardizedProcessor(ABC):
    """
    Abstract base class for standardized pipeline processors.
    All processors must implement the process(data, context) -> Dict[str, Any] interface.
    """
    
    @abstractmethod
    def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process input data with given context.
        
        Args:
            data: Input data to process
            context: Processing context and metadata
            
        Returns:
            Dict containing processing results with standardized keys:
            - success: bool indicating success
            - data: processed data
            - metadata: processing metadata
            - errors: any errors encountered
        """
        pass


class ProcessAdapter:
    """
    Adapter that wraps existing functions to conform to the standardized process interface.
    Handles type coercions and provides telemetry integration points.
    """
    
    def __init__(self, 
                 func: Callable,
                 component_name: str,
                 phase: str = "processing",
                 parameter_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize process adapter.
        
        Args:
            func: Function to wrap
            component_name: Name of the component for telemetry
            phase: Processing phase classification
            parameter_mapping: Custom mapping of process params to function params
        """
        self.func = func
        self.component_name = component_name
        self.phase = phase
        self.parameter_mapping = parameter_mapping or {}
        self.func_signature = inspect.signature(func)
        self.coercions_performed = []
        
    def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the wrapped function with standardized interface.
        
        Args:
            data: Input data
            context: Processing context
            
        Returns:
            Standardized result dictionary
        """
        start_time = datetime.utcnow()
        self.coercions_performed.clear()
        
        try:
            # Prepare function arguments
            func_args, func_kwargs = self._prepare_function_args(data, context)
            
            # Execute function
            result = self.func(*func_args, **func_kwargs)
            
            # Standardize result
            standardized_result = self._standardize_result(result)
            
            # Add adapter metadata
            end_time = datetime.utcnow()
            standardized_result.update({
                'adapter_metadata': {
                    'component_name': self.component_name,
                    'phase': self.phase,
                    'execution_time_ms': (end_time - start_time).total_seconds() * 1000,
                    'coercions_performed': self.coercions_performed.copy(),
                    'original_function': f"{self.func.__module__}.{self.func.__name__}"
                }
            })
            
            return standardized_result
            
        except Exception as e:
            end_time = datetime.utcnow()
            logger.error(f"Error in adapter for {self.component_name}: {e}")
            
            return {
                'success': False,
                'data': None,
                'errors': [str(e)],
                'adapter_metadata': {
                    'component_name': self.component_name,
                    'phase': self.phase,
                    'execution_time_ms': (end_time - start_time).total_seconds() * 1000,
                    'coercions_performed': self.coercions_performed.copy(),
                    'original_function': f"{self.func.__module__}.{self.func.__name__}",
                    'error': str(e)
                }
            }
    
    def _prepare_function_args(self, data: Any, context: Optional[Dict[str, Any]]) -> Tuple[List, Dict]:
        """Prepare arguments for the wrapped function."""
        func_args = []
        func_kwargs = {}
        
        # Get function parameters
        params = list(self.func_signature.parameters.items())
        
        # Handle positional parameters
        if params:
            first_param_name, first_param = params[0]
            
            # If function expects data as first parameter
            if first_param_name in ['data', 'input_data', 'content', 'document', 'text']:
                func_args.append(data)
                remaining_params = params[1:]
            else:
                remaining_params = params
                
            # Handle remaining parameters from context
            if context:
                for param_name, param in remaining_params:
                    # Check parameter mapping first
                    if param_name in self.parameter_mapping:
                        context_key = self.parameter_mapping[param_name]
                        if context_key in context:
                            func_kwargs[param_name] = context[context_key]
                            self.coercions_performed.append({
                                'type': 'parameter_mapping',
                                'from': context_key,
                                'to': param_name,
                                'value_type': type(context[context_key]).__name__
                            })
                    
                    # Direct context key matching
                    elif param_name in context:
                        func_kwargs[param_name] = context[param_name]
                    
                    # Common parameter name variations
                    elif param_name == 'context' and context:
                        func_kwargs[param_name] = context
                    elif param_name in ['config', 'options', 'params'] and 'config' in context:
                        func_kwargs[param_name] = context['config']
                        self.coercions_performed.append({
                            'type': 'parameter_alias',
                            'from': 'config',
                            'to': param_name,
                            'value_type': type(context['config']).__name__
                        })
        
        return func_args, func_kwargs
    
    def _standardize_result(self, result: Any) -> Dict[str, Any]:
        """Convert function result to standardized format."""
        if isinstance(result, dict):
            # If already a dict, ensure it has required keys
            standardized = result.copy()
            if 'success' not in standardized:
                standardized['success'] = 'error' not in standardized and 'errors' not in standardized
            if 'data' not in standardized:
                standardized['data'] = result
            if 'metadata' not in standardized:
                standardized['metadata'] = {}
            if 'errors' not in standardized:
                standardized['errors'] = []
        
        elif result is None:
            standardized = {
                'success': True,
                'data': None,
                'metadata': {},
                'errors': []
            }
        
        else:
            # Wrap non-dict results
            standardized = {
                'success': True,
                'data': result,
                'metadata': {
                    'original_type': type(result).__name__
                },
                'errors': []
            }
            
            self.coercions_performed.append({
                'type': 'result_wrapping',
                'original_type': type(result).__name__,
                'wrapped': True
            })
        
        return standardized


class ProcessAdapterFactory:
    """
    Factory for creating process adapters with automatic function inspection.
    """
    
    @staticmethod
    def create_adapter(func: Callable,
                      component_name: Optional[str] = None,
                      phase: Optional[str] = None,
                      parameter_mapping: Optional[Dict[str, str]] = None) -> ProcessAdapter:
        """
        Create a process adapter for a function.
        
        Args:
            func: Function to adapt
            component_name: Component name (auto-detected if None)
            phase: Processing phase (auto-detected if None)
            parameter_mapping: Custom parameter mappings
            
        Returns:
            ProcessAdapter instance
        """
        if component_name is None:
            component_name = f"{func.__module__}.{func.__name__}"
        
        if phase is None:
            phase = ProcessAdapterFactory._detect_phase(func)
        
        return ProcessAdapter(func, component_name, phase, parameter_mapping)
    
    @staticmethod
    def _detect_phase(func: Callable) -> str:
        """Detect processing phase from function name and module."""
        func_name = func.__name__.lower()
        module_name = func.__module__.lower() if func.__module__ else ""
        
        # Phase detection patterns
        phase_patterns = {
            'ingestion': ['load', 'read', 'ingest', 'import', 'extract'],
            'preparation': ['prepare', 'clean', 'normalize', 'validate'],
            'analysis': ['analyze', 'process', 'parse', 'understand'],
            'retrieval': ['search', 'retrieve', 'find', 'query', 'match'],
            'knowledge': ['extract', 'build', 'construct', 'graph'],
            'classification': ['classify', 'score', 'evaluate', 'rank'],
            'orchestration': ['orchestrate', 'coordinate', 'manage', 'control'],
            'synthesis': ['synthesize', 'generate', 'format', 'output'],
            'integration': ['integrate', 'store', 'save', 'persist'],
            'context': ['context', 'track', 'lineage', 'immutable'],
            'aggregation': ['aggregate', 'compile', 'report', 'audit']
        }
        
        for phase, patterns in phase_patterns.items():
            if any(pattern in func_name or pattern in module_name for pattern in patterns):
                return phase
        
        return 'processing'  # Default phase
    
    @staticmethod
    def wrap_module_functions(module: Any,
                            function_filter: Optional[Callable[[str, Callable], bool]] = None,
                            phase_override: Optional[str] = None) -> Dict[str, ProcessAdapter]:
        """
        Wrap all suitable functions in a module with process adapters.
        
        Args:
            module: Module to inspect
            function_filter: Optional filter function
            phase_override: Override phase detection
            
        Returns:
            Dict mapping function names to adapters
        """
        adapters = {}
        
        for name in dir(module):
            if name.startswith('_'):
                continue
                
            obj = getattr(module, name)
            if not callable(obj) or inspect.isclass(obj):
                continue
                
            # Apply filter if provided
            if function_filter and not function_filter(name, obj):
                continue
            
            # Create adapter
            component_name = f"{module.__name__}.{name}"
            phase = phase_override or ProcessAdapterFactory._detect_phase(obj)
            
            adapter = ProcessAdapter(obj, component_name, phase)
            adapters[name] = adapter
            
        return adapters
    
    @staticmethod
    def create_process_wrapper(func: Callable,
                             component_name: Optional[str] = None,
                             phase: Optional[str] = None) -> Callable:
        """
        Create a wrapped function that conforms to process interface.
        
        Args:
            func: Function to wrap
            component_name: Component name
            phase: Processing phase
            
        Returns:
            Wrapped function with process(data, context) signature
        """
        adapter = ProcessAdapterFactory.create_adapter(func, component_name, phase)
        
        @wraps(func)
        def process_wrapper(data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            return adapter.process(data, context)
        
        # Add adapter as attribute for telemetry access
        process_wrapper._adapter = adapter
        
        return process_wrapper


def standardize_module(module: Any, 
                      phase_override: Optional[str] = None,
                      function_filter: Optional[Callable[[str, Callable], bool]] = None) -> None:
    """
    Modify a module in-place to add standardized process interfaces.
    
    Args:
        module: Module to standardize
        phase_override: Override phase detection
        function_filter: Filter for which functions to wrap
    """
    # Create adapters for all suitable functions
    adapters = ProcessAdapterFactory.wrap_module_functions(
        module, function_filter, phase_override
    )
    
    # Add process method to module if it doesn't exist
    if not hasattr(module, 'process'):
        # Create a default process function that tries to find a suitable main function
        main_func_names = ['main', 'process_document', 'execute', 'run']
        main_adapter = None
        
        for func_name in main_func_names:
            if func_name in adapters:
                main_adapter = adapters[func_name]
                break
        
        if main_adapter:
            module.process = main_adapter.process
        else:
            # Create a generic process function
            def generic_process(data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                return {
                    'success': False,
                    'data': None,
                    'errors': ['No suitable main function found for processing'],
                    'metadata': {
                        'available_functions': list(adapters.keys()),
                        'module': module.__name__
                    }
                }
            
            module.process = generic_process
    
    # Add adapters dict to module for access
    module._process_adapters = adapters
    
    # Add convenience method to list available processors
    def list_processors() -> List[str]:
        return list(adapters.keys())
    
    module.list_processors = list_processors