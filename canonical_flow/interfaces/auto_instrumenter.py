"""
Automatic instrumentation system for canonical flow modules.

Automatically discovers and instruments pipeline modules with standardized
process interfaces and OpenTelemetry telemetry.
"""

import sys
import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from .process_interface import ProcessAdapterFactory, standardize_module
from .telemetry import instrument_module_functions, pipeline_telemetry

logger = logging.getLogger(__name__)


class AutoInstrumenter:
    """
    Automatic instrumentation system for canonical flow modules.
    Discovers modules, creates process adapters, and adds telemetry.
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path(__file__).parent.parent
        self.instrumented_modules: Set[str] = set()
        self.adapters_registry: Dict[str, Dict[str, Any]] = {}
        
    def discover_modules(self, pattern: str = "**/*.py") -> List[Path]:
        """
        Discover Python modules in the canonical flow structure.
        
        Args:
            pattern: Glob pattern for module discovery
            
        Returns:
            List of module paths
        """
        modules = []
        
        for py_file in self.base_path.glob(pattern):
            # Skip __init__.py, test files, and interface files
            if (py_file.name.startswith('__') or 
                py_file.name.startswith('test_') or
                'interfaces' in py_file.parts):
                continue
                
            modules.append(py_file)
        
        return modules
    
    def instrument_module(self, module_path: Path, 
                         force_reinstrument: bool = False) -> Optional[Dict[str, Any]]:
        """
        Instrument a single module with process interface and telemetry.
        
        Args:
            module_path: Path to module file
            force_reinstrument: Force re-instrumentation if already done
            
        Returns:
            Instrumentation result dictionary or None if skipped
        """
        module_name = self._path_to_module_name(module_path)
        
        if module_name in self.instrumented_modules and not force_reinstrument:
            logger.debug(f"Module {module_name} already instrumented, skipping")
            return None
        
        try:
            # Import the module
            module = self._import_module(module_path, module_name)
            if module is None:
                return None
            
            # Detect phase from module path
            phase = self._detect_phase_from_path(module_path)
            
            # Count functions before instrumentation
            original_functions = self._count_functions(module)
            
            # Add process interface
            standardize_module(module, phase_override=phase)
            
            # Add telemetry instrumentation
            instrument_module_functions(
                module, 
                function_filter=self._should_instrument_function
            )
            
            # Count functions after instrumentation
            final_functions = self._count_functions(module)
            
            # Store instrumentation info
            instrumentation_info = {
                'module_name': module_name,
                'module_path': str(module_path),
                'phase': phase,
                'original_functions': original_functions,
                'final_functions': final_functions,
                'has_process_method': hasattr(module, 'process'),
                'adapters': getattr(module, '_process_adapters', {}),
                'instrumented_at': pipeline_telemetry.service_name
            }
            
            self.adapters_registry[module_name] = instrumentation_info
            self.instrumented_modules.add(module_name)
            
            logger.info(f"Instrumented module {module_name} with {original_functions} functions")
            return instrumentation_info
            
        except Exception as e:
            logger.error(f"Failed to instrument module {module_path}: {e}")
            return {
                'module_name': module_name,
                'module_path': str(module_path),
                'error': str(e),
                'instrumented_at': None
            }
    
    def instrument_all_modules(self, 
                              pattern: str = "**/*.py",
                              force_reinstrument: bool = False) -> Dict[str, Any]:
        """
        Discover and instrument all modules in the canonical flow.
        
        Args:
            pattern: Module discovery pattern
            force_reinstrument: Force re-instrumentation
            
        Returns:
            Summary of instrumentation results
        """
        modules = self.discover_modules(pattern)
        
        results = {
            'total_discovered': len(modules),
            'successful_instrumentations': 0,
            'failed_instrumentations': 0,
            'skipped_instrumentations': 0,
            'instrumentation_details': [],
            'errors': []
        }
        
        for module_path in modules:
            result = self.instrument_module(module_path, force_reinstrument)
            
            if result is None:
                results['skipped_instrumentations'] += 1
            elif 'error' in result:
                results['failed_instrumentations'] += 1
                results['errors'].append(result)
            else:
                results['successful_instrumentations'] += 1
            
            if result:
                results['instrumentation_details'].append(result)
        
        logger.info(
            f"Instrumentation complete: {results['successful_instrumentations']} successful, "
            f"{results['failed_instrumentations']} failed, {results['skipped_instrumentations']} skipped"
        )
        
        return results
    
    def get_instrumentation_report(self) -> Dict[str, Any]:
        """Get detailed report of all instrumented modules."""
        return {
            'instrumented_modules_count': len(self.instrumented_modules),
            'instrumented_modules': list(self.instrumented_modules),
            'registry': self.adapters_registry,
            'telemetry_metrics': pipeline_telemetry.get_metrics(),
            'service_name': pipeline_telemetry.service_name
        }
    
    def _path_to_module_name(self, module_path: Path) -> str:
        """Convert file path to Python module name."""
        # Get relative path from base
        try:
            rel_path = module_path.relative_to(self.base_path.parent)
        except ValueError:
            rel_path = module_path
        
        # Convert to module name
        parts = list(rel_path.parts[:-1]) + [rel_path.stem]
        return '.'.join(parts)
    
    def _import_module(self, module_path: Path, module_name: str):
        """Safely import a module from its path."""
        try:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                logger.warning(f"Could not create spec for {module_path}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            
            # Add to sys.modules before executing
            sys.modules[module_name] = module
            
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                # Remove from sys.modules if execution failed
                sys.modules.pop(module_name, None)
                raise e
            
            return module
            
        except Exception as e:
            logger.error(f"Failed to import {module_path}: {e}")
            return None
    
    def _detect_phase_from_path(self, module_path: Path) -> str:
        """Detect processing phase from module path."""
        path_parts = [p.lower() for p in module_path.parts]
        
        # Phase mapping based on canonical flow structure
        phase_mappings = {
            'i_ingestion_preparation': 'ingestion',
            'a_analysis_nlp': 'analysis', 
            'r_search_retrieval': 'retrieval',
            'k_knowledge_extraction': 'knowledge',
            'l_classification_evaluation': 'classification',
            'o_orchestration_control': 'orchestration',
            's_synthesis_output': 'synthesis',
            't_integration_storage': 'integration',
            'x_context_construction': 'context',
            'g_aggregation_reporting': 'aggregation',
            'mathematical_enhancers': 'mathematical'
        }
        
        for path_part in path_parts:
            if path_part in phase_mappings:
                return phase_mappings[path_part]
        
        # Fallback phase detection from file name
        file_name = module_path.stem.lower()
        
        phase_patterns = {
            'ingestion': ['load', 'read', 'ingest', 'import', 'extract'],
            'preparation': ['prepare', 'clean', 'normalize', 'validate'],
            'analysis': ['analyze', 'process', 'parse', 'nlp'],
            'retrieval': ['search', 'retrieve', 'find', 'query', 'index'],
            'knowledge': ['knowledge', 'graph', 'entity', 'concept'],
            'classification': ['classify', 'score', 'evaluate', 'rank'],
            'orchestration': ['orchestrate', 'coordinate', 'manage', 'control'],
            'synthesis': ['synthesize', 'generate', 'format', 'output'],
            'integration': ['integrate', 'store', 'save', 'persist'],
            'context': ['context', 'track', 'lineage', 'immutable'],
            'aggregation': ['aggregate', 'compile', 'report', 'audit']
        }
        
        for phase, patterns in phase_patterns.items():
            if any(pattern in file_name for pattern in patterns):
                return phase
        
        return 'processing'  # Default phase
    
    def _count_functions(self, module) -> int:
        """Count callable functions in a module."""
        count = 0
        for name in dir(module):
            if not name.startswith('_'):
                obj = getattr(module, name)
                if callable(obj) and not type(obj).__name__ == 'type':  # Exclude classes
                    count += 1
        return count
    
    def _should_instrument_function(self, name: str, func) -> bool:
        """Determine if a function should be instrumented."""
        # Skip private functions
        if name.startswith('_'):
            return False
        
        # Skip if already instrumented
        if hasattr(func, '__wrapped__'):
            return False
        
        # Skip certain function types
        import inspect
        if inspect.isclass(func) or inspect.ismodule(func):
            return False
        
        return True


class BatchInstrumenter:
    """
    Batch instrumentation for multiple canonical flow installations.
    """
    
    def __init__(self):
        self.instrumenters: Dict[str, AutoInstrumenter] = {}
    
    def register_canonical_flow(self, name: str, base_path: Path):
        """Register a canonical flow installation for instrumentation."""
        self.instrumenters[name] = AutoInstrumenter(base_path)
    
    def instrument_all_flows(self, force_reinstrument: bool = False) -> Dict[str, Any]:
        """Instrument all registered canonical flows."""
        results = {}
        
        for flow_name, instrumenter in self.instrumenters.items():
            logger.info(f"Instrumenting canonical flow: {flow_name}")
            results[flow_name] = instrumenter.instrument_all_modules(
                force_reinstrument=force_reinstrument
            )
        
        return results
    
    def get_global_report(self) -> Dict[str, Any]:
        """Get global instrumentation report for all flows."""
        global_report = {
            'flows': {},
            'total_modules': 0,
            'total_functions': 0,
            'global_metrics': {}
        }
        
        for flow_name, instrumenter in self.instrumenters.items():
            flow_report = instrumenter.get_instrumentation_report()
            global_report['flows'][flow_name] = flow_report
            global_report['total_modules'] += flow_report['instrumented_modules_count']
        
        return global_report


# Global instrumenter instance
default_instrumenter = AutoInstrumenter()


def instrument_canonical_flow(base_path: Optional[Path] = None,
                            pattern: str = "**/*.py",
                            force_reinstrument: bool = False) -> Dict[str, Any]:
    """
    Convenience function to instrument the canonical flow.
    
    Args:
        base_path: Base path for module discovery
        pattern: Module discovery pattern
        force_reinstrument: Force re-instrumentation
        
    Returns:
        Instrumentation results
    """
    if base_path:
        instrumenter = AutoInstrumenter(base_path)
    else:
        instrumenter = default_instrumenter
    
    return instrumenter.instrument_all_modules(pattern, force_reinstrument)


def get_instrumentation_report() -> Dict[str, Any]:
    """Get instrumentation report from default instrumenter."""
    return default_instrumenter.get_instrumentation_report()


def create_process_wrapper_for_function(func, 
                                       component_name: Optional[str] = None,
                                       phase: Optional[str] = None):
    """
    Create a standardized process wrapper for any function.
    
    Args:
        func: Function to wrap
        component_name: Component name for telemetry
        phase: Processing phase
        
    Returns:
        Wrapped function with process(data, context) interface
    """
    return ProcessAdapterFactory.create_process_wrapper(func, component_name, phase)