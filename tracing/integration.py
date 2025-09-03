"""
Integration system for instrumenting canonical pipeline components
"""

import inspect
import importlib
from typing import Dict, List, Any, Optional
from pathlib import Path

from .decorators import trace_process, auto_trace_process
from .otel_tracer import get_pipeline_tracer, CANONICAL_PHASES


class PipelineInstrumentor:
    """Automatically instruments canonical pipeline components with tracing"""
    
    def __init__(self, canonical_flow_path: str = "canonical_flow"):
        self.canonical_flow_path = canonical_flow_path
        self.instrumented_components: Dict[str, Any] = {}
        
    def instrument_all_components(self):
        """Instrument all components in canonical_flow with tracing"""
        print("Starting automatic instrumentation of canonical pipeline components...")
        
        # Get all phase directories
        phase_mappings = self._discover_phase_components()
        
        instrumented_count = 0
        for phase, components in phase_mappings.items():
            for component_path, component_info in components.items():
                if self._instrument_component(phase, component_path, component_info):
                    instrumented_count += 1
                    
        print(f"Successfully instrumented {instrumented_count} components")
        return instrumented_count
        
    def _discover_phase_components(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Discover all components with process(data, context) interfaces"""
        phase_mappings = {}
        
        for phase in CANONICAL_PHASES:
            phase_mappings[phase] = {}
            phase_dir = Path(self.canonical_flow_path) / phase
            
            if not phase_dir.exists():
                continue
                
            # Find all Python files in phase directory
            for py_file in phase_dir.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                    
                module_path = f"{self.canonical_flow_path}.{phase}.{py_file.stem}"
                
                try:
                    # Load module and check for process function
                    module = importlib.import_module(module_path)
                    
                    process_funcs = self._find_process_functions(module)
                    if process_funcs:
                        phase_mappings[phase][module_path] = {
                            'module': module,
                            'process_functions': process_funcs,
                            'file_path': str(py_file)
                        }
                        
                except Exception as e:
                    print(f"Warning: Could not load module {module_path}: {e}")
                    continue
                    
        return phase_mappings
        
    def _find_process_functions(self, module) -> List[Dict[str, Any]]:
        """Find all process(data, context) functions in a module"""
        process_funcs = []
        
        # Check for module-level process function
        if hasattr(module, 'process'):
            func = getattr(module, 'process')
            if self._is_process_function(func):
                process_funcs.append({
                    'name': 'process',
                    'function': func,
                    'type': 'module_function',
                    'class_name': None
                })
                
        # Check for class-based process methods
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                if hasattr(obj, 'process'):
                    method = getattr(obj, 'process')
                    if self._is_process_function(method):
                        process_funcs.append({
                            'name': f"{name}.process",
                            'function': method,
                            'type': 'class_method',
                            'class_name': name
                        })
                        
        return process_funcs
        
    def _is_process_function(self, func) -> bool:
        """Check if function matches process(data, context) signature"""
        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            
            # Check for various valid signatures
            valid_signatures = [
                ['data', 'context'],
                ['self', 'data', 'context'],  # class method
                ['cls', 'data', 'context'],   # class method
                ['data'],                     # simplified
                ['self', 'data'],            # simplified class method
            ]
            
            return any(
                all(param in params for param in valid_sig[:len(params)])
                for valid_sig in valid_signatures
            )
            
        except Exception:
            return False
            
    def _instrument_component(self, phase: str, module_path: str, 
                            component_info: Dict[str, Any]) -> bool:
        """Instrument a specific component with tracing"""
        try:
            module = component_info['module']
            process_funcs = component_info['process_functions']
            
            # Determine target phase (for now assume same phase, can be refined)
            target_phase = phase
            
            for func_info in process_funcs:
                func_name = func_info['name']
                original_func = func_info['function']
                
                # Create instrumented version
                instrumented_func = self._create_instrumented_function(
                    original_func, phase, target_phase, func_name
                )
                
                # Replace the original function
                if func_info['type'] == 'module_function':
                    setattr(module, 'process', instrumented_func)
                elif func_info['type'] == 'class_method':
                    class_name = func_info['class_name']
                    cls = getattr(module, class_name)
                    setattr(cls, 'process', instrumented_func)
                    
                # Track instrumented component
                self.instrumented_components[f"{module_path}.{func_name}"] = {
                    'phase': phase,
                    'target_phase': target_phase,
                    'original_function': original_func,
                    'instrumented_function': instrumented_func
                }
                
            print(f"Instrumented: {module_path} ({len(process_funcs)} functions)")
            return True
            
        except Exception as e:
            print(f"Failed to instrument {module_path}: {e}")
            return False
            
    def _create_instrumented_function(self, original_func, source_phase: str, 
                                    target_phase: str, component_name: str):
        """Create instrumented version of process function"""
        
        # Use the trace_process decorator
        decorated_func = trace_process(
            source_phase=source_phase,
            target_phase=target_phase, 
            component_name=component_name
        )(original_func)
        
        return decorated_func
        
    def create_phase_transition_hooks(self):
        """Create hooks for detecting phase transitions"""
        
        # This would require deeper integration with the orchestration system
        # For now, we rely on the span data to detect transitions
        pass
        
    def get_instrumentation_report(self) -> Dict[str, Any]:
        """Get report of instrumented components"""
        
        phase_counts = {}
        for component_name, info in self.instrumented_components.items():
            phase = info['phase']
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            
        return {
            'total_instrumented': len(self.instrumented_components),
            'by_phase': phase_counts,
            'components': list(self.instrumented_components.keys()),
            'canonical_phases': CANONICAL_PHASES
        }


def auto_instrument_pipeline(canonical_flow_path: str = "canonical_flow") -> PipelineInstrumentor:
    """Automatically instrument the entire canonical pipeline"""
    instrumentor = PipelineInstrumentor(canonical_flow_path)
    instrumentor.instrument_all_components()
    return instrumentor


# Specific phase instrumentation helpers
def instrument_ingestion_phase():
    """Instrument I_ingestion_preparation phase"""
    return _instrument_phase("I_ingestion_preparation", "X_context_construction")

def instrument_context_phase():
    """Instrument X_context_construction phase"""
    return _instrument_phase("X_context_construction", "K_knowledge_extraction")

def instrument_knowledge_phase():
    """Instrument K_knowledge_extraction phase"""
    return _instrument_phase("K_knowledge_extraction", "A_analysis_nlp")

def instrument_analysis_phase():
    """Instrument A_analysis_nlp phase"""
    return _instrument_phase("A_analysis_nlp", "L_classification_evaluation")

def instrument_classification_phase():
    """Instrument L_classification_evaluation phase"""  
    return _instrument_phase("L_classification_evaluation", "R_search_retrieval")

def instrument_retrieval_phase():
    """Instrument R_search_retrieval phase"""
    return _instrument_phase("R_search_retrieval", "O_orchestration_control")

def instrument_orchestration_phase():
    """Instrument O_orchestration_control phase"""
    return _instrument_phase("O_orchestration_control", "G_aggregation_reporting")

def instrument_aggregation_phase():
    """Instrument G_aggregation_reporting phase"""
    return _instrument_phase("G_aggregation_reporting", "T_integration_storage")

def instrument_integration_phase():
    """Instrument T_integration_storage phase"""
    return _instrument_phase("T_integration_storage", "S_synthesis_output")

def instrument_synthesis_phase():
    """Instrument S_synthesis_output phase"""
    return _instrument_phase("S_synthesis_output", "S_synthesis_output")  # Terminal phase


def _instrument_phase(source_phase: str, target_phase: str) -> int:
    """Helper to instrument a specific phase"""
    instrumentor = PipelineInstrumentor()
    phase_mappings = instrumentor._discover_phase_components()
    
    if source_phase not in phase_mappings:
        return 0
        
    instrumented = 0
    for module_path, component_info in phase_mappings[source_phase].items():
        if instrumentor._instrument_component(source_phase, module_path, component_info):
            instrumented += 1
            
    return instrumented