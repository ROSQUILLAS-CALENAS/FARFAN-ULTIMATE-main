"""
Backward Compatibility Layer for Canonical Flow Components

Provides compatibility adapters and migration utilities to ensure existing
canonical_flow components continue to work while gradually transitioning to
the new phases architecture.
"""

import warnings
from typing import Any, Dict, Optional, Union
from functools import wraps
import importlib


def deprecated_canonical_import(canonical_path: str, phase_equivalent: str):
    """
    Decorator to mark canonical imports as deprecated and suggest phase alternatives.
    
    Args:
        canonical_path: The canonical_flow module path being deprecated
        phase_equivalent: The equivalent phases path to use instead
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"Using {canonical_path} is deprecated. "
                f"Consider migrating to {phase_equivalent} for better isolation.",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


class CanonicalToPhaseAdapter:
    """
    Adapter to bridge canonical_flow interfaces to phases interfaces.
    
    This allows existing code to work with minimal changes while
    encouraging migration to the new phase-based architecture.
    """
    
    def __init__(self):
        self._phase_mappings = {
            'canonical_flow.I_ingestion_preparation': 'phases.I',
            'canonical_flow.X_context_construction': 'phases.X',
            'canonical_flow.K_knowledge_extraction': 'phases.K',
            'canonical_flow.A_analysis_nlp': 'phases.A',
            'canonical_flow.L_classification_evaluation': 'phases.L',
            'canonical_flow.R_search_retrieval': 'phases.R',
            'canonical_flow.O_orchestration_control': 'phases.O',
            'canonical_flow.G_aggregation_reporting': 'phases.G',
            'canonical_flow.T_integration_storage': 'phases.T',
            'canonical_flow.S_synthesis_output': 'phases.S'
        }
    
    def get_phase_equivalent(self, canonical_module: str) -> Optional[str]:
        """Get the phase equivalent for a canonical module."""
        for canonical_path, phase_path in self._phase_mappings.items():
            if canonical_module.startswith(canonical_path):
                return phase_path
        return None
    
    def migrate_context_data(self, canonical_data: Dict[str, Any], phase: str) -> Dict[str, Any]:
        """
        Migrate canonical context data format to phase-compatible format.
        
        Args:
            canonical_data: Data in canonical_flow format
            phase: Target phase identifier
            
        Returns:
            Data in phase-compatible format
        """
        # Base migration common to all phases
        migrated = {
            'base_data': canonical_data.get('base_data', {}),
            'config': canonical_data.get('config', {}),
            'upstream_data': canonical_data.get('upstream_results', {})
        }
        
        # Phase-specific migrations
        if phase == 'I':
            migrated.update({
                'data_path': canonical_data.get('data_path', ''),
                'component_states': canonical_data.get('component_results', {})
            })
        
        elif phase == 'X':
            migrated.update({
                'base_context': canonical_data.get('context', {}),
            })
        
        elif phase == 'K':
            migrated.update({
                'extraction_config': canonical_data.get('config', {}),
                'model_configs': canonical_data.get('model_settings', {})
            })
        
        elif phase == 'A':
            migrated.update({
                'analysis_config': canonical_data.get('config', {}),
                'model_settings': canonical_data.get('model_config', {})
            })
        
        elif phase == 'L':
            migrated.update({
                'classification_config': canonical_data.get('config', {}),
                'evaluation_criteria': canonical_data.get('criteria', {})
            })
        
        elif phase == 'R':
            migrated.update({
                'query_config': canonical_data.get('config', {}),
                'retrieval_settings': canonical_data.get('settings', {})
            })
        
        elif phase == 'O':
            migrated.update({
                'orchestration_config': canonical_data.get('config', {}),
                'control_policies': canonical_data.get('policies', {})
            })
        
        elif phase == 'G':
            migrated.update({
                'aggregation_config': canonical_data.get('config', {}),
                'reporting_settings': canonical_data.get('settings', {})
            })
        
        elif phase == 'T':
            migrated.update({
                'storage_config': canonical_data.get('config', {}),
                'integration_settings': canonical_data.get('settings', {})
            })
        
        elif phase == 'S':
            migrated.update({
                'synthesis_config': canonical_data.get('config', {}),
                'output_settings': canonical_data.get('settings', {})
            })
        
        return migrated
    
    def create_phase_context(self, canonical_context: Dict[str, Any], phase: str) -> Any:
        """
        Create a phase-specific context object from canonical context data.
        
        Args:
            canonical_context: Context data in canonical format
            phase: Target phase identifier
            
        Returns:
            Phase-specific context object
        """
        try:
            # Import phase-specific context class
            phase_module = importlib.import_module(f'phases.{phase}')
            
            # Get context class name
            context_class_name = self._get_context_class_name(phase)
            context_class = getattr(phase_module, context_class_name)
            
            # Migrate data to phase format
            migrated_data = self.migrate_context_data(canonical_context, phase)
            
            # Create context object
            return context_class(**migrated_data)
            
        except (ImportError, AttributeError) as e:
            warnings.warn(f"Could not create {phase} context: {e}")
            # Return a generic dict as fallback
            return self.migrate_context_data(canonical_context, phase)
    
    def _get_context_class_name(self, phase: str) -> str:
        """Get the context class name for a phase."""
        context_mappings = {
            'I': 'IngestionContext',
            'X': 'ContextConstructionContext',
            'K': 'KnowledgeExtractionContext',
            'A': 'AnalysisContext',
            'L': 'ClassificationContext',
            'R': 'RetrievalContext',
            'O': 'OrchestrationContext',
            'G': 'AggregationContext',
            'T': 'IntegrationContext',
            'S': 'SynthesisContext'
        }
        return context_mappings.get(phase, f'{phase}Context')


class CanonicalFlowCompatibilityProxy:
    """
    Proxy class that provides canonical_flow interfaces while using phases backend.
    
    This allows gradual migration by maintaining the old interface while
    delegating to the new phase-based implementation.
    """
    
    def __init__(self, phase: str):
        self.phase = phase
        self.adapter = CanonicalToPhaseAdapter()
    
    def execute_full_pipeline(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute pipeline using phase backend with canonical interface."""
        warnings.warn(
            f"canonical_flow.{self.phase} interface is deprecated. "
            f"Use phases.{self.phase} directly for better performance.",
            DeprecationWarning,
            stacklevel=2
        )
        
        try:
            # Import phase module
            phase_module = importlib.import_module(f'phases.{self.phase}')
            phase_class = getattr(phase_module, f'{self._get_phase_class_name()}')
            
            # Create phase context
            context_data = kwargs.copy()
            context_data['base_data'] = input_data
            phase_context = self.adapter.create_phase_context(context_data, self.phase)
            
            # Execute through phase interface
            result = phase_class.process(input_data, phase_context)
            
            # Convert result back to canonical format
            return self._convert_phase_result_to_canonical(result)
            
        except Exception as e:
            # Fallback to original canonical implementation if available
            warnings.warn(f"Phase execution failed, attempting canonical fallback: {e}")
            return self._canonical_fallback(input_data, **kwargs)
    
    def _get_phase_class_name(self) -> str:
        """Get the phase class name."""
        phase_mappings = {
            'I': 'IngestionPhase',
            'X': 'ContextConstructionPhase', 
            'K': 'KnowledgeExtractionPhase',
            'A': 'AnalysisPhase',
            'L': 'ClassificationPhase',
            'R': 'RetrievalPhase',
            'O': 'OrchestrationPhase',
            'G': 'AggregationPhase',
            'T': 'IntegrationPhase',
            'S': 'SynthesisPhase'
        }
        return phase_mappings.get(self.phase, f'{self.phase}Phase')
    
    def _convert_phase_result_to_canonical(self, phase_result: Any) -> Dict[str, Any]:
        """Convert phase result back to canonical format."""
        if hasattr(phase_result, '__dict__'):
            # Convert dataclass to dict
            result_dict = phase_result.__dict__.copy()
        else:
            result_dict = phase_result
        
        # Map phase result fields to canonical format
        canonical_result = {
            'success': True,
            'results': result_dict,
            'execution_summary': {
                'method': 'phase_backend',
                'phase': self.phase
            }
        }
        
        return canonical_result
    
    def _canonical_fallback(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Fallback to canonical implementation if available."""
        try:
            canonical_module_name = f'canonical_flow.{self._get_canonical_module_name()}'
            canonical_module = importlib.import_module(canonical_module_name)
            
            # Attempt to use orchestrator
            orchestrator_name = f'{self.phase}Orchestrator'
            if hasattr(canonical_module, orchestrator_name):
                orchestrator_class = getattr(canonical_module, orchestrator_name)
                orchestrator = orchestrator_class(**kwargs)
                return orchestrator.execute_full_pipeline(input_data)
            
        except ImportError:
            pass
        
        # Ultimate fallback
        return {
            'success': False,
            'error': f'No implementation available for phase {self.phase}',
            'fallback_used': True
        }
    
    def _get_canonical_module_name(self) -> str:
        """Get canonical module name for phase."""
        canonical_mappings = {
            'I': 'I_ingestion_preparation',
            'X': 'X_context_construction',
            'K': 'K_knowledge_extraction',
            'A': 'A_analysis_nlp',
            'L': 'L_classification_evaluation',
            'R': 'R_search_retrieval',
            'O': 'O_orchestration_control',
            'G': 'G_aggregation_reporting',
            'T': 'T_integration_storage',
            'S': 'S_synthesis_output'
        }
        return canonical_mappings.get(self.phase, f'{self.phase}_module')


def create_compatibility_proxy(phase: str) -> CanonicalFlowCompatibilityProxy:
    """Create a compatibility proxy for a specific phase."""
    return CanonicalFlowCompatibilityProxy(phase)


def install_compatibility_layer():
    """Install the compatibility layer globally."""
    warnings.warn(
        "Installing global compatibility layer. "
        "This is intended for migration only and should be removed eventually.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Install compatibility proxies for all phases
    import sys
    
    for phase in ['I', 'X', 'K', 'A', 'L', 'R', 'O', 'G', 'T', 'S']:
        proxy = create_compatibility_proxy(phase)
        
        # Make proxy available as canonical_flow module
        canonical_module_name = f'canonical_flow_{phase}_compatibility_proxy'
        sys.modules[canonical_module_name] = proxy


__all__ = [
    'deprecated_canonical_import',
    'CanonicalToPhaseAdapter',
    'CanonicalFlowCompatibilityProxy',
    'create_compatibility_proxy',
    'install_compatibility_layer'
]