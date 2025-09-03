"""
Configuration Injection Mechanism

Provides dependency injection system for mathematical enhancers into bridge modules.
Manages enhancer instantiation, configuration, and lifecycle without creating direct
dependencies between bridges and concrete enhancer implementations.
"""

import logging
from typing import Dict, Any, Optional, Type, Protocol, Callable
import importlib
from dataclasses import dataclass, field
from pathlib import Path
import json

from canonical_flow.mathematical_enhancers.api_interfaces import (
    ProcessingPhase,
    MathematicalEnhancerAPI,
    ProcessingContext
)

logger = logging.getLogger(__name__)


@dataclass
class EnhancerConfiguration:
    """Configuration for mathematical enhancer instantiation"""
    enhancer_id: str
    enhancer_class: str
    module_path: str
    config_params: Dict[str, Any] = field(default_factory=dict)
    initialization_params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    phase: Optional[ProcessingPhase] = None


@dataclass
class InjectionConfiguration:
    """Configuration for dependency injection"""
    bridge_id: str
    enhancer_configs: Dict[str, EnhancerConfiguration] = field(default_factory=dict)
    default_enhancer: Optional[str] = None
    injection_strategy: str = "lazy"  # lazy, eager, singleton
    lifecycle_management: bool = True


class EnhancerFactory:
    """
    Factory for creating mathematical enhancer instances.
    
    Provides lazy loading and caching mechanisms to avoid unnecessary
    imports and instantiations of enhancer implementations.
    """
    
    def __init__(self):
        self._enhancer_cache: Dict[str, MathematicalEnhancerAPI] = {}
        self._enhancer_configs: Dict[str, EnhancerConfiguration] = {}
        self._singleton_instances: Dict[str, MathematicalEnhancerAPI] = {}
    
    def register_enhancer_config(self, config: EnhancerConfiguration) -> None:
        """
        Register enhancer configuration.
        
        Args:
            config: Enhancer configuration to register
        """
        self._enhancer_configs[config.enhancer_id] = config
        logger.info(f"Registered enhancer config: {config.enhancer_id}")
    
    def create_enhancer(self, enhancer_id: str, 
                       override_config: Optional[Dict[str, Any]] = None) -> Optional[MathematicalEnhancerAPI]:
        """
        Create mathematical enhancer instance.
        
        Args:
            enhancer_id: ID of enhancer to create
            override_config: Optional configuration overrides
            
        Returns:
            Mathematical enhancer instance or None if creation fails
        """
        config = self._enhancer_configs.get(enhancer_id)
        if not config:
            logger.error(f"Enhancer configuration not found: {enhancer_id}")
            return None
        
        if not config.enabled:
            logger.info(f"Enhancer disabled: {enhancer_id}")
            return None
        
        # Check singleton cache first
        if enhancer_id in self._singleton_instances:
            logger.debug(f"Returning singleton instance for: {enhancer_id}")
            return self._singleton_instances[enhancer_id]
        
        try:
            # Import enhancer module
            enhancer_module = importlib.import_module(config.module_path)
            
            # Get enhancer class
            enhancer_class = getattr(enhancer_module, config.enhancer_class)
            
            # Prepare configuration
            instance_config = config.config_params.copy()
            if override_config:
                instance_config.update(override_config)
            
            # Create instance
            enhancer = enhancer_class(config=instance_config)
            
            # Initialize if needed
            if hasattr(enhancer, 'initialize') and config.initialization_params:
                enhancer.initialize(**config.initialization_params)
            
            # Cache singleton if needed
            if config.enhancer_id.endswith('_singleton') or 'singleton' in instance_config:
                self._singleton_instances[enhancer_id] = enhancer
            
            logger.info(f"Created enhancer instance: {enhancer_id}")
            return enhancer
            
        except ImportError as e:
            logger.error(f"Failed to import enhancer module {config.module_path}: {e}")
        except AttributeError as e:
            logger.error(f"Enhancer class {config.enhancer_class} not found: {e}")
        except Exception as e:
            logger.error(f"Failed to create enhancer {enhancer_id}: {e}")
        
        return None
    
    def get_available_enhancers(self) -> Dict[str, EnhancerConfiguration]:
        """Get all available enhancer configurations"""
        return self._enhancer_configs.copy()
    
    def clear_cache(self) -> None:
        """Clear enhancer cache"""
        self._enhancer_cache.clear()
        self._singleton_instances.clear()


class ConfigurationInjector:
    """
    Dependency injection system for mathematical enhancers.
    
    Manages the injection of enhancer instances into bridge modules
    while maintaining loose coupling and configurable dependencies.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration injector.
        
        Args:
            config_path: Optional path to configuration files
        """
        self.config_path = config_path or Path("canonical_flow/configurations")
        self.enhancer_factory = EnhancerFactory()
        self.injection_configs: Dict[str, InjectionConfiguration] = {}
        self._load_default_configurations()
    
    def _load_default_configurations(self) -> None:
        """Load default enhancer and injection configurations"""
        # Default enhancer configurations
        default_enhancers = [
            EnhancerConfiguration(
                enhancer_id="math_stage1_ingestion",
                enhancer_class="MathStage1IngestionEnhancer",
                module_path="canonical_flow.mathematical_enhancers.ingestion_enhancer",
                phase=ProcessingPhase.INGESTION_PREPARATION,
                config_params={
                    "manifold_dimension": 3,
                    "preservation_threshold": 0.95
                }
            ),
            EnhancerConfiguration(
                enhancer_id="math_stage2_context",
                enhancer_class="MathStage2ContextEnhancer", 
                module_path="canonical_flow.mathematical_enhancers.context_enhancer",
                phase=ProcessingPhase.CONTEXT_CONSTRUCTION,
                config_params={
                    "topology_depth": 2,
                    "quantum_field_strength": 0.8
                }
            ),
            EnhancerConfiguration(
                enhancer_id="math_stage3_knowledge",
                enhancer_class="MathStage3KnowledgeEnhancer",
                module_path="canonical_flow.mathematical_enhancers.knowledge_enhancer",
                phase=ProcessingPhase.KNOWLEDGE_EXTRACTION,
                config_params={
                    "embedding_dimension": 768,
                    "graph_topology_preservation": True
                }
            ),
            EnhancerConfiguration(
                enhancer_id="math_stage4_analysis",
                enhancer_class="MathStage4AnalysisEnhancer",
                module_path="canonical_flow.mathematical_enhancers.analysis_enhancer",
                phase=ProcessingPhase.ANALYSIS_NLP,
                config_params={
                    "information_metric_depth": 3,
                    "spectral_analysis_enabled": True
                }
            ),
            EnhancerConfiguration(
                enhancer_id="math_stage5_scoring",
                enhancer_class="MathStage5ScoringEnhancer",
                module_path="canonical_flow.mathematical_enhancers.scoring_enhancer",
                phase=ProcessingPhase.CLASSIFICATION_EVALUATION,
                config_params={
                    "conformal_prediction_enabled": True,
                    "score_consistency_threshold": 0.9
                }
            ),
            EnhancerConfiguration(
                enhancer_id="math_stage7_orchestration",
                enhancer_class="MathStage7OrchestrationEnhancer",
                module_path="canonical_flow.mathematical_enhancers.orchestration_enhancer",
                phase=ProcessingPhase.ORCHESTRATION_CONTROL,
                config_params={
                    "load_balancing_enabled": True,
                    "optimization_algorithm": "gradient_descent"
                }
            ),
            EnhancerConfiguration(
                enhancer_id="math_stage6_retrieval",
                enhancer_class="MathStage6RetrievalEnhancer",
                module_path="canonical_flow.mathematical_enhancers.retrieval_enhancer",
                phase=ProcessingPhase.SEARCH_RETRIEVAL,
                config_params={
                    "similarity_enhancement_depth": 2,
                    "ranking_optimization_enabled": True
                }
            ),
            EnhancerConfiguration(
                enhancer_id="math_stage11_aggregation",
                enhancer_class="MathStage11AggregationEnhancer",
                module_path="canonical_flow.mathematical_enhancers.aggregation_enhancer",
                phase=ProcessingPhase.AGGREGATION_REPORTING,
                config_params={
                    "multi_dimensional_aggregation": True,
                    "statistical_validation": True
                }
            ),
            EnhancerConfiguration(
                enhancer_id="math_stage12_integration",
                enhancer_class="MathStage12IntegrationEnhancer",
                module_path="canonical_flow.mathematical_enhancers.integration_enhancer",
                phase=ProcessingPhase.INTEGRATION_STORAGE,
                config_params={
                    "consistency_validation": True,
                    "integration_depth": 3
                }
            )
        ]
        
        for config in default_enhancers:
            self.enhancer_factory.register_enhancer_config(config)
        
        # Default injection configurations
        self._create_default_injection_configs()
    
    def _create_default_injection_configs(self) -> None:
        """Create default injection configurations for bridges"""
        injection_mappings = [
            ("ingestion_enhancement_bridge", "math_stage1_ingestion"),
            ("context_enhancement_bridge", "math_stage2_context"),
            ("knowledge_enhancement_bridge", "math_stage3_knowledge"),
            ("analysis_enhancement_bridge", "math_stage4_analysis"),
            ("scoring_enhancement_bridge", "math_stage5_scoring"),
            ("orchestration_enhancement_bridge", "math_stage7_orchestration"),
            ("retrieval_enhancement_bridge", "math_stage6_retrieval"),
            ("aggregation_enhancement_bridge", "math_stage11_aggregation"),
            ("integration_enhancement_bridge", "math_stage12_integration")
        ]
        
        for bridge_id, enhancer_id in injection_mappings:
            config = InjectionConfiguration(
                bridge_id=bridge_id,
                default_enhancer=enhancer_id,
                injection_strategy="lazy",
                lifecycle_management=True
            )
            # Add enhancer config reference
            enhancer_config = self.enhancer_factory._enhancer_configs.get(enhancer_id)
            if enhancer_config:
                config.enhancer_configs[enhancer_id] = enhancer_config
            
            self.injection_configs[bridge_id] = config
    
    def inject_enhancer_into_bridge(self, bridge_instance: Any, 
                                   bridge_id: str,
                                   enhancer_override: Optional[str] = None) -> bool:
        """
        Inject mathematical enhancer into bridge instance.
        
        Args:
            bridge_instance: Bridge instance to inject enhancer into
            bridge_id: ID of the bridge
            enhancer_override: Optional enhancer ID to use instead of default
            
        Returns:
            True if injection successful, False otherwise
        """
        try:
            injection_config = self.injection_configs.get(bridge_id)
            if not injection_config:
                logger.warning(f"No injection configuration found for bridge: {bridge_id}")
                return False
            
            # Determine enhancer to inject
            enhancer_id = enhancer_override or injection_config.default_enhancer
            if not enhancer_id:
                logger.info(f"No enhancer configured for bridge: {bridge_id}")
                return True  # Not an error, just no enhancement
            
            # Create enhancer instance
            enhancer = self.enhancer_factory.create_enhancer(enhancer_id)
            if not enhancer:
                logger.error(f"Failed to create enhancer {enhancer_id} for bridge {bridge_id}")
                return False
            
            # Inject into bridge
            if hasattr(bridge_instance, 'inject_enhancer'):
                bridge_instance.inject_enhancer(enhancer)
                logger.info(f"Injected enhancer {enhancer_id} into bridge {bridge_id}")
                return True
            else:
                logger.error(f"Bridge {bridge_id} does not support enhancer injection")
                return False
                
        except Exception as e:
            logger.error(f"Failed to inject enhancer into bridge {bridge_id}: {e}")
            return False
    
    def create_processing_context_with_enhancer(self, base_context: Dict[str, Any],
                                               enhancer_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create processing context with injected enhancer reference.
        
        Args:
            base_context: Base context dictionary
            enhancer_id: Optional enhancer ID to inject
            
        Returns:
            Enhanced context with injected enhancer
        """
        enhanced_context = base_context.copy()
        
        if enhancer_id:
            enhancer = self.enhancer_factory.create_enhancer(enhancer_id)
            if enhancer:
                enhanced_context['injected_enhancer'] = enhancer
                enhanced_context['enhancer_metadata'] = enhancer.get_enhancement_metadata()
        
        return enhanced_context
    
    def load_configuration_from_file(self, config_file: Path) -> None:
        """
        Load configuration from JSON file.
        
        Args:
            config_file: Path to configuration file
        """
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Load enhancer configurations
            if 'enhancers' in config_data:
                for enhancer_data in config_data['enhancers']:
                    config = EnhancerConfiguration(**enhancer_data)
                    self.enhancer_factory.register_enhancer_config(config)
            
            # Load injection configurations
            if 'injections' in config_data:
                for injection_data in config_data['injections']:
                    config = InjectionConfiguration(**injection_data)
                    self.injection_configs[config.bridge_id] = config
            
            logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file}: {e}")
    
    def export_configuration(self, output_file: Path) -> None:
        """
        Export current configuration to JSON file.
        
        Args:
            output_file: Path to output configuration file
        """
        try:
            config_data = {
                'enhancers': [
                    {
                        'enhancer_id': config.enhancer_id,
                        'enhancer_class': config.enhancer_class,
                        'module_path': config.module_path,
                        'config_params': config.config_params,
                        'initialization_params': config.initialization_params,
                        'enabled': config.enabled,
                        'phase': config.phase.value if config.phase else None
                    }
                    for config in self.enhancer_factory.get_available_enhancers().values()
                ],
                'injections': [
                    {
                        'bridge_id': config.bridge_id,
                        'default_enhancer': config.default_enhancer,
                        'injection_strategy': config.injection_strategy,
                        'lifecycle_management': config.lifecycle_management
                    }
                    for config in self.injection_configs.values()
                ]
            }
            
            with open(output_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Exported configuration to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to export configuration to {output_file}: {e}")
    
    def get_injection_summary(self) -> Dict[str, Any]:
        """Get summary of injection configurations"""
        return {
            'total_enhancers': len(self.enhancer_factory.get_available_enhancers()),
            'total_injections': len(self.injection_configs),
            'injection_mappings': {
                bridge_id: config.default_enhancer
                for bridge_id, config in self.injection_configs.items()
            },
            'enhancer_phases': {
                enhancer_id: config.phase.value if config.phase else 'unspecified'
                for enhancer_id, config in self.enhancer_factory.get_available_enhancers().items()
            }
        }


# Global configuration injector instance
_configuration_injector: Optional[ConfigurationInjector] = None


def get_configuration_injector() -> ConfigurationInjector:
    """Get the global configuration injector instance"""
    global _configuration_injector
    if _configuration_injector is None:
        _configuration_injector = ConfigurationInjector()
    return _configuration_injector


def inject_enhancers_into_context(context: Dict[str, Any], 
                                 phase: ProcessingPhase) -> Dict[str, Any]:
    """
    Convenience function to inject appropriate enhancers into processing context.
    
    Args:
        context: Processing context
        phase: Current processing phase
        
    Returns:
        Enhanced context with injected enhancers
    """
    injector = get_configuration_injector()
    
    # Find appropriate enhancer for phase
    available_enhancers = injector.enhancer_factory.get_available_enhancers()
    phase_enhancers = [
        enhancer_id for enhancer_id, config in available_enhancers.items()
        if config.phase == phase and config.enabled
    ]
    
    if phase_enhancers:
        # Use first available enhancer for the phase
        enhancer_id = phase_enhancers[0]
        return injector.create_processing_context_with_enhancer(context, enhancer_id)
    
    return context