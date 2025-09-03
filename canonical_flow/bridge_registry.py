"""
Bridge Registration System

Provides registration and discovery mechanisms for dependency-inverted bridge modules
within the canonical pipeline flow. Manages bridge instantiation, enhancer injection,
and integration with the existing index.json structure.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Type, Callable
from pathlib import Path
import importlib
from dataclasses import dataclass, asdict

from canonical_flow.mathematical_enhancers.api_interfaces import (
    ProcessingPhase,
    MathematicalEnhancerAPI
)

logger = logging.getLogger(__name__)


@dataclass
class BridgeMetadata:
    """Metadata for bridge module registration"""
    bridge_id: str
    bridge_name: str
    phase: ProcessingPhase
    module_path: str
    api_interface: str
    dependencies: List[str]
    priority: int = 50
    enabled: bool = True


@dataclass
class BridgeRegistration:
    """Complete bridge registration information"""
    metadata: BridgeMetadata
    bridge_class: Optional[Type] = None
    process_function: Optional[Callable] = None
    factory_function: Optional[Callable] = None


class BridgeRegistry:
    """
    Registry for managing dependency-inverted bridge modules.
    
    Provides discovery, instantiation, and configuration injection
    mechanisms for mathematical enhancement bridges.
    """
    
    def __init__(self, canonical_flow_path: Optional[Path] = None):
        """
        Initialize bridge registry.
        
        Args:
            canonical_flow_path: Path to canonical flow directory
        """
        self.canonical_flow_path = canonical_flow_path or Path("canonical_flow")
        self.registered_bridges: Dict[str, BridgeRegistration] = {}
        self.phase_bridges: Dict[ProcessingPhase, List[str]] = {}
        self._initialize_phase_mappings()
    
    def _initialize_phase_mappings(self) -> None:
        """Initialize phase to bridge mappings"""
        for phase in ProcessingPhase:
            self.phase_bridges[phase] = []
    
    def register_bridge(self, metadata: BridgeMetadata) -> None:
        """
        Register a bridge module.
        
        Args:
            metadata: Bridge metadata containing registration information
        """
        try:
            # Import bridge module
            bridge_module = importlib.import_module(metadata.module_path)
            
            # Extract bridge components
            bridge_class = None
            process_function = None
            factory_function = None
            
            # Look for standard bridge class
            if hasattr(bridge_module, f"{metadata.bridge_name}"):
                bridge_class = getattr(bridge_module, metadata.bridge_name)
            
            # Look for standard process function
            if hasattr(bridge_module, 'process'):
                process_function = getattr(bridge_module, 'process')
            
            # Look for factory function
            factory_name = f"create_{metadata.bridge_id.lower()}_bridge"
            if hasattr(bridge_module, factory_name):
                factory_function = getattr(bridge_module, factory_name)
            
            # Create registration
            registration = BridgeRegistration(
                metadata=metadata,
                bridge_class=bridge_class,
                process_function=process_function,
                factory_function=factory_function
            )
            
            # Register bridge
            self.registered_bridges[metadata.bridge_id] = registration
            self.phase_bridges[metadata.phase].append(metadata.bridge_id)
            
            logger.info(f"Registered bridge: {metadata.bridge_id} for phase {metadata.phase.value}")
            
        except ImportError as e:
            logger.error(f"Failed to import bridge module {metadata.module_path}: {e}")
        except Exception as e:
            logger.error(f"Failed to register bridge {metadata.bridge_id}: {e}")
    
    def discover_bridges(self) -> None:
        """Automatically discover and register bridge modules"""
        bridge_configs = [
            # Ingestion phase bridges
            BridgeMetadata(
                bridge_id="ingestion_enhancement_bridge",
                bridge_name="IngestionEnhancementBridge",
                phase=ProcessingPhase.INGESTION_PREPARATION,
                module_path="canonical_flow.I_ingestion_preparation.bridge_ingestion_enhancer",
                api_interface="IngestionEnhancerAPI",
                dependencies=["canonical_flow.mathematical_enhancers.api_interfaces"],
                priority=10
            ),
            
            # Context phase bridges
            BridgeMetadata(
                bridge_id="context_enhancement_bridge",
                bridge_name="ContextEnhancementBridge",
                phase=ProcessingPhase.CONTEXT_CONSTRUCTION,
                module_path="canonical_flow.X_context_construction.bridge_context_enhancer",
                api_interface="ContextEnhancerAPI",
                dependencies=["canonical_flow.mathematical_enhancers.api_interfaces"],
                priority=20
            ),
            
            # Knowledge phase bridges
            BridgeMetadata(
                bridge_id="knowledge_enhancement_bridge",
                bridge_name="KnowledgeEnhancementBridge",
                phase=ProcessingPhase.KNOWLEDGE_EXTRACTION,
                module_path="canonical_flow.K_knowledge_extraction.bridge_knowledge_enhancer",
                api_interface="KnowledgeEnhancerAPI",
                dependencies=["canonical_flow.mathematical_enhancers.api_interfaces"],
                priority=30
            ),
            
            # Analysis phase bridges
            BridgeMetadata(
                bridge_id="analysis_enhancement_bridge",
                bridge_name="AnalysisEnhancementBridge",
                phase=ProcessingPhase.ANALYSIS_NLP,
                module_path="canonical_flow.A_analysis_nlp.bridge_analysis_enhancer",
                api_interface="AnalysisEnhancerAPI",
                dependencies=["canonical_flow.mathematical_enhancers.api_interfaces"],
                priority=40
            ),
            
            # Scoring phase bridges
            BridgeMetadata(
                bridge_id="scoring_enhancement_bridge",
                bridge_name="ScoringEnhancementBridge",
                phase=ProcessingPhase.CLASSIFICATION_EVALUATION,
                module_path="canonical_flow.L_classification_evaluation.bridge_scoring_enhancer",
                api_interface="ScoringEnhancerAPI",
                dependencies=["canonical_flow.mathematical_enhancers.api_interfaces"],
                priority=50
            ),
            
            # Orchestration phase bridges
            BridgeMetadata(
                bridge_id="orchestration_enhancement_bridge",
                bridge_name="OrchestrationEnhancementBridge",
                phase=ProcessingPhase.ORCHESTRATION_CONTROL,
                module_path="canonical_flow.O_orchestration_control.bridge_orchestration_enhancer",
                api_interface="OrchestrationEnhancerAPI",
                dependencies=["canonical_flow.mathematical_enhancers.api_interfaces"],
                priority=60
            ),
            
            # Retrieval phase bridges
            BridgeMetadata(
                bridge_id="retrieval_enhancement_bridge",
                bridge_name="RetrievalEnhancementBridge",
                phase=ProcessingPhase.SEARCH_RETRIEVAL,
                module_path="canonical_flow.R_search_retrieval.bridge_retrieval_enhancer",
                api_interface="RetrievalEnhancerAPI",
                dependencies=["canonical_flow.mathematical_enhancers.api_interfaces"],
                priority=70
            ),
            
            # Synthesis phase bridges
            BridgeMetadata(
                bridge_id="synthesis_enhancement_bridge",
                bridge_name="SynthesisEnhancementBridge",
                phase=ProcessingPhase.SYNTHESIS_OUTPUT,
                module_path="canonical_flow.S_synthesis_output.bridge_synthesis_enhancer",
                api_interface="SynthesisEnhancerAPI",
                dependencies=["canonical_flow.mathematical_enhancers.api_interfaces"],
                priority=80
            ),
            
            # Aggregation phase bridges
            BridgeMetadata(
                bridge_id="aggregation_enhancement_bridge",
                bridge_name="AggregationEnhancementBridge",
                phase=ProcessingPhase.AGGREGATION_REPORTING,
                module_path="canonical_flow.G_aggregation_reporting.bridge_aggregation_enhancer",
                api_interface="AggregationEnhancerAPI",
                dependencies=["canonical_flow.mathematical_enhancers.api_interfaces"],
                priority=90
            ),
            
            # Integration phase bridges
            BridgeMetadata(
                bridge_id="integration_enhancement_bridge",
                bridge_name="IntegrationEnhancementBridge",
                phase=ProcessingPhase.INTEGRATION_STORAGE,
                module_path="canonical_flow.T_integration_storage.bridge_integration_enhancer",
                api_interface="IntegrationEnhancerAPI",
                dependencies=["canonical_flow.mathematical_enhancers.api_interfaces"],
                priority=100
            ),
        ]
        
        for bridge_config in bridge_configs:
            self.register_bridge(bridge_config)
    
    def get_bridges_for_phase(self, phase: ProcessingPhase) -> List[BridgeRegistration]:
        """
        Get all registered bridges for a specific processing phase.
        
        Args:
            phase: Processing phase to get bridges for
            
        Returns:
            List of bridge registrations for the phase
        """
        bridge_ids = self.phase_bridges.get(phase, [])
        bridges = []
        
        for bridge_id in bridge_ids:
            registration = self.registered_bridges.get(bridge_id)
            if registration and registration.metadata.enabled:
                bridges.append(registration)
        
        # Sort by priority
        bridges.sort(key=lambda x: x.metadata.priority)
        return bridges
    
    def create_bridge_instance(self, bridge_id: str, 
                              enhancer: Optional[MathematicalEnhancerAPI] = None) -> Any:
        """
        Create an instance of a registered bridge.
        
        Args:
            bridge_id: ID of the bridge to instantiate
            enhancer: Optional mathematical enhancer to inject
            
        Returns:
            Bridge instance or None if not found
        """
        registration = self.registered_bridges.get(bridge_id)
        if not registration:
            logger.error(f"Bridge not found: {bridge_id}")
            return None
        
        try:
            # Use factory function if available
            if registration.factory_function:
                return registration.factory_function(enhancer)
            
            # Use bridge class constructor
            elif registration.bridge_class:
                return registration.bridge_class(enhancer)
            
            else:
                logger.error(f"No instantiation method available for bridge: {bridge_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create bridge instance {bridge_id}: {e}")
            return None
    
    def update_index_json(self) -> None:
        """Update canonical_flow/index.json with bridge registrations"""
        try:
            index_path = self.canonical_flow_path / "index.json"
            
            # Load existing index
            existing_index = []
            if index_path.exists():
                with open(index_path, 'r') as f:
                    existing_index = json.load(f)
            
            # Add bridge entries
            bridge_entries = []
            next_code = max([int(item.get('code', '0')[:-1]) for item in existing_index if item.get('code')] + [0]) + 1
            
            for bridge_id, registration in self.registered_bridges.items():
                if not registration.metadata.enabled:
                    continue
                
                # Generate unique code
                phase_prefix = self._get_phase_prefix(registration.metadata.phase)
                code = f"{next_code:02d}{phase_prefix}"
                next_code += 1
                
                # Create bridge entry
                bridge_entry = {
                    "code": code,
                    "stage": registration.metadata.phase.value,
                    "alias_path": registration.metadata.module_path.replace('.', '/') + '.py',
                    "original_path": f"bridge_{registration.metadata.bridge_id}.py",
                    "bridge_metadata": {
                        "bridge_id": registration.metadata.bridge_id,
                        "bridge_name": registration.metadata.bridge_name,
                        "api_interface": registration.metadata.api_interface,
                        "dependencies": registration.metadata.dependencies,
                        "priority": registration.metadata.priority,
                        "enhancement_bridge": True
                    }
                }
                
                bridge_entries.append(bridge_entry)
            
            # Merge with existing index
            updated_index = existing_index + bridge_entries
            
            # Write updated index
            with open(index_path, 'w') as f:
                json.dump(updated_index, f, indent=2)
            
            logger.info(f"Updated index.json with {len(bridge_entries)} bridge entries")
            
        except Exception as e:
            logger.error(f"Failed to update index.json: {e}")
    
    def _get_phase_prefix(self, phase: ProcessingPhase) -> str:
        """Get single letter prefix for phase"""
        phase_prefixes = {
            ProcessingPhase.INGESTION_PREPARATION: "I",
            ProcessingPhase.CONTEXT_CONSTRUCTION: "X", 
            ProcessingPhase.KNOWLEDGE_EXTRACTION: "K",
            ProcessingPhase.ANALYSIS_NLP: "A",
            ProcessingPhase.CLASSIFICATION_EVALUATION: "L",
            ProcessingPhase.ORCHESTRATION_CONTROL: "O",
            ProcessingPhase.SEARCH_RETRIEVAL: "R",
            ProcessingPhase.SYNTHESIS_OUTPUT: "S",
            ProcessingPhase.AGGREGATION_REPORTING: "G",
            ProcessingPhase.INTEGRATION_STORAGE: "T"
        }
        return phase_prefixes.get(phase, "B")  # B for Bridge if unknown
    
    def export_bridge_registry(self, output_path: Path) -> None:
        """Export bridge registry to JSON file"""
        try:
            registry_data = {
                "bridges": {
                    bridge_id: {
                        "metadata": asdict(registration.metadata),
                        "has_class": registration.bridge_class is not None,
                        "has_process_function": registration.process_function is not None,
                        "has_factory_function": registration.factory_function is not None
                    }
                    for bridge_id, registration in self.registered_bridges.items()
                },
                "phase_mappings": {
                    phase.value: bridge_ids 
                    for phase, bridge_ids in self.phase_bridges.items()
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(registry_data, f, indent=2, default=str)
            
            logger.info(f"Exported bridge registry to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export bridge registry: {e}")
    
    def get_registry_info(self) -> Dict[str, Any]:
        """Get information about the bridge registry"""
        return {
            "total_bridges": len(self.registered_bridges),
            "enabled_bridges": sum(1 for r in self.registered_bridges.values() 
                                 if r.metadata.enabled),
            "phases_covered": len([phase for phase, bridges in self.phase_bridges.items() 
                                 if bridges]),
            "bridge_summary": {
                bridge_id: {
                    "phase": registration.metadata.phase.value,
                    "enabled": registration.metadata.enabled,
                    "priority": registration.metadata.priority
                }
                for bridge_id, registration in self.registered_bridges.items()
            }
        }


# Global bridge registry instance
_bridge_registry: Optional[BridgeRegistry] = None


def get_bridge_registry() -> BridgeRegistry:
    """Get the global bridge registry instance"""
    global _bridge_registry
    if _bridge_registry is None:
        _bridge_registry = BridgeRegistry()
        _bridge_registry.discover_bridges()
    return _bridge_registry


def initialize_bridge_system() -> BridgeRegistry:
    """Initialize the bridge system and return registry"""
    registry = get_bridge_registry()
    registry.update_index_json()
    return registry