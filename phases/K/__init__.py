"""
Phase K - Knowledge Extraction

Public API for the Knowledge Extraction phase.
Only exports the standard process interface and essential data models.

All external access to this phase must go through these public interfaces.
Direct imports of internal modules are prohibited.
"""

from typing import Any, Dict, List, Protocol, runtime_checkable
from dataclasses import dataclass


@dataclass
class KnowledgeExtractionData:
    """Standard data model for knowledge extraction output."""
    extracted_entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    concepts: Dict[str, Any]
    knowledge_graph: Dict[str, Any]
    extraction_confidence: Dict[str, float]
    metadata: Dict[str, Any]


# Legacy alias for backward compatibility
KnowledgeData = KnowledgeExtractionData


@dataclass
class KnowledgeExtractionContext:
    """Standard context model for knowledge extraction phase."""
    extraction_config: Dict[str, Any]
    upstream_data: Dict[str, Any]
    model_configs: Dict[str, Any]


@runtime_checkable
class KnowledgeExtractionProcessor(Protocol):
    """Standard protocol for knowledge extractors."""
    
    def process(self, data: Any, context: KnowledgeExtractionContext) -> KnowledgeExtractionData:
        """Process data through the knowledge extraction phase."""
        ...


# Legacy alias for backward compatibility  
KnowledgeExtractor = KnowledgeExtractionProcessor


class KnowledgeExtractionPhase:
    """Main processing interface for Knowledge Extraction phase."""
    
    @staticmethod
    def process(data: Any, context: KnowledgeExtractionContext) -> KnowledgeExtractionData:
        """
        Standard process interface for Phase K.
        
        Args:
            data: Input data for knowledge extraction
            context: Processing context with configuration
            
        Returns:
            Extracted knowledge data
        """
        # Import canonical implementation only when needed
        try:
            from canonical_flow.K_knowledge_extraction import knowledge_orchestrator
        except ImportError:
            return _fallback_knowledge_extraction(data, context)
        
        # Execute knowledge extraction
        orchestrator = knowledge_orchestrator.KnowledgeExtractionOrchestrator(
            config=context.extraction_config
        )
        
        results = orchestrator.extract_knowledge(
            input_data=data,
            model_configs=context.model_configs
        )
        
        return KnowledgeExtractionData(
            extracted_entities=results.get('entities', []),
            relationships=results.get('relationships', []),
            concepts=results.get('concepts', {}),
            knowledge_graph=results.get('knowledge_graph', {}),
            extraction_confidence=results.get('confidence_scores', {}),
            metadata=results.get('extraction_metadata', {})
        )


def _fallback_knowledge_extraction(data: Any, context: KnowledgeExtractionContext) -> KnowledgeExtractionData:
    """Fallback knowledge extraction when canonical modules unavailable."""
    return KnowledgeExtractionData(
        extracted_entities=[],
        relationships=[],
        concepts={},
        knowledge_graph={},
        extraction_confidence={},
        metadata={
            'extraction_method': 'fallback',
            'warning': 'Canonical knowledge extraction modules not available'
        }
    )


__all__ = [
    'KnowledgeExtractionData',
    'KnowledgeExtractionContext',
    'KnowledgeExtractionProcessor',
    'KnowledgeExtractionPhase'
]