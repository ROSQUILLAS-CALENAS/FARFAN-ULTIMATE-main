"""
Phase A - Analysis and NLP

Public API for the Analysis and NLP phase.
Only exports the standard process interface and essential data models.

All external access to this phase must go through these public interfaces.
Direct imports of internal modules are prohibited.
"""

from typing import Any, Dict, List, Protocol, runtime_checkable
from dataclasses import dataclass


@dataclass
class AnalysisData:
    """Standard data model for analysis and NLP output."""
    sentiment_analysis: Dict[str, Any]
    semantic_vectors: Dict[str, List[float]]
    language_features: Dict[str, Any]
    nlp_annotations: List[Dict[str, Any]]
    analysis_confidence: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class AnalysisContext:
    """Standard context model for analysis phase."""
    analysis_config: Dict[str, Any]
    model_settings: Dict[str, Any]
    upstream_data: Dict[str, Any]


@runtime_checkable
class AnalysisProcessor(Protocol):
    """Standard protocol for analysis processors."""
    
    def process(self, data: Any, context: AnalysisContext) -> AnalysisData:
        """Process data through the analysis and NLP phase."""
        ...


class AnalysisPhase:
    """Main processing interface for Analysis and NLP phase."""
    
    @staticmethod
    def process(data: Any, context: AnalysisContext) -> AnalysisData:
        """
        Standard process interface for Phase A.
        
        Args:
            data: Input data for analysis
            context: Processing context with configuration
            
        Returns:
            Analyzed data with NLP features
        """
        # Import canonical implementation only when needed
        try:
            from canonical_flow.A_analysis_nlp import analysis_orchestrator
        except ImportError:
            return _fallback_analysis(data, context)
        
        # Execute analysis
        orchestrator = analysis_orchestrator.AnalysisOrchestrator(
            config=context.analysis_config,
            model_settings=context.model_settings
        )
        
        results = orchestrator.analyze(
            input_data=data,
            upstream_data=context.upstream_data
        )
        
        return AnalysisData(
            sentiment_analysis=results.get('sentiment', {}),
            semantic_vectors=results.get('vectors', {}),
            language_features=results.get('features', {}),
            nlp_annotations=results.get('annotations', []),
            analysis_confidence=results.get('confidence', {}),
            metadata=results.get('metadata', {})
        )


def _fallback_analysis(data: Any, context: AnalysisContext) -> AnalysisData:
    """Fallback analysis when canonical modules unavailable."""
    return AnalysisData(
        sentiment_analysis={},
        semantic_vectors={},
        language_features={},
        nlp_annotations=[],
        analysis_confidence={},
        metadata={
            'analysis_method': 'fallback',
            'warning': 'Canonical analysis modules not available'
        }
    )


__all__ = [
    'AnalysisData',
    'AnalysisContext',
    'AnalysisProcessor',
    'AnalysisPhase'
]