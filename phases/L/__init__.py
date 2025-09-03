"""
Phase L - Classification and Evaluation

Public API for the Classification and Evaluation phase.
Only exports the standard process interface and essential data models.

All external access to this phase must go through these public interfaces.
Direct imports of internal modules are prohibited.
"""

from typing import Any, Dict, List, Protocol, runtime_checkable
from dataclasses import dataclass


@dataclass
class ClassificationData:
    """Standard data model for classification and evaluation output."""
    classifications: Dict[str, Any]
    evaluation_scores: Dict[str, float]
    confidence_intervals: Dict[str, List[float]]
    quality_metrics: Dict[str, Any]
    validation_results: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class ClassificationContext:
    """Standard context model for classification phase."""
    classification_config: Dict[str, Any]
    evaluation_criteria: Dict[str, Any]
    upstream_data: Dict[str, Any]


@runtime_checkable
class ClassificationProcessor(Protocol):
    """Standard protocol for classification processors."""
    
    def process(self, data: Any, context: ClassificationContext) -> ClassificationData:
        """Process data through the classification and evaluation phase."""
        ...


class ClassificationPhase:
    """Main processing interface for Classification and Evaluation phase."""
    
    @staticmethod
    def process(data: Any, context: ClassificationContext) -> ClassificationData:
        """
        Standard process interface for Phase L.
        
        Args:
            data: Input data for classification
            context: Processing context with configuration
            
        Returns:
            Classification and evaluation results
        """
        # Import canonical implementation only when needed
        try:
            from canonical_flow.L_classification_evaluation import classification_orchestrator
        except ImportError:
            return _fallback_classification(data, context)
        
        # Execute classification
        orchestrator = classification_orchestrator.ClassificationOrchestrator(
            config=context.classification_config,
            evaluation_criteria=context.evaluation_criteria
        )
        
        results = orchestrator.classify_and_evaluate(
            input_data=data,
            upstream_data=context.upstream_data
        )
        
        return ClassificationData(
            classifications=results.get('classifications', {}),
            evaluation_scores=results.get('scores', {}),
            confidence_intervals=results.get('confidence_intervals', {}),
            quality_metrics=results.get('quality_metrics', {}),
            validation_results=results.get('validation', {}),
            metadata=results.get('metadata', {})
        )


def _fallback_classification(data: Any, context: ClassificationContext) -> ClassificationData:
    """Fallback classification when canonical modules unavailable."""
    return ClassificationData(
        classifications={},
        evaluation_scores={},
        confidence_intervals={},
        quality_metrics={},
        validation_results={},
        metadata={
            'classification_method': 'fallback',
            'warning': 'Canonical classification modules not available'
        }
    )


__all__ = [
    'ClassificationData',
    'ClassificationContext',
    'ClassificationProcessor',
    'ClassificationPhase'
]