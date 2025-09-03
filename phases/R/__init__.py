"""
Phase R - Search and Retrieval

Public API for the Search and Retrieval phase.
Only exports the standard process interface and essential data models.

All external access to this phase must go through these public interfaces.
Direct imports of internal modules are prohibited.
"""

from typing import Any, Dict, List, Protocol, runtime_checkable
from dataclasses import dataclass


@dataclass
class RetrievalData:
    """Standard data model for search and retrieval output."""
    search_results: List[Dict[str, Any]]
    relevance_scores: Dict[str, float]
    retrieval_metrics: Dict[str, Any]
    query_expansions: List[str]
    ranked_documents: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class RetrievalContext:
    """Standard context model for retrieval phase."""
    query_config: Dict[str, Any]
    retrieval_settings: Dict[str, Any]
    upstream_data: Dict[str, Any]


@runtime_checkable
class RetrievalProcessor(Protocol):
    """Standard protocol for retrieval processors."""
    
    def process(self, data: Any, context: RetrievalContext) -> RetrievalData:
        """Process data through the search and retrieval phase."""
        ...


class RetrievalPhase:
    """Main processing interface for Search and Retrieval phase."""
    
    @staticmethod
    def process(data: Any, context: RetrievalContext) -> RetrievalData:
        """
        Standard process interface for Phase R.
        
        Args:
            data: Input data for retrieval (queries, documents)
            context: Processing context with configuration
            
        Returns:
            Search and retrieval results
        """
        # Import canonical implementation only when needed
        try:
            from canonical_flow.R_search_retrieval import retrieval_orchestrator
        except ImportError:
            return _fallback_retrieval(data, context)
        
        # Execute retrieval
        orchestrator = retrieval_orchestrator.RetrievalOrchestrator(
            config=context.query_config,
            retrieval_settings=context.retrieval_settings
        )
        
        results = orchestrator.search_and_retrieve(
            queries=data,
            upstream_data=context.upstream_data
        )
        
        return RetrievalData(
            search_results=results.get('results', []),
            relevance_scores=results.get('scores', {}),
            retrieval_metrics=results.get('metrics', {}),
            query_expansions=results.get('expansions', []),
            ranked_documents=results.get('ranked_docs', []),
            metadata=results.get('metadata', {})
        )


def _fallback_retrieval(data: Any, context: RetrievalContext) -> RetrievalData:
    """Fallback retrieval when canonical modules unavailable."""
    return RetrievalData(
        search_results=[],
        relevance_scores={},
        retrieval_metrics={},
        query_expansions=[],
        ranked_documents=[],
        metadata={
            'retrieval_method': 'fallback',
            'warning': 'Canonical retrieval modules not available'
        }
    )


__all__ = [
    'RetrievalData',
    'RetrievalContext',
    'RetrievalProcessor',
    'RetrievalPhase'
]