"""
Phase G - Aggregation and Reporting

Public API for the Aggregation and Reporting phase.
Only exports the standard process interface and essential data models.

All external access to this phase must go through these public interfaces.
Direct imports of internal modules are prohibited.
"""

from typing import Any, Dict, List, Protocol, runtime_checkable
from dataclasses import dataclass


@dataclass
class AggregationData:
    """Standard data model for aggregation and reporting output."""
    aggregated_results: Dict[str, Any]
    summary_statistics: Dict[str, float]
    reports: List[Dict[str, Any]]
    visualizations: Dict[str, Any]
    consolidated_metrics: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class AggregationContext:
    """Standard context model for aggregation phase."""
    aggregation_config: Dict[str, Any]
    reporting_settings: Dict[str, Any]
    upstream_data: Dict[str, Any]


@runtime_checkable
class AggregationProcessor(Protocol):
    """Standard protocol for aggregation processors."""
    
    def process(self, data: Any, context: AggregationContext) -> AggregationData:
        """Process data through the aggregation and reporting phase."""
        ...


class AggregationPhase:
    """Main processing interface for Aggregation and Reporting phase."""
    
    @staticmethod
    def process(data: Any, context: AggregationContext) -> AggregationData:
        """
        Standard process interface for Phase G.
        
        Args:
            data: Input data for aggregation
            context: Processing context with configuration
            
        Returns:
            Aggregation and reporting results
        """
        # Import canonical implementation only when needed
        try:
            from canonical_flow.G_aggregation_reporting import aggregation_orchestrator
        except ImportError:
            return _fallback_aggregation(data, context)
        
        # Execute aggregation
        orchestrator = aggregation_orchestrator.AggregationOrchestrator(
            config=context.aggregation_config,
            reporting_settings=context.reporting_settings
        )
        
        results = orchestrator.aggregate_and_report(
            input_data=data,
            upstream_data=context.upstream_data
        )
        
        return AggregationData(
            aggregated_results=results.get('aggregated', {}),
            summary_statistics=results.get('statistics', {}),
            reports=results.get('reports', []),
            visualizations=results.get('visualizations', {}),
            consolidated_metrics=results.get('metrics', {}),
            metadata=results.get('metadata', {})
        )


def _fallback_aggregation(data: Any, context: AggregationContext) -> AggregationData:
    """Fallback aggregation when canonical modules unavailable."""
    return AggregationData(
        aggregated_results={},
        summary_statistics={},
        reports=[],
        visualizations={},
        consolidated_metrics={},
        metadata={
            'aggregation_method': 'fallback',
            'warning': 'Canonical aggregation modules not available'
        }
    )


__all__ = [
    'AggregationData',
    'AggregationContext',
    'AggregationProcessor',
    'AggregationPhase'
]