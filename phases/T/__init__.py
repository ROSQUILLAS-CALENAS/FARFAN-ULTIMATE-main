"""
Phase T - Integration and Storage

Public API for the Integration and Storage phase.
Only exports the standard process interface and essential data models.

All external access to this phase must go through these public interfaces.
Direct imports of internal modules are prohibited.
"""

from typing import Any, Dict, List, Protocol, runtime_checkable
from dataclasses import dataclass


@dataclass
class IntegrationData:
    """Standard data model for integration and storage output."""
    stored_artifacts: List[str]
    integration_results: Dict[str, Any]
    storage_metadata: Dict[str, Any]
    data_lineage: Dict[str, Any]
    validation_status: Dict[str, bool]
    metadata: Dict[str, Any]


@dataclass
class IntegrationContext:
    """Standard context model for integration phase."""
    storage_config: Dict[str, Any]
    integration_settings: Dict[str, Any]
    upstream_data: Dict[str, Any]


@runtime_checkable
class IntegrationProcessor(Protocol):
    """Standard protocol for integration processors."""
    
    def process(self, data: Any, context: IntegrationContext) -> IntegrationData:
        """Process data through the integration and storage phase."""
        ...


class IntegrationPhase:
    """Main processing interface for Integration and Storage phase."""
    
    @staticmethod
    def process(data: Any, context: IntegrationContext) -> IntegrationData:
        """
        Standard process interface for Phase T.
        
        Args:
            data: Input data for integration and storage
            context: Processing context with configuration
            
        Returns:
            Integration and storage results
        """
        # Import canonical implementation only when needed
        try:
            from canonical_flow.T_integration_storage import integration_orchestrator
        except ImportError:
            return _fallback_integration(data, context)
        
        # Execute integration
        orchestrator = integration_orchestrator.IntegrationOrchestrator(
            config=context.storage_config,
            integration_settings=context.integration_settings
        )
        
        results = orchestrator.integrate_and_store(
            input_data=data,
            upstream_data=context.upstream_data
        )
        
        return IntegrationData(
            stored_artifacts=results.get('artifacts', []),
            integration_results=results.get('integration', {}),
            storage_metadata=results.get('storage_metadata', {}),
            data_lineage=results.get('lineage', {}),
            validation_status=results.get('validation', {}),
            metadata=results.get('metadata', {})
        )


def _fallback_integration(data: Any, context: IntegrationContext) -> IntegrationData:
    """Fallback integration when canonical modules unavailable."""
    return IntegrationData(
        stored_artifacts=[],
        integration_results={},
        storage_metadata={},
        data_lineage={},
        validation_status={},
        metadata={
            'integration_method': 'fallback',
            'warning': 'Canonical integration modules not available'
        }
    )


__all__ = [
    'IntegrationData',
    'IntegrationContext',
    'IntegrationProcessor',
    'IntegrationPhase'
]