"""
Phase O - Orchestration and Control

Public API for the Orchestration and Control phase.
Only exports the standard process interface and essential data models.

All external access to this phase must go through these public interfaces.
Direct imports of internal modules are prohibited.
"""

from typing import Any, Dict, List, Protocol, runtime_checkable
from dataclasses import dataclass


@dataclass
class OrchestrationData:
    """Standard data model for orchestration and control output."""
    execution_plan: Dict[str, Any]
    control_decisions: List[Dict[str, Any]]
    flow_state: Dict[str, Any]
    scheduling_info: Dict[str, Any]
    resource_allocation: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class OrchestrationContext:
    """Standard context model for orchestration phase."""
    orchestration_config: Dict[str, Any]
    control_policies: Dict[str, Any]
    upstream_data: Dict[str, Any]


@runtime_checkable
class OrchestrationProcessor(Protocol):
    """Standard protocol for orchestration processors."""
    
    def process(self, data: Any, context: OrchestrationContext) -> OrchestrationData:
        """Process data through the orchestration and control phase."""
        ...


class OrchestrationPhase:
    """Main processing interface for Orchestration and Control phase."""
    
    @staticmethod
    def process(data: Any, context: OrchestrationContext) -> OrchestrationData:
        """
        Standard process interface for Phase O.
        
        Args:
            data: Input data for orchestration
            context: Processing context with configuration
            
        Returns:
            Orchestration and control results
        """
        # Import canonical implementation only when needed
        try:
            from canonical_flow.O_orchestration_control import orchestration_controller
        except ImportError:
            return _fallback_orchestration(data, context)
        
        # Execute orchestration
        controller = orchestration_controller.OrchestrationController(
            config=context.orchestration_config,
            policies=context.control_policies
        )
        
        results = controller.orchestrate(
            input_data=data,
            upstream_data=context.upstream_data
        )
        
        return OrchestrationData(
            execution_plan=results.get('plan', {}),
            control_decisions=results.get('decisions', []),
            flow_state=results.get('state', {}),
            scheduling_info=results.get('scheduling', {}),
            resource_allocation=results.get('resources', {}),
            metadata=results.get('metadata', {})
        )


def _fallback_orchestration(data: Any, context: OrchestrationContext) -> OrchestrationData:
    """Fallback orchestration when canonical modules unavailable."""
    return OrchestrationData(
        execution_plan={},
        control_decisions=[],
        flow_state={},
        scheduling_info={},
        resource_allocation={},
        metadata={
            'orchestration_method': 'fallback',
            'warning': 'Canonical orchestration modules not available'
        }
    )


__all__ = [
    'OrchestrationData',
    'OrchestrationContext',
    'OrchestrationProcessor',
    'OrchestrationPhase'
]