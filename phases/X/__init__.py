"""
Phase X - Context Construction

Public API for the Context Construction phase.
Only exports the standard process interface and essential data models.

All external access to this phase must go through these public interfaces.
Direct imports of internal modules are prohibited.
"""

from typing import Any, Dict, List, Protocol, runtime_checkable
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ContextConstructionData:
    """Standard data model for context construction output."""
    immutable_context: Dict[str, Any]
    lineage_trace: List[Dict[str, Any]]
    context_adapters: Dict[str, Any]
    metadata: Dict[str, Any]
    construction_timestamp: str


# Legacy alias for backward compatibility
ContextData = ContextConstructionData


@dataclass
class ContextConstructionContext:
    """Standard context model for context construction phase."""
    base_context: Dict[str, Any]
    config: Dict[str, Any]
    upstream_data: Dict[str, Any]


@runtime_checkable
class ContextConstructionProcessor(Protocol):
    """Standard protocol for context constructors."""
    
    def process(self, data: Any, context: ContextConstructionContext) -> ContextConstructionData:
        """Process data through the context construction phase."""
        ...


# Legacy alias for backward compatibility
ContextConstructor = ContextConstructionProcessor


class ContextConstructionPhase:
    """Main processing interface for Context Construction phase."""
    
    @staticmethod
    def process(data: Any, context: ContextConstructionContext) -> ContextConstructionData:
        """
        Standard process interface for Phase X.
        
        Args:
            data: Input data for context construction
            context: Processing context with configuration
            
        Returns:
            Constructed context data
        """
        # Import canonical implementation only when needed
        try:
            from canonical_flow.X_context_construction.immutable_context import ImmutableContext
            from canonical_flow.X_context_construction.lineage_tracker import LineageTracker
            from canonical_flow.X_context_construction.context_adapter import ContextAdapter
        except ImportError:
            # Fallback implementation if canonical modules not available
            return _fallback_context_construction(data, context)
        
        # Build immutable context
        immutable_ctx = ImmutableContext.from_data(data, context.base_context)
        
        # Track lineage
        lineage_tracker = LineageTracker()
        lineage_trace = lineage_tracker.track_construction(data, context)
        
        # Adapt context
        adapter = ContextAdapter(context.config)
        adapted_context = adapter.adapt(immutable_ctx)
        
        return ContextConstructionData(
            immutable_context=adapted_context,
            lineage_trace=lineage_trace,
            context_adapters=adapter.get_adapters(),
            metadata={
                'construction_method': 'canonical',
                'data_keys': list(data.keys()) if isinstance(data, dict) else [],
                'context_size': len(str(adapted_context))
            },
            construction_timestamp=str(context.config.get('timestamp', 'unknown'))
        )


def _fallback_context_construction(data: Any, context: ContextConstructionContext) -> ContextConstructionData:
    """Fallback context construction when canonical modules unavailable."""
    return ContextConstructionData(
        immutable_context={
            'data': data,
            'base_context': context.base_context,
            'fallback_mode': True
        },
        lineage_trace=[{
            'step': 'fallback_construction',
            'timestamp': str(context.config.get('timestamp', 'unknown')),
            'data_type': type(data).__name__
        }],
        context_adapters={'fallback': True},
        metadata={
            'construction_method': 'fallback',
            'warning': 'Canonical context construction modules not available'
        },
        construction_timestamp=str(context.config.get('timestamp', 'unknown'))
    )


__all__ = [
    'ContextConstructionData',
    'ContextConstructionContext',
    'ContextConstructionProcessor', 
    'ContextConstructionPhase'
]