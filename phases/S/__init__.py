"""
Phase S - Synthesis and Output

Public API for the Synthesis and Output phase.
Only exports the standard process interface and essential data models.

All external access to this phase must go through these public interfaces.
Direct imports of internal modules are prohibited.
"""

from typing import Any, Dict, List, Protocol, runtime_checkable
from dataclasses import dataclass


@dataclass
class SynthesisData:
    """Standard data model for synthesis and output."""
    synthesized_content: Dict[str, Any]
    formatted_outputs: List[Dict[str, Any]]
    final_reports: Dict[str, Any]
    quality_scores: Dict[str, float]
    output_artifacts: List[str]
    metadata: Dict[str, Any]


@dataclass
class SynthesisContext:
    """Standard context model for synthesis phase."""
    synthesis_config: Dict[str, Any]
    output_settings: Dict[str, Any]
    upstream_data: Dict[str, Any]


@runtime_checkable
class SynthesisProcessor(Protocol):
    """Standard protocol for synthesis processors."""
    
    def process(self, data: Any, context: SynthesisContext) -> SynthesisData:
        """Process data through the synthesis and output phase."""
        ...


class SynthesisPhase:
    """Main processing interface for Synthesis and Output phase."""
    
    @staticmethod
    def process(data: Any, context: SynthesisContext) -> SynthesisData:
        """
        Standard process interface for Phase S.
        
        Args:
            data: Input data for synthesis
            context: Processing context with configuration
            
        Returns:
            Synthesis and output results
        """
        # Import canonical implementation only when needed
        try:
            from canonical_flow.S_synthesis_output import synthesis_orchestrator
        except ImportError:
            return _fallback_synthesis(data, context)
        
        # Execute synthesis
        orchestrator = synthesis_orchestrator.SynthesisOrchestrator(
            config=context.synthesis_config,
            output_settings=context.output_settings
        )
        
        results = orchestrator.synthesize_and_output(
            input_data=data,
            upstream_data=context.upstream_data
        )
        
        return SynthesisData(
            synthesized_content=results.get('content', {}),
            formatted_outputs=results.get('outputs', []),
            final_reports=results.get('reports', {}),
            quality_scores=results.get('quality', {}),
            output_artifacts=results.get('artifacts', []),
            metadata=results.get('metadata', {})
        )


def _fallback_synthesis(data: Any, context: SynthesisContext) -> SynthesisData:
    """Fallback synthesis when canonical modules unavailable."""
    return SynthesisData(
        synthesized_content={},
        formatted_outputs=[],
        final_reports={},
        quality_scores={},
        output_artifacts=[],
        metadata={
            'synthesis_method': 'fallback',
            'warning': 'Canonical synthesis modules not available'
        }
    )


__all__ = [
    'SynthesisData',
    'SynthesisContext',
    'SynthesisProcessor',
    'SynthesisPhase'
]