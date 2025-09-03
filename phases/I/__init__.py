"""
Phase I - Ingestion and Preparation

Public API for the Ingestion and Preparation phase.
Only exports the standard process interface and essential data models.

All external access to this phase must go through these public interfaces.
Direct imports of internal modules are prohibited.
"""

from typing import Any, Dict, Protocol, runtime_checkable
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class IngestionData:
    """Standard data model for ingestion phase output."""
    text_extracts: Dict[str, str]
    document_features: Dict[str, Any]
    metadata: Dict[str, Any]
    artifacts_path: str
    processing_timestamp: str


@dataclass
class IngestionContext:
    """Standard context model for ingestion phase."""
    data_path: str
    config: Dict[str, Any]
    component_states: Dict[str, str]


@runtime_checkable
class IngestionProcessor(Protocol):
    """Standard protocol for ingestion processors."""
    
    def process(self, data: Any, context: IngestionContext) -> IngestionData:
        """Process data through the ingestion phase."""
        ...


class IngestionPhase:
    """Main processing interface for Ingestion phase."""
    
    @staticmethod
    def process(data: Any, context: IngestionContext) -> IngestionData:
        """
        Standard process interface for Phase I.
        
        Args:
            data: Input data for ingestion processing
            context: Processing context with configuration
            
        Returns:
            Processed ingestion data
        """
        # Import canonical implementation only when needed
        from canonical_flow.I_ingestion_preparation.ingestion_orchestrator import (
            IngestionPreparationOrchestrator
        )
        
        orchestrator = IngestionPreparationOrchestrator(
            base_data_path=context.data_path,
            enable_strict_mode=context.config.get('strict_mode', True)
        )
        
        # Execute pipeline
        results = orchestrator.execute_full_pipeline(
            input_data={'base_data': data},
            component_configs=context.config.get('component_configs')
        )
        
        # Transform to standard output format
        return IngestionData(
            text_extracts=_extract_text_data(results),
            document_features=_extract_features(results),
            metadata=results.get('execution_summary', {}),
            artifacts_path=context.data_path,
            processing_timestamp=str(results.get('timestamp', 'unknown'))
        )


def _extract_text_data(results: Dict[str, Any]) -> Dict[str, str]:
    """Extract text data from orchestrator results."""
    text_data = {}
    component_results = results.get('component_results', {})
    
    if '01I' in component_results:
        pdf_results = component_results['01I'].get('results', [])
        for result in pdf_results:
            if 'output_file' in result:
                # Would load and extract text here
                text_data[result['pdf_file']] = f"Extracted from {result['output_file']}"
    
    return text_data


def _extract_features(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract features from orchestrator results."""
    features = {}
    component_results = results.get('component_results', {})
    
    for component_id, result in component_results.items():
        if result.get('success') and 'results' in result:
            features[component_id] = result['results']
    
    return features


__all__ = [
    'IngestionData',
    'IngestionContext', 
    'IngestionProcessor',
    'IngestionPhase'
]