"""
Dependency-Inverted Bridge Module for Retrieval Enhancement

Provides a standard process(data, context) -> Dict[str, Any] interface while delegating
mathematical enhancement work to injected enhancer instances. Imports only from
mathematical enhancer APIs to avoid direct dependencies on implementations.
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
import logging

# Only import from API interfaces, not concrete implementations
from canonical_flow.mathematical_enhancers.api_interfaces import (
    RetrievalEnhancerAPI,
    ProcessingContext,
    ProcessingPhase,
    ProcessingResult
)

if TYPE_CHECKING:
    # Type hints for development, but not runtime dependencies
    from canonical_flow.mathematical_enhancers.retrieval_enhancer import MathStage6RetrievalEnhancer

logger = logging.getLogger(__name__)


class RetrievalEnhancementBridge:
    """
    Bridge module for search/retrieval phase mathematical enhancement.
    
    Delegates to injected enhancer instances following the dependency inversion principle.
    Exposes standard process interface while maintaining loose coupling.
    """
    
    def __init__(self, enhancer: Optional[RetrievalEnhancerAPI] = None):
        """
        Initialize bridge with optional enhancer injection.
        
        Args:
            enhancer: Mathematical enhancer implementing RetrievalEnhancerAPI
        """
        self._enhancer: Optional[RetrievalEnhancerAPI] = enhancer
        self._phase = ProcessingPhase.SEARCH_RETRIEVAL
        self._initialized = False
    
    def inject_enhancer(self, enhancer: RetrievalEnhancerAPI) -> None:
        """
        Inject mathematical enhancer instance.
        
        Args:
            enhancer: Mathematical enhancer implementing RetrievalEnhancerAPI
        """
        if not hasattr(enhancer, 'enhance') or not hasattr(enhancer, 'enhance_similarity_metrics'):
            raise ValueError("Enhancer must implement RetrievalEnhancerAPI protocol")
        
        self._enhancer = enhancer
        self._initialized = True
        logger.info(f"Injected enhancer: {type(enhancer).__name__}")
    
    def process(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standard process interface for canonical pipeline integration.
        
        Args:
            data: Input data to process
            context: Processing context dictionary
            
        Returns:
            Enhanced data dictionary following standard format
        """
        try:
            # Convert context dict to ProcessingContext if needed
            if isinstance(context, dict):
                processing_context = ProcessingContext(
                    stage_id=context.get('stage_id', 'retrieval_bridge'),
                    phase=self._phase,
                    pipeline_state=context.get('pipeline_state', {}),
                    metadata=context.get('metadata', {}),
                    upstream_results=context.get('upstream_results', {})
                )
            else:
                processing_context = context
            
            # Apply mathematical enhancement if enhancer is injected
            if self._enhancer is not None:
                # Validate input first
                if not self._enhancer.validate_input(data):
                    logger.warning("Input validation failed, proceeding without enhancement")
                    return self._create_passthrough_result(data, processing_context)
                
                # Apply enhancement
                enhanced_data = self._enhancer.enhance(data, processing_context)
                
                # Wrap in standard result format
                return self._create_enhanced_result(enhanced_data, processing_context)
            else:
                logger.info("No enhancer injected, passing through data unchanged")
                return self._create_passthrough_result(data, processing_context)
                
        except Exception as e:
            logger.error(f"Error in retrieval enhancement bridge: {e}")
            return self._create_error_result(data, processing_context, str(e))
    
    def _create_enhanced_result(self, enhanced_data: Dict[str, Any], 
                              context: ProcessingContext) -> Dict[str, Any]:
        """Create standard result format for enhanced data"""
        return {
            'success': True,
            'data': enhanced_data,
            'metadata': {
                'phase': self._phase.value,
                'stage_id': context.stage_id,
                'enhancement_applied': True,
                'enhancer_type': type(self._enhancer).__name__ if self._enhancer else None,
                'enhancement_metadata': self._enhancer.get_enhancement_metadata() if self._enhancer else {}
            },
            'performance_metrics': {
                'processing_time': 0.0,
                'enhancement_ratio': 1.0
            },
            'error_message': None
        }
    
    def _create_passthrough_result(self, data: Dict[str, Any],
                                  context: ProcessingContext) -> Dict[str, Any]:
        """Create standard result format for passthrough data"""
        return {
            'success': True,
            'data': data,
            'metadata': {
                'phase': self._phase.value,
                'stage_id': context.stage_id,
                'enhancement_applied': False,
                'enhancer_type': None,
                'enhancement_metadata': {}
            },
            'performance_metrics': {
                'processing_time': 0.0,
                'enhancement_ratio': 1.0
            },
            'error_message': None
        }
    
    def _create_error_result(self, data: Dict[str, Any], context: ProcessingContext,
                           error_message: str) -> Dict[str, Any]:
        """Create standard result format for error cases"""
        return {
            'success': False,
            'data': data,
            'metadata': {
                'phase': self._phase.value,
                'stage_id': context.stage_id,
                'enhancement_applied': False,
                'enhancer_type': type(self._enhancer).__name__ if self._enhancer else None,
                'enhancement_metadata': {}
            },
            'performance_metrics': {
                'processing_time': 0.0,
                'enhancement_ratio': 1.0
            },
            'error_message': error_message
        }
    
    def get_bridge_info(self) -> Dict[str, Any]:
        """Get information about the bridge and injected enhancer"""
        return {
            'bridge_type': 'RetrievalEnhancementBridge',
            'phase': self._phase.value,
            'enhancer_injected': self._enhancer is not None,
            'enhancer_type': type(self._enhancer).__name__ if self._enhancer else None,
            'initialized': self._initialized
        }


# Factory function for external instantiation
def create_retrieval_enhancement_bridge(enhancer: Optional[RetrievalEnhancerAPI] = None) -> RetrievalEnhancementBridge:
    """
    Factory function to create retrieval enhancement bridge.
    
    Args:
        enhancer: Optional mathematical enhancer to inject
        
    Returns:
        Configured RetrievalEnhancementBridge instance
    """
    bridge = RetrievalEnhancementBridge(enhancer)
    return bridge


# Standard process function for pipeline integration
def process(data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standard process function for canonical pipeline integration.
    
    This function can be called by the pipeline orchestrator without
    needing to instantiate the bridge class directly.
    """
    # Get enhancer from context if provided
    enhancer = context.get('injected_enhancer')
    bridge = create_retrieval_enhancement_bridge(enhancer)
    return bridge.process(data, context)