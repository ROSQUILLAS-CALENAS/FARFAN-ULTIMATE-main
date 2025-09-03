"""
Mathematical Enhancer API Interfaces

Defines abstract base classes and protocols for mathematical enhancers to enable
dependency inversion and loose coupling in the canonical pipeline flow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum


class ProcessingPhase(Enum):
    """Canonical processing phases"""
    INGESTION_PREPARATION = "ingestion_preparation"
    CONTEXT_CONSTRUCTION = "context_construction"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    ANALYSIS_NLP = "analysis_nlp"
    CLASSIFICATION_EVALUATION = "classification_evaluation"
    ORCHESTRATION_CONTROL = "orchestration_control"
    SEARCH_RETRIEVAL = "search_retrieval"
    SYNTHESIS_OUTPUT = "synthesis_output"
    AGGREGATION_REPORTING = "aggregation_reporting"
    INTEGRATION_STORAGE = "integration_storage"


@dataclass
class ProcessingContext:
    """Standard context object passed between pipeline stages"""
    stage_id: str
    phase: ProcessingPhase
    pipeline_state: Dict[str, Any]
    metadata: Dict[str, Any]
    upstream_results: Dict[str, Any]


@dataclass
class ProcessingResult:
    """Standard result object returned by processing stages"""
    success: bool
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    error_message: Optional[str] = None


@runtime_checkable
class MathematicalEnhancerAPI(Protocol):
    """Protocol defining the mathematical enhancer API contract"""
    
    def enhance(self, data: Dict[str, Any], context: ProcessingContext) -> Dict[str, Any]:
        """
        Apply mathematical enhancement to input data
        
        Args:
            data: Input data to enhance
            context: Processing context with pipeline state
            
        Returns:
            Enhanced data dictionary
        """
        ...
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data format and content"""
        ...
    
    def get_enhancement_metadata(self) -> Dict[str, Any]:
        """Get metadata about enhancement capabilities"""
        ...


class AbstractMathematicalEnhancer(ABC):
    """Abstract base class for mathematical enhancers"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False
    
    @abstractmethod
    def enhance(self, data: Dict[str, Any], context: ProcessingContext) -> Dict[str, Any]:
        """Apply mathematical enhancement to input data"""
        pass
    
    @abstractmethod
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data format and content"""
        pass
    
    def get_enhancement_metadata(self) -> Dict[str, Any]:
        """Get metadata about enhancement capabilities"""
        return {
            "enhancer_type": self.__class__.__name__,
            "config": self.config,
            "initialized": self._initialized
        }
    
    def initialize(self) -> None:
        """Initialize the enhancer"""
        self._initialized = True


@runtime_checkable
class IngestionEnhancerAPI(MathematicalEnhancerAPI, Protocol):
    """API for ingestion phase mathematical enhancement"""
    
    def validate_document_structure(self, document_data: Dict[str, Any]) -> bool:
        """Validate document structure integrity"""
        ...
    
    def compute_manifold_metrics(self, text_data: str) -> Dict[str, float]:
        """Compute Riemannian manifold metrics for text"""
        ...


@runtime_checkable
class ContextEnhancerAPI(MathematicalEnhancerAPI, Protocol):
    """API for context construction mathematical enhancement"""
    
    def build_topological_context(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build topological quantum field theory context"""
        ...
    
    def preserve_causality(self, causal_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Preserve causal relationships in context"""
        ...


@runtime_checkable
class KnowledgeEnhancerAPI(MathematicalEnhancerAPI, Protocol):
    """API for knowledge extraction mathematical enhancement"""
    
    def enhance_embeddings(self, embeddings: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mathematical enhancement to embeddings"""
        ...
    
    def build_knowledge_graph(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Build enhanced knowledge graph structures"""
        ...


@runtime_checkable
class AnalysisEnhancerAPI(MathematicalEnhancerAPI, Protocol):
    """API for analysis NLP mathematical enhancement"""
    
    def compute_information_metrics(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute information-theoretic metrics"""
        ...
    
    def optimize_spectral_transitions(self, transition_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize using spectral analysis"""
        ...


@runtime_checkable
class ScoringEnhancerAPI(MathematicalEnhancerAPI, Protocol):
    """API for classification/evaluation mathematical enhancement"""
    
    def enhance_scoring_metrics(self, scores: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mathematical enhancement to scoring"""
        ...
    
    def validate_score_consistency(self, score_data: Dict[str, Any]) -> bool:
        """Validate scoring consistency"""
        ...


@runtime_checkable
class OrchestrationEnhancerAPI(MathematicalEnhancerAPI, Protocol):
    """API for orchestration control mathematical enhancement"""
    
    def optimize_control_flow(self, flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize orchestration control flow"""
        ...
    
    def balance_computational_load(self, load_data: Dict[str, Any]) -> Dict[str, Any]:
        """Balance computational loads mathematically"""
        ...


@runtime_checkable
class RetrievalEnhancerAPI(MathematicalEnhancerAPI, Protocol):
    """API for search/retrieval mathematical enhancement"""
    
    def enhance_similarity_metrics(self, similarity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance similarity calculations"""
        ...
    
    def optimize_retrieval_ranking(self, ranking_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize retrieval ranking mathematically"""
        ...


@runtime_checkable
class SynthesisEnhancerAPI(MathematicalEnhancerAPI, Protocol):
    """API for synthesis/output mathematical enhancement"""
    
    def enhance_output_coherence(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance output coherence mathematically"""
        ...


@runtime_checkable
class AggregationEnhancerAPI(MathematicalEnhancerAPI, Protocol):
    """API for aggregation/reporting mathematical enhancement"""
    
    def aggregate_multi_dimensional_data(self, aggregation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mathematical aggregation techniques"""
        ...


@runtime_checkable
class IntegrationEnhancerAPI(MathematicalEnhancerAPI, Protocol):
    """API for integration/storage mathematical enhancement"""
    
    def ensure_integration_consistency(self, integration_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure mathematical consistency in integration"""
        ...