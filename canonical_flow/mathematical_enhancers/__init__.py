"""
Mathematical Enhancers Package

Canonical organization of all mathematical enhancement modules for the EGW pipeline.
This package contains the complete 12-stage mathematical enhancement flow from 
ingestion through integration.

Stages:
- Stage 01: Ingestion Enhancement
- Stage 02: Context Enhancement
- Stage 03: Knowledge Enhancement
- Stage 04: Analysis Enhancement
- Stage 05: Scoring Enhancement
- Stage 06: Retrieval Enhancement
- Stage 07: Orchestration Enhancement
- Stage 08: Classification Enhancement (TODO)
- Stage 09: Synthesis Enhancement (TODO)
- Stage 10: Output Enhancement (TODO)  
- Stage 11: Aggregation Enhancement
- Stage 12: Integration Enhancement

Additional modules:
- mathematical_pipeline_coordinator: Orchestrates all stages
- mathematical_compatibility_matrix: Manages stage compatibility
"""

from .mathematical_pipeline_coordinator import (
    MathematicalPipelineCoordinator,
    create_mathematical_pipeline_coordinator,
    StageResult,
    ValidationResult
)

from .mathematical_compatibility_matrix import (
    MathematicalCompatibilityMatrix,
    CompatibilityResult
)

# Import all mathematical enhancer stages
try:
    from .math_stage01_ingestion_enhancer import MathStage1IngestionEnhancer
except ImportError:
    MathStage1IngestionEnhancer = None
try:
    from .math_stage02_context_enhancer import (
        MathematicalStage2ContextEnhancer as MathStage2ContextEnhancer,
        TopologicalQuantumFieldTheoryEnhancer,
        WilsonLoopOperator,
        KnotInvariant,
        ChernSimonsAction,
        TQFTContextFunctor
    )  
except ImportError:
    MathStage2ContextEnhancer = None
    TopologicalQuantumFieldTheoryEnhancer = None
    WilsonLoopOperator = None
    KnotInvariant = None
    ChernSimonsAction = None
    TQFTContextFunctor = None

try:
    from .math_stage03_knowledge_enhancer import MathStage3KnowledgeEnhancer
except ImportError:
    MathStage3KnowledgeEnhancer = None

try:
    from .math_stage04_analysis_enhancer import MathStage4AnalysisEnhancer
except ImportError:
    MathStage4AnalysisEnhancer = None

try:
    from .math_stage05_scoring_enhancer import MathStage5ScoringEnhancer
except ImportError:
    MathStage5ScoringEnhancer = None

try:
    from .math_stage06_retrieval_enhancer import MathStage6RetrievalEnhancer
except ImportError:
    MathStage6RetrievalEnhancer = None

try:
    from .math_stage07_orchestration_enhancer import MathStage7OrchestrationEnhancer
except ImportError:
    MathStage7OrchestrationEnhancer = None

try:
    from .math_stage11_aggregation_enhancer import MathStage11AggregationEnhancer
except ImportError:
    MathStage11AggregationEnhancer = None

try:
    from .math_stage12_integration_enhancer import MathStage12IntegrationEnhancer
except ImportError:
    MathStage12IntegrationEnhancer = None

__all__ = [
    # Core coordination
    'MathematicalPipelineCoordinator',
    'create_mathematical_pipeline_coordinator',
    'StageResult',
    'ValidationResult',
    'MathematicalCompatibilityMatrix',
    'CompatibilityResult',
    
    # All mathematical enhancer stages
    'MathStage1IngestionEnhancer',
    'MathStage2ContextEnhancer',
    'MathStage3KnowledgeEnhancer',
    'MathStage4AnalysisEnhancer',
    'MathStage5ScoringEnhancer',
    'MathStage6RetrievalEnhancer',
    'MathStage7OrchestrationEnhancer',
    'MathStage11AggregationEnhancer',
    'MathStage12IntegrationEnhancer',
    
    # Topological Quantum Field Theory components
    'TopologicalQuantumFieldTheoryEnhancer',
    'WilsonLoopOperator',
    'KnotInvariant',
    'ChernSimonsAction',
    'TQFTContextFunctor',
]