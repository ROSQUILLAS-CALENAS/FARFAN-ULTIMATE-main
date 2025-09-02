"""
Stage: A_analysis_nlp - Analysis NLP Components with Total Ordering

This module contains all analysis_nlp components that inherit from TotalOrderingBase
for deterministic processing, consistent sorting, and canonical JSON serialization.

All components provide:
- Deterministic ID generation using stable hash functions
- Consistent sorting of all data structures and outputs  
- Canonical JSON serialization with sorted keys
- Reproducible identifiers for all artifacts
- Stable ordering in collections and dictionaries
"""

from .adaptive_analyzer import AdaptiveAnalyzer, process as adaptive_analyzer_process
from .decalogo_question_registry import DecalogoQuestionRegistry, create_decalogo_question_registry
from .question_analyzer import QuestionAnalyzer, process as question_analyzer_process
from .implementacion_mapeo import QuestionDecalogoMapper, process as implementacion_mapeo_process
from .extractor_evidencias_contextual import ExtractorEvidenciasContextual, process as extractor_evidencias_process
from .evidence_processor import EvidenceProcessor, process as evidence_processor_process
from .evidence_validation_model import EvidenceValidationModel, process as evidence_validation_process
from .dnp_alignment_adapter import DNPAlignmentAdapter, process as dnp_alignment_process
from .evaluation_driven_processor import EvaluationDrivenProcessor, process as evaluation_driven_process

__all__ = [
    # Main classes
    "AdaptiveAnalyzer",
    "DecalogoQuestionRegistry",
    "QuestionAnalyzer", 
    "QuestionDecalogoMapper",
    "ExtractorEvidenciasContextual",
    "EvidenceProcessor",
    "EvidenceValidationModel",
    "DNPAlignmentAdapter", 
    "EvaluationDrivenProcessor",
    
    # Factory functions
    "create_decalogo_question_registry",
    
    # Process functions for backward compatibility
    "adaptive_analyzer_process",
    "question_analyzer_process",
    "implementacion_mapeo_process", 
    "extractor_evidencias_process",
    "evidence_processor_process",
    "evidence_validation_process",
    "dnp_alignment_process",
    "evaluation_driven_process",
]
