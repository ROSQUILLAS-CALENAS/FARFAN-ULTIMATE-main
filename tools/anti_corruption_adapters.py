"""
Anti-Corruption Adapters Module

This module implements adapter classes to break circular dependencies between retrieval and analysis phases
by translating data transfer objects and blocking backward imports. Provides schema mismatch logging that
captures DTO translation failures and forwards them to a lineage tracking system for monitoring cross-phase
data contract violations.

Design Principles:
1. Enforce unidirectional data flow (retrieval -> analysis)
2. Translate between phase-specific DTOs to prevent tight coupling
3. Log and monitor schema mismatches for data contract violations
4. Raise ImportError for backward dependencies at runtime
5. Provide lineage tracking for cross-phase data transformations
"""

import inspect
import json
import logging
import sys
import threading
import traceback
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

# Import guards to prevent backward dependencies
_BLOCKED_ANALYSIS_MODULES = {
    'analysis_nlp',
    'analysis_nlp.adaptive_analyzer', 
    'analysis_nlp.question_analyzer',
    'analysis_nlp.evidence_processor',
    'analysis_nlp.evaluation_driven_processor',
    'analysis_nlp.dnp_alignment_adapter',
    'analysis_nlp.evidence_validation_model',
    'analysis_nlp.implementacion_mapeo'
}

_BLOCKED_RETRIEVAL_MODULES = {
    'retrieval_engine',
    'retrieval_engine.hybrid_retriever',
    'retrieval_engine.lexical_index', 
    'retrieval_engine.vector_index'
}

# Setup logging
logger = logging.getLogger(__name__)


class BackwardDependencyError(ImportError):
    """Raised when a backward dependency is detected"""
    
    def __init__(self, importing_module: str, blocked_module: str, reason: str):
        self.importing_module = importing_module
        self.blocked_module = blocked_module
        self.reason = reason
        super().__init__(f"Backward dependency detected: {importing_module} -> {blocked_module}. {reason}")


class SchemaMismatchError(Exception):
    """Raised when DTO schema translation fails"""
    
    def __init__(self, source_schema: str, target_schema: str, field_errors: Dict[str, str]):
        self.source_schema = source_schema
        self.target_schema = target_schema
        self.field_errors = field_errors
        super().__init__(f"Schema mismatch: {source_schema} -> {target_schema}")


@dataclass
class SchemaViolation:
    """Record of a schema mismatch event"""
    
    timestamp: datetime
    source_phase: str
    target_phase: str
    source_schema: str
    target_schema: str
    field_errors: Dict[str, str]
    stack_trace: str
    adapter_class: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'source_phase': self.source_phase,
            'target_phase': self.target_phase,
            'source_schema': self.source_schema,
            'target_schema': self.target_schema,
            'field_errors': self.field_errors,
            'stack_trace': self.stack_trace,
            'adapter_class': self.adapter_class
        }


# Data Transfer Objects for cross-phase communication
@dataclass
class RetrievalResultDTO:
    """DTO for retrieval phase results"""
    
    query: str
    documents: List[Dict[str, Any]]
    scores: List[float]
    retrieval_method: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> Dict[str, str]:
        """Validate DTO fields and return errors"""
        errors = {}
        
        if not isinstance(self.query, str) or not self.query.strip():
            errors['query'] = 'Query must be a non-empty string'
            
        if not isinstance(self.documents, list):
            errors['documents'] = 'Documents must be a list'
        elif not all(isinstance(doc, dict) for doc in self.documents):
            errors['documents'] = 'All documents must be dictionaries'
            
        if not isinstance(self.scores, list):
            errors['scores'] = 'Scores must be a list'
        elif len(self.scores) != len(self.documents):
            errors['scores'] = 'Scores length must match documents length'
        elif not all(isinstance(score, (int, float)) for score in self.scores):
            errors['scores'] = 'All scores must be numeric'
            
        if not isinstance(self.retrieval_method, str):
            errors['retrieval_method'] = 'Retrieval method must be a string'
            
        if not isinstance(self.metadata, dict):
            errors['metadata'] = 'Metadata must be a dictionary'
            
        return errors


@dataclass 
class AnalysisInputDTO:
    """DTO for analysis phase input"""
    
    content: str
    document_metadata: Dict[str, Any]
    processing_context: Dict[str, Any]
    analysis_type: str
    priority: int = 1
    
    def validate(self) -> Dict[str, str]:
        """Validate DTO fields and return errors"""
        errors = {}
        
        if not isinstance(self.content, str):
            errors['content'] = 'Content must be a string'
            
        if not isinstance(self.document_metadata, dict):
            errors['document_metadata'] = 'Document metadata must be a dictionary'
            
        if not isinstance(self.processing_context, dict):
            errors['processing_context'] = 'Processing context must be a dictionary'
            
        if not isinstance(self.analysis_type, str) or not self.analysis_type.strip():
            errors['analysis_type'] = 'Analysis type must be a non-empty string'
            
        if not isinstance(self.priority, int) or self.priority < 1:
            errors['priority'] = 'Priority must be a positive integer'
            
        return errors


@dataclass
class AnalysisResultDTO:
    """DTO for analysis phase results"""
    
    analysis_id: str
    results: Dict[str, Any]
    confidence: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> Dict[str, str]:
        """Validate DTO fields and return errors"""
        errors = {}
        
        if not isinstance(self.analysis_id, str) or not self.analysis_id.strip():
            errors['analysis_id'] = 'Analysis ID must be a non-empty string'
            
        if not isinstance(self.results, dict):
            errors['results'] = 'Results must be a dictionary'
            
        if not isinstance(self.confidence, (int, float)) or not 0 <= self.confidence <= 1:
            errors['confidence'] = 'Confidence must be a number between 0 and 1'
            
        if not isinstance(self.processing_time, (int, float)) or self.processing_time < 0:
            errors['processing_time'] = 'Processing time must be a non-negative number'
            
        if not isinstance(self.metadata, dict):
            errors['metadata'] = 'Metadata must be a dictionary'
            
        return errors


class SchemaViolationLogger:
    """Logs and forwards schema mismatches to lineage tracking system"""
    
    def __init__(self, log_file: Optional[Path] = None, enable_lineage: bool = True):
        self.log_file = log_file or Path('schema_violations.log')
        self.enable_lineage = enable_lineage
        self._lock = threading.RLock()
        self._violations: List[SchemaViolation] = []
        
        # Try to import lineage tracker
        self.lineage_tracker = None
        if enable_lineage:
            try:
                from lineage_tracker import LineageTracker
                self.lineage_tracker = LineageTracker()
                logger.info("Schema violation logging connected to lineage tracker")
            except (ImportError, NameError) as e:
                logger.warning(f"Lineage tracker not available ({e}), violations will be logged locally only")
    
    def log_violation(self, violation: SchemaViolation) -> None:
        """Log schema violation and forward to lineage tracking"""
        with self._lock:
            self._violations.append(violation)
            
            # Log to file
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(violation.to_dict()) + '\n')
            except Exception as e:
                logger.error(f"Failed to write violation to log file: {e}")
            
            # Forward to lineage tracker if available
            if self.lineage_tracker:
                try:
                    # Create a trace for schema violations if not exists
                    trace_id = f"schema_violations_{datetime.now().strftime('%Y%m%d')}"
                    
                    # Log processing step for this violation
                    self.lineage_tracker.log_processing_step(
                        trace_id=trace_id,
                        step=f"schema_violation_{violation.adapter_class}",
                        result=violation.to_dict()
                    )
                    logger.debug(f"Schema violation forwarded to lineage tracker: {trace_id}")
                except Exception as e:
                    logger.error(f"Failed to forward violation to lineage tracker: {e}")
            
            logger.warning(f"Schema violation logged: {violation.source_schema} -> {violation.target_schema}")
    
    def get_violations(self, limit: Optional[int] = None) -> List[SchemaViolation]:
        """Get recent violations"""
        with self._lock:
            if limit:
                return self._violations[-limit:]
            return self._violations.copy()
    
    def clear_violations(self) -> None:
        """Clear violation history"""
        with self._lock:
            self._violations.clear()


# Global schema violation logger
_schema_logger = SchemaViolationLogger()


class ImportGuard:
    """Prevents backward dependencies by monitoring import attempts"""
    
    @staticmethod
    def check_import(importing_module: str, target_module: str) -> None:
        """Check if an import should be blocked"""
        
        # Block analysis modules from importing retrieval components
        if (importing_module.startswith('analysis_nlp') and 
            any(target_module.startswith(blocked) for blocked in _BLOCKED_RETRIEVAL_MODULES)):
            raise BackwardDependencyError(
                importing_module=importing_module,
                blocked_module=target_module,
                reason="Analysis modules cannot import retrieval components directly. Use adapters instead."
            )
        
        # Block direct circular imports between phases
        if (importing_module.startswith('retrieval_engine') and
            any(target_module.startswith(blocked) for blocked in _BLOCKED_ANALYSIS_MODULES)):
            raise BackwardDependencyError(
                importing_module=importing_module,
                blocked_module=target_module,
                reason="Retrieval modules cannot import analysis components. Use forward references only."
            )
    
    @staticmethod
    def install_import_hook() -> None:
        """Install import hook to monitor all imports"""
        original_import = __builtins__.__import__
        
        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            # Get the importing module
            if globals and '__name__' in globals:
                importing_module = globals['__name__']
                
                # Check the import
                try:
                    ImportGuard.check_import(importing_module, name)
                except BackwardDependencyError as e:
                    logger.error(f"Blocked backward dependency: {e}")
                    raise e
            
            return original_import(name, globals, locals, fromlist, level)
        
        __builtins__.__import__ = guarded_import
        logger.info("Import guard installed")


class BaseAntiCorruptionAdapter(ABC):
    """Base class for anti-corruption adapters"""
    
    def __init__(self, source_phase: str, target_phase: str):
        self.source_phase = source_phase
        self.target_phase = target_phase
        self._adapter_id = f"{source_phase}_{target_phase}_adapter"
        
    @abstractmethod
    def translate(self, source_dto: Any) -> Any:
        """Translate between DTOs"""
        pass
    
    def _log_schema_violation(self, source_dto: Any, target_dto_class: Type, field_errors: Dict[str, str]) -> None:
        """Log schema violation for monitoring"""
        violation = SchemaViolation(
            timestamp=datetime.now(),
            source_phase=self.source_phase,
            target_phase=self.target_phase,
            source_schema=source_dto.__class__.__name__,
            target_schema=target_dto_class.__name__,
            field_errors=field_errors,
            stack_trace=traceback.format_exc(),
            adapter_class=self.__class__.__name__
        )
        
        _schema_logger.log_violation(violation)
    
    def _validate_translation(self, source_dto: Any, target_dto: Any) -> None:
        """Validate DTO translation and log any issues"""
        if hasattr(source_dto, 'validate'):
            source_errors = source_dto.validate()
            if source_errors:
                logger.warning(f"Source DTO validation errors: {source_errors}")
        
        if hasattr(target_dto, 'validate'):
            target_errors = target_dto.validate()
            if target_errors:
                self._log_schema_violation(source_dto, type(target_dto), target_errors)
                raise SchemaMismatchError(
                    source_schema=type(source_dto).__name__,
                    target_schema=type(target_dto).__name__,
                    field_errors=target_errors
                )


class RetrievalToAnalysisAdapter(BaseAntiCorruptionAdapter):
    """Adapts retrieval results for analysis phase consumption"""
    
    def __init__(self):
        super().__init__('retrieval', 'analysis')
    
    def translate(self, retrieval_result: RetrievalResultDTO) -> List[AnalysisInputDTO]:
        """Translate retrieval results to analysis inputs"""
        try:
            analysis_inputs = []
            
            for i, (doc, score) in enumerate(zip(retrieval_result.documents, retrieval_result.scores)):
                # Extract content from document
                content = self._extract_content(doc)
                
                # Build document metadata
                doc_metadata = {
                    'document_id': doc.get('id', f'doc_{i}'),
                    'title': doc.get('title', ''),
                    'source': doc.get('source', 'unknown'),
                    'retrieval_score': score,
                    'retrieval_rank': i + 1
                }
                
                # Build processing context
                processing_context = {
                    'original_query': retrieval_result.query,
                    'retrieval_method': retrieval_result.retrieval_method,
                    'total_results': len(retrieval_result.documents),
                    'metadata': retrieval_result.metadata
                }
                
                # Determine analysis type based on content
                analysis_type = self._determine_analysis_type(content, doc)
                
                # Calculate priority based on score
                priority = max(1, int(score * 10)) if score > 0 else 1
                
                analysis_input = AnalysisInputDTO(
                    content=content,
                    document_metadata=doc_metadata,
                    processing_context=processing_context,
                    analysis_type=analysis_type,
                    priority=priority
                )
                
                # Validate translation
                self._validate_translation(retrieval_result, analysis_input)
                analysis_inputs.append(analysis_input)
            
            logger.info(f"Translated {len(analysis_inputs)} retrieval results to analysis inputs")
            return analysis_inputs
            
        except Exception as e:
            logger.error(f"Failed to translate retrieval result: {e}")
            self._log_schema_violation(
                retrieval_result, 
                AnalysisInputDTO, 
                {'translation_error': str(e)}
            )
            raise
    
    def _extract_content(self, doc: Dict[str, Any]) -> str:
        """Extract text content from document"""
        # Try various common content fields
        for field in ['content', 'text', 'body', 'description', 'summary']:
            if field in doc and isinstance(doc[field], str):
                return doc[field]
        
        # Fallback to document representation
        return str(doc)
    
    def _determine_analysis_type(self, content: str, doc: Dict[str, Any]) -> str:
        """Determine appropriate analysis type based on content"""
        # Simple heuristics for analysis type detection
        if len(content) > 5000:
            return 'deep_analysis'
        elif 'question' in content.lower() or '?' in content:
            return 'question_analysis'
        elif doc.get('type') == 'evidence':
            return 'evidence_analysis'
        else:
            return 'standard_analysis'


class AnalysisToRetrievalAdapter(BaseAntiCorruptionAdapter):
    """Adapts analysis results for retrieval feedback (limited scope)"""
    
    def __init__(self):
        super().__init__('analysis', 'retrieval')
        
        # This adapter should only be used for limited feedback scenarios
        warnings.warn(
            "AnalysisToRetrievalAdapter should only be used for limited feedback scenarios. "
            "Avoid creating backward dependencies.",
            UserWarning
        )
    
    def translate(self, analysis_result: AnalysisResultDTO) -> Dict[str, Any]:
        """Translate analysis results to retrieval feedback (limited scope)"""
        try:
            # Only provide limited feedback information
            feedback = {
                'analysis_id': analysis_result.analysis_id,
                'confidence': analysis_result.confidence,
                'processing_time': analysis_result.processing_time,
                'quality_indicators': self._extract_quality_indicators(analysis_result.results)
            }
            
            self._validate_translation(analysis_result, feedback)
            logger.info(f"Translated analysis result to retrieval feedback: {analysis_result.analysis_id}")
            return feedback
            
        except Exception as e:
            logger.error(f"Failed to translate analysis result: {e}")
            self._log_schema_violation(
                analysis_result,
                dict,  # Using dict as target type
                {'translation_error': str(e)}
            )
            raise
    
    def _extract_quality_indicators(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract quality indicators that can be used for retrieval improvement"""
        indicators = {}
        
        # Extract safe quality metrics that don't expose analysis internals
        if 'relevance_score' in results:
            indicators['relevance'] = results['relevance_score']
        
        if 'completeness_score' in results:
            indicators['completeness'] = results['completeness_score']
        
        # Add processing success indicator
        indicators['processing_success'] = len(results) > 0
        
        return indicators


class AdapterFactory:
    """Factory for creating anti-corruption adapters"""
    
    _adapters = {
        ('retrieval', 'analysis'): RetrievalToAnalysisAdapter,
        ('analysis', 'retrieval'): AnalysisToRetrievalAdapter,
    }
    
    @classmethod
    def create_adapter(cls, source_phase: str, target_phase: str) -> BaseAntiCorruptionAdapter:
        """Create appropriate adapter for phase transition"""
        key = (source_phase, target_phase)
        
        if key not in cls._adapters:
            raise ValueError(f"No adapter available for {source_phase} -> {target_phase}")
        
        adapter_class = cls._adapters[key]
        return adapter_class()
    
    @classmethod
    def register_adapter(cls, source_phase: str, target_phase: str, 
                        adapter_class: Type[BaseAntiCorruptionAdapter]) -> None:
        """Register custom adapter"""
        key = (source_phase, target_phase)
        cls._adapters[key] = adapter_class
        logger.info(f"Registered custom adapter: {source_phase} -> {target_phase}")


def install_import_guards() -> None:
    """Install import guards to prevent backward dependencies"""
    ImportGuard.install_import_hook()


def get_schema_violations(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get recent schema violations"""
    violations = _schema_logger.get_violations(limit)
    return [v.to_dict() for v in violations]


def clear_schema_violations() -> None:
    """Clear schema violation history"""
    _schema_logger.clear_violations()


# Export public interface
__all__ = [
    'RetrievalResultDTO',
    'AnalysisInputDTO', 
    'AnalysisResultDTO',
    'BaseAntiCorruptionAdapter',
    'RetrievalToAnalysisAdapter',
    'AnalysisToRetrievalAdapter',
    'AdapterFactory',
    'SchemaViolation',
    'SchemaMismatchError',
    'BackwardDependencyError',
    'install_import_guards',
    'get_schema_violations',
    'clear_schema_violations'
]