"""
Question-Level Scoring Pipeline

This module implements a comprehensive question-level scoring pipeline that:
1. Calculates base scores using the mapping Sí=1.0, Parcial=0.5, No/NI=0.0
2. Applies evidence adjustment calculations using weighted averages of evidence quality metrics
3. Produces α-blended final scores that incorporate evidence gap detection logic
4. Integrates with the existing DecalogoQuestionRegistry system
5. Generates deterministic outputs for dimension aggregation

# # # The pipeline consumes question responses and associated evidence from the question_analyzer  # Module not found  # Module not found  # Module not found
output, calculates evidence quality scores using the established relevance/credibility/
recency/authority weighting scheme, and generates per-question scoring artifacts with
evidence gap flags when insufficient evidence is detected.
"""

import json
import logging
import math
# # # from abc import ABC, abstractmethod  # Module not found  # Module not found  # Module not found
# # # from dataclasses import asdict, dataclass, field  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Tuple, Union  # Module not found  # Module not found  # Module not found
# # # from uuid import uuid4  # Module not found  # Module not found  # Module not found

# Import existing components
# # # from evidence_processor import EvidenceScoringSystem, ScoringMetrics, StructuredEvidence  # Module not found  # Module not found  # Module not found
# # # from score_calculator import ScoreCalculator  # Module not found  # Module not found  # Module not found

try:
# # #     from canonical_flow.analysis.audit_logger import get_audit_logger  # Module not found  # Module not found  # Module not found
except ImportError:
    get_audit_logger = None

# Configure logging
logger = logging.getLogger(__name__)


class QuestionResponse(Enum):
    """Standard question response categories with their base scores."""
    SI = ("Sí", 1.0)
    PARCIAL = ("Parcial", 0.5)
    NO = ("No", 0.0)
    NI = ("NI", 0.0)  # No Information
    
    def __init__(self, label: str, base_score: float):
        self.label = label
        self.base_score = base_score


class EvidenceGapSeverity(Enum):
    """Severity levels for evidence gaps."""
    NONE = ("none", 0.0)
    LOW = ("low", 0.1)
    MEDIUM = ("medium", 0.25)
    HIGH = ("high", 0.5)
    CRITICAL = ("critical", 1.0)
    
    def __init__(self, level: str, adjustment_factor: float):
        self.level = level
        self.adjustment_factor = adjustment_factor


@dataclass
class EvidenceQualityWeights:
    """Weighting scheme for evidence quality metrics."""
    relevance: float = 0.4
    credibility: float = 0.3
    recency: float = 0.2
    authority: float = 0.1
    
    def __post_init__(self):
        """Ensure weights sum to 1.0."""
        total = self.relevance + self.credibility + self.recency + self.authority
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Evidence quality weights must sum to 1.0, got {total}")


@dataclass
class EvidenceGapDetection:
    """Configuration for evidence gap detection logic."""
    minimum_evidence_count: int = 2
    minimum_average_quality: float = 0.6
    minimum_total_quality_score: float = 1.2
    critical_dimensions: List[str] = field(default_factory=list)
    
    def detect_gap(self, evidence_list: List[StructuredEvidence]) -> EvidenceGapSeverity:
        """Detect evidence gaps based on configured thresholds."""
        if not evidence_list:
            return EvidenceGapSeverity.CRITICAL
        
        evidence_count = len(evidence_list)
        if evidence_count < self.minimum_evidence_count:
            return EvidenceGapSeverity.HIGH if evidence_count == 1 else EvidenceGapSeverity.CRITICAL
        
        # Calculate average quality
        quality_scores = [e.scoring.overall_score for e in evidence_list]
        average_quality = sum(quality_scores) / len(quality_scores)
        
        if average_quality < self.minimum_average_quality:
            if average_quality < 0.4:
                return EvidenceGapSeverity.HIGH
            else:
                return EvidenceGapSeverity.MEDIUM
        
        # Calculate total quality score
        total_quality = sum(quality_scores)
        if total_quality < self.minimum_total_quality_score:
            return EvidenceGapSeverity.LOW
        
        return EvidenceGapSeverity.NONE


@dataclass
class QuestionScoringResult:
    """Complete scoring result for a single question."""
    question_id: str
    question_text: str
    dimension: str
    base_response: QuestionResponse
    base_score: float
    
    # Evidence metrics
    evidence_count: int
    evidence_quality_score: float
    evidence_adjustment: float
    evidence_gap_severity: EvidenceGapSeverity
    
    # Final scoring
    alpha_blend_factor: float
    final_score: float
    confidence_interval: Tuple[float, float]
    
    # Supporting data
    evidence_artifacts: List[Dict[str, Any]]
    scoring_metadata: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Handle Enum serialization
        result['base_response'] = self.base_response.label
        result['evidence_gap_severity'] = self.evidence_gap_severity.level
        return result


class DecalogoQuestionRegistry:
    """Registry for managing question definitions and mappings."""
    
    def __init__(self):
        self.questions: Dict[str, Dict[str, Any]] = {}
        self._load_default_questions()
    
    def _load_default_questions(self):
# # #         """Load default question definitions from methodology."""  # Module not found  # Module not found  # Module not found
        # Sample questions based on the methodology document
        default_questions = {
            "DE1_C1": {
                "text": "¿Existe identificación clara de causas estructurales?",
                "dimension": "DE1",
                "category": "coherencia",
                "response_type": ["Sí", "Parcial", "No", "NI"],
                "evidence_types": ["causal_analysis", "diagnostic_evidence"]
            },
            "DE1_C2": {
                "text": "¿Se presenta relación lógica entre problemas y causas?",
                "dimension": "DE1",
                "category": "coherencia", 
                "response_type": ["Sí", "Parcial", "No", "NI"],
                "evidence_types": ["logical_framework", "causal_chain"]
            },
            "DE3_G1": {
                "text": "¿Existe identificación de fuentes de financiación diversificadas?",
                "dimension": "DE3",
                "category": "planificacion",
                "response_type": ["Sí", "Parcial", "No", "NI"],
                "evidence_types": ["budget_evidence", "financing_sources"]
            },
            "DE4_CHAIN": {
                "text": "Cadena de valor completa",
                "dimension": "DE4",
                "category": "cadena_valor",
                "response_type": ["Sí", "No"],  # No "Parcial" for DE4
                "evidence_types": ["process_evidence", "value_chain"]
            }
        }
        self.questions.update(default_questions)
    
    def register_question(self, question_id: str, definition: Dict[str, Any]):
        """Register a new question definition."""
        self.questions[question_id] = definition
    
    def get_question(self, question_id: str) -> Optional[Dict[str, Any]]:
        """Get question definition by ID."""
        return self.questions.get(question_id)
    
    def list_questions_by_dimension(self, dimension: str) -> List[str]:
        """List all question IDs for a given dimension."""
        return [qid for qid, qdef in self.questions.items() 
                if qdef.get('dimension') == dimension]


class QuestionLevelScoringPipeline:
    """
    Main pipeline for question-level scoring with evidence integration.
    
    This pipeline implements the complete scoring workflow:
# # #     1. Base score calculation from question responses  # Module not found  # Module not found  # Module not found
    2. Evidence quality assessment using established metrics
    3. Evidence gap detection and severity assessment
    4. α-blended final score calculation with confidence intervals
    5. Deterministic artifact generation for downstream aggregation
    """
    
    def __init__(
        self,
        evidence_scoring_system: Optional[EvidenceScoringSystem] = None,
        score_calculator: Optional[ScoreCalculator] = None,
        question_registry: Optional[DecalogoQuestionRegistry] = None,
        alpha_blend_default: float = 0.3,
        evidence_weights: Optional[EvidenceQualityWeights] = None,
        gap_detection_config: Optional[EvidenceGapDetection] = None
    ):
        """
        Initialize the scoring pipeline.
        
        Args:
            evidence_scoring_system: System for scoring evidence quality
            score_calculator: Calculator for dimension-level scoring
            question_registry: Registry of question definitions
            alpha_blend_default: Default α value for blending base and evidence scores
            evidence_weights: Weights for evidence quality dimensions
            gap_detection_config: Configuration for evidence gap detection
        """
        self.evidence_scoring_system = evidence_scoring_system or EvidenceScoringSystem()
        self.score_calculator = score_calculator or ScoreCalculator()
        self.question_registry = question_registry or DecalogoQuestionRegistry()
        self.alpha_blend_default = alpha_blend_default
        self.evidence_weights = evidence_weights or EvidenceQualityWeights()
        self.gap_detection_config = gap_detection_config or EvidenceGapDetection()
        
        # Internal state
        self._scoring_history: List[QuestionScoringResult] = []
        self._pipeline_metrics = {
            'questions_processed': 0,
            'evidence_gaps_detected': 0,
            'average_final_score': 0.0
        }
    
    def process_question(
        self,
        question_id: str,
        response_text: str,
        evidence_list: List[StructuredEvidence],
        alpha_blend: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> QuestionScoringResult:
        """
        Process a single question through the complete scoring pipeline.
        
        Args:
            question_id: Unique identifier for the question
            response_text: Response text (Sí/Parcial/No/NI)
            evidence_list: List of associated evidence
            alpha_blend: Custom α blending factor (optional)
            metadata: Additional metadata for processing
            
        Returns:
            Complete scoring result with all computed metrics
        """
        # Audit logging setup
        audit_logger = get_audit_logger() if get_audit_logger else None
        input_data = {
            "question_id": question_id,
            "response_text": response_text,
            "evidence_count": len(evidence_list),
            "alpha_blend": alpha_blend or self.alpha_blend_default
        }
        
        if audit_logger:
            with audit_logger.audit_component_execution("question_scoring_pipeline", input_data) as audit_ctx:
                result = self._process_question_internal(
                    question_id, response_text, evidence_list, alpha_blend, metadata
                )
                audit_ctx.set_output({
                    "final_score": result.final_score,
                    "evidence_gap_severity": result.evidence_gap_severity.level,
                    "confidence_interval": result.confidence_interval
                })
                return result
        else:
            return self._process_question_internal(
                question_id, response_text, evidence_list, alpha_blend, metadata
            )
    
    def _process_question_internal(
        self,
        question_id: str,
        response_text: str,
        evidence_list: List[StructuredEvidence],
        alpha_blend: Optional[float],
        metadata: Optional[Dict[str, Any]]
    ) -> QuestionScoringResult:
        """Internal implementation of question processing."""
        
        # Step 1: Get question definition
        question_def = self.question_registry.get_question(question_id)
        if not question_def:
            logger.warning(f"Question {question_id} not found in registry, using defaults")
            question_def = {
                "text": f"Question {question_id}",
                "dimension": "UNKNOWN",
                "category": "general"
            }
        
        question_text = question_def.get("text", f"Question {question_id}")
        dimension = question_def.get("dimension", "UNKNOWN")
        
        # Step 2: Calculate base score
        base_response = self._parse_response(response_text)
        base_score = base_response.base_score
        
        # Step 3: Calculate evidence quality score
        evidence_quality_score = self._calculate_evidence_quality_score(evidence_list)
        
        # Step 4: Detect evidence gaps
        evidence_gap_severity = self.gap_detection_config.detect_gap(evidence_list)
        
        # Step 5: Calculate evidence adjustment
        evidence_adjustment = self._calculate_evidence_adjustment(
            evidence_quality_score, evidence_gap_severity, len(evidence_list)
        )
        
        # Step 6: Calculate α-blended final score
        alpha = alpha_blend if alpha_blend is not None else self.alpha_blend_default
        final_score = self._calculate_alpha_blended_score(
            base_score, evidence_adjustment, alpha, evidence_gap_severity
        )
        
        # Step 7: Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            final_score, evidence_list, evidence_gap_severity
        )
        
        # Step 8: Generate evidence artifacts
        evidence_artifacts = self._generate_evidence_artifacts(evidence_list)
        
        # Step 9: Compile scoring metadata
        scoring_metadata = {
            "pipeline_version": "1.0.0",
            "alpha_blend_factor": alpha,
            "evidence_weights": asdict(self.evidence_weights),
            "gap_detection_config": asdict(self.gap_detection_config),
            "processing_timestamp": datetime.now().isoformat(),
            "question_definition": question_def,
            "input_metadata": metadata or {}
        }
        
        # Create result
        result = QuestionScoringResult(
            question_id=question_id,
            question_text=question_text,
            dimension=dimension,
            base_response=base_response,
            base_score=base_score,
            evidence_count=len(evidence_list),
            evidence_quality_score=evidence_quality_score,
            evidence_adjustment=evidence_adjustment,
            evidence_gap_severity=evidence_gap_severity,
            alpha_blend_factor=alpha,
            final_score=final_score,
            confidence_interval=confidence_interval,
            evidence_artifacts=evidence_artifacts,
            scoring_metadata=scoring_metadata
        )
        
        # Update internal metrics
        self._update_pipeline_metrics(result)
        self._scoring_history.append(result)
        
        return result
    
    def _parse_response(self, response_text: str) -> QuestionResponse:
        """Parse response text into standardized response category."""
        response_text_clean = response_text.strip().lower()
        
        # Map variations to standard responses
        response_mapping = {
            "sí": QuestionResponse.SI,
            "si": QuestionResponse.SI,
            "yes": QuestionResponse.SI,
            "parcial": QuestionResponse.PARCIAL,
            "partial": QuestionResponse.PARCIAL,
            "no": QuestionResponse.NO,
            "ni": QuestionResponse.NI,
            "n/a": QuestionResponse.NI,
            "not applicable": QuestionResponse.NI
        }
        
        return response_mapping.get(response_text_clean, QuestionResponse.NI)
    
    def _calculate_evidence_quality_score(self, evidence_list: List[StructuredEvidence]) -> float:
        """
        Calculate weighted average evidence quality score.
        
        Uses the established relevance/credibility/recency/authority weighting scheme.
        """
        if not evidence_list:
            return 0.0
        
        weighted_scores = []
        total_weight = 0.0
        
        for evidence in evidence_list:
            # Extract individual quality scores
            relevance = evidence.scoring.relevance_score
            credibility = evidence.scoring.credibility_score
            recency = evidence.scoring.recency_score
            authority = evidence.scoring.authority_score
            
            # Apply evidence quality weights
            quality_score = (
                relevance * self.evidence_weights.relevance +
                credibility * self.evidence_weights.credibility +
                recency * self.evidence_weights.recency +
                authority * self.evidence_weights.authority
            )
            
            # Weight by evidence overall score (confidence weighting)
            evidence_weight = evidence.scoring.overall_score
            weighted_scores.append(quality_score * evidence_weight)
            total_weight += evidence_weight
        
        if total_weight == 0.0:
            return 0.0
        
        return sum(weighted_scores) / total_weight
    
    def _calculate_evidence_adjustment(
        self,
        evidence_quality_score: float,
        gap_severity: EvidenceGapSeverity,
        evidence_count: int
    ) -> float:
        """
        Calculate evidence-based adjustment to base score.
        
        The adjustment incorporates both evidence quality and gap penalties.
        """
        if evidence_count == 0:
            return 0.0
        
# # #         # Base adjustment from evidence quality  # Module not found  # Module not found  # Module not found
        quality_adjustment = evidence_quality_score - 0.5  # Neutral point at 0.5
        
        # Apply evidence gap penalty
        gap_penalty = gap_severity.adjustment_factor
        
        # Scale by evidence count (more evidence = more reliable adjustment)
        count_scaling = min(1.0, evidence_count / 3.0)  # Plateau at 3 pieces of evidence
        
        # Final adjustment calculation
        adjustment = (quality_adjustment * count_scaling) - gap_penalty
        
        # Bound the adjustment to reasonable limits
        return max(-0.3, min(0.3, adjustment))
    
    def _calculate_alpha_blended_score(
        self,
        base_score: float,
        evidence_adjustment: float,
        alpha: float,
        gap_severity: EvidenceGapSeverity
    ) -> float:
        """
        Calculate α-blended final score incorporating evidence gap detection.
        
        Formula: final_score = (1-α) * base_score + α * (base_score + evidence_adjustment)
        With gap severity modulation of the α parameter.
        """
        # Modulate α based on evidence gap severity
        if gap_severity == EvidenceGapSeverity.CRITICAL:
            effective_alpha = alpha * 0.1  # Heavy reliance on base score
        elif gap_severity == EvidenceGapSeverity.HIGH:
            effective_alpha = alpha * 0.3
        elif gap_severity == EvidenceGapSeverity.MEDIUM:
            effective_alpha = alpha * 0.6
        elif gap_severity == EvidenceGapSeverity.LOW:
            effective_alpha = alpha * 0.8
        else:  # No gap
            effective_alpha = alpha
        
        # Calculate evidence-adjusted score
        evidence_adjusted_score = base_score + evidence_adjustment
        
        # α-blended calculation
        final_score = (1 - effective_alpha) * base_score + effective_alpha * evidence_adjusted_score
        
        # Ensure score is within [0, 1] bounds
        return max(0.0, min(1.0, final_score))
    
    def _calculate_confidence_interval(
        self,
        final_score: float,
        evidence_list: List[StructuredEvidence],
        gap_severity: EvidenceGapSeverity
    ) -> Tuple[float, float]:
        """Calculate confidence interval for the final score."""
        
        # Base confidence width based on evidence count and quality
        if not evidence_list:
            confidence_width = 0.4  # High uncertainty
        else:
            evidence_count = len(evidence_list)
            avg_confidence = sum(e.scoring.overall_score for e in evidence_list) / evidence_count
            
            # Wider intervals for less evidence or lower quality
            base_width = 0.1 + (1.0 - avg_confidence) * 0.2
            count_factor = max(0.5, 1.0 / math.sqrt(evidence_count))
            confidence_width = base_width * count_factor
        
        # Adjust based on evidence gap severity
        gap_adjustment = gap_severity.adjustment_factor * 0.2
        confidence_width += gap_adjustment
        
        # Ensure interval stays within [0, 1]
        half_width = confidence_width / 2
        lower_bound = max(0.0, final_score - half_width)
        upper_bound = min(1.0, final_score + half_width)
        
        return (lower_bound, upper_bound)
    
    def _generate_evidence_artifacts(self, evidence_list: List[StructuredEvidence]) -> List[Dict[str, Any]]:
        """Generate evidence artifacts for downstream consumption."""
        artifacts = []
        
        for i, evidence in enumerate(evidence_list):
            artifact = {
                "artifact_id": f"evidence_{i+1}_{uuid4().hex[:8]}",
                "evidence_id": getattr(evidence, 'id', f"evidence_{i+1}"),
                "chunk_text": evidence.chunk.text[:200] + "..." if len(evidence.chunk.text) > 200 else evidence.chunk.text,
                "source_document": evidence.metadata.document_title,
                "page_number": evidence.chunk.page_number,
                "quality_scores": {
                    "relevance": evidence.scoring.relevance_score,
                    "credibility": evidence.scoring.credibility_score,
                    "recency": evidence.scoring.recency_score,
                    "authority": evidence.scoring.authority_score,
                    "overall": evidence.scoring.overall_score
                },
                "confidence_level": evidence.scoring.confidence_level.value if hasattr(evidence.scoring.confidence_level, 'value') else str(evidence.scoring.confidence_level),
                "classification_labels": evidence.scoring.classification_labels,
                "extraction_timestamp": evidence.chunk.extraction_timestamp
            }
            artifacts.append(artifact)
        
        return artifacts
    
    def _update_pipeline_metrics(self, result: QuestionScoringResult):
        """Update internal pipeline performance metrics."""
        self._pipeline_metrics['questions_processed'] += 1
        
        if result.evidence_gap_severity != EvidenceGapSeverity.NONE:
            self._pipeline_metrics['evidence_gaps_detected'] += 1
        
        # Update running average of final scores
        total_processed = self._pipeline_metrics['questions_processed']
        current_avg = self._pipeline_metrics['average_final_score']
        new_avg = ((total_processed - 1) * current_avg + result.final_score) / total_processed
        self._pipeline_metrics['average_final_score'] = new_avg
    
    def process_batch(
        self,
        questions_batch: List[Dict[str, Any]],
        batch_metadata: Optional[Dict[str, Any]] = None
    ) -> List[QuestionScoringResult]:
        """
        Process a batch of questions through the scoring pipeline.
        
        Args:
            questions_batch: List of question data dictionaries
            batch_metadata: Optional metadata for the batch
            
        Returns:
            List of scoring results for all questions in the batch
        """
        results = []
        batch_start = datetime.now()
        
        logger.info(f"Processing batch of {len(questions_batch)} questions")
        
        for i, question_data in enumerate(questions_batch):
            try:
                question_id = question_data.get('question_id', f'batch_q_{i+1}')
                response_text = question_data.get('response', 'NI')
                evidence_list = question_data.get('evidence', [])
                alpha_blend = question_data.get('alpha_blend')
                metadata = question_data.get('metadata', {})
                
                # Add batch context to metadata
                if batch_metadata:
                    metadata['batch_metadata'] = batch_metadata
                metadata['batch_position'] = i + 1
                metadata['batch_size'] = len(questions_batch)
                
                result = self.process_question(
                    question_id=question_id,
                    response_text=response_text,
                    evidence_list=evidence_list,
                    alpha_blend=alpha_blend,
                    metadata=metadata
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing question {i+1} in batch: {e}")
                # Create error result
                error_result = QuestionScoringResult(
                    question_id=question_data.get('question_id', f'batch_q_{i+1}'),
                    question_text=question_data.get('question_text', 'Error processing question'),
                    dimension='ERROR',
                    base_response=QuestionResponse.NI,
                    base_score=0.0,
                    evidence_count=0,
                    evidence_quality_score=0.0,
                    evidence_adjustment=0.0,
                    evidence_gap_severity=EvidenceGapSeverity.CRITICAL,
                    alpha_blend_factor=self.alpha_blend_default,
                    final_score=0.0,
                    confidence_interval=(0.0, 0.0),
                    evidence_artifacts=[],
                    scoring_metadata={
                        'error': str(e),
                        'processing_failed': True,
                        'batch_position': i + 1
                    }
                )
                results.append(error_result)
        
        batch_duration = (datetime.now() - batch_start).total_seconds()
        logger.info(f"Batch processing completed in {batch_duration:.2f}s")
        
        return results
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get current pipeline performance metrics."""
        gap_rate = (
            self._pipeline_metrics['evidence_gaps_detected'] / 
            max(1, self._pipeline_metrics['questions_processed'])
        ) * 100
        
        return {
            **self._pipeline_metrics,
            'evidence_gap_rate_percent': gap_rate,
            'scoring_history_length': len(self._scoring_history)
        }
    
    def export_results(self, output_format: str = 'json') -> Union[str, Dict[str, Any]]:
        """
        Export all scoring results in the specified format.
        
        Args:
            output_format: 'json', 'dict', or 'summary'
            
        Returns:
            Exported results in the requested format
        """
        if output_format == 'dict':
            return {
                'pipeline_metrics': self.get_pipeline_metrics(),
                'scoring_results': [result.to_dict() for result in self._scoring_history],
                'export_timestamp': datetime.now().isoformat()
            }
        elif output_format == 'summary':
            return {
                'total_questions': len(self._scoring_history),
                'average_final_score': self._pipeline_metrics['average_final_score'],
                'evidence_gap_rate': self.get_pipeline_metrics()['evidence_gap_rate_percent'],
                'dimensions_covered': list(set(r.dimension for r in self._scoring_history)),
                'score_distribution': {
                    'high (>0.8)': len([r for r in self._scoring_history if r.final_score > 0.8]),
                    'medium (0.5-0.8)': len([r for r in self._scoring_history if 0.5 <= r.final_score <= 0.8]),
                    'low (<0.5)': len([r for r in self._scoring_history if r.final_score < 0.5])
                }
            }
        else:  # json
            data = self.export_results('dict')
            return json.dumps(data, indent=2, ensure_ascii=False)


def create_default_pipeline(**kwargs) -> QuestionLevelScoringPipeline:
    """
    Factory function to create a pipeline with sensible defaults.
    
    Args:
        **kwargs: Override default configuration parameters
        
    Returns:
        Configured QuestionLevelScoringPipeline instance
    """
    config = {
        'alpha_blend_default': 0.3,
        'evidence_weights': EvidenceQualityWeights(),
        'gap_detection_config': EvidenceGapDetection(
            minimum_evidence_count=2,
            minimum_average_quality=0.6,
            critical_dimensions=['DE1', 'DE2', 'DE3', 'DE4']
        )
    }
    config.update(kwargs)
    
    return QuestionLevelScoringPipeline(**config)


if __name__ == "__main__":
    # Demo usage
    pipeline = create_default_pipeline()
    
    # Sample question processing
    sample_evidence = []  # Would normally contain StructuredEvidence objects
    
    result = pipeline.process_question(
        question_id="DE1_C1",
        response_text="Parcial",
        evidence_list=sample_evidence,
        metadata={"demo": True}
    )
    
    print(f"Question: {result.question_text}")
    print(f"Base Score: {result.base_score}")
    print(f"Final Score: {result.final_score:.3f}")
    print(f"Evidence Gap: {result.evidence_gap_severity.level}")
    print(f"Confidence Interval: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]")