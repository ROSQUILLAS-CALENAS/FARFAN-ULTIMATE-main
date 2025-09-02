"""
Confidence and Quality Metrics Propagation System

This module implements a comprehensive confidence and quality metrics system that flows
from individual question evaluations through dimension, point, meso, and macro aggregation
levels with proper evidence density, coverage validation, and model agreement scoring.

Key Features:
- Evidence density calculation (evidence count per question)
- Coverage validation (answered vs total questions ratios) 
- Model agreement scoring (consistency across multiple analysis models)
- Quality metric aggregation with evidence credibility, temporal recency, and authority
- Bounded scores (0.0-1.0) with documented calculation formulas
- Integration with existing artifact generation system
"""

import json
import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict, OrderedDict

logger = logging.getLogger(__name__)


class AggregationLevel(Enum):
    """Hierarchical aggregation levels for confidence and quality metrics"""
    QUESTION = "question_level"
    DIMENSION = "dimension_level" 
    POINT = "point_level"
    MESO = "meso_level"
    MACRO = "macro_level"


class ConfidenceFactorType(Enum):
    """Types of factors that influence confidence scoring"""
    EVIDENCE_DENSITY = "evidence_density"
    COVERAGE_VALIDATION = "coverage_validation"
    MODEL_AGREEMENT = "model_agreement"
    TEMPORAL_RECENCY = "temporal_recency"
    SOURCE_AUTHORITY = "source_authority"
    CREDIBILITY = "evidence_credibility"


@dataclass
class EvidenceDensityMetrics:
    """Metrics for evidence density calculation"""
    total_evidence_count: int = 0
    question_count: int = 0
    density_ratio: float = 0.0
    min_evidence_threshold: int = 3
    optimal_evidence_count: int = 5
    
    def __post_init__(self):
        self.calculate_density()
    
    def calculate_density(self) -> float:
        """
        Calculate evidence density score (0.0-1.0)
        
        Formula: min(evidence_count / (question_count * optimal_evidence_count), 1.0)
        - Penalizes insufficient evidence per question
        - Rewards comprehensive evidence coverage
        """
        if self.question_count == 0:
            self.density_ratio = 0.0
            return self.density_ratio
        
        expected_total = self.question_count * self.optimal_evidence_count
        self.density_ratio = min(self.total_evidence_count / expected_total, 1.0) if expected_total > 0 else 0.0
        return self.density_ratio


@dataclass
class CoverageValidationMetrics:
    """Metrics for coverage validation calculation"""
    answered_questions: int = 0
    total_questions: int = 0
    coverage_ratio: float = 0.0
    critical_gaps: int = 0
    gap_penalty_factor: float = 0.1
    
    def __post_init__(self):
        self.calculate_coverage()
    
    def calculate_coverage(self) -> float:
        """
        Calculate coverage validation score (0.0-1.0)
        
        Formula: (answered_questions / total_questions) * (1 - critical_gaps * gap_penalty_factor)
        - Base score from answer completion ratio
        - Penalty for critical coverage gaps
        """
        if self.total_questions == 0:
            self.coverage_ratio = 0.0
            return self.coverage_ratio
        
        base_coverage = self.answered_questions / self.total_questions
        gap_penalty = min(self.critical_gaps * self.gap_penalty_factor, 0.5)
        self.coverage_ratio = max(0.0, base_coverage - gap_penalty)
        return self.coverage_ratio


@dataclass
class ModelAgreementMetrics:
    """Metrics for model agreement scoring"""
    model_scores: List[float] = field(default_factory=list)
    agreement_threshold: float = 0.1
    consensus_score: float = 0.0
    variance_penalty: float = 0.0
    
    def __post_init__(self):
        if self.model_scores:
            self.calculate_agreement()
    
    def calculate_agreement(self) -> float:
        """
        Calculate model agreement score (0.0-1.0)
        
        Formula: exp(-variance / agreement_threshold) * (1 - outlier_penalty)
        - Higher agreement for consistent model outputs
        - Exponential penalty for high variance
        - Additional penalty for significant outliers
        """
        if len(self.model_scores) < 2:
            self.consensus_score = 1.0 if self.model_scores else 0.0
            return self.consensus_score
        
        # Calculate variance
        variance = statistics.variance(self.model_scores)
        self.variance_penalty = variance
        
        # Exponential decay based on variance
        base_agreement = math.exp(-variance / self.agreement_threshold)
        
        # Outlier penalty
        mean_score = statistics.mean(self.model_scores)
        outliers = sum(1 for score in self.model_scores if abs(score - mean_score) > 2 * self.agreement_threshold)
        outlier_penalty = min(outliers * 0.1, 0.3)
        
        self.consensus_score = max(0.0, base_agreement - outlier_penalty)
        return self.consensus_score


@dataclass
class QualityComponents:
    """Components for quality metric calculation"""
    credibility_score: float = 0.0
    recency_weight: float = 0.0
    authority_ranking: float = 0.0
    composite_score: float = 0.0
    
    # Weighting factors
    credibility_weight: float = 0.4
    recency_weight_factor: float = 0.3
    authority_weight: float = 0.3
    
    def __post_init__(self):
        self.calculate_composite()
    
    def calculate_composite(self) -> float:
        """
        Calculate composite quality score (0.0-1.0)
        
        Formula: credibility_score * credibility_weight + 
                recency_weight * recency_weight_factor + 
                authority_ranking * authority_weight
        """
        self.composite_score = (
            self.credibility_score * self.credibility_weight +
            self.recency_weight * self.recency_weight_factor +
            self.authority_ranking * self.authority_weight
        )
        # Ensure bounded between 0.0 and 1.0
        self.composite_score = max(0.0, min(1.0, self.composite_score))
        return self.composite_score


@dataclass
class ConfidenceQualityScore:
    """Combined confidence and quality score with metadata"""
    level: AggregationLevel
    confidence_score: float = 0.0
    quality_score: float = 0.0
    
    # Component metrics
    evidence_density: Optional[EvidenceDensityMetrics] = None
    coverage_validation: Optional[CoverageValidationMetrics] = None
    model_agreement: Optional[ModelAgreementMetrics] = None
    quality_components: Optional[QualityComponents] = None
    
    # Metadata
    calculation_timestamp: str = ""
    evidence_gaps: List[str] = field(default_factory=list)
    uncertainty_factors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.calculation_timestamp:
            self.calculation_timestamp = datetime.utcnow().isoformat()


class ConfidenceQualityCalculator:
    """Calculator for confidence and quality metrics with propagation rules"""
    
    def __init__(self):
        self.temporal_decay_days = 365  # Days for full temporal decay
        self.authority_sources = {}  # Source ID to authority score mapping
        self.credibility_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.3
        }
        
    def calculate_evidence_credibility(self, evidence_items: List[Dict[str, Any]]) -> float:
        """
        Calculate evidence credibility score based on source quality and validation
        
        Formula: weighted_average(source_credibility * validation_score)
        """
        if not evidence_items:
            return 0.0
        
        credibility_scores = []
        for evidence in evidence_items:
            # Extract source credibility
            source_cred = evidence.get('source_credibility', 0.5)
            if isinstance(source_cred, str):
                source_cred = self.credibility_thresholds.get(source_cred, 0.5)
            
            # Extract validation score
            validation_score = evidence.get('validation_score', 0.5)
            
            # Combine scores
            combined_score = (source_cred + validation_score) / 2
            credibility_scores.append(combined_score)
        
        return statistics.mean(credibility_scores)
    
    def calculate_temporal_recency_weight(self, evidence_items: List[Dict[str, Any]]) -> float:
        """
        Calculate temporal recency weight based on evidence timestamps
        
        Formula: weighted_average(exp(-days_since_creation / temporal_decay_days))
        """
        if not evidence_items:
            return 0.0
        
        current_time = datetime.utcnow()
        recency_weights = []
        
        for evidence in evidence_items:
            timestamp_str = evidence.get('timestamp') or evidence.get('creation_timestamp')
            if not timestamp_str:
                recency_weights.append(0.5)  # Default for missing timestamp
                continue
            
            try:
                evidence_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                days_diff = (current_time - evidence_time).days
                
                # Exponential decay
                recency_weight = math.exp(-days_diff / self.temporal_decay_days)
                recency_weights.append(recency_weight)
                
            except (ValueError, TypeError):
                recency_weights.append(0.5)  # Default for invalid timestamp
        
        return statistics.mean(recency_weights)
    
    def calculate_authority_ranking(self, evidence_items: List[Dict[str, Any]]) -> float:
        """
        Calculate authority ranking based on source authority scores
        
        Formula: weighted_average(source_authority_scores)
        """
        if not evidence_items:
            return 0.0
        
        authority_scores = []
        for evidence in evidence_items:
            source_id = evidence.get('source_id', '')
            authority_score = self.authority_sources.get(source_id, 0.5)
            
            # Also check for direct authority score
            direct_authority = evidence.get('authority_score')
            if direct_authority is not None:
                authority_score = max(authority_score, direct_authority)
            
            authority_scores.append(authority_score)
        
        return statistics.mean(authority_scores)
    
    def calculate_question_level_metrics(self, question_data: Dict[str, Any]) -> ConfidenceQualityScore:
        """Calculate confidence and quality metrics at question level"""
        
        evidence_items = question_data.get('evidence', [])
        if not isinstance(evidence_items, list):
            evidence_items = [evidence_items] if evidence_items else []
        
        # Evidence density
        evidence_density = EvidenceDensityMetrics(
            total_evidence_count=len(evidence_items),
            question_count=1,
            optimal_evidence_count=5
        )
        
        # Coverage validation (single question)
        has_answer = bool(question_data.get('answer') or question_data.get('response'))
        coverage = CoverageValidationMetrics(
            answered_questions=1 if has_answer else 0,
            total_questions=1,
            critical_gaps=0 if has_answer else 1
        )
        
        # Model agreement (from multiple analysis models)
        model_scores = []
        for model_key in ['nlp_score', 'semantic_score', 'context_score']:
            score = question_data.get(model_key)
            if score is not None and isinstance(score, (int, float)):
                model_scores.append(float(score))
        
        agreement = ModelAgreementMetrics(model_scores=model_scores)
        
        # Quality components
        quality = QualityComponents(
            credibility_score=self.calculate_evidence_credibility(evidence_items),
            recency_weight=self.calculate_temporal_recency_weight(evidence_items),
            authority_ranking=self.calculate_authority_ranking(evidence_items)
        )
        
        # Calculate final confidence score
        confidence_components = [
            evidence_density.density_ratio * 0.4,
            coverage.coverage_ratio * 0.4,
            agreement.consensus_score * 0.2
        ]
        confidence_score = sum(confidence_components)
        
        # Identify gaps and uncertainty factors
        evidence_gaps = []
        uncertainty_factors = []
        
        if evidence_density.total_evidence_count < evidence_density.min_evidence_threshold:
            evidence_gaps.append("insufficient_evidence_count")
            uncertainty_factors.append("low_evidence_density")
        
        if not has_answer:
            evidence_gaps.append("missing_answer")
            uncertainty_factors.append("coverage_gap")
        
        if agreement.variance_penalty > 0.2:
            uncertainty_factors.append("model_disagreement")
        
        return ConfidenceQualityScore(
            level=AggregationLevel.QUESTION,
            confidence_score=confidence_score,
            quality_score=quality.composite_score,
            evidence_density=evidence_density,
            coverage_validation=coverage,
            model_agreement=agreement,
            quality_components=quality,
            evidence_gaps=evidence_gaps,
            uncertainty_factors=uncertainty_factors
        )
    
    def propagate_to_dimension_level(self, question_scores: List[ConfidenceQualityScore], 
                                   dimension_data: Dict[str, Any]) -> ConfidenceQualityScore:
        """Propagate question-level metrics to dimension level"""
        
        if not question_scores:
            return ConfidenceQualityScore(level=AggregationLevel.DIMENSION)
        
        # Aggregate evidence density
        total_evidence = sum(q.evidence_density.total_evidence_count for q in question_scores if q.evidence_density)
        total_questions = len(question_scores)
        
        dimension_density = EvidenceDensityMetrics(
            total_evidence_count=total_evidence,
            question_count=total_questions
        )
        
        # Aggregate coverage
        answered_count = sum(1 for q in question_scores if q.coverage_validation and q.coverage_validation.answered_questions > 0)
        critical_gaps = sum(q.coverage_validation.critical_gaps for q in question_scores if q.coverage_validation)
        
        dimension_coverage = CoverageValidationMetrics(
            answered_questions=answered_count,
            total_questions=total_questions,
            critical_gaps=critical_gaps
        )
        
        # Aggregate model agreement (average of consensus scores)
        consensus_scores = [q.model_agreement.consensus_score for q in question_scores if q.model_agreement]
        dimension_agreement = ModelAgreementMetrics(model_scores=consensus_scores)
        
        # Aggregate quality components (weighted by confidence)
        confidence_weights = [q.confidence_score for q in question_scores]
        total_weight = sum(confidence_weights) if confidence_weights else 1.0
        
        if total_weight > 0:
            weighted_credibility = sum(q.quality_components.credibility_score * q.confidence_score 
                                     for q in question_scores if q.quality_components) / total_weight
            weighted_recency = sum(q.quality_components.recency_weight * q.confidence_score 
                                 for q in question_scores if q.quality_components) / total_weight  
            weighted_authority = sum(q.quality_components.authority_ranking * q.confidence_score 
                                   for q in question_scores if q.quality_components) / total_weight
        else:
            weighted_credibility = weighted_recency = weighted_authority = 0.0
        
        dimension_quality = QualityComponents(
            credibility_score=weighted_credibility,
            recency_weight=weighted_recency,
            authority_ranking=weighted_authority
        )
        
        # Calculate dimension-level confidence
        confidence_components = [
            dimension_density.density_ratio * 0.4,
            dimension_coverage.coverage_ratio * 0.4,
            dimension_agreement.consensus_score * 0.2
        ]
        dimension_confidence = sum(confidence_components)
        
        # Aggregate evidence gaps and uncertainty factors
        all_gaps = []
        all_uncertainty = []
        for q in question_scores:
            all_gaps.extend(q.evidence_gaps)
            all_uncertainty.extend(q.uncertainty_factors)
        
        # Remove duplicates while preserving order
        unique_gaps = list(dict.fromkeys(all_gaps))
        unique_uncertainty = list(dict.fromkeys(all_uncertainty))
        
        return ConfidenceQualityScore(
            level=AggregationLevel.DIMENSION,
            confidence_score=dimension_confidence,
            quality_score=dimension_quality.composite_score,
            evidence_density=dimension_density,
            coverage_validation=dimension_coverage,
            model_agreement=dimension_agreement,
            quality_components=dimension_quality,
            evidence_gaps=unique_gaps,
            uncertainty_factors=unique_uncertainty
        )
    
    def propagate_to_point_level(self, dimension_scores: List[ConfidenceQualityScore],
                               point_data: Dict[str, Any]) -> ConfidenceQualityScore:
        """Propagate dimension-level metrics to point level (Decálogo point)"""
        
        if not dimension_scores:
            return ConfidenceQualityScore(level=AggregationLevel.POINT)
        
        # For point level, we use confidence-weighted aggregation
        confidence_weights = [d.confidence_score for d in dimension_scores]
        total_weight = sum(confidence_weights) if confidence_weights else 1.0
        
        if total_weight == 0:
            return ConfidenceQualityScore(level=AggregationLevel.POINT)
        
        # Weighted average of confidence scores
        point_confidence = sum(d.confidence_score * d.confidence_score for d in dimension_scores) / total_weight
        
        # Weighted average of quality scores
        point_quality = sum(d.quality_score * d.confidence_score for d in dimension_scores) / total_weight
        
        # Aggregate all evidence gaps and uncertainty factors
        all_gaps = []
        all_uncertainty = []
        for d in dimension_scores:
            all_gaps.extend(d.evidence_gaps)
            all_uncertainty.extend(d.uncertainty_factors)
        
        unique_gaps = list(dict.fromkeys(all_gaps))
        unique_uncertainty = list(dict.fromkeys(all_uncertainty))
        
        return ConfidenceQualityScore(
            level=AggregationLevel.POINT,
            confidence_score=point_confidence,
            quality_score=point_quality,
            evidence_gaps=unique_gaps,
            uncertainty_factors=unique_uncertainty
        )
    
    def propagate_to_meso_level(self, point_scores: List[ConfidenceQualityScore],
                              meso_data: Dict[str, Any]) -> ConfidenceQualityScore:
        """Propagate point-level metrics to meso level"""
        
        if not point_scores:
            return ConfidenceQualityScore(level=AggregationLevel.MESO)
        
        # Meso level uses harmonic mean to penalize low scores more heavily
        confidence_scores = [p.confidence_score for p in point_scores if p.confidence_score > 0]
        quality_scores = [p.quality_score for p in point_scores if p.quality_score > 0]
        
        if confidence_scores:
            # Harmonic mean formula: n / sum(1/xi)
            meso_confidence = len(confidence_scores) / sum(1/score for score in confidence_scores)
        else:
            meso_confidence = 0.0
        
        if quality_scores:
            meso_quality = len(quality_scores) / sum(1/score for score in quality_scores)
        else:
            meso_quality = 0.0
        
        # Aggregate gaps and uncertainty
        all_gaps = []
        all_uncertainty = []
        for p in point_scores:
            all_gaps.extend(p.evidence_gaps)
            all_uncertainty.extend(p.uncertainty_factors)
        
        unique_gaps = list(dict.fromkeys(all_gaps))
        unique_uncertainty = list(dict.fromkeys(all_uncertainty))
        
        return ConfidenceQualityScore(
            level=AggregationLevel.MESO,
            confidence_score=meso_confidence,
            quality_score=meso_quality,
            evidence_gaps=unique_gaps,
            uncertainty_factors=unique_uncertainty
        )
    
    def propagate_to_macro_level(self, meso_scores: List[ConfidenceQualityScore],
                               macro_data: Dict[str, Any]) -> ConfidenceQualityScore:
        """Propagate meso-level metrics to macro level"""
        
        if not meso_scores:
            return ConfidenceQualityScore(level=AggregationLevel.MACRO)
        
        # Macro level uses geometric mean for balanced aggregation
        confidence_scores = [m.confidence_score for m in meso_scores if m.confidence_score > 0]
        quality_scores = [m.quality_score for m in meso_scores if m.quality_score > 0]
        
        if confidence_scores:
            # Geometric mean formula: (x1 * x2 * ... * xn)^(1/n)
            confidence_product = 1.0
            for score in confidence_scores:
                confidence_product *= score
            macro_confidence = confidence_product ** (1/len(confidence_scores))
        else:
            macro_confidence = 0.0
        
        if quality_scores:
            quality_product = 1.0
            for score in quality_scores:
                quality_product *= score
            macro_quality = quality_product ** (1/len(quality_scores))
        else:
            macro_quality = 0.0
        
        # Aggregate gaps and uncertainty
        all_gaps = []
        all_uncertainty = []
        for m in meso_scores:
            all_gaps.extend(m.evidence_gaps)
            all_uncertainty.extend(m.uncertainty_factors)
        
        unique_gaps = list(dict.fromkeys(all_gaps))
        unique_uncertainty = list(dict.fromkeys(all_uncertainty))
        
        return ConfidenceQualityScore(
            level=AggregationLevel.MACRO,
            confidence_score=macro_confidence,
            quality_score=macro_quality,
            evidence_gaps=unique_gaps,
            uncertainty_factors=unique_uncertainty
        )


class ArtifactMetricsIntegrator:
    """Integrates confidence and quality metrics into existing artifact generation system"""
    
    def __init__(self):
        self.calculator = ConfidenceQualityCalculator()
    
    def add_metrics_to_question_artifact(self, question_artifact: Dict[str, Any]) -> Dict[str, Any]:
        """Add confidence and quality metrics to question-level artifact"""
        
        # Calculate metrics
        metrics = self.calculator.calculate_question_level_metrics(question_artifact)
        
        # Add to artifact
        enhanced_artifact = question_artifact.copy()
        enhanced_artifact.update({
            'confidence_score': metrics.confidence_score,
            'quality_score': metrics.quality_score,
            'metrics_metadata': {
                'evidence_density': {
                    'total_evidence': metrics.evidence_density.total_evidence_count if metrics.evidence_density else 0,
                    'density_ratio': metrics.evidence_density.density_ratio if metrics.evidence_density else 0.0,
                },
                'coverage_validation': {
                    'answered_questions': metrics.coverage_validation.answered_questions if metrics.coverage_validation else 0,
                    'coverage_ratio': metrics.coverage_validation.coverage_ratio if metrics.coverage_validation else 0.0,
                },
                'model_agreement': {
                    'consensus_score': metrics.model_agreement.consensus_score if metrics.model_agreement else 0.0,
                    'model_count': len(metrics.model_agreement.model_scores) if metrics.model_agreement else 0,
                },
                'evidence_gaps': metrics.evidence_gaps,
                'uncertainty_factors': metrics.uncertainty_factors,
                'calculation_timestamp': metrics.calculation_timestamp,
            }
        })
        
        return enhanced_artifact
    
    def add_metrics_to_dimension_artifact(self, dimension_artifact: Dict[str, Any],
                                        question_scores: List[ConfidenceQualityScore]) -> Dict[str, Any]:
        """Add confidence and quality metrics to dimension-level artifact"""
        
        metrics = self.calculator.propagate_to_dimension_level(question_scores, dimension_artifact)
        
        enhanced_artifact = dimension_artifact.copy()
        enhanced_artifact.update({
            'confidence_score': metrics.confidence_score,
            'quality_score': metrics.quality_score,
            'metrics_metadata': {
                'question_count': len(question_scores),
                'evidence_density_ratio': metrics.evidence_density.density_ratio if metrics.evidence_density else 0.0,
                'coverage_ratio': metrics.coverage_validation.coverage_ratio if metrics.coverage_validation else 0.0,
                'evidence_gaps': metrics.evidence_gaps,
                'uncertainty_factors': metrics.uncertainty_factors,
                'calculation_timestamp': metrics.calculation_timestamp,
            }
        })
        
        return enhanced_artifact
    
    def add_metrics_to_point_artifact(self, point_artifact: Dict[str, Any],
                                    dimension_scores: List[ConfidenceQualityScore]) -> Dict[str, Any]:
        """Add confidence and quality metrics to point-level artifact"""
        
        metrics = self.calculator.propagate_to_point_level(dimension_scores, point_artifact)
        
        enhanced_artifact = point_artifact.copy()
        enhanced_artifact.update({
            'confidence_score': metrics.confidence_score,
            'quality_score': metrics.quality_score,
            'metrics_metadata': {
                'dimension_count': len(dimension_scores),
                'evidence_gaps': metrics.evidence_gaps,
                'uncertainty_factors': metrics.uncertainty_factors,
                'calculation_timestamp': metrics.calculation_timestamp,
            }
        })
        
        return enhanced_artifact
    
    def add_metrics_to_meso_artifact(self, meso_artifact: Dict[str, Any],
                                   point_scores: List[ConfidenceQualityScore]) -> Dict[str, Any]:
        """Add confidence and quality metrics to meso-level artifact"""
        
        metrics = self.calculator.propagate_to_meso_level(point_scores, meso_artifact)
        
        enhanced_artifact = meso_artifact.copy()
        enhanced_artifact.update({
            'confidence_score': metrics.confidence_score,
            'quality_score': metrics.quality_score,
            'metrics_metadata': {
                'point_count': len(point_scores),
                'aggregation_method': 'harmonic_mean',
                'evidence_gaps': metrics.evidence_gaps,
                'uncertainty_factors': metrics.uncertainty_factors,
                'calculation_timestamp': metrics.calculation_timestamp,
            }
        })
        
        return enhanced_artifact
    
    def add_metrics_to_macro_artifact(self, macro_artifact: Dict[str, Any],
                                    meso_scores: List[ConfidenceQualityScore]) -> Dict[str, Any]:
        """Add confidence and quality metrics to macro-level artifact"""
        
        metrics = self.calculator.propagate_to_macro_level(meso_scores, macro_artifact)
        
        enhanced_artifact = macro_artifact.copy()
        enhanced_artifact.update({
            'confidence_score': metrics.confidence_score,
            'quality_score': metrics.quality_score,
            'metrics_metadata': {
                'meso_count': len(meso_scores),
                'aggregation_method': 'geometric_mean',
                'evidence_gaps': metrics.evidence_gaps,
                'uncertainty_factors': metrics.uncertainty_factors,
                'calculation_timestamp': metrics.calculation_timestamp,
            }
        })
        
        return enhanced_artifact


def validate_metrics_bounded(score: float, metric_name: str) -> float:
    """Validate that metrics are bounded between 0.0 and 1.0"""
    if not isinstance(score, (int, float)):
        logger.warning(f"Non-numeric score for {metric_name}: {score}, defaulting to 0.0")
        return 0.0
    
    if score < 0.0:
        logger.warning(f"Score below bounds for {metric_name}: {score}, clamping to 0.0")
        return 0.0
    
    if score > 1.0:
        logger.warning(f"Score above bounds for {metric_name}: {score}, clamping to 1.0")  
        return 1.0
    
    return float(score)


# Convenience functions for backward compatibility
def calculate_question_confidence(question_data: Dict[str, Any]) -> float:
    """Calculate confidence score for a single question"""
    calculator = ConfidenceQualityCalculator()
    metrics = calculator.calculate_question_level_metrics(question_data)
    return validate_metrics_bounded(metrics.confidence_score, "question_confidence")


def calculate_question_quality(question_data: Dict[str, Any]) -> float:
    """Calculate quality score for a single question"""
    calculator = ConfidenceQualityCalculator()
    metrics = calculator.calculate_question_level_metrics(question_data)
    return validate_metrics_bounded(metrics.quality_score, "question_quality")