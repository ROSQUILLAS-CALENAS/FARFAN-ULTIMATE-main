"""
Canonical Flow Alias: GA
MesoAggregator with Total Ordering and Deterministic Processing

Source: G_aggregation_reporting/meso_aggregator.py
Stage: aggregation_reporting  
Code: GA
"""

import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from collections import OrderedDict

# Import total ordering base
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from total_ordering_base import TotalOrderingBase, DeterministicCollectionMixin

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Strategy for evidence aggregation"""
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTING = "majority_voting"
    CONFIDENCE_BASED = "confidence_based"
    HIERARCHICAL = "hierarchical"


class AggregationScope(Enum):
    """Scope of aggregation"""
    DIMENSION_LEVEL = "dimension_level"
    QUESTION_LEVEL = "question_level"
    GLOBAL_LEVEL = "global_level"


@dataclass
class AggregationResult:
    """Result of evidence aggregation process"""
    aggregation_id: str
    scope: AggregationScope
    strategy: AggregationStrategy
    aggregated_score: float
    confidence_level: str
    evidence_count: int
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    summary_metrics: Dict[str, Any] = field(default_factory=dict)
    processing_notes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Ensure deterministic ordering
        self.dimension_scores = OrderedDict(sorted(self.dimension_scores.items()))
        self.processing_notes = sorted(self.processing_notes)


class MesoAggregator(TotalOrderingBase, DeterministicCollectionMixin):
    """
    MesoAggregator for evidence aggregation and consolidation with comprehensive audit logging.
    
    This component aggregates evidence from multiple sources and dimensions,
    providing consolidated scores and metrics for comprehensive analysis.
    
    Key Features:
    - Multi-dimensional evidence aggregation
    - Configurable aggregation strategies
    - Confidence-weighted scoring
    - Hierarchical aggregation support
    - Complete execution traceability through audit logs
    """
    
    def __init__(self, strategy: AggregationStrategy = AggregationStrategy.WEIGHTED_AVERAGE):
        super().__init__("MesoAggregator")
        
        self.strategy = strategy
        self.stage_name = "G_aggregation_reporting"  # Set stage name for audit logging
        
        # Configuration parameters
        self.min_evidence_threshold = 3
        self.confidence_weight = 0.3
        self.recency_weight = 0.2
        self.authority_weight = 0.5
        
        # Aggregation statistics
        self.aggregation_stats = {
            "total_aggregations": 0,
            "successful_aggregations": 0,
            "failed_aggregations": 0,
            "average_evidence_per_aggregation": 0.0,
        }
        
        # Update state hash
        self.update_state_hash(self._get_initial_state())
    
    def _get_initial_state(self) -> Dict[str, Any]:
        """Get initial state for hash calculation"""
        return {
            "strategy": self.strategy.value,
            "configuration": {
                "min_evidence_threshold": self.min_evidence_threshold,
                "confidence_weight": self.confidence_weight,
                "recency_weight": self.recency_weight,
                "authority_weight": self.authority_weight,
            }
        }
    
    def process(self, data: Any = None, context: Any = None) -> Dict[str, Any]:
        """
        Main processing function for evidence aggregation with comprehensive audit logging.
        
        Args:
            data: Input data containing evidence to aggregate
            context: Processing context with aggregation parameters
            
        Returns:
            Deterministic aggregation results with audit metadata
        """
        # Use audit-enabled processing if available
        if hasattr(self, 'process_with_audit'):
            return self.process_with_audit(data, context)
        
        # Fallback to standard processing
        operation_id = self.generate_operation_id("process", {"data": data, "context": context})
        
        try:
            # Canonicalize inputs
            canonical_data = self.canonicalize_data(data) if data else {}
            canonical_context = self.canonicalize_data(context) if context else {}
            
            # Extract evidence for aggregation
            evidence_groups = self._extract_evidence_groups_deterministic(canonical_data)
            
            # Perform aggregation for each group
            aggregation_results = []
            for group_id, evidence_list in sorted(evidence_groups.items()):
                result = self._aggregate_evidence_group_deterministic(
                    group_id, evidence_list, canonical_context
                )
                aggregation_results.append(result)
            
            # Generate consolidated output
            output = self._generate_consolidated_output(aggregation_results, operation_id)
            
            # Update statistics
            self._update_aggregation_stats(aggregation_results)
            
            # Update state hash
            self.update_state_hash(output)
            
            return output
            
        except Exception as e:
            error_output = {
                "component": self.component_name,
                "operation_id": operation_id,
                "error": str(e),
                "status": "error",
                "timestamp": self._get_deterministic_timestamp(),
            }
            return self.sort_dict_by_keys(error_output)
    
    def _extract_evidence_groups_deterministic(self, data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract and group evidence for aggregation with stable ordering"""
        evidence_groups = {}
        
        # Handle structured evidence lists
        if "evidence_list" in data:
            evidence_list = data["evidence_list"]
            if isinstance(evidence_list, list):
                for evidence in evidence_list:
                    if isinstance(evidence, dict):
                        dimension = evidence.get("dimension", "default")
                        if dimension not in evidence_groups:
                            evidence_groups[dimension] = []
                        evidence_groups[dimension].append(evidence)
        
        # Handle dimension-based grouping
        if "dimensions" in data:
            dimensions_data = data["dimensions"]
            if isinstance(dimensions_data, dict):
                for dim_name, dim_evidence in sorted(dimensions_data.items()):
                    if isinstance(dim_evidence, list):
                        evidence_groups[dim_name] = dim_evidence
                    elif isinstance(dim_evidence, dict):
                        evidence_groups[dim_name] = [dim_evidence]
        
        # Handle single evidence (create default group)
        if "evidence" in data and not evidence_groups:
            evidence_groups["default"] = [data["evidence"]]
        
        # Ensure all groups have deterministic ordering
        for group_id in evidence_groups:
            evidence_groups[group_id] = sorted(
                evidence_groups[group_id],
                key=lambda x: str(x.get("evidence_id", "")) + str(x.get("text", ""))
            )
        
        return evidence_groups
    
    def _aggregate_evidence_group_deterministic(
        self, 
        group_id: str, 
        evidence_list: List[Dict[str, Any]], 
        context: Dict[str, Any]
    ) -> AggregationResult:
        """Aggregate a group of evidence with deterministic processing"""
        
        if len(evidence_list) < self.min_evidence_threshold:
            # Handle insufficient evidence
            return AggregationResult(
                aggregation_id=self.generate_stable_id(
                    {"group_id": group_id, "count": len(evidence_list)}, 
                    prefix="agg"
                ),
                scope=AggregationScope.DIMENSION_LEVEL,
                strategy=self.strategy,
                aggregated_score=0.0,
                confidence_level="low",
                evidence_count=len(evidence_list),
                processing_notes=[f"Insufficient evidence: {len(evidence_list)} < {self.min_evidence_threshold}"]
            )
        
        # Extract scores and weights
        scores = []
        weights = []
        dimension_scores = {}
        
        for evidence in evidence_list:
            # Extract base score
            base_score = self._extract_evidence_score(evidence)
            scores.append(base_score)
            
            # Calculate weight based on evidence properties
            weight = self._calculate_evidence_weight(evidence)
            weights.append(weight)
            
            # Track dimension-specific scores
            evidence_dim = evidence.get("dimension", group_id)
            if evidence_dim not in dimension_scores:
                dimension_scores[evidence_dim] = []
            dimension_scores[evidence_dim].append(base_score)
        
        # Perform aggregation based on strategy
        aggregated_score = self._apply_aggregation_strategy(scores, weights)
        
        # Calculate dimension averages
        dim_averages = {
            dim: sum(scores) / len(scores) if scores else 0.0
            for dim, scores in dimension_scores.items()
        }
        
        # Determine confidence level
        confidence_level = self._calculate_confidence_level(scores, weights)
        
        # Generate summary metrics
        summary_metrics = {
            "score_variance": self._calculate_variance(scores),
            "weight_sum": sum(weights),
            "max_score": max(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
        }
        
        return AggregationResult(
            aggregation_id=self.generate_stable_id(
                {"group_id": group_id, "evidence_count": len(evidence_list), "strategy": self.strategy.value},
                prefix="agg"
            ),
            scope=AggregationScope.DIMENSION_LEVEL,
            strategy=self.strategy,
            aggregated_score=aggregated_score,
            confidence_level=confidence_level,
            evidence_count=len(evidence_list),
            dimension_scores=dim_averages,
            summary_metrics=summary_metrics,
            processing_notes=[]
        )
    
    def _extract_evidence_score(self, evidence: Dict[str, Any]) -> float:
        """Extract numeric score from evidence with fallbacks"""
        
        # Try different score field names
        score_fields = ["overall_score", "score", "confidence_score", "quality_score", "relevance_score"]
        
        for field in score_fields:
            if field in evidence:
                value = evidence[field]
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    try:
                        return float(value)
                    except ValueError:
                        continue
        
        # Try scoring nested objects
        if "scoring" in evidence and isinstance(evidence["scoring"], dict):
            scoring = evidence["scoring"]
            for field in score_fields:
                if field in scoring:
                    value = scoring[field]
                    if isinstance(value, (int, float)):
                        return float(value)
        
        # Default score based on evidence length/quality
        text = evidence.get("text", "")
        if text:
            return min(1.0, len(text) / 100.0)  # Basic heuristic
        
        return 0.5  # Neutral default
    
    def _calculate_evidence_weight(self, evidence: Dict[str, Any]) -> float:
        """Calculate weight for evidence based on multiple factors"""
        
        base_weight = 1.0
        
        # Authority weight
        authority_score = evidence.get("authority_score", 0.5)
        if isinstance(authority_score, (int, float)):
            base_weight *= (1.0 + self.authority_weight * (authority_score - 0.5))
        
        # Confidence weight
        confidence = evidence.get("confidence_score", evidence.get("confidence", 0.5))
        if isinstance(confidence, (int, float)):
            base_weight *= (1.0 + self.confidence_weight * (confidence - 0.5))
        
        # Recency weight (if timestamp available)
        if "timestamp" in evidence or "creation_timestamp" in evidence:
            base_weight *= (1.0 + self.recency_weight * 0.1)  # Small boost for timestamped
        
        return max(0.1, min(2.0, base_weight))  # Clamp between 0.1 and 2.0
    
    def _apply_aggregation_strategy(self, scores: List[float], weights: List[float]) -> float:
        """Apply the configured aggregation strategy"""
        
        if not scores:
            return 0.0
        
        if self.strategy == AggregationStrategy.WEIGHTED_AVERAGE:
            if sum(weights) > 0:
                return sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            else:
                return sum(scores) / len(scores)
        
        elif self.strategy == AggregationStrategy.MAJORITY_VOTING:
            # Use median as majority representative
            sorted_scores = sorted(scores)
            n = len(sorted_scores)
            if n % 2 == 0:
                return (sorted_scores[n//2 - 1] + sorted_scores[n//2]) / 2
            else:
                return sorted_scores[n//2]
        
        elif self.strategy == AggregationStrategy.CONFIDENCE_BASED:
            # Weight by confidence (using weights as confidence)
            total_confidence = sum(weights)
            if total_confidence > 0:
                return sum(s * w for s, w in zip(scores, weights)) / total_confidence
            else:
                return sum(scores) / len(scores)
        
        elif self.strategy == AggregationStrategy.HIERARCHICAL:
            # Use highest weighted score
            if weights:
                max_weight_idx = weights.index(max(weights))
                return scores[max_weight_idx]
            else:
                return max(scores)
        
        else:
            # Default to simple average
            return sum(scores) / len(scores)
    
    def _calculate_confidence_level(self, scores: List[float], weights: List[float]) -> str:
        """Calculate overall confidence level for aggregation"""
        
        if not scores:
            return "low"
        
        # Calculate score consistency (low variance = high confidence)
        variance = self._calculate_variance(scores)
        avg_weight = sum(weights) / len(weights) if weights else 1.0
        
        # Combine factors
        confidence_score = (1.0 - min(1.0, variance)) * avg_weight
        
        if confidence_score >= 0.8:
            return "high"
        elif confidence_score >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        squared_diffs = [(x - mean) ** 2 for x in values]
        return sum(squared_diffs) / len(values)
    
    def _generate_consolidated_output(self, results: List[AggregationResult], operation_id: str) -> Dict[str, Any]:
        """Generate consolidated output from aggregation results"""
        
        # Sort results by aggregation_id for deterministic ordering
        sorted_results = sorted(results, key=lambda r: r.aggregation_id)
        
        # Calculate global metrics
        total_evidence = sum(r.evidence_count for r in results)
        avg_score = sum(r.aggregated_score for r in results) / len(results) if results else 0.0
        
        # Confidence distribution
        confidence_dist = {"high": 0, "medium": 0, "low": 0}
        for result in results:
            confidence_dist[result.confidence_level] += 1
        
        # Import confidence and quality metrics
        from confidence_quality_metrics import ArtifactMetricsIntegrator
        
        integrator = ArtifactMetricsIntegrator()
        
        # Calculate meso-level confidence and quality metrics
        dimension_scores = []  # Would be populated from dimension-level artifacts
        meso_metrics = integrator.calculator.propagate_to_meso_level(dimension_scores, {})
        
        output = {
            "component": self.component_name,
            "component_id": self.component_id,
            "operation_id": operation_id,
            "aggregation_strategy": self.strategy.value,
            "confidence_score": meso_metrics.confidence_score,
            "quality_score": meso_metrics.quality_score,
            "results": [self._serialize_aggregation_result(r) for r in sorted_results],
            "global_metrics": {
                "total_evidence_aggregated": total_evidence,
                "average_aggregated_score": avg_score,
                "confidence_distribution": confidence_dist,
                "successful_aggregations": len(results),
            },
            "metadata": self.get_deterministic_metadata(),
            "status": "success",
            "timestamp": self._get_deterministic_timestamp(),
        }
        
        return self.sort_dict_by_keys(output)
    
    def _serialize_aggregation_result(self, result: AggregationResult) -> Dict[str, Any]:
        """Convert AggregationResult to serializable dictionary"""
        return {
            "aggregation_id": result.aggregation_id,
            "scope": result.scope.value,
            "strategy": result.strategy.value,
            "aggregated_score": result.aggregated_score,
            "confidence_level": result.confidence_level,
            "evidence_count": result.evidence_count,
            "dimension_scores": self.sort_dict_by_keys(result.dimension_scores),
            "summary_metrics": self.sort_dict_by_keys(result.summary_metrics),
            "processing_notes": result.processing_notes,
        }
    
    def _update_aggregation_stats(self, results: List[AggregationResult]) -> None:
        """Update internal aggregation statistics"""
        self.aggregation_stats["total_aggregations"] += len(results)
        self.aggregation_stats["successful_aggregations"] += len([r for r in results if r.evidence_count >= self.min_evidence_threshold])
        self.aggregation_stats["failed_aggregations"] += len([r for r in results if r.evidence_count < self.min_evidence_threshold])
        
        if results:
            total_evidence = sum(r.evidence_count for r in results)
            self.aggregation_stats["average_evidence_per_aggregation"] = total_evidence / len(results)


# Maintain backward compatibility
def process(data=None, context=None):
    """Backward compatible process function"""
    aggregator = MesoAggregator()
    return aggregator.process(data, context)