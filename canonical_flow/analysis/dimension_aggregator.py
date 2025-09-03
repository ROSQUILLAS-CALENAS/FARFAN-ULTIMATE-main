"""
Dimension Aggregator for DE-1, DE-2, DE-3, DE-4 Analysis

This component aggregates question-level scores into dimension-level metrics,
applying causal correction factors and quality penalties as needed.
Generates deterministic artifacts in canonical_flow/analysis/.
"""

import json
import logging
import os
# # # from collections import OrderedDict  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Set, Tuple  # Module not found  # Module not found  # Module not found
# # # from uuid import uuid4  # Module not found  # Module not found  # Module not found
import hashlib

# import numpy as np  # Removed to avoid dependency

logger = logging.getLogger(__name__)


class DimensionStatus(Enum):
    """Status flags for dimension processing"""
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    NO_QUESTIONS = "no_questions"
    ERROR = "error"


class CorrectionType(Enum):
    """Types of correction factors"""
    BASELINE_DEVIATION = "baseline_deviation"
    QUALITY_PENALTY = "quality_penalty"
    CAUSAL_ADJUSTMENT = "causal_adjustment"


@dataclass
class CorrectionFactor:
    """Represents a correction factor applied to scores"""
    correction_type: CorrectionType
    dimension: str
    original_score: float
    adjustment: float
    final_score: float
    reason: str
    evidence_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionBreakdown:
    """Question-level contribution to dimension score"""
    question_id: str
    raw_score: float
    weighted_score: float
    weight: float
    evidence_count: int
    evidence_quality: float
    contribution_percentage: float


@dataclass
class DimensionResult:
    """Aggregated result for a single dimension"""
    dimension_id: str
    raw_score: float
    adjusted_score: float
    status: DimensionStatus
    question_count: int
    expected_question_count: int
    completion_percentage: float
    
    # Quality metrics
    evidence_density: float
    average_evidence_quality: float
    quality_penalty_applied: float
    
    # Correction factors
    correction_factors: List[CorrectionFactor] = field(default_factory=list)
    question_breakdown: List[QuestionBreakdown] = field(default_factory=list)
    
    # Metadata
    processing_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    artifact_id: str = field(default_factory=lambda: str(uuid4())[:8])


class DimensionAggregator:
    """
    Aggregates question-level scores into dimension-level metrics.
    
    Expected dimension question counts:
    - DE-1: 6 questions
    - DE-2: 21 questions  
    - DE-3: 8 questions
    - DE-4: 8 questions
    
    Features:
    - Weighted mean score computation
    - Causal correction factors for DE-1
    - Quality penalties based on evidence density
    - Deterministic JSON artifact generation
    - Edge case handling for missing questions
    """
    
    # Expected question counts per dimension
    EXPECTED_DIMENSION_COUNTS = {
        "DE-1": 6,
        "DE-2": 21,
        "DE-3": 8,
        "DE-4": 8
    }
    
    # Quality thresholds
    MIN_EVIDENCE_DENSITY = 0.3
    MIN_EVIDENCE_QUALITY = 0.5
    
    # Correction factor parameters
# # #     BASELINE_DEVIATION_THRESHOLD = 0.15  # 15% deviation from baseline  # Module not found  # Module not found  # Module not found
    CAUSAL_CORRECTION_FACTOR = 0.1  # 10% adjustment for DE-1
    QUALITY_PENALTY_FACTOR = 0.2   # 20% penalty for low quality
    
    def __init__(self, output_dir: str = "canonical_flow/analysis"):
        """
        Initialize DimensionAggregator.
        
        Args:
            output_dir: Directory for artifact generation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing statistics
        self.stats = {
            "total_dimensions_processed": 0,
            "corrections_applied": 0,
            "quality_penalties_applied": 0,
            "validation_errors": 0
        }
        
        logger.info(f"DimensionAggregator initialized with output_dir: {self.output_dir}")
    
    def aggregate_dimensions(self, question_data: Dict[str, Any]) -> Dict[str, DimensionResult]:
        """
        Aggregate question-level data into dimension-level results.
        
        Args:
            question_data: Dictionary containing question scores and metadata
                          Expected format: {
                              "questions": [
                                  {
                                      "question_id": str,
                                      "dimension": str,  # DE-1, DE-2, DE-3, DE-4
                                      "score": float,
                                      "weight": float,
                                      "evidence_count": int,
                                      "evidence_quality": float
                                  }
                              ],
                              "metadata": {...}
                          }
        
        Returns:
            Dict mapping dimension IDs to DimensionResult objects
        """
        logger.info("Starting dimension aggregation")
        
        # Validate input data
        self._validate_input_data(question_data)
        
        # Group questions by dimension
        dimension_questions = self._group_questions_by_dimension(question_data["questions"])
        
        # Process each dimension
        results = {}
        for dimension_id in ["DE-1", "DE-2", "DE-3", "DE-4"]:
            questions = dimension_questions.get(dimension_id, [])
            result = self._process_dimension(dimension_id, questions)
            results[dimension_id] = result
            
        # Update statistics
        self.stats["total_dimensions_processed"] = len(results)
        
        # Generate artifacts
        self._generate_artifacts(results, question_data.get("metadata", {}))
        
        logger.info(f"Dimension aggregation completed. Processed {len(results)} dimensions")
        return results
    
    def _validate_input_data(self, data: Dict[str, Any]) -> None:
        """Validate input data structure and content."""
        if "questions" not in data:
            raise ValueError("Input data must contain 'questions' key")
        
        questions = data["questions"]
        if not isinstance(questions, list):
            raise ValueError("Questions must be a list")
        
        # Validate each question
        required_fields = ["question_id", "dimension", "score", "weight", 
                          "evidence_count", "evidence_quality"]
        
        for i, question in enumerate(questions):
            for field in required_fields:
                if field not in question:
                    raise ValueError(f"Question {i} missing required field: {field}")
            
            # Validate dimension ID
            if question["dimension"] not in self.EXPECTED_DIMENSION_COUNTS:
                raise ValueError(f"Invalid dimension ID: {question['dimension']}")
        
        logger.debug(f"Input validation passed for {len(questions)} questions")
    
    def _group_questions_by_dimension(self, questions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group questions by dimension ID."""
        grouped = {dim_id: [] for dim_id in self.EXPECTED_DIMENSION_COUNTS.keys()}
        
        for question in questions:
            dimension = question["dimension"]
            grouped[dimension].append(question)
        
        # Log counts for validation
        for dim_id, dim_questions in grouped.items():
            expected = self.EXPECTED_DIMENSION_COUNTS[dim_id]
            actual = len(dim_questions)
            logger.debug(f"Dimension {dim_id}: {actual}/{expected} questions")
        
        return grouped
    
    def _process_dimension(self, dimension_id: str, questions: List[Dict[str, Any]]) -> DimensionResult:
        """Process a single dimension and compute aggregated metrics."""
        logger.debug(f"Processing dimension {dimension_id} with {len(questions)} questions")
        
        expected_count = self.EXPECTED_DIMENSION_COUNTS[dimension_id]
        
        # Handle edge case: no questions
        if not questions:
            return DimensionResult(
                dimension_id=dimension_id,
                raw_score=0.0,
                adjusted_score=0.0,
                status=DimensionStatus.NO_QUESTIONS,
                question_count=0,
                expected_question_count=expected_count,
                completion_percentage=0.0,
                evidence_density=0.0,
                average_evidence_quality=0.0,
                quality_penalty_applied=0.0
            )
        
        # Compute weighted mean score
        raw_score, question_breakdown = self._compute_weighted_mean(questions)
        
        # Calculate evidence metrics
        evidence_density = self._calculate_evidence_density(questions)
        avg_evidence_quality = self._calculate_average_evidence_quality(questions)
        
        # Initialize result
        result = DimensionResult(
            dimension_id=dimension_id,
            raw_score=raw_score,
            adjusted_score=raw_score,  # Will be modified by corrections
            status=DimensionStatus.COMPLETE if len(questions) == expected_count else DimensionStatus.INCOMPLETE,
            question_count=len(questions),
            expected_question_count=expected_count,
            completion_percentage=len(questions) / expected_count * 100,
            evidence_density=evidence_density,
            average_evidence_quality=avg_evidence_quality,
            quality_penalty_applied=0.0,
            question_breakdown=question_breakdown
        )
        
        # Apply correction factors
        self._apply_correction_factors(result, questions)
        
        return result
    
    def _compute_weighted_mean(self, questions: List[Dict[str, Any]]) -> Tuple[float, List[QuestionBreakdown]]:
        """Compute weighted mean score and generate question breakdown."""
        if not questions:
            return 0.0, []
        
        total_weighted_score = 0.0
        total_weight = 0.0
        breakdown = []
        
        for question in questions:
            score = question["score"]
            weight = question["weight"]
            weighted_score = score * weight
            
            total_weighted_score += weighted_score
            total_weight += weight
            
            breakdown.append(QuestionBreakdown(
                question_id=question["question_id"],
                raw_score=score,
                weighted_score=weighted_score,
                weight=weight,
                evidence_count=question["evidence_count"],
                evidence_quality=question["evidence_quality"],
                contribution_percentage=0.0  # Will be calculated after totals
            ))
        
        # Calculate final weighted mean
        weighted_mean = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Calculate contribution percentages
        for item in breakdown:
            item.contribution_percentage = (item.weighted_score / total_weighted_score * 100) if total_weighted_score > 0 else 0.0
        
        return weighted_mean, breakdown
    
    def _calculate_evidence_density(self, questions: List[Dict[str, Any]]) -> float:
        """Calculate evidence density metric."""
        if not questions:
            return 0.0
        
        total_evidence = sum(q["evidence_count"] for q in questions)
        return total_evidence / len(questions)
    
    def _calculate_average_evidence_quality(self, questions: List[Dict[str, Any]]) -> float:
        """Calculate average evidence quality."""
        if not questions:
            return 0.0
        
        total_quality = sum(q["evidence_quality"] for q in questions)
        return total_quality / len(questions)
    
    def _apply_correction_factors(self, result: DimensionResult, questions: List[Dict[str, Any]]) -> None:
        """Apply correction factors to dimension score."""
        original_score = result.adjusted_score
        
        # Apply quality penalty if evidence density is below threshold
        if result.evidence_density < self.MIN_EVIDENCE_DENSITY:
            penalty = self.QUALITY_PENALTY_FACTOR
            quality_adjustment = -original_score * penalty
            
            correction = CorrectionFactor(
                correction_type=CorrectionType.QUALITY_PENALTY,
                dimension=result.dimension_id,
                original_score=original_score,
                adjustment=quality_adjustment,
                final_score=original_score + quality_adjustment,
                reason=f"Evidence density {result.evidence_density:.3f} below threshold {self.MIN_EVIDENCE_DENSITY}",
                evidence_metrics={
                    "evidence_density": result.evidence_density,
                    "threshold": self.MIN_EVIDENCE_DENSITY,
                    "penalty_factor": penalty
                }
            )
            
            result.correction_factors.append(correction)
            result.adjusted_score += quality_adjustment
            result.quality_penalty_applied = abs(quality_adjustment)
            self.stats["quality_penalties_applied"] += 1
        
        # Apply causal correction for DE-1 if baseline deviation detected
        if result.dimension_id == "DE-1":
            baseline_deviation = self._detect_baseline_deviation(result, questions)
            if baseline_deviation > self.BASELINE_DEVIATION_THRESHOLD:
                causal_adjustment = -result.adjusted_score * self.CAUSAL_CORRECTION_FACTOR
                
                correction = CorrectionFactor(
                    correction_type=CorrectionType.CAUSAL_ADJUSTMENT,
                    dimension=result.dimension_id,
                    original_score=result.adjusted_score,
                    adjustment=causal_adjustment,
                    final_score=result.adjusted_score + causal_adjustment,
                    reason=f"Baseline deviation {baseline_deviation:.3f} exceeds threshold {self.BASELINE_DEVIATION_THRESHOLD}",
                    evidence_metrics={
                        "baseline_deviation": baseline_deviation,
                        "threshold": self.BASELINE_DEVIATION_THRESHOLD,
                        "correction_factor": self.CAUSAL_CORRECTION_FACTOR
                    }
                )
                
                result.correction_factors.append(correction)
                result.adjusted_score += causal_adjustment
                self.stats["corrections_applied"] += 1
        
        logger.debug(f"Applied {len(result.correction_factors)} correction factors to {result.dimension_id}")
    
    def _detect_baseline_deviation(self, result: DimensionResult, questions: List[Dict[str, Any]]) -> float:
        """
        Detect baseline deviation for causal correction.
        
        This is a simplified implementation that computes the standard deviation
        of question scores as a proxy for baseline deviation.
        """
        if len(questions) < 2:
            return 0.0
        
        scores = [q["score"] for q in questions]
        mean_score = sum(scores) / len(scores)
        
        # Calculate standard deviation manually
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        std_dev = variance ** 0.5
        
        # Normalized deviation
        deviation = std_dev / mean_score if mean_score > 0 else 0.0
        return deviation
    
    def _generate_artifacts(self, results: Dict[str, DimensionResult], metadata: Dict[str, Any]) -> None:
        """Generate deterministic JSON artifacts."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive artifact
        artifact = {
            "generation_metadata": {
                "timestamp": timestamp,
                "component": "DimensionAggregator",
                "version": "1.0.0",
                "input_metadata": metadata
            },
            "processing_statistics": dict(self.stats),
            "validation_summary": self._generate_validation_summary(results),
            "dimension_results": self._serialize_results(results),
            "quality_metrics": self._compute_quality_metrics(results),
            "correction_summary": self._summarize_corrections(results)
        }
        
        # Generate artifact hash for deterministic naming
        artifact_content = json.dumps(artifact, sort_keys=True, separators=(',', ':'))
        artifact_hash = hashlib.sha256(artifact_content.encode()).hexdigest()[:12]
        
        # Save main artifact
        artifact_path = self.output_dir / f"dimension_aggregation_{timestamp}_{artifact_hash}.json"
        with open(artifact_path, 'w', encoding='utf-8') as f:
            json.dump(artifact, f, indent=2, sort_keys=True, ensure_ascii=False)
        
        # Save individual dimension artifacts
        for dim_id, result in results.items():
            dim_artifact = {
                "dimension_id": dim_id,
                "result": self._serialize_single_result(result),
                "timestamp": timestamp
            }
            
            dim_path = self.output_dir / f"dimension_{dim_id.lower()}_{timestamp}.json"
            with open(dim_path, 'w', encoding='utf-8') as f:
                json.dump(dim_artifact, f, indent=2, sort_keys=True, ensure_ascii=False)
        
        logger.info(f"Generated artifacts: {artifact_path}")
    
    def _generate_validation_summary(self, results: Dict[str, DimensionResult]) -> Dict[str, Any]:
        """Generate validation summary for dimension counts."""
        summary = {
            "expected_distribution": dict(self.EXPECTED_DIMENSION_COUNTS),
            "actual_distribution": {},
            "validation_status": "PASS",
            "validation_errors": []
        }
        
        for dim_id, result in results.items():
            expected = self.EXPECTED_DIMENSION_COUNTS[dim_id]
            actual = result.question_count
            summary["actual_distribution"][dim_id] = actual
            
            if actual != expected:
                error = f"Dimension {dim_id}: expected {expected} questions, got {actual}"
                summary["validation_errors"].append(error)
                summary["validation_status"] = "FAIL"
                self.stats["validation_errors"] += 1
        
        return summary
    
    def _serialize_results(self, results: Dict[str, DimensionResult]) -> Dict[str, Any]:
        """Serialize results with deterministic ordering."""
        serialized = OrderedDict()
        
        for dim_id in sorted(results.keys()):
            serialized[dim_id] = self._serialize_single_result(results[dim_id])
        
        return serialized
    
    def _serialize_single_result(self, result: DimensionResult) -> Dict[str, Any]:
        """Serialize a single dimension result."""
        return {
            "dimension_id": result.dimension_id,
            "scores": {
                "raw_score": round(result.raw_score, 6),
                "adjusted_score": round(result.adjusted_score, 6),
                "quality_penalty_applied": round(result.quality_penalty_applied, 6)
            },
            "status": result.status.value,
            "question_metrics": {
                "question_count": result.question_count,
                "expected_question_count": result.expected_question_count,
                "completion_percentage": round(result.completion_percentage, 2)
            },
            "evidence_metrics": {
                "evidence_density": round(result.evidence_density, 6),
                "average_evidence_quality": round(result.average_evidence_quality, 6)
            },
            "correction_factors": [
                {
                    "type": cf.correction_type.value,
                    "original_score": round(cf.original_score, 6),
                    "adjustment": round(cf.adjustment, 6),
                    "final_score": round(cf.final_score, 6),
                    "reason": cf.reason,
                    "evidence_metrics": cf.evidence_metrics
                }
                for cf in result.correction_factors
            ],
            "question_breakdown": [
                {
                    "question_id": qb.question_id,
                    "raw_score": round(qb.raw_score, 6),
                    "weighted_score": round(qb.weighted_score, 6),
                    "weight": round(qb.weight, 6),
                    "evidence_count": qb.evidence_count,
                    "evidence_quality": round(qb.evidence_quality, 6),
                    "contribution_percentage": round(qb.contribution_percentage, 2)
                }
                for qb in result.question_breakdown
            ],
            "metadata": {
                "processing_timestamp": result.processing_timestamp,
                "artifact_id": result.artifact_id
            }
        }
    
    def _compute_quality_metrics(self, results: Dict[str, DimensionResult]) -> Dict[str, Any]:
        """Compute aggregate quality metrics across all dimensions."""
        metrics = {
            "overall_completion_percentage": 0.0,
            "average_evidence_density": 0.0,
            "average_evidence_quality": 0.0,
            "dimensions_with_penalties": 0,
            "dimensions_with_corrections": 0
        }
        
        if not results:
            return metrics
        
        # Compute averages
        total_completion = sum(r.completion_percentage for r in results.values())
        total_evidence_density = sum(r.evidence_density for r in results.values())
        total_evidence_quality = sum(r.average_evidence_quality for r in results.values())
        
        metrics["overall_completion_percentage"] = round(total_completion / len(results), 2)
        metrics["average_evidence_density"] = round(total_evidence_density / len(results), 6)
        metrics["average_evidence_quality"] = round(total_evidence_quality / len(results), 6)
        
        # Count penalties and corrections
        metrics["dimensions_with_penalties"] = sum(1 for r in results.values() if r.quality_penalty_applied > 0)
        metrics["dimensions_with_corrections"] = sum(1 for r in results.values() if r.correction_factors)
        
        return metrics
    
    def _summarize_corrections(self, results: Dict[str, DimensionResult]) -> Dict[str, Any]:
        """Summarize correction factors applied."""
        summary = {
            "total_corrections": 0,
            "corrections_by_type": {},
            "corrections_by_dimension": {}
        }
        
        for result in results.values():
            for correction in result.correction_factors:
                summary["total_corrections"] += 1
                
                # By type
                corr_type = correction.correction_type.value
                if corr_type not in summary["corrections_by_type"]:
                    summary["corrections_by_type"][corr_type] = 0
                summary["corrections_by_type"][corr_type] += 1
                
                # By dimension
                dim_id = result.dimension_id
                if dim_id not in summary["corrections_by_dimension"]:
                    summary["corrections_by_dimension"][dim_id] = 0
                summary["corrections_by_dimension"][dim_id] += 1
        
        return summary


def create_sample_question_data() -> Dict[str, Any]:
    """Create sample question data for testing."""
    questions = []
    
    # DE-1: 6 questions
    for i in range(6):
        questions.append({
            "question_id": f"DE1_Q{i+1:02d}",
            "dimension": "DE-1",
            "score": 0.75 + (i * 0.05),
            "weight": 1.0,
            "evidence_count": 3 + i,
            "evidence_quality": 0.6 + (i * 0.05)
        })
    
    # DE-2: 21 questions
    for i in range(21):
        questions.append({
            "question_id": f"DE2_Q{i+1:02d}",
            "dimension": "DE-2", 
            "score": 0.65 + (i * 0.01),
            "weight": 1.0,
            "evidence_count": 2 + (i % 3),
            "evidence_quality": 0.55 + (i * 0.01)
        })
    
    # DE-3: 8 questions
    for i in range(8):
        questions.append({
            "question_id": f"DE3_Q{i+1:02d}",
            "dimension": "DE-3",
            "score": 0.80 - (i * 0.02),
            "weight": 1.0,
            "evidence_count": 4 + i,
            "evidence_quality": 0.70 - (i * 0.02)
        })
    
    # DE-4: 8 questions
    for i in range(8):
        questions.append({
            "question_id": f"DE4_Q{i+1:02d}",
            "dimension": "DE-4",
            "score": 0.70 + (i * 0.03),
            "weight": 1.0,
            "evidence_count": 1 + (i % 2),  # Low evidence count to trigger penalty
            "evidence_quality": 0.40 + (i * 0.03)  # Low quality to trigger penalty
        })
    
    return {
        "questions": questions,
        "metadata": {
            "source": "sample_data",
            "timestamp": datetime.utcnow().isoformat(),
            "total_questions": len(questions)
        }
    }


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    # Create aggregator
    aggregator = DimensionAggregator()
    
    # Generate sample data
    sample_data = create_sample_question_data()
    
    # Process dimensions
    results = aggregator.aggregate_dimensions(sample_data)
    
    # Print summary
    print("\nDimension Aggregation Results:")
    print("=" * 50)
    
    for dim_id, result in results.items():
        print(f"\n{dim_id}:")
        print(f"  Raw Score: {result.raw_score:.3f}")
        print(f"  Adjusted Score: {result.adjusted_score:.3f}")
        print(f"  Status: {result.status.value}")
        print(f"  Questions: {result.question_count}/{result.expected_question_count}")
        print(f"  Evidence Density: {result.evidence_density:.3f}")
        print(f"  Corrections Applied: {len(result.correction_factors)}")
    
    print(f"\nArtifacts generated in: {aggregator.output_dir}")
    print(f"Processing stats: {aggregator.stats}")