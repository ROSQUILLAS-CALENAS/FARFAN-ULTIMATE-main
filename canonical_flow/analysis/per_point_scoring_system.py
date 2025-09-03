"""
Per-Point Scoring System for Decálogo Human Rights Assessment

This module calculates weighted scores across DE-1, DE-2, DE-3, DE-4 dimensions,
applies compliance thresholds, and generates explainability artifacts for each
of the 10 Decálogo points.

Scoring weights:
- DE-1: 0.30 (Logic of Intervention and Internal Coherence)
- DE-2: 0.25 (Thematic Inclusion)
- DE-3: 0.25 (Planning and Budget Adequacy) 
- DE-4: 0.20 (Value Chain)

Compliance classifications:
- CUMPLE: ≥ 0.80
- CUMPLE_PARCIAL: 0.50-0.79
- NO_CUMPLE: < 0.50
"""

import json
import logging
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from decimal import Decimal, ROUND_HALF_UP  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, List, Any, Optional, Tuple  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, asdict  # Module not found  # Module not found  # Module not found

logger = logging.getLogger(__name__)

class ComplianceLevel(Enum):
    """Compliance classification levels."""
    CUMPLE = "CUMPLE"
    CUMPLE_PARCIAL = "CUMPLE_PARCIAL"
    NO_CUMPLE = "NO_CUMPLE"

@dataclass
class DimensionScore:
    """Score for a single dimension."""
    dimension: str
    raw_score: float
    weighted_score: float
    weight: float
    question_count: int
    questions_answered: int
    top_contributors: List[Dict[str, Any]]

@dataclass  
class PointScoreResult:
    """Complete scoring result for a Decálogo point."""
    point_id: int
    final_score: float
    compliance_level: ComplianceLevel
    dimension_scores: Dict[str, DimensionScore]
    total_questions: int
    total_answered: int
    completion_rate: float
    top_contributing_questions: List[Dict[str, Any]]
    evidence_summary: Dict[str, Any]
    calculation_metadata: Dict[str, Any]

class PerPointScoringSystem:
    """
    Deterministic scoring system for Decálogo points across DE dimensions.
    
    Implements weighted aggregation across all 47 questions per point,
    applies compliance thresholds, and generates explainability artifacts.
    """
    
    # Dimension weights as specified
    DIMENSION_WEIGHTS = {
        "DE-1": Decimal("0.30"),
        "DE-2": Decimal("0.25"), 
        "DE-3": Decimal("0.25"),
        "DE-4": Decimal("0.20")
    }
    
    # Compliance thresholds
    COMPLIANCE_THRESHOLDS = {
        ComplianceLevel.CUMPLE: Decimal("0.80"),
        ComplianceLevel.CUMPLE_PARCIAL: Decimal("0.50")
    }
    
    # Question counts per dimension (47 total)
    DIMENSION_QUESTION_COUNTS = {
        "DE-1": 6,   # Q1-Q6
        "DE-2": 21,  # D1-D6, O1-O6, T1-T5, S1-S4
        "DE-3": 8,   # G1-G2, A1-A2, R1-R2, S1-S2  
        "DE-4": 8,   # 8 value chain elements
        "DE-5": 4    # Additional questions (if present)
    }
    
    def __init__(self, output_dir: str = "canonical_flow/analysis/"):
        """Initialize the scoring system."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate dimension weights sum to 1.0
        weight_sum = sum(self.DIMENSION_WEIGHTS.values())
        if not (Decimal("0.999") <= weight_sum <= Decimal("1.001")):
            raise ValueError(f"Dimension weights must sum to 1.0, got {weight_sum}")
            
        logger.info(f"Initialized PerPointScoringSystem with output dir: {self.output_dir}")
    
    def calculate_point_score(
        self,
        point_id: int,
        dimension_scores: Dict[str, Dict[str, Any]],
        evidence_data: Optional[Dict[str, Any]] = None
    ) -> PointScoreResult:
        """
        Calculate weighted score for a single Decálogo point.
        
        Args:
            point_id: Point number (1-10)
            dimension_scores: Dict of dimension -> {questions, scores, evidence}
            evidence_data: Optional evidence metadata
            
        Returns:
            PointScoreResult with scores, compliance, and explainability
        """
        if not (1 <= point_id <= 10):
            raise ValueError(f"Point ID must be 1-10, got {point_id}")
            
        logger.info(f"Calculating score for Point {point_id}")
        
        # Calculate dimension-level scores
        calculated_dimensions = {}
        total_weighted_score = Decimal("0")
        total_questions = 0
        total_answered = 0
        all_contributors = []
        
        for dim, weight in self.DIMENSION_WEIGHTS.items():
            if dim not in dimension_scores:
                logger.warning(f"Missing dimension {dim} for Point {point_id}")
                continue
                
            dim_data = dimension_scores[dim]
            dim_score = self._calculate_dimension_score(dim, dim_data)
            calculated_dimensions[dim] = dim_score
            
            # Accumulate weighted score
            weighted_contribution = Decimal(str(dim_score.raw_score)) * weight
            total_weighted_score += weighted_contribution
            
            # Track question completion
            total_questions += dim_score.question_count
            total_answered += dim_score.questions_answered
            
            # Collect top contributors
            all_contributors.extend(dim_score.top_contributors)
            
        # Calculate final score
        final_score = float(total_weighted_score.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))
        
        # Determine compliance level
        compliance = self._classify_compliance(final_score)
        
        # Calculate completion rate
        completion_rate = (total_answered / total_questions) if total_questions > 0 else 0.0
        
        # Rank top contributing questions
        top_contributors = self._rank_contributors(all_contributors, limit=10)
        
        # Generate evidence summary
        evidence_summary = self._generate_evidence_summary(evidence_data, dimension_scores)
        
        # Calculation metadata
        calculation_metadata = {
            "calculation_timestamp": datetime.now().isoformat(),
            "dimension_weights": {k: float(v) for k, v in self.DIMENSION_WEIGHTS.items()},
            "compliance_thresholds": {
                "CUMPLE": float(self.COMPLIANCE_THRESHOLDS[ComplianceLevel.CUMPLE]),
                "CUMPLE_PARCIAL": float(self.COMPLIANCE_THRESHOLDS[ComplianceLevel.CUMPLE_PARCIAL])
            },
            "total_questions_expected": 47,
            "scoring_algorithm": "weighted_deterministic_v1.0"
        }
        
        result = PointScoreResult(
            point_id=point_id,
            final_score=final_score,
            compliance_level=compliance,
            dimension_scores=calculated_dimensions,
            total_questions=total_questions,
            total_answered=total_answered,
            completion_rate=completion_rate,
            top_contributing_questions=top_contributors,
            evidence_summary=evidence_summary,
            calculation_metadata=calculation_metadata
        )
        
        logger.info(f"Point {point_id} final score: {final_score:.4f} ({compliance.value})")
        return result
    
    def _calculate_dimension_score(
        self,
        dimension: str,
        dim_data: Dict[str, Any]
    ) -> DimensionScore:
        """Calculate score for a single dimension."""
        questions = dim_data.get("questions", [])
        question_count = len(questions) if questions else self.DIMENSION_QUESTION_COUNTS.get(dimension, 0)
        
# # #         # Calculate raw score from questions  # Module not found  # Module not found  # Module not found
        if questions:
# # #             # Sum scores from answered questions  # Module not found  # Module not found  # Module not found
            total_score = Decimal("0")
            answered_count = 0
            contributors = []
            
            for question in questions:
                q_id = question.get("question_id", "")
                q_score = question.get("score", 0)
                q_weight = question.get("weight", 1.0)
                q_evidence = question.get("evidence", [])
                
                if q_score is not None:
                    normalized_score = max(0, min(1, float(q_score)))
                    weighted_score = Decimal(str(normalized_score)) * Decimal(str(q_weight))
                    total_score += weighted_score
                    answered_count += 1
                    
                    # Track contributors
                    contributors.append({
                        "question_id": q_id,
                        "question_text": question.get("text", ""),
                        "score": normalized_score,
                        "weight": q_weight,
                        "contribution": float(weighted_score),
                        "evidence_count": len(q_evidence),
                        "dimension": dimension
                    })
            
            # Calculate average score
            raw_score = float(total_score / answered_count) if answered_count > 0 else 0.0
        else:
# # #             # Use completion percentage from existing data  # Module not found  # Module not found  # Module not found
            raw_score = dim_data.get("completion_percentage", 0.0) / 100.0
            answered_count = dim_data.get("actual_count", 0)
            contributors = []
        
        # Apply dimension weight
        weight = float(self.DIMENSION_WEIGHTS.get(dimension, Decimal("0")))
        weighted_score = raw_score * weight
        
        # Rank top contributors
        top_contributors = sorted(contributors, key=lambda x: x["contribution"], reverse=True)[:5]
        
        return DimensionScore(
            dimension=dimension,
            raw_score=raw_score,
            weighted_score=weighted_score,
            weight=weight,
            question_count=question_count,
            questions_answered=answered_count,
            top_contributors=top_contributors
        )
    
    def _classify_compliance(self, score: float) -> ComplianceLevel:
        """Classify compliance level based on score."""
        score_decimal = Decimal(str(score))
        
        if score_decimal >= self.COMPLIANCE_THRESHOLDS[ComplianceLevel.CUMPLE]:
            return ComplianceLevel.CUMPLE
        elif score_decimal >= self.COMPLIANCE_THRESHOLDS[ComplianceLevel.CUMPLE_PARCIAL]:
            return ComplianceLevel.CUMPLE_PARCIAL
        else:
            return ComplianceLevel.NO_CUMPLE
    
    def _rank_contributors(self, contributors: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        """Rank and return top contributing questions."""
        sorted_contributors = sorted(contributors, key=lambda x: x.get("contribution", 0), reverse=True)
        return sorted_contributors[:limit]
    
    def _generate_evidence_summary(
        self,
        evidence_data: Optional[Dict[str, Any]],
        dimension_scores: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate summary of evidence supporting the scores."""
        summary = {
            "total_evidence_pieces": 0,
            "high_quality_evidence": 0,
            "evidence_by_dimension": {},
            "evidence_types": {},
            "coverage_analysis": {}
        }
        
        if not evidence_data:
            return summary
        
        # Analyze evidence across dimensions
        for dim, dim_data in dimension_scores.items():
            questions = dim_data.get("questions", [])
            dim_evidence = []
            
            for question in questions:
                evidence_list = question.get("evidence", [])
                dim_evidence.extend(evidence_list)
            
            summary["evidence_by_dimension"][dim] = {
                "count": len(dim_evidence),
                "high_quality": len([e for e in dim_evidence if e.get("score", 0) > 0.8])
            }
            
            summary["total_evidence_pieces"] += len(dim_evidence)
        
        return summary
    
    def process_all_points(
        self,
        points_data: Dict[int, Dict[str, Any]],
        output_filename: Optional[str] = None
    ) -> Dict[int, PointScoreResult]:
        """
        Process scoring for all 10 Decálogo points.
        
        Args:
            points_data: Dict mapping point_id -> dimension_scores
            output_filename: Optional custom output filename
            
        Returns:
            Dict mapping point_id -> PointScoreResult
        """
        logger.info("Processing scores for all 10 Decálogo points")
        
        results = {}
        processing_summary = {
            "processed_points": 0,
            "compliance_distribution": {level.value: 0 for level in ComplianceLevel},
            "average_score": 0.0,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        total_score = 0.0
        
        for point_id in range(1, 11):
            if point_id not in points_data:
                logger.warning(f"Missing data for Point {point_id}")
                continue
                
            try:
                result = self.calculate_point_score(
                    point_id=point_id,
                    dimension_scores=points_data[point_id]
                )
                results[point_id] = result
                
                # Update summary stats
                processing_summary["processed_points"] += 1
                processing_summary["compliance_distribution"][result.compliance_level.value] += 1
                total_score += result.final_score
                
            except Exception as e:
                logger.error(f"Error processing Point {point_id}: {e}")
                continue
        
        # Calculate average score
        if processing_summary["processed_points"] > 0:
            processing_summary["average_score"] = total_score / processing_summary["processed_points"]
        
        # Save results to JSON
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"per_point_scores_{timestamp}.json"
        
        output_path = self.output_dir / output_filename
        self._save_results(results, processing_summary, output_path)
        
        logger.info(f"Processed {processing_summary['processed_points']} points, saved to {output_path}")
        return results
    
    def _save_results(
        self,
        results: Dict[int, PointScoreResult], 
        summary: Dict[str, Any],
        output_path: Path
    ) -> None:
        """Save scoring results to JSON file."""
        output_data = {
            "scoring_metadata": {
                "system_version": "per_point_scoring_v1.0",
                "dimension_weights": {k: float(v) for k, v in self.DIMENSION_WEIGHTS.items()},
                "compliance_thresholds": {
                    level.value: float(threshold) 
                    for level, threshold in self.COMPLIANCE_THRESHOLDS.items()
                },
                "total_questions_per_point": 47
            },
            "processing_summary": summary,
            "point_scores": {}
        }
        
        # Convert results to dict format
        for point_id, result in results.items():
            output_data["point_scores"][str(point_id)] = {
                "point_id": result.point_id,
                "final_score": result.final_score,
                "compliance_level": result.compliance_level.value,
                "dimension_scores": {
                    dim: asdict(score) for dim, score in result.dimension_scores.items()
                },
                "total_questions": result.total_questions,
                "total_answered": result.total_answered,
                "completion_rate": result.completion_rate,
                "top_contributing_questions": result.top_contributing_questions,
                "evidence_summary": result.evidence_summary,
                "calculation_metadata": result.calculation_metadata
            }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results to {output_path}: {e}")
            raise
    
    def generate_explainability_report(
        self,
        results: Dict[int, PointScoreResult],
        output_filename: Optional[str] = None
    ) -> Path:
        """
        Generate detailed explainability report for all points.
        
        Args:
# # #             results: Results from process_all_points()  # Module not found  # Module not found  # Module not found
            output_filename: Optional custom filename
            
        Returns:
            Path to generated report
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"explainability_report_{timestamp}.json"
        
        report_path = self.output_dir / output_filename
        
        explainability_data = {
            "report_metadata": {
                "generation_timestamp": datetime.now().isoformat(),
                "report_type": "per_point_explainability",
                "points_analyzed": len(results)
            },
            "methodology_summary": {
                "scoring_approach": "weighted_deterministic_aggregation",
                "dimension_weights": {k: float(v) for k, v in self.DIMENSION_WEIGHTS.items()},
                "compliance_classification": {
                    "CUMPLE": "≥ 0.80",
                    "CUMPLE_PARCIAL": "0.50-0.79", 
                    "NO_CUMPLE": "< 0.50"
                }
            },
            "point_explanations": {}
        }
        
        for point_id, result in results.items():
            explanation = self._generate_point_explanation(result)
            explainability_data["point_explanations"][str(point_id)] = explanation
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(explainability_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Explainability report saved to {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"Error generating explainability report: {e}")
            raise
    
    def _generate_point_explanation(self, result: PointScoreResult) -> Dict[str, Any]:
        """Generate detailed explanation for a single point."""
        explanation = {
            "point_id": result.point_id,
            "final_score": result.final_score,
            "compliance_level": result.compliance_level.value,
            "score_breakdown": {
                "calculation_steps": [],
                "dimension_contributions": {},
                "key_factors": []
            },
            "top_contributors": result.top_contributing_questions,
            "evidence_analysis": result.evidence_summary,
            "recommendations": []
        }
        
        # Detailed calculation steps
        for dim, score_obj in result.dimension_scores.items():
            contribution = {
                "dimension": dim,
                "raw_score": score_obj.raw_score,
                "weight": score_obj.weight,
                "weighted_contribution": score_obj.weighted_score,
                "questions_answered": f"{score_obj.questions_answered}/{score_obj.question_count}",
                "completion_rate": score_obj.questions_answered / score_obj.question_count if score_obj.question_count > 0 else 0
            }
            explanation["score_breakdown"]["dimension_contributions"][dim] = contribution
            
            explanation["score_breakdown"]["calculation_steps"].append(
                f"{dim}: {score_obj.raw_score:.3f} × {score_obj.weight:.2f} = {score_obj.weighted_score:.3f}"
            )
        
        # Key factors affecting the score
        explanation["score_breakdown"]["key_factors"] = [
            f"Overall completion rate: {result.completion_rate:.1%}",
            f"Questions answered: {result.total_answered}/{result.total_questions}",
            f"Compliance classification: {result.compliance_level.value}"
        ]
        
        # Generate recommendations based on compliance level
        explanation["recommendations"] = self._generate_recommendations(result)
        
        return explanation
    
    def _generate_recommendations(self, result: PointScoreResult) -> List[str]:
        """Generate recommendations based on scoring results."""
        recommendations = []
        
        if result.compliance_level == ComplianceLevel.NO_CUMPLE:
            recommendations.append("Priority action required - score below minimum compliance threshold")
            
            # Identify weakest dimensions
            weak_dimensions = [
                dim for dim, score_obj in result.dimension_scores.items()
                if score_obj.raw_score < 0.5
            ]
            if weak_dimensions:
                recommendations.append(f"Focus improvement efforts on dimensions: {', '.join(weak_dimensions)}")
        
        elif result.compliance_level == ComplianceLevel.CUMPLE_PARCIAL:
            recommendations.append("Moderate improvement needed to reach full compliance")
            
        # General recommendations based on completion rate
        if result.completion_rate < 0.8:
            recommendations.append("Increase question coverage and evidence collection")
            
        # Dimension-specific recommendations
        for dim, score_obj in result.dimension_scores.items():
            if score_obj.raw_score < 0.6:
                dim_name = {
                    "DE-1": "Logic of Intervention",
                    "DE-2": "Thematic Inclusion", 
                    "DE-3": "Planning and Budget",
                    "DE-4": "Value Chain"
                }.get(dim, dim)
                recommendations.append(f"Strengthen {dim_name} components")
        
        return recommendations[:5]  # Limit to top 5 recommendations


def main():
    """Demo usage of the PerPointScoringSystem."""
    import random
    
    # Initialize scoring system
    scoring_system = PerPointScoringSystem()
    
    # Generate mock data for demonstration
    mock_data = {}
    for point_id in range(1, 11):
        mock_data[point_id] = {
            "DE-1": {
                "questions": [
                    {
                        "question_id": f"Q{i}",
                        "text": f"Sample question {i}",
                        "score": random.uniform(0.3, 1.0),
                        "weight": 1.0,
                        "evidence": [{"type": "document", "score": 0.8}]
                    }
                    for i in range(1, 7)
                ]
            },
            "DE-2": {
                "questions": [
                    {
                        "question_id": f"D{i}",
                        "text": f"Diagnostic question {i}",
                        "score": random.uniform(0.2, 0.9),
                        "weight": 1.0,
                        "evidence": []
                    }
                    for i in range(1, 22)
                ]
            },
            "DE-3": {
                "completion_percentage": random.uniform(40, 95),
                "actual_count": random.randint(4, 8),
                "questions": []
            },
            "DE-4": {
                "completion_percentage": random.uniform(30, 90),
                "actual_count": random.randint(3, 8),
                "questions": []
            }
        }
    
    # Process all points
    results = scoring_system.process_all_points(mock_data)
    
    # Generate explainability report
    report_path = scoring_system.generate_explainability_report(results)
    
    print(f"✅ Processed {len(results)} points")
    print(f"📊 Results saved to: canonical_flow/analysis/")
    print(f"📋 Explainability report: {report_path}")
    
    # Display summary
    compliance_counts = {}
    for result in results.values():
        level = result.compliance_level.value
        compliance_counts[level] = compliance_counts.get(level, 0) + 1
    
    print(f"\n📈 Compliance Distribution:")
    for level, count in compliance_counts.items():
        print(f"  {level}: {count} points")


if __name__ == "__main__":
    main()