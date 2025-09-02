"""
Macro Aggregation System for Decálogo Compliance Analysis

Implements a macro aggregation system that computes global Decálogo scores by applying
configurable weights to 10 point-level scores (P1-P10), classifies overall compliance
using CUMPLE (≥0.8), CUMPLE_PARCIAL (0.5-0.79), NO_CUMPLE (<0.5) thresholds, and
detects systemic contradictions by identifying conflicting evidence or scores across
related points within the same cluster.

Generates prioritized action recommendations by ranking lowest-scoring points with
their associated evidence gaps and writes results to canonical_flow/analysis/macro_alignment.json
with deterministic ordering and comprehensive audit trails.
"""

import json
import logging
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from pathlib import Path
from collections import OrderedDict

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)


class ComplianceLevel(Enum):
    """Compliance levels based on threshold analysis"""
    CUMPLE = "CUMPLE"
    CUMPLE_PARCIAL = "CUMPLE_PARCIAL"
    NO_CUMPLE = "NO_CUMPLE"


class ContradictionType(Enum):
    """Types of systemic contradictions"""
    SCORE_CONFLICT = "score_conflict"
    EVIDENCE_CONFLICT = "evidence_conflict"
    CLUSTER_INCONSISTENCY = "cluster_inconsistency"
    TEMPORAL_CONTRADICTION = "temporal_contradiction"


@dataclass
class PointScore:
    """Individual point-level score with metadata"""
    point_id: str
    point_name: str
    score: float
    confidence: float
    evidence_count: int
    evidence_gaps: List[str] = field(default_factory=list)
    cluster_id: int = 1
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def __post_init__(self):
        self.evidence_gaps = sorted(self.evidence_gaps)


@dataclass
class SystemicContradiction:
    """Systemic contradiction detection result"""
    contradiction_id: str
    contradiction_type: ContradictionType
    affected_points: List[str]
    cluster_id: int
    severity: str  # "high", "medium", "low"
    description: str
    conflicting_evidence: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    
    def __post_init__(self):
        self.affected_points = sorted(self.affected_points)
        self.conflicting_evidence = sorted(self.conflicting_evidence)


@dataclass
class ActionRecommendation:
    """Prioritized action recommendation"""
    priority_rank: int
    point_id: str
    point_name: str
    current_score: float
    improvement_potential: float
    required_actions: List[str]
    evidence_gaps: List[str]
    estimated_impact: float
    implementation_complexity: str  # "low", "medium", "high"
    
    def __post_init__(self):
        self.required_actions = sorted(self.required_actions)
        self.evidence_gaps = sorted(self.evidence_gaps)


@dataclass
class MacroAlignmentResult:
    """Complete macro alignment analysis result"""
    global_decalogo_score: float
    compliance_level: ComplianceLevel
    weighted_scores: Dict[str, float]
    point_scores: Dict[str, PointScore]
    systemic_contradictions: List[SystemicContradiction]
    action_recommendations: List[ActionRecommendation]
    cluster_analysis: Dict[int, Dict[str, float]]
    audit_trail: Dict[str, Any]
    processing_metadata: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def __post_init__(self):
        # Ensure deterministic ordering
        self.weighted_scores = dict(sorted(self.weighted_scores.items()))
        self.point_scores = dict(sorted(self.point_scores.items()))
        self.systemic_contradictions = sorted(self.systemic_contradictions, key=lambda x: x.contradiction_id)
        self.action_recommendations = sorted(self.action_recommendations, key=lambda x: x.priority_rank)
        self.cluster_analysis = dict(sorted(self.cluster_analysis.items()))


class MacroAggregationSystem:
    """
    Macro aggregation system for Decálogo compliance analysis.
    
    Computes global scores, classifies compliance, detects contradictions,
    and generates actionable recommendations with audit trails.
    """
    
    # Default Decálogo point weights (configurable)
    DEFAULT_WEIGHTS = {
        "P1": 0.15,  # Vida y seguridad - highest weight
        "P2": 0.10,  # Dignidad humana
        "P3": 0.10,  # Igualdad
        "P4": 0.08,  # Participación
        "P5": 0.12,  # Servicios básicos
        "P6": 0.09,  # Protección ambiental
        "P7": 0.11,  # Desarrollo económico
        "P8": 0.07,  # Derechos culturales
        "P9": 0.09,  # Acceso a justicia
        "P10": 0.09  # Transparencia
    }
    
    # Cluster definitions based on Decálogo documentation
    CLUSTER_DEFINITIONS = {
        1: {  # PAZ, SEGURIDAD Y PROTECCIÓN DE DEFENSORES
            "name": "Paz, Seguridad y Protección",
            "points": ["P1", "P5", "P8"],
            "weight": 0.30
        },
        2: {  # DERECHOS SOCIALES FUNDAMENTALES
            "name": "Derechos Sociales Fundamentales",
            "points": ["P2", "P3", "P4"],
            "weight": 0.25
        },
        3: {  # IGUALDAD Y NO DISCRIMINACIÓN
            "name": "Igualdad y No Discriminación",
            "points": ["P6", "P7"],
            "weight": 0.20
        },
        4: {  # DERECHOS TERRITORIALES Y AMBIENTALES
            "name": "Derechos Territoriales y Ambientales",
            "points": ["P9", "P10"],
            "weight": 0.25
        }
    }
    
    # Compliance thresholds
    COMPLIANCE_THRESHOLDS = {
        ComplianceLevel.CUMPLE: 0.8,
        ComplianceLevel.CUMPLE_PARCIAL: 0.5,
        ComplianceLevel.NO_CUMPLE: 0.0
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """Initialize macro aggregation system with optional custom weights"""
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._normalize_weights()
        
        # Initialize processing state
        self.processing_id = self._generate_processing_id()
        self.audit_trail = []
        
        logger.info(f"Initialized MacroAggregationSystem with processing_id: {self.processing_id}")
    
    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0"""
        total_weight = sum(self.weights.values())
        if total_weight != 1.0:
            for point_id in self.weights:
                self.weights[point_id] /= total_weight
            self._log_audit_event("weight_normalization", {
                "original_total": total_weight,
                "normalized_weights": self.weights
            })
    
    def _generate_processing_id(self) -> str:
        """Generate deterministic processing ID"""
        content = f"{datetime.now(timezone.utc).date()}_{id(self)}"
        return f"macro_agg_{hashlib.md5(content.encode()).hexdigest()[:8]}"
    
    def _log_audit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log audit event with timestamp"""
        self.audit_trail.append({
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data
        })
    
    def _classify_compliance(self, score: float) -> ComplianceLevel:
        """Classify compliance level based on score and thresholds"""
        if score >= self.COMPLIANCE_THRESHOLDS[ComplianceLevel.CUMPLE]:
            return ComplianceLevel.CUMPLE
        elif score >= self.COMPLIANCE_THRESHOLDS[ComplianceLevel.CUMPLE_PARCIAL]:
            return ComplianceLevel.CUMPLE_PARCIAL
        else:
            return ComplianceLevel.NO_CUMPLE
    
    def _detect_score_conflicts(self, point_scores: Dict[str, PointScore]) -> List[SystemicContradiction]:
        """Detect score conflicts within clusters"""
        contradictions = []
        
        for cluster_id, cluster_info in self.CLUSTER_DEFINITIONS.items():
            cluster_points = cluster_info["points"]
            cluster_scores = [point_scores[p].score for p in cluster_points if p in point_scores]
            
            if len(cluster_scores) < 2:
                continue
            
            # Calculate coefficient of variation (CV)
            if HAS_NUMPY:
                mean_score = np.mean(cluster_scores)
                std_score = np.std(cluster_scores)
            else:
                mean_score = sum(cluster_scores) / len(cluster_scores)
                variance = sum((x - mean_score) ** 2 for x in cluster_scores) / len(cluster_scores)
                std_score = variance ** 0.5
            cv = std_score / mean_score if mean_score > 0 else 0
            
            # Flag high variation as potential contradiction
            if cv > 0.3:  # 30% coefficient of variation threshold
                severity = "high" if cv > 0.5 else "medium"
                
                contradiction_id = f"score_conflict_cluster_{cluster_id}_{hashlib.md5(str(cluster_scores).encode()).hexdigest()[:6]}"
                
                contradiction = SystemicContradiction(
                    contradiction_id=contradiction_id,
                    contradiction_type=ContradictionType.SCORE_CONFLICT,
                    affected_points=cluster_points,
                    cluster_id=cluster_id,
                    severity=severity,
                    description=f"High score variation in cluster {cluster_info['name']} (CV={cv:.3f})",
                    confidence_score=min(1.0, cv)
                )
                contradictions.append(contradiction)
        
        return contradictions
    
    def _detect_evidence_conflicts(self, point_scores: Dict[str, PointScore]) -> List[SystemicContradiction]:
        """Detect evidence conflicts and gaps"""
        contradictions = []
        
        # Detect points with high scores but low evidence count
        for point_id, point_score in point_scores.items():
            if point_score.score > 0.7 and point_score.evidence_count < 3:
                contradiction_id = f"evidence_conflict_{point_id}_{hashlib.md5(str(point_score.evidence_count).encode()).hexdigest()[:6]}"
                
                contradiction = SystemicContradiction(
                    contradiction_id=contradiction_id,
                    contradiction_type=ContradictionType.EVIDENCE_CONFLICT,
                    affected_points=[point_id],
                    cluster_id=point_score.cluster_id,
                    severity="medium",
                    description=f"High score ({point_score.score:.3f}) with insufficient evidence ({point_score.evidence_count} items)",
                    conflicting_evidence=point_score.evidence_gaps,
                    confidence_score=0.8
                )
                contradictions.append(contradiction)
        
        return contradictions
    
    def _detect_cluster_inconsistencies(self, point_scores: Dict[str, PointScore]) -> List[SystemicContradiction]:
        """Detect inconsistencies within clusters"""
        contradictions = []
        
        for cluster_id, cluster_info in self.CLUSTER_DEFINITIONS.items():
            cluster_points = cluster_info["points"]
            available_points = [p for p in cluster_points if p in point_scores]
            
            if len(available_points) < 2:
                continue
            
            # Check for extreme outliers (z-score > 2)
            scores = [point_scores[p].score for p in available_points]
            if HAS_NUMPY:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
            else:
                mean_score = sum(scores) / len(scores)
                variance = sum((x - mean_score) ** 2 for x in scores) / len(scores)
                std_score = variance ** 0.5
            
            if std_score > 0:
                outliers = []
                for point_id in available_points:
                    z_score = abs(point_scores[point_id].score - mean_score) / std_score
                    if z_score > 2.0:
                        outliers.append(point_id)
                
                if outliers:
                    contradiction_id = f"cluster_inconsistency_{cluster_id}_{hashlib.md5(''.join(outliers).encode()).hexdigest()[:6]}"
                    
                    contradiction = SystemicContradiction(
                        contradiction_id=contradiction_id,
                        contradiction_type=ContradictionType.CLUSTER_INCONSISTENCY,
                        affected_points=outliers,
                        cluster_id=cluster_id,
                        severity="high" if len(outliers) > 1 else "medium",
                        description=f"Statistical outliers detected in cluster {cluster_info['name']}",
                        confidence_score=0.9
                    )
                    contradictions.append(contradiction)
        
        return contradictions
    
    def _generate_action_recommendations(self, point_scores: Dict[str, PointScore]) -> List[ActionRecommendation]:
        """Generate prioritized action recommendations"""
        recommendations = []
        
        # Sort points by score (lowest first) and impact potential
        sorted_points = sorted(
            point_scores.items(),
            key=lambda x: (x[1].score, -len(x[1].evidence_gaps))
        )
        
        for rank, (point_id, point_score) in enumerate(sorted_points[:5], 1):  # Top 5 priorities
            # Calculate improvement potential
            improvement_potential = min(1.0 - point_score.score, 0.5)  # Max 50% improvement
            
            # Estimate impact based on weight and current gap
            point_weight = self.weights.get(point_id, 0.1)
            estimated_impact = improvement_potential * point_weight
            
            # Determine implementation complexity
            complexity = "low" if len(point_score.evidence_gaps) <= 2 else "medium" if len(point_score.evidence_gaps) <= 5 else "high"
            
            # Generate specific actions based on evidence gaps
            required_actions = []
            for gap in point_score.evidence_gaps[:3]:  # Top 3 gaps
                required_actions.append(f"Address evidence gap: {gap}")
            
            if point_score.confidence < 0.7:
                required_actions.append("Improve data quality and validation")
            
            if point_score.evidence_count < 5:
                required_actions.append("Collect additional supporting evidence")
            
            recommendation = ActionRecommendation(
                priority_rank=rank,
                point_id=point_id,
                point_name=point_score.point_name,
                current_score=point_score.score,
                improvement_potential=improvement_potential,
                required_actions=required_actions,
                evidence_gaps=point_score.evidence_gaps,
                estimated_impact=estimated_impact,
                implementation_complexity=complexity
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _analyze_clusters(self, point_scores: Dict[str, PointScore]) -> Dict[int, Dict[str, float]]:
        """Analyze cluster-level performance"""
        cluster_analysis = {}
        
        for cluster_id, cluster_info in self.CLUSTER_DEFINITIONS.items():
            cluster_points = cluster_info["points"]
            available_scores = [point_scores[p].score for p in cluster_points if p in point_scores]
            
            if available_scores:
                if HAS_NUMPY:
                    mean_score = np.mean(available_scores)
                    min_score = np.min(available_scores)
                    max_score = np.max(available_scores)
                    score_variance = np.var(available_scores)
                else:
                    mean_score = sum(available_scores) / len(available_scores)
                    min_score = min(available_scores)
                    max_score = max(available_scores)
                    variance = sum((x - mean_score) ** 2 for x in available_scores) / len(available_scores)
                    score_variance = variance
                
                analysis = {
                    "average_score": mean_score,
                    "min_score": min_score,
                    "max_score": max_score,
                    "score_variance": score_variance,
                    "completion_rate": len(available_scores) / len(cluster_points),
                    "weighted_contribution": cluster_info["weight"] * mean_score
                }
            else:
                analysis = {
                    "average_score": 0.0,
                    "min_score": 0.0,
                    "max_score": 0.0,
                    "score_variance": 0.0,
                    "completion_rate": 0.0,
                    "weighted_contribution": 0.0
                }
            
            cluster_analysis[cluster_id] = analysis
        
        return cluster_analysis
    
    def compute_macro_alignment(
        self,
        point_level_data: Dict[str, Any],
        custom_weights: Optional[Dict[str, float]] = None
    ) -> MacroAlignmentResult:
        """
        Compute comprehensive macro alignment analysis.
        
        Args:
            point_level_data: Dictionary containing P1-P10 scores and metadata
            custom_weights: Optional custom weights for points
            
        Returns:
            Complete macro alignment result with audit trail
        """
        start_time = datetime.now(timezone.utc)
        
        # Apply custom weights if provided
        if custom_weights:
            self.weights.update(custom_weights)
            self._normalize_weights()
        
        self._log_audit_event("computation_start", {
            "weights": self.weights,
            "input_keys": list(point_level_data.keys())
        })
        
        # Parse point-level scores
        point_scores = {}
        for point_id in ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"]:
            point_data = point_level_data.get(point_id, {})
            
            # Extract or compute cluster ID
            cluster_id = 1  # Default cluster
            for cid, cinfo in self.CLUSTER_DEFINITIONS.items():
                if point_id in cinfo["points"]:
                    cluster_id = cid
                    break
            
            point_score = PointScore(
                point_id=point_id,
                point_name=point_data.get("name", f"Punto {point_id}"),
                score=float(point_data.get("score", 0.0)),
                confidence=float(point_data.get("confidence", 0.5)),
                evidence_count=int(point_data.get("evidence_count", 0)),
                evidence_gaps=point_data.get("evidence_gaps", []),
                cluster_id=cluster_id
            )
            point_scores[point_id] = point_score
        
        # Compute weighted scores
        weighted_scores = {}
        total_weighted_score = 0.0
        
        for point_id, point_score in point_scores.items():
            weight = self.weights.get(point_id, 0.0)
            weighted_score = point_score.score * weight
            weighted_scores[point_id] = weighted_score
            total_weighted_score += weighted_score
        
        # Classify compliance
        compliance_level = self._classify_compliance(total_weighted_score)
        
        self._log_audit_event("score_computation", {
            "global_score": total_weighted_score,
            "compliance_level": compliance_level.value,
            "weighted_scores": weighted_scores
        })
        
        # Detect systemic contradictions
        contradictions = []
        contradictions.extend(self._detect_score_conflicts(point_scores))
        contradictions.extend(self._detect_evidence_conflicts(point_scores))
        contradictions.extend(self._detect_cluster_inconsistencies(point_scores))
        
        self._log_audit_event("contradiction_detection", {
            "total_contradictions": len(contradictions),
            "by_type": {ct.value: len([c for c in contradictions if c.contradiction_type == ct]) 
                       for ct in ContradictionType}
        })
        
        # Generate action recommendations
        recommendations = self._generate_action_recommendations(point_scores)
        
        # Analyze clusters
        cluster_analysis = self._analyze_clusters(point_scores)
        
        # Create processing metadata
        end_time = datetime.now(timezone.utc)
        processing_metadata = {
            "processing_id": self.processing_id,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "processing_duration_seconds": (end_time - start_time).total_seconds(),
            "weights_used": self.weights,
            "thresholds_used": {level.value: threshold for level, threshold in self.COMPLIANCE_THRESHOLDS.items()},
            "cluster_definitions": self.CLUSTER_DEFINITIONS
        }
        
        # Create final result
        result = MacroAlignmentResult(
            global_decalogo_score=total_weighted_score,
            compliance_level=compliance_level,
            weighted_scores=weighted_scores,
            point_scores=point_scores,
            systemic_contradictions=contradictions,
            action_recommendations=recommendations,
            cluster_analysis=cluster_analysis,
            audit_trail={"events": self.audit_trail},
            processing_metadata=processing_metadata
        )
        
        self._log_audit_event("computation_complete", {
            "final_score": total_weighted_score,
            "compliance_level": compliance_level.value,
            "contradictions_found": len(contradictions),
            "recommendations_generated": len(recommendations)
        })
        
        return result
    
    def save_results(self, result: MacroAlignmentResult, output_path: Optional[str] = None) -> str:
        """
        Save macro alignment results to JSON file with deterministic ordering.
        
        Args:
            result: MacroAlignmentResult to save
            output_path: Optional custom output path
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = "canonical_flow/analysis/macro_alignment.json"
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary with custom serialization
        result_dict = self._serialize_result(result)
        
        # Write with deterministic formatting
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, sort_keys=True, ensure_ascii=False)
        
        self._log_audit_event("results_saved", {
            "output_path": str(output_file),
            "file_size_bytes": output_file.stat().st_size
        })
        
        logger.info(f"Macro alignment results saved to: {output_file}")
        return str(output_file)
    
    def _serialize_result(self, result: MacroAlignmentResult) -> Dict[str, Any]:
        """Serialize result with custom handling for complex types"""
        result_dict = asdict(result)
        
        # Convert enums to strings
        result_dict["compliance_level"] = result.compliance_level.value
        
        # Serialize point scores
        point_scores_serialized = {}
        for point_id, point_score in result.point_scores.items():
            point_scores_serialized[point_id] = asdict(point_score)
        result_dict["point_scores"] = point_scores_serialized
        
        # Serialize contradictions
        contradictions_serialized = []
        for contradiction in result.systemic_contradictions:
            contradiction_dict = asdict(contradiction)
            contradiction_dict["contradiction_type"] = contradiction.contradiction_type.value
            contradictions_serialized.append(contradiction_dict)
        result_dict["systemic_contradictions"] = contradictions_serialized
        
        return result_dict


def create_sample_data() -> Dict[str, Any]:
    """Create sample point-level data for testing"""
    sample_data = {}
    point_names = [
        "Vida y Seguridad",
        "Dignidad Humana", 
        "Igualdad",
        "Participación",
        "Servicios Básicos",
        "Protección Ambiental",
        "Desarrollo Económico",
        "Derechos Culturales",
        "Acceso a Justicia",
        "Transparencia"
    ]
    
    # Generate varied sample scores
    base_scores = [0.75, 0.82, 0.45, 0.68, 0.91, 0.33, 0.77, 0.59, 0.85, 0.42]
    
    for i, (score, name) in enumerate(zip(base_scores, point_names)):
        point_id = f"P{i+1}"
        sample_data[point_id] = {
            "name": name,
            "score": score,
            "confidence": min(0.95, score + 0.1),
            "evidence_count": max(1, int(score * 10)),
            "evidence_gaps": [f"Gap_{j}" for j in range(max(0, 5 - int(score * 5)))]
        }
    
    return sample_data


if __name__ == "__main__":
    # Demonstration usage
    logger.basicConfig(level=logging.INFO)
    
    # Create system with default weights
    system = MacroAggregationSystem()
    
    # Generate sample data
    sample_data = create_sample_data()
    
    # Compute macro alignment
    result = system.compute_macro_alignment(sample_data)
    
    # Save results
    output_path = system.save_results(result)
    
    print(f"Macro alignment analysis complete!")
    print(f"Global Decálogo Score: {result.global_decalogo_score:.3f}")
    print(f"Compliance Level: {result.compliance_level.value}")
    print(f"Contradictions Found: {len(result.systemic_contradictions)}")
    print(f"Action Recommendations: {len(result.action_recommendations)}")
    print(f"Results saved to: {output_path}")