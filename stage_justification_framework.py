#!/usr/bin/env python3
"""
Stage Justification Framework

Provides comprehensive framework for measuring and validating pipeline stage value contribution.
Includes metrics for artifact uniqueness, processing efficiency ratios, and downstream dependency analysis.
"""

import json
import logging
import time
# # # from abc import ABC, abstractmethod  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime, timedelta  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Set, Tuple, Union  # Module not found  # Module not found  # Module not found
import hashlib
# Optional numpy import
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JustificationMetricType(Enum):
    """Types of justification metrics"""
    ARTIFACT_UNIQUENESS = "artifact_uniqueness"
    PROCESSING_EFFICIENCY = "processing_efficiency" 
    DEPENDENCY_IMPACT = "dependency_impact"
    COMPUTATIONAL_COMPLEXITY = "computational_complexity"
    OUTPUT_DIFFERENTIATION = "output_differentiation"
    DOWNSTREAM_VALUE = "downstream_value"


@dataclass
class MetricThreshold:
    """Threshold configuration for justification metrics"""
    metric_type: JustificationMetricType
    minimum_acceptable: float
    target_value: float
    weight: float
    description: str


@dataclass
class JustificationResult:
    """Result of stage justification analysis"""
    stage_name: str
    overall_score: float
    metric_scores: Dict[JustificationMetricType, float]
    threshold_violations: List[MetricThreshold]
    justification_level: str  # 'excellent', 'good', 'acceptable', 'questionable', 'unjustified'
    recommendation: str
    evidence: List[str]


class JustificationMetricCalculator(ABC):
    """Abstract base class for justification metric calculators"""
    
    @abstractmethod
    def calculate(self, stage_data: Dict[str, Any]) -> float:
        """Calculate the metric value for a given stage"""
        pass
    
    @abstractmethod
    def get_metric_type(self) -> JustificationMetricType:
        """Return the type of metric this calculator handles"""
        pass


class ArtifactUniquenessCalculator(JustificationMetricCalculator):
    """Calculates artifact uniqueness metric"""
    
    def __init__(self, all_stages_data: Dict[str, Dict[str, Any]]):
        self.all_stages_data = all_stages_data
    
    def calculate(self, stage_data: Dict[str, Any]) -> float:
        """Calculate uniqueness based on output artifact differentiation"""
        stage_outputs = set(stage_data.get('outputs', {}).keys())
        
        if not stage_outputs:
            return 0.0
        
        # Count overlaps with other stages
        total_overlap = 0
        other_stages = 0
        
        for other_stage_name, other_data in self.all_stages_data.items():
            if other_stage_name != stage_data.get('stage_name', ''):
                other_outputs = set(other_data.get('outputs', {}).keys())
                if other_outputs:
                    overlap = len(stage_outputs.intersection(other_outputs))
                    total_overlap += overlap / len(stage_outputs)
                    other_stages += 1
        
        if other_stages == 0:
            return 1.0
        
        # Higher uniqueness = lower average overlap
        average_overlap = total_overlap / other_stages
        uniqueness_score = max(0.0, 1.0 - average_overlap)
        
        return uniqueness_score
    
    def get_metric_type(self) -> JustificationMetricType:
        return JustificationMetricType.ARTIFACT_UNIQUENESS


class ProcessingEfficiencyCalculator(JustificationMetricCalculator):
    """Calculates processing time vs output differentiation ratio"""
    
    def calculate(self, stage_data: Dict[str, Any]) -> float:
        """Calculate efficiency as output value per unit processing time"""
        processing_time = stage_data.get('processing_metrics', {}).get('execution_time', 1.0)
        transformation_ratio = stage_data.get('processing_metrics', {}).get('transformation_ratio', 0.1)
        output_count = len(stage_data.get('outputs', {}))
        
        # Efficiency = (transformation_ratio * output_count) / processing_time
        efficiency = (transformation_ratio * max(1, output_count)) / max(0.1, processing_time)
        
        # Normalize to 0-1 range
        return min(1.0, efficiency / 5.0)
    
    def get_metric_type(self) -> JustificationMetricType:
        return JustificationMetricType.PROCESSING_EFFICIENCY


class DependencyImpactCalculator(JustificationMetricCalculator):
    """Calculates downstream dependency impact"""
    
    def calculate(self, stage_data: Dict[str, Any]) -> float:
        """Calculate impact based on downstream dependencies"""
        downstream_count = len(stage_data.get('dependency_analysis', {}).get('downstream_dependents', []))
        removal_impact = stage_data.get('dependency_analysis', {}).get('removal_impact_score', 0.0)
        critical_path = stage_data.get('dependency_analysis', {}).get('critical_path_position', False)
        
# # #         # Base score from downstream count  # Module not found  # Module not found  # Module not found
        base_score = min(1.0, downstream_count / 5.0)
        
        # Boost for critical path position
        critical_boost = 0.3 if critical_path else 0.0
        
        # Include removal impact
        impact_score = base_score + critical_boost + min(0.4, removal_impact)
        
        return min(1.0, impact_score)
    
    def get_metric_type(self) -> JustificationMetricType:
        return JustificationMetricType.DEPENDENCY_IMPACT


class ComputationalComplexityCalculator(JustificationMetricCalculator):
    """Calculates computational complexity contribution"""
    
    def calculate(self, stage_data: Dict[str, Any]) -> float:
        """Calculate complexity metric"""
        complexity = stage_data.get('processing_metrics', {}).get('computational_complexity', 1.0)
        
        # Normalize complexity score (assuming max reasonable complexity is 10)
        normalized_complexity = min(1.0, complexity / 10.0)
        
        return normalized_complexity
    
    def get_metric_type(self) -> JustificationMetricType:
        return JustificationMetricType.COMPUTATIONAL_COMPLEXITY


class OutputDifferentiationCalculator(JustificationMetricCalculator):
# # #     """Calculates output differentiation from inputs"""  # Module not found  # Module not found  # Module not found
    
    def calculate(self, stage_data: Dict[str, Any]) -> float:
# # #         """Calculate how much the outputs differ from inputs"""  # Module not found  # Module not found  # Module not found
        input_keys = stage_data.get('artifact_profile', {}).get('input_keys', set())
        output_keys = stage_data.get('artifact_profile', {}).get('output_keys', set())
        
        if not input_keys and not output_keys:
            return 0.0
        
        if not input_keys:
            return 1.0  # Pure generation
        
        if not output_keys:
            return 0.0  # No outputs
        
        # Calculate differentiation as ratio of unique outputs to total inputs
        unique_outputs = output_keys - input_keys
        differentiation_ratio = len(unique_outputs) / len(input_keys) if input_keys else 1.0
        
        return min(1.0, differentiation_ratio)
    
    def get_metric_type(self) -> JustificationMetricType:
        return JustificationMetricType.OUTPUT_DIFFERENTIATION


class DownstreamValueCalculator(JustificationMetricCalculator):
    """Calculates value provided to downstream stages"""
    
    def calculate(self, stage_data: Dict[str, Any]) -> float:
        """Calculate downstream value contribution"""
        outputs = stage_data.get('outputs', {})
        downstream_deps = stage_data.get('dependency_analysis', {}).get('downstream_dependents', [])
        
        if not outputs or not downstream_deps:
            return 0.0
        
        # Value = (number of outputs) * (number of downstream dependents) / max_expected
        value_score = (len(outputs) * len(downstream_deps)) / 25.0  # Assuming max 5 outputs * 5 dependents
        
        return min(1.0, value_score)
    
    def get_metric_type(self) -> JustificationMetricType:
        return JustificationMetricType.DOWNSTREAM_VALUE


class StageJustificationFramework:
    """Main framework for stage justification analysis"""
    
    def __init__(self, all_stages_data: Dict[str, Dict[str, Any]]):
        self.all_stages_data = all_stages_data
        
        # Initialize metric calculators
        self.calculators = {
            JustificationMetricType.ARTIFACT_UNIQUENESS: ArtifactUniquenessCalculator(all_stages_data),
            JustificationMetricType.PROCESSING_EFFICIENCY: ProcessingEfficiencyCalculator(),
            JustificationMetricType.DEPENDENCY_IMPACT: DependencyImpactCalculator(),
            JustificationMetricType.COMPUTATIONAL_COMPLEXITY: ComputationalComplexityCalculator(),
            JustificationMetricType.OUTPUT_DIFFERENTIATION: OutputDifferentiationCalculator(),
            JustificationMetricType.DOWNSTREAM_VALUE: DownstreamValueCalculator()
        }
        
        # Define metric thresholds
        self.thresholds = {
            JustificationMetricType.ARTIFACT_UNIQUENESS: MetricThreshold(
                JustificationMetricType.ARTIFACT_UNIQUENESS,
                minimum_acceptable=0.3,
                target_value=0.7,
                weight=0.25,
                description="Artifacts should be reasonably unique across stages"
            ),
            JustificationMetricType.PROCESSING_EFFICIENCY: MetricThreshold(
                JustificationMetricType.PROCESSING_EFFICIENCY,
                minimum_acceptable=0.2,
                target_value=0.6,
                weight=0.20,
                description="Processing time should justify output value"
            ),
            JustificationMetricType.DEPENDENCY_IMPACT: MetricThreshold(
                JustificationMetricType.DEPENDENCY_IMPACT,
                minimum_acceptable=0.1,
                target_value=0.5,
                weight=0.20,
                description="Stage should have meaningful downstream impact"
            ),
            JustificationMetricType.COMPUTATIONAL_COMPLEXITY: MetricThreshold(
                JustificationMetricType.COMPUTATIONAL_COMPLEXITY,
                minimum_acceptable=0.2,
                target_value=0.5,
                weight=0.15,
                description="Stage should perform meaningful computation"
            ),
            JustificationMetricType.OUTPUT_DIFFERENTIATION: MetricThreshold(
                JustificationMetricType.OUTPUT_DIFFERENTIATION,
                minimum_acceptable=0.3,
                target_value=0.8,
                weight=0.15,
# # #                 description="Outputs should meaningfully differ from inputs"  # Module not found  # Module not found  # Module not found
            ),
            JustificationMetricType.DOWNSTREAM_VALUE: MetricThreshold(
                JustificationMetricType.DOWNSTREAM_VALUE,
                minimum_acceptable=0.1,
                target_value=0.4,
                weight=0.05,
                description="Stage should provide value to downstream consumers"
            )
        }
    
    def justify_stage(self, stage_name: str) -> JustificationResult:
        """Perform comprehensive justification analysis for a stage"""
        
        if stage_name not in self.all_stages_data:
            raise ValueError(f"Stage {stage_name} not found in provided data")
        
        stage_data = self.all_stages_data[stage_name]
        
        # Calculate all metrics
        metric_scores = {}
        threshold_violations = []
        evidence = []
        
        for metric_type, calculator in self.calculators.items():
            try:
                score = calculator.calculate(stage_data)
                metric_scores[metric_type] = score
                
                threshold = self.thresholds[metric_type]
                
                # Check for threshold violations
                if score < threshold.minimum_acceptable:
                    threshold_violations.append(threshold)
                    evidence.append(f"{metric_type.value}: {score:.3f} (below minimum {threshold.minimum_acceptable})")
                elif score >= threshold.target_value:
                    evidence.append(f"{metric_type.value}: {score:.3f} (exceeds target)")
                else:
                    evidence.append(f"{metric_type.value}: {score:.3f} (acceptable)")
                    
            except Exception as e:
                logger.warning(f"Failed to calculate {metric_type.value} for {stage_name}: {e}")
                metric_scores[metric_type] = 0.0
        
        # Calculate overall weighted score
        overall_score = sum(
            metric_scores[metric_type] * self.thresholds[metric_type].weight
            for metric_type in metric_scores.keys()
        )
        
        # Determine justification level
        justification_level = self._determine_justification_level(
            overall_score, len(threshold_violations)
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            justification_level, threshold_violations, metric_scores
        )
        
        return JustificationResult(
            stage_name=stage_name,
            overall_score=overall_score,
            metric_scores=metric_scores,
            threshold_violations=threshold_violations,
            justification_level=justification_level,
            recommendation=recommendation,
            evidence=evidence
        )
    
    def _determine_justification_level(self, overall_score: float, violation_count: int) -> str:
        """Determine the justification level based on score and violations"""
        
        if overall_score >= 0.8 and violation_count == 0:
            return "excellent"
        elif overall_score >= 0.6 and violation_count <= 1:
            return "good"
        elif overall_score >= 0.4 and violation_count <= 2:
            return "acceptable"
        elif overall_score >= 0.2 and violation_count <= 3:
            return "questionable"
        else:
            return "unjustified"
    
    def _generate_recommendation(
        self, 
        justification_level: str, 
        violations: List[MetricThreshold],
        metric_scores: Dict[JustificationMetricType, float]
    ) -> str:
        """Generate specific recommendations based on analysis"""
        
        if justification_level == "excellent":
            return "Stage is well-justified and provides excellent value. No changes needed."
        
        if justification_level == "unjustified":
            return "Stage provides minimal value and should be considered for removal or major restructuring."
        
        recommendations = []
        
        # Address specific violations
        for violation in violations:
            if violation.metric_type == JustificationMetricType.ARTIFACT_UNIQUENESS:
                recommendations.append("Enhance output uniqueness by adding stage-specific processing")
            elif violation.metric_type == JustificationMetricType.PROCESSING_EFFICIENCY:
                recommendations.append("Improve processing efficiency or increase output value")
            elif violation.metric_type == JustificationMetricType.DEPENDENCY_IMPACT:
                recommendations.append("Strengthen downstream value or increase critical path position")
            elif violation.metric_type == JustificationMetricType.COMPUTATIONAL_COMPLEXITY:
                recommendations.append("Add more sophisticated processing logic")
            elif violation.metric_type == JustificationMetricType.OUTPUT_DIFFERENTIATION:
                recommendations.append("Increase output differentiation through transformation logic")
            elif violation.metric_type == JustificationMetricType.DOWNSTREAM_VALUE:
                recommendations.append("Enhance value provided to downstream stages")
        
        if not recommendations:
            if justification_level == "questionable":
                recommendations.append("Consider minor enhancements to improve overall value contribution")
            else:
                recommendations.append("Monitor performance and consider optimizations")
        
        return "; ".join(recommendations)
    
    def justify_all_stages(self) -> Dict[str, JustificationResult]:
        """Justify all stages in the pipeline"""
        results = {}
        
        for stage_name in self.all_stages_data.keys():
            try:
                results[stage_name] = self.justify_stage(stage_name)
            except Exception as e:
                logger.error(f"Failed to justify stage {stage_name}: {e}")
        
        return results
    
    def generate_justification_report(self) -> Dict[str, Any]:
        """Generate comprehensive justification report"""
        
        all_results = self.justify_all_stages()
        
        # Summary statistics
        level_counts = {}
        total_violations = 0
        avg_score = 0.0
        
        for result in all_results.values():
            level = result.justification_level
            level_counts[level] = level_counts.get(level, 0) + 1
            total_violations += len(result.threshold_violations)
            avg_score += result.overall_score
        
        if all_results:
            avg_score /= len(all_results)
        
        # Identify most problematic metrics
        metric_violation_counts = {}
        for result in all_results.values():
            for violation in result.threshold_violations:
                metric_type = violation.metric_type.value
                metric_violation_counts[metric_type] = metric_violation_counts.get(metric_type, 0) + 1
        
        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_stages": len(all_results),
                "average_justification_score": avg_score,
                "justification_levels": level_counts,
                "total_threshold_violations": total_violations,
                "most_problematic_metrics": sorted(
                    metric_violation_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
            },
            "stage_results": {
                name: {
                    "justification_level": result.justification_level,
                    "overall_score": result.overall_score,
                    "metric_scores": {k.value: v for k, v in result.metric_scores.items()},
                    "violations_count": len(result.threshold_violations),
                    "recommendation": result.recommendation,
                    "evidence": result.evidence
                } for name, result in all_results.items()
            },
            "metric_thresholds": {
                metric_type.value: {
                    "minimum_acceptable": threshold.minimum_acceptable,
                    "target_value": threshold.target_value,
                    "weight": threshold.weight,
                    "description": threshold.description
                } for metric_type, threshold in self.thresholds.items()
            }
        }


def create_stage_justification_framework(stages_data: Dict[str, Dict[str, Any]]) -> StageJustificationFramework:
    """Factory function to create a stage justification framework"""
    return StageJustificationFramework(stages_data)


def main():
    """Demo usage of the stage justification framework"""
    print("Stage Justification Framework Demo")
    print("=" * 40)
    
# # #     # This would normally be populated from the pipeline analysis system  # Module not found  # Module not found  # Module not found
    sample_data = {
        "stage1": {
            "stage_name": "stage1",
            "outputs": {"output1": str, "output2": dict},
            "processing_metrics": {
                "execution_time": 2.0,
                "transformation_ratio": 1.5,
                "computational_complexity": 3.0
            },
            "artifact_profile": {
                "input_keys": {"input1", "input2"},
                "output_keys": {"output1", "output2"}
            },
            "dependency_analysis": {
                "downstream_dependents": ["stage2", "stage3"],
                "removal_impact_score": 0.7,
                "critical_path_position": True
            }
        }
    }
    
    framework = create_stage_justification_framework(sample_data)
    result = framework.justify_stage("stage1")
    
    print(f"Stage: {result.stage_name}")
    print(f"Overall Score: {result.overall_score:.3f}")
    print(f"Justification Level: {result.justification_level}")
    print(f"Recommendation: {result.recommendation}")
    
    return result


if __name__ == "__main__":
    main()