#!/usr/bin/env python3
"""
Pipeline Value Analysis System

Automatically analyzes all 12 pipeline stages to identify:
- Components with minimal output differentiation
- No-op processing 
- Redundant functionality
- Low value-added operations

Provides consolidation and enhancement recommendations with stage justification framework.
"""

import json
import logging
import time
# # # from collections import defaultdict  # Module not found  # Module not found  # Module not found
# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Set, Tuple  # Module not found  # Module not found  # Module not found
import hashlib
import os

# Optional imports with fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import the comprehensive pipeline orchestrator to analyze
try:
# # #     from comprehensive_pipeline_orchestrator import (  # Module not found  # Module not found  # Module not found
        ComprehensivePipelineOrchestrator, 
        ProcessStage,
        ProcessNode,
        get_canonical_process_graph
    )
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    
    # Mock classes for when orchestrator is not available
# # #     from enum import Enum  # Module not found  # Module not found  # Module not found
# # #     from dataclasses import dataclass  # Module not found  # Module not found  # Module not found
    
    class ProcessStage(Enum):
        INGESTION = "ingestion_preparation"
        CONTEXT_BUILD = "context_construction"
        KNOWLEDGE = "knowledge_extraction"
        ANALYSIS = "analysis_nlp"
        CLASSIFICATION = "classification_evaluation"
        SEARCH = "search_retrieval"
        ORCHESTRATION = "orchestration_control"
        AGGREGATION = "aggregation_reporting"
        INTEGRATION = "integration_storage"
        SYNTHESIS = "synthesis_output"
    
    @dataclass
    class ProcessNode:
        file_path: str
        stage: ProcessStage
        dependencies: List[str]
        outputs: Dict[str, Any]
        process_type: str
        evento_inicio: str
        evento_cierre: str
        value_metrics: Dict[str, float]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValueContributionLevel(Enum):
    """Classification of stage value contribution"""
    HIGH = "high"           # Substantial unique value, clear differentiation
    MEDIUM = "medium"       # Moderate value, some differentiation
    LOW = "low"            # Minimal value, little differentiation
    NO_OP = "no_op"        # Essentially no value added
    REDUNDANT = "redundant" # Duplicates other stage functionality


@dataclass
class ArtifactProfile:
    """Profile of input/output artifacts for a stage"""
    input_keys: Set[str]
    output_keys: Set[str]
    data_types: Dict[str, type]
    content_hash: str
    size_estimate: int
    complexity_score: float
    uniqueness_score: float


@dataclass
class ProcessingMetrics:
    """Metrics for stage processing characteristics"""
    execution_time: float
    cpu_utilization: float
    memory_usage: float
    io_operations: int
    computational_complexity: float
    transformation_ratio: float  # output differentiation / input size


@dataclass
class DependencyAnalysis:
    """Analysis of stage dependencies and downstream impact"""
    direct_dependencies: List[str]
    indirect_dependencies: List[str]
    downstream_dependents: List[str]
    critical_path_position: bool
    bottleneck_potential: float
    removal_impact_score: float


@dataclass
class StageAnalysisResult:
    """Comprehensive analysis result for a single stage"""
    stage_name: str
    stage_type: ProcessStage
    value_contribution: ValueContributionLevel
    artifact_profile: ArtifactProfile
    processing_metrics: ProcessingMetrics
    dependency_analysis: DependencyAnalysis
    justification_score: float
    issues: List[str]
    recommendations: List[str]


@dataclass
class ConsolidationRecommendation:
    """Recommendation to merge adjacent low-value stages"""
    target_stages: List[str]
    consolidation_type: str  # 'merge', 'eliminate', 'simplify'
    expected_benefits: Dict[str, float]
    implementation_complexity: float
    risk_assessment: str


@dataclass
class EnhancementRecommendation:
    """Recommendation to enhance stage value through additional functionality"""
    target_stage: str
    enhancement_type: str  # 'validation', 'transformation', 'quality_check'
    proposed_additions: List[str]
    expected_value_increase: float
    implementation_effort: float


class PipelineValueAnalysisSystem:
    """Main system for analyzing pipeline value and providing recommendations"""
    
    def __init__(self):
        if ORCHESTRATOR_AVAILABLE:
            self.orchestrator = ComprehensivePipelineOrchestrator()
            self.process_graph = get_canonical_process_graph()
        else:
            self.orchestrator = None
            self.process_graph = self._create_mock_process_graph()
        
        self.stage_analyses: Dict[str, StageAnalysisResult] = {}
        self.consolidation_recommendations: List[ConsolidationRecommendation] = []
        self.enhancement_recommendations: List[EnhancementRecommendation] = []
    
    def _create_mock_process_graph(self) -> Dict[str, ProcessNode]:
        """Create a mock process graph for testing purposes"""
        return {
            "pdf_reader.py": ProcessNode(
                file_path="pdf_reader.py",
                stage=ProcessStage.INGESTION,
                dependencies=[],
                outputs={"text": str, "metadata": dict},
                process_type="extraction",
                evento_inicio="PDF file loaded",
                evento_cierre="Text extracted and structured",
                value_metrics={"extraction_rate": 0.8, "quality": 0.9}
            ),
            "advanced_loader.py": ProcessNode(
                file_path="advanced_loader.py",
                stage=ProcessStage.INGESTION,
                dependencies=["pdf_reader.py"],
                outputs={"loaded_docs": list, "metadata": dict},
                process_type="loading",
                evento_inicio="Document loading request",
                evento_cierre="Documents loaded with metadata",
                value_metrics={"load_efficiency": 0.7, "completeness": 0.8}
            ),
            "question_analyzer.py": ProcessNode(
                file_path="question_analyzer.py",
                stage=ProcessStage.ANALYSIS,
                dependencies=["advanced_loader.py"],
                outputs={"questions": list, "intents": dict},
                process_type="analysis",
                evento_inicio="Text available",
                evento_cierre="Questions analyzed",
                value_metrics={"intent_accuracy": 0.85}
            )
        }
        
    def analyze_all_stages(self) -> Dict[str, StageAnalysisResult]:
        """Analyze all 12 pipeline stages for value contribution"""
        logger.info("Starting comprehensive pipeline value analysis...")
        
        # Group stages by ProcessStage enum
        stage_groups = self._group_stages_by_type()
        
        for stage_type, stage_nodes in stage_groups.items():
            logger.info(f"Analyzing {stage_type.value} with {len(stage_nodes)} components")
            
            for node_name, node in stage_nodes.items():
                analysis = self._analyze_single_stage(node_name, node)
                self.stage_analyses[node_name] = analysis
        
        # Generate recommendations based on analysis
        self._generate_consolidation_recommendations()
        self._generate_enhancement_recommendations()
        
        return self.stage_analyses
    
    def _group_stages_by_type(self) -> Dict[ProcessStage, Dict[str, ProcessNode]]:
        """Group pipeline nodes by their stage type"""
        stage_groups = defaultdict(dict)
        
        for node_name, node in self.process_graph.items():
            stage_groups[node.stage][node_name] = node
            
        return dict(stage_groups)
    
    def _analyze_single_stage(self, node_name: str, node: ProcessNode) -> StageAnalysisResult:
        """Perform comprehensive analysis of a single stage"""
        logger.debug(f"Analyzing stage: {node_name}")
        
        # Analyze artifacts (inputs/outputs)
        artifact_profile = self._profile_artifacts(node)
        
        # Analyze processing characteristics
        processing_metrics = self._analyze_processing_metrics(node)
        
        # Analyze dependencies
        dependency_analysis = self._analyze_dependencies(node_name, node)
        
        # Calculate value contribution level
        value_contribution = self._classify_value_contribution(
            artifact_profile, processing_metrics, dependency_analysis
        )
        
        # Calculate justification score
        justification_score = self._calculate_justification_score(
            artifact_profile, processing_metrics, dependency_analysis
        )
        
        # Identify issues
        issues = self._identify_issues(node, artifact_profile, processing_metrics)
        
        # Generate recommendations
        recommendations = self._generate_stage_recommendations(
            node, value_contribution, issues
        )
        
        return StageAnalysisResult(
            stage_name=node_name,
            stage_type=node.stage,
            value_contribution=value_contribution,
            artifact_profile=artifact_profile,
            processing_metrics=processing_metrics,
            dependency_analysis=dependency_analysis,
            justification_score=justification_score,
            issues=issues,
            recommendations=recommendations
        )
    
    def _profile_artifacts(self, node: ProcessNode) -> ArtifactProfile:
        """Profile input/output artifacts for uniqueness and complexity"""
        
# # #         # Extract input keys from dependencies (simulated)  # Module not found  # Module not found  # Module not found
        input_keys = set()
        for dep in node.dependencies:
            dep_node = self.process_graph.get(dep)
            if dep_node:
                input_keys.update(dep_node.outputs.keys())
        
        # Extract output keys
        output_keys = set(node.outputs.keys())
        
        # Analyze data types
        data_types = {k: v for k, v in node.outputs.items()}
        
        # Calculate content hash for uniqueness
        content_str = f"{sorted(input_keys)}:{sorted(output_keys)}:{node.process_type}"
        content_hash = hashlib.md5(content_str.encode()).hexdigest()
        
        # Estimate size and complexity
        size_estimate = len(output_keys) * 100  # Rough estimate
        complexity_score = self._calculate_complexity_score(node)
        uniqueness_score = self._calculate_uniqueness_score(node, output_keys)
        
        return ArtifactProfile(
            input_keys=input_keys,
            output_keys=output_keys,
            data_types=data_types,
            content_hash=content_hash,
            size_estimate=size_estimate,
            complexity_score=complexity_score,
            uniqueness_score=uniqueness_score
        )
    
    def _calculate_complexity_score(self, node: ProcessNode) -> float:
        """Calculate processing complexity score"""
        base_score = len(node.outputs) * 0.2
        
        # Adjust based on process type
        complexity_multipliers = {
            'extraction': 1.5,
            'analysis': 2.0,
            'validation': 1.2,
            'transformation': 1.8,
            'orchestration': 1.3,
            'scoring': 1.7,
            'routing': 1.0,
            'tracking': 0.8,
            'loading': 0.9
        }
        
        multiplier = complexity_multipliers.get(node.process_type, 1.0)
        return min(10.0, base_score * multiplier)
    
    def _calculate_uniqueness_score(self, node: ProcessNode, output_keys: Set[str]) -> float:
        """Calculate how unique this stage's outputs are"""
        # Compare with other stages' outputs
        other_outputs = set()
        for other_name, other_node in self.process_graph.items():
            if other_name != node.file_path:
                other_outputs.update(other_node.outputs.keys())
        
        if not output_keys:
            return 0.0
            
        unique_outputs = output_keys - other_outputs
        uniqueness_score = len(unique_outputs) / len(output_keys)
        
        return uniqueness_score
    
    def _analyze_processing_metrics(self, node: ProcessNode) -> ProcessingMetrics:
        """Analyze processing characteristics and performance"""
        
        # Simulate metrics based on process type and current system state
        process_type = node.process_type
        
        # Base metrics by process type
        base_metrics = {
            'extraction': (2.5, 0.6, 0.3, 50, 1.5),
            'analysis': (5.0, 0.8, 0.5, 30, 2.5),
            'validation': (1.5, 0.4, 0.2, 20, 1.0),
            'transformation': (3.0, 0.7, 0.4, 40, 2.0),
            'orchestration': (1.0, 0.3, 0.1, 10, 0.5),
            'scoring': (4.0, 0.9, 0.6, 35, 2.2),
            'routing': (0.8, 0.2, 0.1, 15, 0.3),
            'tracking': (0.5, 0.1, 0.05, 5, 0.2),
            'loading': (2.0, 0.5, 0.25, 60, 1.2)
        }
        
        exec_time, cpu_util, mem_usage, io_ops, comp_complex = base_metrics.get(
            process_type, (1.0, 0.3, 0.1, 20, 1.0)
        )
        
        # Calculate transformation ratio (output differentiation vs input)
        input_size = len(node.dependencies) * 10  # Rough estimate
        output_size = len(node.outputs) * 10
        transformation_ratio = output_size / max(1, input_size) if input_size > 0 else 1.0
        
        return ProcessingMetrics(
            execution_time=exec_time,
            cpu_utilization=cpu_util,
            memory_usage=mem_usage,
            io_operations=io_ops,
            computational_complexity=comp_complex,
            transformation_ratio=transformation_ratio
        )
    
    def _analyze_dependencies(self, node_name: str, node: ProcessNode) -> DependencyAnalysis:
        """Analyze dependency relationships and downstream impact"""
        
        direct_deps = list(node.dependencies)
        
        # Find indirect dependencies
        indirect_deps = []
        for dep in direct_deps:
            dep_node = self.process_graph.get(dep)
            if dep_node:
                indirect_deps.extend(dep_node.dependencies)
        indirect_deps = list(set(indirect_deps) - set(direct_deps))
        
        # Find downstream dependents
        downstream_dependents = []
        for other_name, other_node in self.process_graph.items():
            if node_name in other_node.dependencies:
                downstream_dependents.append(other_name)
        
        # Calculate critical path position
        critical_path_position = len(downstream_dependents) > 2
        
        # Calculate bottleneck potential
        bottleneck_potential = len(downstream_dependents) / max(1, len(self.process_graph))
        
        # Calculate removal impact score
        removal_impact_score = (
            len(downstream_dependents) * 0.4 +
            len(direct_deps) * 0.2 +
            (1.0 if critical_path_position else 0.0) * 0.4
        )
        
        return DependencyAnalysis(
            direct_dependencies=direct_deps,
            indirect_dependencies=indirect_deps,
            downstream_dependents=downstream_dependents,
            critical_path_position=critical_path_position,
            bottleneck_potential=bottleneck_potential,
            removal_impact_score=removal_impact_score
        )
    
    def _classify_value_contribution(
        self, 
        artifacts: ArtifactProfile, 
        metrics: ProcessingMetrics,
        dependencies: DependencyAnalysis
    ) -> ValueContributionLevel:
        """Classify the value contribution level of a stage"""
        
        # Calculate composite value score
        uniqueness_weight = 0.3
        complexity_weight = 0.2
        transformation_weight = 0.25
        impact_weight = 0.25
        
        value_score = (
            artifacts.uniqueness_score * uniqueness_weight +
            (metrics.computational_complexity / 10.0) * complexity_weight +
            min(1.0, metrics.transformation_ratio) * transformation_weight +
            min(1.0, dependencies.removal_impact_score) * impact_weight
        )
        
        # Classify based on score thresholds
        if value_score >= 0.8:
            return ValueContributionLevel.HIGH
        elif value_score >= 0.6:
            return ValueContributionLevel.MEDIUM
        elif value_score >= 0.3:
            return ValueContributionLevel.LOW
        elif value_score >= 0.1:
            return ValueContributionLevel.NO_OP
        else:
            return ValueContributionLevel.REDUNDANT
    
    def _calculate_justification_score(
        self,
        artifacts: ArtifactProfile,
        metrics: ProcessingMetrics,
        dependencies: DependencyAnalysis
    ) -> float:
        """Calculate overall justification score for maintaining this stage"""
        
        # Artifact uniqueness (30%)
        uniqueness_component = artifacts.uniqueness_score * 0.3
        
        # Processing efficiency (25%) - higher efficiency = higher score
        efficiency_ratio = metrics.transformation_ratio / max(0.1, metrics.execution_time)
        efficiency_component = min(1.0, efficiency_ratio / 2.0) * 0.25
        
        # Downstream dependency impact (25%)
        dependency_component = min(1.0, dependencies.removal_impact_score) * 0.25
        
        # Complexity contribution (20%)
        complexity_component = min(1.0, artifacts.complexity_score / 10.0) * 0.2
        
        return uniqueness_component + efficiency_component + dependency_component + complexity_component
    
    def _identify_issues(
        self, 
        node: ProcessNode, 
        artifacts: ArtifactProfile, 
        metrics: ProcessingMetrics
    ) -> List[str]:
        """Identify specific issues with this stage"""
        issues = []
        
        # Low uniqueness
        if artifacts.uniqueness_score < 0.3:
            issues.append(f"Low output uniqueness ({artifacts.uniqueness_score:.2f})")
        
        # Poor transformation ratio
        if metrics.transformation_ratio < 0.5:
            issues.append(f"Minimal output differentiation (ratio: {metrics.transformation_ratio:.2f})")
        
        # High processing overhead vs value
        efficiency = metrics.transformation_ratio / max(0.1, metrics.execution_time)
        if efficiency < 0.2:
            issues.append(f"High processing overhead vs output value (efficiency: {efficiency:.2f})")
        
        # Empty or minimal outputs
        if len(artifacts.output_keys) <= 1:
            issues.append(f"Minimal outputs ({len(artifacts.output_keys)} keys)")
        
        # Low complexity score
        if artifacts.complexity_score < 1.0:
            issues.append(f"Low computational complexity ({artifacts.complexity_score:.2f})")
            
        return issues
    
    def _generate_stage_recommendations(
        self,
        node: ProcessNode,
        value_level: ValueContributionLevel,
        issues: List[str]
    ) -> List[str]:
        """Generate specific recommendations for improving this stage"""
        recommendations = []
        
        if value_level in [ValueContributionLevel.LOW, ValueContributionLevel.NO_OP]:
            recommendations.append("Consider consolidation with adjacent stages")
            recommendations.append("Add meaningful validation or transformation logic")
        
        if value_level == ValueContributionLevel.REDUNDANT:
            recommendations.append("Consider removal or merger with similar functionality")
        
        if "Low output uniqueness" in str(issues):
            recommendations.append("Enhance output differentiation with stage-specific processing")
        
        if "Minimal output differentiation" in str(issues):
            recommendations.append("Add transformation logic to increase output value")
        
        if "High processing overhead" in str(issues):
            recommendations.append("Optimize processing efficiency or increase output value")
        
        if len(recommendations) == 0:
            recommendations.append("Stage appears well-justified - monitor for regression")
            
        return recommendations
    
    def _generate_consolidation_recommendations(self):
        """Generate recommendations for consolidating adjacent low-value stages"""
        
        # Group by stage type and identify adjacent low-value stages
        stage_groups = self._group_stages_by_type()
        
        for stage_type, stage_nodes in stage_groups.items():
            low_value_stages = []
            for node_name, analysis in self.stage_analyses.items():
                if (analysis.stage_type == stage_type and 
                    analysis.value_contribution in [ValueContributionLevel.LOW, ValueContributionLevel.NO_OP]):
                    low_value_stages.append(node_name)
            
            if len(low_value_stages) >= 2:
                # Create consolidation recommendation
                expected_benefits = {
                    'execution_time_reduction': sum(
                        self.stage_analyses[stage].processing_metrics.execution_time 
                        for stage in low_value_stages
                    ) * 0.3,
                    'complexity_reduction': 0.4,
                    'maintenance_reduction': 0.5
                }
                
                recommendation = ConsolidationRecommendation(
                    target_stages=low_value_stages,
                    consolidation_type='merge',
                    expected_benefits=expected_benefits,
                    implementation_complexity=0.6,
                    risk_assessment='medium'
                )
                self.consolidation_recommendations.append(recommendation)
    
    def _generate_enhancement_recommendations(self):
        """Generate recommendations for enhancing stages to justify their computational overhead"""
        
        for node_name, analysis in self.stage_analyses.items():
            if (analysis.value_contribution in [ValueContributionLevel.LOW, ValueContributionLevel.MEDIUM] and
                analysis.justification_score < 0.6):
                
                # Determine appropriate enhancements
                proposed_additions = []
                enhancement_type = "validation"
                
                if "validation" not in analysis.stage_name.lower():
                    proposed_additions.append("Input validation and sanitization")
                    proposed_additions.append("Output quality checks")
                
                if analysis.processing_metrics.transformation_ratio < 1.0:
                    proposed_additions.append("Data transformation and enrichment")
                    enhancement_type = "transformation"
                
                if analysis.processing_metrics.computational_complexity < 2.0:
                    proposed_additions.append("Advanced analysis algorithms")
                    proposed_additions.append("Quality scoring mechanisms")
                    enhancement_type = "quality_check"
                
                if proposed_additions:
                    expected_increase = min(0.8, len(proposed_additions) * 0.2)
                    
                    recommendation = EnhancementRecommendation(
                        target_stage=node_name,
                        enhancement_type=enhancement_type,
                        proposed_additions=proposed_additions,
                        expected_value_increase=expected_increase,
                        implementation_effort=len(proposed_additions) * 0.3
                    )
                    self.enhancement_recommendations.append(recommendation)
    
    def generate_value_report(self) -> Dict[str, Any]:
        """Generate comprehensive value analysis report"""
        
        # Calculate summary statistics
        total_stages = len(self.stage_analyses)
        value_distribution = defaultdict(int)
        
        for analysis in self.stage_analyses.values():
            value_distribution[analysis.value_contribution.value] += 1
        
        avg_justification = sum(
            analysis.justification_score for analysis in self.stage_analyses.values()
        ) / total_stages if total_stages > 0 else 0.0
        
        # Identify top issues
        all_issues = []
        for analysis in self.stage_analyses.values():
            all_issues.extend(analysis.issues)
        
        issue_frequency = defaultdict(int)
        for issue in all_issues:
            issue_frequency[issue] += 1
        
        top_issues = sorted(issue_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "summary_statistics": {
                "total_stages_analyzed": total_stages,
                "value_distribution": dict(value_distribution),
                "average_justification_score": avg_justification,
                "total_consolidation_opportunities": len(self.consolidation_recommendations),
                "total_enhancement_opportunities": len(self.enhancement_recommendations)
            },
            "top_issues": top_issues,
            "stage_analyses": {
                name: {
                    "value_contribution": analysis.value_contribution.value,
                    "justification_score": analysis.justification_score,
                    "issues": analysis.issues,
                    "recommendations": analysis.recommendations,
                    "artifact_uniqueness": analysis.artifact_profile.uniqueness_score,
                    "processing_efficiency": analysis.processing_metrics.transformation_ratio / max(0.1, analysis.processing_metrics.execution_time)
                } for name, analysis in self.stage_analyses.items()
            },
            "consolidation_recommendations": [
                {
                    "target_stages": rec.target_stages,
                    "consolidation_type": rec.consolidation_type,
                    "expected_benefits": rec.expected_benefits,
                    "implementation_complexity": rec.implementation_complexity,
                    "risk_assessment": rec.risk_assessment
                } for rec in self.consolidation_recommendations
            ],
            "enhancement_recommendations": [
                {
                    "target_stage": rec.target_stage,
                    "enhancement_type": rec.enhancement_type,
                    "proposed_additions": rec.proposed_additions,
                    "expected_value_increase": rec.expected_value_increase,
                    "implementation_effort": rec.implementation_effort
                } for rec in self.enhancement_recommendations
            ]
        }
    
    def save_analysis_report(self, output_file: str = "pipeline_value_analysis_report.json"):
        """Save the analysis report to a JSON file"""
        report = self.generate_value_report()
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Pipeline value analysis report saved to {output_file}")
        return output_file


def main():
    """Run the pipeline value analysis system"""
    print("Pipeline Value Analysis System")
    print("=" * 50)
    
    # Initialize the system
    analyzer = PipelineValueAnalysisSystem()
    
    # Run the analysis
    print("Analyzing all 12 pipeline stages...")
    stage_analyses = analyzer.analyze_all_stages()
    
    # Generate and save report
    report_file = analyzer.save_analysis_report()
    
    # Print summary
    report = analyzer.generate_value_report()
    stats = report["summary_statistics"]
    
    print(f"\nAnalysis Complete!")
    print(f"Stages analyzed: {stats['total_stages_analyzed']}")
    print(f"Average justification score: {stats['average_justification_score']:.2f}")
    print(f"Value distribution: {stats['value_distribution']}")
    print(f"Consolidation opportunities: {stats['total_consolidation_opportunities']}")
    print(f"Enhancement opportunities: {stats['total_enhancement_opportunities']}")
    print(f"\nDetailed report saved to: {report_file}")
    
    # Print top issues
    if report["top_issues"]:
        print("\nTop Issues Identified:")
        for issue, count in report["top_issues"]:
            print(f"  • {issue} ({count} stages)")
    
    return report


if __name__ == "__main__":
    main()