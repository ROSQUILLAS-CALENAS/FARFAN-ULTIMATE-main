#!/usr/bin/env python3
"""
Comprehensive Artifact Generation System
Stage: analysis
Code: AG

Writes structured JSON files for all scoring levels including question-level evaluations,
dimension summaries, point summaries, meso cluster analysis, and macro alignment.
Implements deterministic ordering and UTF-8 encoding with consistent formatting.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from collections import OrderedDict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass 
class EvidenceReference:
    """Evidence reference with source metadata."""
    evidence_id: str
    source_type: str
    page_reference: str
    text_excerpt: str
    confidence_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with deterministic ordering."""
        return OrderedDict([
            ("evidence_id", self.evidence_id),
            ("source_type", self.source_type),
            ("page_reference", self.page_reference),
            ("text_excerpt", self.text_excerpt),
            ("confidence_score", self.confidence_score)
        ])


@dataclass
class QuestionEvaluation:
    """Question-level evaluation with evidence references."""
    question_id: str
    question_text: str
    response: str  # "Sí", "Parcial", "No", "NI"
    base_score: float
    evidence_completeness: float
    page_reference_quality: float
    final_score: float
    evidence_references: List[EvidenceReference]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with deterministic ordering."""
        return OrderedDict([
            ("question_id", self.question_id),
            ("question_text", self.question_text),
            ("response", self.response),
            ("base_score", self.base_score),
            ("evidence_completeness", self.evidence_completeness),
            ("page_reference_quality", self.page_reference_quality),
            ("final_score", self.final_score),
            ("evidence_references", [ref.to_dict() for ref in sorted(self.evidence_references, key=lambda x: x.evidence_id)])
        ])


@dataclass
class DimensionSummary:
    """Dimension-level summary with weighted aggregations."""
    dimension_id: str
    dimension_name: str
    questions: List[QuestionEvaluation]
    weighted_average: float
    total_questions: int
    confidence_interval: Dict[str, float]
    aggregation_weights: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with deterministic ordering."""
        return OrderedDict([
            ("dimension_id", self.dimension_id),
            ("dimension_name", self.dimension_name),
            ("weighted_average", self.weighted_average),
            ("total_questions", self.total_questions),
            ("confidence_interval", self.confidence_interval),
            ("aggregation_weights", self.aggregation_weights),
            ("questions", [q.to_dict() for q in sorted(self.questions, key=lambda x: x.question_id)])
        ])


@dataclass
class PointSummary:
    """Point-level summary with compliance classifications."""
    point_id: int
    point_name: str
    dimensions: List[DimensionSummary]
    final_score: float
    compliance_classification: str
    total_questions: int
    cluster_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with deterministic ordering."""
        return OrderedDict([
            ("point_id", self.point_id),
            ("point_name", self.point_name),
            ("final_score", self.final_score),
            ("compliance_classification", self.compliance_classification),
            ("total_questions", self.total_questions),
            ("cluster_id", self.cluster_id),
            ("dimensions", [d.to_dict() for d in sorted(self.dimensions, key=lambda x: x.dimension_id)])
        ])


@dataclass
class MesoClusterAnalysis:
    """Meso cluster analysis with cross-point linkages."""
    cluster_id: str
    cluster_name: str
    points: List[int]
    cross_point_linkages: Dict[str, List[str]]
    cluster_score: float
    coherence_metrics: Dict[str, float]
    evidence_density: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with deterministic ordering."""
        return OrderedDict([
            ("cluster_id", self.cluster_id),
            ("cluster_name", self.cluster_name),
            ("points", sorted(self.points)),
            ("cluster_score", self.cluster_score),
            ("coherence_metrics", dict(sorted(self.coherence_metrics.items()))),
            ("evidence_density", dict(sorted(self.evidence_density.items()))),
            ("cross_point_linkages", dict(sorted(self.cross_point_linkages.items())))
        ])


@dataclass
class MacroAlignment:
    """Macro alignment with overall Decálogo scores."""
    document_stem: str
    overall_decalogo_score: float
    cluster_scores: Dict[str, float]
    dimensional_alignment: Dict[str, float]
    coverage_metrics: Dict[str, float]
    compliance_distribution: Dict[str, int]
    recommendation_priority: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with deterministic ordering."""
        return OrderedDict([
            ("document_stem", self.document_stem),
            ("overall_decalogo_score", self.overall_decalogo_score),
            ("cluster_scores", dict(sorted(self.cluster_scores.items()))),
            ("dimensional_alignment", dict(sorted(self.dimensional_alignment.items()))),
            ("coverage_metrics", dict(sorted(self.coverage_metrics.items()))),
            ("compliance_distribution", dict(sorted(self.compliance_distribution.items()))),
            ("recommendation_priority", self.recommendation_priority)
        ])


class ArtifactGenerator:
    """
    Comprehensive artifact generation system that writes structured JSON files
    for all scoring levels with deterministic ordering and UTF-8 encoding.
    """
    
    def __init__(self, output_dir: str = "canonical_flow/analysis"):
        """
        Initialize artifact generator.
        
        Args:
            output_dir: Directory for artifact output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def write_json_artifact(self, data: Dict[str, Any], filepath: Path) -> None:
        """
        Write JSON artifact with UTF-8 encoding and consistent formatting.
        
        Args:
            data: Data to write
            filepath: Output file path
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=False)
            logger.info(f"Written artifact: {filepath}")
        except Exception as e:
            logger.error(f"Failed to write artifact {filepath}: {e}")
            raise
    
    def generate_question_artifacts(self, document_stem: str, questions: List[QuestionEvaluation]) -> None:
        """
        Generate question-level evaluation artifacts.
        
        Args:
            document_stem: Document identifier
            questions: List of question evaluations
        """
        # Group questions by dimension for organization
        dimensions_data = {}
        for question in questions:
            dim_id = self._extract_dimension_id(question.question_id)
            if dim_id not in dimensions_data:
                dimensions_data[dim_id] = []
            dimensions_data[dim_id].append(question)
        
        # Create deterministically ordered structure
        artifact_data = OrderedDict([
            ("metadata", OrderedDict([
                ("document_stem", document_stem),
                ("artifact_type", "question_evaluations"),
                ("generated_timestamp", datetime.utcnow().isoformat() + "Z"),
                ("total_questions", len(questions)),
                ("dimensions_included", sorted(dimensions_data.keys()))
            ])),
            ("questions_by_dimension", OrderedDict())
        ])
        
        # Add questions sorted by dimension and question_id
        for dim_id in sorted(dimensions_data.keys()):
            dim_questions = sorted(dimensions_data[dim_id], key=lambda x: x.question_id)
            artifact_data["questions_by_dimension"][dim_id] = [q.to_dict() for q in dim_questions]
        
        # Write artifact
        filepath = self.output_dir / f"{document_stem}_questions.json"
        self.write_json_artifact(artifact_data, filepath)
    
    def generate_dimension_artifacts(self, document_stem: str, dimensions: List[DimensionSummary]) -> None:
        """
        Generate dimension-level summary artifacts.
        
        Args:
            document_stem: Document identifier
            dimensions: List of dimension summaries
        """
        artifact_data = OrderedDict([
            ("metadata", OrderedDict([
                ("document_stem", document_stem),
                ("artifact_type", "dimension_summaries"),
                ("generated_timestamp", datetime.utcnow().isoformat() + "Z"),
                ("total_dimensions", len(dimensions)),
                ("dimensions_included", sorted([d.dimension_id for d in dimensions]))
            ])),
            ("dimension_summaries", OrderedDict())
        ])
        
        # Add dimensions sorted by dimension_id
        for dimension in sorted(dimensions, key=lambda x: x.dimension_id):
            artifact_data["dimension_summaries"][dimension.dimension_id] = dimension.to_dict()
        
        filepath = self.output_dir / f"{document_stem}_dimensions.json"
        self.write_json_artifact(artifact_data, filepath)
    
    def generate_point_artifacts(self, document_stem: str, points: List[PointSummary]) -> None:
        """
        Generate point-level summary artifacts.
        
        Args:
            document_stem: Document identifier
            points: List of point summaries
        """
        artifact_data = OrderedDict([
            ("metadata", OrderedDict([
                ("document_stem", document_stem),
                ("artifact_type", "point_summaries"),
                ("generated_timestamp", datetime.utcnow().isoformat() + "Z"),
                ("total_points", len(points)),
                ("points_included", sorted([p.point_id for p in points]))
            ])),
            ("point_summaries", OrderedDict())
        ])
        
        # Add points sorted by point_id
        for point in sorted(points, key=lambda x: x.point_id):
            artifact_data["point_summaries"][str(point.point_id)] = point.to_dict()
        
        filepath = self.output_dir / f"{document_stem}_points.json"
        self.write_json_artifact(artifact_data, filepath)
    
    def generate_meso_artifacts(self, document_stem: str, clusters: List[MesoClusterAnalysis]) -> None:
        """
        Generate meso cluster analysis artifacts.
        
        Args:
            document_stem: Document identifier
            clusters: List of meso cluster analyses
        """
        artifact_data = OrderedDict([
            ("metadata", OrderedDict([
                ("document_stem", document_stem),
                ("artifact_type", "meso_cluster_analysis"),
                ("generated_timestamp", datetime.utcnow().isoformat() + "Z"),
                ("total_clusters", len(clusters)),
                ("clusters_included", sorted([c.cluster_id for c in clusters]))
            ])),
            ("cluster_analysis", OrderedDict())
        ])
        
        # Add clusters sorted by cluster_id
        for cluster in sorted(clusters, key=lambda x: x.cluster_id):
            artifact_data["cluster_analysis"][cluster.cluster_id] = cluster.to_dict()
        
        filepath = self.output_dir / f"{document_stem}_meso.json"
        self.write_json_artifact(artifact_data, filepath)
    
    def generate_macro_artifacts(self, document_stem: str, macro_alignment: MacroAlignment) -> None:
        """
        Generate macro alignment artifacts.
        
        Args:
            document_stem: Document identifier
            macro_alignment: Macro alignment analysis
        """
        artifact_data = OrderedDict([
            ("metadata", OrderedDict([
                ("document_stem", document_stem),
                ("artifact_type", "macro_alignment"),
                ("generated_timestamp", datetime.utcnow().isoformat() + "Z")
            ])),
            ("macro_alignment", macro_alignment.to_dict())
        ])
        
        filepath = self.output_dir / f"{document_stem}_macro.json"
        self.write_json_artifact(artifact_data, filepath)
    
    def generate_comprehensive_artifacts(
        self, 
        document_stem: str,
        questions: List[QuestionEvaluation],
        dimensions: List[DimensionSummary],
        points: List[PointSummary],
        clusters: List[MesoClusterAnalysis],
        macro_alignment: MacroAlignment
    ) -> Dict[str, str]:
        """
        Generate all artifact types for a document.
        
        Args:
            document_stem: Document identifier
            questions: Question evaluations
            dimensions: Dimension summaries
            points: Point summaries
            clusters: Meso cluster analyses
            macro_alignment: Macro alignment
            
        Returns:
            Dictionary mapping artifact types to file paths
        """
        artifacts = {}
        
        try:
            # Generate all artifact types
            self.generate_question_artifacts(document_stem, questions)
            artifacts["questions"] = str(self.output_dir / f"{document_stem}_questions.json")
            
            self.generate_dimension_artifacts(document_stem, dimensions)
            artifacts["dimensions"] = str(self.output_dir / f"{document_stem}_dimensions.json")
            
            self.generate_point_artifacts(document_stem, points)
            artifacts["points"] = str(self.output_dir / f"{document_stem}_points.json")
            
            self.generate_meso_artifacts(document_stem, clusters)
            artifacts["meso"] = str(self.output_dir / f"{document_stem}_meso.json")
            
            self.generate_macro_artifacts(document_stem, macro_alignment)
            artifacts["macro"] = str(self.output_dir / f"{document_stem}_macro.json")
            
            logger.info(f"Generated {len(artifacts)} artifact types for document: {document_stem}")
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive artifacts for {document_stem}: {e}")
            raise
        
        return artifacts
    
    def validate_artifacts(self, document_stem: str) -> Dict[str, bool]:
        """
        Validate that all expected artifacts exist and are valid JSON.
        
        Args:
            document_stem: Document identifier
            
        Returns:
            Dictionary mapping artifact types to validation status
        """
        suffixes = ["questions", "dimensions", "points", "meso", "macro"]
        validation_results = {}
        
        for suffix in suffixes:
            filepath = self.output_dir / f"{document_stem}_{suffix}.json"
            validation_results[suffix] = self._validate_json_file(filepath)
        
        return validation_results
    
    def discover_artifacts(self, pattern: str = "*") -> Dict[str, List[str]]:
        """
        Discover existing artifacts using filename conventions.
        
        Args:
            pattern: Glob pattern for discovery
            
        Returns:
            Dictionary mapping document stems to available artifact types
        """
        artifacts_by_document = {}
        
        for json_file in self.output_dir.glob(f"{pattern}*.json"):
            # Parse filename to extract document stem and suffix
            parts = json_file.stem.split('_')
            if len(parts) >= 2:
                # Last part is the suffix, rest is document stem
                suffix = parts[-1]
                document_stem = '_'.join(parts[:-1])
                
                if document_stem not in artifacts_by_document:
                    artifacts_by_document[document_stem] = []
                
                artifacts_by_document[document_stem].append(suffix)
        
        # Sort for deterministic results
        for document_stem in artifacts_by_document:
            artifacts_by_document[document_stem].sort()
        
        return dict(sorted(artifacts_by_document.items()))
    
    def _extract_dimension_id(self, question_id: str) -> str:
        """Extract dimension ID from question ID."""
        # Assume question IDs follow pattern like "DE-1-Q1", "DE-2-Q5", etc.
        parts = question_id.split('-')
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1]}"
        return "UNKNOWN"
    
    def _validate_json_file(self, filepath: Path) -> bool:
        """Validate that a JSON file exists and is valid."""
        try:
            if not filepath.exists():
                return False
            
            with open(filepath, 'r', encoding='utf-8') as f:
                json.load(f)
            return True
            
        except Exception:
            return False


def create_sample_data(document_stem: str = "SAMPLE-DOC") -> Tuple[
    List[QuestionEvaluation],
    List[DimensionSummary], 
    List[PointSummary],
    List[MesoClusterAnalysis],
    MacroAlignment
]:
    """Create sample data for testing."""
    
    # Sample evidence references
    evidence_refs = [
        EvidenceReference("E001", "document", "p. 15", "Sample evidence text", 0.85),
        EvidenceReference("E002", "interview", "p. 23", "Additional evidence", 0.92)
    ]
    
    # Sample questions
    questions = [
        QuestionEvaluation(
            "DE-1-Q1", "Sample question 1?", "Sí", 1.0, 0.8, 0.9, 0.95, evidence_refs
        ),
        QuestionEvaluation(
            "DE-1-Q2", "Sample question 2?", "Parcial", 0.5, 0.6, 0.7, 0.58, evidence_refs
        )
    ]
    
    # Sample dimension
    dimensions = [
        DimensionSummary(
            "DE-1", "Productos", questions, 0.77, 2, 
            {"lower": 0.65, "upper": 0.89}, {"equal_weight": 1.0}
        )
    ]
    
    # Sample point
    points = [
        PointSummary(
            1, "Point 1", dimensions, 0.77, "Satisfactory", 2, "CLUSTER-1"
        )
    ]
    
    # Sample cluster
    clusters = [
        MesoClusterAnalysis(
            "CLUSTER-1", "Sample Cluster", [1], 
            {"internal": ["link1"], "external": ["link2"]}, 0.77,
            {"coherence": 0.85, "consistency": 0.78}, {"high": 5, "medium": 3}
        )
    ]
    
    # Sample macro alignment
    macro = MacroAlignment(
        document_stem, 0.77, {"CLUSTER-1": 0.77}, {"DE-1": 0.77}, 
        {"total": 0.85}, {"Satisfactory": 1}, ["Priority 1"]
    )
    
    return questions, dimensions, points, clusters, macro


# CLI Interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate comprehensive artifacts")
    parser.add_argument("document_stem", help="Document stem identifier")
    parser.add_argument("--sample", action="store_true", help="Generate sample data")
    parser.add_argument("--validate", action="store_true", help="Validate artifacts")
    parser.add_argument("--discover", action="store_true", help="Discover artifacts")
    parser.add_argument("--output-dir", default="canonical_flow/analysis", help="Output directory")
    
    args = parser.parse_args()
    
    generator = ArtifactGenerator(args.output_dir)
    
    if args.discover:
        artifacts = generator.discover_artifacts()
        print(json.dumps(artifacts, indent=2))
        
    elif args.validate:
        validation = generator.validate_artifacts(args.document_stem)
        print(json.dumps(validation, indent=2))
        
    elif args.sample:
        questions, dimensions, points, clusters, macro = create_sample_data(args.document_stem)
        
        artifacts = generator.generate_comprehensive_artifacts(
            args.document_stem, questions, dimensions, points, clusters, macro
        )
        
        print(f"Generated sample artifacts for {args.document_stem}:")
        for artifact_type, filepath in artifacts.items():
            print(f"  {artifact_type}: {filepath}")
            
    else:
        print(f"Use --sample, --validate, or --discover with document_stem: {args.document_stem}")