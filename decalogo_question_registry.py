"""
DecalogoQuestionRegistry - Registry for Decálogo Question Management

This module provides a comprehensive registry system for managing Decálogo questions
with stable identifiers, cluster mappings, and dimension distribution validation.
"""

# # # from dataclasses import dataclass, field  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, List, Tuple, Set, Any, Optional  # Module not found  # Module not found  # Module not found
# # # from enum import Enum  # Module not found  # Module not found  # Module not found
import json
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found


class ClusterType(Enum):
    """Four cluster types for Decálogo point classification."""
    PARTICIPATIVO = "participativo"
    PLANIFICACION = "planificacion" 
    TERRITORIAL = "territorial"
    SOSTENIBLE = "sostenible"


class Dimension(Enum):
    """Question dimensions with their question counts per point."""
    DE_1 = ("DE-1", 6)
    DE_2 = ("DE-2", 21)
    DE_3 = ("DE-3", 8)
    DE_4 = ("DE-4", 12)  # Updated to make 47 total (6+21+8+12=47)
    
    def __init__(self, code: str, question_count: int):
        self.code = code
        self.question_count = question_count


@dataclass(frozen=True)
class DecalogoPoint:
    """Immutable representation of a Decálogo point."""
    point_id: int
    title: str
    cluster: ClusterType
    
    def __post_init__(self):
        if not (1 <= self.point_id <= 10):
            raise ValueError(f"Point ID must be between 1-10, got {self.point_id}")


@dataclass(frozen=True)  
class Question:
    """Immutable representation of a question with stable identifiers."""
    question_id: str
    point_id: int
    dimension: Dimension
    sequence: int
    text: str = ""
    
    def __post_init__(self):
        if not self.question_id.startswith(f"P{self.point_id:02d}_{self.dimension.code}"):
            raise ValueError(f"Invalid question ID format: {self.question_id}")
        if not (1 <= self.sequence <= self.dimension.question_count):
            raise ValueError(f"Invalid sequence {self.sequence} for dimension {self.dimension.code}")


class DecalogoQuestionRegistry:
    """
    Registry for managing Decálogo questions with stable identifiers and validation.
    
    Provides exactly 10 Decálogo points mapped to 4 clusters, with 47 questions per point
    distributed across dimensions: DE-1 (6), DE-2 (21), DE-3 (8), DE-4 (12).
    Total of 470 questions (10 points × 47 questions per point).
    """
    
    REGISTRY_VERSION = "1.0.0"
    EXPECTED_TOTAL_QUESTIONS = 470  # 10 points × 47 questions per point
    QUESTIONS_PER_POINT = 47
    
    def __init__(self):
        """Initialize the registry with stable point definitions and questions."""
        self._points = self._define_decalogo_points()
        self._questions = self._generate_all_questions()
        self._cluster_mappings = self._build_cluster_mappings()
        self._validate_registry()
    
    def _define_decalogo_points(self) -> Dict[int, DecalogoPoint]:
        """Define exactly 10 Decálogo points with stable IDs, titles, and cluster mappings."""
        points_data = [
            (1, "Derecho a la vida, seguridad y convivencia", ClusterType.PARTICIPATIVO),
            (2, "Derecho a la salud", ClusterType.PLANIFICACION),
            (3, "Derecho a la educación", ClusterType.PLANIFICACION),
            (4, "Derecho a la alimentación", ClusterType.PLANIFICACION),
            (5, "Derechos de las víctimas y construcción de paz", ClusterType.PARTICIPATIVO),
            (6, "Derechos de las mujeres", ClusterType.PARTICIPATIVO),
            (7, "Derechos de niñas, niños y adolescentes", ClusterType.PARTICIPATIVO),
            (8, "Líderes y defensores de derechos humanos", ClusterType.PARTICIPATIVO),
            (9, "Derechos de los pueblos étnicos", ClusterType.TERRITORIAL),
            (10, "Derecho a un ambiente sano", ClusterType.SOSTENIBLE)
        ]
        
        return {
            point_id: DecalogoPoint(point_id, title, cluster)
            for point_id, title, cluster in points_data
        }
    
    def _generate_all_questions(self) -> Dict[str, Question]:
        """Generate all questions with stable sequential IDs within each point-dimension."""
        questions = {}
        
        for point_id in range(1, 11):  # Points 1-10
            for dimension in Dimension:
                for seq in range(1, dimension.question_count + 1):
                    question_id = f"P{point_id:02d}_{dimension.code}_{seq:02d}"
                    question = Question(
                        question_id=question_id,
                        point_id=point_id,
                        dimension=dimension,
                        sequence=seq,
                        text=f"Question {seq} for Point {point_id}, Dimension {dimension.code}"
                    )
                    questions[question_id] = question
        
        return questions
    
    def _build_cluster_mappings(self) -> Dict[ClusterType, List[int]]:
        """Build cluster to point ID mappings."""
        mappings = {cluster: [] for cluster in ClusterType}
        
        for point_id, point in self._points.items():
            mappings[point.cluster].append(point_id)
        
        # Sort point IDs within each cluster for deterministic ordering
        for cluster in mappings:
            mappings[cluster].sort()
        
        return mappings
    
    def _validate_registry(self):
        """Validate registry consistency and requirements."""
        # Validate exactly 10 points
        if len(self._points) != 10:
            raise ValueError(f"Expected 10 points, found {len(self._points)}")
        
        # Validate total question count
        if len(self._questions) != self.EXPECTED_TOTAL_QUESTIONS:
            raise ValueError(f"Expected {self.EXPECTED_TOTAL_QUESTIONS} questions, found {len(self._questions)}")
        
        # Validate questions per point
        for point_id in range(1, 11):
            point_questions = [q for q in self._questions.values() if q.point_id == point_id]
            if len(point_questions) != self.QUESTIONS_PER_POINT:
                raise ValueError(f"Point {point_id} has {len(point_questions)} questions, expected {self.QUESTIONS_PER_POINT}")
        
        # Validate dimension distribution per point
        for point_id in range(1, 11):
            for dimension in Dimension:
                dim_questions = [
                    q for q in self._questions.values() 
                    if q.point_id == point_id and q.dimension == dimension
                ]
                if len(dim_questions) != dimension.question_count:
                    raise ValueError(
                        f"Point {point_id}, Dimension {dimension.code}: "
                        f"found {len(dim_questions)} questions, expected {dimension.question_count}"
                    )
        
        # Validate all clusters are used
        used_clusters = set(point.cluster for point in self._points.values())
        if used_clusters != set(ClusterType):
            raise ValueError(f"Not all clusters used. Used: {used_clusters}, Required: {set(ClusterType)}")
    
    @property
    def registry_version(self) -> str:
        """Get the registry version for deterministic processing."""
        return self.REGISTRY_VERSION
    
    @property 
    def points(self) -> Dict[int, DecalogoPoint]:
        """Get all Decálogo points."""
        return self._points.copy()
    
    @property
    def questions(self) -> Dict[str, Question]:
        """Get all questions."""
        return self._questions.copy()
    
    @property
    def cluster_mappings(self) -> Dict[ClusterType, List[int]]:
        """Get cluster to point ID mappings."""
        return {k: v.copy() for k, v in self._cluster_mappings.items()}
    
    def get_point(self, point_id: int) -> Optional[DecalogoPoint]:
        """Get a specific Decálogo point."""
        return self._points.get(point_id)
    
    def get_question(self, question_id: str) -> Optional[Question]:
        """Get a specific question."""
        return self._questions.get(question_id)
    
    def get_questions_for_point(self, point_id: int) -> List[Question]:
        """Get all questions for a specific point, ordered by dimension and sequence."""
        point_questions = [q for q in self._questions.values() if q.point_id == point_id]
        # Sort by dimension order and sequence
        dimension_order = {dim: i for i, dim in enumerate(Dimension)}
        return sorted(
            point_questions, 
            key=lambda q: (dimension_order[q.dimension], q.sequence)
        )
    
    def get_questions_for_dimension(self, dimension: Dimension, point_id: Optional[int] = None) -> List[Question]:
        """Get questions for a specific dimension, optionally filtered by point."""
        questions = [q for q in self._questions.values() if q.dimension == dimension]
        if point_id is not None:
            questions = [q for q in questions if q.point_id == point_id]
        return sorted(questions, key=lambda q: (q.point_id, q.sequence))
    
    def get_points_for_cluster(self, cluster: ClusterType) -> List[DecalogoPoint]:
        """Get all points for a specific cluster."""
        point_ids = self._cluster_mappings.get(cluster, [])
        return [self._points[pid] for pid in point_ids]
    
    def validate_total_question_count(self) -> Dict[str, Any]:
        """Validate the total question count matches expectations."""
        actual_count = len(self._questions)
        return {
            "expected_count": self.EXPECTED_TOTAL_QUESTIONS,
            "actual_count": actual_count,
            "is_valid": actual_count == self.EXPECTED_TOTAL_QUESTIONS,
            "discrepancy": actual_count - self.EXPECTED_TOTAL_QUESTIONS
        }
    
    def validate_dimension_distribution(self) -> Dict[str, Any]:
        """Validate dimension distribution per point."""
        results = {}
        all_valid = True
        
        for point_id in range(1, 11):
            point_validation = {}
            for dimension in Dimension:
                dim_questions = [
                    q for q in self._questions.values() 
                    if q.point_id == point_id and q.dimension == dimension
                ]
                expected = dimension.question_count
                actual = len(dim_questions)
                is_valid = actual == expected
                
                point_validation[dimension.code] = {
                    "expected": expected,
                    "actual": actual,
                    "is_valid": is_valid
                }
                
                if not is_valid:
                    all_valid = False
            
            # Calculate total for this point
            total_actual = sum(v["actual"] for v in point_validation.values())
            point_validation["total"] = {
                "expected": self.QUESTIONS_PER_POINT,
                "actual": total_actual,
                "is_valid": total_actual == self.QUESTIONS_PER_POINT
            }
            
            results[f"point_{point_id}"] = point_validation
        
        results["overall_valid"] = all_valid
        return results
    
    def validate_cluster_point_mapping(self) -> Dict[str, Any]:
        """Validate cluster-point mapping consistency."""
        results = {}
        
        # Check each cluster has points
        for cluster in ClusterType:
            cluster_points = self._cluster_mappings.get(cluster, [])
            results[cluster.value] = {
                "point_count": len(cluster_points),
                "points": cluster_points,
                "has_points": len(cluster_points) > 0
            }
        
        # Check all points are mapped
        mapped_points = set()
        for points_list in self._cluster_mappings.values():
            mapped_points.update(points_list)
        
        expected_points = set(range(1, 11))
        results["mapping_complete"] = {
            "all_points_mapped": mapped_points == expected_points,
            "mapped_points": sorted(mapped_points),
            "expected_points": sorted(expected_points),
            "missing_points": sorted(expected_points - mapped_points),
            "extra_points": sorted(mapped_points - expected_points)
        }
        
        return results
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the registry."""
        return {
            "version": self.registry_version,
            "points_count": len(self._points),
            "questions_count": len(self._questions),
            "questions_per_point": self.QUESTIONS_PER_POINT,
            "expected_total_questions": self.EXPECTED_TOTAL_QUESTIONS,
            "dimensions": {
                dim.code: {
                    "questions_per_point": dim.question_count,
                    "total_questions": dim.question_count * 10
                }
                for dim in Dimension
            },
            "clusters": {
                cluster.value: len(points)
                for cluster, points in self._cluster_mappings.items()
            },
            "validation_status": {
                "total_count_valid": self.validate_total_question_count()["is_valid"],
                "dimension_distribution_valid": self.validate_dimension_distribution()["overall_valid"],
                "cluster_mapping_valid": self.validate_cluster_point_mapping()["mapping_complete"]["all_points_mapped"]
            }
        }
    
    def export_to_json(self, file_path: str) -> None:
        """Export registry to JSON file."""
        data = {
            "registry_version": self.registry_version,
            "points": {
                str(pid): {
                    "point_id": point.point_id,
                    "title": point.title,
                    "cluster": point.cluster.value
                }
                for pid, point in self._points.items()
            },
            "questions": {
                qid: {
                    "question_id": q.question_id,
                    "point_id": q.point_id,
                    "dimension": q.dimension.code,
                    "sequence": q.sequence,
                    "text": q.text
                }
                for qid, q in self._questions.items()
            },
            "cluster_mappings": {
                cluster.value: points
                for cluster, points in self._cluster_mappings.items()
            },
            "summary": self.get_registry_summary()
        }
        
        Path(file_path).write_text(json.dumps(data, indent=2, ensure_ascii=False))


# Global registry instance for deterministic access
_registry_instance = None

def get_decalogo_registry() -> DecalogoQuestionRegistry:
    """Get the global registry instance (singleton pattern)."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = DecalogoQuestionRegistry()
    return _registry_instance


if __name__ == "__main__":
    # Test the registry
    registry = get_decalogo_registry()
    print(f"Registry Version: {registry.registry_version}")
    print(f"Total Questions: {len(registry.questions)}")
    print(f"Total Points: {len(registry.points)}")
    
    # Validate total count
    total_validation = registry.validate_total_question_count()
    print(f"Total count validation: {total_validation}")
    
    # Validate dimension distribution
    dim_validation = registry.validate_dimension_distribution()
    print(f"Dimension distribution valid: {dim_validation['overall_valid']}")
    
    # Validate cluster mapping
    cluster_validation = registry.validate_cluster_point_mapping()
    print(f"Cluster mapping valid: {cluster_validation['mapping_complete']['all_points_mapped']}")
    
    # Show cluster distribution
    print("\nCluster distribution:")
    for cluster, points in registry.cluster_mappings.items():
        print(f"{cluster.value}: {points} ({len(points)} points)")
    
    # Show dimensions per point
    print("\nDimension distribution verification:")
    for dim in Dimension:
        print(f"{dim.code}: {dim.question_count} questions per point, {dim.question_count * 10} total")
    
    # Show registry summary
    print("\nRegistry Summary:")
    summary = registry.get_registry_summary()
    print(json.dumps(summary, indent=2, ensure_ascii=False))