"""
Decálogo Question Registry with Validation and Coverage Tracking

Validates exactly 47 questions per Decálogo point with specified dimension distribution:
- DE-1: 6 questions  
- DE-2: 21 questions
- DE-3: 8 questions
- DE-4: 8 questions

Includes preflight validation and coverage artifact generation.
"""

import json
import hashlib
import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from collections import OrderedDict


class DecalogoQuestionValidationError(Exception):
    """Exception raised when question registry validation fails"""
    pass


@dataclass 
class DecalogoQuestion:
    """Single question from Decálogo registry"""
    question_id: str
    point_number: int
    dimension_code: str
    question_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Ensure deterministic question_id generation
        if not self.question_id:
            self.question_id = self._generate_deterministic_id()
    
    def _generate_deterministic_id(self) -> str:
        """Generate stable question ID using hash of point, dimension, and text"""
        id_input = f"{self.point_number}|{self.dimension_code}|{self.question_text}"
        hash_obj = hashlib.sha256(id_input.encode('utf-8'))
        return f"DQ_{self.point_number:02d}_{self.dimension_code}_{hash_obj.hexdigest()[:8]}"


@dataclass
class CoverageCell:
    """Represents a single cell in the coverage matrix"""
    point: int  # Decálogo point (1-10)
    dimension: str  # DE dimension (DE-1, DE-2, DE-3, DE-4)
    expected_count: int  # Expected questions for this cell
    actual_count: int = 0  # Actual questions answered
    completion_percentage: float = 0.0
    has_gap: bool = True
    questions: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.update_metrics()
    
    def update_metrics(self):
        """Update completion percentage and gap status"""
        if self.expected_count > 0:
            self.completion_percentage = (self.actual_count / self.expected_count) * 100
            self.has_gap = self.completion_percentage < 100.0
        else:
            self.completion_percentage = 0.0
            self.has_gap = True


@dataclass
class CoverageMatrix:
    """10x4 coverage matrix for Decálogo points vs DE dimensions"""
    matrix: Dict[str, CoverageCell] = field(default_factory=dict)
    total_expected: int = 470  # 47 questions × 10 points
    total_actual: int = 0
    document_id: str = ""
    
    # Expected distribution per point as specified
    DE_DISTRIBUTION = {
        "DE-1": 6,
        "DE-2": 21, 
        "DE-3": 8,
        "DE-4": 8
    }
    
    def __post_init__(self):
        self.initialize_matrix()
    
    def initialize_matrix(self):
        """Initialize matrix with expected counts"""
        for point in range(1, 11):  # Points 1-10
            for dimension in ["DE-1", "DE-2", "DE-3", "DE-4"]:
                cell_key = f"P{point}_{dimension}"
                expected = self.DE_DISTRIBUTION[dimension]
                self.matrix[cell_key] = CoverageCell(
                    point=point,
                    dimension=dimension,
                    expected_count=expected
                )
    
    def add_question(self, point: int, dimension: str, question_id: str):
        """Add a question to the coverage matrix"""
        if 1 <= point <= 10 and dimension in self.DE_DISTRIBUTION:
            cell_key = f"P{point}_{dimension}"
            if cell_key in self.matrix:
                if question_id not in self.matrix[cell_key].questions:
                    self.matrix[cell_key].questions.append(question_id)
                    self.matrix[cell_key].actual_count += 1
                    self.matrix[cell_key].update_metrics()
                    self.total_actual += 1
    
    def get_gaps(self) -> List[Dict[str, Any]]:
        """Get all cells with coverage gaps"""
        gaps = []
        for cell_key, cell in self.matrix.items():
            if cell.has_gap:
                gaps.append({
                    "cell": cell_key,
                    "point": cell.point,
                    "dimension": cell.dimension,
                    "expected": cell.expected_count,
                    "actual": cell.actual_count,
                    "gap": cell.expected_count - cell.actual_count,
                    "completion_percentage": cell.completion_percentage
                })
        return sorted(gaps, key=lambda x: x["completion_percentage"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert matrix to serializable dictionary"""
        matrix_data = {}
        for cell_key, cell in self.matrix.items():
            matrix_data[cell_key] = {
                "point": cell.point,
                "dimension": cell.dimension,
                "expected_count": cell.expected_count,
                "actual_count": cell.actual_count,
                "completion_percentage": cell.completion_percentage,
                "has_gap": cell.has_gap,
                "questions": cell.questions
            }
        
        gaps = self.get_gaps()
        
        return {
            "document_id": self.document_id,
            "matrix": matrix_data,
            "summary": {
                "total_expected": self.total_expected,
                "total_actual": self.total_actual,
                "completion_percentage": (self.total_actual / self.total_expected * 100) if self.total_expected > 0 else 0,
                "gaps_count": len(gaps),
                "cells_with_gaps": len([g for g in gaps if g["gap"] > 0])
            },
            "gaps": gaps,
            "distribution_per_dimension": {
                dim: sum(cell.actual_count for cell in self.matrix.values() if cell.dimension == dim)
                for dim in self.DE_DISTRIBUTION.keys()
            },
            "timestamp": datetime.datetime.now().isoformat()
        }


class DecalogoQuestionRegistry:
    """Registry of all 470 questions (47 per point across 10 Decálogo points) with validation"""
    
    # Exact dimension distribution as specified
    REQUIRED_DIMENSION_DISTRIBUTION = {
        "DE-1": 6,
        "DE-2": 21, 
        "DE-3": 8,
        "DE-4": 8
    }
    
    REQUIRED_QUESTIONS_PER_POINT = 47
    TOTAL_DECALOGO_POINTS = 10
    TOTAL_EXPECTED_QUESTIONS = REQUIRED_QUESTIONS_PER_POINT * TOTAL_DECALOGO_POINTS  # 470
    
    def __init__(self, validate_on_init: bool = True):
        self.questions = self._build_complete_registry()
        self._ensure_deterministic_ordering()
        
        if validate_on_init:
            self.run_preflight_validation()
    
    def _build_complete_registry(self) -> List[DecalogoQuestion]:
        """Build complete registry of 470 questions with exact distribution"""
        questions = []
        
        # Define the 10 Decálogo points 
        point_templates = {
            1: "Derecho a la vida, a la seguridad y a la convivencia",
            2: "Igualdad de la mujer y equidad de género", 
            3: "Derecho humano al agua, ambiente sano y gestión del riesgo",
            4: "Derecho humano a la salud",
            5: "Derechos de las víctimas y construcción de paz",
            6: "Derechos de la niñez, la juventud y fortalecimiento familiar",
            7: "Ordenamiento territorial, tierras y desarrollo rural",
            8: "Protección y promoción de líderes sociales",
            9: "Personas privadas de la libertad", 
            10: "Trabajo decente y seguridad social"
        }
        
        # Base question patterns for each dimension (exactly matching required counts)
        dimension_patterns = {
            "DE-1": [
                "¿Define productos medibles para {}?",
                "¿Establece indicadores cuantitativos para {}?",
                "¿Incluye metas específicas relacionadas con {}?",
                "¿Delimita la población objetivo para {}?",
                "¿Especifica metodologías de medición para {}?",
                "¿Define cronograma de entrega para productos de {}?"
            ],  # Exactly 6 questions as specified
            "DE-2": [
                "¿Identifica los productos como eficientes para {}?",
                "¿Evalúa la relación costo-beneficio en {}?", 
                "¿Analiza la eficiencia operativa para {}?",
                "¿Compara alternativas de implementación en {}?",
                "¿Optimiza recursos asignados a {}?",
                "¿Minimiza costos de operación en {}?",
                "¿Maximiza cobertura con recursos disponibles para {}?",
                "¿Reduce tiempos de ejecución en {}?",
                "¿Simplifica procesos relacionados con {}?",
                "¿Integra tecnologías para eficiencia en {}?",
                "¿Elimina duplicidades en {}?",
                "¿Automatiza procedimientos de {}?",
                "¿Mide productividad en {}?",
                "¿Evalúa impacto por peso invertido en {}?",
                "¿Optimiza cadena de valor en {}?",
                "¿Reduce desperdicio de recursos en {}?",
                "¿Mejora tiempo de respuesta en {}?",
                "¿Incrementa satisfacción del usuario en {}?",
                "¿Reduce complejidad administrativa en {}?",
                "¿Agiliza procesos de decisión en {}?",
                "¿Facilita acceso a servicios de {}?"
            ],  # Exactly 21 questions as specified
            "DE-3": [
                "¿Los productos generan efectos en {}?",
                "¿Se evidencian cambios atribuibles en {}?",
                "¿Los resultados son sostenibles en {}?",
                "¿Existe progreso medible en {}?",
                "¿Se documentan mejoras en {}?",
                "¿Los cambios se mantienen en el tiempo para {}?",
                "¿Hay transformaciones observables en {}?",
                "¿Se registra evolución positiva en {}?"
            ],  # Exactly 8 questions as specified
            "DE-4": [
                "¿El impacto esperado está definido y alineado al {}?",
                "¿Se proyectan transformaciones de largo plazo en {}?",
                "¿Existe contribución al desarrollo territorial en {}?",
                "¿Se espera cambio estructural en {}?",
                "¿El impacto trasciende la población directa en {}?",
                "¿Contribuye al cierre de brechas en {}?",
                "¿Fortalece capacidades institucionales para {}?",
                "¿Genera sinergias intersectoriales en {}?"
            ]  # Exactly 8 questions as specified
        }
        
        # Generate exactly 47 questions per point with exact dimension distribution
        for point_num, point_name in sorted(point_templates.items()):
            point_questions = []
            
            # Generate questions for each dimension with exact counts
            for dim_code, required_count in self.REQUIRED_DIMENSION_DISTRIBUTION.items():
                patterns = dimension_patterns[dim_code]
                
                # Ensure we have exactly the required number of questions
                if len(patterns) < required_count:
                    raise DecalogoQuestionValidationError(
                        f"Insufficient patterns for {dim_code}: need {required_count}, have {len(patterns)}"
                    )
                
                for i in range(required_count):
                    pattern = patterns[i % len(patterns)]  # Cycle through patterns if needed
                    question_text = pattern.format(point_name)
                    
                    question = DecalogoQuestion(
                        question_id="",  # Will be auto-generated
                        point_number=point_num,
                        dimension_code=dim_code,
                        question_text=question_text,
                        metadata={
                            "pattern_index": i,
                            "point_name": point_name,
                            "category": "dimension_evaluation"
                        }
                    )
                    point_questions.append(question)
            
            # Validate exactly 43 questions so far (6+21+8+8)
            if len(point_questions) != 43:
                raise DecalogoQuestionValidationError(
                    f"Point {point_num} has {len(point_questions)} dimension questions, expected 43"
                )
            
            # Add exactly 4 additional point-specific questions to reach 47 total
            for i in range(4):
                additional_text = f"¿Cómo se garantiza la sostenibilidad específica del punto {point_num} ({point_name}) considerando aspectos particulares del territorio? (Pregunta adicional {i+1})"
                
                question = DecalogoQuestion(
                    question_id="",  # Will be auto-generated
                    point_number=point_num,
                    dimension_code="GEN",  # General/cross-cutting
                    question_text=additional_text,
                    metadata={
                        "pattern_index": i,
                        "point_name": point_name, 
                        "category": "point_specific"
                    }
                )
                point_questions.append(question)
            
            # Final validation: exactly 47 questions per point
            if len(point_questions) != self.REQUIRED_QUESTIONS_PER_POINT:
                raise DecalogoQuestionValidationError(
                    f"Point {point_num} has {len(point_questions)} questions, expected {self.REQUIRED_QUESTIONS_PER_POINT}"
                )
            
            questions.extend(point_questions)
        
        return questions
    
    def run_preflight_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive validation before any question evaluation begins.
        Ensures the registry is properly configured and all 470 total questions are accounted for.
        """
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "summary": {},
            "dimension_validation": {},
            "point_validation": {},
            "total_validation": {}
        }
        
        try:
            # 1. Validate total question count (470)
            total_questions = len(self.questions)
            validation_results["total_validation"] = {
                "expected_total": self.TOTAL_EXPECTED_QUESTIONS,
                "actual_total": total_questions,
                "is_valid": total_questions == self.TOTAL_EXPECTED_QUESTIONS
            }
            
            if total_questions != self.TOTAL_EXPECTED_QUESTIONS:
                validation_results["is_valid"] = False
                validation_results["errors"].append(
                    f"Total question count mismatch: expected {self.TOTAL_EXPECTED_QUESTIONS}, got {total_questions}"
                )
            
            # 2. Validate per-point distribution (47 per point)
            point_counts = {}
            for point_num in range(1, self.TOTAL_DECALOGO_POINTS + 1):
                point_questions = self.get_questions_by_point(point_num)
                point_count = len(point_questions)
                point_counts[point_num] = point_count
                
                point_is_valid = point_count == self.REQUIRED_QUESTIONS_PER_POINT
                validation_results["point_validation"][point_num] = {
                    "expected": self.REQUIRED_QUESTIONS_PER_POINT,
                    "actual": point_count,
                    "is_valid": point_is_valid
                }
                
                if not point_is_valid:
                    validation_results["is_valid"] = False
                    validation_results["errors"].append(
                        f"Point {point_num} question count mismatch: expected {self.REQUIRED_QUESTIONS_PER_POINT}, got {point_count}"
                    )
            
            # 3. Validate dimension distribution per point
            for point_num in range(1, self.TOTAL_DECALOGO_POINTS + 1):
                dimension_counts = {}
                point_questions = self.get_questions_by_point(point_num)
                
                for question in point_questions:
                    dim = question.dimension_code
                    dimension_counts[dim] = dimension_counts.get(dim, 0) + 1
                
                # Check each required dimension
                for dim_code, expected_count in self.REQUIRED_DIMENSION_DISTRIBUTION.items():
                    actual_count = dimension_counts.get(dim_code, 0)
                    dim_key = f"P{point_num}_{dim_code}"
                    
                    is_valid = actual_count == expected_count
                    validation_results["dimension_validation"][dim_key] = {
                        "point": point_num,
                        "dimension": dim_code,
                        "expected": expected_count,
                        "actual": actual_count,
                        "is_valid": is_valid
                    }
                    
                    if not is_valid:
                        validation_results["is_valid"] = False
                        validation_results["errors"].append(
                            f"Point {point_num}, Dimension {dim_code}: expected {expected_count}, got {actual_count}"
                        )
            
            # 4. Validate total dimension distribution across all points
            total_dimension_counts = {}
            for dim_code, expected_per_point in self.REQUIRED_DIMENSION_DISTRIBUTION.items():
                questions = self.get_questions_by_dimension(dim_code)
                actual_total = len(questions)
                expected_total = expected_per_point * self.TOTAL_DECALOGO_POINTS
                
                total_dimension_counts[dim_code] = {
                    "expected_total": expected_total,
                    "actual_total": actual_total,
                    "is_valid": actual_total == expected_total
                }
                
                if actual_total != expected_total:
                    validation_results["is_valid"] = False
                    validation_results["errors"].append(
                        f"Dimension {dim_code} total count mismatch: expected {expected_total}, got {actual_total}"
                    )
            
            validation_results["summary"] = {
                "total_questions": total_questions,
                "points_validated": len(point_counts),
                "dimension_combinations_validated": len(validation_results["dimension_validation"]),
                "error_count": len(validation_results["errors"]),
                "warning_count": len(validation_results["warnings"])
            }
            
        except Exception as e:
            validation_results["is_valid"] = False
            validation_results["errors"].append(f"Validation exception: {str(e)}")
        
        # Raise exception if validation fails
        if not validation_results["is_valid"]:
            error_summary = "; ".join(validation_results["errors"])
            raise DecalogoQuestionValidationError(f"Registry validation failed: {error_summary}")
        
        return validation_results
    
    def validate_coverage_completion(self, document_id: str, answered_questions: List[str]) -> Dict[str, Any]:
        """
        Validate coverage completion for a specific document.
        Tracks which questions have been answered vs expected with dimension-level breakdowns.
        """
        coverage_matrix = CoverageMatrix()
        coverage_matrix.document_id = document_id
        
        # Track answered questions
        answered_set = set(answered_questions)
        
        for question in self.questions:
            question_id = question.question_id
            if question_id in answered_set:
                coverage_matrix.add_question(
                    question.point_number, 
                    question.dimension_code.replace("DE", "DE-"), 
                    question_id
                )
        
        return coverage_matrix.to_dict()
    
    def generate_coverage_artifacts(self, document_id: str, answered_questions: List[str], 
                                  output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Generate coverage_P{n}.json artifacts in canonical_flow/classification/<doc>/ directory.
        Creates one file per Decálogo point with detailed coverage analysis.
        """
        if output_dir is None:
            output_dir = f"canonical_flow/classification/{document_id}"
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        coverage_files = {}
        answered_set = set(answered_questions)
        
        for point_num in range(1, self.TOTAL_DECALOGO_POINTS + 1):
            point_coverage = {
                "document_id": document_id,
                "point_number": point_num,
                "point_name": self._get_point_name(point_num),
                "total_expected_questions": self.REQUIRED_QUESTIONS_PER_POINT,
                "dimensions": {},
                "summary": {},
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            point_questions = self.get_questions_by_point(point_num)
            total_answered = 0
            
            # Process by dimension
            for dim_code, expected_count in self.REQUIRED_DIMENSION_DISTRIBUTION.items():
                dim_questions = [q for q in point_questions if q.dimension_code == dim_code]
                answered_questions_in_dim = [
                    q.question_id for q in dim_questions 
                    if q.question_id in answered_set
                ]
                actual_answered = len(answered_questions_in_dim)
                total_answered += actual_answered
                
                point_coverage["dimensions"][dim_code] = {
                    "expected_count": expected_count,
                    "actual_answered": actual_answered,
                    "completion_percentage": (actual_answered / expected_count * 100) if expected_count > 0 else 0,
                    "has_gap": actual_answered < expected_count,
                    "gap_size": expected_count - actual_answered,
                    "answered_questions": answered_questions_in_dim,
                    "unanswered_count": expected_count - actual_answered
                }
            
            # General questions (non-dimension specific)
            general_questions = [q for q in point_questions if q.dimension_code == "GEN"]
            answered_general = [
                q.question_id for q in general_questions 
                if q.question_id in answered_set
            ]
            actual_general = len(answered_general)
            total_answered += actual_general
            
            point_coverage["dimensions"]["GEN"] = {
                "expected_count": len(general_questions),
                "actual_answered": actual_general,
                "completion_percentage": (actual_general / len(general_questions) * 100) if len(general_questions) > 0 else 0,
                "has_gap": actual_general < len(general_questions),
                "gap_size": len(general_questions) - actual_general,
                "answered_questions": answered_general,
                "unanswered_count": len(general_questions) - actual_general
            }
            
            # Summary statistics
            point_coverage["summary"] = {
                "total_answered": total_answered,
                "completion_percentage": (total_answered / self.REQUIRED_QUESTIONS_PER_POINT * 100),
                "has_gaps": total_answered < self.REQUIRED_QUESTIONS_PER_POINT,
                "total_gaps": self.REQUIRED_QUESTIONS_PER_POINT - total_answered,
                "dimensions_with_gaps": sum(1 for dim_data in point_coverage["dimensions"].values() if dim_data["has_gap"]),
                "fully_completed_dimensions": sum(1 for dim_data in point_coverage["dimensions"].values() if not dim_data["has_gap"])
            }
            
            # Write coverage file
            coverage_file = f"{output_dir}/coverage_P{point_num}.json"
            with open(coverage_file, 'w', encoding='utf-8') as f:
                json.dump(point_coverage, f, indent=2, ensure_ascii=False)
            
            coverage_files[f"P{point_num}"] = coverage_file
        
        return coverage_files
    
    def _get_point_name(self, point_num: int) -> str:
        """Get the name of a Decálogo point"""
        point_names = {
            1: "Derecho a la vida, a la seguridad y a la convivencia",
            2: "Igualdad de la mujer y equidad de género", 
            3: "Derecho humano al agua, ambiente sano y gestión del riesgo",
            4: "Derecho humano a la salud",
            5: "Derechos de las víctimas y construcción de paz",
            6: "Derechos de la niñez, la juventud y fortalecimiento familiar",
            7: "Ordenamiento territorial, tierras y desarrollo rural",
            8: "Protección y promoción de líderes sociales",
            9: "Personas privadas de la libertad", 
            10: "Trabajo decente y seguridad social"
        }
        return point_names.get(point_num, f"Punto {point_num}")
    
    def _ensure_deterministic_ordering(self):
        """Ensure lexicographic ordering within each point-dimension combination"""
        # Sort questions by: point_number, dimension_code, question_text
        self.questions.sort(key=lambda q: (q.point_number, q.dimension_code, q.question_text))
    
    def get_all_questions(self) -> List[DecalogoQuestion]:
        """Get all 470 questions in deterministic order"""
        return self.questions.copy()
    
    def get_questions_by_point(self, point_number: int) -> List[DecalogoQuestion]:
        """Get all 47 questions for a specific point"""
        return [q for q in self.questions if q.point_number == point_number]
    
    def get_questions_by_dimension(self, dimension_code: str) -> List[DecalogoQuestion]:
        """Get all questions for a specific dimension"""
        return [q for q in self.questions if q.dimension_code == dimension_code]
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of the registry validation status"""
        try:
            validation = self.run_preflight_validation()
            return {
                "status": "valid",
                "total_questions": len(self.questions),
                "questions_per_point": self.REQUIRED_QUESTIONS_PER_POINT,
                "dimension_distribution": self.REQUIRED_DIMENSION_DISTRIBUTION,
                "validation_details": validation
            }
        except DecalogoQuestionValidationError as e:
            return {
                "status": "invalid",
                "error": str(e),
                "total_questions": len(self.questions)
            }


# Demo/Test functions
def demo_registry_validation():
    """Demo the registry validation functionality"""
    print("=== Decálogo Question Registry Validation Demo ===")
    
    try:
        registry = DecalogoQuestionRegistry()
        print(f"✓ Registry created successfully with {len(registry.questions)} questions")
        
        # Show validation summary
        summary = registry.get_validation_summary()
        print(f"✓ Validation Status: {summary['status']}")
        print(f"✓ Questions per point: {summary['questions_per_point']}")
        print(f"✓ Dimension distribution: {summary['dimension_distribution']}")
        
        # Test coverage tracking
        test_doc = "TEST_DOC_001"
        sample_answered = [registry.questions[i].question_id for i in range(100)]  # First 100 questions
        
        coverage = registry.validate_coverage_completion(test_doc, sample_answered)
        print(f"✓ Coverage validation completed: {coverage['summary']['completion_percentage']:.1f}% coverage")
        
        # Generate coverage artifacts
        coverage_files = registry.generate_coverage_artifacts(test_doc, sample_answered)
        print(f"✓ Generated {len(coverage_files)} coverage artifact files")
        
    except DecalogoQuestionValidationError as e:
        print(f"✗ Validation failed: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")


if __name__ == "__main__":
    demo_registry_validation()