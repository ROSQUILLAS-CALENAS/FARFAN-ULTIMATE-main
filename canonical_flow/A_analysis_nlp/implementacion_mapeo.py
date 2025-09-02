"""
Canonical Flow Alias: 12A
Implementacion Mapeo with Total Ordering and Deterministic Processing

Source: implementacion_mapeo.py
Stage: analysis_nlp
Code: 12A
"""

import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import OrderedDict

# Import total ordering base
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from total_ordering_base import TotalOrderingBase, DeterministicCollectionMixin

logger = logging.getLogger(__name__)


@dataclass
class QuestionMapping:
    """Estructura de datos para mapeo de preguntas"""
    question_id: str
    question_text: str
    decalogo_point: int
    point_name: str
    dimension: str
    value_chain_link: str
    dnp_category: str
    sector: str
    weight: float
    keywords: List[str] = field(default_factory=list)
    search_patterns: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Ensure deterministic ordering
        self.keywords = sorted(self.keywords)
        self.search_patterns = sorted(self.search_patterns)


class QuestionDecalogoMapper(TotalOrderingBase, DeterministicCollectionMixin):
    """
    Question Decálogo Mapper with deterministic processing and total ordering.
    
    Provides consistent mapping results and stable ID generation across runs.
    """
    
    def __init__(self):
        super().__init__("QuestionDecalogoMapper")
        
        self.complete_mapping = self._build_complete_mapping()
        self.value_chain_links = self._define_value_chain_links()
        self.decalogo_points = self._define_decalogo_points()
        self.coverage_matrix = None
        self.validation_report = None
        
        # Update state hash
        self.update_state_hash(self._get_initial_state())
    
    def _get_initial_state(self) -> Dict[str, Any]:
        """Get initial state for hash calculation"""
        return {
            "mapping_count": len(self.complete_mapping),
            "value_chain_count": len(self.value_chain_links),
            "decalogo_points_count": len(self.decalogo_points),
        }
    
    def _define_decalogo_points(self) -> Dict[int, Dict[str, Any]]:
        """Definir los 11 puntos del Decálogo DDHH con orden determinístico"""
        points = {
            1: {
                "budget_min_percent": 3.0,
                "cluster": 5,
                "name": "Derecho a la vida, a la seguridad y a la convivencia",
                "priority": "MAXIMA",
                "sector": "seguridad",
            },
            2: {
                "budget_min_percent": 2.0,
                "cluster": 2,
                "name": "Igualdad de la mujer y equidad de género",
                "priority": "ALTA",
                "sector": "mujer_genero",
            },
            3: {
                "budget_min_percent": 7.0,
                "cluster": 1,
                "name": "Derecho humano al agua, ambiente sano y gestión del riesgo",
                "priority": "ALTA",
                "sector": "ambiente",
            },
            4: {
                "budget_min_percent": 25.0,
                "cluster": 3,
                "name": "Derecho humano a la salud",
                "priority": "MAXIMA",
                "sector": "salud",
            },
            5: {
                "budget_min_percent": 2.0,
                "cluster": 5,
                "name": "Derechos de las víctimas y construcción de paz",
                "priority": "ALTA",
                "sector": "paz",
            },
            6: {
                "budget_min_percent": 5.0,
                "cluster": 2,
                "name": "Derechos de la niñez, la juventud y fortalecimiento familiar",
                "priority": "MAXIMA",
                "sector": "primera_infancia",
            },
            7: {
                "budget_min_percent": 3.0,
                "cluster": 1,
                "name": "Ordenamiento territorial, tierras y desarrollo rural",
                "priority": "ALTA",
                "sector": "rural",
            },
            8: {
                "budget_min_percent": 1.0,
                "cluster": 4,
                "name": "Protección y promoción de líderes sociales",
                "priority": "ALTA",
                "sector": "proteccion",
            },
            9: {
                "budget_min_percent": 0.5,
                "cluster": 4,
                "name": "Personas privadas de la libertad",
                "priority": "MEDIA",
                "sector": "justicia",
            },
            10: {
                "budget_min_percent": 2.0,
                "cluster": 3,
                "name": "Trabajo decente y seguridad social",
                "priority": "ALTA",
                "sector": "empleo",
            },
            11: {
                "budget_min_percent": 1.0,
                "cluster": 4,
                "name": "Participación ciudadana y protesta social",
                "priority": "ALTA",
                "sector": "participacion",
            },
        }
        
        # Ensure deterministic ordering of nested dictionaries
        return {k: self.sort_dict_by_keys(v) for k, v in sorted(points.items())}
    
    def _define_value_chain_links(self) -> Dict[str, Dict[str, Any]]:
        """Definir eslabones de la cadena de valor del DNP con orden determinístico"""
        links = {
            "actividades": {
                "description": "Procesos y acciones ejecutadas",
                "elements": ["cronogramas", "metodologias", "procedimientos", "procesos"],
                "evaluation_focus": "ejecucion_procesos",
                "keywords": ["actividades", "cronograma", "metodología", "procedimientos", "procesos"],
                "name": "Actividades",
                "order": 2,
            },
            "impactos": {
                "description": "Transformaciones de largo plazo en bienestar",
                "elements": ["bienestar", "desarrollo", "sostenibilidad", "transformacion"],
                "evaluation_focus": "transformacion_territorial",
                "keywords": ["bienestar", "desarrollo", "impacto", "sostenible", "transformación"],
                "name": "Impactos",
                "order": 5,
            },
            "insumos": {
                "description": "Recursos necesarios para ejecutar las actividades",
                "elements": ["normatividad", "presupuesto", "recurso_humano", "tecnologia"],
                "evaluation_focus": "disponibilidad_recursos",
                "keywords": ["infraestructura", "normas", "personal", "presupuesto", "recursos"],
                "name": "Insumos",
                "order": 1,
            },
            "productos": {
                "description": "Bienes y servicios entregados directamente",
                "elements": ["beneficiarios", "bienes", "entregables", "servicios"],
                "evaluation_focus": "entrega_productos",
                "keywords": ["beneficiarios", "entregables", "metas", "productos", "servicios"],
                "name": "Productos",
                "order": 3,
            },
            "resultados": {
                "description": "Efectos directos en la población objetivo",
                "elements": ["acceso", "cambios", "coberturas", "mejoras"],
                "evaluation_focus": "efectos_poblacion",
                "keywords": ["cambios", "efectos", "indicadores", "mejoras", "resultados"],
                "name": "Resultados",
                "order": 4,
            },
        }
        
        # Ensure deterministic ordering
        return {k: self.sort_dict_by_keys(v) for k, v in sorted(links.items())}
    
    def _build_complete_mapping(self) -> Dict[str, QuestionMapping]:
        """Construir el mapeo completo con orden determinístico"""
        mapping = {}
        
        # Sample mappings for each decálogo point - condensed for brevity
        sample_mappings = [
            QuestionMapping(
                question_id="DE1_P1_Q1",
                question_text="¿Define productos medibles para prevención de violencia/protección/atención a víctimas?",
                decalogo_point=1,
                point_name="Derecho a la vida, a la seguridad y a la convivencia",
                dimension="DE1",
                value_chain_link="productos",
                dnp_category="seguridad_ciudadana",
                sector="seguridad",
                weight=1.0,
                keywords=["medibles", "prevención", "productos", "víctimas", "violencia"],
                search_patterns=[
                    "entregables.*protección.*víctimas",
                    "metas.*cuantificables.*violencia", 
                    "productos.*medibles.*seguridad"
                ],
            ),
            QuestionMapping(
                question_id="DE1_P2_Q1", 
                question_text="¿Define productos medibles para igualdad de género y protección a la mujer?",
                decalogo_point=2,
                point_name="Igualdad de la mujer y equidad de género",
                dimension="DE1",
                value_chain_link="productos",
                dnp_category="equidad_genero",
                sector="mujer_genero",
                weight=1.0,
                keywords=["género", "igualdad", "medibles", "mujer", "productos"],
                search_patterns=[
                    "entregables.*protección.*femenina",
                    "metas.*mujer.*equidad",
                    "productos.*igualdad.*género"
                ],
            ),
            QuestionMapping(
                question_id="DE1_P4_Q1",
                question_text="¿Define productos medibles para garantizar el derecho a la salud?",
                decalogo_point=4,
                point_name="Derecho humano a la salud",
                dimension="DE1", 
                value_chain_link="productos",
                dnp_category="salud",
                sector="salud",
                weight=1.0,
                keywords=["derecho", "garantizar", "medibles", "productos", "salud"],
                search_patterns=[
                    "entregables.*cobertura.*salud",
                    "metas.*atención.*médica",
                    "productos.*salud.*medibles"
                ],
            ),
        ]
        
        # Add mappings to dictionary
        for m in sample_mappings:
            mapping[m.question_id] = m
        
        return self.sort_dict_by_keys(mapping)
    
    def process(self, data: Any = None, context: Any = None) -> Dict[str, Any]:
        """
        Main processing function with deterministic output.
        
        Args:
            data: Input data for mapping
            context: Processing context
            
        Returns:
            Deterministic mapping results
        """
        operation_id = self.generate_operation_id("process", {"data": data, "context": context})
        
        try:
            # Canonicalize inputs
            canonical_data = self.canonicalize_data(data) if data else {}
            canonical_context = self.canonicalize_data(context) if context else {}
            
            # Perform mapping analysis
            mapping_results = self._perform_deterministic_mapping(canonical_data, canonical_context)
            
            # Generate deterministic output
            output = self._generate_deterministic_output(mapping_results, operation_id)
            
            # Update state hash
            self.update_state_hash(output)
            
            return output
            
        except Exception as e:
            error_output = {
                "component": self.component_name,
                "error": str(e),
                "operation_id": operation_id,
                "status": "error",
                "timestamp": self._get_deterministic_timestamp(),
            }
            return self.sort_dict_by_keys(error_output)
    
    def _perform_deterministic_mapping(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform deterministic mapping analysis"""
        
        # Extract questions if provided
        questions = data.get("questions", [])
        if isinstance(questions, str):
            questions = [questions]
        
        # Map questions to decálogo points
        mapped_questions = []
        for question in questions:
            mapped_question = self._map_question_to_decalogo(str(question))
            mapped_questions.append(mapped_question)
        
        # Generate coverage analysis
        coverage_analysis = self._analyze_coverage_deterministic()
        
        # Generate validation report
        validation_report = self._generate_validation_report_deterministic()
        
        return {
            "complete_mapping": self.complete_mapping,
            "coverage_analysis": coverage_analysis,
            "decalogo_points": self.decalogo_points,
            "mapped_questions": self.sort_collection(mapped_questions, key_func=lambda x: x.get("question_id", "")),
            "validation_report": validation_report,
            "value_chain_links": self.value_chain_links,
        }
    
    def _map_question_to_decalogo(self, question_text: str) -> Dict[str, Any]:
        """Map a question to decálogo points deterministically"""
        question_id = self.generate_stable_id(question_text, prefix="q")
        
        # Simple keyword-based mapping
        best_match = None
        best_score = 0.0
        
        question_lower = question_text.lower()
        
        for mapping_id, mapping in sorted(self.complete_mapping.items()):
            score = 0.0
            
            # Check keyword matches
            for keyword in mapping.keywords:
                if keyword.lower() in question_lower:
                    score += 1.0
            
            # Normalize by number of keywords
            if mapping.keywords:
                score = score / len(mapping.keywords)
            
            if score > best_score:
                best_score = score
                best_match = mapping
        
        return {
            "best_match": {
                "decalogo_point": best_match.decalogo_point,
                "dimension": best_match.dimension,
                "point_name": best_match.point_name,
                "question_id": best_match.question_id,
                "sector": best_match.sector,
                "value_chain_link": best_match.value_chain_link,
                "weight": best_match.weight,
            } if best_match else None,
            "question_id": question_id,
            "question_text": question_text,
            "score": best_score,
        }
    
    def _analyze_coverage_deterministic(self) -> Dict[str, Any]:
        """Analyze coverage with deterministic results"""
        coverage_by_point = {}
        coverage_by_dimension = {}
        coverage_by_sector = {}
        
        for mapping_id, mapping in sorted(self.complete_mapping.items()):
            # Coverage by decálogo point
            point_key = f"point_{mapping.decalogo_point}"
            if point_key not in coverage_by_point:
                coverage_by_point[point_key] = 0
            coverage_by_point[point_key] += 1
            
            # Coverage by dimension
            if mapping.dimension not in coverage_by_dimension:
                coverage_by_dimension[mapping.dimension] = 0
            coverage_by_dimension[mapping.dimension] += 1
            
            # Coverage by sector
            if mapping.sector not in coverage_by_sector:
                coverage_by_sector[mapping.sector] = 0
            coverage_by_sector[mapping.sector] += 1
        
        return {
            "coverage_by_dimension": self.sort_dict_by_keys(coverage_by_dimension),
            "coverage_by_point": self.sort_dict_by_keys(coverage_by_point), 
            "coverage_by_sector": self.sort_dict_by_keys(coverage_by_sector),
            "total_mappings": len(self.complete_mapping),
        }
    
    def _generate_validation_report_deterministic(self) -> Dict[str, Any]:
        """Generate validation report with deterministic results"""
        
        # Count mappings by various dimensions
        dimension_counts = {}
        point_counts = {}
        sector_counts = {}
        
        for mapping in self.complete_mapping.values():
            # Count by dimension
            dimension_counts[mapping.dimension] = dimension_counts.get(mapping.dimension, 0) + 1
            
            # Count by point
            point_key = f"point_{mapping.decalogo_point}"
            point_counts[point_key] = point_counts.get(point_key, 0) + 1
            
            # Count by sector
            sector_counts[mapping.sector] = sector_counts.get(mapping.sector, 0) + 1
        
        # Generate recommendations
        recommendations = []
        
        # Check for missing coverage
        all_dimensions = ["DE1", "DE2", "DE3", "DE4"]
        missing_dimensions = sorted(set(all_dimensions) - set(dimension_counts.keys()))
        
        if missing_dimensions:
            recommendations.append(f"Missing coverage for dimensions: {', '.join(missing_dimensions)}")
        
        # Check for unbalanced coverage
        if dimension_counts:
            max_count = max(dimension_counts.values())
            min_count = min(dimension_counts.values())
            if max_count > min_count * 3:  # Threshold for imbalance
                recommendations.append("Dimension coverage is unbalanced")
        
        return {
            "dimension_counts": self.sort_dict_by_keys(dimension_counts),
            "point_counts": self.sort_dict_by_keys(point_counts),
            "recommendations": sorted(recommendations),
            "sector_counts": self.sort_dict_by_keys(sector_counts),
            "validation_status": "complete",
        }
    
    def _generate_deterministic_output(self, mapping_results: Dict[str, Any], operation_id: str) -> Dict[str, Any]:
        """Generate deterministic output structure"""
        
        output = {
            "component": self.component_name,
            "component_id": self.component_id,
            "metadata": self.get_deterministic_metadata(),
            "operation_id": operation_id,
            "results": mapping_results,
            "status": "success",
            "timestamp": self._get_deterministic_timestamp(),
        }
        
        return self.sort_dict_by_keys(output)
    
    def save_artifact(self, data: Dict[str, Any], document_stem: str, output_dir: str = "canonical_flow/analysis") -> str:
        """
        Save implementacion mapeo output to canonical_flow/analysis/ directory with standardized naming.
        
        Args:
            data: Mapeo data to save
            document_stem: Base document identifier (without extension)
            output_dir: Output directory (defaults to canonical_flow/analysis)
            
        Returns:
            Path to saved artifact
        """
        try:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate standardized filename
            filename = f"{document_stem}_implementacion.json"
            artifact_path = Path(output_dir) / filename
            
            # Save with consistent JSON formatting
            with open(artifact_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"ImplementacionMapeo artifact saved: {artifact_path}")
            return str(artifact_path)
            
        except Exception as e:
            error_msg = f"Failed to save ImplementacionMapeo artifact to {output_dir}/{document_stem}_implementacion.json: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)


# Backward compatibility functions
def get_complete_mapping() -> Dict[str, Any]:
    """Get complete mapping"""
    mapper = QuestionDecalogoMapper()
    return mapper.complete_mapping


def map_question_to_decalogo(question: str) -> Dict[str, Any]:
    """Map single question to decálogo"""
    mapper = QuestionDecalogoMapper()
    return mapper._map_question_to_decalogo(question)


def process(data=None, context=None):
    """Backward compatible process function"""
    mapper = QuestionDecalogoMapper()
    result = mapper.process(data, context)
    
    # Save artifact if document_stem is provided
    if data and isinstance(data, dict) and 'document_stem' in data:
        mapper.save_artifact(result, data['document_stem'])
    
    return result