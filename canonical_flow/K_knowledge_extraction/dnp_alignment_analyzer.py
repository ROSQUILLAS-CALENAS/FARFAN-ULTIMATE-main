"""
DNP Alignment Analyzer for Knowledge Extraction Stage

This component inherits from TotalOrderingBase and implements the standardized process() API
to consume all existing knowledge stage artifacts and compute compliance deviations
for each of the 10 Decálogo points against DNP baseline standards.
"""

import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple
from collections import OrderedDict
from uuid import uuid4

# Import total ordering base
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from total_ordering_base import TotalOrderingBase, DeterministicCollectionMixin

logger = logging.getLogger(__name__)


@dataclass
class DNPAlignmentEvidence:
    """Evidence structure for DNP alignment analysis"""
    text: str
    source_page: int
    source_document: str
    confidence: float
    evidence_type: str
    page_anchor: str = ""
    
    def __post_init__(self):
        if not self.page_anchor:
            self.page_anchor = f"p{self.source_page}"


@dataclass
class DeviationScore:
    """Compliance deviation score for a Decálogo point"""
    point_id: int
    point_name: str
    compliance_score: float  # 0.0 = non-compliant, 1.0 = fully compliant
    deviation_magnitude: float  # 0.0 = no deviation, 1.0 = maximum deviation
    baseline_score: float
    evidence_count: int
    confidence: float


@dataclass
class DNPBaselineValidation:
    """Validation results for DNP baseline standards"""
    has_baseline_data: bool
    baseline_version: str
    validation_status: str  # "valid", "outdated", "missing", "malformed"
    validation_issues: List[str] = field(default_factory=list)
    fallback_applied: bool = False


@dataclass
class ActionableRecommendation:
    """Actionable recommendation for improving compliance"""
    priority: str  # "high", "medium", "low"
    category: str
    description: str
    impact_estimate: float
    implementation_complexity: str  # "low", "medium", "high"
    resources_required: List[str] = field(default_factory=list)


class DNPAlignmentAnalyzer(TotalOrderingBase, DeterministicCollectionMixin):
    """
    DNP Alignment Analyzer with deterministic processing and total ordering.
    
    Consumes knowledge stage artifacts and computes compliance deviations
    for each Decálogo point against DNP baseline standards.
    """
    
    def __init__(self):
        super().__init__("DNPAlignmentAnalyzer")
        
        # Initialize Decálogo points configuration
        self.decalogo_points = self._initialize_decalogo_points()
        self.dnp_baselines = self._load_dnp_baselines()
        self.baseline_validation = None
        
        # State tracking
        self.processed_artifacts = {}
        self.alignment_results = {}
        
        # Update initial state hash
        self.update_state_hash(self._get_initial_state())
    
    def _get_initial_state(self) -> Dict[str, Any]:
        """Get initial state for hash calculation"""
        return {
            "decalogo_points_count": len(self.decalogo_points),
            "has_baselines": len(self.dnp_baselines) > 0,
            "component_version": "1.0.0"
        }
    
    def _initialize_decalogo_points(self) -> Dict[int, Dict[str, Any]]:
        """Initialize the 10 Decálogo points configuration"""
        return {
            1: {
                "name": "Derecho a la vida, a la seguridad y a la convivencia",
                "cluster": 5,
                "priority": "MAXIMA",
                "budget_min_percent": 3.0,
                "sector": "seguridad"
            },
            2: {
                "name": "Igualdad de la mujer y equidad de género",
                "cluster": 2,
                "priority": "ALTA", 
                "budget_min_percent": 2.0,
                "sector": "mujer_genero"
            },
            3: {
                "name": "Derecho humano al agua, ambiente sano y gestión del riesgo",
                "cluster": 1,
                "priority": "ALTA",
                "budget_min_percent": 7.0,
                "sector": "ambiente"
            },
            4: {
                "name": "Derecho humano a la salud",
                "cluster": 3,
                "priority": "MAXIMA",
                "budget_min_percent": 25.0,
                "sector": "salud"
            },
            5: {
                "name": "Derechos de las víctimas y construcción de paz",
                "cluster": 1,
                "priority": "ALTA",
                "budget_min_percent": 5.0,
                "sector": "victimas_paz"
            },
            6: {
                "name": "Derechos de las mujeres",
                "cluster": 3,
                "priority": "ALTA",
                "budget_min_percent": 3.0,
                "sector": "mujeres"
            },
            7: {
                "name": "Derechos de niñas, niños y adolescentes", 
                "cluster": 3,
                "priority": "MAXIMA",
                "budget_min_percent": 15.0,
                "sector": "ninez"
            },
            8: {
                "name": "Líderes y defensores de derechos humanos sociales y ambientales",
                "cluster": 1,
                "priority": "ALTA",
                "budget_min_percent": 2.0,
                "sector": "defensores"
            },
            9: {
                "name": "Derechos de los pueblos étnicos",
                "cluster": 4,
                "priority": "ALTA",
                "budget_min_percent": 8.0,
                "sector": "etnicos"
            },
            10: {
                "name": "Derecho a un ambiente sano",
                "cluster": 4,
                "priority": "ALTA",
                "budget_min_percent": 5.0,
                "sector": "ambiente_sano"
            }
        }
    
    def _load_dnp_baselines(self) -> Dict[str, Any]:
        """Load DNP baseline standards with error handling"""
        try:
            # Attempt to load DNP baselines from standard locations
            baseline_paths = [
                Path("data/dnp_baselines.json"),
                Path("standards_alignment/dnp_baselines.json"),
                Path("canonical_flow/knowledge/dnp_baselines.json")
            ]
            
            for path in baseline_paths:
                if path.exists():
                    with open(path, 'r', encoding='utf-8') as f:
                        baselines = json.load(f)
                        logger.info(f"Loaded DNP baselines from {path}")
                        return baselines
            
            # If no baseline files found, return empty dict
            logger.warning("No DNP baseline files found, using empty baselines")
            return {}
            
        except Exception as e:
            logger.error(f"Error loading DNP baselines: {e}")
            return {}
    
    def process(self, data: Any = None, context: Any = None) -> Dict[str, Any]:
        """
        Main processing function implementing standardized API.
        
        Args:
            data: Input knowledge artifacts (chunks, embeddings, entities, concepts, graphs)
            context: Processing context including document metadata
            
        Returns:
            Deterministic DNP alignment analysis results
        """
        operation_id = self.generate_operation_id("process", {"data": data, "context": context})
        
        try:
            # Validate baseline data first
            self.baseline_validation = self._validate_baseline_data()
            
            # Canonicalize inputs
            canonical_data = self.canonicalize_data(data) if data else {}
            canonical_context = self.canonicalize_data(context) if context else {}
            
            # Extract knowledge artifacts
            artifacts = self._extract_knowledge_artifacts(canonical_data, canonical_context)
            
            # Compute compliance deviations for each Decálogo point
            alignment_results = self._compute_alignment_deviations(artifacts)
            
            # Generate actionable recommendations
            recommendations = self._generate_recommendations(alignment_results)
            
            # Prepare final results
            results = {
                "operation_id": operation_id,
                "component_metadata": self.get_deterministic_metadata(),
                "baseline_validation": self.baseline_validation.__dict__ if self.baseline_validation else {},
                "alignment_results": alignment_results,
                "recommendations": recommendations,
                "processing_stats": self._get_processing_stats(artifacts),
                "timestamp": datetime.now().isoformat()
            }
            
            # Update state
            self.processed_artifacts = artifacts
            self.alignment_results = alignment_results
            self.update_state_hash(results)
            
            # Save to canonical_flow/knowledge/ directory
            self._save_alignment_artifacts(results, canonical_context)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in DNP alignment processing: {e}")
            error_result = {
                "operation_id": operation_id,
                "error": str(e),
                "component_metadata": self.get_deterministic_metadata(),
                "timestamp": datetime.now().isoformat()
            }
            return error_result
    
    def _validate_baseline_data(self) -> DNPBaselineValidation:
        """Validate DNP baseline data availability and quality"""
        validation_issues = []
        baseline_version = "unknown"
        validation_status = "valid"
        fallback_applied = False
        
        if not self.dnp_baselines:
            validation_issues.append("No DNP baseline data available")
            validation_status = "missing"
            fallback_applied = True
        else:
            # Check baseline data structure
            required_fields = ["version", "points", "standards"]
            missing_fields = [field for field in required_fields 
                            if field not in self.dnp_baselines]
            
            if missing_fields:
                validation_issues.append(f"Missing baseline fields: {missing_fields}")
                validation_status = "malformed"
                fallback_applied = True
            else:
                baseline_version = self.dnp_baselines.get("version", "unknown")
                
                # Check if baseline is outdated (simplified check)
                try:
                    version_parts = baseline_version.split(".")
                    major_version = int(version_parts[0]) if version_parts else 0
                    if major_version < 2:  # Arbitrary threshold
                        validation_issues.append(f"Baseline version {baseline_version} may be outdated")
                        validation_status = "outdated"
                except:
                    validation_issues.append("Cannot parse baseline version")
        
        return DNPBaselineValidation(
            has_baseline_data=len(self.dnp_baselines) > 0,
            baseline_version=baseline_version,
            validation_status=validation_status,
            validation_issues=validation_issues,
            fallback_applied=fallback_applied
        )
    
    def _extract_knowledge_artifacts(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and organize knowledge stage artifacts"""
        artifacts = {
            "chunks": [],
            "embeddings": [],
            "entities": [],
            "concepts": [],
            "knowledge_graph": {},
            "causal_graphs": [],
            "document_metadata": {}
        }
        
        # Extract chunks
        if "chunks" in data:
            artifacts["chunks"] = data["chunks"]
        elif "text_chunks" in data:
            artifacts["chunks"] = data["text_chunks"]
        
        # Extract embeddings
        if "embeddings" in data:
            artifacts["embeddings"] = data["embeddings"]
        elif "vectors" in data:
            artifacts["embeddings"] = data["vectors"]
        
        # Extract entities
        if "entities" in data:
            artifacts["entities"] = data["entities"]
        elif "named_entities" in data:
            artifacts["entities"] = data["named_entities"]
        
        # Extract concepts
        if "concepts" in data:
            artifacts["concepts"] = data["concepts"]
        
        # Extract knowledge graph
        if "knowledge_graph" in data:
            artifacts["knowledge_graph"] = data["knowledge_graph"]
        elif "graph" in data:
            artifacts["knowledge_graph"] = data["graph"]
        
        # Extract causal graphs
        if "causal_graphs" in data:
            artifacts["causal_graphs"] = data["causal_graphs"]
        elif "causal_analysis" in data:
            artifacts["causal_graphs"] = data["causal_analysis"]
        
        # Extract document metadata
        if "document_metadata" in context:
            artifacts["document_metadata"] = context["document_metadata"]
        elif "metadata" in context:
            artifacts["document_metadata"] = context["metadata"]
        
        return artifacts
    
    def _compute_alignment_deviations(self, artifacts: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """Compute compliance deviations for each Decálogo point"""
        alignment_results = {}
        
        for point_id, point_config in self.decalogo_points.items():
            try:
                # Extract relevant evidence for this point
                evidence_items = self._extract_point_evidence(point_id, point_config, artifacts)
                
                # Compute baseline score
                baseline_score = self._get_baseline_score(point_id)
                
                # Compute compliance score
                compliance_score = self._compute_compliance_score(point_id, evidence_items, artifacts)
                
                # Compute deviation magnitude
                deviation_magnitude = abs(compliance_score - baseline_score)
                
                # Create deviation score object
                deviation_score = DeviationScore(
                    point_id=point_id,
                    point_name=point_config["name"],
                    compliance_score=compliance_score,
                    deviation_magnitude=deviation_magnitude,
                    baseline_score=baseline_score,
                    evidence_count=len(evidence_items),
                    confidence=self._compute_confidence(evidence_items)
                )
                
                alignment_results[point_id] = {
                    "deviation_score": deviation_score.__dict__,
                    "evidence_references": [evidence.__dict__ for evidence in evidence_items],
                    "baseline_comparison": {
                        "expected": baseline_score,
                        "actual": compliance_score,
                        "deviation": deviation_magnitude,
                        "status": self._classify_deviation_status(deviation_magnitude)
                    }
                }
                
            except Exception as e:
                logger.error(f"Error computing alignment for point {point_id}: {e}")
                # Provide neutral fallback
                alignment_results[point_id] = self._create_fallback_result(point_id, point_config, str(e))
        
        return alignment_results
    
    def _extract_point_evidence(self, point_id: int, point_config: Dict[str, Any], 
                              artifacts: Dict[str, Any]) -> List[DNPAlignmentEvidence]:
        """Extract evidence relevant to a specific Decálogo point"""
        evidence_items = []
        
        # Define keywords for each point
        point_keywords = self._get_point_keywords(point_id, point_config)
        
        # Search in chunks
        for i, chunk in enumerate(artifacts.get("chunks", [])):
            if isinstance(chunk, dict):
                text = chunk.get("text", "")
                page = chunk.get("page", i + 1)
                source_doc = chunk.get("source", "unknown")
            elif isinstance(chunk, str):
                text = chunk
                page = i + 1
                source_doc = "unknown"
            else:
                continue
            
            # Check for keyword matches
            text_lower = text.lower()
            matches = [kw for kw in point_keywords if kw.lower() in text_lower]
            
            if matches:
                confidence = min(1.0, len(matches) / len(point_keywords))
                evidence = DNPAlignmentEvidence(
                    text=text[:500] + "..." if len(text) > 500 else text,  # Truncate long text
                    source_page=page,
                    source_document=source_doc,
                    confidence=confidence,
                    evidence_type="text_match",
                    page_anchor=f"p{page}_chunk{i}"
                )
                evidence_items.append(evidence)
        
        # Search in entities
        for entity in artifacts.get("entities", []):
            if isinstance(entity, dict):
                entity_text = entity.get("text", entity.get("name", ""))
                if any(kw.lower() in entity_text.lower() for kw in point_keywords):
                    evidence = DNPAlignmentEvidence(
                        text=entity_text,
                        source_page=entity.get("page", 0),
                        source_document=entity.get("source", "unknown"),
                        confidence=entity.get("confidence", 0.7),
                        evidence_type="entity"
                    )
                    evidence_items.append(evidence)
        
        return sorted(evidence_items, key=lambda x: x.confidence, reverse=True)
    
    def _get_point_keywords(self, point_id: int, point_config: Dict[str, Any]) -> List[str]:
        """Get relevant keywords for a Decálogo point"""
        keyword_map = {
            1: ["vida", "seguridad", "convivencia", "violencia", "conflicto", "protección"],
            2: ["mujer", "género", "igualdad", "equidad", "discriminación"],
            3: ["agua", "ambiente", "riesgo", "gestión", "ambiental", "recursos naturales"],
            4: ["salud", "médico", "hospital", "atención", "medicina", "enfermedad"],
            5: ["víctimas", "paz", "conflicto", "reparación", "justicia transicional"],
            6: ["mujeres", "género", "feminicidio", "violencia de género"],
            7: ["niños", "niñas", "adolescentes", "menores", "infancia", "educación"],
            8: ["defensores", "líderes", "derechos humanos", "activistas", "amenazas"],
            9: ["étnicos", "indígenas", "afrodescendientes", "diversidad", "territorial"],
            10: ["ambiente sano", "contaminación", "biodiversidad", "sostenibilidad"]
        }
        
        return keyword_map.get(point_id, [point_config.get("sector", "").replace("_", " ")])
    
    def _get_baseline_score(self, point_id: int) -> float:
        """Get baseline score for a Decálogo point"""
        if self.baseline_validation and self.baseline_validation.fallback_applied:
            # Neutral fallback score
            return 0.5
        
        if "points" in self.dnp_baselines and str(point_id) in self.dnp_baselines["points"]:
            point_baseline = self.dnp_baselines["points"][str(point_id)]
            return point_baseline.get("baseline_score", 0.5)
        
        # Default fallback based on priority
        point_config = self.decalogo_points[point_id]
        priority_scores = {"MAXIMA": 0.8, "ALTA": 0.6, "MEDIA": 0.4}
        return priority_scores.get(point_config.get("priority", "MEDIA"), 0.5)
    
    def _compute_compliance_score(self, point_id: int, evidence_items: List[DNPAlignmentEvidence], 
                                artifacts: Dict[str, Any]) -> float:
        """Compute compliance score based on evidence"""
        if not evidence_items:
            return 0.0
        
        # Weight evidence by confidence and type
        total_weighted_score = 0.0
        total_weight = 0.0
        
        type_weights = {
            "text_match": 0.6,
            "entity": 0.8,
            "statistical": 1.0,
            "expert_opinion": 0.9
        }
        
        for evidence in evidence_items:
            type_weight = type_weights.get(evidence.evidence_type, 0.5)
            weight = evidence.confidence * type_weight
            total_weighted_score += evidence.confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        base_score = total_weighted_score / total_weight
        
        # Boost score based on evidence count
        evidence_boost = min(0.2, len(evidence_items) * 0.02)
        
        return min(1.0, base_score + evidence_boost)
    
    def _compute_confidence(self, evidence_items: List[DNPAlignmentEvidence]) -> float:
        """Compute confidence in the alignment assessment"""
        if not evidence_items:
            return 0.0
        
        avg_confidence = sum(e.confidence for e in evidence_items) / len(evidence_items)
        count_factor = min(1.0, len(evidence_items) / 5)  # Max confidence with 5+ evidence items
        
        return avg_confidence * count_factor
    
    def _classify_deviation_status(self, deviation: float) -> str:
        """Classify deviation magnitude into status categories"""
        if deviation < 0.1:
            return "minimal"
        elif deviation < 0.3:
            return "moderate"
        elif deviation < 0.5:
            return "significant"
        else:
            return "critical"
    
    def _create_fallback_result(self, point_id: int, point_config: Dict[str, Any], error: str) -> Dict[str, Any]:
        """Create fallback result when processing fails"""
        baseline_score = self._get_baseline_score(point_id)
        
        return {
            "deviation_score": {
                "point_id": point_id,
                "point_name": point_config["name"],
                "compliance_score": 0.5,  # Neutral fallback
                "deviation_magnitude": abs(0.5 - baseline_score),
                "baseline_score": baseline_score,
                "evidence_count": 0,
                "confidence": 0.0
            },
            "evidence_references": [],
            "baseline_comparison": {
                "expected": baseline_score,
                "actual": 0.5,
                "deviation": abs(0.5 - baseline_score),
                "status": "unknown"
            },
            "error": error,
            "fallback_applied": True
        }
    
    def _generate_recommendations(self, alignment_results: Dict[int, Dict[str, Any]]) -> List[ActionableRecommendation]:
        """Generate actionable recommendations based on alignment results"""
        recommendations = []
        
        for point_id, result in alignment_results.items():
            deviation_score = result["deviation_score"]
            deviation_magnitude = deviation_score["deviation_magnitude"]
            
            if deviation_magnitude > 0.3:  # Significant deviation threshold
                point_config = self.decalogo_points[point_id]
                priority = "high" if point_config.get("priority") == "MAXIMA" else "medium"
                
                recommendation = ActionableRecommendation(
                    priority=priority,
                    category=f"decalogo_point_{point_id}",
                    description=self._generate_recommendation_text(point_id, deviation_score, point_config),
                    impact_estimate=min(1.0, deviation_magnitude * 1.2),
                    implementation_complexity=self._assess_implementation_complexity(point_id, point_config),
                    resources_required=self._identify_required_resources(point_id, point_config)
                )
                
                recommendations.append(recommendation)
        
        # Sort by priority and impact
        priority_order = {"high": 3, "medium": 2, "low": 1}
        recommendations.sort(
            key=lambda r: (priority_order.get(r.priority, 0), r.impact_estimate), 
            reverse=True
        )
        
        return [rec.__dict__ for rec in recommendations]
    
    def _generate_recommendation_text(self, point_id: int, deviation_score: Dict[str, Any], 
                                    point_config: Dict[str, Any]) -> str:
        """Generate specific recommendation text"""
        compliance_score = deviation_score["compliance_score"]
        point_name = point_config["name"]
        
        if compliance_score < 0.3:
            return f"Urgente: Implementar medidas inmediatas para mejorar {point_name}. " \
                   f"El nivel de cumplimiento actual ({compliance_score:.2f}) está muy por debajo del estándar requerido."
        elif compliance_score < 0.6:
            return f"Mejorar: Fortalecer las acciones relacionadas con {point_name}. " \
                   f"Se requieren mejoras significativas para alcanzar el nivel de cumplimiento esperado."
        else:
            return f"Optimizar: Ajustar estrategias para {point_name} y mantener el nivel de cumplimiento actual."
    
    def _assess_implementation_complexity(self, point_id: int, point_config: Dict[str, Any]) -> str:
        """Assess implementation complexity for a recommendation"""
        sector = point_config.get("sector", "")
        priority = point_config.get("priority", "MEDIA")
        budget_percent = point_config.get("budget_min_percent", 0)
        
        if budget_percent > 15 or priority == "MAXIMA":
            return "high"
        elif budget_percent > 5 or priority == "ALTA":
            return "medium" 
        else:
            return "low"
    
    def _identify_required_resources(self, point_id: int, point_config: Dict[str, Any]) -> List[str]:
        """Identify required resources for implementation"""
        sector = point_config.get("sector", "")
        
        resource_map = {
            "seguridad": ["personal_seguridad", "equipamiento", "coordinacion_interinstitucional"],
            "salud": ["infraestructura_salud", "personal_medico", "equipos_medicos", "medicamentos"],
            "educacion": ["infraestructura_educativa", "docentes", "material_pedagogico"],
            "ambiente": ["estudios_ambientales", "personal_tecnico", "equipos_monitoreo"],
            "mujeres": ["programas_especializados", "personal_capacitado", "refugios"],
            "ninez": ["programas_infancia", "personal_especializado", "espacios_protegidos"]
        }
        
        return resource_map.get(sector, ["recursos_generales", "personal_capacitado", "coordinacion"])
    
    def _get_processing_stats(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "total_chunks": len(artifacts.get("chunks", [])),
            "total_entities": len(artifacts.get("entities", [])),
            "total_concepts": len(artifacts.get("concepts", [])),
            "has_knowledge_graph": len(artifacts.get("knowledge_graph", {})) > 0,
            "has_causal_graphs": len(artifacts.get("causal_graphs", [])) > 0,
            "processed_points": len(self.decalogo_points)
        }
    
    def _save_alignment_artifacts(self, results: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Save alignment results to canonical_flow/knowledge/ directory"""
        try:
            # Extract document stem from context
            doc_stem = "unknown_document"
            if "document_metadata" in context:
                metadata = context["document_metadata"]
                if "file_stem" in metadata:
                    doc_stem = metadata["file_stem"]
                elif "filename" in metadata:
                    filename = metadata["filename"]
                    doc_stem = Path(filename).stem if filename else doc_stem
            
            # Create output directory
            output_dir = Path("canonical_flow/knowledge")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate output filename
            output_file = output_dir / f"{doc_stem}_dnp_alignment.json"
            
            # Save results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved DNP alignment results to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving alignment artifacts: {e}")


# Convenience function for standalone usage
def process(data=None, context=None):
    """Standalone processing function"""
    analyzer = DNPAlignmentAnalyzer()
    return analyzer.process(data, context)


if __name__ == "__main__":
    # Demo usage
    demo_data = {
        "chunks": [
            {"text": "El proyecto incluye medidas de seguridad y protección para la población vulnerable", "page": 1},
            {"text": "Se establecerán programas de salud comunitaria", "page": 2},
            {"text": "El componente de educación contempla mejoras en infraestructura escolar", "page": 3}
        ],
        "entities": [
            {"text": "seguridad ciudadana", "confidence": 0.8, "page": 1},
            {"text": "salud comunitaria", "confidence": 0.9, "page": 2}
        ]
    }
    
    demo_context = {
        "document_metadata": {"file_stem": "demo_project"}
    }
    
    results = process(demo_data, demo_context)
    print(json.dumps(results, indent=2))