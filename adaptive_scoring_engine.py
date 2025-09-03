"""
Motor de Puntuación Adaptativa (AdaptiveScoringEngine)
Implementa modelado predictivo basado en datos históricos y contexto municipal
con integración de DNP alignment validation para estándares de derechos humanos.
"""

# # # from typing import Any, Dict, List, Optional  # Module not found  # Module not found  # Module not found

import json
import logging
import pickle
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Dict, List, Optional, Tuple  # Module not found  # Module not found  # Module not found

import numpy as np
import pandas as pd

try:
# # #     from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # Module not found  # Module not found  # Module not found
# # #     from sklearn.feature_selection import SelectKBest, f_regression  # Module not found  # Module not found  # Module not found
# # #     from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Module not found  # Module not found  # Module not found
# # #     from sklearn.model_selection import cross_val_score, train_test_split  # Module not found  # Module not found  # Module not found
# # #     from sklearn.preprocessing import RobustScaler, StandardScaler  # Module not found  # Module not found  # Module not found

    SKLEARN_AVAILABLE = True
except ImportError:
    logging.warning(
        "scikit-learn not available. AdaptiveScoringEngine will use fallback implementations."
    )
    SKLEARN_AVAILABLE = False

try:
# # #     from causal_dnp_framework import CausalDNPAnalyzer, CausalGraph, Evidence, CausalFactor  # Module not found  # Module not found  # Module not found
    CAUSAL_FRAMEWORK_AVAILABLE = True
except ImportError:
    logging.warning(
        "Causal DNP framework not available. Causal correction will use fallback implementations."
    )
    CAUSAL_FRAMEWORK_AVAILABLE = False

# Try to import centralized configuration
try:
# # #     from config_loader import get_thresholds  # Module not found  # Module not found  # Module not found
    THRESHOLDS_AVAILABLE = True
except ImportError:
    THRESHOLDS_AVAILABLE = False
    logging.warning("Centralized thresholds not available, using hardcoded values")

try:
# # #     from models import (  # Module not found  # Module not found  # Module not found
        AdaptiveScoringResults,
        ComplianceStatus,
        DecalogoPointScore,
        DimensionScore,
        DocumentPackage,
        PDTContext,
    )
except ImportError:
    try:
# # #         from .models import (  # Module not found  # Module not found  # Module not found
            AdaptiveScoringResults,
            ComplianceStatus,
            DecalogoPointScore,
            DimensionScore,
            DocumentPackage,
            PDTContext,
        )
    except ImportError:
        logging.error("Could not import required models. Please ensure models.py is available.")
        raise
    
    @dataclass
    class DimensionScore:
        dimension_id: str = ""


logger = logging.getLogger(__name__)


class AdaptiveScoringEngine:
    """
    Motor de Puntuación Adaptativa que predice puntuaciones basándose en:
    - Contexto municipal granular (PDTContext)
    - Datos históricos de evaluaciones
    - Características documentales extraídas
    """

    def __init__(self, models_path: str = "models/adaptive_scoring"):
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)

        # Inicializar modelos predictivos
        self.dimension_models = {}
        self.decalogo_models = {}
        self.global_model = None

        # Preprocesadores
        self.feature_scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_selector = None

# # #         # Load configuration from centralized thresholds or use defaults  # Module not found  # Module not found  # Module not found
        self._load_configuration()

        # Métricas de importancia
        self.feature_importance = {}
        self.model_performance = {}

        # Initialize DNP causal correction system
        self.dnp_analyzer = None
        self.causal_correction_enabled = CAUSAL_FRAMEWORK_AVAILABLE
        
        if self.causal_correction_enabled:
            try:
                self.dnp_analyzer = CausalDNPAnalyzer()
                logger.info("DNP causal correction system initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize DNP analyzer: {e}")
                self.causal_correction_enabled = False

        self.load_models()
        
        # DNP alignment validation mechanism
        try:
# # #             from dnp_alignment_adapter import DNPAlignmentValidator  # Module not found  # Module not found  # Module not found
            self.dnp_validator = DNPAlignmentValidator()
        except ImportError:
            self.dnp_validator = None
            logger.warning("DNP alignment validator not available")
        
        # Human rights standards cache
        self.human_rights_standards = self._load_human_rights_standards()
        
        # Math Stage5 Enhanced Scoring Integration
        try:
# # #             from canonical_flow.mathematical_enhancers.scoring_enhancer import create_math_stage5_enhancer  # Module not found  # Module not found  # Module not found
            self.math_stage5_enhancer = create_math_stage5_enhancer(entropy_reg=0.1)
            self.has_stage5_enhancement = True
            logger.info("Math Stage5 scoring enhancer loaded successfully")
        except ImportError as e:
            logger.warning(f"Math Stage5 enhancer not available: {e}")
            self.math_stage5_enhancer = None
            self.has_stage5_enhancement = False

    def _load_configuration(self):
# # #         """Load configuration from centralized thresholds or use defaults."""  # Module not found  # Module not found  # Module not found
        if THRESHOLDS_AVAILABLE:
            try:
                config = get_thresholds()
                adaptive_config = config.adaptive_scoring
                decalogo_config = config.decalogo_scoring
                
                # Load model parameters
                self.model_config = adaptive_config.model_parameters.copy()
                
                # Load compliance thresholds
                self.compliance_thresholds = {
                    "DE1": decalogo_config.compliance_thresholds.get("DE1", {"CUMPLE": 0.75, "CUMPLE_PARCIAL": 0.50}),
                    "DE2": decalogo_config.compliance_thresholds.get("DE2", {"CUMPLE": 0.70, "CUMPLE_PARCIAL": 0.45}),
                    "DE3": decalogo_config.compliance_thresholds.get("DE3", {"CUMPLE": 0.80, "CUMPLE_PARCIAL": 0.55}),
                    "DE4": decalogo_config.compliance_thresholds.get("DE4", {"CUMPLE": 0.65, "CUMPLE_PARCIAL": 0.40}),
                    "DECALOGO": decalogo_config.decalogo_point_thresholds
                }
                
                # Load hierarchical weights
                self.dimension_weights = decalogo_config.dimension_weights.copy()
                self.decalogo_weights = decalogo_config.decalogo_point_weights.copy()
                
                # Load correction thresholds and DNP baseline standards
                self.correction_thresholds = adaptive_config.correction_thresholds.copy()
                self.dnp_baseline_standards = adaptive_config.dnp_baseline_standards.copy()
                
# # #                 logger.info("Loaded adaptive scoring configuration from centralized thresholds")  # Module not found  # Module not found  # Module not found
                
            except Exception as e:
                logger.warning(f"Failed to load centralized configuration: {e}, using defaults")
                self._load_default_configuration()
        else:
            self._load_default_configuration()
    
    def _load_default_configuration(self):
        """Load default hardcoded configuration."""
        # Configuración de modelos
        self.model_config = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
        }
        
        # Umbrales de satisfacibilidad
        self.compliance_thresholds = {
            "DE1": {"CUMPLE": 0.75, "CUMPLE_PARCIAL": 0.50},
            "DE2": {"CUMPLE": 0.70, "CUMPLE_PARCIAL": 0.45},
            "DE3": {"CUMPLE": 0.80, "CUMPLE_PARCIAL": 0.55},
            "DE4": {"CUMPLE": 0.65, "CUMPLE_PARCIAL": 0.40},
            "DECALOGO": {
                "P1": {"CUMPLE": 0.75, "CUMPLE_PARCIAL": 0.50},
                "P2": {"CUMPLE": 0.70, "CUMPLE_PARCIAL": 0.45},
                "P3": {"CUMPLE": 0.72, "CUMPLE_PARCIAL": 0.47},
                "P4": {"CUMPLE": 0.68, "CUMPLE_PARCIAL": 0.43},
                "P5": {"CUMPLE": 0.73, "CUMPLE_PARCIAL": 0.48},
                "P6": {"CUMPLE": 0.71, "CUMPLE_PARCIAL": 0.46},
                "P7": {"CUMPLE": 0.74, "CUMPLE_PARCIAL": 0.49},
                "P8": {"CUMPLE": 0.69, "CUMPLE_PARCIAL": 0.44},
                "P9": {"CUMPLE": 0.76, "CUMPLE_PARCIAL": 0.51},
                "P10": {"CUMPLE": 0.67, "CUMPLE_PARCIAL": 0.42},
            },
        }

        # Ponderaciones jerárquicas
        self.dimension_weights = {"DE1": 0.30, "DE2": 0.25, "DE3": 0.25, "DE4": 0.20}
        self.decalogo_weights = {
            "P1": 0.12,
            "P2": 0.11,
            "P3": 0.10,
            "P4": 0.09,
            "P5": 0.08,
            "P6": 0.10,
            "P7": 0.09,
            "P8": 0.11,
            "P9": 0.10,
            "P10": 0.10,
        }
        
        # Correction thresholds
        self.correction_thresholds = {
            "severe_deviation": 0.3,
            "moderate_deviation": 0.5,
            "minor_deviation": 0.7,
            "acceptable_range": 0.8
        }
        
        # DNP baseline standards
        self.dnp_baseline_standards = self._initialize_dnp_standards()

    def _load_human_rights_standards(self) -> Dict[str, Any]:
        """Load Decálogo de Derechos Humanos standards for alignment validation."""
        return {
            "decalogo_standards": {
                "P1": {"title": "Derecho a la vida y seguridad", "weight": 0.12, "min_evidence": 3},
                "P2": {"title": "Dignidad humana", "weight": 0.11, "min_evidence": 2},
                "P3": {"title": "Igualdad y no discriminación", "weight": 0.10, "min_evidence": 3},
                "P4": {"title": "Participación ciudadana", "weight": 0.09, "min_evidence": 2},
                "P5": {"title": "Acceso a servicios básicos", "weight": 0.08, "min_evidence": 4},
                "P6": {"title": "Protección ambiental", "weight": 0.10, "min_evidence": 3},
                "P7": {"title": "Desarrollo económico inclusivo", "weight": 0.09, "min_evidence": 3},
                "P8": {"title": "Derechos culturales y territoriales", "weight": 0.11, "min_evidence": 2},
                "P9": {"title": "Acceso a la justicia", "weight": 0.10, "min_evidence": 3},
                "P10": {"title": "Transparencia y rendición de cuentas", "weight": 0.10, "min_evidence": 4}
            },
            "compliance_thresholds": {
                "CUMPLE": 0.75,
                "CUMPLE_PARCIAL": 0.50,
                "NO_CUMPLE": 0.0
            },
            "evidence_quality_weights": {
                "high_quality": 1.0,
                "medium_quality": 0.7,
                "low_quality": 0.3
            }
        }

    def _initialize_dnp_standards(self) -> Dict[str, Dict[str, float]]:
        """Initialize DNP baseline standards for human rights compliance"""
        return {
            "human_rights_baseline": {
                "P1": 0.75,  # Derecho a la vida, libertad, integridad y seguridad
                "P2": 0.70,  # Derechos de las mujeres
                "P3": 0.72,  # Derechos de la niñez y la adolescencia
                "P4": 0.68,  # Derechos de los jóvenes
                "P5": 0.73,  # Derechos de las personas adultas mayores
                "P6": 0.71,  # Personas con discapacidad
                "P7": 0.74,  # Grupos étnicos: pueblos indígenas y comunidades negras
                "P8": 0.69,  # Personas LGTBI
                "P9": 0.76,  # Víctimas del conflicto armado
                "P10": 0.67, # Líderes y defensores de derechos humanos
            },
            "dimensional_baselines": {
                "DE1": 0.75,  # Dimensión Institucional
                "DE2": 0.70,  # Dimensión Social
                "DE3": 0.80,  # Dimensión Económica
                "DE4": 0.65,  # Dimensión Ambiental
            },
            "causal_strength_requirements": {
                "minimum_causal_validity": 0.6,
                "evidence_alignment_threshold": 0.5,
                "robustness_requirement": 0.4
            }
        }

    def calculate_dnp_causal_correction(
        self,
        current_scores: Dict[str, float],
        dimension_id: str,
        evidence_patterns: List[str] = None
    ) -> Dict[str, Any]:
        """
# # #         Calculate causal correction scores measuring distance from DNP baseline standards  # Module not found  # Module not found  # Module not found
        """
        if not self.causal_correction_enabled:
            return {
                "correction_factor": 1.0,
                "causal_distance": 0.0,
                "dnp_deviation_score": 0.0,
                "correction_status": "DISABLED",
                "recommendations": []
            }

        try:
            # Build causal graph for the dimension
            causal_graph = self._build_dimension_causal_graph(dimension_id, evidence_patterns or [])
            
            # Validate causal logic
            causal_validity = self.dnp_analyzer.validate_causal_logic(causal_graph, dimension_id)
            
            # Extract causal evidence
            evidence = self.dnp_analyzer.extract_causal_evidence(causal_graph, evidence_patterns or [])
            
            # Calculate causal factor
            causal_factor = self.dnp_analyzer.calculate_causal_factor(causal_graph)
            
            # Compute DNP deviation scores
            dnp_deviations = self._compute_dnp_deviations(current_scores, dimension_id)
            
# # #             # Calculate causal distance from baseline  # Module not found  # Module not found  # Module not found
            causal_distance = self._calculate_causal_distance(
                current_scores, 
                self.dnp_baseline_standards, 
                causal_factor
            )
            
            # Determine correction factor
            correction_factor = self._compute_correction_factor(
                causal_validity,
                causal_distance,
                dnp_deviations,
                causal_factor
            )
            
            # Generate recommendations
            recommendations = self._generate_dnp_recommendations(
                dnp_deviations,
                causal_validity,
                evidence,
                dimension_id
            )
            
            # Determine correction status
            correction_status = self._classify_correction_status(causal_distance, causal_validity)
            
            return {
                "correction_factor": correction_factor,
                "causal_distance": causal_distance,
                "dnp_deviation_score": np.mean(list(dnp_deviations.values())),
                "correction_status": correction_status,
                "causal_validity_score": causal_validity,
                "evidence_count": len(evidence),
                "robustness_score": self._extract_robustness_score(causal_factor),
                "recommendations": recommendations,
                "detailed_deviations": dnp_deviations
            }
            
        except Exception as e:
            logger.error(f"DNP causal correction calculation failed: {e}")
            return {
                "correction_factor": 1.0,
                "causal_distance": 0.0,
                "dnp_deviation_score": 0.0,
                "correction_status": "ERROR",
                "recommendations": [f"Correction calculation failed: {str(e)}"]
            }

    def _build_dimension_causal_graph(self, dimension_id: str, evidence_patterns: List[str]) -> 'CausalGraph':
        """Build causal graph for specific dimension"""
        graph = CausalGraph()
        
        # Add nodes based on dimension
        if dimension_id == "DE1":  # Institutional
            nodes = ["governance_quality", "institutional_capacity", "transparency", "participation"]
        elif dimension_id == "DE2":  # Social
            nodes = ["education_access", "health_coverage", "social_protection", "equity"]
        elif dimension_id == "DE3":  # Economic
            nodes = ["economic_growth", "employment", "productivity", "competitiveness"]
        elif dimension_id == "DE4":  # Environmental
            nodes = ["environmental_protection", "sustainability", "climate_adaptation", "resource_management"]
        else:
            nodes = ["generic_input", "generic_process", "generic_outcome"]
        
        for node in nodes:
            graph.add_node(node)
        
        # Add causal edges (simplified structure)
        for i in range(len(nodes) - 1):
            graph.add_edge(nodes[i], nodes[i + 1])
        
        # Add confounders and proxies
        graph.add_confounder("unobserved_factors", nodes)
        for i, node in enumerate(nodes[:2]):  # Add proxies for first two nodes
            graph.add_proxy("unobserved_factors", f"proxy_{node}")
        
        # Set treatment and outcome
        if len(nodes) >= 2:
            graph.set_treatment(nodes[0])
            graph.set_outcome(nodes[-1])
        
        return graph

    def _compute_dnp_deviations(self, current_scores: Dict[str, float], dimension_id: str) -> Dict[str, float]:
# # #         """Compute deviations from DNP baseline standards"""  # Module not found  # Module not found  # Module not found
        deviations = {}
        
        # Check dimensional baseline
        if dimension_id in self.dnp_baseline_standards["dimensional_baselines"]:
            baseline = self.dnp_baseline_standards["dimensional_baselines"][dimension_id]
            current = current_scores.get(f"dimension_{dimension_id}", 0.0)
            deviations[f"dimension_{dimension_id}"] = abs(current - baseline) / baseline
        
        # Check Decálogo points
        for point_id in self.dnp_baseline_standards["human_rights_baseline"]:
            baseline = self.dnp_baseline_standards["human_rights_baseline"][point_id]
            current = current_scores.get(f"decalogo_{point_id}", 0.0)
            if current > 0:  # Only if score exists
                deviations[f"decalogo_{point_id}"] = abs(current - baseline) / baseline
        
        return deviations

    def _calculate_causal_distance(
        self, 
        current_scores: Dict[str, float], 
        baseline_standards: Dict[str, Dict[str, float]], 
        causal_factor: 'CausalFactor'
    ) -> float:
        """Calculate causal distance between current and baseline using causal factor"""
        if not hasattr(causal_factor, 'point_estimate') or np.isnan(causal_factor.point_estimate):
            # Fallback to simple Euclidean distance
            distances = []
            for category, baselines in baseline_standards.items():
                if category == "causal_strength_requirements":
                    continue
                for key, baseline in baselines.items():
                    current_key = f"dimension_{key}" if category == "dimensional_baselines" else f"decalogo_{key}"
                    current = current_scores.get(current_key, 0.0)
                    distances.append((current - baseline) ** 2)
            return np.sqrt(np.mean(distances)) if distances else 0.0
        
        # Use causal factor point estimate weighted by robustness
        base_distance = abs(causal_factor.point_estimate)
        robustness_weight = self._extract_robustness_score(causal_factor)
        
        return base_distance * (1 + robustness_weight)

    def _compute_correction_factor(
        self, 
        causal_validity: float, 
        causal_distance: float, 
        dnp_deviations: Dict[str, float], 
        causal_factor: 'CausalFactor'
    ) -> float:
        """Compute quantitative correction factor for realignment"""
# # #         # Base correction from causal validity  # Module not found  # Module not found  # Module not found
        validity_correction = max(0.5, causal_validity)
        
        # Distance penalty
        distance_penalty = np.exp(-causal_distance)
        
        # Deviation severity
        avg_deviation = np.mean(list(dnp_deviations.values())) if dnp_deviations else 0.0
        deviation_correction = max(0.3, 1.0 - avg_deviation)
        
        # Robustness bonus
        robustness_score = self._extract_robustness_score(causal_factor)
        robustness_bonus = 1.0 + (robustness_score * 0.2)
        
        # Combined correction factor
        correction_factor = validity_correction * distance_penalty * deviation_correction * robustness_bonus
        
        return np.clip(correction_factor, 0.1, 2.0)

    def _extract_robustness_score(self, causal_factor: 'CausalFactor') -> float:
# # #         """Extract robustness score from causal factor"""  # Module not found  # Module not found  # Module not found
        if not hasattr(causal_factor, 'dro_robustness') or not causal_factor.dro_robustness:
            return 0.5  # Default moderate robustness
        
        # Average robustness across epsilon values
        robustness_values = list(causal_factor.dro_robustness.values())
        avg_robustness = np.mean(robustness_values) if robustness_values else 0.5
        
        # Invert because lower values mean more robust
        return max(0.0, 1.0 - avg_robustness)

    def _classify_correction_status(self, causal_distance: float, causal_validity: float) -> str:
        """Classify correction status based on distance and validity"""
        if causal_validity < 0.3 or causal_distance > 1.0:
            return "CRITICAL_DEVIATION"
        elif causal_validity < 0.5 or causal_distance > 0.7:
            return "SIGNIFICANT_DEVIATION"
        elif causal_validity < 0.7 or causal_distance > 0.4:
            return "MODERATE_DEVIATION"
        elif causal_distance > 0.2:
            return "MINOR_DEVIATION"
        else:
            return "COMPLIANT"

    def _generate_dnp_recommendations(
        self, 
        dnp_deviations: Dict[str, float], 
        causal_validity: float, 
        evidence: List['Evidence'], 
        dimension_id: str
    ) -> List[str]:
        """Generate DNP-specific recommendations for realignment"""
        recommendations = []
        
        # Causal validity recommendations
        if causal_validity < 0.5:
            recommendations.append(
                "Strengthen causal logic between interventions and human rights outcomes"
            )
            recommendations.append(
                "Provide more explicit evidence linking plan components to Decálogo compliance"
            )
        
        # Deviation-specific recommendations
        severe_deviations = {k: v for k, v in dnp_deviations.items() if v > 0.5}
        for deviation_key, deviation_value in severe_deviations.items():
            if "decalogo_P1" in deviation_key:
                recommendations.append(
                    "Enhance life, liberty, and security protections in development programs"
                )
            elif "decalogo_P2" in deviation_key:
                recommendations.append(
                    "Strengthen women's rights integration across all development sectors"
                )
            elif "decalogo_P9" in deviation_key:
                recommendations.append(
                    "Improve victim assistance and transitional justice mechanisms"
                )
            elif f"dimension_{dimension_id}" in deviation_key:
                recommendations.append(
                    f"Realign {dimension_id} dimension with DNP baseline standards"
                )
        
        # Evidence-based recommendations
        if len(evidence) < 3:
            recommendations.append(
                "Increase documentary evidence supporting human rights impact pathways"
            )
        
        # General DNP compliance
        if not recommendations:
            recommendations.append(
                "Maintain current DNP alignment and monitor for emerging deviations"
            )
        
        return recommendations[:5]  # Limit to top 5 recommendations

    def validate_dnp_contrast(
        self, 
        current_scores: Dict[str, float], 
        dimension_id: str = None
    ) -> Dict[str, Any]:
        """
        DNP contrast validation to flag fundamental contradictions with human rights frameworks
        """
        validation_results = {
            "fundamental_contradictions": [],
            "contrast_flags": [],
            "compliance_gaps": {},
            "severity_assessment": "LOW",
            "requires_immediate_action": False
        }
        
        try:
            # Check for fundamental contradictions
            contradictions = self._detect_fundamental_contradictions(current_scores)
            validation_results["fundamental_contradictions"] = contradictions
            
            # Flag critical gaps
            critical_gaps = self._identify_critical_gaps(current_scores)
            validation_results["compliance_gaps"] = critical_gaps
            
            # Generate contrast flags
            contrast_flags = self._generate_contrast_flags(current_scores, contradictions, critical_gaps)
            validation_results["contrast_flags"] = contrast_flags
            
            # Assess severity
            severity = self._assess_contrast_severity(contradictions, critical_gaps)
            validation_results["severity_assessment"] = severity
            validation_results["requires_immediate_action"] = severity in ["HIGH", "CRITICAL"]
            
            return validation_results
            
        except Exception as e:
            logger.error(f"DNP contrast validation failed: {e}")
            validation_results["contrast_flags"].append(f"Validation error: {str(e)}")
            return validation_results

    def _detect_fundamental_contradictions(self, current_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Detect fundamental contradictions with human rights frameworks"""
        contradictions = []
        
        # Check for scores that fundamentally contradict human rights principles
        critical_points = ["P1", "P3", "P9"]  # Life/security, children, victims
        
        for point in critical_points:
            score_key = f"decalogo_{point}"
            if score_key in current_scores:
                score = current_scores[score_key]
                baseline = self.dnp_baseline_standards["human_rights_baseline"][point]
                
                # Fundamental contradiction: score is less than 40% of baseline
                if score < (baseline * 0.4):
                    contradictions.append({
                        "type": "FUNDAMENTAL_VIOLATION",
                        "component": point,
                        "current_score": score,
                        "expected_minimum": baseline * 0.4,
                        "severity": "CRITICAL",
                        "description": f"Score for {point} critically below human rights minimum"
                    })
        
        return contradictions

    def _identify_critical_gaps(self, current_scores: Dict[str, float]) -> Dict[str, float]:
        """Identify critical compliance gaps"""
        gaps = {}
        
        # Check all Decálogo points
        for point_id, baseline in self.dnp_baseline_standards["human_rights_baseline"].items():
            score_key = f"decalogo_{point_id}"
            if score_key in current_scores:
                current = current_scores[score_key]
                if current < baseline:
                    gap_magnitude = (baseline - current) / baseline
                    gaps[point_id] = gap_magnitude
        
        # Filter only critical gaps (>30% below baseline)
        critical_gaps = {k: v for k, v in gaps.items() if v > 0.3}
        return critical_gaps

    def _generate_contrast_flags(
        self, 
        current_scores: Dict[str, float], 
        contradictions: List[Dict[str, Any]], 
        critical_gaps: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Generate specific contrast flags"""
        flags = []
        
# # #         # Flags from contradictions  # Module not found  # Module not found  # Module not found
        for contradiction in contradictions:
            flags.append({
                "flag_type": "HUMAN_RIGHTS_VIOLATION",
                "component": contradiction["component"],
                "severity": contradiction["severity"],
                "message": contradiction["description"],
                "action_required": "IMMEDIATE_REMEDIATION"
            })
        
# # #         # Flags from critical gaps  # Module not found  # Module not found  # Module not found
        for component, gap_size in critical_gaps.items():
            flags.append({
                "flag_type": "CRITICAL_GAP",
                "component": component,
                "severity": "HIGH" if gap_size > 0.5 else "MEDIUM",
                "message": f"{component} shows {gap_size:.1%} gap below DNP baseline",
                "action_required": "STRENGTHEN_COMPONENT"
            })
        
        # System-level flags
        if len(critical_gaps) > 5:
            flags.append({
                "flag_type": "SYSTEMIC_FAILURE",
                "component": "OVERALL_PLAN",
                "severity": "CRITICAL",
                "message": "Multiple critical gaps indicate systemic human rights failure",
                "action_required": "COMPREHENSIVE_REDESIGN"
            })
        
        return flags

    def _assess_contrast_severity(
        self, 
        contradictions: List[Dict[str, Any]], 
        critical_gaps: Dict[str, float]
    ) -> str:
        """Assess overall severity of contrasts"""
        if contradictions:
            return "CRITICAL"
        elif len(critical_gaps) > 3:
            return "HIGH"
        elif len(critical_gaps) > 1:
            return "MEDIUM"
        else:
            return "LOW"

    def extract_features(
        self, pdt_context: PDTContext, document_package: DocumentPackage
    ) -> np.ndarray:
        """
        Extrae características granulares para el modelado predictivo
        """
        features = []

        # Características contextuales municipales
        features.extend(
            [
                pdt_context.population,
                pdt_context.area_km2,
                np.log1p(pdt_context.budget) if pdt_context.budget > 0 else 0,
                pdt_context.gdp_per_capita,
                pdt_context.urbanization_rate,
                pdt_context.education_index,
                pdt_context.health_index,
                pdt_context.poverty_index,
                pdt_context.infrastructure_index,
                pdt_context.governance_index,
                pdt_context.environmental_index,
            ]
        )

        # Características documentales
        qi = document_package.quality_indicators
        features.extend(
            [
                qi.completeness_index,
                qi.logical_coherence_hint,
                qi.tables_found,
                qi.ocr_ratio,
                len(qi.mandatory_sections_present),
                len(qi.missing_sections),
                len(document_package.sections),
                len(document_package.tables),
            ]
        )

        # Características históricas (si disponibles)
        if pdt_context.previous_pdt_scores:
            prev_scores = list(pdt_context.previous_pdt_scores.values())
            features.extend(
                [
                    np.mean(prev_scores) if prev_scores else 0.0,
                    np.std(prev_scores) if len(prev_scores) > 1 else 0.0,
                    max(prev_scores) if prev_scores else 0.0,
                    min(prev_scores) if prev_scores else 0.0,
                ]
            )
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        # Características regionales
        if pdt_context.regional_indicators:
            regional_values = list(pdt_context.regional_indicators.values())
            features.extend(
                [
                    np.mean(regional_values) if regional_values else 0.0,
                    len(regional_values),
                ]
            )
        else:
            features.extend([0.0, 0])

        return np.array(features, dtype=np.float32)

    def train_models(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Entrena modelos predictivos usando datos históricos
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Cannot train models without scikit-learn")
            return {"status": "failed", "reason": "scikit-learn not available"}

        logger.info(
            f"Training adaptive scoring models with {len(training_data)} samples"
        )

        try:
            # Preparar datos
            X_features = []
            y_dimensions = {dim: [] for dim in self.dimension_weights.keys()}
            y_decalogo = {point: [] for point in self.decalogo_weights.keys()}
            y_global = []

            for sample in training_data:
                # Extraer características usando modelos Pydantic
                try:
                    context = PDTContext(**sample["pdt_context"])

                    # Handle DocumentPackage creation with proper validation
                    package_data = sample["document_package"]
                    if "quality_indicators" not in package_data:
# # #                         from .models import QualityIndicators  # Module not found  # Module not found  # Module not found

                        package_data["quality_indicators"] = QualityIndicators(
                            completeness_index=0.5,
                            logical_coherence_hint=0.5,
                            tables_found=0,
                            ocr_ratio=0.0,
                        )

                    package = DocumentPackage(**package_data)
                    features = self.extract_features(context, package)
                    X_features.append(features)
                except Exception as e:
                    logger.warning(f"Skipping invalid training sample: {str(e)}")
                    continue

                # Extraer etiquetas objetivo
                scores = sample["scores"]
                for dim in self.dimension_weights.keys():
                    y_dimensions[dim].append(scores.get(f"dimension_{dim}", 0.0))

                for point in self.decalogo_weights.keys():
                    y_decalogo[point].append(scores.get(f"decalogo_{point}", 0.0))

                y_global.append(scores.get("global_score", 0.0))

            X = np.array(X_features)

            # Normalización de características
            X_scaled = self.feature_scaler.fit_transform(X)

            # Selección de características
            self.feature_selector = SelectKBest(f_regression, k=min(15, X.shape[1]))
            X_selected = self.feature_selector.fit_transform(X_scaled, y_global)

            training_results = {}

            # Entrenar modelos por dimensión
            for dim, scores in y_dimensions.items():
                if len(set(scores)) > 1:  # Verificar variabilidad
                    model = RandomForestRegressor(**self.model_config)
                    model.fit(X_selected, scores)

                    # Validación cruzada
                    cv_scores = cross_val_score(
                        model, X_selected, scores, cv=5, scoring="r2"
                    )

                    self.dimension_models[dim] = model
                    self.model_performance[f"dimension_{dim}"] = {
                        "r2_mean": np.mean(cv_scores),
                        "r2_std": np.std(cv_scores),
                        "feature_importance": model.feature_importances_.tolist(),
                    }

                    training_results[f"dimension_{dim}"] = {
                        "r2_score": np.mean(cv_scores),
                        "samples": len(scores),
                    }

            # Entrenar modelos por punto del Decálogo
            for point, scores in y_decalogo.items():
                if len(set(scores)) > 1:
                    model = GradientBoostingRegressor(**self.model_config)
                    model.fit(X_selected, scores)

                    cv_scores = cross_val_score(
                        model, X_selected, scores, cv=5, scoring="r2"
                    )

                    self.decalogo_models[point] = model
                    self.model_performance[f"decalogo_{point}"] = {
                        "r2_mean": np.mean(cv_scores),
                        "r2_std": np.std(cv_scores),
                        "feature_importance": model.feature_importances_.tolist(),
                    }

                    training_results[f"decalogo_{point}"] = {
                        "r2_score": np.mean(cv_scores),
                        "samples": len(scores),
                    }

            # Entrenar modelo global
            if len(set(y_global)) > 1:
                self.global_model = RandomForestRegressor(**self.model_config)
                self.global_model.fit(X_selected, y_global)

                cv_scores = cross_val_score(
                    self.global_model, X_selected, y_global, cv=5, scoring="r2"
                )
                self.model_performance["global"] = {
                    "r2_mean": np.mean(cv_scores),
                    "r2_std": np.std(cv_scores),
                    "feature_importance": self.global_model.feature_importances_.tolist(),
                }

                training_results["global"] = {
                    "r2_score": np.mean(cv_scores),
                    "samples": len(y_global),
                }

            # Calcular importancia de características
            self._calculate_feature_importance()

            # Guardar modelos
            self.save_models()

            logger.info(
                f"Training completed successfully. Models trained: {len(training_results)}"
            )

            return {
                "status": "success",
                "models_trained": len(training_results),
                "performance_summary": training_results,
                "feature_importance": self.feature_importance,
            }

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {"status": "failed", "reason": str(e)}

    def predict_scores(
        self,
        pdt_context: PDTContext,
        document_package: DocumentPackage,
        initial_scores: Dict[str, float],
        enable_dnp_correction: bool = True,
    ) -> AdaptiveScoringResults:
        """
        Genera predicciones adaptativas con corrección causal DNP
        """
        logger.info(
            f"Generating adaptive predictions with DNP correction for municipality {pdt_context.municipality_name}"
        )

        try:
            # Extraer características
            features = self.extract_features(pdt_context, document_package)

            if self.feature_scaler and self.feature_selector:
                features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
                features_selected = self.feature_selector.transform(features_scaled)
            else:
                features_selected = features.reshape(1, -1)

            # Initialize DNP correction results storage
            dnp_corrections = {}
            dnp_validation = {}

            # Predicciones por dimensión con corrección DNP
            dimension_scores = {}
            for dim_id in self.dimension_weights.keys():
                raw_score = initial_scores.get(f"dimension_{dim_id}", 0.0)

                if dim_id in self.dimension_models and SKLEARN_AVAILABLE:
                    predicted_score = self.dimension_models[dim_id].predict(
                        features_selected
                    )[0]
                    predicted_score = np.clip(predicted_score, 0.0, 1.0)
                else:
                    predicted_score = raw_score  # Fallback

                # Apply DNP causal correction
                if enable_dnp_correction:
                    dimension_initial_scores = {
                        k: v for k, v in initial_scores.items() 
                        if k.startswith(f"dimension_{dim_id}") or k.startswith("decalogo_")
                    }
                    
                    dnp_correction = self.calculate_dnp_causal_correction(
                        dimension_initial_scores,
                        dim_id,
                        evidence_patterns=self._extract_evidence_patterns(document_package)
                    )
                    dnp_corrections[dim_id] = dnp_correction
                    
                    # Apply correction factor to predicted score
                    corrected_score = predicted_score * dnp_correction["correction_factor"]
                    corrected_score = np.clip(corrected_score, 0.0, 1.0)
                    
                    # Use corrected score as final prediction
                    final_predicted_score = corrected_score
                else:
                    final_predicted_score = predicted_score
                    dnp_correction = {"correction_factor": 1.0}

                # Calcular puntuación ponderada
                weighted_score = raw_score * self.dimension_weights[dim_id]

                # Determinar estado de satisfacibilidad
                compliance_status = self._determine_compliance(
                    final_predicted_score, self.compliance_thresholds[dim_id]
                )

                dimension_scores[dim_id] = DimensionScore(
                    dimension_id=dim_id,
                    raw_score=raw_score,
                    weighted_score=weighted_score,
                    predicted_score=final_predicted_score,
                    compliance_status=compliance_status,
                    contributing_questions={},
                    quality_metrics={
                        "dnp_correction_factor": dnp_correction["correction_factor"],
                        "causal_distance": dnp_correction.get("causal_distance", 0.0),
                        "correction_status": dnp_correction.get("correction_status", "UNKNOWN")
                    },
                    confidence_level=self.model_performance.get(
                        f"dimension_{dim_id}", {}
                    ).get("r2_mean", 1.0),
                )

            # Predicciones por punto del Decálogo con corrección DNP
            decalogo_scores = {}
            for point_id in self.decalogo_weights.keys():
                raw_score = initial_scores.get(f"decalogo_{point_id}", 0.0)

                if point_id in self.decalogo_models and SKLEARN_AVAILABLE:
                    predicted_score = self.decalogo_models[point_id].predict(
                        features_selected
                    )[0]
                    predicted_score = np.clip(predicted_score, 0.0, 1.0)
                else:
                    predicted_score = raw_score

                # Apply DNP baseline correction
                if enable_dnp_correction:
                    baseline = self.dnp_baseline_standards["human_rights_baseline"][point_id]
                    deviation = abs(predicted_score - baseline) / baseline
                    
                    # Apply correction based on severity of deviation
                    if deviation > 0.5:  # Severe deviation
                        correction_factor = 0.7
                    elif deviation > 0.3:  # Moderate deviation
                        correction_factor = 0.85
                    else:  # Minor or no deviation
                        correction_factor = 1.0
                    
                    corrected_score = predicted_score * correction_factor
                    # Ensure we don't correct below baseline for critical rights
                    if point_id in ["P1", "P3", "P9"]:  # Critical rights
                        corrected_score = max(corrected_score, baseline * 0.6)
                    
                    final_predicted_score = np.clip(corrected_score, 0.0, 1.0)
                else:
                    final_predicted_score = predicted_score

                weighted_score = raw_score * self.decalogo_weights[point_id]

                compliance_status = self._determine_compliance(
                    final_predicted_score, self.compliance_thresholds["DECALOGO"][point_id]
                )

                decalogo_scores[point_id] = DecalogoPointScore(
                    point_id=point_id,
                    raw_score=raw_score,
                    weighted_score=weighted_score,
                    predicted_score=final_predicted_score,
                    compliance_status=compliance_status,
                    dimension_contributions={},
                    thresholds=self.compliance_thresholds["DECALOGO"][point_id],
                    confidence_level=self.model_performance.get(
                        f"decalogo_{point_id}", {}
                    ).get("r2_mean", 1.0),
                )

            # Perform DNP contrast validation
            if enable_dnp_correction:
                all_scores = {**initial_scores}
                for dim_id, dim_score in dimension_scores.items():
                    all_scores[f"dimension_{dim_id}_corrected"] = dim_score.predicted_score
                for point_id, point_score in decalogo_scores.items():
                    all_scores[f"decalogo_{point_id}_corrected"] = point_score.predicted_score
                
                dnp_validation = self.validate_dnp_contrast(all_scores)

            # Predicción global con corrección
            global_raw = sum(dim.weighted_score for dim in dimension_scores.values())

            if self.global_model and SKLEARN_AVAILABLE:
                global_predicted = self.global_model.predict(features_selected)[0]
                global_predicted = np.clip(global_predicted, 0.0, 1.0)
            else:
                global_predicted = global_raw
            
            # Apply global DNP correction if severe issues detected
            if enable_dnp_correction and dnp_validation.get("severity_assessment") in ["HIGH", "CRITICAL"]:
                global_correction_factor = 0.8 if dnp_validation["severity_assessment"] == "HIGH" else 0.6
                global_predicted *= global_correction_factor

            # Calcular confianza del modelo
            model_confidence = self._calculate_model_confidence(features_selected)

            # Métricas de calidad de predicción con DNP
            prediction_quality = {
                "feature_coverage": np.sum(features != 0) / len(features),
                "historical_data_available": len(pdt_context.previous_pdt_scores) > 0,
                "context_completeness": self._assess_context_completeness(pdt_context),
                "model_agreement": self._calculate_model_agreement(
                    dimension_scores, decalogo_scores
                ),
                "dnp_correction_enabled": enable_dnp_correction,
                "dnp_validation_severity": dnp_validation.get("severity_assessment", "UNKNOWN"),
                "causal_corrections_applied": len(dnp_corrections),
            }

            # Create extended results with DNP information
            results_dict = {
                "global_score": global_raw,
                "predicted_global_score": global_predicted,
                "dimension_scores": dimension_scores,
                "decalogo_scores": decalogo_scores,
                "feature_importance": self.feature_importance,
                "model_confidence": model_confidence,
                "prediction_quality_metrics": prediction_quality,
                "dnp_causal_corrections": dnp_corrections,
                "dnp_contrast_validation": dnp_validation,
            }

            # Create results object (extended if possible)
            try:
                results = AdaptiveScoringResults(**results_dict)
            except TypeError:
                # Fallback to original structure if extended fields not supported
                results = AdaptiveScoringResults(
                    global_score=global_raw,
                    predicted_global_score=global_predicted,
                    dimension_scores=dimension_scores,
                    decalogo_scores=decalogo_scores,
                    feature_importance=self.feature_importance,
                    model_confidence=model_confidence,
                    prediction_quality_metrics=prediction_quality,
                )
                # Store DNP results as attributes
                results.dnp_causal_corrections = dnp_corrections
                results.dnp_contrast_validation = dnp_validation

            # Apply Math Stage5 enhancement for causal correction scoring
            if self.has_stage5_enhancement:
                try:
                    enhancement_result = self.math_stage5_enhancer.enhance_causal_correction_scoring(
                        evaluation_scores=initial_scores,
                        pdt_context=pdt_context,
# # #                         document_features=None  # Will be extracted from context  # Module not found  # Module not found  # Module not found
                    )
                    
                    # Update results with transport-enhanced scoring
                    results.predicted_global_score = enhancement_result.transport_enhanced_score
                    
                    # Add mathematical certificates to prediction quality metrics
                    results.prediction_quality_metrics.update({
                        "transport_alignment_confidence": enhancement_result.alignment_confidence,
                        "stability_verified": enhancement_result.stability_verified,
                        "dnp_compliance_evidence": enhancement_result.dnp_compliance_evidence,
                        "optimal_transport_cost": enhancement_result.mathematical_certificates.get("optimal_cost", 0.0),
                        "spectral_radius": enhancement_result.mathematical_certificates.get("spectral_radius", 1.0),
                        "entropy_regularization": enhancement_result.mathematical_certificates.get("entropy_regularization", 0.1)
                    })
                    
                    logger.info(
                        f"Math Stage5 enhancement applied. "
                        f"Score: {global_predicted:.3f} → {enhancement_result.transport_enhanced_score:.3f}, "
                        f"Alignment confidence: {enhancement_result.alignment_confidence:.3f}, "
                        f"Stability: {enhancement_result.stability_verified}"
                    )
                    
                except Exception as e:
                    logger.warning(f"Math Stage5 enhancement failed: {e}")

            logger.info(
                f"Adaptive scoring completed. Global score: {global_raw:.3f} → {results.predicted_global_score:.3f}"
            )

            return results

        except Exception as e:
            logger.error(f"Prediction with DNP correction failed: {str(e)}")
            # Devolver resultados fallback
            return self._create_fallback_results(initial_scores)
    
    def validate_dnp_alignment(
        self,
        scoring_results: 'AdaptiveScoringResults',
        evidence_data: Dict[str, Any],
        cluster_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate alignment with DNP (Decálogo de Derechos Humanos) standards."""
        
        validation_results = {
            "overall_compliance": "PENDING",
            "point_compliance": {},
            "evidence_gaps": [],
            "recommendations": [],
            "compliance_score": 0.0
        }
        
        try:
            # Validate each Decálogo point
            total_weighted_score = 0.0
            compliant_points = 0
            
            for point_id, standard in self.human_rights_standards["decalogo_standards"].items():
                point_validation = self._validate_decalogo_point(
                    point_id, standard, evidence_data, cluster_data, scoring_results
                )
                
                validation_results["point_compliance"][point_id] = point_validation
                
                # Calculate weighted score
                point_score = point_validation.get("score", 0.0)
                weight = standard.get("weight", 0.1)
                total_weighted_score += point_score * weight
                
                if point_validation.get("status") == "CUMPLE":
                    compliant_points += 1
                elif point_validation.get("evidence_gap"):
                    validation_results["evidence_gaps"].append({
                        "point": point_id,
                        "title": standard["title"],
                        "missing_evidence": point_validation.get("missing_evidence", 0)
                    })
            
            # Determine overall compliance
            validation_results["compliance_score"] = total_weighted_score
            
            if total_weighted_score >= self.human_rights_standards["compliance_thresholds"]["CUMPLE"]:
                validation_results["overall_compliance"] = "CUMPLE"
            elif total_weighted_score >= self.human_rights_standards["compliance_thresholds"]["CUMPLE_PARCIAL"]:
                validation_results["overall_compliance"] = "CUMPLE_PARCIAL"
            else:
                validation_results["overall_compliance"] = "NO_CUMPLE"
            
            # Generate recommendations
            validation_results["recommendations"] = self._generate_dnp_recommendations(
                validation_results["point_compliance"],
                validation_results["evidence_gaps"]
            )
            
            logger.info(f"DNP alignment validation completed. Overall compliance: {validation_results['overall_compliance']} (score: {total_weighted_score:.3f})")
            
        except Exception as e:
            logger.error(f"DNP alignment validation failed: {str(e)}")
            validation_results["error"] = str(e)
            validation_results["overall_compliance"] = "ERROR"
            
        return validation_results
    
    def _validate_decalogo_point(
        self,
        point_id: str,
        standard: Dict[str, Any],
        evidence_data: Dict[str, Any],
        cluster_data: Dict[str, Any],
        scoring_results: 'AdaptiveScoringResults'
    ) -> Dict[str, Any]:
        """Validate a specific Decálogo point against available evidence."""
        
        # Get evidence for this point
        point_evidence = []
        if isinstance(evidence_data, dict):
            for evidence_id, evidence_content in evidence_data.items():
                if isinstance(evidence_content, dict):
                    evidence_tags = evidence_content.get("tags", [])
                    if point_id.lower() in [tag.lower() for tag in evidence_tags]:
                        point_evidence.append({
                            "id": evidence_id,
                            "quality": evidence_content.get("quality", "medium"),
                            "relevance": evidence_content.get("relevance_score", 0.5)
                        })
        
        # Check cluster-specific evidence
        cluster_evidence_count = 0
        if isinstance(cluster_data, dict) and "cluster_audit" in cluster_data:
            cluster_processing = cluster_data["cluster_audit"].get("cluster_processing", {})
            for cluster_id, cluster_info in cluster_processing.items():
                evaluation_results = cluster_info.get("evaluation_results", {})
                if point_id in evaluation_results:
                    cluster_evidence_count += evaluation_results[point_id].get("evidence_count", 0)
        
        # Calculate point score
        evidence_score = 0.0
        quality_weights = self.human_rights_standards["evidence_quality_weights"]
        
        for evidence in point_evidence:
            quality = evidence.get("quality", "medium")
            quality_weight = quality_weights.get(f"{quality}_quality", 0.5)
            relevance = evidence.get("relevance", 0.5)
            evidence_score += quality_weight * relevance
        
        # Normalize by expected evidence count
        min_evidence = standard.get("min_evidence", 2)
        if len(point_evidence) + cluster_evidence_count >= min_evidence:
            normalized_score = min(1.0, evidence_score / min_evidence)
        else:
            normalized_score = evidence_score / min_evidence
        
        # Determine compliance status
        if normalized_score >= 0.75:
            status = "CUMPLE"
        elif normalized_score >= 0.50:
            status = "CUMPLE_PARCIAL"
        else:
            status = "NO_CUMPLE"
        
        return {
            "point_id": point_id,
            "title": standard["title"],
            "status": status,
            "score": normalized_score,
            "evidence_count": len(point_evidence) + cluster_evidence_count,
            "min_evidence_required": min_evidence,
            "evidence_gap": (len(point_evidence) + cluster_evidence_count) < min_evidence,
            "missing_evidence": max(0, min_evidence - len(point_evidence) - cluster_evidence_count),
            "evidence_quality_distribution": {
                quality: len([e for e in point_evidence if e.get("quality") == quality])
                for quality in ["high", "medium", "low"]
            }
        }
    
    def _generate_dnp_recommendations(
        self,
        point_compliance: Dict[str, Any],
        evidence_gaps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations for improving DNP compliance."""
        
        recommendations = []
        
        # Evidence gap recommendations
        for gap in evidence_gaps:
            recommendations.append({
                "type": "evidence_collection",
                "priority": "high",
                "point": gap["point"],
                "title": gap["title"],
                "action": f"Recopilar {gap['missing_evidence']} evidencias adicionales para {gap['title']}",
                "specific_actions": [
                    f"Documentar actividades relacionadas con {gap['title'].lower()}",
                    "Obtener testimonios de beneficiarios",
                    "Recopilar datos cuantitativos de impacto"
                ]
            })
        
        # Quality improvement recommendations
        low_scoring_points = [
            (point_id, data) for point_id, data in point_compliance.items()
            if data.get("score", 0) < 0.6
        ]
        
        for point_id, point_data in low_scoring_points:
            recommendations.append({
                "type": "quality_improvement",
                "priority": "medium",
                "point": point_id,
                "title": point_data.get("title", ""),
                "action": f"Mejorar la calidad de evidencias para {point_data.get('title', point_id)}",
                "current_score": point_data.get("score", 0),
                "target_score": 0.75,
                "specific_actions": [
                    "Incluir fuentes primarias de información",
                    "Agregar datos cuantitativos verificables",
                    "Proporcionar contexto y análisis detallado"
                ]
            })
        
        # Cross-cutting recommendations
        fully_compliant = sum(1 for data in point_compliance.values() if data.get("status") == "CUMPLE")
        if fully_compliant < len(point_compliance) * 0.7:  # Less than 70% compliance
            recommendations.append({
                "type": "systematic_improvement",
                "priority": "high",
                "action": "Implementar un sistema integral de seguimiento de derechos humanos",
                "rationale": f"Solo {fully_compliant}/{len(point_compliance)} puntos del Decálogo cumplen completamente",
                "specific_actions": [
                    "Establecer indicadores de seguimiento para cada punto del Decálogo",
                    "Crear un sistema de reporte periódico",
                    "Capacitar al equipo en estándares de derechos humanos",
                    "Implementar mecanismos de participación ciudadana"
                ]
            })
        
        return sorted(recommendations, key=lambda x: {"high": 0, "medium": 1, "low": 2}[x.get("priority", "low")])


class DNPAlignmentValidator:
    """Validates alignment with Decálogo de Derechos Humanos standards."""
    
    def __init__(self):
        self.validation_cache = {}
        
    def validate_alignment(
        self, 
        evidence: Dict[str, Any], 
        standards: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate evidence alignment with DNP standards."""
        
        cache_key = hashlib.md5(
            json.dumps(evidence, sort_keys=True).encode() +
            json.dumps(standards, sort_keys=True).encode()
        ).hexdigest()
        
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
            
        validation_result = {
            "aligned": True,
            "compliance_score": 0.0,
            "issues": []
        }
        
        # Perform validation logic
        # This is a simplified implementation
        
        self.validation_cache[cache_key] = validation_result
        return validation_result

    def _extract_evidence_patterns(self, document_package: DocumentPackage) -> List[str]:
# # #         """Extract evidence patterns from document package for causal analysis"""  # Module not found  # Module not found  # Module not found
        patterns = []
        
# # #         # Extract from sections  # Module not found  # Module not found  # Module not found
        for section in document_package.sections:
            if hasattr(section, 'content') and section.content:
                # Look for causal language patterns
                causal_words = ['porque', 'debido a', 'como resultado', 'genera', 'produce', 'causa']
                if any(word in section.content.lower() for word in causal_words):
                    patterns.append(section.content[:200])  # First 200 chars
        
# # #         # Extract from tables if they contain relevant information  # Module not found  # Module not found  # Module not found
        for table in document_package.tables:
            if hasattr(table, 'caption') and table.caption:
                patterns.append(table.caption)
        
        # Add quality indicators as patterns
        qi = document_package.quality_indicators
        if qi.completeness_index > 0.7:
            patterns.append("High document completeness detected")
        if qi.logical_coherence_hint > 0.6:
            patterns.append("Strong logical coherence in document structure")
        
        return patterns[:10]  # Limit to 10 patterns for performance

    def _determine_compliance(
        self, score: float, thresholds: Dict[str, float]
    ) -> ComplianceStatus:
        """Determina el estado de satisfacibilidad basado en umbrales"""
        if score >= thresholds["CUMPLE"]:
            return ComplianceStatus.CUMPLE
        elif score >= thresholds["CUMPLE_PARCIAL"]:
            return ComplianceStatus.CUMPLE_PARCIAL
        else:
            return ComplianceStatus.NO_CUMPLE

    def _calculate_feature_importance(self) -> None:
        """Calcula la importancia agregada de características"""
        if not SKLEARN_AVAILABLE:
            return

        importance_scores = {}
        model_count = 0

        # Agregar importancias de todos los modelos
        for model_name, model in {
            **self.dimension_models,
            **self.decalogo_models,
        }.items():
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                for i, importance in enumerate(importances):
                    feature_name = f"feature_{i}"
                    if feature_name not in importance_scores:
                        importance_scores[feature_name] = 0.0
                    importance_scores[feature_name] += importance
                model_count += 1

        # Incluir modelo global
        if self.global_model and hasattr(self.global_model, "feature_importances_"):
            importances = self.global_model.feature_importances_
            for i, importance in enumerate(importances):
                feature_name = f"feature_{i}"
                if feature_name not in importance_scores:
                    importance_scores[feature_name] = 0.0
                importance_scores[feature_name] += importance
            model_count += 1

        # Promediar importancias
        if model_count > 0:
            for feature_name in importance_scores:
                importance_scores[feature_name] /= model_count

        self.feature_importance = importance_scores

    def _calculate_model_confidence(self, features: np.ndarray) -> float:
        """Calcula la confianza agregada del modelo"""
        if not self.model_performance:
            return 0.5

        r2_scores = [perf.get("r2_mean", 0) for perf in self.model_performance.values()]
        return float(np.mean([max(0, score) for score in r2_scores]))

    def _assess_context_completeness(self, context: PDTContext) -> float:
        """Evalúa la completitud del contexto municipal"""
        total_fields = 0
        complete_fields = 0

        for field_name, field_value in asdict(context).items():
            if not field_name.startswith("_"):
                total_fields += 1
                if field_value is not None and field_value != 0:
                    complete_fields += 1

        return complete_fields / total_fields if total_fields > 0 else 0.0

    def _calculate_model_agreement(self, dim_scores: Dict, dec_scores: Dict) -> float:
        """Calcula el acuerdo entre predicciones de diferentes modelos"""
        all_predictions = []

        for score_obj in dim_scores.values():
            if hasattr(score_obj, "predicted_score"):
                all_predictions.append(score_obj.predicted_score)

        for score_obj in dec_scores.values():
            if hasattr(score_obj, "predicted_score"):
                all_predictions.append(score_obj.predicted_score)

        if len(all_predictions) < 2:
            return 1.0

        return (
            1.0 - (np.std(all_predictions) / np.mean(all_predictions))
            if np.mean(all_predictions) > 0
            else 0.0
        )

    def _create_fallback_results(
        self, initial_scores: Dict[str, float]
    ) -> AdaptiveScoringResults:
        """Crea resultados fallback cuando falla la predicción"""
        dimension_scores = {}
        for dim_id in self.dimension_weights.keys():
            score = initial_scores.get(f"dimension_{dim_id}", 0.0)
            dimension_scores[dim_id] = DimensionScore(
                dimension_id=dim_id,
                raw_score=score,
                weighted_score=score * self.dimension_weights[dim_id],
                predicted_score=score,
                compliance_status=ComplianceStatus.NO_CUMPLE,
                confidence_level=0.5,
            )

        decalogo_scores = {}
        for point_id in self.decalogo_weights.keys():
            score = initial_scores.get(f"decalogo_{point_id}", 0.0)
            decalogo_scores[point_id] = DecalogoPointScore(
                point_id=point_id,
                raw_score=score,
                weighted_score=score * self.decalogo_weights[point_id],
                predicted_score=score,
                compliance_status=ComplianceStatus.NO_CUMPLE,
                confidence_level=0.5,
            )

        return AdaptiveScoringResults(
            global_score=sum(initial_scores.values()) / len(initial_scores)
            if initial_scores
            else 0.0,
            predicted_global_score=0.0,
            dimension_scores=dimension_scores,
            decalogo_scores=decalogo_scores,
            model_confidence=0.0,
            analysis_timestamp=datetime.now(),
        )

    def save_models(self) -> bool:
        """Guarda modelos entrenados en disco"""
        try:
            # Guardar modelos de dimensiones
            for dim, model in self.dimension_models.items():
                model_path = self.models_path / f"dimension_{dim}_model.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

            # Guardar modelos del Decálogo
            for point, model in self.decalogo_models.items():
                model_path = self.models_path / f"decalogo_{point}_model.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

            # Guardar modelo global
            if self.global_model:
                with open(self.models_path / "global_model.pkl", "wb") as f:
                    pickle.dump(self.global_model, f)

            # Guardar preprocesadores
            if self.feature_scaler:
                with open(self.models_path / "feature_scaler.pkl", "wb") as f:
                    pickle.dump(self.feature_scaler, f)

            if self.feature_selector:
                with open(self.models_path / "feature_selector.pkl", "wb") as f:
                    pickle.dump(self.feature_selector, f)

            # Guardar metadatos
            metadata = {
                "feature_importance": self.feature_importance,
                "model_performance": self.model_performance,
                "compliance_thresholds": self.compliance_thresholds,
                "dimension_weights": self.dimension_weights,
                "decalogo_weights": self.decalogo_weights,
                "model_config": self.model_config,
                "saved_at": datetime.now().isoformat(),
            }

            with open(self.models_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Models saved to {self.models_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save models: {str(e)}")
            return False

    def load_models(self) -> bool:
        """Carga modelos desde disco"""
        try:
            metadata_path = self.models_path / "metadata.json"
            if not metadata_path.exists():
                logger.info("No existing models found. Using default configuration.")
                return False

            # Cargar metadatos
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.feature_importance = metadata.get("feature_importance", {})
            self.model_performance = metadata.get("model_performance", {})

            # Cargar modelos de dimensiones
            for dim in self.dimension_weights.keys():
                model_path = self.models_path / f"dimension_{dim}_model.pkl"
                if model_path.exists():
                    with open(model_path, "rb") as f:
                        self.dimension_models[dim] = pickle.load(f)

            # Cargar modelos del Decálogo
            for point in self.decalogo_weights.keys():
                model_path = self.models_path / f"decalogo_{point}_model.pkl"
                if model_path.exists():
                    with open(model_path, "rb") as f:
                        self.decalogo_models[point] = pickle.load(f)

            # Cargar modelo global
            global_path = self.models_path / "global_model.pkl"
            if global_path.exists():
                with open(global_path, "rb") as f:
                    self.global_model = pickle.load(f)

            # Cargar preprocesadores
            scaler_path = self.models_path / "feature_scaler.pkl"
            if scaler_path.exists() and SKLEARN_AVAILABLE:
                with open(scaler_path, "rb") as f:
                    self.feature_scaler = pickle.load(f)

            selector_path = self.models_path / "feature_selector.pkl"
            if selector_path.exists() and SKLEARN_AVAILABLE:
                with open(selector_path, "rb") as f:
                    self.feature_selector = pickle.load(f)

# # #             logger.info(f"Models loaded from {self.models_path}")  # Module not found  # Module not found  # Module not found
            return True

        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            return False

    def get_model_status(self) -> Dict[str, Any]:
        """Devuelve el estado actual de los modelos"""
        return {
            "models_loaded": {
                "dimension_models": len(self.dimension_models),
                "decalogo_models": len(self.decalogo_models),
                "global_model": self.global_model is not None,
            },
            "performance_metrics": self.model_performance,
            "feature_importance": self.feature_importance,
            "sklearn_available": SKLEARN_AVAILABLE,
            "models_path": str(self.models_path),
        }
