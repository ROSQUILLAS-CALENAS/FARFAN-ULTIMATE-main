"""
Decálogo Evaluation Scoring System
Implements deterministic scoring with exact values, evidence quality multipliers,
dimension-level aggregation, and point-level composition for the Decálogo framework.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import logging
from decimal import Decimal, ROUND_HALF_UP

logger = logging.getLogger(__name__)

# Try to import centralized configuration
try:
    from config_loader import get_thresholds
    THRESHOLDS_AVAILABLE = True
except ImportError:
    THRESHOLDS_AVAILABLE = False
    logger.warning("Centralized thresholds not available, using hardcoded values")


@dataclass
class QuestionResponse:
    """Individual question response with evidence quality assessment"""
    question_id: str
    response: str  # "Sí", "Parcial", "No", "NI" 
    base_score: float
    evidence_completeness: float  # 0.0 to 1.0
    page_reference_quality: float  # 0.0 to 1.0
    final_score: float


@dataclass
class DimensionScore:
    """Score for a dimension (DE-1, DE-2, DE-3, DE-4)"""
    dimension_id: str
    question_responses: List[QuestionResponse]
    weighted_average: float
    total_questions: int


@dataclass
class PointScore:
    """Final score for a Decálogo point"""
    point_id: int
    dimension_scores: List[DimensionScore]
    final_score: float
    total_questions: int


class ScoringSystem:
    """
    Deterministic scoring system for Decálogo evaluation with exact values,
    evidence quality multipliers, dimension-level aggregation, and point-level composition.
    """
    
    def __init__(self, precision: int = 4):
        """
        Initialize scoring system with specified precision for rounding.
        Load configuration from centralized thresholds if available.
        
        Args:
            precision: Number of decimal places for consistent rounding
        """
        self.precision = precision
        self.rounding_context = Decimal(10) ** (-precision)
        
        # Load configuration or use defaults
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from centralized thresholds or use defaults."""
        if THRESHOLDS_AVAILABLE:
            try:
                config = get_thresholds()
                decalogo_config = config.decalogo_scoring
                evidence_config = config.evidence_multipliers
                
                # Load base scores
                self.BASE_SCORES = {
                    k: Decimal(str(v)) for k, v in decalogo_config.base_scores.items()
                }
                
                # Load dimension weights  
                self.DECALOGO_WEIGHTS = {
                    k: Decimal(str(v)) for k, v in decalogo_config.dimension_weights.items()
                }
                
                # Load multiplier ranges
                self.MIN_MULTIPLIER = Decimal(str(evidence_config.MIN_MULTIPLIER))
                self.MAX_MULTIPLIER = Decimal(str(evidence_config.MAX_MULTIPLIER))
                
                # Load compliance thresholds
                self.compliance_thresholds = decalogo_config.compliance_thresholds
                self.decalogo_point_thresholds = decalogo_config.decalogo_point_thresholds
                
                # Load evidence weights
                self.completeness_weight = Decimal(str(evidence_config.completeness_weight))
                self.reference_quality_weight = Decimal(str(evidence_config.reference_quality_weight))
                
                logger.info("Loaded Decálogo scoring configuration from centralized thresholds")
                
            except Exception as e:
                logger.warning(f"Failed to load centralized configuration: {e}, using defaults")
                self._load_default_configuration()
        else:
            self._load_default_configuration()
    
    def _load_default_configuration(self):
        """Load default hardcoded configuration."""
        # Exact scoring values
        self.BASE_SCORES = {
            "Sí": Decimal("1.0"),
            "Parcial": Decimal("0.5"),
            "No": Decimal("0.0"),
            "NI": Decimal("0.0")
        }
        
        # Predefined Decálogo weights for point-level composition
        self.DECALOGO_WEIGHTS = {
            "DE-1": Decimal("0.30"),  # Productos
            "DE-2": Decimal("0.25"),  # Diagnóstico
            "DE-3": Decimal("0.25"),  # Seguimiento
            "DE-4": Decimal("0.20")   # Evaluación
        }
        
        # Evidence quality multiplier ranges
        self.MIN_MULTIPLIER = Decimal("0.5")
        self.MAX_MULTIPLIER = Decimal("1.2")
        
        # Default evidence weights
        self.completeness_weight = Decimal("0.7")
        self.reference_quality_weight = Decimal("0.3")
        
        # Default compliance thresholds
        self.compliance_thresholds = {
            "DE1": {"CUMPLE": 0.75, "CUMPLE_PARCIAL": 0.50},
            "DE2": {"CUMPLE": 0.70, "CUMPLE_PARCIAL": 0.45},
            "DE3": {"CUMPLE": 0.80, "CUMPLE_PARCIAL": 0.55},
            "DE4": {"CUMPLE": 0.65, "CUMPLE_PARCIAL": 0.40},
        }
        
        self.decalogo_point_thresholds = {
            "P1": {"CUMPLE": 0.75, "CUMPLE_PARCIAL": 0.50},
            "P2": {"CUMPLE": 0.70, "CUMPLE_PARCIAL": 0.45},
            "P3": {"CUMPLE": 0.72, "CUMPLE_PARCIAL": 0.47},
            "P4": {"CUMPLE": 0.68, "CUMPLE_PARCIAL": 0.43},
            "P5": {"CUMPLE": 0.73, "CUMPLE_PARCIAL": 0.48},
            "P6": {"CUMPLE": 0.71, "CUMPLE_PARCIAL": 0.46},
            "P7": {"CUMPLE": 0.74, "CUMPLE_PARCIAL": 0.49},
            "P8": {"CUMPLE": 0.69, "CUMPLE_PARCIAL": 0.44},
            "P9": {"CUMPLE": 0.76, "CUMPLE_PARCIAL": 0.51},
            "P10": {"CUMPLE": 0.67, "CUMPLE_PARCIAL": 0.42}
        }
    

        
    def calculate_base_score(self, response: str) -> Decimal:
        """
        Calculate deterministic base score for a response.
        
        Args:
            response: Question response ("Sí", "Parcial", "No", "NI")
            
        Returns:
            Exact base score as Decimal
        """
        if response not in self.BASE_SCORES:
            logger.warning(f"Unknown response '{response}', treating as 'NI'")
            return self.BASE_SCORES["NI"]
        
        return self.BASE_SCORES[response]
    
    def calculate_evidence_multiplier(
        self, 
        evidence_completeness: float, 
        page_reference_quality: float
    ) -> Decimal:
        """
        Calculate evidence quality multiplier based on completeness and reference quality.
        
        Args:
            evidence_completeness: Evidence completeness score (0.0 to 1.0)
            page_reference_quality: Page reference quality score (0.0 to 1.0)
            
        Returns:
            Evidence quality multiplier (0.5 to 1.2)
        """
        # Convert to Decimal for precision
        completeness = Decimal(str(max(0.0, min(1.0, evidence_completeness))))
        reference_quality = Decimal(str(max(0.0, min(1.0, page_reference_quality))))
        
        # Combined evidence quality (weighted average)
        evidence_quality = (completeness * self.completeness_weight + 
                          reference_quality * self.reference_quality_weight)
        
        # Map to multiplier range [0.5, 1.2]
        multiplier_range = self.MAX_MULTIPLIER - self.MIN_MULTIPLIER
        multiplier = self.MIN_MULTIPLIER + (evidence_quality * multiplier_range)
        
        return self._round_decimal(multiplier)
    
    def calculate_final_score(
        self,
        response: str,
        evidence_completeness: float,
        page_reference_quality: float
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """
        Calculate final score with evidence quality adjustment.
        
        Args:
            response: Question response
            evidence_completeness: Evidence completeness score
            page_reference_quality: Page reference quality score
            
        Returns:
            Tuple of (base_score, multiplier, final_score)
        """
        base_score = self.calculate_base_score(response)
        multiplier = self.calculate_evidence_multiplier(
            evidence_completeness, page_reference_quality
        )
        
        final_score = self._round_decimal(base_score * multiplier)
        
        return base_score, multiplier, final_score
    
    def aggregate_dimension_score(
        self, 
        question_responses: List[QuestionResponse]
    ) -> Decimal:
        """
        Calculate dimension-level weighted average score.
        
        Args:
            question_responses: List of question responses for the dimension
            
        Returns:
            Weighted average score for the dimension
        """
        if not question_responses:
            return Decimal("0.0")
        
        # Calculate weighted average (all questions have equal weight)
        total_score = sum(Decimal(str(resp.final_score)) for resp in question_responses)
        dimension_score = total_score / Decimal(str(len(question_responses)))
        
        return self._round_decimal(dimension_score)
    
    def compose_point_score(self, dimension_scores: Dict[str, Decimal]) -> Decimal:
        """
        Combine dimension scores using predefined Decálogo weights.
        
        Args:
            dimension_scores: Dictionary mapping dimension IDs to scores
            
        Returns:
            Final point score
        """
        weighted_sum = Decimal("0.0")
        total_weight = Decimal("0.0")
        
        for dimension_id, score in dimension_scores.items():
            if dimension_id in self.DECALOGO_WEIGHTS:
                weight = self.DECALOGO_WEIGHTS[dimension_id]
                weighted_sum += Decimal(str(score)) * weight
                total_weight += weight
            else:
                logger.warning(f"Unknown dimension '{dimension_id}' in point composition")
        
        if total_weight == 0:
            return Decimal("0.0")
        
        point_score = weighted_sum / total_weight
        return self._round_decimal(point_score)
    
    def validate_question_count(
        self, 
        point_id: int, 
        total_questions: int, 
        expected_count: int = 47
    ) -> bool:
        """
        Validate that all expected questions are accounted for in the aggregation.
        
        Args:
            point_id: Decálogo point ID
            total_questions: Actual number of questions processed
            expected_count: Expected number of questions per point
            
        Returns:
            True if validation passes
        """
        if total_questions != expected_count:
            logger.error(
                f"Point {point_id}: Expected {expected_count} questions, "
                f"got {total_questions}"
            )
            return False
        
        return True
    
    def process_point_evaluation(
        self,
        point_id: int,
        evaluation_data: Dict[str, List[Dict[str, Any]]]
    ) -> PointScore:
        """
        Process complete evaluation for a Decálogo point.
        
        Args:
            point_id: Decálogo point ID (1-11)
            evaluation_data: Dictionary with dimension data
                Format: {
                    "DE-1": [{"question_id": "...", "response": "...", ...}],
                    "DE-2": [...],
                    ...
                }
                
        Returns:
            Complete point score with all dimension details
        """
        dimension_scores = []
        dimension_score_values = {}
        total_questions = 0
        
        # Process each dimension
        for dimension_id in ["DE-1", "DE-2", "DE-3", "DE-4"]:
            if dimension_id not in evaluation_data:
                logger.warning(f"Missing dimension {dimension_id} for point {point_id}")
                continue
            
            question_responses = []
            
            # Process each question in the dimension
            for question_data in evaluation_data[dimension_id]:
                response = question_data.get("response", "NI")
                evidence_completeness = question_data.get("evidence_completeness", 0.0)
                page_reference_quality = question_data.get("page_reference_quality", 0.0)
                
                base_score, multiplier, final_score = self.calculate_final_score(
                    response, evidence_completeness, page_reference_quality
                )
                
                question_response = QuestionResponse(
                    question_id=question_data.get("question_id", ""),
                    response=response,
                    base_score=float(base_score),
                    evidence_completeness=evidence_completeness,
                    page_reference_quality=page_reference_quality,
                    final_score=float(final_score)
                )
                
                question_responses.append(question_response)
            
            # Aggregate dimension score
            dimension_avg = self.aggregate_dimension_score(question_responses)
            
            dimension_score = DimensionScore(
                dimension_id=dimension_id,
                question_responses=question_responses,
                weighted_average=float(dimension_avg),
                total_questions=len(question_responses)
            )
            
            dimension_scores.append(dimension_score)
            dimension_score_values[dimension_id] = dimension_avg
            total_questions += len(question_responses)
        
        # Validate question count
        self.validate_question_count(point_id, total_questions)
        
        # Compose final point score
        final_score = self.compose_point_score(dimension_score_values)
        
        return PointScore(
            point_id=point_id,
            dimension_scores=dimension_scores,
            final_score=float(final_score),
            total_questions=total_questions
        )
    
    def _round_decimal(self, value: Decimal) -> Decimal:
        """Round decimal to specified precision using consistent rounding."""
        return value.quantize(self.rounding_context, rounding=ROUND_HALF_UP)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the scoring system configuration."""
        return {
            "base_scores": {k: float(v) for k, v in self.BASE_SCORES.items()},
            "decalogo_weights": {k: float(v) for k, v in self.DECALOGO_WEIGHTS.items()},
            "evidence_multiplier_range": [float(self.MIN_MULTIPLIER), float(self.MAX_MULTIPLIER)],
            "precision": self.precision,
            "deterministic": True,
            "validation_enabled": True
        }