"""
Calibration Artifact Data Models

Defines the standardized structure for calibration reports across pipeline stages.
"""

# # # from dataclasses import dataclass, asdict, field  # Module not found  # Module not found  # Module not found
# # # from typing import Dict, Any, Optional, List, Union  # Module not found  # Module not found  # Module not found
# # # from datetime import datetime  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
import json


@dataclass
class CalibrationArtifact:
    """Base class for all calibration artifacts with common fields."""
    
    # Core calibration metrics
    calibration_quality_score: float
    coverage_percentage: float
    decisions_made: int
    quality_gate_status: str  # "PASS", "FAIL", "WARNING"
    
    # Metadata sections
    timestamp: str
    stage_version: str
    calibration_parameters: Dict[str, Any]
    
    # Optional quality gate thresholds
    quality_thresholds: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert artifact to dictionary for JSON serialization."""
        return asdict(self)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save artifact to JSON file."""
        filepath_obj: Path = Path(filepath)
        filepath_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath_obj, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationArtifact':
# # #         """Create artifact from dictionary."""  # Module not found  # Module not found  # Module not found
        return cls(**data)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'CalibrationArtifact':
# # #         """Load artifact from JSON file."""  # Module not found  # Module not found  # Module not found
        with open(filepath, 'r') as f:
            data: Dict[str, Any] = json.load(f)
        return cls.from_dict(data)
    
    def check_quality_gates(self) -> bool:
        """Check if all quality gates pass."""
        return self.quality_gate_status == "PASS"


@dataclass
class RetrievalCalibrationArtifact(CalibrationArtifact):
    """Calibration artifact for retrieval stage with temperature stability metrics."""
    
    # Stage-specific metrics
    temperature_stability: float = 0.0
    entropy_calibration_score: float = 0.0
    retrieval_precision: float = 0.0
    retrieval_recall: float = 0.0
    fusion_coherence: float = 0.0
    
    # Stage parameters
    temperature_parameter: float = 1.0
    fusion_weights: Optional[List[float]] = None
    retrieval_k_values: Optional[List[int]] = None
    
    # Additional retrieval metrics
    retrieval_latency_ms: float = 0.0
    index_coherence_score: float = 0.0
    cross_encoder_agreement: float = 0.0
    
    def __post_init__(self) -> None:
        if self.fusion_weights is None:
            self.fusion_weights = [0.33, 0.33, 0.34]
        if self.retrieval_k_values is None:
            self.retrieval_k_values = [10, 50, 100]


@dataclass
class ConfidenceCalibrationArtifact(CalibrationArtifact):
    """Calibration artifact for confidence estimation stage with interval coverage."""
    
    # Stage-specific metrics
    interval_coverage: float = 0.0
    calibration_error: float = 0.0
    sharpness_score: float = 0.0
    reliability_score: float = 0.0
    prediction_efficiency: float = 0.0
    
    # Conformal prediction metrics
    adaptive_coverage_gap: float = 0.0
    prediction_set_size_avg: float = 0.0
    nonconformity_score_stats: Optional[Dict[str, float]] = None
    
    # Stage parameters
    confidence_level: float = 0.9
    alpha_significance: float = 0.1
    adaptive_quantile_level: float = 0.9
    
    # Distribution analysis
    distribution_shift_detected: bool = False
    ks_test_p_value: float = 1.0
    bootstrap_intervals: Optional[Dict[str, List[float]]] = None
    
    def __post_init__(self) -> None:
        if self.nonconformity_score_stats is None:
            self.nonconformity_score_stats = {}
        if self.bootstrap_intervals is None:
            self.bootstrap_intervals = {}


@dataclass
class AggregationCalibrationArtifact(CalibrationArtifact):
    """Calibration artifact for aggregation stage with convergence metrics."""
    
    # Stage-specific metrics
    convergence_rate: float = 0.0
    aggregation_stability: float = 0.0
    consensus_score: float = 0.0
    uncertainty_quantification_quality: float = 0.0
    
    # Mathematical rigor metrics  
    spectral_gap: float = 0.0
    eigenvalue_stability: float = 0.0
    manifold_coherence: float = 0.0
    
    # Stage parameters
    aggregation_method: str = "consensus"
    convergence_tolerance: float = 1e-6
    max_iterations: int = 1000
    regularization_strength: float = 0.01
    
    # Uncertainty metrics
    epistemic_uncertainty: float = 0.0
    aleatoric_uncertainty: float = 0.0
    total_uncertainty: float = 0.0
    calibration_curve_auc: float = 0.0