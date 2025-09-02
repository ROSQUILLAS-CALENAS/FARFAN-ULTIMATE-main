"""
<<<<<<< HEAD
Centralized Configuration Loader for Threshold Management

Provides typed access to threshold values from thresholds.json with validation,
error handling, and caching for consistent parameter usage across the pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

T = TypeVar('T')


class ConfigError(Exception):
    """Base exception for configuration-related errors."""
    pass


class ConfigValidationError(ConfigError):
    """Exception raised when configuration validation fails."""
    pass


class ConfigNotFoundError(ConfigError):
    """Exception raised when configuration file is not found."""
    pass


@dataclass
class ScoringBounds:
    """Scoring bounds configuration with validation."""
    min_score: float
    max_score: float
    base_score_range: tuple[float, float]
    evidence_quality_bound: tuple[float, float]
    
    def __post_init__(self):
        if self.min_score >= self.max_score:
            raise ConfigValidationError("min_score must be less than max_score")
        if len(self.base_score_range) != 2:
            raise ConfigValidationError("base_score_range must be a tuple of 2 values")
        if self.base_score_range[0] >= self.base_score_range[1]:
            raise ConfigValidationError("base_score_range must be ordered (min, max)")


@dataclass
class EvidenceMultipliers:
    """Evidence quality multiplier configuration."""
    min_multiplier: float
    max_multiplier: float
    completeness_weight: float
    reference_quality_weight: float
    
    def __post_init__(self):
        if self.min_multiplier >= self.max_multiplier:
            raise ConfigValidationError("min_multiplier must be less than max_multiplier")
        if not (0.0 <= self.completeness_weight <= 1.0):
            raise ConfigValidationError("completeness_weight must be in [0, 1]")
        if not (0.0 <= self.reference_quality_weight <= 1.0):
            raise ConfigValidationError("reference_quality_weight must be in [0, 1]")
        if abs(self.completeness_weight + self.reference_quality_weight - 1.0) > 1e-6:
            raise ConfigValidationError("completeness_weight + reference_quality_weight must equal 1.0")


@dataclass
class TemperatureRange:
    """Temperature range configuration."""
    min: float
    default: float
    max: float
    
    def __post_init__(self):
        if not (self.min <= self.default <= self.max):
            raise ConfigValidationError("Temperature values must satisfy min <= default <= max")
    
    def clamp(self, value: float) -> float:
        """Clamp a temperature value to the valid range."""
        return max(self.min, min(self.max, value))


@dataclass
class TemperatureRanges:
    """All temperature range configurations."""
    retrieval: TemperatureRange
    entropy_calibration: TemperatureRange
    classification: TemperatureRange
    conformal_risk: TemperatureRange
    
    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, float]]) -> 'TemperatureRanges':
        """Create TemperatureRanges from dictionary data."""
        ranges = {}
        for name, range_data in data.items():
            if 'range' in range_data:
                # Handle range format like [min, max] with default
                range_vals = range_data['range']
                ranges[name] = TemperatureRange(
                    min=range_vals[0],
                    default=range_data.get('default', 1.0),
                    max=range_vals[1]
                )
            else:
                # Handle min/default/max format
                ranges[name] = TemperatureRange(
                    min=range_data.get('min', 0.1),
                    default=range_data.get('default', 1.0),
                    max=range_data.get('max', 5.0)
                )
        return cls(**ranges)


@dataclass
class FusionWeights:
    """Fusion weight configurations."""
    hybrid_retrieval: Dict[str, float]
    dimension_aggregation: Dict[str, float]
    sparse_dense_projection: Dict[str, float]
    
    def __post_init__(self):
        # Validate that weights sum to 1.0 (with tolerance)
        for name, weights in [
            ("hybrid_retrieval", self.hybrid_retrieval),
            ("dimension_aggregation", self.dimension_aggregation),
            ("sparse_dense_projection", self.sparse_dense_projection)
        ]:
            weight_sum = sum(weights.values())
            if abs(weight_sum - 1.0) > 1e-6:
                raise ConfigValidationError(f"{name} weights must sum to 1.0, got {weight_sum}")


@dataclass
class QualityThresholds:
    """Quality assessment threshold configurations."""
    coverage_tolerance: float
    confidence_levels: Dict[str, float]
    evidence_quality: Dict[str, float]
    risk_control: Dict[str, float]
    
    def __post_init__(self):
        # Validate confidence levels are in [0, 1]
        for level, value in self.confidence_levels.items():
            if not (0.0 <= value <= 1.0):
                raise ConfigValidationError(f"confidence_levels.{level} must be in [0, 1], got {value}")


@dataclass
class PipelineStageThresholds:
    """Pipeline stage specific thresholds."""
    ingestion: Dict[str, float]
    retrieval: Dict[str, Union[float, int]]
    scoring: Dict[str, Union[float, int, str]]
    risk_assessment: Dict[str, Union[float, int]]


@dataclass
class MonitoringBounds:
    """System monitoring threshold bounds."""
    cpu_usage: Dict[str, float]
    memory_usage: Dict[str, float]
    error_rates: Dict[str, float]
    response_times: Dict[str, float]


@dataclass
class AdaptiveControl:
    """Adaptive control parameters."""
    learning_rates: Dict[str, float]
    confidence_thresholds: Dict[str, float]
    update_intervals: Dict[str, float]


@dataclass
class ThresholdConfiguration:
    """Complete threshold configuration with all sections."""
    version: str
    description: str
    scoring_bounds: ScoringBounds
    evidence_quality_multipliers: EvidenceMultipliers
    temperature_ranges: TemperatureRanges
    fusion_weights: FusionWeights
    quality_assessment_thresholds: QualityThresholds
    pipeline_stage_thresholds: PipelineStageThresholds
    monitoring_bounds: MonitoringBounds
    adaptive_control: AdaptiveControl
    _source_hash: str = field(default="", init=False)


class ConfigurationLoader:
    """
    Centralized configuration loader with validation and caching.
    
    Provides typed access to threshold values with proper error handling
    for missing or invalid parameters.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to thresholds.json. If None, searches standard locations.
        """
        self.config_path = self._find_config_path(config_path)
        self._config: Optional[ThresholdConfiguration] = None
        self._config_hash: Optional[str] = None
        self._load_config()
    
    def _find_config_path(self, config_path: Optional[Union[str, Path]]) -> Path:
        """Find the configuration file in standard locations."""
        if config_path is not None:
            path = Path(config_path)
            if path.exists():
                return path
            raise ConfigNotFoundError(f"Configuration file not found: {path}")
        
        # Search standard locations
        search_paths = [
            Path.cwd() / "thresholds.json",
            Path(__file__).parent / "thresholds.json",
            Path.cwd() / "config" / "thresholds.json",
            Path.home() / ".config" / "egw_query_expansion" / "thresholds.json"
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        raise ConfigNotFoundError(
            f"Configuration file 'thresholds.json' not found in any of: "
            f"{[str(p) for p in search_paths]}"
        )
    
    def _load_config(self) -> None:
        """Load and validate configuration from file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Calculate hash for change detection
            config_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            
            # Skip reload if unchanged
            if self._config_hash == config_hash and self._config is not None:
                return
            
            raw_config = json.loads(content)
            self._config = self._parse_configuration(raw_config)
            self._config._source_hash = config_hash
            self._config_hash = config_hash
            
        except json.JSONDecodeError as e:
            raise ConfigValidationError(f"Invalid JSON in configuration file: {e}") from e
        except Exception as e:
            raise ConfigError(f"Failed to load configuration: {e}") from e
    
    def _parse_configuration(self, data: Dict[str, Any]) -> ThresholdConfiguration:
        """Parse raw configuration data into typed objects."""
        try:
            # Parse scoring bounds
            scoring_bounds_data = data["scoring_bounds"]
            scoring_bounds = ScoringBounds(
                min_score=scoring_bounds_data["min_score"],
                max_score=scoring_bounds_data["max_score"],
                base_score_range=tuple(scoring_bounds_data["base_score_range"]),
                evidence_quality_bound=tuple(scoring_bounds_data["evidence_quality_bound"])
            )
            
            # Parse evidence multipliers
            multipliers_data = data["evidence_quality_multipliers"]
            evidence_multipliers = EvidenceMultipliers(
                min_multiplier=multipliers_data["min_multiplier"],
                max_multiplier=multipliers_data["max_multiplier"],
                completeness_weight=multipliers_data["completeness_weight"],
                reference_quality_weight=multipliers_data["reference_quality_weight"]
            )
            
            # Parse temperature ranges
            temp_ranges = TemperatureRanges.from_dict(data["temperature_ranges"])
            
            # Parse fusion weights
            fusion_weights = FusionWeights(
                hybrid_retrieval=data["fusion_weights"]["hybrid_retrieval"],
                dimension_aggregation=data["fusion_weights"]["dimension_aggregation"],
                sparse_dense_projection=data["fusion_weights"]["sparse_dense_projection"]
            )
            
            # Parse quality thresholds
            quality_thresholds = QualityThresholds(
                coverage_tolerance=data["quality_assessment_thresholds"]["coverage_tolerance"],
                confidence_levels=data["quality_assessment_thresholds"]["confidence_levels"],
                evidence_quality=data["quality_assessment_thresholds"]["evidence_quality"],
                risk_control=data["quality_assessment_thresholds"]["risk_control"]
            )
            
            # Parse pipeline stage thresholds
            pipeline_thresholds = PipelineStageThresholds(
                ingestion=data["pipeline_stage_thresholds"]["ingestion"],
                retrieval=data["pipeline_stage_thresholds"]["retrieval"],
                scoring=data["pipeline_stage_thresholds"]["scoring"],
                risk_assessment=data["pipeline_stage_thresholds"]["risk_assessment"]
            )
            
            # Parse monitoring bounds
            monitoring_bounds = MonitoringBounds(
                cpu_usage=data["monitoring_bounds"]["cpu_usage"],
                memory_usage=data["monitoring_bounds"]["memory_usage"],
                error_rates=data["monitoring_bounds"]["error_rates"],
                response_times=data["monitoring_bounds"]["response_times"]
            )
            
            # Parse adaptive control
            adaptive_control = AdaptiveControl(
                learning_rates=data["adaptive_control"]["learning_rates"],
                confidence_thresholds=data["adaptive_control"]["confidence_thresholds"],
                update_intervals=data["adaptive_control"]["update_intervals"]
            )
            
            return ThresholdConfiguration(
                version=data["version"],
                description=data["description"],
                scoring_bounds=scoring_bounds,
                evidence_quality_multipliers=evidence_multipliers,
                temperature_ranges=temp_ranges,
                fusion_weights=fusion_weights,
                quality_assessment_thresholds=quality_thresholds,
                pipeline_stage_thresholds=pipeline_thresholds,
                monitoring_bounds=monitoring_bounds,
                adaptive_control=adaptive_control
            )
            
        except KeyError as e:
            raise ConfigValidationError(f"Missing required configuration key: {e}") from e
        except Exception as e:
            raise ConfigValidationError(f"Configuration validation failed: {e}") from e
    
    def get_config(self) -> ThresholdConfiguration:
        """Get the complete configuration object."""
        if self._config is None:
            self._load_config()
        return self._config
    
    def reload_if_changed(self) -> bool:
        """Reload configuration if file has changed. Returns True if reloaded."""
        old_hash = self._config_hash
        self._load_config()
        return old_hash != self._config_hash
    
    def get_scoring_bounds(self) -> ScoringBounds:
        """Get scoring bounds configuration."""
        return self.get_config().scoring_bounds
    
    def get_evidence_multipliers(self) -> EvidenceMultipliers:
        """Get evidence quality multiplier configuration."""
        return self.get_config().evidence_quality_multipliers
    
    def get_temperature_range(self, component: str) -> TemperatureRange:
        """
        Get temperature range for a specific component.
        
        Args:
            component: One of 'retrieval', 'entropy_calibration', 'classification', 'conformal_risk'
            
        Returns:
            TemperatureRange object
            
        Raises:
            ConfigError: If component is not found
        """
        temp_ranges = self.get_config().temperature_ranges
        if not hasattr(temp_ranges, component):
            raise ConfigError(f"Unknown temperature component: {component}")
        return getattr(temp_ranges, component)
    
    def get_fusion_weights(self, category: str) -> Dict[str, float]:
        """
        Get fusion weights for a specific category.
        
        Args:
            category: One of 'hybrid_retrieval', 'dimension_aggregation', 'sparse_dense_projection'
            
        Returns:
            Dictionary of weights
        """
        fusion_weights = self.get_config().fusion_weights
        if not hasattr(fusion_weights, category):
            raise ConfigError(f"Unknown fusion weight category: {category}")
        return getattr(fusion_weights, category)
    
    def get_quality_threshold(self, category: str, key: Optional[str] = None) -> Union[float, Dict[str, float]]:
        """
        Get quality assessment thresholds.
        
        Args:
            category: One of 'confidence_levels', 'evidence_quality', 'risk_control'
            key: Optional specific key within category
            
        Returns:
            Float value if key is specified, otherwise dictionary of values
        """
        quality_thresholds = self.get_config().quality_assessment_thresholds
        
        if category == 'coverage_tolerance':
            return quality_thresholds.coverage_tolerance
        
        category_data = getattr(quality_thresholds, category, None)
        if category_data is None:
            raise ConfigError(f"Unknown quality threshold category: {category}")
        
        if key is not None:
            if key not in category_data:
                raise ConfigError(f"Unknown key '{key}' in category '{category}'")
            return category_data[key]
        
        return category_data
    
    def get_pipeline_threshold(self, stage: str, key: Optional[str] = None) -> Union[float, int, str, Dict[str, Any]]:
        """
        Get pipeline stage threshold values.
        
        Args:
            stage: One of 'ingestion', 'retrieval', 'scoring', 'risk_assessment'
            key: Optional specific key within stage
            
        Returns:
            Value if key is specified, otherwise dictionary of values
        """
        pipeline_thresholds = self.get_config().pipeline_stage_thresholds
        stage_data = getattr(pipeline_thresholds, stage, None)
        
        if stage_data is None:
            raise ConfigError(f"Unknown pipeline stage: {stage}")
        
        if key is not None:
            if key not in stage_data:
                raise ConfigError(f"Unknown key '{key}' in stage '{stage}'")
            return stage_data[key]
        
        return stage_data
    
    def get_monitoring_bound(self, metric: str, threshold_type: str) -> float:
        """
        Get monitoring threshold bounds.
        
        Args:
            metric: One of 'cpu_usage', 'memory_usage', 'error_rates', 'response_times'
            threshold_type: One of 'critical', 'optimal', 'warning'
            
        Returns:
            Threshold value
        """
        monitoring_bounds = self.get_config().monitoring_bounds
        metric_data = getattr(monitoring_bounds, metric, None)
        
        if metric_data is None:
            raise ConfigError(f"Unknown monitoring metric: {metric}")
        
        if threshold_type not in metric_data:
            raise ConfigError(f"Unknown threshold type '{threshold_type}' for metric '{metric}'")
        
        return metric_data[threshold_type]
    
    def validate_score(self, score: float) -> bool:
        """Validate that a score is within configured bounds."""
        bounds = self.get_scoring_bounds()
        return bounds.min_score <= score <= bounds.max_score
    
    def clamp_score(self, score: float) -> float:
        """Clamp a score to configured bounds."""
        bounds = self.get_scoring_bounds()
        return max(bounds.min_score, min(bounds.max_score, score))
    
    def validate_temperature(self, temperature: float, component: str) -> bool:
        """Validate that temperature is within configured range for component."""
        temp_range = self.get_temperature_range(component)
        return temp_range.min <= temperature <= temp_range.max
    
    def clamp_temperature(self, temperature: float, component: str) -> float:
        """Clamp temperature to configured range for component."""
        temp_range = self.get_temperature_range(component)
        return temp_range.clamp(temperature)


# Singleton instance for global access
_config_loader: Optional[ConfigurationLoader] = None


def get_config_loader(config_path: Optional[Union[str, Path]] = None) -> ConfigurationLoader:
    """
    Get the global configuration loader instance.
    
    Args:
        config_path: Path to configuration file (only used on first call)
        
    Returns:
        ConfigurationLoader instance
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigurationLoader(config_path)
    return _config_loader


# Convenience functions for common access patterns
def get_scoring_bounds() -> ScoringBounds:
    """Get scoring bounds from global configuration."""
    return get_config_loader().get_scoring_bounds()


def get_evidence_multipliers() -> EvidenceMultipliers:
    """Get evidence multipliers from global configuration."""
    return get_config_loader().get_evidence_multipliers()


def get_fusion_weights(category: str) -> Dict[str, float]:
    """Get fusion weights for category from global configuration."""
    return get_config_loader().get_fusion_weights(category)


def get_temperature_range(component: str) -> TemperatureRange:
    """Get temperature range for component from global configuration."""
    return get_config_loader().get_temperature_range(component)


def validate_score(score: float) -> bool:
    """Validate score against global configuration bounds."""
    return get_config_loader().validate_score(score)


def clamp_score(score: float) -> float:
    """Clamp score to global configuration bounds."""
    return get_config_loader().clamp_score(score)


def validate_temperature(temperature: float, component: str) -> bool:
    """Validate temperature against global configuration bounds."""
    return get_config_loader().validate_temperature(temperature, component)


def clamp_temperature(temperature: float, component: str) -> float:
    """Clamp temperature to global configuration bounds."""
    return get_config_loader().clamp_temperature(temperature, component)
=======
Configuration Loader Utility for Centralized Thresholds

Provides typed access to threshold configuration parameters with:
- Schema validation on startup
- Type safety and error handling  
- Caching for performance
- Environment variable overrides
- Fallback values for missing parameters
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import jsonschema
    from jsonschema import ValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    ValidationError = ValueError  # Fallback

logger = logging.getLogger(__name__)


@dataclass
class TemperatureConfig:
    """Temperature bounds and defaults."""
    min_temperature: float = 0.5
    max_temperature: float = 2.0
    default_temperature: float = 1.0
    classification_temperature: float = 1.0
    entropy_calibration_temperature: float = 1.0


@dataclass
class ScoringBoundsConfig:
    """Score bounds and ranges."""
    min_score: float = 0.0
    max_score: float = 1.2
    default_score: float = 0.5
    neutral_score: float = 0.5
    score_tolerance: float = 0.01


@dataclass
class FusionWeightsConfig:
    """Multi-modal retrieval fusion weights."""
    lexical: float = 0.3
    vector: float = 0.4
    late_interaction: float = 0.2
    rrf: float = 0.1
    bm25_weight: float = 0.3
    dense_weight: float = 0.4
    colbert_weight: float = 0.2
    hybrid_weight: float = 0.1


@dataclass
class EvidenceMultipliersConfig:
    """Evidence quality multiplier bounds."""
    MIN_MULTIPLIER: float = 0.5
    MAX_MULTIPLIER: float = 1.2
    high_quality_multiplier: float = 1.0
    medium_quality_multiplier: float = 0.7
    low_quality_multiplier: float = 0.3
    completeness_weight: float = 0.7
    reference_quality_weight: float = 0.3


@dataclass
class QualityThresholdsConfig:
    """Quality assessment thresholds."""
    coverage_tolerance: float = 0.01
    min_coverage_rate: float = 0.90
    confidence_level: float = 0.95
    min_confidence_threshold: float = 0.50
    high_quality_threshold: float = 0.75
    medium_quality_threshold: float = 0.50
    low_quality_threshold: float = 0.25
    evidence_completeness_threshold: float = 0.5
    page_reference_quality_threshold: float = 0.5


@dataclass
class ConformalPredictionConfig:
    """Conformal risk control parameters."""
    alpha: float = 0.1
    lambda_reg: float = 0.0
    calibration_ratio: float = 0.5
    validation_size: int = 1000
    test_ratio: float = 0.2
    distribution_shift_bound: float = 0.1
    recalibration_threshold: float = 0.05
    confidence_level: float = 0.95
    min_calibration_size: int = 100
    max_set_size_ratio: float = 0.8
    coverage_tolerance: float = 0.01
    bootstrap_samples: int = 1000
    adaptive_quantile_clt_threshold: float = 0.05
    hoeffding_confidence: float = 0.95
    clt_confidence: float = 0.95


@dataclass
class RetrievalThresholdsConfig:
    """Retrieval and ranking thresholds."""
    rrf_k_parameter: int = 60
    top_k_default: int = 10
    min_similarity_threshold: float = 0.1
    max_similarity_threshold: float = 1.0
    score_variance_threshold: float = 0.1
    rank_correlation_threshold: float = 0.3
    candidate_filter_threshold: float = 0.2
    normalization_epsilon: float = 1e-8


@dataclass
class DecalogoScoringConfig:
    """Decálogo evaluation system configuration."""
    base_scores: Dict[str, float] = field(default_factory=lambda: {
        "Sí": 1.0, "Parcial": 0.5, "No": 0.0, "NI": 0.0
    })
    dimension_weights: Dict[str, float] = field(default_factory=lambda: {
        "DE-1": 0.30, "DE-2": 0.25, "DE-3": 0.25, "DE-4": 0.20
    })
    compliance_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    decalogo_point_weights: Dict[str, float] = field(default_factory=dict)
    decalogo_point_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    rounding_precision: int = 4


@dataclass
class AdaptiveScoringConfig:
    """Adaptive scoring engine configuration."""
    model_parameters: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    })
    correction_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "severe_deviation": 0.3,
        "moderate_deviation": 0.5,
        "minor_deviation": 0.7,
        "acceptable_range": 0.8
    })
    dnp_baseline_standards: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DNPAlignmentConfig:
    """DNP alignment and compliance configuration."""
    default_alignment_scores: Dict[str, float] = field(default_factory=lambda: {
        "gpr_alignment": 0.5,
        "sgp_compliance": 0.5,
        "sinergia_integration": 0.5,
        "competencias_score": 0.5,
        "territorial_coherence": 0.5,
        "overall_alignment": 0.5
    })
    human_rights_standards: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StatisticalBoundsConfig:
    """Statistical bounds and validation parameters."""
    hoeffding_confidence: float = 0.95
    clt_confidence: float = 0.95
    bootstrap_confidence: float = 0.95
    conservative_bound_min: float = 0.0
    conservative_bound_max: float = 1.0
    significance_level: float = 0.05
    p_value_threshold: float = 0.05
    statistical_power: float = 0.8


@dataclass
class AggregationThresholdsConfig:
    """G-aggregation and meso-level configuration."""
    aggregation_weights: Dict[str, float] = field(default_factory=lambda: {
        "dimension_institutional": 0.30,
        "dimension_social": 0.25,
        "dimension_economic": 0.25,
        "dimension_environmental": 0.20
    })
    consolidation_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "excellent": 0.85,
        "good": 0.70,
        "acceptable": 0.55,
        "poor": 0.40,
        "critical": 0.25
    })
    meso_aggregation_bounds: Dict[str, Any] = field(default_factory=lambda: {
        "min_aggregation_size": 10,
        "max_aggregation_size": 1000,
        "default_aggregation_size": 100,
        "convergence_tolerance": 0.001,
        "max_iterations": 1000
    })


@dataclass
class ValidationConfig:
    """Validation and error handling parameters."""
    timeout_seconds: int = 300
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    health_check_interval: int = 60
    error_rate_threshold: float = 0.05
    warning_rate_threshold: float = 0.10
    memory_limit_mb: int = 1024
    cpu_limit_percent: float = 80.0


@dataclass
class ThresholdsConfig:
    """Complete thresholds configuration container."""
    version: str = "1.0.0"
    last_updated: str = "2024-01-01T00:00:00Z"
    
    temperature: TemperatureConfig = field(default_factory=TemperatureConfig)
    scoring_bounds: ScoringBoundsConfig = field(default_factory=ScoringBoundsConfig)
    fusion_weights: FusionWeightsConfig = field(default_factory=FusionWeightsConfig)
    evidence_multipliers: EvidenceMultipliersConfig = field(default_factory=EvidenceMultipliersConfig)
    quality_thresholds: QualityThresholdsConfig = field(default_factory=QualityThresholdsConfig)
    conformal_prediction: ConformalPredictionConfig = field(default_factory=ConformalPredictionConfig)
    retrieval_thresholds: RetrievalThresholdsConfig = field(default_factory=RetrievalThresholdsConfig)
    decalogo_scoring: DecalogoScoringConfig = field(default_factory=DecalogoScoringConfig)
    adaptive_scoring: AdaptiveScoringConfig = field(default_factory=AdaptiveScoringConfig)
    dnp_alignment: DNPAlignmentConfig = field(default_factory=DNPAlignmentConfig)
    statistical_bounds: StatisticalBoundsConfig = field(default_factory=StatisticalBoundsConfig)
    aggregation_thresholds: AggregationThresholdsConfig = field(default_factory=AggregationThresholdsConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


class ConfigLoader:
    """
    Configuration loader with schema validation and caching.
    
    Features:
    - JSON schema validation
    - Environment variable overrides
    - Configuration caching
    - Type-safe parameter access
    - Error handling with fallbacks
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration loader."""
        self.config_path = Path(config_path or "thresholds.json")
        self._config: Optional[ThresholdsConfig] = None
        self._raw_config: Optional[Dict[str, Any]] = None
        self._schema: Optional[Dict[str, Any]] = None
        
    def _load_schema(self) -> Dict[str, Any]:
        """Load JSON schema for validation."""
        schema = {
            "$schema": "https://json-schema.org/draft/2019-09/schema",
            "type": "object",
            "required": ["version", "temperature", "scoring_bounds", "fusion_weights"],
            "properties": {
                "version": {"type": "string"},
                "last_updated": {"type": "string"},
                "temperature": {
                    "type": "object",
                    "properties": {
                        "min_temperature": {"type": "number", "minimum": 0.1, "maximum": 5.0},
                        "max_temperature": {"type": "number", "minimum": 0.1, "maximum": 5.0},
                        "default_temperature": {"type": "number", "minimum": 0.1, "maximum": 5.0}
                    }
                },
                "scoring_bounds": {
                    "type": "object",
                    "properties": {
                        "min_score": {"type": "number", "minimum": 0.0},
                        "max_score": {"type": "number", "maximum": 2.0},
                        "score_tolerance": {"type": "number", "minimum": 0.0, "maximum": 0.1}
                    }
                },
                "fusion_weights": {
                    "type": "object",
                    "properties": {
                        "lexical": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "vector": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "late_interaction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rrf": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                    }
                },
                "evidence_multipliers": {
                    "type": "object",
                    "properties": {
                        "MIN_MULTIPLIER": {"type": "number", "minimum": 0.1, "maximum": 1.0},
                        "MAX_MULTIPLIER": {"type": "number", "minimum": 1.0, "maximum": 2.0}
                    }
                },
                "conformal_prediction": {
                    "type": "object",
                    "properties": {
                        "alpha": {"type": "number", "minimum": 0.01, "maximum": 0.5},
                        "calibration_ratio": {"type": "number", "minimum": 0.1, "maximum": 0.9},
                        "confidence_level": {"type": "number", "minimum": 0.5, "maximum": 0.999}
                    }
                }
            }
        }
        return schema
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        # Common environment variable mappings
        env_mappings = {
            "EGW_TEMPERATURE": ("temperature", "default_temperature"),
            "EGW_ALPHA": ("conformal_prediction", "alpha"),
            "EGW_CONFIDENCE": ("quality_thresholds", "confidence_level"),
            "EGW_MIN_MULTIPLIER": ("evidence_multipliers", "MIN_MULTIPLIER"),
            "EGW_MAX_MULTIPLIER": ("evidence_multipliers", "MAX_MULTIPLIER"),
            "EGW_RRF_K": ("retrieval_thresholds", "rrf_k_parameter"),
            "EGW_TOP_K": ("retrieval_thresholds", "top_k_default"),
        }
        
        for env_var, (section, key) in env_mappings.items():
            if env_var in os.environ:
                try:
                    value = float(os.environ[env_var])
                    if section not in config:
                        config[section] = {}
                    config[section][key] = value
                    logger.info(f"Applied environment override: {env_var} = {value}")
                except ValueError as e:
                    logger.warning(f"Invalid environment variable {env_var}: {e}")
        
        return config
    
    def _convert_to_dataclass(self, config_dict: Dict[str, Any]) -> ThresholdsConfig:
        """Convert configuration dictionary to typed dataclass."""
        try:
            # Extract and convert each section
            temperature = TemperatureConfig(**config_dict.get("temperature", {}))
            scoring_bounds = ScoringBoundsConfig(**config_dict.get("scoring_bounds", {}))
            fusion_weights = FusionWeightsConfig(**config_dict.get("fusion_weights", {}))
            evidence_multipliers = EvidenceMultipliersConfig(**config_dict.get("evidence_multipliers", {}))
            quality_thresholds = QualityThresholdsConfig(**config_dict.get("quality_thresholds", {}))
            conformal_prediction = ConformalPredictionConfig(**config_dict.get("conformal_prediction", {}))
            retrieval_thresholds = RetrievalThresholdsConfig(**config_dict.get("retrieval_thresholds", {}))
            decalogo_scoring = DecalogoScoringConfig(**config_dict.get("decalogo_scoring", {}))
            adaptive_scoring = AdaptiveScoringConfig(**config_dict.get("adaptive_scoring", {}))
            dnp_alignment = DNPAlignmentConfig(**config_dict.get("dnp_alignment", {}))
            statistical_bounds = StatisticalBoundsConfig(**config_dict.get("statistical_bounds", {}))
            aggregation_thresholds = AggregationThresholdsConfig(**config_dict.get("aggregation_thresholds", {}))
            validation = ValidationConfig(**config_dict.get("validation", {}))
            
            return ThresholdsConfig(
                version=config_dict.get("version", "1.0.0"),
                last_updated=config_dict.get("last_updated", "2024-01-01T00:00:00Z"),
                temperature=temperature,
                scoring_bounds=scoring_bounds,
                fusion_weights=fusion_weights,
                evidence_multipliers=evidence_multipliers,
                quality_thresholds=quality_thresholds,
                conformal_prediction=conformal_prediction,
                retrieval_thresholds=retrieval_thresholds,
                decalogo_scoring=decalogo_scoring,
                adaptive_scoring=adaptive_scoring,
                dnp_alignment=dnp_alignment,
                statistical_bounds=statistical_bounds,
                aggregation_thresholds=aggregation_thresholds,
                validation=validation
            )
        except Exception as e:
            logger.error(f"Failed to convert configuration to dataclass: {e}")
            # Return default configuration
            return ThresholdsConfig()
    
    def load_config(self, validate_schema: bool = True) -> ThresholdsConfig:
        """
        Load and validate configuration from file.
        
        Args:
            validate_schema: Whether to validate against JSON schema
            
        Returns:
            Validated configuration object
            
        Raises:
            FileNotFoundError: If configuration file not found
            ValidationError: If configuration invalid
            ValueError: If configuration malformed
        """
        if self._config is not None:
            return self._config
        
        try:
            # Load configuration file
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                raw_config = json.load(f)
            
            self._raw_config = raw_config
            
            # Apply environment variable overrides
            raw_config = self._apply_env_overrides(raw_config)
            
            # Validate schema if requested
            if validate_schema and JSONSCHEMA_AVAILABLE:
                try:
                    self._schema = self._load_schema()
                    jsonschema.validate(raw_config, self._schema)
                    logger.info("Configuration schema validation passed")
                except ValidationError as e:
                    logger.error(f"Configuration validation failed: {e.message}")
                    raise ValueError(f"Invalid configuration: {e.message}") from e
            elif validate_schema and not JSONSCHEMA_AVAILABLE:
                logger.warning("jsonschema not available, skipping schema validation")
            
            # Convert to typed configuration
            self._config = self._convert_to_dataclass(raw_config)
            
            logger.info(f"Successfully loaded configuration from {self.config_path}")
            return self._config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Return default configuration for robustness
            logger.warning("Using default configuration due to load failure")
            self._config = ThresholdsConfig()
            return self._config
    
    def get_config(self) -> ThresholdsConfig:
        """Get loaded configuration, loading if necessary."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def reload_config(self) -> ThresholdsConfig:
        """Force reload configuration from file."""
        self._config = None
        self._raw_config = None
        return self.load_config()
    
    def get_raw_config(self) -> Dict[str, Any]:
        """Get raw configuration dictionary."""
        if self._raw_config is None:
            self.load_config()
        return self._raw_config or {}
    
    def validate_thresholds(self, config: ThresholdsConfig) -> List[str]:
        """
        Validate threshold value consistency and relationships.
        
        Returns:
            List of validation warnings/errors
        """
        issues = []
        
        # Temperature bounds validation
        temp = config.temperature
        if temp.min_temperature >= temp.max_temperature:
            issues.append(f"Invalid temperature bounds: min={temp.min_temperature} >= max={temp.max_temperature}")
        
        if not (temp.min_temperature <= temp.default_temperature <= temp.max_temperature):
            issues.append(f"Default temperature {temp.default_temperature} outside bounds [{temp.min_temperature}, {temp.max_temperature}]")
        
        # Score bounds validation
        bounds = config.scoring_bounds
        if bounds.min_score >= bounds.max_score:
            issues.append(f"Invalid score bounds: min={bounds.min_score} >= max={bounds.max_score}")
        
        # Fusion weights validation (should sum to 1.0)
        fusion = config.fusion_weights
        fusion_sum = fusion.lexical + fusion.vector + fusion.late_interaction + fusion.rrf
        if abs(fusion_sum - 1.0) > 0.01:
            issues.append(f"Fusion weights sum to {fusion_sum:.3f}, expected 1.0")
        
        # Evidence multiplier validation
        mult = config.evidence_multipliers
        if mult.MIN_MULTIPLIER >= mult.MAX_MULTIPLIER:
            issues.append(f"Invalid multiplier bounds: MIN={mult.MIN_MULTIPLIER} >= MAX={mult.MAX_MULTIPLIER}")
        
        # Conformal prediction validation
        cp = config.conformal_prediction
        if not (0 < cp.alpha < 1):
            issues.append(f"Invalid alpha value: {cp.alpha} (must be in (0,1))")
        
        if not (0 < cp.calibration_ratio < 1):
            issues.append(f"Invalid calibration_ratio: {cp.calibration_ratio} (must be in (0,1))")
        
        # Decálogo weights validation
        decalogo = config.decalogo_scoring
        dim_weights_sum = sum(decalogo.dimension_weights.values())
        if abs(dim_weights_sum - 1.0) > 0.01:
            issues.append(f"Dimension weights sum to {dim_weights_sum:.3f}, expected 1.0")
        
        point_weights_sum = sum(decalogo.decalogo_point_weights.values())
        if abs(point_weights_sum - 1.0) > 0.01:
            issues.append(f"Decálogo point weights sum to {point_weights_sum:.3f}, expected 1.0")
        
        return issues


# Global configuration loader instance
_config_loader: Optional[ConfigLoader] = None


def get_config_loader(config_path: Optional[Union[str, Path]] = None) -> ConfigLoader:
    """Get global configuration loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader


def get_thresholds() -> ThresholdsConfig:
    """Get loaded thresholds configuration."""
    return get_config_loader().get_config()


def reload_thresholds() -> ThresholdsConfig:
    """Reload thresholds configuration from file."""
    return get_config_loader().reload_config()


def validate_config_on_startup() -> bool:
    """
    Validate configuration on application startup.
    
    Returns:
        True if validation passes, False if issues found
    """
    try:
        loader = get_config_loader()
        config = loader.load_config(validate_schema=True)
        issues = loader.validate_thresholds(config)
        
        if issues:
            logger.warning("Configuration validation issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
            return False
        
        logger.info("Configuration validation passed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


if __name__ == "__main__":
    # Test configuration loading
    import sys
    
    try:
        valid = validate_config_on_startup()
        config = get_thresholds()
        
        print(f"Configuration loaded: {config.version}")
        print(f"Temperature range: [{config.temperature.min_temperature}, {config.temperature.max_temperature}]")
        print(f"Score bounds: [{config.scoring_bounds.min_score}, {config.scoring_bounds.max_score}]")
        print(f"Alpha: {config.conformal_prediction.alpha}")
        print(f"MIN_MULTIPLIER: {config.evidence_multipliers.MIN_MULTIPLIER}")
        print(f"MAX_MULTIPLIER: {config.evidence_multipliers.MAX_MULTIPLIER}")
        
        if valid:
            print("\n✅ Configuration validation passed")
            sys.exit(0)
        else:
            print("\n⚠️  Configuration validation failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Configuration loading failed: {e}")
        sys.exit(1)
>>>>>>> 4a50c97 (Create centralized thresholds configuration file and update all modules to use shared parameters)
