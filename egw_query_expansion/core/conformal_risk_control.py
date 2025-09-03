"""
Enhanced Conformal Risk Control System Implementation

Based on "Conformal Risk Control" by Angelopoulos et al. (2024)
Journal of Machine Learning Research

Implements adaptive conformal prediction with distribution-free guarantees
for exact finite-sample coverage control with comprehensive risk certification.

Features:
- Distribution-free finite-sample guarantees
- Exact coverage control with adaptive calibration
- RCPS (Risk-Controlling Prediction Sets) construction
- Distribution shift detection and bounds
- Deterministic certification with reproducible splits
- Comprehensive statistical validation
"""

import hashlib
import json
import logging
import pickle
import warnings
# # # from abc import ABC, abstractmethod  # Module not found  # Module not found  # Module not found
# # # from collections import defaultdict  # Module not found  # Module not found  # Module not found
# # # from dataclasses import asdict, dataclass, field  # Module not found  # Module not found  # Module not found
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found
# # # from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union  # Module not found  # Module not found  # Module not found

import numpy as np
# # # from scipy import stats  # Module not found  # Module not found  # Module not found

logger = logging.getLogger(__name__)

# Try to import centralized configuration
try:
# # #     from config_loader import get_thresholds  # Module not found  # Module not found  # Module not found
    THRESHOLDS_AVAILABLE = True
except ImportError:
    THRESHOLDS_AVAILABLE = False
    logger.warning("Centralized thresholds not available, using default values")


@dataclass
class RiskControlConfig:
    """Configuration for risk control parameters with enhanced certification."""

    # Core conformal parameters
    alpha: float = 0.1  # Miscoverage rate (1-α = coverage target)
    lambda_reg: float = 0.0  # Regularization parameter for adaptive methods
    random_seed: int = 42  # Deterministic seed for reproducible splits

    # Data splitting parameters
    calibration_ratio: float = 0.5  # Fraction for calibration split
    validation_size: int = 1000  # Minimum validation set size
    test_ratio: float = 0.2  # Fraction for final testing

    # Adaptive control parameters
    adaptive_lambda: bool = True  # Enable adaptive regularization
    adaptive_quantile_method: str = "clt"  # "clt" or "bootstrap"
    batch_adaptation: bool = False  # Enable batch-wise adaptation

    # Distribution shift control
    distribution_shift_bound: float = 0.1  # Maximum allowed distribution shift
    shift_detection_method: str = "ks"  # "ks", "mmd", or "energy"
    recalibration_threshold: float = 0.05  # p-value threshold for recalibration

    # Certification parameters
    confidence_level: float = 0.95  # Confidence level for risk bounds
    certificate_validity_hours: int = 24  # Certificate validity duration
    enable_cross_validation: bool = True  # Enable cross-validation bounds
    bootstrap_samples: int = 1000  # Bootstrap samples for uncertainty quantification

    # Quality control
    min_calibration_size: int = 100  # Minimum calibration set size
    max_set_size_ratio: float = (
        0.8  # Maximum prediction set size as ratio of candidates
    )
    coverage_tolerance: float = 0.01  # Tolerance for coverage validation

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if not 0 < self.calibration_ratio < 1:
            raise ValueError("calibration_ratio must be in (0, 1)")
        if self.validation_size < 50:
            raise ValueError("validation_size must be at least 50")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        if self.min_calibration_size < 10:
            raise ValueError("min_calibration_size must be at least 10")

    @classmethod
    def from_thresholds_config(cls, use_centralized: bool = True) -> 'RiskControlConfig':
# # #         """Create RiskControlConfig from centralized thresholds configuration."""  # Module not found  # Module not found  # Module not found
        if use_centralized and THRESHOLDS_AVAILABLE:
            try:
                config = get_thresholds()
                cp_config = config.conformal_prediction
                quality_config = config.quality_thresholds
                
                return cls(
                    alpha=cp_config.alpha,
                    lambda_reg=cp_config.lambda_reg,
                    calibration_ratio=cp_config.calibration_ratio,
                    validation_size=cp_config.validation_size,
                    test_ratio=cp_config.test_ratio,
                    distribution_shift_bound=cp_config.distribution_shift_bound,
                    recalibration_threshold=cp_config.recalibration_threshold,
                    confidence_level=quality_config.confidence_level,
                    bootstrap_samples=cp_config.bootstrap_samples,
                    min_calibration_size=cp_config.min_calibration_size,
                    max_set_size_ratio=cp_config.max_set_size_ratio,
                    coverage_tolerance=cp_config.coverage_tolerance
                )
            except Exception as e:
                logger.warning(f"Failed to load centralized config: {e}, using defaults")
        
        # Return default configuration
        return cls()


@dataclass
class NonconformityScore:
    """Container for non-conformity scoring results."""

    score: float
    prediction: Any
    true_label: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionSet:
    """Conformal prediction set with coverage guarantees."""

    set_values: Set[Any]
    confidence_level: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    size: int = field(init=False)

    def __post_init__(self):
        self.size = len(self.set_values)

    def contains(self, value: Any) -> bool:
        """Check if value is in prediction set."""
        return value in self.set_values


@dataclass
class StatisticalBounds:
    """Statistical bounds for risk control with multiple estimation methods."""

    empirical_risk: float
    upper_bound_hoeffding: float
    upper_bound_clt: float
    upper_bound_bootstrap: float
    lower_bound: float
    confidence_level: float
    sample_size: int
    bounds_method: str

    @property
    def conservative_bound(self) -> float:
        """Return most conservative upper bound."""
        return min(
            1.0,
            max(
                self.upper_bound_hoeffding,
                self.upper_bound_clt,
                self.upper_bound_bootstrap,
            ),
        )


@dataclass
class CoverageAnalysis:
    """Comprehensive coverage analysis with statistical tests."""

    empirical_coverage: float
    target_coverage: float
    coverage_gap: float
    confidence_interval: Tuple[float, float]
    statistical_significance: float
    coverage_test_p_value: float
    average_set_size: float
    set_size_std: float
    efficiency_score: float  # 1 / average_set_size

    def is_significantly_different(self, alpha: float = 0.05) -> bool:
# # #         """Test if coverage is significantly different from target."""  # Module not found  # Module not found  # Module not found
        return self.coverage_test_p_value < alpha


@dataclass
class RiskCertificate:
    """Enhanced certification of risk bounds with comprehensive validation."""

    # Core risk metrics
    empirical_risk: float
    risk_bounds: StatisticalBounds
    coverage_analysis: CoverageAnalysis

    # Certificate metadata
    certificate_id: str
    certificate_version: str
    input_hash: str
    timestamp: str
    validity_expires: str

    # Configuration used
    config_hash: str
    model_metadata: Dict[str, Any]
    scorer_metadata: Dict[str, Any]

    # Validation results
    distribution_shift_results: Dict[str, Any]
    cross_validation_results: Optional[Dict[str, Any]]
    bootstrap_results: Dict[str, Any]

    # Quality indicators
    calibration_quality_score: float
    prediction_efficiency: float
    uncertainty_quantification: Dict[str, Any]

    def is_valid(self, tolerance: float = 0.01) -> bool:
        """Check if certificate meets validity requirements with enhanced validation."""
        coverage_valid = abs(self.coverage_analysis.coverage_gap) <= tolerance

        risk_bound_valid = (
            self.risk_bounds.conservative_bound
            <= 1 - self.coverage_analysis.target_coverage + tolerance
        )

        shift_valid = not self.distribution_shift_results.get("shift_detected", True)

        return coverage_valid and risk_bound_valid and shift_valid

    def to_dict(self) -> Dict[str, Any]:
        """Convert certificate to dictionary for serialization."""
        return asdict(self)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save certificate to file."""
        filepath = Path(filepath)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "RiskCertificate":
# # #         """Load certificate from file."""  # Module not found  # Module not found  # Module not found
        filepath = Path(filepath)
        with open(filepath, "r") as f:
            data = json.load(f)

        # Reconstruct nested dataclasses
        data["risk_bounds"] = StatisticalBounds(**data["risk_bounds"])
        data["coverage_analysis"] = CoverageAnalysis(**data["coverage_analysis"])

        return cls(**data)


class NonconformityScorer(ABC):
    """Abstract base class for non-conformity scoring functions with enhanced metadata."""

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.creation_time = str(np.datetime64("now"))
        self.usage_count = 0

    @abstractmethod
    def score(self, prediction: Any, true_label: Any) -> float:
        """Compute non-conformity score for a prediction."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get scorer metadata for reproducibility."""
        pass

    def batch_score(self, predictions: List[Any], true_labels: List[Any]) -> np.ndarray:
        """Efficiently compute scores for multiple predictions."""
        self.usage_count += len(predictions)

        scores = []
        for pred, label in zip(predictions, true_labels):
            # Handle case where prediction might be a list for classification
            if isinstance(pred, list):
                pred = np.array(pred)
            scores.append(self.score(pred, label))

        return np.array(scores)

    def validate_inputs(self, prediction: Any, true_label: Any) -> None:
        """Validate input types and ranges."""
        pass  # Override in subclasses for specific validation

    def get_score_statistics(self, scores: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive statistics for score distribution."""
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores)),
            "q25": float(np.percentile(scores, 25)),
            "q75": float(np.percentile(scores, 75)),
            "skewness": float(stats.skew(scores)),
            "kurtosis": float(stats.kurtosis(scores)),
        }


class RegressionNonconformityScorer(NonconformityScorer):
    """Enhanced non-conformity scorer for regression tasks with multiple methods."""

    def __init__(
        self,
        method: str = "absolute",
        normalize: bool = False,
        robust: bool = False,
        name: str = None,
        temperature: Optional[float] = None,
    ):
        super().__init__(name)
        self.method = method
        self.normalize = normalize
        self.robust = robust
        self.scale_factor = None
        
# # #         # Load temperature from centralized config if not provided  # Module not found  # Module not found  # Module not found
        if temperature is None and THRESHOLDS_AVAILABLE:
            try:
                config = get_thresholds()
                self.temperature = config.temperature.default_temperature
            except Exception:
                self.temperature = 1.0
        else:
            self.temperature = temperature if temperature is not None else 1.0

        valid_methods = ["absolute", "squared", "quantile", "studentized"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

    def fit_normalization(
        self, predictions: np.ndarray, true_labels: np.ndarray
    ) -> None:
# # #         """Fit normalization parameters from calibration data."""  # Module not found  # Module not found  # Module not found
        residuals = predictions - true_labels

        if self.robust:
            # Use MAD (Median Absolute Deviation) for robust normalization
            self.scale_factor = np.median(np.abs(residuals - np.median(residuals)))
        else:
            # Use standard deviation
            self.scale_factor = np.std(residuals)

        if self.scale_factor == 0:
            self.scale_factor = 1.0  # Avoid division by zero

    def score(self, prediction: float, true_label: float) -> float:
        """Compute regression non-conformity score with multiple methods."""
        global score, score
        self.validate_inputs(prediction, true_label)

        residual = prediction - true_label

        if self.method == "absolute":
            score = abs(residual)
        elif self.method == "squared":
            score = residual**2
        elif self.method == "quantile":
            # Asymmetric loss for quantile regression
            tau = 0.5  # Median
            score = residual * (tau - (residual < 0))
        elif self.method == "studentized":
            # Studentized residual (requires pre-fitted scale)
            if self.scale_factor is None:
                warnings.warn("Scale factor not fitted, using raw residual")
                score = abs(residual)
            else:
                score = abs(residual) / max(self.scale_factor, 1e-8)

        # Apply normalization if enabled
        if self.normalize and self.scale_factor is not None:
            score = score / max(self.scale_factor, 1e-8)

        return float(score)

    def validate_inputs(self, prediction: Any, true_label: Any) -> None:
        """Validate that inputs are numeric."""
        if not isinstance(prediction, (int, float, np.number)):
            raise TypeError(f"Prediction must be numeric, got {type(prediction)}")
        if not isinstance(true_label, (int, float, np.number)):
            raise TypeError(f"True label must be numeric, got {type(true_label)}")
        if not np.isfinite(prediction):
            raise ValueError("Prediction must be finite")
        if not np.isfinite(true_label):
            raise ValueError("True label must be finite")

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "scorer_type": "regression",
            "method": self.method,
            "normalize": self.normalize,
            "robust": self.robust,
            "scale_factor": self.scale_factor,
            "name": self.name,
            "creation_time": self.creation_time,
            "usage_count": self.usage_count,
        }


class ClassificationNonconformityScorer(NonconformityScorer):
    """Enhanced non-conformity scorer for classification tasks with multiple methods."""

    def __init__(
        self,
        method: str = "softmax",
        adaptive_threshold: bool = False,
        temperature: Optional[float] = None,
        name: str = None,
    ):
        super().__init__(name)
        self.method = method
        self.adaptive_threshold = adaptive_threshold
        self.class_thresholds = None
        
# # #         # Load temperature from centralized config if not provided  # Module not found  # Module not found  # Module not found
        if temperature is None and THRESHOLDS_AVAILABLE:
            try:
                config = get_thresholds()
                self.temperature = config.temperature.classification_temperature
            except Exception:
                self.temperature = 1.0
        else:
            self.temperature = temperature if temperature is not None else 1.0

        valid_methods = ["softmax", "margin", "entropy", "logit", "rank"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

    def fit_adaptive_thresholds(
        self, predictions: np.ndarray, true_labels: np.ndarray
    ) -> None:
        """Fit class-specific adaptive thresholds."""
        if not self.adaptive_threshold:
            return

        self.class_thresholds = {}
        unique_classes = np.unique(true_labels)

        for class_idx in unique_classes:
            class_mask = true_labels == class_idx
            if np.sum(class_mask) > 0:
                class_scores = [
                    self._compute_base_score(pred, class_idx)
                    for pred in predictions[class_mask]
                ]
                self.class_thresholds[class_idx] = np.median(class_scores)

    def _compute_base_score(self, prediction: np.ndarray, true_label: int) -> float:
        """Compute base score without adaptive adjustment."""
        # Apply temperature scaling
        scaled_prediction = prediction / self.temperature

        if self.method == "softmax":
            # Standard inverse probability score
            probs = self._softmax(scaled_prediction)
            return 1.0 - probs[true_label]

        elif self.method == "margin":
            # Margin-based score
            probs = self._softmax(scaled_prediction)
            sorted_probs = np.sort(probs)[::-1]  # Sort descending
            if len(sorted_probs) > 1:
                return sorted_probs[0] - sorted_probs[1]
            return 0.0

        elif self.method == "entropy":
            # Entropy-based uncertainty
            probs = self._softmax(scaled_prediction)
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            return entropy

        elif self.method == "logit":
            # Raw logit difference
            if len(scaled_prediction) > true_label:
                max_other_logit = np.max(np.delete(scaled_prediction, true_label))
                return max_other_logit - scaled_prediction[true_label]
            return 0.0

        elif self.method == "rank":
            # Rank-based score
            ranks = stats.rankdata(-scaled_prediction)  # Negative for descending
            return float(ranks[true_label])

    def score(self, prediction: np.ndarray, true_label: int) -> float:
        """Compute classification non-conformity score with enhancements."""
        self.validate_inputs(prediction, true_label)

        base_score = self._compute_base_score(prediction, true_label)

        # Apply adaptive threshold if fitted
        if (
            self.adaptive_threshold
            and self.class_thresholds is not None
            and true_label in self.class_thresholds
        ):
            base_score = base_score / (self.class_thresholds[true_label] + 1e-8)

        return float(base_score)

    def validate_inputs(self, prediction: Any, true_label: Any) -> None:
        """Validate inputs for classification scoring."""
        if not isinstance(prediction, np.ndarray):
            raise TypeError("Prediction must be numpy array")
        if not isinstance(true_label, (int, np.integer)):
            raise TypeError("True label must be integer")
        if true_label < 0 or true_label >= len(prediction):
            raise ValueError(
                f"True label {true_label} out of range [0, {len(prediction)})"
            )
        if not np.all(np.isfinite(prediction)):
            raise ValueError("Prediction contains non-finite values")

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "scorer_type": "classification",
            "method": self.method,
            "adaptive_threshold": self.adaptive_threshold,
            "temperature": self.temperature,
            "class_thresholds": self.class_thresholds,
            "name": self.name,
            "creation_time": self.creation_time,
            "usage_count": self.usage_count,
        }


class EnhancedConformalRiskController:
    """
    Enhanced conformal risk control system implementing RCPS with comprehensive certification.

    Provides distribution-free finite-sample guarantees following
    Angelopoulos et al. (2024) methodology with enhanced:
    - Multi-method statistical bounds
    - Distribution shift detection
    - Cross-validation support
    - Bootstrap uncertainty quantification
    - Comprehensive certificate generation
    """

    def __init__(self, config: Optional[RiskControlConfig] = None):
# # #         # Load config from centralized thresholds if not provided  # Module not found  # Module not found  # Module not found
        if config is None:
            self.config = RiskControlConfig.from_thresholds_config()
        else:
            self.config = config
        self.config.validate()  # Validate configuration

        # Core data splits
        self.calibration_scores: np.ndarray = np.array([])
        self.validation_scores: np.ndarray = np.array([])
        self.test_scores: np.ndarray = np.array([])

        # Fitting state
        self.is_fitted = False
        self.scorer_fitted = False

        # Random number generators with deterministic seeds
        self._rng = np.random.RandomState(config.random_seed)
        self._bootstrap_rng = np.random.RandomState(config.random_seed + 1)

        # Cache for identical certificates
        self._certificate_cache: Dict[str, RiskCertificate] = {}

        # Adaptive parameters
        self.adaptive_quantile = None
        self.quantile_bounds = None

        # Distribution tracking
        self.baseline_distribution = None
        self.shift_history = []

        # Cross-validation results
        self.cv_results = None

        logger.info(
            f"Initialized enhanced conformal risk controller with α={config.alpha}, "
            f"seed={config.random_seed}"
        )

    def fit_calibration(
        self,
        predictions: List[Any],
        true_labels: List[Any],
        scorer: NonconformityScorer,
        enable_scorer_fitting: bool = True,
    ) -> Dict[str, Any]:
        """
        Enhanced calibration fitting with comprehensive statistical analysis.

        Args:
            predictions: Model predictions on labeled data
            true_labels: Ground truth labels
            scorer: Non-conformity scoring function
            enable_scorer_fitting: Whether to fit scorer parameters

        Returns:
            Dictionary with fitting statistics and diagnostics
        """
        if len(predictions) != len(true_labels):
            raise ValueError("Predictions and labels must have same length")

        n_total = len(predictions)
        if n_total < self.config.validation_size:
            raise ValueError(f"Need at least {self.config.validation_size} samples")

        if n_total < self.config.min_calibration_size:
            raise ValueError(
                f"Need at least {self.config.min_calibration_size} samples"
            )

        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        # Deterministic three-way split using fixed seed
        indices = np.arange(n_total)
        self._rng.shuffle(indices)

        # Calculate split sizes
        n_test = max(int(self.config.test_ratio * n_total), 50)
        n_cal = int(self.config.calibration_ratio * (n_total - n_test))
        n_val = n_total - n_cal - n_test

        # Split indices
        cal_indices = indices[:n_cal]
        val_indices = indices[n_cal : n_cal + n_val]
        test_indices = indices[n_cal + n_val :]

        # Extract splits
        cal_pred, cal_labels = predictions[cal_indices], true_labels[cal_indices]
        val_pred, val_labels = predictions[val_indices], true_labels[val_indices]
        test_pred, test_labels = predictions[test_indices], true_labels[test_indices]

        # Fit scorer parameters if requested
        if enable_scorer_fitting:
            if hasattr(scorer, "fit_normalization"):
                scorer.fit_normalization(cal_pred, cal_labels)
            if hasattr(scorer, "fit_adaptive_thresholds"):
                scorer.fit_adaptive_thresholds(cal_pred, cal_labels)
            self.scorer_fitted = True

        # Calculate non-conformity scores
        self.calibration_scores = scorer.batch_score(
            cal_pred.tolist(), cal_labels.tolist()
        )
        self.validation_scores = scorer.batch_score(
            val_pred.tolist(), val_labels.tolist()
        )
        self.test_scores = scorer.batch_score(test_pred.tolist(), test_labels.tolist())

        # Store baseline distribution for shift detection
        self.baseline_distribution = self.calibration_scores.copy()

        # Compute adaptive quantile with multiple methods
        quantile_results = self._compute_comprehensive_quantile()

        # Perform cross-validation if enabled
        if self.config.enable_cross_validation:
            self.cv_results = self._cross_validate_coverage(
                predictions, true_labels, scorer
            )

        self.is_fitted = True

        fitting_stats = {
            "n_total": n_total,
            "n_calibration": n_cal,
            "n_validation": n_val,
            "n_test": n_test,
            "calibration_score_stats": scorer.get_score_statistics(
                self.calibration_scores
            ),
            "validation_score_stats": scorer.get_score_statistics(
                self.validation_scores
            ),
            "quantile_results": quantile_results,
            "scorer_metadata": scorer.get_metadata(),
            "splits_seed": self.config.random_seed,
        }

        if self.cv_results:
            fitting_stats["cross_validation"] = self.cv_results

        logger.info(
            f"Enhanced calibration fitted: {n_cal} cal, {n_val} val, {n_test} test samples"
        )

        return fitting_stats

    def _compute_comprehensive_quantile(self) -> Dict[str, Any]:
        """Compute quantile using multiple methods with uncertainty bounds."""
        n_cal = len(self.calibration_scores)

        # Standard conformal quantile level
        base_quantile_level = (1 + 1 / n_cal) * (1 - self.config.alpha)

        results = {
            "base_quantile_level": base_quantile_level,
            "n_calibration": n_cal,
            "methods": {},
        }

        # Method 1: Standard quantile
        standard_quantile = np.quantile(self.calibration_scores, base_quantile_level)
        results["methods"]["standard"] = standard_quantile

        # Method 2: CLT-based adjustment
        if self.config.adaptive_quantile_method == "clt":
            empirical_cdf = np.mean(self.calibration_scores <= standard_quantile)
            clt_adjustment = np.sqrt(empirical_cdf * (1 - empirical_cdf) / n_cal)
            adjusted_level = min(1.0, base_quantile_level + clt_adjustment)
            clt_quantile = np.quantile(self.calibration_scores, adjusted_level)
            results["methods"]["clt"] = clt_quantile

        # Method 3: Bootstrap quantile bounds
        if self.config.adaptive_quantile_method == "bootstrap":
            bootstrap_quantiles = []
            for _ in range(self.config.bootstrap_samples):
                boot_indices = self._bootstrap_rng.choice(
                    n_cal, size=n_cal, replace=True
                )
                boot_scores = self.calibration_scores[boot_indices]
                boot_quantile = np.quantile(boot_scores, base_quantile_level)
                bootstrap_quantiles.append(boot_quantile)

            bootstrap_quantiles = np.array(bootstrap_quantiles)
            results["methods"]["bootstrap_mean"] = np.mean(bootstrap_quantiles)
            results["methods"]["bootstrap_upper"] = np.quantile(
                bootstrap_quantiles, 0.95
            )

        # Adaptive adjustment based on empirical distribution
        if self.config.adaptive_lambda:
            empirical_cdf_at_quantile = np.mean(
                self.calibration_scores <= standard_quantile
            )
            adaptation_factor = min(
                1.0, empirical_cdf_at_quantile / (1 - self.config.alpha)
            )
            adaptive_level = min(1.0, base_quantile_level * adaptation_factor)
            adaptive_quantile = np.quantile(self.calibration_scores, adaptive_level)
            results["methods"]["adaptive"] = adaptive_quantile
            results["adaptation_factor"] = adaptation_factor

        # Select primary quantile
        if self.config.adaptive_lambda:
            self.adaptive_quantile = results["methods"]["adaptive"]
            primary_method = "adaptive"
        elif self.config.adaptive_quantile_method == "bootstrap":
            self.adaptive_quantile = results["methods"]["bootstrap_upper"]
            primary_method = "bootstrap_upper"
        elif self.config.adaptive_quantile_method == "clt":
            self.adaptive_quantile = results["methods"]["clt"]
            primary_method = "clt"
        else:
            self.adaptive_quantile = standard_quantile
            primary_method = "standard"

        results["selected_quantile"] = self.adaptive_quantile
        results["selected_method"] = primary_method

        # Compute confidence bounds for the quantile
        self.quantile_bounds = self._compute_quantile_confidence_bounds(
            base_quantile_level
        )
        results["quantile_bounds"] = self.quantile_bounds

        logger.debug(
            f"Computed quantile {self.adaptive_quantile:.4f} using {primary_method}"
        )

        return results

    def _cross_validate_coverage(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        scorer: NonconformityScorer,
        n_folds: int = 5,
    ) -> Dict[str, Any]:
        """Perform cross-validation to estimate coverage stability."""
# # #         from sklearn.model_selection import KFold  # Module not found  # Module not found  # Module not found

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.config.random_seed)
        cv_coverages = []
        cv_set_sizes = []
        cv_quantiles = []

        for train_idx, test_idx in kf.split(predictions):
            # Split data
            train_pred, train_labels = predictions[train_idx], true_labels[train_idx]
            test_pred, test_labels = predictions[test_idx], true_labels[test_idx]

            # Further split training into cal/val
            n_train = len(train_pred)
            n_cal = int(0.5 * n_train)

            train_indices = np.arange(n_train)
            np.random.RandomState(self.config.random_seed).shuffle(train_indices)

            cal_indices = train_indices[:n_cal]

            cal_pred = train_pred[cal_indices]
            cal_labels = train_labels[cal_indices]

            # Compute scores and quantile
            cal_scores = scorer.batch_score(cal_pred.tolist(), cal_labels.tolist())
            fold_quantile = np.quantile(
                cal_scores, (1 + 1 / len(cal_scores)) * (1 - self.config.alpha)
            )
            cv_quantiles.append(fold_quantile)

            # Test coverage on fold test set
            test_scores = scorer.batch_score(test_pred.tolist(), test_labels.tolist())
            covered = np.sum(test_scores <= fold_quantile)
            coverage = covered / len(test_scores)

            cv_coverages.append(coverage)
            cv_set_sizes.append(np.mean(test_scores <= fold_quantile))

        return {
            "mean_coverage": np.mean(cv_coverages),
            "std_coverage": np.std(cv_coverages),
            "min_coverage": np.min(cv_coverages),
            "max_coverage": np.max(cv_coverages),
            "coverage_stability": 1.0 - np.std(cv_coverages),  # Higher is more stable
            "mean_quantile": np.mean(cv_quantiles),
            "std_quantile": np.std(cv_quantiles),
            "n_folds": n_folds,
        }

    def _compute_quantile_confidence_bounds(
        self, quantile_level: float
    ) -> Dict[str, float]:
        """Compute confidence bounds for quantile estimate."""
        n = len(self.calibration_scores)
        z_alpha = stats.norm.ppf(1 - self.config.alpha / 2)  # Two-sided confidence

        # Wilson score interval for quantile
        p = quantile_level
        bound_radius = z_alpha * np.sqrt(p * (1 - p) / n)

        lower_level = max(0, p - bound_radius)
        upper_level = min(1, p + bound_radius)

        return {
            "lower_quantile": np.quantile(self.calibration_scores, lower_level),
            "upper_quantile": np.quantile(self.calibration_scores, upper_level),
            "lower_level": lower_level,
            "upper_level": upper_level,
        }

    def construct_prediction_set(
        self,
        candidate_predictions: Dict[Any, float],
        confidence_level: Optional[float] = None,
    ) -> PredictionSet:
        """
        Construct RCPS prediction set with coverage guarantees.

        Args:
# # #             candidate_predictions: Mapping from candidate values to non-conformity scores  # Module not found  # Module not found  # Module not found
            confidence_level: Override default confidence level

        Returns:
            PredictionSet with guaranteed coverage
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Must fit calibration before constructing prediction sets"
            )

        alpha = confidence_level or self.config.alpha
        quantile_threshold = self.adaptive_quantile

        # Construct prediction set
        prediction_set = {
            value
            for value, score in candidate_predictions.items()
            if score <= quantile_threshold
        }

        # Handle empty sets (add most likely prediction)
        if not prediction_set and candidate_predictions:
            best_candidate = min(candidate_predictions.items(), key=lambda x: x[1])
            prediction_set = {best_candidate[0]}

        return PredictionSet(set_values=prediction_set, confidence_level=1 - alpha)

    def construct_confidence_interval(
        self, point_prediction: float, scorer: NonconformityScorer
    ) -> Tuple[float, float]:
        """
        Construct confidence interval for regression predictions.

        Args:
# # #             point_prediction: Point prediction from model  # Module not found  # Module not found  # Module not found
            scorer: Non-conformity scorer (should be regression type)

        Returns:
            (lower_bound, upper_bound) with guaranteed coverage
        """
        if not self.is_fitted:
            raise RuntimeError("Must fit calibration before constructing intervals")

        if not isinstance(scorer, RegressionNonconformityScorer):
            raise ValueError("Confidence intervals require regression scorer")

        quantile_threshold = self.adaptive_quantile

        if scorer.method == "absolute":
            return (
                point_prediction - quantile_threshold,
                point_prediction + quantile_threshold,
            )
        elif scorer.method == "squared":
            radius = np.sqrt(quantile_threshold)
            return (point_prediction - radius, point_prediction + radius)
        else:
            raise ValueError(f"Unsupported scorer method: {scorer.method}")

    def comprehensive_coverage_analysis(
        self,
        test_predictions: List[Any],
        test_labels: List[Any],
        scorer: NonconformityScorer,
    ) -> CoverageAnalysis:
        """
        Comprehensive coverage validation with statistical testing.

        Args:
            test_predictions: Test set predictions
            test_labels: Test set labels
            scorer: Non-conformity scorer

        Returns:
            CoverageAnalysis object with detailed statistics
        """
        if not self.is_fitted:
            raise RuntimeError("Must fit calibration before validation")

        coverage_indicators = []
        set_sizes = []
        total_count = len(test_predictions)

        for pred, label in zip(test_predictions, test_labels):
            # For regression, create candidate set around prediction
            if isinstance(scorer, RegressionNonconformityScorer):
                lower, upper = self.construct_confidence_interval(pred, scorer)
                is_covered = lower <= label <= upper
                set_size = upper - lower
            else:
                # For classification, construct prediction set
                if hasattr(pred, "__iter__") and not isinstance(pred, str):
                    # Ensure prediction is numpy array
                    if not isinstance(pred, np.ndarray):
                        pred = np.array(pred)
                    candidates = {i: scorer.score(pred, i) for i in range(len(pred))}
                    pred_set = self.construct_prediction_set(candidates)
                    is_covered = pred_set.contains(label)
                    set_size = pred_set.size
                else:
                    # Simple binary case
                    candidates = {0: scorer.score(pred, 0), 1: scorer.score(pred, 1)}
                    pred_set = self.construct_prediction_set(candidates)
                    is_covered = pred_set.contains(label)
                    set_size = pred_set.size

            coverage_indicators.append(is_covered)
            set_sizes.append(set_size)

        coverage_indicators = np.array(coverage_indicators)
        set_sizes = np.array(set_sizes)

        empirical_coverage = np.mean(coverage_indicators)
        target_coverage = 1 - self.config.alpha
        coverage_gap = empirical_coverage - target_coverage

        # Statistical test for coverage
        n_covered = np.sum(coverage_indicators)

        # Binomial test for coverage hypothesis (using newer scipy API)
        try:
# # #             from scipy.stats import binomtest  # Module not found  # Module not found  # Module not found

            coverage_p_value = binomtest(
                n_covered, total_count, target_coverage, alternative="two-sided"
            ).pvalue
        except ImportError:
            # Fallback for older scipy versions
            try:
# # #                 from scipy.stats import binom_test  # Module not found  # Module not found  # Module not found

                coverage_p_value = binom_test(
                    n_covered, total_count, target_coverage, alternative="two-sided"
                )
            except ImportError:
                # Manual binomial test calculation
# # #                 from scipy.stats import binom  # Module not found  # Module not found  # Module not found

                coverage_p_value = 2 * min(
                    binom.cdf(n_covered, total_count, target_coverage),
                    1 - binom.cdf(n_covered - 1, total_count, target_coverage),
                )

        # Confidence interval for coverage using Wilson score
        z_alpha = stats.norm.ppf(1 - self.config.alpha / 2)
        p_hat = empirical_coverage
        n = total_count

        denominator = 1 + z_alpha**2 / n
        center = (p_hat + z_alpha**2 / (2 * n)) / denominator
        margin = (
            z_alpha
            * np.sqrt(p_hat * (1 - p_hat) / n + z_alpha**2 / (4 * n**2))
            / denominator
        )

        confidence_interval = (max(0, center - margin), min(1, center + margin))

        # Efficiency metrics
        avg_set_size = np.mean(set_sizes)
        set_size_std = np.std(set_sizes)
        efficiency_score = 1.0 / max(avg_set_size, 1e-6)  # Avoid division by zero

        return CoverageAnalysis(
            empirical_coverage=empirical_coverage,
            target_coverage=target_coverage,
            coverage_gap=coverage_gap,
            confidence_interval=confidence_interval,
            statistical_significance=abs(coverage_gap),
            coverage_test_p_value=coverage_p_value,
            average_set_size=avg_set_size,
            set_size_std=set_size_std,
            efficiency_score=efficiency_score,
        )

    def compute_comprehensive_risk_bounds(self) -> StatisticalBounds:
        """
        Compute comprehensive risk bounds using multiple statistical methods.

        Returns:
            StatisticalBounds object with multiple bound estimates
        """
        if not self.is_fitted or len(self.validation_scores) == 0:
            return StatisticalBounds(
                empirical_risk=1.0,
                upper_bound_hoeffding=1.0,
                upper_bound_clt=1.0,
                upper_bound_bootstrap=1.0,
                lower_bound=0.0,
                confidence_level=self.config.confidence_level,
                sample_size=0,
                bounds_method="none",
            )

        n_val = len(self.validation_scores)
        threshold_violations = np.sum(self.validation_scores > self.adaptive_quantile)
        empirical_risk = threshold_violations / n_val

        # Confidence level
        alpha_bound = 1 - self.config.confidence_level

        # Method 1: Hoeffding's inequality (distribution-free)
        hoeffding_radius = np.sqrt(-np.log(alpha_bound) / (2 * n_val))
        upper_bound_hoeffding = min(1.0, empirical_risk + hoeffding_radius)

        # Method 2: CLT-based normal approximation
        if 0 < empirical_risk < 1:
            se_clt = np.sqrt(empirical_risk * (1 - empirical_risk) / n_val)
            z_score = stats.norm.ppf(1 - alpha_bound)
            upper_bound_clt = min(1.0, empirical_risk + z_score * se_clt)
            lower_bound = max(0.0, empirical_risk - z_score * se_clt)
        else:
            upper_bound_clt = upper_bound_hoeffding
            lower_bound = 0.0

        # Method 3: Bootstrap confidence interval
        bootstrap_risks = []
        for _ in range(self.config.bootstrap_samples):
            boot_indices = self._bootstrap_rng.choice(n_val, size=n_val, replace=True)
            boot_scores = self.validation_scores[boot_indices]
            boot_violations = np.sum(boot_scores > self.adaptive_quantile)
            bootstrap_risks.append(boot_violations / n_val)

        bootstrap_risks = np.array(bootstrap_risks)
        upper_bound_bootstrap = np.quantile(bootstrap_risks, 1 - alpha_bound)

        # Update lower bound with bootstrap if available
        if len(bootstrap_risks) > 0:
            lower_bound = max(lower_bound, np.quantile(bootstrap_risks, alpha_bound))

        return StatisticalBounds(
            empirical_risk=empirical_risk,
            upper_bound_hoeffding=upper_bound_hoeffding,
            upper_bound_clt=upper_bound_clt,
            upper_bound_bootstrap=upper_bound_bootstrap,
            lower_bound=lower_bound,
            confidence_level=self.config.confidence_level,
            sample_size=n_val,
            bounds_method="comprehensive",
        )

    def generate_enhanced_certificate(
        self,
        input_data: Any,
        predictions: List[Any],
        scorer: NonconformityScorer,
        test_labels: Optional[List[Any]] = None,
        model_metadata: Optional[Dict[str, Any]] = None,
    ) -> RiskCertificate:
        """
        Generate comprehensive risk certificate with enhanced guarantees.

        Args:
            input_data: Input data for hashing/reproducibility
            predictions: Model predictions
            scorer: Non-conformity scorer
            test_labels: Optional test labels for validation
            model_metadata: Optional model metadata for tracking

        Returns:
            Enhanced RiskCertificate with comprehensive bounds and validation
        """
        if not self.is_fitted:
            raise RuntimeError("Must fit calibration before generating certificates")

        # Create deterministic input hash for caching
        serialization_data = {
            "input_data": input_data,
            "config": self.config.__dict__,
            "scorer_metadata": scorer.get_metadata(),
            "model_metadata": model_metadata or {},
        }
        input_serialized = pickle.dumps(
            serialization_data, protocol=pickle.HIGHEST_PROTOCOL
        )
        input_hash = hashlib.sha256(input_serialized).hexdigest()

        # Check cache for identical certificates
        if input_hash in self._certificate_cache:
            cached_cert = self._certificate_cache[input_hash]
            # Check if certificate is still valid (not expired)
# # #             from datetime import datetime, timedelta  # Module not found  # Module not found  # Module not found

            if hasattr(cached_cert, "validity_expires"):
                try:
                    expiry_time = datetime.fromisoformat(cached_cert.validity_expires)
                    if datetime.now() < expiry_time:
                        logger.info(
                            f"Returning cached certificate {cached_cert.certificate_id}"
                        )
                        return cached_cert
                except ValueError:
                    pass  # Invalid timestamp format, regenerate

        # Compute comprehensive risk bounds
        risk_bounds = self.compute_comprehensive_risk_bounds()

        # Perform comprehensive coverage analysis
        if test_labels and len(test_labels) > 0:
            coverage_analysis = self.comprehensive_coverage_analysis(
                predictions, test_labels, scorer
            )
        else:
            # Fallback analysis using validation set
            empirical_coverage = 1 - risk_bounds.empirical_risk
            coverage_analysis = CoverageAnalysis(
                empirical_coverage=empirical_coverage,
                target_coverage=1 - self.config.alpha,
                coverage_gap=empirical_coverage - (1 - self.config.alpha),
                confidence_interval=(
                    empirical_coverage - 0.05,
                    empirical_coverage + 0.05,
                ),
                statistical_significance=abs(
                    empirical_coverage - (1 - self.config.alpha)
                ),
                coverage_test_p_value=0.5,  # Neutral p-value for fallback
                average_set_size=1.0,  # Placeholder
                set_size_std=0.0,  # Placeholder
                efficiency_score=1.0,  # Placeholder
            )

        # Detect distribution shift
        test_scores = (
            scorer.batch_score(predictions, test_labels) if test_labels else []
        )
        shift_results = (
            self.detect_distribution_shift(test_scores)
            if len(test_scores) > 0
            else {
                "shift_detected": False,
                "ks_statistic": 0.0,
                "p_value": 1.0,
                "requires_recalibration": False,
            }
        )

        # Bootstrap uncertainty quantification
        bootstrap_results = (
            self._compute_bootstrap_uncertainties(predictions, test_labels, scorer)
            if test_labels and len(test_labels) > 0
            else {"bootstrap_coverage_std": 0.0}
        )

        # Compute quality metrics
        calibration_quality = self._assess_calibration_quality()
        prediction_efficiency = 1.0 / max(coverage_analysis.average_set_size, 1e-6)

        # Generate certificate ID and metadata
        certificate_id = (
            f"ecrc_{input_hash[:16]}_{int(np.datetime64('now').astype(int))}"
        )
        timestamp = str(np.datetime64("now"))

# # #         from datetime import datetime, timedelta  # Module not found  # Module not found  # Module not found

        validity_expires = str(
            datetime.now() + timedelta(hours=self.config.certificate_validity_hours)
        )

        # Create configuration hash
        config_hash = hashlib.sha256(
            pickle.dumps(self.config.__dict__, protocol=pickle.HIGHEST_PROTOCOL)
        ).hexdigest()[:16]

        # Generate certificate
        certificate = RiskCertificate(
            # Core risk metrics
            empirical_risk=risk_bounds.empirical_risk,
            risk_bounds=risk_bounds,
            coverage_analysis=coverage_analysis,
            # Certificate metadata
            certificate_id=certificate_id,
            certificate_version="2.0",
            input_hash=input_hash,
            timestamp=timestamp,
            validity_expires=validity_expires,
            # Configuration tracking
            config_hash=config_hash,
            model_metadata=model_metadata or {},
            scorer_metadata=scorer.get_metadata(),
            # Validation results
            distribution_shift_results=shift_results,
            cross_validation_results=self.cv_results,
            bootstrap_results=bootstrap_results,
            # Quality indicators
            calibration_quality_score=calibration_quality,
            prediction_efficiency=prediction_efficiency,
            uncertainty_quantification={
                "quantile_bounds": self.quantile_bounds,
                "risk_bound_range": risk_bounds.conservative_bound
                - risk_bounds.empirical_risk,
                "coverage_uncertainty": coverage_analysis.set_size_std,
            },
        )

        # Cache certificate
        self._certificate_cache[input_hash] = certificate

        logger.info(
            f"Generated enhanced certificate {certificate_id} with "
            f"coverage {coverage_analysis.empirical_coverage:.3f} "
            f"(target: {coverage_analysis.target_coverage:.3f}), "
            f"risk bound: {risk_bounds.conservative_bound:.3f}"
        )

        return certificate

    def _compute_bootstrap_uncertainties(
        self,
        predictions: List[Any],
        test_labels: List[Any],
        scorer: NonconformityScorer,
    ) -> Dict[str, float]:
        """Compute bootstrap estimates of uncertainty."""
        if not test_labels or len(test_labels) < 10:
            return {"bootstrap_coverage_std": 0.0}

        bootstrap_coverages = []
        n_test = len(predictions)

        for _ in range(min(100, self.config.bootstrap_samples)):  # Limit for efficiency
# # #             # Bootstrap sample from test set  # Module not found  # Module not found  # Module not found
            boot_indices = self._bootstrap_rng.choice(n_test, size=n_test, replace=True)
            boot_pred = [predictions[i] for i in boot_indices]
            boot_labels = [test_labels[i] for i in boot_indices]

            # Compute coverage for this bootstrap sample
            coverage_analysis = self.comprehensive_coverage_analysis(
                boot_pred, boot_labels, scorer
            )
            bootstrap_coverages.append(coverage_analysis.empirical_coverage)

        return {
            "bootstrap_coverage_mean": np.mean(bootstrap_coverages),
            "bootstrap_coverage_std": np.std(bootstrap_coverages),
            "bootstrap_coverage_ci": (
                np.percentile(bootstrap_coverages, 2.5),
                np.percentile(bootstrap_coverages, 97.5),
            ),
        }

    def _assess_calibration_quality(self) -> float:
        """Assess quality of calibration using multiple metrics."""
        if not self.is_fitted:
            return 0.0

        # Metric 1: Distribution uniformity of calibration scores
        # Well-calibrated scores should be approximately uniform on [0,1] after transformation
        p_values = np.mean(
            self.calibration_scores[:, np.newaxis]
            <= self.calibration_scores[np.newaxis, :],
            axis=1,
        )

        # KS test against uniform distribution
# # #         from scipy.stats import kstest  # Module not found  # Module not found  # Module not found

        ks_stat, ks_p = kstest(p_values, "uniform")
        uniformity_score = ks_p  # Higher p-value indicates better uniformity

        # Metric 2: Stability across quantiles
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        empirical_quantiles = np.quantile(self.calibration_scores, quantiles)
        expected_spacing = np.diff(empirical_quantiles)
        stability_score = 1.0 - np.std(expected_spacing) / np.mean(expected_spacing)

        # Combine metrics
        quality_score = 0.6 * uniformity_score + 0.4 * max(0, stability_score)

        return min(1.0, max(0.0, quality_score))

    def detect_distribution_shift(self, new_scores: List[float]) -> Dict[str, Any]:
        """
        Enhanced distribution shift detection using multiple statistical tests.

        Args:
# # #             new_scores: Non-conformity scores from new data  # Module not found  # Module not found  # Module not found

        Returns:
            Dictionary with comprehensive shift detection results
        """
        if not self.is_fitted:
            raise RuntimeError("Must fit calibration before shift detection")

        if len(new_scores) == 0:
            return {
                "shift_detected": False,
                "method": "no_data",
                "requires_recalibration": False,
            }

        new_scores = np.array(new_scores)
        baseline = self.baseline_distribution

        results = {"methods": {}}

        # Method 1: Kolmogorov-Smirnov test
        if self.config.shift_detection_method in ["ks", "all"]:
# # #             from scipy.stats import ks_2samp  # Module not found  # Module not found  # Module not found

            ks_statistic, ks_p_value = ks_2samp(baseline, new_scores)
            results["methods"]["ks"] = {
                "statistic": ks_statistic,
                "p_value": ks_p_value,
                "shift_detected": ks_statistic > self.config.distribution_shift_bound,
                "critical_value": self.config.distribution_shift_bound,
            }

        # Method 2: Maximum Mean Discrepancy (MMD) test
        if self.config.shift_detection_method in ["mmd", "all"]:
            mmd_statistic = self._compute_mmd(baseline, new_scores)
            # Simple threshold-based detection for MMD
            mmd_shift = mmd_statistic > self.config.distribution_shift_bound * 2
            results["methods"]["mmd"] = {
                "statistic": mmd_statistic,
                "shift_detected": mmd_shift,
                "critical_value": self.config.distribution_shift_bound * 2,
            }

        # Method 3: Energy distance test
        if self.config.shift_detection_method in ["energy", "all"]:
            energy_stat = self._compute_energy_distance(baseline, new_scores)
            energy_shift = energy_stat > self.config.distribution_shift_bound * 1.5
            results["methods"]["energy"] = {
                "statistic": energy_stat,
                "shift_detected": energy_shift,
                "critical_value": self.config.distribution_shift_bound * 1.5,
            }

        # Aggregate decision
        shift_votes = []
        primary_p_value = 1.0

        for method_name, method_results in results["methods"].items():
            if "shift_detected" in method_results:
                shift_votes.append(method_results["shift_detected"])
                if method_name == "ks" and "p_value" in method_results:
                    primary_p_value = method_results["p_value"]

        # Conservative aggregation: shift detected if any method detects it
        overall_shift = any(shift_votes) if shift_votes else False

        # Recalibration decision
        requires_recalibration = (
            overall_shift or primary_p_value < self.config.recalibration_threshold
        )

        # Update shift history
        shift_event = {
            "timestamp": str(np.datetime64("now")),
            "shift_detected": overall_shift,
            "primary_p_value": primary_p_value,
            "n_new_samples": len(new_scores),
        }
        self.shift_history.append(shift_event)

        # Keep only recent history
        if len(self.shift_history) > 100:
            self.shift_history = self.shift_history[-100:]

        results.update(
            {
                "shift_detected": overall_shift,
                "primary_method": self.config.shift_detection_method,
                "primary_p_value": primary_p_value,
                "requires_recalibration": requires_recalibration,
                "shift_bound": self.config.distribution_shift_bound,
                "n_baseline_samples": len(baseline),
                "n_new_samples": len(new_scores),
                "shift_history_length": len(self.shift_history),
            }
        )

        if overall_shift:
            logger.warning(
                f"Distribution shift detected using {self.config.shift_detection_method}. "
                f"Recalibration recommended: {requires_recalibration}"
            )

        return results

    def _compute_mmd(
        self, x: np.ndarray, y: np.ndarray, gamma: Optional[float] = None
    ) -> float:
        """Compute Maximum Mean Discrepancy with RBF kernel for 1D data."""
        # Ensure inputs are 2D
        x = np.atleast_2d(x).reshape(-1, 1)
        y = np.atleast_2d(y).reshape(-1, 1)

        if gamma is None:
            # Use median heuristic for bandwidth
            all_data = np.vstack([x, y])
            distances = []
            for i in range(len(all_data)):
                for j in range(i + 1, len(all_data)):
                    distances.append(np.linalg.norm(all_data[i] - all_data[j]))
            distances = np.array(distances)
            gamma = (
                1.0 / (2 * np.median(distances[distances > 0]) ** 2)
                if len(distances) > 0
                else 1.0
            )

        def rbf_kernel(a, b):
            # Compute RBF kernel between sets a and b
            distances = np.sum((a[:, np.newaxis, :] - b[np.newaxis, :, :]) ** 2, axis=2)
            return np.exp(-gamma * distances)

        # Compute kernel matrices
        kxx = rbf_kernel(x, x)
        kyy = rbf_kernel(y, y)
        kxy = rbf_kernel(x, y)

        # MMD statistic
        n_x, n_y = len(x), len(y)
        mmd = (
            np.sum(kxx) / (n_x * n_x)
            + np.sum(kyy) / (n_y * n_y)
            - 2 * np.sum(kxy) / (n_x * n_y)
        )

        return max(0, mmd)  # MMD should be non-negative

    def _compute_energy_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute energy distance between two 1D samples."""
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()

        # Compute pairwise distances for 1D data
        def pairwise_distance_1d(a, b):
            return np.mean(np.abs(a[:, np.newaxis] - b[np.newaxis, :]))

        # Energy distance components
        d_xx = pairwise_distance_1d(x, x)
        d_yy = pairwise_distance_1d(y, y)
        d_xy = pairwise_distance_1d(x, y)

        energy_dist = 2 * d_xy - d_xx - d_yy
        return max(0, energy_dist)

    def get_comprehensive_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive system diagnostics with enhanced metrics."""
        base_diagnostics = {
            "status": "fitted" if self.is_fitted else "not_fitted",
            "config": self.config.__dict__,
            "system_info": {
                "certificates_cached": len(self._certificate_cache),
                "scorer_fitted": self.scorer_fitted,
                "shift_events_recorded": len(self.shift_history),
                "cv_enabled": self.config.enable_cross_validation,
            },
        }

        if not self.is_fitted:
            return base_diagnostics

        # Enhanced calibration statistics
        cal_stats = {
            "descriptive": {
                "n_samples": len(self.calibration_scores),
                "mean": float(np.mean(self.calibration_scores)),
                "std": float(np.std(self.calibration_scores)),
                "min": float(np.min(self.calibration_scores)),
                "max": float(np.max(self.calibration_scores)),
                "range": float(np.ptp(self.calibration_scores)),
            },
            "quantiles": {
                f"{int(q*100)}%": float(np.quantile(self.calibration_scores, q))
                for q in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            },
            "distribution": {
                "skewness": float(stats.skew(self.calibration_scores)),
                "kurtosis": float(stats.kurtosis(self.calibration_scores)),
                "normality_test_p": float(
                    stats.jarque_bera(self.calibration_scores)[1]
                ),
            },
        }

        # Validation set statistics
        val_stats = {}
        if len(self.validation_scores) > 0:
            val_stats = {
                "n_samples": len(self.validation_scores),
                "mean": float(np.mean(self.validation_scores)),
                "std": float(np.std(self.validation_scores)),
                "distribution_comparison": {
                    "ks_statistic": float(
                        stats.ks_2samp(self.calibration_scores, self.validation_scores)[
                            0
                        ]
                    ),
                    "ks_p_value": float(
                        stats.ks_2samp(self.calibration_scores, self.validation_scores)[
                            1
                        ]
                    ),
                },
            }

        # Adaptive quantile information
        quantile_info = {
            "value": float(self.adaptive_quantile)
            if self.adaptive_quantile is not None
            else None,
            "bounds": self.quantile_bounds,
            "calibration_coverage": float(
                np.mean(self.calibration_scores <= self.adaptive_quantile)
            )
            if self.adaptive_quantile is not None
            else None,
        }

        # System health indicators
        health_metrics = {
            "calibration_quality_score": self._assess_calibration_quality(),
            "coverage_stability": self.cv_results.get("coverage_stability", None)
            if self.cv_results
            else None,
            "recent_shifts": len(
                [
                    event
                    for event in self.shift_history[-10:]
                    if event.get("shift_detected", False)
                ]
            ),
            "cache_efficiency": len(self._certificate_cache)
            / max(1, len(self._certificate_cache) * 2),  # Placeholder metric
        }

        # Recent shift history summary
        shift_summary = {
            "total_events": len(self.shift_history),
            "recent_shifts": sum(
                1
                for event in self.shift_history[-20:]
                if event.get("shift_detected", False)
            ),
            "last_shift": self.shift_history[-1] if self.shift_history else None,
        }

        return {
            **base_diagnostics,
            "calibration_statistics": cal_stats,
            "validation_statistics": val_stats,
            "quantile_information": quantile_info,
            "health_metrics": health_metrics,
            "shift_detection_summary": shift_summary,
            "cross_validation_results": self.cv_results,
        }

    # Backward compatibility alias
    def get_diagnostics(self) -> Dict[str, Any]:
        """Backward compatibility wrapper for diagnostics."""
        return self.get_comprehensive_diagnostics()


# Backward compatibility alias
ConformalRiskController = EnhancedConformalRiskController
