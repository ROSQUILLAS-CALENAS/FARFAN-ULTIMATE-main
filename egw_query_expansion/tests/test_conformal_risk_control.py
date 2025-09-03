"""
Test suite for Conformal Risk Control system

Tests implementation of adaptive conformal prediction with
distribution-free guarantees following Angelopoulos et al. (2024).
"""

import pickle
import tempfile
# # # from typing import Dict, List  # Module not found  # Module not found  # Module not found

import numpy as np
import pytest

# # # from egw_query_expansion.core.conformal_risk_control import (  # Module not found  # Module not found  # Module not found
    ClassificationNonconformityScorer,
    ConformalRiskController,
    PredictionSet,
    RegressionNonconformityScorer,
    RiskCertificate,
    RiskControlConfig,
)


class TestConformalRiskControl:
    @pytest.fixture
    def config(self) -> RiskControlConfig:
        """Standard test configuration."""
        return RiskControlConfig(
            alpha=0.1, random_seed=42, calibration_ratio=0.5, validation_size=100
        )

    @pytest.fixture
    def regression_data(self) -> tuple:
        """Generate synthetic regression data."""
        np.random.seed(42)
        n_samples = 1000

        # True function: y = 2x + noise
        X = np.random.uniform(-5, 5, n_samples)
        noise = np.random.normal(0, 0.5, n_samples)
        y_true = 2 * X + noise

        # Predictions with some bias
        y_pred = 2 * X + 0.1 + np.random.normal(0, 0.1, n_samples)

        return y_pred.tolist(), y_true.tolist()

    @pytest.fixture
    def classification_data(self) -> tuple:
        """Generate synthetic classification data."""
        np.random.seed(42)
        n_samples = 1000
        n_classes = 3

        # Random softmax predictions
        logits = np.random.randn(n_samples, n_classes)
        predictions = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

        # True labels
        labels = np.random.randint(0, n_classes, n_samples)

        return predictions.tolist(), labels.tolist()

    def test_risk_control_config(self):
        """Test risk control configuration."""
        config = RiskControlConfig(alpha=0.05, random_seed=123)

        assert config.alpha == 0.05
        assert config.random_seed == 123
        assert config.calibration_ratio == 0.5
        assert config.adaptive_lambda is True

    def test_regression_nonconformity_scorer(self):
        """Test regression non-conformity scorer."""
        scorer = RegressionNonconformityScorer(method="absolute")

        score = scorer.score(prediction=2.0, true_label=1.5)
        assert score == 0.5

        metadata = scorer.get_metadata()
        assert metadata["scorer_type"] == "regression"
        assert metadata["method"] == "absolute"

        # Test squared method
        squared_scorer = RegressionNonconformityScorer(method="squared")
        score = squared_scorer.score(prediction=3.0, true_label=1.0)
        assert score == 4.0

    def test_classification_nonconformity_scorer(self):
        """Test classification non-conformity scorer."""
        scorer = ClassificationNonconformityScorer(method="softmax")

        prediction = np.array([0.1, 0.7, 0.2])
        score = scorer.score(prediction=prediction, true_label=1)
        assert score == pytest.approx(1.0 - 0.7)

        metadata = scorer.get_metadata()
        assert metadata["scorer_type"] == "classification"
        assert metadata["method"] == "softmax"

    def test_prediction_set(self):
        """Test prediction set functionality."""
        pred_set = PredictionSet(set_values={0, 1, 2}, confidence_level=0.9)

        assert pred_set.size == 3
        assert pred_set.contains(1) is True
        assert pred_set.contains(5) is False
        assert pred_set.confidence_level == 0.9

    def test_controller_initialization(self, config):
        """Test controller initialization."""
        controller = ConformalRiskController(config)

        assert controller.config.alpha == 0.1
        assert controller.is_fitted is False
        assert len(controller.calibration_scores) == 0

    def test_regression_calibration_and_prediction(self, config, regression_data):
        """Test regression calibration and interval construction."""
        predictions, labels = regression_data
        controller = ConformalRiskController(config)
        scorer = RegressionNonconformityScorer(method="absolute")

        # Fit calibration
        controller.fit_calibration(predictions, labels, scorer)

        assert controller.is_fitted is True
        assert len(controller.calibration_scores) > 0
        assert len(controller.validation_scores) > 0
        assert controller.adaptive_quantile is not None

        # Construct confidence interval
        test_prediction = 5.0
        lower, upper = controller.construct_confidence_interval(test_prediction, scorer)

        assert lower < upper
        assert lower <= test_prediction <= upper

        # Interval should have reasonable width
        width = upper - lower
        assert 0.5 < width < 10.0

    def test_classification_calibration_and_sets(self, config, classification_data):
        """Test classification calibration and prediction sets."""
        predictions, labels = classification_data
        controller = ConformalRiskController(config)
        scorer = ClassificationNonconformityScorer(method="softmax")

        # Fit calibration
        controller.fit_calibration(predictions, labels, scorer)

        assert controller.is_fitted is True

        # Construct prediction set for test example
        test_pred = np.array([0.2, 0.5, 0.3])
        candidates = {i: scorer.score(test_pred, i) for i in range(3)}
        pred_set = controller.construct_prediction_set(candidates)

        assert isinstance(pred_set, PredictionSet)
        assert pred_set.confidence_level == 0.9
        assert pred_set.size >= 1  # Should contain at least one class

    def test_coverage_validation_regression(self, config, regression_data):
        """Test coverage validation for regression."""
        predictions, labels = regression_data
        controller = ConformalRiskController(config)
        scorer = RegressionNonconformityScorer(method="absolute")

        # Split data for test
        n_train = len(predictions) // 2
        train_preds, test_preds = predictions[:n_train], predictions[n_train:]
        train_labels, test_labels = labels[:n_train], labels[n_train:]

        # Fit and validate
        controller.fit_calibration(train_preds, train_labels, scorer)
        coverage_stats = controller.validate_coverage(test_preds, test_labels, scorer)

        # Check coverage is reasonable (within bounds for finite samples)
        assert 0.0 <= coverage_stats["empirical_coverage"] <= 1.0
        assert coverage_stats["target_coverage"] == 0.9
        assert "coverage_gap" in coverage_stats
        assert "average_set_size" in coverage_stats

        # Coverage should be close to nominal for large samples
        coverage_gap = abs(coverage_stats["coverage_gap"])
        assert coverage_gap < 0.15  # Allow some deviation for finite samples

    def test_risk_bounds_computation(self, config, regression_data):
        """Test risk bounds computation."""
        predictions, labels = regression_data
        controller = ConformalRiskController(config)
        scorer = RegressionNonconformityScorer(method="absolute")

        controller.fit_calibration(predictions, labels, scorer)
        empirical_risk, upper_bound = controller.compute_risk_bounds()

        assert 0.0 <= empirical_risk <= 1.0
        assert empirical_risk <= upper_bound <= 1.0
        assert upper_bound >= empirical_risk  # Upper bound should be conservative

    def test_certificate_generation(self, config, regression_data):
        """Test risk certificate generation."""
        predictions, labels = regression_data
        controller = ConformalRiskController(config)
        scorer = RegressionNonconformityScorer(method="absolute")

        # Split for training and testing
        n_train = len(predictions) // 2
        train_preds, test_preds = predictions[:n_train], predictions[n_train:]
        train_labels, test_labels = labels[:n_train], labels[n_train:]

        controller.fit_calibration(train_preds, train_labels, scorer)

        # Generate certificate
        certificate = controller.generate_certificate(
            input_data=test_preds[:10],
            predictions=test_preds[:10],
            scorer=scorer,
            test_labels=test_labels[:10],
        )

        assert isinstance(certificate, RiskCertificate)
        assert certificate.certificate_id.startswith("crc_")
        assert len(certificate.input_hash) == 64  # SHA256 hash
        assert 0.0 <= certificate.empirical_risk <= 1.0
        assert certificate.risk_bound >= certificate.empirical_risk
        assert certificate.confidence_level == 0.95

        # Certificate should be valid
        assert isinstance(certificate.is_valid(), bool)

    def test_certificate_caching(self, config, regression_data):
        """Test certificate caching for identical inputs."""
        predictions, labels = regression_data
        controller = ConformalRiskController(config)
        scorer = RegressionNonconformityScorer(method="absolute")

        controller.fit_calibration(predictions, labels, scorer)

        input_data = [1.0, 2.0, 3.0]

        # Generate certificate twice with identical inputs
        cert1 = controller.generate_certificate(
            input_data=input_data, predictions=input_data, scorer=scorer
        )

        cert2 = controller.generate_certificate(
            input_data=input_data, predictions=input_data, scorer=scorer
        )

        # Should be identical due to caching
        assert cert1.certificate_id == cert2.certificate_id
        assert cert1.input_hash == cert2.input_hash
        assert cert1.empirical_risk == cert2.empirical_risk

    def test_distribution_shift_detection(self, config):
        """Test distribution shift detection."""
        # Create controller with initial data
        np.random.seed(42)
        initial_predictions = np.random.normal(0, 1, 500).tolist()
        initial_labels = np.random.normal(0, 1, 500).tolist()

        controller = ConformalRiskController(config)
        scorer = RegressionNonconformityScorer(method="absolute")
        controller.fit_calibration(initial_predictions, initial_labels, scorer)

        # Test with similar distribution (no shift)
        similar_scores = np.abs(np.random.normal(0, 1, 100)).tolist()
        shift_results = controller.detect_distribution_shift(similar_scores)

        assert "ks_statistic" in shift_results
        assert "p_value" in shift_results
        assert "shift_detected" in shift_results
        assert shift_results["shift_bound"] == config.distribution_shift_bound

        # Test with very different distribution (shift detected)
        shifted_scores = np.abs(np.random.normal(10, 1, 100)).tolist()
        shift_results = controller.detect_distribution_shift(shifted_scores)

        # Should detect significant shift
        assert shift_results["shift_detected"] == True
        assert shift_results["requires_recalibration"] == True

    def test_adaptive_quantile_computation(self, config, regression_data):
        """Test adaptive quantile computation."""
        predictions, labels = regression_data

        # Test with adaptive lambda enabled
        config.adaptive_lambda = True
        controller_adaptive = ConformalRiskController(config)
        scorer = RegressionNonconformityScorer(method="absolute")
        controller_adaptive.fit_calibration(predictions, labels, scorer)
        adaptive_quantile = controller_adaptive.adaptive_quantile

        # Test with adaptive lambda disabled
        config.adaptive_lambda = False
        controller_standard = ConformalRiskController(config)
        controller_standard.fit_calibration(predictions, labels, scorer)
        standard_quantile = controller_standard.adaptive_quantile

        # Both should be reasonable quantiles
        assert 0.0 < adaptive_quantile < 10.0
        assert 0.0 < standard_quantile < 10.0

        # They may be different due to adaptation
        # (not asserting equality since adaptation may or may not change the value)

    def test_diagnostics(self, config, regression_data):
        """Test system diagnostics."""
        predictions, labels = regression_data
        controller = ConformalRiskController(config)

        # Diagnostics before fitting
        diag_unfitted = controller.get_diagnostics()
        assert diag_unfitted["status"] == "not_fitted"

        # Diagnostics after fitting
        scorer = RegressionNonconformityScorer(method="absolute")
        controller.fit_calibration(predictions, labels, scorer)

        diag_fitted = controller.get_diagnostics()
        assert diag_fitted["status"] == "fitted"
        assert "calibration_size" in diag_fitted
        assert "validation_size" in diag_fitted
        assert "adaptive_quantile" in diag_fitted
        assert "calibration_stats" in diag_fitted

        # Check calibration statistics
        cal_stats = diag_fitted["calibration_stats"]
        required_stats = ["mean", "std", "min", "max", "quantiles"]
        for stat in required_stats:
            assert stat in cal_stats

    def test_error_handling(self, config):
        """Test error handling and edge cases."""
        controller = ConformalRiskController(config)
        scorer = RegressionNonconformityScorer(method="absolute")

        # Test prediction without fitting
        with pytest.raises(RuntimeError, match="Must fit calibration"):
            controller.construct_confidence_interval(5.0, scorer)

        with pytest.raises(RuntimeError, match="Must fit calibration"):
            controller.construct_prediction_set({0: 0.1, 1: 0.9})

        with pytest.raises(RuntimeError, match="Must fit calibration"):
            controller.generate_certificate([1, 2, 3], [1, 2, 3], scorer)

        # Test mismatched data lengths
        with pytest.raises(ValueError, match="same length"):
            controller.fit_calibration([1, 2, 3], [1, 2], scorer)

        # Test insufficient data
        with pytest.raises(ValueError, match="Need at least"):
            controller.fit_calibration([1, 2], [1, 2], scorer)

    def test_deterministic_behavior(self, config):
        """Test that identical inputs produce identical results."""
        np.random.seed(123)
        predictions = np.random.normal(0, 1, 500).tolist()
        labels = np.random.normal(0, 1, 500).tolist()
        scorer = RegressionNonconformityScorer(method="absolute")

        # Create two controllers with same config
        controller1 = ConformalRiskController(config)
        controller2 = ConformalRiskController(config)

        # Fit both with same data
        controller1.fit_calibration(predictions, labels, scorer)
        controller2.fit_calibration(predictions, labels, scorer)

        # Should produce identical quantiles
        assert controller1.adaptive_quantile == controller2.adaptive_quantile

        # Should produce identical certificates
        test_input = [1.0, 2.0, 3.0]
        cert1 = controller1.generate_certificate(test_input, test_input, scorer)
        cert2 = controller2.generate_certificate(test_input, test_input, scorer)

        assert cert1.input_hash == cert2.input_hash
        assert cert1.empirical_risk == cert2.empirical_risk
        assert cert1.risk_bound == cert2.risk_bound


if __name__ == "__main__":
    pytest.main([__file__])
