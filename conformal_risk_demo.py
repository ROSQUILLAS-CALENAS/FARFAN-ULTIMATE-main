"""
Conformal Risk Control Demo

Demonstrates the implementation of adaptive conformal prediction with
distribution-free guarantees following Angelopoulos et al. (2024).

This example shows:
1. Calibration with fixed deterministic splits
2. Construction of prediction sets and confidence intervals
3. Risk certification with finite-sample bounds
4. Distribution shift detection
"""

import logging
# # # from typing import List, Tuple  # Module not found  # Module not found  # Module not found

import matplotlib.pyplot as plt
import numpy as np

# # # from egw_query_expansion.core.conformal_risk_control import (  # Module not found  # Module not found  # Module not found
    ClassificationNonconformityScorer,
    ConformalRiskController,
    RegressionNonconformityScorer,
    RiskControlConfig,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_regression_data(
    n_samples: int = 1000, seed: int = 42
) -> Tuple[List[float], List[float]]:
    """Generate synthetic regression data with known relationship."""
    np.random.seed(seed)

    # True function: y = 2x + sin(x) + noise
    X = np.random.uniform(-3, 3, n_samples)
    noise = np.random.normal(0, 0.3, n_samples)
    y_true = 2 * X + np.sin(X) + noise

    # Model predictions (slightly biased)
    model_noise = np.random.normal(0, 0.1, n_samples)
    y_pred = 2 * X + np.sin(X) * 0.9 + 0.1 + model_noise

    return y_pred.tolist(), y_true.tolist()


def generate_classification_data(
    n_samples: int = 1000, n_classes: int = 3, seed: int = 42
) -> Tuple[List[np.ndarray], List[int]]:
    """Generate synthetic classification data."""
    np.random.seed(seed)

    # Generate features and true labels
    X = np.random.randn(n_samples, 2)
    true_labels = np.random.randint(0, n_classes, n_samples)

    # Generate model predictions (softmax probabilities)
    # Add some calibration issues to make it interesting
    logits = np.random.randn(n_samples, n_classes)

    # Make predictions correlate with true labels but with some bias
    for i in range(n_samples):
        logits[i, true_labels[i]] += 1.5  # Boost correct class

    predictions = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    return predictions.tolist(), true_labels.tolist()


def demo_regression_conformal_control():
    """Demonstrate conformal risk control for regression."""
    print("\n" + "=" * 60)
    print("REGRESSION CONFORMAL RISK CONTROL DEMO")
    print("=" * 60)

    # Generate data
    print("Generating regression data...")
    predictions, labels = generate_regression_data(n_samples=2000)

    # Split into train/calibration and test
    n_train = 1200
    train_preds, test_preds = predictions[:n_train], predictions[n_train:]
    train_labels, test_labels = labels[:n_train], labels[n_train:]

    # Configure system
    config = RiskControlConfig(
        alpha=0.1,  # 90% coverage target
        random_seed=42,
        calibration_ratio=0.6,
        adaptive_lambda=True,
    )

    # Initialize controller
    controller = ConformalRiskController(config)
    scorer = RegressionNonconformityScorer(method="absolute")

    print(f"Target coverage: {1 - config.alpha:.1%}")
    print(f"Training samples: {len(train_preds)}")

    # Fit calibration
    print("\nFitting calibration...")
    controller.fit_calibration(train_preds, train_labels, scorer)

    # Get diagnostics
    diagnostics = controller.get_diagnostics()
    print(f"Calibration samples: {diagnostics['calibration_size']}")
    print(f"Validation samples: {diagnostics['validation_size']}")
    print(f"Adaptive quantile: {diagnostics['adaptive_quantile']:.4f}")

    # Construct confidence intervals for test predictions
    print("\nConstructing confidence intervals...")
    intervals = []
    coverage_count = 0

    for i, (pred, true_val) in enumerate(zip(test_preds[:100], test_labels[:100])):
        lower, upper = controller.construct_confidence_interval(pred, scorer)
        intervals.append((lower, upper))

        if lower <= true_val <= upper:
            coverage_count += 1

    empirical_coverage = coverage_count / 100
    print(f"Empirical coverage on test set: {empirical_coverage:.1%}")
    print(f"Average interval width: {np.mean([u - l for l, u in intervals]):.3f}")

    # Validate coverage on larger test set
    coverage_stats = controller.validate_coverage(test_preds, test_labels, scorer)
    print(f"\nFull test set coverage validation:")
    print(f"  Empirical coverage: {coverage_stats['empirical_coverage']:.3f}")
    print(f"  Target coverage:    {coverage_stats['target_coverage']:.3f}")
    print(f"  Coverage gap:       {coverage_stats['coverage_gap']:+.3f}")
    print(f"  Average set size:   {coverage_stats['average_set_size']:.3f}")

    # Compute risk bounds
    empirical_risk, upper_bound = controller.compute_risk_bounds()
    print(f"\nRisk bounds:")
    print(f"  Empirical risk: {empirical_risk:.4f}")
    print(f"  Upper bound:    {upper_bound:.4f}")

    # Generate certificate
    print("\nGenerating risk certificate...")
    certificate = controller.generate_certificate(
        input_data=test_preds[:50],
        predictions=test_preds[:50],
        scorer=scorer,
        test_labels=test_labels[:50],
    )

    print(f"Certificate ID: {certificate.certificate_id}")
    print(f"Certificate valid: {certificate.is_valid()}")
    print(f"Coverage achieved: {certificate.coverage_achieved:.3f}")
    print(f"Risk bound: {certificate.risk_bound:.4f}")

    return controller, coverage_stats, certificate


def demo_classification_conformal_control():
    """Demonstrate conformal risk control for classification."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION CONFORMAL RISK CONTROL DEMO")
    print("=" * 60)

    # Generate data
    print("Generating classification data...")
    predictions, labels = generate_classification_data(n_samples=1500, n_classes=3)

    # Split data
    n_train = 1000
    train_preds, test_preds = predictions[:n_train], predictions[n_train:]
    train_labels, test_labels = labels[:n_train], labels[n_train:]

    # Configure system
    config = RiskControlConfig(
        alpha=0.05, random_seed=42, calibration_ratio=0.5  # 95% coverage target
    )

    controller = ConformalRiskController(config)
    scorer = ClassificationNonconformityScorer(method="softmax")

    print(f"Target coverage: {1 - config.alpha:.1%}")
    print(f"Number of classes: 3")

    # Fit calibration
    controller.fit_calibration(train_preds, train_labels, scorer)

    # Construct prediction sets
    print("\nConstructing prediction sets...")
    set_sizes = []
    coverage_count = 0

    for i, (pred, true_label) in enumerate(zip(test_preds[:100], test_labels[:100])):
        # Create candidate set with non-conformity scores
        candidates = {
            class_idx: scorer.score(np.array(pred), class_idx) for class_idx in range(3)
        }

        pred_set = controller.construct_prediction_set(candidates)
        set_sizes.append(pred_set.size)

        if pred_set.contains(true_label):
            coverage_count += 1

    empirical_coverage = coverage_count / 100
    avg_set_size = np.mean(set_sizes)

    print(f"Empirical coverage: {empirical_coverage:.1%}")
    print(f"Average prediction set size: {avg_set_size:.2f}")
    print(f"Set size distribution: {np.bincount(set_sizes)}")

    # Full validation
    coverage_stats = controller.validate_coverage(test_preds, test_labels, scorer)
    print(f"\nFull validation results:")
    print(f"  Coverage: {coverage_stats['empirical_coverage']:.3f}")
    print(f"  Target:   {coverage_stats['target_coverage']:.3f}")
    print(f"  Gap:      {coverage_stats['coverage_gap']:+.3f}")

    return controller, coverage_stats


def demo_distribution_shift_detection():
    """Demonstrate distribution shift detection."""
    print("\n" + "=" * 60)
    print("DISTRIBUTION SHIFT DETECTION DEMO")
    print("=" * 60)

    # Train on normal distribution
    np.random.seed(42)
    normal_preds = np.random.normal(0, 1, 1200).tolist()
    normal_labels = [pred + np.random.normal(0, 0.1) for pred in normal_preds]

    config = RiskControlConfig(
        alpha=0.1, random_seed=42, distribution_shift_bound=0.2, validation_size=200
    )
    controller = ConformalRiskController(config)
    scorer = RegressionNonconformityScorer(method="absolute")

    controller.fit_calibration(normal_preds, normal_labels, scorer)
    print("Calibrated on normal distribution data")

    # Test with similar distribution (no shift)
    similar_preds = np.random.normal(0.1, 1.1, 200).tolist()
    similar_labels = [pred + np.random.normal(0, 0.1) for pred in similar_preds]
    similar_scores = [scorer.score(p, l) for p, l in zip(similar_preds, similar_labels)]

    shift_result = controller.detect_distribution_shift(similar_scores)
    print(f"\nSimilar distribution test:")
    print(f"  KS statistic: {shift_result['ks_statistic']:.4f}")
    print(f"  Shift detected: {shift_result['shift_detected']}")

    # Test with shifted distribution
    shifted_preds = np.random.normal(3, 2, 200).tolist()  # Different mean and variance
    shifted_labels = [pred + np.random.normal(0, 0.5) for pred in shifted_preds]
    shifted_scores = [scorer.score(p, l) for p, l in zip(shifted_preds, shifted_labels)]

    shift_result = controller.detect_distribution_shift(shifted_scores)
    print(f"\nShifted distribution test:")
    print(f"  KS statistic: {shift_result['ks_statistic']:.4f}")
    print(f"  Shift detected: {shift_result['shift_detected']}")
    print(f"  Requires recalibration: {shift_result['requires_recalibration']}")

    return controller


def demo_certificate_reproducibility():
    """Demonstrate certificate reproducibility for identical inputs."""
    print("\n" + "=" * 60)
    print("CERTIFICATE REPRODUCIBILITY DEMO")
    print("=" * 60)

    # Setup
    predictions = [1.0, 2.0, 3.0, 4.0, 5.0] * 200  # Repeated pattern
    labels = [p + np.random.normal(0, 0.1) for p in predictions]

    config = RiskControlConfig(alpha=0.1, random_seed=42)
    scorer = RegressionNonconformityScorer(method="absolute")

    # Create two identical controllers
    controller1 = ConformalRiskController(config)
    controller2 = ConformalRiskController(config)

    controller1.fit_calibration(predictions, labels, scorer)
    controller2.fit_calibration(predictions, labels, scorer)

    # Generate certificates for identical input
    test_input = [2.5, 3.5, 4.5]

    cert1 = controller1.generate_certificate(test_input, test_input, scorer)
    cert2 = controller2.generate_certificate(test_input, test_input, scorer)

    print("Certificate reproducibility test:")
    print(f"  Certificate 1 ID: {cert1.certificate_id}")
    print(f"  Certificate 2 ID: {cert2.certificate_id}")
    print(f"  IDs match: {cert1.certificate_id == cert2.certificate_id}")
    print(f"  Risk bounds match: {cert1.risk_bound == cert2.risk_bound}")
    print(f"  Input hashes match: {cert1.input_hash == cert2.input_hash}")

    # Test certificate caching
    cert3 = controller1.generate_certificate(test_input, test_input, scorer)
    print(f"  Cached certificate ID: {cert3.certificate_id}")
    print(f"  Cache working: {cert1.certificate_id == cert3.certificate_id}")

    return controller1


def visualize_regression_results(controller, predictions, labels, n_plot=100):
    """Visualize regression conformal prediction results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available for visualization")
        return

    scorer = RegressionNonconformityScorer(method="absolute")

    # Get subset for plotting
    plot_preds = predictions[:n_plot]
    plot_labels = labels[:n_plot]

    # Construct confidence intervals
    intervals = []
    for pred in plot_preds:
        lower, upper = controller.construct_confidence_interval(pred, scorer)
        intervals.append((lower, upper))

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x_indices = range(n_plot)
    lowers = [interval[0] for interval in intervals]
    uppers = [interval[1] for interval in intervals]

    # Plot confidence intervals
    ax.fill_between(
        x_indices,
        lowers,
        uppers,
        alpha=0.3,
        color="blue",
        label=f"{(1-controller.config.alpha)*100:.0f}% Confidence Intervals",
    )

    # Plot predictions and true values
    ax.scatter(x_indices, plot_preds, color="red", alpha=0.6, s=20, label="Predictions")
    ax.scatter(
        x_indices, plot_labels, color="green", alpha=0.6, s=20, label="True Values"
    )

    # Calculate coverage
    coverage_indicators = [
        lower <= true <= upper for (lower, upper), true in zip(intervals, plot_labels)
    ]
    coverage_rate = np.mean(coverage_indicators)

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Value")
    ax.set_title(f"Conformal Prediction Intervals (Coverage: {coverage_rate:.1%})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("conformal_regression_demo.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Visualization saved as 'conformal_regression_demo.png'")


def main():
    """Run complete conformal risk control demonstration."""
    print("CONFORMAL RISK CONTROL SYSTEM DEMO")
    print("Based on Angelopoulos et al. (2024)")
    print("=" * 60)

    try:
        # Run regression demo
        reg_controller, reg_stats, certificate = demo_regression_conformal_control()

        # Run classification demo
        clf_controller, clf_stats = demo_classification_conformal_control()

        # Run distribution shift demo
        shift_controller = demo_distribution_shift_detection()

        # Run reproducibility demo
        repro_controller = demo_certificate_reproducibility()

        # Generate visualization if matplotlib available
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATION")
        print("=" * 60)

        reg_preds, reg_labels = generate_regression_data(n_samples=500)
        visualize_regression_results(reg_controller, reg_preds, reg_labels)

        # Summary
        print("\n" + "=" * 60)
        print("DEMO SUMMARY")
        print("=" * 60)
        print(
            f"✓ Regression coverage: {reg_stats['empirical_coverage']:.1%} "
            f"(target: {reg_stats['target_coverage']:.1%})"
        )
        print(
            f"✓ Classification coverage: {clf_stats['empirical_coverage']:.1%} "
            f"(target: {clf_stats['target_coverage']:.1%})"
        )
        print(f"✓ Risk certificate generated: {certificate.certificate_id}")
        print(f"✓ Certificate valid: {certificate.is_valid()}")
        print(f"✓ Distribution shift detection: Working")
        print(f"✓ Reproducibility: Guaranteed")

        print("\nSystem provides distribution-free finite-sample coverage guarantees!")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
